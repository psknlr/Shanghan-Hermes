"""ServiceContext — the framework-agnostic API surface behind the web console.

Every method returns a JSON-serializable dict and reuses the existing engine
(RAG, apps, agent, council, paper, research). Artifacts are lazy-loaded once
and shared; the HTTP layer is a thin adapter over this.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .. import config
from ..schemas import read_jsonl


class ServiceContext:
    def __init__(self):
        self._art = None
        self._clause_rag = None
        self._skill_rag = None
        self._matcher = None
        self._registry = None
        self._llm = None

    # -- lazy resources -------------------------------------------------
    @property
    def art(self):
        if self._art is None:
            from ..orchestrator import Artifacts
            self._art = Artifacts()
        return self._art

    @property
    def clause_rag(self):
        if self._clause_rag is None:
            from ..rag.clause_rag import ClauseRAG
            self._clause_rag = ClauseRAG.load()
        return self._clause_rag

    @property
    def matcher(self):
        if self._matcher is None:
            from ..apps.doctor import FormulaMatcher
            self._matcher = FormulaMatcher(self.art.formula_rules, self.art.clause_store())
        return self._matcher

    @property
    def registry(self):
        if self._registry is None:
            from ..agent.tools import get_registry
            self._registry = get_registry()
        return self._registry

    @property
    def llm(self):
        if self._llm is None:
            from ..llm.client import get_client
            self._llm = get_client()
        return self._llm

    def warm(self):
        _ = self.clause_rag, self.art.formula_rules, self.registry
        # 段落級引文邊（歷代引用條目用）：啟動時預建，首個抽屜點擊不再等掃描
        try:
            from ..trace.passages import load_full_edges
            load_full_edges()
        except Exception:
            pass

    @staticmethod
    def ready() -> bool:
        return (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists()

    # -- dashboard ------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        rules = read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
        clauses = read_jsonl(config.CLAUSE_DIR / "clauses.jsonl")
        levels = Counter(r["autonomous_review"]["release_level"] for r in rules)
        types = Counter(r["rule_type"] for r in rules)
        return {
            "clauses": len(clauses),
            "canonical": sum(1 for c in clauses if c["text_type"] == "original_clause"),
            "initial_rules": len(rules),
            "release_levels": dict(levels),
            "rule_types": dict(types.most_common()),
            "formula_pattern_rules": len(self.art.formula_rules),
            "six_channel_rules": len(self.art.six_channel_rules),
            "therapy_rules": len(read_jsonl(config.RULES_THERAPY_DIR / "therapy_rules.jsonl")),
            "mistreatment_rules": len(self.art.mistreatment_rules),
            "differential_rules": len(self.art.differential_rules),
            "merged_rules": len(self.art.merged_rules),
            "variant_rules": len(self.art.variant_rules),
            "commentary_rules": len(self.art.commentary_rules),
            "clause_relations": len(read_jsonl(config.RELATION_DIR / "clause_relations.jsonl")),
            "audits": len(read_jsonl(config.AUDIT_DIR / "audit_log.jsonl")),
            "skills": self._skill_count(),
        }

    def _skill_count(self) -> int:
        import json
        m = config.SKILLS_DIR / "skills_manifest.json"
        if m.exists():
            try:
                return json.loads(m.read_text(encoding="utf-8")).get("total_dirs", 0)
            except Exception:
                return 0
        return 0

    def llm_status(self) -> Dict[str, Any]:
        from ..llm.config import RECOMMENDED_MODELS
        st = self.llm.status()
        st["recommended_models"] = RECOMMENDED_MODELS
        return st

    # -- retrieval / clause --------------------------------------------
    def search(self, query: str, top_k: int = 8, six_channel: str = None,
               formula: str = None, field: str = None, expand: bool = False) -> Dict:
        hits = self.clause_rag.search(query, top_k=top_k, six_channel=six_channel or None,
                                      formula=formula or None, field=field or None,
                                      expand_relations=expand)
        return {"query": query, "hits": hits, "count": len(hits)}

    def explain_clause(self, ref, role: str = "student") -> Dict:
        from .. import safety
        c = self.clause_rag.get_clause(ref)
        if c is None:
            return {"error": f"未找到條文 {ref}"}
        rules = [r for r in read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
                 if r["clause_id"] == c.clause_id]
        variants = [v for v in self.art.variant_rules if v.clause_id == c.clause_id]
        comments = [v for v in self.art.commentary_rules if v.clause_id == c.clause_id]
        payload = {
            "clause_id": c.clause_id, "clause_number": c.clause_number,
            "chapter": c.chapter, "six_channel": c.six_channel,
            "layer_label": config.LAYER_LABEL.get(c.layer, ""),
            "text": c.clean_text,
            "entities": {"symptoms": c.symptoms, "negated_findings": c.negated_findings,
                         "pulse": c.pulse, "formulas": c.formula_names,
                         "disease_patterns": c.disease_patterns,
                         "therapy": c.therapy_terms,
                         "contraindications": c.contraindication_terms,
                         "mistreatment": c.mistreatment_terms,
                         "prognosis": c.prognosis_terms},
            "formula_blocks": [fb.to_dict() for fb in c.formula_blocks],
            "initial_rules": [{"id": r["initial_rule_id"], "type": r["rule_type"],
                               "strength": r.get("prescription_strength", ""),
                               "release": r["autonomous_review"]["release_level"],
                               "interpretation": r.get("interpretation", "")}
                              for r in rules],
            "relations": self.clause_rag.related(c.clause_id, limit=10),
            "variants": [{"book": v.variant_book, "text": v.variant_text,
                          "similarity": v.similarity,
                          "differences": v.notable_differences} for v in variants],
            "commentaries": [{"commentator": v.commentator,
                              "book": v.book, "chapter": v.chapter,
                              "text": v.commentary_text[:400]} for v in comments],
        }
        # 十六輪：注家解釋智能化——貼近原文度/學派/分析取徑逐家標注
        #（複用注家爭議鏈的確定性指標），另附歷代古籍段落級引用。
        # 患者端不附：引用段落多含方藥劑量原文（可執行診療信息不出患者面），
        # 序列化出口的 PATIENT_FORBIDDEN_KEYS 投影亦兜底
        if comments and role != "patient":
            try:
                from ..trace.chains import dispute_chain
                dc = dispute_chain(c.clause_id)
                if "error" not in dc:
                    payload["commentary_analysis"] = {
                        "views": dc.get("views", []),
                        "divergence_types_present":
                            dc.get("divergence_types_present", []),
                        "term_divergence": dc.get("term_divergence"),
                        "note": "貼近原文度=注文與條文字二元組重合率；"
                                "學派歸屬為 posthoc_induction；只呈現結構，"
                                "不裁決對錯。"}
            except Exception:
                pass
        if role != "patient":
            try:
                from ..trace.passages import clause_citing_passages
                payload["historical_citations"] = clause_citing_passages(
                    c.clause_id, per_book=2, max_books=30)
            except Exception as exc:
                payload["historical_citations"] = {"error": type(exc).__name__}
        return safety.governed(payload, role)

    # -- apps -----------------------------------------------------------
    def match(self, symptoms: List[str], pulse: List[str] = None,
              six_channel: str = None, top_k: int = 5) -> Dict:
        return self.matcher.match(symptoms=symptoms, pulse=pulse or [],
                                  six_channel=six_channel or None, top_k=top_k)

    def differential(self, formulas: List[str], use_llm: bool = True) -> Dict:
        from .. import safety
        from ..textutil import normalize_query
        names = [normalize_query(f) for f in formulas]
        cands = [d for d in self.art.differential_rules if set(names) <= set(d.formulas)]
        if not cands:
            cands = [d for d in self.art.differential_rules
                     if len(set(names) & set(d.formulas)) >= 2]
        if not cands:
            from ..induce.differential import DifferentialInducer
            one = DifferentialInducer(self.art.formula_rules)._build_one(names, 999)
            cands = [one] if one else []
        if not cands:
            return {"error": "無法構建該鑒別對"}
        d = cands[0].to_dict()
        # 十六輪：規則歸類可錯——逐格回源核驗 + 模型對抗審校（引用過核驗）
        from ..apps.differential_audit import model_review, verify_differential
        store = self.art.clause_store()
        verification = verify_differential(d, self.art.formula_rules, store)
        payload = {"differential": d, "verification": verification}
        if use_llm:
            payload["model_review"] = model_review(
                d, self.art.formula_rules, store, self.llm,
                verification=verification)
        return safety.governed(payload, "doctor")

    def teach(self, channel: str) -> Dict:
        from ..apps.teaching import TeachingBuilder
        tb = TeachingBuilder(self.art.clauses, self.art.six_channel_rules,
                             self.art.formula_rules, self.art.mistreatment_rules)
        return tb.lesson(channel)

    def quiz(self, channel: str = "", n: int = 8, seed: int = 1,
             use_llm: bool = False) -> Dict:
        """練習題（十八輪）：多題型確定性題庫（seed 換批）；
        use_llm=True 且接真模型時由模型自主命題（證據強制綁定給定條文）。"""
        from ..apps.quiz import QuizBuilder, model_quiz
        qb = QuizBuilder(self.art.clauses, self.art.six_channel_rules,
                         self.art.formula_rules, self.art.mistreatment_rules,
                         self.art.differential_rules)
        if use_llm:
            return model_quiz(qb, self.llm, channel=channel, n=n, seed=seed)
        out = qb.build(channel=channel, n=n, seed=seed)
        out["backend"] = "bank"
        return out

    def charmap(self) -> Dict:
        """繁→簡顯示映射（十八輪 UI 簡繁切換；顯示層轉換，原文以繁體為準）。"""
        from ..textutil import T2S
        return {"t2s": T2S,
                "note": "顯示層自動轉換（領域字表，非全量簡繁轉換）；"
                        "古籍原文以繁體為準。"}

    def mistreatment(self, query: str = None) -> Dict:
        return self.registry.call("shanghan_mistreatment", {"query": query or ""})

    def teaching_case(self, mistreatment: str, resulting_pattern: str = "",
                      use_llm: bool = True) -> Dict:
        """誤治傳變 → 教學案例（二十輪）：確定性骨架（規則+證據條文逐字
        取證）恆有；接真模型時另生成敘事層病案，所引 clause_id 過
        CitationGuard。案例為虛構教學情景，不構成診療建議。"""
        from ..textutil import fold_variants, normalize_query
        mt = normalize_query(mistreatment or "")
        rp = normalize_query(resulting_pattern or "")
        rules = [m for m in self.art.mistreatment_rules
                 if (not mt or fold_variants(mt)
                     in fold_variants(m.mistreatment_type))
                 and (not rp or fold_variants(rp)
                      in fold_variants(m.resulting_pattern))]
        if not rules:
            return {"error": f"未找到誤治規則：{mistreatment}"
                             + (f" → {resulting_pattern}" if rp else ""),
                    "available_types": sorted(
                        {m.mistreatment_type
                         for m in self.art.mistreatment_rules})}
        r = rules[0]
        store = self.art.clause_store()
        evidence = []
        for cid in r.supporting_clauses[:6]:
            c = store.get(cid)
            if c is not None:
                evidence.append({"clause_id": cid, "chapter": c.chapter,
                                 "text": c.clean_text})
        channel = (r.six_channel_scope or ["太陽病"])[0]
        rescue = "、".join(r.rescue_formulas) or "（無明文救逆方）"
        manifest = "、".join(r.manifestations[:6]) or "（原文未列具體表現）"
        path_desc = (f"{channel} 誤用「{r.mistreatment_type}」→ 變證"
                     f"「{r.resulting_pattern}」（見證：{manifest}）"
                     f"→ 救逆：{rescue}")
        # 確定性骨架：情景/要點/思考題全部由規則字段拼裝，條文逐字附後
        case = {
            "title": f"{r.mistreatment_type}致{r.resulting_pattern}案（教學）",
            "scenario": (f"【教學案例·虛構】患者初患{channel}，醫者誤用"
                         f"「{r.mistreatment_type}」；隨後出現{manifest}，"
                         f"轉為「{r.resulting_pattern}」。"
                         f"救逆方向：{rescue}。"),
            "key_manifestations": r.manifestations[:6],
            "rescue_formulas": r.rescue_formulas,
            "teaching_points": [
                f"誤治環節：{channel}不當施以「{r.mistreatment_type}」",
                f"變證辨識：{r.resulting_pattern}——關鍵見證 {manifest}",
                f"救逆思路：{rescue}（依據見證據條文）"],
            "discussion_questions": [
                f"本案誤用「{r.mistreatment_type}」為何不當？"
                "請從證據條文中找出原文依據。",
                f"變證「{r.resulting_pattern}」與原證如何鑒別？"
                "哪些表現是轉變的信號？",
                f"為何以 {rescue} 救逆？其方證核心指徵是什麼？"],
        }
        out = {"mistreatment": r.mistreatment_type,
               "resulting_pattern": r.resulting_pattern,
               "channel": channel,
               "release_level": r.release_level,
               "n_matched_rules": len(rules),
               "case": case,
               "evidence": evidence,
               "note": "教學案例為虛構情景：骨架由誤治規則（D 層）確定性"
                       "拼裝、條文逐字回源；不構成診療建議。"}
        if use_llm and self.llm.available:
            out["model_narrative"] = self._teaching_case_narrative(
                path_desc, evidence)
        return out

    def _teaching_case_narrative(self, path_desc: str,
                                 evidence: List[Dict]) -> Dict:
        """模型敘事層病案（E 層）：事實只取規則路徑與證據條文，引用核驗。"""
        from ..agent.citation_guard import CitationGuard
        from ..llm.prompts import (teaching_case_system_prompt,
                                   teaching_case_user_prompt)
        allowed = [e["clause_id"] for e in evidence]
        block = "\n".join(f"- [{e['clause_id']}] {e['text'][:200]}"
                          for e in evidence)
        try:
            res = self.llm.json_complete(teaching_case_system_prompt(),
                                         teaching_case_user_prompt(path_desc,
                                                                   block),
                                         task="synthesize")
        except Exception as exc:
            return {"backend": "error", "error": type(exc).__name__}
        narrative = str(res.get("narrative", ""))[:2500]
        analysis = str(res.get("analysis", ""))[:2000]
        guard = CitationGuard(self.art.clause_store())
        rep = guard.check(narrative + "\n" + analysis, allowed_ids=allowed)
        return {"backend": self.llm.backend,
                "title": str(res.get("title", ""))[:60],
                "narrative": narrative,
                "analysis": analysis,
                "discussion_questions":
                    [str(q)[:160] for q in
                     (res.get("discussion_questions") or [])[:4]],
                "citation_report": rep.to_dict(),
                "note": "敘事層屬 E 層模型生成；引用已逐一過 CitationGuard，"
                        "未核實編號請勿採信。"}

    def patient(self, question: str) -> Dict:
        from ..apps.patient import PatientEducator
        edu = PatientEducator(self.art.six_channel_rules, self.art.clause_store())
        return edu.explain(question)

    def formula_rule(self, formula: str) -> Dict:
        return self.registry.call("shanghan_formula_rule", {"formula": formula})

    def list_formulas(self) -> Dict:
        return {"formulas": sorted(r.formula for r in self.art.formula_rules)}

    def channels(self) -> Dict:
        return {"channels": [r.six_channel for r in self.art.six_channel_rules]}

    def skills(self) -> Dict:
        from ..rag.skill_rag import SkillRAG
        return {"skills": SkillRAG().describe()}

    # -- research / paper ----------------------------------------------
    def research(self, topic: str, outputs: List[str] = None) -> Dict:
        from ..apps.research import ResearchMiner
        miner = ResearchMiner(self.art.clauses, self.art.formula_rules,
                              self.art.mistreatment_rules)
        return miner.run_topic(topic, outputs=outputs or ["rules", "network", "paper_outline"])

    def paper(self, paper_type: str = "formula_pattern", topic: str = "",
              use_llm: bool = True) -> Dict:
        from ..paper.writer import PaperWriter
        writer = PaperWriter(self.art.clauses, self.art.initial_rules,
                             self.art.formula_rules, self.art.six_channel_rules,
                             self.art.mistreatment_rules, self.art.differential_rules,
                             commentary_rules=self.art.commentary_rules,
                             llm_client=self.llm)
        path = writer.generate(paper_type=paper_type, topic=topic or "",
                               use_llm=use_llm)
        meta = {}
        meta_path = path.parent / "paper_meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # 導出（十九輪）：docx（純標準庫 OOXML）+ 全件 zip（稿+SVG+CSV），
        # 經 /api/artifact/download 下載（papers/ 在允許集內）
        downloads = {}
        try:
            from ..paper.exporter import export_bundle
            names = export_bundle(path)
            rel = path.parent.relative_to(config.SHANGHAN_DIR)
            downloads = {fmt: str(rel / name) for fmt, name in names.items()}
        except Exception as exc:
            downloads = {"error": type(exc).__name__}
        manuscript = path.read_text(encoding="utf-8")
        return {"manuscript_path": str(path),
                "manuscript": manuscript,
                "manuscript_chars": len(manuscript),
                "downloads": downloads,
                "meta": meta}

    def complex(self, question: str, role: str = None) -> Dict:
        from ..agent.complex_agent import ComplexAgent
        return ComplexAgent(client=self.llm,
                            registry=self.registry).solve(question, role=role)

    # 會話治理（九輪 P0-6）：無 session_id 不再共用 "default"——服務端生成
    # 獨立 id 並隨響應回傳；會話鍵含服務端主體命名空間（防 fixation/串話）；
    # TTL + 容量上限防無界增長
    SESSION_TTL_S = int(os.environ.get("HERMES_SESSION_TTL", "3600"))
    SESSION_MAX = 256

    def _gc_sessions(self) -> None:
        import time
        now = time.time()
        stale = [k for k, v in self._sessions.items()
                 if now - v["last"] > self.SESSION_TTL_S]
        for k in stale:
            self._sessions.pop(k, None)
        while len(self._sessions) >= self.SESSION_MAX:
            oldest = min(self._sessions, key=lambda k: self._sessions[k]["last"])
            self._sessions.pop(oldest, None)

    def chat(self, question: str, session_id: str = "",
             role: str = None, subject: str = "anonymous") -> Dict:
        import threading
        import time
        import uuid
        from ..agent.session import AgentSession
        if not hasattr(self, "_sessions"):
            self._sessions = {}
            self._sessions_lock = threading.Lock()
        sid = str(session_id or "").strip()
        generated = False
        if not sid or sid == "default":
            sid = uuid.uuid4().hex[:16]
            generated = True
        key = f"{subject}:{sid}"
        # 併發安全（十一輪 九）：會話表加鎖；同一會話的兩個併發請求
        # 經 per-session 鎖串行化（history/ledger 不被交叉寫壞）
        with self._sessions_lock:
            self._gc_sessions()
            entry = self._sessions.get(key)
            if entry is None:
                sess = AgentSession(client=self.llm, registry=self.registry,
                                    namespace=subject)
                # 語義恢復（十四輪 九）：服務重啟/新實例對已持久化會話
                # 重建真實上下文（history/主語/錨點/糾正），不只展示記錄
                try:
                    self._restore_session(sess, subject, sid)
                except Exception:
                    pass
                entry = {"sess": sess, "last": time.time(),
                         "lock": threading.Lock()}
                self._sessions[key] = entry
            entry["last"] = time.time()
        with entry["lock"]:
            out = entry["sess"].ask(question, role=role)
            out.setdefault("session", {})
            out["session"]["session_id"] = sid
            out["session"]["namespace"] = subject
            # 持久化在 per-session 鎖內（十四輪 十：並發寫不再丟回合/
            # 競態 FileNotFoundError）；失敗如實暴露在響應元數據
            try:
                self._persist_turn(subject, sid, question, out)
                out["session"]["persisted"] = True
            except Exception as exc:
                out["session"]["persisted"] = False
                out["session"]["persist_error"] = str(exc)[:120]
        if generated:
            out["session"]["note"] = ("服務端已生成獨立 session_id；"
                                      "續接上下文請在後續請求回傳該 id")
        return out

    def deep_research(self, topic: str, rounds: int = 3) -> Dict:
        from ..agent.research_loop import DeepResearcher
        from .. import safety
        d = DeepResearcher(client=self.llm, registry=self.registry,
                           max_rounds=rounds).run(topic)
        return safety.governed(d, "researcher")

    # -- agent / council -----------------------------------------------
    def agent(self, question: str, role: str = None, max_steps: int = 5) -> Dict:
        from ..agent.agent import ShanghanAgent
        return ShanghanAgent(client=self.llm, registry=self.registry,
                             max_steps=max_steps).ask(question, role=role)

    def council(self, question: str, role: str = None) -> Dict:
        from ..agent.multi_agent import Council
        return Council(client=self.llm, registry=self.registry).deliberate(question, role=role)

    def tool_call(self, name: str, arguments: Dict, role: str = "",
                  subject: str = "") -> Dict:
        # /api/tool 按角色限權：patient 經 ScopedRegistry 硬裁剪工具面。
        # role 已由 http 層 Policy 按服務端身份鉗制（請求體不可提權）；
        # subject 進入審計台賬
        reg = self.registry.for_role(role) if role else self.registry
        out = reg.call(name, arguments or {})
        if subject and isinstance(out, dict):
            out.setdefault("_audit", {})["subject"] = subject
        return out

    def trace(self, query_type: str, ref: str, synthesize: bool = True) -> Dict:
        from ..trace.chains import trace_dispatch
        out = trace_dispatch(query_type, ref)
        # 十六輪：規則檢索之上加模型綜合層（引用過 CitationGuard；
        # local 後端給確定性摘要，同一出口離線可測）
        if synthesize and isinstance(out, dict) and "error" not in out:
            try:
                out["model_synthesis"] = self._trace_synthesis(query_type, out)
            except Exception as exc:
                out["model_synthesis"] = {"backend": "error",
                                          "error": type(exc).__name__}
        return out

    @staticmethod
    def _report_clause_ids(report: Dict) -> List[str]:
        import re
        blob = __import__("json").dumps(report, ensure_ascii=False, default=str)
        ids = re.findall(r"SHL_SONGBEN_(?:AUX_)?\d{4}", blob)
        return sorted(set(ids))

    def _trace_synthesis(self, query_type: str, report: Dict) -> Dict:
        """溯源報告 → 綜述。真模型：撰寫並核驗引用；local：確定性摘要。"""
        from ..agent.citation_guard import CitationGuard
        allowed = self._report_clause_ids(report)
        chain_type = report.get("chain_type", query_type)
        if not self.llm.available:
            bits = []
            clause = report.get("clause") or {}
            if clause.get("clause_id"):
                bits.append(f"本鏈錨定條文 {clause['clause_id']}")
            cit = report.get("citations") or {}
            if cit.get("n_citing_books"):
                bits.append(f"歷代 {cit['n_citing_books']} 部著作存在引用")
            if report.get("commentaries"):
                bits.append(f"{len(report['commentaries'])} 家注家有對齊注文")
            if report.get("variants"):
                bits.append(f"{len(report['variants'])} 部異文本可對勘")
            if report.get("matches"):
                bits.append(f"文本回源命中 {len(report['matches'])} 條")
            text = (f"【確定性摘要】{chain_type}：" + "；".join(bits) + "。"
                    if bits else f"【確定性摘要】{chain_type}：見結構化報告各節。")
            return {"backend": "local", "synthesis": text,
                    "evidence_layer": "D 計量/檢索歸納",
                    "note": "未接真實模型；接入後將生成引用經核驗的溯源綜述。"}
        import json as _json
        from ..llm.prompts import (trace_synth_system_prompt,
                                   trace_synth_user_prompt)
        compact = {k: v for k, v in report.items()
                   if k not in ("model_synthesis",)}
        blob = _json.dumps(compact, ensure_ascii=False, default=str)[:6000]
        text = self.llm.complete(trace_synth_system_prompt(),
                                 trace_synth_user_prompt(chain_type, blob),
                                 task="synthesize")
        guard = CitationGuard(self.art.clause_store())
        rep = guard.check(text, allowed_ids=allowed)
        if not rep.ok:
            text = guard.annotate(text, rep)
        return {"backend": self.llm.backend, "synthesis": text,
                "evidence_layer": "E 模型綜合（事實僅取結構化報告）",
                "citation_report": rep.to_dict()}

    def tools(self) -> Dict:
        from ..integrations.tool_specs import openai_tool_specs
        return {"tools": openai_tool_specs()}

    def gold_sample(self, n: int = 20, stratify: bool = True) -> Dict:
        from ..trace.goldset import build_sample
        return build_sample(n=n, stratify=stratify)   # 不落盤，rows 隨響應返回

    def gold_eval(self, rows) -> Dict:
        from ..trace.goldset import evaluate_rows
        return evaluate_rows(rows or [])

    def herb(self, name: str) -> Dict:
        return self.registry.call("shanghan_herb_profile", {"herb": name})

    def trace_passages(self, book_dir: str, clause_ids: List[str],
                       offset: int = 0, limit: int = 8) -> Dict:
        """歷代引用的段落級點閱（方劑源流/原文溯源的「某書引用」展開）。"""
        from ..trace.passages import book_citing_passages
        return book_citing_passages(book_dir, clause_ids or [],
                                    offset=offset, limit=limit)

    # -- 勘誤提交（十九輪：用戶對原文/轉寫錯誤的反饋閉環）----------------
    def errata_submit(self, clause_ref: str, quote: str, suggestion: str,
                      note: str = "", subject: str = "anonymous") -> Dict:
        """勘誤落盤 data/shanghan/errata/errata.jsonl（不入庫，人工複核）。
        提交不改動語料——語料以 manifest sha256 為版本錨，勘誤經維護者
        審定後才進入下一版底本。"""
        import json as _json
        import time
        import uuid
        clause_ref = str(clause_ref or "").strip()[:40]
        quote = str(quote or "").strip()[:400]
        suggestion = str(suggestion or "").strip()[:400]
        note = str(note or "").strip()[:400]
        if not quote or not suggestion:
            return {"error": "須同時提供原文片段（quote）與勘誤建議（suggestion）"}
        c = self.clause_rag.get_clause(clause_ref) if clause_ref else None
        from ..textutil import contains_verbatim
        rec = {
            "erratum_id": "ERR_" + uuid.uuid4().hex[:10],
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "subject": str(subject)[:60],
            "clause_ref": clause_ref,
            "clause_id": c.clause_id if c else "",
            "quote": quote,
            "suggestion": suggestion,
            "note": note,
            # 逐字核驗：片段是否真在所指條文中（在=可定位；不在=如實標記）
            "quote_found_in_clause": bool(c and contains_verbatim(
                c.clean_text, quote)),
            "status": "pending",
        }
        d = config.SHANGHAN_DIR / "errata"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "errata.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True, "erratum_id": rec["erratum_id"],
                "quote_found_in_clause": rec["quote_found_in_clause"],
                "note": "已登記，待人工複核；勘誤不即時改動語料"
                        "（語料版本以 manifest sha256 為錨）。"
                        + ("" if rec["quote_found_in_clause"] else
                           "提示：所引片段未能在該條文中逐字定位，"
                           "複核時將優先人工核對。")}

    def errata_list(self, limit: int = 50) -> Dict:
        from ..schemas import read_jsonl
        rows = read_jsonl(config.SHANGHAN_DIR / "errata" / "errata.jsonl")
        return {"n_total": len(rows), "errata": rows[-max(1, limit):][::-1],
                "note": "僅維護者（doctor 角色）可查閱；status 由人工複核更新"}

    def library_read(self, book: str, section: str = "", offset: int = 0,
                     max_chars: int = 3000) -> Dict:
        """笈成全庫章節全文點閱（二十輪）：全文命中標題點擊 → 分頁續讀
        該章節（或全書）原文。旁證層，僅供文獻查閱，不進入證據閘門。"""
        from ..corpus import library
        if not library.ensure_available(verbose=False):
            return {"available": False,
                    "error": "全庫未下載：運行 `python3 -m hermes_shanghan "
                             "library fetch`（約 69MB）"}
        lib = library.Library()
        out = lib.read(str(book or "").strip()[:80],
                       section=str(section or "").strip()[:80],
                       max_chars=max(400, min(8000, int(max_chars or 3000))),
                       offset=max(0, int(offset or 0)))
        if "error" in out:
            return {"available": True, **out}
        return {"available": True, **out,
                "evidence_layer": "文獻旁證層（非經文層）：出處僅供文獻查閱，"
                                  "不進入證據閘門"}

    def trace_mentions(self, name: str, book_dir: str,
                       offset: int = 0, limit: int = 6) -> Dict:
        """方名傳播的點閱：某書中該方名的逐字提及段落（十九輪）。"""
        from ..trace.passages import name_mention_passages
        return name_mention_passages(name, book_dir, offset=offset,
                                     limit=limit)

    def term_passages(self, term: str, book: str,
                      offset: int = 0, limit: int = 6) -> Dict:
        """注家使用譜的點閱：某注本中含該術語的注文用例（十九輪）。"""
        from ..textutil import fold_variants, normalize_query
        t = fold_variants(normalize_query(term))
        if len(t) < 2:
            return {"error": "術語至少 2 字"}
        rows = []
        for r in self.art.commentary_rules:
            if book and r.book != book:
                continue
            folded = fold_variants(r.commentary_text)
            pos = folded.find(t)
            if pos < 0:
                continue
            lo, hi = max(0, pos - 40), min(len(r.commentary_text),
                                           pos + len(t) + 40)
            rows.append({"clause_id": r.clause_id,
                         "commentator": r.commentator,
                         "book": r.book, "chapter": r.chapter,
                         "excerpt": ("…" if lo else "")
                         + r.commentary_text[lo:hi]
                         + ("…" if hi < len(r.commentary_text) else "")})
        total = len(rows)
        page = rows[max(0, offset):max(0, offset) + max(1, limit)]
        return {"term": term, "book": book, "n_passages": total,
                "offset": offset, "has_more": offset + len(page) < total,
                "passages": page,
                "evidence_layer": "C 注家解釋（用例逐字檢出，所注條文可回源）"}

    def source_passage(self, book: str, ref: str) -> Dict:
        """讀取語料書的原始段落（十八輪：條文關係目標可點閱）。

        ref 兩種形態：``pN``（第 N 段——注文對齊關係的定位）或章節名
        （異文關係的定位，返回該章節段落）。只讀 corpus_raw 隨庫語料。"""
        import re as _re
        from ..corpus import segmenter
        from ..textutil import fold_variants
        book = str(book or "").strip()[:60]
        ref = str(ref or "").strip()[:80]
        if not book:
            return {"error": "缺少書名"}
        try:
            paragraphs = segmenter.segment_paragraphs(book)
        except FileNotFoundError:
            return {"error": f"語料中無此書：{book}"}
        m = _re.fullmatch(r"p(\d{1,5})", ref)
        if m:
            i = int(m.group(1))
            if not 0 <= i < len(paragraphs):
                return {"error": f"段號越界：{ref}（全書 {len(paragraphs)} 段）"}
            chapter = paragraphs[i][0]
            rows = [{"para_seq": i, "text": paragraphs[i][1]}]
            # 注文常跨 1-2 段：同章節緊隨其後的段落一併給出
            for j in (i + 1, i + 2):
                if j < len(paragraphs) and paragraphs[j][0] == chapter:
                    rows.append({"para_seq": j, "text": paragraphs[j][1]})
            return {"book": book, "chapter": chapter, "ref": ref,
                    "paragraphs": rows,
                    "note": "定位段落＋同章節緊隨段（注文可能跨段）"}
        want = fold_variants(ref)
        rows, total = [], 0
        for seq, (ch, text) in enumerate(paragraphs):
            if want and want not in fold_variants(ch):
                continue
            rows.append({"para_seq": seq, "text": text})
            total += len(text)
            if total > 2600 or len(rows) >= 12:
                break
        if not rows:
            chapters = sorted({ch for ch, _ in paragraphs})
            return {"error": f"《{book}》查無章節：{ref}",
                    "available_chapters": chapters[:20]}
        return {"book": book, "chapter": ref, "ref": ref,
                "paragraphs": rows, "truncated": total > 2600,
                "note": "該章節段落（超長截取前段）"}

    # -- 辨證閉環（十七輪：規則 + 模型雙層，不再全靠規則）----------------
    def intake(self, text: str, use_llm: bool = True) -> Dict:
        """四診採集：確定性詞表抽取為底座；真模型補語義層抽取——模型抽出的
        每個表現都要能在患者敘述（含口語→古籍映射後）中找到依據，找不到
        進 unverified、不併入。"""
        base = self.registry.call("shanghan_intake", {"text": text})
        if not (use_llm and self.llm.available):
            return base
        from ..apps.bianzheng import modernize
        from ..llm.prompts import (intake_extract_system_prompt,
                                   intake_extract_user_prompt)
        from ..textutil import fold_variants, normalize_query
        try:
            out = self.llm.json_complete(intake_extract_system_prompt(),
                                         intake_extract_user_prompt(text),
                                         task="extract_rule")
        except Exception as exc:
            base["model_extraction"] = {"backend": "error",
                                        "error": type(exc).__name__}
            return base
        haystack = fold_variants(modernize(text)) + "\n" + fold_variants(
            normalize_query(text))
        already = {fold_variants(s) for key in
                   ("cold_heat", "sweating", "thirst_drinking", "stool_urine",
                    "chest_hypochondrium", "epigastrium_abdomen",
                    "pain_location", "sleep", "tongue", "other_findings")
                   for s in (base.get(key) or [])}
        added, unverified = [], []
        for t in (out.get("findings") or [])[:16]:
            if not isinstance(t, str) or not t.strip():
                continue
            tf = fold_variants(normalize_query(t))
            if tf in already:
                continue
            # 驗證線：模型詞須逐字在敘述中，或其肯定/否定核在敘述中
            core = tf.lstrip("無不")
            if tf in haystack or (len(core) >= 2 and core in haystack):
                added.append(t)
            else:
                unverified.append(t)
        base["model_extraction"] = {
            "backend": self.llm.backend,
            "added_findings": added,
            "unverified": unverified,
            "model_pulse": [p for p in (out.get("pulse") or [])[:4]
                            if isinstance(p, str)],
            "notes": str(out.get("notes", ""))[:200],
            "note": "模型抽取須逐詞回驗患者敘述（含口語→古籍映射）；"
                    "unverified 中的表現敘述裡找不到依據，未併入四診表。"}
        return base

    def adjudicate(self, symptoms: List[str], pulse: List[str] = None,
                   six_channel: str = "", use_llm: bool = True) -> Dict:
        """多假設裁決：規則三態裁決為底座；真模型作語義級審校（漏診方向/
        裁決穩妥性/關鍵追問），所引條文過 CitationGuard。local 附確定性說明。"""
        base = self.registry.call("shanghan_adjudicate",
                                  {"symptoms": symptoms or [],
                                   "pulse": pulse or [],
                                   "six_channel": six_channel or ""})
        if not use_llm or not isinstance(base, dict) or "error" in base:
            return base
        base["model_review"] = self._adjudicate_review(base)
        return base

    def _adjudicate_review(self, adjudication: Dict) -> Dict:
        from ..agent.citation_guard import CitationGuard
        allowed = self._report_clause_ids(adjudication)
        if not self.llm.available:
            cands = adjudication.get("candidates", [])
            return {"backend": "local",
                    "agrees_with_verdict": True,
                    "assessment": (f"確定性說明：{len(cands)} 個規則候選，"
                                   f"裁決「{adjudication.get('verdict', '')}」"
                                   "由評分差距+反證+禁忌規則得出（D 層詞表"
                                   "匹配）；接入真實模型後將補語義級審校"
                                   "（漏診方向/非典型表述）。"),
                    "missed_patterns": [],
                    "additional_questions": [],
                    "citation_report": None}
        import json as _json
        from ..llm.prompts import (adjudicate_review_system_prompt,
                                   adjudicate_review_user_prompt)
        store = self.art.clause_store()
        rows, seen = [], set()
        for cid in allowed[:14]:
            c = store.get(cid)
            if c is None or cid in seen:
                continue
            seen.add(cid)
            rows.append(f"- [{cid}] {c.clean_text[:200]}")
        compact = {k: adjudication.get(k) for k in
                   ("input", "verdict", "rationale", "why_not_prescribe",
                    "key_questions")}
        compact["candidates"] = [
            {k: h.get(k) for k in ("formula", "support", "against",
                                   "missing_key_findings",
                                   "contraindication_hits")}
            for h in adjudication.get("candidates", [])[:3]]
        out = self.llm.json_complete(
            adjudicate_review_system_prompt(),
            adjudicate_review_user_prompt(
                _json.dumps(compact, ensure_ascii=False, indent=1)[:4000],
                "\n".join(rows)),
            task="critic")
        guard = CitationGuard(store)
        missed = []
        for it in (out.get("missed_patterns") or [])[:6]:
            if not isinstance(it, dict):
                continue
            cids = [c for c in (it.get("clause_ids") or [])
                    if isinstance(c, str)]
            rep = guard.check("、".join(cids), allowed_ids=allowed)
            missed.append({"formula": str(it.get("formula", ""))[:24],
                           "reason": str(it.get("reason", ""))[:200],
                           "clause_ids": rep.verified_ids,
                           "unverified_clause_ids": (rep.unsupported_ids
                                                     + rep.outside_evidence_ids)})
        assessment = str(out.get("assessment", ""))[:500]
        arep = guard.check(assessment, allowed_ids=allowed)
        return {"backend": self.llm.backend,
                "agrees_with_verdict": bool(out.get("agrees_with_verdict",
                                                    True)),
                "assessment": assessment,
                "missed_patterns": missed,
                "additional_questions":
                    [str(q)[:120] for q in
                     (out.get("additional_questions") or [])[:4]],
                "citation_report": arep.to_dict(),
                "note": "模型審校屬 E 層；漏診方向所引 clause_id 已逐一核驗，"
                        "unverified_clause_ids 請勿採信；不構成處方建議。"}

    def formula_explain(self, name: str) -> Dict:
        return self.registry.call("shanghan_formula_explain", {"formula": name})

    # -- 運行中心（十二輪：Harness 控制面進 UI）--------------------------
    def runs_list(self, limit: int = 30) -> Dict:
        from ..agent.harness import list_runs
        return {"runs": list_runs(limit=limit)}

    def run_detail(self, run_id: str) -> Dict:
        """Run 摘要（十四輪 十二：不再全量返回 node_outputs/全部台賬/
        全部 spans——大字段走 /spans /evidence /output/<node> 端點）。"""
        from ..agent.harness.runner import load_run, run_dir
        from ..agent.harness.tracing import TraceStore
        try:
            st = load_run(run_id)
        except Exception:
            return {"run_id": run_id, "status": "corrupt",
                    "error": "state.json 損壞（可修復性見磁盤文件）"}
        if st is None:
            return {"error": f"未找到 run {run_id}", "_status": 404}
        n_ledger = sum(len(v) for v in st.evidence_ledger.values())
        n_spans = len(TraceStore(run_dir(run_id)).read())
        return {
            "spec": st.spec.to_dict(), "status": st.status,
            "trace_id": st.trace_id,
            "nodes": {k: v.to_dict() for k, v in st.nodes.items()},
            "tool_calls": st.tool_calls[-60:],
            "guardrail_events": st.guardrail_events[-30:],
            "errors": st.errors[-10:],
            "final_answer": (st.final_answer or "")[:4000],
            "release": st.release,
            "pending_review": st.pending_review,
            "approval_requests": st.approval_requests,
            "budget_snapshot": st.budget_snapshot,
            "counts": {"evidence_records": n_ledger, "spans": n_spans,
                       "node_outputs": len(st.node_outputs)},
            "links": {"spans": f"/api/runs/{run_id}/spans?offset=0&limit=60",
                      "evidence": f"/api/runs/{run_id}/evidence?limit=100",
                      "output": f"/api/runs/{run_id}/output/<node_id>"},
        }

    def run_node_output(self, run_id: str, node_id: str) -> Dict:
        from ..agent.harness.runner import load_run
        st = load_run(run_id)
        if st is None:
            return {"error": f"未找到 run {run_id}", "_status": 404}
        if node_id not in st.node_outputs:
            return {"error": f"節點 {node_id} 無輸出",
                    "available": list(st.node_outputs), "_status": 404}
        return {"run_id": run_id, "node_id": node_id,
                "output": st.node_outputs[node_id]}

    # 有界執行器（十三輪 九：後台線程→受控任務池；隊列/lease/多 worker
    # 屬 SQLite 路線，見 PLATFORM.md）
    RUN_WORKERS = int(os.environ.get("HERMES_RUN_WORKERS", "2"))
    MAX_QUERY_CHARS = 20_000

    RUN_QUEUE_SIZE = int(os.environ.get("HERMES_RUN_QUEUE", "8"))

    def _run_executor(self):
        if not hasattr(self, "_executor"):
            import threading
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=self.RUN_WORKERS,
                thread_name_prefix="hermes-run")
            # 背壓（十四輪 七）：ThreadPoolExecutor 的內部隊列無界——
            # 用提交信號量限容量（workers+queue），滿載回 429 而非默默排隊
            self._run_slots = threading.BoundedSemaphore(
                self.RUN_WORKERS + self.RUN_QUEUE_SIZE)
        return self._executor

    def close(self) -> None:
        """關閉任務池（十四輪 八：測試 tearDown/serve finally/Notebook
        清理均應調用，避免線程滯留與進程無法退出）。"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False, cancel_futures=True)
            del self._executor

    def run_start(self, query: str, mode: str = "agent",
                  role: str = "researcher", max_steps: int = 6,
                  max_tool_calls: int = 12) -> Dict:
        """創建前校驗（十三輪 十：非法請求 400，不創建注定失敗的任務）→
        queued 狀態**同步落盤**（幽靈 run 根除）→ 提交有界任務池 →
        前端輪詢 run_detail。"""
        from ..agent.harness import HarnessRunner
        from ..agent.harness.state import RUN_MODES
        if not (query or "").strip():
            return {"error": "query 不能為空", "_status": 400}
        if len(query) > self.MAX_QUERY_CHARS:
            return {"error": f"query 超長（>{self.MAX_QUERY_CHARS}）",
                    "_status": 400}
        if mode not in RUN_MODES:
            return {"error": f"未知模式 {mode!r}", "supported": RUN_MODES,
                    "_status": 400}
        max_steps = max(1, min(50, int(max_steps)))
        max_tool_calls = max(0, min(100, int(max_tool_calls)))
        executor = self._run_executor()
        # 背壓在建立 run 目錄**之前**：拒絕時不留幽靈 queued
        if not self._run_slots.acquire(blocking=False):
            return {"error": "任務隊列已滿（workers+queue 容量耗盡），"
                             "請稍後重試", "_status": 429}
        try:
            runner = HarnessRunner()
            state = runner.prepare(query, mode=mode, role=role,
                                   max_steps=max_steps,
                                   max_tool_calls=max_tool_calls)
        except ValueError as exc:
            self._run_slots.release()
            return {"error": str(exc), "_status": 400}
        except Exception:
            self._run_slots.release()
            raise
        rid = state.spec.run_id           # 此刻 state.json 已持久化（queued）

        def _work():
            import threading
            import time as _time
            import traceback
            try:
                runner.execute_prepared(rid)
            except Exception as exc:
                # 十四輪 六：worker 崩潰必須落盤——不留永久 queued 幽靈
                traceback.print_exc()
                try:
                    from ..agent.harness.runner import load_run, save_state
                    from ..agent.harness.tracing import sanitize_error
                    st = load_run(rid)
                    if st is not None and st.status in ("queued", "created",
                                                        "running"):
                        st.status = "failed"
                        st.errors.append(sanitize_error(exc))
                        st.guardrail_events.append(
                            {"event": "worker_crash",
                             "worker_id": threading.current_thread().name,
                             "at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "error": sanitize_error(exc)})
                        st.release = {"decision": "failed_closed",
                                      "reasons": ["worker 異常，運行未完成"]}
                        save_state(st)
                except Exception:
                    traceback.print_exc()
            finally:
                self._run_slots.release()

        executor.submit(_work)
        return {"run_id": rid, "status": "queued",
                "hint": "輪詢 GET /api/runs/<run_id> 查看節點軌跡與發布裁定"}

    def run_action(self, run_id: str, action: str, approver: str = "",
                   reason: str = "", trigger: str = "") -> Dict:
        self._approve_trigger = trigger
        from ..agent.harness import HarnessRunner
        from ..agent.harness.runner import export_run
        if action == "approve":
            st = HarnessRunner().resume(run_id, approve=True,
                                        approver=approver or "console",
                                        reason=reason,
                                        trigger=getattr(self, "_approve_trigger",
                                                        ""))
        elif action == "reject":
            st = HarnessRunner().resume(run_id, reject=True,
                                        approver=approver or "console",
                                        reason=reason)
        elif action == "resume":
            st = HarnessRunner().resume(run_id)
        elif action == "cancel":
            ok, why = HarnessRunner.request_cancel(run_id)
            if ok:
                return {"run_id": run_id, "cancel_requested": True,
                        "note": "協作式取消：節點邊界生效（節點內工具只讀原子）"}
            if why == "not_found":
                return {"error": f"未找到 run {run_id}", "_status": 404}
            return {"run_id": run_id, "cancel_requested": False,
                    "reason": why, "_status": 409}
        elif action == "replay":
            out = HarnessRunner().replay(run_id)
            return out or {"error": f"未找到 run {run_id}"}
        elif action == "export":
            md = export_run(run_id, "md")
            return {"run_id": run_id, "markdown": md} if md else \
                {"error": f"未找到 run {run_id}"}
        else:
            return {"error": f"未知動作 {action}",
                    "supported": ["approve", "reject", "resume", "cancel",
                                  "replay", "export"]}
        if st is None:
            return {"error": f"未找到 run {run_id}"}
        return {"run_id": run_id, "status": st.status,
                "release": st.release, "pending_review": st.pending_review}

    def run_spans(self, run_id: str, offset: int = 0,
                  limit: int = 60) -> Dict:
        """span 分頁讀取（十三輪 十二：大運行詳情不可一次性全量返回）。"""
        from ..agent.harness.runner import run_dir
        from ..agent.harness.tracing import TraceStore
        events = TraceStore(run_dir(run_id)).read()
        offset = max(0, int(offset)); limit = max(1, min(200, int(limit)))
        return {"run_id": run_id, "total": len(events),
                "offset": offset, "limit": limit,
                "spans": events[offset:offset + limit]}

    def run_evidence(self, run_id: str, offset: int = 0,
                     limit: int = 100) -> Dict:
        from ..agent.harness.runner import load_run
        st = load_run(run_id)
        if st is None:
            return {"error": f"未找到 run {run_id}"}
        recs = [dict(r, node=n) for n, v in st.evidence_ledger.items()
                for r in v]
        offset = max(0, int(offset)); limit = max(1, min(400, int(limit)))
        return {"run_id": run_id, "total": len(recs),
                "offset": offset, "limit": limit,
                "records": recs[offset:offset + limit]}

    # -- 評測（十二輪：評測運行進 UI）------------------------------------
    def eval_trajectory(self) -> Dict:
        from ..eval.trajectory import trajectory_eval
        return trajectory_eval()

    def eval_perturbation(self) -> Dict:
        from ..eval.trajectory import perturbation_eval
        return perturbation_eval()

    # -- Artifact（十二輪：論文/運行導出下載，防路徑穿越）----------------
    def artifacts(self) -> Dict:
        out = []
        for base, kind in ((config.PAPER_DIR, "paper"),
                           (config.RUNS_DIR, "run")):
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix in (".md", ".json", ".csv",
                                                ".jsonl", ".svg"):
                    out.append({"kind": kind,
                                "path": str(p.relative_to(config.SHANGHAN_DIR)),
                                "bytes": p.stat().st_size})
        return {"artifacts": out[:200],
                "note": "下載走 /api/artifact?path=…（僅限 papers/ 與 runs/，"
                        "路徑穿越一律拒絕）"}

    def _artifact_target(self, rel_path: str):
        base = config.SHANGHAN_DIR.resolve()
        target = (base / (rel_path or "")).resolve()
        allowed = (base / "papers", base / "runs")
        if not any(str(target).startswith(str(a.resolve()) + os.sep)
                   or target == a.resolve() for a in allowed) \
                or not target.is_file():
            return None
        return target

    def artifact_read(self, rel_path: str) -> Dict:
        target = self._artifact_target(rel_path)
        if target is None:
            return {"error": "路徑不合法或文件不存在（僅限 papers/ 與 runs/）"}
        if target.stat().st_size > 1_500_000:
            return {"error": "文件超過下載上限 1.5MB，請用倉庫/磁盤方式獲取"}
        return {"path": rel_path,
                "content": target.read_text(encoding="utf-8",
                                            errors="replace")}

    def artifact_meta(self, rel_path: str) -> Dict:
        """Artifact 元數據（十三輪 十三）：哈希/大小/MIME/語料指紋。"""
        import hashlib
        import mimetypes
        target = self._artifact_target(rel_path)
        if target is None:
            return {"error": "路徑不合法或文件不存在（僅限 papers/ 與 runs/）"}
        meta = {"path": rel_path, "filename": target.name,
                "bytes": target.stat().st_size,
                "mime_type": mimetypes.guess_type(target.name)[0]
                or "text/plain",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        # 生成時指紋（十四輪 十七）：runs/ 下的 Artifact 從其 run state 讀
        # 創建時語料/代碼指紋；讀不到時如實標 current（不冒充生成時值）
        frozen = None
        parts = rel_path.replace("\\", "/").split("/")
        if parts and parts[0] == "runs" and len(parts) > 1:
            try:
                from ..agent.harness.runner import load_run
                st = load_run(parts[1])
                if st is not None:
                    frozen = {"corpus_fingerprint_at_creation":
                              st.spec.corpus_version,
                              "code_fingerprint_at_creation":
                              st.spec.code_fingerprint,
                              "created_by_run": parts[1],
                              "created_at": st.spec.created_at}
            except Exception:
                frozen = None
        if frozen:
            meta.update(frozen)
        else:
            from ..agent.harness.state import spec_versions
            meta["corpus_fingerprint_current"] = \
                spec_versions()["corpus_version"]
            meta["provenance_note"] = ("無生成時記錄——此為**當前**語料指紋，"
                                       "不代表生成時版本")
        return meta

    def artifact_download(self, rel_path: str):
        """返回 (filename, mime, bytes)——http 層以 Content-Disposition:
        attachment 下發；None = 不合法。"""
        import mimetypes
        target = self._artifact_target(rel_path)
        if target is None or target.stat().st_size > 8_000_000:
            return None
        return (target.name,
                mimetypes.guess_type(target.name)[0]
                or "application/octet-stream", target.read_bytes())

    # -- 會話持久化（十三輪 十五：刷新不丟、可列可刪可複核逐輪解析）------
    def _session_file(self, subject: str, sid: str):
        # 十四輪 十一：哈希文件名——長 subject 截斷不會使不同 session
        # 碰撞同一文件；可讀元數據在文件內容（namespace/session_id）
        import hashlib
        safe = hashlib.sha256(f"{subject}\0{sid}".encode()).hexdigest()[:32]
        d = config.SHANGHAN_DIR / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.json"

    def _persist_turn(self, subject: str, sid: str, question: str,
                      out: Dict) -> None:
        import json
        import time
        p = self._session_file(subject, sid)
        doc = {"session_id": sid, "namespace": subject, "turns": []}
        if p.exists():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        s = out.get("session", {})
        doc["turns"].append({
            "turn_id": len(doc["turns"]) + 1,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_message": question[:2000],
            "reference_resolution": s.get("reference_resolution"),
            "anchors": s.get("anchors", []),
            "answer": (out.get("answer") or out.get("message", ""))[:2000],
            "evidence_ids": out.get("evidence_clause_ids", [])[:12],
        })
        doc["turns"] = doc["turns"][-50:]
        import uuid
        tmp = p.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")   # 唯一 tmp 防競態
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(p)

    def _restore_session(self, sess, subject: str, sid: str) -> None:
        import json
        p = self._session_file(subject, sid)
        if not p.exists():
            return
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("namespace") != subject:
            return
        for t in doc.get("turns", [])[-8:]:
            q = t.get("user_message", "")
            sess._record_corrections(q)
            sess._remember(q, {"answer": t.get("answer", ""),
                               "evidence_clause_ids":
                               t.get("evidence_ids", [])})
        sess.restored_turns = len(doc.get("turns", []))

    def sessions_list(self, subject: str) -> Dict:
        import json
        d = config.SHANGHAN_DIR / "sessions"
        out = []
        if d.exists():
            for p in sorted(d.glob("*.json"), reverse=True):
                try:
                    doc = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if doc.get("namespace") != subject:
                    continue        # 命名空間隔離：只列本主體的會話
                turns = doc.get("turns", [])
                out.append({"session_id": doc.get("session_id"),
                            "n_turns": len(turns),
                            "last_at": turns[-1]["at"] if turns else "",
                            "preview": (turns[-1]["user_message"][:40]
                                        if turns else "")})
        return {"sessions": out[:50]}

    def session_turns(self, subject: str, sid: str) -> Dict:
        import json
        p = self._session_file(subject, sid)
        if not p.exists():
            return {"error": f"未找到會話 {sid}"}
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("namespace") != subject:
            return {"error": "會話不屬於當前主體"}
        return doc

    def session_delete(self, subject: str, sid: str) -> Dict:
        p = self._session_file(subject, sid)
        if not p.exists():
            return {"error": f"未找到會話 {sid}"}
        p.unlink()
        if hasattr(self, "_sessions"):
            self._sessions.pop(f"{subject}:{sid}", None)
        return {"deleted": sid}

    # -- 治理面板 ---------------------------------------------------------
    def governance(self) -> Dict:
        from .._version import __version__
        from ..agent.harness.release_gate import ROLE_RELEASE_POLICY
        from ..health import readyz
        return {"version": __version__,
                "readyz": readyz(),
                "role_release_policy": ROLE_RELEASE_POLICY,
                "tool_audit_tail": self.registry.audit_tail(30),
                "note": "角色上限由服務端身份綁定（HERMES_API_KEYS）；"
                        "前端角色選擇只是請求，不是權限"}


_SERVICE: Optional[ServiceContext] = None


def get_service() -> ServiceContext:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ServiceContext()
    return _SERVICE
