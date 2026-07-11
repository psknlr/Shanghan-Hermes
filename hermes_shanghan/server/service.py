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

    def mistreatment(self, query: str = None) -> Dict:
        return self.registry.call("shanghan_mistreatment", {"query": query or ""})

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
        return {"manuscript_path": str(path),
                "manuscript": path.read_text(encoding="utf-8"),
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
