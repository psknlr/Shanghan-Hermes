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
        return safety.governed(payload, role)

    # -- apps -----------------------------------------------------------
    def match(self, symptoms: List[str], pulse: List[str] = None,
              six_channel: str = None, top_k: int = 5) -> Dict:
        return self.matcher.match(symptoms=symptoms, pulse=pulse or [],
                                  six_channel=six_channel or None, top_k=top_k)

    def differential(self, formulas: List[str]) -> Dict:
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
        return safety.governed({"differential": cands[0].to_dict()}, "doctor")

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
                entry = {"sess": AgentSession(client=self.llm,
                                              registry=self.registry,
                                              namespace=subject),
                         "last": time.time(),
                         "lock": threading.Lock()}
                self._sessions[key] = entry
            entry["last"] = time.time()
        with entry["lock"]:
            out = entry["sess"].ask(question, role=role)
        out.setdefault("session", {})
        out["session"]["session_id"] = sid
        out["session"]["namespace"] = subject
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

    def trace(self, query_type: str, ref: str) -> Dict:
        from ..trace.chains import trace_dispatch
        return trace_dispatch(query_type, ref)

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
        from ..agent.harness.runner import load_run, run_dir
        from ..agent.harness.tracing import TraceStore
        st = load_run(run_id)
        if st is None:
            return {"error": f"未找到 run {run_id}"}
        d = st.to_dict()
        events = TraceStore(run_dir(run_id)).read()
        d["spans"] = [{k: e.get(k) for k in
                       ("span_id", "parent_span_id", "span_type", "name",
                        "duration_ms", "error", "evidence_ids", "metadata")}
                      for e in events][-120:]
        return d

    def run_start(self, query: str, mode: str = "agent",
                  role: str = "researcher", max_steps: int = 6,
                  max_tool_calls: int = 12) -> Dict:
        """異步啟動：先返回 run_id，後台線程執行；前端輪詢 run_detail。"""
        import threading
        from ..agent.harness import HarnessRunner
        from ..agent.harness.state import new_run_id
        if not (query or "").strip():
            return {"error": "query 不能為空"}
        rid = new_run_id(query)

        def _work():
            try:
                HarnessRunner().start(query, mode=mode, role=role,
                                      max_steps=max_steps,
                                      max_tool_calls=max_tool_calls,
                                      run_id=rid)
            except Exception:
                import traceback
                traceback.print_exc()

        threading.Thread(target=_work, daemon=True).start()
        return {"run_id": rid, "status": "started",
                "hint": "輪詢 GET /api/runs/<run_id> 查看節點軌跡與發布裁定"}

    def run_action(self, run_id: str, action: str,
                   approver: str = "") -> Dict:
        from ..agent.harness import HarnessRunner
        from ..agent.harness.runner import export_run
        if action == "approve":
            st = HarnessRunner().resume(run_id, approve=True,
                                        approver=approver or "console")
        elif action == "reject":
            st = HarnessRunner().resume(run_id, reject=True,
                                        approver=approver or "console")
        elif action == "resume":
            st = HarnessRunner().resume(run_id)
        elif action == "replay":
            out = HarnessRunner().replay(run_id)
            return out or {"error": f"未找到 run {run_id}"}
        elif action == "export":
            md = export_run(run_id, "md")
            return {"run_id": run_id, "markdown": md} if md else \
                {"error": f"未找到 run {run_id}"}
        else:
            return {"error": f"未知動作 {action}",
                    "supported": ["approve", "reject", "resume", "replay",
                                  "export"]}
        if st is None:
            return {"error": f"未找到 run {run_id}"}
        return {"run_id": run_id, "status": st.status,
                "release": st.release, "pending_review": st.pending_review}

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

    def artifact_read(self, rel_path: str) -> Dict:
        base = config.SHANGHAN_DIR.resolve()
        target = (base / (rel_path or "")).resolve()
        allowed = (base / "papers", base / "runs")
        if not any(str(target).startswith(str(a.resolve()) + os.sep)
                   or target == a.resolve() for a in allowed) \
                or not target.is_file():
            return {"error": "路徑不合法或文件不存在（僅限 papers/ 與 runs/）"}
        if target.stat().st_size > 1_500_000:
            return {"error": "文件超過下載上限 1.5MB，請用倉庫/磁盤方式獲取"}
        return {"path": rel_path,
                "content": target.read_text(encoding="utf-8",
                                            errors="replace")}

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
