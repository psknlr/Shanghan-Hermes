"""ComplexAgent — compound-question orchestration（任務分解 → 子代理派遣）.

A compound question（「桂枝湯與麻黃湯如何鑒別？各自劑量比是多少？注家對
第12條有何分歧？」）is decomposed into typed subtasks; each subtask runs a
ShanghanAgent whose ToolRegistry is SCOPED to that task type (least
privilege), with guard-driven reflection on; a research-scale subtask may
dispatch the DeepResearcher loop instead. The synthesizer merges subtask
answers and the merged answer passes the CitationGuard once more.

Decomposition itself follows the trusted-base/augmentation split: a real
model decomposes via JSON planning; the local backend decomposes
deterministically (sentence split + task-type classification), so the whole
orchestration tree runs and tests offline through the same code path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import safety
from ..llm.client import LLMClient, get_client
from ..llm.prompts import EVIDENCE_CONTRACT
from ..textutil import normalize_query
from .agent import ShanghanAgent
from .citation_guard import CitationGuard
from .planner import Planner, execution_order
from .tools import ScopedRegistry, ToolRegistry, get_registry

# subtask type → (description for the planner, allowed tool scope)
TASK_TYPES: Dict[str, Dict] = {
    "differential": {"desc": "方證鑒別/對比",
                     "tools": ["shanghan_differential", "shanghan_formula_rule",
                               "shanghan_search"]},
    "dose": {"desc": "劑量/藥量/折算",
             "tools": ["shanghan_dose", "shanghan_dose_convert",
                       "shanghan_formula_rule"]},
    "commentary": {"desc": "注家/注本/詮釋分歧/異文",
                   "tools": ["shanghan_divergence_atlas", "shanghan_variants",
                             "shanghan_search", "shanghan_get_clause"]},
    "mistreatment": {"desc": "誤治/傳變/壞病",
                     "tools": ["shanghan_mistreatment", "shanghan_relations",
                               "shanghan_search"]},
    "therapy": {"desc": "治法法度（汗吐下和溫補/禁例）",
                "tools": ["shanghan_therapy", "shanghan_search"]},
    "safety_check": {"desc": "用方禁忌檢查",
                     "tools": ["shanghan_contraindication_check",
                               "shanghan_formula_rule"]},
    "case": {"desc": "歷史醫案旁證",
             "tools": ["shanghan_case_search", "shanghan_formula_rule"]},
    "six_channel": {"desc": "六經/提綱/經證",
                    "tools": ["shanghan_six_channel", "shanghan_search"]},
    "match": {"desc": "據症狀脈象選方",
              "tools": ["shanghan_match_formula", "shanghan_formula_rule",
                        "shanghan_contraindication_check", "shanghan_search"]},
    "stats": {"desc": "全庫統計/頻次/評測指標",
              "tools": ["shanghan_corpus_stats", "shanghan_eval_metrics"]},
    "research": {"desc": "跨維度學術溯源（派遣深度研究循環）", "tools": []},
    "literature": {"desc": "全庫文獻查閱（中醫笈成 800+ 部，旁證層）",
                   "tools": ["shanghan_library", "shanghan_search"]},
    "general": {"desc": "一般查證",
                "tools": ["shanghan_search", "shanghan_get_clause",
                          "shanghan_formula_rule"]},
}

# segment classification patterns and split regex live in planner.py now —
# the Planner is the single decomposition brain for this orchestrator


class ComplexAgent:
    def __init__(self, client: Optional[LLMClient] = None,
                 registry: Optional[ToolRegistry] = None,
                 max_subtasks: int = 5, subagent_steps: int = 4):
        self.client = client or get_client()
        # 依賴注入原則：harness 下傳入 TracedRegistry（span/台賬/預算），
        # 子代理與 ScopedRegistry 一律基於它派生，不得自行 get_registry()
        self.registry = registry or get_registry()
        self.max_subtasks = max_subtasks
        self.subagent_steps = subagent_steps

    # ------------------------------------------------------------------
    def solve(self, question: str, role: Optional[str] = None) -> Dict[str, Any]:
        role = role if role in safety.ROLES else "doctor"
        if role == "patient":
            triage = safety.red_flag_triage(question)
            if triage:
                return safety.governed(triage, "patient")
            guard = safety.patient_intent_guard(question)
            if guard:
                return safety.governed(guard, "patient")

        # 配置即契約：max_subtasks 不再被 max(...,5) 靜默覆蓋（九輪 P1）
        plan = Planner(client=self.client, task_types=TASK_TYPES,
                       max_subtasks=self.max_subtasks).plan(question)
        ordered = execution_order(plan["subtasks"])
        trace: List[Dict] = [{"step": "decompose",
                              "planner": plan["planner"], "goal": plan["goal"],
                              "success_criteria": plan["success_criteria"],
                              "subtasks": [{"id": t["id"], "kind": t["kind"],
                                            "question": t["question"],
                                            "depends_on": t["depends_on"]}
                                           for t in ordered]}]
        results: List[Dict] = []
        by_id: Dict[str, Dict] = {}
        for t in ordered:
            sub_q = t["question"]
            deps = [by_id[d] for d in t.get("depends_on", []) if d in by_id]
            if deps:
                # a dependent task sees its dependencies' verified evidence —
                # the join task reasons over gathered proof, not from scratch
                ctx = "\n".join(
                    f"（已取證 {d['id']}：{self._strip_footer(d['answer'])[:200]}"
                    f"｜證據：{'、'.join(d.get('evidence_clause_ids', [])[:4])}）"
                    for d in deps)
                sub_q = ctx + "\n綜合任務：" + t["question"]
            r = self._dispatch({"kind": t["kind"], "question": sub_q}, role, trace)
            r["id"] = t["id"]
            r["depends_on"] = t.get("depends_on", [])
            r["planned_question"] = t["question"]
            by_id[t["id"]] = r
            results.append(r)

        final = self._synthesize(question, role, results)
        # strict round grounding for the merged answer: only clause_ids that
        # some subtask actually retrieved are legal citations here
        allowed = sorted({cid for r in results
                          for cid in r.get("evidence_clause_ids", [])})
        guard = CitationGuard(self.registry.art.clause_store())
        report = guard.check(final, allowed_ids=allowed if allowed else None)
        criteria = self._criteria_check(plan, final)
        if criteria["unmet"]:
            final += ("\n\n⚠️ 覆蓋提示：本回答尚未明確覆蓋——"
                      + "；".join(criteria["unmet"]))
        trace.append({"step": "criteria_check", **criteria})
        final = guard.annotate(final, report)
        payload = {
            "question": question, "role": role,
            "backend": self.client.backend,
            "answer": final,
            "plan": {"goal": plan["goal"], "planner": plan["planner"],
                     "success_criteria": plan["success_criteria"]},
            "subtasks": [{"id": r.get("id"), "kind": r["kind"],
                          "question": r["question"],
                          "depends_on": r.get("depends_on", []),
                          "tools_used": r.get("tools_used", []),
                          "evidence_clause_ids": r.get("evidence_clause_ids", []),
                          "reflection_rounds": r.get("reflection_rounds", 0)}
                         for r in results],
            "criteria_check": criteria,
            "evidence_clause_ids": report.verified_ids,
            "citation_report": report.to_dict(),
            "orchestrator_trace": trace,
        }
        return safety.governed(payload, role)

    # ------------------------------------------------------------------
    @staticmethod
    def _criteria_check(plan: Dict, final: str) -> Dict:
        """Deterministic success-criteria audit: coverage criteria
        （「必須分別覆蓋：X、Y」）are checked against the merged answer;
        citation criteria are enforced by the CitationGuard itself."""
        unmet: List[str] = []
        norm = normalize_query(final)
        for c in plan.get("success_criteria", []):
            if "必須分別覆蓋：" in c:
                for obj in c.split("：", 1)[1].split("、"):
                    key = obj.strip()
                    # 「少陰病寒化」 counts as covered if 寒化 appears
                    stem = key[-2:] if len(key) >= 2 else key
                    if key and key not in norm and stem not in norm:
                        unmet.append(key)
        return {"criteria": plan.get("success_criteria", []), "unmet": unmet}

    # ------------------------------------------------------------------
    def _decompose_local(self, question: str) -> List[Dict]:
        """Deterministic decomposition（保留舊接口：返回 {kind, question}）."""
        plan = Planner(task_types=TASK_TYPES,
                       max_subtasks=self.max_subtasks)._plan_local(question)
        return [{"kind": t["kind"], "question": t["question"]}
                for t in plan["subtasks"]]

    # ------------------------------------------------------------------
    def _dispatch(self, task: Dict, role: str, trace: List[Dict]) -> Dict:
        kind = task["kind"]
        if kind == "research":
            from .research_loop import DeepResearcher
            # role isolation holds for research dispatch too: a patient-role
            # research loop only sees the patient-safe tool surface
            dossier = DeepResearcher(client=self.client,
                                     registry=self.registry.for_role(role)
                                     ).run(task["question"])
            trace.append({"step": "subagent", "kind": kind,
                          "dispatch": "deep_research",
                          "rounds": dossier["n_rounds"]})
            summary = "\n".join(f"- [{f['dimension']}] {f['summary']}"
                                for f in dossier["findings"])
            return {"kind": kind, "question": task["question"],
                    "answer": summary,
                    "tools_used": ["deep_research"],
                    "evidence_clause_ids": dossier["evidence_clause_ids"]}
        scope = ScopedRegistry(self.registry, TASK_TYPES[kind]["tools"])
        agent = ShanghanAgent(client=self.client, registry=scope,
                              max_steps=self.subagent_steps)
        out = agent.ask(task["question"], role=role)
        trace.append({"step": "subagent", "kind": kind,
                      "tool_scope": scope.names(),
                      "tools_used": out.get("tools_used", []),
                      "reflection_rounds": out.get("reflection_rounds", 0)})
        return {"kind": kind, "question": task["question"],
                "answer": out.get("answer", ""),
                "tools_used": out.get("tools_used", []),
                "evidence_clause_ids": out.get("evidence_clause_ids", []),
                "reflection_rounds": out.get("reflection_rounds", 0)}

    # ------------------------------------------------------------------
    @staticmethod
    def _strip_footer(answer: str) -> str:
        return answer.split("—" * 12)[0].rstrip()

    def _synthesize(self, question: str, role: str, results: List[Dict]) -> str:
        if self.client.available:
            try:
                block = "\n\n".join(
                    f"【子任務：{r['question']}】\n{self._strip_footer(r['answer'])}"
                    for r in results)
                text = self.client.complete(
                    EVIDENCE_CONTRACT + "\n\n任務：把各子任務的已核驗回答綜合為"
                    "一個連貫答覆。只可使用子任務回答中的事實與 clause_id，"
                    "不得新增結論。",
                    f"原始問題：{question}\n\n{block}", task="synthesize").strip()
                if text:
                    return text
            except Exception:
                pass
        parts = [f"（複合問題編排 · {self.client.backend} 後端 · "
                 f"{len(results)} 個子任務）"]
        for i, r in enumerate(results, 1):
            parts.append(f"\n■ 子任務{i}（{TASK_TYPES[r['kind']]['desc']}）："
                         f"{r['question']}\n{self._strip_footer(r['answer'])}")
        return "\n".join(parts)
