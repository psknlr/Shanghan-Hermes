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

import json
import re
from typing import Any, Dict, List, Optional

from .. import lexicon, safety
from ..llm.client import LLMClient, get_client
from ..llm.prompts import EVIDENCE_CONTRACT
from ..textutil import normalize_query
from .agent import ShanghanAgent
from .citation_guard import CitationGuard
from .tools import ScopedRegistry, ToolRegistry, get_registry

# subtask type → (description for the planner, allowed tool scope)
TASK_TYPES: Dict[str, Dict] = {
    "differential": {"desc": "方證鑒別/對比",
                     "tools": ["shanghan_differential", "shanghan_formula_rule",
                               "shanghan_search"]},
    "dose": {"desc": "劑量/藥量/折算",
             "tools": ["shanghan_dose", "shanghan_formula_rule"]},
    "commentary": {"desc": "注家/注本/詮釋分歧",
                   "tools": ["shanghan_divergence_atlas", "shanghan_search",
                             "shanghan_get_clause"]},
    "mistreatment": {"desc": "誤治/傳變/壞病",
                     "tools": ["shanghan_mistreatment", "shanghan_search"]},
    "six_channel": {"desc": "六經/提綱/經證",
                    "tools": ["shanghan_six_channel", "shanghan_search"]},
    "match": {"desc": "據症狀脈象選方",
              "tools": ["shanghan_match_formula", "shanghan_formula_rule",
                        "shanghan_search"]},
    "stats": {"desc": "全庫統計/頻次/評測指標",
              "tools": ["shanghan_corpus_stats", "shanghan_eval_metrics"]},
    "research": {"desc": "跨維度學術溯源（派遣深度研究循環）", "tools": []},
    "general": {"desc": "一般查證",
                "tools": ["shanghan_search", "shanghan_get_clause",
                          "shanghan_formula_rule"]},
}

_KIND_PATTERNS = [
    ("differential", r"鑒別|對比|區別|區分|不同|vs"),
    ("dose", r"劑量|藥量|用量|折算|幾兩|幾克|銖"),
    ("commentary", r"注家|注本|詮釋|分歧|成無己|柯琴|尤怡|方有執"),
    ("mistreatment", r"誤治|誤下|誤汗|誤吐|傳變|壞病|變證|救逆"),
    ("six_channel", r"提綱|六經|經證|欲解時"),
    ("stats", r"統計|頻次|多少條|基準|評測|接地率"),
    ("research", r"溯源|源流|演化史|全面研究|綜述"),
]
RE_SPLIT = re.compile(r"[？?；;。]\s*")


class ComplexAgent:
    def __init__(self, client: Optional[LLMClient] = None,
                 registry: Optional[ToolRegistry] = None,
                 max_subtasks: int = 4, subagent_steps: int = 4):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        self.max_subtasks = max_subtasks
        self.subagent_steps = subagent_steps

    # ------------------------------------------------------------------
    def solve(self, question: str, role: Optional[str] = None) -> Dict[str, Any]:
        role = role if role in safety.ROLES else "doctor"
        if role == "patient":
            guard = safety.patient_intent_guard(question)
            if guard:
                return safety.governed(guard, "patient")

        subtasks = self._decompose(question)
        trace: List[Dict] = [{"step": "decompose",
                              "subtasks": [{"kind": t["kind"], "question": t["question"]}
                                           for t in subtasks]}]
        results: List[Dict] = []
        for t in subtasks:
            results.append(self._dispatch(t, role, trace))

        final = self._synthesize(question, role, results)
        guard = CitationGuard(self.registry.art.clause_store())
        report = guard.check(final)
        final = guard.annotate(final, report)
        payload = {
            "question": question, "role": role,
            "backend": self.client.backend,
            "answer": final,
            "subtasks": [{"kind": r["kind"], "question": r["question"],
                          "tools_used": r.get("tools_used", []),
                          "evidence_clause_ids": r.get("evidence_clause_ids", []),
                          "reflection_rounds": r.get("reflection_rounds", 0)}
                         for r in results],
            "evidence_clause_ids": report.verified_ids,
            "citation_report": report.to_dict(),
            "orchestrator_trace": trace,
        }
        return safety.governed(payload, role)

    # ------------------------------------------------------------------
    def _decompose(self, question: str) -> List[Dict]:
        if self.client.available:
            catalog = "\n".join(f"- {k}：{v['desc']}" for k, v in TASK_TYPES.items())
            try:
                plan = self.client.json_complete(
                    EVIDENCE_CONTRACT + "\n\n任務：把用戶的複合問題分解為可獨立"
                    "查證的子任務（1-4 個），每個標注類型。嚴格輸出 JSON："
                    "{\"subtasks\":[{\"kind\":\"…\",\"question\":\"…\"}]}",
                    f"複合問題：{question}\n可用類型：\n{catalog}",
                    task="synthesize")
                tasks = [t for t in plan.get("subtasks", [])
                         if t.get("kind") in TASK_TYPES and t.get("question")]
                if tasks:
                    return tasks[:self.max_subtasks]
            except Exception:
                pass
        return self._decompose_local(question)

    def _decompose_local(self, question: str) -> List[Dict]:
        q = normalize_query(question)
        anchors = [n for n in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True)
                   if n in q][:3]
        segments = [s.strip() for s in RE_SPLIT.split(question) if s.strip()]
        tasks: List[Dict] = []
        for seg in segments or [question]:
            kind = next((k for k, pat in _KIND_PATTERNS
                         if re.search(pat, normalize_query(seg))), "general")
            sub_q = seg
            # a fragment like「各自劑量比是多少」loses its subjects in the
            # split — re-anchor it with the formulas named in the full question
            if anchors and not any(a in normalize_query(seg) for a in anchors):
                sub_q = f"{seg}（涉及：{'、'.join(anchors)}）"
            tasks.append({"kind": kind, "question": sub_q})
        # merge consecutive duplicates of the same kind
        merged: List[Dict] = []
        for t in tasks:
            if merged and merged[-1]["kind"] == t["kind"]:
                merged[-1]["question"] += "；" + t["question"]
            else:
                merged.append(t)
        return merged[:self.max_subtasks]

    # ------------------------------------------------------------------
    def _dispatch(self, task: Dict, role: str, trace: List[Dict]) -> Dict:
        kind = task["kind"]
        if kind == "research":
            from .research_loop import DeepResearcher
            dossier = DeepResearcher(client=self.client,
                                     registry=self.registry).run(task["question"])
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
