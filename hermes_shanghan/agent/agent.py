"""ShanghanAgent — provider-agnostic tool-calling agent.

Loop: system(role) → user(question) → [tool_call → tool_result]* → answer.
The same loop runs on a real model (litellm) or the deterministic `local`
brain. Before returning, every answer passes the citation guard and the
role-aware safety governor. Patient questions hit the intent guard first and
never reach a model that could prescribe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import safety
from ..llm.client import LLMClient, get_client
from ..llm.prompts import agent_system_prompt
from .citation_guard import CitationGuard
from .tools import ToolRegistry, get_registry


@dataclass
class AgentTrace:
    steps: List[Dict] = field(default_factory=list)

    def add(self, kind: str, **data):
        self.steps.append({"step": len(self.steps) + 1, "kind": kind, **data})


class ShanghanAgent:
    def __init__(self, client: Optional[LLMClient] = None,
                 registry: Optional[ToolRegistry] = None, max_steps: int = 5,
                 max_repair_rounds: int = 1):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        self.max_steps = max_steps
        # reflection: when the citation guard rejects an answer, feed the
        # verdict back and let the model retry (guard as controller, not
        # just annotator)
        self.max_repair_rounds = max_repair_rounds

    def _infer_role(self, question: str, role: Optional[str]) -> str:
        if role in safety.ROLES:
            return role
        from ..rag.skill_rag import SkillRAG
        try:
            return SkillRAG().infer_role(question, role)
        except Exception:
            # conservative: prescription/dosage/diagnosis intent → patient
            if (safety.RE_PRESCRIPTION_INTENT.search(question)
                    or safety.RE_DOSAGE_INTENT.search(question)
                    or safety.RE_DIAGNOSIS_INTENT.search(question)):
                return "patient"
            return "doctor"

    def ask(self, question: str, role: Optional[str] = None) -> Dict[str, Any]:
        role = self._infer_role(question, role)
        trace = AgentTrace()

        # patient safety: intent guard BEFORE any model/tool call
        if role == "patient":
            guard = safety.patient_intent_guard(question)
            if guard:
                trace.add("safety_block", intents=guard["refused_intents"])
                out = safety.governed(guard, "patient")
                out["agent_trace"] = trace.steps
                out["backend"] = self.client.backend
                return out

        messages: List[Dict] = [
            {"role": "system", "content": agent_system_prompt(role)},
            {"role": "user", "content": question},
        ]
        tool_results: List[Dict] = []
        final = self._react(messages, trace, tool_results)
        if not final:
            # ran out of steps: synthesize from whatever we gathered
            final = self.client.synthesize(question, self._evidence_from(tool_results), role)
            trace.add("final_forced")

        # citation guard, with guard-driven reflection: an answer that cites
        # unverifiable clause_ids (or none at all despite gathered evidence)
        # is sent back with the verdict for another bounded attempt
        guard = CitationGuard(self.registry.art.clause_store())
        report = guard.check(final)
        rounds = 0
        while rounds < self.max_repair_rounds and \
                (report.unsupported_ids or
                 (not report.has_any_citation and tool_results)):
            rounds += 1
            trace.add("reflection", round=rounds,
                      unsupported=report.unsupported_ids,
                      has_citation=report.has_any_citation)
            feedback = "⚠️ 引用核驗未通過："
            if report.unsupported_ids:
                feedback += ("以下條文編號無法核實："
                             + "、".join(report.unsupported_ids) + "。")
            if not report.has_any_citation:
                feedback += "回答未附任何條文編號。"
            feedback += ("請重新作答：只可引用已檢索工具結果中出現的 clause_id，"
                         "必要時可再調用工具補充取證；沒有證據的結論必須刪去。")
            messages.append({"role": "assistant", "content": final})
            messages.append({"role": "user", "content": feedback})
            retry = self._react(messages, trace, tool_results, budget=3)
            if not retry:
                break
            final = retry
            report = guard.check(final)
        final = guard.annotate(final, report)
        trace.add("citation_check", **report.to_dict())

        payload = {
            "question": question,
            "answer": final,
            "backend": self.client.backend,
            "tools_used": [t["tool"] for t in tool_results],
            "evidence_clause_ids": report.verified_ids,
            "citation_report": report.to_dict(),
            "reflection_rounds": rounds,
            "agent_trace": trace.steps,
        }
        return safety.governed(payload, role)

    def _react(self, messages: List[Dict], trace: AgentTrace,
               tool_results: List[Dict], budget: Optional[int] = None) -> str:
        """One bounded tool-calling loop; returns final text ('' if budget
        ran out). Shared by the first pass and every reflection round."""
        specs = self.registry.specs()
        for _ in range(budget or self.max_steps):
            res = self.client.chat(messages, tools=specs)
            if res.tool_calls:
                assistant_msg = {"role": "assistant", "content": res.content or None,
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.name,
                                                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                                                for tc in res.tool_calls]}
                messages.append(assistant_msg)
                for tc in res.tool_calls:
                    result = self.registry.call(tc.name, tc.arguments)
                    tool_results.append({"tool": tc.name, "arguments": tc.arguments,
                                         "result": result})
                    trace.add("tool_call", tool=tc.name, arguments=tc.arguments)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "name": tc.name,
                                     "content": json.dumps(result, ensure_ascii=False)})
                continue
            trace.add("final", backend=res.backend)
            return res.content
        return ""

    @staticmethod
    def _evidence_from(tool_results: List[Dict]) -> List[Dict]:
        evidence: List[Dict] = []
        for tr in tool_results:
            r = tr.get("result", {})
            for h in r.get("hits", []) or []:
                evidence.append(h)
            if r.get("clause"):
                evidence.append(r["clause"])
            for m in r.get("matched_formula_patterns", []) or []:
                evidence.extend(m.get("evidence", []))
        return evidence
