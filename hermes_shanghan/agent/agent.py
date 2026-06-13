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
                 registry: Optional[ToolRegistry] = None, max_steps: int = 5):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        self.max_steps = max_steps

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
        specs = self.registry.specs()
        tool_results: List[Dict] = []
        final = ""
        for _ in range(self.max_steps):
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
            final = res.content
            trace.add("final", backend=res.backend)
            break
        else:
            # ran out of steps: synthesize from whatever we gathered
            final = self.client.synthesize(question, self._evidence_from(tool_results), role)
            trace.add("final_forced")

        # citation guard
        guard = CitationGuard(self.registry.art.clause_store())
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
            "agent_trace": trace.steps,
        }
        return safety.governed(payload, role)

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
