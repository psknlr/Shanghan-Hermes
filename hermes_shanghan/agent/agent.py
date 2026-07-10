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
                 max_repair_rounds: int = 1, max_tool_calls: int = 12):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        self.max_steps = max_steps
        # reflection: when the citation guard rejects an answer, feed the
        # verdict back and let the model retry (guard as controller, not
        # just annotator)
        self.max_repair_rounds = max_repair_rounds
        # hard per-question tool budget across the first pass AND all
        # reflection rounds — a confused model cannot retrieve forever
        self.max_tool_calls = max_tool_calls

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

        # patient safety: red-flag triage, then intent guard — both BEFORE
        # any model/tool call
        if role == "patient":
            triage = safety.red_flag_triage(question)
            if triage:
                trace.add("red_flag_triage", flags=triage["red_flags"],
                          urgent=triage["urgent"])
                out = safety.governed(triage, "patient")
                out["agent_trace"] = trace.steps
                out["backend"] = self.client.backend
                return out
            guard = safety.patient_intent_guard(question)
            if guard:
                trace.add("safety_block", intents=guard["refused_intents"])
                out = safety.governed(guard, "patient")
                out["agent_trace"] = trace.steps
                out["backend"] = self.client.backend
                return out

        # hard role isolation: the tool surface itself is scoped by role
        registry = self.registry.for_role(role)
        if registry is not self.registry:
            trace.add("tool_scope", role=role, tools=registry.names())

        messages: List[Dict] = [
            {"role": "system", "content": agent_system_prompt(role)},
            {"role": "user", "content": question},
        ]
        tool_results: List[Dict] = []
        final = self._react(messages, trace, tool_results, registry)
        if not final:
            # ran out of steps: synthesize from whatever we gathered
            final = self.client.synthesize(question, self._evidence_from(tool_results), role)
            trace.add("final_forced")

        # citation guard, with guard-driven reflection: an answer that cites
        # unverifiable clause_ids (or none at all despite gathered evidence)
        # is sent back with the verdict for another bounded attempt
        guard = CitationGuard(registry.art.clause_store())
        # strict grounding: citations must come from THIS round's tool
        # evidence, not merely exist somewhere in the corpus
        allowed = self._clause_ids_from(tool_results)
        report = guard.check(final, allowed_ids=allowed if tool_results else None)
        rounds = 0
        while rounds < self.max_repair_rounds and \
                (report.unsupported_ids or report.outside_evidence_ids or
                 (not report.has_any_citation and tool_results)):
            rounds += 1
            trace.add("reflection", round=rounds,
                      unsupported=report.unsupported_ids,
                      outside_evidence=report.outside_evidence_ids,
                      has_citation=report.has_any_citation)
            feedback = "⚠️ 引用核驗未通過："
            if report.unsupported_ids:
                feedback += ("以下條文編號無法核實："
                             + "、".join(report.unsupported_ids) + "。")
            if report.outside_evidence_ids:
                feedback += ("以下條文未出現在本輪工具證據中，須先檢索取證再引用："
                             + "、".join(report.outside_evidence_ids) + "。")
            if not report.has_any_citation:
                feedback += "回答未附任何條文編號。"
            feedback += ("請重新作答：只可引用已檢索工具結果中出現的 clause_id，"
                         "必要時可再調用工具補充取證；沒有證據的結論必須刪去。")
            messages.append({"role": "assistant", "content": final})
            messages.append({"role": "user", "content": feedback})
            retry = self._react(messages, trace, tool_results, registry, budget=3)
            if not retry:
                break
            final = retry
            allowed = self._clause_ids_from(tool_results)
            report = guard.check(final, allowed_ids=allowed if tool_results else None)

        # multi-hypothesis layer: when this round did formula matching,
        # upgrade top-k into parallel hypotheses + 鑒別追問 (亮點功能一)
        hyp_payload = self._hypotheses_from(tool_results, registry, role)
        if hyp_payload:
            from .hypothesis import render_hypotheses
            final = final.rstrip() + "\n\n" + render_hypotheses(hyp_payload)
            trace.add("hypotheses", n=len(hyp_payload["hypotheses"]),
                      needs_clarification=hyp_payload["needs_clarification"])
            allowed = self._clause_ids_from(tool_results)
            report = guard.check(final, allowed_ids=allowed if tool_results else None)

        # claim→evidence binding: sentence-level audit of what supports what
        from .evidence_binder import EvidenceBinder
        binding = EvidenceBinder(registry.art.clause_store()).bind(final, tool_results)
        trace.add("claim_binding", n_claims=binding["n_claims"],
                  grounding_rate=binding["claim_grounding_rate"])

        final = guard.annotate(final, report)
        trace.add("citation_check", **report.to_dict())

        payload = {
            "question": question,
            "answer": final,
            "backend": self.client.backend,
            "tools_used": [t["tool"] for t in tool_results],
            "evidence_clause_ids": report.verified_ids,
            "citation_report": report.to_dict(),
            "claims": binding,
            "reflection_rounds": rounds,
            "agent_trace": trace.steps,
        }
        if hyp_payload:
            payload["hypotheses"] = hyp_payload["hypotheses"]
            payload["decision"] = hyp_payload["decision"]
            if hyp_payload["needs_clarification"]:
                payload["clarification"] = {
                    "reason": hyp_payload["clarification_reason"],
                    "questions": hyp_payload["clarifying_questions"]}
        return safety.governed(payload, role)

    def _hypotheses_from(self, tool_results: List[Dict], registry,
                         role: str) -> Optional[Dict]:
        """Reuse this round's match/hypotheses call to attach the parallel-
        hypothesis analysis (doctor/student/researcher surfaces only)."""
        if role == "patient":
            return None
        for tr in tool_results:
            if tr["tool"] == "shanghan_hypotheses" and \
                    tr["result"].get("hypotheses"):
                return tr["result"]
        match = next((t for t in tool_results
                      if t["tool"] == "shanghan_match_formula"
                      and t.get("result", {}).get("matched_formula_patterns")),
                     None)
        if match is None:
            return None
        try:
            from .hypothesis import HypothesisManager

            def as_list(v):
                # raw tool arguments may predate registry coercion
                if isinstance(v, str):
                    return [s for s in
                            (x.strip() for x in
                             v.replace("，", ",").replace("、", ",").split(","))
                            if s]
                return list(v or [])

            args = match.get("arguments", {}) or {}
            out = HypothesisManager(registry).analyze(
                symptoms=as_list(args.get("symptoms")),
                pulse=as_list(args.get("pulse")),
                six_channel=args.get("six_channel"))
            return out if out.get("hypotheses") else None
        except Exception:
            return None    # hypothesis layer is additive; never break ask()

    def _react(self, messages: List[Dict], trace: AgentTrace,
               tool_results: List[Dict], registry=None,
               budget: Optional[int] = None) -> str:
        """One bounded tool-calling loop; returns final text ('' if budget
        ran out). Shared by the first pass and every reflection round."""
        registry = registry or self.registry
        specs = registry.specs()
        for _ in range(budget or self.max_steps):
            # tool budget is global to the question: once exhausted, the
            # model is asked to answer from gathered evidence only
            over_budget = len(tool_results) >= self.max_tool_calls
            if over_budget:
                trace.add("tool_budget_exhausted", used=len(tool_results),
                          budget=self.max_tool_calls)
            res = self.client.chat(messages, tools=None if over_budget else specs)
            if res.tool_calls:
                assistant_msg = {"role": "assistant", "content": res.content or None,
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.name,
                                                              "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                                                for tc in res.tool_calls]}
                messages.append(assistant_msg)
                for tc in res.tool_calls:
                    # 預算逐個檢查：模型單輪返回的批量 tool_calls 不能突破
                    # 預算（九輪 P0-3）。超限的調用不執行，回 BUDGET_EXHAUSTED
                    # 信封（協議仍要求每個 tool_call_id 有響應）。
                    if len(tool_results) >= self.max_tool_calls:
                        trace.add("tool_budget_denied", tool=tc.name,
                                  used=len(tool_results),
                                  budget=self.max_tool_calls)
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": json.dumps(
                                {"error": "BUDGET_EXHAUSTED：本問題工具預算"
                                          "已用盡，該調用未執行；請基於已"
                                          "取證作答"}, ensure_ascii=False)})
                        continue
                    result = registry.call(tc.name, tc.arguments)
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
    def _clause_ids_from(tool_results: List[Dict]) -> List[str]:
        """Every clause_id present anywhere in this round's tool results —
        the only ids an answer is allowed to cite."""
        from .citation_guard import RE_CLAUSE_ID
        blob = json.dumps([t.get("result", {}) for t in tool_results],
                          ensure_ascii=False)
        return list(dict.fromkeys(RE_CLAUSE_ID.findall(blob)))

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
