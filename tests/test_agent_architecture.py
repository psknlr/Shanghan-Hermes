"""Agent-architecture tests: guard-driven reflection, scoped registries,
compound-question orchestration, and session memory."""
import json
import unittest

from hermes_shanghan import config
from hermes_shanghan.llm.client import LLMClient
from hermes_shanghan.llm.config import LLMSettings
from hermes_shanghan.llm.providers import ScriptedProvider


def _ensure_artifacts():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


def _client(queue):
    return LLMClient(settings=LLMSettings(cache=False),
                     provider=ScriptedProvider(queue))


class TestGuardDrivenReflection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def _agent(self, queue, **kw):
        from hermes_shanghan.agent.agent import ShanghanAgent
        return ShanghanAgent(client=_client(queue), **kw)

    def test_fabricated_citation_triggers_retry_and_recovers(self):
        queue = [
            {"tool_calls": [{"id": "t1", "name": "shanghan_get_clause",
                             "arguments": {"ref": "12"}}]},
            "太陽中風，見 SHL_SONGBEN_9999。",              # fabricated
            "太陽中風用桂枝湯，見 SHL_SONGBEN_0012。",       # corrected
        ]
        out = self._agent(queue).ask("桂枝湯主治？", role="doctor")
        self.assertEqual(out["reflection_rounds"], 1)
        self.assertEqual(out["citation_report"]["unsupported"], [])
        self.assertIn("SHL_SONGBEN_0012", out["evidence_clause_ids"])
        self.assertTrue(any(s["kind"] == "reflection"
                            for s in out["agent_trace"]))

    def test_reflection_budget_is_bounded(self):
        queue = ["見 SHL_SONGBEN_9999。", "還是 SHL_SONGBEN_9998。",
                 "仍然 SHL_SONGBEN_9997。"]
        out = self._agent(queue, max_repair_rounds=1).ask("問", role="researcher")
        self.assertEqual(out["reflection_rounds"], 1)
        # cap reached → the bad answer ships, loudly annotated
        self.assertIn("請勿採信", out["answer"])

    def test_reflection_can_gather_more_evidence(self):
        # the retry round is a full ReAct continuation: it may call tools
        queue = [
            "太陽病如何。",                                   # no citation
            {"tool_calls": [{"id": "t2", "name": "shanghan_get_clause",
                             "arguments": {"ref": "1"}}]},
            "太陽之為病，脈浮。見 SHL_SONGBEN_0001。",
        ]
        # force a tool result before first answer so reflection triggers
        queue.insert(0, {"tool_calls": [{"id": "t1", "name": "shanghan_search",
                                         "arguments": {"query": "太陽"}}]})
        out = self._agent(queue).ask("太陽病提綱？", role="student")
        self.assertIn("SHL_SONGBEN_0001", out["evidence_clause_ids"])
        self.assertIn("shanghan_get_clause", out["tools_used"])


class TestScopedRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_scope_limits_specs_and_calls(self):
        from hermes_shanghan.agent.tools import ScopedRegistry, get_registry
        scope = ScopedRegistry(get_registry(), ["shanghan_dose", "nonexistent"])
        self.assertEqual(scope.names(), ["shanghan_dose"])
        self.assertEqual(len(scope.specs()), 1)
        out = scope.call("shanghan_search", {"query": "x"})
        self.assertIn("out of scope", out["error"])
        out = scope.call("shanghan_dose", {"formula": "桂枝湯"})
        self.assertIn("ratio", out)


class TestComplexAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_compound_question_offline(self):
        from hermes_shanghan.agent.complex_agent import ComplexAgent
        out = ComplexAgent().solve(
            "桂枝湯與麻黃湯如何鑒別？各自劑量比是多少？注家對第12條有何分歧？",
            role="researcher")
        kinds = [t["kind"] for t in out["subtasks"]]
        self.assertEqual(kinds, ["differential", "dose", "commentary"])
        # anchor re-attachment: the dose fragment names no formula itself
        self.assertIn("涉及：桂枝湯", out["subtasks"][1]["question"])
        self.assertTrue(out["citation_report"]["ok"])
        self.assertTrue(out["evidence_clause_ids"])
        self.assertEqual(out["orchestrator_trace"][0]["step"], "decompose")

    def test_llm_decomposition_filters_invalid_kinds(self):
        from hermes_shanghan.agent.complex_agent import ComplexAgent
        plan = json.dumps({"subtasks": [
            {"kind": "dose", "question": "桂枝湯劑量比？"},
            {"kind": "not_a_kind", "question": "應被過濾"}]}, ensure_ascii=False)
        queue = [plan,
                 "桂枝湯藥量比 1.5:1.5:1（SHL_SONGBEN_0012）。",
                 "綜合：桂枝湯劑量比見 SHL_SONGBEN_0012。"]
        client = _client(queue)
        client._backend = "litellm"
        out = ComplexAgent(client=client).solve("桂枝湯劑量？", role="doctor")
        self.assertEqual([t["kind"] for t in out["subtasks"]], ["dose"])
        self.assertIn("SHL_SONGBEN_0012", out["evidence_clause_ids"])

    def test_patient_guard_precedes_orchestration(self):
        from hermes_shanghan.agent.complex_agent import ComplexAgent
        out = ComplexAgent().solve("給我開個方？劑量多少？", role="patient")
        self.assertTrue(out.get("refused"))


class TestAgentSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_followup_resolves_against_history(self):
        from hermes_shanghan.agent.session import AgentSession
        sess = AgentSession(role="researcher")
        first = sess.ask("桂枝湯的方證要點是什麼？")
        self.assertIn("桂枝湯", sess.anchors)
        second = sess.ask("它的劑量比呢？")
        self.assertTrue(second["session"]["contextualized"])
        self.assertIn("shanghan_dose", second["tools_used"])
        self.assertIn("桂枝1.50", second["answer"])
        self.assertEqual(second["session"]["turn"], 2)
        self.assertTrue(sess.ledger)   # evidence accumulated across turns
        self.assertTrue(first["evidence_clause_ids"])

    def test_service_keeps_sessions_apart(self):
        from hermes_shanghan.server.service import ServiceContext
        svc = ServiceContext()
        a1 = svc.chat("桂枝湯的方證要點？", session_id="a", role="doctor")
        b1 = svc.chat("麻黃湯的方證要點？", session_id="b", role="doctor")
        a2 = svc.chat("它的劑量比呢？", session_id="a", role="doctor")
        self.assertEqual(a1["session"]["turn"], 1)
        self.assertEqual(b1["session"]["turn"], 1)
        self.assertEqual(a2["session"]["turn"], 2)
        self.assertIn("桂枝", a2["answer"])       # session a's anchor, not b's


if __name__ == "__main__":
    unittest.main()
