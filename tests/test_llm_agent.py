"""LLM layer + agent tests.

Two modes are covered without any network:
  * local backend  — the deterministic 'brain' drives the real ReAct loop;
  * scripted backend — canned model responses exercise the real-model code
    path (tool-calling parse, citation guard catching fabrication, fallback).
"""
import unittest

from hermes_shanghan import config, safety
from hermes_shanghan.llm.client import LLMClient
from hermes_shanghan.llm.providers import (ChatResult, LocalProvider,
                                           ScriptedProvider, ToolCall)


def _ensure_artifacts():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestLLMClientOffline(unittest.TestCase):
    def test_local_backend_default(self):
        c = LLMClient()
        self.assertIn(c.backend, ("local", "litellm"))
        # in this sandbox there is no litellm/key → local
        self.assertEqual(c.backend, "local")
        self.assertFalse(c.available)

    def test_status(self):
        st = LLMClient().status()
        self.assertIn("backend", st)
        self.assertIn("reason", st)

    def test_local_extract_then_review_guard(self):
        # the evidence gate is the safety net: every surviving LLM rule must be
        # evidence_verified; anything ungrounded is rejected, never silently kept
        _ensure_artifacts()
        from hermes_shanghan.rag.clause_rag import ClauseRAG
        from hermes_shanghan.extract.llm_extractor import LLMRuleExtractor
        from hermes_shanghan.review.pipeline import ReviewPipeline
        rag = ClauseRAG.load()
        c = rag.get_clause(12)
        cands = LLMRuleExtractor(LLMClient()).extract_clause(c)
        self.assertTrue(cands)
        store = {cc.clause_id: cc for cc in rag.clauses}
        rp = ReviewPipeline(store)
        for r in cands:
            reviewed = rp.review_rule(r)
            if reviewed.autonomous_review.release_level != "rejected":
                self.assertTrue(reviewed.autonomous_review.evidence_verified)
        # the canonical 桂枝湯 formula rule should survive as gold
        formula_rules = [rp.review_rule(r) for r in
                         LLMRuleExtractor(LLMClient()).extract_clause(c)
                         if r.rule_type == "formula_pattern_rule"]
        self.assertTrue(any(r.autonomous_review.release_level in ("gold", "silver")
                            and "桂枝湯" in r.then_conclusions.get("formula", [])
                            for r in formula_rules))


class TestScriptedToolCalling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def _agent_with(self, queue):
        from hermes_shanghan.agent.agent import ShanghanAgent
        client = LLMClient(provider=ScriptedProvider(queue))
        return ShanghanAgent(client=client)

    def test_real_path_toolcall_then_answer(self):
        # 1st model turn: ask for a tool; 2nd turn: final answer citing a clause
        queue = [
            {"tool_calls": [{"id": "t1", "name": "shanghan_get_clause",
                             "arguments": {"ref": "12"}}]},
            {"content": "太陽中風用桂枝湯，見第12條（SHL_SONGBEN_0012）。"},
        ]
        out = self._agent_with(queue).ask("桂枝湯主治什麼？", role="doctor")
        self.assertEqual(out["backend"], "scripted")
        self.assertIn("shanghan_get_clause", out["tools_used"])
        self.assertIn("SHL_SONGBEN_0012", out["evidence_clause_ids"])
        self.assertTrue(out["citation_report"]["ok"])

    def test_citation_guard_flags_fabricated_clause(self):
        # model fabricates a non-existent clause id → guard must flag it
        queue = [{"content": "依據 SHL_SONGBEN_9999，此方主之。"}]
        out = self._agent_with(queue).ask("隨便問", role="researcher")
        self.assertIn("SHL_SONGBEN_9999", out["citation_report"]["unsupported"])
        self.assertIn("未能核實", out["answer"])

    def test_patient_guard_precedes_model(self):
        # scripted model would happily prescribe, but the guard fires first
        queue = [{"content": "给你开桂枝汤三两。"}]
        agent = self._agent_with(queue)
        out = agent.ask("给我开个方，剂量多少？", role="patient")
        self.assertTrue(out.get("refused"))
        # the scripted response must never have been consumed
        self.assertEqual(len(agent.client.provider.calls), 0)


class TestLLMCriticGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_llm_critic_can_downgrade_not_promote(self):
        from hermes_shanghan.corpus import segmenter
        from hermes_shanghan.extract.entities import EntityExtractor, annotate_clause
        from hermes_shanghan.review.pipeline import ReviewPipeline
        from hermes_shanghan.schemas import AutonomousReview, InitialRule

        clauses = segmenter.segment_canonical()
        ex = EntityExtractor(segmenter.harvest_formula_names(clauses))
        for c in clauses:
            annotate_clause(c, ex)
        store = {c.clause_id: c for c in clauses}

        class HostileCritic:
            def review(self, rule, clause_store):
                return "fail", ["llm:always_hostile"], "test"

        good = InitialRule(
            initial_rule_id="IR_SHL_0012_701", clause_id="SHL_SONGBEN_0012",
            six_channel="太陽病", rule_type="formula_pattern_rule",
            if_conditions={"disease": ["太陽中風"], "symptoms": [], "pulse": ["陽浮而陰弱"]},
            then_conclusions={"formula": ["桂枝湯"]},
            evidence_span=store["SHL_SONGBEN_0012"].clean_text,
            evidence_type="original_text", interpretation="x",
            interpretation_level="normalized", model_confidence=0.9,
            prescription_strength="主之", autonomous_review=AutonomousReview())
        rp = ReviewPipeline(store, llm_critic=HostileCritic())
        r = rp.review_rule(good)
        # hostile LLM verdict downgrades but evidence-true rule is not rejected
        self.assertTrue(r.autonomous_review.evidence_verified)
        self.assertNotEqual(r.autonomous_review.release_level, "gold")
        self.assertIn("llm:always_hostile", r.autonomous_review.critic_flags)


class TestToolRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_specs_and_calls(self):
        from hermes_shanghan.agent.tools import get_registry
        reg = get_registry()
        self.assertEqual(len(reg.specs()), 36)   # 28 領域 + 8 classics
        out = reg.call("shanghan_search", {"query": "桂枝湯", "top_k": 3})
        self.assertTrue(out["hits"])
        self.assertTrue(all(h["clause_id"] for h in out["hits"]))

    def test_unknown_tool_safe(self):
        from hermes_shanghan.agent.tools import get_registry
        out = get_registry().call("nope", {})
        self.assertIn("error", out)

    def test_mcp_handle(self):
        from hermes_shanghan.integrations.mcp_server import handle
        init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "hermes-shanghanlun")
        listed = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertGreaterEqual(len(listed["result"]["tools"]), 9)

    def test_openai_specs_export(self):
        from hermes_shanghan.integrations.tool_specs import openai_tool_specs
        specs = openai_tool_specs()
        self.assertTrue(all(s["type"] == "function" for s in specs))


if __name__ == "__main__":
    unittest.main()
