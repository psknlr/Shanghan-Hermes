"""Tests for the second-line refinements:

  * MCP isError on unknown tools + -32700 on malformed JSON;
  * therapy overview advertises only sub-Skills that exist on disk;
  * graph-variant folding (脇/鞕/欬/濇) restores extraction recall;
  * formula matching: hallmark triads rank decisively, 提綱證 credited;
  * LLM extractor accepts every clause-level rule type;
  * council specialists add citation-checked LLM remarks when available.
"""
import io
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


class TestMCPErrors(unittest.TestCase):
    def test_unknown_tool_marked_is_error(self):
        from hermes_shanghan.integrations.mcp_server import handle
        resp = handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                       "params": {"name": "no_such_tool", "arguments": {}}})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("unknown tool", resp["result"]["content"][0]["text"])

    def test_domain_error_marked_is_error(self):
        _ensure_artifacts()
        from hermes_shanghan.integrations.mcp_server import handle
        resp = handle({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                       "params": {"name": "shanghan_six_channel",
                                  "arguments": {"channel": "不存在經"}}})
        self.assertTrue(resp["result"].get("isError"))

    def test_malformed_json_gets_parse_error(self):
        from hermes_shanghan.integrations.mcp_server import serve
        out = io.StringIO()
        serve(stdin=io.StringIO("{not json}\n"), stdout=out)
        resp = json.loads(out.getvalue().strip())
        self.assertEqual(resp["error"]["code"], -32700)
        self.assertIsNone(resp["id"])


class TestTherapyOverviewNoPhantoms(unittest.TestCase):
    def test_advertised_subskills_exist(self):
        _ensure_artifacts()
        base = config.SKILLS_DIR / "hermes.shanghan.therapy"
        text = (base / "SKILL.md").read_text(encoding="utf-8")
        import re
        advertised = re.findall(r"hermes\.shanghan\.therapy\.(\w+)", text)
        self.assertTrue(advertised)
        for slug in advertised:
            self.assertTrue((base / slug / "SKILL.md").exists(),
                            f"phantom sub-skill advertised: {slug}")


class TestVariantFolding(unittest.TestCase):
    def test_fold_and_contains_verbatim(self):
        from hermes_shanghan.textutil import contains_verbatim, fold_variants
        self.assertEqual(fold_variants("胸脇苦滿，心下痞鞕，脈濇，欬者"),
                         "胸脅苦滿，心下痞硬，脈澀，咳者")
        self.assertTrue(contains_verbatim("傷寒五六日，胸脇苦滿", "胸脅苦滿"))

    def test_extractor_recall_on_variant_glyphs(self):
        from hermes_shanghan.extract.entities import EntityExtractor
        res = EntityExtractor().extract("往來寒熱，胸脇苦滿，心下痞鞕，脈濇而欬。")
        self.assertIn("胸脅苦滿", res.symptoms)
        self.assertIn("心下痞硬", res.symptoms)
        self.assertTrue(any("澀" in p for p in res.pulse))

    def test_clause_96_symptoms_include_xiongxie(self):
        _ensure_artifacts()
        for line in open(config.CLAUSE_DIR / "clauses.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d["clause_id"] == "SHL_SONGBEN_0096":
                self.assertIn("胸脅苦滿", d["symptoms"])
                return
        self.fail("clause 96 not found")


class TestMatcherTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.apps.doctor import FormulaMatcher
        from hermes_shanghan.orchestrator import Artifacts
        art = Artifacts()
        cls.matcher = FormulaMatcher(art.formula_rules, art.clause_store())

    def test_xiaochaihu_triad_ranks_first_decisively(self):
        out = self.matcher.match(["往來寒熱", "胸脅苦滿", "口苦"])
        top = out["matched_formula_patterns"][0]
        second = out["matched_formula_patterns"][1]
        self.assertEqual(top["formula"], "小柴胡湯")
        self.assertGreaterEqual(top["match_score"], 0.6)
        self.assertGreaterEqual(top["match_score"] - second["match_score"], 0.15)
        self.assertTrue(any(h.startswith("提綱證：口苦") for h in top["matched_findings"]))

    def test_outline_boost_respects_channel_scope(self):
        # 口苦 is the 少陽 outline symptom — a formula never scoped to 少陽
        # must not receive the boost
        out = self.matcher.match(["口苦"])
        for m in out["matched_formula_patterns"]:
            for h in m["matched_findings"]:
                if h.startswith("提綱證："):
                    self.assertIn("少陽病", m["six_channel"])

    def test_mahuang_tang_classic_presentation(self):
        out = self.matcher.match(["惡寒", "發熱", "無汗", "身疼痛"], pulse=["浮緊"])
        self.assertEqual(out["matched_formula_patterns"][0]["formula"], "麻黃湯")


class TestLLMExtractorAllTypes(unittest.TestCase):
    def test_all_clause_level_types_allowed(self):
        from hermes_shanghan.extract.llm_extractor import RULE_TYPES_ALLOWED
        from hermes_shanghan.schemas import RULE_TYPES
        self.assertEqual(RULE_TYPES_ALLOWED,
                         RULE_TYPES - {"variant_rule", "commentary_rule"})

    def test_administration_rule_accepted_from_model(self):
        _ensure_artifacts()
        from hermes_shanghan.extract.llm_extractor import LLMRuleExtractor
        from hermes_shanghan.rag.clause_rag import ClauseRAG
        rag = ClauseRAG.load()
        clause = rag.get_clause(12)
        draft = {"rules": [{
            "rule_type": "administration_rule",
            "if_conditions": {"symptoms": []},
            "then_conclusions": {"administration": ["溫服一升"]},
            "prescription_strength": "",
            "evidence_span": "溫服一升",
            "interpretation": "服法",
            "interpretation_level": "literal",
            "model_confidence": 0.8}]}
        client = LLMClient(settings=LLMSettings(cache=False),
                           provider=ScriptedProvider([json.dumps(draft, ensure_ascii=False)]))
        rules = LLMRuleExtractor(client).extract_clause(clause)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rule_type, "administration_rule")


class TestCouncilSpecialistLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def _council(self, queue, **kw):
        from hermes_shanghan.agent.multi_agent import Council
        client = LLMClient(settings=LLMSettings(cache=False),
                           provider=ScriptedProvider(queue))
        client._backend = "litellm"  # make `available` True for the test
        return Council(client=client, **kw)

    def test_specialist_remark_appended_and_verified(self):
        queue = ["依 SHL_SONGBEN_0096，少陽樞機不利，宜和解。",
                 "綜合結論：小柴胡湯（SHL_SONGBEN_0096）。"]
        out = self._council(queue).deliberate("往來寒熱，胸脅苦滿，口苦，用什麼方？",
                                              role="doctor")
        analyze = [m for m in out["council"] if m["action"] == "analyze"]
        self.assertTrue(any("💬" in m["content"] for m in analyze))
        remarked = next(m for m in analyze if "💬" in m["content"])
        self.assertIn("SHL_SONGBEN_0096", remarked["evidence_ids"])

    def test_fabricated_specialist_citation_flagged(self):
        queue = ["依 SHL_SONGBEN_9999 云云。", "答案。"]
        out = self._council(queue).deliberate("往來寒熱，用什麼方？", role="doctor")
        analyze = [m for m in out["council"] if "💬" in m.get("content", "")]
        self.assertTrue(analyze)
        self.assertIn("未核實", analyze[0]["content"])

    def test_llm_specialists_can_be_disabled(self):
        queue = ["這句不應出現。", "答案。"]
        out = self._council(queue, llm_specialists=False).deliberate(
            "往來寒熱，用什麼方？", role="doctor")
        analyze = [m for m in out["council"] if m["action"] == "analyze"]
        self.assertFalse(any("💬" in m["content"] for m in analyze))


if __name__ == "__main__":
    unittest.main()
