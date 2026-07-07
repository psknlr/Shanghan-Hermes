"""Deep-research loop, module auto-selection, SVG charts, provenance paper."""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from hermes_shanghan import config
from hermes_shanghan.llm.client import LLMClient
from hermes_shanghan.llm.config import LLMSettings
from hermes_shanghan.llm.providers import ScriptedProvider


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestResearchModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_registry_has_twelve_tools(self):
        self.assertEqual(len(self.reg.specs()), 12)
        names = self.reg.names()
        for t in ("shanghan_divergence_atlas", "shanghan_dose",
                  "shanghan_corpus_stats", "shanghan_eval_metrics"):
            self.assertIn(t, names)

    def test_research_modules_return_data(self):
        out = self.reg.call("shanghan_divergence_atlas", {})
        self.assertEqual(out["n_books"], 9)
        out = self.reg.call("shanghan_dose", {"formula": "桂枝加芍藥湯"})
        self.assertIn("芍藥3", out["ratio"]["ratio"])
        out = self.reg.call("shanghan_corpus_stats", {})
        self.assertGreater(out["initial_rules"], 1000)

    def test_llm_auto_selects_module_offline(self):
        # 語言模型自動選擇調用模塊：the deterministic brain routes a dose
        # question to shanghan_dose through the SAME tool-choice loop a real
        # model uses (function calling over the same specs)
        from hermes_shanghan.agent.agent import ShanghanAgent
        out = ShanghanAgent().ask("桂枝湯的劑量折算是多少克？", role="researcher")
        self.assertIn("shanghan_dose", out["tools_used"])
        out = ShanghanAgent().ask("第12條各注家的詮釋有何分歧？", role="researcher")
        self.assertIn("shanghan_divergence_atlas", out["tools_used"])


class TestDeepResearcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_loop_converges_with_full_coverage(self):
        from hermes_shanghan.agent.research_loop import (DIMENSIONS,
                                                         DeepResearcher)
        d = DeepResearcher(max_rounds=3).run("桂枝湯類方的劑量演化")
        self.assertLessEqual(d["n_rounds"], 3)
        self.assertEqual(d["uncovered_dimensions"], [])
        for dim in DIMENSIONS:
            self.assertGreaterEqual(d["coverage"][dim], 1)
        self.assertTrue(d["evidence_clause_ids"])
        for f in d["findings"]:
            self.assertIn("verified_clause_ids", f)

    def test_no_duplicate_module_calls(self):
        from hermes_shanghan.agent.research_loop import DeepResearcher
        d = DeepResearcher(max_rounds=3).run("小柴胡湯")
        calls = [(t["module"], json.dumps(t["args"], sort_keys=True))
                 for r in d["rounds"] for t in r["tasks"]]
        self.assertEqual(len(calls), len(set(calls)))

    def test_llm_planner_path_with_scripted_model(self):
        from hermes_shanghan.agent.research_loop import DeepResearcher
        plan = json.dumps({"tasks": [
            {"module": "shanghan_dose", "args": {"formula": "桂枝湯"},
             "reason": "劑量"},
            {"module": "not_a_module", "args": {}, "reason": "應被過濾"}]},
            ensure_ascii=False)
        queue = [plan, "劑量發現：桂枝三兩（SHL_SONGBEN_0012）。",
                 json.dumps({"tasks": []})]
        client = LLMClient(settings=LLMSettings(cache=False),
                           provider=ScriptedProvider(queue))
        client._backend = "litellm"
        d = DeepResearcher(client=client, max_rounds=2).run("桂枝湯")
        mods = [t["module"] for r in d["rounds"] for t in r["tasks"]]
        self.assertEqual(mods, ["shanghan_dose"])   # invalid module filtered
        self.assertIn("SHL_SONGBEN_0012",
                      d["findings"][0]["verified_clause_ids"])


class TestCharts(unittest.TestCase):
    def test_svgs_wellformed_and_escaped(self):
        from hermes_shanghan.paper.charts import (grouped_hbar_chart, heatmap,
                                                  hbar_chart)
        bars = [
            hbar_chart([("桂枝湯", 26), ("A&B<湯>", 3)], "測試", "副題"),
            grouped_hbar_chart([("桂枝湯", [171.9, 153.1, 33.0])],
                               ["考古", "度量衡", "折算"], "總量"),
        ]
        for svg in bars:
            ET.fromstring(svg)              # parses → well-formed XML
            self.assertIn("#2a78d6", svg)   # validated palette slot 1
        hm = heatmap(["甲", "乙"], {("甲", "乙"): 0.9}, "矩陣")
        ET.fromstring(hm)
        self.assertIn("0.90", hm)           # every cell direct-labeled

    def test_deterministic(self):
        from hermes_shanghan.paper.charts import hbar_chart
        a = hbar_chart([("x", 1.0)], "t")
        b = hbar_chart([("x", 1.0)], "t")
        self.assertEqual(a, b)


class TestProvenancePaper(unittest.TestCase):
    def test_provenance_paper_generated_with_charts(self):
        _ensure_artifacts()
        from hermes_shanghan.orchestrator import Artifacts
        from hermes_shanghan.paper.writer import PaperWriter
        art = Artifacts()
        w = PaperWriter(art.clauses, art.initial_rules, art.formula_rules,
                        art.six_channel_rules, art.mistreatment_rules,
                        art.differential_rules,
                        commentary_rules=art.commentary_rules)
        with tempfile.TemporaryDirectory() as tmp:
            path = w.generate(paper_type="provenance", topic="桂枝湯類方源流",
                              out_dir=Path(tmp))
            text = path.read_text(encoding="utf-8")
            figs = sorted(p.name for p in Path(tmp).glob("fig*.svg"))
        self.assertIn("深度研究循環溯源發現", text)
        self.assertIn("循環軌跡", text)
        self.assertIn("fig5_formula_frequency.svg", text)
        self.assertIn("fig5_formula_frequency.svg", " ".join(figs))
        self.assertIn("fig6_commentator_agreement.svg", " ".join(figs))
        self.assertIn("fig7_dose_totals.svg", " ".join(figs))


if __name__ == "__main__":
    unittest.main()
