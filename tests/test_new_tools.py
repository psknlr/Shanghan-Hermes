"""Six new agent tools: variants, relations, therapy, contraindication
check, dose converter, case search — plus routing and loop integration."""
import unittest

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestNewTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_registry_tool_count(self):
        names = self.reg.names()
        self.assertEqual(len(names), 36)   # 28 領域 + 8 classics
        for t in ("shanghan_variants", "shanghan_relations", "shanghan_therapy",
                  "shanghan_contraindication_check", "shanghan_dose_convert",
                  "shanghan_case_search", "shanghan_trace",
                  "shanghan_citation_network"):
            self.assertIn(t, names)

    def test_variants_b_layer(self):
        out = self.reg.call("shanghan_variants", {"ref": "12"})
        self.assertEqual(out["clause_id"], "SHL_SONGBEN_0012")
        books = {v["book"] for v in out["variants"]}
        self.assertIn("傷寒雜病論_桂本", books)

    def test_relations_graph_traversal(self):
        out = self.reg.call("shanghan_relations",
                            {"ref": "15",
                             "relation_type": "mistreatment_transformation"})
        self.assertGreaterEqual(out["n_edges"], 1)
        self.assertTrue(all(e["relation_type"] == "mistreatment_transformation"
                            for e in out["edges"]))
        # B/C layer edges are excluded (dedicated tools exist)
        out = self.reg.call("shanghan_relations", {"ref": "12"})
        self.assertFalse(any(e["relation_type"] in ("variant", "commentary_support")
                             for e in out["edges"]))

    def test_therapy_rules(self):
        out = self.reg.call("shanghan_therapy", {"method": "禁汗"})
        self.assertTrue(out["rules"])
        self.assertTrue(all("禁汗" in r["method"] for r in out["rules"]))
        out = self.reg.call("shanghan_therapy", {"method": "不存在法"})
        self.assertIn("available", out)

    def test_contraindication_check_composite_reasoning(self):
        out = self.reg.call("shanghan_contraindication_check",
                            {"formula": "麻黃湯", "symptoms": ["汗出"]})
        self.assertTrue(any(c["presented"] == "汗出"
                            for c in out["symptom_conflicts"]))
        bans = [b["method"] for b in out["therapy_law_bans"]]
        self.assertIn("禁汗", bans)
        self.assertEqual(len(bans), len(set(bans)))   # deduped
        # 酒客不可與桂枝湯 comes straight from the formula rule
        out = self.reg.call("shanghan_contraindication_check",
                            {"formula": "桂枝湯"})
        self.assertTrue(any("酒客" in c["condition"]
                            for c in out["formula_contraindications"]))

    def test_dose_convert_calculator(self):
        out = self.reg.call("shanghan_dose_convert", {"dose": "一兩半"})
        self.assertEqual(out["zhu"], 36.0)
        self.assertAlmostEqual(out["grams_by_school"]["kaogu"], 23.44, places=2)
        out = self.reg.call("shanghan_dose_convert", {"dose": "半升"})
        self.assertEqual(out["ml"], 100.0)
        out = self.reg.call("shanghan_dose_convert", {"dose": "亂寫"})
        self.assertIn("error", out)

    def test_case_search_with_canonical_anchor(self):
        out = self.reg.call("shanghan_case_search", {"formula": "桂枝湯"})
        self.assertGreaterEqual(out["n_matched"], 3)
        first = out["cases"][0]
        self.assertIn("桂枝湯", first["formula"])
        # 醫案屬旁證，必須附經文錨點
        self.assertIn("SHL_SONGBEN_0012", first["canonical_support"])
        self.assertIn("非經文層", out["evidence_layer"])


class TestNewToolRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def _tools_used(self, q, role="researcher"):
        from hermes_shanghan.agent.agent import ShanghanAgent
        return ShanghanAgent().ask(q, role=role)["tools_used"]

    def test_offline_auto_selection(self):
        self.assertIn("shanghan_dose_convert",
                      self._tools_used("一兩半折合多少克？"))
        self.assertIn("shanghan_contraindication_check",
                      self._tools_used("病人有汗出，能不能用麻黃湯？", "doctor"))
        self.assertIn("shanghan_case_search",
                      self._tools_used("有沒有桂枝湯的醫案？"))
        self.assertIn("shanghan_variants",
                      self._tools_used("第12條桂本有什麼異文？"))
        self.assertIn("shanghan_therapy",
                      self._tools_used("禁汗的法度有哪些？"))


class TestSixDimensionLoop(unittest.TestCase):
    def test_research_loop_covers_case_dimension(self):
        _ensure_artifacts()
        from hermes_shanghan.agent.research_loop import DeepResearcher
        d = DeepResearcher(max_rounds=3).run("桂枝湯類方源流")
        self.assertEqual(d["uncovered_dimensions"], [])
        self.assertGreaterEqual(d["coverage"]["醫案例證"], 1)
        case_f = next(f for f in d["findings"] if f["dimension"] == "醫案例證")
        self.assertIn("醫案", case_f["summary"])


if __name__ == "__main__":
    unittest.main()
