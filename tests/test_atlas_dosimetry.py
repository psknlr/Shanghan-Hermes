"""Multi-commentator alignment / divergence atlas + dosimetric layer tests."""
import json
import tempfile
import unittest
from pathlib import Path

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestDoseParser(unittest.TestCase):
    def test_units_and_compounds(self):
        from hermes_shanghan.apps.dosimetry import parse_dose
        self.assertEqual(parse_dose("三兩，去皮")["zhu"], 72)
        self.assertEqual(parse_dose("一兩十六銖，去皮")["zhu"], 40)
        self.assertEqual(parse_dose("三分")["zhu"], 18)          # 1分=6銖
        self.assertEqual(parse_dose("一斤，碎")["zhu"], 384)
        self.assertEqual(parse_dose("半升")["ml"], 100.0)
        p = parse_dose("十二枚，擘")
        self.assertEqual((p["kind"], p["count"]), ("count", 12))
        # unit chars inside processing tails are NOT doses
        p = parse_dose("一枚，炮，去皮，破八片")
        self.assertEqual((p["kind"], p["count"]), ("count", 1))

    def test_school_conversions(self):
        from hermes_shanghan.apps.dosimetry import parse_dose
        g = parse_dose("一兩")["grams"]
        self.assertAlmostEqual(g["kaogu"], 15.62, places=1)
        self.assertAlmostEqual(g["duliangheng"], 13.92, places=2)
        self.assertAlmostEqual(g["zhezhuan"], 3.0, places=2)

    def test_each_group_resolution(self):
        from hermes_shanghan.apps.dosimetry import parse_dose, resolve_each_groups
        parsed = [parse_dose(""), parse_dose(""), parse_dose("炙，各十八銖")]
        resolve_each_groups(parsed)
        self.assertEqual(parsed[0]["zhu"], 18)
        self.assertTrue(parsed[0]["shared_from_each"])
        self.assertEqual(parsed[1]["zhu"], 18)

    def test_unparsed_reported_not_hidden(self):
        from hermes_shanghan.apps.dosimetry import parse_dose
        self.assertIn("unparsed_head", parse_dose("一錢匕"))
        self.assertIn("unparsed_head", parse_dose("如雞子大，碎"))
        self.assertNotIn("unparsed_head", parse_dose("炙"))  # processing-only


class TestDosimetryMiner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.apps.dosimetry import DosimetryMiner
        from hermes_shanghan.orchestrator import Artifacts
        art = Artifacts()
        cls.miner = DosimetryMiner(art.clauses, art.formula_rules)
        cls.table = cls.miner.dose_table()

    def test_ratio_is_school_independent(self):
        ratios = self.miner.dose_ratios(self.table)
        gz = next(f for f in ratios["formulas"] if f["formula"] == "桂枝湯")
        # ratio computed in 銖-equivalents; totals differ by school, ratio not
        self.assertIn("甘草1", gz["ratio"])
        self.assertNotEqual(gz["total_weight_g"]["kaogu"],
                            gz["total_weight_g"]["zhezhuan"])

    def test_family_dose_evolution_detects_shaoyao_doubling(self):
        evo = self.miner.family_dose_evolution(self.table)
        edge = next(e for e in evo["edges"]
                    if e["base"] == "桂枝湯" and e["modified"] == "桂枝加芍藥湯")
        self.assertEqual(edge["edge_kind"], "增減量")   # dose-only, no add/remove
        d = next(x for x in edge["dose_deltas"] if x["herb"] == "芍藥")
        self.assertEqual(d["factor"], 2.0)
        self.assertGreaterEqual(evo["n_dose_only_edges"], 1)

    def test_coverage_accounting(self):
        t = self.table
        self.assertEqual(t["n_rows"], sum(t["kind_counts"].values()))
        self.assertEqual(t["n_unparsed"], len(t["unparsed"]))
        self.assertGreaterEqual(t["kind_counts"]["weight"], 300)


class TestMultiCommentatorAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.orchestrator import Artifacts
        cls.art = Artifacts()

    def test_nine_books_aligned(self):
        books = {r.book for r in self.art.commentary_rules}
        self.assertEqual(len(books), 9)
        ids = [r.commentary_rule_id for r in self.art.commentary_rules]
        self.assertEqual(len(ids), len(set(ids)))

    def test_laisu_quote_comment_split_recovers_coverage(self):
        laisu = [r for r in self.art.commentary_rules if r.book == "傷寒來蘇集"]
        self.assertGreaterEqual(len(laisu), 200)   # was 28 before the split
        # inline commentary carries the '[' marker from the source
        self.assertTrue(any(r.commentary_text.startswith("[") for r in laisu))

    def test_clause_12_has_multiple_commentators(self):
        comms = {r.commentator for r in self.art.commentary_rules
                 if r.clause_id == "SHL_SONGBEN_0012"}
        self.assertGreaterEqual(len(comms), 4)
        self.assertIn("成無己", comms)


class TestDivergenceAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        cls.atlas = json.loads(
            (config.RESEARCH_DIR / "commentary_divergence.json")
            .read_text(encoding="utf-8"))

    def test_structure_and_coverage(self):
        a = self.atlas
        self.assertEqual(a["n_books"], 9)
        self.assertGreaterEqual(a["n_clauses_multi_commentator"], 300)
        for row in a["top_divergent_clauses"]:
            self.assertGreaterEqual(row["n_commentators"], 2)
            self.assertTrue(row["clause_text"])

    def test_zhangqingzi_chengwuji_affinity(self):
        # 張卿子傷寒論 is an edition built ON 成無己's annotations — the
        # atlas must rediscover this filiation from the data alone
        pair = next(p for p in self.atlas["agreement_matrix"]
                    if {p["a"], p["b"]} == {"張卿子", "成無己"})
        others = [p["mean_term_agreement"] for p in self.atlas["agreement_matrix"]
                  if {p["a"], p["b"]} != {"張卿子", "成無己"}]
        self.assertGreater(pair["mean_term_agreement"], 0.7)
        self.assertGreater(pair["mean_term_agreement"], max(others))

    def test_fingerprints_present(self):
        fp = self.atlas["commentator_fingerprints"]
        self.assertGreaterEqual(len(fp), 5)
        for comm, rows in fp.items():
            for r in rows:
                self.assertGreaterEqual(r["n"], 3)


class TestPaperIntegration(unittest.TestCase):
    def _generate(self, ptype, topic=""):
        _ensure_artifacts()
        from hermes_shanghan.orchestrator import Artifacts
        from hermes_shanghan.paper.writer import PaperWriter
        art = Artifacts()
        w = PaperWriter(art.clauses, art.initial_rules, art.formula_rules,
                        art.six_channel_rules, art.mistreatment_rules,
                        art.differential_rules,
                        commentary_rules=art.commentary_rules)
        with tempfile.TemporaryDirectory() as tmp:
            return w.generate(paper_type=ptype, topic=topic,
                              out_dir=Path(tmp)).read_text(encoding="utf-8")

    def test_commentary_compare_paper_has_atlas(self):
        text = self._generate("commentary_compare", topic="桂枝湯歷代注釋")
        self.assertIn("注家分歧圖譜", text)
        self.assertIn("一致度矩陣", text)
        self.assertIn("張卿子", text)

    def test_network_pharmacology_paper_has_dosimetry(self):
        text = self._generate("network_pharmacology")
        self.assertIn("劑量計量層", text)
        self.assertIn("銖當量比", text)
        self.assertIn("劑量演化", text)


if __name__ == "__main__":
    unittest.main()
