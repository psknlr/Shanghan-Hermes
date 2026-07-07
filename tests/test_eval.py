"""Evaluation-suite tests: prescription-cloze (LOCO), case replay,
grounding metrics, ablation flags, and the benchmark paper type."""
import json
import tempfile
import unittest
from pathlib import Path

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


def _ensure_eval():
    if not (config.SHANGHAN_DIR / "eval" / "cloze_results.json").exists():
        from hermes_shanghan.eval.runner import run_suites
        run_suites(suites=("cloze", "cases", "grounding"), verbose=False)


class TestClozeBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.eval.cloze import ClozeBenchmark
        from hermes_shanghan.orchestrator import Artifacts
        art = Artifacts()
        cls.bench = ClozeBenchmark(art.clauses, art.initial_rules)

    def test_loco_no_leakage(self):
        # the held-out clause's rules must be absent from the fold rule set
        from hermes_shanghan.eval.cloze import build_instances
        instances, _ = build_instances(self.bench.initial_rules)
        cid = instances[0]["clause_id"]
        fold = self.bench._loco_rules(cid)
        for r in fold:
            self.assertNotIn(cid, r.supporting_clauses,
                             "held-out clause leaked into LOCO rule set")

    def test_metrics_structure_and_monotonicity(self):
        res = self.bench.run(limit=30)
        m = res["metrics"]
        self.assertEqual(m["all"]["n"],
                         m["attainable"]["n"] + m["singleton_unattainable"]["n"])
        for split in ("all", "attainable"):
            self.assertLessEqual(m[split]["top1"], m[split]["top3"])
            self.assertLessEqual(m[split]["top3"], m[split]["top5"])
        # singleton folds are impossible by construction
        self.assertEqual(m["singleton_unattainable"]["top5"], 0.0)

    def test_deterministic(self):
        a = self.bench.run(limit=15)
        b = self.bench.run(limit=15)
        self.assertEqual(a["metrics"], b["metrics"])


class TestCaseBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_parse_jingfang_shiyanlu(self):
        from hermes_shanghan.eval.cases import parse_cases
        from hermes_shanghan.extract.entities import EntityExtractor
        cases, non_formula = parse_cases(EntityExtractor())
        self.assertGreaterEqual(len(cases), 70)
        self.assertGreaterEqual(non_formula, 15)
        first = next(c for c in cases if c["title"].startswith("第一案"))
        self.assertEqual(first["gold"], "桂枝湯")
        # attribution markup (<z>穎師醫案</z>) must not pollute gold labels
        self.assertFalse(any("穎師" in c["gold"] or "佐景" in c["gold"]
                             for c in cases))

    def test_replay_metrics(self):
        from hermes_shanghan.eval.cases import CaseBenchmark
        from hermes_shanghan.orchestrator import Artifacts
        art = Artifacts()
        res = CaseBenchmark(art.formula_rules, art.clause_store()).run()
        self.assertGreaterEqual(res["metrics"]["n_scored"], 30)
        # honesty accounting adds up
        self.assertEqual(res["n_cases_parsed"],
                         res["metrics"]["n_scored"] + res["n_out_of_scope"]
                         + res["n_insufficient_findings"])
        self.assertLessEqual(res["metrics"]["top1"], res["metrics"]["top5"])


class TestGroundingBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_local_backend_fully_grounded(self):
        from hermes_shanghan.eval.grounding import (GroundingBenchmark,
                                                    build_question_bank)
        from hermes_shanghan.orchestrator import Artifacts
        art = Artifacts()
        qs = build_question_bank(art.formula_rules, art.six_channel_rules)
        self.assertGreaterEqual(len(qs), 20)
        res = GroundingBenchmark().run(qs, limit=6)
        self.assertEqual(res["backend"], "local")
        self.assertEqual(res["metrics"]["grounded_answer_rate"], 1.0)
        self.assertEqual(res["metrics"]["unsupported_citation_rate"], 0.0)


class TestAblationFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.orchestrator import Artifacts
        cls.art = Artifacts()

    def test_outline_boost_flag_changes_scoring(self):
        from hermes_shanghan.apps.doctor import FormulaMatcher
        store = self.art.clause_store()
        on = FormulaMatcher(self.art.formula_rules, store)
        off = FormulaMatcher(self.art.formula_rules, store,
                             use_outline_boost=False)
        q = ["往來寒熱", "胸脅苦滿", "口苦"]
        hit_on = on.match(q)["matched_formula_patterns"][0]
        hit_off = off.match(q)["matched_formula_patterns"][0]
        self.assertTrue(any(h.startswith("提綱證")
                            for h in hit_on["matched_findings"]))
        self.assertFalse(any(h.startswith("提綱證")
                             for h in hit_off["matched_findings"]))


class TestBenchmarkPaper(unittest.TestCase):
    def test_benchmark_paper_includes_eval_tables(self):
        _ensure_artifacts()
        _ensure_eval()
        from hermes_shanghan.orchestrator import Artifacts
        from hermes_shanghan.paper.writer import PaperWriter
        art = Artifacts()
        w = PaperWriter(art.clauses, art.initial_rules, art.formula_rules,
                        art.six_channel_rules, art.mistreatment_rules,
                        art.differential_rules,
                        commentary_rules=art.commentary_rules)
        with tempfile.TemporaryDirectory() as tmp:
            path = w.generate(paper_type="benchmark", out_dir=Path(tmp))
            text = path.read_text(encoding="utf-8")
        self.assertIn("客觀評測結果", text)
        self.assertIn("遮方預測（留一條文，自監督）", text)
        self.assertIn("醫案回放", text)
        self.assertIn("完全接地率", text)
        # the augmentation layer interprets the benchmark numbers
        self.assertIn("遮方預測基準", text.split("計量結果解讀")[1])


if __name__ == "__main__":
    unittest.main()
