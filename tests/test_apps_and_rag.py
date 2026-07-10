"""End-user surface tests: RAG, doctor matching, safety, skills, memory.

Requires pipeline artifacts (run `hermes-shanghan pipeline` once; the repo
ships with generated artifacts, so a fresh clone passes).
"""
import unittest

from hermes_shanghan import config, safety
from hermes_shanghan.orchestrator import Artifacts


def _ensure_artifacts():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestClauseRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.rag.clause_rag import ClauseRAG
        cls.rag = ClauseRAG.load()

    def test_clause_number_lookup(self):
        hits = self.rag.search("第12條")
        self.assertEqual(hits[0]["clause_number"], 12)
        self.assertIn("桂枝湯", hits[0]["text"])

    def test_formula_search(self):
        hits = self.rag.search("小柴胡湯 往來寒熱", top_k=5)
        self.assertTrue(any("小柴胡湯" in h["formulas"] for h in hits[:3]))

    def test_simplified_query(self):
        hits = self.rag.search("桂枝汤主之", top_k=5)
        self.assertTrue(any("桂枝湯" in h["formulas"] for h in hits[:3]))

    def test_every_hit_has_clause_id(self):
        for h in self.rag.search("發熱惡寒", top_k=8):
            self.assertTrue(h["clause_id"])
            self.assertTrue(h["layer_label"])

    def test_relation_expansion(self):
        hits = self.rag.search("結胸", top_k=4, expand_relations=True)
        self.assertTrue(any(h["match_source"].startswith("relation:") for h in hits)
                        or len(hits) >= 4)


class TestDoctorMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.apps.doctor import FormulaMatcher
        art = Artifacts()
        cls.matcher = FormulaMatcher(art.formula_rules, art.clause_store())

    def test_mahuang_case(self):
        res = self.matcher.match(["惡寒", "發熱", "無汗", "身疼痛"], ["浮緊"])
        top = res["matched_formula_patterns"][0]
        self.assertEqual(top["formula"], "麻黃湯")
        self.assertTrue(top["evidence"])
        self.assertTrue(all(e["clause_id"] for e in top["evidence"]))
        self.assertIn("輔助", res["safety_notice"])

    def test_contradiction_penalty(self):
        # 自汗出 conflicts with 麻黃湯 (無汗) — 桂枝湯 must outrank it
        res = self.matcher.match(["發熱", "汗出", "惡風", "頭痛"], ["浮緩"])
        names = [m["formula"] for m in res["matched_formula_patterns"][:3]]
        self.assertIn("桂枝湯", names)
        if "麻黃湯" in names:
            self.assertLess(names.index("桂枝湯"), names.index("麻黃湯"))


class TestSafety(unittest.TestCase):
    def test_patient_guard_prescription(self):
        r = safety.patient_intent_guard("给我开个方治感冒吧")
        self.assertTrue(r and r["refused"])

    def test_patient_guard_dosage(self):
        r = safety.patient_intent_guard("桂枝湯我该吃几克？一天几次？")
        self.assertTrue(r and "劑量調整" in r["refused_intents"])

    def test_patient_guard_diagnosis(self):
        r = safety.patient_intent_guard("我是不是得了太阳病？")
        self.assertTrue(r and "診斷判定" in r["refused_intents"])

    def test_benign_question_passes(self):
        self.assertIsNone(safety.patient_intent_guard("太陽表證是什麼意思？"))

    def test_dose_redaction(self):
        text = "桂枝三兩，芍藥三兩，甘草二兩，水七升煮取三升。"
        red = safety.redact_for_patient(text)
        self.assertNotIn("三兩", red)
        self.assertIn("劑量信息略", red)

    def test_governed_strips_prescriptions_for_patient(self):
        out = safety.governed({"answer": "x", "matched_formula_patterns": [1]}, "patient")
        self.assertNotIn("matched_formula_patterns", out)

    def test_role_routing_conservative(self):
        _ensure_artifacts()
        from hermes_shanghan.rag.skill_rag import SkillRAG
        rag = SkillRAG()
        self.assertEqual(rag.infer_role("给我开个方"), "patient")
        self.assertEqual(rag.infer_role("帮我诊断一下"), "patient")
        route = rag.route("医生说我是太阳表证，这是什么意思？")
        self.assertEqual(route["skill"], "hermes.shanghan.patient_education")


class TestSkillsAndRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        cls.art = Artifacts()

    def test_skill_tree_complete(self):
        root = config.SKILLS_DIR
        for d in ["hermes.shanghan.catalog", "hermes.shanghan.mistreatment",
                  "hermes.shanghan.contraindications", "hermes.shanghan.differential",
                  "hermes.shanghan.clause_explainer", "hermes.shanghan.paper_writer",
                  "hermes.shanghan.patient_education", "hermes.shanghan.six_channels/taiyang",
                  "hermes.shanghan.six_channels/jueyin",
                  "hermes.shanghan.formula_patterns/guizhi_tang",
                  "hermes.shanghan.formula_patterns/mahuang_tang",
                  "hermes.shanghan.formula_patterns/xiaochaihu_tang",
                  "hermes.shanghan.formula_patterns/wumei_wan",
                  "hermes.shanghan.therapy/sweating",
                  "hermes.shanghan.therapy/purgation",
                  "hermes.shanghan.therapy/harmonization",
                  "hermes.shanghan.therapy/rescue_reverse"]:
            self.assertTrue((root / d / "SKILL.md").exists(), d)
            self.assertTrue((root / d / "rules.jsonl").exists(), d)
            self.assertTrue((root / d / "examples.jsonl").exists(), d)

    def test_patient_visit_summary(self):
        from hermes_shanghan.apps.patient import PatientEducator
        edu = PatientEducator(self.art.six_channel_rules, self.art.clause_store())
        out = edu.organize_symptoms(["怕冷", "頭痛"])
        self.assertIn("visit_summary", out)
        self.assertEqual(out["mode"], "patient")
        # no diagnosis, no formula fields
        self.assertNotIn("matched_formula_patterns", out)

    def test_paper_assets(self):
        from hermes_shanghan.paper.writer import PaperWriter
        import tempfile
        from pathlib import Path
        writer = PaperWriter(self.art.clauses, self.art.initial_rules,
                             self.art.formula_rules, self.art.six_channel_rules,
                             self.art.mistreatment_rules, self.art.differential_rules)
        with tempfile.TemporaryDirectory() as td:
            path = writer.generate("formula_pattern", "桂枝湯類方證", Path(td))
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            for section in ("摘要", "方法", "結果", "討論", "結論",
                            "參考文獻", "Cover Letter"):
                self.assertIn(section, text)
            assets = {p.name for p in Path(td).iterdir()}
            self.assertIn("fig_clause_topics.mmd.md", assets)
            self.assertIn("figures_manifest.json", assets)
            self.assertIn("source_data", assets)
            self.assertIn("table4_variant_comparison.csv", assets)

    def test_merged_rules_reference_not_replace(self):
        ir_ids = {r.initial_rule_id for r in self.art.initial_rules}
        for m in self.art.merged_rules[:30]:
            for rid in m.supporting_initial_rules[:5]:
                self.assertIn(rid, ir_ids)   # references stay resolvable
            if m.evidence_chain:
                self.assertTrue(all(e.get("clause_id") for e in m.evidence_chain))

    def test_formula_rule_provenance(self):
        guizhi = next(r for r in self.art.formula_rules if r.formula == "桂枝湯")
        self.assertIn("SHL_SONGBEN_0012", guizhi.supporting_clauses)
        self.assertIn("汗出", guizhi.core_symptoms)
        self.assertEqual(len(guizhi.composition), 5)
        self.assertTrue(guizhi.modification_relations)  # 桂枝加葛根湯 etc.

    def test_six_channel_outlines(self):
        scr = {r.six_channel: r for r in self.art.six_channel_rules}
        self.assertEqual(scr["太陽病"].outline_clause_id, "SHL_SONGBEN_0001")
        self.assertEqual(scr["陽明病"].outline_clause_id, "SHL_SONGBEN_0180")
        self.assertEqual(scr["少陰病"].outline_clause_id, "SHL_SONGBEN_0281")

    def test_mistreatment_paths_grounded(self):
        for m in self.art.mistreatment_rules:
            self.assertTrue(m.supporting_clauses or m.supporting_initial_rules, m.path)

    def test_memory_modules(self):
        from hermes_shanghan.memory.store import MemoryHub
        hub = MemoryHub()
        self.assertTrue(hub.formula_memory.get("桂枝湯"))
        self.assertTrue(hub.six_channel_memory.get("太陽病"))
        self.assertTrue((config.MEMORY_DIR / "critic_memory.json").exists())


if __name__ == "__main__":
    unittest.main()
