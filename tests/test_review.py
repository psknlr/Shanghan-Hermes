"""Adversarial tests of the autonomous review pipeline.

These inject deliberately broken rules and assert the pipeline rejects or
repairs them — the heart of "無原文，不成規則".
"""
import unittest

from hermes_shanghan.corpus import segmenter
from hermes_shanghan.extract.entities import EntityExtractor, annotate_clause
from hermes_shanghan.review.pipeline import ReviewPipeline
from hermes_shanghan.schemas import AutonomousReview, InitialRule


def make_rule(**kw) -> InitialRule:
    base = dict(
        initial_rule_id="IR_SHL_0012_901",
        clause_id="SHL_SONGBEN_0012",
        six_channel="太陽病",
        rule_type="formula_pattern_rule",
        if_conditions={"disease": ["太陽中風"], "symptoms": [], "pulse": []},
        then_conclusions={"formula": ["桂枝湯"]},
        evidence_span="太陽中風，陽浮而陰弱，陽浮者，熱自發，陰弱者，汗自出。"
                      "嗇嗇惡寒，淅淅惡風，翕翕發熱，鼻鳴乾嘔者，桂枝湯主之。",
        evidence_type="original_text",
        interpretation="測試規則",
        interpretation_level="normalized",
        model_confidence=0.9,
        prescription_strength="主之",
    )
    base.update(kw)
    return InitialRule(**base)


class TestReviewPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clauses = segmenter.segment_canonical()
        ex = EntityExtractor(segmenter.harvest_formula_names(clauses))
        for c in clauses:
            annotate_clause(c, ex)
        cls.store = {c.clause_id: c for c in clauses}

    def _review(self, rule):
        return ReviewPipeline(self.store).review_rule(rule)

    def test_good_rule_passes(self):
        r = self._review(make_rule())
        self.assertIn(r.autonomous_review.release_level, ("gold", "silver"))
        self.assertTrue(r.autonomous_review.evidence_verified)

    def test_fabricated_evidence_rejected(self):
        r = self._review(make_rule(
            evidence_span="太陽中風，營衛不和，桂枝湯調和營衛主之。"))  # not in text
        self.assertEqual(r.autonomous_review.release_level, "rejected")

    def test_wrong_clause_id_rejected(self):
        r = self._review(make_rule(clause_id="SHL_SONGBEN_9999"))
        self.assertEqual(r.autonomous_review.release_level, "rejected")

    def test_invented_symptom_repaired_or_rejected(self):
        r = self._review(make_rule(
            if_conditions={"disease": ["太陽中風"],
                           "symptoms": ["潮熱", "譫語"],  # not in clause 12
                           "pulse": []}))
        # repair must drop the fabricated symptoms; rule may survive without them
        self.assertNotIn("潮熱", r.if_conditions.get("symptoms", []))
        self.assertNotIn("譫語", r.if_conditions.get("symptoms", []))

    def test_formula_clause_mismatch_rejected_or_stripped(self):
        r = self._review(make_rule(then_conclusions={"formula": ["大承氣湯"]}))
        # 大承氣湯 does not appear in clause 12 — must not survive as conclusion
        self.assertNotIn("大承氣湯", r.then_conclusions.get("formula", []))

    def test_keyu_inflation_downgraded(self):
        # clause 15: 「太陽病，下之後，其氣上衝者，可與桂枝湯…」 — only 可與
        clause15 = self.store["SHL_SONGBEN_0015"]
        r = self._review(make_rule(
            clause_id="SHL_SONGBEN_0015",
            evidence_span=clause15.clean_text,
            prescription_strength="主之"))   # inflated claim
        self.assertNotEqual(r.prescription_strength, "主之")
        self.assertTrue(any("strength_downgraded" in x
                            for x in r.autonomous_review.repairs))

    def test_posthoc_term_blocked(self):
        r = self._review(make_rule(
            if_conditions={"disease": ["太陽中風", "營衛不和"], "symptoms": [], "pulse": []}))
        self.assertNotIn("營衛不和", r.if_conditions.get("disease", []))
        self.assertEqual(r.interpretation_level, "model_inference")

    def test_schema_violation_rejected(self):
        r = self._review(make_rule(rule_type="not_a_rule_type"))
        self.assertEqual(r.autonomous_review.release_level, "rejected")

    def test_contraindication_annotated(self):
        # clause 17 酒客 forbids 桂枝湯: a formula rule built on it must carry
        # the contraindication, not present 桂枝湯 unconditionally
        clause17 = self.store["SHL_SONGBEN_0017"]
        r = self._review(InitialRule(
            initial_rule_id="IR_SHL_0017_901", clause_id="SHL_SONGBEN_0017",
            six_channel="太陽病", rule_type="formula_pattern_rule",
            if_conditions={"disease": [], "symptoms": [], "pulse": []},
            then_conclusions={"formula": ["桂枝湯"]},
            evidence_span=clause17.clean_text, evidence_type="original_text",
            interpretation="x", interpretation_level="normalized",
            model_confidence=0.8, prescription_strength="與"))
        flags = r.autonomous_review.critic_flags + r.autonomous_review.repairs
        self.assertTrue(any("contraindication" in f for f in flags) or
                        r.if_conditions.get("contraindications"))


if __name__ == "__main__":
    unittest.main()
