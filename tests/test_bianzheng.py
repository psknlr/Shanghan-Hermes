"""方證辨證閉環（B 組）測試：四診採集/多假設裁決/衝突審計/誤治模擬。"""
import unittest

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestIntake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_modern_narrative_structured(self):
        from hermes_shanghan.apps.bianzheng import intake_parse
        r = intake_parse("发热，怕冷，出汗，头痛，二三日，服退烧药后腹泻")
        self.assertIn("惡寒", r["cold_heat"])       # 怕冷 → 惡寒（映射透明）
        self.assertIn("汗出", r["sweating"])        # 出汗 → 汗出
        self.assertIn("下利", r["stool_urine"])     # 腹泻 → 下利
        self.assertIn("二三日", r["timeline"])
        self.assertTrue(r["medication_response"])   # 服藥後反應被捕獲
        self.assertIn("pulse", r["missing_key_findings"])
        self.assertTrue(r["next_questions"])
        self.assertIn("不構成診斷", r["note"])

    def test_missing_axes_generate_questions(self):
        from hermes_shanghan.apps.bianzheng import intake_parse
        r = intake_parse("頭項強痛")
        self.assertIn("cold_heat", r["missing_key_findings"])
        self.assertTrue(any("惡寒" in q or "汗" in q for q in r["next_questions"]))

    def test_intake_patient_safe_tool(self):
        from hermes_shanghan.agent.tools import get_registry
        reg = get_registry()
        scoped = reg.for_role("patient")
        self.assertIn("shanghan_intake", scoped.names())
        out = scoped.call("shanghan_intake", {"text": "怕冷發熱無汗"})
        self.assertIn("惡寒", out["cold_heat"])
        # 患者端其餘三個辨證工具不暴露
        for t in ("shanghan_adjudicate", "shanghan_conflict_audit",
                  "shanghan_mistreatment_simulate"):
            self.assertNotIn(t, scoped.names())


class TestAdjudicate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_three_state_verdict(self):
        out = self.reg.call("shanghan_adjudicate",
                            {"symptoms": ["發熱", "惡寒", "無汗", "身疼痛"],
                             "pulse": ["浮緊"]})
        self.assertIn("verdict", out)
        self.assertTrue(out["verdict"].startswith(("傾向", "不能裁決")))
        self.assertTrue(out["rationale"])
        self.assertLessEqual(len(out["key_questions"]), 3)
        # 每個候選附禁忌衝突檢查
        for h in out["candidates"]:
            self.assertIn("contraindication_hits", h)

    def test_insufficient_evidence(self):
        out = self.reg.call("shanghan_adjudicate", {"symptoms": ["某不存在證"]})
        self.assertEqual(out["verdict"], "不能裁決")


class TestConflictAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_guizhi_vs_wuhan(self):
        # 評審典型用例：桂枝湯證 vs 無汗
        out = self.reg.call("shanghan_conflict_audit",
                            {"formula": "桂枝湯", "symptoms": ["無汗", "發熱"]})
        conflict = next(c for c in out["conflicts"] if c["presented"] == "無汗")
        self.assertEqual(conflict["strength"], "核心證衝突")
        self.assertTrue(conflict["supporting_clauses"])
        self.assertIn("高", out["severity"])
        # 改判候選含以無汗為核心證的方（如麻黃湯類）
        self.assertTrue(out["reassign_candidates"])

    def test_mahuang_vs_zihan(self):
        out = self.reg.call("shanghan_conflict_audit",
                            {"formula": "麻黃湯", "symptoms": ["自汗出"]})
        self.assertTrue(out["conflicts"])

    def test_no_conflict_clean(self):
        out = self.reg.call("shanghan_conflict_audit",
                            {"formula": "桂枝湯", "symptoms": ["汗出", "惡風"]})
        self.assertEqual(out["conflicts"], [])
        self.assertEqual(out["severity"], "無衝突")


class TestMistreatmentSimulate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_taiyang_wuxia_branches(self):
        out = self.reg.call("shanghan_mistreatment_simulate",
                            {"channel": "太陽病", "mistreatment": "誤下"})
        self.assertGreater(out["n_branches"], 5)
        p0 = out["paths"][0]
        self.assertEqual(p0["path"][0], "太陽病")
        self.assertTrue(p0["supporting_clauses"])   # 逐條錨定原文

    def test_multi_step_marked_hypothetical(self):
        out = self.reg.call("shanghan_mistreatment_simulate",
                            {"channel": "太陽病", "mistreatment": "誤汗",
                             "steps": 2})
        self.assertIn("further_steps", out)
        self.assertIn("非原文直述", out["further_steps"]["note"])

    def test_unknown_type_lists_available(self):
        out = self.reg.call("shanghan_mistreatment_simulate",
                            {"channel": "太陽病", "mistreatment": "誤灸不存在"})
        self.assertIn("available_types", out)


class TestTeachingCase(unittest.TestCase):
    """二十輪：誤治傳變 → 教學案例（確定性骨架 + 條文逐字回源）。"""

    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.server.service import ServiceContext
        cls.svc = ServiceContext()

    def test_case_skeleton_anchored(self):
        out = self.svc.teaching_case("誤下", "結胸", use_llm=False)
        self.assertNotIn("error", out)
        case = out["case"]
        # 虛構教學情景須明示，不冒充真實病案
        self.assertIn("【教學案例·虛構】", case["scenario"])
        self.assertTrue(case["teaching_points"])
        self.assertEqual(len(case["discussion_questions"]), 3)
        # 證據條文逐字回源（clause_id + 原文都在）
        self.assertTrue(out["evidence"])
        for e in out["evidence"]:
            self.assertTrue(e["clause_id"].startswith("SHL_SONGBEN"))
            self.assertTrue(e["text"])
        self.assertIn("不構成診療建議", out["note"])

    def test_unknown_path_lists_available(self):
        out = self.svc.teaching_case("不存在的誤治", use_llm=False)
        self.assertIn("error", out)
        self.assertTrue(out["available_types"])

    def test_filter_by_resulting_pattern(self):
        out = self.svc.teaching_case("誤下", "痞", use_llm=False)
        if "error" not in out:      # 規則庫中存在該變證時必須命中對應規則
            self.assertIn("痞", out["resulting_pattern"])


if __name__ == "__main__":
    unittest.main()
