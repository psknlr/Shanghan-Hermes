"""Negation-aware entity extraction tests."""
import unittest

from hermes_shanghan.extract.entities import EntityExtractor


class TestEntities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = EntityExtractor()

    def test_negation_trap(self):
        # 「不惡寒」 must NOT register positive 惡寒
        res = self.ex.extract("太陽病，發熱而渴，不惡寒者，為溫病。")
        self.assertIn("不惡寒", res.symptoms)       # canonical negated form
        self.assertNotIn("惡寒", res.symptoms)
        self.assertIn("渴", res.symptoms)
        self.assertIn("溫病", res.disease_patterns)

    def test_wuhan_first_class(self):
        res = self.ex.extract("頭痛發熱，身疼腰痛，骨節疼痛，惡風，無汗而喘者。")
        self.assertIn("無汗而喘", res.symptoms)
        self.assertNotIn("汗出", res.symptoms)
        self.assertIn("惡風", res.symptoms)

    def test_pulse(self):
        res = self.ex.extract("太陽中風，陽浮而陰弱。")
        self.assertIn("陽浮而陰弱", res.pulse)
        res2 = self.ex.extract("脈浮緊者，麻黃湯主之。")
        self.assertTrue(any("浮緊" in p for p in res2.pulse))

    def test_formula_strength(self):
        res = self.ex.extract("嗇嗇惡寒，淅淅惡風，翕翕發熱，鼻鳴乾嘔者，桂枝湯主之。")
        m = [x for x in res.formula_mentions if x["name"] == "桂枝湯"][0]
        self.assertEqual(m["strength"], "主之")
        res2 = self.ex.extract("其氣上衝者，可與桂枝湯。")
        m2 = [x for x in res2.formula_mentions if x["name"] == "桂枝湯"][0]
        self.assertEqual(m2["strength"], "可與")

    def test_negated_formula(self):
        res = self.ex.extract("若酒客病，不可與桂枝湯，得之則嘔。")
        self.assertIn("桂枝湯", res.contraindicated_formulas)
        self.assertNotIn("桂枝湯", res.formulas)

    def test_mistreatment_and_outcome(self):
        res = self.ex.extract("太陽病，醫反下之，因爾腹滿時痛者。")
        self.assertIn("誤下", res.mistreatment_types)
        res2 = self.ex.extract("火逆下之，因燒針煩躁者。")
        self.assertIn("火逆", res2.mistreatment_types)

    def test_simplified_query_normalization(self):
        from hermes_shanghan.textutil import s2t
        self.assertEqual(s2t("恶寒发热无汗"), "惡寒發熱無汗")
        self.assertEqual(s2t("桂枝汤"), "桂枝湯")
        self.assertEqual(s2t("误下"), "誤下")


if __name__ == "__main__":
    unittest.main()
