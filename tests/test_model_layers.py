"""十六輪測試：模型增益層與段落級溯源。

1. 方證鑒別：逐格回源核驗（含否定語境誤歸類）+ 模型審校（local 確定性
   降級 / Scripted 真模型分支的引用核驗）。
2. 科研挖掘：共現網絡/頻次/家族樹/論文大綱以真實數據隨響應返回。
3. 溯源工作台：模型綜合層（local 摘要 / 真模型引用守衛，偽造編號必被
   標記）；全庫候選出處攜帶可點閱定位（book_id）。
4. 條文全息：注家智能分析（貼近度/學派/取徑）+ 歷代古籍段落級引用；
   患者端不得收到含劑量原文的引用段落。
"""
import json
import unittest

from hermes_shanghan.llm.client import LLMClient
from hermes_shanghan.llm.providers import ScriptedProvider
from hermes_shanghan.orchestrator import Artifacts
from hermes_shanghan.server.service import ServiceContext


ART = Artifacts()


def _diff_dict(*names):
    return next(x for x in ART.differential_rules
                if set(x.formulas) == set(names)).to_dict()


class TestDifferentialVerification(unittest.TestCase):
    def test_canonical_pair_fully_verified(self):
        from hermes_shanghan.apps.differential_audit import verify_differential
        d = _diff_dict("桂枝湯", "麻黃湯")
        v = verify_differential(d, ART.formula_rules, ART.clause_store())
        self.assertGreater(v["n_checked"], 10)
        self.assertEqual(v["flagged"], [],
                         "隨庫鑒別規則的表述應全部可回源")

    def test_fabricated_term_flagged(self):
        from hermes_shanghan.apps.differential_audit import verify_differential
        d = _diff_dict("桂枝湯", "麻黃湯")
        for row in d["contrast_table"]:
            if row["axis"] == "核心症狀":
                row["桂枝湯"] += "、潮熱"          # 桂枝湯條文絕無潮熱
                row["麻黃湯"] += "、汗出"          # 麻黃湯條文只有「無汗」
        v = verify_differential(d, ART.formula_rules, ART.clause_store())
        by_term = {(f["formula"], f["term"]): f["status"]
                   for f in v["flagged"]}
        self.assertEqual(by_term.get(("桂枝湯", "潮熱")), "unverified")
        self.assertIn(("麻黃湯", "汗出"), by_term)

    def test_local_model_review_mirrors_verification(self):
        from hermes_shanghan.apps.differential_audit import model_review
        d = _diff_dict("桂枝湯", "麻黃湯")
        client = LLMClient(provider=ScriptedProvider())  # available=False
        out = model_review(d, ART.formula_rules, ART.clause_store(), client)
        self.assertEqual(out["backend"], "local")
        self.assertEqual(out["verdict"], "pass")
        self.assertEqual(out["issues"], [])

    def test_scripted_model_review_guards_citations(self):
        from hermes_shanghan.apps.differential_audit import model_review
        d = _diff_dict("桂枝湯", "麻黃湯")
        support = set()
        for r in ART.formula_rules:
            if r.formula in d["formulas"]:
                support |= set(r.supporting_clauses)
        good = sorted(support)[0]
        fake = "SHL_SONGBEN_9999"
        scripted = ScriptedProvider([json.dumps({
            "verdict": "warn",
            "issues": [{"formula": "麻黃湯", "axis": "汗之有無",
                        "problem": "測試問題",
                        "clause_ids": [good, fake]}],
            "summary": f"見 {good}。"}, ensure_ascii=False)])
        client = LLMClient(provider=scripted)
        client._backend = "litellm"  # make `available` True for the test
        client.settings.cache = False    # 測試不落磁盤緩存
        out = model_review(d, ART.formula_rules, ART.clause_store(), client)
        self.assertEqual(out["verdict"], "warn")
        issue = out["issues"][0]
        self.assertIn(good, issue["clause_ids"])
        self.assertIn(fake, issue["unverified_clause_ids"],
                      "偽造編號必須被標記，不得混入已核實引用")
        self.assertTrue(out["citation_report"]["ok"])

    def test_service_differential_payload(self):
        svc = ServiceContext()
        r = svc.differential(["桂枝湯", "麻黃湯"])
        self.assertIn("verification", r)
        self.assertIn("model_review", r)
        self.assertGreater(r["verification"]["n_checked"], 0)


class TestResearchPayload(unittest.TestCase):
    def test_real_assets_in_response(self):
        svc = ServiceContext()
        r = svc.research("桂枝湯類方證演化")
        nw = r["networks"]
        self.assertGreater(len(nw["top_symptom_edges"]), 10)
        e0 = nw["top_symptom_edges"][0]
        self.assertIn("formula", e0)
        self.assertIn("weight", e0)
        self.assertEqual(nw["focus_formulas"], ["桂枝湯"])
        self.assertTrue(all(e["formula"] == "桂枝湯"
                            for e in nw["focus_edges"]))
        fq = r["frequency"]
        self.assertGreater(len(fq["symptom_frequency"]), 10)
        self.assertGreater(len(fq["channel_formula"]), 5)
        ft = r["family_tree"]
        self.assertGreater(ft["n_families"], 3)
        self.assertTrue(any(f["base"] == "桂枝湯" for f in ft["families"]))
        self.assertIn("sections", r["paper_outline"])


class TestTraceSynthesis(unittest.TestCase):
    def test_local_synthesis_deterministic(self):
        svc = ServiceContext()
        r = svc.trace("clause", "12")
        ms = r["model_synthesis"]
        self.assertEqual(ms["backend"], "local")
        self.assertIn("SHL_SONGBEN_0012", ms["synthesis"])

    def test_synthesis_can_be_disabled(self):
        svc = ServiceContext()
        r = svc.trace("clause", "12", synthesize=False)
        self.assertNotIn("model_synthesis", r)

    def test_scripted_synthesis_fabrication_flagged(self):
        svc = ServiceContext()
        scripted = ScriptedProvider(
            ["本條源流見 SHL_SONGBEN_0012 與 SHL_SONGBEN_8888。"])
        client = LLMClient(provider=scripted)
        client._backend = "litellm"
        client.settings.cache = False    # 測試不落磁盤緩存
        svc._llm = client
        r = svc.trace("clause", "12")
        ms = r["model_synthesis"]
        rep = ms["citation_report"]
        self.assertFalse(rep["ok"])
        self.assertIn("SHL_SONGBEN_8888", rep["unsupported"])
        self.assertIn("請勿採信", ms["synthesis"])

    def test_library_candidates_carry_locator(self):
        # 全庫未下載時如實聲明；已下載時 hits 必須攜帶可點閱的 book_id
        from hermes_shanghan.corpus import library
        svc = ServiceContext()
        r = svc.trace("text", "此句庫內絕無此文亦非後世歸納語測試")
        lc = r.get("library_candidates", {})
        if not library.is_available():
            self.assertFalse(lc.get("available"))
        else:
            for h in lc.get("hits", []):
                self.assertIn("book_id", h)
                self.assertIn("excerpt", h)


class TestClauseHolism(unittest.TestCase):
    def test_commentary_analysis_and_historical_citations(self):
        svc = ServiceContext()
        r = svc.explain_clause("12", role="student")
        ca = r["commentary_analysis"]
        self.assertGreaterEqual(len(ca["views"]), 5)
        v0 = ca["views"][0]
        for key in ("commentator", "dynasty", "closeness_to_original",
                    "analytic_focus"):
            self.assertIn(key, v0)
        hc = r["historical_citations"]
        self.assertGreater(hc["n_books"], 10)
        self.assertGreater(hc["n_edges"], 50)
        dyn = [d["dynasty"] for d in hc["by_dynasty"]]
        self.assertIn("宋", dyn)
        self.assertIn("清", dyn)
        p0 = hc["by_dynasty"][0]["books"][0]["passages"][0]
        self.assertIn("mode", p0)
        self.assertTrue(p0.get("excerpt") or p0.get("matched_span"))

    def test_patient_role_gets_no_dose_bearing_passages(self):
        from hermes_shanghan.server import policy
        svc = ServiceContext()
        r = svc.explain_clause("12", role="patient")
        self.assertNotIn("historical_citations", r)
        self.assertNotIn("commentary_analysis", r)
        # 序列化出口投影兜底：即便業務層忘了，鍵也會被強制移除
        projected = policy.project_for_role(
            {"historical_citations": {"x": 1}, "text": "y"}, "patient")
        self.assertNotIn("historical_citations", projected)

    def test_passages_cache_roundtrip(self):
        from hermes_shanghan.trace import passages
        r1 = passages.clause_citing_passages("SHL_SONGBEN_0012")
        passages.invalidate_cache()
        r2 = passages.clause_citing_passages("SHL_SONGBEN_0012")
        self.assertEqual(r1["n_edges"], r2["n_edges"])
        self.assertEqual(r1["n_books"], r2["n_books"])

    def test_uncited_clause_reports_empty(self):
        from hermes_shanghan.trace import passages
        out = passages.clause_citing_passages("SHL_SONGBEN_NOPE")
        self.assertEqual(out["n_edges"], 0)
        self.assertEqual(out["by_dynasty"], [])


if __name__ == "__main__":
    unittest.main()
