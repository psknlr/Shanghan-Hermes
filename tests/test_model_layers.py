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


# ---------------------------------------------------------------------------
# 十七輪：歷代引用點閱 · 爭議文本檢索 · 藥檔分頁 · 注文出處 · 辨證模型層
# ---------------------------------------------------------------------------
class TestBookCitingPassages(unittest.TestCase):
    def test_passages_paginated(self):
        from hermes_shanghan.trace import passages
        p1 = passages.book_citing_passages("傷寒來蘇集",
                                           ["SHL_SONGBEN_0012"], limit=1)
        self.assertGreaterEqual(p1["n_passages"], 1)
        e0 = p1["passages"][0]
        for key in ("mode", "chapter", "clause_id", "excerpt"):
            self.assertIn(key, e0)
        if p1["has_more"]:
            p2 = passages.book_citing_passages(
                "傷寒來蘇集", ["SHL_SONGBEN_0012"], offset=1, limit=1)
            self.assertNotEqual(p1["passages"][0]["excerpt"],
                                p2["passages"][0]["excerpt"])

    def test_formula_citations_carry_locators(self):
        svc = ServiceContext()
        r = svc.trace("formula", "桂枝湯", synthesize=False)
        cit = r["citations_of_clauses"]
        self.assertGreater(len(cit["cited_clause_ids"]), 5)
        b0 = cit["by_dynasty"][0]["books"][0]
        self.assertIn("book_dir", b0)
        p = svc.trace_passages(b0["book_dir"], cit["cited_clause_ids"],
                               limit=3)
        self.assertGreater(p["n_passages"], 0)
        self.assertLessEqual(len(p["passages"]), 3)


class TestDisputeTextResolution(unittest.TestCase):
    def test_text_sentence_resolves_to_clause(self):
        svc = ServiceContext()
        r = svc.trace("dispute", "觀其脈證，知犯何逆，隨證治之",
                      synthesize=False)
        self.assertEqual(r["clause"]["clause_id"], "SHL_SONGBEN_0016")
        self.assertIn("resolved_from_text", r)
        self.assertGreater(r["n_commentators"], 0)

    def test_number_still_works_without_resolution_note(self):
        svc = ServiceContext()
        r = svc.trace("dispute", "12", synthesize=False)
        self.assertEqual(r["clause"]["clause_id"], "SHL_SONGBEN_0012")
        self.assertNotIn("resolved_from_text", r)

    def test_garbage_text_errors_honestly(self):
        svc = ServiceContext()
        r = svc.trace("dispute", "quantum blockchain 42", synthesize=False)
        self.assertIn("error", r)


class TestHerbPagination(unittest.TestCase):
    def test_clause_pages_disjoint(self):
        from hermes_shanghan.agent.tools import get_registry
        reg = get_registry()
        p1 = reg.call("shanghan_herb_profile",
                      {"herb": "桂枝", "clause_offset": 0, "clause_limit": 5})
        p2 = reg.call("shanghan_herb_profile",
                      {"herb": "桂枝", "clause_offset": 5, "clause_limit": 5})
        self.assertEqual(len(p1["clause_ids"]), 5)
        self.assertTrue(p1["clauses_has_more"])
        self.assertFalse(set(p1["clause_ids"]) & set(p2["clause_ids"]))
        self.assertEqual(p1["n_clauses"], p2["n_clauses"])

    def test_bencao_layer_honest_or_locatable(self):
        from hermes_shanghan.apps.herbal import bencao_evidence
        from hermes_shanghan.corpus import library
        bc = bencao_evidence("桂枝", limit=2)
        if library.is_available():
            self.assertTrue(bc["available"])
            for e in bc["excerpts"]:
                self.assertIn("book_id", e)
        else:
            self.assertFalse(bc["available"])


class TestCommentaryProvenance(unittest.TestCase):
    def test_explain_clause_commentaries_have_book_chapter(self):
        svc = ServiceContext()
        r = svc.explain_clause("12", role="student")
        self.assertGreater(len(r["commentaries"]), 3)
        for c in r["commentaries"]:
            self.assertTrue(c["book"], "注文必須帶書名")
            self.assertTrue(c["chapter"], "注文必須帶章節")

    def test_dispute_views_have_chapter(self):
        svc = ServiceContext()
        r = svc.trace("dispute", "12", synthesize=False)
        self.assertTrue(all(v.get("book") for v in r["views"]))
        self.assertTrue(any(v.get("chapter") for v in r["views"]))


class TestBianzhengModelLayer(unittest.TestCase):
    NARRATIVE = "发热，怕冷，没有汗，头痛，身上疼，脉浮紧"

    def test_intake_local_unchanged(self):
        svc = ServiceContext()
        r = svc.intake(self.NARRATIVE)
        self.assertNotIn("model_extraction", r)
        self.assertIn("惡寒", r["cold_heat"])

    def test_intake_model_findings_verified_against_narrative(self):
        svc = ServiceContext()
        scripted = ScriptedProvider([json.dumps({
            "findings": ["頭痛", "潮熱譫語"],   # 後者敘述中無依據
            "pulse": ["浮緊"],
            "notes": "測試"}, ensure_ascii=False)])
        client = LLMClient(provider=scripted)
        client._backend = "litellm"
        client.settings.cache = False
        svc._llm = client
        r = svc.intake(self.NARRATIVE)
        mx = r["model_extraction"]
        self.assertIn("潮熱譫語", mx["unverified"],
                      "敘述中無依據的模型抽取必須被攔下")
        self.assertNotIn("潮熱譫語", mx["added_findings"])

    def test_adjudicate_local_review(self):
        svc = ServiceContext()
        r = svc.adjudicate(["發熱", "惡寒", "無汗", "身疼痛"],
                           pulse=["浮緊"])
        mr = r["model_review"]
        self.assertEqual(mr["backend"], "local")
        self.assertIn("assessment", mr)

    def test_adjudicate_scripted_review_guards_citations(self):
        svc = ServiceContext()
        base = svc.adjudicate(["發熱", "惡寒", "無汗", "身疼痛"],
                              pulse=["浮緊"], use_llm=False)
        allowed = svc._report_clause_ids(base)
        good = allowed[0] if allowed else "SHL_SONGBEN_0035"
        scripted = ScriptedProvider([json.dumps({
            "agrees_with_verdict": False,
            "assessment": f"依 {good} 當考慮麻黃湯。",
            "missed_patterns": [{"formula": "大青龍湯", "reason": "測試",
                                 "clause_ids": [good, "SHL_SONGBEN_9998"]}],
            "additional_questions": ["有無煩躁？"]}, ensure_ascii=False)])
        client = LLMClient(provider=scripted)
        client._backend = "litellm"
        client.settings.cache = False
        svc._llm = client
        r = svc.adjudicate(["發熱", "惡寒", "無汗", "身疼痛"],
                           pulse=["浮緊"])
        mr = r["model_review"]
        self.assertFalse(mr["agrees_with_verdict"])
        mp = mr["missed_patterns"][0]
        self.assertIn("SHL_SONGBEN_9998", mp["unverified_clause_ids"])
        self.assertNotIn("SHL_SONGBEN_9998", mp["clause_ids"])
        self.assertEqual(mr["additional_questions"], ["有無煩躁？"])


# ---------------------------------------------------------------------------
# 十八輪：關係目標點閱 · 練習題引擎 · 簡繁映射 · 條文智能體問答
# ---------------------------------------------------------------------------
class TestSourcePassage(unittest.TestCase):
    def test_paragraph_ref(self):
        svc = ServiceContext()
        r = svc.source_passage("註解傷寒論", "p1282")
        self.assertNotIn("error", r)
        self.assertTrue(r["paragraphs"][0]["text"])
        self.assertTrue(r["chapter"])

    def test_chapter_ref(self):
        svc = ServiceContext()
        r = svc.source_passage("傷寒雜病論_桂本", "辨少陽病脈證並治")
        self.assertNotIn("error", r)
        self.assertGreater(len(r["paragraphs"]), 2)

    def test_errors_honest(self):
        svc = ServiceContext()
        self.assertIn("error", svc.source_passage("不存在的書", "p1"))
        bad = svc.source_passage("註解傷寒論", "p999999")
        self.assertIn("error", bad)
        miss = svc.source_passage("註解傷寒論", "不存在的章節名")
        self.assertIn("available_chapters", miss)


class TestQuizEngine(unittest.TestCase):
    def test_bank_multi_type_and_grounded(self):
        svc = ServiceContext()
        r = svc.quiz(channel="太陽", n=8, seed=1)
        self.assertEqual(r["backend"], "bank")
        self.assertGreaterEqual(len(r["types_present"]), 3)
        store = ART.clause_store()
        for q in r["questions"]:
            if q.get("options"):
                self.assertIn(q["answer"], q["options"],
                              "選擇題答案必須在選項中")
            if q.get("evidence_clause"):
                self.assertIn(q["evidence_clause"], store,
                              "證據條文必須真實存在")

    def test_seed_changes_batch(self):
        svc = ServiceContext()
        a = [q["question"] for q in svc.quiz("太陽", n=8, seed=1)["questions"]]
        b = [q["question"] for q in svc.quiz("太陽", n=8, seed=2)["questions"]]
        self.assertNotEqual(a, b)
        # 同 seed 確定性
        a2 = [q["question"] for q in svc.quiz("太陽", n=8, seed=1)["questions"]]
        self.assertEqual(a, a2)

    def test_model_quiz_local_fallback(self):
        svc = ServiceContext()
        r = svc.quiz(channel="太陽", n=5, use_llm=True)
        self.assertEqual(r["backend"], "local")
        self.assertGreater(r["n"], 0)

    def test_model_quiz_rejects_out_of_pool_evidence(self):
        from hermes_shanghan.apps.quiz import QuizBuilder, model_quiz
        qb = QuizBuilder(ART.clauses, ART.six_channel_rules,
                         ART.formula_rules, ART.mistreatment_rules,
                         ART.differential_rules)
        scr = qb.scrs["太陽病"]
        good = scr.outline_clause_id
        scripted = ScriptedProvider([json.dumps({"questions": [
            {"type": "選擇", "question": "合規題", "options": ["A", "B"],
             "answer": "A", "evidence_clause": good},
            {"type": "選擇", "question": "越池題", "options": ["A", "B"],
             "answer": "A", "evidence_clause": "SHL_SONGBEN_0999"},
            {"type": "選擇", "question": "答案不在選項", "options": ["A", "B"],
             "answer": "C", "evidence_clause": good},
        ]}, ensure_ascii=False)])
        client = LLMClient(provider=scripted)
        client._backend = "litellm"
        client.settings.cache = False
        out = model_quiz(qb, client, channel="太陽", n=5)
        self.assertEqual(out["n"], 1, "只有合規題可以進卷")
        reasons = [x["reject_reason"] for x in out["rejected_questions"]]
        self.assertTrue(any("不在給定條文集" in r for r in reasons))
        self.assertTrue(any("不在選項" in r for r in reasons))


class TestCharmapAndT2S(unittest.TestCase):
    def test_t2s_domain(self):
        from hermes_shanghan.textutil import t2s
        self.assertEqual(t2s("傷寒論"), "伤寒论")
        self.assertEqual(t2s("觀其脈證，知犯何逆，隨證治之"),
                         "观其脉证，知犯何逆，随证治之")

    def test_charmap_endpoint_payload(self):
        svc = ServiceContext()
        cm = svc.charmap()
        self.assertGreater(len(cm["t2s"]), 200)
        self.assertEqual(cm["t2s"].get("傷"), "伤")
        self.assertIn("繁體為準", cm["note"])


class TestClauseAgentQA(unittest.TestCase):
    def test_agent_resolves_and_cites_clause(self):
        svc = ServiceContext()
        r = svc.agent("請解讀第271條的辨證要點", role="student")
        rep = r["citation_report"]
        self.assertTrue(rep["ok"])
        self.assertIn("SHL_SONGBEN_0271", rep["verified"])


# ---------------------------------------------------------------------------
# 十九輪：簡繁檢索修復 · 主題感知挖掘 · 長論文與導出 · 點閱端點 · 勘誤 ·
#        推薦處方列表 · 口語映射最長優先
# ---------------------------------------------------------------------------
class TestJijiSearchFix(unittest.TestCase):
    def test_fold_unifies_ji(self):
        from hermes_shanghan.textutil import fold_variants, normalize_query
        self.assertEqual(fold_variants(normalize_query("项背强几几")),
                         "項背強几几")

    def test_short_query_resolves(self):
        from hermes_shanghan.trace.chains import text_trace
        r = text_trace("项背强几几")
        ids = [m["clause_id"] for m in r.get("matches", [])]
        self.assertIn("SHL_SONGBEN_0031", ids)
        self.assertIn("SHL_SONGBEN_0014", ids)


class TestTopicAwareResearch(unittest.TestCase):
    def test_scoped_stats_vary_by_topic(self):
        svc = ServiceContext()
        r = svc.research("少陰病寒化證研究")
        self.assertTrue(r["topic_analysis"]["scoped"])
        self.assertIn("少陰病", r["topic_analysis"]["channels"])
        tops = [f for f, _ in r["statistics"]["top_formulas"][:5]]
        self.assertIn("四逆湯", tops)
        self.assertNotIn("桂枝湯", tops[:1],
                         "少陰主題不得再以桂枝湯居榜首")

    def test_unparsable_topic_falls_back_honestly(self):
        svc = ServiceContext()
        r = svc.research("量子區塊鏈")
        self.assertFalse(r["topic_analysis"]["scoped"])
        self.assertIn("回退全書", r["topic_analysis"]["note"])


class TestPaperLengthAndExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        svc = ServiceContext()
        cls.r = svc.paper("formula_pattern", topic="桂枝湯類方證",
                          use_llm=False)

    def test_manuscript_length_target(self):
        self.assertGreaterEqual(self.r["manuscript_chars"], 5000)
        for sec in ("方證各論", "計量結果分述", "誤治傳變分述"):
            self.assertIn(sec, self.r["manuscript"])

    def test_downloads_present_and_valid(self):
        import zipfile
        from pathlib import Path
        dl = self.r["downloads"]
        for k in ("md", "docx", "zip"):
            self.assertIn(k, dl)
        rev = Path(self.r["manuscript_path"]).parent
        with zipfile.ZipFile(rev / "manuscript.docx") as z:
            names = z.namelist()
            self.assertIn("word/document.xml", names)
            doc = z.read("word/document.xml").decode()
            self.assertIn("Heading1", doc)
        with zipfile.ZipFile(rev / "paper_bundle.zip") as z:
            self.assertTrue(any(n.endswith(".svg") for n in z.namelist()))
            self.assertIn("manuscript.md", z.namelist())

    def test_docx_xml_escapes(self):
        from hermes_shanghan.paper.exporter import markdown_to_docx_xml
        xml = markdown_to_docx_xml("# T\n\nA<B&C\n\n- 項目")
        self.assertIn("A&lt;B&amp;C", xml)
        self.assertIn("• 項目", xml)


class TestClickThroughEndpoints(unittest.TestCase):
    def test_name_mentions_paginated(self):
        svc = ServiceContext()
        p1 = svc.trace_mentions("桂枝湯", "傷寒溯源集", limit=2)
        self.assertGreater(p1["n_paragraphs"], 10)
        self.assertTrue(p1["has_more"])
        p2 = svc.trace_mentions("桂枝湯", "傷寒溯源集", offset=2, limit=2)
        self.assertNotEqual(p1["passages"][0]["para_seq"],
                            p2["passages"][0]["para_seq"])

    def test_term_passages_anchor_clauses(self):
        svc = ServiceContext()
        t = svc.term_passages("榮衛", "註解傷寒論", limit=3)
        self.assertGreater(t["n_passages"], 0)
        for p in t["passages"]:
            self.assertTrue(p["clause_id"].startswith("SHL_"))

    def test_formula_explain_admin_source(self):
        svc = ServiceContext()
        fe = svc.formula_explain("桂枝湯")
        src = fe["administration"]["source"]
        self.assertEqual(src["clause_id"], "SHL_SONGBEN_0012")
        self.assertIn("宋本", src["book"])


class TestErrata(unittest.TestCase):
    def test_submit_and_list_roundtrip(self):
        import tempfile
        from unittest import mock
        from pathlib import Path
        from hermes_shanghan import config
        svc = ServiceContext()
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(config, "SHANGHAN_DIR", Path(td)):
                r = svc.errata_submit("12", "陽浮而陰弱", "測試建議", note="n")
                self.assertTrue(r["ok"])
                self.assertTrue(r["quote_found_in_clause"])
                bad = svc.errata_submit("12", "此片段絕不在條文中", "x")
                self.assertTrue(bad["ok"])
                self.assertFalse(bad["quote_found_in_clause"])
                miss = svc.errata_submit("12", "", "x")
                self.assertIn("error", miss)
                lst = svc.errata_list()
                self.assertEqual(lst["n_total"], 2)

class TestAdjudicateRecommendations(unittest.TestCase):
    def test_ranked_with_followups(self):
        svc = ServiceContext()
        r = svc.adjudicate(["發熱", "惡寒", "無汗", "身疼痛"],
                           pulse=["浮緊"], use_llm=False)
        recs = r["recommendations"]
        self.assertGreaterEqual(len(recs), 2)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[0]["recommendation_pct"], 100)
        pcts = [x["recommendation_pct"] for x in recs]
        self.assertEqual(pcts, sorted(pcts, reverse=True))
        self.assertTrue(any(x["follow_up_questions"] for x in recs))
        for x in recs:
            self.assertIn("support", x)
            self.assertIn("against", x)
            self.assertIn("missing_key_findings", x)


class TestModernizeLongestFirst(unittest.TestCase):
    def test_negated_sweat_not_hijacked(self):
        from hermes_shanghan.apps.bianzheng import intake_parse
        r = intake_parse("发热，怕冷，不出汗，身疼痛")
        self.assertIn("無汗", r["sweating"])
        self.assertNotIn("汗出", r["sweating"])
