"""十五輪測試：獨立全量古籍智能體。

覆蓋：classics_* 工具族（分層檢索/布爾/座標/計數/誠實封頂）、Passage
穩定 ID（跨進程）、P 層 EvidenceRecord 逐字重驗、證據包、按結論類型的
最低證據層策略、ClassicsAgent 研究留痕、harness classics 模式全鏈路
（Broker P 台賬 + 外層 psg 引用複核 + 偽造 psg 阻斷）、統計語義分離、
DomainPlugin 可執行、深度研究全庫維度誠實跳過、全庫驗收審計、UI 在位。
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hermes_shanghan import config
from hermes_shanghan.corpus import library

try:
    from tests.test_library import make_fixture
except ImportError:                       # 直接以 tests/ 為根運行時
    from test_library import make_fixture


def _artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class _FixtureLib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        make_fixture(Path(cls._tmp.name))
        cls._saved = config.LIBRARY_DIR
        config.LIBRARY_DIR = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        config.LIBRARY_DIR = cls._saved
        cls._tmp.cleanup()


class TestPassageModel(_FixtureLib):
    def test_stable_id_is_cross_process(self):
        # P0（論文 P0-1 同源問題）：ID 不得依賴帶隨機種子的內置 hash()
        code = ("from hermes_shanghan.classics.model import stable_id;"
                "print(stable_id('psg', '甲/1.txt#0'))")
        outs = {subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True,
                               env={"PYTHONHASHSEED": seed,
                                    "PATH": "/usr/bin:/bin"}).stdout.strip()
                for seed in ("1", "2")}
        self.assertEqual(len(outs), 1)
        self.assertRegex(outs.pop(), r"^psg_[0-9a-f]{12}$")

    def test_passages_have_span_identity(self):
        from hermes_shanghan.classics.model import PassageIndex
        lib = library.Library(config.LIBRARY_DIR)
        idx = PassageIndex(lib)
        ps = idx.unit_passages(lib._by_id["乙部方書"])
        self.assertTrue(ps)
        p = ps[0]
        self.assertRegex(p.passage_id, r"^psg_[0-9a-f]{12}$")
        self.assertEqual(idx.get(p.passage_id).flat_text, p.flat_text)


class TestLayeredSearch(_FixtureLib):
    def setUp(self):
        from hermes_shanghan.classics.search import PassageSearcher
        self.s = PassageSearcher(library.Library(config.LIBRARY_DIR))

    def test_boolean_and_not(self):
        r = self.s.search(query="桂枝湯 中風")           # AND：同段共現
        self.assertEqual(r["n_hits"], 1)
        r2 = self.s.search(query="桂枝湯", not_terms=["中風"])
        self.assertEqual(r2["n_hits"], 0)               # NOT 排除

    def test_occurrence_counts_and_coordinates(self):
        r = self.s.search(query="奔豚")
        h = r["hits"][0]
        self.assertGreaterEqual(h["n_occurrences"], 1)
        # 座標可回切原文（扁平化正文座標，fold 1:1 對齊）
        from hermes_shanghan.classics.model import PassageIndex
        p = PassageIndex(self.s.lib).get(h["passage_id"], work=h["work_id"])
        from hermes_shanghan.textutil import fold_variants
        self.assertEqual(
            fold_variants(p.flat_text[h["char_start"]:h["char_end"]]), "奔豚")

    def test_layers_are_explained_and_capping_honest(self):
        r = self.s.search(query="奔豚", max_scan=1, per_book=1, limit=1)
        self.assertIn("L0_metadata", r["retrieval_layers"])
        self.assertIn("L1_char_index", r["retrieval_layers"])
        self.assertIn("L2_verbatim_scan", r["retrieval_layers"])
        # 未實現的層如實聲明，不冒充
        self.assertIn("未實現", r["retrieval_layers"]["L3_plus"])

    def test_metadata_filter_l0(self):
        r = self.s.search(query="奔豚", dynasty="清")
        self.assertTrue(all(h["dynasty"] == "清" for h in r["hits"]))


class TestClassicsTools(_FixtureLib):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from hermes_shanghan.agent.tools import ToolRegistry
        cls.reg = ToolRegistry()

    def test_eight_tools_registered_with_P_contracts(self):
        from hermes_shanghan.classics.tools import CLASSICS_TOOL_NAMES
        names = set(self.reg.names())
        for t in CLASSICS_TOOL_NAMES:
            self.assertIn(t, names)
        cs = {c["name"]: c for c in self.reg.contracts()}
        for t in CLASSICS_TOOL_NAMES:
            self.assertEqual(cs[t]["evidence_level"], "P")

    def test_search_returns_verifiable_p_evidence(self):
        out = self.reg.call("classics_search_passages", {"query": "奔豚"})
        self.assertTrue(out["passage_evidence"])
        ev = out["passage_evidence"][0]
        for key in ("evidence_level", "work_id", "passage_id", "verbatim_text",
                    "char_start", "char_end", "quote_hash", "retrieval_query",
                    "retrieval_rank"):
            self.assertIn(key, ev)
        self.assertEqual(ev["evidence_level"], "P")
        # 逐字重驗：書名/章節/摘錄/hash 對應性不靠信任
        from hermes_shanghan.classics.evidence import verify_records
        from hermes_shanghan.classics.tools import _searcher
        v = verify_records(out["passage_evidence"], _searcher().index)
        self.assertTrue(v["ok"])
        # 篡改即被抓
        bad = dict(ev, verbatim_text=ev["verbatim_text"][:-1] + "偽")
        self.assertFalse(verify_records([bad], _searcher().index)["ok"])

    def test_trace_citation_time_ordered_with_counter_search(self):
        out = self.reg.call("classics_trace_citation",
                            {"quote": "奔豚者，氣上衝胸也"})
        ranks = [h["dynasty_rank"] for h in out["attestations_time_ordered"]]
        self.assertEqual(ranks, sorted(ranks))
        self.assertIn("counter_search", out)
        self.assertIn("在庫首現≠歷史首現", out["honesty"])
        self.assertTrue(out["conclusion_policy"])       # 政策表隨結果出廠

    def test_compare_witnesses_and_concept_drift(self):
        w = self.reg.call("classics_compare_witnesses", {"work": "丙氏全書"})
        self.assertGreaterEqual(w["n_witnesses"], 1)
        d = self.reg.call("classics_concept_drift", {"term": "奔豚"})
        self.assertTrue(d["series_by_dynasty"])
        self.assertIn("頻次漂移≠語義漂移", d["honesty"])

    def test_stats_semantics_separated(self):
        lib_stats = self.reg.call("classics_library_stats", {})
        self.assertEqual(lib_stats["n_books"], 3)
        self.assertIn("非", lib_stats["semantic"])       # 明示不是規則庫統計
        _artifacts()
        dom = self.reg.call("shanghan_corpus_stats", {})
        self.assertIn("semantic", dom)
        self.assertNotIn("n_books", dom)                 # 各答各的，不混同

    def test_export_packet_verifies(self):
        out = self.reg.call("classics_search_passages", {"query": "奔豚"})
        pid = out["passage_evidence"][0]["passage_id"]
        pkt = self.reg.call("classics_export_evidence_packet",
                            {"passage_ids": [pid, "psg_000000000000"]})
        self.assertTrue(pkt["packet"]["verification"]["ok"])
        self.assertIn("psg_000000000000", pkt["missing_passage_ids"])

    def test_unavailable_library_hints_fetch(self):
        with tempfile.TemporaryDirectory() as empty:
            saved = config.LIBRARY_DIR
            config.LIBRARY_DIR = Path(empty)
            try:
                out = self.reg.call("classics_search_passages",
                                    {"query": "奔豚"})
            finally:
                config.LIBRARY_DIR = saved
        self.assertFalse(out["available"])
        self.assertIn("library fetch", out["hint"])


class TestClassicsAgent(_FixtureLib):
    def test_earliest_question_full_research_log(self):
        from hermes_shanghan.classics.agent import ClassicsAgent
        out = ClassicsAgent().ask("歷代醫書裡「奔豚」最早見於哪部書？")
        self.assertIn("在庫首現", out["answer"])
        self.assertIn("psg_", out["answer"])             # 引用帶 passage_id
        self.assertEqual(out["tools_used"], ["classics_trace_citation"])
        log = out["research_log"]
        for key in ("plan", "queried_works", "unqueried_candidates",
                    "supporting_evidence_count", "counter_candidates",
                    "first_candidates", "needs_human_review"):
            self.assertIn(key, log)
        self.assertTrue(out["audit"]["quote_verification"]["ok"])
        self.assertEqual(out["audit"]["policy_violations"], [])

    def test_refuses_honestly_without_library(self):
        from hermes_shanghan.classics.agent import ClassicsAgent
        with tempfile.TemporaryDirectory() as empty:
            saved = config.LIBRARY_DIR
            config.LIBRARY_DIR = Path(empty)
            try:
                out = ClassicsAgent().ask("「奔豚」最早見於哪部書？")
            finally:
                config.LIBRARY_DIR = saved
        self.assertTrue(out["refused"])
        self.assertEqual(out["refusal_reason"], "library_unavailable")

    def test_conclusion_policy_table(self):
        from hermes_shanghan.classics.evidence import conclusion_policy_check
        # 「最早」結論但沒跑時間有序檢索 → 違例
        v = conclusion_policy_check("此語最早見於《千金方》。", [], ["classics_search_passages"])
        self.assertTrue(any(x["conclusion_type"] == "最早提出" for x in v))
        # 「普遍討論」但只有 1 個著作來源 → 違例
        v2 = conclusion_policy_check("後世醫家普遍討論之。",
                                     [{"work_id": "a"}], [])
        self.assertTrue(any("普遍" in x["conclusion_type"] for x in v2))
        # 宋本原文宣稱但無 A 層編號 → 違例
        v3 = conclusion_policy_check("宋本原文記載此證。", [], [])
        self.assertTrue(any(x["conclusion_type"] == "宋本原文記載" for x in v3))


class TestClassicsHarness(_FixtureLib):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _artifacts()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(config.RUNS_DIR, ignore_errors=True)
        super().tearDownClass()

    def test_classics_mode_full_chain_passes_gate(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        st = HarnessRunner().start("歷代醫書裡「奔豚」最早見於哪部書？",
                                   mode="classics", role="researcher")
        self.assertEqual(st.status, "completed")
        self.assertEqual(st.release["decision"], "pass")
        # Broker P 台賬：passage 記錄帶完整綁定字段
        p_recs = [r for recs in st.evidence_ledger.values()
                  for r in recs if r.get("passage_id")]
        self.assertTrue(p_recs)
        for r in p_recs:
            self.assertEqual(r["evidence_level"], "P")
            self.assertEqual(r["evidence_role"], "primary_text_returned")
            self.assertEqual(r["registered_by"], "capability_broker")
            self.assertTrue(r["quote_hash"] and r["span_id"])
        # 外層獨立複核：引用的 psg 全部在台賬允許集中
        cr = st.node_outputs["execute"]["citation_report"]
        self.assertTrue(cr["passage_citations"]["verified"])
        self.assertEqual(cr["passage_citations"]["unsupported"], [])

    def test_forged_passage_citation_is_blocked(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        from hermes_shanghan.classics import agent as ca
        real_ask = ca.ClassicsAgent.ask

        def forged(self, q, role="researcher"):
            out = real_ask(self, q, role=role)
            out["answer"] += "\n另據〔psg_deadbeef0000〕亦可證。"   # 台賬外引用
            return out
        ca.ClassicsAgent.ask = forged
        try:
            st = HarnessRunner().start("「奔豚」最早見於哪部書？",
                                       mode="classics", role="researcher")
        finally:
            ca.ClassicsAgent.ask = real_ask
        self.assertEqual(st.release["decision"], "blocked")   # 偽造引用=硬阻斷
        self.assertIn("psg_deadbeef0000",
                      st.node_outputs["execute"]["citation_report"]["unsupported"])

    def test_earliest_claim_without_trace_tool_fails_gate(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        from hermes_shanghan.classics import agent as ca
        real_plan = ca.ClassicsAgent._plan
        real_compose = ca.ClassicsAgent._compose

        def plan_search_only(self, q, topic):
            return [{"intent": "search", "tool": "classics_search_passages",
                     "args": {"query": topic, "limit": 4}}]

        def compose_overclaim(self, q, topic, plan, results, evidence):
            base = real_compose(self, q, topic, plan, results, evidence)
            return base + "\n可斷言：此語最早由本書提出。"     # 未經時間有序+反證
        ca.ClassicsAgent._plan = plan_search_only
        ca.ClassicsAgent._compose = compose_overclaim
        try:
            st = HarnessRunner().start("「奔豚」的記載情況？",
                                       mode="classics", role="researcher")
        finally:
            ca.ClassicsAgent._plan = real_plan
            ca.ClassicsAgent._compose = real_compose
        self.assertIn(st.release["decision"], ("review_required", "blocked"))
        cr = st.node_outputs["execute"]["citation_report"]
        self.assertTrue(cr.get("conclusion_policy_violations"))
        # citation_failure 不可審批豁免（十四輪 P0 + 十五輪政策聯動）
        self.assertIn("citation_failure", st.pending_review)

    def test_classics_mode_declared_and_dispatched(self):
        from hermes_shanghan.agent.harness.state import RUN_MODES
        self.assertIn("classics", RUN_MODES)


class TestPlatformSeams(_FixtureLib):
    def test_domain_plugins_executable(self):
        from hermes_shanghan.domains import DOMAINS, active_domains
        active = {d.domain_id for d in active_domains()}
        self.assertEqual(active, {"shanghan", "classics"})
        for d in active_domains():
            self.assertTrue(d.executable(), d.domain_id)
            self.assertTrue(d.load_normalizer())
            self.assertTrue(d.load_passage_parser())
            self.assertTrue(d.load_citation_parser())
        # planned 插件如實聲明未實現（不偽裝）
        self.assertIsNone(DOMAINS["jingui"].tool_factory)
        self.assertFalse(DOMAINS["jingui"].executable())

    def test_deep_research_library_dimension_honest(self):
        from hermes_shanghan.agent.research_loop import (DeepResearcher,
                                                         LIBRARY_DIMENSION)
        _artifacts()
        r = DeepResearcher(max_rounds=1)
        self.assertIn(LIBRARY_DIMENSION, r.dimensions)   # 庫就緒→常規維度
        with tempfile.TemporaryDirectory() as empty:
            saved = config.LIBRARY_DIR
            config.LIBRARY_DIR = Path(empty)
            try:
                r2 = DeepResearcher(max_rounds=1)
            finally:
                config.LIBRARY_DIR = saved
        self.assertNotIn(LIBRARY_DIMENSION, r2.dimensions)  # 未就緒→如實跳過

    def test_acceptance_audit_on_fixture(self):
        from hermes_shanghan.classics.audit import acceptance_report
        rep = acceptance_report(root=config.LIBRARY_DIR, sample=3)
        self.assertTrue(rep["available"])
        for key in ("n_books", "n_units", "n_parsed", "n_empty_text",
                    "n_encoding_anomalies", "n_missing_author",
                    "n_missing_dynasty", "depth_histogram", "max_depth",
                    "n_duplicate_text_groups", "toc_recognition_rate",
                    "unreadable_files", "largest_file"):
            self.assertIn(key, rep)
        self.assertEqual(rep["max_depth"], 3)
        self.assertEqual(len(rep["gold_sample"]), 3)     # 分層抽樣金標準
        self.assertTrue(all(s["stratum"] for s in rep["gold_sample"]))

    def test_classics_ui_page_served_and_linked(self):
        from hermes_shanghan.server import http_server as hs
        page = hs.STATIC_DIR / "classics.html"
        self.assertTrue(page.exists())
        con = page.read_text(encoding="utf-8")
        for section in ("書庫管理", "研究檢索", "古籍閱讀", "智能體工作台",
                        "證據籃", "classics_library_stats",
                        "classics_export_evidence_packet"):
            self.assertIn(section, con)
        self.assertIn("classics.html",
                      (hs.STATIC_DIR / "console.html").read_text(encoding="utf-8"))
        self.assertIn("classics.html",
                      (hs.STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    def test_library_stats_question_routes_to_classics(self):
        _artifacts()
        from hermes_shanghan.agent.agent import ShanghanAgent
        out = ShanghanAgent().ask("笈成全庫一共收錄多少部書？",
                                  role="researcher")
        self.assertIn("classics_library_stats", out["tools_used"])


class TestRealLibraryAcceptance(unittest.TestCase):
    def test_real_library_acceptance_report(self):
        if not library.is_available():
            self.skipTest("full library not fetched (run `library fetch`)")
        from hermes_shanghan.classics.audit import acceptance_report
        rep = acceptance_report(sample=100)
        self.assertGreaterEqual(rep["n_books"], 800)
        self.assertGreaterEqual(rep["n_parsed"] / rep["n_units"], 0.98)
        self.assertGreaterEqual(rep["toc_recognition_rate"], 0.8)
        self.assertEqual(len(rep["gold_sample"]), 100)
        self.assertLessEqual(len(rep["unreadable_files"]), 0)


if __name__ == "__main__":
    unittest.main()
