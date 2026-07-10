"""十輪測試：work_type 分類（證據層不由目錄名決定）/ 全庫供應鏈安全 /
EvidenceRecord 逐證據來源對象 / 方證論證結構（反證與爭議 Harness）。"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


# ---------------------------------------------------------------------------
class TestWorkType(unittest.TestCase):
    def test_registered_books_keep_curated_layer(self):
        from hermes_shanghan.corpus.worktype import classify
        wt, layer, basis, _ = classify(config.PRIMARY_BOOK)
        self.assertEqual((wt, layer, basis),
                         ("canonical_text", "A", "registered"))
        wt, layer, basis, _ = classify("傷寒來蘇集")
        self.assertEqual((wt, layer, basis), ("commentary", "C", "registered"))
        wt, layer, _, _ = classify("傷寒論輯義")
        self.assertEqual(wt, "collation")     # 輯義如實登記為校勘/輯佚

    def test_unregistered_fail_closed_to_p(self):
        # 十輪 六.2：目錄名/標題推斷不能授予 C/D 資格
        from hermes_shanghan.corpus.worktype import classify
        wt, layer, basis, inferred = classify("某某傷寒新注", "shanghan")
        self.assertEqual((wt, layer), ("unclassified", "P"))
        self.assertEqual(basis, "fail_closed_unregistered")
        self.assertEqual(inferred, "commentary")   # 推斷僅供編目複核

    def test_inference_taxonomy(self):
        from hermes_shanghan.corpus.worktype import infer_work_type
        self.assertEqual(infer_work_type("經方實驗錄醫案"), "medical_case")
        self.assertEqual(infer_work_type("傷寒論類方彙參"), "formula_family")
        self.assertEqual(infer_work_type("傷寒論歌括淺注"), "teaching_summary")
        self.assertEqual(infer_work_type("無線索書名"), "unclassified")

    def test_manifest_carries_basis_and_no_guessed_cd(self):
        _ensure_artifacts()
        m = json.loads((config.MANIFEST_DIR / "corpus_manifest.json")
                       .read_text(encoding="utf-8"))
        for b in m["books"]:
            self.assertIn("work_type", b)
            self.assertIn("layer_basis", b)
            if b["hermes_layer"] in ("A", "B", "C", "D"):
                # C/D 資格必須來自人工登記，不得來自目錄猜測
                self.assertEqual(b["layer_basis"], "registered", b["book_dir"])
            else:
                self.assertEqual(b["hermes_layer"], "P")


# ---------------------------------------------------------------------------
class TestLibrarySupplyChain(unittest.TestCase):
    def test_custom_url_fail_closed(self):
        from hermes_shanghan.corpus import library
        os.environ.pop("HERMES_LIBRARY_ALLOW_CUSTOM", None)
        with self.assertRaises(library.SupplyChainError):
            library.resolve_source("https://evil.example/x.7z")
        os.environ["HERMES_LIBRARY_ALLOW_CUSTOM"] = "1"
        try:
            with self.assertRaises(library.SupplyChainError):
                library.resolve_source("https://mirror.example/x.7z")  # 無哈希
            url, sha = library.resolve_source("https://mirror.example/x.7z",
                                              sha256="a" * 64)
            self.assertEqual(sha, "a" * 64)
        finally:
            os.environ.pop("HERMES_LIBRARY_ALLOW_CUSTOM", None)
        # 默認源：固定哈希
        url, sha = library.resolve_source(None)
        self.assertEqual((url, sha), (config.LIBRARY_URL, config.LIBRARY_SHA256))

    def test_member_name_validation(self):
        from hermes_shanghan.corpus import library
        library.validate_member_names(["books/a/index.txt", "b/1.txt"])
        for bad in (["../etc/passwd"], ["/abs/path"], ["C:\\win"],
                    ["ok/../../up"]):
            with self.assertRaises(library.SupplyChainError):
                library.validate_member_names(bad)

    def test_extracted_tree_validation(self):
        from hermes_shanghan.corpus import library
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "book1").mkdir()
            (root / "book1" / "index.txt").write_text("x", encoding="utf-8")
            stats = library.validate_extracted_tree(root)
            self.assertEqual(stats["n_files"], 1)
            os.symlink("/etc/passwd", root / "book1" / "evil")
            with self.assertRaises(library.SupplyChainError):
                library.validate_extracted_tree(root)


# ---------------------------------------------------------------------------
class TestEvidenceRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_clause_record_complete_and_honest(self):
        from hermes_shanghan.trace.evidence import EVIDENCE_RECORD_FIELDS
        out = self.reg.call("shanghan_get_clause", {"ref": "12"})
        rec = out["evidence_record"]
        for f in EVIDENCE_RECORD_FIELDS:
            self.assertIn(f, rec)
        self.assertEqual(rec["passage_id"], "SHL_SONGBEN_0012")
        self.assertEqual(rec["provenance_layer"], "A")
        self.assertEqual(rec["work_type"], "canonical_text")
        self.assertRegex(rec["quote_hash"], r"^[0-9a-f]{16}$")
        self.assertTrue(rec["edition_fingerprint"])   # 版本指紋（換版即變）
        # 誠實記 null：未保留的字段不編造
        self.assertIsNone(rec["char_start"])
        self.assertIsNone(rec["volume_id"])
        # 品質語義必須有說明：來源校對程度標注，或 unmeasured（≠0 分）
        self.assertTrue("校對程度" in rec["quality_note"]
                        or "unmeasured" in rec["quality_note"])

    def test_search_records_carry_retrieval_context(self):
        out = self.reg.call("shanghan_search", {"query": "惡寒發熱", "top_k": 3})
        recs = out["evidence_records"]
        self.assertEqual(len(recs), len(out["hits"]))
        self.assertEqual(recs[0]["retrieval_query"], "惡寒發熱")
        self.assertEqual([r["retrieval_rank"] for r in recs],
                         list(range(1, len(recs) + 1)))

    def test_edge_record(self):
        from hermes_shanghan.trace.evidence import evidence_record_for_edge
        rec = evidence_record_for_edge(
            {"book": "某注本", "book_dir": "某注本", "dynasty": "清",
             "layer": "P", "clause_id": "SHL_SONGBEN_0012",
             "max_coverage": 0.83, "modes": {"節引": 2},
             "first_chapter": "卷一"})
        self.assertEqual(rec["provenance_layer"], "P")
        self.assertEqual(rec["quality_score"], 0.83)
        self.assertIsNone(rec["verbatim_text"])       # 聚合層不存逐字（如實）
        self.assertEqual(rec["citation_modes"], {"節引": 2})


# ---------------------------------------------------------------------------
class TestArgumentChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.trace.chains import trace_dispatch
        cls.d = trace_dispatch("argument", "桂枝湯")

    def test_seven_sections_present(self):
        for key in ("songben_direct", "contradicting_or_caution_clauses",
                    "variant_forks", "commentator_common_points",
                    "commentator_dispute_points", "hidden_assumptions",
                    "posthoc_induction", "model_synthesis", "undecidable",
                    "confidence_by_layer"):
            self.assertIn(key, self.d)

    def test_contradicting_clauses_found(self):
        # 酒客不可與桂枝湯（第17條）必須作為反證/邊界證據出現
        ids = [c["clause_id"]
               for c in self.d["contradicting_or_caution_clauses"]]
        self.assertIn("SHL_SONGBEN_0017", ids)

    def test_no_unified_verdict_and_e_layer_marked(self):
        self.assertTrue(self.d["undecidable"])        # 不能裁決是正式輸出
        self.assertIn("E 層", self.d["model_synthesis"])
        self.assertEqual(
            set(self.d["confidence_by_layer"]), {"A", "B", "C", "D", "E"})
        self.assertIn("置信最低", self.d["confidence_by_layer"]["E"]["note"])

    def test_hidden_assumptions_have_no_textual_anchor(self):
        # 隱含假設的術語必須確實不在支持條文原文中
        from hermes_shanghan.textutil import fold_variants
        from hermes_shanghan.trace.chains import _clauses
        clauses = _clauses()
        blob = fold_variants("".join(
            (clauses.get(c["clause_id"]) or {}).get("clean_text", "")
            for c in self.d["songben_direct"]))
        for h in self.d["hidden_assumptions"]:
            self.assertNotIn(fold_variants(h["term"]), blob)
            self.assertGreaterEqual(len(h["used_by"]), 2)

    def test_clause_ref_falls_back_to_formula(self):
        from hermes_shanghan.trace.chains import argument_chain
        d = argument_chain("12")
        self.assertEqual(d.get("formula"), "桂枝湯")
        err = argument_chain("不存在的方")
        self.assertIn("dispute", err.get("error", ""))


if __name__ == "__main__":
    unittest.main()
