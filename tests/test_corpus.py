"""Segmentation & catalog tests against the real corpus."""
import unittest

from hermes_shanghan.corpus import catalog, downloader, segmenter


class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clauses = segmenter.segment_canonical()

    def test_398_canonical_clauses(self):
        self.assertEqual(len(self.clauses), 398)
        nums = [c.clause_number for c in self.clauses]
        self.assertEqual(nums, list(range(1, 399)))

    def test_chapter_channel_mapping(self):
        c1 = self.clauses[0]
        self.assertEqual(c1.six_channel, "太陽病")
        self.assertIn("太陽之為病", c1.clean_text)
        c180 = self.clauses[179]
        self.assertEqual(c180.six_channel, "陽明病")
        self.assertIn("胃家實", c180.clean_text)
        c281 = self.clauses[280]
        self.assertEqual(c281.six_channel, "少陰病")
        self.assertIn("脈微細", c281.clean_text)

    def test_formula_blocks(self):
        n_blocks = sum(len(c.formula_blocks) for c in self.clauses)
        self.assertEqual(n_blocks, 116)
        c12 = self.clauses[11]
        self.assertEqual(c12.formula_blocks[0].formula_name, "桂枝湯")
        herbs = [x["herb"] for x in c12.formula_blocks[0].composition]
        self.assertEqual(herbs, ["桂枝", "芍藥", "甘草", "生薑", "大棗"])
        self.assertIn("三兩", c12.formula_blocks[0].composition[0]["dose_processing"])
        self.assertIn("微火煮取三升", c12.formula_blocks[0].preparation)
        self.assertIn("禁生冷", c12.formula_blocks[0].administration)

    def test_auxiliary_chapters(self):
        aux = segmenter.segment_auxiliary()
        self.assertGreater(len(aux), 100)
        chapters = {c.chapter for c in aux}
        self.assertTrue(any("傷寒例" in ch for ch in chapters))
        self.assertTrue(any("不可發汗" in ch for ch in chapters))
        # canonical chapters must NOT appear in aux
        self.assertFalse(any("辨太陽病脈證並治上" == ch for ch in chapters))

    def test_manifest(self):
        manifest_path = downloader.run()
        self.assertTrue(manifest_path.exists())
        m = downloader.load_manifest()
        self.assertGreaterEqual(m["book_count"], 60)
        primary = [b for b in m["books"] if b["book_dir"] == "傷寒論_條文版"]
        self.assertEqual(primary[0]["hermes_layer"], "A")
        gui = [b for b in m["books"] if b["book_dir"] == "傷寒雜病論_桂本"]
        self.assertEqual(gui[0]["hermes_layer"], "B")


if __name__ == "__main__":
    unittest.main()
