"""十五輪測試：論文 Figure Factory。

覆蓋：跨進程（PYTHONHASHSEED）字節級可復現、每種論文類型獨立圖組、
版本化輸出不覆蓋、正文引用圖、正式 Figure Legends、逐 panel Source
Data、SVG 物理尺寸/無障礙/字號下限、GraphML+layout 導出、CVD 檢查
真實落盤、fail-fast 非法論文類型、劑量圖為假設區間非確定柱。
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hermes_shanghan import config


def _artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


def _writer():
    from hermes_shanghan.orchestrator import Artifacts
    from hermes_shanghan.paper.writer import PaperWriter
    art = Artifacts()
    return PaperWriter(art.clauses, art.initial_rules, art.formula_rules,
                       art.six_channel_rules, art.mistreatment_rules,
                       art.differential_rules,
                       commentary_rules=art.commentary_rules)


class TestFigureFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _artifacts()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        w = _writer()
        cls.fp = w.generate("formula_pattern", out_dir=cls.out / "fp",
                            use_llm=False)
        cls.bm = w.generate("benchmark", out_dir=cls.out / "bm", use_llm=False)
        cls.fp_text = cls.fp.read_text(encoding="utf-8")
        cls.fp_manifest = json.loads(
            (cls.fp.parent / "figures_manifest.json").read_text(encoding="utf-8"))
        cls.qa = json.loads((cls.fp.parent / "figure_qa" / "qa_report.json")
                            .read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_mermaid_ids_stable_across_hash_seeds(self):
        # P0-1：abs(hash()) 帶進程隨機種子——已換 sha256 穩定 ID
        code = ("from hermes_shanghan.paper.figspec import stable_id;"
                "print(stable_id('f', '桂枝湯'))")
        outs = {subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True,
                               env={"PYTHONHASHSEED": seed,
                                    "PATH": "/usr/bin:/bin"}).stdout.strip()
                for seed in ("1", "2", "0")}
        self.assertEqual(len(outs), 1)
        mmd = (self.fp.parent / "fig_channel_formula.mmd.md") \
            .read_text(encoding="utf-8")
        self.assertNotRegex(mmd, r"\bf\d{1,4}\[")   # 舊 hash 模式 ID 已根除

    def test_each_paper_type_has_own_figure_plan(self):
        # P0-2：八種論文不再共用同一套圖
        from hermes_shanghan.paper.figspec import FIGURE_PLANS
        from hermes_shanghan.paper.writer import PAPER_TYPES
        self.assertEqual(set(FIGURE_PLANS), set(PAPER_TYPES))
        self.assertEqual(len({tuple(v) for v in FIGURE_PLANS.values()}),
                         len(FIGURE_PLANS))          # 兩兩不同
        bm_manifest = json.loads(
            (self.bm.parent / "figures_manifest.json").read_text(encoding="utf-8"))
        self.assertNotEqual(self.fp_manifest["plan"], bm_manifest["plan"])

    def test_versioned_output_no_silent_overwrite(self):
        # P0-3：默認輸出按內容指紋分修訂目錄 + revisions.json 追加
        w = _writer()
        saved = config.PAPER_DIR
        config.PAPER_DIR = self.out / "papers"
        try:
            p1 = w.generate("six_channel_kg", use_llm=False)
            p2 = w.generate("six_channel_kg", use_llm=False)   # 同輸入=冪等
        finally:
            config.PAPER_DIR = saved
        self.assertEqual(p1, p2)
        self.assertTrue(p1.parent.name.startswith("rev_"))
        revs = json.loads((p1.parent.parent / "revisions.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(len(revs), 1)               # 冪等重生成不重複記錄

    def test_figures_cited_in_body_with_legends(self):
        # P0-5/P0-6：圖入論證鏈 + 正式圖例
        body = self.fp_text.split("## 圖表清單")[0]
        self.assertIn("### 4.0 圖組導覽", body)
        for f in self.fp_manifest["figures"]:
            if not f.get("skipped"):
                self.assertIn(f["figure_id"], body)
        self.assertIn("## 圖例（Figure Legends）", self.fp_text)
        for token in ("n：", "數據來源：", "誤差/統計定義：", "證據層級：",
                      "Source Data："):
            self.assertIn(token, self.fp_text)

    def test_source_data_per_panel(self):
        sd = self.fp.parent / "source_data"
        files = sorted(x.name for x in sd.iterdir())
        self.assertTrue(files)
        for f in self.fp_manifest["figures"]:
            if not f.get("skipped"):
                for src in f["source_data"]:
                    self.assertIn(src, files)
                self.assertTrue(f["data_sha256"])    # 逐圖數據指紋

    def test_svg_physical_size_accessibility_min_font(self):
        svg = (self.fp.parent / "fig_formula_frequency.svg") \
            .read_text(encoding="utf-8")
        self.assertIn("mm'", svg)                    # 物理尺寸
        self.assertIn("role='img'", svg)
        self.assertIn("<title id='fig-title'>", svg)
        self.assertIn("<desc id='fig-desc'>", svg)
        self.assertIn("載方條文數", svg)              # 軸標題（量綱）
        import re
        sizes = [float(s) for s in re.findall(r"font-size='([\d.]+)'", svg)]
        self.assertTrue(all(s >= 9 for s in sizes))
        # 標籤不再靜默截斷：Source Data 中每個方名都完整出現在 SVG 里
        rows = (self.fp.parent / "source_data").glob("*formula_frequency.csv")
        for row in list(rows)[0].read_text(encoding="utf-8").splitlines()[1:]:
            self.assertIn(row.split(",")[0], svg)

    def test_network_figures_export_graphml_and_layout(self):
        # P0-4：DOT 是源代碼不是投稿圖——隨附 GraphML+凍結佈局+邊表 CSV
        parent = self.fp.parent
        self.assertTrue((parent / "fig_formula_family.graphml").exists())
        layout = json.loads((parent / "fig_formula_family.layout.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(layout["layout"], "deterministic-circular")
        import xml.etree.ElementTree as ET
        ET.parse(parent / "fig_formula_family.graphml")   # 合法 XML

    def test_cvd_check_is_real_asset(self):
        # 十五輪 十二：「已驗證」必須有可復核資產
        cvd = self.qa["palette_cvd"]
        for vt in ("protanopia", "deuteranopia", "tritanopia"):
            self.assertIn(vt, cvd["vision_types"])
            self.assertTrue(cvd["vision_types"][vt]["adjacent_delta_e"])
        self.assertIn("grayscale", cvd)
        self.assertTrue(cvd["ok"])
        self.assertTrue(self.qa["ok"], self.qa["hard_violations"])

    def test_invalid_paper_type_fails_fast(self):
        with self.assertRaises(ValueError):
            _writer().generate("totally_bogus", out_dir=self.out / "x")

    def test_dose_figure_is_scenario_interval_not_bars(self):
        w = _writer()
        p = w.generate("network_pharmacology", out_dir=self.out / "np",
                       use_llm=False)
        svg = (p.parent / "fig_dose_totals.svg").read_text(encoding="utf-8")
        self.assertIn("折算假設區間", svg)
        self.assertIn("不構成臨床劑量建議", svg)
        text = p.read_text(encoding="utf-8")
        self.assertIn("研究定位聲明", text)          # 不冒充網絡藥理學
        self.assertIn("共現網絡", text)

    def test_heatmap_has_scale_legend_and_shared_n(self):
        w = _writer()
        p = w.generate("commentary_compare", out_dir=self.out / "cc",
                       use_llm=False)
        svg = (p.parent / "fig_commentator_agreement.svg") \
            .read_text(encoding="utf-8")
        self.assertIn("色階", svg)                    # 色標圖例
        self.assertIn("(n=", svg)                     # 每格共注條數
        self.assertIn("對角=自身（非缺失）", svg)      # 對角語義區分

    def test_abstract_no_overclaim(self):
        self.assertNotIn("每條結論均可追溯至條文編號", self.fp_text)
        self.assertIn("證據分層聲明", self.fp_text)


if __name__ == "__main__":
    unittest.main()
