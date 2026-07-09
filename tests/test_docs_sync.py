"""文檔同步守衛（評審建議：防止 README=253 / TEST_REPORT=240 類漂移）。

用測試強制文檔數字與代碼實況一致：測試總數、工具數在 README 與
TEST_REPORT 中的表述必須等於運行時實測值，不一致即紅。"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _discovered_test_count() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py",
                            top_level_dir=str(ROOT))
    return suite.countTestCases()


class TestDocsSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_tests = _discovered_test_count()
        from hermes_shanghan.agent.tools import get_registry
        cls.n_tools = len(get_registry().names())
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.report = (ROOT / "docs" / "TEST_REPORT.md").read_text(encoding="utf-8")

    def test_readme_test_count_matches(self):
        m = re.search(r"測試（(\d+) 項", self.readme)
        self.assertIsNotNone(m, "README 應包含「測試（N 項」")
        self.assertEqual(int(m.group(1)), self.n_tests,
                         f"README 測試數 {m.group(1)} ≠ 實測 {self.n_tests}")
        m2 = re.search(r"tests/\s+(\d+) 項測試", self.readme)
        if m2:
            self.assertEqual(int(m2.group(1)), self.n_tests)

    def test_readme_tool_count_matches(self):
        m = re.search(r"（(\d+) 個只讀回源工具", self.readme)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), self.n_tools,
                         f"README 工具數 {m.group(1)} ≠ 實測 {self.n_tools}")

    def test_report_test_count_matches(self):
        m = re.search(r"測試總數 \| (\d+) 項", self.report)
        self.assertIsNotNone(m, "TEST_REPORT 應包含「測試總數 | N 項」")
        self.assertEqual(int(m.group(1)), self.n_tests,
                         f"TEST_REPORT 測試數 {m.group(1)} ≠ 實測 {self.n_tests}")

    def test_llm_agent_doc_tool_count_matches(self):
        doc = (ROOT / "docs" / "LLM_AGENT.md").read_text(encoding="utf-8")
        m = re.search(r"## (\d+) 個可調用工具", doc)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), self.n_tools)

    def test_committed_tool_specs_asset_in_sync(self):
        # 隨庫 tool_specs.json 必須與運行時註冊表同步（七輪評審抓到 18≠28）
        import json
        spec = json.loads((ROOT / "data" / "shanghan" / "tool_specs.json")
                          .read_text(encoding="utf-8"))
        names = {t["function"]["name"] for t in spec["openai_tools"]}
        from hermes_shanghan.agent.tools import get_registry
        self.assertEqual(names, set(get_registry().names()),
                         "運行 export-tools --out data/shanghan/tool_specs.json 同步")
        self.assertEqual(len(spec["anthropic_tools"]), self.n_tools)


if __name__ == "__main__":
    unittest.main()
