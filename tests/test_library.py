"""中醫笈成全庫接入：解析（元數據/多卷/小數卷號/嵌套子書/menu 排除）、
編目與字符索引、快速查閱（search/grep/toc/read）、工具與路由集成。

Tests run against a synthetic fixture library exercising every layout the
real archive contains, so CI never needs the 69MB download. When the real
library happens to be fetched locally, an extra sanity check runs too.
"""
import tempfile
import unittest
from pathlib import Path

from hermes_shanghan import config
from hermes_shanghan.corpus import library


def make_fixture(root: Path) -> None:
    books = root / library.BOOKS_SUBDIR
    # 1) single-file book: body lives in index.txt (like 傷寒論_宋本)
    a = books / "甲乙經考"
    a.mkdir(parents=True)
    a.joinpath("index.txt").write_text(
        "======甲乙經考======\n\n<book>\n書名=甲乙經考\n作者=王試\n"
        "朝代=清\n年份=1700\n分類=醫經\n品質=90%\n備考=測試用\n</book>\n\n"
        "=====總論=====\n\n人參味甘，微寒。主補五臟，\n安精神。\n",
        encoding="utf-8")
    # 2) multi-file book with decimal stems (like 曹氏傷寒金匱發微合刊)
    #    + menu.txt that must be excluded from reading order
    b = books / "乙部方書"
    b.mkdir()
    b.joinpath("index.txt").write_text(
        "<book>\n書名=乙部方書\n作者=李試\n朝代=明\n分類=方書\n</book>\n",
        encoding="utf-8")
    b.joinpath("menu.txt").write_text("導航頁不入正文", encoding="utf-8")
    b.joinpath("2-0.3.txt").write_text(
        "=====卷二序三=====\n\n奔豚者，氣上衝胸也。\n", encoding="utf-8")
    b.joinpath("1.txt").write_text(
        "=====卷一=====\n\n桂枝湯主治中風，\n胸脅苦滿者刺期門。\n",
        encoding="utf-8")
    b.joinpath("2-15.txt").write_text(
        "=====卷二之十五=====\n\n半夏瀉心湯。\n", encoding="utf-8")
    # 3) nested container book with a sub-book (like 醫宗金鑑)
    c = books / "丙氏全書"
    c.mkdir()
    c.joinpath("index.txt").write_text(
        "<book>\n書名=丙氏全書\n作者=張試\n朝代=清\n分類=綜合\n</book>\n",
        encoding="utf-8")
    sub = c / "丙氏傷寒注"
    sub.mkdir()
    sub.joinpath("index.txt").write_text(
        "<book>\n書名=丙氏全書·丙氏傷寒注\n作者=張試\n</book>\n",
        encoding="utf-8")
    sub.joinpath("1.txt").write_text(
        "=====太陽篇=====\n\n往來寒熱，胸脇苦滿。\n", encoding="utf-8")
    # 4) depth-3 nested sub-sub-book（十五輪 P0-3：遞歸解析守衛——
    #    以前的一層遍歷會把它整個丟掉）
    deep = sub / "丙氏傷寒注補遺"
    deep.mkdir()
    deep.joinpath("index.txt").write_text(
        "<book>\n書名=丙氏傷寒注補遺\n</book>\n\n=====補遺=====\n\n"
        "奔豚氣上衝，甚則腹痛。\n", encoding="utf-8")
    catalog = library.build_catalog(root, archive_sha256="fixture")
    library.build_char_index(root, catalog)


class TestLibraryParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        make_fixture(cls.root)
        cls.lib = library.Library(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_catalog_units_and_inheritance(self):
        cat = self.lib.catalog
        self.assertEqual(cat["n_books"], 3)
        self.assertEqual(cat["n_units"], 5)          # 嵌套子書+三層子子書
        sub = self.lib.info("丙氏全書/丙氏傷寒注")
        self.assertEqual(sub["parent"], "丙氏全書")
        self.assertEqual(sub["category"], "綜合")     # inherited from parent
        self.assertEqual(self.lib.info("甲乙經考")["extra"]["備考"], "測試用")

    def test_recursive_nesting_any_depth(self):
        # 十五輪 P0-3：A/B/C 三層每級都是文本單元，元數據沿最近祖先繼承
        cat = self.lib.catalog
        self.assertEqual(cat["max_depth"], 3)
        deep = self.lib.info("丙氏全書/丙氏傷寒注/丙氏傷寒注補遺")
        self.assertIsNotNone(deep)
        self.assertEqual(deep["parent"], "丙氏全書/丙氏傷寒注")
        self.assertEqual(deep["category"], "綜合")    # 隔代繼承
        self.assertEqual(deep["author"], "張試")
        # 父/子正文互不重複計入：各單元 files 只含本目錄文件
        mid = self.lib.info("丙氏全書/丙氏傷寒注")
        self.assertEqual(mid["files"], ["index.txt", "1.txt"])
        self.assertEqual(deep["files"], ["index.txt"])
        # 三層正文可被全文檢索到
        out = self.lib.grep("奔豚氣上衝")
        self.assertEqual(out["hits"][0]["book_id"],
                         "丙氏全書/丙氏傷寒注/丙氏傷寒注補遺")

    def test_reading_order_decimal_stems_menu_excluded(self):
        files = self.lib.info("乙部方書")["files"]
        self.assertEqual(files, ["index.txt", "1.txt", "2-0.3.txt", "2-15.txt"])
        self.assertNotIn("menu.txt", files)

    def test_catalog_search_and_categories(self):
        hits = self.lib.search("方書")
        self.assertEqual(hits[0]["title"], "乙部方書")
        self.assertTrue(self.lib.search("李試"))      # author match
        self.assertEqual(self.lib.categories()["方書"], 1)
        self.assertEqual([h["title"] for h in self.lib.search("", category="醫經")],
                         ["甲乙經考"])

    def test_toc_and_section_read(self):
        toc = self.lib.toc("甲乙經考")
        self.assertIn("總論", [t["title"] for t in toc])
        out = self.lib.read("《甲乙經考》", section="總論")
        self.assertIn("人參味甘", out["text"])
        out = self.lib.read("甲乙經考", section="不存在章節")
        self.assertIn("error", out)
        self.assertIn("總論", out["toc"])

    def test_read_offset_pagination(self):
        # 二十輪：章節全文點閱的「載入更多」——offset 窗口續讀，兩頁拼接
        # 等於全文；truncated 表示窗口之後仍有餘文
        full = self.lib.read("乙部方書", max_chars=100000)
        total = full["total_chars"]
        half = max(1, total // 2)
        p1 = self.lib.read("乙部方書", max_chars=half)
        self.assertTrue(p1["truncated"])
        self.assertEqual(p1["offset"], 0)
        p2 = self.lib.read("乙部方書", max_chars=100000,
                           offset=len(p1["text"]))
        self.assertEqual(p2["offset"], half)
        self.assertFalse(p2["truncated"])
        self.assertEqual(p1["text"] + p2["text"], full["text"])
        # 章節模式同樣支持 offset；越界 offset 收斂為空窗口
        sec = self.lib.read("甲乙經考", section="總論", max_chars=100000)
        paged = self.lib.read("甲乙經考", section="總論",
                              max_chars=100000, offset=3)
        self.assertEqual(sec["text"][3:], paged["text"])
        beyond = self.lib.read("乙部方書", offset=10 ** 9)
        self.assertEqual(beyond["text"], "")
        self.assertFalse(beyond["truncated"])

    def test_grep_unwraps_lines_folds_variants_locates_section(self):
        # match spans a hard line-wrap（主治中風，\n胸脅苦滿）
        out = self.lib.grep("胸脅苦滿")
        self.assertEqual(out["n_hits"], 2)            # 脅 and 脇 both fold
        sections = {h["section"] for h in out["hits"]}
        self.assertEqual(sections, {"卷一", "太陽篇"})
        self.assertFalse(out["scan_capped"])
        # absent characters short-circuit without scanning
        out = self.lib.grep("龍膽瀉肝")
        self.assertEqual((out["n_hits"], out["n_books_scanned"]), (0, 0))
        self.assertIn("error", self.lib.grep("短"))

    def test_status_and_availability(self):
        st = library.status(self.root)
        self.assertTrue(st["available"])
        self.assertEqual(st["n_books"], 3)
        with tempfile.TemporaryDirectory() as empty:
            self.assertFalse(library.is_available(Path(empty)))
            self.assertFalse(library.ensure_available(Path(empty), auto=False))
            self.assertIn("hint", library.status(Path(empty)))


class TestLibraryTool(unittest.TestCase):
    """Tool #19 against the fixture (config.LIBRARY_DIR monkeypatched)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        make_fixture(Path(cls._tmp.name))
        cls._saved = config.LIBRARY_DIR
        config.LIBRARY_DIR = Path(cls._tmp.name)
        from hermes_shanghan.agent.tools import ToolRegistry
        cls.reg = ToolRegistry()

    @classmethod
    def tearDownClass(cls):
        config.LIBRARY_DIR = cls._saved
        cls._tmp.cleanup()

    def test_tool_search_mode(self):
        out = self.reg.call("shanghan_library", {"query": "胸脅苦滿"})
        self.assertTrue(out["available"])
        self.assertEqual(out["mode"], "search")
        self.assertGreaterEqual(out["n_text_hits"], 2)
        self.assertIn("非經文層", out["evidence_layer"])

    def test_tool_read_and_overview(self):
        out = self.reg.call("shanghan_library", {"book": "甲乙經考"})
        self.assertEqual(out["mode"], "read")
        self.assertIn("人參", out["text"])
        out = self.reg.call("shanghan_library", {})
        self.assertEqual(out["mode"], "overview")
        self.assertEqual(out["n_books"], 3)

    def test_tool_unavailable_hints_fetch(self):
        with tempfile.TemporaryDirectory() as empty:
            saved = config.LIBRARY_DIR
            config.LIBRARY_DIR = Path(empty)
            try:
                out = self.reg.call("shanghan_library", {"query": "人參"})
            finally:
                config.LIBRARY_DIR = saved
        self.assertFalse(out["available"])
        self.assertIn("library fetch", out["hint"])


class TestLibraryRouting(unittest.TestCase):
    """Offline auto-selection routes literature questions to the tool."""

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

    def test_routes_to_library(self):
        from hermes_shanghan.agent.agent import ShanghanAgent
        out = ShanghanAgent().ask("歷代醫書裡有哪些書記載奔豚？", role="researcher")
        self.assertIn("shanghan_library", out["tools_used"])

    def test_stats_still_routes_to_corpus_stats(self):
        from hermes_shanghan.agent.agent import ShanghanAgent
        out = ShanghanAgent().ask("全庫統計一共多少條規則？", role="researcher")
        self.assertIn("shanghan_corpus_stats", out["tools_used"])

    def test_complex_agent_literature_kind(self):
        from hermes_shanghan.agent.complex_agent import TASK_TYPES, ComplexAgent
        self.assertIn("shanghan_library", TASK_TYPES["literature"]["tools"])
        agent = ComplexAgent()
        sub = agent._decompose_local("笈成全庫書目裡查一下奔豚；桂枝湯的劑量折算")
        kinds = [t["kind"] for t in sub]
        self.assertIn("literature", kinds)


class TestRealLibraryIfPresent(unittest.TestCase):
    def test_real_library_sanity(self):
        if not library.is_available():
            self.skipTest("full library not fetched (run `library fetch`)")
        lib = library.Library()
        self.assertGreaterEqual(lib.catalog["n_books"], 800)
        self.assertTrue(lib.search("傷寒論"))
        self.assertTrue(lib.read("傷寒論_宋本", max_chars=200)["text"])


if __name__ == "__main__":
    unittest.main()
