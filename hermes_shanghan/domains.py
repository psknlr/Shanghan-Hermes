"""領域插件（十五輪 P1-5：從「聲明清單」升級為**可執行插件**）。

平台層（與領域無關）：harness、server/policy、health、corpus 供應鏈、
classics 全庫工具族/Passage 模型/P 層證據面、eval/trajectory 骨架、MCP。
領域層由 DomainPlugin 顯式掛載可執行接縫（工廠均為 import 路徑惰性
解析，加載失敗即插件不健康——不是願望字段）：

    tool_factory      向 ToolRegistry 註冊本領域工具
    agent_factory     構造本領域智能體
    passage_parser    文本 → 段落切分
    normalizer        字符規範化
    citation_parser   從回答抽取本領域引用 ID
    evidence_policy   本領域證據閘門口徑
    evaluation_suites 本領域測試套件
    ui_manifest       工作台頁面清單

誠實現狀：shanghan 與 classics 兩個插件 **active 且可執行**；
jingui / neijing 仍是 planned——字段顯式為 None，不偽裝已實現。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config


def _resolve(path: Optional[str]) -> Optional[Any]:
    """惰性解析 'module:attr' 工廠路徑；插件字段不是字符串裝飾。"""
    if not path:
        return None
    mod, _, attr = path.partition(":")
    return getattr(import_module(mod, package=None), attr)


@dataclass(frozen=True)
class DomainPlugin:
    domain_id: str
    name: str
    status: str = "active"                     # active | planned
    canonical_books: List[str] = field(default_factory=list)
    tool_prefix: str = ""
    corpus_categories: List[str] = field(default_factory=list)
    notes: str = ""
    # —— 可執行接縫（import 路徑；None = 未提供，如實聲明）——
    tool_factory: Optional[str] = None         # callable(add) 註冊工具
    agent_factory: Optional[str] = None        # 智能體類
    passage_parser: Optional[str] = None       # callable(text)->segments
    normalizer: Optional[str] = None           # callable(str)->str
    citation_parser: Optional[str] = None      # 正則或 callable
    evidence_policy: str = ""                  # 人可讀口徑聲明
    evaluation_suites: Tuple[str, ...] = ()
    ui_manifest: Dict[str, str] = field(default_factory=dict)

    def load_agent(self):
        return _resolve(self.agent_factory)

    def load_tool_factory(self):
        return _resolve(self.tool_factory)

    def load_normalizer(self):
        return _resolve(self.normalizer)

    def load_passage_parser(self):
        return _resolve(self.passage_parser)

    def load_citation_parser(self):
        return _resolve(self.citation_parser)

    def executable(self) -> bool:
        """active 插件必須至少可解析 agent/tool 工廠之一。"""
        if self.status != "active":
            return False
        try:
            return bool(self.load_agent() or self.load_tool_factory())
        except Exception:
            return False


# 向後兼容別名（十二輪引入的名稱）
DomainSpec = DomainPlugin

SHANGHAN = DomainPlugin(
    domain_id="shanghan",
    name="傷寒論",
    canonical_books=[config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK],
    tool_prefix="shanghan_",
    corpus_categories=["shanghan"],
    status="active",
    notes="領域插件一：398 條核心 + 異文/九注本/類方 + 28 領域工具",
    agent_factory="hermes_shanghan.agent.agent:ShanghanAgent",
    passage_parser="hermes_shanghan.corpus.segmenter:segment_canonical",
    normalizer="hermes_shanghan.textutil:normalize_query",
    citation_parser="hermes_shanghan.agent.citation_guard:RE_CLAUSE_ID",
    evidence_policy="strict_round：非拒答回答必須引用本輪 Broker 台賬中的 "
                    "A 層條文（primary_text_returned）",
    evaluation_suites=("tests/test_harness.py", "tests/test_evidence_integrity.py"),
    ui_manifest={"workbench": "/static/index.html",
                 "console": "/static/console.html"},
)

CLASSICS = DomainPlugin(
    domain_id="classics",
    name="全量古籍（中醫笈成全庫）",
    canonical_books=[],
    tool_prefix="classics_",
    corpus_categories=[],
    status="active",
    notes="領域插件二（十五輪）：803 部級全庫、Passage/Span 模型、"
          "P 層證據面、8 個 classics 工具、獨立 ClassicsAgent",
    tool_factory="hermes_shanghan.classics.tools:register_classics_tools",
    agent_factory="hermes_shanghan.classics.agent:ClassicsAgent",
    passage_parser="hermes_shanghan.classics.model:segment_file",
    normalizer="hermes_shanghan.textutil:fold_variants",
    citation_parser="hermes_shanghan.classics.model:RE_PASSAGE_ID",
    evidence_policy="P 層分層證據：引用只認 Broker 台賬 passage 記錄"
                    "（verbatim+座標+quote_hash 可重驗）；按結論類型執行"
                    "最低證據層（宋本原文→A；最早提出→時間有序+反證）",
    evaluation_suites=("tests/test_classics.py",),
    ui_manifest={"workbench": "/static/classics.html"},
)

JINGUI = DomainPlugin(
    domain_id="jingui", name="金匱要略", tool_prefix="jingui_",
    corpus_categories=["jingui"], status="planned",
    notes="語料已在 corpus_raw/jingui（P 層）；切分/實體/規則歸納器待建"
          "——工廠字段顯式為 None，不偽裝已實現")

NEIJING = DomainPlugin(
    domain_id="neijing", name="黃帝內經", tool_prefix="neijing_",
    status="planned",
    notes="全庫（笈成）已含相關書目，可經 classics 插件檢索；"
          "專屬領域插件待建")

DOMAINS: Dict[str, DomainPlugin] = {d.domain_id: d for d in
                                    (SHANGHAN, CLASSICS, JINGUI, NEIJING)}


def get_plugin(domain_id: str) -> Optional[DomainPlugin]:
    return DOMAINS.get(domain_id)


def active_domains() -> List[DomainPlugin]:
    return [d for d in DOMAINS.values() if d.status == "active"]
