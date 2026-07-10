"""領域插件 seam（十二輪 P2 起步：通用古籍平台 = Core + Harness + 領域插件）。

誠實現狀：本倉庫當前只有一個領域（傷寒論），大量 API/工具名帶 shanghan
前綴——**尚不是**通用平台。本模塊做的是把「哪些東西屬於領域、哪些屬於
平台」顯式化為註冊表：

- 平台層（與領域無關，已可複用）：harness（RunSpec/狀態圖/預算/發布閘門/
  span 軌跡）、server/policy（Principal/RequestContext/投影）、health、
  corpus/worktype、corpus/library 供應鏈、trace/evidence 記錄結構、
  eval/trajectory 骨架、MCP 協議層；
- 領域層（傷寒論專屬）：語料書目與切分規則、實體詞表、方證/六經/誤治
  規則歸納器、工具面、提示詞、UI 文案。

新領域（金匱/內經）按 DomainSpec 落位；遷移計劃見 docs/PLATFORM.md。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from . import config


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    name: str
    canonical_books: List[str] = field(default_factory=list)
    tool_prefix: str = ""
    corpus_categories: List[str] = field(default_factory=list)
    status: str = "active"          # active | planned
    notes: str = ""


SHANGHAN = DomainSpec(
    domain_id="shanghan",
    name="傷寒論",
    canonical_books=[config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK],
    tool_prefix="shanghan_",
    corpus_categories=["shanghan"],
    status="active",
    notes="當前唯一已落地領域：398 條核心 + 異文/九注本/類方 + 28 工具",
)

JINGUI = DomainSpec(
    domain_id="jingui", name="金匱要略", tool_prefix="jingui_",
    corpus_categories=["jingui"], status="planned",
    notes="語料已在 corpus_raw/jingui（P 層）；切分/實體/規則歸納器待建")

NEIJING = DomainSpec(
    domain_id="neijing", name="黃帝內經", tool_prefix="neijing_",
    status="planned", notes="全庫（笈成）已含相關書目；領域插件待建")

DOMAINS: Dict[str, DomainSpec] = {d.domain_id: d
                                  for d in (SHANGHAN, JINGUI, NEIJING)}


def active_domains() -> List[DomainSpec]:
    return [d for d in DOMAINS.values() if d.status == "active"]
