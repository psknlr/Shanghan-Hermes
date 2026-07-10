"""文獻類型分類（十輪評審 六.2：證據層不得由目錄名默認決定）。

舊行為 `"C" if category == "shanghan" else "D"` 把注本/類方書/醫案/校勘書/
綜合著作/現代整理本一律劃成 C 或 D。本模塊改為：

1. **顯式註冊優先**：config 中登記的書目（宋本/異文/九注本/類方）保持
   人工核定的 work_type 與證據層；
2. **未登記書目 fail-closed 到 P 層（旁證）**：寧可降級也不冒充注家層——
   C 層意味着「可作注家解釋證據」，這個資格必須人工登記，不能由目錄猜；
3. 同時給出**確定性的 work_type 推斷**（標題/元數據線索）作為編目輔助，
   但推斷結果只寫進 ``work_type_inferred``，**不影響證據層**；
4. 每本書記錄 ``layer_basis``：這一層是怎麼來的（registered / fail_closed），
   審計可查。

證據等級的完整決定式為 文獻類型 + 版本可靠性(品質元數據) + 引用方式 +
文本質量——前兩項在本模塊與 manifest；引用方式與逐字質量在引文掃描層
（trace/quotation 的 mode/coverage），四者在 EvidenceRecord 匯合。
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from .. import config

# 十類 work type（評審建議清單）
WORK_TYPES = (
    "canonical_text",      # 經文底本（宋本條文版/宋本全帙）
    "variant_edition",     # 版本異文（桂本/千金翼）
    "commentary",          # 注本（成注/條辨/來蘇集…）
    "collation",           # 校勘/輯佚（輯義類）
    "formula_family",      # 類方書（傷寒論類方）
    "medical_case",        # 醫案
    "teaching_summary",    # 講義/歌括/入門
    "modern_research",     # 現代整理/研究
    "dictionary",          # 字書/辭書
    "secondary_quote",     # 轉引彙編
    "unclassified",        # 未能歸類（fail-closed）
)

# 顯式註冊：書目 → work_type（證據層沿用 config.LAYER_OF_BOOK 的人工核定）
WORK_TYPE_OF_BOOK: Dict[str, str] = {
    config.PRIMARY_BOOK: "canonical_text",
    config.SONGBEN_FULL_BOOK: "canonical_text",
    **{b: "variant_edition" for b in config.VARIANT_BOOKS},
    **{b: "commentary" for b in config.COMMENTARY_BOOKS},
    # 輯義：丹波元簡輯諸家之義，兼校勘性質——註冊為 collation 更如實，
    # 其注文仍經 COMMENTARY_BOOKS 顯式管道進 C 層（雙重身份如實記錄）
    "傷寒論輯義": "collation",
    **{b: "formula_family" for b in config.FORMULA_FAMILY_BOOKS},
}

# 確定性推斷線索（僅供編目輔助，不決定證據層）
_INFER_PATTERNS = [
    ("medical_case", r"醫案|驗案|治驗|診籍|臨證指南"),
    ("formula_family", r"類方|方論|方解|方考"),
    ("collation", r"輯義|輯注|校注|校勘|考證|考異"),
    ("dictionary", r"字典|辭典|音義|釋名"),
    ("teaching_summary", r"歌括|歌訣|淺注|入門|白話|講義|啟蒙"),
    ("modern_research", r"研究|新解|譯釋|語譯"),
    ("commentary", r"[注註箋疏]|條辨|直解|集解|發微|懸解|貫珠|溯源集"),
]


def infer_work_type(title: str, meta: Optional[Dict] = None) -> str:
    """標題/元數據線索的確定性推斷——只作編目輔助（layer 不採信）。"""
    blob = title + " " + " ".join((meta or {}).values())
    for wt, pat in _INFER_PATTERNS:
        if re.search(pat, blob):
            return wt
    return "unclassified"


def classify(book_dir: str, category: str = "",
             meta: Optional[Dict] = None) -> Tuple[str, str, str, str]:
    """返回 (work_type, hermes_layer, layer_basis, work_type_inferred)。

    - 已註冊：人工核定的類型與層；
    - 未註冊：layer 一律 **P（旁證）**——fail-closed，目錄名與標題推斷
      都不能授予 C/D 資格；推斷類型另存 ``work_type_inferred`` 供人工
      複核後升格註冊。
    """
    registered = WORK_TYPE_OF_BOOK.get(book_dir)
    if registered:
        layer = config.LAYER_OF_BOOK.get(book_dir, "P")
        return registered, layer, "registered", registered
    inferred = infer_work_type(book_dir, meta)
    return "unclassified", "P", "fail_closed_unregistered", inferred
