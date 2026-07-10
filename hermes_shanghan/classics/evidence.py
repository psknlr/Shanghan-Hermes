"""P 層（全庫文獻）正式證據對象（十五輪 P0-2）。

十四輪之前，全庫文獻是「證據系統之外的文本」：可讀、可答，但無法嚴格
證明回答引用了哪一段。本模塊把 P 層升格為**一等證據**——不是混入 A 層，
而是分層並行：

    EvidenceRecord(P)：work/edition/passage 身份 + verbatim_text +
    字符座標 + quote_hash + 檢索上下文（query/rank）——逐字可重驗。

發布時按**結論類型**決定最低證據層（CONCLUSION_EVIDENCE_POLICY）：
「宋本原文記載」必須 A 層；「最早提出」必須時間有序檢索 + 反證搜索。
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Sequence

from .model import Passage, PassageIndex, stable_id

NORMALIZATION_NOTE = ("verbatim_text 為扁平化正文原字（未折疊）；"
                      "char_start/char_end 為扁平化正文座標；"
                      "匹配時折疊異體字（1:1 映射，座標對齊）")


def quote_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def passage_evidence(p: Passage, unit: Dict, char_start: int, char_end: int,
                     retrieval_query: str = "", retrieval_rank: int = 0,
                     max_chars: int = 120) -> Dict:
    """從一個 Passage 命中構造 P 層 EvidenceRecord。"""
    lo = max(0, char_start - 30)
    hi = min(len(p.flat_text), max(char_end + 30, lo + 1))
    verbatim = p.flat_text[lo:hi][:max_chars]
    return {
        "evidence_level": "P",
        "work_id": unit["id"],
        "work_title": unit["title"],
        "author": unit["author"],
        "dynasty": unit["dynasty"],
        "category": unit["category"],
        "edition_id": unit.get("edition", "") or unit["id"],
        "passage_id": p.passage_id,
        "file": p.file,
        "seq": p.seq,
        "section": p.section,
        "verbatim_text": verbatim,
        "char_start": lo,
        "char_end": lo + len(verbatim),
        "quote_hash": quote_hash(verbatim),
        "normalization": NORMALIZATION_NOTE,
        "retrieval_query": retrieval_query,
        "retrieval_rank": retrieval_rank,
        "visible_to_model": True,
    }


def evidence_from_hit(index: PassageIndex, hit: Dict, query: str,
                      rank: int) -> Optional[Dict]:
    unit = index.lib._by_id.get(hit["work_id"])
    p = index.get(hit["passage_id"], work=hit["work_id"])
    if unit is None or p is None:
        return None
    return passage_evidence(p, unit, hit["char_start"], hit["char_end"],
                            retrieval_query=query, retrieval_rank=rank)


def verify_records(records: Sequence[Dict], index: PassageIndex) -> Dict:
    """逐字重驗：回到庫中該 Passage，按座標切片對照 verbatim + quote_hash。
    書名/章節/摘錄/結論之間是否對應——不再靠信任，靠重驗。"""
    failures: List[Dict] = []
    n_ok = 0
    for r in records:
        p = index.get(r.get("passage_id", ""), work=r.get("work_id", ""))
        if p is None:
            failures.append({"passage_id": r.get("passage_id"),
                             "reason": "passage_not_found"})
            continue
        sliced = p.flat_text[r.get("char_start", 0):r.get("char_end", 0)]
        if sliced != r.get("verbatim_text") or \
                quote_hash(sliced) != r.get("quote_hash"):
            failures.append({"passage_id": r.get("passage_id"),
                             "reason": "verbatim_mismatch"})
            continue
        n_ok += 1
    return {"ok": not failures, "n_verified": n_ok,
            "n_failed": len(failures), "failures": failures}


def build_packet(records: Sequence[Dict], index: PassageIndex,
                 topic: str = "") -> Dict:
    """證據包導出：記錄 + 重驗結果 + 庫指紋——論文/審計可直接引用。"""
    recs = list(records)
    verification = verify_records(recs, index)
    cat = index.lib.catalog
    body = "|".join(sorted(r.get("quote_hash", "") for r in recs))
    return {"packet_id": stable_id("pkt", f"{topic}|{body}"),
            "topic": topic, "n_records": len(recs),
            "n_works": len({r.get("work_id") for r in recs}),
            "records": recs, "verification": verification,
            "library_fingerprint": cat.get("archive_sha256", ""),
            "note": "P 層文獻旁證包：逐條 verbatim+座標+quote_hash 已重驗；"
                    "不進入 A 層經文規則閘門"}


# ---------------------------------------------------------------------------
# 按結論類型的最低證據層策略（十五輪 P0-2 評審表，逐行落地）
# ---------------------------------------------------------------------------
CONCLUSION_EVIDENCE_POLICY = (
    {"conclusion_type": "宋本原文記載", "minimum": "A 層條文引用（SHL_SONGBEN_*）"},
    {"conclusion_type": "某注家認為", "minimum": "對應 C/P 層原文（注文對齊或全庫段落）"},
    {"conclusion_type": "後世醫家普遍討論", "minimum": "≥2 個不同著作的 P 層來源"},
    {"conclusion_type": "系統綜合推斷", "minimum": "顯式標明 E 層並鏈接基礎證據"},
    {"conclusion_type": "最早提出", "minimum": "時間有序全庫檢索 + 反證搜索"
                                              "（classics_trace_citation）"},
)

RE_SONGBEN_CLAIM = re.compile(r"宋本(?:原文)?(?:記載|明言|直述|载|記)")
RE_EARLIEST_CLAIM = re.compile(r"(?:最早|最先|首見|首见|首載|首载|首現|首现|首倡)")
RE_CONSENSUS_CLAIM = re.compile(r"(?:後世|历代|歷代)?(?:醫家|注家|諸家)"
                                r"(?:普遍|多數|多数|咸|皆|均)")
RE_SHL_ID = re.compile(r"SHL_SONGBEN_(?:AUX_)?\d{4}")


def conclusion_policy_check(answer: str, p_records: Sequence[Dict],
                            tools_used: Sequence[str]) -> List[Dict]:
    """可檢測結論類型的最低證據層審查（確定性啟發，寧嚴勿鬆）。

    返回違例清單；空列表=未檢出違例（≠證明全部結論充分，剩餘結論
    類型的判定屬人工審核範圍，如實聲明）。"""
    answer = answer or ""
    violations: List[Dict] = []
    tools = set(tools_used or ())
    if RE_SONGBEN_CLAIM.search(answer) and not RE_SHL_ID.search(answer):
        violations.append({"conclusion_type": "宋本原文記載",
                           "violation": "宣稱宋本原文記載但未引 A 層條文編號"})
    if RE_EARLIEST_CLAIM.search(answer) and \
            "classics_trace_citation" not in tools:
        violations.append({"conclusion_type": "最早提出",
                           "violation": "含「最早/首見」類結論但未執行時間"
                                        "有序檢索+反證搜索（classics_trace_citation）"})
    if RE_CONSENSUS_CLAIM.search(answer):
        n_works = len({r.get("work_id") for r in p_records
                       if r.get("work_id")})
        if n_works < 2:
            violations.append({"conclusion_type": "後世醫家普遍討論",
                               "violation": f"「普遍/多數」類結論僅有 "
                                            f"{n_works} 個著作來源（<2）"})
    return violations
