"""藥證檔案（C10 藥解）：單味藥在《傷寒論》中的可計算畫像。

全部字段由既有確定性資產推導：<F> 方塊組成（A 層）、劑量計量層、
方證規則、條文實體標註。刻意不做的事（如實聲明）：
藥性/功效解釋屬本草層與注文層，非傷寒論原文直述，本檔案不編造；
「角色變化」（君臣佐使）屬後世方論歸納，僅給出可計算的配伍事實。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from .. import config
from ..schemas import read_jsonl
from ..textutil import fold_variants, normalize_query


def herb_profile(name: str, clause_offset: int = 0, clause_limit: int = 20,
                 bencao_offset: int = 0, bencao_limit: int = 4) -> Dict:
    """單味藥檔案。條文與本草摘錄均分頁（offset/limit + has_more），
    UI「載入更多」即續讀（十七輪）。"""
    q = normalize_query(name)
    formula_rules = read_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")

    # 出現方劑 + 配伍網絡（同方共現計數）
    formulas: List[Dict] = []
    partners: Dict[str, int] = {}
    canonical_name = ""
    for r in formula_rules:
        herbs = [c.get("herb", "") for c in r.get("composition", [])]
        hit = next((h for h in herbs if fold_variants(h) == q
                    or q in fold_variants(h)), "")
        if not hit:
            continue
        canonical_name = canonical_name or hit
        formulas.append({"formula": r.get("formula", ""),
                         "supporting_clauses": r.get("supporting_clauses", [])[:3],
                         "core_pattern": r.get("core_pattern", "")[:40]})
        for h in herbs:
            if h and h != hit:
                partners[h] = partners.get(h, 0) + 1
    if not formulas:
        return {"error": f"未在方劑組成中找到藥物 {name}"}

    # 劑量計量層：劑量範圍與眾數
    dose_rows = []
    dose_path = config.RESEARCH_DIR / "dose_table.json"
    if dose_path.exists():
        table = json.loads(dose_path.read_text(encoding="utf-8"))
        dose_rows = [row for row in table.get("rows", [])
                     if fold_variants(row.get("herb", "")) == fold_variants(canonical_name)]
    weights = sorted({row.get("raw", "") for row in dose_rows if row.get("raw")})

    # 條文出現（實體標註層）
    clause_ids = []
    for c in read_jsonl(config.CLAUSE_DIR / "clauses.jsonl"):
        if any(fold_variants(h) == fold_variants(canonical_name)
               for h in c.get("herbs", [])):
            clause_ids.append(c["clause_id"])

    top_partners = sorted(partners.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    bencao = bencao_evidence(canonical_name, offset=bencao_offset,
                             limit=bencao_limit)
    roles = role_evidence(canonical_name)
    c_off = max(0, int(clause_offset))
    c_lim = max(1, min(100, int(clause_limit)))
    return {
        "herb": canonical_name,
        "n_formulas": len(formulas),
        "formulas": formulas,
        "n_clauses": len(clause_ids),
        "clause_ids": clause_ids[c_off:c_off + c_lim],
        "clause_offset": c_off,
        "clauses_has_more": c_off + c_lim < len(clause_ids),
        "dose_variants": weights[:15],
        "n_dose_records": len(dose_rows),
        "top_partners": [{"herb": h, "n_formulas_together": n}
                         for h, n in top_partners],
        "bencao_layer": bencao,
        "role_evidence": roles,
        "section_evidence_levels": {
            "formulas": "A 原文直述（<F> 方塊組成）",
            "clause_ids": "A 條文實體標註",
            "dose_variants": "A 原文劑量寫法（折算屬 D 層）",
            "top_partners": "同方共現計數（可計算事實）",
            "bencao_layer": "本草層（旁證/文獻查閱，不入經文閘門）",
            "role_evidence": "A 層可計算事件（加減味/劑量調整，不做君臣佐使歸納）",
        },
        "warnings": ["藥性/功效解釋屬本草層（見 bencao_layer，需 library "
                     "fetch），與傷寒 A 層事實嚴格分層；君臣佐使等角色歸納"
                     "屬後世方論，本檔案不編造。"],
    }


# ---------------------------------------------------------------------------
# 本草證據層（旁證：神農本草經等原文摘錄，嚴格分層，不入經文閘門）
# ---------------------------------------------------------------------------
BENCAO_BOOKS = ["神農本草經", "名醫別錄", "本草經集注", "證類本草", "本草綱目"]


def bencao_evidence(herb: str, offset: int = 0, limit: int = 4) -> Dict:
    """從中醫笈成全庫的本草類書中取該藥的原文摘錄（書·章節定位）。

    嚴格分層：傷寒 A 層=方劑/劑量/配伍事實；本草層=藥性功效（旁證，
    出處供查閱，不進入經文層證據閘門）。庫未下載時如實返回不可用。
    摘錄攜帶 book_id/section——UI 可點擊展開該書該節全文；offset/limit
    分頁續讀更多本草書。"""
    from ..corpus import library
    if not library.is_available():
        return {"available": False,
                "note": "本草層需先下載全庫（`library fetch`）；"
                        "傷寒 A 層事實不受影響。"}
    lib = library.Library()
    offset = max(0, int(offset))
    limit = max(1, min(12, int(limit)))
    res = lib.grep(herb, category="本草", limit=offset + limit * 2, per_book=1)
    wanted = []
    for h in res.get("hits", []):
        rank = next((i for i, b in enumerate(BENCAO_BOOKS)
                     if b in h.get("title", "")), len(BENCAO_BOOKS))
        wanted.append((rank, h))
    wanted.sort(key=lambda x: (x[0], x[1].get("title", "")))
    page = wanted[offset:offset + limit]
    excerpts = []
    for _, h in page:
        text = h.get("excerpt", "")
        entry = {"book": h.get("title", ""), "book_id": h.get("book_id", ""),
                 "author": h.get("author", ""),
                 "dynasty": h.get("dynasty", ""), "section": h.get("section", ""),
                 "excerpt": text[:120]}
        # 性味結構化（僅逐字提取「味X…寒/熱/溫/涼/平」句式；提不到不編造）
        m = re.search(r"味([㐀-鿿]{1,4}?)[，、]?\s*((?:微|大)?[寒熱温溫涼凉平])",
                      fold_variants(text))
        if m:
            entry["nature_flavor"] = {"flavor": m.group(1), "nature": m.group(2),
                                      "source_layer": "本草層逐字提取（旁證）"}
        excerpts.append(entry)
    return {"available": True, "n_hits": res.get("n_hits", 0),
            "excerpts": excerpts,
            "offset": offset,
            "has_more": offset + len(page) < len(wanted),
            "note": "本草層＝旁證（藥性功效屬本草文獻，非傷寒原文直述）；"
                    "摘錄按書·章節定位（book_id 可經 shanghan_library 點閱"
                    "全節原文）；nature_flavor 僅在原文出現「味X性Y」"
                    "句式時逐字提取，提不到不編造。"}


def role_evidence(herb: str) -> List[Dict]:
    """方中作用的可計算證據：家族劑量演化中該藥的增減量/加減味事件。

    「量變致新方」（桂枝湯→桂枝加桂湯 = 桂枝×1.67）是該藥在方中作用的
    A 層可計算證據——不做君臣佐使歸納，只列事件。"""
    import json as _json
    evo_path = config.RESEARCH_DIR / "dose_family_evolution.json"
    if not evo_path.exists():
        return []
    data = _json.loads(evo_path.read_text(encoding="utf-8"))
    out = []
    hq = fold_variants(herb)
    for e in data.get("edges", []):
        added = fold_variants(e.get("added_herbs", "") or "")
        removed = fold_variants(e.get("removed_herbs", "") or "")
        deltas = [d for d in e.get("dose_deltas", [])
                  if fold_variants(d.get("herb", "")) == hq]
        if hq in added or hq in removed or deltas:
            out.append({"base": e.get("base", ""), "modified": e.get("modified", ""),
                        "edge_kind": e.get("edge_kind", ""),
                        "event": ("加味" if hq in added else
                                  "減味" if hq in removed else "劑量調整"),
                        "dose_deltas": deltas})
    return out[:10]
