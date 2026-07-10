"""分層全庫檢索（十五輪 P1-2）：每一層可解釋，不輸出混合黑箱分數。

    L0  元數據篩選（朝代/作者/分類/著作）——編目字段，零 IO
    L1  字符倒排剪枝（精確：候選集可證明包含全部真命中）
    L2  逐段逐字驗證：布爾（AND/OR/NOT）+ 鄰近窗口 + 全量命中計數
        + 字符座標（扁平化正文座標，fold 為 1:1 映射故座標對齊原字）

誠實邊界（如實返回，不偽裝全庫掃描）：
- ``scan_capped=True`` 時零命中只說明「前 max_scan 個候選單元中沒有」；
- 通假字/古今詞/同義詞擴展、BM25、語義向量與學術重要度重排未實現，
  屬 L3-L6 路線（見 docs/PLATFORM.md），本層不冒充。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..corpus import library as _lib
from ..textutil import find_all, fold_variants
from .model import Passage, PassageIndex, dynasty_rank

MAX_POSITIONS_PER_PASSAGE = 20      # 座標列表封頂（計數不封頂）


class PassageSearcher:
    def __init__(self, lib: "_lib.Library"):
        self.lib = lib
        self.index = PassageIndex(lib)

    # ------------------------------------------------------------------
    def search(self, query: str = "", any_terms: Sequence[str] = (),
               not_terms: Sequence[str] = (), near: int = 0,
               category: str = "", dynasty: str = "", author: str = "",
               work: str = "", limit: int = 12, per_book: int = 3,
               max_scan: int = 200, order: str = "relevance") -> Dict:
        """布爾檢索：``query`` 空白分詞為 AND 項；``any_terms`` 至少命中
        其一；``not_terms`` 排除；``near>0`` 時前兩個 AND 項須在該字符
        窗口內共現。返回逐層解釋 + 帶座標/計數的段級命中。"""
        and_terms = [fold_variants(t) for t in (query or "").split() if t]
        anys = [fold_variants(t) for t in any_terms if t]
        nots = [fold_variants(t) for t in not_terms if t]
        if not and_terms and not anys:
            return {"error": "至少提供一個檢索項（query 或 any_terms）"}
        if any(len(t) < 2 for t in and_terms + anys):
            return {"error": "檢索項至少 2 字（單字歧義過大）"}
        limit = max(1, min(int(limit or 12), 100))
        per_book = max(1, min(int(per_book or 3), 20))
        max_scan = max(1, min(int(max_scan or 200), 2000))

        # L0 元數據篩選
        l0_units = []
        for u in self.lib.units:
            if category and category not in u["category"]:
                continue
            if dynasty and dynasty not in u["dynasty"]:
                continue
            if author and fold_variants(author) not in fold_variants(u["author"]):
                continue
            if work and fold_variants(work) not in fold_variants(u["title"]) \
                    and not u["id"].startswith(work):
                continue
            l0_units.append(u)
        l0_ords = {u["id"] for u in l0_units}

        # L1 字符倒排剪枝（對 AND 項全部剪枝；純 OR 查詢取並集）
        probe_terms = and_terms or anys
        cand_ord = self._l1_candidates(probe_terms, union=not and_terms)
        cands = [self.lib.units[i] for i in cand_ord
                 if self.lib.units[i]["id"] in l0_ords]

        # L2 逐段逐字驗證
        hits: List[Dict] = []
        scanned = 0
        capped = False
        total_occurrences = 0
        for u in cands:
            if len(hits) >= limit:
                break
            if scanned >= max_scan:
                capped = True
                break
            scanned += 1
            book_hits = 0
            for p in self.index.unit_passages(u):
                if book_hits >= per_book or len(hits) >= limit:
                    break
                h = self._match_passage(p, u, and_terms, anys, nots, near)
                if h:
                    total_occurrences += h["n_occurrences"]
                    hits.append(h)
                    book_hits += 1
        if order == "dynasty":
            hits.sort(key=lambda h: (h["dynasty_rank"], h["work_id"],
                                     h["file"], h["seq"]))
        layers = {
            "L0_metadata": {"n_units_after": len(l0_units),
                            "filters": {k: v for k, v in
                                        (("category", category),
                                         ("dynasty", dynasty),
                                         ("author", author),
                                         ("work", work)) if v}},
            "L1_char_index": {"n_candidates": len(cands),
                              "mode": "intersection" if and_terms else "union"},
            "L2_verbatim_scan": {"n_units_scanned": scanned,
                                 "max_scan": max_scan,
                                 "scan_capped": capped},
            "L3_plus": "通假/同義擴展、BM25、語義召回、重要度重排未實現"
                       "（規劃層，不冒充）",
        }
        note = ("scan_capped=true：零命中僅說明前 max_scan 個候選中沒有，"
                "非全庫不存在——調大 max_scan 或加過濾條件"
                if capped and not hits else "")
        return {"query": query, "any_terms": list(any_terms),
                "not_terms": list(not_terms), "near": near,
                "n_hits": len(hits), "n_occurrences_in_hits": total_occurrences,
                "scan_capped": capped, "order": order,
                "retrieval_layers": layers, "hits": hits,
                **({"note": note} if note else {})}

    # ------------------------------------------------------------------
    def _l1_candidates(self, terms: Sequence[str], union: bool) -> List[int]:
        chars_index = self.lib._load_charindex()["chars"]
        if not chars_index:
            return list(range(len(self.lib.units)))
        cjk = sorted({ch for t in terms for ch in t if ord(ch) >= 128})
        if not cjk:
            return list(range(len(self.lib.units)))
        if union:
            result: set = set()
            for ch in cjk:
                result |= set(chars_index.get(ch, []))
            return sorted(result)
        if any(ch not in chars_index for ch in cjk):
            return []
        rare = sorted((len(chars_index[ch]), ch) for ch in cjk)
        result = set(chars_index[rare[0][1]])
        for _, ch in rare[1:4]:
            result &= set(chars_index[ch])
        return sorted(result)

    def _match_passage(self, p: Passage, u: Dict, and_terms, anys, nots,
                       near: int) -> Optional[Dict]:
        folded = fold_variants(p.flat_text)
        positions: Dict[str, List[int]] = {}
        for t in and_terms:
            pos = find_all(folded, t)
            if not pos:
                return None
            positions[t] = pos
        matched_any = ""
        if anys:
            for t in anys:
                pos = find_all(folded, t)
                if pos:
                    positions[t] = pos
                    matched_any = t
                    break
            else:
                return None
        for t in nots:
            if t in folded:
                return None
        if near > 0 and len(and_terms) >= 2:
            a, b = and_terms[0], and_terms[1]
            if min(abs(i - j) for i in positions[a]
                   for j in positions[b]) > near:
                return None
        anchor = and_terms[0] if and_terms else matched_any
        first = positions[anchor][0]
        n_occ = sum(len(v) for v in positions.values())
        lo, hi = max(0, first - 40), first + len(anchor) + 40
        return {"passage_id": p.passage_id, "work_id": u["id"],
                "title": u["title"], "author": u["author"],
                "dynasty": u["dynasty"], "dynasty_rank": dynasty_rank(u["dynasty"]),
                "category": u["category"], "file": p.file, "seq": p.seq,
                "section": p.section,
                "n_occurrences": n_occ,
                "occurrence_positions": {
                    t: v[:MAX_POSITIONS_PER_PASSAGE]
                    for t, v in positions.items()},
                "char_start": first, "char_end": first + len(anchor),
                "excerpt": p.flat_text[lo:hi],       # 原字（未折疊）
                "matched_terms": sorted(positions)}
