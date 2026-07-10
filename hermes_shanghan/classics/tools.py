"""classics_* 工具族（十五輪 P0-1）：全庫古籍研究的通用工具面。

與 shanghan_* 領域工具並列註冊進同一個 ToolRegistry——因此自動獲得
Broker 台賬登記、span 軌跡、預算扣減、契約導出（OpenAI/Anthropic/MCP）。
所有檢索/閱讀類結果攜帶 ``passage_evidence``（P 層 EvidenceRecord，
verbatim+座標+quote_hash 可重驗），由 Capability Broker 寫入證據台賬。

統計語義分離（P1-3）：classics_library_stats 統計**中醫笈成全庫**
（803 部醫籍），shanghan_corpus_stats 統計**傷寒論規則庫**——兩者不再
在產品層混為一談。
"""
from __future__ import annotations

import difflib
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..corpus import library as _libmod
from ..textutil import fold_variants
from .evidence import (CONCLUSION_EVIDENCE_POLICY, build_packet,
                       evidence_from_hit, passage_evidence)
from .model import dynasty_rank, work_base_title
from .search import PassageSearcher

EVIDENCE_LAYER_NOTE = ("P（文獻旁證層）：非仲景經文層；書·章節·逐字摘錄"
                       "可回源重驗，不進入 A 層規則閘門")

_SEARCHER_CACHE: Dict[Tuple[str, float], PassageSearcher] = {}


def _searcher() -> Optional[PassageSearcher]:
    """按（庫根目錄, 編目 mtime）緩存 Searcher——測試換庫/重建編目自動失效。"""
    root = _libmod.library_root()
    cat = root / _libmod.CATALOG_NAME
    if not cat.exists():
        return None
    key = (str(root), cat.stat().st_mtime)
    if key not in _SEARCHER_CACHE:
        _SEARCHER_CACHE.clear()
        _SEARCHER_CACHE[key] = PassageSearcher(_libmod.Library(root))
    return _SEARCHER_CACHE[key]


def _unavailable(tool: str) -> Dict:
    return {"tool": tool, "available": False,
            "hint": "全庫未就緒：請先運行 `python3 -m hermes_shanghan "
                    "library fetch`"}


def _attach_evidence(out: Dict, s: PassageSearcher, hits: List[Dict],
                     query: str) -> None:
    evs = []
    for rank, h in enumerate(hits):
        ev = evidence_from_hit(s.index, h, query, rank)
        if ev:
            evs.append(ev)
    out["passage_evidence"] = evs
    out["evidence_layer"] = EVIDENCE_LAYER_NOTE


# ---------------------------------------------------------------------------
# tool implementations
# ---------------------------------------------------------------------------
def t_search_passages(query: str = "", any_terms: Optional[List[str]] = None,
                      not_terms: Optional[List[str]] = None, near: int = 0,
                      category: str = "", dynasty: str = "", author: str = "",
                      work: str = "", limit: int = 8, per_book: int = 3,
                      max_scan: int = 200, order: str = "relevance") -> Dict:
    s = _searcher()
    if s is None:
        return _unavailable("classics_search_passages")
    out = s.search(query=query, any_terms=any_terms or [],
                   not_terms=not_terms or [], near=near, category=category,
                   dynasty=dynasty, author=author, work=work, limit=limit,
                   per_book=per_book, max_scan=max_scan, order=order)
    if "error" in out:
        return out
    out["tool"] = "classics_search_passages"
    out["available"] = True
    _attach_evidence(out, s, out["hits"], query)
    return out


def t_read_passage(passage_id: str = "", work: str = "",
                   section: str = "", max_chars: int = 4000) -> Dict:
    s = _searcher()
    if s is None:
        return _unavailable("classics_read_passage")
    if passage_id:
        p = s.index.get(passage_id, work=work)
        if p is None:
            return {"error": f"未找到段落 {passage_id}"
                             + ("" if work else "（未提供 work 提示，掃描封頂"
                                                "400 單元——可補 work 參數重試）")}
        unit = s.lib._by_id[p.work_id]
    else:
        u = s.lib._resolve(work)
        if u is None:
            return {"error": f"全庫查無此書：{work}"}
        sec = fold_variants(section)
        p = next((x for x in s.index.unit_passages(u)
                  if not sec or sec in fold_variants(x.section)), None)
        if p is None:
            return {"error": f"《{u['title']}》查無章節：{section}",
                    "toc": [t["title"] for t in s.lib.toc(u["id"])][:40]}
        unit = u
    text = p.flat_text[:max(200, min(int(max_chars or 4000), 20000))]
    ev = passage_evidence(p, unit, 0, len(p.flat_text),
                          retrieval_query=f"read:{passage_id or work}")
    return {"tool": "classics_read_passage", "available": True,
            "locator": p.locator(),
            "work": {k: unit[k] for k in ("id", "title", "author", "dynasty",
                                          "category")},
            "text": text, "truncated": len(p.flat_text) > len(text),
            "total_chars": len(p.flat_text),
            "passage_evidence": [ev], "evidence_layer": EVIDENCE_LAYER_NOTE}


def t_compare_witnesses(work: str, query: str = "", limit: int = 6) -> Dict:
    """同一著作不同傳本對照：按折疊書名歸組 Witness；帶 query 時取各
    傳本命中段對照 + 兩兩相似度（difflib，確定性）。"""
    s = _searcher()
    if s is None:
        return _unavailable("classics_compare_witnesses")
    base = work_base_title(work)
    if not base:
        return {"error": "須提供著作名"}
    wits = [u for u in s.lib.units
            if work_base_title(u["title"]) == base
            or fold_variants(u["title"]).startswith(base)][:max(2, limit)]
    if not wits:
        return {"error": f"全庫未找到《{work}》的任何傳本"}
    out: Dict = {"tool": "classics_compare_witnesses", "available": True,
                 "work_base": base,
                 "n_witnesses": len(wits),
                 "witnesses": [{**{k: u[k] for k in
                                   ("id", "title", "author", "dynasty",
                                    "category", "edition", "quality")},
                                "n_files": len(u["files"]),
                                "approx_chars": u["approx_chars"]}
                               for u in wits]}
    if query and len(wits) >= 1:
        probes = []
        for u in wits:
            r = s.search(query=query, work=u["id"], limit=1, max_scan=50)
            h = (r.get("hits") or [None])[0]
            if h:
                probes.append(h)
        pairs = []
        for i in range(len(probes)):
            for j in range(i + 1, len(probes)):
                ratio = difflib.SequenceMatcher(
                    None, probes[i]["excerpt"], probes[j]["excerpt"]).ratio()
                pairs.append({"a": probes[i]["work_id"],
                              "b": probes[j]["work_id"],
                              "similarity": round(ratio, 3)})
        out["probe_query"] = query
        out["probe_hits"] = probes
        out["pairwise_similarity"] = pairs
        _attach_evidence(out, s, probes, query)
    else:
        out["passage_evidence"] = []
        out["evidence_layer"] = EVIDENCE_LAYER_NOTE
    out["note"] = ("Witness 歸組按折疊書名（同名異書需人工消歧）；"
                   "影印頁對齊未實現，屬路線層")
    return out


def t_trace_citation(quote: str, max_scan: int = 300, top: int = 12) -> Dict:
    """引文溯源：全庫時間有序檢索 + 反證搜索（部分匹配探針找更早候選）。

    誠實邊界：「在庫首現」≠「歷史首現」——庫外文獻與亡佚著作不可見；
    「首次明確表述」與「首次系統化」的區分屬人工判定，本工具給候選。"""
    s = _searcher()
    if s is None:
        return _unavailable("classics_trace_citation")
    quote = (quote or "").strip()
    flat_len = len(fold_variants("".join(quote.split())))
    if flat_len < 2:
        return {"error": "引文溯源至少 2 字"}
    # <4 字＝術語級首現（時間有序全量載錄）；≥8 字才做截半反證探針
    mode = "term_attestation" if flat_len < 4 else "quote_trace"
    r = s.search(query=quote, limit=top, per_book=2, max_scan=max_scan,
                 order="dynasty")
    if "error" in r:
        return r
    hits = r["hits"]
    earliest = hits[0] if hits else None
    # 反證搜索：截半探針找「更早的相近表述」（部分匹配，需人工核驗）
    counter: List[Dict] = []
    flat_q = "".join(quote.split())
    if earliest and len(flat_q) >= 8:
        half = max(4, len(flat_q) // 2)
        for probe in dict.fromkeys((flat_q[:half], flat_q[-half:])):
            cr = s.search(query=probe, limit=5, per_book=1,
                          max_scan=max_scan, order="dynasty")
            for h in cr.get("hits", []):
                if h["dynasty_rank"] < earliest["dynasty_rank"] and \
                        h["passage_id"] != earliest["passage_id"]:
                    counter.append({**h, "probe": probe,
                                    "match_kind": "partial"})
    out = {"tool": "classics_trace_citation", "available": True,
           "quote": quote, "mode": mode, "n_attestations": len(hits),
           "attestations_time_ordered": hits,
           "earliest_in_library": earliest,
           "counter_search": {"n_probes": 2 if counter or len(flat_q) >= 8 else 0,
                              "earlier_partial_candidates": counter[:8],
                              "note": "部分匹配候選需人工核驗（match_kind="
                                      "partial 不構成首現反證的定論）"},
           "scan_capped": r["scan_capped"],
           "retrieval_layers": r["retrieval_layers"],
           "honesty": "在庫首現≠歷史首現；首次明確表述/首次系統化須人工判定",
           "conclusion_policy": list(CONCLUSION_EVIDENCE_POLICY)}
    _attach_evidence(out, s, hits, quote)
    return out


def t_resolve_term(term: str, max_scan: int = 120) -> Dict:
    """術語解析：異體字折疊形、涉及的變體字符、全庫出現概況（按分類）。"""
    s = _searcher()
    if s is None:
        return _unavailable("classics_resolve_term")
    term = (term or "").strip()
    if len(term) < 2:
        return {"error": "術語至少 2 字"}
    folded = fold_variants(term)
    variant_chars = [{"raw": a, "folded": b}
                     for a, b in zip(term, folded) if a != b]
    r = s.search(query=term, limit=24, per_book=1, max_scan=max_scan)
    if "error" in r:
        return r
    by_cat = Counter(h["category"] for h in r["hits"])
    by_dyn = Counter(h["dynasty"] or "（無朝代）" for h in r["hits"])
    out = {"tool": "classics_resolve_term", "available": True,
           "term": term, "folded_form": folded,
           "variant_chars": variant_chars,
           "n_works_with_hits": len({h["work_id"] for h in r["hits"]}),
           "by_category": dict(by_cat.most_common()),
           "by_dynasty": dict(by_dyn.most_common()),
           "sample_hits": r["hits"][:6],
           "scan_capped": r["scan_capped"],
           "note": "通假字/古今詞/同義詞映射未實現（規劃層）；"
                   "此處僅做異體折疊與出現概況"}
    _attach_evidence(out, s, r["hits"][:6], term)
    return out


def t_concept_drift(term: str, category: str = "",
                    max_scan: int = 300) -> Dict:
    """概念漂移計量：術語出現按朝代分桶（著作數/命中段數/代表著作）。"""
    s = _searcher()
    if s is None:
        return _unavailable("classics_concept_drift")
    if len((term or "").strip()) < 2:
        return {"error": "術語至少 2 字"}
    r = s.search(query=term, category=category, limit=100, per_book=2,
                 max_scan=max_scan, order="dynasty")
    if "error" in r:
        return r
    buckets: Dict[str, Dict] = {}
    for h in r["hits"]:
        d = h["dynasty"] or "（無朝代）"
        b = buckets.setdefault(d, {"dynasty": d,
                                   "dynasty_rank": h["dynasty_rank"],
                                   "n_passages": 0, "n_occurrences": 0,
                                   "works": Counter()})
        b["n_passages"] += 1
        b["n_occurrences"] += h["n_occurrences"]
        b["works"][h["title"]] += h["n_occurrences"]
    series = sorted(buckets.values(), key=lambda b: b["dynasty_rank"])
    for b in series:
        b["n_works"] = len(b["works"])
        b["top_works"] = [w for w, _ in b["works"].most_common(3)]
        del b["works"]
    out = {"tool": "classics_concept_drift", "available": True,
           "term": term, "series_by_dynasty": series,
           "n_hits_total": r["n_hits"], "scan_capped": r["scan_capped"],
           "note": "計數基於命中段（per_book=2 封頂）+ scan_capped 如實標注"
                   "——是分佈證據，不是全庫窮舉普查",
           "honesty": "頻次漂移≠語義漂移；語義級漂移需注家釋義對齊（規劃層）"}
    _attach_evidence(out, s, r["hits"][:8], term)
    return out


def t_library_stats() -> Dict:
    """中醫笈成全庫統計（**非**傷寒論規則庫統計——那是 shanghan_corpus_stats）。"""
    s = _searcher()
    if s is None:
        return _unavailable("classics_library_stats")
    cat = s.lib.catalog
    dyn = Counter((u["dynasty"] or "（無朝代）")
                  for u in s.lib.units if not u["parent"])
    return {"tool": "classics_library_stats", "available": True,
            "semantic": "中醫笈成全庫（803 部級醫籍庫）書目統計，**非**傷寒論"
                        "規則庫統計——規則庫統計請用 shanghan_corpus_stats",
            "n_books": cat["n_books"], "n_units": cat["n_units"],
            "max_depth": cat.get("max_depth", 2),
            "categories": cat["categories"],
            "dynasties": dict(dyn.most_common()),
            "total_approx_chars": sum(u["approx_chars"] for u in s.lib.units),
            "archive_sha256": cat.get("archive_sha256", ""),
            "source_url": cat.get("source_url", "")}


def t_export_evidence_packet(passage_ids: List[str], topic: str = "") -> Dict:
    """證據包導出：按 passage_id 物化整段 P 層記錄並逐字重驗。"""
    s = _searcher()
    if s is None:
        return _unavailable("classics_export_evidence_packet")
    if not passage_ids:
        return {"error": "須提供 passage_ids"}
    records, missing = [], []
    for pid in list(dict.fromkeys(passage_ids))[:40]:
        p = s.index.get(pid)
        if p is None:
            missing.append(pid)
            continue
        unit = s.lib._by_id[p.work_id]
        records.append(passage_evidence(p, unit, 0, len(p.flat_text),
                                        retrieval_query=f"packet:{topic}"))
    packet = build_packet(records, s.index, topic=topic)
    out = {"tool": "classics_export_evidence_packet", "available": True,
           "packet": packet, "missing_passage_ids": missing,
           "passage_evidence": records,
           "evidence_layer": EVIDENCE_LAYER_NOTE}
    return out


# ---------------------------------------------------------------------------
# registration（由 ToolRegistry._register_all 調用）
# ---------------------------------------------------------------------------
def register_classics_tools(add) -> None:
    """``add(name, description, parameters, func)`` 註冊 classics 工具族。"""
    add("classics_search_passages",
        "全量古籍分層檢索（L0 元數據→L1 字符倒排→L2 逐字驗證，逐層可解釋）："
        "布爾 AND/OR/NOT、鄰近窗口、全量命中計數與字符座標；"
        "返回 P 層段級證據（verbatim+quote_hash 可重驗）。",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "AND 檢索項（空白分詞）"},
            "any_terms": {"type": "array", "items": {"type": "string"},
                          "description": "OR 項：至少命中其一"},
            "not_terms": {"type": "array", "items": {"type": "string"},
                          "description": "排除項"},
            "near": {"type": "integer", "default": 0,
                     "description": ">0 時前兩個 AND 項須在該字符窗口內共現"},
            "category": {"type": "string"}, "dynasty": {"type": "string"},
            "author": {"type": "string"}, "work": {"type": "string"},
            "limit": {"type": "integer", "default": 8},
            "per_book": {"type": "integer", "default": 3},
            "max_scan": {"type": "integer", "default": 200},
            "order": {"type": "string", "description": "relevance|dynasty"}},
         "required": []},
        t_search_passages)
    add("classics_read_passage",
        "按 passage_id（或 著作+章節）閱讀全庫某一段：整段扁平化正文 + "
        "P 層證據記錄（可重驗）。",
        {"type": "object", "properties": {
            "passage_id": {"type": "string"},
            "work": {"type": "string", "description": "著作名/單元 id"},
            "section": {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000}},
         "required": []},
        t_read_passage)
    add("classics_compare_witnesses",
        "同一著作不同傳本（Witness）對照：按書名歸組傳本清單；帶探針詞時"
        "返回各傳本命中段對照與兩兩相似度。",
        {"type": "object", "properties": {
            "work": {"type": "string"},
            "query": {"type": "string", "description": "可選探針詞句"},
            "limit": {"type": "integer", "default": 6}},
         "required": ["work"]},
        t_compare_witnesses)
    add("classics_trace_citation",
        "引文溯源：全庫時間有序檢索（按朝代排序的逐字命中）+ 反證搜索"
        "（截半探針找更早部分匹配候選）。誠實標注：在庫首現≠歷史首現。",
        {"type": "object", "properties": {
            "quote": {"type": "string", "description": "≥4 字引文"},
            "max_scan": {"type": "integer", "default": 300},
            "top": {"type": "integer", "default": 12}},
         "required": ["quote"]},
        t_trace_citation)
    add("classics_resolve_term",
        "術語解析：異體字折疊形、變體字符、全庫出現概況（分類/朝代分佈）。",
        {"type": "object", "properties": {
            "term": {"type": "string"},
            "max_scan": {"type": "integer", "default": 120}},
         "required": ["term"]},
        t_resolve_term)
    add("classics_concept_drift",
        "概念漂移計量：術語出現按朝代分桶（著作數/命中段數/代表著作），"
        "時間有序；掃描封頂如實標注。",
        {"type": "object", "properties": {
            "term": {"type": "string"},
            "category": {"type": "string"},
            "max_scan": {"type": "integer", "default": 300}},
         "required": ["term"]},
        t_concept_drift)
    add("classics_library_stats",
        "中醫笈成全庫統計：書目數/文本單元數/嵌套深度/分類/朝代分佈/指紋。"
        "（傷寒論規則庫統計請用 shanghan_corpus_stats——語義嚴格分離。）",
        {"type": "object", "properties": {}},
        t_library_stats)
    add("classics_export_evidence_packet",
        "P 層證據包導出：按 passage_id 物化整段記錄並逐字重驗"
        "（verbatim+座標+quote_hash），附庫指紋——供論文與人工複核。",
        {"type": "object", "properties": {
            "passage_ids": {"type": "array", "items": {"type": "string"}},
            "topic": {"type": "string"}},
         "required": ["passage_ids"]},
        t_export_evidence_packet)


CLASSICS_TOOL_NAMES = (
    "classics_search_passages", "classics_read_passage",
    "classics_compare_witnesses", "classics_trace_citation",
    "classics_resolve_term", "classics_concept_drift",
    "classics_library_stats", "classics_export_evidence_packet",
)
