"""十六輪：條文 → 歷代古籍相關條目（段落級引文定位層）。

(著作×條文) 聚合邊只回答「哪部書引了」；本模塊補回段落級定位——給定
clause_id，返回歷代著作中引用該條的**具體段落**（引用模式/覆蓋率/章節/
逐字摘錄），按朝代分組，供條文全息（explain_clause）與注家解釋層使用。

段落級全量邊（4 萬+）體積大，不作為提交資產：首次調用時由掃描器
確定性重建（與 pipeline 同一代碼路徑，字節級可復現），並緩存於
``data/shanghan/trace/citation_edges_full.jsonl``（gitignored，可隨時
刪除重建）。緩存以 clauses.jsonl 的 mtime 做新鮮度守衛——語料重建後
自動重掃，不會用舊邊冒充新語料。
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from .. import config
from ..schemas import read_jsonl, write_jsonl
from ..textutil import fold_variants
from . import builder
from .ids import dynasty_order

FULL_EDGES_NAME = "citation_edges_full.jsonl"

# 落盤只保留展示/定位所需字段（全量字段可由掃描器隨時重建）
_PERSIST_KEYS = ("citation_edge_id", "target_kind", "clause_id", "mode",
                 "matched_span", "longest_run", "coverage", "book_dir",
                 "book", "author", "dynasty", "chapter", "para_seq",
                 "via_book", "via_commentator")

_LOCK = threading.Lock()
_EDGES: Optional[List[Dict]] = None
_BY_CLAUSE: Optional[Dict[str, List[Dict]]] = None
_PARA_CACHE: Dict[str, List[Tuple[str, str]]] = {}


def _full_edges_path():
    return builder.trace_dir() / FULL_EDGES_NAME


def _cache_fresh() -> bool:
    p = _full_edges_path()
    clauses = config.CLAUSE_DIR / "clauses.jsonl"
    return (p.exists() and clauses.exists()
            and p.stat().st_mtime >= clauses.stat().st_mtime)


def _rebuild() -> List[Dict]:
    from .quotation import scan_corpus
    commentary_rules = read_jsonl(config.RULES_COMMENTARY_DIR
                                  / "commentary_rules.jsonl")
    scan = scan_corpus(builder._clause_texts(), commentary_rules)
    rows = [{k: e[k] for k in _PERSIST_KEYS if k in e} for e in scan["edges"]]
    builder.trace_dir().mkdir(parents=True, exist_ok=True)
    write_jsonl(_full_edges_path(), rows)
    return rows


def load_full_edges() -> List[Dict]:
    """段落級引文邊（進程內緩存；磁盤緩存過期自動確定性重建）。"""
    global _EDGES, _BY_CLAUSE
    with _LOCK:
        if _EDGES is None:
            _EDGES = (read_jsonl(_full_edges_path()) if _cache_fresh()
                      else _rebuild())
            by: Dict[str, List[Dict]] = {}
            for e in _EDGES:
                by.setdefault(e.get("clause_id", ""), []).append(e)
            _BY_CLAUSE = by
        return _EDGES


def invalidate_cache() -> None:
    global _EDGES, _BY_CLAUSE
    with _LOCK:
        _EDGES = None
        _BY_CLAUSE = None
        _PARA_CACHE.clear()


def _paragraphs(book_dir: str) -> List[Tuple[str, str]]:
    if book_dir not in _PARA_CACHE:
        from ..corpus import segmenter
        try:
            _PARA_CACHE[book_dir] = segmenter.segment_paragraphs(book_dir)
        except FileNotFoundError:
            _PARA_CACHE[book_dir] = []
    return _PARA_CACHE[book_dir]


def _excerpt(edge: Dict, radius: int = 60) -> str:
    """引用段落摘錄：以命中片段為中心截取上下文；段落不可讀時退回
    matched_span（掃描時已存的逐字片段）。"""
    paras = _paragraphs(edge.get("book_dir", ""))
    seq = edge.get("para_seq", -1)
    if 0 <= seq < len(paras):
        text = paras[seq][1]
        span = edge.get("matched_span", "")
        pos = fold_variants(text).find(fold_variants(span)) if span else -1
        if pos < 0:
            return text[: radius * 2 + len(span)]
        lo = max(0, pos - radius)
        hi = min(len(text), pos + len(span) + radius)
        return ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else "")
    return edge.get("matched_span", "")


def book_citing_passages(book_dir: str, clause_ids: List[str],
                         offset: int = 0, limit: int = 8,
                         with_excerpt: bool = True) -> Dict:
    """某部書對一組條文的引用段落（方劑源流「歷代引用」的點閱視圖）。

    按覆蓋率降序分頁；每段標所引條文/模式/章節/逐字摘錄。"""
    load_full_edges()
    wanted = set(clause_ids or [])
    rows = [e for e in (_EDGES or [])
            if e.get("book_dir") == book_dir and e.get("clause_id") in wanted]
    rows.sort(key=lambda r: (-r.get("coverage", 0.0),
                             -r.get("longest_run", 0),
                             r.get("citation_edge_id", "")))
    total = len(rows)
    page = rows[max(0, offset):max(0, offset) + max(1, limit)]
    passages = []
    for e in page:
        p = {"clause_id": e.get("clause_id", ""),
             "mode": e.get("mode", ""), "chapter": e.get("chapter", ""),
             "coverage": e.get("coverage", 0.0),
             "matched_span": e.get("matched_span", "")}
        if e.get("mode") == "轉引注文":
            p["via_commentator"] = e.get("via_commentator", "")
            p["via_book"] = e.get("via_book", "")
        if with_excerpt:
            p["excerpt"] = _excerpt(e)
        passages.append(p)
    meta = next(({"book": e.get("book", book_dir),
                  "author": e.get("author", ""),
                  "dynasty": e.get("dynasty", "")} for e in rows),
                {"book": book_dir, "author": "", "dynasty": ""})
    return {"book_dir": book_dir, **meta,
            "n_passages": total, "offset": offset,
            "has_more": offset + len(page) < total,
            "passages": passages,
            "evidence_layer": "跨書引文邊（逐字回源）"}


def clause_citing_passages(clause_id: str, per_book: int = 2,
                           max_books: int = 30,
                           with_excerpt: bool = True) -> Dict:
    """某條文在歷代（隨庫 57 部）著作中的引用段落，按朝代→著作分組。

    每部書取覆蓋率最高的 ``per_book`` 段；摘錄為引用段落的逐字上下文。
    全庫（笈成 803 部）擴展屬旁證層，走 ``trace-scan-library``，不在此列。
    """
    load_full_edges()
    edges = (_BY_CLAUSE or {}).get(clause_id, [])
    if not edges:
        return {"clause_id": clause_id, "n_edges": 0, "n_books": 0,
                "by_dynasty": [],
                "note": "隨庫 57 部後世著作中未檢出對本條的引用段落。"}

    by_book: Dict[str, List[Dict]] = {}
    for e in edges:
        by_book.setdefault(e["book_dir"], []).append(e)

    books = []
    for bdir, rows in by_book.items():
        rows.sort(key=lambda r: (-r.get("coverage", 0.0),
                                 -r.get("longest_run", 0),
                                 r.get("citation_edge_id", "")))
        top = rows[:per_book]
        passages = []
        for e in top:
            p = {"mode": e.get("mode", ""), "chapter": e.get("chapter", ""),
                 "coverage": e.get("coverage", 0.0),
                 "matched_span": e.get("matched_span", "")}
            if e.get("mode") == "轉引注文":
                p["via_commentator"] = e.get("via_commentator", "")
                p["via_book"] = e.get("via_book", "")
            if with_excerpt:
                p["excerpt"] = _excerpt(e)
            passages.append(p)
        books.append({"book_dir": bdir, "book": rows[0].get("book", bdir),
                      "author": rows[0].get("author", ""),
                      "dynasty": rows[0].get("dynasty", ""),
                      "n_citing_paragraphs": len(rows),
                      "passages": passages})
    # 排序：朝代序 → 引用段數多者先；每朝代內書名穩定序
    books.sort(key=lambda b: (dynasty_order(b["dynasty"]),
                              -b["n_citing_paragraphs"], b["book"]))
    books = books[:max_books]

    by_dyn: Dict[str, Dict] = {}
    for b in books:
        dyn = b["dynasty"] or "未詳"
        s = by_dyn.setdefault(dyn, {"dynasty": dyn,
                                    "dynasty_order": dynasty_order(dyn),
                                    "books": []})
        s["books"].append(b)
    out = sorted(by_dyn.values(), key=lambda s: (s["dynasty_order"],
                                                 s["dynasty"]))
    for s in out:
        s.pop("dynasty_order", None)
    return {
        "clause_id": clause_id,
        "n_edges": len(edges),
        "n_books": len(by_book),
        "n_books_shown": len(books),
        "by_dynasty": out,
        "evidence_layer": "跨書引文邊（逐字回源，C/引文層；非原文直述）",
        "note": "隨庫 57 部傷寒/金匱著作的段落級引用；全庫（803 部）擴展"
                "見 trace-scan-library（旁證層，不入庫）。",
    }
