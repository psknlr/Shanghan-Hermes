"""EvidenceRecord —— 逐證據來源對象（十輪評審 六.1）。

把「文獻層級」細化到每條證據：來源/著作/版本指紋/篇章/段落/逐字文本/
引文哈希/朝代作者/證據層/解析器版本/質量/檢索上下文——使系統能回答：

  該引文來自哪一版（edition_fingerprint = 該書全部文件 sha256 的聚合）；
  原文還是後人轉述（provenance_layer + work_type）；
  是否發生刪改（quote_hash 逐字對比；版本敏感處見 variant 規則）；
  「最早」是歷史最早還是在庫首現（本系統一律「在庫首現」口徑）；
  結論能否被其他版本復核（variant_books 指針）。

**誠實邊界**（缺什麼記 null，不編造）：
- ``volume_id`` / ``char_start`` / ``char_end``：條文切分管道未保留卷號與
  源文件字符偏移，記 null；聚合引文邊只保留段落級定位（chapter/段數）。
- ``quality_score``：來源品質元數據多數書目為空——記 null + unmeasured，
  絕不寫 0（0 分是測過而不及格，與沒測過是兩回事）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from .. import config

EVIDENCE_RECORD_FIELDS = (
    "source_id", "work_id", "work_type", "edition_id", "edition_fingerprint",
    "volume_id", "chapter_id", "passage_id", "char_start", "char_end",
    "verbatim_text", "normalized_text", "quote_hash", "dynasty", "author",
    "edition", "provenance_layer", "parser_version", "quality_score",
    "quality_note", "retrieval_query", "retrieval_rank")

_MANIFEST_CACHE: Optional[Dict] = None


def _manifest_book(book_dir: str) -> Dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        p = config.MANIFEST_DIR / "corpus_manifest.json"
        _MANIFEST_CACHE = json.loads(p.read_text(encoding="utf-8")) \
            if p.exists() else {"books": []}
    return next((b for b in _MANIFEST_CACHE.get("books", [])
                 if b.get("book_dir") == book_dir), {})


def _edition_fingerprint(book: Dict) -> str:
    """版本指紋：該書全部源文件 sha256 的穩定聚合（換一版即變）。"""
    hashes = book.get("file_sha256") or {}
    if not hashes:
        return ""
    blob = ";".join(f"{k}={v}" for k, v in sorted(hashes.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def quote_hash(text: str) -> str:
    return hashlib.sha256("".join((text or "").split()).encode()) \
        .hexdigest()[:16]


def _parser_version() -> str:
    try:
        from ..corpus.source_registry import PARSER_VERSION
        return PARSER_VERSION
    except Exception:
        return ""


def evidence_record(clause, retrieval_query: Optional[str] = None,
                    retrieval_rank: Optional[int] = None) -> Dict:
    """為一條宋本條文構造逐證據來源記錄。``clause`` 為 ShanghanClause。"""
    work = (config.SONGBEN_FULL_BOOK if "AUX" in clause.clause_id
            else config.PRIMARY_BOOK)
    book = _manifest_book(work)
    quality = book.get("quality")
    return {
        "source_id": "corpus_raw_shanghan",
        "work_id": work,
        "work_type": book.get("work_type", "canonical_text"),
        "edition_id": book.get("edition") or None,
        "edition_fingerprint": _edition_fingerprint(book),
        "volume_id": None,               # 切分管道未保留卷號（如實）
        "chapter_id": clause.chapter,
        "passage_id": clause.clause_id,
        "char_start": None,              # 未保留源文件字符偏移（如實）
        "char_end": None,
        "verbatim_text": clause.raw_text or clause.clean_text,
        "normalized_text": clause.clean_text,
        "quote_hash": quote_hash(clause.clean_text),
        "dynasty": book.get("dynasty") or "東漢",
        "author": book.get("author") or "張仲景",
        "edition": book.get("edition", ""),
        "provenance_layer": clause.layer,
        "parser_version": _parser_version(),
        "quality_score": quality if quality not in ("", None) else None,
        "quality_note": book.get("quality_note",
                                 "來源未提供品質元數據（unmeasured）"),
        "retrieval_query": retrieval_query,
        "retrieval_rank": retrieval_rank,
        "cross_check": {
            "variant_books": list(config.VARIANT_BOOKS),
            "note": "版本復核走 shanghan_variants（B 層對勘）；"
                    "「最早」一律為在庫首現口徑",
        },
    }


def evidence_record_for_edge(edge: Dict) -> Dict:
    """為一條聚合引文邊構造來源記錄（引用方著作側）。

    聚合邊只保留段落級定位（首見篇章/段數/模式/覆蓋率），字符偏移與
    逐字全文不在聚合層——記 null，如實。coverage/longest_run 作為
    文本質量信號進 quality_score。"""
    return {
        "source_id": "jicheng_20180111" if edge.get("from_library")
                     else "corpus_raw_shanghan",
        "work_id": edge.get("book_dir") or edge.get("book", ""),
        "work_type": edge.get("work_type", "unclassified"),
        "edition_id": None,
        "edition_fingerprint": _edition_fingerprint(
            _manifest_book(edge.get("book_dir", ""))),
        "volume_id": None,
        "chapter_id": edge.get("first_chapter", ""),
        "passage_id": f"{edge.get('book', '')}→{edge.get('clause_id', '')}",
        "char_start": None,
        "char_end": None,
        "verbatim_text": None,           # 聚合層不存逐字（重掃可得）
        "normalized_text": None,
        "quote_hash": None,
        "dynasty": edge.get("dynasty", ""),
        "author": edge.get("author", ""),
        "edition": "",
        "provenance_layer": edge.get("layer", "P"),
        "parser_version": _parser_version(),
        "quality_score": edge.get("max_coverage"),
        "quality_note": "quality_score=引文逐字覆蓋率（0-1，確定性）；"
                        "引用方式見 modes",
        "retrieval_query": None,
        "retrieval_rank": None,
        "citation_modes": edge.get("modes", {}),
    }


def records_for_hits(hits: List[Dict], store: Dict, query: str) -> List[Dict]:
    """為檢索命中批量構造記錄（帶 retrieval_query/rank）。"""
    out = []
    for rank, h in enumerate(hits, 1):
        c = store.get(h.get("clause_id"))
        if c is not None:
            out.append(evidence_record(c, retrieval_query=query,
                                       retrieval_rank=rank))
    return out
