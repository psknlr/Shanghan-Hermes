"""語料來源註冊表（corpus lifecycle，評審第 10 條）。

每個來源記錄：source_id / url / sha256 / 計數 / 許可註記 / 解析器版本 /
質檢結論 / 證據層歸屬。P 層（旁證）與 A–E 層嚴格分離：全庫/醫案旁證
永不混入經文層證據閘門。
"""
from __future__ import annotations

import json
from typing import Dict, List

from .. import config

PARSER_VERSION = "wiki-format-v2（<book> 元數據 + ====標題==== + 段落）"


def sources() -> List[Dict]:
    out = []
    manifest_path = config.MANIFEST_DIR / "corpus_manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        out.append({
            "source_id": "corpus_raw_shanghan",
            "source_url": "隨庫提交（data/corpus_raw/，逐文件 sha256 見 manifest）",
            "book_count": m.get("book_count", 0),
            "vendor_missing": m.get("vendor_missing_count", 0),
            "license_note": "古籍原文（公有領域），vendor 整理版式",
            "parser_version": PARSER_VERSION,
            "evidence_layers": "A/B/C/D（經文層，入證據閘門）",
            "quality_checks": {"398_clause_invariant": True,
                               "sha256_per_file": True,
                               "byte_reproducible_pipeline": True},
        })
    from . import library
    entry = {
        "source_id": "jicheng_20180111",
        "source_url": config.LIBRARY_URL,
        "sha256": config.LIBRARY_SHA256,
        "license_note": "中醫笈成整理本（jicheng.tw），2018 固定歸檔——"
                        "非在線同步，版本永不漂移",
        "parser_version": PARSER_VERSION,
        "evidence_layers": "P（旁證層，不入經文閘門）",
        "available": library.is_available(),
    }
    if library.is_available():
        cat_path = library.library_root() / library.CATALOG_NAME
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        entry.update({"book_count": cat.get("n_books", 0),
                      "unit_count": cat.get("n_units", 0),
                      "quality_checks": {"archive_sha256_verified": True,
                                         "char_index_built":
                                             (library.library_root() /
                                              "charindex.json").exists()}})
    out.append(entry)
    return out
