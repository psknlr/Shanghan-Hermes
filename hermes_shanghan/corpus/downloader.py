"""ShanghanDownloaderAgent — corpus acquisition & version management.

Imports the raw corpus packages under ``data/corpus_raw`` (already vendored
in this repository from the user-provided 7z archives), identifies every
book, assigns Hermes evidence layers (A 宋本原文 / B 異文 / C 注釋 /
D 類方歸納), and writes a version manifest with sha256 checksums so every
downstream artifact can be traced back to an exact source file.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from ..textutil import sha256_text

RE_BOOK_META = re.compile(r"<book>(.*?)</book>", re.S)

# some archive extractors (p7zip/unzip under a C locale) mangle non-ASCII
# file names into #Uxxxx escapes; decode so a corpus extracted that way is
# still discoverable (無論解壓工具如何轉義，語料都能重建)
RE_UESC = re.compile(r"#U([0-9A-Fa-f]{4,5})")


def decode_u_escapes(name: str) -> str:
    return RE_UESC.sub(lambda m: chr(int(m.group(1), 16)), name)


def iter_book_dirs(corpus_root: Path):
    """Yield (category, book_dir, decoded_book_name) for every book directory,
    tolerating #Uxxxx-escaped path names at any level."""
    if not corpus_root.exists():
        return
    for category_dir in sorted(corpus_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for child in sorted(category_dir.iterdir()):
            if not (child.is_dir() and decode_u_escapes(child.name) == "書籍"):
                continue
            for book_dir in sorted(p for p in child.iterdir() if p.is_dir()):
                yield category_dir.name, book_dir, decode_u_escapes(book_dir.name)


def parse_book_meta(index_text: str) -> Dict[str, str]:
    m = RE_BOOK_META.search(index_text)
    meta: Dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_books(corpus_root: Path) -> List[Dict]:
    """Walk corpus_raw and return one manifest entry per book directory.

    ``book_dir`` in each entry is the *decoded* directory name so downstream
    lookups (LAYER_OF_BOOK, PRIMARY_BOOK …) work even when the corpus was
    extracted with #Uxxxx-mangled names; ``path`` keeps the on-disk location.
    """
    books: List[Dict] = []
    for category, book_dir, decoded in iter_book_dirs(corpus_root):
        index = book_dir / "index.txt"
        meta: Dict[str, str] = {}
        if index.exists():
            try:
                meta = parse_book_meta(index.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                meta = {}
        files = sorted(p.name for p in book_dir.glob("*.txt"))
        # 證據層不由目錄名默認決定（十輪 六.2）：未註冊書目 fail-closed
        # 到 P（旁證），推斷類型僅供編目，layer_basis 記錄依據
        from .worktype import classify
        work_type, layer, basis, inferred = classify(decoded, category, meta)
        try:
            rel_path = str(book_dir.relative_to(config.REPO_ROOT))
        except ValueError:
            rel_path = str(book_dir)
        books.append({
            "book_dir": decoded,
            "category": category,
            "title": meta.get("書名", decoded),
            "author": meta.get("作者", ""),
            "dynasty": meta.get("朝代", ""),
            "year": meta.get("年份", ""),
            "edition": meta.get("版本", ""),
            # 品質字段語義如實標注：笈成「品質」=校對程度（0%=已錄入未
            # 校對），不是文本質量評分；缺失記 None（unmeasured ≠ 0）
            "quality": meta.get("品質", "") or None,
            "quality_note": ("來源標注的校對程度（0%=未校對，非質量評分）"
                             if meta.get("品質") else
                             "來源未提供品質元數據（unmeasured，非 0 分）"),
            "work_type": work_type,
            "work_type_inferred": inferred,
            "hermes_layer": layer,
            "layer_basis": basis,
            "layer_label": config.LAYER_LABEL.get(layer, ""),
            "files": files,
            "file_sha256": {f: file_sha256(book_dir / f) for f in files},
            "path": rel_path,
        })
    return books


def reconcile_vendor_manifests(corpus_root: Path) -> Dict:
    """Compare the source archives' own book lists with what is vendored.

    The upstream 7z archives ship per-category manifest_*.json files listing
    every book they contained. Not all of those book directories were vendored
    into this repository, so the corpus manifest records the discrepancy
    explicitly instead of silently under-counting: which titles the vendor
    lists, which are on disk, and which are missing per category.
    """
    listed_total = 0
    missing: List[Dict] = []
    for vendor_file in sorted(corpus_root.glob("*/manifest_*.json")):
        category = vendor_file.parent.name
        try:
            entries = json.loads(vendor_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        on_disk = {decoded for cat, _, decoded in iter_book_dirs(corpus_root)
                   if cat == category}
        listed_total += len(entries)
        for e in entries:
            title = e.get("title", "")
            if title and not any(str(v) in on_disk for v in e.values()):
                missing.append({"category": category, "title": title})
    return {"vendor_listed_count": listed_total,
            "vendor_missing_count": len(missing),
            "vendor_missing_books": sorted(missing,
                                           key=lambda b: (b["category"], b["title"]))}


def run(corpus_root: Optional[Path] = None) -> Path:
    """Build and persist the corpus manifest. Returns the manifest path.

    Refuses to overwrite an existing manifest with an empty or load-bearing-
    incomplete discovery (a mis-pointed corpus path must fail loudly, never
    silently zero out a previously good manifest).
    """
    config.ensure_dirs()
    corpus_root = corpus_root or config.CORPUS_RAW_DIR
    books = discover_books(corpus_root)
    if not books:
        raise RuntimeError(
            f"語料發現為空：{corpus_root} 下未找到任何書籍目錄。"
            "請檢查語料路徑（HERMES_SHANGHAN_ROOT/HERMES_SHANGHAN_DATA）"
            "或解壓後的目錄編碼；已拒絕覆蓋現有 manifest。")
    found = {b["book_dir"] for b in books}
    required = {config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK}
    missing = required - found
    if missing:
        raise RuntimeError(
            f"語料缺少關鍵書目：{'、'.join(sorted(missing))}（僅發現 {len(books)} 部）。"
            "已拒絕覆蓋現有 manifest。")
    manifest = {
        "system": "Hermes-Shanghanlun",
        "primary_book": config.PRIMARY_BOOK,
        "songben_full_book": config.SONGBEN_FULL_BOOK,
        "variant_books": config.VARIANT_BOOKS,
        "commentary_books": config.COMMENTARY_BOOKS,
        "formula_family_books": config.FORMULA_FAMILY_BOOKS,
        "layer_legend": config.LAYER_LABEL,
        "book_count": len(books),
        **reconcile_vendor_manifests(corpus_root),
        "books": books,
    }
    out = config.MANIFEST_DIR / "corpus_manifest.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out)                      # atomic: never leave a torn manifest
    return out


def load_manifest() -> Dict:
    path = config.MANIFEST_DIR / "corpus_manifest.json"
    if not path.exists():
        run()
    return json.loads(path.read_text(encoding="utf-8"))


def book_path(book_dir_name: str) -> Optional[Path]:
    target = decode_u_escapes(book_dir_name)
    for _, book_dir, decoded in iter_book_dirs(config.CORPUS_RAW_DIR):
        if decoded == target or book_dir.name == book_dir_name:
            return book_dir
    return None


def read_book_text(book_dir_name: str) -> str:
    """Concatenate a book's text files in reading order (index, 1..n)."""
    path = book_path(book_dir_name)
    if path is None:
        raise FileNotFoundError(f"book not found in corpus: {book_dir_name}")
    parts: List[str] = []
    index = path / "index.txt"
    if index.exists():
        parts.append(index.read_text(encoding="utf-8", errors="replace"))
    # stems may be plain ("3") or volume-chapter ("2-15") — order numerically
    nums = sorted((tuple(int(x) for x in p.stem.split("-")), p)
                  for p in path.glob("*.txt")
                  if p.stem.replace("-", "").isdigit())
    for _, p in nums:
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)
