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
    """Walk corpus_raw and return one manifest entry per book directory."""
    books: List[Dict] = []
    for shuji in sorted(corpus_root.glob("*/書籍")):
        category = shuji.parent.name
        for book_dir in sorted(p for p in shuji.iterdir() if p.is_dir()):
            index = book_dir / "index.txt"
            meta: Dict[str, str] = {}
            if index.exists():
                try:
                    meta = parse_book_meta(index.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    meta = {}
            files = sorted(p.name for p in book_dir.glob("*.txt"))
            layer = config.LAYER_OF_BOOK.get(book_dir.name, "C" if category == "shanghan" else "D")
            books.append({
                "book_dir": book_dir.name,
                "category": category,
                "title": meta.get("書名", book_dir.name),
                "author": meta.get("作者", ""),
                "dynasty": meta.get("朝代", ""),
                "year": meta.get("年份", ""),
                "edition": meta.get("版本", ""),
                "quality": meta.get("品質", ""),
                "hermes_layer": layer,
                "layer_label": config.LAYER_LABEL.get(layer, ""),
                "files": files,
                "file_sha256": {f: file_sha256(book_dir / f) for f in files},
                "path": str(book_dir.relative_to(config.REPO_ROOT)),
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
        on_disk = {p.name for p in (vendor_file.parent / "書籍").glob("*")
                   if p.is_dir()}
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
    """Build and persist the corpus manifest. Returns the manifest path."""
    config.ensure_dirs()
    corpus_root = corpus_root or config.CORPUS_RAW_DIR
    books = discover_books(corpus_root)
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
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_manifest() -> Dict:
    path = config.MANIFEST_DIR / "corpus_manifest.json"
    if not path.exists():
        run()
    return json.loads(path.read_text(encoding="utf-8"))


def book_path(book_dir_name: str) -> Optional[Path]:
    for shuji in config.CORPUS_RAW_DIR.glob("*/書籍"):
        cand = shuji / book_dir_name
        if cand.exists():
            return cand
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
