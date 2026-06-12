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
    nums = sorted((int(p.stem), p) for p in path.glob("*.txt")
                  if p.stem.isdigit())
    for _, p in nums:
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)
