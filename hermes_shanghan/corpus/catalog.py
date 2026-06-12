"""ShanghanCatalogAgent — book/version/chapter identification.

Parses the wiki-format used by the corpus:
``======書名======`` (book), ``=====卷/篇=====`` (level-1 section),
``====篇名====`` (level-2 section), paragraphs separated by blank lines.

Produces a chapter catalog and the chapter→six-channel mapping used by the
segmenter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import config
from . import downloader

RE_HEADING = re.compile(r"^(={4,6})([^=]+)\1\s*$")


@dataclass
class Section:
    title: str
    level: int
    paragraphs: List[str] = field(default_factory=list)


def parse_sections(text: str) -> List[Section]:
    """Split a book text into sections with their paragraph lists."""
    sections: List[Section] = []
    current = Section(title="(前言)", level=0)
    in_meta = False
    buf: List[str] = []

    def flush_par():
        nonlocal buf
        if buf:
            current.paragraphs.append("\n".join(buf).strip())
            buf = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "<book>":
            in_meta = True
            continue
        if stripped == "</book>":
            in_meta = False
            continue
        if in_meta:
            continue
        m = RE_HEADING.match(stripped)
        if m:
            flush_par()
            if current.paragraphs or current.level:
                sections.append(current)
            level = len(m.group(1))
            current = Section(title=m.group(2).strip(), level=level)
            continue
        if not stripped:
            flush_par()
            continue
        buf.append(line.rstrip())
    flush_par()
    if current.paragraphs or current.level:
        sections.append(current)
    return sections


def chapter_channel(chapter: str) -> str:
    """Map a chapter title to a six-channel tag (empty if not applicable)."""
    for key, channel in config.CHAPTER_TO_CHANNEL.items():
        if key in chapter:
            return channel
    # fuzzy: 辨XX病脈證并治 style in other versions
    m = re.search(r"辨?(太陽|陽明|少陽|太陰|少陰|厥陰|霍亂)病?", chapter)
    if m:
        name = m.group(1)
        return "霍亂病" if name == "霍亂" else name + "病"
    if "陰陽易" in chapter or "勞復" in chapter:
        return "陰陽易差後勞復病"
    return ""


def catalog_book(book_dir_name: str) -> Dict:
    """Return {title, meta, chapters:[{title, level, n_paragraphs, six_channel}]}"""
    text = downloader.read_book_text(book_dir_name)
    meta = downloader.parse_book_meta(text)
    sections = parse_sections(text)
    chapters = [{
        "title": s.title,
        "level": s.level,
        "n_paragraphs": len(s.paragraphs),
        "six_channel": chapter_channel(s.title),
    } for s in sections]
    return {"book_dir": book_dir_name, "title": meta.get("書名", book_dir_name),
            "meta": meta, "chapters": chapters}
