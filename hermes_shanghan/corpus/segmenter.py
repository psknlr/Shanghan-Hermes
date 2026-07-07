"""ClauseSegmenterAgent — clause-level segmentation with stable clause_id.

Canonical layer (A):
  * 傷寒論_條文版 — the Songben text with the standard modern numbering.
    Each ``<#/>`` line is one numbered clause (1–398). ``<F>…</F>`` blocks
    carry formula composition, preparation, administration and 方後注 and are
    attached to the preceding clause.
  * 傷寒論_宋本 — auxiliary chapters not covered by the numbering
    (辨脈法/平脈法/傷寒例/辨痙濕暍/可與不可諸篇), segmented as auxiliary
    clauses. These power therapy & contraindication rules.

Logic words (若/不可/反/誤/或/勿/必/但/急/慎) are tagged for downstream
rule extraction.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .. import config
from ..schemas import FormulaBlock, ShanghanClause
from ..textutil import extract_j_notes, sha256_text, strip_markup
from ..lexicon import canonical_formula
from . import catalog, downloader

RE_CLAUSE_MARK = re.compile(r"^<#[^>]*/>\s*(.*)$")
RE_HEADING = re.compile(r"^(={4,6})([^=]+)\1\s*$")
RE_F_NAME = re.compile(r"\*\*(.+?)\*\*")
RE_HERB_TOKEN = re.compile(r"([^　<>\s，。]+)(?:<l>(.*?)</l>)?")
RE_PREP_START = re.compile(r"^[右上][一二三四五六七八九十]{1,3}味")
LOGIC_WORDS = ["不可", "可與", "勿", "若", "反", "誤", "或", "必", "但", "急", "慎", "莫", "得之", "屬", "宜", "主之"]

CANONICAL_CHAPTER_KEYS = list(config.CHAPTER_TO_CHANNEL.keys())


# ---------------------------------------------------------------------------
# Formula block parsing
# ---------------------------------------------------------------------------
def parse_formula_block(block_text: str) -> FormulaBlock:
    fb = FormulaBlock(raw_text=block_text.strip())
    name = ""
    comp: List[Dict[str, str]] = []
    prep_lines: List[str] = []
    post: List[str] = []
    for raw_line in block_text.splitlines():
        line = raw_line.strip()
        if not line or line in ("<F>", "</F>"):
            continue
        m = RE_F_NAME.search(line)
        if m and not name:
            name = m.group(1).strip()
            continue
        if RE_PREP_START.match(strip_markup(line)):
            prep_lines.append(strip_markup(line, keep_notes=True))
            continue
        if "<l>" in line or "　" in line:
            # composition line: herb tokens separated by full-width space
            for tok in line.split("　"):
                tok = tok.strip()
                if not tok:
                    continue
                hm = re.match(r"^([^<\s]+)(?:<l>(.*?)</l>)?$", tok)
                if hm:
                    herb = strip_markup(hm.group(1)).strip("，。、")
                    if herb:
                        comp.append({"herb": herb, "dose_processing": (hm.group(2) or "").strip()})
            continue
        if line.startswith("本云") or line.startswith("一方") or line.startswith("臣億") or line.startswith("疑非"):
            post.append(strip_markup(line, keep_notes=True))
        elif prep_lines:
            prep_lines.append(strip_markup(line, keep_notes=True))
        else:
            post.append(strip_markup(line, keep_notes=True))
    # split preparation (煎法) from administration (服法) at the first 服 sentence
    prep_text = "".join(prep_lines)
    fb.formula_name = canonical_formula(re.sub(r"方$", "", name)) if name else ""
    fb.composition = comp
    if prep_text:
        m = re.search(r"(溫服|頓服|分溫|服一升|和服|飲服|白飲和服|每服)", prep_text)
        if m:
            fb.preparation = prep_text[:m.start()].strip()
            fb.administration = prep_text[m.start():].strip()
        else:
            fb.preparation = prep_text.strip()
    fb.post_notes = post
    return fb


# ---------------------------------------------------------------------------
# Canonical numbered edition (條文版)
# ---------------------------------------------------------------------------
def segment_canonical() -> List[ShanghanClause]:
    text = downloader.read_book_text(config.PRIMARY_BOOK)
    clauses: List[ShanghanClause] = []
    chapter = ""
    number = 0
    in_meta = False
    in_f = False
    f_buf: List[str] = []
    current: Optional[ShanghanClause] = None

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
        if stripped == "<F>":
            in_f = True
            f_buf = []
            continue
        if stripped == "</F>":
            in_f = False
            if current is not None:
                fb = parse_formula_block("\n".join(f_buf))
                current.formula_blocks.append(fb)
                if fb.formula_name and fb.formula_name not in current.formula_names:
                    current.formula_names.append(fb.formula_name)
                current.contains_formula = True
            continue
        if in_f:
            f_buf.append(line)
            continue
        hm = RE_HEADING.match(stripped)
        if hm:
            title = hm.group(2).strip()
            if catalog.chapter_channel(title):
                chapter = title
            continue
        cm = RE_CLAUSE_MARK.match(stripped)
        if cm:
            number += 1
            raw = cm.group(1).strip()
            clean = strip_markup(raw)
            current = ShanghanClause(
                clause_id=f"{config.ID_PREFIX_CLAUSE}{number:04d}",
                book_title="傷寒論（宋本）",
                version="songben",
                chapter=chapter,
                six_channel=catalog.chapter_channel(chapter),
                clause_number=number,
                raw_text=raw,
                clean_text=clean,
                text_type="original_clause",
                layer="A",
                collation_notes=extract_j_notes(raw),
                logic_words=[w for w in LOGIC_WORDS if w in clean],
                sha256=sha256_text(clean),
            )
            clauses.append(current)
            continue
        # stray content line: append to current clause as continuation
        if current is not None and stripped:
            current.raw_text += "\n" + stripped
            current.clean_text = strip_markup(current.raw_text)
            current.sha256 = sha256_text(current.clean_text)
    return clauses


# ---------------------------------------------------------------------------
# Songben auxiliary chapters
# ---------------------------------------------------------------------------
def is_canonical_chapter(title: str) -> bool:
    return any(key in title for key in CANONICAL_CHAPTER_KEYS)


def segment_auxiliary() -> List[ShanghanClause]:
    text = downloader.read_book_text(config.SONGBEN_FULL_BOOK)
    sections = catalog.parse_sections(text)
    clauses: List[ShanghanClause] = []
    idx = 0
    for sec in sections:
        if sec.level != 4:           # chapters are ====…==== inside 卷
            continue
        if is_canonical_chapter(sec.title):
            continue                 # covered by the numbered edition
        for para in sec.paragraphs:
            for line in (l.strip() for l in para.splitlines()):
                if not line:
                    continue
                clean = strip_markup(line)
                if len(clean) < 6:
                    continue
                idx += 1
                clauses.append(ShanghanClause(
                    clause_id=f"{config.ID_PREFIX_AUX}{idx:04d}",
                    book_title="傷寒論（宋本）",
                    version="songben",
                    chapter=sec.title,
                    six_channel=catalog.chapter_channel(sec.title),
                    clause_number=0,
                    raw_text=line,
                    clean_text=clean,
                    text_type="auxiliary_clause",
                    layer="A",
                    collation_notes=extract_j_notes(line),
                    logic_words=[w for w in LOGIC_WORDS if w in clean],
                    sha256=sha256_text(clean),
                ))
    return clauses


# ---------------------------------------------------------------------------
# Generic paragraph segmentation for variant / commentary books
# ---------------------------------------------------------------------------
def segment_paragraphs(book_dir_name: str) -> List[Tuple[str, str]]:
    """Return [(chapter_title, clean_paragraph)] for any corpus book."""
    text = downloader.read_book_text(book_dir_name)
    out: List[Tuple[str, str]] = []
    for sec in catalog.parse_sections(text):
        for para in sec.paragraphs:
            clean = strip_markup(para)
            if len(clean) >= 6:
                out.append((sec.title, clean))
    return out


def harvest_formula_names(clauses: List[ShanghanClause]) -> List[str]:
    names = []
    for c in clauses:
        for fb in c.formula_blocks:
            if fb.formula_name:
                names.append(fb.formula_name)
    return sorted(set(names), key=lambda t: (-len(t), t))
