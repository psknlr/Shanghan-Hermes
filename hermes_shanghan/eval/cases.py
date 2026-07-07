"""醫案回放基準 (Historical Case-Replay Benchmark) — 經方實驗錄.

曹穎甫《經方實驗錄》(1937) records ~100 real consultations whose section
titles name the formula actually prescribed (「第一案 桂枝湯證 其一」).
Each case is replayed: the patient presentation is extracted from the
narrative BEFORE the first formula mention (so neither the prescription nor
the master's discussion can leak into the query), the matcher predicts, and
the master's own prescription is the gold label.

Unlike 2025-era TCM benchmarks built from modern hospital records or exam
items, the ground truth here is a century-old master's real clinical
decision on the very formula system this codebase models — a direct
external-validity test of rules mined from the 傷寒論 itself.

Honesty accounting (no silent caps): cases whose gold formula is outside
the mined rule inventory (金匱方 such as 皂莢丸) are reported as
out-of-scope; cases whose narrative yields fewer than `min_findings`
extractable findings are reported as insufficient — both counted, neither
scored nor hidden.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import lexicon
from ..apps.doctor import FormulaMatcher
from ..corpus import downloader
from ..extract.entities import EntityExtractor
from ..schemas import FormulaPatternRule, ShanghanClause

CASE_BOOK = "經方實驗錄"
RE_SECTION = re.compile(r"^=====(.+?)=====\s*$", re.M)
RE_CASE_TITLE = re.compile(r"第[〇○一二三四五六七八九十百\d]+案[\s　]*(.+)")
RE_ANNOT = re.compile(r"<z>.*?</z>", re.S)   # attribution notes incl. content
RE_MARKUP = re.compile(r"<[^>]+>")           # any remaining bare tags
RE_FORMULA_LIKE = re.compile(r"[湯丸散飲導方]$")
TOP_KS = (1, 3, 5)


def _clean_title(raw: str) -> str:
    return RE_MARKUP.sub("", RE_ANNOT.sub("", raw)).strip()


def _gold_formula(title: str) -> Optional[str]:
    """「桂枝湯證 其一」→ 桂枝湯 (canonicalized).

    Returns None for non-formula case titles (病名案 such as 奔豚/肺癰) —
    those are counted separately, not scored.
    """
    m = RE_CASE_TITLE.search(title)
    if not m:
        return None
    body = m.group(1)
    body = re.split(r"[\s　]", body)[0]
    body = re.sub(r"(證|证)$", "", body)
    if not body:
        return None
    name = lexicon.canonical_formula(body)
    if not RE_FORMULA_LIKE.search(name) and name not in lexicon.FORMULA_SEEDS:
        return None
    return name


def parse_cases(extractor: EntityExtractor) -> List[Dict]:
    """Split the book into cases; extract presentation entities per case."""
    text = downloader.read_book_text(CASE_BOOK)
    parts = RE_SECTION.split(text)
    cases: List[Dict] = []
    non_formula = 0
    # parts = [preamble, title1, body1, title2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        title = _clean_title(parts[i])
        gold = _gold_formula(title)
        if not gold:
            if RE_CASE_TITLE.search(title):
                non_formula += 1   # 病名案（奔豚/肺癰…），計數不評分
            continue               # else 序/跋 etc.
        body = parts[i + 1]
        presentation = _presentation_segment(body, extractor)
        res = extractor.extract(presentation)
        cases.append({
            "title": title, "gold": gold,
            "presentation_chars": len(presentation),
            "symptoms": res.symptoms, "pulse": res.pulse,
        })
    return cases, non_formula


RE_HERB_DOSE_LINE = re.compile(
    r"^[^\n。]{0,60}?[一-鿿]{1,6}[一二三四五六七八九十錢兩分半]{1,4}"
    r"[錢兩分片枚匙]\S{0,8}[一-鿿]{1,6}[一二三四五六七八九十錢兩分半]{1,4}"
    r"[錢兩分片枚匙]", re.M)
RE_RX_MARKER = re.compile(r"(隨疏方|疏方|擬方|方用|處方|用：)")


def _presentation_segment(body: str, extractor: EntityExtractor) -> str:
    """Narrative up to the first prescription — the herb-dose line, an
    explicit 疏方/擬方 marker, or a formula-name mention, whichever comes
    first. Everything after is the master's answer and rationale (often
    quoting the gold formula's hallmark clause) and must not leak into
    the query."""
    body = RE_MARKUP.sub("", body)
    cuts = [len(body)]
    mentions = extractor.extract_formula_mentions(body)
    if mentions:
        cuts.append(mentions[0]["start"])
    m = RE_HERB_DOSE_LINE.search(body)
    if m:
        cuts.append(m.start())
    m = RE_RX_MARKER.search(body)
    if m:
        cuts.append(m.start())
    return body[:min(cuts)]


class CaseBenchmark:
    def __init__(self, formula_rules: List[FormulaPatternRule],
                 clause_store: Dict[str, ShanghanClause],
                 extractor: Optional[EntityExtractor] = None,
                 use_outline_boost: bool = True,
                 use_near_match: bool = True):
        self.extractor = extractor or EntityExtractor()
        self.matcher = FormulaMatcher(formula_rules, clause_store,
                                      use_outline_boost=use_outline_boost,
                                      use_near_match=use_near_match)
        self.known = {r.formula for r in formula_rules}
        self.config = {"use_outline_boost": use_outline_boost,
                       "use_near_match": use_near_match}

    def run(self, limit: Optional[int] = None, min_findings: int = 2) -> Dict:
        cases, non_formula = parse_cases(self.extractor)
        if limit:
            cases = cases[:limit]
        scored: List[Dict] = []
        out_of_scope, insufficient = [], []
        for c in cases:
            if c["gold"] not in self.known:
                out_of_scope.append(c["gold"])
                continue
            if len(c["symptoms"]) + len(c["pulse"]) < min_findings:
                insufficient.append(c["title"])
                continue
            out = self.matcher.match(c["symptoms"], pulse=c["pulse"],
                                     top_k=max(TOP_KS),
                                     need_original_evidence=False)
            matches = out["matched_formula_patterns"]
            rank = next((i for i, m in enumerate(matches, 1)
                         if m["formula"] == c["gold"]), None)
            scored.append({"title": c["title"], "gold": c["gold"],
                           "n_findings": len(c["symptoms"]) + len(c["pulse"]),
                           "rank": rank,
                           "top1": matches[0]["formula"] if matches else ""})

        n = len(scored)
        metrics: Dict = {"n_scored": n}
        for k in TOP_KS:
            metrics[f"top{k}"] = round(sum(1 for r in scored
                                           if r["rank"] and r["rank"] <= k) / n, 4) if n else 0.0
        metrics["mrr"] = round(sum(1.0 / r["rank"] for r in scored
                                   if r["rank"]) / n, 4) if n else 0.0
        return {"benchmark": "case_replay_jingfang_shiyanlu",
                "source": CASE_BOOK,
                "config": self.config,
                "n_cases_parsed": len(cases),
                "n_non_formula_titles": non_formula,
                "n_out_of_scope": len(out_of_scope),
                "out_of_scope_formulas": sorted(set(out_of_scope)),
                "n_insufficient_findings": len(insufficient),
                "metrics": metrics,
                "records": scored}
