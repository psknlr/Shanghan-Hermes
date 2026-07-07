"""EntityExtractorAgent — negation-aware entity extraction.

Extracts: six channels, disease patterns, symptoms (incl. negated canonical
forms), pulse images, formulas, herbs, doses, administration, therapy terms,
contraindications, mistreatment markers, transformations, prognosis and
time-course expressions from a clause.

Matching is longest-first and non-overlapping so 「不惡寒」 wins over 「惡寒」
and 「頭項強痛」 wins over 「頭痛」+「項強」. Any positive lexicon term whose
match is immediately preceded by an uncovered negation character (不/無/未/
非/勿/莫) is recorded as a negated finding, never as the positive symptom.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .. import lexicon
from ..schemas import ShanghanClause
from ..textutil import fold_variants


@dataclass
class Match:
    term: str
    start: int
    end: int
    negated: bool = False
    contrary: bool = False   # 反X (unexpected finding)


def _match_terms(text: str, terms: Sequence[str], taken: Optional[List[bool]] = None) -> List[Match]:
    """Longest-first non-overlapping matching over `text`.

    `taken` is a shared occupancy mask so different entity passes don't
    claim the same span twice (e.g. formula names vs disease patterns).
    """
    if taken is None:
        taken = [False] * len(text)
    matches: List[Match] = []
    for term in terms:           # caller must pass longest-first ordering
        if not term:
            continue
        start = 0
        while True:
            i = text.find(term, start)
            if i < 0:
                break
            j = i + len(term)
            if not any(taken[i:j]):
                neg = False
                contrary = False
                if i > 0 and text[i - 1] in lexicon.NEGATION_PREFIX and not taken[i - 1]:
                    neg = True
                if i > 0 and text[i - 1] in lexicon.CONTRARY_PREFIX and not taken[i - 1]:
                    contrary = True
                for k in range(i, j):
                    taken[k] = True
                if neg:
                    taken[i - 1] = True
                matches.append(Match(term=term, start=i, end=j, negated=neg, contrary=contrary))
            start = i + 1
    matches.sort(key=lambda m: m.start)
    return matches


@dataclass
class EntityResult:
    symptoms: List[str] = field(default_factory=list)
    negated_findings: List[str] = field(default_factory=list)
    pulse: List[str] = field(default_factory=list)
    disease_patterns: List[str] = field(default_factory=list)
    six_channels: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    contraindicated_formulas: List[str] = field(default_factory=list)
    therapy_terms: List[str] = field(default_factory=list)
    therapy_methods: List[str] = field(default_factory=list)
    contraindication_terms: List[str] = field(default_factory=list)
    mistreatment_terms: List[str] = field(default_factory=list)
    mistreatment_types: List[str] = field(default_factory=list)
    transformation_terms: List[str] = field(default_factory=list)
    transmission_terms: List[str] = field(default_factory=list)
    prognosis_terms: List[str] = field(default_factory=list)
    time_course: List[str] = field(default_factory=list)
    formula_mentions: List[Dict] = field(default_factory=list)  # name/strength/negated/pos


class EntityExtractor:
    """Stateful extractor configured with the harvested formula inventory."""

    def __init__(self, formula_names: Optional[Sequence[str]] = None):
        names: Set[str] = set(lexicon.FORMULA_SEEDS)
        names.update(lexicon.FORMULA_ALIASES.keys())
        if formula_names:
            names.update(formula_names)
        self.formula_terms = sorted(names, key=lambda t: (-len(t), t))

    # -- formulas with prescription strength ------------------------------
    def extract_formula_mentions(self, text: str) -> List[Dict]:
        text = fold_variants(text)
        mentions: List[Dict] = []
        taken = [False] * len(text)
        for m in _match_terms(text, self.formula_terms, taken):
            name = lexicon.canonical_formula(m.term)
            before = text[max(0, m.start - 4):m.start]
            after = text[m.end:m.end + 4]
            strength = ""
            negated = False
            if after.startswith("主之"):
                strength = "主之"
            elif after.startswith("不中與"):
                negated = True
            elif before.endswith("不可與") or before.endswith("不可服"):
                negated = True
            elif before.endswith("可與") or before.endswith("可服") \
                    or after.startswith("可服") or after.startswith("亦可服"):
                strength = "可與"
            elif before.endswith("宜") or before.endswith("宜服"):
                strength = "宜"
            elif before.endswith("屬") or after.startswith("證"):
                strength = "屬"
            elif before.endswith("與") or before.endswith("用") or before.endswith("服"):
                strength = "與"
            elif after.startswith("和之") or after.startswith("發汗") or after.startswith("攻"):
                strength = "與"
            mentions.append({
                "name": name, "surface": m.term, "start": m.start, "end": m.end,
                "strength": strength, "negated": negated,
            })
        return mentions

    # -- main entry --------------------------------------------------------
    def extract(self, text: str) -> EntityResult:
        text = fold_variants(text)   # length-preserving; offsets stay valid
        res = EntityResult()
        taken = [False] * len(text)

        # 1. formulas first (longest spans; avoids 桂枝湯→桂枝 partial hits)
        res.formula_mentions = self.extract_formula_mentions(text)
        for fm in res.formula_mentions:
            for k in range(fm["start"], fm["end"]):
                taken[k] = True
            if fm["negated"]:
                if fm["name"] not in res.contraindicated_formulas:
                    res.contraindicated_formulas.append(fm["name"])
            elif fm["name"] not in res.formulas:
                res.formulas.append(fm["name"])

        # 2+4. disease patterns & symptoms — ONE longest-first pass across
        # both vocabularies, so a short disease term (痞) can never claim a
        # span inside a longer symptom (心下痞硬). Terms in both lists count
        # as disease (previous precedence preserved).
        disease_set = set(lexicon.DISEASE_PATTERNS)
        combined = sorted(disease_set | set(lexicon.SYMPTOMS),
                          key=lambda t: (-len(t), t))
        for m in _match_terms(text, combined, taken):
            if m.term in disease_set:
                if m.negated:
                    continue
                if m.term not in res.disease_patterns:
                    res.disease_patterns.append(m.term)
                for stem, channel in lexicon.CHANNEL_IN_TEXT.items():
                    if m.term.startswith(stem) and channel not in res.six_channels:
                        res.six_channels.append(channel)
                    # 合病 mentions both channels
                    if stem in m.term and channel not in res.six_channels and ("合病" in m.term or "併病" in m.term):
                        res.six_channels.append(channel)
                continue
            # symptom (negation-aware)
            if m.negated:
                neg_term = "不" + m.term
                if neg_term not in res.negated_findings:
                    res.negated_findings.append(neg_term)
                continue
            term = ("反" if m.contrary else "") + m.term
            # canonical negated forms (無汗/不惡寒…) are first-class symptoms
            if m.term not in res.symptoms and term not in res.symptoms:
                res.symptoms.append(term if m.contrary else m.term)

        # 3. pulse — named patterns then 脈-phrases
        for m in _match_terms(text, lexicon.PULSE_NAMED_PATTERNS, taken):
            t = m.term
            if t.startswith("脈"):
                t = t[1:]
            entry = ("不" if m.negated else "") + t
            if entry and entry not in res.pulse:
                res.pulse.append(entry)
        for pm in lexicon.RE_PULSE_PHRASE.finditer(text):
            seg = pm.group(0)
            quals = "".join(q for q in lexicon.PULSE_QUALITIES if q in seg[1:8])
            if quals and quals not in res.pulse and not any(quals in p for p in res.pulse):
                res.pulse.append(quals)

        # 5. mistreatment markers
        for mtype, pats in lexicon.MISTREATMENT_PATTERNS.items():
            for pat in sorted(pats, key=len, reverse=True):
                if pat in text:
                    if pat not in res.mistreatment_terms:
                        res.mistreatment_terms.append(pat)
                    if mtype not in res.mistreatment_types:
                        res.mistreatment_types.append(mtype)
                    break

        # 6. contraindications
        for cm in lexicon.RE_CONTRA.finditer(text):
            t = cm.group(0)
            if t not in res.contraindication_terms:
                res.contraindication_terms.append(t)

        # 7. therapy terms
        for method, terms in lexicon.THERAPY_METHODS.items():
            for t in sorted(terms, key=len, reverse=True):
                if t in text:
                    # skip if actually a contraindication (不可發汗) or
                    # mistreatment context handled separately
                    i = text.find(t)
                    before = text[max(0, i - 3):i]
                    if before.endswith("不可") or before.endswith("勿") or before.endswith("慎不可"):
                        continue
                    if t not in res.therapy_terms:
                        res.therapy_terms.append(t)
                    if method not in res.therapy_methods:
                        res.therapy_methods.append(method)
                    break

        # 8. transformations & transmission
        for t in lexicon.TRANSFORMATION_OUTCOMES:
            if t in text and t not in res.transformation_terms:
                if t in res.symptoms or any(t in s for s in res.symptoms):
                    res.transformation_terms.append(t)
                elif t in text:
                    res.transformation_terms.append(t)
        for t in lexicon.TRANSMISSION_MARKERS:
            if t in text and t not in res.transmission_terms:
                if t == "傳" and ("不傳" in text and text.count("傳") == text.count("不傳")):
                    res.transmission_terms.append("不傳")
                else:
                    res.transmission_terms.append(t)

        # 9. prognosis
        for cls_, terms in lexicon.PROGNOSIS_MARKERS.items():
            for t in sorted(terms, key=len, reverse=True):
                if t in text:
                    if t == "死" and ("不死" in text):
                        continue
                    if t == "愈" and any(x in res.prognosis_terms for x in ("自愈", "欲愈")):
                        break
                    if t not in res.prognosis_terms:
                        res.prognosis_terms.append(t)
                    break

        # 10. time course
        for tm in lexicon.RE_TIME_COURSE.finditer(text):
            t = tm.group(1)
            if t not in res.time_course:
                res.time_course.append(t)
        return res


def annotate_clause(clause: ShanghanClause, extractor: EntityExtractor) -> ShanghanClause:
    """Fill a clause's entity fields from its clean text + formula blocks."""
    res = extractor.extract(clause.clean_text)
    clause.symptoms = res.symptoms
    clause.negated_findings = res.negated_findings
    clause.pulse = res.pulse
    clause.disease_patterns = res.disease_patterns
    clause.therapy_terms = res.therapy_terms
    clause.contraindication_terms = res.contraindication_terms
    clause.mistreatment_terms = res.mistreatment_terms
    clause.transformation_terms = res.transformation_terms
    clause.prognosis_terms = res.prognosis_terms
    clause.time_course = res.time_course
    for fm in res.formulas:
        if fm not in clause.formula_names:
            clause.formula_names.append(fm)
    clause.contains_formula = bool(clause.formula_names or clause.formula_blocks)
    herbs = []
    for fb in clause.formula_blocks:
        herbs.extend(c["herb"] for c in fb.composition)
    clause.herbs = sorted(set(herbs))
    return clause
