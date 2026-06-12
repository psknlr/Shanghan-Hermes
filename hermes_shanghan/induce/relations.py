"""ClauseRelation graph builder + variant/commentary alignment.

Relation types produced here:
  sequence                    — adjacent clauses in the same chapter (並列/遞進)
  same_formula_family         — clauses sharing a normalized formula
  mistreatment_transformation — 誤治條 → 救治方條
  contraindication            — 禁忌條 → 該方主證條
  differential                — same-channel clauses sharing symptoms but
                                concluding different formulas
  transmission                — 傳變條 → 目標經提綱條
  variant                     — 桂本/千金翼方版 aligned paragraphs (layer B)
  commentary_support          — 註解傷寒論 aligned commentary (layer C)

Variant and commentary alignment use char-bigram Dice similarity with an
inverted-index prefilter, so the whole graph builds in seconds.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .. import config
from ..corpus import segmenter
from ..schemas import (ClauseRelation, CommentaryRule, ShanghanClause,
                       VariantRule, write_jsonl)
from ..textutil import bigram_set, similarity

VARIANT_SIM_THRESHOLD = 0.62
COMMENT_SIM_THRESHOLD = 0.60


class RelationBuilder:
    def __init__(self, clauses: List[ShanghanClause]):
        self.clauses = clauses
        self.canonical = [c for c in clauses if c.text_type == "original_clause"]
        self.relations: List[ClauseRelation] = []
        self._n = 0

    def _add(self, src: str, tgt: str, rtype: str, desc: str, conf: float, **ev):
        self._n += 1
        self.relations.append(ClauseRelation(
            relation_id=f"REL_{self._n:05d}", source_clause_id=src,
            target_clause_id=tgt, relation_type=rtype, description=desc,
            evidence=ev or {}, confidence=round(conf, 3)))

    # ------------------------------------------------------------------
    def build_sequence(self):
        prev: Optional[ShanghanClause] = None
        for c in self.canonical:
            if prev is not None and prev.chapter == c.chapter:
                self._add(prev.clause_id, c.clause_id, "sequence",
                          "同篇相鄰條文（並列/遞進語境）", 0.7)
            prev = c

    def build_formula_family(self):
        by_formula: Dict[str, List[ShanghanClause]] = defaultdict(list)
        for c in self.canonical:
            for f in c.formula_names:
                by_formula[f].append(c)
        for f, group in by_formula.items():
            for a, b in zip(group, group[1:]):
                self._add(a.clause_id, b.clause_id, "same_formula_family",
                          f"兩條均涉及{f}方證，但症狀條件不同。", 0.86, formula=f)

    def build_mistreatment(self):
        for c in self.canonical:
            if not c.mistreatment_terms or not c.formula_names:
                continue
            # 誤治條 itself names the rescue formula — link to that formula's
            # 主之 anchor clause (the formula's first 主之 occurrence)
            for f in c.formula_names:
                anchor = self._formula_anchor(f)
                if anchor and anchor.clause_id != c.clause_id:
                    self._add(c.clause_id, anchor.clause_id,
                              "mistreatment_transformation",
                              f"誤治變證（{ '、'.join(c.mistreatment_terms[:2])}）以{f}救治，"
                              f"與該方主證條互參。", 0.8, formula=f)

    def _formula_anchor(self, formula: str) -> Optional[ShanghanClause]:
        for c in self.canonical:
            if formula in c.formula_names and f"{formula}主之" in c.clean_text:
                return c
        for c in self.canonical:
            if formula in c.formula_names:
                return c
        return None

    def build_contraindication(self):
        for c in self.canonical:
            ctext = c.clean_text
            for f in set(c.formula_names):
                if f"不可與{f}" in ctext or f"{f}不中與" in ctext or f"不可服{f}" in ctext:
                    anchor = self._formula_anchor(f)
                    if anchor and anchor.clause_id != c.clause_id:
                        self._add(c.clause_id, anchor.clause_id, "contraindication",
                                  f"本條為{f}之禁例，與其主證條對勘。", 0.85, formula=f)

    def build_differential(self):
        # same-channel clause pairs sharing ≥2 symptoms but different formulas
        seen = set()
        by_channel: Dict[str, List[ShanghanClause]] = defaultdict(list)
        for c in self.canonical:
            if c.formula_names and c.symptoms:
                by_channel[c.six_channel].append(c)
        for channel, group in by_channel.items():
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if set(a.formula_names) == set(b.formula_names):
                        continue
                    shared = set(a.symptoms) & set(b.symptoms)
                    if len(shared) >= 2:
                        key = (a.clause_id, b.clause_id)
                        if key in seen:
                            continue
                        seen.add(key)
                        self._add(a.clause_id, b.clause_id, "differential",
                                  f"二條共見{'、'.join(sorted(shared)[:3])}而結論方不同"
                                  f"（{'、'.join(a.formula_names[:1])} vs "
                                  f"{'、'.join(b.formula_names[:1])}），構成鑒別。",
                                  0.72, shared=sorted(shared))

    def build_transmission(self):
        outline = {ch: f"{config.ID_PREFIX_CLAUSE}{n:04d}"
                   for ch, n in config.CHANNEL_OUTLINE_CLAUSE.items()}
        for c in self.canonical:
            t = c.clean_text
            for stem, channel in (("陽明", "陽明病"), ("少陽", "少陽病"),
                                  ("太陰", "太陰病"), ("少陰", "少陰病"),
                                  ("厥陰", "厥陰病")):
                if (f"轉屬{stem}" in t or f"轉入{stem}" in t or f"屬{stem}" in t) \
                        and c.six_channel != channel and channel in outline:
                    self._add(c.clause_id, outline[channel], "transmission",
                              f"本條言傳變：{c.six_channel}→{channel}。", 0.8,
                              target_channel=channel)

    # ------------------------------------------------------------------
    # Variant alignment (layer B)
    # ------------------------------------------------------------------
    def _align(self, paragraphs: List[Tuple[str, str]],
               threshold: float) -> List[Tuple[ShanghanClause, str, str, float]]:
        """Align corpus paragraphs to canonical clauses via bigram index."""
        index: Dict[str, List[int]] = defaultdict(list)
        for pi, (_, para) in enumerate(paragraphs):
            for bg in bigram_set(para):
                index[bg].append(pi)
        out = []
        for c in self.canonical:
            bgs = bigram_set(c.clean_text)
            counts: Dict[int, int] = defaultdict(int)
            for bg in bgs:
                for pi in index.get(bg, ()):
                    counts[pi] += 1
            if not counts:
                continue
            best = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            best_pi, best_sim = -1, 0.0
            for pi, _ in best:
                sim = similarity(c.clean_text, paragraphs[pi][1])
                if sim > best_sim:
                    best_pi, best_sim = pi, sim
            if best_pi >= 0 and best_sim >= threshold:
                chapter, para = paragraphs[best_pi]
                out.append((c, chapter, para, best_sim))
        return out

    def build_variants(self) -> List[VariantRule]:
        rules: List[VariantRule] = []
        n = 0
        for book in config.VARIANT_BOOKS:
            try:
                paragraphs = segmenter.segment_paragraphs(book)
            except FileNotFoundError:
                continue
            version = "guiben" if "桂本" in book else "qianjinyi"
            for c, chapter, para, sim in self._align(paragraphs, VARIANT_SIM_THRESHOLD):
                n += 1
                diffs: List[str] = []
                if sim < 0.97:
                    a_set, b_set = set(c.clean_text), set(para)
                    only_b = "".join(sorted(b_set - a_set))[:30]
                    only_a = "".join(sorted(a_set - b_set))[:30]
                    if only_a:
                        diffs.append(f"宋本獨有用字：{only_a}")
                    if only_b:
                        diffs.append(f"{book}獨有用字：{only_b}")
                rules.append(VariantRule(
                    variant_rule_id=f"VR_{version.upper()}_{n:04d}",
                    clause_id=c.clause_id, variant_version=version,
                    variant_book=book, base_text=c.clean_text,
                    variant_text=para, similarity=round(sim, 3),
                    notable_differences=diffs,
                    release_level="silver" if sim >= 0.8 else "bronze"))
                self._add(c.clause_id, f"{book}:{chapter}", "variant",
                          f"{book}存在對應異文（相似度{sim:.2f}）。", sim,
                          book=book)
        return rules

    # ------------------------------------------------------------------
    # Commentary alignment (layer C, 註解傷寒論)
    # ------------------------------------------------------------------
    def build_commentary(self) -> List[CommentaryRule]:
        try:
            paragraphs = segmenter.segment_paragraphs(config.COMMENTARY_ALIGN_BOOK)
        except FileNotFoundError:
            return []
        # In 註解傷寒論 the original clause paragraph is followed by Cheng
        # Wuji's commentary paragraph(s). Identify original-clause paragraphs
        # by alignment, then attach the paragraphs that follow them.
        matched: Dict[int, Tuple[ShanghanClause, float]] = {}
        index: Dict[str, List[int]] = defaultdict(list)
        for pi, (_, para) in enumerate(paragraphs):
            for bg in bigram_set(para):
                index[bg].append(pi)
        for c in self.canonical:
            counts: Dict[int, int] = defaultdict(int)
            for bg in bigram_set(c.clean_text):
                for pi in index.get(bg, ()):
                    counts[pi] += 1
            best = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            for pi, _ in best:
                sim = similarity(c.clean_text, paragraphs[pi][1])
                if sim >= COMMENT_SIM_THRESHOLD and \
                        (pi not in matched or sim > matched[pi][1]):
                    matched[pi] = (c, sim)
        rules: List[CommentaryRule] = []
        n = 0
        matched_pis = sorted(matched.keys())
        for k, pi in enumerate(matched_pis):
            clause, sim = matched[pi]
            end = matched_pis[k + 1] if k + 1 < len(matched_pis) else min(pi + 3, len(paragraphs))
            commentary_parts = [paragraphs[q][1] for q in range(pi + 1, end)
                                if paragraphs[q][0] == paragraphs[pi][0]][:2]
            if not commentary_parts:
                continue
            n += 1
            rules.append(CommentaryRule(
                commentary_rule_id=f"CR_ZHUJIE_{n:04d}",
                clause_id=clause.clause_id, commentator="成無己",
                book=config.COMMENTARY_ALIGN_BOOK,
                commentary_text="\n".join(commentary_parts),
                alignment_similarity=round(sim, 3)))
            self._add(clause.clause_id, f"註解傷寒論:p{pi}", "commentary_support",
                      f"成無己《註解傷寒論》對本條有注（對齊相似度{sim:.2f}）。", sim)
        return rules

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, int]:
        self.build_sequence()
        self.build_formula_family()
        self.build_mistreatment()
        self.build_contraindication()
        self.build_differential()
        self.build_transmission()
        variants = self.build_variants()
        commentaries = self.build_commentary()
        config.ensure_dirs()
        write_jsonl(config.RELATION_DIR / "clause_relations.jsonl", self.relations)
        write_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl", variants)
        write_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl", commentaries)
        return {"relations": len(self.relations), "variants": len(variants),
                "commentaries": len(commentaries)}
