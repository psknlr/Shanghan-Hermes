"""MergedShanghanRule builder — top of the rule hierarchy.

InitialRule → FormulaPatternRule → SixChannelRule → TherapyRule →
MistreatmentTransformationRule → MergedShanghanRule

A merged rule is an *aggregation view*: it references the lower layers by ID
and carries the full evidence chain. It never mutates or replaces them
(合併規則不能覆蓋初始條文規則). Conflicts (e.g. a formula prescribed in one
clause and forbidden in another) are surfaced, not silently resolved.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import config
from ..schemas import (FormulaPatternRule, InitialRule, MergedShanghanRule,
                       MistreatmentTransformationRule, ShanghanClause,
                       SixChannelRule, TherapyRule, VariantRule,
                       CommentaryRule, write_jsonl)


class MergedRuleBuilder:
    def __init__(self,
                 clauses: List[ShanghanClause],
                 initial_rules: List[InitialRule],
                 formula_rules: List[FormulaPatternRule],
                 six_channel_rules: List[SixChannelRule],
                 therapy_rules: List[TherapyRule],
                 mistreatment_rules: List[MistreatmentTransformationRule],
                 variant_rules: Optional[List[VariantRule]] = None,
                 commentary_rules: Optional[List[CommentaryRule]] = None):
        self.clause_store = {c.clause_id: c for c in clauses}
        self.initial_rules = initial_rules
        self.formula_rules = formula_rules
        self.six_channel_rules = six_channel_rules
        self.therapy_rules = therapy_rules
        self.mistreatment_rules = mistreatment_rules
        self.variants = variant_rules or []
        self.commentaries = commentary_rules or []
        self._v_by_clause: Dict[str, List[VariantRule]] = {}
        for v in self.variants:
            self._v_by_clause.setdefault(v.clause_id, []).append(v)
        self._c_by_clause: Dict[str, List[CommentaryRule]] = {}
        for cm in self.commentaries:
            self._c_by_clause.setdefault(cm.clause_id, []).append(cm)

    def _evidence_chain(self, clause_ids: List[str], limit: int = 8) -> List[Dict]:
        chain = []
        for cid in clause_ids[:limit]:
            cl = self.clause_store.get(cid)
            if cl:
                chain.append({
                    "clause_id": cid,
                    "book": cl.book_title,
                    "chapter": cl.chapter,
                    "clause_number": cl.clause_number,
                    "text": cl.clean_text,
                    "layer": cl.layer,
                    "text_type": cl.text_type,
                })
        return chain

    def _conflicts_for_formula(self, formula: str,
                               fpr: FormulaPatternRule) -> List[Dict]:
        conflicts = []
        for c in fpr.contraindications:
            conflicts.append({
                "type": "prescribed_vs_forbidden",
                "description": f"{formula}既有主治條文，亦有禁例（{c['clause_id']}），"
                               f"使用必須同時滿足主證並排除禁忌。",
                "clause_id": c["clause_id"],
            })
        return conflicts

    def build(self) -> List[MergedShanghanRule]:
        out: List[MergedShanghanRule] = []
        n = 0
        scr_by_channel = {r.six_channel: r for r in self.six_channel_rules}

        # —— per (channel × formula) merged rules for principal formulas ——
        for fpr in self.formula_rules:
            if fpr.release_level not in ("gold", "silver"):
                continue
            channel = fpr.six_channel_scope[0] if fpr.six_channel_scope else ""
            scr = scr_by_channel.get(channel)
            mtrs = [m for m in self.mistreatment_rules if fpr.formula in m.rescue_formulas]
            trs = [t for t in self.therapy_rules if fpr.formula in t.representative_formulas]
            n += 1
            pinyin = config.CHANNEL_PINYIN.get(channel, "misc").upper()
            variants = []
            commentaries = []
            for cid in fpr.supporting_clauses:
                variants += [v.variant_rule_id for v in self._v_by_clause.get(cid, [])]
                commentaries += [c.commentary_rule_id for c in self._c_by_clause.get(cid, [])]
            chapters = sorted({self.clause_store[cid].chapter
                               for cid in fpr.supporting_clauses
                               if cid in self.clause_store})
            core_desc = "、".join(fpr.core_symptoms[:5])
            pulse_desc = "、".join(fpr.core_pulse[:3])
            claim = (f"{fpr.formula}可作為{fpr.core_pattern}"
                     f"（{core_desc}{'，脈' + pulse_desc if pulse_desc else ''}）的核心方。")
            conflicts = self._conflicts_for_formula(fpr.formula, fpr)
            score = min(0.96, fpr.consensus_score + 0.02 * min(3, len(fpr.supporting_clauses)))
            out.append(MergedShanghanRule(
                merged_rule_id=f"MHR_SHL_{pinyin}_{n:04d}",
                title=f"{channel + ' ' if channel else ''}{fpr.formula}方證合併規則",
                claim=claim,
                source_scope={
                    "main_version": "傷寒論（宋本）",
                    "chapters": chapters,
                    "six_channels": fpr.six_channel_scope,
                },
                supporting_initial_rules=fpr.supporting_initial_rules,
                supporting_formula_pattern_rules=[fpr.formula_pattern_rule_id],
                supporting_six_channel_rules=[scr.six_channel_rule_id] if scr else [],
                supporting_therapy_rules=[t.therapy_rule_id for t in trs],
                supporting_mistreatment_rules=[m.mistreatment_rule_id for m in mtrs],
                variants=variants[:10],
                commentaries=commentaries[:10],
                conflicts=conflicts,
                evidence_chain=self._evidence_chain(fpr.supporting_clauses),
                release_level=fpr.release_level,
                consensus_score=round(score, 3),
            ))

        # —— per-channel merged rules ——
        for scr in self.six_channel_rules:
            n += 1
            pinyin = config.CHANNEL_PINYIN.get(scr.six_channel, "misc").upper()
            fprs = [f for f in self.formula_rules if scr.six_channel in f.six_channel_scope]
            mtrs = [m for m in self.mistreatment_rules if scr.six_channel in m.six_channel_scope]
            out.append(MergedShanghanRule(
                merged_rule_id=f"MHR_SHL_{pinyin}_{n:04d}",
                title=f"{scr.six_channel}六經合併規則",
                claim=scr.summary,
                source_scope={
                    "main_version": "傷寒論（宋本）",
                    "six_channels": [scr.six_channel],
                },
                supporting_initial_rules=scr.supporting_initial_rules[:100],
                supporting_formula_pattern_rules=[f.formula_pattern_rule_id for f in fprs][:30],
                supporting_six_channel_rules=[scr.six_channel_rule_id],
                supporting_mistreatment_rules=[m.mistreatment_rule_id for m in mtrs][:20],
                evidence_chain=self._evidence_chain(scr.core_clauses, limit=6),
                release_level=scr.release_level,
                consensus_score=scr.consensus_score,
            ))
        return out

    def run(self) -> List[MergedShanghanRule]:
        rules = self.build()
        config.ensure_dirs()
        write_jsonl(config.RULES_MERGED_DIR / "merged_rules.jsonl", rules)
        return rules
