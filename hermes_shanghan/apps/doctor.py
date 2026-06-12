"""Doctor-mode 方證匹配 (formula pattern matching).

Scores verified FormulaPatternRules against the presented findings:
  + core symptom hit ×2.0      + associated symptom hit ×1.0
  + core pulse hit ×2.0        + associated pulse hit ×1.0
  − contradiction ×2.5 (e.g. presented 無汗 vs pattern's 汗出)
  − contraindication conflict ×2.0

Every match returns the verbatim supporting clauses (evidence chain) and an
assistive-only safety notice. 無原文，不成規則；無條文編號，不成證據。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import config, lexicon, safety
from ..schemas import FormulaPatternRule, ShanghanClause
from ..textutil import normalize_query


def _normalize_findings(items: List[str]) -> List[str]:
    return [normalize_query(x) for x in items if x and x.strip()]


def _contradicts(finding: str, pattern_terms: List[str]) -> Optional[str]:
    for a, b in lexicon.CONTRADICTORY_SYMPTOMS:
        if finding == a and b in pattern_terms:
            return b
        if finding == b and a in pattern_terms:
            return a
    return None


class FormulaMatcher:
    def __init__(self, formula_rules: List[FormulaPatternRule],
                 clause_store: Dict[str, ShanghanClause]):
        self.rules = [r for r in formula_rules if r.release_level != "rejected"]
        self.clauses = clause_store

    def match(self, symptoms: List[str], pulse: Optional[List[str]] = None,
              six_channel: Optional[str] = None, top_k: int = 5,
              need_original_evidence: bool = True) -> Dict:
        symptoms = _normalize_findings(symptoms or [])
        pulse = _normalize_findings(pulse or [])
        results = []
        for r in self.rules:
            if six_channel and six_channel not in r.six_channel_scope:
                continue
            score, hits, conflicts = 0.0, [], []
            pattern_syms = r.core_symptoms + r.associated_symptoms
            for s in symptoms:
                matched = False
                for cs in r.core_symptoms:
                    if s == cs or s in cs or cs in s:
                        score += 2.0
                        hits.append(f"核心證：{cs}")
                        matched = True
                        break
                if not matched:
                    for asym in r.associated_symptoms:
                        if s == asym or s in asym or asym in s:
                            score += 1.0
                            hits.append(f"兼證：{asym}")
                            matched = True
                            break
                if not matched:
                    contra = _contradicts(s, pattern_syms)
                    if contra:
                        score -= 2.5
                        conflicts.append(f"所述「{s}」與本方證之「{contra}」相反")
            for p in pulse:
                body = p.lstrip("脈")
                matched = False
                for cp in r.core_pulse:
                    if body == cp or body in cp or cp in body:
                        score += 2.0
                        hits.append(f"核心脈：{cp}")
                        matched = True
                        break
                if not matched:
                    for ap in r.associated_pulse:
                        if body == ap or body in ap or ap in body:
                            score += 1.0
                            hits.append(f"兼脈：{ap}")
                            break
            if score <= 0:
                continue
            # evidence-thickness bonus: better-attested patterns win ties
            score += min(0.3, 0.05 * len(r.supporting_clauses))
            denom = 2.0 * (len(symptoms) + len(pulse)) or 1.0
            norm = max(0.0, min(1.0, score / denom))
            results.append((norm, score, r, hits, conflicts))

        results.sort(key=lambda t: (-t[0], -t[1], -len(t[2].supporting_clauses)))
        matches = []
        for norm, raw, r, hits, conflicts in results[:top_k]:
            evidence = []
            if need_original_evidence:
                for cid in r.supporting_clauses[:3]:
                    c = self.clauses.get(cid)
                    if c:
                        evidence.append({
                            "book": c.book_title, "chapter": c.chapter,
                            "clause_id": c.clause_id,
                            "clause_number": c.clause_number,
                            "text": c.clean_text,
                        })
            matches.append({
                "formula": r.formula,
                "match_score": round(norm, 2),
                "six_channel": "、".join(r.six_channel_scope),
                "core_pattern": r.core_pattern,
                "core_reason": (
                    f"{'、'.join(h.split('：')[1] for h in hits[:6])}"
                    f"與{r.core_pattern}（{r.formula}）相關度較高。" if hits else ""),
                "matched_findings": hits,
                "conflicts": conflicts,
                "contraindications": r.contraindications[:3],
                "source_level": r.source_level,
                "release_level": r.release_level,
                "interpretation_warning": r.interpretation_warning,
                "evidence": evidence,
            })
        payload = {
            "input": {"symptoms": symptoms, "pulse": pulse, "six_channel": six_channel},
            "matched_formula_patterns": matches,
            "match_count": len(matches),
        }
        return safety.governed(payload, "doctor")
