"""FormulaPatternAgent — induces FormulaPatternRules from verified
InitialRules (never the other way round; merged rules never overwrite
clause-level rules).

Core/associated split: a finding is *core* if it appears in ≥2 supporting
clauses, or appears in a 主之-strength clause; otherwise associated.
加減方 relations are detected from composition diffs + name morphology
(桂枝湯 → 桂枝加葛根湯 etc.).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .. import config, lexicon
from ..schemas import FormulaPatternRule, InitialRule, ShanghanClause, write_jsonl


def _patient_pattern_name(formula: str, channels: List[str], diseases: Counter) -> str:
    main_channel = channels[0] if channels else ""
    top = [d for d, _ in diseases.most_common(2) if d not in ("傷寒",)]
    if top:
        return f"{top[0]}類證"
    return f"{main_channel}{formula}證" if main_channel else f"{formula}證"


class FormulaPatternInducer:
    def __init__(self, clauses: List[ShanghanClause], initial_rules: List[InitialRule]):
        self.clause_store = {c.clause_id: c for c in clauses}
        self.rules = initial_rules

    def induce(self) -> List[FormulaPatternRule]:
        # group accepted formula_pattern_rules by formula
        by_formula: Dict[str, List[InitialRule]] = defaultdict(list)
        comp_rules: Dict[str, InitialRule] = {}
        admin_rules: Dict[str, List[InitialRule]] = defaultdict(list)
        contra_clauses: Dict[str, List[InitialRule]] = defaultdict(list)
        for r in self.rules:
            if r.autonomous_review.release_level == "rejected":
                continue
            if r.rule_type == "formula_pattern_rule":
                for f in r.then_conclusions.get("formula", []):
                    by_formula[f].append(r)
            elif r.rule_type == "formula_composition_rule":
                f = (r.if_conditions.get("formula") or [""])[0]
                if f and f not in comp_rules:
                    comp_rules[f] = r
            elif r.rule_type == "administration_rule":
                f = (r.if_conditions.get("formula") or [""])[0]
                if f:
                    admin_rules[f].append(r)
            elif r.rule_type == "contraindication_rule":
                for f in r.then_conclusions.get("contraindicated_formulas", []):
                    contra_clauses[f].append(r)

        out: List[FormulaPatternRule] = []
        for n, (formula, group) in enumerate(sorted(by_formula.items()), 1):
            sym_counter: Counter = Counter()
            pulse_counter: Counter = Counter()
            disease_counter: Counter = Counter()
            zhuzhi_sym, zhuzhi_pulse = set(), set()
            channels: Counter = Counter()
            clause_ids: List[str] = []
            for r in group:
                clause_ids.append(r.clause_id)
                if r.six_channel:
                    channels[r.six_channel] += 1
                for s in r.if_conditions.get("symptoms", []):
                    sym_counter[s] += 1
                    if r.prescription_strength == "主之":
                        zhuzhi_sym.add(s)
                for p in r.if_conditions.get("pulse", []):
                    pulse_counter[p] += 1
                    if r.prescription_strength == "主之":
                        zhuzhi_pulse.add(p)
                for d in r.if_conditions.get("disease", []):
                    disease_counter[d] += 1

            core_sym = [s for s, c in sym_counter.most_common()
                        if c >= 2 or s in zhuzhi_sym][:8]
            assoc_sym = [s for s, c in sym_counter.most_common()
                         if s not in core_sym][:8]
            core_pulse = [p for p, c in pulse_counter.most_common()
                          if c >= 2 or p in zhuzhi_pulse][:4]
            assoc_pulse = [p for p in pulse_counter if p not in core_pulse][:4]
            channel_scope = [ch for ch, _ in channels.most_common()]

            comp = comp_rules.get(formula)
            admin_notes: List[str] = []
            for ar in admin_rules.get(formula, [])[:2]:
                t = (ar.then_conclusions.get("administration") or "")
                if t:
                    admin_notes.append(t[:120])
                for pn in ar.then_conclusions.get("post_notes", [])[:2]:
                    admin_notes.append(pn[:120])

            contras = []
            for cr in contra_clauses.get(formula, []):
                cl = self.clause_store.get(cr.clause_id)
                if cl:
                    contras.append({"clause_id": cr.clause_id,
                                    "condition": cl.clean_text[:60]})

            scores = [r.autonomous_review.consensus_score for r in group]
            avg = sum(scores) / len(scores)
            zhuzhi_n = sum(1 for r in group if r.prescription_strength == "主之")
            level = "gold" if (len(group) >= 2 and zhuzhi_n >= 1 and avg >= 0.85) else \
                    ("silver" if avg >= 0.78 else "bronze")

            rule = FormulaPatternRule(
                formula_pattern_rule_id=f"FPR_{n:04d}",
                formula=formula,
                formula_family=lexicon.formula_family(formula) or "",
                six_channel_scope=channel_scope,
                core_pattern=_patient_pattern_name(formula, channel_scope, disease_counter),
                core_symptoms=core_sym,
                core_pulse=core_pulse,
                associated_symptoms=assoc_sym,
                associated_pulse=assoc_pulse,
                contraindications=contras,
                composition=(comp.then_conclusions.get("composition") if comp else []) or [],
                administration_notes=admin_notes,
                supporting_initial_rules=[r.initial_rule_id for r in group],
                supporting_clauses=sorted(set(clause_ids)),
                source_level="inductive_from_original_clauses",
                interpretation_warning=(
                    "核心證候為跨條文歸納結果；病機類術語（如營衛不和）屬後世歸納，"
                    "非所有條文原文直述。"),
                consensus_score=round(avg, 3),
                release_level=level,
            )
            out.append(rule)

        self._attach_modifications(out)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _attach_modifications(rules: List[FormulaPatternRule]):
        """加減方 relations via name morphology + composition diff."""
        by_name = {r.formula: r for r in rules}
        for r in rules:
            for other_name, other in by_name.items():
                if other_name == r.formula:
                    continue
                base_stem = r.formula.replace("湯", "").replace("丸", "").replace("散", "")
                if other_name.startswith(base_stem) and \
                        any(k in other_name for k in ("加", "去")) and \
                        len(other_name) > len(r.formula):
                    a = {c["herb"] for c in r.composition}
                    b = {c["herb"] for c in other.composition}
                    added = sorted(b - a)
                    removed = sorted(a - b)
                    r.modification_relations.append({
                        "modified_formula": other_name,
                        "relation": "加減方",
                        "added_herbs": "、".join(added),
                        "removed_herbs": "、".join(removed),
                    })

    # ------------------------------------------------------------------
    def run(self) -> List[FormulaPatternRule]:
        rules = self.induce()
        config.ensure_dirs()
        write_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl", rules)
        return rules
