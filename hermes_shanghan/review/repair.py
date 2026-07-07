"""AutoRepairAgent — fixes machine-actionable review flags, then the rule is
re-verified. Unfixable rules fall through to rejection.

Repairs never invent evidence; they only remove or downgrade content:
  * strip condition terms that are not attested in the clause;
  * downgrade inflated prescription strength to what the text supports;
  * move posthoc vocabulary out of if/then into the interpretation;
  * re-label interpretation_level when latter-day terms are present;
  * convert positive symptoms that only occur negated into negated findings.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .. import lexicon
from ..textutil import fold_variants
from ..schemas import InitialRule, ShanghanClause


def repair(rule: InitialRule, flags: List[str],
           clause_store: Dict[str, ShanghanClause]) -> Tuple[InitialRule, List[str]]:
    clause = clause_store.get(rule.clause_id)
    if clause is None:
        return rule, []
    text = fold_variants(clause.clean_text
                         + "\n".join(fb.raw_text for fb in clause.formula_blocks))
    applied: List[str] = []

    for flag in flags:
        # --- evidence: unattested condition terms → drop --------------------
        if flag.startswith("evidence:condition_not_in_text:"):
            _, _, rest = flag.partition("evidence:condition_not_in_text:")
            key, _, term = rest.partition(":")
            vals = rule.if_conditions.get(key) or []
            if term in vals:
                vals.remove(term)
                rule.if_conditions[key] = vals
                applied.append(f"repair:dropped_condition:{key}:{term}")

        elif flag.startswith("evidence:pulse_not_in_text:"):
            term = flag.rsplit(":", 1)[1]
            vals = rule.if_conditions.get("pulse") or []
            if term in vals:
                vals.remove(term)
                rule.if_conditions["pulse"] = vals
                applied.append(f"repair:dropped_pulse:{term}")

        elif flag.startswith("evidence:formula_not_in_clause:"):
            term = flag.rsplit(":", 1)[1]
            vals = rule.then_conclusions.get("formula") or []
            if term in vals:
                vals.remove(term)
                rule.then_conclusions["formula"] = vals
                applied.append(f"repair:dropped_formula:{term}")

        # --- strength inflation → downgrade to attested marker -------------
        elif "keyu_inflated_to_zhuzhi" in flag or "strength_overclaimed" in flag:
            f_list = rule.then_conclusions.get("formula") or []
            attested = ""
            if f_list:
                f = f_list[0]
                surfaces = [f] + [a for a, c in lexicon.FORMULA_ALIASES.items() if c == f]
                for s in surfaces:
                    if ("宜" + s) in text or ("宜服" + s) in text:
                        attested = "宜"
                        break
                    if ("可與" + s) in text or ("可服" + s) in text:
                        attested = "可與"
                        break
                    if ("與" + s) in text or ("屬" + s) in text:
                        attested = "與" if ("與" + s) in text else "屬"
                        break
            rule.prescription_strength = attested or "可與"
            rule.interpretation += f"（原文證據強度經審核降級為「{rule.prescription_strength}」。）"
            applied.append(f"repair:strength_downgraded:{rule.prescription_strength}")

        # --- posthoc terms in rule body → move to interpretation -----------
        elif flag.startswith("critic:posthoc_term_in_rule_body:"):
            ph = flag.rsplit(":", 1)[1]
            for cond in (rule.if_conditions, rule.then_conclusions):
                for key, vals in list(cond.items()):
                    if isinstance(vals, list):
                        kept = [v for v in vals if not (isinstance(v, str) and ph in v)]
                        if len(kept) != len(vals):
                            cond[key] = kept
            note = f"後世歸納術語「{ph}」非原文直述，僅作為模型解釋。"
            if note not in rule.interpretation:
                rule.interpretation += note
            rule.interpretation_level = "model_inference"
            applied.append(f"repair:posthoc_moved_to_interpretation:{ph}")

        elif flag.startswith("critic:posthoc_term_as_literal:"):
            rule.interpretation_level = "model_inference"
            applied.append("repair:interpretation_level_downgraded")

        # --- negation traps -------------------------------------------------
        elif flag.startswith("critic:negated_symptom_recorded_positive:"):
            s = flag.rsplit(":", 1)[1]
            syms = rule.if_conditions.get("symptoms") or []
            if s in syms:
                syms.remove(s)
                neg = rule.if_conditions.setdefault("negated_findings", [])
                if ("不" + s) not in neg:
                    neg.append("不" + s)
                applied.append(f"repair:symptom_negated:{s}")

        # --- ignored contraindication → annotate ----------------------------
        elif flag.startswith("critic:contraindication_ignored:"):
            term = flag.rsplit(":", 1)[1]
            lst = rule.if_conditions.setdefault("contraindications", [])
            if term not in lst:
                lst.append(term)
            rule.interpretation += f"本條同時載明禁忌「{term}」，用方須以該禁忌為前提。"
            applied.append(f"repair:contraindication_annotated:{term}")

        # --- scope expansion → restore negated qualifiers -------------------
        elif flag.startswith("critic:zhuzhi_scope_expanded:"):
            missing = flag.rsplit(":", 1)[1].split(",")
            neg = rule.if_conditions.setdefault("negated_findings", [])
            for m in missing:
                if m and m not in neg and m.lstrip("不") in text:
                    neg.append(m)
                    applied.append(f"repair:negated_qualifier_restored:{m}")

        # --- channel label confusion → fix the interpretation ---------------
        elif "shaoyin_heat_mislabeled_cold" in flag:
            rule.interpretation = rule.interpretation.replace("寒化", "熱化").replace("回陽", "育陰清熱")
            applied.append("repair:shaoyin_label_fixed")
        elif "shaoyin_cold_mislabeled_heat" in flag:
            rule.interpretation = rule.interpretation.replace("熱化", "寒化")
            applied.append("repair:shaoyin_label_fixed")
        elif "yangming_jing_fu_confusion" in flag:
            rule.interpretation += "（陽明經證/腑證之分屬後世歸納，須以原文燥屎、潮熱、譫語等指徵為準。）"
            rule.interpretation_level = "model_inference"
            applied.append("repair:yangming_label_annotated")

    return rule, applied
