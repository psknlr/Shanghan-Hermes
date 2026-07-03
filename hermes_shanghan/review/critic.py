"""ShanghanCriticAgent — adversarial reviewer specialized for Shanghan Lun.

The critic exists to find exactly the error classes named in the protocol:

  * latter-day pattern vocabulary (營衛不和…) passed off as original text;
  * ignored contraindication conditions in the same clause;
  * 「可與」 inflated into 「主之」-strength claims;
  * 「主之」 scope expanded beyond the clause's stated conditions;
  * 太陽中風 confused with 太陽傷寒 (汗出/無汗 axis);
  * 少陰寒化 confused with 少陰熱化;
  * 陽明經證 confused with 陽明腑證;
  * symptoms invented by the model (not present in the clause).

Every flag carries a machine-actionable code so AutoRepair can attempt a fix.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .. import lexicon
from ..textutil import fold_variants
from ..schemas import InitialRule, ShanghanClause

# formulas that anchor the classic confusion axes
ZHONGFENG_FORMULAS = {"桂枝湯", "桂枝加葛根湯", "桂枝加附子湯"}
SHANGHAN_FORMULAS = {"麻黃湯", "大青龍湯", "葛根湯"}
SHAOYIN_HEAT_FORMULAS = {"黃連阿膠湯", "豬苓湯"}
SHAOYIN_COLD_FORMULAS = {"四逆湯", "通脈四逆湯", "白通湯", "白通加豬膽汁湯",
                         "真武湯", "附子湯", "桃花湯", "吳茱萸湯"}
YANGMING_JING_FORMULAS = {"白虎湯", "白虎加人參湯"}
YANGMING_FU_FORMULAS = {"大承氣湯", "小承氣湯", "調胃承氣湯", "麻子仁丸"}


def _all_terms(d: Dict) -> List[str]:
    out: List[str] = []
    for v in d.values():
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend(x for x in v if isinstance(x, str))
    return out


def criticize(rule: InitialRule, clause_store: Dict[str, ShanghanClause]) -> Tuple[str, List[str]]:
    """Return (pass|warn|fail, flags)."""
    flags: List[str] = []
    clause = clause_store.get(rule.clause_id)
    if clause is None:
        return "fail", ["critic:no_clause"]
    text = fold_variants(clause.clean_text
                         + "\n".join(fb.raw_text for fb in clause.formula_blocks))

    # 1 —— posthoc vocabulary leaking into evidence-grounded fields
    grounded_terms = _all_terms(rule.if_conditions) + _all_terms(rule.then_conclusions)
    for t in grounded_terms:
        for ph in lexicon.POSTHOC_TERMS:
            if ph in t and ph not in text:
                flags.append(f"critic:posthoc_term_in_rule_body:{ph}")

    # interpretation may use them, but only with explicit non-literal level
    if rule.interpretation_level == "literal":
        for ph in lexicon.POSTHOC_TERMS:
            if ph in rule.interpretation and ph not in text:
                flags.append(f"critic:posthoc_term_as_literal:{ph}")

    # 2 —— ignored contraindications in the same clause
    if clause.contraindication_terms and rule.rule_type == "formula_pattern_rule":
        noted = (rule.then_conclusions.get("contraindicated_formulas") or []) + \
                (rule.if_conditions.get("contraindications") or [])
        formulas = rule.then_conclusions.get("formula") or []
        for term in clause.contraindication_terms:
            # the clause both prescribes AND forbids — the rule must not
            # present the formula unconditionally
            if "不可" in term and not noted and "禁" not in rule.interpretation \
               and "不可" not in rule.interpretation:
                flags.append(f"critic:contraindication_ignored:{term}")
                break

    # 3 —— 「可與」/「與」 inflated to 主之
    if rule.rule_type == "formula_pattern_rule":
        f_list = rule.then_conclusions.get("formula") or []
        if f_list:
            f = f_list[0]
            surfaces = [f] + [a for a, c in lexicon.FORMULA_ALIASES.items() if c == f]
            has_zhuzhi = any((s + "主之") in text for s in surfaces)
            if rule.prescription_strength == "主之" and not has_zhuzhi:
                flags.append("critic:keyu_inflated_to_zhuzhi")
            # 4 —— 主之 scope expansion: rule must not drop the clause's own
            # negated qualifiers (e.g. 反汗出/無汗 conditions)
            if has_zhuzhi:
                cond_neg = set(rule.if_conditions.get("negated_findings") or [])
                clause_neg = set(clause.negated_findings)
                missing = clause_neg - cond_neg
                cond_sym = set(rule.if_conditions.get("symptoms") or [])
                missing = {m for m in missing if m.lstrip("不") not in cond_sym}
                if missing and len(clause_neg) > 0 and not cond_neg:
                    flags.append("critic:zhuzhi_scope_expanded:" + ",".join(sorted(missing)))

    # 5 —— 太陽中風 vs 太陽傷寒 confusion
    cond_sym = set(rule.if_conditions.get("symptoms") or [])
    cond_dis = set(rule.if_conditions.get("disease") or [])
    f_set = set(rule.then_conclusions.get("formula") or [])
    if f_set & ZHONGFENG_FORMULAS and "無汗" in cond_sym and "桂枝" in "".join(f_set):
        if "汗出" not in text and "無汗" in text and "桂枝湯" in f_set:
            flags.append("critic:zhongfeng_shanghan_confusion:桂枝湯+無汗")
    if f_set & SHANGHAN_FORMULAS and ("汗出" in cond_sym or "自汗出" in cond_sym):
        if "麻黃湯" in f_set and "汗出" in cond_sym:
            flags.append("critic:zhongfeng_shanghan_confusion:麻黃湯+汗出")
    if "太陽中風" in cond_dis and "無汗" in cond_sym and "無汗" not in text:
        flags.append("critic:zhongfeng_with_wuhan_unattested")

    # 6 —— 少陰寒化 vs 熱化 confusion (label must follow the formula axis)
    interp = rule.interpretation or ""
    if f_set & SHAOYIN_HEAT_FORMULAS and ("寒化" in interp or "回陽" in interp):
        flags.append("critic:shaoyin_heat_mislabeled_cold")
    if f_set & SHAOYIN_COLD_FORMULAS and "熱化" in interp:
        flags.append("critic:shaoyin_cold_mislabeled_heat")

    # 7 —— 陽明經證 vs 腑證 confusion
    if f_set & YANGMING_JING_FORMULAS and ("腑實" in interp or "燥屎" in interp) \
       and "燥屎" not in text:
        flags.append("critic:yangming_jing_fu_confusion:白虎類標腑實")
    if f_set & YANGMING_FU_FORMULAS and "經證" in interp:
        flags.append("critic:yangming_jing_fu_confusion:承氣類標經證")

    # 8 —— invented symptoms (redundant with the evidence verifier, but the
    # critic re-checks negation traps: 不惡寒 must not ground 惡寒)
    for s in cond_sym:
        i = text.find(s)
        all_negated = True
        start = 0
        while True:
            i = text.find(s, start)
            if i < 0:
                break
            if i == 0 or text[i - 1] not in lexicon.NEGATION_PREFIX:
                all_negated = False
                break
            start = i + 1
        if i == -1 and start == 0:
            continue  # not found at all → evidence verifier's job
        if all_negated and text.find(s) >= 0:
            flags.append(f"critic:negated_symptom_recorded_positive:{s}")

    if not flags:
        return "pass", flags
    hard = [f for f in flags if any(k in f for k in (
        "posthoc_term_in_rule_body", "negated_symptom_recorded_positive",
        "keyu_inflated_to_zhuzhi", "no_clause"))]
    return ("fail" if hard else "warn"), flags
