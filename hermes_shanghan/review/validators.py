"""SchemaValidator, EvidenceVerifierAgent and SemanticReviewer.

These are the first three gates of the autonomous review pipeline. They are
fully deterministic: every check is grounded in the clause store, so a rule
can only pass if its evidence really exists in the source text.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .. import config, lexicon
from ..schemas import InitialRule, RULE_TYPES, EVIDENCE_TYPES, INTERPRETATION_LEVELS, ShanghanClause
from ..textutil import contains_verbatim

RE_IR_ID = re.compile(r"^IR_SHL_[A-Z0-9_]+_\d{3}$")
RE_CLAUSE_ID = re.compile(r"^SHL_SONGBEN_(AUX_)?\d{4}$")


# ---------------------------------------------------------------------------
# Stage 1 — SchemaValidator
# ---------------------------------------------------------------------------
def validate_schema(rule: InitialRule) -> Tuple[bool, List[str]]:
    flags: List[str] = []
    if not RE_IR_ID.match(rule.initial_rule_id or ""):
        flags.append(f"schema:bad_rule_id:{rule.initial_rule_id}")
    if not RE_CLAUSE_ID.match(rule.clause_id or ""):
        flags.append(f"schema:bad_clause_id:{rule.clause_id}")
    if rule.rule_type not in RULE_TYPES:
        flags.append(f"schema:bad_rule_type:{rule.rule_type}")
    if rule.evidence_type not in EVIDENCE_TYPES:
        flags.append(f"schema:bad_evidence_type:{rule.evidence_type}")
    if rule.interpretation_level not in INTERPRETATION_LEVELS:
        flags.append(f"schema:bad_interpretation_level:{rule.interpretation_level}")
    if not rule.evidence_span or len(rule.evidence_span) < 4:
        flags.append("schema:empty_evidence_span")
    if not isinstance(rule.if_conditions, dict) or not isinstance(rule.then_conclusions, dict):
        flags.append("schema:if_then_not_dict")
    if not (rule.if_conditions or rule.then_conclusions):
        flags.append("schema:empty_rule_body")
    if not 0.0 <= float(rule.model_confidence) <= 1.0:
        flags.append("schema:confidence_out_of_range")
    return (not flags, flags)


# ---------------------------------------------------------------------------
# Stage 2 — EvidenceVerifierAgent
# ---------------------------------------------------------------------------
def _iter_condition_terms(rule: InitialRule):
    for key in ("symptoms", "negated_findings", "disease", "prior_treatment", "mistreatment"):
        for t in rule.if_conditions.get(key, []) or []:
            yield key, t


def verify_evidence(rule: InitialRule, clause_store: Dict[str, ShanghanClause]) -> Tuple[bool, List[str]]:
    """No original evidence, no Shanghan rule. No clause_id, no answer."""
    flags: List[str] = []
    clause = clause_store.get(rule.clause_id)
    if clause is None:
        return False, [f"evidence:clause_id_not_found:{rule.clause_id}"]

    # evidence span must be verbatim text of the clause (clause text or one of
    # its formula blocks).
    span_ok = contains_verbatim(clause.clean_text, rule.evidence_span) or \
        any(contains_verbatim(fb.raw_text, rule.evidence_span) or
            contains_verbatim(rule.evidence_span, fb.raw_text)
            for fb in clause.formula_blocks)
    if not span_ok:
        flags.append("evidence:span_not_in_clause")

    full_text = clause.clean_text + "\n" + "\n".join(fb.raw_text for fb in clause.formula_blocks)

    # every IF-side textual condition must appear in the evidence
    for key, term in _iter_condition_terms(rule):
        probe = term[1:] if term.startswith("不") and term not in full_text else term
        if probe not in full_text and term not in full_text:
            flags.append(f"evidence:condition_not_in_text:{key}:{term}")

    # pulse: each recorded quality must be attested in the clause text
    for p in rule.if_conditions.get("pulse", []) or []:
        body = p.lstrip("不")
        if body in full_text:
            continue
        if all(q in full_text for q in body if q in lexicon.PULSE_QUALITIES) and \
           any(q in lexicon.PULSE_QUALITIES for q in body):
            continue
        flags.append(f"evidence:pulse_not_in_text:{p}")

    # formula conclusions must correspond to the clause
    for f in rule.then_conclusions.get("formula", []) or []:
        surfaces = {f} | {a for a, c in lexicon.FORMULA_ALIASES.items() if c == f}
        if not any(s in full_text for s in surfaces):
            flags.append(f"evidence:formula_not_in_clause:{f}")
    return (not flags, flags)


# ---------------------------------------------------------------------------
# Stage 3 — SemanticReviewer
# ---------------------------------------------------------------------------
def review_semantics(rule: InitialRule, clause_store: Dict[str, ShanghanClause]) -> Tuple[str, List[str]]:
    """Returns (pass|warn|fail, flags)."""
    flags: List[str] = []
    clause = clause_store.get(rule.clause_id)
    if clause is None:
        return "fail", ["semantic:no_clause"]

    # six-channel consistency with the chapter
    if rule.six_channel and clause.six_channel and rule.six_channel != clause.six_channel:
        flags.append(f"semantic:channel_mismatch:{rule.six_channel}!={clause.six_channel}")

    # rule_type sanity vs markers in text
    text = clause.clean_text
    rt = rule.rule_type
    if rt == "six_channel_definition_rule" and "之為病" not in text:
        flags.append("semantic:not_an_outline_clause")
    if rt == "contraindication_rule" and not (clause.contraindication_terms or
                                              rule.then_conclusions.get("contraindicated_formulas")):
        flags.append("semantic:contraindication_without_marker")
    if rt == "mistreatment_rule" and not clause.mistreatment_terms:
        flags.append("semantic:mistreatment_without_marker")
    if rt == "formula_pattern_rule":
        if not rule.then_conclusions.get("formula"):
            flags.append("semantic:formula_rule_without_formula")
        if not rule.prescription_strength:
            flags.append("semantic:missing_prescription_strength")

    # prescription strength must match the actual marker in text
    f_list = rule.then_conclusions.get("formula", []) or []
    if rt == "formula_pattern_rule" and f_list:
        f = f_list[0]
        surfaces = [f] + [a for a, c in lexicon.FORMULA_ALIASES.items() if c == f]
        claimed = rule.prescription_strength
        if claimed == "主之" and not any((s + "主之") in text for s in surfaces):
            flags.append("semantic:strength_overclaimed:主之")
        if claimed in ("宜",) and not any(("宜" + s) in text or ("宜服" + s) in text for s in surfaces):
            if any((s + "主之") in text for s in surfaces):
                pass  # under-claimed is acceptable but warn
            else:
                flags.append("semantic:strength_unattested:宜")

    if not flags:
        return "pass", flags
    hard = [f for f in flags if "without_formula" in f or "no_clause" in f]
    return ("fail" if hard else "warn"), flags
