"""LLM-augmented rule extraction.

Produces candidate InitialRules from a clause via the LLM, then hands them to
the SAME autonomous review pipeline. The evidence verifier is the safety net:
an LLM-invented symptom or fabricated span is rejected or repaired exactly
like any other rule. This is how 「結合語言模型」 mining stays trustworthy —
the model widens recall, the deterministic gates protect precision.

With the `local` backend the candidates are rule-derived, so the plumbing is
exercised end-to-end offline; with a real model the same code path uses the
model's structured output.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..llm.client import LLMClient, get_client
from ..schemas import RULE_TYPES, AutonomousReview, InitialRule, ShanghanClause

# every clause-level rule type is fair game for the LLM (the evidence gates
# protect precision); variant/commentary rules are separate dataclasses built
# by B/C-layer alignment, not InitialRules
RULE_TYPES_ALLOWED = RULE_TYPES - {"variant_rule", "commentary_rule"}


class LLMRuleExtractor:
    def __init__(self, client: Optional[LLMClient] = None,
                 formula_names: Optional[List[str]] = None):
        self.client = client or get_client()
        self.formula_names = formula_names
        self._counter: Dict[str, int] = {}

    def _new_id(self, clause: ShanghanClause) -> str:
        n = self._counter.get(clause.clause_id, 0) + 1
        self._counter[clause.clause_id] = n
        stem = clause.clause_id.replace("SHL_SONGBEN_", "")
        return f"IR_SHL_{stem}_L{n:02d}"      # 'L' marks LLM provenance

    def extract_clause(self, clause: ShanghanClause) -> List[InitialRule]:
        data = self.client.extract_rules(clause, formula_names=self.formula_names)
        out: List[InitialRule] = []
        ev_type = "original_text" if clause.text_type == "original_clause" else "auxiliary_text"
        for raw in data.get("rules", []) or []:
            rtype = raw.get("rule_type", "")
            if rtype not in RULE_TYPES_ALLOWED:
                continue
            level = raw.get("interpretation_level", "normalized")
            if level not in ("literal", "normalized", "model_inference"):
                level = "normalized"
            try:
                conf = float(raw.get("model_confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            out.append(InitialRule(
                initial_rule_id=self._new_id(clause),
                clause_id=clause.clause_id,
                six_channel=clause.six_channel,
                rule_type=rtype,
                if_conditions=raw.get("if_conditions", {}) or {},
                then_conclusions=raw.get("then_conclusions", {}) or {},
                evidence_span=raw.get("evidence_span", "") or clause.clean_text,
                evidence_type=ev_type,
                interpretation=(raw.get("interpretation", "") or "") + "（來源：LLM 抽取）",
                interpretation_level=level,
                model_confidence=max(0.0, min(1.0, conf)),
                prescription_strength=raw.get("prescription_strength", "") or "",
                autonomous_review=AutonomousReview(),
            ))
        return out

    def extract_all(self, clauses: List[ShanghanClause],
                    limit: Optional[int] = None) -> List[InitialRule]:
        out: List[InitialRule] = []
        for c in clauses[:limit] if limit else clauses:
            out.extend(self.extract_clause(c))
        return out
