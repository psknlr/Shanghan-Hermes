"""LLM adversarial critic — an OPTIONAL extra review gate.

It runs after the deterministic ShanghanCritic and is advisory: it can flag
subtle semantic errors the regex critic misses, but it can never override the
hard evidence gate (a rule the evidence verifier rejects stays rejected; a
rule it accepts is at most *downgraded* by a hostile LLM verdict, not silently
promoted). Defaults to off so the deterministic pipeline stays reproducible.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..llm.client import LLMClient, get_client
from ..schemas import InitialRule, ShanghanClause

VALID_VERDICTS = {"pass", "warn", "fail"}


class LLMCritic:
    def __init__(self, client: LLMClient = None):
        self.client = client or get_client()

    def review(self, rule: InitialRule,
               clause_store: Dict[str, ShanghanClause]) -> Tuple[str, List[str], str]:
        clause = clause_store.get(rule.clause_id)
        if clause is None:
            return "fail", ["llm_critic:no_clause"], ""
        try:
            data = self.client.critic_review(clause, rule)
        except Exception as exc:
            return "pass", [f"llm_critic:error:{type(exc).__name__}"], ""
        verdict = data.get("verdict", "pass")
        if verdict not in VALID_VERDICTS:
            verdict = "warn"
        flags = [f"llm:{f}" for f in (data.get("flags") or []) if isinstance(f, str)][:8]
        rationale = (data.get("rationale") or "")[:300]
        return verdict, flags, rationale
