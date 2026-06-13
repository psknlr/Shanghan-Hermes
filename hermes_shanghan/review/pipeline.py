"""Autonomous review pipeline:

SchemaValidator → EvidenceVerifier → SemanticReviewer → ShanghanCritic
→ AutoRepair (one round) → re-verification → ConsensusJudge → ReleaseGate.

Outputs:
  data/shanghan/rules_initial/initial_rules.jsonl   (gold/silver/bronze)
  data/shanghan/rejected/rejected_rules.jsonl
  data/shanghan/audit/audit_log.jsonl
Critic statistics are persisted into critic_memory for self-improvement.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from .. import config
from ..schemas import AuditRecord, InitialRule, ShanghanClause, write_jsonl
from . import critic as critic_mod
from . import repair as repair_mod
from . import validators


class ReviewPipeline:
    def __init__(self, clause_store: Dict[str, ShanghanClause], llm_critic=None):
        self.clauses = clause_store
        self.audits: List[AuditRecord] = []
        self._audit_n = 0
        self.critic_counter: Counter = Counter()
        # Optional extra adversarial gate (advisory: can downgrade, never
        # promote past the hard evidence gate). Defaults off for determinism.
        self.llm_critic = llm_critic

    def _audit(self, rule_id: str, stage: str, result: str, flags: List[str], **details):
        self._audit_n += 1
        self.audits.append(AuditRecord(
            audit_id=f"AUD_{self._audit_n:06d}", target_id=rule_id,
            target_kind="initial_rule", stage=stage, result=result,
            flags=flags, details=details or {}))

    # ------------------------------------------------------------------
    def review_rule(self, rule: InitialRule) -> InitialRule:
        ar = rule.autonomous_review

        # Stage 1 — schema
        ok, flags = validators.validate_schema(rule)
        ar.schema_valid = ok
        self._audit(rule.initial_rule_id, "schema", "pass" if ok else "fail", flags)
        if not ok:
            ar.release_level = "rejected"
            ar.consensus_score = 0.0
            return rule

        # Stage 2 — evidence
        ok, ev_flags = validators.verify_evidence(rule, self.clauses)
        ar.evidence_verified = ok
        self._audit(rule.initial_rule_id, "evidence", "pass" if ok else "fail", ev_flags)

        # Stage 3 — semantics
        sem_result, sem_flags = validators.review_semantics(rule, self.clauses)
        ar.semantic_result = sem_result
        self._audit(rule.initial_rule_id, "semantic", sem_result, sem_flags)

        # Stage 4 — adversarial critic
        crit_result, crit_flags = critic_mod.criticize(rule, self.clauses)
        ar.critic_result = crit_result
        ar.critic_flags = crit_flags
        for f in crit_flags:
            self.critic_counter[f.split(":")[1] if ":" in f else f] += 1
        self._audit(rule.initial_rule_id, "critic", crit_result, crit_flags)

        # Stage 5 — automatic repair (single round) + re-verification
        all_flags = ev_flags + sem_flags + crit_flags
        repaired = False
        if all_flags:
            rule, applied = repair_mod.repair(rule, all_flags, self.clauses)
            if applied:
                repaired = True
                ar.repairs = applied
                self._audit(rule.initial_rule_id, "repair", "repaired", applied)
                ok, ev_flags = validators.verify_evidence(rule, self.clauses)
                ar.evidence_verified = ok
                sem_result, sem_flags = validators.review_semantics(rule, self.clauses)
                ar.semantic_result = sem_result
                crit_result, crit_flags = critic_mod.criticize(rule, self.clauses)
                ar.critic_result = crit_result
                ar.critic_flags = crit_flags
                self._audit(rule.initial_rule_id, "reverify",
                            "pass" if (ok and crit_result != "fail") else "fail",
                            ev_flags + sem_flags + crit_flags)

        # Stage 4b — optional LLM adversarial critic (advisory downgrade only)
        if self.llm_critic is not None and ar.evidence_verified and crit_result != "fail":
            try:
                llm_verdict, llm_flags, rationale = self.llm_critic.review(rule, self.clauses)
            except Exception as exc:
                llm_verdict, llm_flags, rationale = "pass", [f"llm_critic:error:{type(exc).__name__}"], ""
            if llm_flags:
                ar.critic_flags = list(ar.critic_flags) + llm_flags
            for f in llm_flags:
                self.critic_counter[f.split(":", 1)[1] if ":" in f else f] += 1
            # hostile LLM verdict downgrades, but cannot reject past evidence gate
            if llm_verdict == "fail" and crit_result == "pass":
                crit_result = "warn"
            elif llm_verdict == "warn" and crit_result == "pass":
                crit_result = "warn"
            ar.critic_result = crit_result
            self._audit(rule.initial_rule_id, "llm_critic", llm_verdict, llm_flags,
                        rationale=rationale)

        # Stage 6 — consensus + release gate
        score = self._consensus(rule, ar.evidence_verified, sem_result, crit_result, repaired)
        ar.consensus_score = round(score, 3)
        ar.release_level = self._release_gate(score, ar.evidence_verified, crit_result)
        self._audit(rule.initial_rule_id, "release", ar.release_level, [],
                    consensus_score=ar.consensus_score)
        return rule

    # ------------------------------------------------------------------
    @staticmethod
    def _consensus(rule: InitialRule, evidence_ok: bool, sem: str, crit: str,
                   repaired: bool) -> float:
        score = 0.0
        if evidence_ok:
            score += 0.50
        if sem == "pass":
            score += 0.12
        elif sem == "warn":
            score += 0.06
        if crit == "pass":
            score += 0.16
        elif crit == "warn":
            score += 0.08
        # evidence strength: 主之 and literal structural rules are firmest
        literal_types = {"formula_composition_rule", "administration_rule",
                         "dosage_processing_rule", "six_channel_definition_rule"}
        if rule.prescription_strength == "主之" or rule.rule_type in literal_types:
            score += 0.08
        elif rule.prescription_strength in ("宜", "屬"):
            score += 0.05
        elif rule.prescription_strength in ("與", "可與"):
            score += 0.02
        elif rule.rule_type in ("contraindication_rule", "disease_pattern_rule"):
            score += 0.06
        # condition richness: a rule that binds explicit 脈/證 conditions is
        # stronger than one with a bare conclusion
        cond_n = sum(len(rule.if_conditions.get(k) or [])
                     for k in ("symptoms", "pulse", "disease", "negated_findings",
                               "mistreatment", "formula"))
        score += min(0.06, 0.02 * cond_n)
        score += 0.04 * min(1.0, float(rule.model_confidence))
        if rule.evidence_type == "auxiliary_text":
            score -= 0.04
        if repaired:
            score = min(score, 0.92)   # repaired rules carry a small ceiling
        return max(0.0, min(score, 0.98))

    @staticmethod
    def _release_gate(score: float, evidence_ok: bool, crit: str) -> str:
        if not evidence_ok or crit == "fail":
            return "rejected"
        if score >= config.RELEASE_GOLD:
            return "gold"
        if score >= config.RELEASE_SILVER:
            return "silver"
        if score >= config.RELEASE_BRONZE:
            return "bronze"
        return "rejected"

    # ------------------------------------------------------------------
    def run(self, rules: List[InitialRule]) -> Tuple[List[InitialRule], List[InitialRule]]:
        accepted, rejected = [], []
        for r in rules:
            r = self.review_rule(r)
            (accepted if r.autonomous_review.release_level != "rejected" else rejected).append(r)
        return accepted, rejected

    def persist(self, accepted: List[InitialRule], rejected: List[InitialRule]) -> Dict[str, int]:
        config.ensure_dirs()
        n_acc = write_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl", accepted)
        n_rej = write_jsonl(config.REJECTED_DIR / "rejected_rules.jsonl", rejected)
        n_aud = write_jsonl(config.AUDIT_DIR / "audit_log.jsonl", self.audits)
        return {"accepted": n_acc, "rejected": n_rej, "audits": n_aud}
