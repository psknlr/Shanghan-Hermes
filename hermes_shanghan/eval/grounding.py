"""證據接地率基準 (Evidence-Grounding Benchmark).

2025-era studies report LLM citation-fabrication rates from 18% (GPT-4) to
94% under adversarial prompting. This suite measures the opposite property
for any backend plugged into this system: over a deterministic question bank
generated from the rule base itself, what fraction of agent answers are
fully grounded — every cited clause_id resolves, every quoted passage
verifies verbatim (the CitationGuard report the agent already attaches).

Metrics per backend: grounded-answer rate, unsupported-citation rate,
mean verified citations per answer, refusal/no-citation rate. Because the
question bank is derived from the rule inventory, the suite regenerates
deterministically with the corpus — no external annotation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas import FormulaPatternRule, SixChannelRule


def build_question_bank(formula_rules: List[FormulaPatternRule],
                        six_channel_rules: List[SixChannelRule],
                        n_formula: int = 12, n_diff: int = 6) -> List[Dict]:
    """Deterministic questions spanning the agent's tool surface."""
    qs: List[Dict] = []
    ranked = sorted(formula_rules, key=lambda r: -len(r.supporting_clauses))
    for r in ranked[:n_formula]:
        qs.append({"kind": "formula", "question": f"{r.formula}的方證要點與禁忌是什麼？"})
    for a, b in zip(ranked[:n_diff * 2:2], ranked[1:n_diff * 2:2]):
        qs.append({"kind": "differential",
                   "question": f"{a.formula}與{b.formula}如何鑒別？"})
    for scr in six_channel_rules:
        qs.append({"kind": "six_channel",
                   "question": f"{scr.six_channel}的提綱與主方是什麼？"})
    for r in ranked[:4]:
        sy = "、".join(r.core_symptoms[:3])
        if sy:
            qs.append({"kind": "match", "question": f"病人{sy}，考慮什麼方？"})
    return qs


class GroundingBenchmark:
    def __init__(self, agent=None):
        if agent is None:
            from ..agent.agent import ShanghanAgent
            agent = ShanghanAgent()
        self.agent = agent

    def run(self, questions: List[Dict], limit: Optional[int] = None,
            role: str = "doctor") -> Dict:
        if limit:
            questions = questions[:limit]
        records: List[Dict] = []
        for q in questions:
            out = self.agent.ask(q["question"], role=role)
            rep = out.get("citation_report") or {}
            records.append({
                "kind": q["kind"], "question": q["question"],
                "cited": len(rep.get("cited", [])),
                "verified": len(rep.get("verified", [])),
                "unsupported": len(rep.get("unsupported", [])),
                "quote_mismatches": len(rep.get("quote_mismatches", [])),
                "has_citation": bool(rep.get("has_any_citation")),
                "grounded": bool(rep.get("ok")) and bool(rep.get("has_any_citation")),
            })
        n = len(records)
        cited_total = sum(r["cited"] for r in records)
        metrics = {
            "n_questions": n,
            "grounded_answer_rate": round(sum(1 for r in records if r["grounded"]) / n, 4) if n else 0.0,
            "no_citation_rate": round(sum(1 for r in records if not r["has_citation"]) / n, 4) if n else 0.0,
            "unsupported_citation_rate": round(
                sum(r["unsupported"] for r in records) / cited_total, 4) if cited_total else 0.0,
            "mean_verified_per_answer": round(
                sum(r["verified"] for r in records) / n, 4) if n else 0.0,
        }
        return {"benchmark": "evidence_grounding",
                "backend": self.agent.client.backend,
                "metrics": metrics, "records": records}
