"""遮方預測基準 (Prescription-Cloze Benchmark) — leave-one-clause-out.

Ithaca/Aeneas (Nature 2022/2025) evaluate ancient-text models by masking
characters and restoring them. Here the masked unit is the CLINICAL DECISION:
for every prescription-bearing canonical clause, the clause's presented
findings (症狀/脈象, taken from its branch segment) are the query, the
prescribed formula is the gold label, and the matcher must answer WITHOUT
that clause — every initial rule mined from it is removed and the
FormulaPatternRule layer is re-induced per fold (strict LOCO: no leakage of
the gold clause into the knowledge base it is scored against).

The benchmark is fully self-supervised (zero annotation cost), deterministic
and reproducible from the corpus alone. Metrics follow the TCM formula-
recommendation literature (Top-K hit, MRR, herb-level P/R/F1) plus an
honesty split: folds whose gold formula survives LOCO ("attainable") vs
singleton formulas attested by that one clause only (impossible by
construction, reported separately, never silently dropped).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..apps.doctor import FormulaMatcher
from ..induce.formula_patterns import FormulaPatternInducer
from ..schemas import FormulaPatternRule, InitialRule, ShanghanClause

TOP_KS = (1, 3, 5)


def build_instances(initial_rules: List[InitialRule],
                    min_findings: int = 2):
    """One instance per prescription-bearing formula_pattern rule.

    Returns (instances, n_skipped_low_findings) — under-specified clauses
    (fewer than min_findings presented findings) are counted, not hidden.
    """
    instances: List[Dict] = []
    skipped = 0
    for r in initial_rules:
        if r.rule_type != "formula_pattern_rule":
            continue
        if r.autonomous_review.release_level == "rejected":
            continue
        gold = (r.then_conclusions.get("formula") or [""])[0]
        symptoms = list(r.if_conditions.get("symptoms") or [])
        pulse = list(r.if_conditions.get("pulse") or [])
        if not gold:
            continue
        if len(symptoms) + len(pulse) < min_findings:
            skipped += 1
            continue
        instances.append({"clause_id": r.clause_id, "gold": gold,
                          "symptoms": symptoms, "pulse": pulse,
                          "strength": r.prescription_strength})
    return instances, skipped


class ClozeBenchmark:
    def __init__(self, clauses: List[ShanghanClause],
                 initial_rules: List[InitialRule],
                 use_outline_boost: bool = True,
                 use_near_match: bool = True):
        self.clauses = clauses
        self.clause_store = {c.clause_id: c for c in clauses}
        self.initial_rules = initial_rules
        self.use_outline_boost = use_outline_boost
        self.use_near_match = use_near_match
        # full-corpus compositions for herb-level scoring
        full_rules = FormulaPatternInducer(clauses, initial_rules).induce()
        self.composition = {r.formula: {c["herb"] for c in r.composition}
                            for r in full_rules if r.composition}

    # ------------------------------------------------------------------
    def _loco_rules(self, held_out_clause: str) -> List[FormulaPatternRule]:
        kept = [r for r in self.initial_rules if r.clause_id != held_out_clause]
        return FormulaPatternInducer(self.clauses, kept).induce()

    def _rank_of(self, matches: List[Dict], gold: str) -> Optional[int]:
        for i, m in enumerate(matches, 1):
            if m["formula"] == gold:
                return i
        return None

    def _herb_prf(self, predicted: str, gold: str) -> Optional[Dict]:
        p, g = self.composition.get(predicted), self.composition.get(gold)
        if not p or not g:
            return None
        tp = len(p & g)
        prec = tp / len(p)
        rec = tp / len(g)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f1": f1}

    # ------------------------------------------------------------------
    def run(self, limit: Optional[int] = None) -> Dict:
        instances, skipped = build_instances(self.initial_rules)
        if limit:
            instances = instances[:limit]

        records: List[Dict] = []
        loco_cache: Dict[str, List[FormulaPatternRule]] = {}
        for inst in instances:
            cid = inst["clause_id"]
            if cid not in loco_cache:
                loco_cache[cid] = self._loco_rules(cid)
            fold_rules = loco_cache[cid]
            attainable = any(r.formula == inst["gold"] for r in fold_rules)
            matcher = FormulaMatcher(fold_rules, self.clause_store,
                                     use_outline_boost=self.use_outline_boost,
                                     use_near_match=self.use_near_match)
            out = matcher.match(inst["symptoms"], pulse=inst["pulse"],
                                top_k=max(TOP_KS), need_original_evidence=False)
            matches = out["matched_formula_patterns"]
            rank = self._rank_of(matches, inst["gold"])
            top1 = matches[0]["formula"] if matches else ""
            records.append({
                "clause_id": cid, "gold": inst["gold"],
                "strength": inst["strength"], "attainable": attainable,
                "rank": rank, "top1": top1,
                "herb_prf": self._herb_prf(top1, inst["gold"]) if top1 else None,
            })

        return {"benchmark": "prescription_cloze_loco",
                "config": {"use_outline_boost": self.use_outline_boost,
                           "use_near_match": self.use_near_match},
                "n_instances": len(records),
                "n_skipped_low_findings": skipped,
                "metrics": summarize(records),
                "records": records}


def summarize(records: List[Dict]) -> Dict:
    def _metrics(rs: List[Dict]) -> Dict:
        n = len(rs)
        if not n:
            return {"n": 0}
        out: Dict = {"n": n}
        for k in TOP_KS:
            out[f"top{k}"] = round(sum(1 for r in rs
                                       if r["rank"] and r["rank"] <= k) / n, 4)
        out["mrr"] = round(sum(1.0 / r["rank"] for r in rs if r["rank"]) / n, 4)
        prfs = [r["herb_prf"] for r in rs if r.get("herb_prf")]
        if prfs:
            out["herb_precision"] = round(sum(p["precision"] for p in prfs) / len(prfs), 4)
            out["herb_recall"] = round(sum(p["recall"] for p in prfs) / len(prfs), 4)
            out["herb_f1"] = round(sum(p["f1"] for p in prfs) / len(prfs), 4)
        return out

    attainable = [r for r in records if r["attainable"]]
    singleton = [r for r in records if not r["attainable"]]
    zhuzhi = [r for r in attainable if r["strength"] == "主之"]
    return {"all": _metrics(records),
            "attainable": _metrics(attainable),
            "singleton_unattainable": _metrics(singleton),
            "attainable_zhuzhi_only": _metrics(zhuzhi)}
