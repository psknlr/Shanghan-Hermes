"""MistreatmentTransformationAgent — 誤治傳變圖譜.

Builds (誤治方式 → 變證 → 症狀表現 → 救治方 → 原文證據) paths in two passes:
  1. auto-detection from clauses carrying mistreatment markers, adverse
     outcomes and a rescue formula in the same clause;
  2. canonical path templates (誤下→痞→瀉心類 etc.) verified against the
     detected evidence so every path keeps its clause IDs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .. import config, lexicon
from ..schemas import (InitialRule, MistreatmentTransformationRule,
                       ShanghanClause, write_jsonl)

# canonical path templates: (誤治, 變證模式, anchor formulas)
PATH_TEMPLATES: List[Tuple[str, str, List[str]]] = [
    ("誤下", "結胸", ["大陷胸湯", "大陷胸丸", "小陷胸湯"]),
    ("誤下", "痞證", ["半夏瀉心湯", "生薑瀉心湯", "甘草瀉心湯", "大黃黃連瀉心湯", "附子瀉心湯"]),
    ("誤下", "協熱利", ["葛根黃芩黃連湯", "桂枝人參湯"]),
    ("誤下", "虛煩", ["梔子豉湯", "梔子甘草豉湯", "梔子生薑豉湯"]),
    ("誤下", "氣上衝", ["桂枝湯"]),
    ("誤汗", "亡陽漏汗", ["桂枝加附子湯"]),
    ("誤汗", "亡陽厥逆", ["四逆湯", "乾薑附子湯", "茯苓四逆湯"]),
    ("誤汗", "心陽虛悸", ["桂枝甘草湯"]),
    ("誤汗", "奔豚/欲作奔豚", ["桂枝加桂湯", "茯苓桂枝甘草大棗湯"]),
    ("誤汗", "陽虛水泛", ["真武湯", "茯苓桂枝白朮甘草湯"]),
    ("誤汗", "胃中乾煩躁", ["五苓散"]),
    ("誤吐", "虛煩", ["梔子豉湯"]),
    ("火逆", "驚狂", ["桂枝去芍藥加蜀漆牡蠣龍骨救逆湯"]),
    ("火逆", "煩躁", ["桂枝甘草龍骨牡蠣湯"]),
    ("火逆", "奔豚", ["桂枝加桂湯"]),
]


class MistreatmentInducer:
    def __init__(self, clauses: List[ShanghanClause], initial_rules: List[InitialRule]):
        self.clauses = [c for c in clauses if c.text_type == "original_clause"]
        self.rules = [r for r in initial_rules
                      if r.autonomous_review.release_level != "rejected"
                      and r.rule_type == "mistreatment_rule"]
        self.clause_store = {c.clause_id: c for c in clauses}

    def induce(self) -> List[MistreatmentTransformationRule]:
        out: List[MistreatmentTransformationRule] = []
        n = 0

        # Index detected mistreatment evidence by (type, formula)
        evidence: Dict[Tuple[str, str], List[InitialRule]] = defaultdict(list)
        for r in self.rules:
            mtypes = r.if_conditions.get("mistreatment_type", [])
            formulas = r.then_conclusions.get("rescue_formula", [])
            for mt in mtypes:
                for f in formulas or [""]:
                    evidence[(mt, f)].append(r)

        # Pass 1 — canonical templates grounded in detected evidence
        for mtype, pattern, formulas in PATH_TEMPLATES:
            support_rules: List[InitialRule] = []
            hit_formulas: List[str] = []
            for f in formulas:
                rs = evidence.get((mtype, f), [])
                if rs:
                    support_rules.extend(rs)
                    hit_formulas.append(f)
            # also accept clauses where formula present + pattern keyword
            kw = pattern.split("/")[0].replace("證", "")
            extra_clauses = [c.clause_id for c in self.clauses
                             if any(f in c.formula_names for f in formulas)
                             and (kw in c.clean_text or not kw)]
            if not support_rules and not extra_clauses:
                continue
            n += 1
            manifestations: List[str] = []
            channels: List[str] = []
            clause_ids = sorted({r.clause_id for r in support_rules} | set(extra_clauses))
            for cid in clause_ids:
                cl = self.clause_store.get(cid)
                if cl:
                    manifestations.extend(cl.symptoms[:3])
                    if cl.six_channel and cl.six_channel not in channels:
                        channels.append(cl.six_channel)
            seen = set()
            manifestations = [m for m in manifestations if not (m in seen or seen.add(m))][:10]
            out.append(MistreatmentTransformationRule(
                mistreatment_rule_id=f"MTR_{n:03d}",
                mistreatment_type=mtype,
                resulting_pattern=pattern,
                manifestations=manifestations,
                rescue_formulas=hit_formulas or formulas,
                six_channel_scope=channels,
                path=[mtype, pattern, "、".join(hit_formulas or formulas)],
                supporting_clauses=clause_ids[:15],
                supporting_initial_rules=[r.initial_rule_id for r in support_rules][:15],
                source_level="inductive_from_original_clauses",
                consensus_score=0.88 if support_rules else 0.78,
                release_level="gold" if len(support_rules) >= 2 else
                              ("silver" if support_rules or extra_clauses else "bronze"),
            ))

        # Pass 2 — auto paths not covered by templates
        covered = {(r.mistreatment_type, f) for r in out for f in r.rescue_formulas}
        for (mtype, formula), rs in sorted(evidence.items()):
            if not formula or (mtype, formula) in covered:
                continue
            outcomes: List[str] = []
            for r in rs:
                outcomes.extend(r.then_conclusions.get("adverse_outcomes", []))
            seen = set()
            outcomes = [o for o in outcomes if not (o in seen or seen.add(o))]
            n += 1
            clause_ids = sorted({r.clause_id for r in rs})
            channels = sorted({r.six_channel for r in rs if r.six_channel})
            out.append(MistreatmentTransformationRule(
                mistreatment_rule_id=f"MTR_{n:03d}",
                mistreatment_type=mtype,
                resulting_pattern="、".join(outcomes[:3]) or "變證",
                manifestations=outcomes[:10],
                rescue_formulas=[formula],
                six_channel_scope=channels,
                path=[mtype, "、".join(outcomes[:2]) or "變證", formula],
                supporting_clauses=clause_ids,
                supporting_initial_rules=[r.initial_rule_id for r in rs],
                source_level="inductive_from_original_clauses",
                consensus_score=0.82,
                release_level="silver" if len(rs) >= 2 else "bronze",
            ))
        return out

    def run(self) -> List[MistreatmentTransformationRule]:
        rules = self.induce()
        config.ensure_dirs()
        write_jsonl(config.RULES_MISTREATMENT_DIR / "mistreatment_rules.jsonl", rules)
        return rules
