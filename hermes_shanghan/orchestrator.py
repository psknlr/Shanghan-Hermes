"""Orchestrator — runs the five Hermes-Shanghanlun workflows end to end.

Workflow 1  條文級規則挖掘   ingest → segment → extract → autonomous review
Workflow 2  六經體系構建     SixChannelRules + channel skills
Workflow 3  方證體系構建     FormulaPatternRules + formula skills
Workflow 4  誤治傳變圖譜     MistreatmentTransformationRules
Workflow 5  方證鑒別         DifferentialRules
…then ClauseRelations / variants / commentaries, MergedRules, skill
compilation, research assets and memory updates.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from . import config
from .corpus import downloader, segmenter
from .extract.entities import EntityExtractor, annotate_clause
from .extract.initial_rules import InitialRuleExtractor
from .induce.differential import DifferentialInducer
from .induce.formula_patterns import FormulaPatternInducer
from .induce.merged import MergedRuleBuilder
from .induce.mistreatment import MistreatmentInducer
from .induce.relations import RelationBuilder
from .induce.six_channels import SixChannelInducer
from .induce.therapy import TherapyInducer
from .memory.store import MemoryHub
from .review.pipeline import ReviewPipeline
from .schemas import (CommentaryRule, DifferentialRule, FormulaPatternRule,
                      InitialRule, MergedShanghanRule,
                      MistreatmentTransformationRule, ShanghanClause,
                      SixChannelRule, TherapyRule, VariantRule, read_jsonl,
                      write_jsonl)
from .skills.builder import SkillBuilder


class Artifacts:
    """Lazy loader for persisted pipeline artifacts."""

    def __init__(self):
        self._cache: Dict[str, object] = {}

    def _load(self, key, path, cls):
        if key not in self._cache:
            self._cache[key] = [cls.from_dict(d) for d in read_jsonl(path)]
        return self._cache[key]

    @property
    def clauses(self) -> List[ShanghanClause]:
        return self._load("clauses", config.CLAUSE_DIR / "clauses.jsonl", ShanghanClause)

    @property
    def initial_rules(self) -> List[InitialRule]:
        return self._load("initial", config.RULES_INITIAL_DIR / "initial_rules.jsonl", InitialRule)

    @property
    def formula_rules(self) -> List[FormulaPatternRule]:
        return self._load("formula", config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl", FormulaPatternRule)

    @property
    def six_channel_rules(self) -> List[SixChannelRule]:
        return self._load("scr", config.RULES_SIX_CHANNEL_DIR / "six_channel_rules.jsonl", SixChannelRule)

    @property
    def therapy_rules(self) -> List[TherapyRule]:
        return self._load("therapy", config.RULES_THERAPY_DIR / "therapy_rules.jsonl", TherapyRule)

    @property
    def mistreatment_rules(self) -> List[MistreatmentTransformationRule]:
        return self._load("mtr", config.RULES_MISTREATMENT_DIR / "mistreatment_rules.jsonl", MistreatmentTransformationRule)

    @property
    def differential_rules(self) -> List[DifferentialRule]:
        return self._load("diff", config.RULES_DIFFERENTIAL_DIR / "differential_rules.jsonl", DifferentialRule)

    @property
    def merged_rules(self) -> List[MergedShanghanRule]:
        return self._load("merged", config.RULES_MERGED_DIR / "merged_rules.jsonl", MergedShanghanRule)

    @property
    def variant_rules(self) -> List[VariantRule]:
        return self._load("variant", config.RULES_VARIANT_DIR / "variant_rules.jsonl", VariantRule)

    @property
    def commentary_rules(self) -> List[CommentaryRule]:
        return self._load("commentary", config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl", CommentaryRule)

    def clause_store(self) -> Dict[str, ShanghanClause]:
        return {c.clause_id: c for c in self.clauses}


def run_pipeline(verbose: bool = True) -> Dict:
    """Run everything. Returns a stats summary."""
    t0 = time.time()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    config.ensure_dirs()
    stats: Dict = {}
    memory = MemoryHub()

    # ---- Workflow 1: corpus → clauses → initial rules → review ----------
    log("[1/8] 語料導入與版本 manifest …")
    downloader.run()

    log("[2/8] 條文切分（條文版 398 條 + 宋本輔助篇章）…")
    canonical = segmenter.segment_canonical()
    aux = segmenter.segment_auxiliary()
    clauses = canonical + aux
    formula_names = segmenter.harvest_formula_names(canonical)
    extractor = EntityExtractor(formula_names)
    for c in clauses:
        annotate_clause(c, extractor)
    write_jsonl(config.CLAUSE_DIR / "clauses.jsonl", clauses)
    stats["clauses_canonical"] = len(canonical)
    stats["clauses_auxiliary"] = len(aux)
    stats["formulas_harvested"] = len(formula_names)
    for c in canonical:
        memory.clause_memory.update(c.clause_id, chapter=c.chapter,
                                    six_channel=c.six_channel,
                                    entities_extracted=True,
                                    n_symptoms=len(c.symptoms),
                                    n_formulas=len(c.formula_names))

    log("[3/8] 條文級 InitialRule 抽取 …")
    rule_extractor = InitialRuleExtractor(extractor)
    raw_rules = rule_extractor.extract_all(clauses)
    stats["initial_rules_raw"] = len(raw_rules)

    log("[4/8] 自主審核（Schema→證據回源→語義→批評→修復→共識分級）…")
    store = {c.clause_id: c for c in clauses}
    pipeline = ReviewPipeline(store)
    accepted, rejected = pipeline.run(raw_rules)
    counts = pipeline.persist(accepted, rejected)
    stats.update({"initial_rules_accepted": counts["accepted"],
                  "initial_rules_rejected": counts["rejected"],
                  "audit_records": counts["audits"]})
    for flag, n in pipeline.critic_counter.most_common():
        memory.critic_memory.update(flag, count=n)
    memory.critic_memory.set("_note", "高頻錯誤模式用於下一輪抽取的先驗約束，"
                                      "例如「營衛不和」類後世術語禁止落入規則主體。")

    # ---- relations / variants / commentaries ----------------------------
    log("[5/8] 條文關係圖譜 + 異文/注釋對齊 …")
    rel_builder = RelationBuilder(clauses)
    rel_stats = rel_builder.run()
    stats.update(rel_stats)

    # ---- Workflows 2–5: induction ---------------------------------------
    log("[6/8] 方證/六經/治法/誤治/鑒別規則歸納 …")
    formula_rules = FormulaPatternInducer(clauses, accepted).run()
    six_rules = SixChannelInducer(clauses, accepted).run()
    therapy_rules = TherapyInducer(clauses, accepted).run()
    mistreatment_rules = MistreatmentInducer(clauses, accepted).run()
    differential_rules = DifferentialInducer(formula_rules).run()
    stats.update({
        "formula_pattern_rules": len(formula_rules),
        "six_channel_rules": len(six_rules),
        "therapy_rules": len(therapy_rules),
        "mistreatment_rules": len(mistreatment_rules),
        "differential_rules": len(differential_rules),
    })
    for f in formula_rules:
        memory.formula_memory.update(
            f.formula, family=f.formula_family,
            n_clauses=len(f.supporting_clauses),
            composition=[c["herb"] for c in f.composition],
            modifications=[m["modified_formula"] for m in f.modification_relations])
    for r in six_rules:
        memory.six_channel_memory.update(
            r.six_channel, outline=r.outline_clause_id,
            subtypes=[s["name"] for s in r.subtypes],
            main_formulas=[f["formula"] for f in r.main_formulas[:6]])
    for m in mistreatment_rules:
        memory.mistreatment_memory.update(
            m.mistreatment_rule_id, type=m.mistreatment_type,
            pattern=m.resulting_pattern, rescue=m.rescue_formulas)

    # ---- merged rules -----------------------------------------------------
    artifacts_variant = [VariantRule.from_dict(d) for d in
                         read_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl")]
    artifacts_comment = [CommentaryRule.from_dict(d) for d in
                         read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")]
    log("[7/8] MergedShanghanRule 合併（不覆蓋初始規則）…")
    merged = MergedRuleBuilder(clauses, accepted, formula_rules, six_rules,
                               therapy_rules, mistreatment_rules,
                               artifacts_variant, artifacts_comment).run()
    stats["merged_rules"] = len(merged)

    # ---- skills + research assets ----------------------------------------
    log("[8/8] Skill 編譯 + 科研資產 …")
    builder = SkillBuilder(clauses, accepted, formula_rules, six_rules,
                           therapy_rules, mistreatment_rules,
                           differential_rules, merged, artifacts_variant,
                           artifacts_comment)
    skill_counts = builder.build_all()
    stats["skills"] = skill_counts
    for name, n in skill_counts.items():
        memory.skill_memory.update(name, built=n,
                                   built_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    from .apps.research import ResearchMiner
    miner = ResearchMiner(clauses, formula_rules, mistreatment_rules)
    miner.run_topic("全書方證體系", outputs=["network"])

    memory.paper_memory.set("last_pipeline_stats", {
        k: v for k, v in stats.items() if isinstance(v, (int, str))})
    memory.save_all()
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    log(f"✅ pipeline 完成，用時 {stats['elapsed_sec']}s")
    return stats
