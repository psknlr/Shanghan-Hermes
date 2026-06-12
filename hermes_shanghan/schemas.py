"""Core data objects of Hermes-Shanghanlun.

Implements the protocol's data model:
ShanghanClause, FormulaBlock, InitialRule, ClauseRelation, FormulaPatternRule,
SixChannelRule, TherapyRule, MistreatmentTransformationRule, DifferentialRule,
VariantRule, CommentaryRule, MergedShanghanRule, AuditRecord.

Everything serializes to/from plain JSON dicts (jsonl on disk).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Enumerations (kept as plain strings + validation sets for JSON friendliness)
# ---------------------------------------------------------------------------
TEXT_TYPES = {
    "original_clause",      # numbered canonical clause (layer A)
    "auxiliary_clause",     # Songben auxiliary chapters (脈法/傷寒例/可不可…)
    "formula_block",        # <F> block: composition + administration
    "variant_clause",       # layer B paragraph aligned to a canonical clause
    "commentary",           # layer C commentary paragraph
}

RULE_TYPES = {
    "six_channel_definition_rule",   # 01 六經綱領
    "disease_pattern_rule",          # 02 病證定義
    "formula_pattern_rule",          # 03 方證對應
    "pulse_symptom_rule",            # 04 脈證關係
    "therapy_selection_rule",        # 05 治法選擇
    "contraindication_rule",         # 06 禁忌
    "mistreatment_rule",             # 07 誤治
    "transformation_rule",           # 08 傳變
    "prognosis_rule",                # 09 預後
    "administration_rule",           # 10 煎服法
    "formula_composition_rule",      # 11 方藥組成
    "dosage_processing_rule",        # 12 劑量炮製
    "differential_rule",             # 13 鑒別
    "rescue_reverse_rule",           # 14 救逆
    "recurrence_rule",               # 15 復發/勞復
    "variant_rule",                  # 16 版本異文
    "commentary_rule",               # 17 注家解釋
}

EVIDENCE_TYPES = {"original_text", "auxiliary_text", "variant_text", "commentary_text"}
INTERPRETATION_LEVELS = {"literal", "normalized", "model_inference"}
RELEASE_LEVELS = {"gold", "silver", "bronze", "rejected"}
RELATION_TYPES = {
    "same_formula_family", "differential", "mistreatment_transformation",
    "contraindication", "variant", "commentary_support", "sequence",
    "transmission",
}
SOURCE_LEVELS = {
    "original_clause", "inductive_from_original_clauses",
    "chapter_level_induction", "auxiliary_text", "variant_text",
    "commentary", "posthoc_induction", "model_inference",
}


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    return obj


class JsonRecord:
    """Mixin: serialize dataclass <-> dict/JSON line."""

    def to_dict(self) -> Dict[str, Any]:
        return _asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        # nested dataclasses
        for f in dataclasses.fields(cls):
            if f.name in kwargs and dataclasses.is_dataclass(f.type) and isinstance(kwargs[f.name], dict):
                kwargs[f.name] = f.type.from_dict(kwargs[f.name])  # type: ignore[union-attr]
        return cls(**kwargs)


def write_jsonl(path: Path, records: Iterable["JsonRecord | Dict[str, Any]"]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write((rec.to_json() if isinstance(rec, JsonRecord) else json.dumps(rec, ensure_ascii=False)) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 1. ShanghanClause — minimal evidence unit
# ---------------------------------------------------------------------------
@dataclass
class FormulaBlock(JsonRecord):
    formula_name: str = ""
    composition: List[Dict[str, str]] = field(default_factory=list)  # {herb, dose_processing}
    preparation: str = ""            # 煎法
    administration: str = ""         # 服法 + 將息法
    post_notes: List[str] = field(default_factory=list)  # 方後注 (本云…/校注)
    raw_text: str = ""


@dataclass
class ShanghanClause(JsonRecord):
    clause_id: str = ""
    book_title: str = ""
    version: str = "songben"
    chapter: str = ""
    six_channel: str = ""
    clause_number: int = 0
    raw_text: str = ""
    clean_text: str = ""
    text_type: str = "original_clause"
    layer: str = "A"
    contains_formula: bool = False
    formula_names: List[str] = field(default_factory=list)
    formula_blocks: List[FormulaBlock] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    negated_findings: List[str] = field(default_factory=list)
    pulse: List[str] = field(default_factory=list)
    disease_patterns: List[str] = field(default_factory=list)
    therapy_terms: List[str] = field(default_factory=list)
    contraindication_terms: List[str] = field(default_factory=list)
    mistreatment_terms: List[str] = field(default_factory=list)
    transformation_terms: List[str] = field(default_factory=list)
    prognosis_terms: List[str] = field(default_factory=list)
    herbs: List[str] = field(default_factory=list)
    time_course: List[str] = field(default_factory=list)
    collation_notes: List[str] = field(default_factory=list)
    logic_words: List[str] = field(default_factory=list)   # 若/不可/反/誤/或…
    sha256: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        rec = super().from_dict(d)
        rec.formula_blocks = [FormulaBlock.from_dict(b) if isinstance(b, dict) else b
                              for b in (d.get("formula_blocks") or [])]
        return rec


# ---------------------------------------------------------------------------
# 2. InitialRule — extracted from a single clause only
# ---------------------------------------------------------------------------
@dataclass
class AutonomousReview(JsonRecord):
    evidence_verified: bool = False
    schema_valid: bool = False
    semantic_result: str = "pending"     # pass | warn | fail
    critic_result: str = "pending"       # pass | warn | fail
    critic_flags: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    consensus_score: float = 0.0
    release_level: str = "rejected"


@dataclass
class InitialRule(JsonRecord):
    initial_rule_id: str = ""
    clause_id: str = ""
    six_channel: str = ""
    rule_type: str = "formula_pattern_rule"
    if_conditions: Dict[str, Any] = field(default_factory=dict)
    then_conclusions: Dict[str, Any] = field(default_factory=dict)
    evidence_span: str = ""
    evidence_type: str = "original_text"
    interpretation: str = ""
    interpretation_level: str = "normalized"
    model_confidence: float = 0.0
    prescription_strength: str = ""     # 主之 / 宜 / 屬 / 與 / 可與 / ""
    autonomous_review: AutonomousReview = field(default_factory=AutonomousReview)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        rec = super().from_dict(d)
        ar = d.get("autonomous_review")
        if isinstance(ar, dict):
            rec.autonomous_review = AutonomousReview.from_dict(ar)
        return rec


# ---------------------------------------------------------------------------
# 3. ClauseRelation
# ---------------------------------------------------------------------------
@dataclass
class ClauseRelation(JsonRecord):
    relation_id: str = ""
    source_clause_id: str = ""
    target_clause_id: str = ""
    relation_type: str = "sequence"
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# 4. FormulaPatternRule
# ---------------------------------------------------------------------------
@dataclass
class FormulaPatternRule(JsonRecord):
    formula_pattern_rule_id: str = ""
    formula: str = ""
    formula_family: str = ""
    six_channel_scope: List[str] = field(default_factory=list)
    core_pattern: str = ""
    core_symptoms: List[str] = field(default_factory=list)
    core_pulse: List[str] = field(default_factory=list)
    associated_symptoms: List[str] = field(default_factory=list)
    associated_pulse: List[str] = field(default_factory=list)
    contraindications: List[Dict[str, str]] = field(default_factory=list)
    composition: List[Dict[str, str]] = field(default_factory=list)
    administration_notes: List[str] = field(default_factory=list)
    modification_relations: List[Dict[str, str]] = field(default_factory=list)  # 加減方
    supporting_initial_rules: List[str] = field(default_factory=list)
    supporting_clauses: List[str] = field(default_factory=list)
    source_level: str = "inductive_from_original_clauses"
    interpretation_warning: str = ""
    consensus_score: float = 0.0
    release_level: str = "bronze"


# ---------------------------------------------------------------------------
# 5. SixChannelRule
# ---------------------------------------------------------------------------
@dataclass
class SixChannelRule(JsonRecord):
    six_channel_rule_id: str = ""
    six_channel: str = ""
    outline_clause_id: str = ""
    outline_text: str = ""
    summary: str = ""
    core_clauses: List[str] = field(default_factory=list)
    subtypes: List[Dict[str, Any]] = field(default_factory=list)
    main_formulas: List[Dict[str, Any]] = field(default_factory=list)
    contraindication_clauses: List[str] = field(default_factory=list)
    mistreatment_clauses: List[str] = field(default_factory=list)
    resolution_time: str = ""
    supporting_initial_rules: List[str] = field(default_factory=list)
    source_level: str = "chapter_level_induction"
    consensus_score: float = 0.0
    release_level: str = "silver"


# ---------------------------------------------------------------------------
# 6. TherapyRule (汗/吐/下/和/溫/清/補/救逆 + 禁/誤)
# ---------------------------------------------------------------------------
@dataclass
class TherapyRule(JsonRecord):
    therapy_rule_id: str = ""
    therapy_method: str = ""           # 汗法/下法/…/禁汗/禁下/禁吐/誤汗/誤下/誤吐
    polarity: str = "indicated"        # indicated | contraindicated | mistreatment
    summary: str = ""
    indications: List[str] = field(default_factory=list)
    contraindication_conditions: List[str] = field(default_factory=list)
    representative_formulas: List[str] = field(default_factory=list)
    supporting_clauses: List[str] = field(default_factory=list)
    supporting_initial_rules: List[str] = field(default_factory=list)
    source_level: str = "inductive_from_original_clauses"
    consensus_score: float = 0.0
    release_level: str = "silver"


# ---------------------------------------------------------------------------
# 7. MistreatmentTransformationRule
# ---------------------------------------------------------------------------
@dataclass
class MistreatmentTransformationRule(JsonRecord):
    mistreatment_rule_id: str = ""
    mistreatment_type: str = ""        # 誤汗/誤下/誤吐/火逆
    resulting_pattern: str = ""        # 結胸/痞/奔豚/亡陽/…
    manifestations: List[str] = field(default_factory=list)
    rescue_formulas: List[str] = field(default_factory=list)
    six_channel_scope: List[str] = field(default_factory=list)
    path: List[str] = field(default_factory=list)  # [誤治, 變證, 救治方]
    supporting_clauses: List[str] = field(default_factory=list)
    supporting_initial_rules: List[str] = field(default_factory=list)
    source_level: str = "inductive_from_original_clauses"
    consensus_score: float = 0.0
    release_level: str = "silver"


# ---------------------------------------------------------------------------
# 8. DifferentialRule
# ---------------------------------------------------------------------------
@dataclass
class DifferentialRule(JsonRecord):
    differential_rule_id: str = ""
    formulas: List[str] = field(default_factory=list)
    six_channels: List[str] = field(default_factory=list)
    shared_features: List[str] = field(default_factory=list)
    contrast_table: List[Dict[str, Any]] = field(default_factory=list)
    key_discriminators: List[str] = field(default_factory=list)
    composition_diff: Dict[str, List[str]] = field(default_factory=dict)
    supporting_clauses: List[str] = field(default_factory=list)
    source_level: str = "inductive_from_original_clauses"
    consensus_score: float = 0.0
    release_level: str = "silver"


# ---------------------------------------------------------------------------
# 9. VariantRule / CommentaryRule
# ---------------------------------------------------------------------------
@dataclass
class VariantRule(JsonRecord):
    variant_rule_id: str = ""
    clause_id: str = ""
    base_version: str = "songben"
    variant_version: str = ""
    variant_book: str = ""
    base_text: str = ""
    variant_text: str = ""
    similarity: float = 0.0
    notable_differences: List[str] = field(default_factory=list)
    source_level: str = "variant_text"
    release_level: str = "silver"


@dataclass
class CommentaryRule(JsonRecord):
    commentary_rule_id: str = ""
    clause_id: str = ""
    commentator: str = ""
    book: str = ""
    commentary_text: str = ""
    alignment_similarity: float = 0.0
    source_level: str = "commentary"
    release_level: str = "silver"


# ---------------------------------------------------------------------------
# 10. MergedShanghanRule — top of the hierarchy; never overwrites InitialRules
# ---------------------------------------------------------------------------
@dataclass
class MergedShanghanRule(JsonRecord):
    merged_rule_id: str = ""
    title: str = ""
    claim: str = ""
    source_scope: Dict[str, Any] = field(default_factory=dict)
    supporting_initial_rules: List[str] = field(default_factory=list)
    supporting_formula_pattern_rules: List[str] = field(default_factory=list)
    supporting_six_channel_rules: List[str] = field(default_factory=list)
    supporting_therapy_rules: List[str] = field(default_factory=list)
    supporting_mistreatment_rules: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    commentaries: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    evidence_chain: List[Dict[str, Any]] = field(default_factory=list)
    release_level: str = "silver"
    consensus_score: float = 0.0


# ---------------------------------------------------------------------------
# AuditRecord
# ---------------------------------------------------------------------------
@dataclass
class AuditRecord(JsonRecord):
    audit_id: str = ""
    target_id: str = ""
    target_kind: str = "initial_rule"
    stage: str = ""               # schema|evidence|semantic|critic|repair|consensus|release
    result: str = ""              # pass|warn|fail|repaired|gold|silver|bronze|rejected
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
