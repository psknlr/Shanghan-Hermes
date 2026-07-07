"""DifferentialDiagnosisAgent — 方證鑒別 generator.

For the protocol's canonical contrast pairs plus auto-discovered pairs
(formulas sharing ≥3 core findings), produces a structured contrast table
covering: 六經歸屬 / 核心症狀(同/異) / 核心脈象 / 寒熱虛實傾向 / 裏實與否 /
汗之有無 / 惡寒 / 煩渴 / 下利 / 方藥組成差異 / 禁忌差異 / 原文條文.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

from .. import config
from ..schemas import DifferentialRule, FormulaPatternRule, write_jsonl

CANONICAL_PAIRS: List[List[str]] = [
    ["桂枝湯", "麻黃湯"],
    ["桂枝湯", "桂枝加葛根湯"],
    ["麻黃湯", "大青龍湯"],
    ["小柴胡湯", "大柴胡湯"],
    ["白虎湯", "大承氣湯"],
    ["調胃承氣湯", "小承氣湯", "大承氣湯"],
    ["四逆湯", "通脈四逆湯"],
    ["真武湯", "四逆湯"],
    ["半夏瀉心湯", "生薑瀉心湯", "甘草瀉心湯"],
    ["五苓散", "豬苓湯"],
    ["梔子豉湯", "白虎湯"],
    ["葛根湯", "桂枝加葛根湯"],
    ["小青龍湯", "大青龍湯"],
    ["黃連阿膠湯", "梔子豉湯"],
    ["當歸四逆湯", "四逆湯"],
]

AXES = [
    ("汗之有無", ["汗出", "自汗出", "無汗"]),
    ("惡寒惡風", ["惡寒", "惡風", "不惡寒"]),
    ("煩渴", ["煩渴", "大煩渴不解", "渴", "消渴", "不渴", "口苦"]),
    ("下利", ["下利", "自利", "下利清穀", "熱利下重"]),
    ("裏實指徵", ["不大便", "燥屎", "潮熱", "譫語", "腹滿", "心下急"]),
    ("厥逆", ["厥逆", "手足厥冷", "手足厥寒", "四逆"]),
    ("胸脅心下", ["胸脅苦滿", "心下痞", "心下痞硬", "結胸", "心下悸"]),
]


def _axis_value(fpr: FormulaPatternRule, terms: List[str]) -> str:
    hits = [t for t in terms if t in fpr.core_symptoms or t in fpr.associated_symptoms]
    return "、".join(hits) if hits else "—"


def _cold_heat_tendency(fpr: FormulaPatternRule) -> str:
    """Coarse 寒熱虛實 tendency from composition (normalized knowledge)."""
    herbs = {c["herb"] for c in fpr.composition}
    warm = herbs & {"附子", "乾薑", "吳茱萸", "桂枝", "細辛", "蜀椒"}
    cold = herbs & {"石膏", "知母", "黃連", "黃芩", "黃檗", "大黃", "芒消", "梔子", "白頭翁", "秦皮"}
    if warm and cold:
        return "寒熱並用"
    if warm:
        return "溫熱（治寒證）"
    if cold:
        return "寒涼（治熱證）"
    return "平和/調和"


class DifferentialInducer:
    def __init__(self, formula_rules: List[FormulaPatternRule]):
        self.by_name: Dict[str, FormulaPatternRule] = {r.formula: r for r in formula_rules}

    def _build_one(self, names: List[str], rid: int) -> Optional[DifferentialRule]:
        rules = [self.by_name.get(n) for n in names]
        if any(r is None for r in rules):
            return None
        rules = [r for r in rules if r]
        shared = set(rules[0].core_symptoms + rules[0].associated_symptoms)
        for r in rules[1:]:
            shared &= set(r.core_symptoms + r.associated_symptoms)

        table: List[Dict] = []
        table.append({"axis": "六經歸屬",
                      **{r.formula: "、".join(r.six_channel_scope) or "—" for r in rules}})
        table.append({"axis": "核心症狀",
                      **{r.formula: "、".join(r.core_symptoms[:6]) or "—" for r in rules}})
        table.append({"axis": "核心脈象",
                      **{r.formula: "、".join(r.core_pulse[:4]) or "—" for r in rules}})
        table.append({"axis": "寒熱虛實傾向（後世歸納）",
                      **{r.formula: _cold_heat_tendency(r) for r in rules}})
        for axis_name, terms in AXES:
            row = {"axis": axis_name, **{r.formula: _axis_value(r, terms) for r in rules}}
            if any(v not in ("—",) for k, v in row.items() if k != "axis"):
                table.append(row)
        table.append({"axis": "禁忌",
                      **{r.formula: (r.contraindications[0]["condition"][:30] + "…"
                                     if r.contraindications else "—") for r in rules}})

        # composition diff
        comp_diff: Dict[str, List[str]] = {}
        all_herbs = [({c["herb"] for c in r.composition}, r.formula) for r in rules]
        common = set.intersection(*[h for h, _ in all_herbs]) if all_herbs else set()
        comp_diff["共有藥物"] = sorted(common)
        for herbs, name in all_herbs:
            comp_diff[f"{name}獨有"] = sorted(herbs - common - set().union(
                *[h for h, n2 in all_herbs if n2 != name]) if len(all_herbs) > 1 else herbs - common)

        # key discriminators: core symptoms unique to exactly one formula
        discriminators = []
        for r in rules:
            others = set()
            for r2 in rules:
                if r2.formula != r.formula:
                    others |= set(r2.core_symptoms + r2.associated_symptoms)
            uniq = [s for s in r.core_symptoms if s not in others][:3]
            if uniq:
                discriminators.append(f"{r.formula}：{'、'.join(uniq)}")

        clause_ids = sorted({cid for r in rules for cid in r.supporting_clauses})
        return DifferentialRule(
            differential_rule_id=f"DR_{rid:03d}",
            formulas=[r.formula for r in rules],
            six_channels=sorted({ch for r in rules for ch in r.six_channel_scope}),
            shared_features=sorted(shared)[:8],
            contrast_table=table,
            key_discriminators=discriminators,
            composition_diff=comp_diff,
            supporting_clauses=clause_ids[:20],
            consensus_score=0.86,
            release_level="silver" if all(r.release_level in ("gold", "silver")
                                          for r in rules) else "bronze",
        )

    def induce(self) -> List[DifferentialRule]:
        out: List[DifferentialRule] = []
        rid = 0
        done = set()
        for names in CANONICAL_PAIRS:
            rid += 1
            r = self._build_one(names, rid)
            if r:
                out.append(r)
                done.add(tuple(sorted(names)))
        # auto-discovery: pairs with ≥2 shared core symptoms and ≥3 shared
        # overall (core+associated). Sharing counts core+associated so the
        # core[:8] cap ordering doesn't decide whether a pair is discoverable,
        # while the 2-core floor keeps out background overlaps (發熱 etc.)
        names = sorted(self.by_name)
        for a, b in combinations(names, 2):
            key = tuple(sorted([a, b]))
            if key in done:
                continue
            ra, rb = self.by_name[a], self.by_name[b]
            core_shared = set(ra.core_symptoms) & set(rb.core_symptoms)
            shared = (set(ra.core_symptoms + ra.associated_symptoms)
                      & set(rb.core_symptoms + rb.associated_symptoms))
            if len(shared) >= 3 and len(core_shared) >= 2:
                rid += 1
                r = self._build_one([a, b], rid)
                if r:
                    out.append(r)
                    done.add(key)
        return out

    def run(self) -> List[DifferentialRule]:
        rules = self.induce()
        config.ensure_dirs()
        write_jsonl(config.RULES_DIFFERENTIAL_DIR / "differential_rules.jsonl", rules)
        return rules
