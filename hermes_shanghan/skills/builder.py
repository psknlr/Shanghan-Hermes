"""SkillBuilderAgent — compiles all rule layers into the Hermes skill tree.

data/skills/shanghanlun/
├─ hermes.shanghan.catalog/
├─ hermes.shanghan.six_channels/{taiyang…jueyin}/
├─ hermes.shanghan.formula_patterns/<formula_slug>/
├─ hermes.shanghan.mistreatment/
├─ hermes.shanghan.contraindications/
├─ hermes.shanghan.therapy/
├─ hermes.shanghan.transformation/
├─ hermes.shanghan.differential/
├─ hermes.shanghan.clause_explainer/
├─ hermes.shanghan.variants/
├─ hermes.shanghan.paper_writer/
└─ hermes.shanghan.patient_education/

Each skill = SKILL.md (YAML frontmatter + usage doc) + rules.jsonl +
examples.jsonl. Skills are pure data — they can be loaded by any agent
runtime (including Claude Code skills) or by the built-in Skill RAG.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import config
from ..schemas import (CommentaryRule, DifferentialRule, FormulaPatternRule,
                       InitialRule, JsonRecord, MergedShanghanRule,
                       MistreatmentTransformationRule, ShanghanClause,
                       SixChannelRule, TherapyRule, VariantRule)
from .pinyin import formula_slug


def _frontmatter(name: str, description: str, **extra) -> str:
    lines = ["---", f"name: {name}", f"description: {description}"]
    for k, v in extra.items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}")
    lines += ["---", ""]
    return "\n".join(lines)


def _write_skill(skill_dir: Path, skill_md: str,
                 rules: Sequence[JsonRecord | Dict],
                 examples: Sequence[Dict]):
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    with (skill_dir / "rules.jsonl").open("w", encoding="utf-8") as fh:
        for r in rules:
            fh.write((r.to_json() if isinstance(r, JsonRecord)
                      else json.dumps(r, ensure_ascii=False)) + "\n")
    with (skill_dir / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


CORE_PRINCIPLES = """
## 核心原則

- 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
- 合併規則不能覆蓋初始條文規則（本 Skill 中所有規則均保留 supporting_initial_rules / supporting_clauses 回鏈）。
- 輸出必須區分：原文直述（A）／版本異文（B）／注家解釋（C）／後世歸納（D）／模型推理（E）。
- 患者端禁止自動診斷、自動處方和劑量建議。
"""


class SkillBuilder:
    def __init__(self,
                 clauses: List[ShanghanClause],
                 initial_rules: List[InitialRule],
                 formula_rules: List[FormulaPatternRule],
                 six_channel_rules: List[SixChannelRule],
                 therapy_rules: List[TherapyRule],
                 mistreatment_rules: List[MistreatmentTransformationRule],
                 differential_rules: List[DifferentialRule],
                 merged_rules: List[MergedShanghanRule],
                 variant_rules: Optional[List[VariantRule]] = None,
                 commentary_rules: Optional[List[CommentaryRule]] = None):
        self.clauses = clauses
        self.clause_store = {c.clause_id: c for c in clauses}
        self.initial_rules = initial_rules
        self.formula_rules = formula_rules
        self.six_channel_rules = six_channel_rules
        self.therapy_rules = therapy_rules
        self.mistreatment_rules = mistreatment_rules
        self.differential_rules = differential_rules
        self.merged_rules = merged_rules
        self.variant_rules = variant_rules or []
        self.commentary_rules = commentary_rules or []
        self.root = config.SKILLS_DIR

    # ------------------------------------------------------------------
    def build_all(self) -> Dict[str, int]:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        counts = {
            "catalog": self._build_catalog(),
            "six_channels": self._build_six_channels(),
            "formula_patterns": self._build_formula_patterns(),
            "mistreatment": self._build_mistreatment(),
            "contraindications": self._build_contraindications(),
            "therapy": self._build_therapy(),
            "transformation": self._build_transformation(),
            "differential": self._build_differential(),
            "clause_explainer": self._build_clause_explainer(),
            "variants": self._build_variants(),
            "paper_writer": self._build_paper_writer(),
            "patient_education": self._build_patient_education(),
        }
        manifest = {
            "skill_tree": "hermes.shanghan",
            "root": str(self.root.relative_to(config.REPO_ROOT)),
            "skills_built": counts,
            "total_dirs": sum(1 for _ in self.root.rglob("SKILL.md")),
        }
        (self.root / "skills_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        return counts

    # ------------------------------------------------------------------
    def _build_catalog(self) -> int:
        canonical = [c for c in self.clauses if c.text_type == "original_clause"]
        chapters: Dict[str, Dict] = {}
        for c in canonical:
            ch = chapters.setdefault(c.chapter, {"chapter": c.chapter,
                                                 "six_channel": c.six_channel,
                                                 "first": c.clause_number,
                                                 "last": c.clause_number,
                                                 "n": 0})
            ch["last"] = max(ch["last"], c.clause_number)
            ch["first"] = min(ch["first"], c.clause_number)
            ch["n"] += 1
        md = _frontmatter(
            "hermes.shanghan.catalog",
            "《傷寒論》目錄與版本層級總覽：宋本398條編號體系、十篇章節、輔助篇章、版本/注本分層。",
            roles=["doctor", "researcher", "student", "patient"]) + f"""
# 《傷寒論》目錄 Skill

主底本：傷寒論（宋本，趙開美本），按現代通行編號共 **{len(canonical)} 條**。
輔助層：宋本辨脈法/平脈法/傷寒例/痙濕暍/可與不可諸篇（auxiliary_clause）。
異文層（B）：桂林古本、千金翼方版。注釋層（C）：成無己《註解傷寒論》等。
類方層（D）：《傷寒論類方》。

## 篇章結構
| 篇章 | 六經 | 條文範圍 | 條數 |
|---|---|---|---|
""" + "\n".join(
            f"| {ch['chapter']} | {ch['six_channel']} | {ch['first']}–{ch['last']} | {ch['n']} |"
            for ch in chapters.values()) + CORE_PRINCIPLES
        rules = list(chapters.values())
        examples = [{
            "query": "傷寒論宋本一共多少條？太陽病篇的範圍是？",
            "answer_outline": f"共{len(canonical)}條；太陽病分上中下三篇（1–178條），證據見篇章結構表。",
        }]
        _write_skill(self.root / "hermes.shanghan.catalog", md, rules, examples)
        return 1

    # ------------------------------------------------------------------
    def _build_six_channels(self) -> int:
        base = self.root / "hermes.shanghan.six_channels"
        n = 0
        for scr in self.six_channel_rules:
            pinyin = config.CHANNEL_PINYIN.get(scr.six_channel)
            if not pinyin:
                continue
            n += 1
            skill_name = f"hermes.shanghan.{pinyin}"
            related_initial = [r for r in self.initial_rules
                               if r.six_channel == scr.six_channel and
                               r.autonomous_review.release_level in ("gold", "silver")]
            subtype_md = "\n".join(
                f"- **{s['name']}**（錨定方：{'、'.join(s['anchor_formulas'])}；"
                f"證據條文：{'、'.join(s['evidence_clauses'][:3]) or '—'}）"
                for s in scr.subtypes)
            formula_md = "\n".join(
                f"| {f['formula']} | {f['clause_count']} |" for f in scr.main_formulas[:10])
            md = _frontmatter(
                skill_name,
                f"{scr.six_channel}六經規則：提綱、亞型、主方、禁忌、誤治與條文證據。",
                six_channel=scr.six_channel,
                release_level=scr.release_level) + f"""
# {scr.six_channel} Skill

## 提綱（原文直述）
> {scr.outline_text}
（{scr.outline_clause_id}）

## 總括（chapter_level_induction，模型歸納）
{scr.summary}

欲解時：{scr.resolution_time or '原文未及/未提取'}

## 內部結構（亞型名稱為後世歸納，證據條文為原文）
{subtype_md}

## 主要方劑（按條文頻次）
| 方劑 | 條文數 |
|---|---|
{formula_md}

## 禁忌條文
{'、'.join(scr.contraindication_clauses[:10]) or '—'}

## 誤治相關條文
{'、'.join(scr.mistreatment_clauses[:12]) or '—'}
""" + CORE_PRINCIPLES
            examples = [
                {"query": f"幫我講清楚{scr.six_channel}的內部結構。",
                 "route": "teaching.lesson", "args": {"channel": scr.six_channel}},
                {"query": f"{scr.six_channel}的提綱條文是哪一條？",
                 "answer_outline": f"{scr.outline_clause_id}：{scr.outline_text}"},
            ]
            _write_skill(base / pinyin, md, [scr] + related_initial[:200], examples)
        return n

    # ------------------------------------------------------------------
    def _build_formula_patterns(self) -> int:
        base = self.root / "hermes.shanghan.formula_patterns"
        ir_by_id = {r.initial_rule_id: r for r in self.initial_rules}
        n = 0
        for fpr in self.formula_rules:
            if fpr.release_level == "rejected" or not fpr.supporting_clauses:
                continue
            n += 1
            slug = formula_slug(fpr.formula)
            skill_name = f"hermes.formula.{slug}"
            comp_md = "　".join(
                f"{c['herb']}（{c['dose_processing']}）" if c.get("dose_processing") else c["herb"]
                for c in fpr.composition) or "（本方組成見他條或未載）"
            clause_md = ""
            for cid in fpr.supporting_clauses[:6]:
                c = self.clause_store.get(cid)
                if c:
                    clause_md += f"\n> {c.clean_text}\n（{cid}，{c.chapter}）\n"
            mods = "\n".join(
                f"- {m['modified_formula']}：加 {m['added_herbs'] or '—'}；減 {m['removed_herbs'] or '—'}"
                for m in fpr.modification_relations) or "—"
            contra_md = "\n".join(
                f"- {c['condition']}（{c['clause_id']}）" for c in fpr.contraindications) or "—"
            md = _frontmatter(
                skill_name,
                f"{fpr.formula}方證規則：核心證候、脈象、組成、煎服法、加減方與禁忌，全部回源條文。",
                formula=fpr.formula, formula_family=fpr.formula_family,
                six_channel_scope=fpr.six_channel_scope,
                release_level=fpr.release_level) + f"""
# {fpr.formula}方證 Skill

| 項目 | 內容 | 證據層 |
|---|---|---|
| 核心病類 | {fpr.core_pattern} | 歸納（D/E） |
| 六經歸屬 | {'、'.join(fpr.six_channel_scope) or '—'} | 原文章節（A） |
| 核心症狀 | {'、'.join(fpr.core_symptoms) or '—'} | 跨條歸納自原文（A→D） |
| 核心脈象 | {'、'.join(fpr.core_pulse) or '—'} | 跨條歸納自原文（A→D） |
| 兼證 | {'、'.join(fpr.associated_symptoms) or '—'} | 跨條歸納自原文（A→D） |

## 組成（原文直述）
{comp_md}

## 煎服法要點（原文直述）
{chr(10).join('- ' + a for a in fpr.administration_notes[:3]) or '—'}

## 加減方關係
{mods}

## 禁忌
{contra_md}

## 支持條文（原文）
{clause_md}

> ⚠️ {fpr.interpretation_warning}
""" + CORE_PRINCIPLES
            support = [ir_by_id[i] for i in fpr.supporting_initial_rules if i in ir_by_id]
            examples = [
                {"query": f"{fpr.formula}的方證要點和原文依據？",
                 "answer_outline": f"核心證：{'、'.join(fpr.core_symptoms[:4])}；"
                                   f"條文：{'、'.join(fpr.supporting_clauses[:3])}"},
                {"query": f"{fpr.formula}和同類方如何鑒別？",
                 "route": "differential", "args": {"formula": fpr.formula}},
            ]
            _write_skill(base / slug, md, [fpr] + support, examples)
        return n

    # ------------------------------------------------------------------
    def _build_mistreatment(self) -> int:
        md = _frontmatter(
            "hermes.shanghan.mistreatment",
            "誤治傳變圖譜：誤汗/誤下/誤吐/火逆 → 變證 → 救治方，全部路徑帶條文證據。") + """
# 誤治傳變 Skill

誤治方式 → 變證 → 症狀表現 → 救治方 → 原文證據。

| 誤治 | 變證 | 救治方 | 證據條文（前3） | 等級 |
|---|---|---|---|---|
""" + "\n".join(
            f"| {m.mistreatment_type} | {m.resulting_pattern} | "
            f"{'、'.join(m.rescue_formulas[:3])} | "
            f"{'、'.join(m.supporting_clauses[:3])} | {m.release_level} |"
            for m in self.mistreatment_rules) + CORE_PRINCIPLES
        examples = [
            {"query": "誤下以後出現心下痞怎麼辦？",
             "answer_outline": "誤下→痞證→瀉心湯類；證據見規則 MTR（誤下/痞證）。"},
            {"query": "火逆燒針後驚狂的救治？",
             "answer_outline": "火逆→驚狂→桂枝去芍藥加蜀漆牡蠣龍骨救逆湯。"},
        ]
        _write_skill(self.root / "hermes.shanghan.mistreatment", md,
                     self.mistreatment_rules, examples)
        return 1

    # ------------------------------------------------------------------
    def _build_contraindications(self) -> int:
        contra_therapy = [t for t in self.therapy_rules if t.polarity == "contraindicated"]
        contra_initial = [r for r in self.initial_rules
                          if r.rule_type == "contraindication_rule"
                          and r.autonomous_review.release_level != "rejected"]
        rows = []
        for r in contra_initial[:40]:
            c = self.clause_store.get(r.clause_id)
            if c:
                rows.append(f"| {r.clause_id} | {c.clean_text[:38]}… | "
                            f"{'、'.join(r.then_conclusions.get('contraindicated_actions', [])[:2]) or '禁與'} |")
        md = _frontmatter(
            "hermes.shanghan.contraindications",
            "禁忌法度：禁汗/禁下/禁吐諸條與方劑禁例，逐條回源。") + """
# 禁忌法度 Skill

## 治法之禁（含宋本可/不可專篇）
""" + "\n".join(f"- **{t.therapy_method}**：{t.summary}（條文 {len(t.supporting_clauses)} 條）"
                for t in contra_therapy) + """

## 禁例條文（節選）
| 條文 | 原文 | 禁忌 |
|---|---|---|
""" + "\n".join(rows) + CORE_PRINCIPLES
        examples = [
            {"query": "哪些人不可發汗？", "answer_outline": "咽喉乾燥者、淋家、瘡家、衄家、亡血家、汗家等——見禁汗規則與其條文。"},
            {"query": "酒客能不能用桂枝湯？", "answer_outline": "不可——「若酒客病，不可與桂枝湯，得之則嘔」（SHL_SONGBEN_0017）。"},
        ]
        _write_skill(self.root / "hermes.shanghan.contraindications", md,
                     contra_therapy + contra_initial, examples)
        return 1

    # ------------------------------------------------------------------
    THERAPY_SUBSKILLS = {
        "汗法": ("sweating", "汗法規則：邪在表者汗而發之；適應指徵、代表方與禁例。"),
        "下法": ("purgation", "下法規則：實在裏者下之；承氣輩指徵、急下三證與禁下諸條。"),
        "和法": ("harmonization", "和法規則：半表半裏樞機不利，小柴胡湯和解；瀉心輩辛開苦降。"),
        "吐法": ("vomiting", "吐法規則：邪實胸中者越之，瓜蒂散主之；虛家禁用。"),
        "溫法": ("warming", "溫法規則：裏虛寒者溫之，四逆輩、理中輩。"),
        "清法": ("clearing", "清法規則：無形之熱清之，白虎輩、梔子豉輩、芩連柏輩。"),
        "補法": ("tonifying", "補法規則：正虛者補之，小建中湯、炙甘草湯。"),
        "救逆": ("rescue_reverse", "救逆規則：誤治壞病、陽亡陰竭之急救回逆法度。"),
        "利水": ("water_regulation", "利水規則：水飲內停者利其小便，五苓散/豬苓湯/真武湯。"),
    }

    def _build_therapy(self) -> int:
        base = self.root / "hermes.shanghan.therapy"
        indicated = [t for t in self.therapy_rules if t.polarity == "indicated"]

        # decide which subskills actually have rules before advertising them
        # in the overview — no phantom sub-Skill references
        sub_plan = []
        for method, (slug, desc) in self.THERAPY_SUBSKILLS.items():
            group = [t for t in self.therapy_rules if t.therapy_method == method]
            related_mist = {"汗法": "誤汗", "下法": "誤下", "吐法": "誤吐"}.get(method)
            mist = [t for t in self.therapy_rules
                    if related_mist and t.therapy_method == related_mist]
            prohib = [t for t in self.therapy_rules
                      if t.therapy_method == {"汗法": "禁汗", "下法": "禁下",
                                              "吐法": "禁吐"}.get(method)]
            if group or prohib:
                sub_plan.append((method, slug, desc, group, mist, prohib))

        sub_list = " /\n".join(f"hermes.shanghan.therapy.{slug}"
                               for _, slug, *_ in sub_plan)
        md = _frontmatter(
            "hermes.shanghan.therapy",
            "治法規則總覽：汗/吐/下/和/溫/清/補/救逆/利水的適應指徵與代表方。") + f"""
# 治法 Skill（總覽）

各治法獨立子 Skill：
{sub_list}。

""" + "\n".join(
            f"## {t.therapy_method}\n{t.summary}\n- 指徵：{'、'.join(t.indications[:8]) or '—'}\n"
            f"- 代表方：{'、'.join(t.representative_formulas) or '—'}\n"
            f"- 證據條文：{len(t.supporting_clauses)} 條\n"
            for t in indicated) + CORE_PRINCIPLES
        examples = [
            {"query": "什麼情況當用下法？", "answer_outline": "潮熱、譫語、腹滿痛、不大便、燥屎內結——承氣輩；表未解者不可下。"},
        ]
        _write_skill(base, md, self.therapy_rules, examples)

        n = 1
        for method, slug, desc, group, mist, prohib in sub_plan:
            n += 1
            body = []
            for t in group:
                clause_refs = "、".join(t.supporting_clauses[:6])
                body.append(f"## 適應（{t.therapy_method}）\n{t.summary}\n"
                            f"- 指徵：{'、'.join(t.indications[:10]) or '—'}\n"
                            f"- 代表方：{'、'.join(t.representative_formulas) or '—'}\n"
                            f"- 條文：{clause_refs}")
            for t in prohib:
                body.append(f"## 禁例（{t.therapy_method}）\n{t.summary}\n"
                            + "\n".join(f"- {c}" for c in t.contraindication_conditions[:8]))
            for t in mist:
                body.append(f"## 誤施之變（{t.therapy_method}）\n{t.summary}\n"
                            f"- 變證：{'、'.join(t.indications[:8]) or '—'}\n"
                            f"- 救治方：{'、'.join(t.representative_formulas) or '—'}")
            md_sub = _frontmatter(f"hermes.shanghan.therapy.{slug}", desc,
                                  therapy_method=method) + \
                f"# {method} Skill\n\n" + "\n\n".join(body) + CORE_PRINCIPLES
            examples_sub = [{"query": f"{method}的適應證和禁忌？",
                             "answer_outline": f"見本 Skill 適應/禁例/誤施之變三節，均帶條文證據。"}]
            _write_skill(base / slug, md_sub, group + prohib + mist, examples_sub)
        return n

    # ------------------------------------------------------------------
    def _build_transformation(self) -> int:
        trans_rules = [r for r in self.initial_rules
                       if r.rule_type in ("transformation_rule",)
                       and r.autonomous_review.release_level != "rejected"]
        rows = []
        for r in trans_rules:
            c = self.clause_store.get(r.clause_id)
            if c:
                rows.append(f"| {r.clause_id} | {c.clean_text[:42]}… | "
                            f"{'、'.join(r.then_conclusions.get('transmission', []))} |")
        md = _frontmatter(
            "hermes.shanghan.transformation",
            "傳變規則：六經傳變、轉屬轉入與不傳之判斷。") + """
# 傳變 Skill

| 條文 | 原文 | 傳變標記 |
|---|---|---|
""" + "\n".join(rows) + """

## 傳變判斷要點（原文歸納）
- 「傷寒一日，太陽受之，脈若靜者為不傳；頗欲吐，若躁煩，脈數急者，為傳也。」（SHL_SONGBEN_0004）
- 「傷寒二三日，陽明、少陽證不見者，為不傳也。」（SHL_SONGBEN_0005）
""" + CORE_PRINCIPLES
        examples = [{"query": "怎麼判斷太陽病傳不傳？",
                     "answer_outline": "以脈證為據：脈靜為不傳；躁煩、脈數急為傳（第4、5條）。"}]
        _write_skill(self.root / "hermes.shanghan.transformation", md, trans_rules, examples)
        return 1

    # ------------------------------------------------------------------
    def _build_differential(self) -> int:
        md_rows = []
        for d in self.differential_rules:
            md_rows.append(f"## {' vs '.join(d.formulas)}\n")
            md_rows.append("| 鑒別軸 | " + " | ".join(d.formulas) + " |")
            md_rows.append("|" + "---|" * (len(d.formulas) + 1))
            for row in d.contrast_table:
                md_rows.append("| " + row["axis"] + " | " +
                               " | ".join(str(row.get(f, "—")) for f in d.formulas) + " |")
            if d.key_discriminators:
                md_rows.append("\n關鍵鑒別點：" + "；".join(d.key_discriminators))
            md_rows.append("")
        md = _frontmatter(
            "hermes.shanghan.differential",
            "方證鑒別：桂枝湯vs麻黃湯、三承氣、三瀉心等鑒別對，多軸對比表。") + \
            "# 方證鑒別 Skill\n\n" + "\n".join(md_rows) + CORE_PRINCIPLES
        examples = [
            {"query": "桂枝湯和麻黃湯怎麼區分？",
             "route": "differential", "args": {"formulas": ["桂枝湯", "麻黃湯"]}},
            {"query": "三個瀉心湯的區別？",
             "route": "differential", "args": {"formulas": ["半夏瀉心湯", "生薑瀉心湯", "甘草瀉心湯"]}},
        ]
        _write_skill(self.root / "hermes.shanghan.differential", md,
                     self.differential_rules, examples)
        return 1

    # ------------------------------------------------------------------
    def _build_clause_explainer(self) -> int:
        md = _frontmatter(
            "hermes.shanghan.clause_explainer",
            "條文解釋：按條文號回源原文，附實體標註、初始規則、條文關係、異文與注釋。") + """
# 條文解釋 Skill

輸入條文號（1–398）或 clause_id，輸出：
1. 原文（A層，verbatim）+ 篇章 + 六經歸屬；
2. 實體標註：症狀/脈象/方劑/治法/禁忌/誤治/預後；
3. 本條抽取的 InitialRules（含審核等級）；
4. 條文關係：同方族/鑒別/誤治傳變/禁忌/傳變；
5. 版本異文（B層）與成無己注（C層）；
6. 模型解讀（E層，明確標註）。

調用：`hermes-shanghan explain-clause 12`
""" + CORE_PRINCIPLES
        examples = [
            {"query": "解釋第12條", "route": "explain_clause", "args": {"clause": 12}},
            {"query": "第317條講什麼？", "route": "explain_clause", "args": {"clause": 317}},
        ]
        sample_rules = [r for r in self.initial_rules[:50]]
        _write_skill(self.root / "hermes.shanghan.clause_explainer", md, sample_rules, examples)
        return 1

    # ------------------------------------------------------------------
    def _build_variants(self) -> int:
        rows = []
        for v in self.variant_rules[:60]:
            if v.notable_differences:
                rows.append(f"| {v.clause_id} | {v.variant_book} | {v.similarity} | "
                            f"{'；'.join(v.notable_differences)[:50]} |")
        md = _frontmatter(
            "hermes.shanghan.variants",
            "版本異文（B層）：桂林古本/千金翼方版與宋本的條文級對齊與差異。") + """
# 版本異文 Skill

| 宋本條文 | 異文底本 | 相似度 | 主要差異 |
|---|---|---|---|
""" + "\n".join(rows) + """

注意：異文層永不覆蓋宋本原文層；引用異文必須標註 variant_text。
""" + CORE_PRINCIPLES
        examples = [{"query": "第12條桂本有沒有異文？",
                     "answer_outline": "查 variants 規則中 clause_id=SHL_SONGBEN_0012 的對齊記錄。"}]
        # stratified commentary sample: rules are written in book order, so a
        # flat [:200] slice would contain only the first book (成無己)
        by_book: Dict[str, List] = {}
        for r in self.commentary_rules:
            by_book.setdefault(r.book, []).append(r)
        per_book = max(1, 200 // max(1, len(by_book)))
        comm_sample = [r for rs in by_book.values() for r in rs[:per_book]]
        _write_skill(self.root / "hermes.shanghan.variants", md,
                     self.variant_rules[:400] + comm_sample, examples)
        return 1

    # ------------------------------------------------------------------
    def _build_paper_writer(self) -> int:
        md = _frontmatter(
            "hermes.shanghan.paper_writer",
            "論文寫作：方證規律挖掘/六經知識圖譜/誤治傳變研究等論文的自動生成。") + """
# Paper Writer Skill

支持的論文類型：
1. 《傷寒論》方證規律挖掘
2. 《傷寒論》六經辨證知識圖譜
3. 《傷寒論》誤治傳變規則研究
4. 《傷寒論》方劑網絡藥理學前置研究
5. 《傷寒論》某方劑歷代注釋比較
6. 古籍數據挖掘與智能體方法學論文

自動生成模塊：Title / Abstract / Introduction / Methods / Results /
Discussion / Conclusion / Figures / Tables / References / Supplementary /
Cover Letter。

調用：`hermes-shanghan paper --type formula_pattern --topic 桂枝湯類方`
所有結果性陳述自動掛接規則 ID 與條文 ID（證據鏈）。
""" + CORE_PRINCIPLES
        examples = [
            {"query": "生成一篇桂枝湯類方證演化的論文大綱",
             "route": "paper", "args": {"type": "formula_pattern", "topic": "桂枝湯類方證演化"}},
        ]
        _write_skill(self.root / "hermes.shanghan.paper_writer", md, [], examples)
        return 1

    # ------------------------------------------------------------------
    def _build_patient_education(self) -> int:
        md = _frontmatter(
            "hermes.shanghan.patient_education",
            "患者教育：中醫術語通俗解釋、就診症狀整理、風險信號提醒。禁止診斷/處方/劑量。") + """
# 患者教育 Skill

## 能做
- 通俗解釋《傷寒論》術語（太陽表證、六經、方證、和解……）
- 幫患者把症狀整理成就診清單
- 提供需要及時就醫的風險信號提醒

## 不做（硬性安全邊界）
- ❌ 判斷用戶是否屬於某證型（自動診斷）
- ❌ 推薦任何方劑或藥物（自動處方）
- ❌ 任何劑量建議（劑量文本自動脫敏）

意圖守衛：診斷/處方/劑量類請求 → 禮貌拒絕 + 就醫指引。
""" + CORE_PRINCIPLES
        examples = [
            {"query": "醫生說我是太陽表證，這是什麼意思？",
             "route": "patient.explain",
             "expected": "通俗解釋+常見理解方向+建議結合醫師判斷，不判斷用戶是否屬於該證。"},
            {"query": "給我開個方治感冒", "route": "patient.explain",
             "expected": "拒絕（處方意圖），給出可替代的幫助。"},
        ]
        _write_skill(self.root / "hermes.shanghan.patient_education", md, [], examples)
        return 1
