"""InitialRuleExtractorAgent — clause-level rule extraction.

Hard constraints (Hermes core rules):
  * one clause → its own rules only; NO cross-clause induction here;
  * evidence_span is always the verbatim clause text;
  * latter-day pathogenesis vocabulary never enters if/then fields — the
    model's reading goes to `interpretation` with an explicit level.

A single clause may yield several InitialRules (e.g. clause 12 produces a
formula_pattern_rule plus composition/administration/dosage rules from its
<F> block).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import config, lexicon
from ..schemas import InitialRule, ShanghanClause
from ..textutil import fold_variants
from .entities import EntityExtractor, EntityResult

RE_OUTLINE = re.compile(r"(太陽|陽明|少陽|太陰|少陰|厥陰)之為病")
RE_NAMING = re.compile(r"(?:名為|名曰|為|此名|是名)([^，。；]{1,6}?)(?:也)?[，。；]")
RE_RESOLUTION = lexicon.RE_RESOLUTION_TIME


def _conditions_text(text: str, mention: Optional[Dict]) -> str:
    """Everything before the formula mention = the IF side.

    Classical clause grammar puts the prescription at the end
    (「…者，桂枝湯主之」), so the run-up — including the symptom list right
    before 「者」 — all belongs to the conditions.
    """
    if not mention:
        return text
    cond = text[:mention["start"]]
    return cond.rstrip("，者、 ")


class InitialRuleExtractor:
    def __init__(self, extractor: EntityExtractor):
        self.extractor = extractor
        self._counters: Dict[str, int] = {}

    def _new_id(self, clause: ShanghanClause) -> str:
        n = self._counters.get(clause.clause_id, 0) + 1
        self._counters[clause.clause_id] = n
        stem = clause.clause_id.replace("SHL_SONGBEN_", "")
        return f"IR_SHL_{stem}_{n:03d}"

    # ------------------------------------------------------------------
    def extract_clause_rules(self, clause: ShanghanClause) -> List[InitialRule]:
        text = fold_variants(clause.clean_text)
        res = self.extractor.extract(text)
        rules: List[InitialRule] = []
        ev_type = "original_text" if clause.text_type == "original_clause" else "auxiliary_text"

        def base_rule(rule_type: str) -> InitialRule:
            return InitialRule(
                initial_rule_id=self._new_id(clause),
                clause_id=clause.clause_id,
                six_channel=clause.six_channel,
                rule_type=rule_type,
                evidence_span=text,
                evidence_type=ev_type,
                interpretation_level="normalized",
            )

        positive_mentions = [m for m in res.formula_mentions if not m["negated"]]
        # main formula = strongest prescription marker, latest on ties — a
        # trailing post-note formula (157條末「理中者…」) must not outrank
        # the mid-clause 「生薑瀉心湯主之」
        _rank = {"主之": 5, "宜": 4, "屬": 3, "與": 2, "可與": 1}
        main_mention = max(positive_mentions,
                           key=lambda m: (_rank.get(m["strength"], 0), m["start"])) \
            if positive_mentions else None

        # branch prescriptions: one per formula (strongest mention), in text
        # order. A clause like 149 條 prescribes 大陷胸湯主之 for the 結胸
        # branch AND 宜半夏瀉心湯 for the 痞 branch — each gets its own rule
        # with conditions taken from its own branch segment.
        best_by_name: Dict[str, Dict] = {}
        for m in positive_mentions:
            if not m["strength"]:
                continue
            cur = best_by_name.get(m["name"])
            if cur is None or (_rank[m["strength"]], m["start"]) > \
                    (_rank[cur["strength"]], cur["start"]):
                best_by_name[m["name"]] = m
        prescriptions = sorted(best_by_name.values(), key=lambda m: m["start"])
        if not prescriptions and main_mention:
            prescriptions = [main_mention]

        # ---- 01 six_channel_definition_rule --------------------------------
        m_out = RE_OUTLINE.search(text)
        if m_out:
            r = base_rule("six_channel_definition_rule")
            channel = m_out.group(1) + "病"
            r.if_conditions = {"disease": [channel], "symptoms": res.symptoms, "pulse": res.pulse}
            r.then_conclusions = {"definition": text, "channel": channel}
            r.interpretation = f"本條為{channel}提綱條文，定義{channel}的脈證基準。"
            r.interpretation_level = "literal"
            r.model_confidence = 0.95
            rules.append(r)

        # ---- 02 disease_pattern_rule ---------------------------------------
        for nm in RE_NAMING.finditer(text):
            name = nm.group(1).strip()
            if name in {"也", "之", "病"} or len(name) < 2:
                continue
            if any(name in dp or dp in name for dp in lexicon.DISEASE_PATTERNS) or name.endswith("病") \
               or name in ("中風", "傷寒", "溫病", "風溫", "剛痙", "柔痙", "霍亂", "縱", "橫"):
                r = base_rule("disease_pattern_rule")
                r.if_conditions = {"disease": res.disease_patterns, "symptoms": res.symptoms,
                                   "negated_findings": res.negated_findings, "pulse": res.pulse}
                r.then_conclusions = {"pattern_name": name}
                r.interpretation = f"本條以脈證界定病名「{name}」。"
                r.interpretation_level = "literal"
                r.model_confidence = 0.9
                rules.append(r)
                break

        # ---- 03 formula_pattern_rule (one per branch prescription) ----------
        prev_end = 0
        for pm in prescriptions:
            cond_text = text[prev_end:pm["start"]].rstrip("，者、 ")
            prev_end = pm["end"]
            cond = self.extractor.extract(cond_text)
            r = base_rule("formula_pattern_rule")
            r.prescription_strength = pm["strength"] or "與"
            r.if_conditions = {
                "disease": cond.disease_patterns,
                "symptoms": cond.symptoms,
                "negated_findings": cond.negated_findings,
                "pulse": cond.pulse,
                "prior_treatment": cond.mistreatment_terms,
                "time_course": cond.time_course,
            }
            r.then_conclusions = {
                "formula": [pm["name"]],
                "treatment_principle": res.therapy_terms,
                "alternatives": [m["name"] for m in positive_mentions
                                 if m["name"] != pm["name"]],
            }
            strength_note = {"主之": "主治關係（主之）", "宜": "建議用方（宜）",
                             "可與": "斟酌可用（可與）", "與": "給予（與）",
                             "屬": "證屬（屬）"}.get(r.prescription_strength, "")
            r.interpretation = (
                f"該條可作為{pm['name']}相關方證的原文證據，原文用語為"
                f"「{pm['surface']}{ '主之' if r.prescription_strength=='主之' else ''}」，"
                f"證據強度：{strength_note}。")
            r.model_confidence = 0.9 if r.prescription_strength == "主之" else 0.82
            rules.append(r)

        # ---- 06 contraindication_rule --------------------------------------
        if res.contraindication_terms or res.contraindicated_formulas:
            r = base_rule("contraindication_rule")
            r.if_conditions = {
                "disease": res.disease_patterns, "symptoms": res.symptoms,
                "negated_findings": res.negated_findings, "pulse": res.pulse,
            }
            r.then_conclusions = {
                "contraindicated_actions": res.contraindication_terms,
                "contraindicated_formulas": res.contraindicated_formulas,
            }
            r.interpretation = "本條給出明確禁忌條件，違之則生變證。"
            r.interpretation_level = "literal"
            r.model_confidence = 0.88
            rules.append(r)

        # ---- 07 mistreatment_rule -------------------------------------------
        adverse = [t for t in res.transformation_terms if t in text]
        if res.mistreatment_types and (adverse or res.contraindication_terms or main_mention):
            r = base_rule("mistreatment_rule")
            r.if_conditions = {
                "mistreatment": res.mistreatment_terms,
                "mistreatment_type": res.mistreatment_types,
                "disease": res.disease_patterns,
            }
            r.then_conclusions = {
                "adverse_outcomes": adverse,
                "rescue_formula": [main_mention["name"]] if main_mention else [],
            }
            r.interpretation = (
                f"本條記載{'、'.join(res.mistreatment_types)}後的變證"
                + (f"，以{main_mention['name']}救治。" if main_mention else "。"))
            r.model_confidence = 0.84
            rules.append(r)

        # ---- 08 transformation_rule -----------------------------------------
        if res.transmission_terms and not res.mistreatment_types:
            r = base_rule("transformation_rule")
            r.if_conditions = {"disease": res.disease_patterns, "symptoms": res.symptoms,
                               "pulse": res.pulse, "time_course": res.time_course}
            r.then_conclusions = {"transmission": res.transmission_terms}
            r.interpretation = "本條論六經傳變或不傳之判斷。"
            r.model_confidence = 0.8
            rules.append(r)

        # ---- 09 prognosis_rule ----------------------------------------------
        if res.prognosis_terms:
            mres = RE_RESOLUTION.search(text)
            r = base_rule("prognosis_rule")
            r.if_conditions = {"disease": res.disease_patterns, "symptoms": res.symptoms,
                               "pulse": res.pulse, "time_course": res.time_course}
            r.then_conclusions = {"prognosis": res.prognosis_terms,
                                  "resolution_time": (f"從{mres.group(1)}至{mres.group(2)}上" if mres else "")}
            r.interpretation = "本條給出預後或欲解時判斷。"
            r.model_confidence = 0.8
            rules.append(r)

        # ---- 04 pulse_symptom_rule ------------------------------------------
        if res.pulse and not main_mention and not res.contraindication_terms \
           and not res.prognosis_terms and not m_out:
            r = base_rule("pulse_symptom_rule")
            r.if_conditions = {"pulse": res.pulse, "disease": res.disease_patterns}
            r.then_conclusions = {"symptoms": res.symptoms,
                                  "negated_findings": res.negated_findings}
            r.interpretation = "本條論脈與證之對應關係。"
            r.model_confidence = 0.75
            rules.append(r)

        # ---- 05 therapy_selection_rule --------------------------------------
        if res.therapy_methods:
            r = base_rule("therapy_selection_rule")
            r.if_conditions = {"disease": res.disease_patterns, "symptoms": res.symptoms,
                               "negated_findings": res.negated_findings, "pulse": res.pulse}
            r.then_conclusions = {"therapy_methods": res.therapy_methods,
                                  "therapy_terms": res.therapy_terms,
                                  "formula": [main_mention["name"]] if main_mention else []}
            r.interpretation = f"本條提示治法：{'、'.join(res.therapy_methods)}。"
            r.model_confidence = 0.8
            rules.append(r)

        # ---- 13 differential_rule (two-formula contrast inside one clause) ---
        distinct = {m["name"] for m in positive_mentions}
        if len(distinct) >= 2:
            r = base_rule("differential_rule")
            r.if_conditions = {"symptoms": res.symptoms, "pulse": res.pulse,
                               "disease": res.disease_patterns}
            r.then_conclusions = {"formulas": sorted(distinct)}
            r.interpretation = "本條於同一條文內對舉兩方以上，提示方證鑒別。"
            r.model_confidence = 0.8
            rules.append(r)

        # ---- 14 rescue_reverse_rule ------------------------------------------
        rescue_markers = ("救逆", "亡陽", "厥逆", "脈微欲絕", "四逆")
        if main_mention and any(k in text for k in rescue_markers) and \
           any(k in main_mention["name"] for k in ("四逆", "救逆", "白通", "茯苓四逆", "通脈")):
            r = base_rule("rescue_reverse_rule")
            r.if_conditions = {"symptoms": res.symptoms, "pulse": res.pulse,
                               "prior_treatment": res.mistreatment_terms}
            r.then_conclusions = {"rescue_formula": [main_mention["name"]]}
            r.interpretation = "本條屬救逆法度：陽衰陰盛或誤治壞病之挽救。"
            r.model_confidence = 0.82
            rules.append(r)

        # ---- 15 recurrence_rule ----------------------------------------------
        if clause.six_channel == "陰陽易差後勞復病" or "勞復" in text or "差後" in text:
            if clause.clause_number >= 382 or "勞復" in text or "差後" in text:
                r = base_rule("recurrence_rule")
                r.if_conditions = {"disease": res.disease_patterns or ["差後勞復"],
                                   "symptoms": res.symptoms, "pulse": res.pulse}
                r.then_conclusions = {"formula": [main_mention["name"]] if main_mention else [],
                                      "therapy_terms": res.therapy_terms}
                r.interpretation = "本條論病差後復發、勞復或陰陽易之治。"
                r.model_confidence = 0.8
                rules.append(r)

        # ---- 10/11/12 from formula blocks -------------------------------------
        for fb in clause.formula_blocks:
            if not fb.formula_name:
                continue
            if fb.composition:
                r = base_rule("formula_composition_rule")
                r.if_conditions = {"formula": [fb.formula_name]}
                r.then_conclusions = {"composition": fb.composition}
                r.evidence_span = fb.raw_text
                r.interpretation = f"{fb.formula_name}藥物組成，凡{len(fb.composition)}味。"
                r.interpretation_level = "literal"
                r.model_confidence = 0.95
                rules.append(r)
                doses = [c for c in fb.composition if c.get("dose_processing")]
                if doses:
                    r = base_rule("dosage_processing_rule")
                    r.if_conditions = {"formula": [fb.formula_name]}
                    r.then_conclusions = {"dosage_processing": doses}
                    r.evidence_span = fb.raw_text
                    r.interpretation = f"{fb.formula_name}各藥劑量與炮製要求。"
                    r.interpretation_level = "literal"
                    r.model_confidence = 0.95
                    rules.append(r)
            if fb.preparation or fb.administration:
                r = base_rule("administration_rule")
                r.if_conditions = {"formula": [fb.formula_name]}
                r.then_conclusions = {"preparation": fb.preparation,
                                      "administration": fb.administration,
                                      "post_notes": fb.post_notes}
                r.evidence_span = fb.raw_text
                r.interpretation = f"{fb.formula_name}煎法、服法與將息禁忌。"
                r.interpretation_level = "literal"
                r.model_confidence = 0.93
                rules.append(r)
        return rules

    # ------------------------------------------------------------------
    def extract_all(self, clauses: List[ShanghanClause]) -> List[InitialRule]:
        out: List[InitialRule] = []
        for c in clauses:
            out.extend(self.extract_clause_rules(c))
        return out
