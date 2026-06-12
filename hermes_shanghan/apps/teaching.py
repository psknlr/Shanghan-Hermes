"""Teaching-mode 六經學習 lesson builder.

Produces a structured lesson per channel: 綱領 → 亞型 → 主方 → 誤治變證 →
禁忌法度 → 條文證據 → 自動練習題 (quiz generated from verified rules with
answers grounded in clause IDs).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from .. import config, safety
from ..schemas import (FormulaPatternRule, MistreatmentTransformationRule,
                       ShanghanClause, SixChannelRule)


class TeachingBuilder:
    def __init__(self, clauses: List[ShanghanClause],
                 six_channel_rules: List[SixChannelRule],
                 formula_rules: List[FormulaPatternRule],
                 mistreatment_rules: List[MistreatmentTransformationRule]):
        self.clauses = {c.clause_id: c for c in clauses}
        self.scrs = {r.six_channel: r for r in six_channel_rules}
        self.fprs = formula_rules
        self.mtrs = mistreatment_rules

    # ------------------------------------------------------------------
    def _quiz(self, channel: str, scr: SixChannelRule,
              fprs: List[FormulaPatternRule], seed: int = 7) -> List[Dict]:
        rng = random.Random(seed)
        quiz: List[Dict] = []
        # Q1: outline cloze
        if scr.outline_text:
            quiz.append({
                "type": "條文填空",
                "question": f"{channel}提綱：「{scr.outline_text[:6]}……」請補全本條並指出其辨證要點。",
                "answer": scr.outline_text,
                "evidence_clause": scr.outline_clause_id,
            })
        # Q2-3: clause → formula
        candidates = [f for f in fprs if f.supporting_clauses]
        rng.shuffle(candidates)
        for f in candidates[:2]:
            cid = f.supporting_clauses[0]
            c = self.clauses.get(cid)
            if not c:
                continue
            blanked = c.clean_text.replace(f.formula, "______")
            quiz.append({
                "type": "據證選方",
                "question": f"「{blanked}」此處當用何方？",
                "answer": f.formula,
                "evidence_clause": cid,
            })
        # Q4: true/false from contradictory features
        tf_bank = {
            "太陽病": ("太陽中風證以無汗、脈浮緊為特徵。", False,
                     "太陽中風為汗出、惡風、脈浮緩；無汗脈浮緊屬太陽傷寒。"),
            "陽明病": ("陽明病提綱為『胃家實』。", True, "陽明之為病，胃家實是也。"),
            "少陽病": ("少陽病可用汗、吐、下三法治療。", False,
                     "少陽禁汗吐下，當以小柴胡湯和解。"),
            "太陰病": ("太陰病以自利不渴、腹滿時痛為特點，治宜溫之。", True,
                     "自利不渴者屬太陰，當溫之，宜服四逆輩。"),
            "少陰病": ("少陰病提綱為脈微細、但欲寐。", True, "少陰之為病，脈微細，但欲寐也。"),
            "厥陰病": ("烏梅丸僅用於治蛔，不主久利。", False, "烏梅丸……又主久利。"),
        }
        if channel in tf_bank:
            stmt, ans, why = tf_bank[channel]
            quiz.append({"type": "判斷題", "question": stmt,
                         "answer": "正確" if ans else "錯誤", "explanation": why})
        # Q5: mistreatment path
        paths = [m for m in self.mtrs if channel in m.six_channel_scope]
        if paths:
            m = paths[0]
            quiz.append({
                "type": "誤治傳變",
                "question": f"{m.mistreatment_type}後出現{m.resulting_pattern}，救治當選何方？",
                "answer": "、".join(m.rescue_formulas),
                "evidence_clause": m.supporting_clauses[0] if m.supporting_clauses else "",
            })
        return quiz

    # ------------------------------------------------------------------
    def lesson(self, channel: str) -> Dict:
        if not channel.endswith("病"):
            channel = channel + "病"
        scr = self.scrs.get(channel)
        if scr is None:
            return safety.governed({
                "error": f"未找到{channel}的六經規則；可選：{'、'.join(self.scrs)}"}, "student")
        fprs = [f for f in self.fprs if channel in f.six_channel_scope]
        fprs.sort(key=lambda f: -len(f.supporting_clauses))
        mtrs = [m for m in self.mtrs if channel in m.six_channel_scope]

        outline_clause = self.clauses.get(scr.outline_clause_id)
        contra_examples = []
        for cid in scr.contraindication_clauses[:5]:
            c = self.clauses.get(cid)
            if c:
                contra_examples.append({"clause_id": cid, "text": c.clean_text})
        mist_examples = []
        for m in mtrs[:6]:
            mist_examples.append({
                "path": " → ".join(m.path),
                "manifestations": m.manifestations[:5],
                "clauses": m.supporting_clauses[:3],
            })

        payload = {
            "channel": channel,
            "lesson": {
                "一、綱領": {
                    "outline_clause_id": scr.outline_clause_id,
                    "outline_text": scr.outline_text,
                    "summary": scr.summary,
                    "resolution_time": scr.resolution_time,
                },
                "二、內部結構（亞型）": [
                    {"name": s["name"],
                     "anchor_formulas": s["anchor_formulas"],
                     "evidence_clauses": s["evidence_clauses"][:4],
                     "note": s.get("note", "")} for s in scr.subtypes],
                "三、主要方劑": [
                    {"formula": f.formula,
                     "core_pattern": f.core_pattern,
                     "core_symptoms": f.core_symptoms[:6],
                     "core_pulse": f.core_pulse[:3],
                     "clauses": f.supporting_clauses[:3]} for f in fprs[:10]],
                "四、誤治變證": mist_examples,
                "五、禁忌法度": contra_examples,
                "六、條文證據": [
                    {"clause_id": cid,
                     "text": self.clauses[cid].clean_text}
                    for cid in scr.core_clauses[:8] if cid in self.clauses],
                "七、練習題": self._quiz(channel, scr, fprs),
            },
            "source_levels": {
                "綱領/條文": "原文直述",
                "亞型名稱": "後世歸納（已標註）",
                "總結": "模型歸納（chapter_level_induction）",
            },
        }
        return safety.governed(payload, "student")
