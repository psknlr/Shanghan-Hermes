"""Patient-education mode — 通俗解釋, with hard safety rails.

Capabilities (per protocol): explain TCM terms in plain language, organize
symptoms for a doctor visit, give risk-signal reminders.
Forbidden: diagnosis, prescription, dosage advice — enforced by
safety.patient_intent_guard before any content generation, plus dosage
redaction on the way out.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import safety
from ..schemas import ShanghanClause, SixChannelRule
from ..textutil import normalize_query

# Plain-language glossary (model-authored educational content, labelled E層).
GLOSSARY: Dict[str, Dict] = {
    "太陽表證": {
        "plain": "「太陽表證」是中醫對一類剛起病、病位偏「表」(體表層面)狀態的叫法，"
                 "常見於受涼感冒初期。典型表現是怕冷、發熱、頭痛、脖子後面發緊、脈摸起來偏浮。",
        "directions": ["有汗、怕風的一類，古書稱「中風」（與現代腦中風不是一回事）",
                       "無汗、怕冷明顯、全身痠痛的一類，古書稱「傷寒」"],
        "channel": "太陽病",
    },
    "太陽病": {"alias": "太陽表證"},
    "表證": {"alias": "太陽表證"},
    "六經": {
        "plain": "「六經」是《傷寒論》把外感病按深淺和性質分成的六個階段/類型：太陽、陽明、少陽、"
                 "太陰、少陰、厥陰。它是一個分類框架，幫助醫生判斷病走到哪一步、該往哪個方向治。",
        "directions": ["太陽：病在表（初期）", "陽明：熱與實為主", "少陽：半表半裏",
                       "太陰：脾胃虛寒", "少陰：心腎虛衰", "厥陰：寒熱錯雜"],
    },
    "六經辨證": {"alias": "六經"},
    "陽明病": {
        "plain": "「陽明病」指病邪入裏化熱的階段，常見高熱、大汗、口渴、便祕等表現。"
                 "醫生說到它，多半是描述發熱性疾病進入「熱、實」為主的階段。",
        "channel": "陽明病",
    },
    "少陽病": {
        "plain": "「少陽病」指病在「半表半裏」的狀態，典型表現是一陣冷一陣熱、口苦、咽乾、"
                 "頭暈、胸脅悶脹、不想吃東西。",
        "channel": "少陽病",
    },
    "少陰病": {
        "plain": "「少陰病」多指身體機能（特別是心腎陽氣）明顯虛衰的階段，"
                 "人會特別怕冷、沒精神、總想睡、脈摸起來微弱。這通常提示病情較深，需要及時就醫。",
        "channel": "少陰病",
    },
    "中風": {
        "plain": "在《傷寒論》裏，「中風」指受風邪引起的有汗型感冒樣表證（發熱、出汗、怕風），"
                 "和現代醫學說的「腦中風/卒中」完全是兩回事，不要混淆。",
    },
    "傷寒": {
        "plain": "「傷寒」在古書裏有兩個用法：廣義指各種外感熱病；狹義指無汗、怕冷明顯的"
                 "受寒型表證。它和現代醫學的「傷寒桿菌感染（腸傷寒）」不是同一概念。",
    },
    "方證": {
        "plain": "「方證」是指某個經方對應的一組典型表現。醫生說「你是桂枝湯證」，意思是"
                 "你的表現組合與古籍裏桂枝湯所治的那一類情況相似，而不是給病貼了現代診斷標籤。",
    },
    "和解": {"plain": "「和解」是中醫治法之一，用於病位在半表半裏、不適合單純發汗或瀉下的情況，"
                  "代表思路是調和身體內外，使邪氣有出路。"},
    "誤治": {"plain": "「誤治」指治療方法與病情不匹配（比如該發汗時用了瀉下），"
                  "古書記載誤治後病情可能發生變化，需要重新辨證。這提醒我們用藥要遵醫囑。"},
}

RISK_SIGNALS = [
    "持續高熱不退（超過3天）或體溫超過39.5℃",
    "精神萎靡、嗜睡、煩躁不安或神志改變",
    "嚴重嘔吐腹瀉導致無法進食飲水、尿量明顯減少",
    "呼吸困難、胸痛、口唇發紫",
    "皮膚黏膜出血點、劇烈頭痛、頸項強直",
    "孕婦、嬰幼兒、高齡或有慢性基礎病者症狀加重",
]


class PatientEducator:
    def __init__(self, six_channel_rules: Optional[List[SixChannelRule]] = None,
                 clause_store: Optional[Dict[str, ShanghanClause]] = None):
        self.scrs = {r.six_channel: r for r in (six_channel_rules or [])}
        self.clauses = clause_store or {}

    def _lookup(self, term: str) -> Optional[Dict]:
        entry = GLOSSARY.get(term)
        seen = set()
        while entry and "alias" in entry and entry["alias"] not in seen:
            seen.add(entry["alias"])
            entry = GLOSSARY.get(entry["alias"])
        return entry

    def explain(self, question: str) -> Dict:
        # 1 — intent guard first: refuse diagnosis/prescription/dosage asks
        refusal = safety.patient_intent_guard(question)
        if refusal:
            return safety.governed(refusal, "patient")

        q = normalize_query(question)
        matched_term, entry = None, None
        for term in sorted(GLOSSARY, key=len, reverse=True):
            if term in q or term in question:
                matched_term, entry = term, self._lookup(term)
                break

        if entry is None:
            payload = {
                "answer": ("這個术语暂时不在患者教育词库中。建议您把医生的原话记下来，"
                           "下次就诊时请医生当面解释；也可以问我「六經」「太陽表證」「方證」"
                           "这类《傷寒論》常见概念。"),
                "can_explain": sorted({k for k, v in GLOSSARY.items() if "alias" not in v}),
                "risk_reminders": RISK_SIGNALS[:3],
            }
            return safety.governed(payload, "patient")

        explanation = entry["plain"]
        payload: Dict = {
            "term": matched_term,
            "answer": explanation,
            "common_understandings": entry.get("directions", []),
            "what_this_is": "這是中醫辨證術語的通俗介紹（教育內容，模型解釋層）",
            "what_to_do": [
                "把醫生的判斷和您自己的感受記錄下來，複診時帶上",
                "如果對診斷或用藥有疑問，直接向開方醫師確認",
                "服藥期間出現新的不適，及時告知醫生",
            ],
            "risk_reminders": RISK_SIGNALS,
            "not_provided": ["診斷判定", "處方建議", "劑量調整"],
        }
        # channel-grounded original text shown as cultural reference only
        channel = entry.get("channel")
        scr = self.scrs.get(channel) if channel else None
        if scr and scr.outline_text:
            payload["classical_reference"] = {
                "note": "古籍原文僅供了解概念來源，不用於自我判斷",
                "clause_id": scr.outline_clause_id,
                "text": scr.outline_text,
            }
        return safety.governed(payload, "patient")

    def organize_symptoms(self, symptoms: List[str]) -> Dict:
        """Help a patient prepare a visit summary — no interpretation."""
        cleaned = [s.strip() for s in symptoms if s and s.strip()]
        payload = {
            "visit_summary": {
                "主要不適（按您提供的順序）": cleaned,
                "建議補充給醫生的信息": [
                    "每個症狀開始的時間和變化趨勢",
                    "是否怕冷/怕風、出汗情況、口渴與否、大小便情況、睡眠",
                    "已用過的藥物及效果",
                    "既往疾病和過敏史",
                ],
            },
            "answer": "已幫您把症狀整理成就診清單。請帶給醫生面診時參考，"
                      "我不會也不能據此判斷您屬於什麼證型。",
            "risk_reminders": RISK_SIGNALS,
        }
        return safety.governed(payload, "patient")
