"""SafetyGovernanceAgent — role-aware output governance.

Protocol requirements:
  * doctor mode    — every answer marked as 輔助性質, never a substitute for
                     clinical judgement;
  * patient mode   — never diagnose, never prescribe, never give dosages;
                     intent guard refuses diagnosis/prescription requests and
                     redirects to professional care; dosage text is redacted;
  * research mode  — every statement labelled 原文/異文/注釋/歸納/模型推理;
  * student mode   — teaching aid notice.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

ROLES = ("doctor", "researcher", "student", "patient")

DOCTOR_NOTICE = "本結果僅為古籍方證輔助匹配，不能替代醫師臨床判斷。"
PATIENT_NOTICE = ("以上內容只是中醫古籍知識的通俗介紹，不構成診斷或治療建議；"
                  "是否屬於某種證型、如何用藥，請務必由執業中醫師當面判斷。")
RESEARCH_NOTICE = "本輸出區分原文直述／版本異文／注家解釋／後世歸納／模型推理五個證據層級。"
STUDENT_NOTICE = "本內容為《傷寒論》教學輔助材料，臨床應用須在執業醫師指導下進行。"

ROLE_NOTICE = {
    "doctor": DOCTOR_NOTICE,
    "patient": PATIENT_NOTICE,
    "researcher": RESEARCH_NOTICE,
    "student": STUDENT_NOTICE,
}

# —— patient-side intent guard ————————————————————————————————
RE_DIAGNOSIS_INTENT = re.compile(
    r"(我是不是|我得了|我患了|我這是|我这是|幫我診斷|帮我诊断|診斷一下|诊断一下|"
    r"我有沒有|我有没有|是什麼病|是什么病|我該吃|我该吃)")
RE_PRESCRIPTION_INTENT = re.compile(
    r"(給我開|给我开|開個方|开个方|吃什麼藥|吃什么药|用什麼方|用什么方|"
    r"怎麼治|怎么治|怎麼用藥|怎么用药|喝什麼湯|喝什么汤|推薦.{0,4}(方|藥|药))")
RE_DOSAGE_INTENT = re.compile(
    r"(劑量|剂量|用量|幾克|几克|多少克|多少錢|吃幾|吃几|喝幾|喝几|一天.{0,3}次)")

# dose expressions to redact in patient-facing text
RE_DOSE_TEXT = re.compile(
    r"[一二三四五六七八九十百半]+(兩|两|錢|钱|銖|铢|升|合|枚|個|个|片|斤|克|分(?!類))")


def patient_intent_guard(question: str) -> Optional[Dict]:
    """Return a refusal payload if the patient question asks for
    diagnosis / prescription / dosage; otherwise None."""
    reasons = []
    if RE_DIAGNOSIS_INTENT.search(question):
        reasons.append("診斷判定")
    if RE_PRESCRIPTION_INTENT.search(question):
        reasons.append("處方用藥")
    if RE_DOSAGE_INTENT.search(question):
        reasons.append("劑量調整")
    if not reasons:
        return None
    return {
        "refused": True,
        "refused_intents": reasons,
        "message": (
            f"很抱歉，這個問題涉及{ '、'.join(reasons)}，屬於必須由執業醫師當面完成的部分，"
            "我不能在患者模式下提供。\n"
            "我可以幫您做的是：\n"
            "1. 用通俗語言解釋醫生提到的中醫術語（如「太陽表證」「六經」）；\n"
            "2. 幫您把症狀按時間和部位整理成就診時可以直接給醫生看的清單；\n"
            "3. 提醒哪些情況（如高熱不退、神志改變、嚴重脫水）需要儘快就醫。"),
        "safety_notice": PATIENT_NOTICE,
    }


def redact_for_patient(text: str) -> str:
    """Remove dosage expressions from patient-facing text."""
    return RE_DOSE_TEXT.sub("（劑量信息略，須遵醫囑）", text)


def governed(payload: Dict, role: str) -> Dict:
    """Attach role-appropriate safety annotations to an answer payload."""
    role = role if role in ROLES else "patient"
    payload = dict(payload)
    payload["mode"] = role
    payload["safety_notice"] = ROLE_NOTICE[role]
    if role == "patient":
        for key in ("answer", "explanation", "message"):
            if isinstance(payload.get(key), str):
                payload[key] = redact_for_patient(payload[key])
        # patient answers must not carry actionable prescriptions
        payload.pop("matched_formula_patterns", None)
        payload.pop("recommended_formulas", None)
    if role == "doctor":
        payload["assistive_only"] = True
    return payload
