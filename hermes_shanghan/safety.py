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
    r"我有沒有|我有没有|是什麼病|是什么病|我該吃|我该吃|"
    r"(我|我這|我这)[^。？?]{0,12}是不是[^。？?]{0,8}(證|证|湯證|汤证))")
RE_PRESCRIPTION_INTENT = re.compile(
    r"(給我開|给我开|開個方|开个方|吃什麼藥|吃什么药|用什麼方|用什么方|"
    r"怎麼治|怎么治|怎麼用藥|怎么用药|喝什麼湯|喝什么汤|推薦.{0,4}(方|藥|药)|"
    r"能不能(喝|吃|用|服)|可不可以(喝|吃|用|服)|(適|适)不(適|适)合我|"
    r"我能.{0,4}(喝|吃|用|服))")
RE_DOSAGE_INTENT = re.compile(
    r"(劑量|剂量|用量|幾克|几克|多少克|多少錢|吃幾|吃几|喝幾|喝几|一天.{0,3}次|"
    r"每日.{0,3}次|加量|減量|减量|停藥|停药)")

# —— patient-side red-flag triage ————————————————————————————————
# Danger signs that must escalate to 立即就醫 BEFORE any classical-text
# discussion. Two tiers: always-urgent signs, and vulnerable populations that
# become urgent when combined with symptom/medication context.
RE_RED_FLAG_URGENT = re.compile(
    r"(高[熱燒烧]不退|持續高[熱烧燒]|39\.?5|40\s*度|呼吸困難|呼吸困难|喘不過氣|"
    r"喘不过气|胸痛|胸口痛|口唇發[紫绀]|口唇发[紫绀]|意識(不清|模糊)|意识(不清|模糊)|"
    r"神志(不清|改變|改变)|昏迷|昏厥|抽搐|驚厥|惊厥|嘔血|呕血|便血|咯血|吐血|"
    r"劇烈頭痛|剧烈头痛|頸項強直|颈项强直|尿量明顯減少|尿量明显减少|無法進食|无法进食|"
    r"嚴重脫水|严重脱水|出血點|出血点)")
RE_VULNERABLE = re.compile(
    r"(孕婦|孕妇|懷孕|怀孕|妊娠|哺乳|新生兒|新生儿|嬰兒|婴儿|嬰幼兒|婴幼儿|"
    r"幼兒|幼儿|寶寶|宝宝|(我|家)?(小)?孩子?|兒童|儿童|老人|高齡|高龄)")
RE_SYMPTOM_CONTEXT = re.compile(
    r"(發[熱燒烧]|发[热烧]|惡寒|恶寒|怕冷|咳|嘔|呕|吐|下利|腹瀉|腹泻|腹痛|頭痛|头痛|"
    r"出汗|無汗|无汗|喝.{0,6}[湯汤藥药]|吃.{0,6}藥|吃.{0,6}药|用藥|用药)")


def red_flag_triage(question: str) -> Optional[Dict]:
    """Return an urgent-care triage payload when the patient question carries
    danger signs（紅旗症狀）or a vulnerable population + symptom/medication
    context. Runs BEFORE the intent guard: 就醫優先於一切古籍討論."""
    urgent = RE_RED_FLAG_URGENT.search(question)
    vulnerable = RE_VULNERABLE.search(question) and (
        RE_SYMPTOM_CONTEXT.search(question)
        or RE_PRESCRIPTION_INTENT.search(question)
        or RE_DOSAGE_INTENT.search(question))
    if not urgent and not vulnerable:
        return None
    flags: List[str] = []
    if urgent:
        flags.append(f"危險徵象：{urgent.group(0)}")
    if vulnerable:
        flags.append(f"重點人群：{RE_VULNERABLE.search(question).group(0)}")
    return {
        "refused": True,
        "urgent": bool(urgent),
        "red_flags": flags,
        "message": (
            "您描述的情況包含需要優先由醫生當面評估的信號（"
            + "；".join(flags) + "）。\n"
            "請不要依靠古籍知識自行處理："
            + ("建議儘快就醫（必要時急診）。\n" if urgent
               else "重點人群用藥風險更高，請務必先諮詢執業醫師。\n")
            + "就診前我可以幫您：\n"
              "1. 把症狀按「開始時間—部位—性質—伴隨表現」整理成清單，供醫生快速了解；\n"
              "2. 用通俗語言解釋醫生提到的中醫術語；\n"
              "3. 提醒就診時值得主動告知的信息（過敏史、正在服用的藥物等）。"),
        "safety_notice": PATIENT_NOTICE,
    }

# dose expressions to redact in patient-facing text — covers classical units
# (三兩/半升), arabic-numeral doses (3克/10g/5 ml), and frequency schedules
# (每日三次/一天2次/bid/tid)
RE_DOSE_TEXT = re.compile(
    r"[一二三四五六七八九十百半]+(兩|两|錢|钱|銖|铢|升|合|枚|個|个|片|斤|克|分(?!類))"
    r"|\d+(?:\.\d+)?\s*(克|毫克|毫升|兩|两|錢|钱|[gG]|mg|ml|mL)(?![a-zA-Z])"
    r"|(每日|每天|一天|一日)\s*[一二三四五六七八九十\d]+\s*次"
    r"|\b[bt]id\b")


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
        # claim-binding payload carries answer sentences — redact those too
        claims = payload.get("claims")
        if isinstance(claims, dict):
            claims = dict(claims)
            claims["claims"] = [
                {**c, "claim": redact_for_patient(c.get("claim", ""))}
                for c in claims.get("claims", [])]
            claims["ungrounded_claims"] = [
                redact_for_patient(c) for c in claims.get("ungrounded_claims", [])]
            payload["claims"] = claims
        # patient answers must not carry actionable prescriptions —
        # composition/administration blocks are dropped wholesale
        for key in ("matched_formula_patterns", "recommended_formulas",
                    "formula_blocks", "composition", "dose_processing",
                    "administration", "hypotheses", "clarification"):
            payload.pop(key, None)
    if role == "doctor":
        payload["assistive_only"] = True
    return payload
