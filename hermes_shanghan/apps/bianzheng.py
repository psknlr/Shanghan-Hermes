"""方證辨證閉環（B 組）：四診採集 · 多假設裁決 · 方證衝突審計 · 誤治傳變模擬。

四個確定性能力，全部錨定 clause_id、沿用既有規則資產：

1. ``intake_parse``      自然敘述 → 結構化四診表 + 缺失關鍵信息 + 追問
                         （患者端唯一暴露項：只整理就診信息，不做匹配）
2. ``adjudicate``        多假設 → 三態裁決（傾向 A / 傾向 B / 不能裁決）
                         + 為什麼還不能定方 + 三個關鍵追問
3. ``conflict_audit``    候選方 × 呈現表現 → 衝突項/衝突條文/強度/是否禁忌/
                         改判候選/應補問（比 top-k 更安全的醫師輔助定位）
4. ``mistreatment_simulate``  誤治 → 變證分支 → 救逆方 → 條文依據；
                         多步為組合視圖（每步單獨錨定原文，鏈本身非原文連續敘述）
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import config, lexicon
from ..schemas import read_jsonl
from ..textutil import fold_variants, normalize_query

# ---------------------------------------------------------------------------
# 1. 四診信息採集（信息整理，非診斷）
# ---------------------------------------------------------------------------
_AXIS_KEYS = {
    "cold_heat": ("寒", "熱", "厥", "惡風"),
    "sweating": ("汗",),
    "thirst_drinking": ("渴", "飲水", "欲飲"),
    "stool_urine": ("大便", "小便", "下利", "溏", "秘", "溲"),
    "chest_hypochondrium": ("胸", "脅"),
    "epigastrium_abdomen": ("心下", "腹", "嘔", "吐", "食"),
    "pain_location": ("痛", "疼"),
    "sleep": ("眠", "寐", "煩躁"),
    "tongue": ("舌",),
}
_MISSING_QUESTIONS = {
    "cold_heat": "是否惡寒/惡風或發熱？寒熱是否往來？",
    "sweating": "有汗還是無汗？（此為表虛/表實關鍵鑒別）",
    "thirst_drinking": "口渴嗎？喜熱飲還是冷飲、飲多飲少？",
    "stool_urine": "大便、小便情況如何？",
    "pulse": "脈象如何（浮/沉、遲/數、有力/無力）？",
    "tongue": "舌質舌苔如何？",
}
# 現代口語 → 古籍術語（僅覆蓋高頻就診用語；映射透明可審）
MODERN_TO_CLASSICAL = {
    "怕冷": "惡寒", "怕風": "惡風", "出汗": "汗出", "沒有汗": "無汗",
    "不出汗": "無汗", "腹瀉": "下利", "拉肚子": "下利", "噁心": "欲嘔",
    "惡心": "欲嘔", "想吐": "欲嘔", "睡不著": "不得眠", "失眠": "不得眠",
    "沒胃口": "不欲飲食", "吃不下": "不欲飲食", "便秘": "不大便",
    "心慌": "悸", "頭暈": "眩",
}
RE_MISTREAT_HISTORY = re.compile(
    r"(發汗後|汗後|下之後|下後|吐後|吐下後|誤汗|誤下|誤吐|燒針|溫針|火劫)")
RE_MED_RESPONSE = re.compile(
    r"([服吃][^，。；]{0,10}?(藥|湯|丸|散)[^，。；]{0,4}?[後后][^，。；]{0,12})")


def modernize(text: str) -> str:
    """口語→古籍術語映射（**最長優先**——十九輪修復：「不出汗」必須先於
    「出汗」匹配，否則被劫持為陽性「汗出」）。"""
    out = normalize_query(text)
    for modern, classical in sorted(MODERN_TO_CLASSICAL.items(),
                                    key=lambda kv: -len(kv[0])):
        out = out.replace(modern, classical)
    return out


def intake_parse(text: str) -> Dict:
    """自然敘述 → 結構化四診表。只整理信息，不做任何診斷/方證匹配。"""
    raw = modernize(text)
    folded = fold_variants(raw)

    found: List[str] = []
    consumed = folded
    for term in lexicon.SYMPTOMS:          # 詞表已按最長優先排序
        if term in consumed:
            found.append(term)
            consumed = consumed.replace(term, "□" * len(term))
    pulses = [p for p in lexicon.PULSE_NAMED_PATTERNS if p in folded]
    if not pulses:
        m = re.search(r"脈([㐀-鿿]{1,6})", folded)
        if m:
            pulses = ["脈" + m.group(1)]

    table: Dict[str, List[str]] = {k: [] for k in _AXIS_KEYS}
    other: List[str] = []
    for s in found:
        axis = next((k for k, keys in _AXIS_KEYS.items()
                     if any(x in s for x in keys)), None)
        (table[axis] if axis else other).append(s)

    timeline = lexicon.RE_TIME_COURSE.findall(folded)
    mistreat = RE_MISTREAT_HISTORY.findall(folded)
    med = [m[0] for m in RE_MED_RESPONSE.findall(raw)]

    missing = [k for k in ("cold_heat", "sweating", "thirst_drinking",
                           "stool_urine") if not table[k]]
    if not pulses:
        missing.append("pulse")
    if not table["tongue"]:
        missing.append("tongue")
    return {
        "chief_complaint": "、".join(found[:3]) or raw[:20],
        "timeline": timeline[:6],
        **{k: v for k, v in table.items()},
        "other_findings": other[:8],
        "pulse": pulses,
        "prior_mistreatment": sorted(set(mistreat)),
        "medication_response": med[:3],
        "missing_key_findings": missing,
        "next_questions": [_MISSING_QUESTIONS[k] for k in missing
                           if k in _MISSING_QUESTIONS][:4],
        "note": "本表僅為就診信息整理（確定性詞表抽取），不構成診斷；"
                "四診axes留空表示敘述中未提及，正是應補問之處。",
    }


# ---------------------------------------------------------------------------
# 2. 方證多假設裁決（為什麼還不能定方）
# ---------------------------------------------------------------------------
def adjudicate(symptoms: List[str], pulse: Optional[List[str]] = None,
               six_channel: str = "", registry=None) -> Dict:
    from ..agent.hypothesis import HypothesisManager
    if registry is None:
        from ..agent.tools import get_registry
        registry = get_registry()
    base = HypothesisManager(registry).analyze(
        symptoms, pulse=pulse, six_channel=six_channel or None, top_k=3)
    hyps = base.get("hypotheses", [])
    for h in hyps[:3]:
        chk = registry.call("shanghan_contraindication_check",
                            {"formula": h.get("formula", ""),
                             "symptoms": symptoms})
        h["contraindication_hits"] = (
            chk.get("symptom_conflicts", []) if isinstance(chk, dict) else [])

    def _score(h):
        return float(h.get("normalized_score") or h.get("score")
                     or h.get("match_score") or 0)

    if not hyps:
        verdict, rationale = "不能裁決", "無候選方證（證據不足）"
    elif base.get("needs_clarification"):
        verdict = "不能裁決"
        rationale = ("關鍵鑒別信息缺失：" + (base.get("clarification_reason") or "")
                     + "；候選接近，先補問再定。")
    else:
        s0 = _score(hyps[0])
        s1 = _score(hyps[1]) if len(hyps) > 1 else 0.0
        h0_clean = not hyps[0].get("against") and not hyps[0]["contraindication_hits"]
        if s0 >= s1 * 1.3 and h0_clean:
            verdict = f"傾向 {hyps[0].get('formula', '')}"
            rationale = "首選評分明顯領先且無反證/禁忌衝突。"
        elif len(hyps) > 1 and not hyps[1].get("against") and hyps[0].get("against"):
            verdict = f"傾向 {hyps[1].get('formula', '')}"
            rationale = "首選存在反證/衝突，次選證據面更乾淨。"
        else:
            verdict = "不能裁決"
            rationale = "候選評分接近或均有缺失關鍵證，須補充四診後再判。"
    # 推薦處方列表（十九輪）：歸一化推薦度 + 各候選的支持/反證/缺失
    # 證據 + 該方專屬追問點——「為什麼是它、還差什麼、該問什麼」一屏呈現
    max_s = max((_score(h) for h in hyps), default=0.0) or 1.0
    recommendations = []
    for rank, h in enumerate(hyps, 1):
        penal = 0.0
        if h.get("against"):
            penal += 0.25
        if h.get("contraindication_hits"):
            penal += 0.35
        pct = max(0.0, round(_score(h) / max_s * (1 - penal) * 100))
        questions = [f"是否見「{m}」？（{h.get('formula', '')} 的關鍵指徵）"
                     for m in h.get("missing_key_findings", [])[:3]]
        for a in h.get("against", [])[:1]:
            questions.append(f"「{a}」是否確切？此為該方反證，須再核。")
        recommendations.append({
            "rank": rank,
            "formula": h.get("formula", ""),
            "recommendation_pct": pct,
            "support": h.get("support", []),
            "against": h.get("against", []),
            "missing_key_findings": h.get("missing_key_findings", []),
            "contraindication_hits": h.get("contraindication_hits", []),
            "supporting_clauses": (h.get("supporting_clauses")
                                   or h.get("evidence_clauses") or [])[:4],
            "follow_up_questions": questions[:4],
        })
    return {
        "input": base.get("input", {}),
        "candidates": hyps,
        "recommendations": recommendations,
        "verdict": verdict,
        "rationale": rationale,
        "why_not_prescribe": [
            f"{h.get('formula', '')}：缺 " + "、".join(
                h.get("missing_key_findings", [])[:3])
            for h in hyps if h.get("missing_key_findings")][:3],
        "key_questions": base.get("clarifying_questions", [])[:3],
        "note": "裁決為 D 層確定性規則（評分差距+反證+禁忌），核心目的是"
                "說明「為什麼還不能定方」；推薦度為域內歸一化排序值，"
                "非療效概率；不替代臨床判斷、不構成處方。",
    }


# ---------------------------------------------------------------------------
# 3. 方證衝突審計（比匹配更安全的定位）
# ---------------------------------------------------------------------------
def conflict_audit(formula: str, symptoms: List[str],
                   pulse: Optional[List[str]] = None, registry=None) -> Dict:
    if registry is None:
        from ..agent.tools import get_registry
        registry = get_registry()
    q = normalize_query(formula)
    rules = read_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")
    rule = next((r for r in rules
                 if fold_variants(r.get("formula", "")) == fold_variants(q)), None)
    if rule is None:
        return {"error": f"未找到方劑 {formula}"}
    presented = [normalize_query(s) for s in symptoms or []]
    expected = set(rule.get("core_symptoms", []))
    assoc = set(rule.get("associated_symptoms", []))

    mutex: Dict[str, str] = {}
    for a, b in lexicon.CONTRADICTORY_SYMPTOMS:
        mutex[a] = b
        mutex[b] = a
    conflicts = []
    for p in presented:
        partner = mutex.get(p)
        opp = None
        if partner and partner in expected | assoc:
            opp = partner
        elif p.startswith(("無", "不")) and p.lstrip("無不") in expected | assoc:
            opp = p.lstrip("無不")
        elif ("無" + p) in expected | assoc or ("不" + p) in expected | assoc:
            opp = next(x for x in ("無" + p, "不" + p) if x in expected | assoc)
        if opp:
            strength = "核心證衝突" if opp in expected else "兼證衝突"
            conflicts.append({
                "presented": p, "pattern_expects": opp, "strength": strength,
                "supporting_clauses": rule.get("supporting_clauses", [])[:3]})

    chk = registry.call("shanghan_contraindication_check",
                        {"formula": rule["formula"], "symptoms": presented})
    contra = chk if isinstance(chk, dict) else {}

    # 改判候選：核心證包含衝突表現的其他方（如 無汗 → 麻黃湯）
    alternatives = []
    for c in conflicts:
        for r in rules:
            if r["formula"] != rule["formula"] and \
                    any(c["presented"] in cs
                        for cs in r.get("core_symptoms", [])):
                alternatives.append({"conflict": c["presented"],
                                     "candidate": r["formula"],
                                     "supporting_clauses":
                                         r.get("supporting_clauses", [])[:2]})
    seen = set()
    alternatives = [a for a in alternatives
                    if not (a["candidate"] in seen or seen.add(a["candidate"]))][:4]

    ask = [mutex[e] and f"是「{e}」還是「{mutex[e]}」？"
           for e in sorted(expected) if e in mutex
           and e not in presented and mutex[e] not in presented][:3]
    # 嚴重度只看「呈現表現觸發的」衝突：方劑固有禁例（如桂枝湯之酒客）
    # 屬常備信息，未被本次呈現觸發時不升級嚴重度
    triggered = contra.get("symptom_conflicts", [])
    severity = ("高（存在核心證衝突或觸發禁例）"
                if any(c["strength"] == "核心證衝突" for c in conflicts)
                or triggered else
                ("中（兼證衝突）" if conflicts else "無衝突"))
    return {
        "formula": rule["formula"],
        "presented": presented,
        "conflicts": conflicts,
        "contraindications": contra.get("formula_contraindications", []),
        "therapy_law_bans": contra.get("therapy_law_bans", []),
        "severity": severity,
        "reassign_candidates": alternatives,
        "should_ask": ask,
        "note": "衝突判定基於互斥證對與方證規則（D 層），條文可回源；"
                "「改判候選」僅為定位提示，不構成處方建議。",
    }


# ---------------------------------------------------------------------------
# 4. 誤治傳變路徑模擬
# ---------------------------------------------------------------------------
def mistreatment_simulate(channel: str = "太陽病", mistreatment: str = "",
                          steps: int = 1) -> Dict:
    rules = read_jsonl(config.RULES_MISTREATMENT_DIR / "mistreatment_rules.jsonl")
    channel = normalize_query(channel) or "太陽病"
    mtype = normalize_query(mistreatment)

    def _branches(mt_filter: str) -> List[Dict]:
        out = []
        for r in rules:
            scope = r.get("six_channel_scope") or []
            if scope and channel not in scope:
                continue
            if mt_filter and mt_filter not in r.get("mistreatment_type", ""):
                continue
            out.append({
                "mistreatment": r.get("mistreatment_type", ""),
                "resulting_pattern": r.get("resulting_pattern", ""),
                "manifestations": r.get("manifestations", [])[:5],
                "rescue_formulas": r.get("rescue_formulas", []),
                "supporting_clauses": r.get("supporting_clauses", [])[:3],
            })
        out.sort(key=lambda b: (b["mistreatment"], b["resulting_pattern"]))
        return out

    step1 = _branches(mtype)
    if not step1:
        available = sorted({r.get("mistreatment_type", "") for r in rules
                            if not r.get("six_channel_scope")
                            or channel in r["six_channel_scope"]})
        return {"channel": channel, "mistreatment": mtype,
                "error": "該經無此誤治規則", "available_types": available}

    result = {
        "channel": channel,
        "mistreatment": mtype or "（全部誤治類型）",
        "n_branches": len(step1),
        "paths": [{"path": [channel, b["mistreatment"],
                            b["resulting_pattern"],
                            "、".join(b["rescue_formulas"]) or "（無明文救逆方）"],
                   **b} for b in step1],
        "note": "每條路徑（誤治→變證→救逆方）逐條錨定原文條文。",
    }
    if steps > 1:
        # 多步為組合視圖：每步各自有原文依據，但「連續誤治」的鏈本身
        # 非原文連續敘述——如實標註，不冒充 A 層。
        others = sorted({r.get("mistreatment_type", "") for r in rules
                         if (not r.get("six_channel_scope")
                             or channel in r["six_channel_scope"])
                         and r.get("mistreatment_type") != (mtype or None)})
        result["further_steps"] = {
            "available_second_mistreatments": [o for o in others if o != mtype],
            "note": "多步鏈為組合視圖（假設路徑）：每步單獨錨定條文，"
                    "但鏈的連續性屬推演而非原文直述，使用時須聲明。"}
    return result
