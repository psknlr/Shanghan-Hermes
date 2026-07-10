"""發布閘門：evidence / safety / role / uncertainty / human-review 五道。

九輪評審重構——決策五態，且**fail-closed**：

  pass                     全部閘門通過
  pass_with_warning        放行但響亮標注（如無條文引用、句級接地率偏低）
  review_required          需人工審核 → run 轉 paused，生成 ApprovalRequest
  blocked                  硬阻斷（偽造引用/患者端方藥指令）——人工批准
                           **不可**放行，必須修復後重跑
  failed_closed            關鍵核驗對象缺失（citation_report 不存在等）——
                           缺什麼都不能當「通過」處理

人工批准後由 runner 重新執行 evidence_audit 與本閘門（帶 approved 集合）
再放行，不是簡單改狀態；批准通過的決策記 pass_after_human_review。
候選方檢測用結構化信號（hypotheses / 方證匹配類工具在調用台賬中），
不再用「湯/丸/散」關鍵詞掃描（易誤報漏報）。
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List

# 需要人工確認的場景（觸發 review_required）
HUMAN_REVIEW_TRIGGERS = {
    "doctor_formula_candidates": "醫師端給出候選方（結構化信號：方證匹配/多假設"
                                 "工具產出）——輔助定位，需人工確認後發布",
    "unresolved_conflict": "方證衝突/鑒別未決——暫停並生成追問",
    "paper_generation": "論文生成——標題/論點/參考證據需人工確認",
    "citation_failure": "引用未能全部核驗（引用未接地/引文不匹配）——"
                        "自動修復失敗時不得靜默發布",
}

# 結構化候選方信號：這些工具出現在調用台賬 = 本輪產生了方劑推薦類輸出
FORMULA_CANDIDATE_TOOLS = frozenset(
    {"shanghan_match_formula", "shanghan_hypotheses", "shanghan_adjudicate"})

CLAIM_GROUNDING_WARN = 0.6      # 句級接地率低於此值 → 響亮標注（不改變決策）

# 結構化臨床動作抽取（十一輪 六：角色安全不靠關鍵詞黑名單——先把輸出
# 轉成動作對象，再由策略裁定）。確定性正則，覆蓋推薦/劑量/煎服/加減。
import re as _re

_ACTION_PATTERNS = [
    ("medication_recommendation",
     _re.compile(r"[一-鿿]{1,6}(?:湯|丸|散|飲)(?:主之|可服|宜服|服之|"
                 r"可與|可考慮)|建議(?:服用|使用)")),
    ("dosing_instruction",
     _re.compile(r"(?:每日|一日)\s*[一二三四五六七八九十\d]+\s*(?:次|服)|"
                 r"[一二三四五六七八九十百\d]+(?:兩|銖|升|枚|克|g)\b|劑量")),
    ("administration_instruction",
     _re.compile(r"煎服|溫服|頓服|分溫|水煎|先煮|去滓")),
    ("modification_plan", _re.compile(r"加減|[去加][一-鿿]{1,4}(?:一兩|二兩|三兩)")),
    ("treatment_directive", _re.compile(r"主之|處方|服用")),
]


def clinical_actions(text: str) -> List[Dict]:
    """把回答文本掃描為結構化臨床動作清單（確定性）。"""
    out: List[Dict] = []
    for action_type, rx in _ACTION_PATTERNS:
        m = rx.search(text or "")
        if m:
            out.append({"action_type": action_type, "cue": m.group(0)[:24]})
    return out


# 角色 × 最低發布要求（十一輪 六：按角色配置發布策略，不再一刀切）
ROLE_RELEASE_POLICY = {
    "patient": "不得出現任何可執行診療動作（clinical_actions 非空即 blocked）",
    "student": "非拒答回答須有可核驗引用（無引用→review_required）",
    "researcher": "非拒答回答須有可核驗引用（無引用→review_required）",
    "doctor": "方證候選必須人工確認（doctor_formula_candidates→paused）",
}


def evaluate(spec, output: Dict[str, Any],
             approved: FrozenSet[str] = frozenset(),
             tool_names: List[str] = ()) -> Dict[str, Any]:
    """對最終輸出做發布裁定。

    ``approved``：已獲人工批准的 review 觸發鍵（由 runner 在 resume
    --approve 後重新調用本函數時傳入）。``tool_names``：本輪工具調用台賬
    中的工具名（結構化候選方檢測）。
    """
    gates: Dict[str, Dict] = {}
    reasons: List[str] = []
    review: List[str] = []
    blocked: List[str] = []

    refused = bool(output.get("refused"))
    cr = output.get("citation_report")

    # 0. fail-closed：關鍵核驗對象缺失時不得推定通過
    if not isinstance(cr, dict) and not refused:
        return {"decision": "failed_closed", "gates": {
                    "evidence_gate": {"ok": False, "missing": "citation_report"}},
                "review_required": [], "blocked_reasons": [],
                "reasons": ["citation_report 缺失——關鍵核驗對象不存在時"
                            "一律 fail-closed，不推定 ok=True"],
                "approved": sorted(approved)}
    cr = cr or {}

    # 1. evidence gate：引用核驗（默認值一律取「未通過」——fail-closed）
    ev_ok = bool(cr.get("ok", False))
    has_cite = bool(cr.get("has_any_citation", False))
    unsupported = list(cr.get("unsupported") or [])
    gates["evidence_gate"] = {"ok": ev_ok and (has_cite or refused),
                              "verified": cr.get("verified", []),
                              "unsupported": unsupported}
    if unsupported:
        # 引用了庫中不存在的條文編號 = 偽造引用：硬阻斷，不可人工放行
        blocked.append("偽造引用：條文編號無法在語料中核實（"
                       + "、".join(unsupported[:5]) + "）——必須修復後重跑")
    elif not ev_ok and not refused:
        review.append("citation_failure")
        reasons.append(HUMAN_REVIEW_TRIGGERS["citation_failure"])

    # 2. safety gate：上游攔截即記錄（攔截本身就是安全結論，可直接發布）
    gates["safety_gate"] = {"ok": True, "refused": refused,
                            "refused_intents": output.get("refused_intents", [])}

    # 3. role gate：患者端輸出不得含**任何可執行診療動作**——先抽取
    #    結構化動作再由策略裁定（不再是四個關鍵詞的黑名單）
    role_ok = True
    actions: List[Dict] = []
    if spec.role == "patient" and not refused:
        actions = clinical_actions(str(output.get("answer", "")))
        role_ok = not actions
    gates["role_gate"] = {"ok": role_ok, "clinical_actions": actions,
                          "policy": ROLE_RELEASE_POLICY.get(spec.role, "")}
    if not role_ok:
        blocked.append("role_violation：患者端輸出含可執行診療動作（"
                       + "、".join(a["action_type"] for a in actions)
                       + "）——角色隔離失效屬硬故障，人工批准不可放行")

    # 4. uncertainty gate：多假設未決/需要補問
    needs = bool(output.get("needs_clarification")) or \
        output.get("decision") in ("needs_more_information",
                                   "insufficient_evidence")
    gates["uncertainty_gate"] = {"ok": not needs}
    if needs and spec.role == "doctor":
        review.append("unresolved_conflict")
        reasons.append(HUMAN_REVIEW_TRIGGERS["unresolved_conflict"])

    # 5. human review gate：結構化場景觸發（非關鍵詞掃描）
    used = set(tool_names or []) | {t for t in
                                    (output.get("tools_used") or [])}
    if spec.role == "doctor" and not refused and \
            (output.get("hypotheses") or used & FORMULA_CANDIDATE_TOOLS):
        review.append("doctor_formula_candidates")
        reasons.append(HUMAN_REVIEW_TRIGGERS["doctor_formula_candidates"])
    if output.get("manuscript_path"):
        review.append("paper_generation")
        reasons.append(HUMAN_REVIEW_TRIGGERS["paper_generation"])

    # 角色發布策略（十一輪 六）：strict_round 下無任何可核驗引用的非拒答
    # 回答不再 pass_with_warning——古籍事實性結論必須有證據，交人工審核
    if not has_cite and not refused and not unsupported:
        review.append("citation_failure")
        reasons.append("回答未含任何可核驗條文編號（evidence_policy="
                       "strict_round：無證據鏈不放行）")

    review = sorted(set(review) - set(approved))
    warnings: List[str] = []
    grounding = (output.get("claims") or {}).get("claim_grounding_rate")
    if grounding is not None and grounding < CLAIM_GROUNDING_WARN and not refused:
        warnings.append(f"句級接地率 {grounding} 低於 {CLAIM_GROUNDING_WARN}"
                        "（詞彙級下界指標）——結論須逐句對照證據台賬")
    attribution = list(cr.get("attribution_warnings") or [])
    if attribution:
        warnings.append(f"引文歸屬存疑 {len(attribution)} 處（文字錯掛條文）"
                        "——見 citation_report.attribution_warnings")

    if blocked:
        decision = "blocked"
    elif review:
        decision = "review_required"
    elif approved:
        decision = "pass_after_human_review"
    elif warnings:
        decision = "pass_with_warning"
    else:
        decision = "pass"
    return {"decision": decision, "gates": gates,
            "review_required": review, "blocked_reasons": blocked,
            "reasons": reasons + warnings, "approved": sorted(approved),
            "note": "review_required→paused，run-resume --approve 後重新過閘"
                    "（pass_after_human_review）；blocked/failed_closed 不可"
                    "人工放行，必須修復後重跑。"}
