"""發布閘門：evidence / safety / role / uncertainty / human-review 五道。

決策三態：release（放行）· release_with_warnings（放行但響亮標注）·
needs_human_review（暫停等待人工確認，run 狀態轉 paused）。
拒絕類（患者端診斷/處方）在上游安全層已攔截，此處記錄 guardrail 事件。
"""
from __future__ import annotations

from typing import Any, Dict, List

# 需要人工確認的場景（評審第 8 條）
HUMAN_REVIEW_TRIGGERS = {
    "doctor_formula_candidates": "醫師端給出候選方——輔助定位，需人工確認後發布",
    "unresolved_conflict": "方證衝突/鑒別未決——暫停並生成追問",
    "paper_generation": "論文生成——標題/論點/參考證據需人工確認",
    "citation_failure": "引用未能全部核驗——自動修復失敗時不得靜默發布",
}


def evaluate(spec, output: Dict[str, Any]) -> Dict[str, Any]:
    """對最終輸出做發布裁定。返回 {decision, gates, review_required, reasons}。"""
    gates: Dict[str, Dict] = {}
    reasons: List[str] = []
    review: List[str] = []

    # 1. evidence gate：引用核驗
    cr = output.get("citation_report") or {}
    ev_ok = bool(cr.get("ok", True))
    has_cite = bool(cr.get("has_any_citation", True))
    gates["evidence_gate"] = {"ok": ev_ok and has_cite,
                              "verified": cr.get("verified", []),
                              "unsupported": cr.get("unsupported", [])}
    if not ev_ok:
        review.append("citation_failure")
        reasons.append(HUMAN_REVIEW_TRIGGERS["citation_failure"])

    # 2. safety gate：上游攔截即記錄（攔截本身就是安全結論，可直接發布）
    refused = bool(output.get("refused"))
    gates["safety_gate"] = {"ok": True, "refused": refused,
                            "refused_intents": output.get("refused_intents", [])}

    # 3. role gate：患者端輸出不得含方劑推薦字段（上游已脫敏，此處複核）
    role_ok = True
    if spec.role == "patient":
        blob = str(output.get("answer", ""))
        role_ok = not any(k in blob for k in ("主之", "劑量", "服用", "處方"))
    gates["role_gate"] = {"ok": role_ok}
    if not role_ok:
        review.append("citation_failure")
        reasons.append("患者端輸出疑似含方藥指令，需人工複核")

    # 4. uncertainty gate：多假設未決/需要補問
    needs = bool(output.get("needs_clarification")) or \
        output.get("decision") in ("needs_more_information", "insufficient_evidence")
    gates["uncertainty_gate"] = {"ok": not needs}
    if needs and spec.role == "doctor":
        review.append("unresolved_conflict")
        reasons.append(HUMAN_REVIEW_TRIGGERS["unresolved_conflict"])

    # 5. human review gate：場景觸發
    if spec.role == "doctor" and not refused:
        blob = str(output.get("answer", "")) + str(output.get("hypotheses", ""))
        if "湯" in blob or "丸" in blob or "散" in blob:
            review.append("doctor_formula_candidates")
            reasons.append(HUMAN_REVIEW_TRIGGERS["doctor_formula_candidates"])
    if spec.mode == "paper":
        review.append("paper_generation")
        reasons.append(HUMAN_REVIEW_TRIGGERS["paper_generation"])

    review = sorted(set(review))
    if not gates["evidence_gate"]["ok"] or not role_ok:
        decision = "needs_human_review"
    elif review:
        decision = "needs_human_review"
    elif not has_cite and not refused:
        decision = "release_with_warnings"
        reasons.append("回答未含可核驗條文編號——已響亮標注")
    else:
        decision = "release"
    return {"decision": decision, "gates": gates,
            "review_required": review, "reasons": reasons,
            "note": "needs_human_review 時 run 轉 paused；"
                    "run-resume --approve 由人確認後放行（審批人記錄在案）。"}
