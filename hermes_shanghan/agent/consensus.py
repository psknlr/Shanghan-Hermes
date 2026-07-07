"""ConsensusJudge — 把多專家輸出從「流程展示」變成「真實爭議解決」.

每位專家先給出獨立判斷（judgment）：

    {"agent": "FormulaAnalyst", "hypothesis": "麻黃湯證可能性較高",
     "support": [...], "against": [...], "evidence": [clause_ids],
     "confidence": 0.86}

裁決官再按固定評分規則合議：

    證據直接性 0-3   判斷是否錨定 A 層條文
    條文數量   0-2   證據厚度
    支持覆蓋   0-3   支持點的數量與質量
    反證衝突  −3-0   專家間分歧與判斷內反證
    安全風險  −3-0   禁忌/衝突提示
    完整度     0-2   到場專家的多樣性

輸出共識（各專家一致同意的方向）、分歧（誰與誰不一致、為什麼）、
must_verify（下一步必須確認的四診）與 final_confidence/decision——
醫生端看到的是一份可審計的合議記錄，而不是單一答案。
"""
from __future__ import annotations

from typing import Dict, List, Optional


class ConsensusJudge:
    # ------------------------------------------------------------------
    def adjudicate(self, judgments: List[Dict],
                   contraindication_notes: Optional[List[str]] = None,
                   must_verify: Optional[List[str]] = None) -> Dict:
        notes = contraindication_notes or []
        must = list(dict.fromkeys(must_verify or []))[:5]
        by_agent = {j["agent"]: j for j in judgments}
        dominant = self._dominant(judgments)

        consensus, disagreements = self._alignments(judgments, by_agent)

        # —— rubric ————————————————————————————————————————————
        directness = 3 if (dominant and dominant.get("evidence")) else \
            (1 if any(j.get("evidence") for j in judgments) else 0)
        n_ev = len({cid for j in judgments for cid in j.get("evidence", [])})
        clause_score = min(2.0, n_ev / 3.0)
        coverage = min(3.0, len(dominant.get("support", [])) * 0.75) \
            if dominant else 0.0
        conflict_penalty = min(3.0,
                               1.5 * len(disagreements)
                               + 0.5 * len(dominant.get("against", [])
                                           if dominant else []))
        safety_penalty = min(3.0, 1.0 * len(notes))
        completeness = min(2.0, 0.5 * len(judgments))
        raw = (directness + clause_score + coverage + completeness
               - conflict_penalty - safety_penalty)
        final_confidence = round(max(0.0, min(1.0, raw / 10.0)), 2)

        if dominant is None:
            decision = "insufficient_evidence"
        elif final_confidence >= 0.65 and not must and not disagreements:
            decision = "probable"
        elif final_confidence >= 0.35:
            decision = "probable_but_needs_more_information"
        else:
            decision = "insufficient_evidence"

        return {
            "dominant_hypothesis": dominant.get("hypothesis") if dominant else None,
            "final_confidence": final_confidence,
            "decision": decision,
            "score_breakdown": {
                "evidence_directness": directness,
                "clause_count": round(clause_score, 2),
                "support_coverage": round(coverage, 2),
                "conflict_penalty": round(conflict_penalty, 2),
                "safety_penalty": round(safety_penalty, 2),
                "completeness": round(completeness, 2),
            },
            "consensus": consensus,
            "disagreements": disagreements,
            "must_verify": must,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _dominant(judgments: List[Dict]) -> Optional[Dict]:
        """FormulaAnalyst leads when present（方證判斷是臨床決策主軸）；
        otherwise the most confident evidence-backed judgment."""
        formula = next((j for j in judgments
                        if j["agent"] == "FormulaAnalyst"
                        and j.get("hypothesis")), None)
        if formula:
            return formula
        backed = [j for j in judgments if j.get("hypothesis")]
        return max(backed, key=lambda j: j.get("confidence", 0), default=None)

    # ------------------------------------------------------------------
    @staticmethod
    def _alignments(judgments: List[Dict], by_agent: Dict):
        consensus: List[str] = []
        disagreements: List[str] = []
        fa = by_agent.get("FormulaAnalyst")
        ca = by_agent.get("ChannelAnalyst")
        ma = by_agent.get("MistreatmentAnalyst")
        da = by_agent.get("DifferentialAnalyst")
        # formula ↔ channel: does the leading formula sit in the located 經?
        if fa and ca and fa.get("data_channel_scope") is not None:
            channel = (ca.get("data_channel") or "").rstrip("病")
            scope = fa.get("data_channel_scope") or ""
            if channel and channel in scope:
                consensus.append(f"方證與六經定位一致：{fa['hypothesis']}屬"
                                 f"{ca.get('data_channel')}方向")
            elif channel:
                disagreements.append(
                    f"FormulaAnalyst 傾向 {fa['hypothesis']}（經屬 {scope}），"
                    f"而 ChannelAnalyst 定位 {ca.get('data_channel')}——需覆核")
        # in-judgment counter-evidence and alternates from the formula side
        if fa:
            for alt in fa.get("close_alternatives", []):
                disagreements.append(
                    f"候選接近：{alt}與{fa['hypothesis']}評分相近，"
                    "須以鑒別追問區分")
        if da and da.get("support"):
            consensus.append("鑒別分析已給出關鍵鑒別軸："
                             + "；".join(da["support"][:2]))
        if ma and ma.get("support"):
            consensus.append("誤治風險已標註：" + "；".join(ma["support"][:1]))
        return consensus[:4], disagreements[:4]


def render_adjudication(adj: Dict) -> str:
    """Deterministic 共識/分歧/需要補充 text block for the synthesizer."""
    lines: List[str] = []
    if adj.get("consensus"):
        lines.append("◎ 共識：")
        lines += [f"  - {c}" for c in adj["consensus"]]
    if adj.get("disagreements"):
        lines.append("◎ 分歧：")
        lines += [f"  - {d}" for d in adj["disagreements"]]
    if adj.get("must_verify"):
        lines.append("◎ 需要補充確認：")
        lines += [f"  - {m}" for m in adj["must_verify"]]
    lines.append(f"◎ 合議置信度：{adj.get('final_confidence')}"
                 f"（{adj.get('decision')}）")
    return "\n".join(lines)
