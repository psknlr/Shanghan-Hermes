"""HypothesisManager — 多假設並行的方證推理（不做單一答案）.

FormulaMatcher gives a ranked candidate list; this module upgrades it into
parallel hypotheses a clinician can audit:

  * support        本輪呈現中命中的核心證/兼證/脈
  * against        呈現與方證相反的表現（如 無汗 vs 桂枝湯之汗出）
  * counter_evidence_would_be
                   若出現則削弱本假設的表現（確定性生成自互斥證對）
  * missing_key_findings
                   方證核心證中尚未確認的部分——正是該追問的四診
  * confidence     高/中/低——由歸一化匹配分與反證扣減而來（D 層啟發式）

When top candidates are close, or key discriminators are unknown, the
ClarificationAgent side of this module emits 鑒別追問 instead of letting the
agent hand back a single top-1 —「先問清汗出與否，再談麻黃桂枝」.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .. import lexicon
from ..textutil import normalize_query

# tiers for the heuristic confidence label
_TIERS = [(0.6, "高"), (0.35, "中"), (0.0, "低")]


def _tier(score: float, n_conflicts: int) -> str:
    for floor, label in _TIERS:
        if score >= floor:
            # any hard conflict caps confidence at 低
            return "低" if n_conflicts and label != "低" else label
    return "低"


class HypothesisManager:
    def __init__(self, registry):
        # needs the full registry (matcher + artifacts); unwrap a scoped view
        self.reg = getattr(registry, "_base", registry)

    # ------------------------------------------------------------------
    def analyze(self, symptoms: List[str], pulse: Optional[List[str]] = None,
                six_channel: Optional[str] = None, top_k: int = 4) -> Dict:
        symptoms = [normalize_query(s) for s in (symptoms or []) if s.strip()]
        pulse = [normalize_query(p) for p in (pulse or []) if p.strip()]
        match = self.reg.matcher.match(symptoms=symptoms, pulse=pulse,
                                       six_channel=six_channel,
                                       top_k=max(top_k, 3))
        cands = match.get("matched_formula_patterns", [])[:top_k]
        rules = {r.formula: r for r in self.reg.art.formula_rules}
        presented = set(symptoms) | {p.lstrip("脈") for p in pulse}

        hypotheses = [self._hypothesis(m, rules.get(m["formula"]), presented)
                      for m in cands]
        questions = self._clarifying_questions(hypotheses, rules, presented)
        needs, why = self._needs_clarification(hypotheses, presented)
        if not hypotheses:
            decision = "insufficient_evidence"
        elif needs:
            decision = "needs_more_information"
        else:
            decision = "probable"
        return {
            "tool": "shanghan_hypotheses",
            "input": {"symptoms": symptoms, "pulse": pulse,
                      "six_channel": six_channel},
            "hypotheses": hypotheses,
            "clarifying_questions": questions,
            "needs_clarification": needs,
            "clarification_reason": why,
            "decision": decision,
            "notice": "多假設分析屬 D 層歸納，僅供醫師/教學參考，不替代臨床判斷。",
        }

    # ------------------------------------------------------------------
    def _hypothesis(self, m: Dict, rule, presented: set) -> Dict:
        hit_terms = {h.split("：", 1)[1].split("≈")[0]
                     for h in m.get("matched_findings", []) if "：" in h}
        missing: List[str] = []
        counters: List[str] = []
        if rule is not None:
            pattern = rule.core_symptoms + rule.associated_symptoms
            missing = [cs for cs in (rule.core_symptoms + rule.core_pulse)
                       if cs not in hit_terms
                       and not any(cs in p or p in cs for p in presented)][:4]
            # deterministic 反證 from mutually exclusive finding pairs:
            # the pattern expects 汗出 → 若見無汗則不支持本方
            for a, b in lexicon.CONTRADICTORY_SYMPTOMS:
                if a in pattern and b not in presented:
                    counters.append(f"若見「{b}」則不支持")
                elif b in pattern and a not in presented:
                    counters.append(f"若見「{a}」則不支持")
        return {
            "formula": m["formula"],
            "score": m.get("match_score", 0.0),
            "confidence": _tier(m.get("match_score", 0.0),
                                len(m.get("conflicts", []))),
            "six_channel": m.get("six_channel", ""),
            "support": m.get("matched_findings", []),
            "against": m.get("conflicts", []),
            "counter_evidence_would_be": counters[:3],
            "missing_key_findings": missing,
            "contraindications": m.get("contraindications", []),
            "evidence": [e.get("clause_id") for e in m.get("evidence", [])
                         if e.get("clause_id")],
        }

    # ------------------------------------------------------------------
    def _clarifying_questions(self, hyps: List[Dict], rules: Dict,
                              presented: set) -> List[str]:
        """鑒別追問：優先問互斥證對（汗出/無汗），其次問 top 假設之間
        互不共享的核心證，最後問首選假設缺失的核心證。"""
        questions: List[str] = []
        seen = set()

        def add(q):
            if q not in seen:
                seen.add(q)
                questions.append(q)

        top = [h for h in hyps[:3] if rules.get(h["formula"])]
        # 1 — mutually exclusive axes separating any two top candidates
        for i, ha in enumerate(top):
            for hb in top[i + 1:]:
                pa = set(rules[ha["formula"]].core_symptoms
                         + rules[ha["formula"]].associated_symptoms)
                pb = set(rules[hb["formula"]].core_symptoms
                         + rules[hb["formula"]].associated_symptoms)
                for a, b in lexicon.CONTRADICTORY_SYMPTOMS:
                    if a in presented or b in presented:
                        continue
                    if (a in pa and b in pb):
                        add(f"是「{a}」還是「{b}」？（{a}→{ha['formula']}；"
                            f"{b}→{hb['formula']}）")
                    elif (b in pa and a in pb):
                        add(f"是「{b}」還是「{a}」？（{b}→{ha['formula']}；"
                            f"{a}→{hb['formula']}）")
                # 2 — core findings unique to one side (iterate in the
                # rule's declared order so output is deterministic)
                only_a = [f for f in rules[ha["formula"]].core_symptoms
                          if f not in pb and f not in presented][:2]
                for f in only_a:
                    add(f"是否見「{f}」？（支持{ha['formula']}）")
                only_b = [f for f in rules[hb["formula"]].core_symptoms
                          if f not in pa and f not in presented][:2]
                for f in only_b:
                    add(f"是否見「{f}」？（支持{hb['formula']}）")
        # 3 — unconfirmed core findings of the leading hypothesis
        if top:
            for f in top[0]["missing_key_findings"][:2]:
                add(f"是否見「{f}」？（{top[0]['formula']}核心證，尚未確認）")
        return questions[:6]

    @staticmethod
    def _needs_clarification(hyps: List[Dict], presented: set):
        if not hyps:
            return True, "無候選方證，證據不足"
        top = hyps[0]
        if len(presented) < 3:
            return True, "呈現的四診信息過少（不足3項）"
        if top["score"] < 0.35:
            return True, "首選假設匹配度偏低"
        if len(hyps) >= 2 and top["score"] - hyps[1]["score"] < 0.15:
            return True, (f"前兩位候選（{top['formula']}/{hyps[1]['formula']}）"
                          "評分接近，需鑒別追問")
        if len(top["missing_key_findings"]) >= 2:
            return True, f"{top['formula']}的核心證尚有未確認項"
        return False, ""


def render_hypotheses(payload: Dict) -> str:
    """Deterministic text block for answers / teaching output."""
    lines = ["【多假設方證分析（D 層歸納，輔助性質）】"]
    for i, h in enumerate(payload.get("hypotheses", []), 1):
        lines.append(f"假設{i}：{h['formula']}（匹配度 {h['score']}，"
                     f"置信 {h['confidence']}）")
        if h["support"]:
            lines.append(f"  支持：{'；'.join(h['support'][:4])}")
        if h["against"]:
            lines.append(f"  反證：{'；'.join(h['against'][:2])}")
        if h["counter_evidence_would_be"]:
            lines.append(f"  何種表現會削弱本假設：{'；'.join(h['counter_evidence_would_be'][:2])}")
        if h["missing_key_findings"]:
            lines.append(f"  尚未確認：{'、'.join(h['missing_key_findings'])}")
        if h["evidence"]:
            lines.append(f"  證據條文：{'、'.join(h['evidence'][:3])}")
    qs = payload.get("clarifying_questions", [])
    if payload.get("needs_clarification") and qs:
        lines.append(f"【鑒別追問】（{payload.get('clarification_reason','')}）")
        for q in qs:
            lines.append(f"  - {q}")
    return "\n".join(lines)
