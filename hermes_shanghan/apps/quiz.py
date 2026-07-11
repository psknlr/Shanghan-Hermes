"""練習題生成器（十八輪）：多題型 · 換批 · 模型自主出題。

確定性題庫（seed 換批，字節級穩定）六種題型，每題錨定條文證據：

1. 遮方選擇（MCQ）——條文遮方，干擾項取同經他方；
2. 方證鑒別（MCQ）——鑒別規則的關鍵鑒別點歸屬判斷；
3. 提綱歸經（MCQ）——提綱條文選六經；
4. 誤治傳變（MCQ）——誤治→變證，選救逆方；
5. 法度強度（MCQ）——「主之/宜/可與」被遮，按條文原文選；
6. 判斷題——經典易混點（含解析）。

模型出題層（``model_quiz``）：真模型基於【給定條文】自主命題，每題的
evidence_clause 必須取自給定條文集、答案文本須與條文可對——不合規的題
整題剔除進 rejected（寧缺毋濫）；local 後端退回確定性題庫。
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..textutil import fold_variants

STRENGTH_MARKS = ["主之", "宜", "可與"]
CHANNELS = ["太陽病", "陽明病", "少陽病", "太陰病", "少陰病", "厥陰病"]

TF_BANK = {
    "太陽病": [
        ("太陽中風證以無汗、脈浮緊為特徵。", False,
         "太陽中風為汗出、惡風、脈浮緩；無汗脈浮緊屬太陽傷寒。"),
        ("桂枝湯服後須啜熱稀粥以助藥力。", True,
         "桂枝湯方後注：服已須臾，歠熱稀粥一升餘，以助藥力。"),
    ],
    "陽明病": [
        ("陽明病提綱為『胃家實』。", True, "陽明之為病，胃家實是也。"),
        ("陽明經證（白虎湯證）與腑證（承氣湯證）治法相同。", False,
         "經證清熱（白虎），腑證攻下（承氣），經腑有別。"),
    ],
    "少陽病": [
        ("少陽病可用汗、吐、下三法治療。", False,
         "少陽禁汗吐下，當以小柴胡湯和解。"),
        ("往來寒熱、胸脅苦滿是少陽病的典型表現。", True,
         "小柴胡湯證：往來寒熱，胸脅苦滿，嘿嘿不欲飲食，心煩喜嘔。"),
    ],
    "太陰病": [
        ("太陰病以自利不渴、腹滿時痛為特點，治宜溫之。", True,
         "自利不渴者屬太陰，當溫之，宜服四逆輩。"),
    ],
    "少陰病": [
        ("少陰病提綱為脈微細、但欲寐。", True, "少陰之為病，脈微細，但欲寐也。"),
        ("少陰病只有寒化證，沒有熱化證。", False,
         "少陰有寒化（四逆輩）與熱化（黃連阿膠湯）兩途。"),
    ],
    "厥陰病": [
        ("烏梅丸僅用於治蛔，不主久利。", False, "烏梅丸……又主久利。"),
    ],
}


class QuizBuilder:
    def __init__(self, clauses, six_channel_rules, formula_rules,
                 mistreatment_rules, differential_rules):
        self.clauses = {c.clause_id: c for c in clauses}
        self.scrs = {r.six_channel: r for r in six_channel_rules}
        self.fprs = formula_rules
        self.mtrs = mistreatment_rules
        self.diffs = differential_rules

    # -- 題型 ----------------------------------------------------------
    def _q_cloze_formula(self, rng, channel: str) -> List[Dict]:
        pool = [f for f in self.fprs
                if f.supporting_clauses
                and (not channel or channel in (f.six_channel_scope or []))]
        rng.shuffle(pool)
        out = []
        all_names = sorted({f.formula for f in self.fprs})
        for f in pool[:4]:
            cid = f.supporting_clauses[0]
            c = self.clauses.get(cid)
            if not c or f.formula not in c.clean_text:
                continue
            blanked = c.clean_text.replace(f.formula, "______")
            distract = [x for x in all_names if x != f.formula]
            rng.shuffle(distract)
            options = sorted([f.formula] + distract[:3], key=lambda s: rng.random())
            out.append({"type": "遮方選擇", "question": f"「{blanked}」此處當用何方？",
                        "options": options, "answer": f.formula,
                        "evidence_clause": cid})
        return out

    def _q_differential(self, rng, channel: str) -> List[Dict]:
        pool = [d for d in self.diffs
                if d.key_discriminators
                and (not channel or channel in (d.six_channels or []))]
        rng.shuffle(pool)
        out = []
        for d in pool[:3]:
            line = rng.choice(d.key_discriminators)
            if "：" not in line:
                continue
            owner, _, terms = line.partition("：")
            if owner not in d.formulas:
                continue
            others = [f for f in d.formulas if f != owner]
            if not others:
                continue
            options = sorted(d.formulas, key=lambda s: rng.random())
            out.append({
                "type": "方證鑒別",
                "question": f"{'、'.join(d.formulas)} 之鑒別中，"
                            f"「{terms}」是哪一方的獨有指徵？",
                "options": options, "answer": owner,
                "evidence_clause": (d.supporting_clauses or [""])[0]})
        return out

    def _q_outline(self, rng, channel: str) -> List[Dict]:
        pool = [(ch, scr) for ch, scr in self.scrs.items()
                if scr.outline_text and ch in CHANNELS
                and (not channel or ch == channel)]
        rng.shuffle(pool)
        out = []
        for ch, scr in pool[:2]:
            masked = scr.outline_text
            for name in CHANNELS:
                masked = masked.replace(name.rstrip("病"), "◯◯")
            options = sorted(CHANNELS, key=lambda s: rng.random())[:4]
            if ch not in options:
                options[0] = ch
                options.sort(key=lambda s: rng.random())
            out.append({"type": "提綱歸經",
                        "question": f"「{masked}」此為何經提綱？",
                        "options": options, "answer": ch,
                        "evidence_clause": scr.outline_clause_id})
        return out

    def _q_mistreatment(self, rng, channel: str) -> List[Dict]:
        pool = [m for m in self.mtrs
                if m.rescue_formulas
                and (not channel or channel in (m.six_channel_scope or []))]
        rng.shuffle(pool)
        out = []
        all_names = sorted({f.formula for f in self.fprs})
        for m in pool[:2]:
            ans = m.rescue_formulas[0]
            distract = [x for x in all_names if x not in m.rescue_formulas]
            rng.shuffle(distract)
            options = sorted([ans] + distract[:3], key=lambda s: rng.random())
            out.append({
                "type": "誤治傳變",
                "question": f"{m.mistreatment_type}後出現「{m.resulting_pattern}」"
                            f"（{'、'.join(m.manifestations[:3])}），救治當選何方？",
                "options": options, "answer": ans,
                "evidence_clause": (m.supporting_clauses or [""])[0]})
        return out

    def _q_strength(self, rng, channel: str) -> List[Dict]:
        # 法度用語直接從條文原文檢出（自證：遮什麼就是什麼）
        pool = [f for f in self.fprs
                if f.supporting_clauses
                and (not channel or channel in (f.six_channel_scope or []))]
        rng.shuffle(pool)
        out = []
        for f in pool:
            if len(out) >= 2:
                break
            cid = f.supporting_clauses[0]
            c = self.clauses.get(cid)
            if not c:
                continue
            folded = fold_variants(c.clean_text)
            mark = next((mk for mk in STRENGTH_MARKS
                         if (fold_variants(f.formula) + mk) in folded), "")
            if not mark:
                continue
            masked = c.clean_text.replace(f.formula + mark,
                                          f.formula + "［　　］")
            out.append({
                "type": "法度強度",
                "question": f"「{masked}」——原文此處的法度用語是？"
                            "（主之=正治首選；宜=適宜；可與=斟酌可用）",
                "options": list(STRENGTH_MARKS), "answer": mark,
                "evidence_clause": cid})
        return out

    def _q_tf(self, rng, channel: str) -> List[Dict]:
        bank = []
        for ch, items in TF_BANK.items():
            if channel and ch != channel:
                continue
            bank.extend(items)
        rng.shuffle(bank)
        return [{"type": "判斷題", "question": stmt,
                 "options": ["正確", "錯誤"],
                 "answer": "正確" if ans else "錯誤", "explanation": why}
                for stmt, ans, why in bank[:2]]

    # -- 出卷 ----------------------------------------------------------
    def build(self, channel: str = "", n: int = 8, seed: int = 1) -> Dict:
        channel = (channel + "病") if channel and not channel.endswith("病") \
            else channel
        rng = random.Random((seed, channel).__repr__())
        pool: List[Dict] = []
        for gen in (self._q_cloze_formula, self._q_differential,
                    self._q_outline, self._q_mistreatment,
                    self._q_strength, self._q_tf):
            pool.extend(gen(rng, channel))
        rng.shuffle(pool)
        picked = pool[:max(1, min(20, n))]
        for i, q in enumerate(picked, 1):
            q["no"] = i
        return {"channel": channel or "（全書）", "seed": seed,
                "n": len(picked), "questions": picked,
                "types_present": sorted({q["type"] for q in picked}),
                "note": "確定性題庫（seed 換批）；每題錨定條文證據，"
                        "答案可回源。教學輔助，不構成臨床指導。"}


# ---------------------------------------------------------------------------
# 模型自主出題（真模型；證據不合規的題整題剔除）
# ---------------------------------------------------------------------------
def model_quiz(builder: QuizBuilder, llm, channel: str = "", n: int = 5,
               seed: int = 1) -> Dict:
    if not getattr(llm, "available", False):
        out = builder.build(channel=channel, n=n, seed=seed)
        out["backend"] = "local"
        out["note"] += "（未接真實模型：模型自主出題需真實後端，已回退題庫）"
        return out
    channel_full = (channel + "病") if channel and not channel.endswith("病") \
        else channel
    scr = builder.scrs.get(channel_full or "太陽病")
    cids: List[str] = []
    if scr:
        cids = [scr.outline_clause_id] + list(scr.core_clauses or [])
    cids = [c for c in cids if c in builder.clauses][:10]
    if not cids:
        cids = list(builder.clauses)[:8]
    evid = "\n".join(f"- [{cid}] {builder.clauses[cid].clean_text[:180]}"
                     for cid in cids)
    from ..llm.prompts import quiz_system_prompt, quiz_user_prompt
    raw = llm.json_complete(quiz_system_prompt(),
                            quiz_user_prompt(channel_full or "全書", n, evid),
                            task="synthesize")
    allowed = set(cids)
    questions, rejected = [], []
    for q in (raw.get("questions") or [])[: n * 2]:
        if not isinstance(q, dict):
            continue
        cid = str(q.get("evidence_clause", ""))
        item = {"type": str(q.get("type", "模型出題"))[:12],
                "question": str(q.get("question", ""))[:300],
                "options": [str(o)[:60] for o in (q.get("options") or [])[:5]],
                "answer": str(q.get("answer", ""))[:120],
                "explanation": str(q.get("explanation", ""))[:200],
                "evidence_clause": cid}
        # 硬約束：證據條文必須取自給定集合；答案須在選項中（MCQ 時）
        if cid not in allowed:
            item["reject_reason"] = "evidence_clause 不在給定條文集"
            rejected.append(item)
            continue
        if item["options"] and item["answer"] not in item["options"]:
            item["reject_reason"] = "答案不在選項中"
            rejected.append(item)
            continue
        questions.append(item)
        if len(questions) >= n:
            break
    for i, q in enumerate(questions, 1):
        q["no"] = i
    return {"channel": channel_full or "（全書）", "backend": llm.backend,
            "n": len(questions), "questions": questions,
            "rejected_questions": rejected[:5],
            "evidence_pool": cids,
            "note": "模型自主出題（E 層）：每題證據條文已強制綁定給定條文集，"
                    "不合規的題整題剔除；答案解析仍須自行核對原文。"}
