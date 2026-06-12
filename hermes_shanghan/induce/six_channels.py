"""SixChannelInducerAgent — chapter-level induction of SixChannelRules.

For each channel: 提綱 clause, subtype structure (curated taxonomy whose
membership is *verified against the corpus* — a subtype only appears when
its anchor formulas/keywords actually occur in that channel's clauses),
main formulas by clause frequency, contraindication & mistreatment clause
sets, and 欲解時.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .. import config, lexicon
from ..schemas import InitialRule, ShanghanClause, SixChannelRule, write_jsonl

# Curated subtype taxonomy (normalized knowledge, labelled as such).
# anchors = formulas or keywords whose presence in the channel's clauses
# confirms the subtype for this corpus.
CHANNEL_SUBTYPES: Dict[str, List[Dict]] = {
    "太陽病": [
        {"name": "太陽中風（表虛）", "anchor_formulas": ["桂枝湯"], "keywords": ["中風", "汗出", "惡風"]},
        {"name": "太陽傷寒（表實）", "anchor_formulas": ["麻黃湯"], "keywords": ["傷寒", "無汗", "體痛"]},
        {"name": "太陽蓄水", "anchor_formulas": ["五苓散"], "keywords": ["消渴", "小便不利", "水入則吐"]},
        {"name": "太陽蓄血", "anchor_formulas": ["桃核承氣湯", "抵當湯", "抵當丸"], "keywords": ["如狂", "少腹急結", "下血"]},
        {"name": "太陽變證（誤治壞病）", "anchor_formulas": ["桂枝加附子湯", "梔子豉湯", "半夏瀉心湯", "大陷胸湯"], "keywords": ["結胸", "心下痞", "亡陽"]},
    ],
    "陽明病": [
        {"name": "陽明熱證（經表之熱）", "anchor_formulas": ["白虎湯", "白虎加人參湯"], "keywords": ["大煩渴", "口乾舌燥"]},
        {"name": "陽明腑實", "anchor_formulas": ["大承氣湯", "小承氣湯", "調胃承氣湯"], "keywords": ["潮熱", "譫語", "燥屎", "不大便"]},
        {"name": "陽明發黃", "anchor_formulas": ["茵陳蒿湯", "梔子檗皮湯", "麻黃連軺赤小豆湯"], "keywords": ["發黃", "身黃"]},
        {"name": "脾約", "anchor_formulas": ["麻子仁丸"], "keywords": ["脾約", "大便硬"]},
    ],
    "少陽病": [
        {"name": "少陽本證", "anchor_formulas": ["小柴胡湯"], "keywords": ["往來寒熱", "胸脅苦滿", "口苦"]},
        {"name": "少陽兼裏實", "anchor_formulas": ["大柴胡湯"], "keywords": ["心下急", "鬱鬱微煩"]},
        {"name": "少陽兼表", "anchor_formulas": ["柴胡桂枝湯"], "keywords": ["支節煩疼"]},
        {"name": "少陽誤治變證", "anchor_formulas": ["柴胡加龍骨牡蠣湯"], "keywords": ["胸滿煩驚"]},
    ],
    "太陰病": [
        {"name": "太陰虛寒本證", "anchor_formulas": ["四逆湯", "理中丸"], "keywords": ["自利不渴", "腹滿", "藏有寒"]},
        {"name": "太陰兼表", "anchor_formulas": ["桂枝湯"], "keywords": ["脈浮"]},
        {"name": "太陰腹痛", "anchor_formulas": ["桂枝加芍藥湯", "桂枝加大黃湯"], "keywords": ["腹滿時痛", "大實痛"]},
    ],
    "少陰病": [
        {"name": "少陰寒化", "anchor_formulas": ["四逆湯", "真武湯", "附子湯", "白通湯", "通脈四逆湯"], "keywords": ["脈微細", "但欲寐", "下利清穀"]},
        {"name": "少陰熱化", "anchor_formulas": ["黃連阿膠湯", "豬苓湯"], "keywords": ["心中煩", "不得臥"]},
        {"name": "少陰咽痛", "anchor_formulas": ["豬膚湯", "甘草湯", "桔梗湯", "苦酒湯", "半夏散及湯"], "keywords": ["咽痛", "咽中痛"]},
        {"name": "少陰急下", "anchor_formulas": ["大承氣湯"], "keywords": ["急下之"]},
        {"name": "少陰兼表", "anchor_formulas": ["麻黃細辛附子湯", "麻黃附子甘草湯"], "keywords": ["反發熱", "始得之"]},
    ],
    "厥陰病": [
        {"name": "厥陰寒熱錯雜", "anchor_formulas": ["烏梅丸", "乾薑黃芩黃連人參湯", "麻黃升麻湯"], "keywords": ["消渴", "氣上撞心", "飢而不欲食"]},
        {"name": "厥陰寒證", "anchor_formulas": ["當歸四逆湯", "吳茱萸湯"], "keywords": ["手足厥寒", "脈細欲絕", "乾嘔吐涎沫"]},
        {"name": "厥陰熱利", "anchor_formulas": ["白頭翁湯"], "keywords": ["熱利下重"]},
        {"name": "厥逆辨治", "anchor_formulas": ["四逆湯", "瓜蒂散"], "keywords": ["厥", "厥逆"]},
    ],
    "霍亂病": [
        {"name": "霍亂吐利", "anchor_formulas": ["五苓散", "理中丸", "四逆湯"], "keywords": ["吐利", "霍亂"]},
    ],
    "陰陽易差後勞復病": [
        {"name": "陰陽易", "anchor_formulas": ["燒褌散"], "keywords": ["陰陽易"]},
        {"name": "差後勞復", "anchor_formulas": ["枳實梔子豉湯", "牡蠣澤瀉散", "竹葉石膏湯"], "keywords": ["勞復", "差後"]},
    ],
}

CHANNEL_SUMMARY: Dict[str, str] = {
    "太陽病": "太陽病以表證為主，提綱為脈浮、頭項強痛而惡寒；分中風（表虛有汗）與傷寒（表實無汗），"
            "另有蓄水、蓄血及誤治後諸變證，方以桂枝湯、麻黃湯為兩大主軸。",
    "陽明病": "陽明病提綱為胃家實；熱在經者主以白虎輩清之，實在腑者以三承氣輩下之，"
            "兼有發黃、脾約諸證，辨潮熱、譫語、燥屎為下法之指徵。",
    "少陽病": "少陽病提綱為口苦、咽乾、目眩；樞機之病，禁汗吐下，主以小柴胡湯和解，"
            "兼裏實者大柴胡湯，誤下則生胸滿煩驚諸變。",
    "太陰病": "太陰病提綱為腹滿而吐、食不下、自利益甚、時腹自痛；屬裏虛寒，"
            "當溫之，宜服四逆輩；兼表者仍可先桂枝湯解外。",
    "少陰病": "少陰病提綱為脈微細、但欲寐；分寒化（四逆湯、真武湯輩回陽）與熱化"
            "（黃連阿膠湯育陰清熱），另有咽痛諸方與三急下證，死證條文最多，最重判別預後。",
    "厥陰病": "厥陰病提綱為消渴、氣上撞心、心中疼熱、飢而不欲食；寒熱錯雜，"
            "烏梅丸為代表；厥熱勝復判預後，熱利白頭翁湯，寒厥當歸四逆湯。",
    "霍亂病": "霍亂以暴起吐利為主證，熱多欲飲水者五苓散，寒多不用水者理中丸，"
            "吐利止而身痛不休者桂枝湯小和之，陽亡者四逆輩救之。",
    "陰陽易差後勞復病": "病後餘熱未清、正氣未復，論陰陽易與差後勞復食復之治，"
            "以枳實梔子豉湯、竹葉石膏湯等清解調養。",
}


class SixChannelInducer:
    def __init__(self, clauses: List[ShanghanClause], initial_rules: List[InitialRule]):
        self.clauses = [c for c in clauses if c.text_type == "original_clause"]
        self.rules = [r for r in initial_rules
                      if r.autonomous_review.release_level != "rejected"]

    def induce(self) -> List[SixChannelRule]:
        by_channel: Dict[str, List[ShanghanClause]] = defaultdict(list)
        for c in self.clauses:
            if c.six_channel:
                by_channel[c.six_channel].append(c)
        rules_by_channel: Dict[str, List[InitialRule]] = defaultdict(list)
        for r in self.rules:
            if r.six_channel:
                rules_by_channel[r.six_channel].append(r)

        out: List[SixChannelRule] = []
        for n, channel in enumerate(config.SIX_CHANNELS + config.EXTRA_CHANNELS, 1):
            group = by_channel.get(channel, [])
            if not group:
                continue
            channel_formulas = Counter()
            for c in group:
                for f in c.formula_names:
                    channel_formulas[f] += 1

            outline_num = config.CHANNEL_OUTLINE_CLAUSE.get(channel)
            outline_clause: Optional[ShanghanClause] = None
            if outline_num:
                outline_clause = next((c for c in group if c.clause_number == outline_num), None)
            if outline_clause is None:
                outline_clause = next((c for c in group if "之為病" in c.clean_text), group[0])

            subtypes = []
            for st in CHANNEL_SUBTYPES.get(channel, []):
                anchor_hits = [f for f in st["anchor_formulas"] if channel_formulas.get(f)]
                kw_clauses = [c.clause_id for c in group
                              if any(k in c.clean_text for k in st["keywords"])][:6]
                if anchor_hits or kw_clauses:
                    subtypes.append({
                        "name": st["name"],
                        "anchor_formulas": anchor_hits or st["anchor_formulas"],
                        "evidence_clauses": kw_clauses,
                        "source_level": "posthoc_induction",
                        "note": "亞型名稱為後世歸納，證據條文為原文。",
                    })

            res_time = ""
            for c in group:
                m = lexicon.RE_RESOLUTION_TIME.search(c.clean_text)
                if m:
                    res_time = f"從{m.group(1)}至{m.group(2)}上"
                    break

            contra_ids = [c.clause_id for c in group if c.contraindication_terms][:20]
            mist_ids = [c.clause_id for c in group if c.mistreatment_terms][:30]
            core = [outline_clause.clause_id] + [
                c.clause_id for c in group
                if c.clause_id != outline_clause.clause_id and any(
                    f"{f}主之" in c.clean_text for f in c.formula_names)][:11]

            r_ids = [r.initial_rule_id for r in rules_by_channel.get(channel, [])]
            out.append(SixChannelRule(
                six_channel_rule_id=f"SCR_{config.CHANNEL_PINYIN[channel].upper()}_{n:03d}",
                six_channel=channel,
                outline_clause_id=outline_clause.clause_id,
                outline_text=outline_clause.clean_text,
                summary=CHANNEL_SUMMARY.get(channel, ""),
                core_clauses=core,
                subtypes=subtypes,
                main_formulas=[{"formula": f, "clause_count": c}
                               for f, c in channel_formulas.most_common(12)],
                contraindication_clauses=contra_ids,
                mistreatment_clauses=mist_ids,
                resolution_time=res_time,
                supporting_initial_rules=r_ids[:400],
                source_level="chapter_level_induction",
                consensus_score=0.9 if len(group) > 20 else 0.84,
                release_level="silver",
            ))
        return out

    def run(self) -> List[SixChannelRule]:
        rules = self.induce()
        config.ensure_dirs()
        write_jsonl(config.RULES_SIX_CHANNEL_DIR / "six_channel_rules.jsonl", rules)
        return rules
