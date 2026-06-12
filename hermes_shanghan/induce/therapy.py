"""Therapy rule induction — 汗吐下和溫清補救逆 + 禁汗/禁下/禁吐 + 誤汗/誤下/誤吐.

Draws on two evidence pools:
  * canonical clauses whose markers indicate a therapy or its prohibition;
  * Songben auxiliary chapters (辨不可發汗病脈證並治 etc.) — the dedicated
    可/不可 chapters, which are the densest source of 法度 rules.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .. import config
from ..schemas import InitialRule, ShanghanClause, TherapyRule, write_jsonl

THERAPY_REPRESENTATIVE = {
    "汗法": ["桂枝湯", "麻黃湯", "葛根湯", "大青龍湯", "小青龍湯"],
    "下法": ["大承氣湯", "小承氣湯", "調胃承氣湯", "大柴胡湯", "桃核承氣湯", "大陷胸湯"],
    "吐法": ["瓜蒂散"],
    "和法": ["小柴胡湯", "半夏瀉心湯", "黃連湯"],
    "溫法": ["四逆湯", "理中丸", "吳茱萸湯", "附子湯"],
    "清法": ["白虎湯", "梔子豉湯", "黃芩湯", "白頭翁湯"],
    "補法": ["小建中湯", "炙甘草湯"],
    "救逆": ["四逆湯", "通脈四逆湯", "白通湯", "茯苓四逆湯", "桂枝去芍藥加蜀漆牡蠣龍骨救逆湯"],
    "利水": ["五苓散", "豬苓湯", "真武湯"],
}

AUX_CHAPTER_POLARITY = [
    ("不可發汗", "禁汗", "contraindicated"),
    ("可發汗", "汗法", "indicated"),
    ("發汗後", "誤汗", "mistreatment"),
    ("不可吐", "禁吐", "contraindicated"),
    ("可吐", "吐法", "indicated"),
    ("不可下", "禁下", "contraindicated"),
    ("可下", "下法", "indicated"),
    ("發汗吐下後", "誤治綜合", "mistreatment"),
]

METHOD_SUMMARY = {
    "汗法": "邪在表者汗而發之：脈浮、頭痛、惡寒無汗者可發汗，以麻黃湯類；表虛有汗者解肌，以桂枝湯類。",
    "下法": "實在裏者下之：潮熱、譫語、腹滿痛、不大便、燥屎內結為下法指徵，以承氣輩；下不厭遲，表未解者不可下。",
    "吐法": "邪實胸中、痰實在上脘者，其高者因而越之，瓜蒂散主之；虛家、亡血家禁用。",
    "和法": "邪在半表半裏、樞機不利者，禁汗吐下，以小柴胡湯和解之；寒熱錯雜痞證以瀉心輩辛開苦降。",
    "溫法": "裏虛寒者溫之：自利不渴屬太陰、下利清穀、脈微肢厥者，四逆輩、理中輩溫陽祛寒。",
    "清法": "熱而無形者清之：表解裏熱、煩渴引飲以白虎輩；虛煩懊憹以梔子豉輩；熱利以芩連柏輩。",
    "補法": "正虛者補之：傷寒裏虛、悸而煩者小建中湯；脈結代、心動悸者炙甘草湯。",
    "救逆": "誤治壞病、陽亡陰竭者急救回逆：亡陽漏汗、厥逆脈微者四逆湯輩；火逆驚狂者救逆湯。",
    "利水": "水飲內停者利其小便：脈浮、小便不利、微熱消渴者五苓散；陰虛有熱者豬苓湯；陽虛水泛者真武湯。",
    "禁汗": "咽喉乾燥、淋家、瘡家、衄家、亡血家、汗家、病人有寒等，不可發汗；尺中遲、脈微者亦禁。",
    "禁下": "表未解者不可下；虛家、陽明病面合色赤、嘔多雖有陽明證不可攻之；脈浮虛者當發汗不當下。",
    "禁吐": "太陽病惡寒發熱在表者不可吐；少陰病膈上有寒飲者當溫之不可吐。",
    "誤汗": "不當汗而汗、或汗之太過，則亡陽漏汗、筋惕肉瞤、心下悸、奔豚諸變生焉。",
    "誤下": "不當下而下之，則結胸、痞、協熱利、虛煩、驚悸諸壞病作矣。",
    "誤吐": "誤吐則傷胃氣，腹脹滿、不欲食、朝食暮吐，或內煩懊憹。",
    "誤治綜合": "汗吐下相迭誤施，正虛邪陷，當觀其脈證，知犯何逆，隨證治之。",
    "誤火": "火劫、燒針、熏熨之屬誤施，則驚狂、煩躁、發黃、奔豚諸火逆變證起。",
}


class TherapyInducer:
    def __init__(self, clauses: List[ShanghanClause], initial_rules: List[InitialRule]):
        self.clauses = clauses
        self.rules = [r for r in initial_rules
                      if r.autonomous_review.release_level != "rejected"]
        self.clause_store = {c.clause_id: c for c in clauses}

    def induce(self) -> List[TherapyRule]:
        out: List[TherapyRule] = []
        n = 0

        # --- indicated methods from therapy_selection_rules ----------------
        by_method: Dict[str, List[InitialRule]] = defaultdict(list)
        for r in self.rules:
            if r.rule_type == "therapy_selection_rule":
                for m in r.then_conclusions.get("therapy_methods", []):
                    by_method[m].append(r)
        for method, group in sorted(by_method.items()):
            n += 1
            indications = []
            for r in group[:40]:
                cl = self.clause_store.get(r.clause_id)
                if cl and cl.symptoms:
                    indications.extend(cl.symptoms[:3])
            seen = set()
            indications = [x for x in indications if not (x in seen or seen.add(x))][:12]
            out.append(TherapyRule(
                therapy_rule_id=f"TR_{n:03d}",
                therapy_method=method, polarity="indicated",
                summary=METHOD_SUMMARY.get(method, ""),
                indications=indications,
                representative_formulas=THERAPY_REPRESENTATIVE.get(method, []),
                supporting_clauses=sorted({r.clause_id for r in group}),
                supporting_initial_rules=[r.initial_rule_id for r in group],
                consensus_score=0.88, release_level="silver"))

        # --- prohibitions from contraindication_rules ----------------------
        prohib_map = {"禁汗": ("發汗", "汗"), "禁下": ("攻", "下"), "禁吐": ("吐",)}
        contra_rules = [r for r in self.rules if r.rule_type == "contraindication_rule"]
        for label, keys in prohib_map.items():
            group = []
            for r in contra_rules:
                actions = "".join(r.then_conclusions.get("contraindicated_actions", []))
                if any(k in actions for k in keys):
                    group.append(r)
            if not group:
                continue
            n += 1
            conds = []
            for r in group[:60]:
                cl = self.clause_store.get(r.clause_id)
                if cl:
                    conds.append(cl.clean_text[:40])
            out.append(TherapyRule(
                therapy_rule_id=f"TR_{n:03d}",
                therapy_method=label, polarity="contraindicated",
                summary=METHOD_SUMMARY.get(label, ""),
                contraindication_conditions=conds[:15],
                supporting_clauses=sorted({r.clause_id for r in group}),
                supporting_initial_rules=[r.initial_rule_id for r in group],
                consensus_score=0.9, release_level="gold" if len(group) >= 5 else "silver"))

        # --- mistreatment classes from mistreatment_rules -------------------
        mist_map = {"誤汗": "誤汗", "誤下": "誤下", "誤吐": "誤吐", "火逆": "誤火"}
        by_mtype: Dict[str, List[InitialRule]] = defaultdict(list)
        for r in self.rules:
            if r.rule_type == "mistreatment_rule":
                for t in r.if_conditions.get("mistreatment_type", []):
                    by_mtype[t].append(r)
        for mtype, group in sorted(by_mtype.items()):
            label = mist_map.get(mtype, mtype)
            n += 1
            outcomes = []
            for r in group:
                outcomes.extend(r.then_conclusions.get("adverse_outcomes", []))
            seen = set()
            outcomes = [x for x in outcomes if not (x in seen or seen.add(x))][:12]
            rescue = []
            for r in group:
                rescue.extend(r.then_conclusions.get("rescue_formula", []))
            seen = set()
            rescue = [x for x in rescue if not (x in seen or seen.add(x))][:10]
            out.append(TherapyRule(
                therapy_rule_id=f"TR_{n:03d}",
                therapy_method=label, polarity="mistreatment",
                summary=METHOD_SUMMARY.get(label, ""),
                indications=outcomes,
                representative_formulas=rescue,
                supporting_clauses=sorted({r.clause_id for r in group}),
                supporting_initial_rules=[r.initial_rule_id for r in group],
                consensus_score=0.86, release_level="silver"))

        # --- auxiliary 可/不可 chapters --------------------------------------
        aux = [c for c in self.clauses if c.text_type == "auxiliary_clause"]
        for key, label, polarity in AUX_CHAPTER_POLARITY:
            group = [c for c in aux if key in c.chapter and
                     (("不可" in c.chapter) == ("不可" in key))]
            if not group:
                continue
            n += 1
            out.append(TherapyRule(
                therapy_rule_id=f"TR_{n:03d}",
                therapy_method=label, polarity=polarity,
                summary=f"宋本《{group[0].chapter}》專篇法度（{METHOD_SUMMARY.get(label, '')[:40]}…）"
                        if METHOD_SUMMARY.get(label) else f"宋本《{group[0].chapter}》專篇法度。",
                contraindication_conditions=[c.clean_text[:50] for c in group[:12]]
                if polarity == "contraindicated" else [],
                indications=[c.clean_text[:50] for c in group[:12]]
                if polarity != "contraindicated" else [],
                supporting_clauses=[c.clause_id for c in group],
                source_level="auxiliary_text",
                consensus_score=0.84, release_level="silver"))
        return out

    def run(self) -> List[TherapyRule]:
        rules = self.induce()
        config.ensure_dirs()
        write_jsonl(config.RULES_THERAPY_DIR / "therapy_rules.jsonl", rules)
        return rules
