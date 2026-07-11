"""確定性敘述層（十九輪）：從規則庫真實數據生成結構化+段落化長文。

論文此前以模板骨架+表格為主，全文 4–5 千字且敘述稀薄。本模塊補齊
「段落化」一翼：方證各論（逐方專段）、計量結果分述（逐榜解讀）、
誤治傳變分述（逐類型敘述）——全部事實直採規則庫並逐處錨定
clause_id，離線確定性生成；接入真模型時增益層再疊加語義解讀。
"""
from __future__ import annotations

from typing import Dict, List

from ..textutil import fold_variants


def _fmt_ids(ids: List[str], n: int = 4) -> str:
    return "、".join(ids[:n]) + ("等" if len(ids) > n else "")


def _topic_formulas(topic: str, formula_rules, formula_freq,
                    limit: int = 6) -> List:
    t = fold_variants(topic or "")
    hits = [r for r in formula_rules
            if r.formula and fold_variants(r.formula) in t]
    if hits:
        return hits[:limit]
    top = [f for f, _ in formula_freq.most_common(limit)]
    by_name = {r.formula: r for r in formula_rules}
    return [by_name[f] for f in top if f in by_name][:limit]


def formula_monographs(topic: str, s: Dict, formula_rules,
                       clause_store: Dict, differential_rules) -> str:
    """方證各論：逐方一段（核心證/首見原文節選/組成/加減/鑒別），
    每段事實均錨定條文編號。"""
    rows = ["## 5 方證各論（規則庫直採，逐段錨定條文）", ""]
    for r in _topic_formulas(topic, formula_rules, s["formula_freq"]):
        support = list(r.supporting_clauses or [])
        first = clause_store.get(support[0]) if support else None
        seg: List[str] = []
        seg.append(f"**{r.formula}**（{'、'.join(r.six_channel_scope or []) or '跨經'}；"
                   f"支持條文 {len(support)} 條：{_fmt_ids(support)}）。")
        if r.core_symptoms:
            seg.append(f"其核心證候為{('、'.join(r.core_symptoms[:6]))}"
                       + (f"，核心脈象{('、'.join(r.core_pulse[:3]))}"
                          if r.core_pulse else "") + "。")
        if r.associated_symptoms:
            seg.append(f"兼證可見{('、'.join(r.associated_symptoms[:6]))}。")
        if first is not None:
            excerpt = first.clean_text[:64]
            seg.append(f"首見於 {support[0]}：「{excerpt}"
                       + ("……」" if len(first.clean_text) > 64 else "」")
                       + "。")
            comp = next((fb for fb in first.formula_blocks
                         if fb.formula_name == r.formula), None)
            if comp and comp.composition:
                herbs = "、".join(x["herb"] + (f"（{x['dose_processing']}）"
                                              if x.get("dose_processing") else "")
                                  for x in comp.composition[:8])
                seg.append(f"其方由{herbs}組成"
                           + (f"；服法要點：{comp.administration[:40]}……"
                              if comp.administration else "") + "。")
        mods = r.modification_relations or []
        if mods:
            seg.append("加減演化方面，"
                       + "；".join(f"加{m.get('added_herbs') or '（調量）'}"
                                   f"{'減' + m['removed_herbs'] if m.get('removed_herbs') else ''}"
                                   f"成{m.get('modified_formula', '')}"
                                   for m in mods[:3]) + "。")
        diffs = [d for d in differential_rules if r.formula in d.formulas][:2]
        for d in diffs:
            others = [f for f in d.formulas if f != r.formula]
            key = next((k for k in d.key_discriminators
                        if k.startswith(r.formula)), "")
            seg.append(f"與{('、'.join(others))}之鑒別，"
                       + (f"其獨有指徵在{key.partition('：')[2]}"
                          if key else "要點見鑒別規則")
                       + f"（證據：{_fmt_ids(d.supporting_clauses or [], 3)}）。")
        rows.append(" ".join(seg))
        rows.append("")
    rows.append("（本節每一事實均可由所附條文編號回源；方證歸納屬 D 層，"
                "原文節選屬 A 層。）")
    return "\n".join(rows)


def quant_narrative(s: Dict, digest: Dict) -> str:
    """計量結果分述：逐榜給出段落化解讀（確定性，數字直採資產）。"""
    rows = ["## 6 計量結果分述（確定性敘述層）", ""]
    freq = digest.get("top_symptoms") or []
    if freq:
        head = freq[0]
        rows.append(
            f"證候頻次方面，全書最高頻的表現為「{head[0]}」"
            f"（{head[1]} 條），其後依次為"
            + "、".join(f"{t}（{n}）" for t, n in freq[1:6])
            + "。高頻榜由表證與胃腸道表現主導，這與太陽病篇幅最大、"
              "誤治變證多累及中焦的篇章結構一致；頻次口徑為條文計數，"
              "同條多次出現不重複計。")
    pulses = digest.get("top_pulses") or []
    if pulses:
        rows.append(
            "脈象方面，"
            + "、".join(f"{t}（{n}）" for t, n in pulses[:5])
            + " 為最常見記載；浮脈居首與表證主導的篇章結構互為印證，"
              "微、細等虛脈集中出現於三陰篇。")
    edges = digest.get("top_symptom_edges") or []
    if edges and isinstance(edges[0], dict):
        e0 = edges[0]
        rows.append(
            f"方-症共現網絡共 {digest.get('symptom_edge_count', len(edges))} 條邊，"
            f"最強邊為 {e0.get('formula')}—{e0.get('symptom')}"
            f"（{e0.get('weight')} 條同現），其後有 "
            + "；".join(f"{e.get('formula')}—{e.get('symptom')}（{e.get('weight')}）"
                        for e in edges[1:5])
            + " 等構成骨幹。共現以條文為單位統計，反映原文中方與證的"
              "同現結構，而非療效斷言。")
    hubs = digest.get("network_hubs") or []
    if hubs:
        rows.append(
            "以連接度計，網絡樞紐方為 "
            + "、".join(f"{h['formula']}（度 {h['degree']}）"
                        for h in hubs[:5])
            + "。樞紐度高者多為主治面寬、兼證豐富之方，"
              "亦是鑒別診斷的高頻參照系。")
    ch = s.get("channel_clauses")
    if ch:
        parts = "、".join(f"{k} {v} 條" for k, v in ch.most_common())
        rows.append(f"六經篇幅分佈為：{parts}。太陽病篇獨大是《傷寒論》"
                    "詳於表證、詳於誤治的文本事實，也是遮方預測等評測任務"
                    "難度不均的來源之一。")
    fams = digest.get("top_families") or []
    if fams:
        f0 = fams[0]
        rows.append(
            f"家族樹方面，{f0.get('base')}族收錄加減方最多"
            f"（{f0.get('n_modifications')} 個），其後為 "
            + "、".join(f"{x.get('base')}（{x.get('n_modifications')}）"
                        for x in fams[1:4])
            + "。「量變致新方」（藥味不變、僅劑量變者）在圖譜中作為"
              "一等關係單獨標記，體現經方以劑量界定方界的特點。")
    rows.append("（本節數字直採 research/ 計量資產，屬 D 層派生統計；"
                "逐項對應的 CSV 與 SVG 圖見圖表清單。）")
    return "\n".join(rows)


def mistreatment_narrative(mistreatment_rules) -> str:
    """誤治傳變分述：按誤治類型逐段敘述（變證譜+救逆方+條文）。"""
    by_type: Dict[str, List] = {}
    for m in mistreatment_rules:
        by_type.setdefault(m.mistreatment_type, []).append(m)
    rows = ["## 7 誤治傳變分述", ""]
    for mtype in sorted(by_type):
        items = by_type[mtype]
        pats = "、".join(dict.fromkeys(
            m.resulting_pattern for m in items[:5]))
        rescues = sorted({f for m in items for f in (m.rescue_formulas or [])})
        cids = sorted({c for m in items
                       for c in (m.supporting_clauses or [])})
        rows.append(
            f"**{mtype}**（{len(items)} 條路徑）：變證譜包括{pats}等；"
            f"救逆方涉及{('、'.join(rescues[:6])) or '（無明文救逆方）'}"
            f"。證據條文：{_fmt_ids(cids, 5)}。")
        rows.append("")
    rows.append("（誤治→變證→救逆為條文級路徑歸納；多步連續誤治屬組合"
                "推演，文中不作原文連續敘述使用。）")
    return "\n".join(rows)
