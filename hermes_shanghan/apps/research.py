"""Research-mode mining outputs: 方證譜系 / 共現網絡 / 頻次統計 / 論文大綱.

Generates machine-readable research assets under data/shanghan/research/:
  formula_symptom_network.json   formula-symptom co-occurrence (+DOT export)
  formula_pulse_network.json     formula-pulse co-occurrence
  mistreatment_paths.json        誤治→變證→救治方 path list
  frequency_tables.csv           symptom/pulse/formula frequencies
  formula_family_tree.json       加減方 family tree
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .. import config, safety
from ..schemas import (FormulaPatternRule, MistreatmentTransformationRule,
                       ShanghanClause)


class ResearchMiner:
    def __init__(self, clauses: List[ShanghanClause],
                 formula_rules: List[FormulaPatternRule],
                 mistreatment_rules: List[MistreatmentTransformationRule]):
        self.clauses = [c for c in clauses if c.text_type == "original_clause"]
        self.formula_rules = formula_rules
        self.mistreatment_rules = mistreatment_rules

    # ------------------------------------------------------------------
    def cooccurrence(self, kind: str = "symptom") -> Dict:
        edges: Counter = Counter()
        for c in self.clauses:
            terms = c.symptoms if kind == "symptom" else c.pulse
            for f in c.formula_names:
                for t in terms:
                    edges[(f, t)] += 1
        nodes_f = sorted({f for (f, _t) in edges})
        nodes_t = sorted({t for (_f, t) in edges})
        return {
            "kind": f"formula_{kind}_cooccurrence",
            "formula_nodes": nodes_f,
            f"{kind}_nodes": nodes_t,
            "edges": [{"formula": f, kind: t, "weight": w}
                      for (f, t), w in edges.most_common()],
        }

    def to_dot(self, network: Dict, kind: str, min_weight: int = 2) -> str:
        lines = ["graph cooccurrence {", '  rankdir=LR;',
                 '  node [fontname="Noto Sans CJK SC"];']
        for e in network["edges"]:
            if e["weight"] >= min_weight:
                lines.append(f'  "{e["formula"]}" -- "{e[kind]}" [weight={e["weight"]}, '
                             f'penwidth={min(4, e["weight"])}];')
        lines.append("}")
        return "\n".join(lines)

    def frequency_tables(self) -> Dict[str, List]:
        sym, pul, form, channel_form = Counter(), Counter(), Counter(), Counter()
        for c in self.clauses:
            sym.update(c.symptoms)
            pul.update(c.pulse)
            form.update(c.formula_names)
            for f in c.formula_names:
                channel_form[(c.six_channel, f)] += 1
        return {
            "symptom_frequency": sym.most_common(),
            "pulse_frequency": pul.most_common(),
            "formula_frequency": form.most_common(),
            "channel_formula": [(ch, f, n) for (ch, f), n in channel_form.most_common()],
        }

    def family_tree(self) -> Dict:
        tree: Dict[str, List[Dict]] = defaultdict(list)
        for r in self.formula_rules:
            for m in r.modification_relations:
                tree[r.formula].append(m)
        return {"families": [{"base": k, "modifications": v} for k, v in sorted(tree.items())]}

    def mistreatment_paths(self) -> List[Dict]:
        return [{
            "mistreatment": m.mistreatment_type,
            "resulting_pattern": m.resulting_pattern,
            "manifestations": m.manifestations,
            "rescue_formulas": m.rescue_formulas,
            "clauses": m.supporting_clauses,
            "release_level": m.release_level,
        } for m in self.mistreatment_rules]

    # ------------------------------------------------------------------
    def run_topic(self, topic: str, scope: str = "傷寒論",
                  outputs: Optional[List[str]] = None) -> Dict:
        outputs = outputs or ["rules", "network", "paper_outline"]
        config.ensure_dirs()
        payload: Dict = {"research_topic": topic, "scope": scope,
                         "evidence_layers": config.LAYER_LABEL}

        sym_net = self.cooccurrence("symptom")
        pulse_net = self.cooccurrence("pulse")
        freq = self.frequency_tables()
        paths = self.mistreatment_paths()
        tree = self.family_tree()

        out_dir = config.RESEARCH_DIR
        (out_dir / "formula_symptom_network.json").write_text(
            json.dumps(sym_net, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "formula_symptom_network.dot").write_text(
            self.to_dot(sym_net, "symptom"), encoding="utf-8")
        (out_dir / "formula_pulse_network.json").write_text(
            json.dumps(pulse_net, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "mistreatment_paths.json").write_text(
            json.dumps(paths, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "formula_family_tree.json").write_text(
            json.dumps(tree, ensure_ascii=False, indent=1), encoding="utf-8")
        with (out_dir / "frequency_tables.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["table", "term", "count"])
            for name in ("symptom_frequency", "pulse_frequency", "formula_frequency"):
                for term, n in freq[name]:
                    w.writerow([name, term, n])
            for ch, f, n in freq["channel_formula"]:
                w.writerow(["channel_formula", f"{ch}|{f}", n])

        # 主題聚焦（十六輪）：主題中出現的方名決定網絡/家族樹的聚焦視圖
        focus = sorted({r.formula for r in self.formula_rules
                        if r.formula and r.formula in topic})

        if "network" in outputs:
            payload["networks"] = {
                "formula_symptom_edges": len(sym_net["edges"]),
                "formula_pulse_edges": len(pulse_net["edges"]),
                # 真實數據隨響應返回（十六輪：UI 不再只有計數與文件名）
                "top_symptom_edges": sym_net["edges"][:60],
                "top_pulse_edges": pulse_net["edges"][:24],
                "files": ["formula_symptom_network.json", "formula_symptom_network.dot",
                          "formula_pulse_network.json"],
            }
            if focus:
                payload["networks"]["focus_formulas"] = focus
                payload["networks"]["focus_edges"] = [
                    e for e in sym_net["edges"] if e["formula"] in focus][:40]
        payload["frequency"] = {
            "symptom_frequency": freq["symptom_frequency"][:30],
            "pulse_frequency": freq["pulse_frequency"][:20],
            "formula_frequency": freq["formula_frequency"][:30],
            "channel_formula": [
                {"six_channel": ch, "formula": f, "n_clauses": n}
                for ch, f, n in freq["channel_formula"][:24]],
            "note": "頻次以宋本 398 條正文為口徑（D 層計量，證據錨定 A 層條文）",
        }
        fam = tree["families"]
        if focus:
            fam = [f for f in fam
                   if f["base"] in focus
                   or any(f["base"] in x for x in focus)
                   or any(m.get("modified_formula", "") in focus
                          for m in f["modifications"])] or tree["families"]
        payload["family_tree"] = {
            "n_families": len(tree["families"]),
            "families": fam[:20],
            "note": "加減方家族樹（modification_relations，D 層歸納）",
        }
        if "rules" in outputs:
            topic_formulas = [r for r in self.formula_rules if r.formula in topic
                              or (r.formula_family and r.formula_family in topic)]
            payload["topic_formula_rules"] = [{
                "formula": r.formula, "core_symptoms": r.core_symptoms,
                "core_pulse": r.core_pulse,
                "supporting_clauses": r.supporting_clauses,
                "release_level": r.release_level,
            } for r in (topic_formulas or self.formula_rules[:10])]
        if "paper_outline" in outputs:
            payload["paper_outline"] = {
                "title": f"基於規則挖掘與證據回源的{scope}{topic}研究",
                "sections": [
                    "1 引言：方證對應與六經辨證的可計算化",
                    "2 數據與方法：宋本條文層、自主審核流水線、規則分級",
                    "3 結果 3.1 方證規則庫 3.2 共現網絡 3.3 誤治傳變圖譜",
                    "4 討論：原文直述與後世歸納的邊界、版本異文的影響",
                    "5 結論與展望",
                ],
                "figures": ["六經-方劑分佈圖", "方劑-症狀共現網絡", "誤治-變證路徑圖", "方劑家族樹"],
                "tables": ["高頻症狀表", "高頻脈象表", "方證規則分級統計", "版本異文對比表"],
            }
        payload["statistics"] = {
            "clauses": len(self.clauses),
            "formula_rules": len(self.formula_rules),
            "mistreatment_paths": len(paths),
            "top_symptoms": freq["symptom_frequency"][:10],
            "top_formulas": freq["formula_frequency"][:10],
        }
        return safety.governed(payload, "researcher")
