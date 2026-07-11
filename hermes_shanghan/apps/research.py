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
    # 主題解析與域界定（十九輪：挖掘真正按題收斂，不再恆為全書榜單）
    # ------------------------------------------------------------------
    def parse_topic(self, topic: str) -> Dict[str, List[str]]:
        """從主題文本解析方/證/脈/藥/六經詞（詞表確定性匹配，透明可審）。"""
        from .. import lexicon
        from ..textutil import fold_variants, normalize_query
        t = fold_variants(normalize_query(topic or ""))
        formulas = sorted({r.formula for r in self.formula_rules
                           if r.formula and fold_variants(r.formula) in t})
        for alias, canon in lexicon.FORMULA_ALIASES.items():
            if fold_variants(alias) in t:
                formulas.append(canon)
        formulas = sorted(set(formulas))
        symptoms = sorted({s for s in lexicon.SYMPTOMS
                           if len(s) >= 2 and fold_variants(s) in t})
        pulses = sorted({p for p in lexicon.PULSE_NAMED_PATTERNS
                         if fold_variants(p) in t})
        channels = sorted({ch for ch in ("太陽病", "陽明病", "少陽病",
                                         "太陰病", "少陰病", "厥陰病")
                           if ch.rstrip("病") in t})
        all_herbs = {h for c in self.clauses for h in (c.herbs or [])}
        herbs = sorted({h for h in all_herbs
                        if len(h) >= 2 and fold_variants(h) in t
                        and not any(h in f for f in formulas)})
        return {"formulas": formulas, "symptoms": symptoms, "pulses": pulses,
                "channels": channels, "herbs": herbs}

    def scope_clauses(self, parsed: Dict[str, List[str]]):
        """主題域條文子集：命中任一解析詞的條文（並集）。"""
        from ..textutil import fold_variants
        out = []
        for c in self.clauses:
            text = fold_variants(c.clean_text)
            hit = (any(f in c.formula_names for f in parsed["formulas"])
                   or any(s in c.symptoms or fold_variants(s) in text
                          for s in parsed["symptoms"])
                   or any(p in c.pulse for p in parsed["pulses"])
                   or c.six_channel in parsed["channels"]
                   or any(h in (c.herbs or []) for h in parsed["herbs"]))
            if hit:
                out.append(c)
        return out

    def cooccurrence(self, kind: str = "symptom", clauses=None) -> Dict:
        edges: Counter = Counter()
        for c in (clauses if clauses is not None else self.clauses):
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

    def frequency_tables(self, clauses=None) -> Dict[str, List]:
        sym, pul, form, channel_form = Counter(), Counter(), Counter(), Counter()
        for c in (clauses if clauses is not None else self.clauses):
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

        # 主題感知（十九輪）：解析方/證/脈/藥/六經詞，統計域收斂到主題
        # 條文子集——不再不論輸入為何都返回全書榜單
        parsed = self.parse_topic(topic)
        scoped_clauses = self.scope_clauses(parsed)
        scoped = bool(scoped_clauses)
        dom = scoped_clauses if scoped else self.clauses
        s_sym_net = self.cooccurrence("symptom", dom) if scoped else sym_net
        s_pulse_net = self.cooccurrence("pulse", dom) if scoped else pulse_net
        s_freq = self.frequency_tables(dom) if scoped else freq
        payload["topic_analysis"] = {
            **parsed,
            "n_scope_clauses": len(scoped_clauses),
            "scope_clause_ids": [c.clause_id for c in scoped_clauses][:40],
            "scoped": scoped,
            "note": ("統計域＝主題命中條文子集（並集）"
                     if scoped else
                     "主題未解析出方/證/脈/藥/六經詞——已回退全書口徑，"
                     "請在主題中包含具體方名或證候詞"),
        }
        focus = parsed["formulas"]

        if "network" in outputs:
            payload["networks"] = {
                "formula_symptom_edges": len(s_sym_net["edges"]),
                "formula_pulse_edges": len(s_pulse_net["edges"]),
                "top_symptom_edges": s_sym_net["edges"][:60],
                "top_pulse_edges": s_pulse_net["edges"][:24],
                "files": ["formula_symptom_network.json", "formula_symptom_network.dot",
                          "formula_pulse_network.json"],
            }
            if focus:
                payload["networks"]["focus_formulas"] = focus
                payload["networks"]["focus_edges"] = [
                    e for e in s_sym_net["edges"] if e["formula"] in focus][:40]
        payload["frequency"] = {
            "symptom_frequency": s_freq["symptom_frequency"][:30],
            "pulse_frequency": s_freq["pulse_frequency"][:20],
            "formula_frequency": s_freq["formula_frequency"][:30],
            "channel_formula": [
                {"six_channel": ch, "formula": f, "n_clauses": n}
                for ch, f, n in s_freq["channel_formula"][:24]],
            "note": ("頻次口徑＝主題域 " + str(len(dom)) + " 條"
                     if scoped else
                     "頻次以宋本 398 條正文為口徑")
                    + "（D 層計量，證據錨定 A 層條文）",
        }
        fam = tree["families"]
        scope_formulas = ({f for c in scoped_clauses for f in c.formula_names}
                          if scoped else set())
        if focus or scope_formulas:
            keys = set(focus) | scope_formulas
            fam = [f for f in fam
                   if f["base"] in keys
                   or any(m.get("modified_formula", "") in keys
                          for m in f["modifications"])] or tree["families"]
        payload["family_tree"] = {
            "n_families": len(tree["families"]),
            "families": fam[:20],
            "note": "加減方家族樹（modification_relations，D 層歸納"
                    + ("，已按主題域過濾" if scoped else "") + "）",
        }
        if "rules" in outputs:
            topic_formulas = [r for r in self.formula_rules
                              if r.formula in focus
                              or r.formula in scope_formulas]
            payload["topic_formula_rules"] = [{
                "formula": r.formula, "core_symptoms": r.core_symptoms,
                "core_pulse": r.core_pulse,
                "supporting_clauses": r.supporting_clauses,
                "release_level": r.release_level,
            } for r in (topic_formulas or self.formula_rules)[:10]]
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
            "clauses": len(dom),
            "clauses_whole_book": len(self.clauses),
            "formula_rules": len(self.formula_rules),
            "mistreatment_paths": len(paths),
            "top_symptoms": s_freq["symptom_frequency"][:10],
            "top_formulas": s_freq["formula_frequency"][:10],
        }
        return safety.governed(payload, "researcher")
