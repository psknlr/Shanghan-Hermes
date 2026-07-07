"""劑量計量層 (Dosimetric layer) — 經方本源劑量的可計算化.

The segmenter already parses every <F> block's per-herb 劑量炮製 string
(「三兩，去皮」「各十八銖」「半升」…) but nothing downstream ever computed
with it. This module turns those strings into:

  dose_table            per-herb structured doses + gram/mL conversions
  dose_ratios           within-formula weight ratios（銖當量，學派無關）
  dose_family_evolution family-tree edges annotated with per-herb dose
                        deltas — distinguishing 加味 (new herb) from 增量
                        (same herb, dose changed): 桂枝湯→桂枝加芍藥湯 is
                        芍藥 三兩→六兩 (×2), invisible to a herb-set view
  summary               modal dose per herb, total formula mass per school,
                        parse coverage (unparsed strings listed, not hidden)

Uncertainty is explicit, like the B-layer variants: absolute masses are
reported under multiple 度量衡 conversion schools simultaneously (考古實測
1兩=15.625g；吳承洛度量衡史 1兩=13.92g；明清折算 1兩≈3g) and labelled
posthoc (D/E layer). Weight RATIOS, computed in 銖-equivalents, are
school-independent — a deliberately conversion-free invariant. Counts
(枚/個/把…) and lengths (尺) are preserved as-is, never converted through
unattested per-herb mass assumptions.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from ..schemas import FormulaPatternRule, ShanghanClause

# 漢制單位（銖當量 / 合當量）—— 1分=6銖, 1兩=4分=24銖, 1斤=16兩,
# 1升=10合, 1斗=10升（《廣雅》：四分成兩）
WEIGHT_ZHU = {"銖": 1, "分": 6, "兩": 24, "斤": 24 * 16}
VOLUME_GE = {"合": 1, "升": 10, "斗": 100}
COUNT_UNITS = "枚個莖把株挺具粒顆片"
LENGTH_UNITS = "尺寸"

# 度量衡折算學派（後世考證，D/E 層；並存標註，不欽定一家）
SCHOOLS = {
    "kaogu": {"label": "考古實測（東漢1斤=250g，光和大司農銅權；柯雪帆等）",
              "liang_g": 15.625},
    "duliangheng": {"label": "度量衡史（吳承洛：東漢1斤=222.73g）",
                    "liang_g": 13.92},
    "zhezhuan": {"label": "明清折算（古之一兩，今用一錢≈3g）",
                 "liang_g": 3.0},
}
SHENG_ML = 200.0   # 漢代量器實測約 198–204 mL，取 200

_NUM_CHAR = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
             "八": 8, "九": 9}


def _cn_int(s: str) -> Optional[int]:
    """Chinese numeral → int（一~九百，涵蓋經方劑量域，如烏梅「三百枚」）."""
    if not s:
        return None
    if s == "十":
        return 10
    if s == "百":
        return 100
    if "百" in s:
        hundreds, _, rest = s.partition("百")
        h = _NUM_CHAR.get(hundreds)
        if h is None:
            return None
        r = _cn_int(rest) if rest else 0
        return None if r is None else h * 100 + r
    if "十" in s:
        tens, _, ones = s.partition("十")
        if tens and tens not in _NUM_CHAR:
            return None
        total = (_NUM_CHAR.get(tens, 1)) * 10
        if ones:
            if ones not in _NUM_CHAR:
                return None
            total += _NUM_CHAR[ones]
        return total
    return _NUM_CHAR.get(s) if len(s) == 1 else None


RE_DOSE_TOKEN = re.compile(
    r"(各)?(半|[一二三四五六七八九十百]+)([銖分兩斤升合斗" + COUNT_UNITS + LENGTH_UNITS + r"])")


def parse_dose(dose_processing: str) -> Dict:
    """「一兩十六銖，去皮」→ {kind: weight, zhu: 40, each: False, raw: …}.

    Only the LEADING dose expression is parsed; unit characters inside the
    processing tail (「破八片」) are not doses. Returns kind ∈
    weight|volume|count|length|none, with `unparsed` echoing anything the
    grammar refused (counted upstream, never silently dropped).
    """
    raw = (dose_processing or "").strip()
    out: Dict = {"raw": dose_processing, "kind": "none", "each": False}
    if not raw:
        return out
    # the dose expression may follow processing notes（「炙，各十八銖」）—
    # take the first comma-separated segment that BEGINS with a dose token;
    # tail fragments like「破八片」begin with a verb and are never doses
    s = raw
    for seg in re.split(r"[，、,]", raw):
        seg = re.sub(r"^(大者|小者)", "", seg.strip())  # 「大者一枚」尺寸修飾
        if seg and RE_DOSE_TOKEN.match(seg):
            s = seg
            break
    pos = 0
    each = False
    zhu = 0.0
    ge = 0.0
    count: Optional[float] = None
    length: Optional[Tuple[float, str]] = None
    kind = "none"
    while True:
        m = RE_DOSE_TOKEN.match(s, pos)
        if not m:
            break
        if m.group(1):
            each = True
        num = 0.5 if m.group(2) == "半" else _cn_int(m.group(2))
        if num is None:
            break
        unit = m.group(3)
        if unit in WEIGHT_ZHU and kind in ("none", "weight"):
            zhu += num * WEIGHT_ZHU[unit]
            kind = "weight"
        elif unit in VOLUME_GE and kind in ("none", "volume"):
            ge += num * VOLUME_GE[unit]
            kind = "volume"
        elif unit in COUNT_UNITS and kind == "none":
            count = num
            kind = "count"
        elif unit in LENGTH_UNITS and kind == "none":
            length = (num, unit)
            kind = "length"
        else:
            break
        pos = m.end()
        # suffix 半 =「再加半個前一單位」：一兩半=36銖, 二合半=2.5合
        if pos < len(s) and s[pos] == "半":
            if kind == "weight":
                zhu += 0.5 * WEIGHT_ZHU[unit]
            elif kind == "volume":
                ge += 0.5 * VOLUME_GE[unit]
            elif kind == "count":
                count += 0.5
            pos += 1
    out["each"] = each
    out["kind"] = kind
    if kind == "weight":
        out["zhu"] = zhu
        out["grams"] = {k: round(zhu / 24 * v["liang_g"], 2)
                        for k, v in SCHOOLS.items()}
    elif kind == "volume":
        out["ge"] = ge
        out["ml"] = round(ge / 10 * SHENG_ML, 1)
    elif kind == "count":
        out["count"] = count
        out["count_unit"] = s[pos - 1]
    elif kind == "length":
        out["length"], out["length_unit"] = length
    if kind == "none" and s and not s.startswith(("去", "炙", "洗", "切", "碎",
                                                  "擘", "炮", "泡", "熬", "研")):
        out["unparsed_head"] = s[:12]
    return out


def resolve_each_groups(parsed: List[Dict]) -> None:
    """「芍藥　麻黃　甘草〈各十八銖〉」— herbs listed dose-less before a
    各-dose share it. Mutates the parsed list in place (adds shared dose)."""
    run: List[int] = []
    for i, p in enumerate(parsed):
        if p["kind"] == "none" and "unparsed_head" not in p:
            run.append(i)
        elif p["each"] and run:
            for j in run:
                shared = {k: v for k, v in p.items() if k != "raw"}
                shared["shared_from_each"] = True
                parsed[j].update(shared)
            run = []
        else:
            run = []


class DosimetryMiner:
    def __init__(self, clauses: List[ShanghanClause],
                 formula_rules: Optional[List[FormulaPatternRule]] = None):
        self.clauses = [c for c in clauses if c.text_type == "original_clause"]
        self.formula_rules = formula_rules or []
        self._blocks: Dict[str, Dict] = {}       # formula -> first <F> block
        self._occurrences: Counter = Counter()
        for c in self.clauses:
            for fb in c.formula_blocks:
                self._occurrences[fb.formula_name] += 1
                self._blocks.setdefault(fb.formula_name, {
                    "clause_id": c.clause_id, "composition": fb.composition})

    # ------------------------------------------------------------------
    def dose_table(self) -> Dict:
        rows: List[Dict] = []
        unparsed: List[Dict] = []
        for formula in sorted(self._blocks):
            blk = self._blocks[formula]
            parsed = [parse_dose(c.get("dose_processing", ""))
                      for c in blk["composition"]]
            resolve_each_groups(parsed)
            for c, p in zip(blk["composition"], parsed):
                row = {"formula": formula, "clause_id": blk["clause_id"],
                       "herb": c["herb"], **p}
                rows.append(row)
                if "unparsed_head" in p:
                    unparsed.append({"formula": formula, "herb": c["herb"],
                                     "raw": p["raw"]})
        kinds = Counter(r["kind"] for r in rows)
        return {"schools": {k: v["label"] for k, v in SCHOOLS.items()},
                "sheng_ml": SHENG_ML,
                "interpretation_level": "posthoc_conversion(D/E)",
                "n_rows": len(rows), "kind_counts": dict(kinds),
                "n_unparsed": len(unparsed), "unparsed": unparsed,
                "rows": rows}

    # ------------------------------------------------------------------
    def dose_ratios(self, table: Optional[Dict] = None) -> Dict:
        """Within-formula weight ratios in 銖-equivalents — school-free."""
        table = table or self.dose_table()
        by_formula: Dict[str, List[Dict]] = defaultdict(list)
        for r in table["rows"]:
            if r["kind"] == "weight":
                by_formula[r["formula"]].append(r)
        out = []
        for formula, rows in sorted(by_formula.items()):
            base = min(r["zhu"] for r in rows)
            if base <= 0:
                continue
            ratio = "：".join(f"{r['herb']}{_fmt(r['zhu'] / base)}" for r in rows)
            total_g = {k: round(sum(r["grams"][k] for r in rows), 1)
                       for k in SCHOOLS}
            out.append({"formula": formula, "clause_id": rows[0]["clause_id"],
                        "n_weight_herbs": len(rows), "ratio": ratio,
                        "total_weight_g": total_g})
        return {"note": "比例以銖當量計，與折算學派無關（school-independent）",
                "formulas": out}

    # ------------------------------------------------------------------
    def family_dose_evolution(self, table: Optional[Dict] = None) -> Dict:
        """Annotate 加減方 edges with per-herb dose deltas."""
        table = table or self.dose_table()
        doses: Dict[Tuple[str, str], Dict] = {
            (r["formula"], r["herb"]): r for r in table["rows"]}
        herbs_of: Dict[str, List[str]] = defaultdict(list)
        for r in table["rows"]:
            herbs_of[r["formula"]].append(r["herb"])

        edges: List[Dict] = []
        for fr in self.formula_rules:
            for m in fr.modification_relations:
                mod = m.get("modified_formula", "")
                if fr.formula not in herbs_of or mod not in herbs_of:
                    continue
                deltas: List[Dict] = []
                for herb in herbs_of[fr.formula]:
                    a = doses.get((fr.formula, herb))
                    b = doses.get((mod, herb))
                    if not a or not b:
                        continue
                    if a["kind"] == b["kind"] == "weight" and a["zhu"] and \
                            abs(a["zhu"] - b["zhu"]) > 1e-9:
                        deltas.append({"herb": herb,
                                       "base_raw": a["raw"], "mod_raw": b["raw"],
                                       "factor": round(b["zhu"] / a["zhu"], 3)})
                    elif a["kind"] == b["kind"] == "count" and a.get("count") and \
                            a["count"] != b.get("count"):
                        deltas.append({"herb": herb,
                                       "base_raw": a["raw"], "mod_raw": b["raw"],
                                       "factor": round((b.get("count") or 0)
                                                       / a["count"], 3)})
                kind = []
                if m.get("added_herbs"):
                    kind.append("加味")
                if m.get("removed_herbs"):
                    kind.append("減味")
                if deltas:
                    kind.append("增減量")
                edges.append({"base": fr.formula, "modified": mod,
                              "edge_kind": "+".join(kind) or "同藥同量",
                              "added_herbs": m.get("added_herbs", ""),
                              "removed_herbs": m.get("removed_herbs", ""),
                              "dose_deltas": deltas})
        n_dose_only = sum(1 for e in edges
                          if e["dose_deltas"] and not e["added_herbs"]
                          and not e["removed_herbs"])
        return {"note": "劑量演化：加味≠增量，僅劑量變化的方對(dose-only)單獨計數",
                "n_edges": len(edges), "n_with_dose_delta":
                    sum(1 for e in edges if e["dose_deltas"]),
                "n_dose_only_edges": n_dose_only,
                "edges": edges}

    # ------------------------------------------------------------------
    def summary(self, table: Optional[Dict] = None) -> Dict:
        table = table or self.dose_table()
        herb_doses: Dict[str, Counter] = defaultdict(Counter)
        for r in table["rows"]:
            if r["kind"] == "weight":
                herb_doses[r["herb"]][r["zhu"]] += 1
        modal = []
        for herb, cnt in sorted(herb_doses.items(),
                                key=lambda kv: -sum(kv[1].values()))[:15]:
            zhu, n = cnt.most_common(1)[0]
            modal.append({"herb": herb, "n_formulas": sum(cnt.values()),
                          "modal_dose": _zhu_str(zhu), "modal_n": n})
        ratios = self.dose_ratios(table)
        heaviest = sorted(ratios["formulas"],
                          key=lambda f: -f["total_weight_g"]["kaogu"])[:8]
        return {"herb_modal_doses": modal,
                "heaviest_formulas_kaogu_g": [
                    {"formula": f["formula"],
                     "total_g": f["total_weight_g"]} for f in heaviest],
                "parse_coverage": {
                    "n_rows": table["n_rows"],
                    "kind_counts": table["kind_counts"],
                    "n_unparsed": table["n_unparsed"]}}


def _fmt(x: float) -> str:
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def _zhu_str(zhu: float) -> str:
    liang = zhu / 24
    if abs(liang - round(liang)) < 1e-9:
        return f"{_fmt(liang)}兩"
    return f"{_fmt(zhu)}銖"
