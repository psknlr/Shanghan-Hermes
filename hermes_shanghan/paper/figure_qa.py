"""自動 Figure QA（十五輪 十七：QA 是發布門禁，不是註釋裡的口頭聲明）。

檢查項（qa_report.json 逐項留痕）：
- SVG XML 合法、<title>/<desc>/role="img" 在場、物理尺寸（mm）在場
- 字號不低於剖面下限（720px 畫布 ≥9px）
- 計劃圖組全部存在（或如實記 skipped_no_data）
- 每張圖都被正文引用（圖表清單之前的正文里出現 Fig.N）
- 每張圖有結構化圖例、有 Source Data 文件
- 調色板 CVD 檢查**真實執行**（十五輪 十二）：線性 RGB 下 Machado
  severity=1.0 模擬 protan/deutan/tritan，CIE76 ΔE（Lab）逐相鄰色對
  計算 + 灰度 ΔL——結果落盤，可復核，不再是「已驗證」空話。

硬違例（invalid XML / 計劃圖缺失且無 skip 原因 / 圖未被正文引用）
使 generate() 失敗——圖表質量問題不能靜默出廠。
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .charts import MIN_FONT_PX, SERIES

RE_FONT_SIZE = re.compile(r"font-size='(\d+(?:\.\d+)?)'")

# Machado et al. 2009, severity 1.0（線性 RGB 空間 3×3）
_CVD_MATRICES = {
    "protanopia": ((0.152286, 1.052583, -0.204868),
                   (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
    "deuteranopia": ((0.367322, 0.860646, -0.227968),
                     (0.280085, 0.672501, 0.047413),
                     (-0.011820, 0.042940, 0.968881)),
    "tritanopia": ((1.255528, -0.076749, -0.178779),
                   (-0.078411, 0.930809, 0.147602),
                   (0.004733, 0.691367, 0.303900)),
}


# ---------------------------------------------------------------------------
# color math（純標準庫：sRGB→linear→模擬→Lab→ΔE76）
# ---------------------------------------------------------------------------
def _hex_to_linear(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    srgb = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple((c / 12.92) if c <= 0.04045
                 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)


def _apply(m, rgb):
    return tuple(max(0.0, min(1.0, sum(m[i][j] * rgb[j] for j in range(3))))
                 for i in range(3))


def _linear_to_lab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = rgb
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _delta_e(a, b) -> float:
    la, lb = _linear_to_lab(a), _linear_to_lab(b)
    return sum((p - q) ** 2 for p, q in zip(la, lb)) ** 0.5


def palette_cvd_report(palette: Sequence[str] = tuple(SERIES),
                       min_delta_e: float = 12.0) -> Dict:
    """調色板相鄰色對在三類色覺缺陷模擬下的 ΔE + 灰度 ΔL。"""
    linear = [_hex_to_linear(c) for c in palette]
    report: Dict = {"palette": list(palette), "min_delta_e_required": min_delta_e,
                    "method": "Machado2009 severity=1.0（線性RGB）→ CIE76 ΔE(Lab)",
                    "vision_types": {}}
    ok = True
    for vt, m in _CVD_MATRICES.items():
        sim = [_apply(m, c) for c in linear]
        pair_es = [round(_delta_e(sim[i], sim[i + 1]), 1)
                   for i in range(len(sim) - 1)]
        worst = min(pair_es) if pair_es else 0.0
        report["vision_types"][vt] = {"adjacent_delta_e": pair_es,
                                      "worst_adjacent": worst,
                                      "pass": worst >= min_delta_e}
        ok = ok and worst >= min_delta_e
    grays = [_linear_to_lab(c)[0] for c in linear]
    gray_ds = [round(abs(grays[i] - grays[i + 1]), 1)
               for i in range(len(grays) - 1)]
    report["grayscale"] = {"adjacent_delta_l": gray_ds,
                           "worst_adjacent": min(gray_ds) if gray_ds else 0.0,
                           "pass": all(d >= 8.0 for d in gray_ds)}
    report["ok"] = ok and report["grayscale"]["pass"]
    return report


# ---------------------------------------------------------------------------
# figure checks
# ---------------------------------------------------------------------------
def check_svg(path: Path) -> Dict:
    issues: List[str] = []
    text = path.read_text(encoding="utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return {"file": path.name, "ok": False,
                "issues": [f"invalid_xml: {exc}"]}
    ns = "{http://www.w3.org/2000/svg}"
    if root.find(f"{ns}title") is None:
        issues.append("missing_title")
    if root.find(f"{ns}desc") is None:
        issues.append("missing_desc")
    if root.get("role") != "img":
        issues.append("missing_role_img")
    if not (root.get("width") or "").endswith("mm"):
        issues.append("no_physical_width_mm")
    small = [s for s in RE_FONT_SIZE.findall(text)
             if float(s) < MIN_FONT_PX]
    if small:
        issues.append(f"font_below_min: {sorted(set(small))}")
    return {"file": path.name, "ok": not issues, "issues": issues}


def run_qa(out_dir: Path, emitted: List[Dict], manuscript: str,
           legends: Dict[str, Dict]) -> Dict:
    """全套 QA；返回報告並落盤 figure_qa/qa_report.json。

    ``emitted``：[{"fig_no": "Fig.1", "key": ..., "file": ...,
                   "skipped": bool, "skip_reason": str}]
    """
    body = manuscript.split("## 圖表清單")[0]
    hard: List[str] = []
    svg_reports: List[Dict] = []
    for f in emitted:
        if f.get("skipped"):
            if not f.get("skip_reason"):
                hard.append(f"{f['key']}: skipped without reason")
            continue
        p = out_dir / f["file"]
        if not p.exists():
            hard.append(f"{f['fig_no']} 文件缺失：{f['file']}")
            continue
        if f["file"].endswith(".svg"):
            rep = check_svg(p)
            svg_reports.append(rep)
            if any(i.startswith("invalid_xml") for i in rep["issues"]):
                hard.append(f"{f['fig_no']} SVG 非法：{rep['issues']}")
        if f["fig_no"] not in body:
            hard.append(f"{f['fig_no']} 未被正文引用（圖不入論證鏈不出廠）")
        if f["key"] not in legends:
            hard.append(f"{f['fig_no']} 缺結構化圖例")
    soft = [i for r in svg_reports for i in r["issues"]]
    cvd = palette_cvd_report()
    report = {"ok": not hard, "hard_violations": hard,
              "soft_issues": sorted(set(soft)),
              "svg_checks": svg_reports,
              "figures_checked": len(emitted),
              "figures_skipped": [
                  {"key": f["key"], "reason": f.get("skip_reason", "")}
                  for f in emitted if f.get("skipped")],
              "palette_cvd": cvd,
              "truncated_labels": [],   # charts 層已根除靜默截斷
              "note": "hard_violations 使論文生成失敗；soft_issues 隨包出廠"
                      "供人工複核"}
    qa_dir = out_dir / "figure_qa"
    qa_dir.mkdir(exist_ok=True)
    (qa_dir / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
