"""Pure-stdlib SVG chart generation for paper figures.

Follows the data-viz method: form picked by the data's job (horizontal bars
for ranked magnitude, a one-hue sequential heatmap for the agreement matrix,
an interval chart for scenario assumptions), color assigned by role from a
palette whose CVD safety is **checked by paper/figure_qa.py and shipped as
figure_qa/qa_report.json**（十五輪 十二：不再留下不可復核的「已驗證」註釋
——驗證本身是資產）。

十五輪加固：物理尺寸（mm 寬度，期刊剖面）、<title>/<desc>/role="img"
無障礙、標籤**不再靜默截斷**（標籤列寬按最長標籤動態計算）、熱圖色階
圖例 + 每格 n、坐標刻度軸、確定性輸出（無隨機/時間依賴）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]   # fixed categorical order
SEQ_BLUE = ["#eef4fc", "#c9dcf4", "#9cc0ea", "#659ada", "#2a78d6", "#1a54a0"]
FONT = "font-family='Noto Sans CJK SC, PingFang SC, sans-serif'"
MIN_FONT_PX = 9                      # QA 執行的字號下限（720px 畫布）


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _svg_open(width: int, height: int, title: str, desc: str,
              width_mm: float) -> List[str]:
    """帶物理尺寸與無障礙信息的 SVG 開頭（十五輪 九.1/九.5）。"""
    height_mm = width_mm * height / width
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{width_mm:.0f}mm' height='{height_mm:.1f}mm' "
        f"viewBox='0 0 {width} {height}' role='img' "
        f"aria-labelledby='fig-title fig-desc' {FONT}>",
        f"<title id='fig-title'>{_esc(title)}</title>",
        f"<desc id='fig-desc'>{_esc(desc or title)}</desc>",
        f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
    ]


def _label_width(labels: Sequence[str], font_px: int = 12,
                 pad: int = 20, cap: int = 320) -> int:
    """標籤列寬按最長標籤計算——**不截斷**（十五輪 九.3）。"""
    longest = max((len(str(x)) for x in labels), default=4)
    return min(cap, pad + font_px * longest + 8)


def _axis_ticks(out: List[str], x0: int, x1: int, y0: int, y1: int,
                vmax: float, value_fmt: str, x_label: str) -> None:
    """底部刻度軸 + 軸標題（十五輪 九.4：直接標籤不替代量綱說明）。"""
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + (x1 - x0) * frac
        out.append(f"<line x1='{x:.1f}' y1='{y0}' x2='{x:.1f}' y2='{y1}' "
                   f"stroke='{GRID}' stroke-width='1'/>")
        out.append(f"<text x='{x:.1f}' y='{y1 + 14}' font-size='9' "
                   f"text-anchor='middle' fill='{INK_2}'>"
                   f"{_esc(value_fmt.format(vmax * frac))}</text>")
    if x_label:
        out.append(f"<text x='{(x0 + x1) / 2:.1f}' y='{y1 + 30}' "
                   f"font-size='10' text-anchor='middle' fill='{INK_2}'>"
                   f"{_esc(x_label)}</text>")


def _rbar(x: float, y: float, w: float, h: float, fill: str, r: float = 4) -> str:
    """Horizontal bar anchored at the left baseline, rounded DATA end only."""
    r = min(r, w / 2, h / 2)
    return (f"<path d='M{x:.1f},{y:.1f} h{w - r:.1f} q{r},0 {r},{r} "
            f"v{h - 2 * r:.1f} q0,{r} -{r},{r} h-{w - r:.1f} z' fill='{fill}'/>")


def hbar_chart(pairs: Sequence[Tuple[str, float]], title: str,
               subtitle: str = "", value_fmt: str = "{:.0f}",
               color: str = SERIES[0], width: int = 720,
               width_mm: float = 89.0, desc: str = "",
               x_label: str = "") -> str:
    """Ranked magnitude → horizontal bars, direct-labeled + 刻度軸。"""
    pairs = list(pairs)
    n = len(pairs)
    bar_h, gap, pad = 22, 8, 16
    label_w = _label_width([p[0] for p in pairs])
    top = 56 if subtitle else 40
    bottom = 40 if x_label else 34
    height = top + n * (bar_h + gap) + bottom
    vmax = max((v for _, v in pairs), default=1) or 1
    plot_w = width - label_w - 90
    out = _svg_open(width, height, title, desc or subtitle, width_mm)
    out.append(f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
               f"fill='{INK}'>{_esc(title)}</text>")
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    _axis_ticks(out, label_w, label_w + plot_w, top - 6,
                top + n * (bar_h + gap), vmax, value_fmt, x_label)
    for i, (label, v) in enumerate(pairs):
        y = top + i * (bar_h + gap)
        w = max(2.0, plot_w * v / vmax)
        out.append(f"<text x='{label_w - 8}' y='{y + bar_h - 6}' font-size='12' "
                   f"text-anchor='end' fill='{INK}'>{_esc(label)}</text>")
        out.append(_rbar(label_w + 1, y, w, bar_h, color))
        out.append(f"<text x='{label_w + w + 8}' y='{y + bar_h - 6}' "
                   f"font-size='11' fill='{INK_2}'>"
                   f"{_esc(value_fmt.format(v))}</text>")
    out.append("</svg>")
    return "\n".join(out)


def grouped_hbar_chart(rows: Sequence[Tuple[str, Sequence[float]]],
                       series_names: Sequence[str], title: str,
                       subtitle: str = "", value_fmt: str = "{:.0f}",
                       width: int = 720, width_mm: float = 183.0,
                       desc: str = "", x_label: str = "") -> str:
    """Small-N grouped comparison, ≤3 series in fixed categorical order."""
    rows = list(rows)
    ns = len(series_names)
    bar_h, gap_in, gap_out, pad = 14, 2, 12, 16
    label_w = _label_width([r[0] for r in rows])
    group_h = ns * (bar_h + gap_in) + gap_out
    top = 66
    bottom = 40 if x_label else 16
    height = top + len(rows) * group_h + bottom
    vmax = max((v for _, vs in rows for v in vs), default=1) or 1
    plot_w = width - label_w - 100
    out = _svg_open(width, height, title, desc or subtitle, width_mm)
    out.append(f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
               f"fill='{INK}'>{_esc(title)}</text>")
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    lx = pad
    for k, name in enumerate(series_names[:3]):
        out.append(f"<rect x='{lx}' y='50' width='10' height='10' rx='2' "
                   f"fill='{SERIES[k]}'/>")
        out.append(f"<text x='{lx + 14}' y='59' font-size='11' "
                   f"fill='{INK}'>{_esc(name)}</text>")
        lx += 14 + 11 * len(str(name)) + 18
    _axis_ticks(out, label_w, label_w + plot_w, top - 6,
                top + len(rows) * group_h, vmax, value_fmt, x_label)
    for i, (label, vs) in enumerate(rows):
        gy = top + i * group_h
        out.append(f"<text x='{label_w - 8}' y='{gy + group_h / 2}' font-size='12' "
                   f"text-anchor='end' fill='{INK}'>{_esc(label)}</text>")
        for k, v in enumerate(list(vs)[:3]):
            y = gy + k * (bar_h + gap_in)
            w = max(2.0, plot_w * v / vmax)
            out.append(_rbar(label_w + 1, y, w, bar_h, SERIES[k]))
            out.append(f"<text x='{label_w + w + 6}' y='{y + bar_h - 3}' "
                       f"font-size='10' fill='{INK_2}'>"
                       f"{_esc(value_fmt.format(v))}</text>")
    out.append("</svg>")
    return "\n".join(out)


def interval_chart(rows: Sequence[Tuple[str, Sequence[float]]],
                   scenario_names: Sequence[str], title: str,
                   subtitle: str = "", value_fmt: str = "{:.0f}",
                   width: int = 720, width_mm: float = 183.0,
                   desc: str = "", x_label: str = "") -> str:
    """情景假設區間圖（十五輪 十.Fig7）：一行一項，橫線=情景範圍，
    彩點=各情景取值——**不把假設畫成三根確定性柱子**。"""
    rows = list(rows)
    row_h, pad = 34, 16
    label_w = _label_width([r[0] for r in rows])
    top = 84
    bottom = 40 if x_label else 16
    height = top + len(rows) * row_h + bottom
    vmax = max((v for _, vs in rows for v in vs), default=1) or 1
    plot_w = width - label_w - 100
    out = _svg_open(width, height, title, desc or subtitle, width_mm)
    out.append(f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
               f"fill='{INK}'>{_esc(title)}</text>")
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    lx = pad
    for k, name in enumerate(scenario_names[:3]):
        out.append(f"<circle cx='{lx + 5}' cy='{60}' r='5' fill='{SERIES[k]}'/>")
        out.append(f"<text x='{lx + 14}' y='64' font-size='11' "
                   f"fill='{INK}'>{_esc(name)}</text>")
        lx += 14 + 11 * len(str(name)) + 18
    _axis_ticks(out, label_w, label_w + plot_w, top - 6,
                top + len(rows) * row_h, vmax, value_fmt, x_label)
    for i, (label, vs) in enumerate(rows):
        cy = top + i * row_h + row_h / 2
        vals = list(vs)[:3]
        xs = [label_w + plot_w * v / vmax for v in vals]
        out.append(f"<text x='{label_w - 8}' y='{cy + 4}' font-size='12' "
                   f"text-anchor='end' fill='{INK}'>{_esc(label)}</text>")
        out.append(f"<line x1='{min(xs):.1f}' y1='{cy}' x2='{max(xs):.1f}' "
                   f"y2='{cy}' stroke='{INK_2}' stroke-width='2'/>")
        for k, (v, x) in enumerate(zip(vals, xs)):
            out.append(f"<circle cx='{x:.1f}' cy='{cy}' r='5' "
                       f"fill='{SERIES[k]}'/>")
        out.append(f"<text x='{max(xs) + 10:.1f}' y='{cy + 4}' font-size='10' "
                   f"fill='{INK_2}'>{_esc(value_fmt.format(min(vals)))}–"
                   f"{_esc(value_fmt.format(max(vals)))}</text>")
    out.append("</svg>")
    return "\n".join(out)


def heatmap(labels: Sequence[str], values: Dict[Tuple[str, str], float],
            title: str, subtitle: str = "", width: int = 720,
            value_fmt: str = "{:.2f}", width_mm: float = 183.0,
            desc: str = "",
            cell_n: Optional[Dict[Tuple[str, str], int]] = None) -> str:
    """Symmetric matrix → one-hue sequential heatmap；十五輪 十.Fig6：
    色階圖例 + 每格 n（共注條數）+ 標籤不截斷（列標籤旋轉呈現）。"""
    labels = list(labels)
    n = len(labels)
    cell, gap, pad = 52, 2, 16
    label_w = _label_width(labels)
    top = 96 if subtitle else 84
    legend_h = 46
    height = top + n * (cell + gap) + legend_h + pad
    lo = min(values.values(), default=0.0)
    hi = max(values.values(), default=1.0)
    span = (hi - lo) or 1.0
    out = _svg_open(width, height, title, desc or subtitle, width_mm)
    out.append(f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
               f"fill='{INK}'>{_esc(title)}</text>")
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    for j, lab in enumerate(labels):
        x = label_w + j * (cell + gap) + cell / 2
        out.append(f"<text x='{x}' y='{top - 8}' font-size='10' "
                   f"text-anchor='start' fill='{INK_2}' "
                   f"transform='rotate(-38 {x} {top - 8})'>{_esc(lab)}</text>")
    for i, row_lab in enumerate(labels):
        y = top + i * (cell + gap)
        out.append(f"<text x='{label_w - 8}' y='{y + cell / 2 + 4}' font-size='11' "
                   f"text-anchor='end' fill='{INK}'>{_esc(row_lab)}</text>")
        for j, col_lab in enumerate(labels):
            x = label_w + j * (cell + gap)
            if i == j:
                out.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                           f"rx='3' fill='{GRID}'/>")
                out.append(f"<text x='{x + cell / 2}' y='{y + cell / 2 + 4}' "
                           f"font-size='9' text-anchor='middle' "
                           f"fill='{INK_2}'>—</text>")
                continue
            key = (row_lab, col_lab) if (row_lab, col_lab) in values \
                else (col_lab, row_lab)
            v = values.get(key)
            if v is None:
                continue
            step = min(len(SEQ_BLUE) - 1, int((v - lo) / span * len(SEQ_BLUE)))
            fill = SEQ_BLUE[step]
            ink = "#ffffff" if step >= 4 else INK
            out.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                       f"rx='3' fill='{fill}'/>")
            out.append(f"<text x='{x + cell / 2}' y='{y + cell / 2}' "
                       f"font-size='10' text-anchor='middle' fill='{ink}'>"
                       f"{_esc(value_fmt.format(v))}</text>")
            if cell_n:
                nn = cell_n.get(key)
                if nn is not None:
                    out.append(f"<text x='{x + cell / 2}' y='{y + cell / 2 + 13}' "
                               f"font-size='9' text-anchor='middle' fill='{ink}'>"
                               f"(n={nn})</text>")
    # 色階圖例（對角灰=自身，非缺失值）
    ly = top + n * (cell + gap) + 14
    out.append(f"<text x='{pad}' y='{ly + 10}' font-size='10' "
               f"fill='{INK_2}'>色階（{_esc(value_fmt.format(lo))} → "
               f"{_esc(value_fmt.format(hi))}）：</text>")
    sw = 26
    for k, c in enumerate(SEQ_BLUE):
        out.append(f"<rect x='{pad + 150 + k * (sw + 2)}' y='{ly}' width='{sw}' "
                   f"height='14' rx='2' fill='{c}'/>")
    out.append(f"<rect x='{pad + 150 + 6 * (sw + 2) + 10}' y='{ly}' width='{sw}' "
               f"height='14' rx='2' fill='{GRID}'/>")
    out.append(f"<text x='{pad + 150 + 6 * (sw + 2) + 10 + sw + 6}' y='{ly + 11}' "
               f"font-size='9' fill='{INK_2}'>對角=自身（非缺失）</text>")
    out.append("</svg>")
    return "\n".join(out)
