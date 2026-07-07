"""Pure-stdlib SVG chart generation for paper figures.

Follows the data-viz method: form picked by the data's job (horizontal bars
for ranked magnitude, a one-hue sequential heatmap for the agreement matrix),
color assigned by role from a CVD-validated palette (worst adjacent ΔE 47.2,
validated with the six-checks script), thin marks with baseline-anchored
rounded data-ends, 2px surface gaps, recessive grid, direct value labels on
every mark (the relief rule for the sub-3:1 hues — papers also ship the CSV
table view alongside). Static light-surface figures, committed like print.
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


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _rbar(x: float, y: float, w: float, h: float, fill: str, r: float = 4) -> str:
    """Horizontal bar anchored at the left baseline, rounded DATA end only."""
    r = min(r, w / 2, h / 2)
    return (f"<path d='M{x:.1f},{y:.1f} h{w - r:.1f} q{r},0 {r},{r} "
            f"v{h - 2 * r:.1f} q0,{r} -{r},{r} h-{w - r:.1f} z' fill='{fill}'/>")


def hbar_chart(pairs: Sequence[Tuple[str, float]], title: str,
               subtitle: str = "", value_fmt: str = "{:.0f}",
               color: str = SERIES[0], width: int = 720) -> str:
    """Ranked magnitude → horizontal bars, direct-labeled."""
    pairs = list(pairs)
    n = len(pairs)
    bar_h, gap, label_w, pad = 22, 8, 150, 16
    top = 56 if subtitle else 40
    height = top + n * (bar_h + gap) + pad
    vmax = max((v for _, v in pairs), default=1) or 1
    plot_w = width - label_w - 90
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
           f"height='{height}' viewBox='0 0 {width} {height}' {FONT}>",
           f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
           f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
           f"fill='{INK}'>{_esc(title)}</text>"]
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    out.append(f"<line x1='{label_w}' y1='{top - 6}' x2='{label_w}' "
               f"y2='{height - pad}' stroke='{GRID}' stroke-width='1'/>")
    for i, (label, v) in enumerate(pairs):
        y = top + i * (bar_h + gap)
        w = max(2.0, plot_w * v / vmax)
        out.append(f"<text x='{label_w - 8}' y='{y + bar_h - 6}' font-size='12' "
                   f"text-anchor='end' fill='{INK}'>{_esc(label[:12])}</text>")
        out.append(_rbar(label_w + 1, y, w, bar_h, color))
        out.append(f"<text x='{label_w + w + 8}' y='{y + bar_h - 6}' "
                   f"font-size='11' fill='{INK_2}'>"
                   f"{_esc(value_fmt.format(v))}</text>")
    out.append("</svg>")
    return "\n".join(out)


def grouped_hbar_chart(rows: Sequence[Tuple[str, Sequence[float]]],
                       series_names: Sequence[str], title: str,
                       subtitle: str = "", value_fmt: str = "{:.0f}",
                       width: int = 720) -> str:
    """Small-N grouped comparison (e.g. one bar per 折算學派), ≤3 series in
    fixed categorical order, legend + direct labels."""
    rows = list(rows)
    ns = len(series_names)
    bar_h, gap_in, gap_out, label_w, pad = 14, 2, 12, 150, 16
    group_h = ns * (bar_h + gap_in) + gap_out
    top = 66
    height = top + len(rows) * group_h + pad
    vmax = max((v for _, vs in rows for v in vs), default=1) or 1
    plot_w = width - label_w - 100
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
           f"height='{height}' viewBox='0 0 {width} {height}' {FONT}>",
           f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
           f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
           f"fill='{INK}'>{_esc(title)}</text>"]
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    lx = pad
    for k, name in enumerate(series_names[:3]):
        out.append(f"<rect x='{lx}' y='50' width='10' height='10' rx='2' "
                   f"fill='{SERIES[k]}'/>")
        out.append(f"<text x='{lx + 14}' y='59' font-size='11' "
                   f"fill='{INK}'>{_esc(name)}</text>")
        lx += 14 + 11 * min(len(str(name)), 14) + 18
    for i, (label, vs) in enumerate(rows):
        gy = top + i * group_h
        out.append(f"<text x='{label_w - 8}' y='{gy + group_h / 2}' font-size='12' "
                   f"text-anchor='end' fill='{INK}'>{_esc(label[:12])}</text>")
        for k, v in enumerate(list(vs)[:3]):
            y = gy + k * (bar_h + gap_in)
            w = max(2.0, plot_w * v / vmax)
            out.append(_rbar(label_w + 1, y, w, bar_h, SERIES[k]))
            out.append(f"<text x='{label_w + w + 6}' y='{y + bar_h - 3}' "
                       f"font-size='10' fill='{INK_2}'>"
                       f"{_esc(value_fmt.format(v))}</text>")
    out.append("</svg>")
    return "\n".join(out)


def heatmap(labels: Sequence[str], values: Dict[Tuple[str, str], float],
            title: str, subtitle: str = "", width: int = 720,
            value_fmt: str = "{:.2f}") -> str:
    """Symmetric matrix (agreement) → one-hue sequential heatmap, every cell
    direct-labeled; ink flips to white on dark steps."""
    labels = list(labels)
    n = len(labels)
    cell, gap, label_w, pad = 52, 2, 96, 16
    top = 66 if subtitle else 52
    height = top + n * (cell + gap) + pad
    lo = min(values.values(), default=0.0)
    hi = max(values.values(), default=1.0)
    span = (hi - lo) or 1.0
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
           f"height='{height}' viewBox='0 0 {width} {height}' {FONT}>",
           f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
           f"<text x='{pad}' y='24' font-size='15' font-weight='600' "
           f"fill='{INK}'>{_esc(title)}</text>"]
    if subtitle:
        out.append(f"<text x='{pad}' y='42' font-size='11' "
                   f"fill='{INK_2}'>{_esc(subtitle)}</text>")
    for j, lab in enumerate(labels):
        x = label_w + j * (cell + gap)
        out.append(f"<text x='{x + cell / 2}' y='{top - 8}' font-size='10' "
                   f"text-anchor='middle' fill='{INK_2}'>{_esc(lab[:5])}</text>")
    for i, row_lab in enumerate(labels):
        y = top + i * (cell + gap)
        out.append(f"<text x='{label_w - 8}' y='{y + cell / 2 + 4}' font-size='11' "
                   f"text-anchor='end' fill='{INK}'>{_esc(row_lab[:6])}</text>")
        for j, col_lab in enumerate(labels):
            x = label_w + j * (cell + gap)
            if i == j:
                out.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                           f"rx='3' fill='{GRID}'/>")
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
            out.append(f"<text x='{x + cell / 2}' y='{y + cell / 2 + 4}' "
                       f"font-size='10' text-anchor='middle' fill='{ink}'>"
                       f"{_esc(value_fmt.format(v))}</text>")
    out.append("</svg>")
    return "\n".join(out)
