"""引文識別金標準工具（A3 Quotation Gold-standard Builder）。

`selfcheck()` 是合成基準（算法下限標尺）；走向論文級可信度需要
**人工標註金標準**。本模塊提供閉環的兩半：

1. ``build_sample(n)``：對全語料段落做確定性等距抽樣（零隨機、可復現），
   導出標註表 CSV——附算法預測列（供對比）與空白人工列（供標註）；
2. ``evaluate(csv)``：讀回已標註的表，按條文級命中計 precision/recall/F1，
   另計引用模式一致率，輸出評估報告。

標註口徑：human_clause_id 填被引條文號（如 12 / AUX_0222），確認無引用
填 0；human_mode 從｛明引 節引 暗引 化用 改寫 轉引注文 無｝中選。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from .. import config

CSV_FIELDS = ["sample_id", "stratum", "book", "chapter", "para_seq",
              "paragraph", "algo_clause_id", "algo_mode", "algo_longest_run",
              "human_clause_id", "human_mode", "notes"]


def _iter_paragraphs():
    from ..corpus import downloader, segmenter
    manifest = downloader.load_manifest()
    skip = {config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK, *config.VARIANT_BOOKS}
    for book in sorted(manifest.get("books", []),
                       key=lambda b: (b.get("category", ""), b.get("book_dir", ""))):
        bdir = book.get("book_dir", "")
        if bdir in skip:
            continue
        try:
            paragraphs = segmenter.segment_paragraphs(bdir)
        except FileNotFoundError:
            continue
        for seq, (chapter, para) in enumerate(paragraphs):
            yield bdir, chapter, seq, para


def build_sample(n: int = 50, out_path: Optional[Path] = None,
                 stratify: bool = False) -> Dict:
    """確定性抽樣 n 個段落，導出標註表（含算法預測列）。

    默認等距抽樣（最快、可復現）。``stratify=True`` 啟用分層抽樣
    （論文級評測用）：層 = 朝代 × 算法預測模式（含「無」負例層），
    份額按層規模比例分配、每層至少 1 個、層內等距——樣本不再被大部頭
    或單一文本類型主導，且仍零隨機、可復現。按*預測*模式分層是評測
    慣例（真實模式在標註前未知），最終評測建議雙人標註計一致率。"""
    from ..corpus import downloader
    from .builder import _clause_texts
    from .ids import dynasty_of
    from .quotation import QuotationScanner

    paragraphs = [p for p in _iter_paragraphs() if len(p[3]) >= 20]
    if not paragraphs:
        return {"error": "語料為空"}
    scanner = QuotationScanner(_clause_texts())

    def _predict(para: str) -> Dict:
        edges, _ = scanner.scan_paragraph(para)
        clause_edges = [e for e in edges if e.get("target_kind") == "clause"]
        return max(clause_edges, key=lambda e: (e.get("longest_run", 0),
                                                e.get("coverage", 0.0)),
                   default={})

    if stratify:
        dyn_of = {b.get("book_dir", ""): dynasty_of(b)
                  for b in downloader.load_manifest().get("books", [])}
        strata: Dict[str, List] = {}
        for bdir, chapter, seq, para in paragraphs:
            best = _predict(para)
            key = f"{dyn_of.get(bdir, '') or '未詳'}×{best.get('mode', '無')}"
            strata.setdefault(key, []).append(
                (bdir, chapter, seq, para, best))
        # 輪轉配額：層按規模降序，各層先做層內等距候選（每層至多
        # ceil(n/層數)+1 個），再逐輪各取 1 直到湊滿 n——層多於 n 時
        # 恰取最大的 n 層各 1 個，不超額返回；全程零隨機
        ordered = sorted(strata.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        per = max(1, -(-n // len(ordered)))          # ceil(n/層數)
        cand = {}
        for key, pool in ordered:
            k = min(len(pool), per + 1)
            stride = max(1, len(pool) // k)
            cand[key] = pool[::stride][:k]
        sample = []
        rnd = 0
        while len(sample) < n and any(rnd < len(c) for c in cand.values()):
            for key, _ in ordered:
                if len(sample) >= n:
                    break
                if rnd < len(cand[key]):
                    sample.append((key, *cand[key][rnd]))
            rnd += 1
        sampling_note = (f"分層抽樣：{len(strata)} 層（朝代×預測模式，含負例層），"
                         "按層規模輪轉配額（層多於 n 時取最大 n 層各 1），"
                         "層內等距，零隨機。")
    else:
        stride = max(1, len(paragraphs) // max(1, n))
        sample = [("等距", bdir, chapter, seq, para, _predict(para))
                  for bdir, chapter, seq, para in paragraphs[::stride][:n]]
        sampling_note = "等距抽樣（零隨機、可復現）；論文級評測請用 --stratify。"

    rows: List[Dict] = []
    for i, (stratum, bdir, chapter, seq, para, best) in enumerate(sample, 1):
        rows.append({
            "sample_id": f"GS_{i:03d}", "stratum": stratum,
            "book": bdir, "chapter": chapter,
            "para_seq": seq, "paragraph": para[:160],
            "algo_clause_id": best.get("clause_id", "無"),
            "algo_mode": best.get("mode", "無"),
            "algo_longest_run": best.get("longest_run", 0),
            "human_clause_id": "", "human_mode": "", "notes": "",
        })
    out = {"n_sampled": len(rows), "n_paragraph_pool": len(paragraphs),
           "n_strata": len({r["stratum"] for r in rows}),
           "note": sampling_note + " algo_* 列為算法預測，human_* 列留空"
                   "供標註；標註後用 trace-gold-eval 評估。"}
    if out_path:
        out_path = Path(out_path)
        with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        out["out"] = str(out_path)
    else:
        out["rows"] = rows
    return out


def _norm_ref(ref: str) -> str:
    ref = (ref or "").strip()
    if not ref or ref in ("0", "無", "无", "none", "None"):
        return ""
    if ref.startswith("SHL_SONGBEN"):
        return ref
    if ref.upper().startswith("AUX"):
        return config.ID_PREFIX_AUX + f"{int(ref.split('_')[-1]):04d}"
    if ref.isdigit():
        return config.ID_PREFIX_CLAUSE + f"{int(ref):04d}"
    return ref


def evaluate(csv_path: Path) -> Dict:
    """讀回已標註的金標準 CSV，計 precision / recall / F1 與模式一致率。"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"error": f"文件不存在：{csv_path}"}
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return evaluate_rows(rows)


def evaluate_rows(rows: List[Dict]) -> Dict:
    """行級評估（Web 標註工作台走此入口：瀏覽器內標註→直接回傳評估）。"""
    annotated = [r for r in rows if (r.get("human_clause_id") or "").strip()]
    if not annotated:
        return {"error": "無已標註行（human_clause_id 全為空）；"
                         "確認無引用請填 0，引用請填條文號。"}

    tp = fp = fn = tn = 0
    mode_pairs = 0
    mode_agree = 0
    disagreements: List[Dict] = []
    for r in annotated:
        human = _norm_ref(r["human_clause_id"])
        algo = r.get("algo_clause_id", "無")
        algo = "" if algo in ("無", "", "0") else _norm_ref(algo)
        if human and algo:
            if human == algo:
                tp += 1
                mode_pairs += 1
                if (r.get("human_mode") or "").strip() == r.get("algo_mode", ""):
                    mode_agree += 1
            else:
                fp += 1
                fn += 1
                disagreements.append({"sample_id": r["sample_id"],
                                      "human": human, "algo": algo})
        elif human and not algo:
            fn += 1
            disagreements.append({"sample_id": r["sample_id"],
                                  "human": human, "algo": "無"})
        elif algo and not human:
            fp += 1
            disagreements.append({"sample_id": r["sample_id"],
                                  "human": "無", "algo": algo})
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"n_annotated": len(annotated),
            "clause_level": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "precision": round(precision, 3),
                             "recall": round(recall, 3),
                             "f1": round(f1, 3)},
            "mode_agreement": (round(mode_agree / mode_pairs, 3)
                               if mode_pairs else None),
            "disagreements": disagreements[:20],
            "note": "條文級命中=人工與算法條文號一致；模式一致率僅在條文"
                    "命中的樣本上計算。人工標註是金標準，分歧樣本供誤差分析。"}
