"""跨書引文模式識別（古籍引用模式的確定性檢測）。

中醫古籍引用大多不帶規範參考文獻格式。本模塊在折疊異體字、剝離標點的
字符層上，用逐字 8-gram 倒排索引 + 對角線合併找出「後世著作 ↔ 宋本條文」
之間的最長公共片段，再按「引述標記 × 覆蓋率」分類引用模式：

| 模式   | 判定 |
|--------|------|
| 明引   | 引述標記（某曰/某云/《書名》）+ 條文覆蓋率 ≥ 0.7 |
| 節引   | 引述標記 + 覆蓋率 < 0.7（只引片段） |
| 暗引   | 無標記 + 覆蓋率 ≥ 0.7（未標出處而全文化用） |
| 化用   | 無標記 + 逐字片段 ≥ 8 字（化用原文片段） |
| 改寫   | 有標記、無逐字片段，但字二元組 Dice ≥ 0.45（意引/改寫） |
| 轉引注文 | 逐字片段命中某注家注文而非經文（經由注本轉引） |
| 存疑引用 | 有引述標記但在傷寒論條文中無可回源匹配（多為引《內經》 |
|        | 等他書或佚文；只作統計，不猜出處） |

深度改寫（無標記且無逐字片段的意譯）超出確定性方法的可靠邊界，
留給 LLM 增益層並在文檔中如實聲明，不在此偽裝成檢測能力。
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from .. import config
from ..textutil import cjk_chars, fold_variants, similarity

# ---------------------------------------------------------------------------
# 參數（全部為模塊級常量，保證可復現）
# ---------------------------------------------------------------------------
SHINGLE_CLAUSE = 8        # 條文逐字匹配最小長度
SHINGLE_COMMENT = 10      # 注文匹配 shingle 長度
MIN_COMMENT_RUN = 12      # 轉引注文最小逐字片段
FULL_QUOTE_COVERAGE = 0.7  # 全文引用覆蓋率門檻
REWRITE_SIM_LOW = 0.45    # 改寫判定的 Dice 下限
GENERIC_GRAM_CAP = 20     # 一個 8-gram 出現於超過此數條文即視為套語
MAX_SHARED_SOURCES = 3    # 同一片段可歸屬條文數上限（超過視為套語，不計邊）
MARKER_WINDOW = 16        # 引述標記回看窗口（原始字符）
REWRITE_SEGMENT = 40      # 標記後用於改寫匹配的片段長度

# 引述標記：窗口末尾的「某某曰/云/謂」
RE_MARKER_TAIL = re.compile(r"([㐀-鿿]{1,8})(曰|云|謂)[：:，。、「『]?\s*$")
# 書名號引用《傷寒論》《內經》等
RE_BOOK_TITLE = re.compile(r"《([^》]{1,12})》")
# 段落內尋找全部標記位置（改寫層）；排除對話體（問曰/答曰/師曰…非引用）
RE_MARKER_ANY = re.compile(r"([㐀-鿿])(曰|云)[：:，。、「『]?")
DIALOGUE_PREFIX = set("問答師或者故亦又對譯注釋批按")

# 標記歸屬中指向本論的線索（存疑引用再細分用）
SHANGHAN_HINTS = ("仲景", "傷寒", "本論", "論曰", "經文")


class ClauseIndex:
    """條文逐字 8-gram 倒排索引（折疊異體字、僅 CJK 字符）。"""

    def __init__(self, clause_texts: Dict[str, str]):
        # clause_id -> 折疊後純字符文本
        self.texts: Dict[str, str] = {
            cid: fold_variants("".join(cjk_chars(t))) for cid, t in clause_texts.items()
        }
        self.shingle: Dict[str, List[Tuple[str, int]]] = {}
        self.n_generic_grams = 0
        counts: Dict[str, set] = {}
        for cid in sorted(self.texts):
            t = self.texts[cid]
            for i in range(len(t) - SHINGLE_CLAUSE + 1):
                g = t[i:i + SHINGLE_CLAUSE]
                counts.setdefault(g, set()).add(cid)
                self.shingle.setdefault(g, []).append((cid, i))
        # 剔除套語 gram（出現於過多條文，無鑒別力）
        for g, cids in counts.items():
            if len(cids) > GENERIC_GRAM_CAP:
                del self.shingle[g]
                self.n_generic_grams += 1
        # 字二元組倒排 + 預計算二元組集合（改寫層的候選剪枝與快速 Dice）
        self.bigram: Dict[str, List[str]] = {}
        self.bigram_sets: Dict[str, frozenset] = {}
        for cid in sorted(self.texts):
            t = self.texts[cid]
            bgs = {a + b for a, b in zip(t, t[1:])}
            self.bigram_sets[cid] = frozenset(bgs)
            for bg in bgs:
                self.bigram.setdefault(bg, []).append(cid)


def _merge_runs(hits: List[Tuple[int, int]], k: int) -> List[Tuple[int, int, int]]:
    """把 (段落偏移, 條文偏移) 命中點按對角線合併為最長逐字片段。

    返回 [(p_start, c_start, length)]。"""
    by_diag: Dict[int, List[Tuple[int, int]]] = {}
    for p, c in hits:
        by_diag.setdefault(p - c, []).append((p, c))
    runs: List[Tuple[int, int, int]] = []
    for diag in sorted(by_diag):
        pts = sorted(by_diag[diag])
        start_p, start_c = pts[0]
        prev_p = pts[0][0]
        for p, c in pts[1:]:
            if p == prev_p + 1:
                prev_p = p
                continue
            runs.append((start_p, start_c, prev_p - start_p + k))
            start_p, start_c, prev_p = p, c, p
        runs.append((start_p, start_c, prev_p - start_p + k))
    return runs


def _covered(runs: List[Tuple[int, int, int]]) -> int:
    """條文側被覆蓋的字符數（區間并集）。"""
    ivs = sorted((c, c + ln) for _, c, ln in runs)
    total, cur_s, cur_e = 0, -1, -1
    for s, e in ivs:
        if s > cur_e:
            total += cur_e - cur_s if cur_e > cur_s else 0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s if cur_e > cur_s else 0
    return total


class ClauseMatcher:
    """單段文本 → 條文匹配（供掃描器、溯源鏈與現代文獻接口共用）。"""

    def __init__(self, index: ClauseIndex):
        self.index = index

    def fold(self, text: str) -> Tuple[str, List[int]]:
        """返回 (折疊純字符文本, 每個字符在原文中的位置)。"""
        chars, origins = [], []
        for i, ch in enumerate(text):
            if "㐀" <= ch <= "鿿":
                chars.append(ch)
                origins.append(i)
        return fold_variants("".join(chars)), origins

    def candidate_runs(self, pchars: str) -> Dict[str, List[Tuple[int, int, int]]]:
        hits: Dict[str, List[Tuple[int, int]]] = {}
        get = self.index.shingle.get
        for i in range(len(pchars) - SHINGLE_CLAUSE + 1):
            for cid, off in get(pchars[i:i + SHINGLE_CLAUSE], ()):
                hits.setdefault(cid, []).append((i, off))
        return {cid: _merge_runs(h, SHINGLE_CLAUSE) for cid, h in hits.items()}

    def fuzzy_best(self, segment: str, top: int = 1) -> List[Tuple[str, float]]:
        """改寫層：用最稀有二元組剪枝後對候選條文做 Dice 相似度。"""
        seg_bgs = frozenset(a + b for a, b in zip(segment, segment[1:]))
        if not seg_bgs:
            return []
        # 最稀有且非空桶的二元組作候選剪枝（改寫產生的新二元組桶為空，須跳過）
        bigrams = sorted((bg for bg in seg_bgs if self.index.bigram.get(bg)),
                         key=lambda bg: (len(self.index.bigram[bg]), bg))
        cand: List[str] = []
        for bg in bigrams[:4]:
            cand.extend(self.index.bigram[bg])
        scored = []
        for cid in sorted(set(cand))[:200]:
            cbgs = self.index.bigram_sets[cid]
            dice = 2 * len(seg_bgs & cbgs) / (len(seg_bgs) + len(cbgs))
            scored.append((cid, dice))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top]

    def match_text(self, text: str, limit: int = 5) -> List[Dict]:
        """把任意輸入文本回源到條文（溯源鏈 / 現代文獻接口入口）。"""
        pchars, _ = self.fold(text)
        # 短查詢（如「項背強几几」5 字）低於 8-gram 索引窗口：
        # 直接逐字子串掃描（681 條，毫秒級），十九輪修復
        if 2 <= len(pchars) < SHINGLE_CLAUSE:
            rows = []
            for cid in sorted(self.index.texts):
                ctext = self.index.texts[cid]
                if pchars in ctext:
                    rows.append({"clause_id": cid,
                                 "longest_run": len(pchars),
                                 "coverage": round(len(pchars)
                                                   / max(1, len(ctext)), 3),
                                 "matched_span": pchars})
            rows.sort(key=lambda r: (-r["coverage"], r["clause_id"]))
            return rows[:limit]
        rows = []
        for cid, runs in self.candidate_runs(pchars).items():
            longest = max(ln for _, _, ln in runs)
            ctext = self.index.texts[cid]
            rows.append({
                "clause_id": cid,
                "longest_run": longest,
                "coverage": round(_covered(runs) / max(1, len(ctext)), 3),
                "matched_span": next(pchars[p:p + ln] for p, _, ln in runs
                                     if ln == longest),
            })
        if not rows and len(pchars) >= 6:
            for cid, sim in self.fuzzy_best(pchars, top=limit):
                if sim >= REWRITE_SIM_LOW:
                    rows.append({"clause_id": cid, "longest_run": 0,
                                 "coverage": 0.0, "fuzzy_similarity": round(sim, 3)})
        rows.sort(key=lambda r: (-r["longest_run"], -r["coverage"], r["clause_id"]))
        return rows[:limit]


class CommentaryIndex:
    """注文逐字索引：檢測「轉引注文」（經由注本而非原典的間接引用）。"""

    def __init__(self, commentary_rules: List[Dict]):
        self.meta: Dict[str, Dict] = {}
        self.shingle: Dict[str, List[Tuple[str, int]]] = {}
        counts: Dict[str, set] = {}
        for r in commentary_rules:
            rid = r["commentary_rule_id"]
            t = fold_variants("".join(cjk_chars(r.get("commentary_text", ""))))
            if len(t) < SHINGLE_COMMENT:
                continue
            self.meta[rid] = {"book": r.get("book", ""),
                              "commentator": r.get("commentator", ""),
                              "clause_id": r.get("clause_id", ""),
                              "length": len(t)}
            for i in range(len(t) - SHINGLE_COMMENT + 1):
                g = t[i:i + SHINGLE_COMMENT]
                counts.setdefault(g, set()).add(rid)
                self.shingle.setdefault(g, []).append((rid, i))
        for g, rids in counts.items():
            if len(rids) > GENERIC_GRAM_CAP:
                del self.shingle[g]


def _marker_before(raw: str, pos: int) -> Tuple[str, str]:
    """檢查原始文本 pos 之前是否有引述標記。返回 (marker, attribution)。"""
    window = raw[max(0, pos - MARKER_WINDOW):pos]
    m = RE_MARKER_TAIL.search(window)
    if m:
        return m.group(1) + m.group(2), m.group(1)
    tm = RE_BOOK_TITLE.search(window)
    if tm:
        return "《" + tm.group(1) + "》", tm.group(1)
    return "", ""


class QuotationScanner:
    """全語料引文掃描器：後世著作 → 宋本條文的引文邊。"""

    def __init__(self, clause_texts: Dict[str, str],
                 commentary_rules: Optional[List[Dict]] = None):
        self.index = ClauseIndex(clause_texts)
        self.matcher = ClauseMatcher(self.index)
        self.comment_index = (CommentaryIndex(commentary_rules)
                              if commentary_rules else None)

    # -- 單段掃描 ----------------------------------------------------------
    def scan_paragraph(self, raw: str) -> Tuple[List[Dict], List[Dict]]:
        """返回 (條文引文邊, 未回源標記)。邊不含 work 級字段（由調用方補）。"""
        pchars, origins = self.matcher.fold(raw)
        if len(pchars) < SHINGLE_CLAUSE:
            return [], []
        per_clause = self.matcher.candidate_runs(pchars)

        # 套語剔除：同一段落片段可歸屬過多條文則放棄
        span_owners: Dict[Tuple[int, int], List[str]] = {}
        for cid, runs in per_clause.items():
            p, _, ln = max(runs, key=lambda r: r[2])
            span_owners.setdefault((p, ln), []).append(cid)

        edges: List[Dict] = []
        matched_positions: List[Tuple[int, int]] = []
        for cid in sorted(per_clause):
            runs = per_clause[cid]
            p_start, c_start, longest = max(runs, key=lambda r: r[2])
            shared = len(span_owners.get((p_start, longest), [cid]))
            if shared > MAX_SHARED_SOURCES:
                continue
            ctext = self.index.texts[cid]
            coverage = _covered(runs) / max(1, len(ctext))
            marker, attribution = _marker_before(raw, origins[p_start])
            if marker:
                mode = "明引" if coverage >= FULL_QUOTE_COVERAGE else "節引"
            else:
                mode = "暗引" if coverage >= FULL_QUOTE_COVERAGE else "化用"
            edges.append({
                "target_kind": "clause",
                "clause_id": cid,
                "mode": mode,
                "marker": marker,
                "attribution": attribution,
                "matched_span": pchars[p_start:p_start + longest],
                "longest_run": longest,
                "coverage": round(coverage, 3),
                "n_runs": len(runs),
                "ambiguity": shared,
            })
            matched_positions.append((p_start, p_start + longest))

        # 轉引注文層
        if self.comment_index is not None:
            edges.extend(self._scan_commentary(pchars))

        # 改寫 / 存疑引用層：段落中有標記但附近無逐字匹配
        unresolved: List[Dict] = []
        for m in RE_MARKER_ANY.finditer(raw):
            if m.group(1) in DIALOGUE_PREFIX:
                continue
            mpos = m.end()
            # 標記後的折疊片段
            seg_chars = [(i, ch) for i, ch in enumerate(raw[mpos:mpos + REWRITE_SEGMENT * 2])
                         if "㐀" <= ch <= "鿿"][:REWRITE_SEGMENT]
            if len(seg_chars) < 6:
                continue
            seg = fold_variants("".join(ch for _, ch in seg_chars))
            # 已有逐字匹配覆蓋此標記 → 略過
            fold_pos = next((fi for fi, oi in enumerate(origins) if oi >= mpos), None)
            if fold_pos is not None and any(s <= fold_pos + 4 and fold_pos - 4 <= e
                                            for s, e in matched_positions):
                continue
            best = self.matcher.fuzzy_best(seg, top=1)
            marker_text = raw[max(0, m.start() - 8):m.end()].strip()
            if best and best[0][1] >= REWRITE_SIM_LOW:
                cid, sim = best[0]
                edges.append({
                    "target_kind": "clause", "clause_id": cid, "mode": "改寫",
                    "marker": marker_text, "attribution": "",
                    "matched_span": seg[:24], "longest_run": 0,
                    "coverage": 0.0, "similarity": round(sim, 3),
                    "n_runs": 0, "ambiguity": 1,
                })
            else:
                unresolved.append({
                    "marker": marker_text,
                    "segment": seg[:20],
                    "shanghan_hint": any(h in marker_text for h in SHANGHAN_HINTS),
                })
        return edges, unresolved

    def _scan_commentary(self, pchars: str) -> List[Dict]:
        hits: Dict[str, List[Tuple[int, int]]] = {}
        get = self.comment_index.shingle.get
        for i in range(len(pchars) - SHINGLE_COMMENT + 1):
            for rid, off in get(pchars[i:i + SHINGLE_COMMENT], ()):
                hits.setdefault(rid, []).append((i, off))
        rows = []
        for rid in sorted(hits):
            runs = _merge_runs(hits[rid], SHINGLE_COMMENT)
            p_start, _, longest = max(runs, key=lambda r: r[2])
            if longest < MIN_COMMENT_RUN:
                continue
            meta = self.comment_index.meta[rid]
            rows.append({
                "target_kind": "commentary",
                "commentary_rule_id": rid,
                "via_book": meta["book"],
                "via_commentator": meta["commentator"],
                "clause_id": meta["clause_id"],
                "mode": "轉引注文",
                "marker": "", "attribution": "",
                "matched_span": pchars[p_start:p_start + longest][:30],
                "longest_run": longest,
                "coverage": round(_covered(runs) / max(1, meta["length"]), 3),
                "n_runs": len(runs),
                "ambiguity": 1,
            })
        rows.sort(key=lambda r: (-r["longest_run"], r["commentary_rule_id"]))
        return rows[:3]


# ---------------------------------------------------------------------------
# 全語料掃描
# ---------------------------------------------------------------------------
def scan_corpus(clause_texts: Dict[str, str],
                commentary_rules: Optional[List[Dict]] = None,
                verbose: bool = False) -> Dict:
    """掃描 manifest 中全部後世著作（排除 A/B 層底本），產出引文邊。

    返回 {"edges": [...], "book_stats": [...], "params": {...}}。
    掃描順序與邊編號全確定，字節級可復現。"""
    from ..corpus import downloader, segmenter
    from .ids import dynasty_of

    manifest = downloader.load_manifest()
    skip_books = {config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK, *config.VARIANT_BOOKS}
    scanner = QuotationScanner(clause_texts, commentary_rules)

    edges: List[Dict] = []
    book_stats: List[Dict] = []
    n_edge = 0
    books = sorted(manifest.get("books", []),
                   key=lambda b: (b.get("category", ""), b.get("book_dir", "")))
    for book in books:
        bdir = book.get("book_dir", "")
        if bdir in skip_books:
            continue
        try:
            paragraphs = segmenter.segment_paragraphs(bdir)
        except FileNotFoundError:
            continue
        n_book_edges = 0
        unresolved_total: List[Dict] = []
        commentator_self = ""
        info = config.COMMENTARY_BOOK_INFO.get(bdir)
        if info:
            commentator_self = info[1]
        for seq, (chapter, para) in enumerate(paragraphs):
            para_edges, unresolved = scanner.scan_paragraph(para)
            for e in para_edges:
                # 注本掃描自身注文 / 同一注家自引 → 非轉引
                if e["target_kind"] == "commentary" and (
                        e["via_book"] == bdir or
                        (commentator_self and e["via_commentator"] == commentator_self)):
                    continue
                n_edge += 1
                e.update({
                    "citation_edge_id": f"CITE_{n_edge:06d}",
                    "book_dir": bdir,
                    "book": book.get("title", bdir),
                    "author": book.get("author", ""),
                    "dynasty": dynasty_of(book),
                    "layer": book.get("hermes_layer", ""),
                    "chapter": chapter,
                    "para_seq": seq,
                })
                edges.append(e)
                n_book_edges += 1
            unresolved_total.extend(unresolved)
        book_stats.append({
            "book_dir": bdir,
            "book": book.get("title", bdir),
            "author": book.get("author", ""),
            "dynasty": dynasty_of(book),
            "layer": book.get("hermes_layer", ""),
            "n_paragraphs": len(paragraphs),
            "n_edges": n_book_edges,
            "n_marker_unresolved": len(unresolved_total),
            "unresolved_examples": unresolved_total[:3],
        })
        if verbose:
            print(f"    [trace] {bdir}: {n_book_edges} 邊 / "
                  f"{len(unresolved_total)} 存疑標記")
    return {
        "edges": edges,
        "book_stats": book_stats,
        "params": {
            "shingle_clause": SHINGLE_CLAUSE,
            "full_quote_coverage": FULL_QUOTE_COVERAGE,
            "rewrite_sim_low": REWRITE_SIM_LOW,
            "generic_gram_cap": GENERIC_GRAM_CAP,
            "n_generic_grams_dropped": scanner.index.n_generic_grams,
            "note": "存疑引用=有引述標記但傷寒論內無可回源匹配（多為引《內經》等他書），"
                    "只計數不猜出處；深度意譯檢測留給 LLM 增益層。",
        },
    }


def formula_mentions(formula_names: Iterable[str], verbose: bool = False) -> Dict:
    """方名源流：各方劑名在全部後世著作中的出現計量（長名優先防止子串誤計）。"""
    from ..corpus import downloader, segmenter
    from .ids import dynasty_of

    manifest = downloader.load_manifest()
    skip_books = {config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK, *config.VARIANT_BOOKS}
    names = sorted({n for n in formula_names if len(n) >= 3}, key=lambda n: (-len(n), n))
    per_formula: Dict[str, Dict[str, int]] = {n: {} for n in names}
    books_meta: Dict[str, Dict] = {}
    for book in sorted(manifest.get("books", []),
                       key=lambda b: (b.get("category", ""), b.get("book_dir", ""))):
        bdir = book.get("book_dir", "")
        if bdir in skip_books:
            continue
        try:
            paragraphs = segmenter.segment_paragraphs(bdir)
        except FileNotFoundError:
            continue
        text = fold_variants("\n".join(p for _, p in paragraphs))
        books_meta[bdir] = {"title": book.get("title", bdir),
                            "author": book.get("author", ""),
                            "dynasty": dynasty_of(book)}
        for name in names:  # 長名在前；計數後遮蔽，防止「四逆湯」重計「當歸四逆湯」
            folded = fold_variants(name)
            n = text.count(folded)
            if n:
                per_formula[name][bdir] = n
                text = text.replace(folded, "□" * len(folded))
    rows = []
    for name in sorted(per_formula, key=lambda n: (-sum(per_formula[n].values()), n)):
        by_book = per_formula[name]
        if not by_book:
            continue
        rows.append({
            "formula": name,
            "total_mentions": sum(by_book.values()),
            "n_books": len(by_book),
            "by_book": [{"book_dir": b, **books_meta[b], "n": by_book[b]}
                        for b in sorted(by_book, key=lambda b: (-by_book[b], b))],
        })
    return {"note": "方名逐字計量（折疊異體字；長名優先遮蔽防子串誤計），"
                    "不含 A/B 層底本自身。",
            "n_formulas_mentioned": len(rows), "formulas": rows}


# ---------------------------------------------------------------------------
# 引文邊審計（A2 Citation Evidence Auditor：每條邊的可靠性逐項核查）
# ---------------------------------------------------------------------------
def audit_citation(book_dir: str, clause_id: str,
                   clause_texts: Dict[str, str],
                   commentary_rules: Optional[List[Dict]] = None) -> Dict:
    """重掃單書，對「某書 × 某條文」的全部引文邊出具審計報告。

    每條邊給出：模式、最長逐字片段、覆蓋率、歸屬歧義（同片段可歸屬條文數）、
    是否僅達套語邊界（run==8）、是否片段引用（覆蓋率低，存在斷章風險，
    需結合上下文人工核）、是否經由注文轉引，並給出確定性可靠性分級。"""
    from ..corpus import segmenter

    scanner = QuotationScanner(clause_texts, commentary_rules)
    try:
        paragraphs = segmenter.segment_paragraphs(book_dir)
    except FileNotFoundError:
        return {"error": f"語料中無此書：{book_dir}"}
    ctext = scanner.index.texts.get(clause_id, "")
    if not ctext:
        return {"error": f"未找到條文 {clause_id}"}
    # 文獻學語義：被審計書自身（或同一注家）的注文命中是「本書注文解釋」
    # （self_commentary），只有他書命中注文才是「後世轉引」（relay_commentary）
    self_info = config.COMMENTARY_BOOK_INFO.get(book_dir)
    self_commentator = self_info[1] if self_info else ""

    rows = []
    for seq, (chapter, para) in enumerate(paragraphs):
        edges, _ = scanner.scan_paragraph(para)
        for e in edges:
            if e.get("clause_id") != clause_id:
                continue
            run = e.get("longest_run", 0)
            cov = e.get("coverage", 0.0)
            amb = e.get("ambiguity", 1)
            flags = []
            if amb > 1:
                flags.append(f"歸屬歧義：同片段可歸屬 {amb} 條條文")
            if 0 < run <= SHINGLE_CLAUSE:
                flags.append("僅達套語邊界（8 字），證據強度弱")
            if e["target_kind"] == "clause" and 0 < cov < 0.3:
                flags.append("片段引用（覆蓋率<0.3）：存在斷章風險，"
                             "結論部分未必被引及，需人工核上下文")
            mode_label = e["mode"]
            if e["target_kind"] == "commentary":
                is_self = (e.get("via_book") == book_dir or
                           (self_commentator and
                            e.get("via_commentator") == self_commentator))
                if is_self:
                    mode_label = "本書注文（self_commentary）"
                    flags.append("本注本自身注文解釋條文，非後世轉引")
                else:
                    mode_label = "轉引注文（relay_commentary）"
                    flags.append(f"經由注文轉引（{e.get('via_commentator', '')}"
                                 f"《{e.get('via_book', '')}》），非直接引經文")
            if e["mode"] == "改寫":
                flags.append("改寫判定為相似度提示（無逐字片段），可靠性低")
            if run >= 16 and amb == 1 and e["target_kind"] == "clause":
                reliability = "高"
            elif run >= 10 or cov >= 0.5:
                reliability = "中"
            else:
                reliability = "低"
            rows.append({"chapter": chapter, "para_seq": seq,
                         "paragraph_excerpt": para[:80],
                         "mode": mode_label, "marker": e.get("marker", ""),
                         "longest_run": run, "coverage": cov,
                         "ambiguity": amb, "matched_span": e.get("matched_span", ""),
                         "reliability": reliability, "flags": flags})
    rows.sort(key=lambda r: ({"高": 0, "中": 1, "低": 2}[r["reliability"]],
                             r["para_seq"]))
    counts = {}
    for r in rows:
        counts[r["reliability"]] = counts.get(r["reliability"], 0) + 1
    return {"book_dir": book_dir, "clause_id": clause_id,
            "clause_text": clause_texts.get(clause_id, "")[:80],
            "n_edges": len(rows),
            "reliability_counts": {k: counts[k] for k in sorted(counts)},
            "edges": rows,
            "note": "可靠性分級為確定性規則（片段長度/覆蓋率/歧義度/轉引），"
                    "「斷章風險」僅為提示，語義層面的斷章取義需人工判定。"}


# ---------------------------------------------------------------------------
# 全庫掃描（中醫笈成 800+ 部，文獻旁證層；`library fetch` 後可用）
# ---------------------------------------------------------------------------
def scan_library(clause_texts: Dict[str, str],
                 category: str = "", limit: int = 0,
                 root=None, verbose: bool = False) -> Dict:
    """把引文掃描的「引用方」擴展到中醫笈成全庫（803 部醫籍）。

    被引靶集仍為傷寒論條文；引用方覆蓋全庫任意分類（本草/方書/醫案/
    溫病/內科…），故可回答「傷寒條文在整個醫籍傳統中的傳播」。庫屬
    文獻旁證層：邊照常逐字回源，但出處標 layer=旁證，不進入經文閘門。
    庫未下載時如實返回不可用狀態（不自動觸發 69MB 下載）。"""
    from ..corpus import library

    if not library.is_available(root):
        return {"available": False,
                "note": "中醫笈成全庫未下載：運行 `python3 -m hermes_shanghan "
                        "library fetch`（約 69MB，sha256 校驗）後重試。",
                "edges": [], "book_stats": []}
    import json as _json
    catalog = _json.loads((library.library_root(root) / library.CATALOG_NAME)
                          .read_text(encoding="utf-8"))
    from ..corpus.catalog import parse_sections
    from ..textutil import strip_markup
    scanner = QuotationScanner(clause_texts)

    # 每個 unit 只讀自身文件（非遞歸），父書與子書不會重複掃描
    units = list(catalog.get("units", []))
    if category:
        units = [u for u in units if category in (u.get("category") or "")]
    units.sort(key=lambda u: u["id"])
    if limit:
        units = units[:limit]

    edges: List[Dict] = []
    book_stats: List[Dict] = []
    n_edge = 0
    for u in units:
        unit_dir = library.books_dir(root) / u["id"]
        try:
            text = library.read_unit_text(unit_dir)
        except OSError:
            continue
        n_book_edges = 0
        n_unresolved = 0
        seq = 0
        for sec in parse_sections(text):
            for para in sec.paragraphs:
                clean = strip_markup(para)
                if len(clean) < SHINGLE_CLAUSE:
                    seq += 1
                    continue
                para_edges, unresolved = scanner.scan_paragraph(clean)
                n_unresolved += len(unresolved)
                for e in para_edges:
                    n_edge += 1
                    e.update({
                        "citation_edge_id": f"LCITE_{n_edge:06d}",
                        "book_dir": u["id"],
                        "book": u.get("title", u["id"]),
                        "author": u.get("author", ""),
                        "dynasty": u.get("dynasty", ""),
                        "category": u.get("category", ""),
                        "layer": "旁證",
                        "chapter": sec.title,
                        "para_seq": seq,
                    })
                    edges.append(e)
                    n_book_edges += 1
                seq += 1
        if n_book_edges or verbose:
            book_stats.append({
                "book_id": u["id"], "book": u.get("title", u["id"]),
                "author": u.get("author", ""), "dynasty": u.get("dynasty", ""),
                "category": u.get("category", ""),
                "n_edges": n_book_edges,
                "n_marker_unresolved": n_unresolved,
            })
        if verbose and n_book_edges:
            print(f"    [trace/library] {u['id']}: {n_book_edges} 邊")
    book_stats.sort(key=lambda b: (-b["n_edges"], b["book_id"]))
    return {"available": True, "n_units_scanned": len(units),
            "n_edges": len(edges), "edges": edges, "book_stats": book_stats,
            "note": "全庫掃描屬文獻旁證層（layer=旁證）：邊逐字回源到條文，"
                    "但出處不進入經文層證據閘門。朝代為漢/東漢的單元是仲景書"
                    "自身的庫內版本（版本見證），其邊不是後世引用，按朝代過濾。"}


# ---------------------------------------------------------------------------
# 引文識別自檢基準（評價體系：引文識別能力）
# ---------------------------------------------------------------------------
def selfcheck(clause_texts: Dict[str, str], step: int = 20) -> Dict:
    """確定性合成 明引/節引/暗引/改寫 正例與負例，計各模式檢出率。

    合成變換全部確定（取樣步長、截斷位置、刪字步長固定），零隨機性；
    「檢出」= 掃描產出指向正確條文的引文邊；另計模式判定一致率與
    負例誤報率。這是合成基準（衡量算法下限），非人工標註金標準。"""
    scanner = QuotationScanner(clause_texts)
    folded = scanner.index.texts
    sample = [cid for i, cid in enumerate(sorted(folded))
              if i % step == 0 and len(folded[cid]) >= 24]

    def _synth(cid: str, mode: str) -> Optional[str]:
        t = folded[cid]
        if mode == "明引":
            return f"故仲景曰：{t}。此其旨也。"
        if mode == "節引":
            frag = t[len(t) // 3: len(t) // 3 + 10]
            return f"經云{frag}，即此謂也。"
        if mode == "暗引":
            return f"蓋{t}，學者所當識也。"
        if mode == "改寫":
            if len(t) > 45:      # 改寫比對段有界，長條文如實跳過
                return None
            body = "".join(ch for i, ch in enumerate(t) if i % 5 != 4)
            return f"論云：{body}。"
        return None

    per_mode: Dict[str, Dict] = {}
    for mode in ("明引", "節引", "暗引", "改寫"):
        n = detected = mode_ok = 0
        for cid in sample:
            para = _synth(cid, mode)
            if para is None:
                continue
            n += 1
            edges, _ = scanner.scan_paragraph(para)
            hit = next((e for e in edges if e.get("clause_id") == cid), None)
            if hit:
                detected += 1
                if hit["mode"] == mode:
                    mode_ok += 1
        per_mode[mode] = {"n": n,
                          "detection_rate": round(detected / n, 3) if n else 0.0,
                          "mode_agreement": round(mode_ok / n, 3) if n else 0.0}

    # 負例：條文倒序文本（共享字符但無逐字片段/無標記）不應產生指向原條文的邊
    n_neg = fp = 0
    for cid in sample:
        para = folded[cid][::-1]
        n_neg += 1
        edges, _ = scanner.scan_paragraph(para)
        if any(e.get("clause_id") == cid and e.get("longest_run", 0) >= SHINGLE_CLAUSE
               for e in edges):
            fp += 1
    return {"note": "確定性合成基準（算法下限標尺，非人工金標準）。",
            "n_sampled_clauses": len(sample),
            "per_mode": per_mode,
            "negative": {"n": n_neg,
                         "false_positive_rate": round(fp / n_neg, 3) if n_neg else 0.0}}
