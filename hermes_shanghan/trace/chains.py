"""五類溯源鏈：結構化溯源報告生成。

| 鏈 | 回答 | 鏈路 |
|----|------|------|
| 原文溯源鏈 | 這句話從哪裡來 | 條文 → 異文 → 上下文 → 注家 → 後世引用 → 計量 → 現代 |
| 方劑源流鏈 | 這個方如何演變 | 首見條文 → 組成/劑量 → 類方演化 → 方名傳播 → 方證觀點 |
| 方證觀點鏈 | 某方為何對應某證 | 原文直述檢驗 → 注家首倡時間線 → 學派立場 → 現代回聲 |
| 注家解釋鏈 | 後世如何解釋原文 | 注家 → 學派 → 對齊覆蓋 → 術語指紋 → 被轉引樞紐度 |
| 學派觀點鏈 | 不同學派為何不同 | 範式 → 成員著作 → 派內/跨派一致度 → 對立學派 |

每份報告攜帶 evidence_grade（實際用到的證據層）與 warnings（後世歸納
與原文直述的區分提示），所有 clause_id 可被 CitationGuard 逐字核驗。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .. import config
from ..schemas import read_jsonl
from ..textutil import fold_variants, normalize_query
from . import builder
from .ids import dynasty_order
from .modern import modern_echo_for

EXCERPT = 80

RE_NUM = re.compile(r"^\d{1,4}$")


# ---------------------------------------------------------------------------
# 公共取數
# ---------------------------------------------------------------------------
def _clauses() -> Dict[str, Dict]:
    return {c["clause_id"]: c for c in read_jsonl(config.CLAUSE_DIR / "clauses.jsonl")}


def _resolve_clause(ref: str, clauses: Dict[str, Dict]) -> Optional[Dict]:
    ref = (ref or "").strip()
    if ref in clauses:
        return clauses[ref]
    m = RE_NUM.match(ref.lstrip("第").rstrip("條条"))
    if m:
        cid = config.ID_PREFIX_CLAUSE + f"{int(m.group(0)):04d}"
        return clauses.get(cid)
    return None


def _citations_by_dynasty(clause_ids: List[str], max_books: int = 6) -> Dict:
    """（著作,條文）聚合邊 → 按朝代分組的引用概覽。"""
    wanted = set(clause_ids)
    rows = [r for r in builder.load_agg_edges() if r["clause_id"] in wanted]
    by_dyn: Dict[str, Dict] = {}
    for r in rows:
        dyn = r["dynasty"] or "未詳"
        s = by_dyn.setdefault(dyn, {"dynasty": dyn,
                                    "dynasty_order": dynasty_order(dyn),
                                    "books": {}})
        b = s["books"].setdefault(r["book_dir"], {
            "book": r["book"], "book_dir": r["book_dir"],
            "author": r["author"], "n_paragraphs": 0,
            "modes": {}, "max_coverage": 0.0})
        b["n_paragraphs"] += r["n_paragraphs"]
        b["max_coverage"] = max(b["max_coverage"], r["max_coverage"])
        for m, n in r["modes"].items():
            b["modes"][m] = b["modes"].get(m, 0) + n
    out = []
    for dyn in sorted(by_dyn, key=lambda d: (by_dyn[d]["dynasty_order"], d)):
        s = by_dyn[dyn]
        books = sorted(s["books"].values(),
                       key=lambda b: (-b["n_paragraphs"], b["book"]))
        out.append({"dynasty": dyn, "n_books": len(books),
                    "books": [{**b, "modes": {m: b["modes"][m]
                                              for m in sorted(b["modes"])}}
                              for b in books[:max_books]]})
    return {"n_citing_books": len({b for d in by_dyn.values() for b in d["books"]}),
            # 十七輪：UI 據此把「某書引用」點開為段落級查閱
            # （POST /api/trace/passages {book_dir, clause_ids}）
            "cited_clause_ids": sorted(wanted),
            "by_dynasty": out}


def _main_path(clause_id: str) -> List[Dict]:
    per_dyn: Dict[str, Dict] = {}
    for r in builder.load_agg_edges():
        if r["clause_id"] != clause_id:
            continue
        dyn = r["dynasty"] or "未詳"
        cur = per_dyn.get(dyn)
        key = (r["max_coverage"], r["max_run"])
        if cur is None or key > (cur["max_coverage"], cur["max_run"]):
            per_dyn[dyn] = {"dynasty": dyn, "dynasty_order": dynasty_order(dyn),
                            "book": r["book"], "author": r["author"],
                            "max_coverage": r["max_coverage"],
                            "max_run": r["max_run"]}
    chain = sorted(per_dyn.values(), key=lambda x: (x["dynasty_order"], x["book"]))
    return ([{"dynasty": "東漢", "book": "傷寒論", "author": "張仲景",
              "max_coverage": 1.0, "max_run": 0}] + chain)


def _commentaries_for(clause_id: str, schools_reg: Dict) -> List[Dict]:
    rows = []
    seen = set()
    member_school = schools_reg.get("commentator_school", {})
    registry = builder.load_registry()
    dyn_of_dir = {w["book_dir"]: w["dynasty"] for w in registry["works"]}
    for r in read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl"):
        if r.get("clause_id") != clause_id:
            continue
        commentator = r.get("commentator", "")
        if commentator in seen:
            continue
        seen.add(commentator)
        dyn = dyn_of_dir.get(r.get("book", ""), "")
        rows.append({"commentator": commentator, "book": r.get("book", ""),
                     "chapter": r.get("chapter", ""),
                     "dynasty": dyn, "dynasty_order": dynasty_order(dyn),
                     "school_id": member_school.get(commentator, ""),
                     "excerpt": r.get("commentary_text", "")[:EXCERPT]})
    rows.sort(key=lambda x: (x["dynasty_order"], x["commentator"]))
    return rows


# ---------------------------------------------------------------------------
# 1. 原文溯源鏈
# ---------------------------------------------------------------------------
def clause_chain(ref: str) -> Dict:
    clauses = _clauses()
    c = _resolve_clause(ref, clauses)
    if c is None:
        return {"error": f"未找到條文 {ref}（可用條文號 1-398 或 clause_id）"}
    cid = c["clause_id"]
    schools_reg = builder.load_schools()

    variants = [v for v in read_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl")
                if v.get("clause_id") == cid]
    n = c.get("clause_number", 0)
    prev_id = config.ID_PREFIX_CLAUSE + f"{n-1:04d}" if n > 1 else ""
    next_id = config.ID_PREFIX_CLAUSE + f"{n+1:04d}" if 0 < n < 398 else ""
    commentaries = _commentaries_for(cid, schools_reg)
    citations = _citations_by_dynasty([cid])
    network = builder.load_network()
    bursts = [b for b in network.get("bursts", []) if b["clause_id"] == cid]
    modern = modern_echo_for([cid])

    grade = ["A 原文直述"]
    if variants:
        grade.append("B 版本異文")
    if commentaries:
        grade.append("C 注家解釋")
    if citations["n_citing_books"]:
        grade.append("後世引文邊（逐字回源）")
    if modern.get("available") and modern.get("n_citations"):
        grade.append("現代學術引用（導入層）")

    return {
        "chain_type": "原文溯源鏈",
        "query": ref,
        "clause": {"clause_id": cid, "clause_number": n,
                   "chapter": c.get("chapter", ""),
                   "six_channel": c.get("six_channel", ""),
                   "text": c.get("clean_text", "")},
        "variants": [{"variant_book": v.get("variant_book", ""),
                      "similarity": v.get("similarity", 0.0),
                      "notable_differences": v.get("notable_differences", []),
                      "variant_text": v.get("variant_text", "")[:EXCERPT]}
                     for v in variants],
        "context": {"prev_clause_id": prev_id if prev_id in clauses else "",
                    "next_clause_id": next_id if next_id in clauses else ""},
        "commentaries": commentaries,
        "citations": citations,
        "main_path": _main_path(cid),
        "bursts": bursts,
        "modern": modern,
        "evidence_grade": grade,
        "section_evidence_levels": {
            "clause": "A 原文直述",
            "variants": "B 版本異文",
            "context": "A 原文直述（篇次相鄰）",
            "commentaries": "C 注家解釋",
            "citations": "引文邊（跨書逐字回源）",
            "main_path": "計量推導（基於引文邊）",
            "bursts": "計量推導（基於引文邊）",
            "modern": "現代導入層（用戶自備，不隨庫分發）",
        },
        "warnings": ["注家解釋與後世引用均屬 C/D 層，不得回填為原文直述；"
                     "「化用/暗引」為逐字片段證據，改寫判定僅為相似度提示。"],
    }


# ---------------------------------------------------------------------------
# 2. 方劑源流鏈
# ---------------------------------------------------------------------------
def formula_chain(name: str) -> Dict:
    q = normalize_query(name)
    rules = read_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")
    rule = next((r for r in rules if fold_variants(r.get("formula", "")) == q), None)
    if rule is None:
        rule = next((r for r in rules if q and q in fold_variants(r.get("formula", ""))), None)
    if rule is None:
        return {"error": f"未找到方劑 {name}"}
    formula = rule["formula"]
    supporting = rule.get("supporting_clauses", [])
    canonical_ids = sorted(c for c in supporting if "AUX" not in c)
    auxiliary_ids = sorted(c for c in supporting if "AUX" in c)
    first_id = canonical_ids[0] if canonical_ids else (
        auxiliary_ids[0] if auxiliary_ids else "")

    # 劑量與類方演化（劑量計量層資產）
    ratios = {}
    ratios_path = config.RESEARCH_DIR / "dose_ratios.json"
    if ratios_path.exists():
        data = json.loads(ratios_path.read_text(encoding="utf-8"))
        ratios = next((r for r in data.get("formulas", [])
                       if r.get("formula") == formula), {})
    evolution = []
    evo_path = config.RESEARCH_DIR / "dose_family_evolution.json"
    if evo_path.exists():
        data = json.loads(evo_path.read_text(encoding="utf-8"))
        evolution = [e for e in data.get("edges", [])
                     if formula in (e.get("base", ""), e.get("modified", ""))]

    all_mentions = builder.load_formula_mentions().get("formulas", [])
    mentions = next((f for f in all_mentions if f.get("formula") == formula), None)
    mention_rows = []
    if mentions:
        mention_rows = sorted(
            mentions["by_book"],
            key=lambda b: (dynasty_order(b.get("dynasty", "")), b.get("book_dir", "")))
    # 異名歸並（編輯性對照表 + 組成比對）：異名計量與正名分列，不混計
    from .aliases import aliases_for
    alias_info = []
    for a in aliases_for(formula):
        am = next((f for f in all_mentions if f.get("formula") == a["alias"]), None)
        alias_info.append({**a,
                           "alias_mentions": (am or {}).get("total_mentions", 0),
                           "alias_n_books": (am or {}).get("n_books", 0)})

    claims = [c for c in builder.load_claims().get("claims", [])
              if c.get("formula") == formula]
    citations = _citations_by_dynasty(supporting)
    modern = modern_echo_for(supporting)
    anchor = first_id
    schools_reg = builder.load_schools()

    return {
        "chain_type": "方劑源流鏈",
        "query": name,
        "formula": formula,
        # 「首見」指宋本條文序中的首次出現；支持條文全集另列並分正文/輔助
        # （不與首見混同——跨書史源（如湯液經法之爭）庫外不作臆斷）
        "first_attestation": {"work": "傷寒論（宋本）", "clause_id": first_id,
                              "core_pattern": rule.get("core_pattern", ""),
                              "note": "首見=宋本條文序中的首次出現，"
                                      "非跨書史源判定"},
        "supporting_clauses": {"canonical": canonical_ids,
                               "auxiliary": auxiliary_ids},
        "composition": rule.get("composition", []),
        "administration_notes": rule.get("administration_notes", [])[:3],
        "dose_ratios": ratios,
        "modification_relations": rule.get("modification_relations", []),
        "family_dose_evolution": evolution,
        "name_transmission": {"total_mentions": (mentions or {}).get("total_mentions", 0),
                              "n_books": (mentions or {}).get("n_books", 0),
                              "by_book": mention_rows[:15],
                              "aliases": alias_info},
        "claims": [{"claim_id": c["claim_id"], "claim": c["claim"],
                    "evidence_grade": c["evidence_grade"]} for c in claims],
        "anchor_commentaries": _commentaries_for(anchor, schools_reg)[:6] if anchor else [],
        "citations_of_clauses": citations,
        "modern": modern,
        "evidence_grade": ["A 原文直述（條文與組成）", "D 劑量/類方計量",
                           "方名逐字計量", "後世引文邊"],
        "section_evidence_levels": {
            "first_attestation": "A 原文直述（宋本條文序首見）",
            "supporting_clauses": "A 原文直述（正文/輔助分列）",
            "composition": "A 原文直述（<F> 方塊）",
            "administration_notes": "A 原文直述",
            "dose_ratios": "A 銖當量藥量比（克數折算屬 D 層假設）",
            "modification_relations": "D 類方歸納",
            "family_dose_evolution": "D 劑量計量歸納",
            "name_transmission": "方名逐字計量（跨書）",
            "claims": "方證觀點（分級見各條 evidence_grade）",
            "anchor_commentaries": "C 注家解釋",
            "citations_of_clauses": "引文邊（跨書逐字回源）",
            "modern": "現代導入層（用戶自備，不隨庫分發）",
        },
        "warnings": ["主治演變與方義解釋屬注文層歸納；方名計量為逐字統計；"
                     "異名（如陽旦湯）經編輯性對照表歸並且與正名分列計量，"
                     "組成存疑者標不可合併（見 name_transmission.aliases）。"],
    }


# ---------------------------------------------------------------------------
# 3. 方證觀點演化鏈
# ---------------------------------------------------------------------------
def claim_chain(key: str) -> Dict:
    q = normalize_query(key)
    claims = builder.load_claims().get("claims", [])
    claim = next((c for c in claims if c["claim_id"] == key), None)
    if claim is None:
        claim = next((c for c in claims
                      if q and (q in fold_variants(c["formula"])
                                or q in fold_variants(c["claim"]))), None)
    if claim is None:
        available = [c["claim_id"] + " " + c["claim"] for c in claims]
        return {"error": f"未找到方證觀點 {key}", "available_claims": available}

    schools_reg = builder.load_schools()
    school_names = {s["school_id"]: s["name"] for s in schools_reg.get("schools", [])}
    citations = _citations_by_dynasty(claim.get("classical_evidence", []))
    modern = modern_echo_for(claim.get("classical_evidence", []))
    return {
        "chain_type": "方證觀點演化鏈",
        "query": key,
        **claim,
        "school_views_named": [{"school_id": s, "name": school_names.get(s, s)}
                               for s in claim.get("school_views", [])],
        "citations_of_evidence": citations,
        "modern": modern,
        "section_evidence_levels": {
            "classical_evidence": "A 原文條文（clause_id）",
            "terms_verbatim_in_original": "A 原文逐字檢驗",
            "commentarial_chronology": "C 注家解釋（按朝代排序）",
            "school_views_named": "posthoc_induction 學派歸納",
            "controversies": "posthoc_induction 編輯性整理",
            "citations_of_evidence": "引文邊（跨書逐字回源）",
            "modern": "現代導入層（用戶自備，不隨庫分發）",
        },
        "warnings": [claim.get("warning", ""),
                     "多觀點並存：學派立場不做對錯裁決。"],
    }


# ---------------------------------------------------------------------------
# 4. 注家解釋鏈
# ---------------------------------------------------------------------------
def commentator_chain(name: str) -> Dict:
    q = normalize_query(name)
    schools_reg = builder.load_schools()
    member_school = schools_reg.get("commentator_school", {})
    match = next((c for c in sorted(member_school)
                  if q and q in fold_variants(c)), None)
    atlas_path = config.RESEARCH_DIR / "commentary_divergence.json"
    atlas = (json.loads(atlas_path.read_text(encoding="utf-8"))
             if atlas_path.exists() else {})
    coverage = [
        {"book": b, **info} for b, info in sorted(atlas.get("book_coverage", {}).items())
        if q and q in fold_variants(info.get("commentator", ""))]
    if match is None and not coverage:
        return {"error": f"未找到注家 {name}",
                "known_commentators": sorted(member_school)}
    commentator = match or coverage[0]["commentator"]
    school_id = member_school.get(commentator, "")
    school = next((s for s in schools_reg.get("schools", [])
                   if s["school_id"] == school_id), None)

    fingerprints = atlas.get("commentator_fingerprints", {}).get(commentator, [])
    agreements = [row for row in atlas.get("agreement_matrix", [])
                  if commentator in (row.get("a"), row.get("b"))]
    agreements.sort(key=lambda r: -r.get("mean_term_agreement", 0.0))

    # 被轉引樞紐度：後世著作經由該注家注文轉引的計量
    relayed = [r for r in builder.load_relay_edges()
               if r.get("via_commentator") == commentator]
    relayed.sort(key=lambda r: (-r["n_paragraphs"], r["book_dir"]))

    return {
        "chain_type": "注家解釋鏈",
        "query": name,
        "commentator": commentator,
        "school": ({"school_id": school_id, "name": school["name"],
                    "paradigm": school["paradigm"]} if school else {}),
        "aligned_books": coverage,
        "fingerprint_terms": fingerprints[:10],
        "agreement_with_peers": agreements[:6],
        "relay_hub": {"n_relaying_books": len({r["book_dir"] for r in relayed}),
                      "top": [{"book": r["book"], "dynasty": r["dynasty"],
                               "n_paragraphs": r["n_paragraphs"]}
                              for r in relayed[:8]]},
        "evidence_grade": ["C 注家解釋（條文級對齊）", "轉引邊（逐字回源）",
                           "學派歸屬（posthoc_induction）"],
        "section_evidence_levels": {
            "aligned_books": "C 注家對齊（計算資產）",
            "fingerprint_terms": "C 層計算資產（詞彙 lift）",
            "agreement_with_peers": "C 層計算資產（實測一致度）",
            "relay_hub": "轉引邊（跨書逐字回源）",
            "school": "posthoc_induction 學派歸納",
        },
        "warnings": ["注家指紋與一致度為 C 層計算資產；學派歸屬為編輯性元數據。"],
    }


# ---------------------------------------------------------------------------
# 5. 學派觀點鏈
# ---------------------------------------------------------------------------
def school_chain(key: str) -> Dict:
    q = normalize_query(key)
    schools_reg = builder.load_schools()
    schools = schools_reg.get("schools", [])
    school = next((s for s in schools if s["school_id"] == key), None)
    if school is None:
        school = next((s for s in schools
                       if q and (q in fold_variants(s["name"])
                                 or any(q in fold_variants(m["name"])
                                        for m in s["members"]))), None)
    if school is None:
        return {"error": f"未找到學派 {key}",
                "available_schools": [{"school_id": s["school_id"], "name": s["name"]}
                                      for s in schools]}
    school_names = {s["school_id"]: s["name"] for s in schools}
    network = builder.load_network()
    breadth = {w["book_dir"]: w for w in network.get("citing_works", [])}
    member_works = []
    for m in school["members"]:
        for bdir in m["book_dirs"]:
            w = breadth.get(bdir)
            member_works.append({
                "member": m["name"], "book_dir": bdir,
                "n_clauses_cited": (w or {}).get("n_clauses_cited", 0),
                "n_edges": (w or {}).get("n_edges", 0)})
    return {
        "chain_type": "學派觀點鏈",
        "query": key,
        **{k: school[k] for k in ("school_id", "name", "paradigm", "scope",
                                  "members", "agreement", "source_level")},
        "opposed_to": [{"school_id": s, "name": school_names.get(s, s)}
                       for s in school.get("opposed_to", [])],
        "member_citation_breadth": member_works,
        "basis": schools_reg.get("note", ""),
        "section_evidence_levels": {
            "paradigm": "posthoc_induction 學派歸納",
            "members": "posthoc_induction（僅收語料在庫著者）",
            "agreement": "C 層實測一致度（分歧圖譜）",
            "member_citation_breadth": "引文邊計量",
        },
        "warnings": ["學派歸屬為後世歸納（posthoc_induction）；"
                     "一致度證據來自注家分歧圖譜實測。"],
    }


# ---------------------------------------------------------------------------
# 6. 任意文本回源（原文溯源鏈入口）
# ---------------------------------------------------------------------------
def text_trace(text: str) -> Dict:
    matcher = builder.get_matcher()
    matches = matcher.match_text(normalize_query(text), limit=5)
    if not matches:
        out = {"chain_type": "原文溯源鏈", "query": text,
               "matches": [],
               "note": "傷寒論條文（含輔助篇章）內無可回源匹配；"
                       "該句可能出自他書（如《內經》）或為後世歸納語。"}
        out["library_candidates"] = _library_candidates(text)
        return out
    best = matches[0]
    chain = clause_chain(best["clause_id"])
    chain["query"] = text
    chain["matches"] = matches
    return chain


def _library_candidates(text: str, limit: int = 6) -> Dict:
    """傷寒論內無匹配時，退到中醫笈成全庫（若已下載）找候選出處。

    只做逐字全文檢索並回報「書·章節」定位（文獻旁證層），不臆斷首出。"""
    from ..corpus import library
    if not library.is_available():
        return {"available": False,
                "note": "全庫未下載（`library fetch` 後，可在 800+ 部醫籍中"
                        "檢索該句的候選出處）。"}
    q = normalize_query(text)
    q = "".join(ch for ch in q if "㐀" <= ch <= "鿿")[:20]
    if len(q) < 4:
        return {"available": True, "hits": [],
                "note": "查詢過短，不作全庫檢索。"}
    res = library.Library().grep(q, limit=limit)
    return {"available": True,
            "query": q,
            "n_hits": res.get("n_hits", 0),
            "scan_capped": res.get("scan_capped", False),
            # book_id/section/excerpt 隨返回（十六輪）：候選出處可經
            # shanghan_library(book=…, section=…) 直接點閱章節原文
            "hits": [{k: h.get(k, "") for k in
                      ("book_id", "title", "author", "dynasty", "category",
                       "section", "excerpt")}
                     for h in res.get("hits", [])],
            "note": "文獻旁證層：按書·章節定位候選出處，可點閱原文、"
                    "需人工核對；全庫檢索不臆斷「最早出處」"
                    "（庫外文獻與版本先後未覆蓋）。"}


# ---------------------------------------------------------------------------
# 6b. 方解（C11 formula-explain：一站式方劑檔案，組合既有確定性資產）
# ---------------------------------------------------------------------------
def formula_explain(name: str) -> Dict:
    """方解一站式檔案：源流鏈 + 方證規則 + 類方鑒別 + 禁忌 + 煎服法。"""
    chain = formula_chain(name)
    if "error" in chain:
        return chain
    formula = chain["formula"]
    rule = next((r for r in read_jsonl(config.RULES_FORMULA_DIR /
                                       "formula_pattern_rules.jsonl")
                 if r.get("formula") == formula), {})
    differentials = []
    for d in read_jsonl(config.RULES_DIFFERENTIAL_DIR / "differential_rules.jsonl"):
        if formula in d.get("formulas", []):
            differentials.append({
                "vs": [f for f in d["formulas"] if f != formula],
                "key_discriminators": d.get("key_discriminators", [])[:3],
                "supporting_clauses": d.get("supporting_clauses", [])[:3]})
        if len(differentials) >= 3:
            break

    # 三層症狀口徑（評審問題 5）：首見方證核心證 ≠ 全書相關表現 ≠ 特殊
    # 上下文（誤治/禁忌/傳變）證——混排會讓醫師把全書聚合誤讀為標準核心證
    clauses = _clauses()
    first_id = chain["first_attestation"]["clause_id"]
    first_c = clauses.get(first_id, {})
    aggregate: Dict[str, int] = {}
    special = []
    all_support = (chain["supporting_clauses"]["canonical"]
                   + chain["supporting_clauses"]["auxiliary"])
    for cid in all_support:
        c = clauses.get(cid)
        if not c:
            continue
        ctx = []
        if c.get("mistreatment_terms"):
            ctx.append("誤治")
        if c.get("contraindication_terms"):
            ctx.append("禁忌")
        if c.get("transformation_terms"):
            ctx.append("傳變")
        for s in c.get("symptoms", []):
            aggregate[s] = aggregate.get(s, 0) + 1
        if ctx:
            special.append({"clause_id": cid, "context": ctx,
                            "symptoms": c.get("symptoms", [])[:6]})
    symptom_layers = {
        "first_attestation": {"clause_id": first_id,
                              "symptoms": first_c.get("symptoms", []),
                              "pulse": first_c.get("pulse", [])},
        "rule_induced_core": {"symptoms": rule.get("core_symptoms", []),
                              "pulse": rule.get("core_pulse", []),
                              "associated": rule.get("associated_symptoms", [])[:8],
                              "source": "D 方證規則歸納（跨條聚合，證據錨定 A 層）"},
        "aggregate_all_clauses": [
            {"symptom": s, "n_clauses": n}
            for s, n in sorted(aggregate.items(),
                               key=lambda kv: (-kv[1], kv[0]))[:15]],
        "special_context": special[:10],
        "note": "四層口徑：首見層=首見條文直接所載（默認優先展示）；"
                "規則歸納層=方證規則跨條歸納（D 層，非原文首見核心證）；"
                "聚合層=全書相關條文表現總和（含誤治/禁忌/變證上下文），"
                "不得徑作標準方證核心證；特殊上下文層單列以防誤讀。",
    }
    return {
        "chain_type": "方解檔案",
        "formula": formula,
        "first_attestation": chain["first_attestation"],
        "supporting_clauses": chain["supporting_clauses"],
        # 症狀只經 symptom_layers 四層口徑輸出（首見層在前，默認優先展示）；
        # 頂層不再放 core_symptoms——規則歸納核心證易被誤讀為原文首見核心證
        "symptom_layers": symptom_layers,
        "composition": chain["composition"],
        "dose_ratios": chain["dose_ratios"],
        # 煎服法獨立成節：方後注本身就是治療法度的一部分
        "administration": {
            "preparation": next((fb.get("preparation", "")
                                 for fb in first_c.get("formula_blocks", [])
                                 if formula in fb.get("formula_name", "")), ""),
            "administration": next((fb.get("administration", "")
                                    for fb in first_c.get("formula_blocks", [])
                                    if formula in fb.get("formula_name", "")), ""),
            "post_notes": next((fb.get("post_notes", [])
                                for fb in first_c.get("formula_blocks", [])
                                if formula in fb.get("formula_name", "")), []),
            "rule_notes": rule.get("administration_notes", [])[:5],
            # 出處錨點（十九輪）：煎法/服法皆出自首見條文的 <F> 方塊
            "source": {"clause_id": first_id,
                       "book": "傷寒論（宋本，趙開美本）",
                       "chapter": first_c.get("chapter", "")},
            "warning": "古籍煎服法 ≠ 現代可直接執行醫囑：劑量制式、藥材炮製與"
                       "服藥調護均需專業醫師按現代規範轉換。",
        },
        "contraindications": rule.get("contraindications", [])[:5],
        "modification_relations": chain["modification_relations"],
        "family_dose_evolution": chain["family_dose_evolution"],
        "differentials": differentials,
        "claims": chain["claims"],
        "anchor_commentaries": chain["anchor_commentaries"],
        "name_transmission": chain["name_transmission"],
        "citations_of_clauses": chain["citations_of_clauses"],
        "section_evidence_levels": {
            **chain["section_evidence_levels"],
            "symptom_layers": "四層口徑分列（首見 A / 規則歸納 D / 聚合 A 標註 /"
                              "特殊上下文 A 標註），見其 note",
            "administration": "A 原文煎服法（rule_notes 屬 D 歸納；"
                              "附「非現代醫囑」警示）",
            "contraindications": "A 原文禁例",
            "differentials": "D 鑒別歸納（supporting_clauses 回源）",
        },
        "warnings": chain["warnings"] + [
            "方義解釋（如調和營衛）見 claims 分級，不混入原文字段。"],
    }


# ---------------------------------------------------------------------------
# 6d. 注家爭議結構化（呈現證據結構，不裁決對錯）
# ---------------------------------------------------------------------------
_DISPUTE_TYPE_CUES = {
    "訓詁": ["音", "義", "讀", "字", "謂之", "訓", "古文", "作"],
    "方證": ["湯", "方", "證", "主之", "宜"],
    "病機": ["氣", "陽", "陰", "虛", "實", "營", "衛", "榮", "樞機", "表裏", "寒熱"],
    "治法": ["汗", "下法", "吐法", "和解", "溫", "清", "補", "當", "法"],
    "劑量": ["兩", "銖", "升", "枚", "劑"],
}


def dispute_chain(ref: str) -> Dict:
    """某條文的注家爭議結構化呈現：分歧類型 · 各家觀點 · 貼近原文程度 ·
    後世發揮程度 · 不可裁決處。只呈現證據結構，不判對錯。"""
    from ..lexicon import POSTHOC_TERMS

    clauses = _clauses()
    c = _resolve_clause(ref, clauses)
    resolved_from_text = None
    if c is None:
        # 十七輪：不只認序號——文本句子經回源匹配到最佳條文再入爭議鏈
        matches = builder.get_matcher().match_text(normalize_query(ref),
                                                   limit=3)
        if matches:
            c = clauses.get(matches[0]["clause_id"])
            resolved_from_text = {
                "matched_clause_id": matches[0]["clause_id"],
                "longest_run": matches[0].get("longest_run", 0),
                "coverage": matches[0].get("coverage", 0.0),
                "alternatives": [m["clause_id"] for m in matches[1:]],
                "note": "輸入為文本句子：已回源到最相近條文（片段逐字匹配），"
                        "備選見 alternatives。"}
    if c is None:
        return {"error": f"未找到條文 {ref}（可用條文號 1-398、clause_id "
                         "或條文文本句子）"}
    cid = c["clause_id"]
    ctext_folded = fold_variants(c.get("clean_text", ""))
    schools_reg = builder.load_schools()
    member_school = schools_reg.get("commentator_school", {})
    school_names = {s["school_id"]: s["name"] for s in schools_reg.get("schools", [])}
    registry = builder.load_registry()
    dyn_of_dir = {w["book_dir"]: w["dynasty"] for w in registry["works"]}

    atlas_path = config.RESEARCH_DIR / "commentary_divergence.json"
    atlas_row = {}
    if atlas_path.exists():
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        atlas_row = next((r for r in atlas.get("clauses", [])
                          if r.get("clause_id") == cid), {})

    views = []
    seen = set()
    for r in read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl"):
        if r.get("clause_id") != cid or r.get("commentator") in seen:
            continue
        seen.add(r.get("commentator"))
        text = r.get("commentary_text", "")
        folded = fold_variants(text)
        # 貼近原文程度：注文與條文的字二元組重合率（可計算指標）
        closeness = round(similarity_pct(folded, ctext_folded), 3)
        posthoc = [t for t in POSTHOC_TERMS if fold_variants(t) in folded]
        cues = {k: [w for w in ws if w in folded]
                for k, ws in _DISPUTE_TYPE_CUES.items()}
        focus = sorted((k for k in cues if cues[k]),
                       key=lambda k: -len(cues[k]))[:2]
        sid = member_school.get(r.get("commentator", ""), "")
        views.append({
            "commentator": r.get("commentator", ""),
            "book": r.get("book", ""),
            "chapter": r.get("chapter", ""),
            "dynasty": dyn_of_dir.get(r.get("book", ""), ""),
            "school": school_names.get(sid, ""),
            "excerpt": text[:100],
            "closeness_to_original": closeness,
            "posthoc_terms": posthoc[:5],
            "posthoc_degree": round(len(posthoc) / max(1, len(folded) / 50), 3),
            "analytic_focus": focus,
        })
    views.sort(key=lambda v: (dynasty_order(v["dynasty"]), v["commentator"]))
    focus_types = sorted({f for v in views for f in v["analytic_focus"]})
    out_head: Dict = {
        "chain_type": "注家爭議結構化",
        "query": ref,
    }
    if resolved_from_text:
        out_head["resolved_from_text"] = resolved_from_text
    return {
        **out_head,
        "clause": {"clause_id": cid, "text": c.get("clean_text", "")},
        "n_commentators": len(views),
        "term_divergence": atlas_row.get("term_divergence"),
        "distinctive_terms": atlas_row.get("distinctive_terms", {}),
        "views": views,
        "divergence_types_present": focus_types,
        "undecidable_note": "各家分歧屬解釋範式差異（訓詁/方證/病機/治法取徑"
                            "不同），文本層面不可裁決對錯；「貼近原文程度」為"
                            "字面重合率，高≠正確、低≠錯誤。",
        "paper_writing_hint": "論文寫法建議：按朝代列各家觀點→報告 term_divergence"
                              " 與指紋術語→分析解釋範式差異→不下對錯結論，"
                              "以「證據結構」與「適用邊界」收束。",
        "section_evidence_levels": {
            "views": "C 注家解釋 + 可計算指標（重合率/術語密度）",
            "divergence_types_present": "E 啟發式分類（提示詞表，僅供定位）",
            "distinctive_terms": "C 層計算資產（分歧圖譜）",
        },
        "warnings": ["分歧類型為提示詞表啟發式（E 層），僅供研究定位；"
                     "多觀點並存，不做裁決。"],
    }


def similarity_pct(a: str, b: str) -> float:
    from ..textutil import similarity
    return similarity(a, b)


# ---------------------------------------------------------------------------
# 6e. 學派/注家比較（「柯琴 vs 尤怡」）
# ---------------------------------------------------------------------------
def compare_chain(ref: str) -> Dict:
    """兩注家（或兩學派）對照：範式 · 指紋術語 · 一致度 · 高分歧條文 ·
    引用網絡差異。輸入如「柯琴 vs 尤怡」「錯簡重訂 vs 以法類證」。"""
    parts = [p.strip() for p in
             re.split(r"\s*(?:vs|VS|對比|对比|×)\s*", ref) if p.strip()]
    if len(parts) != 2:
        return {"error": "輸入格式：A vs B（兩注家名或兩學派名）", "query": ref}

    schools_reg = builder.load_schools()
    member_school = schools_reg.get("commentator_school", {})
    schools = {s["school_id"]: s for s in schools_reg.get("schools", [])}

    atlas_path = config.RESEARCH_DIR / "commentary_divergence.json"
    atlas = (json.loads(atlas_path.read_text(encoding="utf-8"))
             if atlas_path.exists() else {})

    def _side(name: str) -> Dict:
        q = normalize_query(name)
        # 注家名 or 學派名
        commentator = next((c for c in sorted(member_school)
                            if q in fold_variants(c)), "")
        school = None
        if commentator:
            school = schools.get(member_school.get(commentator, ""))
        else:
            school = next((s for s in schools.values()
                           if q in fold_variants(s["name"])), None)
        fingerprints = (atlas.get("commentator_fingerprints", {})
                        .get(commentator, [])[:8] if commentator else [])
        return {"name": name, "commentator": commentator,
                "school": (school or {}).get("name", ""),
                "paradigm": (school or {}).get("paradigm", ""),
                "works": ([m for s in ([school] if school else [])
                           for m in s.get("members", [])
                           if not commentator or m["name"] == commentator]),
                "fingerprint_terms": [f.get("term", "") for f in fingerprints]}

    a, b = _side(parts[0]), _side(parts[1])
    if not (a["commentator"] or a["school"]) or not (b["commentator"] or b["school"]):
        return {"error": f"未識別比較對象：{parts}", "query": ref,
                "known_commentators": sorted(member_school)}

    agreement = next(
        (row for row in atlas.get("agreement_matrix", [])
         if {row.get("a"), row.get("b")} == {a["commentator"], b["commentator"]}),
        None) if a["commentator"] and b["commentator"] else None

    # 高分歧條文（兩家同注且術語剖面差異最大）
    top_divergent = []
    if a["commentator"] and b["commentator"]:
        for row in atlas.get("clauses", []):
            if a["commentator"] in row.get("commentators", []) \
                    and b["commentator"] in row.get("commentators", []):
                top_divergent.append({"clause_id": row["clause_id"],
                                      "term_divergence": row.get("term_divergence"),
                                      "clause_text": row.get("clause_text", "")[:40]})
        top_divergent.sort(key=lambda r: -(r["term_divergence"] or 0))
        top_divergent = top_divergent[:8]

    return {
        "chain_type": "學派/注家比較",
        "query": ref,
        "a": a, "b": b,
        "agreement": agreement,
        "top_divergent_clauses": top_divergent,
        "reading_note": "一致度與分歧條文為 C 層實測；範式/學派歸屬為"
                        " posthoc_induction；比較呈現證據結構，不裁決高下。",
        "section_evidence_levels": {
            "agreement": "C 層實測一致度（分歧圖譜）",
            "top_divergent_clauses": "C 層實測（術語剖面 Jaccard）",
            "paradigm": "posthoc_induction 學派歸納",
        },
    }


# ---------------------------------------------------------------------------
# 6c. 術語譜系鏈（某術語是否原文、最早在庫注家、學派分佈、傳播路徑）
# ---------------------------------------------------------------------------
def term_chain(term: str) -> Dict:
    """回答「營衛不和最早何時出現？」「少陽樞機不利是否原文？」類問題。"""
    matcher = builder.get_matcher()
    q = fold_variants("".join(ch for ch in normalize_query(term)
                              if "㐀" <= ch <= "鿿"))
    if len(q) < 2:
        return {"error": "術語過短"}

    # A 層逐字檢驗
    verbatim = [cid for cid in sorted(matcher.index.texts)
                if q in matcher.index.texts[cid]][:10]

    # C 層使用譜：哪些注家用過、最早何時（以在庫九注本為限）
    registry = builder.load_registry()
    dyn_of_dir = {w["book_dir"]: w["dynasty"] for w in registry["works"]}
    schools_reg = builder.load_schools()
    member_school = schools_reg.get("commentator_school", {})
    usage: Dict[str, Dict] = {}
    for r in read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl"):
        if q not in fold_variants(r.get("commentary_text", "")):
            continue
        commentator = r.get("commentator", "")
        dyn = dyn_of_dir.get(r.get("book", ""), "")
        entry = usage.setdefault(commentator, {
            "commentator": commentator, "book": r.get("book", ""),
            "dynasty": dyn, "dynasty_order": dynasty_order(dyn),
            "school_id": member_school.get(commentator, ""),
            "n_passages": 0, "clause_ids": []})
        entry["n_passages"] += 1
        if r.get("clause_id") not in entry["clause_ids"]:
            entry["clause_ids"].append(r.get("clause_id", ""))
    chronology = sorted(usage.values(),
                        key=lambda e: (e["dynasty_order"], e["commentator"]))
    for e in chronology:
        e["clause_ids"] = e["clause_ids"][:5]

    # 關聯方證觀點與相關原文表達
    related_claims = []
    for c in builder.load_claims().get("claims", []):
        if any(fold_variants(t) == q or q in fold_variants(t)
               or fold_variants(t) in q for t in c.get("interpretive_terms", [])):
            related_claims.append({
                "claim_id": c["claim_id"], "formula": c["formula"],
                "evidence_grade": c["evidence_grade"],
                "related_original_terms": c.get("terms_verbatim_in_original", {}),
                "term_first_use": c.get("term_first_use", {})})

    school_dist: Dict[str, int] = {}
    for e in chronology:
        if e["school_id"]:
            school_dist[e["school_id"]] = school_dist.get(e["school_id"], 0) + 1

    if verbatim:
        grade = "原文逐字（A 層）"
        citable = (f"「{term}」逐字見於《傷寒論》原文"
                   f"（{verbatim[0]} 等 {len(verbatim)} 條）。")
    elif chronology:
        first = chronology[0]
        grade = (f"後世術語：在庫首現 {first['commentator']}"
                 f"（{first['dynasty']}《{first['book']}》）")
        # 論文/UI 直接可引的規範句式：「最早」必須限定「在庫」，
        # 庫外文獻、未收錄注本與版本先後都可能改變「歷史首現」結論
        citable = (f"在當前收錄九種注本中，「{term}」逐字首見於"
                   f"{first['commentator']}《{first['book']}》"
                   f"（{first['dynasty']}）。")
    else:
        grade = "庫內未見（原文與九注本皆無逐字出現）"
        citable = f"「{term}」在當前語料（宋本原文與九注本）中無逐字出現。"
    modern = modern_echo_for(verbatim) if verbatim else {
        "available": False, "note": "無 A 層錨點，現代回聲不適用。"}

    return {
        "chain_type": "術語譜系鏈",
        "query": term,
        "verbatim_in_original": verbatim,
        "commentarial_chronology": chronology,
        "school_distribution": {k: school_dist[k] for k in sorted(school_dist)},
        "related_claims": related_claims,
        "evidence_grade": grade,
        "citable_statement": citable,
        "modern": modern,
        "section_evidence_levels": {
            "verbatim_in_original": "A 原文逐字檢驗",
            "commentarial_chronology": "C 注家使用譜（按朝代排序）",
            "school_distribution": "posthoc_induction 學派歸納",
            "related_claims": "方證觀點庫（分級見各條）",
        },
        "warnings": ["「最早」以在庫九注本為限，散佚注釋不可考；"
                     "術語未見於庫內不等於歷史上不存在。"],
    }


# ---------------------------------------------------------------------------
# 7. 誤引檢測（Misquotation Detection：引文是否可作原文直引）
# ---------------------------------------------------------------------------
def quote_check(text: str) -> Dict:
    """逐片段檢驗一段「引文」能否作為《傷寒論》原文直引。

    典型輸入「營衛不和，桂枝湯主之」應得到：「桂枝湯主之」逐字見於原文、
    「營衛不和」原文無此四字（屬後世方證歸納語，可回源到第 53/54 條
    「榮氣和/衛氣不和」相關表述）→ 整句不能作為原文直引。"""
    from ..lexicon import POSTHOC_TERMS
    from ..textutil import split_subclauses

    matcher = builder.get_matcher()
    folded_texts = matcher.index.texts
    claims = builder.load_claims().get("claims", [])

    def _verbatim_clauses(frag: str) -> List[str]:
        return [cid for cid in sorted(folded_texts)
                if frag in folded_texts[cid]][:5]

    fragments = []
    for frag in split_subclauses(normalize_query(text)):
        frag_folded = fold_variants("".join(
            ch for ch in frag if "㐀" <= ch <= "鿿"))
        if len(frag_folded) < 3:
            continue
        hits = _verbatim_clauses(frag_folded)
        entry: Dict = {"fragment": frag, "verbatim_in": hits,
                       "verdict": "原文逐字" if hits else "原文無此表述"}
        if not hits:
            # 後世歸納語檢驗：方證觀點術語 / 後世術語表
            related = []
            for c in claims:
                terms = [t for t in c.get("interpretive_terms",
                                          list(c.get("terms_verbatim_in_original", {})))
                         or [] if fold_variants(t) in frag_folded
                         or frag_folded in fold_variants(t)]
                # claims.json 未存 interpretive_terms 時退回命題文本匹配
                if terms or frag_folded in fold_variants(c.get("claim", "")):
                    related.append({
                        "claim_id": c["claim_id"], "formula": c["formula"],
                        "evidence_grade": c["evidence_grade"],
                        "related_original_terms":
                            c.get("terms_verbatim_in_original", {})})
            posthoc = [t for t in POSTHOC_TERMS
                       if fold_variants(t) in frag_folded]
            if related or posthoc:
                entry["verdict"] = "後世歸納語（非原文）"
                entry["posthoc_terms"] = posthoc
                entry["related_claims"] = related
        fragments.append(entry)

    n_verbatim = sum(1 for f in fragments if f["verbatim_in"])
    if not fragments:
        verdict = "輸入過短，無法檢驗"
    elif n_verbatim == len(fragments):
        verdict = "全部片段逐字見於原文，可作原文直引（附條文號）"
    elif n_verbatim == 0:
        verdict = "傷寒論原文無此表述，不能作為原文直引"
    else:
        verdict = ("混合引用：部分片段為原文逐字、部分為後世歸納語——"
                   "不能整句作為原文直引，需拆分標註")
    return {"chain_type": "誤引檢測",
            "query": text,
            "fragments": fragments,
            "verdict": verdict,
            "section_evidence_levels": {
                "fragments.verbatim_in": "A 原文逐字檢驗",
                "fragments.related_claims": "方證觀點庫（分級見各條）",
                "fragments.posthoc_terms": "後世術語表（posthoc）"},
            "warnings": ["逐字檢驗折疊異體字並剝離標點；「原文無此表述」"
                         "僅就傷寒論（含輔助篇章）而言，不排除出自他書。"]}


# ---------------------------------------------------------------------------
# 6f. 反證與爭議論證結構（十輪評審 六.5）
# ---------------------------------------------------------------------------
_CAUTION_TOKENS = ("不可", "勿", "禁", "非其治", "難治", "不中與")


def argument_chain(ref: str) -> Dict:
    """方證論證圖：不輸出一個統一結論，而是七段分層陳述——

        宋本直接可見 / 反證與慎用條文 / 版本差異導致的解釋分叉 /
        注家共同點 / 注家爭議點 / 後世經方家歸納 / 模型綜合（E 層）
        + 隱含假設（注文依賴而原文未見的病機概念）+ 尚不能裁決的問題
        + 分層置信

    各段只呈現證據結構；「不能裁決」是正式輸出而非缺陷。"""
    from ..lexicon import POSTHOC_TERMS

    q = normalize_query(ref)
    rules = read_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")
    rule = next((r for r in rules if fold_variants(r.get("formula", "")) == q),
                None) or next((r for r in rules
                               if q and q in fold_variants(r.get("formula", ""))),
                              None)
    if rule is None:
        clauses = _clauses()
        c = _resolve_clause(ref, clauses)
        if c and c.get("formula_names"):
            return argument_chain(c["formula_names"][0])
        return {"error": f"argument 以方劑為論證對象，未找到方劑 {ref}；"
                         "條文級爭議請用 dispute"}
    formula = rule["formula"]
    f_folded = fold_variants(formula)
    supporting = rule.get("supporting_clauses", [])
    canonical_ids = sorted(c for c in supporting if "AUX" not in c)
    clauses = _clauses()

    # 1. 宋本直接可見（A）
    direct = [{"clause_id": cid,
               "text": (clauses.get(cid) or {}).get("clean_text", "")[:80]}
              for cid in canonical_ids[:8]]

    # 2. 反證與慎用（A）：含方名且含禁例語氣的條文——結論的邊界證據
    contra = []
    for cid, c in clauses.items():
        text = fold_variants(c.get("clean_text", ""))
        if f_folded in text and any(t in text for t in
                                    (fold_variants(t) for t in _CAUTION_TOKENS)):
            tok = next(t for t in _CAUTION_TOKENS
                       if fold_variants(t) in text)
            contra.append({"clause_id": cid, "caution_token": tok,
                           "text": c.get("clean_text", "")[:80]})
    contra.sort(key=lambda x: x["clause_id"])

    # 3. 版本差異導致的解釋分叉（B）
    variant_forks = []
    for v in read_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl"):
        if v.get("clause_id") in set(supporting) and v.get("notable_differences"):
            variant_forks.append({
                "clause_id": v["clause_id"], "book": v.get("variant_book", ""),
                "differences": v.get("notable_differences", [])[:4]})

    # 4/5. 注家共同點與爭議點（C）：錨定前 3 條核心條文的注文，
    # 詞表 = 分歧圖譜的分析術語表（八綱/病機/榮衛…）∪ 後世病機術語表
    from ..apps.commentary_atlas import analytic_terms
    vocab = sorted(set(analytic_terms()) | set(POSTHOC_TERMS),
                   key=lambda t: (-len(t), t))
    anchor_ids = canonical_ids[:3]
    term_users: Dict[str, set] = {}
    commentators: set = set()
    for r in read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl"):
        if r.get("clause_id") not in anchor_ids:
            continue
        who = r.get("commentator", "")
        commentators.add(who)
        folded = fold_variants(r.get("commentary_text", ""))
        for t in vocab:
            if fold_variants(t) in folded:
                term_users.setdefault(t, set()).add(who)
    # 只計「解釋性詞彙」：原文自身含有的詞（太陽病/下之…）不是注家的
    # 解釋貢獻，剔除後剩下的才是各家帶入的概念框架
    anchor_texts = fold_variants("".join(
        (clauses.get(cid) or {}).get("clean_text", "") for cid in supporting))
    interpretive = {t: u for t, u in term_users.items()
                    if fold_variants(t) not in anchor_texts}
    half = max(2, len(commentators) // 2)
    common = sorted((t for t, u in interpretive.items() if len(u) >= half),
                    key=lambda t: -len(interpretive[t]))
    disputed = sorted(t for t, u in interpretive.items() if len(u) == 1)
    common_points = [{"term": t, "commentators": sorted(interpretive[t])}
                     for t in common[:8]]
    dispute_points = [{"term": t, "only_used_by": sorted(interpretive[t]),
                       "silent_commentators":
                           sorted(commentators - interpretive[t])[:6]}
                      for t in disputed[:8]]

    # 隱含假設：≥2 家共同依賴、而全部支持條文原文均未見的概念——
    # 整個解釋傳統賴以成立、卻無文本錨點的前提
    hidden = [{"term": t, "used_by": sorted(u),
               "note": "後世解釋概念——原文無此語，解釋依賴的隱含假設"}
              for t, u in sorted(interpretive.items())
              if len(u) >= 2][:8]

    # 6. 後世經方家歸納（D）
    claims = [c for c in builder.load_claims().get("claims", [])
              if c.get("formula") == formula]
    posthoc_ind = {
        "modification_relations": rule.get("modification_relations", [])[:6],
        "claims": [{"claim_id": c["claim_id"], "claim": c["claim"],
                    "evidence_grade": c["evidence_grade"]} for c in claims]}

    # 7. 模型綜合（E）——顯式標層，置信最低
    synthesis = (f"{formula}：核心證 {'、'.join(rule.get('core_pattern', '').split('+')[:4]) or rule.get('core_pattern', '')}"
                 f"；A 層支持條文 {len(canonical_ids)} 條，慎用/禁例 "
                 f"{len(contra)} 條，注家共同概念 {len(common_points)} 個，"
                 f"爭議概念 {len(dispute_points)} 個。此段為系統綜合（E 層），"
                 "僅在上列各段證據範圍內成立。")

    # 尚不能裁決的問題（正式輸出）
    undecidable = []
    if variant_forks:
        undecidable.append("版本異文是否改變醫義：庫內僅呈現差異，無裁決依據"
                           f"（{len(variant_forks)} 處）")
    if dispute_points:
        undecidable.append("注家範式分歧（訓詁/方證/病機取徑不同）文本層面"
                           "不可裁決對錯")
    if hidden:
        undecidable.append("隱含病機假設無原文錨點，成立與否超出文本證據")
    if contra:
        undecidable.append("適用邊界（禁例與主治的張力）屬臨床判斷，"
                           "系統只列證據")

    return {
        "chain_type": "方證論證結構",
        "query": ref,
        "formula": formula,
        "songben_direct": direct,
        "contradicting_or_caution_clauses": contra[:8],
        "variant_forks": variant_forks[:8],
        "commentator_common_points": common_points,
        "commentator_dispute_points": dispute_points,
        "hidden_assumptions": hidden,
        "posthoc_induction": posthoc_ind,
        "model_synthesis": synthesis,
        "undecidable": undecidable,
        "confidence_by_layer": {
            "A": {"n_evidence": len(canonical_ids) + len(contra),
                  "note": "逐字可回源（支持+反證）"},
            "B": {"n_evidence": len(variant_forks), "note": "版本對勘"},
            "C": {"n_evidence": len(commentators),
                  "note": "注家解釋（共同點置信高於爭議點）"},
            "D": {"n_evidence": len(claims) + len(posthoc_ind["modification_relations"]),
                  "note": "後世歸納（posthoc）"},
            "E": {"n_evidence": 1, "note": "模型綜合，置信最低，不得單獨引用"},
        },
        "section_evidence_levels": {
            "songben_direct": "A 原文直述",
            "contradicting_or_caution_clauses": "A 原文直述（禁例語氣詞表定位）",
            "variant_forks": "B 版本異文",
            "commentator_common_points": "C 注家解釋（跨家共現統計）",
            "commentator_dispute_points": "C 注家解釋（獨用術語）",
            "hidden_assumptions": "C→E 邊界（注文概念，原文無錨點）",
            "posthoc_induction": "D 後世歸納",
            "model_synthesis": "E 模型綜合",
        },
        "warnings": ["反證條文由禁例語氣詞表定位（確定性），語氣≠絕對禁忌，"
                     "須回源原文；共同點/爭議點為術語共現統計，非語義判定；"
                     "本鏈不輸出統一結論——多觀點並存，「不能裁決」是正式輸出。"],
    }


def trace_dispatch(query_type: str, ref: str) -> Dict:
    """統一入口（CLI / 工具 / 服務端共用）。"""
    dispatch = {"clause": clause_chain, "formula": formula_chain,
                "claim": claim_chain, "school": school_chain,
                "commentator": commentator_chain, "text": text_trace,
                "quote": quote_check, "term": term_chain,
                "dispute": dispute_chain, "compare": compare_chain,
                "argument": argument_chain}
    fn = dispatch.get(query_type)
    if fn is None:
        return {"error": f"未知溯源對象類型 {query_type}",
                "supported": sorted(dispatch)}
    return fn(ref)
