"""確定性科學計量分析（引文網絡 / 共引 / 文獻耦合 / 時間切片 / 突現 / 主路徑）。

scope 一致性契約（評審方案 A）：scope=canonical/auxiliary 時，報告中
**所有** scope 敏感字段（被引榜/時間切片焦點/突現/主路徑/共引）只含
該域條文；`audit_scope_consistency` 對任意輸出做遞歸審計，違例即報。

分析對象是中醫古籍知識單元（條文、方劑、注文）在歷代文獻中的引用與
傳播，全部從引文邊確定性推導，純標準庫、無隨機性：

| 方法     | 實現 |
|----------|------|
| 引文網絡 | 著作→條文邊的度分佈：被引最多的條文、引用最廣的著作 |
| 共引分析 | 同一段落共同引用的條文對（條文互參的文獻學證據） |
| 文獻耦合 | 共享被引條文集的著作對（Jaccard，解釋群的雛形） |
| 時間切片 | 按朝代切片的引用強度與焦點條文 |
| 突現分析 | 條文在某朝代的引用份額相對全期基線的提升（lift） |
| 主路徑   | 某條文從原典到近代的傳播鏈：逐朝代取最強引用著作 |

「現代」切片誠實聲明：語料最晚一層為民國（1937《經方實驗錄》）；
現代論文/教材引用經 modern.py 接口導入後參與同一網絡，不隨庫捏造。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from .ids import dynasty_order

MIN_BURST_EDGES = 8      # 突現分析的條文最小被引邊數
TOP_N = 20

RE_ANY_CLAUSE = re.compile(r"SHL_SONGBEN_(AUX_)?\d{4}")


def audit_scope_consistency(payload: Dict, scope: str) -> Dict:
    """遞歸審計一份計量輸出是否符合 scope 契約（A1 Scope Auditor）。

    canonical 輸出中出現任何 AUX 條文號、auxiliary 輸出中出現任何正文
    條文號，均計違例。對整個 JSON 序列化文本掃描，杜絕漏檢字段。"""
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    n_aux = n_canon = 0
    for m in RE_ANY_CLAUSE.finditer(blob):
        if m.group(1):
            n_aux += 1
        else:
            n_canon += 1
    violations = 0
    if scope == "canonical":
        violations = n_aux
    elif scope == "auxiliary":
        violations = n_canon
    return {"scope": scope, "n_canonical_ids": n_canon, "n_aux_ids": n_aux,
            "violations": violations, "ok": violations == 0}


def aggregate_edges(edges: List[Dict]) -> List[Dict]:
    """把段落級引文邊聚合為 (著作, 條文) 級行（提交用緊湊資產）。"""
    agg: Dict[Tuple[str, str], Dict] = {}
    for e in edges:
        if e.get("target_kind") != "clause" or not e.get("clause_id"):
            continue
        key = (e["book_dir"], e["clause_id"])
        row = agg.setdefault(key, {
            "book_dir": e["book_dir"], "book": e["book"],
            "author": e["author"], "dynasty": e["dynasty"],
            "layer": e["layer"], "clause_id": e["clause_id"],
            "n_paragraphs": 0, "modes": {}, "max_coverage": 0.0,
            "max_run": 0, "first_chapter": e.get("chapter", ""),
        })
        row["n_paragraphs"] += 1
        row["modes"][e["mode"]] = row["modes"].get(e["mode"], 0) + 1
        row["max_coverage"] = max(row["max_coverage"], e.get("coverage", 0.0))
        row["max_run"] = max(row["max_run"], e.get("longest_run", 0))
    out = [agg[k] for k in sorted(agg)]
    for row in out:
        row["modes"] = {m: row["modes"][m] for m in sorted(row["modes"])}
    return out


def aggregate_relay(edges: List[Dict]) -> List[Dict]:
    """轉引注文邊聚合為 (著作, 經由注本) 級行——注本的樞紐作用計量。"""
    agg: Dict[Tuple[str, str], Dict] = {}
    for e in edges:
        if e.get("target_kind") != "commentary":
            continue
        key = (e["book_dir"], e["via_book"])
        row = agg.setdefault(key, {
            "book_dir": e["book_dir"], "book": e["book"], "dynasty": e["dynasty"],
            "via_book": e["via_book"], "via_commentator": e["via_commentator"],
            "n_paragraphs": 0, "max_run": 0,
        })
        row["n_paragraphs"] += 1
        row["max_run"] = max(row["max_run"], e.get("longest_run", 0))
    return [agg[k] for k in sorted(agg)]


def build_network(edges: List[Dict], book_stats: List[Dict]) -> Dict:
    """從段落級引文邊構建雙層計量網絡摘要（全確定）。"""
    clause_edges = [e for e in edges if e.get("target_kind") == "clause"]

    # ---- 引文網絡：條文被引度 -------------------------------------------
    clause_stats: Dict[str, Dict] = {}
    for e in clause_edges:
        s = clause_stats.setdefault(e["clause_id"], {
            "clause_id": e["clause_id"], "n_edges": 0, "books": set(),
            "modes": {}, "dynasties": {}})
        s["n_edges"] += 1
        s["books"].add(e["book_dir"])
        s["modes"][e["mode"]] = s["modes"].get(e["mode"], 0) + 1
        dyn = e["dynasty"] or "未詳"
        s["dynasties"][dyn] = s["dynasties"].get(dyn, 0) + 1
    def _expand(s: Dict) -> Dict:
        return {**s, "n_books": len(s["books"]), "books": sorted(s["books"]),
                "modes": {m: s["modes"][m] for m in sorted(s["modes"])},
                "dynasties": {d: s["dynasties"][d]
                              for d in sorted(s["dynasties"])}}

    ranked = sorted(clause_stats.values(),
                    key=lambda s: (-s["n_edges"], s["clause_id"]))
    # 正文/輔助篇章分榜：輔助篇章（辨脈法/傷寒例…）篇幅長、被綱目/輯義類
    # 著作整段徵引，混排會壓過正文 398 條的學術中心性——產品展示必須分開。
    top_clauses = [_expand(s) for s in ranked[:TOP_N]]
    top_canonical = [_expand(s) for s in ranked
                     if "AUX" not in s["clause_id"]][:TOP_N]
    top_auxiliary = [_expand(s) for s in ranked
                     if "AUX" in s["clause_id"]][:TOP_N]

    # ---- 著作引用廣度 ----------------------------------------------------
    work_stats: Dict[str, Dict] = {}
    for e in clause_edges:
        w = work_stats.setdefault(e["book_dir"], {
            "book_dir": e["book_dir"], "book": e["book"], "author": e["author"],
            "dynasty": e["dynasty"], "layer": e["layer"],
            "n_edges": 0, "clauses": set()})
        w["n_edges"] += 1
        w["clauses"].add(e["clause_id"])
    works_ranked = sorted(work_stats.values(),
                          key=lambda w: (-len(w["clauses"]), w["book_dir"]))
    works_out = [{**w, "n_clauses_cited": len(w["clauses"]),
                  "clauses": None} for w in works_ranked]
    for w in works_out:
        w.pop("clauses")

    # ---- scope 謂詞：正文 / 輔助篇章 / 混排 -------------------------------
    # 評審共識（方案 A）：scope 必須貫穿全部 scope 敏感字段，
    # 不能只過濾被引榜而讓 time_slices/bursts/main_paths/共引混入另一域。
    scope_pred = {
        "canonical": lambda cid: "AUX" not in cid,
        "auxiliary": lambda cid: "AUX" in cid,
        "all": lambda cid: True,
    }

    # ---- 共引：同段落共同引用的條文對（逐 scope，兩端須同域） -------------
    para_clauses: Dict[Tuple[str, int], set] = {}
    for e in clause_edges:
        para_clauses.setdefault((e["book_dir"], e["para_seq"]), set()).add(e["clause_id"])
    cocitation_scoped: Dict[str, List[Dict]] = {}
    for scope, pred in scope_pred.items():
        cocite: Dict[Tuple[str, str], int] = {}
        for cids in para_clauses.values():
            cl = sorted(c for c in cids if pred(c))
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    cocite[(cl[i], cl[j])] = cocite.get((cl[i], cl[j]), 0) + 1
        cocitation_scoped[scope] = [
            {"a": a, "b": b, "n": n}
            for (a, b), n in sorted(cocite.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:50]]
    cocitation = cocitation_scoped["all"]

    # ---- 文獻耦合：共享被引條文集的著作對（逐 scope：條文集先按域過濾） ----
    coupling_scoped: Dict[str, List[Dict]] = {}
    for scope, pred in scope_pred.items():
        rows = []
        scoped_sets = [(w["book_dir"], {c for c in w["clauses"] if pred(c)})
                       for w in works_ranked]
        for i in range(len(scoped_sets)):
            for j in range(i + 1, len(scoped_sets)):
                (da, sa), (db, sb) = scoped_sets[i], scoped_sets[j]
                inter = len(sa & sb)
                union = len(sa | sb)
                if inter >= 10 and union:
                    rows.append({"a": da, "b": db, "shared_clauses": inter,
                                 "jaccard": round(inter / union, 3)})
        rows.sort(key=lambda r: (-r["jaccard"], r["a"], r["b"]))
        coupling_scoped[scope] = rows[:50]
    coupling = coupling_scoped["all"]

    # ---- 時間切片 --------------------------------------------------------
    slices: Dict[str, Dict] = {}
    for e in clause_edges:
        dyn = e["dynasty"] or "未詳"
        s = slices.setdefault(dyn, {"dynasty": dyn,
                                    "dynasty_order": dynasty_order(dyn),
                                    "n_edges": 0, "books": set(), "focus": {}})
        s["n_edges"] += 1
        s["books"].add(e["book_dir"])
        s["focus"][e["clause_id"]] = s["focus"].get(e["clause_id"], 0) + 1
    time_slices = []
    for dyn in sorted(slices, key=lambda d: (slices[d]["dynasty_order"], d)):
        s = slices[dyn]
        row = {"dynasty": dyn, "n_edges": s["n_edges"],
               "n_works": len(s["books"])}
        for scope, pred in scope_pred.items():
            focus = sorted(((c, n) for c, n in s["focus"].items() if pred(c)),
                           key=lambda kv: (-kv[1], kv[0]))[:5]
            key = {"canonical": "top_canonical", "auxiliary": "top_auxiliary",
                   "all": "top_clauses"}[scope]
            row[key] = [{"clause_id": c, "n": n} for c, n in focus]
            row["n_edges_" + scope] = sum(n for c, n in s["focus"].items()
                                          if pred(c))
        time_slices.append(row)

    # ---- 突現分析：條文在某朝代的引用份額 lift（逐 scope 排榜） -----------
    total_edges = len(clause_edges) or 1
    dyn_totals = {s["dynasty"]: s["n_edges"] for s in time_slices}
    bursts_scoped: Dict[str, List[Dict]] = {}
    for scope, pred in scope_pred.items():
        rows = []
        for cid in sorted(clause_stats):
            if not pred(cid):
                continue
            s = clause_stats[cid]
            if s["n_edges"] < MIN_BURST_EDGES:
                continue
            base_share = s["n_edges"] / total_edges
            for dyn, n in sorted(s["dynasties"].items()):
                dyn_total = dyn_totals.get(dyn, 0)
                if dyn_total < 20:
                    continue
                share = n / dyn_total
                lift = share / base_share if base_share else 0.0
                if lift >= 2.0:
                    rows.append({"clause_id": cid, "dynasty": dyn, "n": n,
                                 "lift": round(lift, 2)})
        rows.sort(key=lambda b: (-b["lift"], b["clause_id"], b["dynasty"]))
        bursts_scoped[scope] = rows[:30]
    bursts = bursts_scoped["all"]

    # ---- 主路徑：各 scope 被引最多條文的跨朝代傳播鏈 ----------------------
    main_paths_scoped = {
        "canonical": [main_path_for(c["clause_id"], clause_edges)
                      for c in top_canonical[:10]],
        "auxiliary": [main_path_for(c["clause_id"], clause_edges)
                      for c in top_auxiliary[:10]],
        "all": [main_path_for(c["clause_id"], clause_edges)
                for c in top_clauses[:10]],
    }
    main_paths = main_paths_scoped["canonical"]

    mode_counts: Dict[str, int] = {}
    for e in edges:
        mode_counts[e["mode"]] = mode_counts.get(e["mode"], 0) + 1

    return {
        "note": "全部指標由引文邊確定性推導；「未詳」朝代為 vendor 元數據缺失且"
                "無書志補注者。現代切片需經 modern 接口導入，不隨庫捏造。",
        "overview": {
            "n_edges_total": len(edges),
            "n_clause_edges": len(clause_edges),
            "n_relay_edges": len(edges) - len(clause_edges),
            "n_clauses_cited": len(clause_stats),
            "n_citing_works": len(work_stats),
            "mode_distribution": {m: mode_counts[m] for m in sorted(mode_counts)},
            "n_marker_unresolved": sum(b.get("n_marker_unresolved", 0)
                                       for b in book_stats),
        },
        "top_cited_clauses": top_clauses,
        "top_cited_canonical": top_canonical,
        "top_cited_auxiliary": top_auxiliary,
        "ranking_note": "混排榜中輔助篇章（AUX，辨脈法/傷寒例等）因篇幅長、"
                        "被整段徵引而居前，不代表其學術中心性高於正文條文；"
                        "展示時應按 scope（canonical/auxiliary/all）分榜。",
        "citing_works": works_out,
        "cocitation_pairs": cocitation,
        "bibliographic_coupling": coupling,
        "time_slices": time_slices,
        "bursts": bursts,
        "main_paths": main_paths,
        # scope 一致視圖：canonical/auxiliary/all 各自的共引/耦合/突現/主路徑
        # （頂層同名字段為向後兼容視圖：cocitation/coupling/bursts=all，
        #   main_paths=canonical）
        "scoped": {scope: {"cocitation_pairs": cocitation_scoped[scope],
                           "bibliographic_coupling": coupling_scoped[scope],
                           "bursts": bursts_scoped[scope],
                           "main_paths": main_paths_scoped[scope]}
                   for scope in ("canonical", "auxiliary", "all")},
    }


def main_path_for(clause_id: str, clause_edges: List[Dict]) -> Dict:
    """某條文的主路徑：逐朝代取最強引用（覆蓋率、片段長度）著作成鏈。"""
    per_dyn: Dict[str, Dict] = {}
    for e in clause_edges:
        if e.get("clause_id") != clause_id or e.get("target_kind") != "clause":
            continue
        dyn = e["dynasty"] or "未詳"
        cur = per_dyn.get(dyn)
        key = (e.get("coverage", 0.0), e.get("longest_run", 0))
        if cur is None or key > (cur.get("coverage", 0.0), cur.get("longest_run", 0)):
            per_dyn[dyn] = {"dynasty": dyn, "dynasty_order": dynasty_order(dyn),
                            "book": e["book"], "book_dir": e["book_dir"],
                            "author": e["author"], "mode": e["mode"],
                            "coverage": e.get("coverage", 0.0),
                            "longest_run": e.get("longest_run", 0)}
    chain = sorted(per_dyn.values(), key=lambda r: (r["dynasty_order"], r["book_dir"]))
    return {"clause_id": clause_id,
            "path": [{"dynasty": "東漢", "book": "傷寒論", "author": "張仲景",
                      "mode": "原典", "coverage": 1.0, "longest_run": 0}] + chain}
