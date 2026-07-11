"""方證鑒別審計（十六輪）：規則歸類可錯，故逐格回源核驗 + 模型審校。

兩層，缺一不可：

1. **確定性核驗層**（``verify_differential``）——鑒別表由 D 層規則歸納
   生成，歸類可能有錯。本層把對比表每格的每個證候詞逐一回源到該方的
   支持條文：詞在條文實體層（否定感知抽取）出現 → verified；只在條文
   否定語境出現（如「不渴」被歸入「渴」軸）→ negated_context（疑似
   歸類錯誤）；支持條文中根本找不到 → unverified（規則層污染）。
2. **模型審校層**（``model_review``）——把對比表與支持條文全文交給
   大模型作對抗式審校（軸值錯掛/漏軸/鑒別點不成立）；模型指出的每個
   問題所引 clause_id 都經 CitationGuard 逐一核驗，未核實引用如實標記。
   無真實模型時降級為確定性審校（由核驗層結果構造，離線同構可測）。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..textutil import contains_verbatim, fold_variants

# 表中不作逐詞回源的行：非證候軸（歸屬/傾向為歸納口徑，禁忌為節錄）
_META_AXES = ("六經歸屬", "寒熱虛實傾向（後世歸納）", "禁忌")

_NEG_PREFIXES = ("不", "無", "未", "非")


def _clause_of(store, cid):
    c = store.get(cid)
    if c is None:
        return None
    return c


def _term_status(term: str, clause) -> Optional[str]:
    """單條文內判定：verified / negated_context / None（未見）。"""
    t = fold_variants(term)
    pos = {fold_variants(x) for x in (list(clause.symptoms or [])
                                      + list(clause.pulse or []))}
    neg = {fold_variants(x) for x in (clause.negated_findings or [])}
    if t in pos:
        return "verified"
    if t in neg or any(n.endswith(t) and n[: -len(t)] in _NEG_PREFIXES
                       for n in neg):
        return "negated_context"
    text = fold_variants(clause.clean_text or "")
    if t and t in text:
        # 逐字在文，但要排除被否定前綴貼身包裹的假陽性（「不惡寒」含「惡寒」）
        i = text.find(t)
        while i >= 0:
            if not (i > 0 and text[i - 1] in "不無未非勿"):
                return "verified"
            i = text.find(t, i + 1)
        return "negated_context"
    return None


def _verify_term(term: str, clause_ids: List[str], store) -> Dict:
    """詞級核驗：聚合該方全部支持條文的判定。"""
    verified_in, negated_in = [], []
    for cid in clause_ids:
        c = _clause_of(store, cid)
        if c is None:
            continue
        st = _term_status(term, c)
        if st == "verified":
            verified_in.append(cid)
        elif st == "negated_context":
            negated_in.append(cid)
    if verified_in:
        return {"status": "verified", "clauses": verified_in[:4]}
    if negated_in:
        return {"status": "negated_context", "clauses": negated_in[:4]}
    return {"status": "unverified", "clauses": []}


def _supporting_of(formula: str, formula_rules) -> List[str]:
    r = next((r for r in formula_rules if r.formula == formula), None)
    return list(r.supporting_clauses) if r else []


def verify_differential(diff: Dict, formula_rules, clause_store) -> Dict:
    """對比表逐格逐詞回源核驗。返回逐項結果與問題清單。"""
    formulas = diff.get("formulas", [])
    support = {f: _supporting_of(f, formula_rules) for f in formulas}
    checks: List[Dict] = []
    for row in diff.get("contrast_table", []):
        axis = row.get("axis", "")
        if axis in _META_AXES:
            continue
        for f in formulas:
            val = row.get(f, "—")
            if not val or val == "—":
                continue
            for term in [t for t in val.split("、") if t]:
                res = _verify_term(term, support[f], clause_store)
                checks.append({"axis": axis, "formula": f, "term": term,
                               **res})
    # 關鍵鑒別點（「方名：a、b」格式）同樣核驗
    for line in diff.get("key_discriminators", []) or []:
        if "：" not in line:
            continue
        f, _, terms = line.partition("：")
        if f not in support:
            continue
        for term in [t for t in terms.split("、") if t]:
            res = _verify_term(term, support[f], clause_store)
            checks.append({"axis": "關鍵鑒別點", "formula": f, "term": term,
                           **res})
    flagged = [c for c in checks if c["status"] != "verified"]
    return {
        "n_checked": len(checks),
        "n_verified": len(checks) - len(flagged),
        "flagged": flagged,
        "note": "逐格逐詞回源：verified=支持條文實體層可證；"
                "negated_context=僅見於否定語境（疑似規則歸類錯誤）；"
                "unverified=支持條文中無此表述（D 層歸納污染，勿作原文陳述）。",
    }


# ---------------------------------------------------------------------------
# 模型審校層
# ---------------------------------------------------------------------------
def _evidence_block(formulas: List[str], support: Dict[str, List[str]],
                    store, max_per_formula: int = 6,
                    max_chars: int = 260) -> str:
    rows, seen = [], set()
    for f in formulas:
        rows.append(f"◆ {f} 支持條文：")
        for cid in support[f][:max_per_formula]:
            c = store.get(cid)
            if c is None or cid in seen:
                continue
            seen.add(cid)
            rows.append(f"- [{cid}] {c.clean_text[:max_chars]}")
    return "\n".join(rows)


def model_review(diff: Dict, formula_rules, clause_store, llm,
                 verification: Optional[Dict] = None) -> Dict:
    """對抗式審校鑒別表。真模型：結構化審校 + 引用核驗；
    local：由確定性核驗結果構造同構審校（離線可測）。"""
    from ..agent.citation_guard import CitationGuard
    from ..llm.prompts import diff_review_system_prompt, diff_review_user_prompt

    formulas = diff.get("formulas", [])
    support = {f: _supporting_of(f, formula_rules) for f in formulas}
    allowed = sorted({cid for ids in support.values() for cid in ids})
    verification = verification or verify_differential(diff, formula_rules,
                                                       clause_store)

    if not getattr(llm, "available", False):
        issues = [{"formula": c["formula"], "axis": c["axis"],
                   "problem": (f"「{c['term']}」"
                               + ("僅見於支持條文的否定語境，疑似歸類反了"
                                  if c["status"] == "negated_context"
                                  else "在支持條文中未找到，屬規則層歸納，"
                                       "不可當原文陳述")),
                   "clause_ids": c["clauses"], "source": "deterministic"}
                  for c in verification["flagged"]]
        return {"backend": "local",
                "verdict": "warn" if issues else "pass",
                "issues": issues,
                "summary": (f"確定性審校：{verification['n_checked']} 項核驗，"
                            f"{len(issues)} 項存疑（未接真實模型；"
                            "接入後將由大模型作語義級對抗審校）。"),
                "citation_report": None}

    table = json.dumps({"formulas": formulas,
                        "contrast_table": diff.get("contrast_table", []),
                        "key_discriminators":
                            diff.get("key_discriminators", [])},
                       ensure_ascii=False, indent=1)
    out = llm.json_complete(
        diff_review_system_prompt(),
        diff_review_user_prompt(table,
                                _evidence_block(formulas, support,
                                                clause_store)),
        task="critic",
        context={"formulas": formulas})
    issues = []
    guard = CitationGuard(clause_store)
    for it in (out.get("issues") or [])[:12]:
        if not isinstance(it, dict):
            continue
        cids = [c for c in (it.get("clause_ids") or []) if isinstance(c, str)]
        rep = guard.check("、".join(cids), allowed_ids=allowed)
        issues.append({"formula": str(it.get("formula", ""))[:24],
                       "axis": str(it.get("axis", ""))[:24],
                       "problem": str(it.get("problem", ""))[:200],
                       "clause_ids": rep.verified_ids,
                       "unverified_clause_ids": (rep.unsupported_ids
                                                 + rep.outside_evidence_ids),
                       "source": "model"})
    summary = str(out.get("summary", ""))[:600]
    srep = guard.check(summary, allowed_ids=allowed)
    verdict = out.get("verdict", "")
    if verdict not in ("pass", "warn", "fail"):
        verdict = "warn" if issues else "pass"
    return {"backend": getattr(llm, "backend", "litellm"),
            "verdict": verdict,
            "issues": issues,
            "summary": summary,
            "citation_report": srep.to_dict(),
            "note": "模型審校屬 E 層；所引 clause_id 已逐一核驗，"
                    "unverified_clause_ids 中的編號請勿採信。"}
