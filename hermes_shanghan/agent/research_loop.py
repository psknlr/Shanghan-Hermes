"""DeepResearcher — loop-engineered autonomous scholarship（學術溯源引擎）.

Plan → dispatch subagents → critique coverage → iterate until converged:

  Planner    picks which research MODULES to call next round (a real model
             chooses via JSON planning over the module catalog — 語言模型
             自動選擇調用模塊; the local backend plans deterministically by
             coverage, so the loop runs offline through the same code path)
  Subagents  one per task: execute the module via the read-only ToolRegistry,
             then write a short evidence-cited finding (LLM prose when
             available, deterministic formatting otherwise — the same
             trusted-base/augmentation split as everywhere else)
  Critic     checks the six provenance dimensions (原文源流/異文注家/方證
             計量/劑量計量/客觀評測/醫案例證) for gaps; uncovered dimensions become
             next round's plan; convergence = full coverage or max_rounds
  Ledger     every finding passes the CitationGuard; the dossier carries
             verified clause_ids per finding — 無證據鏈，不成回答 holds for
             machine scholarship too

The dossier feeds the `provenance` paper type: a 溯源論文 whose every
section is a round-stamped, citation-verified finding.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .. import lexicon
from ..llm.client import LLMClient, get_client
from ..llm.prompts import EVIDENCE_CONTRACT
from ..textutil import normalize_query
from .citation_guard import CitationGuard
from .tools import ToolRegistry, get_registry

# module catalog: name → (provenance dimension, planner-facing description)
MODULES = {
    "shanghan_search": ("原文源流", "檢索相關條文（帶 clause_id）"),
    "shanghan_formula_rule": ("原文源流", "取某方的方證規則與支持條文"),
    "shanghan_six_channel": ("原文源流", "取某經提綱/亞型/主方"),
    "shanghan_differential": ("方證計量", "2-3 方多軸鑒別對比"),
    "shanghan_corpus_stats": ("方證計量", "全庫頻次/分級/六經分佈統計"),
    "shanghan_mistreatment": ("方證計量", "誤治→變證→救治路徑"),
    "shanghan_divergence_atlas": ("異文注家", "9 注本對齊/分歧榜/一致度矩陣"),
    "shanghan_dose": ("劑量計量", "藥量比/三家折算/家族劑量演化"),
    "shanghan_eval_metrics": ("客觀評測", "遮方/醫案回放/接地率基準指標"),
    "shanghan_variants": ("異文注家", "某條文的桂本/千金翼異文對勘"),
    "shanghan_relations": ("原文源流", "條文關係圖譜鄰接邊"),
    "shanghan_therapy": ("方證計量", "治法法度：適應/禁例/誤施"),
    "shanghan_case_search": ("醫案例證", "經方實驗錄真實診案（旁證+經文錨點）"),
    "shanghan_trace": ("引文傳播", "深度溯源鏈：條文/方劑/方證觀點/注家/學派"),
    "shanghan_citation_network": ("引文傳播", "歷代引文網絡/共引/時間切片/主路徑"),
}
DIMENSIONS = ["原文源流", "異文注家", "方證計量", "劑量計量", "客觀評測",
              "醫案例證", "引文傳播"]

# 聚合統計類模塊：合法產出是數字/比例/矩陣而非條文引用（藥量比、頻次、
# 一致度、計量網絡）——覆蓋要求是「工具成功 + 有數據」（DATA_FOUND），
# 不強求 clause_id，也不冒充 VERIFIED
AGGREGATE_MODULES = {"shanghan_corpus_stats", "shanghan_eval_metrics",
                     "shanghan_divergence_atlas", "shanghan_citation_network",
                     "shanghan_dose"}

# 發現覆蓋狀態（九輪 P1：「調用過」≠「覆蓋完成」）：
#   FAILED          工具返回錯誤——不計入覆蓋
#   EMPTY           無錯誤但無條文證據（非聚合模塊）——不計入覆蓋
#   DATA_FOUND      聚合模塊有數據（無條文引用要求）
#   EVIDENCE_FOUND  有本模塊自產的已核實 clause_id
#   VERIFIED        引用核驗全通過且有引用
FINDING_STATUSES = ("FAILED", "EMPTY", "DATA_FOUND", "EVIDENCE_FOUND",
                    "VERIFIED")
_COVERING = {"DATA_FOUND", "EVIDENCE_FOUND", "VERIFIED"}

# actionable follow-ups per uncovered dimension — a research gap is only
# useful if it says HOW to close it
GAP_SUGGESTIONS = {
    "原文源流": "調用 shanghan_search / shanghan_formula_rule 補充 A 層條文取證",
    "異文注家": "調用 shanghan_variants 對勘桂林古本/千金翼方，或 shanghan_divergence_atlas 取注家分歧",
    "方證計量": "調用 shanghan_corpus_stats / shanghan_differential 補計量與鑒別",
    "劑量計量": "調用 shanghan_dose（藥量比/家族演化）或 shanghan_dose_convert",
    "客觀評測": "先運行 evaluate 生成基準，再調 shanghan_eval_metrics",
    "醫案例證": "調用 shanghan_case_search（經方實驗錄旁證 + 經文錨點）",
    "引文傳播": "調用 shanghan_trace（溯源鏈）或 shanghan_citation_network（歷代引文計量）",
}


def refine_questions(topic: str, formulas: List[str]) -> List[str]:
    """Research-question refiner: expand a bare topic（「桂枝湯類方源流」）
    into the concrete questions the loop should answer — deterministic so
    the dossier is reproducible."""
    f0 = formulas[0] if formulas else ""
    subject = f0 or topic
    return [
        f"{topic}在宋本中的原文基礎（相關條文與方證規則）是什麼？",
        f"{subject}的加減/類方變化體現哪些證候邊界？",
        f"{subject}的藥量比與家族劑量演化呈何種模式？",
        f"注家對{topic}核心條文的詮釋有哪些分歧？版本異文是否影響釋讀？",
        f"{topic}相關的誤治傳變與禁例法度有哪些？",
        f"現有規則庫的計量統計與客觀評測基準對{topic}的結論有何約束？",
    ]


class DeepResearcher:
    def __init__(self, client: Optional[LLMClient] = None,
                 registry: Optional[ToolRegistry] = None, max_rounds: int = 3):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        self.max_rounds = max_rounds

    # ------------------------------------------------------------------
    def run(self, topic: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {"topic": topic, "findings": [],
                                 "called": set(), "rounds": []}
        formulas = [n for n in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True)
                    if n in normalize_query(topic)][:2]
        for rnd in range(1, self.max_rounds + 1):
            tasks = self._plan(topic, formulas, state)
            if not tasks:
                break
            round_log = {"round": rnd, "tasks": []}
            for t in tasks:
                key = (t["module"], json.dumps(t.get("args", {}),
                                               ensure_ascii=False, sort_keys=True))
                if key in state["called"]:
                    continue
                state["called"].add(key)
                finding = self._subagent(topic, t)
                state["findings"].append(finding)
                round_log["tasks"].append({"module": t["module"],
                                           "args": t.get("args", {}),
                                           "reason": t.get("reason", ""),
                                           "dimension": finding["dimension"]})
            state["rounds"].append(round_log)
            if not self._gaps(state):
                break
        guard = CitationGuard(self.registry.art.clause_store())
        all_ids: List[str] = []
        for f in state["findings"]:
            # strict round grounding: a finding may only cite clause_ids that
            # appeared in its OWN module result
            own = f.pop("_result_ids", None)
            rep = guard.check(f["summary"], allowed_ids=own)
            f["verified_clause_ids"] = rep.verified_ids
            # citation_ok = 核驗通過 **且** 確有引用（無引用不再默認 ok）；
            # 聚合統計類發現如實標 aggregate（引用要求不同，不冒充）
            f["has_citation"] = rep.has_any_citation
            f["citation_ok"] = rep.ok and rep.has_any_citation
            f["status"] = self._finding_status(f, rep)
            all_ids += rep.verified_ids
        status_by_dim = {d: self._dimension_status(state["findings"], d)
                         for d in DIMENSIONS}
        # coverage 保持計數口徑（只計「真覆蓋」的發現：FAILED/EMPTY 不算）
        coverage = {d: sum(1 for f in state["findings"]
                           if f["dimension"] == d
                           and f.get("status") in _COVERING)
                    for d in DIMENSIONS}
        gaps = self._gaps(state)
        return {"topic": topic, "backend": self.client.backend,
                "research_questions": refine_questions(topic, formulas),
                "n_rounds": len(state["rounds"]), "rounds": state["rounds"],
                "coverage": coverage,
                "coverage_status": status_by_dim,
                "coverage_note": "覆蓋=工具成功且有證據（FAILED/EMPTY 不計）；"
                                 "聚合統計模塊為 DATA_FOUND（無條文引用要求，"
                                 "如實分層不冒充 VERIFIED）",
                "uncovered_dimensions": gaps,
                "gap_report": [{"dimension": d,
                                "suggestion": GAP_SUGGESTIONS.get(d, "")}
                               for d in gaps],
                "evidence_clause_ids": sorted(set(all_ids)),
                "findings": state["findings"]}

    # ------------------------------------------------------------------
    @staticmethod
    def _finding_status(f: Dict, rep=None) -> str:
        if f.get("error"):
            return "FAILED"
        if rep is not None and rep.ok and rep.has_any_citation:
            return "VERIFIED"
        # 循環中期 verified_clause_ids 尚未計算，用本模塊自產 id 作證據代理
        if f.get("verified_clause_ids") or f.get("_result_ids"):
            return "EVIDENCE_FOUND"
        if f.get("module") in AGGREGATE_MODULES:
            return "DATA_FOUND"
        return "EMPTY"

    @staticmethod
    def _dimension_status(findings, dim: str) -> Dict:
        rows = [f for f in findings if f["dimension"] == dim]
        order = {s: i for i, s in enumerate(FINDING_STATUSES)}
        best = max((f.get("status", "EMPTY") for f in rows),
                   key=lambda s: order.get(s, 0), default="NOT_STARTED")
        return {"n_findings": len(rows), "status": best}

    def _gaps(self, state) -> List[str]:
        """未覆蓋維度：只有「工具成功且有證據/數據」的發現才算覆蓋——
        工具報錯或空手而歸的維度仍是缺口（九輪 P1：調用過≠覆蓋）。"""
        covered = set()
        for f in state["findings"]:
            status = f.get("status") or self._finding_status(f)
            if status in _COVERING:
                covered.add(f["dimension"])
        return [d for d in DIMENSIONS if d not in covered]

    def _plan(self, topic: str, formulas: List[str], state) -> List[Dict]:
        """LLM plans module calls; local backend plans by coverage gaps."""
        if self.client.available:
            catalog = "\n".join(f"- {m}（維度：{dim}）：{desc}"
                                for m, (dim, desc) in MODULES.items())
            done = "\n".join(f"- 已調 {t['module']}({json.dumps(t['args'], ensure_ascii=False)})"
                             for r in state["rounds"] for t in r["tasks"]) or "（尚未調用）"
            try:
                plan = self.client.json_complete(
                    EVIDENCE_CONTRACT + "\n\n任務：為《傷寒論》學術溯源研究規劃下一輪"
                    "模塊調用。六個溯源維度都應覆蓋；不要重複已調用的組合。"
                    "嚴格輸出 JSON：{\"tasks\":[{\"module\":\"…\",\"args\":{…},"
                    "\"reason\":\"…\"}]}，tasks 為空表示研究已完備。",
                    f"研究主題：{topic}\n可用模塊：\n{catalog}\n\n已完成調用：\n{done}\n"
                    f"未覆蓋維度：{self._gaps(state) or '（已全覆蓋）'}",
                    task="synthesize")
                tasks = [t for t in plan.get("tasks", [])
                         if t.get("module") in MODULES][:5]
                if tasks or state["rounds"]:
                    return tasks
            except Exception:
                pass
        return self._plan_local(topic, formulas, state)

    def _plan_local(self, topic: str, formulas: List[str], state) -> List[Dict]:
        gaps = self._gaps(state)
        tasks: List[Dict] = []
        f0 = formulas[0] if formulas else ""
        if "原文源流" in gaps:
            tasks.append({"module": "shanghan_search",
                          "args": {"query": topic, "top_k": 6, "expand": True},
                          "reason": "取主題相關條文作 A 層源流"})
            if f0:
                tasks.append({"module": "shanghan_formula_rule",
                              "args": {"formula": f0}, "reason": "主題方證規則"})
        if "異文注家" in gaps:
            tasks.append({"module": "shanghan_divergence_atlas", "args": {},
                          "reason": "注家詮釋史與分歧"})
        if "方證計量" in gaps:
            tasks.append({"module": "shanghan_corpus_stats", "args": {},
                          "reason": "全庫計量背景"})
        if "劑量計量" in gaps:
            tasks.append({"module": "shanghan_dose",
                          "args": {"formula": f0} if f0 else {},
                          "reason": "劑量比與家族演化"})
        if "客觀評測" in gaps:
            tasks.append({"module": "shanghan_eval_metrics", "args": {},
                          "reason": "方法可信度基準"})
        if "醫案例證" in gaps:
            tasks.append({"module": "shanghan_case_search",
                          "args": {"formula": f0} if f0 else {},
                          "reason": "歷史醫案旁證"})
        if "引文傳播" in gaps:
            tasks.append({"module": "shanghan_trace",
                          "args": ({"query_type": "formula", "ref": f0} if f0
                                   else {"query_type": "text", "ref": topic}),
                          "reason": "歷代引用與傳播路徑"})
        return tasks[:7]

    # ------------------------------------------------------------------
    def _subagent(self, topic: str, task: Dict) -> Dict:
        module = task["module"]
        args = task.get("args", {}) or {}
        result = self.registry.call(module, args)
        dimension = MODULES[module][0]
        summary = self._summarize(topic, module, result)
        from .citation_guard import RE_CLAUSE_ID
        own_ids = list(dict.fromkeys(RE_CLAUSE_ID.findall(
            json.dumps(result, ensure_ascii=False, default=str))))
        return {"dimension": dimension, "module": module, "args": args,
                "summary": summary,
                "_result_ids": own_ids,
                "error": result.get("error") if isinstance(result, dict) else None}

    def _summarize(self, topic: str, module: str, result: Dict) -> str:
        if isinstance(result, dict) and result.get("error"):
            return f"（{module} 無數據：{result['error']}）"
        if self.client.available:
            try:
                text = self.client.complete(
                    EVIDENCE_CONTRACT + "\n\n任務：作為溯源研究子代理，把下方工具"
                    "結果凝練成 2-4 句研究發現。只可使用結果中的事實；引用條文附 "
                    "clause_id（僅可取自結果）。",
                    f"研究主題：{topic}\n模塊：{module}\n結果（JSON，截斷）：\n"
                    + json.dumps(result, ensure_ascii=False)[:3000],
                    task="synthesize").strip()
                if text:
                    return text
            except Exception:
                pass
        return self._summarize_local(module, result)

    @staticmethod
    def _summarize_local(module: str, r: Dict) -> str:
        if module == "shanghan_search":
            ids = "、".join(h["clause_id"] for h in r.get("hits", [])[:4])
            return f"檢得相關條文 {len(r.get('hits', []))} 條（{ids}），構成 A 層源流基礎。"
        if module == "shanghan_formula_rule":
            return (f"{r.get('formula', '')} 方證：核心證 "
                    f"{'、'.join(r.get('core_symptoms', [])[:4]) or '—'}；支持條文 "
                    f"{'、'.join(r.get('supporting_clauses', [])[:3])}。")
        if module == "shanghan_divergence_atlas":
            ag = r.get("agreement_matrix", [])
            hi = max(ag, key=lambda x: x["mean_term_agreement"], default=None)
            lo = min(ag, key=lambda x: x["mean_term_agreement"], default=None)
            seg = (f"九注本共 {r.get('n_commentary_rules', 0)} 條注文，"
                   f"{r.get('n_clauses_multi_commentator', 0)} 條條文多注家。")
            if hi and lo:
                seg += (f"一致度最高 {hi['a']}×{hi['b']}（{hi['mean_term_agreement']}），"
                        f"最低 {lo['a']}×{lo['b']}（{lo['mean_term_agreement']}）。")
            return seg
        if module == "shanghan_dose":
            if r.get("ratio"):
                return (f"{r['formula']} 藥量比 {r['ratio']['ratio']}（銖當量，學派無關）；"
                        f"家族劑量邊 {len(r.get('evolution_edges', []))} 條。")
            return (f"全庫劑量摘要：dose-only 家族邊 {r.get('n_dose_only_edges', 0)} 條；"
                    f"{r.get('note', '')}")
        if module == "shanghan_corpus_stats":
            top = "、".join(f"{f}({n})" for f, n in r.get("top_formulas", [])[:4])
            return (f"全庫 {r.get('initial_rules', 0)} 條初始規則；高頻方 {top}。")
        if module == "shanghan_eval_metrics":
            cz = (r.get("suites", {}).get("cloze", {})
                  .get("metrics", {}).get("attainable", {}))
            gr = r.get("suites", {}).get("grounding", {}).get("metrics", {})
            return (f"遮方基準（可達折）Top-1 {cz.get('top1', '—')}、MRR "
                    f"{cz.get('mrr', '—')}；接地率 "
                    f"{gr.get('grounded_answer_rate', '—')}——方法可信度可查證。")
        if module == "shanghan_mistreatment":
            p = (r.get("paths") or [{}])[0]
            return (f"誤治路徑 {len(r.get('paths', []))} 條，典型如 "
                    f"{p.get('mistreatment', '')}→{p.get('resulting_pattern', '')}"
                    f"（{'、'.join(p.get('clauses', [])[:2])}）。")
        if module == "shanghan_case_search":
            c0 = (r.get("cases") or [{}])[0]
            return (f"醫案旁證 {r.get('n_matched', 0)} 案（{r.get('source', '')}），"
                    f"如「{c0.get('title', '')[:16]}」證見 "
                    f"{'、'.join(c0.get('symptoms', [])[:3])}，經文錨點 "
                    f"{'、'.join(c0.get('canonical_support', [])[:2])}。")
        if module == "shanghan_trace":
            t = r.get("trace", {}) or {}
            cit = t.get("citations") or t.get("citations_of_clauses") or {}
            anchors = (t.get("supporting_clauses", {}).get("canonical")
                       or [t.get("first_attestation", {}).get("clause_id", "")]
                       or [t.get("clause", {}).get("clause_id", "")])
            anchors = [a for a in anchors if a][:3]
            return (f"{t.get('chain_type', '溯源鏈')}（{t.get('formula', '') or t.get('query', '')}）："
                    f"歷代 {cit.get('n_citing_books', 0)} 部著作存在逐字引文邊；"
                    f"錨點條文 {'、'.join(anchors) or '—'}；證據分級 "
                    f"{'；'.join(t.get('evidence_grade', [])[:3])}。")
        if module == "shanghan_citation_network":
            ov = r.get("overview", {})
            return (f"引文網絡：{ov.get('n_clause_edges', 0)} 條條文引文邊、"
                    f"{ov.get('n_citing_works', 0)} 部引用著作、"
                    f"覆蓋條文 {ov.get('n_clauses_cited', 0)} 條；"
                    f"存疑標記 {ov.get('n_marker_unresolved', 0)} 處（多為引他書）。")
        if module in ("shanghan_six_channel", "shanghan_differential"):
            d = r.get("differential") or {}
            if d:
                return (f"鑒別 {' vs '.join(d.get('formulas', []))}：" +
                        "；".join(d.get("key_discriminators", [])[:2]))
            return (f"{r.get('six_channel', '')}：{r.get('summary', '')[:60]}"
                    f"（{r.get('outline_clause_id', '')}）")
        return json.dumps(result, ensure_ascii=False)[:200]
