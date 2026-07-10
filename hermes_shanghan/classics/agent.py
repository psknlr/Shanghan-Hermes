"""ClassicsAgent — 第二套智能體：全量古籍研究（獨立於傷寒論智能體）。

與 ShanghanAgent 的關係：**並列**，不是附庸。工具面只用 classics_*
（平台層，全庫 803 部），不觸傷寒論規則庫；證據層是 P 層 EvidenceRecord
（verbatim+座標+quote_hash），不是 A 層條文編號。

研究過程全程留痕（智能體工作台四要素）：
    research_log.plan                檢索計劃（意圖→步驟）
    research_log.queried_works       已查書目
    research_log.unqueried_candidates 未查候選（掃描封頂剩餘）
    research_log.supporting / counter_candidates / first_candidates
    research_log.needs_human_review  待人工核驗清單

回答中的每一段引用帶 ``psg_<hash>`` 標記——外層 Harness 獨立複核時
只認 Broker 台賬中 primary_text_returned 的 P 層記錄（無證據鏈，不成
回答，對全庫文獻同樣成立）。

當前為確定性規劃器（離線可復現）；LLM 增益層（自由式規劃/綜述撰寫）
沿用平台的 trusted-base/augmentation 分離，屬下一步。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..corpus import library as _libmod
from .evidence import conclusion_policy_check, verify_records
from .tools import _searcher

RE_QUOTED = re.compile(r"[「『“]([^」』”]{2,24})[」』”]")
RE_BOOK = re.compile(r"《([^》]{2,14})》")
RE_EARLIEST = re.compile(r"最早|最先|首見|首见|首載|首载|首現|首现|首倡|源出")
RE_DRIFT = re.compile(r"演變|演变|流變|流变|變遷|变迁|歷代.{0,6}變化|朝代分佈|朝代分布")
RE_WITNESS = re.compile(r"版本|傳本|传本|異文|异文|校勘|對勘|哪些本")
RE_STATS = re.compile(r"多少部|幾部|几部|收錄.{0,4}書|藏書量|全庫統計|全库统计|書目統計")
RE_TERM = re.compile(r"是甚麼|是什麼|是什么|何謂|何谓|甚麼意思|什麼意思|什么意思|術語|术语")
_STOP = ("歷代醫書", "歷代", "哪些書", "哪部書", "全庫", "全库", "古籍",
         "醫籍", "文獻", "記載", "提到", "如何論述", "怎麼論述", "論述",
         "如何", "怎麼", "檢索", "請問", "請", "一下", "裡", "中的", "中",
         "有", "了", "嗎", "呢", "？", "?", "。", "，")
_EDGE = "在的與与和或於于對对把被就也又"


def _extract_topic(question: str) -> str:
    m = RE_QUOTED.search(question)
    if m:
        return m.group(1)
    m = RE_BOOK.search(question)
    if m:
        return m.group(1)
    q = question
    for w in _STOP:
        q = q.replace(w, " ")
    runs = re.findall(r"[㐀-鿿]{2,}", q)
    topic = max(runs, key=len) if runs else question[:8]
    return topic.strip(_EDGE) or topic


class ClassicsAgent:
    """全量古籍研究智能體（確定性規劃 + 工具取證 + P 層引用自審）。"""

    def __init__(self, registry=None):
        if registry is None:
            from ..agent.tools import get_registry
            registry = get_registry()
        self.registry = registry

    # ------------------------------------------------------------------
    def ask(self, question: str, role: str = "researcher") -> Dict[str, Any]:
        question = (question or "").strip()
        if not _libmod.is_available():
            return {"question": question,
                    "answer": "全量古籍庫尚未就緒：請先運行 "
                              "`python3 -m hermes_shanghan library fetch`。"
                              "（誠實拒答：無庫不作全庫結論）",
                    "refused": True, "refusal_reason": "library_unavailable",
                    "tools_used": [], "passage_evidence": [],
                    "backend": "classics-deterministic"}

        topic = _extract_topic(question)
        plan = self._plan(question, topic)
        tool_results: List[Dict] = []
        tools_used: List[str] = []
        for step in plan:
            out = self.registry.call(step["tool"], step["args"])
            tools_used.append(step["tool"])
            step["ok"] = not (isinstance(out, dict) and out.get("error"))
            tool_results.append(out if isinstance(out, dict)
                                else {"error": "non-dict result"})

        evidence: List[Dict] = []
        for r in tool_results:
            evidence.extend(r.get("passage_evidence") or [])
        answer = self._compose(question, topic, plan, tool_results, evidence)

        # 自審（外層 Harness 還會獨立複核——這裡是業務層 self report）
        s = _searcher()
        verification = (verify_records(evidence, s.index)
                        if s and evidence else
                        {"ok": not evidence, "n_verified": 0, "n_failed": 0,
                         "failures": []})
        violations = conclusion_policy_check(answer, evidence, tools_used)
        log = self._research_log(plan, tool_results, evidence, violations)
        return {"question": question, "topic": topic, "answer": answer,
                "backend": "classics-deterministic", "role": role,
                "tools_used": tools_used,
                "passage_evidence": evidence,
                "research_log": log,
                "audit": {"quote_verification": verification,
                          "policy_violations": violations,
                          "authority": "agent_self_report（外層 Harness 另有"
                                       "獨立複核）"},
                "refused": False}

    # ------------------------------------------------------------------
    def _plan(self, q: str, topic: str) -> List[Dict]:
        if RE_STATS.search(q):
            return [{"intent": "library_stats",
                     "tool": "classics_library_stats", "args": {}}]
        if RE_EARLIEST.search(q):
            return [{"intent": "trace_citation",
                     "tool": "classics_trace_citation",
                     "args": {"quote": topic}}]
        if RE_DRIFT.search(q):
            return [{"intent": "concept_drift",
                     "tool": "classics_concept_drift", "args": {"term": topic}},
                    {"intent": "context_search",
                     "tool": "classics_search_passages",
                     "args": {"query": topic, "limit": 4}}]
        if RE_WITNESS.search(q) and (RE_BOOK.search(q) or topic):
            return [{"intent": "compare_witnesses",
                     "tool": "classics_compare_witnesses",
                     "args": {"work": (RE_BOOK.search(q) or [None]) and
                              (RE_BOOK.search(q).group(1)
                               if RE_BOOK.search(q) else topic),
                              "query": ""}}]
        if RE_TERM.search(q):
            return [{"intent": "resolve_term",
                     "tool": "classics_resolve_term", "args": {"term": topic}},
                    {"intent": "context_search",
                     "tool": "classics_search_passages",
                     "args": {"query": topic, "limit": 4}}]
        return [{"intent": "search",
                 "tool": "classics_search_passages",
                 "args": {"query": topic, "limit": 6}}]

    # ------------------------------------------------------------------
    @staticmethod
    def _cite(ev: Dict) -> str:
        who = "·".join(x for x in (ev.get("dynasty"), ev.get("author")) if x)
        sec = f"·{ev['section']}" if ev.get("section") else ""
        return (f"《{ev.get('work_title', '')}》{sec}"
                f"（{who or '朝代作者不詳'}）〔{ev.get('passage_id', '')}〕")

    def _compose(self, question: str, topic: str, plan: List[Dict],
                 results: List[Dict], evidence: List[Dict]) -> str:
        intent = plan[0]["intent"]
        r0 = results[0] if results else {}
        lines: List[str] = []
        if r0.get("error"):
            return (f"檢索未能完成：{r0['error']}。"
                    "（如實報錯，不以空結果冒充結論。）")
        if intent == "library_stats" and r0.get("available"):
            lines.append(f"中醫笈成全庫共收書 {r0['n_books']} 部"
                         f"（{r0['n_units']} 個文本單元，最大嵌套 "
                         f"{r0['max_depth']} 層）。")
            cats = list(r0.get("categories", {}).items())[:6]
            lines.append("分類分佈（前 6）：" +
                         "、".join(f"{c} {n} 部" for c, n in cats) + "。")
            lines.append("（此為全庫書目統計；傷寒論規則庫統計請問"
                         "「規則庫有多少條規則」。）")
        elif intent == "trace_citation" and r0.get("available"):
            e = r0.get("earliest_in_library")
            if e:
                ev0 = next((v for v in evidence
                            if v["passage_id"] == e["passage_id"]), None)
                lines.append(f"「{r0['quote']}」在庫中最早見於"
                             f"{self._cite(ev0) if ev0 else e['title']}"
                             f"——此為**在庫首現**，不等於歷史首現"
                             f"（庫外與亡佚文獻不可見）。")
                later = r0.get("attestations_time_ordered", [])[1:4]
                if later:
                    lines.append("其後時間有序的載錄：" + "；".join(
                        f"{h['dynasty'] or '?'}《{h['title']}》"
                        for h in later) + "。")
                counter = (r0.get("counter_search") or {}) \
                    .get("earlier_partial_candidates", [])
                if counter:
                    lines.append(f"反證搜索發現 {len(counter)} 個更早的部分"
                                 "匹配候選（需人工核驗，見 research_log）。")
                else:
                    lines.append("反證搜索（截半探針）未發現更早的部分匹配"
                                 "候選。")
            else:
                lines.append(f"「{r0.get('quote', topic)}」在掃描範圍內未見"
                             "逐字載錄" +
                             ("（掃描封頂，非全庫定論）"
                              if r0.get("scan_capped") else "。"))
        elif intent == "concept_drift" and r0.get("available"):
            series = r0.get("series_by_dynasty", [])
            if series:
                lines.append(f"「{topic}」的朝代分佈（命中段數）：" + "；".join(
                    f"{b['dynasty']} {b['n_passages']} 段"
                    f"（{b['n_works']} 部，如《{b['top_works'][0]}》）"
                    for b in series[:6]) + "。")
                lines.append("頻次漂移≠語義漂移——語義級結論需逐段研讀"
                             "（見引用段落）。")
            else:
                lines.append(f"「{topic}」在掃描範圍內未見載錄。")
        elif intent == "compare_witnesses" and r0.get("available"):
            wits = r0.get("witnesses", [])
            lines.append(f"《{r0.get('work_base', topic)}》在庫傳本 "
                         f"{len(wits)} 種：" + "；".join(
                             f"{w['title']}（{w['dynasty'] or '?'}"
                             f"{'·' + w['edition'] if w['edition'] else ''}）"
                             for w in wits[:6]) + "。")
        elif intent == "resolve_term" and r0.get("available"):
            lines.append(f"「{r0['term']}」折疊形「{r0['folded_form']}」，"
                         f"見於 {r0['n_works_with_hits']} 部著作；分類分佈："
                         + "、".join(f"{c}({n})" for c, n in
                                     list(r0.get("by_category", {}).items())[:5])
                         + "。")
        else:
            hits_ev = evidence[:5]
            if hits_ev:
                lines.append(f"「{topic}」全庫檢得以下載錄：")
                for ev in hits_ev:
                    lines.append(f"- {self._cite(ev)}："
                                 f"「{ev['verbatim_text'][:60]}…」")
            else:
                capped = any(r.get("scan_capped") for r in results
                             if isinstance(r, dict))
                lines.append(f"「{topic}」在掃描範圍內未檢得載錄" +
                             ("（掃描封頂 max_scan，非全庫不存在——"
                              "可調大 max_scan 重試）" if capped else "。"))
        # 統一附引用段落（P 層證據）
        cited = {m for m in re.findall(r"psg_[0-9a-f]{12}", "\n".join(lines))}
        extra = [ev for ev in evidence[:4] if ev["passage_id"] not in cited]
        if extra and intent not in ("search",):
            lines.append("依據段落：" + "；".join(self._cite(ev)
                                                  for ev in extra))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _research_log(plan, results, evidence, violations) -> Dict:
        queried = sorted({ev["work_id"] for ev in evidence})
        capped = any(isinstance(r, dict) and r.get("scan_capped")
                     for r in results)
        counter = []
        first_candidates = []
        for r in results:
            if not isinstance(r, dict):
                continue
            cs = (r.get("counter_search") or {}) \
                .get("earlier_partial_candidates", [])
            counter.extend(cs)
            if r.get("earliest_in_library"):
                first_candidates.append(r["earliest_in_library"])
        needs_review = [{"item": "earlier_partial_candidate",
                         "detail": f"{c['title']}（{c.get('probe', '')}）"}
                        for c in counter]
        needs_review += [{"item": "policy_violation", "detail": v["violation"]}
                         for v in violations]
        return {"plan": [{k: s[k] for k in ("intent", "tool", "args", "ok")
                          if k in s} for s in plan],
                "queried_works": queried,
                "unqueried_candidates": ("掃描封頂——存在未掃描候選（調大 "
                                         "max_scan 可續查）" if capped
                                         else "本輪候選集已掃描完"),
                "supporting_evidence_count": len(evidence),
                "counter_candidates": counter[:8],
                "first_candidates": first_candidates,
                "needs_human_review": needs_review}
