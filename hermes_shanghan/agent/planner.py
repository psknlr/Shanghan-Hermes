"""Planner — 從「關鍵詞路由」升級為「任務圖規劃」.

輸出結構化計劃而非扁平子任務串：

    {"goal": …,
     "subtasks": [{"id": "T1", "kind": "general", "question": …,
                   "required_tools": […], "depends_on": []},
                  {"id": "T3", "kind": "differential", "question": …,
                   "depends_on": ["T1", "T2"]}],
     "success_criteria": ["每個結論須有 clause_id", "須分別覆蓋 寒化 與 熱化", …]}

對比類問題（「少陰寒化與熱化怎麼區分？」「桂枝湯與麻黃湯各自…」）自動展開
為「逐對象取證 → 匯總對比」的依賴圖，對比任務能看到取證任務的證據；
其餘複合問題退化為句段分類（與舊 _decompose_local 行為兼容）。
真實模型可用時由 JSON 規劃產生同樣結構，local 後端走確定性路徑——
同一套執行器，離線在線同構。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import lexicon
from ..llm.prompts import EVIDENCE_CONTRACT
from ..textutil import normalize_query

RE_SPLIT = re.compile(r"[？?；;。]\s*")
RE_COMPARATIVE = re.compile(r"鑒別|區別|區分|對比|比較|異同|不同|vs")
# aspect pairs that expand into per-aspect retrieval subtasks even without
# two formula anchors（少陰「寒化 vs 熱化」、陽明「經證 vs 腑證」…）
ASPECT_PAIRS = [("寒化", "熱化"), ("經證", "腑證"), ("表證", "裏證")]

_KIND_PATTERNS = [
    ("safety_check", r"能不能用|可不可以用|禁忌嗎|犯不犯禁"),
    ("case", r"醫案|案例|實驗錄|診案"),
    ("therapy", r"[汗下吐和溫清補]法|禁[汗下吐]|治法"),
    ("differential", r"鑒別|對比|區別|區分|不同|vs"),
    ("dose", r"劑量|藥量|用量|折算|幾兩|幾克|銖"),
    ("commentary", r"注家|注本|詮釋|分歧|成無己|柯琴|尤怡|方有執"),
    ("mistreatment", r"誤治|誤下|誤汗|誤吐|傳變|壞病|變證|救逆"),
    ("six_channel", r"提綱|六經|經證|欲解時"),
    ("stats", r"統計|頻次|多少條|基準|評測|接地率"),
    ("literature", r"笈成|全庫文獻|古籍|醫籍|歷代醫書|後世醫[家書]|哪些書|哪部書|書目"),
    ("research", r"溯源|源流|演化史|全面研究|綜述"),
]

BASE_CRITERIA = [
    "每個結論必須有 clause_id 支撐",
    "引用只可來自本輪工具證據（不能只是語料庫中存在）",
    "後世病機歸納（D/E 層）不得當作原文（A 層）陳述",
]


class Planner:
    """task_types: kind → {"desc", "tools"}（來自 ComplexAgent.TASK_TYPES，
    以參數注入避免循環依賴）。"""

    def __init__(self, client=None, task_types: Optional[Dict] = None,
                 max_subtasks: int = 6):
        self.client = client
        self.task_types = task_types or {}
        self.max_subtasks = max_subtasks

    # ------------------------------------------------------------------
    def plan(self, question: str) -> Dict:
        if self.client is not None and getattr(self.client, "available", False):
            planned = self._plan_llm(question)
            if planned:
                return planned
        return self._plan_local(question)

    # ------------------------------------------------------------------
    def _tools_for(self, kind: str) -> List[str]:
        return list(self.task_types.get(kind, {}).get("tools", []))

    def _plan_local(self, question: str) -> Dict:
        q = normalize_query(question)
        anchors = [n for n in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True)
                   if n in q][:3]
        # —— comparative expansion: per-object retrieval + join ————————
        # Only for aspect comparisons（寒化/熱化、經證/腑證…）: formula-pair
        # comparison already has a dedicated tool (shanghan_differential),
        # but no tool answers an aspect contrast directly — so the planner
        # decomposes it into per-aspect retrieval and a dependent join.
        compare_objects: List[str] = []
        if RE_COMPARATIVE.search(q) and len(anchors) < 2:
            for a, b in ASPECT_PAIRS:
                if a in q and b in q:
                    subject = next((c for c in lexicon.CHANNEL_IN_TEXT.values()
                                    if c in q or c[:-1] in q), "")
                    compare_objects = [f"{subject}{a}", f"{subject}{b}"]
                    break
        if compare_objects:
            subtasks: List[Dict] = []
            for i, obj in enumerate(compare_objects, 1):
                subtasks.append({
                    "id": f"T{i}", "kind": "general",
                    "question": f"{obj}的證候要點、主方與條文依據",
                    "required_tools": self._tools_for("general"),
                    "depends_on": []})
            # extra facets asked alongside the comparison（誤治/劑量/注家…）
            for seg in RE_SPLIT.split(question):
                seg = seg.strip()
                if not seg:
                    continue
                kind = next((k for k, pat in _KIND_PATTERNS
                             if re.search(pat, normalize_query(seg))), None)
                if kind in ("mistreatment", "dose", "commentary", "therapy",
                            "case", "literature") and kind in self.task_types \
                        and not any(t["kind"] == kind for t in subtasks):
                    subtasks.append({
                        "id": f"T{len(subtasks) + 1}", "kind": kind,
                        "question": f"{seg}（涉及：{'、'.join(compare_objects)}）",
                        "required_tools": self._tools_for(kind),
                        "depends_on": []})
            subtasks = subtasks[:self.max_subtasks - 1]
            subtasks.append({
                "id": f"T{len(subtasks) + 1}", "kind": "general",
                "question": question,
                "required_tools": self._tools_for("general"),
                "depends_on": [t["id"] for t in subtasks]})
            criteria = BASE_CRITERIA + [
                f"必須分別覆蓋：{'、'.join(compare_objects)}",
                "對比結論須落在關鍵鑒別軸上（如 寒/熱、二便、煩躁、脈象、治法、主方）",
            ]
            return {"goal": question, "planner": "local_task_graph",
                    "subtasks": subtasks,
                    "success_criteria": criteria}

        # —— fallback: segment classification（與舊行為兼容）——————
        segments = [s.strip() for s in RE_SPLIT.split(question) if s.strip()]
        tasks: List[Dict] = []
        for seg in segments or [question]:
            kind = next((k for k, pat in _KIND_PATTERNS
                         if re.search(pat, normalize_query(seg))), "general")
            if kind not in self.task_types:
                kind = "general"
            sub_q = seg
            if anchors and not any(a in normalize_query(seg) for a in anchors):
                sub_q = f"{seg}（涉及：{'、'.join(anchors)}）"
            tasks.append({"kind": kind, "question": sub_q})
        merged: List[Dict] = []
        for t in tasks:
            if merged and merged[-1]["kind"] == t["kind"]:
                merged[-1]["question"] += "；" + t["question"]
            else:
                merged.append(t)
        subtasks = [{"id": f"T{i}", "kind": t["kind"], "question": t["question"],
                     "required_tools": self._tools_for(t["kind"]),
                     "depends_on": []}
                    for i, t in enumerate(merged[:self.max_subtasks], 1)]
        return {"goal": question, "planner": "local_segments",
                "subtasks": subtasks,
                "success_criteria": list(BASE_CRITERIA)}

    # ------------------------------------------------------------------
    def _plan_llm(self, question: str) -> Optional[Dict]:
        catalog = "\n".join(f"- {k}：{v['desc']}"
                            for k, v in self.task_types.items())
        try:
            plan = self.client.json_complete(
                EVIDENCE_CONTRACT + "\n\n任務：把複合問題規劃為任務圖（1-6 個"
                "子任務）。對比類問題應先逐對象取證、再加一個 depends_on 全部"
                "取證任務的對比任務。嚴格輸出 JSON：{\"goal\":\"…\","
                "\"subtasks\":[{\"id\":\"T1\",\"kind\":\"…\",\"question\":\"…\","
                "\"depends_on\":[]}],\"success_criteria\":[\"…\"]}",
                f"複合問題：{question}\n可用任務類型：\n{catalog}",
                task="synthesize")
        except Exception:
            return None
        subtasks = []
        ids = set()
        for i, t in enumerate(plan.get("subtasks", []), 1):
            if t.get("kind") not in self.task_types or not t.get("question"):
                continue
            tid = str(t.get("id") or f"T{i}")
            subtasks.append({
                "id": tid, "kind": t["kind"], "question": t["question"],
                "required_tools": self._tools_for(t["kind"]),
                "depends_on": [d for d in (t.get("depends_on") or [])]})
            ids.add(tid)
        # drop dangling dependencies so the executor cannot deadlock
        for t in subtasks:
            t["depends_on"] = [d for d in t["depends_on"] if d in ids]
        if not subtasks:
            return None
        criteria = [str(c) for c in (plan.get("success_criteria") or [])]
        return {"goal": plan.get("goal", question), "planner": "llm_task_graph",
                "subtasks": subtasks[:self.max_subtasks],
                "success_criteria": criteria or list(BASE_CRITERIA)}


def execution_order(subtasks: List[Dict]) -> List[Dict]:
    """Dependency-respecting order (stable topological sort); cycles are
    broken by falling back to listed order for the remainder."""
    done: set = set()
    ordered: List[Dict] = []
    pending = list(subtasks)
    while pending:
        progress = False
        for t in list(pending):
            if all(d in done for d in t.get("depends_on", [])):
                ordered.append(t)
                done.add(t["id"])
                pending.remove(t)
                progress = True
        if not progress:            # cycle: execute remainder as listed
            ordered.extend(pending)
            break
    return ordered
