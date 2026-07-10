"""AgentSession — multi-turn conversation with an evidence ledger.

Agents were stateless: every question started from zero. A session keeps
(question, answer, evidence) history plus a clause-id ledger, and resolves
follow-up references（「它的劑量比呢？」「上面那條的注家分歧？」）by
prepending a compact context block before dispatch — so the deterministic
router and a real model alike see the anchors (方名/條文號) from earlier
turns. Compound questions route to the ComplexAgent orchestrator, simple
ones to the plain ReAct agent.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .. import lexicon
from ..llm.client import LLMClient, get_client
from ..textutil import normalize_query
from .complex_agent import ComplexAgent
from .agent import ShanghanAgent
from .tools import ToolRegistry, get_registry

RE_FOLLOWUP = re.compile(r"它|其(?![他餘])|這個|那個|上面|上述|前面|剛才|此方|該方|呢[？?]?$")
RE_COMPOUND = re.compile(r"[？?；;].+\S|(鑒別|對比).+(劑量|注家|誤治)|"
                         r"(劑量|注家).+(鑒別|誤治|提綱)")
# user corrections（「這裡不是桂枝加芍藥湯，而是桂枝去芍藥湯」）are remembered
# for the rest of the session so the same slip is not repeated
RE_CORRECTION = re.compile(r"不是\s*([^，,。；;\s]{2,14})\s*[，,]?\s*(?:而是|應是|应是|是)\s*([^，,。；;\s]{2,14})")


class AgentSession:
    def __init__(self, client: Optional[LLMClient] = None,
                 registry: Optional[ToolRegistry] = None,
                 role: Optional[str] = None, max_history: int = 8,
                 namespace: str = "anon"):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        self.role = role
        self.max_history = max_history
        # 記憶命名空間 = 服務端主體：糾正只在本會話內生效；持久化僅作
        # 帶來源的登記（見 _record_corrections），防跨用戶記憶投毒
        self.namespace = namespace
        self.history: List[Dict] = []          # {question, answer, evidence}
        self.ledger: Dict[str, str] = {}       # clause_id -> snippet
        self.anchors: List[str] = []           # formulas mentioned so far
        self.corrections: List[Dict] = []      # {wrong, right} user fixes

    # ------------------------------------------------------------------
    def ask(self, question: str, role: Optional[str] = None) -> Dict[str, Any]:
        role = role or self.role
        self._record_corrections(question)
        resolution = self._resolve_reference(question)
        contextual = self._contextualize(question, resolution)
        if RE_COMPOUND.search(question):
            agent = ComplexAgent(client=self.client, registry=self.registry)
            out = agent.solve(contextual, role=role)
        else:
            agent = ShanghanAgent(client=self.client, registry=self.registry)
            out = agent.ask(contextual, role=role)
        self._remember(question, out)
        out["session"] = {"turn": len(self.history),
                          "contextualized": contextual != question,
                          "anchors": list(self.anchors),
                          "ledger_size": len(self.ledger),
                          # 結構化指代解析（十二輪：不再只報「成功=True」——
                          # 解析到什麼、依據什麼、置信多少，全部可審）
                          "reference_resolution": resolution}
        return out

    def _resolve_reference(self, question: str) -> Dict[str, Any]:
        """結構化指代解析：mention → 解析對象 + 依據 + 啟發式置信。

        關鍵區分（十二輪「偽成功」修復）：**用戶問句中的主語錨點**優先於
        回答文本裡順帶出現的方名——「問桂枝湯 → 答文提到附子湯」時，
        「它」解析為桂枝湯而不是回答裡最後出現的方名。
        誠實邊界：詞表+錨點啟發式（非語義消解）；無錨點如實報 unresolved。"""
        m = RE_FOLLOWUP.search(question)
        if not m or not self.history:
            return {"mention": None, "resolved": None, "confidence": None,
                    "status": "no_reference"}
        mention = m.group(0)
        subject = next((h["subject"] for h in reversed(self.history)
                        if h.get("subject")), None)
        if subject:
            return {"mention": mention, "resolved": subject,
                    "candidates": list(self.anchors[-3:]),
                    "confidence": 0.85, "status": "resolved",
                    "basis": "最近**問句主語**錨點（優先於回答文本中順帶"
                             "出現的方名；詞表級，非語義消解）",
                    "evidence_carried": self.history[-1]["evidence"][:4]}
        if not self.anchors:
            return {"mention": mention, "resolved": None, "confidence": 0.0,
                    "status": "unresolved",
                    "note": "有指代詞但先前輪次無方名/條文錨點——無從解析"}
        return {"mention": mention, "resolved": self.anchors[-1],
                "candidates": list(self.anchors[-3:]),
                "confidence": 0.4, "status": "resolved",
                "basis": "回答文本錨點回退（主語未知，置信降檔）",
                "evidence_carried": self.history[-1]["evidence"][:4]}

    # ------------------------------------------------------------------
    def _record_corrections(self, question: str) -> None:
        """「不是X，而是Y」→ remember {wrong: X, right: Y}。

        生效範圍只在本會話（self.corrections）；持久化到 correction_memory
        僅為**帶來源的登記**（namespace/信任級/時間），信任級一律
        unverified_user_correction——模型/其他會話不得把它當事實應用
        （九輪 P0-6：防跨用戶記憶投毒；用戶級刪除按 namespace 過濾即可）。"""
        for wrong, right in RE_CORRECTION.findall(normalize_query(question)):
            entry = {"wrong": wrong, "right": right}
            if entry not in self.corrections:
                self.corrections.append(entry)
                try:
                    import time
                    from ..memory.store import MemoryStore
                    mem = MemoryStore("correction_memory")
                    record = {**entry, "namespace": self.namespace,
                              "trust": "unverified_user_correction",
                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    existing = mem.get("user_corrections", [])
                    if not any(e.get("wrong") == wrong
                               and e.get("right") == right
                               and e.get("namespace", "anon") == self.namespace
                               for e in existing):
                        mem.append("user_corrections", record, max_items=100)
                        mem.save()
                except Exception:
                    pass       # persistence is best-effort; session memory holds

    def _correction_note(self) -> str:
        if not self.corrections:
            return ""
        pairs = "；".join(f"{c['wrong']}→{c['right']}"
                          for c in self.corrections[-3:])
        return f"（用戶已糾正，請勿再犯：{pairs}）"

    # 可被直接替換為主語的代詞（呢/上面 等語氣與方位詞不可直替）
    _SUBSTITUTABLE = ("它", "此方", "該方", "這個", "那個", "其")

    def _contextualize(self, question: str, resolution: Optional[Dict] = None
                       ) -> str:
        """把指代解析結果作為**硬約束**注入改寫問題（十三輪 P0-六：
        「元數據解析對、答案答錯方」的根因是舊實現只給模型看歷史錨點
        列表——第一輪回答裡順帶出現的類方（桂枝加芍藥湯…）會污染路由。
        現在：解析成功時直接把代詞改寫為主語方名，且**不再**注入多方名
        錨點列表，路由只能看到主語。"""
        note = self._correction_note()
        if not self.history or not RE_FOLLOWUP.search(question):
            return (note + "\n" + question) if note else question
        last = self.history[-1]
        if resolution and resolution.get("status") == "resolved":
            subject = resolution["resolved"]
            mention = resolution.get("mention") or ""
            rewritten = question
            if mention in self._SUBSTITUTABLE and mention in question:
                rewritten = question.replace(mention, subject, 1)
            elif subject not in question:
                rewritten = f"{subject}{question}" if question.startswith(
                    ("的", "之")) else f"{subject}：{question}"
            ctx = (f"（指代解析：「{mention}」={subject}。請圍繞{subject}"
                   f"本方作答，勿切換到其加減方或類方")
            if last["evidence"]:
                ctx += f"；上輪已核實條文：{'、'.join(last['evidence'][:4])}"
            ctx += "）"
            return ctx + (note or "") + "\n當前追問：" + rewritten
        ctx = [f"（先前對話：問「{last['question'][:40]}」，"
               f"答及 {'、'.join(self.anchors[:3]) or '（無方名）'}"]
        if last["evidence"]:
            ctx.append(f"；已核實條文：{'、'.join(last['evidence'][:4])}")
        ctx.append("）")
        if note:
            ctx.append(note)
        return "".join(ctx) + "\n當前追問：" + question

    def _remember(self, question: str, out: Dict) -> None:
        evidence = out.get("evidence_clause_ids", []) or []
        store = self.registry.art.clause_store()
        for cid in evidence:
            c = store.get(cid)
            if c and cid not in self.ledger:
                self.ledger[cid] = c.clean_text[:60]
        # 問句主語（用戶提到的方名）與回答順帶方名分開記——指代解析
        # 以主語為準，防「問桂枝湯、答文含附子湯→它=附子湯」的偽解析
        q_norm = normalize_query(question)
        subject = next((n for n in sorted(lexicon.FORMULA_SEEDS,
                                          key=len, reverse=True)
                        if n in q_norm), None)
        blob = normalize_query(question + " " + out.get("answer", "")[:400])
        for name in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True):
            if name in blob and name not in self.anchors:
                self.anchors.append(name)
        self.anchors = self.anchors[-6:]
        self.history.append({"question": question,
                             "answer": out.get("answer", "")[:400],
                             "subject": subject,
                             "evidence": evidence})
        self.history = self.history[-self.max_history:]

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {"turns": len(self.history), "anchors": self.anchors,
                "ledger": self.ledger,
                "corrections": self.corrections,
                "history": [{"question": h["question"],
                             "evidence": h["evidence"]} for h in self.history]}
