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
        contextual = self._contextualize(question)
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
                          "ledger_size": len(self.ledger)}
        return out

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

    def _contextualize(self, question: str) -> str:
        note = self._correction_note()
        if not self.history or not RE_FOLLOWUP.search(question):
            return (note + "\n" + question) if note else question
        last = self.history[-1]
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
        blob = normalize_query(question + " " + out.get("answer", "")[:400])
        for name in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True):
            if name in blob and name not in self.anchors:
                self.anchors.append(name)
        self.anchors = self.anchors[-6:]
        self.history.append({"question": question,
                             "answer": out.get("answer", "")[:400],
                             "evidence": evidence})
        self.history = self.history[-self.max_history:]

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {"turns": len(self.history), "anchors": self.anchors,
                "ledger": self.ledger,
                "corrections": self.corrections,
                "history": [{"question": h["question"],
                             "evidence": h["evidence"]} for h in self.history]}
