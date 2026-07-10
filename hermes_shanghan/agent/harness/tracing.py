"""Span 級軌跡（OpenTelemetry 風格的本地 JSONL 實現，純標準庫）。

每個事件統一為 span：trace_id / span_id / parent_span_id / span_type /
started_at / ended_at / duration_ms / input_hash / output_hash / tokens /
cost / error / evidence_ids / metadata。落盤 `runs/<run_id>/events.jsonl`，
可直接被外部 OTel 管道轉譯。local 後端無 token/cost 計量時如實記 null。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ... import config
from ..citation_guard import RE_CLAUSE_ID

RE_ABS_PATH = re.compile(r"(/[\w.\-]+){3,}")


def sanitize_error(exc: BaseException) -> str:
    """異常入軌跡前脫敏：截斷 + 去絕對路徑（軌跡可能被導出/共享，
    不應洩露文件系統佈局或用戶輸入全文）。"""
    msg = str(exc).replace(str(config.REPO_ROOT), "<repo>")
    msg = RE_ABS_PATH.sub("<path>", msg)
    return f"{type(exc).__name__}: {msg[:200]}"


def _digest(obj: Any) -> str:
    try:
        blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        blob = str(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class TraceStore:
    def __init__(self, run_dir: Path, trace_id: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self.trace_id = trace_id or uuid.uuid4().hex[:16]

    def _write(self, span: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(span, ensure_ascii=False) + "\n")

    def span(self, span_type: str, name: str,
             parent_span_id: Optional[str] = None) -> "Span":
        return Span(self, span_type, name, parent_span_id)

    def read(self) -> List[Dict]:
        """讀取 span 事件；損壞行**跳過並計數**，不拖垮整個詳情/列表
        （十四輪 十九：一個殘行不應讓 Run Detail 500）。"""
        if not self.path.exists():
            return []
        out: List[Dict] = []
        corrupt = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                corrupt += 1
        if corrupt:
            out.append({"span_type": "corrupt_records", "name": "trace",
                        "corrupt_lines": corrupt, "trace_id": self.trace_id,
                        "span_id": "corrupt", "parent_span_id": None,
                        "started_at": "", "ended_at": "", "duration_ms": 0,
                        "input_hash": "", "output_hash": "", "tokens": None,
                        "cost": None, "error": f"{corrupt} 行損壞已跳過",
                        "mentioned_clause_ids": [], "metadata": {}})
        return out


class Span:
    def __init__(self, store: TraceStore, span_type: str, name: str,
                 parent_span_id: Optional[str]):
        self.store = store
        self.span_type = span_type
        self.name = name
        self.parent_span_id = parent_span_id
        self.span_id = uuid.uuid4().hex[:16]
        self.metadata: Dict[str, Any] = {}
        self.tokens: Optional[Dict[str, int]] = None
        self.cost: Optional[float] = None
        self._input_hash = ""
        self._output_hash = ""
        self._evidence: List[str] = []
        self._error: Optional[str] = None

    def set_input(self, obj: Any) -> None:
        self._input_hash = _digest(obj)

    def set_output(self, obj: Any) -> None:
        self._output_hash = _digest(obj)
        try:
            blob = json.dumps(obj, ensure_ascii=False, default=str)
            self._evidence = sorted(set(RE_CLAUSE_ID.findall(blob)))[:40]
        except Exception:
            pass

    def set_error(self, exc: BaseException) -> None:
        self._error = sanitize_error(exc)

    def __enter__(self) -> "Span":
        self._t0 = time.time()
        self._started = time.strftime("%Y-%m-%dT%H:%M:%S")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and self._error is None:
            self.set_error(exc)
        self.store._write({
            "trace_id": self.store.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "span_type": self.span_type,
            "name": self.name,
            "started_at": self._started,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_ms": int((time.time() - self._t0) * 1000),
            "input_hash": self._input_hash,
            "output_hash": self._output_hash,
            "tokens": self.tokens,
            "cost": self.cost,
            "error": self._error,
            # 十四輪 十四：這是「文本中提到的編號」（含模型可能偽造的），
            # 不是已核驗證據——核驗證據只在 Broker 台賬
            "mentioned_clause_ids": self._evidence,
            "metadata": self.metadata,
        })
        return False   # 不吞異常，交由節點重試策略處理


class TracedRegistry:
    """工具註冊表的 tracing + 預算 + **證據登記**代理。

    每次 call 產生一個 tool span；執行前向 RunBudget 原子扣減（九輪
    P0-3）；成功執行後把工具結果中的條文證據以結構化 EvidenceRecord 寫入
    run 的 evidence_ledger——**這是台賬唯一的寫入口**（十一輪 P0-1：模型
    輸出不能自我登記為證據，只有 Capability Broker 在工具成功執行後可
    登記，且每條記錄綁定 tool_call_id/span_id/source_hash/語料指紋）。
    budget 跨 for_role 副本共享。"""

    MAX_LEDGER_RECORDS = 400

    def __init__(self, base, store: TraceStore, parent_span_id: Optional[str],
                 state=None, budget=None, node_id: str = "execute"):
        self._base = base
        self._store = store
        self._parent = parent_span_id
        self._state = state
        self._budget = budget
        self._node = node_id

    def names(self):
        return self._base.names()

    def specs(self):
        return self._base.specs()

    def for_role(self, role):
        return TracedRegistry(self._base.for_role(role), self._store,
                              self._parent, self._state, self._budget,
                              self._node)

    @property
    def art(self):
        return self._base.art

    @property
    def matcher(self):
        return self._base.matcher

    @property
    def clause_rag(self):
        return self._base.clause_rag

    def resolve_formula(self, formula):
        return self._base.resolve_formula(formula)

    def call(self, name, arguments):
        with self._store.span("tool", name, self._parent) as sp:
            sp.set_input(arguments)
            if self._budget is not None and \
                    not self._budget.reserve_tool_call(name):
                out = {"error": "BUDGET_EXHAUSTED：本次運行工具預算已用盡，"
                                "剩餘調用一律拒絕執行（達到預算即停，"
                                "請基於已取證作答）",
                       "budget": self._budget.snapshot()}
                sp.metadata["budget_denied"] = True
                sp.set_output(out)
                if self._state is not None:
                    self._state.tool_calls.append(
                        {"tool": name, "span_id": sp.span_id,
                         "args_hash": sp._input_hash,
                         "error": out["error"], "budget_denied": True})
                return out
            out = self._base.call(name, arguments or {})
            sp.set_output(out)
            if isinstance(out, dict):
                if out.get("error"):
                    sp.metadata["tool_error"] = out["error"][:200]
                if out.get("cache_hit"):
                    sp.metadata["cache_hit"] = True
            if self._state is not None:
                self._state.tool_calls.append(
                    {"tool": name, "span_id": sp.span_id,
                     "args_hash": sp._input_hash,
                     "error": (out or {}).get("error") if isinstance(out, dict) else None})
                if isinstance(out, dict) and not out.get("error"):
                    self._register_evidence(name, sp.span_id, out,
                                            arguments=arguments)
            return out

    def _register_evidence(self, tool: str, span_id: str, out: Dict,
                           arguments: Optional[Dict] = None) -> None:
        """Broker 證據登記：只登記**存在於條文庫**且出自工具結果的 id，
        每條帶強不變量字段（tool_call_id/span_id/source_hash/語料指紋）。

        十三輪 五：登記時區分證據角色——
          primary_text_returned  條文正文（≥12 連續字）確實出現在工具輸出
                                 中（模型可讀到原文）
          id_mention_only        僅編號出現，正文未返回——「編號出現≠證據
                                 被返回」，外層審計對此類引用響亮標注
        並保留 excerpt / retrieval_query 供 UI 逐條複核。逐工具聲明式
        evidence_records 契約見路線圖（現階段为分類式登記，如實標注）。"""
        query = ""
        if isinstance(arguments, dict):
            query = str(arguments.get("query")
                        or arguments.get("formula")
                        or arguments.get("ref") or "")[:60]
        ledger = self._state.evidence_ledger.setdefault(self._node, [])

        # —— P 層（全庫文獻）證據登記（十五輪 P0-2）：classics 工具在結果
        # 中攜帶結構化 passage_evidence（verbatim+座標+quote_hash，工具層
        # 構造保證正文確實返回）——Broker 綁定 tool_call_id/span_id 後入賬
        p_seen = {(r.get("passage_id"), r["tool"]) for r in ledger
                  if r.get("passage_id")}
        for rec in (out.get("passage_evidence") or []):
            if not (isinstance(rec, dict) and rec.get("passage_id")
                    and rec.get("verbatim_text") and rec.get("quote_hash")):
                continue
            if (rec["passage_id"], tool) in p_seen:
                continue
            if len(ledger) >= self.MAX_LEDGER_RECORDS:
                break
            ledger.append({
                "passage_id": rec["passage_id"],
                "evidence_level": "P",
                "work_id": rec.get("work_id"),
                "work_title": rec.get("work_title"),
                "section": rec.get("section"),
                "char_start": rec.get("char_start"),
                "char_end": rec.get("char_end"),
                "quote_hash": rec["quote_hash"],
                "excerpt": rec["verbatim_text"][:60],
                "tool": tool,
                "tool_call_id": span_id,
                "span_id": span_id,
                "source_hash": rec["quote_hash"],
                "evidence_role": "primary_text_returned",
                "retrieval_query": rec.get("retrieval_query") or query or None,
                "corpus_fingerprint": self._state.spec.corpus_version,
                "registered_by": "capability_broker",
            })
            p_seen.add((rec["passage_id"], tool))

        try:
            blob = json.dumps(out, ensure_ascii=False, default=str)
        except Exception:
            return
        ids = sorted(set(RE_CLAUSE_ID.findall(blob)))
        if not ids:
            return
        try:
            store = self._base.art.clause_store()
        except Exception:
            return
        seen = {(r.get("clause_id"), r["tool"]) for r in ledger
                if r.get("clause_id")}
        for cid in ids:
            c = store.get(cid)
            if c is None or (cid, tool) in seen:
                continue    # 庫中不存在的 id 不得成為證據（fail-closed）
            if len(ledger) >= self.MAX_LEDGER_RECORDS:
                break
            text = getattr(c, "clean_text", "") or ""
            probe = text[:12]
            text_returned = bool(probe) and probe in blob
            ledger.append({
                "clause_id": cid,
                "tool": tool,
                "tool_call_id": span_id,
                "span_id": span_id,
                "source_hash": _digest(text),
                "evidence_role": ("primary_text_returned" if text_returned
                                  else "id_mention_only"),
                "excerpt": text[:40] if text_returned else None,
                "retrieval_query": query or None,
                "corpus_fingerprint": self._state.spec.corpus_version,
                "registered_by": "capability_broker",
            })
