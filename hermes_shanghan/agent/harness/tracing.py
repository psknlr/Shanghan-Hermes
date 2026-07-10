"""Span 級軌跡（OpenTelemetry 風格的本地 JSONL 實現，純標準庫）。

每個事件統一為 span：trace_id / span_id / parent_span_id / span_type /
started_at / ended_at / duration_ms / input_hash / output_hash / tokens /
cost / error / evidence_ids / metadata。落盤 `runs/<run_id>/events.jsonl`，
可直接被外部 OTel 管道轉譯。local 後端無 token/cost 計量時如實記 null。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..citation_guard import RE_CLAUSE_ID


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
        if not self.path.exists():
            return []
        return [json.loads(x) for x in
                self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


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
        self._error = f"{type(exc).__name__}: {exc}"

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
            "evidence_ids": self._evidence,
            "metadata": self.metadata,
        })
        return False   # 不吞異常，交由節點重試策略處理


class TracedRegistry:
    """工具註冊表的 tracing 代理：每次 call 產生一個 tool span。"""

    def __init__(self, base, store: TraceStore, parent_span_id: Optional[str],
                 state=None):
        self._base = base
        self._store = store
        self._parent = parent_span_id
        self._state = state

    def names(self):
        return self._base.names()

    def specs(self):
        return self._base.specs()

    def for_role(self, role):
        return TracedRegistry(self._base.for_role(role), self._store,
                              self._parent, self._state)

    @property
    def art(self):
        return self._base.art

    @property
    def matcher(self):
        return self._base.matcher

    @property
    def clause_rag(self):
        return self._base.clause_rag

    def call(self, name, arguments):
        with self._store.span("tool", name, self._parent) as sp:
            sp.set_input(arguments)
            out = self._base.call(name, arguments or {})
            sp.set_output(out)
            if isinstance(out, dict) and out.get("error"):
                sp.metadata["tool_error"] = out["error"]
            if self._state is not None:
                self._state.tool_calls.append(
                    {"tool": name, "span_id": sp.span_id,
                     "args_hash": sp._input_hash,
                     "error": (out or {}).get("error") if isinstance(out, dict) else None})
            return out
