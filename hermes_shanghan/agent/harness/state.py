"""統一運行對象：RunSpec（不可變規格）與 RunState（可恢復狀態）。"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ... import config

RUN_MODES = ("agent", "council", "deep-research", "solve", "tool")
RUN_STATUSES = ("created", "running", "paused", "failed", "completed")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_run_id(query: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(f"{query}{time.time_ns()}".encode()).hexdigest()[:6]
    return f"run_{stamp}_{digest}"


@dataclass
class RunSpec:
    run_id: str
    user_query: str
    role: str = "researcher"
    mode: str = "agent"
    max_steps: int = 6
    max_tool_calls: int = 12
    safety_policy: str = "default"         # 紅旗分診+意圖守衛+角色治理
    evidence_policy: str = "strict_round"  # 引用必須綁定本輪工具證據
    created_at: str = field(default_factory=_now)
    corpus_version: str = ""
    tool_spec_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeSpec:
    node_id: str
    node_type: str                      # intake|execute|guard|release|...
    inputs: List[str] = field(default_factory=list)
    tool_policy: List[str] = field(default_factory=list)   # 空=按角色默認
    retry_policy: int = 1               # 失敗重試次數
    fallback_policy: str = "fail"       # fail | skip | degrade
    evidence_requirement: str = ""      # 該節點必須產出的證據說明
    release_condition: str = ""         # 進入下一步的條件說明

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeResult:
    node_id: str
    status: str = "pending"             # pending|running|ok|failed|skipped|degraded
    attempts: int = 0
    started_at: str = ""
    duration_ms: int = 0
    output_digest: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    spec: RunSpec
    status: str = "created"
    plan: List[NodeSpec] = field(default_factory=list)
    nodes: Dict[str, NodeResult] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    evidence_ledger: Dict[str, List[str]] = field(default_factory=dict)
    tool_calls: List[Dict] = field(default_factory=list)
    guardrail_events: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    release: Dict[str, Any] = field(default_factory=dict)
    pending_review: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status,
            "plan": [n.to_dict() for n in self.plan],
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "node_outputs": self.node_outputs,
            "evidence_ledger": self.evidence_ledger,
            "tool_calls": self.tool_calls,
            "guardrail_events": self.guardrail_events,
            "errors": self.errors,
            "final_answer": self.final_answer,
            "release": self.release,
            "pending_review": self.pending_review,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunState":
        spec = RunSpec(**{k: v for k, v in d["spec"].items()})
        st = cls(spec=spec, status=d.get("status", "created"))
        st.plan = [NodeSpec(**n) for n in d.get("plan", [])]
        st.nodes = {k: NodeResult(**v) for k, v in d.get("nodes", {}).items()}
        st.node_outputs = d.get("node_outputs", {})
        st.evidence_ledger = d.get("evidence_ledger", {})
        st.tool_calls = d.get("tool_calls", [])
        st.guardrail_events = d.get("guardrail_events", [])
        st.errors = d.get("errors", [])
        st.final_answer = d.get("final_answer")
        st.release = d.get("release", {})
        st.pending_review = d.get("pending_review", [])
        return st


def spec_versions() -> Dict[str, str]:
    """RunSpec 的語料/工具版本指紋（審計可追）。"""
    import json
    corpus = ""
    manifest = config.MANIFEST_DIR / "corpus_manifest.json"
    if manifest.exists():
        corpus = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    tool_v = ""
    spec_path = config.SHANGHAN_DIR / "tool_specs.json"
    if spec_path.exists():
        try:
            n = len(json.loads(spec_path.read_text(encoding="utf-8"))["openai_tools"])
            tool_v = f"v1+{n}tools"
        except Exception:
            tool_v = "unknown"
    return {"corpus_version": corpus, "tool_spec_version": tool_v}
