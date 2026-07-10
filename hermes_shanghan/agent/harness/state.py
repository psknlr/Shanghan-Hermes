"""統一運行對象：RunSpec（不可變規格）、RunBudget（統一預算）、RunState。"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from ... import config

RUN_MODES = ("agent", "council", "deep-research", "solve", "tool", "classics")
RUN_STATUSES = ("queued", "created", "running", "paused", "failed",
                "completed", "blocked", "rejected", "cancelling", "cancelled")


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
    # 環境指紋（九輪：replay 對比的前提是凍結並記錄環境）
    corpus_version: str = ""
    tool_spec_version: str = ""
    python_version: str = ""
    backend: str = ""                      # llm 後端（local / litellm 模型名）
    code_fingerprint: str = ""             # git HEAD（不可得時為空，如實）
    # 代碼樹內容哈希：覆蓋 dirty 工作區與 ZIP（無 git）場景——同 commit
    # 本地改動/解壓包改動都會變（十四輪 十六）
    code_tree_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RunBudget:
    """統一預算：由 Harness 控制器持有並在每次工具執行前**原子扣減**——
    批量 tool_calls 逐個檢查，超限的調用返回 BUDGET_EXHAUSTED 不執行。
    計數器跨 for_role/Scoped 副本共享（與 FaultInjectionRegistry 同一教訓）。"""

    def __init__(self, max_tool_calls: int = 12, max_wall_ms: int = 0):
        self.max_tool_calls = max_tool_calls
        self.max_wall_ms = max_wall_ms          # 0 = 不限
        self.used_tool_calls = 0
        self.denied_tool_calls = 0
        self._t0 = time.time()
        self._lock = threading.Lock()

    def reserve_tool_call(self, tool_name: str = "") -> bool:
        with self._lock:
            if self.max_wall_ms and \
                    (time.time() - self._t0) * 1000 > self.max_wall_ms:
                self.denied_tool_calls += 1
                return False
            if self.used_tool_calls >= self.max_tool_calls:
                self.denied_tool_calls += 1
                return False
            self.used_tool_calls += 1
            return True

    def snapshot(self) -> Dict[str, int]:
        return {"max_tool_calls": self.max_tool_calls,
                "used_tool_calls": self.used_tool_calls,
                "denied_tool_calls": self.denied_tool_calls}


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
    trace_id: str = ""                  # resume 沿用同一 trace（軌跡跨恢復延續）
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
    approval_requests: List[Dict] = field(default_factory=list)
    approved_items: List[str] = field(default_factory=list)
    budget_snapshot: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status,
            "trace_id": self.trace_id,
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
            "approval_requests": self.approval_requests,
            "approved_items": self.approved_items,
            "budget_snapshot": self.budget_snapshot,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunState":
        known = {f.name for f in fields(RunSpec)}
        spec = RunSpec(**{k: v for k, v in d["spec"].items() if k in known})
        st = cls(spec=spec, status=d.get("status", "created"))
        st.trace_id = d.get("trace_id", "")
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
        st.approval_requests = d.get("approval_requests", [])
        st.approved_items = d.get("approved_items", [])
        st.budget_snapshot = d.get("budget_snapshot", {})
        return st


_CODE_TREE_CACHE: Dict[str, str] = {}


def _code_tree_fingerprint() -> str:
    """hermes_shanghan/ 全部 .py 內容的聚合哈希（進程內緩存）。
    覆蓋 git dirty 與 archive 場景；prompts/路由/發布策略都在代碼樹內，
    其變更均反映於此。語料內容指紋=manifest（含逐文件 sha256）——直接
    改數據文件而不重建 manifest 的漂移由 readyz 條數/規則檢查兜底（如實
    標注局限）。"""
    if "v" in _CODE_TREE_CACHE:
        return _CODE_TREE_CACHE["v"]
    h = hashlib.sha256()
    pkg = config.REPO_ROOT / "hermes_shanghan"
    try:
        for f in sorted(pkg.rglob("*.py")):
            h.update(str(f.relative_to(pkg)).encode())
            h.update(f.read_bytes())
        _CODE_TREE_CACHE["v"] = h.hexdigest()[:12]
    except OSError:
        _CODE_TREE_CACHE["v"] = ""
    return _CODE_TREE_CACHE["v"]


def spec_versions() -> Dict[str, str]:
    """RunSpec 的環境指紋（審計/replay 可追）。code_fingerprint 在非 git
    環境如實留空，不編造。"""
    import json
    import platform
    import subprocess
    corpus = ""
    manifest = config.MANIFEST_DIR / "corpus_manifest.json"
    if manifest.exists():
        corpus = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    tool_v = ""
    spec_path = config.SHANGHAN_DIR / "tool_specs.json"
    if spec_path.exists():
        try:
            raw = spec_path.read_bytes()
            n = len(json.loads(raw.decode("utf-8"))["openai_tools"])
            # 內容哈希（十四輪 十六）：28 個工具的 Schema/契約變更即使
            # 數量不變也會改變指紋——「數量相同≠規格相同」
            tool_v = f"{n}tools@{hashlib.sha256(raw).hexdigest()[:12]}"
        except Exception:
            tool_v = "unknown"
    code = ""
    try:
        code = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_ROOT,
            capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        code = ""
    backend = ""
    try:
        from ...llm.client import get_client
        backend = get_client().backend
    except Exception:
        backend = ""
    return {"corpus_version": corpus, "tool_spec_version": tool_v,
            "code_tree_fingerprint": _code_tree_fingerprint(),
            "python_version": platform.python_version(),
            "backend": backend, "code_fingerprint": code}
