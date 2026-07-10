"""Harness 運行器：顯式節點圖執行 + checkpoint/resume/replay/export。

v1 節點圖（四模式同構）：

    intake（安全預檢+角色確認）
      → execute（模式引擎：agent/council/deep-research/solve；
                 工具經 TracedRegistry 逐調用產 span）
      → evidence_audit（CitationGuard 對最終回答複核）
      → release_gate（五道閘門；needs_human_review → paused）

每節點帶 retry_policy / fallback_policy；每步落 checkpoint
（`runs/<run_id>/state.json`），中斷後 `run-resume` 從未完成節點續跑。
更細粒度的圖原生編排（檢索/專家/批評各自成節點）見 docs/HARNESS.md 路線。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ... import config
from ..citation_guard import CitationGuard, RE_CLAUSE_ID
from .release_gate import evaluate as gate_evaluate
from .state import (NodeResult, NodeSpec, RunSpec, RunState, new_run_id,
                    spec_versions)
from .tracing import TracedRegistry, TraceStore, _digest


def run_dir(run_id: str) -> Path:
    return config.RUNS_DIR / run_id


def save_state(state: RunState) -> None:
    d = run_dir(state.spec.run_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(d / "state.json")


def load_run(run_id: str) -> Optional[RunState]:
    p = run_dir(run_id) / "state.json"
    if not p.exists():
        return None
    return RunState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def list_runs(limit: int = 30) -> List[Dict]:
    if not config.RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(config.RUNS_DIR.iterdir(), reverse=True)[:limit]:
        st = load_run(d.name)
        if st:
            out.append({"run_id": d.name, "status": st.status,
                        "mode": st.spec.mode, "role": st.spec.role,
                        "query": st.spec.user_query[:40],
                        "created_at": st.spec.created_at})
    return out


def _default_plan(mode: str) -> List[NodeSpec]:
    return [
        NodeSpec("intake", "intake", retry_policy=0,
                 evidence_requirement="紅旗分診+意圖守衛結論",
                 release_condition="未被安全攔截（攔截則直接進 release）"),
        NodeSpec("execute", "execute", inputs=["intake"], retry_policy=1,
                 fallback_policy="degrade",
                 evidence_requirement="回答 + 本輪工具證據 clause_id",
                 release_condition="產出非空回答或明確拒答"),
        NodeSpec("evidence_audit", "guard", inputs=["execute"], retry_policy=0,
                 evidence_requirement="CitationGuard 覆核報告",
                 release_condition="核驗報告生成"),
        NodeSpec("release_gate", "release", inputs=["evidence_audit"],
                 retry_policy=0,
                 release_condition="五道閘門裁定（needs_human_review→paused）"),
    ]


class HarnessRunner:
    def __init__(self, registry=None):
        from ..tools import get_registry
        self.base_registry = registry or get_registry()

    # ------------------------------------------------------------------
    def start(self, query: str, mode: str = "agent", role: str = "researcher",
              max_steps: int = 6) -> RunState:
        versions = spec_versions()
        spec = RunSpec(run_id=new_run_id(query), user_query=query, role=role,
                       mode=mode, max_steps=max_steps, **versions)
        state = RunState(spec=spec, plan=_default_plan(mode))
        state.nodes = {n.node_id: NodeResult(node_id=n.node_id)
                       for n in state.plan}
        save_state(state)
        return self._execute(state)

    def resume(self, run_id: str, approve: bool = False,
               approver: str = "") -> Optional[RunState]:
        state = load_run(run_id)
        if state is None:
            return None
        if state.status == "paused" and approve:
            state.guardrail_events.append(
                {"event": "human_review_approved", "approver": approver or "cli",
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "review_items": state.pending_review})
            state.release["decision"] = "released_after_human_review"
            state.pending_review = []
            state.status = "completed"
            save_state(state)
            return state
        if state.status in ("completed",):
            return state
        return self._execute(state)

    def replay(self, run_id: str) -> Optional[Dict]:
        """local 後端全確定：重跑同一 RunSpec，對比最終回答指紋。"""
        old = load_run(run_id)
        if old is None:
            return None
        new_state = self.start(old.spec.user_query, mode=old.spec.mode,
                               role=old.spec.role, max_steps=old.spec.max_steps)
        return {"original_run": run_id, "replay_run": new_state.spec.run_id,
                "original_digest": _digest(old.final_answer),
                "replay_digest": _digest(new_state.final_answer),
                "deterministic_match":
                    _digest(old.final_answer) == _digest(new_state.final_answer),
                "note": "local 確定性後端下應一致；真實 LLM 後端不保證。"}

    # ------------------------------------------------------------------
    def _execute(self, state: RunState) -> RunState:
        spec = state.spec
        state.status = "running"
        save_state(state)
        trace = TraceStore(run_dir(spec.run_id))
        with trace.span("run", f"{spec.mode}:{spec.run_id}") as root:
            root.set_input(spec.to_dict())
            for node in state.plan:
                res = state.nodes[node.node_id]
                if res.status == "ok":            # resume：已完成節點跳過
                    continue
                if any(state.nodes[d].status not in ("ok", "degraded")
                       for d in node.inputs):
                    res.status = "skipped"
                    save_state(state)
                    continue
                self._run_node(state, node, res, trace, root.span_id)
                save_state(state)
                if state.status in ("failed", "paused"):
                    break
            root.set_output({"status": state.status,
                             "answer_digest": _digest(state.final_answer)})
        save_state(state)
        return state

    def _run_node(self, state: RunState, node: NodeSpec, res: NodeResult,
                  trace: TraceStore, parent: str) -> None:
        spec = state.spec
        for attempt in range(node.retry_policy + 1):
            res.attempts = attempt + 1
            res.status = "running"
            res.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            t0 = time.time()
            try:
                with trace.span(node.node_type, node.node_id, parent) as sp:
                    out = self._dispatch(state, node, trace, sp.span_id)
                    sp.set_output(out)
                res.duration_ms = int((time.time() - t0) * 1000)
                res.output_digest = _digest(out)
                blob = json.dumps(out, ensure_ascii=False, default=str)
                res.evidence_ids = sorted(set(RE_CLAUSE_ID.findall(blob)))[:40]
                state.evidence_ledger[node.node_id] = res.evidence_ids
                state.node_outputs[node.node_id] = out
                res.status = "ok"
                res.error = None
                return
            except Exception as exc:
                res.error = f"{type(exc).__name__}: {exc}"
                if attempt < node.retry_policy:
                    continue
                if node.fallback_policy == "degrade":
                    state.node_outputs[node.node_id] = {
                        "answer": "（該步驟執行失敗，降級為無結果；請勿採信本次運行）",
                        "error": res.error, "citation_report": {
                            "ok": False, "has_any_citation": False}}
                    res.status = "degraded"
                    state.errors.append(f"{node.node_id}: {res.error}")
                    return
                if node.fallback_policy == "skip":
                    res.status = "skipped"
                    return
                res.status = "failed"
                state.status = "failed"
                state.errors.append(f"{node.node_id}: {res.error}")
                return

    # ------------------------------------------------------------------
    def _dispatch(self, state: RunState, node: NodeSpec, trace: TraceStore,
                  span_id: str) -> Dict:
        spec = state.spec
        if node.node_type == "intake":
            from ... import safety
            triage = getattr(safety, "red_flag_triage", None)
            flag = triage(spec.user_query) if callable(triage) else None
            out = {"role": spec.role, "red_flag": bool(flag),
                   "triage": flag or None}
            if flag:
                state.guardrail_events.append({"event": "red_flag_triage",
                                               "detail": str(flag)[:200]})
            return out

        if node.node_type == "execute":
            reg = TracedRegistry(self.base_registry, trace, span_id, state)
            if spec.mode == "agent":
                from ..agent import ShanghanAgent
                out = ShanghanAgent(registry=reg, max_steps=spec.max_steps) \
                    .ask(spec.user_query, role=spec.role)
            elif spec.mode == "council":
                from ..multi_agent import Council
                out = Council(registry=reg).deliberate(spec.user_query,
                                                       role=spec.role)
            elif spec.mode == "deep-research":
                from ..research_loop import DeepResearcher
                out = DeepResearcher(registry=reg,
                                     max_rounds=spec.max_steps).run(spec.user_query)
                out.setdefault("answer", "；".join(
                    f.get("summary", "") for f in out.get("findings", [])[:4]))
            elif spec.mode == "solve":
                from ..complex_agent import ComplexAgent
                out = ComplexAgent().solve(spec.user_query, role=spec.role)
            else:
                raise ValueError(f"未知模式 {spec.mode}")
            state.final_answer = out.get("answer") or out.get("message", "")
            if out.get("refused"):
                state.guardrail_events.append(
                    {"event": "intent_guard_refused",
                     "intents": out.get("refused_intents", [])})
            return out

        if node.node_type == "guard":
            exec_out = state.node_outputs.get("execute", {})
            report = exec_out.get("citation_report")
            if not report:
                allowed = sorted({i for ids in state.evidence_ledger.values()
                                  for i in ids})
                guard = CitationGuard(self.base_registry.art.clause_store())
                rep = guard.check(state.final_answer or "", allowed_ids=allowed)
                report = {"ok": rep.ok, "has_any_citation": rep.has_any_citation,
                          "verified": rep.verified_ids,
                          "unsupported": rep.unsupported_ids}
                exec_out["citation_report"] = report
            return {"citation_report": report}

        if node.node_type == "release":
            exec_out = state.node_outputs.get("execute", {})
            verdict = gate_evaluate(spec, exec_out)
            state.release = verdict
            if verdict["decision"] == "needs_human_review":
                state.status = "paused"
                state.pending_review = verdict["review_required"]
            else:
                state.status = "completed"
            return verdict

        raise ValueError(f"未知節點類型 {node.node_type}")


# ---------------------------------------------------------------------------
# 導出
# ---------------------------------------------------------------------------
def export_run(run_id: str, fmt: str = "md") -> Optional[str]:
    state = load_run(run_id)
    if state is None:
        return None
    if fmt == "json":
        events = TraceStore(run_dir(run_id)).read()
        return json.dumps({"state": state.to_dict(), "events": events},
                          ensure_ascii=False, indent=1)
    lines = [f"# Run {run_id}", "",
             f"- 查詢：{state.spec.user_query}",
             f"- 模式/角色：{state.spec.mode} / {state.spec.role}",
             f"- 狀態：{state.status}",
             f"- 語料版本：{state.spec.corpus_version} · 工具版本：{state.spec.tool_spec_version}",
             "", "## 節點軌跡", ""]
    for n in state.plan:
        r = state.nodes[n.node_id]
        lines.append(f"- **{n.node_id}**（{n.node_type}）：{r.status}，"
                     f"{r.attempts} 次嘗試，{r.duration_ms}ms"
                     + (f"，錯誤 {r.error}" if r.error else ""))
    lines += ["", "## 工具調用", ""]
    for t in state.tool_calls:
        lines.append(f"- {t['tool']}（span {t['span_id']}）"
                     + (f" ⚠ {t['error']}" if t.get("error") else ""))
    lines += ["", "## 發布裁定", "",
              f"- 決策：{state.release.get('decision', '—')}"]
    for r in state.release.get("reasons", []):
        lines.append(f"- {r}")
    lines += ["", "## 最終回答", "", state.final_answer or "（無）"]
    ids = sorted({i for ids in state.evidence_ledger.values() for i in ids})
    lines += ["", "## 證據台賬", "", "、".join(ids) or "（無）"]
    return "\n".join(lines)
