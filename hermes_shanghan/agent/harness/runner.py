"""Harness 運行器：顯式節點圖執行 + checkpoint/resume/replay/export。

v1 節點圖（四模式同構）：

    intake（安全預檢+角色確認）
      → execute（模式引擎：agent/council/deep-research/solve；
                 工具一律經 TracedRegistry：逐調用產 span + 統一預算扣減）
      → evidence_audit（CitationGuard 對最終回答複核）
      → release_gate（五道閘門；review_required → paused；
                      blocked/failed_closed 不可人工放行）

每節點帶 retry_policy / fallback_policy；每步落 checkpoint
（`runs/<run_id>/state.json`），中斷後 `run-resume` 從未完成節點續跑；
運行目錄帶 run.lock（單寫者），trace_id 跨 resume 延續。
人工批准不是改狀態：resume --approve 會**重新執行** evidence_audit 與
release_gate（帶 approved 集合）再放行。
更細粒度的圖原生編排（檢索/專家/批評各自成節點）見 docs/HARNESS.md 路線。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ... import config
from ..citation_guard import CitationGuard, RE_CLAUSE_ID
from .release_gate import HUMAN_REVIEW_TRIGGERS, evaluate as gate_evaluate
from .state import (RUN_MODES, NodeResult, NodeSpec, RunBudget, RunSpec,
                    RunState, new_run_id, spec_versions)
from .tracing import TracedRegistry, TraceStore, _digest

LOCK_STALE_S = 600      # run.lock 超過此秒數視為殘留（進程崩潰未清理）


def run_dir(run_id: str) -> Path:
    return config.RUNS_DIR / run_id


def _technical_failures(state) -> List[str]:
    """不可審批覆蓋的技術失敗（十三輪 P0-四）：非法模式/節點異常/
    輸出為空/降級執行——這些是系統完整性問題，人工批准不能把它們變成
    「成功完成」。"""
    problems: List[str] = []
    exec_out = state.node_outputs.get("execute", {}) or {}
    refused = bool(exec_out.get("refused"))
    for node_id in ("intake", "execute"):
        res = state.nodes.get(node_id)
        if res and res.status in ("failed", "degraded"):
            problems.append(f"必經節點 {node_id} 狀態 {res.status}"
                            + (f"（{res.error}）" if res.error else ""))
    if not refused and not (state.final_answer or "").strip():
        problems.append("final_answer 為空且非拒答——空輸出不可發布")
    if exec_out.get("error") and not refused:
        problems.append(f"execute 錯誤：{str(exec_out['error'])[:120]}")
    return problems


def _ledger_ids_verified(state) -> List[str]:
    """證據台賬完整性校驗（十一輪 P0-1 強不變量）：每條記錄必須由
    Capability Broker 登記且綁定 tool_call_id / span_id / source_hash /
    本 run 的語料指紋——違反即拋錯（寧可炸也不放行偽證據）。"""
    ids: List[str] = []
    for node_id, recs in state.evidence_ledger.items():
        for r in recs:
            if not (isinstance(r, dict) and r.get("tool_call_id")
                    and r.get("span_id") and r.get("source_hash")
                    and r.get("registered_by") == "capability_broker"
                    and r.get("corpus_fingerprint")
                    == state.spec.corpus_version):
                raise RuntimeError(
                    f"evidence ledger 完整性違例（node={node_id}）："
                    "記錄缺少 Broker 綁定字段或語料指紋不符——"
                    "台賬只能由工具執行後的 Broker 寫入")
            ids.append(r["clause_id"])
    return sorted(set(ids))


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
                 release_condition="五道閘門裁定（review_required→paused；"
                                   "blocked/failed_closed 不可人工放行）"),
    ]


class _RunLock:
    """單寫者鎖：防多進程對同一 run 目錄交錯寫入（checkpoint/JSONL）。"""

    def __init__(self, d: Path):
        self.path = d / "run.lock"
        self._fd = None
        self._hb = None
        self._stop = threading.Event()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.path),
                               os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime
            if age < LOCK_STALE_S:
                raise RuntimeError(
                    f"run 正在被另一進程執行（{self.path.name} 存在且未過期，"
                    f"age={int(age)}s）；如確認殘留可刪除該文件")
            self.path.unlink()          # 殘留鎖：接管
            self._fd = os.open(str(self.path),
                               os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self._fd, f"pid={os.getpid()} at={time.strftime('%FT%T')}"
                 .encode())
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb.start()
        return self

    HEARTBEAT_S = 30

    def touch(self) -> None:
        try:
            os.utime(self.path)
        except OSError:
            pass

    def _heartbeat_loop(self) -> None:
        # 獨立心跳線程（十三輪 七）：節點執行 10 分鐘也不會被殘留判定
        # 接管——心跳與節點時長解耦；CAS/lease 版本號屬 SQLite 路線
        while not self._stop.wait(self.HEARTBEAT_S):
            self.touch()

    def __exit__(self, *exc):
        self._stop.set()
        if self._hb is not None:
            self._hb.join(timeout=2)
        if self._fd is not None:
            os.close(self._fd)
        self.path.unlink(missing_ok=True)
        return False


class HarnessRunner:
    def __init__(self, registry=None, client=None):
        from ..tools import get_registry
        self.base_registry = registry or get_registry()
        # 依賴注入：模式引擎的 client 由控制器決定（測試可注入假後端）
        self.client = client

    # ------------------------------------------------------------------
    def prepare(self, query: str, mode: str = "agent",
                role: str = "researcher", max_steps: int = 6,
                max_tool_calls: int = 12, run_id: str = "") -> RunState:
        """建立 queued 狀態並**同步落盤**（十三輪 十一：幽靈 run 根除——
        API 返回前狀態必須已持久化）。非法模式在此 fail-fast，
        不創建注定失敗的後台任務。"""
        if mode not in RUN_MODES:
            raise ValueError(f"未知模式 {mode!r}（可用：{RUN_MODES}）")
        versions = spec_versions()
        spec = RunSpec(run_id=run_id or new_run_id(query), user_query=query,
                       role=role, mode=mode, max_steps=max_steps,
                       max_tool_calls=max_tool_calls, **versions)
        state = RunState(spec=spec, plan=_default_plan(mode), status="queued")
        state.nodes = {n.node_id: NodeResult(node_id=n.node_id)
                       for n in state.plan}
        save_state(state)
        return state

    def start(self, query: str, mode: str = "agent", role: str = "researcher",
              max_steps: int = 6, max_tool_calls: int = 12,
              run_id: str = "") -> RunState:
        return self._execute(self.prepare(query, mode=mode, role=role,
                                          max_steps=max_steps,
                                          max_tool_calls=max_tool_calls,
                                          run_id=run_id))

    def execute_prepared(self, run_id: str) -> Optional[RunState]:
        state = load_run(run_id)
        if state is None or state.status not in ("queued", "created"):
            return state
        return self._execute(state)

    @staticmethod
    def request_cancel(run_id: str) -> bool:
        """協作式取消：寫 cancel.flag，執行器在節點邊界檢查（工具只讀
        原子，無中斷點——與 MCP tasks 同一誠實口徑）。"""
        d = run_dir(run_id)
        if not (d / "state.json").exists():
            return False
        (d / "cancel.flag").write_text("cancel", encoding="utf-8")
        return True

    def resume(self, run_id: str, approve: bool = False, reject: bool = False,
               approver: str = "", reason: str = "") -> Optional[RunState]:
        state = load_run(run_id)
        if state is None:
            return None
        if state.status == "paused" and reject:
            state.guardrail_events.append(
                {"event": "human_review_rejected",
                 "approver": approver or "cli",
                 "reason": reason or "（未填理由）",
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "review_items": state.pending_review})
            for a in state.approval_requests:
                a["status"] = "rejected"
            state.status = "rejected"
            state.release["decision"] = "rejected_by_human_review"
            save_state(state)
            return state
        if state.status == "paused" and approve:
            # 審批邊界（十三輪 P0-四）：技術失敗不可經 approve 洗白——
            # 審批通道只裁決學術/臨床審核項，不豁免系統完整性約束
            failures = _technical_failures(state)
            if failures:
                state.guardrail_events.append(
                    {"event": "approval_refused_technical_failure",
                     "approver": approver or "cli",
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "failures": failures})
                state.release = {**state.release,
                                 "decision": "failed_closed",
                                 "reasons": state.release.get("reasons", [])
                                 + ["審批被拒：存在不可覆蓋的技術失敗——"
                                    + "；".join(failures)]}
                state.status = "failed"
                save_state(state)
                return state
            # 批准 ≠ 改狀態：記錄審批人與理由，然後**重新執行**下游閘門
            state.guardrail_events.append(
                {"event": "human_review_approved", "approver": approver or "cli",
                 "reason": reason or "（未填理由）",
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "review_items": state.pending_review})
            state.approved_items = sorted(
                set(state.approved_items) | set(state.pending_review))
            for a in state.approval_requests:
                if a.get("trigger") in state.approved_items:
                    a["status"] = "approved"
                    a["approver"] = approver or "cli"
            state.pending_review = []
            for node_id in ("evidence_audit", "release_gate"):
                if node_id in state.nodes:
                    state.nodes[node_id] = NodeResult(node_id=node_id)
            save_state(state)
            return self._execute(state)
        if state.status in ("completed", "blocked", "rejected", "cancelled"):
            return state
        return self._execute(state)

    def replay(self, run_id: str) -> Optional[Dict]:
        """重放：先對比環境指紋（語料/工具/代碼/Python/後端），再重跑同一
        RunSpec 對比回答指紋。指紋不一致時如實標 comparable=False——
        「當前代碼+當前語料重跑一遍」不等於可復現 replay。"""
        old = load_run(run_id)
        if old is None:
            return None
        current = spec_versions()
        mismatches = {}
        for k, v in current.items():
            recorded = getattr(old.spec, k, "")
            if recorded and recorded != v:
                mismatches[k] = {"recorded": recorded, "current": v}
        new_state = self.start(old.spec.user_query, mode=old.spec.mode,
                               role=old.spec.role, max_steps=old.spec.max_steps,
                               max_tool_calls=old.spec.max_tool_calls)
        comparable = (not mismatches
                      and current.get("backend") == "local"
                      and getattr(old.spec, "backend", "local")
                      in ("", "local"))
        return {"original_run": run_id, "replay_run": new_state.spec.run_id,
                "original_digest": _digest(old.final_answer),
                "replay_digest": _digest(new_state.final_answer),
                "deterministic_match":
                    _digest(old.final_answer) == _digest(new_state.final_answer),
                "comparable": comparable,
                "fingerprint_mismatches": mismatches,
                "note": "comparable=True（指紋一致+local 後端）時指紋必一致；"
                        "指紋不一致或真實 LLM 後端下的差異不構成回歸信號。"}

    # ------------------------------------------------------------------
    def _execute(self, state: RunState) -> RunState:
        spec = state.spec
        with _RunLock(run_dir(spec.run_id)) as lock:
            state.status = "running"
            save_state(state)
            trace = TraceStore(run_dir(spec.run_id),
                               trace_id=state.trace_id or None)
            state.trace_id = trace.trace_id      # resume 沿用同一 trace_id
            budget = RunBudget(max_tool_calls=spec.max_tool_calls)
            # resume：先前已執行的工具調用計入預算（預算屬於 run，不屬於進程）
            budget.used_tool_calls = len(
                [t for t in state.tool_calls if not t.get("budget_denied")])
            with trace.span("run", f"{spec.mode}:{spec.run_id}") as root:
                root.set_input(spec.to_dict())
                upstream_reran = False
                for node in state.plan:
                    res = state.nodes[node.node_id]
                    # 上游節點在本次恢復中重跑過 → 下游舊結果失效必須重算
                    # （否則失敗節點重試成功後，guard/release 仍持舊報告）
                    if upstream_reran and res.status == "ok":
                        res = state.nodes[node.node_id] = \
                            NodeResult(node_id=node.node_id)
                    # resume：已完成節點跳過；triage 分支標記的節點不執行
                    if res.status in ("ok", "skipped_by_triage"):
                        continue
                    if any(state.nodes[d].status not in
                           ("ok", "degraded", "skipped_by_triage")
                           for d in node.inputs):
                        res.status = "skipped"
                        save_state(state)
                        continue
                    if (run_dir(spec.run_id) / "cancel.flag").exists():
                        state.status = "cancelled"
                        state.guardrail_events.append(
                            {"event": "run_cancelled",
                             "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
                        save_state(state)
                        break
                    self._run_node(state, node, res, trace, root.span_id,
                                   budget)
                    upstream_reran = True
                    state.budget_snapshot = budget.snapshot()
                    save_state(state)
                    # 圖分支（十一輪 P0-4）：intake 的控制決策說停就停——
                    # execute/evidence_audit 被跳過，直接進發布閘門
                    if node.node_id == "intake":
                        dec = (state.node_outputs.get("intake", {})
                               .get("triage_decision") or {})
                        if dec and not dec.get("continue_execution", True):
                            state.final_answer = dec.get("message", "")
                            state.node_outputs["execute"] = {
                                "refused": True,
                                "message": state.final_answer,
                                "refused_intents": dec.get("intents", []),
                                "triage_outcome": dec.get("outcome")}
                            for skip_id in ("execute", "evidence_audit"):
                                if skip_id in state.nodes:
                                    state.nodes[skip_id].status = \
                                        "skipped_by_triage"
                            save_state(state)
                    if state.status in ("failed", "paused", "blocked"):
                        break
                root.set_output({"status": state.status,
                                 "answer_digest": _digest(state.final_answer)})
            state.budget_snapshot = budget.snapshot()
            save_state(state)
        return state

    def _run_node(self, state: RunState, node: NodeSpec, res: NodeResult,
                  trace: TraceStore, parent: str, budget: RunBudget) -> None:
        for attempt in range(node.retry_policy + 1):
            res.attempts = attempt + 1
            res.status = "running"
            res.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            t0 = time.time()
            try:
                with trace.span(node.node_type, node.node_id, parent) as sp:
                    out = self._dispatch(state, node, trace, sp.span_id, budget)
                    sp.set_output(out)
                    if isinstance(out, dict) and out.get("backend"):
                        sp.metadata["backend"] = out["backend"]
                res.duration_ms = int((time.time() - t0) * 1000)
                res.output_digest = _digest(out)
                # 十一輪 P0-1：**不再**用正則從節點輸出提取 clause_id 進
                # 台賬——模型輸出不能自我登記為證據。台賬唯一寫入口是
                # TracedRegistry（Broker 在工具成功執行後登記結構化記錄），
                # 節點只回讀本節點名下已登記的證據 id 作摘要
                res.evidence_ids = sorted({
                    r["clause_id"]
                    for r in state.evidence_ledger.get(node.node_id, [])})[:40]
                state.node_outputs[node.node_id] = out
                res.status = "ok"
                res.error = None
                return
            except Exception as exc:
                from .tracing import sanitize_error
                res.error = sanitize_error(exc)
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
                  span_id: str, budget: RunBudget) -> Dict:
        spec = state.spec
        if node.node_type == "intake":
            # 十一輪 P0-4：intake 輸出**強類型控制決策**，由圖執行器分支
            # ——不再只記事件然後繼續執行（安全決策屬控制器，不屬提示詞）
            from ... import safety
            decision = {"outcome": "safe", "continue_execution": True,
                        "message": "", "intents": []}
            flag = safety.red_flag_triage(spec.user_query) \
                if spec.role == "patient" else None
            if flag:
                state.guardrail_events.append({"event": "red_flag_triage",
                                               "detail": str(flag)[:200]})
                payload = safety.governed(dict(flag), "patient")
                decision = {"outcome": "emergency_redirect",
                            "continue_execution": False,
                            "message": payload.get("message")
                            or payload.get("answer")
                            or "檢測到急症紅旗信號，請立即就醫。",
                            "intents": flag.get("red_flags", [])}
            elif spec.role == "patient":
                guard = safety.patient_intent_guard(spec.user_query)
                if guard:
                    state.guardrail_events.append(
                        {"event": "intent_guard_refused",
                         "intents": guard.get("refused_intents", [])})
                    payload = safety.governed(dict(guard), "patient")
                    decision = {"outcome": "refused_intent",
                                "continue_execution": False,
                                "message": payload.get("message")
                                or payload.get("answer") or "該請求已被拒絕。",
                                "intents": guard.get("refused_intents", [])}
            return {"role": spec.role, "red_flag": bool(flag),
                    "triage": flag or None, "triage_decision": decision}

        if node.node_type == "execute":
            # 所有模式的依賴只能從此注入（不得自行 get_registry()）：
            # 工具面統一經 TracedRegistry → span 樹 + 工具台賬 + 預算扣減
            reg = TracedRegistry(self.base_registry, trace, span_id, state,
                                 budget)
            if spec.mode == "agent":
                from ..agent import ShanghanAgent
                out = ShanghanAgent(client=self.client, registry=reg,
                                    max_steps=spec.max_steps,
                                    max_tool_calls=spec.max_tool_calls) \
                    .ask(spec.user_query, role=spec.role)
            elif spec.mode == "council":
                from ..multi_agent import Council
                out = Council(client=self.client,
                              registry=reg).deliberate(spec.user_query,
                                                       role=spec.role)
            elif spec.mode == "deep-research":
                from ..research_loop import DeepResearcher
                out = DeepResearcher(client=self.client, registry=reg,
                                     max_rounds=spec.max_steps).run(spec.user_query)
                # 全部發現進入回答（不只前 4 條——七維研究不得靜默丟維度）
                out.setdefault("answer", "；".join(
                    f.get("summary", "") for f in out.get("findings", [])))
            elif spec.mode == "solve":
                from ..complex_agent import ComplexAgent
                out = ComplexAgent(client=self.client,
                                   registry=reg).solve(spec.user_query,
                                                       role=spec.role)
            else:
                raise ValueError(f"未知模式 {spec.mode}")
            state.final_answer = out.get("answer") or out.get("message", "")
            if out.get("refused"):
                state.guardrail_events.append(
                    {"event": "intent_guard_refused",
                     "intents": out.get("refused_intents", [])})
            return out

        if node.node_type == "guard":
            # 十三輪 P0：外層 Harness **無條件獨立複核**——被審計對象
            # （業務智能體）提交的 citation_report 只能作 agent_self_report，
            # 不得作為發布依據（否則偽造 ok=True 即可空台賬過閘）。
            # 權威報告 = CitationGuard(最終回答, 允許集=Broker 台賬)。
            exec_out = state.node_outputs.get("execute", {})
            inner = exec_out.get("citation_report")
            refused = bool(exec_out.get("refused"))
            allowed = _ledger_ids_verified(state)
            guard = CitationGuard(self.base_registry.art.clause_store())
            rep = guard.check(state.final_answer or "", allowed_ids=allowed)
            outer = {"ok": rep.ok, "has_any_citation": rep.has_any_citation,
                     "verified": rep.verified_ids,
                     "unsupported": rep.unsupported_ids,
                     "outside_evidence": rep.outside_evidence_ids,
                     "attribution_warnings": rep.attribution_warnings,
                     "authority": "harness_independent_audit"}
            disagreement = bool(inner) and (
                bool(inner.get("ok")) != rep.ok
                or set(inner.get("verified") or []) != set(rep.verified_ids))
            if disagreement:
                state.guardrail_events.append(
                    {"event": "citation_report_disagreement",
                     "detail": "業務智能體自報與外層獨立複核不一致——"
                               "以外層為準（自報僅存檔）",
                     "inner_ok": bool(inner.get("ok")),
                     "outer_ok": rep.ok})
            # 強不變量：strict_round 下非拒答回答必須有 Broker 台賬證據
            if spec.evidence_policy == "strict_round" and not refused \
                    and not allowed and rep.has_any_citation:
                outer["ok"] = False
            exec_out["agent_self_report"] = inner
            exec_out["citation_report"] = outer      # 權威報告覆蓋
            # 台賬中被引用證據若僅以編號出現（正文未返回），響亮標注
            cited = set(rep.verified_ids)
            id_only = sorted({r["clause_id"]
                              for recs in state.evidence_ledger.values()
                              for r in recs
                              if r.get("evidence_role") == "id_mention_only"
                              and r["clause_id"] in cited})
            if id_only:
                outer["id_only_cited"] = id_only
            return {"citation_report": outer, "agent_self_report": inner,
                    "disagreement": disagreement,
                    "ledger_records": sum(len(v) for v in
                                          state.evidence_ledger.values())}

        if node.node_type == "release":
            exec_out = state.node_outputs.get("execute", {})
            verdict = gate_evaluate(
                spec, exec_out,
                approved=frozenset(state.approved_items),
                tool_names=[t["tool"] for t in state.tool_calls
                            if not t.get("budget_denied")])
            # 發布不變量（十三輪 P0-四）：pass* 必須 (a) 有答案或明確拒答
            # (b) 必經節點無 failed/degraded (c) 證據閘通過或屬拒答。
            # 技術失敗不進人工審核隊列（那是「裁決學術爭議」的通道，
            # 不是「豁免系統故障」的通道）——一律 failed_closed，
            # **審批集合對其無效**
            failures = _technical_failures(state)
            if failures and verdict["decision"] != "blocked":
                verdict = {**verdict, "decision": "failed_closed",
                           "reasons": verdict.get("reasons", []) + [
                               "發布不變量違例（不可審批覆蓋）：" +
                               "；".join(failures)],
                           "technical_failures": failures}
            state.release = verdict
            decision = verdict["decision"]
            if decision == "review_required":
                state.status = "paused"
                state.pending_review = verdict["review_required"]
                now = time.strftime("%Y-%m-%dT%H:%M:%S")
                state.approval_requests = [
                    {"approval_id": f"{spec.run_id}:{trig}",
                     "run_id": spec.run_id, "node_id": "release_gate",
                     "trigger": trig,
                     "reason": HUMAN_REVIEW_TRIGGERS.get(trig, ""),
                     "action_digest": _digest(state.final_answer),
                     "evidence_digest": _digest(state.evidence_ledger),
                     "requested_at": now, "required_role": "human_reviewer",
                     "status": "pending"} for trig in state.pending_review]
            elif decision == "blocked":
                state.status = "blocked"
                state.errors.extend(verdict.get("blocked_reasons", []))
            elif decision == "failed_closed":
                state.status = "failed"
                state.errors.append("release_gate fail-closed："
                                    + "；".join(verdict.get("reasons", [])))
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
             f"- 預算：{state.budget_snapshot or '—'}",
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
    for a in state.approval_requests:
        lines.append(f"- 審批：{a['trigger']} → {a.get('status', 'pending')}"
                     + (f"（{a.get('approver', '')}）" if a.get("approver") else ""))
    lines += ["", "## 最終回答", "", state.final_answer or "（無）"]
    ids = sorted({r["clause_id"] for recs in state.evidence_ledger.values()
                  for r in recs if isinstance(r, dict)})
    lines += ["", "## 證據台賬（Broker 登記，含 tool_call/span 綁定）", "",
              "、".join(ids) or "（無）"]
    return "\n".join(lines)
