"""九輪治理測試：鏡像評審的動態探針——

  發布閘門 fail-closed / blocked 不可批准 / 駁回流
  預算：模型單輪批量 tool_calls 不可突破 max_tool_calls
  solve 模式工具調用進 Harness 台賬與 span 樹
  角色自提權被服務端 Policy 拒絕
  session 隔離（無 id 不共用 default；namespace 防串話）
  readyz 假健康防護 + 資產缺失響亮失敗
  工具契約運行時執行（超時/輸出形狀/版本化緩存鍵/審計）
  planner 圖編譯（環路/懸空依賴/重複 ID fail-closed）
  深研覆蓋狀態（FAILED/EMPTY 不算覆蓋）
"""
import json
import shutil
import unittest

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


# ---------------------------------------------------------------------------
class TestReleaseGateFailClosed(unittest.TestCase):
    def _spec(self, role="researcher", mode="agent"):
        from hermes_shanghan.agent.harness.state import RunSpec
        return RunSpec(run_id="t", user_query="q", role=role, mode=mode)

    def test_missing_citation_report_fails_closed(self):
        # 動態探針復現：缺失 citation report + 無引用回答 ≠ release
        from hermes_shanghan.agent.harness.release_gate import evaluate
        out = evaluate(self._spec(), {"answer": "桂枝湯治太陽中風。"})
        self.assertEqual(out["decision"], "failed_closed")

    def test_defaults_are_fail_closed(self):
        from hermes_shanghan.agent.harness.release_gate import evaluate
        # citation_report 在但字段缺失 → 默認取「未通過」而非 ok=True
        out = evaluate(self._spec(), {"answer": "…", "citation_report": {}})
        self.assertIn(out["decision"], ("review_required", "blocked"))
        self.assertFalse(out["gates"]["evidence_gate"]["ok"])

    def test_fabricated_citation_blocks_and_approval_cannot_release(self):
        from hermes_shanghan.agent.harness.release_gate import evaluate
        payload = {"answer": "見 SHL_SONGBEN_9999。",
                   "citation_report": {"ok": False, "has_any_citation": True,
                                       "unsupported": ["SHL_SONGBEN_9999"]}}
        out = evaluate(self._spec(), payload)
        self.assertEqual(out["decision"], "blocked")
        # 人工批准也不能放行 blocked（approved 集合無效）
        out2 = evaluate(self._spec(), payload,
                        approved=frozenset({"citation_failure"}))
        self.assertEqual(out2["decision"], "blocked")

    def test_patient_role_violation_is_role_gate_not_citation(self):
        from hermes_shanghan.agent.harness.release_gate import evaluate
        out = evaluate(self._spec(role="patient"),
                       {"answer": "桂枝湯主之，服用三劑。",
                        "citation_report": {"ok": True,
                                            "has_any_citation": True}})
        self.assertEqual(out["decision"], "blocked")
        self.assertFalse(out["gates"]["role_gate"]["ok"])
        self.assertNotIn("citation_failure", out["review_required"])

    def test_refusal_is_a_safe_conclusion(self):
        from hermes_shanghan.agent.harness.release_gate import evaluate
        out = evaluate(self._spec(role="patient"),
                       {"refused": True, "message": "…"})
        self.assertEqual(out["decision"], "pass")

    def test_structured_trigger_not_keyword(self):
        from hermes_shanghan.agent.harness.release_gate import evaluate
        report = {"ok": True, "has_any_citation": True}
        # 鑒別解說提到「桂枝湯」但無方劑推薦類工具/假設 → 不觸發人工審核
        out = evaluate(self._spec(role="doctor"),
                       {"answer": "桂枝湯與麻黃湯之別在汗之有無。",
                        "citation_report": report,
                        "tools_used": ["shanghan_differential"]})
        self.assertNotIn("doctor_formula_candidates", out["review_required"])
        # 方證匹配工具在台賬 → 觸發
        out2 = evaluate(self._spec(role="doctor"),
                        {"answer": "候選方如上。", "citation_report": report,
                         "tools_used": ["shanghan_match_formula"]})
        self.assertIn("doctor_formula_candidates", out2["review_required"])

    def test_reject_flow(self):
        _ensure_artifacts()
        from hermes_shanghan.agent.harness import HarnessRunner
        st = HarnessRunner().start("惡寒發熱，汗出，脈浮緩，應當用什麼方？",
                                   mode="agent", role="doctor")
        try:
            self.assertEqual(st.status, "paused")
            st2 = HarnessRunner().resume(st.spec.run_id, reject=True,
                                         approver="reviewer-x")
            self.assertEqual(st2.status, "rejected")
            self.assertEqual(st2.release["decision"],
                             "rejected_by_human_review")
            self.assertTrue(any(e["event"] == "human_review_rejected"
                                for e in st2.guardrail_events))
        finally:
            shutil.rmtree(config.RUNS_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
class _BatchToolClient:
    """單輪返回 3 個 tool_calls 的假客戶端（復現預算突破探針）。"""
    backend = "fake-batch"
    available = False

    class _TC:
        def __init__(self, i):
            self.id = f"call_{i}"
            self.name = "shanghan_search"
            self.arguments = {"query": f"桂枝湯{i}"}

    class _Res:
        def __init__(self, tool_calls, content=""):
            self.tool_calls = tool_calls
            self.content = content
            self.backend = "fake-batch"

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools=None):
        self.turn += 1
        if self.turn == 1 and tools:
            return self._Res([self._TC(i) for i in range(3)])
        return self._Res([], content="已依據取得的證據作答。")

    def synthesize(self, question, evidence, role):
        return "（合成回答）"


class TestBudgetEnforcement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_batch_tool_calls_cannot_break_budget(self):
        # 動態探針復現：max_tool_calls=1，模型單輪返回 3 個工具調用
        from hermes_shanghan.agent.agent import ShanghanAgent
        agent = ShanghanAgent(client=_BatchToolClient(), max_tool_calls=1)
        out = agent.ask("桂枝湯的證據？", role="researcher")
        self.assertEqual(len(out["tools_used"]), 1)     # 只執行 1 次
        denied = [s for s in out["agent_trace"]
                  if s["kind"] == "tool_budget_denied"]
        self.assertEqual(len(denied), 2)                # 其餘 2 次顯式拒絕

    def test_harness_budget_at_registry_layer(self):
        # Harness 控制器統一預算：TracedRegistry 原子扣減，超限回
        # BUDGET_EXHAUSTED（子 agent 自己的計數器不再是唯一防線）
        from hermes_shanghan.agent.harness.state import RunBudget
        from hermes_shanghan.agent.harness.tracing import (TracedRegistry,
                                                           TraceStore)
        from hermes_shanghan.agent.tools import get_registry
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            budget = RunBudget(max_tool_calls=2)
            reg = TracedRegistry(get_registry(), TraceStore(td), None,
                                 budget=budget)
            r1 = reg.call("shanghan_search", {"query": "桂枝湯"})
            r2 = reg.for_role("researcher").call("shanghan_search",
                                                 {"query": "麻黃湯"})
            r3 = reg.call("shanghan_search", {"query": "小柴胡湯"})
            self.assertNotIn("error", r1)
            self.assertNotIn("error", r2)      # 預算跨 for_role 副本共享
            self.assertIn("BUDGET_EXHAUSTED", r3.get("error", ""))
            self.assertEqual(budget.snapshot()["used_tool_calls"], 2)
            self.assertEqual(budget.snapshot()["denied_tool_calls"], 1)

    def test_solve_mode_tools_enter_ledger_and_spans(self):
        # 動態探針復現：solve 模式 tool_calls 不得為 0
        from hermes_shanghan.agent.harness import HarnessRunner
        from hermes_shanghan.agent.harness.tracing import TraceStore
        st = HarnessRunner().start(
            "桂枝湯與麻黃湯如何鑒別？劑量比各是多少？", mode="solve",
            role="doctor")
        try:
            self.assertTrue(st.tool_calls, "solve 模式工具調用必須進台賬")
            events = TraceStore(config.RUNS_DIR / st.spec.run_id).read()
            self.assertTrue([e for e in events if e["span_type"] == "tool"])
        finally:
            shutil.rmtree(config.RUNS_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
class TestRolePolicy(unittest.TestCase):
    def test_role_escalation_denied(self):
        from hermes_shanghan.server import policy
        keys = policy.parse_api_keys("tokA:patient:alice,tokB:doctor:bob")
        p = policy.resolve_principal("tokA", keys, "")
        self.assertEqual(p.role_ceiling, "patient")
        with self.assertRaises(policy.PolicyDenied):
            policy.effective_role(p, "doctor")      # 患者 key 不可自稱醫師
        self.assertEqual(policy.effective_role(p, None), "patient")  # 鉗到上限
        doc = policy.resolve_principal("tokB", keys, "")
        self.assertEqual(policy.effective_role(doc, "student"), "student")

    def test_invalid_token_unauthorized(self):
        from hermes_shanghan.server import policy
        keys = policy.parse_api_keys("tokA:patient")
        self.assertIsNone(policy.resolve_principal("wrong", keys, ""))
        self.assertIsNone(policy.resolve_principal("", keys, ""))

    def test_endpoint_capability_matrix(self):
        from hermes_shanghan.server import policy
        p = policy.PrincipalContext("alice", "patient", "api_key")
        self.assertFalse(policy.allow_min_role(p, "student"))
        self.assertTrue(policy.allow_min_role(p, "patient"))

    def test_malformed_key_entries_dropped(self):
        from hermes_shanghan.server import policy
        keys = policy.parse_api_keys("tok:king,:doctor,ok:doctor")
        self.assertEqual(list(keys), ["ok"])      # 非法角色 fail-closed 丟棄

    def test_clinical_routes_declare_min_role(self):
        from hermes_shanghan.server import http_server as hs
        by_path = {rx.pattern: mr for _, rx, _, mr, _ in hs.ROUTES}
        for path in (r"^/api/match$", r"^/api/differential$",
                     r"^/api/formula$", r"^/api/mistreatment$"):
            self.assertEqual(by_path[path], "student", path)
        self.assertEqual(by_path[r"^/api/deep-research$"], "researcher")


# ---------------------------------------------------------------------------
class TestSessionIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        import tempfile
        from pathlib import Path
        cls._tmp = tempfile.TemporaryDirectory()
        cls._old = config.MEMORY_DIR
        config.MEMORY_DIR = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        config.MEMORY_DIR = cls._old
        cls._tmp.cleanup()

    def test_no_session_id_no_shared_default(self):
        from hermes_shanghan.server.service import ServiceContext
        svc = ServiceContext()
        a = svc.chat("桂枝湯的方證要點？", role="doctor")
        b = svc.chat("麻黃湯的方證要點？", role="doctor")
        sa, sb = a["session"]["session_id"], b["session"]["session_id"]
        self.assertNotEqual(sa, sb)          # 兩個匿名請求不共用會話
        self.assertNotEqual(sa, "default")
        # 回傳的 id 可續接：追問看見前一輪錨點
        a2 = svc.chat("它的劑量比呢？", session_id=sa, role="doctor")
        self.assertTrue(a2["session"]["contextualized"])

    def test_namespace_prevents_fixation(self):
        from hermes_shanghan.server.service import ServiceContext
        svc = ServiceContext()
        a = svc.chat("桂枝湯的方證要點？", session_id="shared-id",
                     role="doctor", subject="alice")
        b = svc.chat("它的劑量比呢？", session_id="shared-id",
                     role="doctor", subject="mallory")
        # 同一 session_id、不同主體：mallory 看不到 alice 的上下文
        self.assertFalse(b["session"]["contextualized"])
        self.assertEqual(a["session"]["namespace"], "alice")

    def test_correction_persisted_with_provenance(self):
        from hermes_shanghan.agent.session import AgentSession
        from hermes_shanghan.memory.store import MemoryStore
        s = AgentSession(namespace="alice")
        s.ask("不是桂枝加芍藥湯，而是桂枝去芍藥湯", role="doctor")
        rows = MemoryStore("correction_memory").get("user_corrections", [])
        row = next(r for r in rows if r["wrong"] == "桂枝加芍藥湯")
        self.assertEqual(row["namespace"], "alice")
        self.assertEqual(row["trust"], "unverified_user_correction")


# ---------------------------------------------------------------------------
class TestReadyz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_readyz_green_in_repo(self):
        from hermes_shanghan.health import livez, readyz
        self.assertTrue(livez()["ok"])
        out = readyz()
        self.assertTrue(out["ready"], out)
        names = {c["check"] for c in out["checks"]}
        for need in ("manifest", "clauses", "initial_rules", "formula_rules",
                     "tool_specs"):
            self.assertIn(need, names)

    def test_missing_assets_fail_loud(self):
        # 動態探針復現：資產缺失時不得「假健康」空運行
        import tempfile
        from pathlib import Path
        from hermes_shanghan import health
        saved = {k: getattr(config, k) for k in
                 ("CLAUSE_DIR", "RULES_INITIAL_DIR", "RULES_FORMULA_DIR")}
        with tempfile.TemporaryDirectory() as td:
            try:
                for k in saved:
                    setattr(config, k, Path(td))
                with self.assertRaises(health.MissingAssetsError):
                    health.assert_ready(context="test")
                self.assertFalse(health.readyz()["ready"])
            finally:
                for k, v in saved.items():
                    setattr(config, k, v)


# ---------------------------------------------------------------------------
class TestContractEnforcement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_timeout_enforced(self):
        import time as _t
        from hermes_shanghan.agent.tools import get_registry, Tool
        reg = get_registry()
        slow = Tool("_slow_probe", "test", {"type": "object", "properties": {}},
                    lambda: (_t.sleep(1.0), {"ok": True})[1])
        reg._tools["_slow_probe"] = slow
        import os
        os.environ["HERMES_TOOL_TIMEOUT"] = "0.2"
        try:
            out = reg.call("_slow_probe", {})
            self.assertIn("timeout", out.get("error", ""))
        finally:
            os.environ.pop("HERMES_TOOL_TIMEOUT", None)
            reg._tools.pop("_slow_probe", None)

    def test_output_shape_contract(self):
        from hermes_shanghan.agent.tools import get_registry, Tool
        reg = get_registry()
        bad = Tool("_bad_probe", "test", {"type": "object", "properties": {}},
                   lambda: ["not-a-dict"])
        reg._tools["_bad_probe"] = bad
        try:
            out = reg.call("_bad_probe", {})
            self.assertIn("輸出契約違例", out.get("error", ""))
        finally:
            reg._tools.pop("_bad_probe", None)

    def test_cache_key_versioned_and_audited(self):
        from hermes_shanghan.agent.tools import get_registry
        reg = get_registry()
        reg.call("shanghan_search", {"query": "契約緩存探針"})
        out = reg.call("shanghan_search", {"query": "契約緩存探針"})
        self.assertTrue(out.get("cache_hit"))
        keys = [k for k in reg._cache if "契約緩存探針" in k]
        self.assertTrue(keys)
        self.assertIn(reg._corpus_fp, keys[0])      # 語料指紋入緩存鍵
        tail = reg.audit_tail(5)
        self.assertTrue(tail and tail[-1]["tool"] == "shanghan_search")
        self.assertTrue(tail[-1]["cache_hit"])


# ---------------------------------------------------------------------------
class TestPlannerCompiler(unittest.TestCase):
    def test_compile_rejects_defects(self):
        from hermes_shanghan.agent.planner import compile_plan
        types = {"general": {"tools": []}}
        base = {"subtasks": [
            {"id": "T1", "kind": "general", "question": "a",
             "depends_on": ["T2"]},
            {"id": "T2", "kind": "general", "question": "b",
             "depends_on": ["T1"]}]}
        errs = compile_plan(base, types)
        self.assertTrue(any("環路" in e for e in errs))
        errs = compile_plan({"subtasks": [
            {"id": "T1", "kind": "general", "question": "a",
             "depends_on": ["T9"]}]}, types)
        self.assertTrue(any("懸空依賴" in e for e in errs))
        errs = compile_plan({"subtasks": [
            {"id": "T1", "kind": "general", "question": "a", "depends_on": []},
            {"id": "T1", "kind": "general", "question": "b",
             "depends_on": []}]}, types)
        self.assertTrue(any("重複" in e for e in errs))
        errs = compile_plan({"subtasks": [
            {"id": "T1", "kind": "nope", "question": "a",
             "depends_on": []}]}, types)
        self.assertTrue(any("未知任務類型" in e for e in errs))

    def test_execution_order_raises_on_cycle(self):
        from hermes_shanghan.agent.planner import execution_order
        with self.assertRaises(ValueError):
            execution_order([
                {"id": "T1", "depends_on": ["T2"]},
                {"id": "T2", "depends_on": ["T1"]}])

    def test_local_plans_compile_clean(self):
        _ensure_artifacts()
        from hermes_shanghan.agent.complex_agent import TASK_TYPES
        from hermes_shanghan.agent.planner import Planner, compile_plan
        for q in ("少陰病寒化證與熱化證如何區別？誤治後如何救逆？",
                  "桂枝湯的劑量比？注家對第12條有何分歧？"):
            plan = Planner(task_types=TASK_TYPES, max_subtasks=5)._plan_local(q)
            self.assertEqual(compile_plan(plan, TASK_TYPES, 5), [])

    def test_max_subtasks_respected(self):
        from hermes_shanghan.agent.complex_agent import ComplexAgent
        self.assertEqual(ComplexAgent(max_subtasks=3).max_subtasks, 3)


# ---------------------------------------------------------------------------
class TestResearchCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_failed_tool_does_not_count_as_covered(self):
        from hermes_shanghan.agent.research_loop import DeepResearcher
        d = DeepResearcher(max_rounds=1)
        state = {"findings": [
            {"dimension": "原文源流", "module": "shanghan_search",
             "error": "tool failed", "summary": "（無數據）"},
            {"dimension": "異文注家", "module": "shanghan_variants",
             "error": None, "summary": "…", "_result_ids": []}]}
        gaps = d._gaps(state)
        self.assertIn("原文源流", gaps)       # FAILED 不算覆蓋
        self.assertIn("異文注家", gaps)       # EMPTY（無證據）不算覆蓋

    def test_full_run_statuses_and_no_silent_empty(self):
        from hermes_shanghan.agent.research_loop import (DeepResearcher,
                                                         FINDING_STATUSES)
        d = DeepResearcher(max_rounds=3).run("桂枝湯的源流")
        for f in d["findings"]:
            self.assertIn(f["status"], FINDING_STATUSES)
            if not f["has_citation"]:
                self.assertFalse(f["citation_ok"])   # 無引用不再默認 ok
        self.assertIn("coverage_status", d)


if __name__ == "__main__":
    unittest.main()
