"""Harness 測試：狀態圖執行 / checkpoint·resume·replay / span 軌跡 /
發布閘門（人工審核）/ 工具契約 / MCP resources·prompts / 軌跡評測 /
擾動注入 / API 治理。"""
import json
import shutil
import unittest

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestHarnessRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.harness import HarnessRunner
        cls.runner = HarnessRunner()
        # 症狀選方問題：local 路由到 shanghan_match_formula → 多假設層
        # 附着 → **結構化**候選方信號觸發人工審核（不再靠「湯」字關鍵詞）
        cls.st = cls.runner.start("惡寒發熱，汗出，脈浮緩，應當用什麼方？",
                                  mode="agent", role="doctor")
        cls.run_id = cls.st.spec.run_id

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(config.RUNS_DIR, ignore_errors=True)

    def test_nodes_all_executed_with_checkpoint(self):
        self.assertEqual({k: v.status for k, v in self.st.nodes.items()},
                         {"intake": "ok", "execute": "ok",
                          "evidence_audit": "ok", "release_gate": "ok"})
        state_file = config.RUNS_DIR / self.run_id / "state.json"
        self.assertTrue(state_file.exists())
        d = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertEqual(d["spec"]["evidence_policy"], "strict_round")
        self.assertTrue(d["spec"]["corpus_version"])   # 語料版本指紋在案
        self.assertTrue(d["spec"]["python_version"])   # 環境指紋（replay 前提）
        self.assertTrue(d["trace_id"])                 # trace 跨 resume 延續

    def test_doctor_formula_triggers_human_review(self):
        # 評審第 8 條：醫師端候選方（結構化信號）→ 人工審核 → paused
        self.assertEqual(self.st.status, "paused")
        self.assertIn("doctor_formula_candidates", self.st.pending_review)
        self.assertEqual(self.st.release["decision"], "review_required")
        # ApprovalRequest 結構化在案（審什麼、證據指紋、時間）
        req = next(a for a in self.st.approval_requests
                   if a["trigger"] == "doctor_formula_candidates")
        for key in ("approval_id", "run_id", "action_digest",
                    "evidence_digest", "requested_at", "status"):
            self.assertIn(key, req)
        self.assertEqual(req["status"], "pending")

    def test_resume_with_approval_completes(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        st2 = HarnessRunner().resume(self.run_id, approve=True,
                                     approver="unit-test")
        self.assertEqual(st2.status, "completed")
        # 批准≠改狀態：evidence_audit/release_gate 帶 approved 集合重新執行
        self.assertEqual(st2.release["decision"], "pass_after_human_review")
        approvals = [e for e in st2.guardrail_events
                     if e["event"] == "human_review_approved"]
        self.assertEqual(approvals[0]["approver"], "unit-test")  # 審批人在案
        self.assertTrue(all(a["status"] == "approved"
                            for a in st2.approval_requests))

    def test_spans_schema_and_tool_span(self):
        from hermes_shanghan.agent.harness.tracing import TraceStore
        events = TraceStore(config.RUNS_DIR / self.run_id).read()
        self.assertTrue(events)
        for sp in events:
            for key in ("trace_id", "span_id", "span_type", "started_at",
                        "duration_ms", "input_hash", "output_hash",
                        "error", "evidence_ids", "metadata"):
                self.assertIn(key, sp)
        tool_spans = [s for s in events if s["span_type"] == "tool"]
        self.assertTrue(tool_spans)
        self.assertTrue(all(s["parent_span_id"] for s in tool_spans))
        # 工具 span 攜帶證據 clause_id
        self.assertTrue(any(s["evidence_ids"] for s in tool_spans))

    def test_evidence_ledger_and_tool_calls_recorded(self):
        self.assertTrue(self.st.evidence_ledger.get("execute"))
        self.assertIn("shanghan_match_formula",
                      [t["tool"] for t in self.st.tool_calls])
        self.assertTrue(self.st.budget_snapshot)   # 預算快照在案

    def test_replay_deterministic_local(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        out = HarnessRunner().replay(self.run_id)
        self.assertTrue(out["deterministic_match"])
        # 環境指紋對比：同倉庫同語料同後端 → comparable
        self.assertEqual(out["fingerprint_mismatches"], {})
        self.assertIn("comparable", out)

    def test_export_md_and_json(self):
        from hermes_shanghan.agent.harness.runner import export_run
        md = export_run(self.run_id, "md")
        self.assertIn("## 節點軌跡", md)
        self.assertIn("## 證據台賬", md)
        j = json.loads(export_run(self.run_id, "json"))
        self.assertIn("events", j)

    def test_patient_refusal_flow(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        st = HarnessRunner().start("给我开个方治感冒", mode="agent",
                                   role="patient")
        # 拒答本身是安全結論，可直接完成發布（guardrail 事件在案）
        self.assertIn(st.status, ("completed", "paused"))
        self.assertTrue(any(e["event"] == "intent_guard_refused"
                            for e in st.guardrail_events)
                        or st.node_outputs["execute"].get("refused"))


class TestToolContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_contract_fields(self):
        cs = self.reg.contracts()
        self.assertEqual(len(cs), len(self.reg.names()))
        for c in cs:
            for key in ("version", "permission_level", "evidence_level",
                        "side_effect", "timeout_s", "cacheable", "idempotent",
                        "max_result_bytes", "schema_hash", "error_schema",
                        "enforced"):
                self.assertIn(key, c)
            self.assertEqual(c["side_effect"], "read")   # 只讀不變式
            # 契約條款必須聲明執行方式（runtime / by-construction），
            # 不得只是願望清單（九輪 P0-8）
            for clause in ("timeout_s", "max_result_bytes", "args_schema",
                           "cacheable", "side_effect_read"):
                self.assertIn(clause, c["enforced"])
        intake = next(c for c in cs if c["name"] == "shanghan_intake")
        self.assertEqual(intake["permission_level"], "patient_safe")

    def test_committed_specs_carry_contracts(self):
        spec = json.loads((config.SHANGHAN_DIR / "tool_specs.json")
                          .read_text(encoding="utf-8"))
        self.assertIn("contracts", spec)
        self.assertEqual(len(spec["contracts"]), len(self.reg.names()))
        self.assertIn("tool_spec_version", spec)


class TestMCPExtensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def _call(self, method, params=None, id_=1):
        from hermes_shanghan.integrations.mcp_server import handle
        return handle({"jsonrpc": "2.0", "id": id_, "method": method,
                       "params": params or {}})

    def test_initialize_negotiation(self):
        r = self._call("initialize", {"protocolVersion": "2025-06-18"})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
        caps = r["result"]["capabilities"]
        for cap in ("tools", "resources", "prompts"):
            self.assertIn(cap, caps)
        # 未知版本回退基線
        r2 = self._call("initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(r2["result"]["protocolVersion"], "2024-11-05")

    def test_resources_list_and_read(self):
        r = self._call("resources/list")
        uris = {x["uri"] for x in r["result"]["resources"]}
        self.assertIn("shanghan://clauses", uris)
        self.assertIn("shanghan://trace/claims", uris)
        rd = self._call("resources/read", {"uri": "shanghan://trace/claims"})
        text = rd["result"]["contents"][0]["text"]
        self.assertIn("CLAIM_GZT_YINGWEI", text)
        err = self._call("resources/read", {"uri": "shanghan://nope"})
        self.assertIn("error", err)

    def test_prompts_list_and_get(self):
        r = self._call("prompts/list")
        names = {p["name"] for p in r["result"]["prompts"]}
        self.assertIn("misquote-review", names)
        g = self._call("prompts/get", {"name": "misquote-review",
                                       "arguments": {"quote": "營衛不和"}})
        msg = g["result"]["messages"][0]["content"]["text"]
        self.assertIn("營衛不和", msg)

    def test_tasks_submit_status_result_cancel(self):
        # 實驗性長任務：submit → running/completed → result；cancel 協作式
        import time
        r = self._call("initialize", {"protocolVersion": "2025-06-18"})
        self.assertIn("tasks", r["result"]["capabilities"]["experimental"])
        sub = self._call("tasks/submit", {"name": "shanghan_search",
                                          "arguments": {"query": "桂枝湯"}})
        tid = sub["result"]["task_id"]
        for _ in range(100):
            st = self._call("tasks/status", {"task_id": tid})["result"]
            if st["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(st["status"], "completed")
        res = self._call("tasks/result", {"task_id": tid})["result"]
        self.assertIn("content", res)
        lst = self._call("tasks/list")["result"]
        self.assertIn(tid, [t["task_id"] for t in lst["tasks"]])
        # 已終態任務 cancel 無效果（如實返回）
        c = self._call("tasks/cancel", {"task_id": tid})["result"]
        self.assertFalse(c["cancelled"])
        err = self._call("tasks/status", {"task_id": "nope"})
        self.assertIn("error", err)


class TestTrajectoryEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_trajectory_metrics(self):
        from hermes_shanghan.eval.trajectory import trajectory_eval
        m = trajectory_eval()
        self.assertEqual(m["tool_name_accuracy"], 1.0)
        self.assertEqual(m["trajectory_validity_rate"], 1.0)
        self.assertEqual(m["refusal_precision"], 1.0)

    def test_fault_injection_recovery(self):
        from hermes_shanghan.eval.trajectory import perturbation_eval
        m = perturbation_eval()
        self.assertEqual(m["recovery_success_rate"], 1.0)
        self.assertTrue(all(s["injected"] >= 1 or s["fault"] == "empty"
                            for s in m["scenarios"]))


class TestApiGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.server.service import ServiceContext
        cls.svc = ServiceContext()

    def test_tool_call_role_scoped(self):
        # patient 角色經 ScopedRegistry 硬裁剪：劑量類工具不可達
        out = self.svc.tool_call("shanghan_dose", {"formula": "桂枝湯"},
                                 role="patient")
        self.assertIn("error", out)
        ok = self.svc.tool_call("shanghan_intake", {"text": "怕冷發熱"},
                                role="patient")
        self.assertNotIn("error", ok)

    def test_source_registry(self):
        from hermes_shanghan.corpus.source_registry import sources
        srcs = {s["source_id"]: s for s in sources()}
        self.assertIn("corpus_raw_shanghan", srcs)
        self.assertIn("jicheng_20180111", srcs)
        self.assertIn("P（旁證層", srcs["jicheng_20180111"]["evidence_layers"])
        self.assertEqual(srcs["jicheng_20180111"]["sha256"],
                         config.LIBRARY_SHA256)


if __name__ == "__main__":
    unittest.main()
