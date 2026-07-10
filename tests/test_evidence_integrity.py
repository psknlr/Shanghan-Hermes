"""十一輪對抗回歸：鏡像評審探針——

  P0-1 模型輸出不能自我登記為證據（零檢索猜中真實編號≠通過 strict_round）
  P0-2 患者身份貫穿路由（GET clause 不再回退 student；序列化出口投影）
  P0-4 intake 紅旗形成真正圖分支（不依賴業務引擎自行攔截）
  台賬 Broker 強不變量（tool_call_id/span_id/source_hash/語料指紋）
  結構化臨床動作 / 引文歸屬綁定 / 無引用不放行 / 參數深校驗 / 版本單源
"""
import json
import shutil
import threading
import unittest
import urllib.error
import urllib.request

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class _NoToolClient:
    """零工具調用、直接輸出真實條文編號的假後端（P0-1 探針復現）。"""
    backend = "fake-no-tool"
    available = False

    class _Res:
        def __init__(self):
            self.tool_calls = []
            self.content = "結論見 SHL_SONGBEN_0012。"
            self.backend = "fake-no-tool"

    def chat(self, messages, tools=None):
        return self._Res()

    def synthesize(self, q, e, r):
        return "（合成）"

    def complete(self, *a, **k):
        return ""


# ---------------------------------------------------------------------------
class TestEvidenceSelfRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_zero_tool_guess_fails_strict_round(self):
        # 探針復現：tool_calls=0 + 猜中真實編號 → 引用核驗必須不通過
        from hermes_shanghan.agent.agent import ShanghanAgent
        out = ShanghanAgent(client=_NoToolClient()).ask("桂枝湯？",
                                                        role="researcher")
        rep = out["citation_report"]
        self.assertEqual(out["tools_used"], [])
        self.assertFalse(rep["ok"])
        self.assertIn("SHL_SONGBEN_0012", rep["outside_evidence"])

    def test_harness_zero_tool_pauses_with_empty_ledger(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        st = HarnessRunner(client=_NoToolClient()).start(
            "桂枝湯？", mode="agent", role="researcher")
        try:
            # 台賬只能由 Broker 寫入：零工具調用 → 台賬空
            self.assertEqual(
                sum(len(v) for v in st.evidence_ledger.values()), 0)
            self.assertNotIn(st.release.get("decision"),
                             ("pass", "pass_with_warning"))
            self.assertEqual(st.status, "paused")   # fail-closed，交人工
        finally:
            shutil.rmtree(config.RUNS_DIR, ignore_errors=True)

    def test_ledger_records_carry_broker_invariants(self):
        from hermes_shanghan.agent.harness import HarnessRunner
        from hermes_shanghan.agent.harness.runner import _ledger_ids_verified
        st = HarnessRunner().start("惡寒發熱，汗出，脈浮緩，應當用什麼方？",
                                   mode="agent", role="doctor")
        try:
            recs = [r for v in st.evidence_ledger.values() for r in v]
            self.assertTrue(recs)
            for r in recs:
                self.assertTrue(r["tool_call_id"])       # 綁定工具調用
                self.assertTrue(r["span_id"])            # 綁定 span
                self.assertTrue(r["source_hash"])        # 綁定原文哈希
                self.assertEqual(r["corpus_fingerprint"],
                                 st.spec.corpus_version)
                self.assertEqual(r["registered_by"], "capability_broker")
            self.assertTrue(_ledger_ids_verified(st))    # 完整性校驗通過
        finally:
            shutil.rmtree(config.RUNS_DIR, ignore_errors=True)

    def test_ledger_integrity_violation_raises(self):
        from hermes_shanghan.agent.harness.runner import _ledger_ids_verified
        from hermes_shanghan.agent.harness.state import RunSpec, RunState
        st = RunState(spec=RunSpec(run_id="t", user_query="q",
                                   corpus_version="abc"))
        st.evidence_ledger = {"execute": [
            {"clause_id": "SHL_SONGBEN_0012"}]}   # 手寫記錄=偽證據
        with self.assertRaises(RuntimeError):
            _ledger_ids_verified(st)

    def test_intake_red_flag_is_a_graph_branch(self):
        # P0-4：紅旗由控制器分支，execute 不再運行
        from hermes_shanghan.agent.harness import HarnessRunner
        st = HarnessRunner().start("突然胸痛徹背，大汗不止，怎麼辦？",
                                   mode="agent", role="patient")
        try:
            self.assertEqual(st.nodes["execute"].status, "skipped_by_triage")
            self.assertEqual(st.nodes["evidence_audit"].status,
                             "skipped_by_triage")
            self.assertTrue(st.final_answer)
            self.assertEqual(st.status, "completed")
            self.assertEqual(st.release["decision"], "pass")   # 攔截=安全結論
            self.assertTrue(any(e["event"] == "red_flag_triage"
                                for e in st.guardrail_events))
        finally:
            shutil.rmtree(config.RUNS_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
class TestPatientRoleEndToEnd(unittest.TestCase):
    """P0-2：患者 API key 全鏈路——GET clause 不回退 student，
    序列化出口強制投影。"""

    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.server import http_server as hs, policy
        from hermes_shanghan.server.service import ServiceContext
        from http.server import ThreadingHTTPServer
        cls.hs = hs
        cls._saved_keys = hs.API_KEYS
        hs.API_KEYS = policy.parse_api_keys("pkey:patient:pat1,dkey:doctor:dr1")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                        hs.make_handler(ServiceContext()))
        cls.port = cls.httpd.server_address[1]
        cls.th = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.th.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.th.join()
        cls.hs.API_KEYS = cls._saved_keys

    def _get(self, path, token):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_patient_key_gets_no_formula_structures(self):
        # 探針復現：患者 key、不帶 role 參數 → 不得拿到 student 默認視圖
        # （字段級契約：處方結構鍵一律被序列化出口投影移除）
        out = self._get("/api/clause/12", "pkey")
        # 業務層按患者身份治理（不再是 student 默認視圖）……
        self.assertEqual(out.get("mode"), "patient")
        # ……且處方結構鍵不出患者面（序列化出口投影兜底；業務層已移除時
        # 出口無事可做——雙保險，見 test_request_context_projection）
        self.assertNotIn("formula_blocks", out)

    def test_doctor_key_unaffected(self):
        out = self._get("/api/clause/12", "dkey")
        self.assertIn("formula_blocks", out)

    def test_patient_key_cannot_claim_doctor_via_query(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._get("/api/clause/12?role=doctor", "pkey")
        cm.exception.close()
        self.assertEqual(cm.exception.code, 403)

    def test_invalid_json_returns_400(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/search",
            data=b"{not json", method="POST",
            headers={"Authorization": "Bearer dkey",
                     "Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        cm.exception.close()
        self.assertEqual(cm.exception.code, 400)


# ---------------------------------------------------------------------------
class TestGatePolicyUpgrades(unittest.TestCase):
    def _spec(self, role="researcher"):
        from hermes_shanghan.agent.harness.state import RunSpec
        return RunSpec(run_id="t", user_query="q", role=role)

    def test_clinical_actions_structured_detection(self):
        from hermes_shanghan.agent.harness.release_gate import clinical_actions
        acts = clinical_actions("桂枝湯可考慮，每日三次，溫服，可加減。")
        types = {a["action_type"] for a in acts}
        # 舊四關鍵詞（主之/劑量/服用/處方）一個都沒出現，仍須全部檢出
        self.assertIn("medication_recommendation", types)
        self.assertIn("dosing_instruction", types)
        self.assertIn("administration_instruction", types)
        self.assertIn("modification_plan", types)
        self.assertEqual(clinical_actions("太陽病的提綱是什麼。"), [])

    def test_patient_blocked_without_old_keywords(self):
        from hermes_shanghan.agent.harness.release_gate import evaluate
        out = evaluate(self._spec("patient"),
                       {"answer": "可考慮桂枝湯，每日三次溫服。",
                        "citation_report": {"ok": True,
                                            "has_any_citation": True}})
        self.assertEqual(out["decision"], "blocked")

    def test_no_citation_no_release(self):
        # strict_round：無引用的非拒答回答不再 pass_with_warning
        from hermes_shanghan.agent.harness.release_gate import evaluate
        out = evaluate(self._spec(), {"answer": "太陽病以脈浮為要。",
                                      "citation_report": {
                                          "ok": True,
                                          "has_any_citation": False}})
        self.assertEqual(out["decision"], "review_required")
        self.assertIn("citation_failure", out["review_required"])

    def test_quote_attribution_binding(self):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        from hermes_shanghan.agent.citation_guard import CitationGuard
        store = get_registry().art.clause_store()
        t16 = store["SHL_SONGBEN_0016"].clean_text[:12]
        # 第16條的文字被錯掛到第1條的引用標記旁
        answer = (f"SHL_SONGBEN_0001 曰「{t16}」；另見 SHL_SONGBEN_0016。")
        rep = CitationGuard(store).check(
            answer, allowed_ids=["SHL_SONGBEN_0001", "SHL_SONGBEN_0016"])
        self.assertTrue(rep.attribution_warnings)
        w = rep.attribution_warnings[0]
        self.assertEqual(w["bound_to"], "SHL_SONGBEN_0001")
        self.assertIn("SHL_SONGBEN_0016", w["actually_in"])

    def test_deep_arg_validation_and_no_banana_bool(self):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry, Tool
        reg = get_registry()
        probe = Tool("_schema_probe", "t",
                     {"type": "object", "properties": {
                         "mode": {"type": "string", "enum": ["a", "b"]},
                         "k": {"type": "integer", "minimum": 1, "maximum": 5},
                         "flag": {"type": "boolean"},
                         "items": {"type": "array",
                                   "items": {"type": "string"},
                                   "maxItems": 2}},
                      "required": []},
                     lambda **kw: {"ok": True, **kw})
        reg._tools["_schema_probe"] = probe
        try:
            self.assertIn("枚舉", reg.call("_schema_probe",
                                           {"mode": "c"})["error"])
            self.assertIn("上限", reg.call("_schema_probe",
                                           {"k": 9})["error"])
            # "banana" 不再被靜默修成 False——顯式類型錯誤
            self.assertIn("應為 boolean",
                          reg.call("_schema_probe",
                                   {"flag": "banana"})["error"])
            self.assertIn("元素數", reg.call("_schema_probe",
                                             {"items": ["a", "b", "c"]})["error"])
            ok = reg.call("_schema_probe", {"mode": "a", "k": 3,
                                            "flag": "true"})
            self.assertNotIn("error", ok)
        finally:
            reg._tools.pop("_schema_probe", None)

    def test_single_version_source(self):
        from hermes_shanghan._version import __version__
        from hermes_shanghan.integrations.mcp_server import SERVER_INFO
        self.assertEqual(SERVER_INFO["version"], __version__)

    def test_request_context_projection(self):
        from hermes_shanghan.server.policy import (RequestContext,
                                                   project_for_role)
        ctx = RequestContext("p1", "local", "patient", "patient", "r1")
        self.assertEqual(ctx.role_or("student"), "patient")   # 不回退 student
        out = project_for_role({"a": 1, "formula_blocks": [1],
                                "nested": {"composition": ["桂枝"]}},
                               "patient")
        self.assertNotIn("formula_blocks", out)
        self.assertNotIn("composition", out["nested"])
        self.assertIn("_role_projection", out)


if __name__ == "__main__":
    unittest.main()
