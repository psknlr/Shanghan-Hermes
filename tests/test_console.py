"""十二輪測試：運行中心 API（whoami/runs 生命週期/審批/導出）、
Artifact 防穿越、評測端點、雙 UI 在位、會話結構化指代解析、
Colab notebook P0 守衛（固定版本/ensure_server/無硬編碼統計/冪等克隆）。"""
import json
import shutil
import time
import unittest

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestRunCenterApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.server.service import ServiceContext
        cls.svc = ServiceContext()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(config.RUNS_DIR, ignore_errors=True)

    def test_run_lifecycle_via_api(self):
        # 異步啟動 → 輪詢 → paused（審批請求在案）→ approve → completed
        out = self.svc.run_start("惡寒發熱，汗出，脈浮緩，應當用什麼方？",
                                 mode="agent", role="doctor")
        rid = out["run_id"]
        self.assertTrue(rid.startswith("run_"))
        detail = {}
        for _ in range(120):
            detail = self.svc.run_detail(rid)
            if detail.get("status") in ("paused", "completed", "failed",
                                        "blocked"):
                break
            time.sleep(0.25)
        self.assertEqual(detail["status"], "paused")
        self.assertTrue(detail["approval_requests"])
        self.assertTrue(detail["spans"])            # span 軌跡隨詳情返回
        # 台賬記錄帶 Broker 綁定字段
        recs = [r for v in detail["evidence_ledger"].values() for r in v]
        self.assertTrue(all(r["registered_by"] == "capability_broker"
                            for r in recs))
        # 任務列表可見
        self.assertIn(rid, [r["run_id"] for r in
                            self.svc.runs_list()["runs"]])
        # 審批 → 重跑下游閘門 → completed
        acted = self.svc.run_action(rid, "approve", approver="console-test")
        self.assertEqual(acted["status"], "completed")
        # 導出
        md = self.svc.run_action(rid, "export")
        self.assertIn("節點軌跡", md["markdown"])
        # 未知動作明確報錯
        self.assertIn("error", self.svc.run_action(rid, "detonate"))

    def test_run_start_rejects_empty_query(self):
        self.assertIn("error", self.svc.run_start(""))

    def test_eval_endpoints(self):
        m = self.svc.eval_trajectory()
        self.assertEqual(m["trajectory_validity_rate"], 1.0)
        p = self.svc.eval_perturbation()
        self.assertEqual(p["recovery_success_rate"], 1.0)

    def test_artifact_traversal_denied(self):
        self.assertIn("error", self.svc.artifact_read("../../../etc/passwd"))
        self.assertIn("error", self.svc.artifact_read("clauses/clauses.jsonl"))
        self.assertIn("error", self.svc.artifact_read(""))

    def test_governance_panel(self):
        g = self.svc.governance()
        self.assertTrue(g["version"])
        self.assertIn("patient", g["role_release_policy"])
        self.assertIn("readyz", g)

    def test_whoami_route_registered_and_dual_ui_served(self):
        from hermes_shanghan.server import http_server as hs
        patterns = {rx.pattern for _, rx, _, _, _ in hs.ROUTES}
        for need in (r"^/api/whoami$", r"^/api/runs$",
                     r"^/api/runs/([A-Za-z0-9_\-]+)$", r"^/api/artifacts$",
                     r"^/api/governance$", r"^/api/eval/trajectory$"):
            self.assertIn(need, patterns)
        static = hs.STATIC_DIR
        self.assertTrue((static / "index.html").exists())     # 舊 UI 保留
        self.assertTrue((static / "console.html").exists())   # 新運行中心
        idx = (static / "index.html").read_text(encoding="utf-8")
        self.assertIn("console.html", idx)                    # 互通入口
        con = (static / "console.html").read_text(encoding="utf-8")
        for section in ("運行中心", "會話", "評測", "Artifact", "治理"):
            self.assertIn(section, con)
        self.assertIn("Authorization", con)                   # 認證頭在場
        appjs = (static / "app.js").read_text(encoding="utf-8")
        self.assertIn("authHeaders", appjs)                   # 舊 UI 修 401


class TestSessionReferenceResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_structured_resolution(self):
        from hermes_shanghan.agent.session import AgentSession
        s = AgentSession()
        out1 = s.ask("桂枝湯的方證要點？", role="doctor")
        self.assertEqual(out1["session"]["reference_resolution"]["status"],
                         "no_reference")
        out2 = s.ask("它的劑量比呢？", role="doctor")
        rr = out2["session"]["reference_resolution"]
        self.assertEqual(rr["status"], "resolved")
        self.assertEqual(rr["resolved"], "桂枝湯")
        self.assertIn("mention", rr)
        self.assertGreater(rr["confidence"], 0.5)
        self.assertIn("basis", rr)      # 解析依據可審，不是裸 True

    def test_unresolved_reported_honestly(self):
        from hermes_shanghan.agent.session import AgentSession
        s = AgentSession()
        s.ask("陽明病的提綱是什麼？", role="student")   # 無方名錨點
        rr = s.ask("它呢？", role="student")["session"]["reference_resolution"]
        self.assertIn(rr["status"], ("unresolved", "resolved"))
        if rr["status"] == "unresolved":
            self.assertEqual(rr["confidence"], 0.0)


class TestNotebookP0Guards(unittest.TestCase):
    """Colab P0 守衛：固定版本、統一服務生命週期、冪等克隆、零硬編碼統計。"""

    @classmethod
    def setUpClass(cls):
        p = config.REPO_ROOT / "notebooks" / "Hermes_Shanghanlun_Colab.ipynb"
        cls.nb = json.loads(p.read_text(encoding="utf-8"))
        cls.blob = "".join("".join(c["source"]) for c in cls.nb["cells"])

    def test_branch_pinned_to_main_with_version_echo(self):
        self.assertIn("BRANCH = 'main'", self.blob)
        self.assertNotIn("claude/project-review", self.blob)   # 死分支已除
        for echo in ("Commit ID", "Data Version", "COMMIT_SHA"):
            self.assertIn(echo, self.blob)

    def test_ensure_server_single_lifecycle(self):
        self.assertIn("def ensure_server", self.blob)
        self.assertIn("find_free_port", self.blob)
        # ngrok 節不再自起第二個服務進程
        self.assertNotIn("'serve',\n                           '--port'",
                         self.blob)
        self.assertEqual(self.blob.count("def ensure_server"), 1)

    def test_idempotent_clone(self):
        # 已在倉庫內（pyproject 在場）不再克隆——嵌套克隆根除
        self.assertIn("pyproject.toml", self.blob)

    def test_no_hardcoded_stats(self):
        for stale in ("138 項測試", "19 個工具", "11 模塊", "11 個模塊",
                      "253", "297 項"):
            self.assertNotIn(stale, self.blob)
        self.assertIn("test_docs_sync", self.blob)   # 指向守衛而非寫死數字


if __name__ == "__main__":
    unittest.main()
