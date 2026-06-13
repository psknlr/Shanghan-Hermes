"""Web console: ServiceContext API, multi-agent council, and a live HTTP smoke
test (all offline, local backend)."""
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from hermes_shanghan import config
from hermes_shanghan.server.service import ServiceContext


def _ensure():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure()
        cls.svc = ServiceContext()

    def test_stats(self):
        s = self.svc.stats()
        self.assertEqual(s["canonical"], 398)
        self.assertGreater(s["skills"], 100)

    def test_search_and_clause(self):
        r = self.svc.search("往來寒熱 胸脅苦滿", top_k=5)
        self.assertTrue(r["hits"])
        self.assertTrue(all(h["clause_id"] for h in r["hits"]))
        c = self.svc.explain_clause(12)
        self.assertEqual(c["clause_id"], "SHL_SONGBEN_0012")
        self.assertTrue(c["formula_blocks"])
        self.assertTrue(c["variants"])  # 桂本/千金翼 alignment

    def test_match(self):
        r = self.svc.match(["惡寒", "發熱", "無汗", "身疼痛"], ["浮緊"])
        self.assertEqual(r["matched_formula_patterns"][0]["formula"], "麻黃湯")

    def test_differential(self):
        r = self.svc.differential(["桂枝湯", "麻黃湯"])
        self.assertEqual(r["differential"]["formulas"], ["桂枝湯", "麻黃湯"])

    def test_teach(self):
        r = self.svc.teach("少陰病")
        self.assertEqual(r["mode"], "student")
        self.assertTrue(r["lesson"]["七、練習題"])

    def test_paper(self):
        r = self.svc.paper("mistreatment", "誤治路徑")
        self.assertIn("摘要", r["manuscript"])
        self.assertTrue(r["meta"]["figures"])


class TestCouncil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure()
        cls.svc = ServiceContext()

    def test_council_pipeline(self):
        out = self.svc.council("病人往來寒熱、胸脅苦滿、口苦，考慮什麼方？", role="doctor")
        agents = [m["agent"] for m in out["council"]]
        self.assertEqual(agents[0], "Planner")
        self.assertIn("Retriever", agents)
        self.assertIn("Critic", agents)
        self.assertEqual(agents[-1], "Synthesizer")
        self.assertTrue(out["evidence_clause_ids"])
        self.assertTrue(out["citation_report"]["ok"])

    def test_council_differential(self):
        out = self.svc.council("桂枝湯和麻黃湯怎麼鑒別？", role="doctor")
        self.assertIn("DifferentialAnalyst", out["specialists"])

    def test_council_patient_guard(self):
        out = self.svc.council("我是不是得了少阴病？给我开方", role="patient")
        self.assertTrue(out.get("refused"))


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure()
        from hermes_shanghan.server.http_server import make_handler
        svc = ServiceContext()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return json.loads(r.read())

    def _post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def test_health_and_static(self):
        self.assertTrue(self._get("/api/health")["ok"])
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=10) as r:
            html = r.read().decode("utf-8")
        self.assertIn("傷寒論", html)
        self.assertIn("/app.js", html)

    def test_static_app_js(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/app.js", timeout=10) as r:
            self.assertEqual(r.headers.get_content_type(), "text/javascript")
            self.assertIn(b"renderBotAnswer", r.read())

    def test_api_council_over_http(self):
        out = self._post("/api/council", {"question": "太陽中風用什麼方？", "role": "doctor"})
        self.assertTrue(out["citation_report"]["ok"] or out["evidence_clause_ids"])

    def test_path_traversal_blocked(self):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/../config.py", timeout=10)
            blocked = False
        except urllib.error.HTTPError as e:
            blocked = e.code == 404
        self.assertTrue(blocked)


if __name__ == "__main__":
    unittest.main()
