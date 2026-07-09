"""Stdlib HTTP server for the Hermes-Shanghanlun web console.

No third-party dependencies. Serves a single-page app from ./static and a
JSON API backed by ServiceContext. Concurrency via ThreadingHTTPServer.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .service import ServiceContext, get_service

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 256 * 1024        # JSON request bodies are tiny; cap hard
# optional bearer-token auth for non-localhost deployments:
#   HERMES_SERVER_TOKEN=... python3 -m hermes_shanghan serve --host 0.0.0.0
AUTH_TOKEN = os.environ.get("HERMES_SERVER_TOKEN", "")


def _json_body(handler: BaseHTTPRequestHandler) -> Dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length <= 0:
        return {}
    if length > MAX_BODY_BYTES:
        raise ValueError("body_too_large")
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


# route table: (method, regex) -> handler(service, body, match, query) -> dict
ROUTES: list = []


def route(method: str, pattern: str):
    rx = re.compile(f"^{pattern}$")

    def deco(fn):
        ROUTES.append((method, rx, fn))
        return fn
    return deco


# --------------------------------------------------------------------------
@route("GET", r"/api/health")
def _health(svc, body, m, q):
    return {"ok": True, "ready": svc.ready(), "backend": svc.llm.backend}


@route("GET", r"/api/stats")
def _stats(svc, body, m, q):
    return svc.stats()


@route("GET", r"/api/llm/status")
def _llm_status(svc, body, m, q):
    return svc.llm_status()


@route("GET", r"/api/formulas")
def _formulas(svc, body, m, q):
    return svc.list_formulas()


@route("GET", r"/api/channels")
def _channels(svc, body, m, q):
    return svc.channels()


@route("GET", r"/api/skills")
def _skills(svc, body, m, q):
    return svc.skills()


@route("POST", r"/api/search")
def _search(svc, body, m, q):
    return svc.search(body.get("query", ""), top_k=int(body.get("top_k", 8)),
                      six_channel=body.get("six_channel"), formula=body.get("formula"),
                      field=body.get("field"), expand=bool(body.get("expand")))


@route("GET", r"/api/clause/([^/]+)")
def _clause(svc, body, m, q):
    return svc.explain_clause(m.group(1), role=(q.get("role", ["student"])[0]))


@route("POST", r"/api/explain")
def _explain(svc, body, m, q):
    return svc.explain_clause(body.get("ref"), role=body.get("role", "student"))


@route("POST", r"/api/match")
def _match(svc, body, m, q):
    return svc.match(body.get("symptoms", []), pulse=body.get("pulse", []),
                     six_channel=body.get("six_channel"), top_k=int(body.get("top_k", 5)))


@route("POST", r"/api/differential")
def _diff(svc, body, m, q):
    return svc.differential(body.get("formulas", []))


@route("POST", r"/api/teach")
def _teach(svc, body, m, q):
    return svc.teach(body.get("channel", "太陽病"))


@route("POST", r"/api/mistreatment")
def _mistreat(svc, body, m, q):
    return svc.mistreatment(body.get("query"))


@route("POST", r"/api/formula")
def _formula(svc, body, m, q):
    return svc.formula_rule(body.get("formula", ""))


@route("POST", r"/api/research")
def _research(svc, body, m, q):
    return svc.research(body.get("topic", ""), outputs=body.get("outputs"))


@route("POST", r"/api/paper")
def _paper(svc, body, m, q):
    return svc.paper(body.get("type", "formula_pattern"), topic=body.get("topic", ""),
                     use_llm=body.get("use_llm", True))


@route("POST", r"/api/complex")
def _complex(svc, body, m, q):
    return svc.complex(body.get("question", ""), role=body.get("role"))


@route("POST", r"/api/chat")
def _chat(svc, body, m, q):
    return svc.chat(body.get("question", ""),
                    session_id=str(body.get("session_id", "default")),
                    role=body.get("role"))


@route("POST", r"/api/deep-research")
def _deep_research(svc, body, m, q):
    return svc.deep_research(body.get("topic", ""),
                             rounds=int(body.get("rounds", 3)))


@route("POST", r"/api/patient")
def _patient(svc, body, m, q):
    return svc.patient(body.get("question", ""))


@route("POST", r"/api/agent")
def _agent(svc, body, m, q):
    return svc.agent(body.get("question", ""), role=body.get("role"),
                     max_steps=int(body.get("max_steps", 5)))


@route("POST", r"/api/council")
def _council(svc, body, m, q):
    return svc.council(body.get("question", ""), role=body.get("role"))


@route("POST", r"/api/tool")
def _tool(svc, body, m, q):
    return svc.tool_call(body.get("name", ""), body.get("arguments", {}))


@route("POST", r"/api/trace")
def _trace(svc, body, m, q):
    return svc.trace(body.get("type", body.get("query_type", "text")),
                     body.get("ref", ""))


@route("GET", r"/api/tools")
def _tools(svc, body, m, q):
    return svc.tools()


@route("POST", r"/api/gold-sample")
def _gold_sample(svc, body, m, q):
    return svc.gold_sample(n=int(body.get("n", 20)),
                           stratify=bool(body.get("stratify", True)))


@route("POST", r"/api/gold-eval")
def _gold_eval(svc, body, m, q):
    return svc.gold_eval(body.get("rows", []))


@route("POST", r"/api/herb")
def _herb(svc, body, m, q):
    return svc.herb(body.get("name", body.get("herb", "")))


@route("POST", r"/api/formula-explain")
def _formula_explain(svc, body, m, q):
    return svc.formula_explain(body.get("name", body.get("formula", "")))


# --------------------------------------------------------------------------
def make_handler(service: ServiceContext):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HermesShanghan/0.1"

        def log_message(self, *a):  # quiet by default
            pass

        def _send(self, code: int, payload: Any, ctype="application/json"):
            if isinstance(payload, (dict, list)):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            elif isinstance(payload, bytes):
                data = payload
            else:
                data = str(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8"
                             if ctype.startswith(("text", "application/json")) else ""))
            self.send_header("Content-Length", str(len(data)))
            if not AUTH_TOKEN:      # open CORS only in tokenless local mode
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, b"")

        def _serve_static(self, path: str):
            if path == "/" or path == "":
                path = "/index.html"
            target = (STATIC_DIR / path.lstrip("/")).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                self._send(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype=ctype)

        def _dispatch(self, method: str):
            parsed = urlparse(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            if not path.startswith("/api/"):
                if method == "GET":
                    self._serve_static(path)
                else:
                    self._send(404, {"error": "not found"})
                return
            if AUTH_TOKEN and path != "/api/health":
                supplied = (self.headers.get("Authorization", "")
                            .removeprefix("Bearer ").strip()
                            or self.headers.get("X-Auth-Token", ""))
                if supplied != AUTH_TOKEN:
                    self._send(401, {"error": "unauthorized"})
                    return
            try:
                body = _json_body(self) if method == "POST" else {}
            except ValueError:
                self._send(413, {"error": "request body too large"})
                return
            for rmethod, rx, fn in ROUTES:
                if rmethod != method:
                    continue
                mt = rx.match(path)
                if mt:
                    try:
                        result = fn(service, body, mt, query)
                        self._send(200, result)
                    except Exception as exc:
                        traceback.print_exc()   # full detail server-side only
                        self._send(500, {"error": type(exc).__name__})
                    return
            self._send(404, {"error": f"no route: {method} {path}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, warm: bool = True) -> None:
    if not ServiceContext.ready():
        print("規則庫未生成，請先運行: python3 -m hermes_shanghan pipeline", file=sys.stderr)
        sys.exit(2)
    service = get_service()
    if warm:
        print("預熱規則庫與索引 …", file=sys.stderr)
        service.warm()
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    url = f"http://{host}:{port}/"
    print(f"\n  傷寒論 · Hermes 控制台已啟動", file=sys.stderr)
    print(f"  ▶ {url}", file=sys.stderr)
    print(f"  LLM 後端：{service.llm.backend}（{service.llm.status()['reason']}）", file=sys.stderr)
    print("  Ctrl+C 退出\n", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
        httpd.shutdown()


if __name__ == "__main__":
    serve()
