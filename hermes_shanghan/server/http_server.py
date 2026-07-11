"""Stdlib HTTP server for the Hermes-Shanghanlun web console.

No third-party dependencies. Serves a single-page app from ./static and a
JSON API backed by ServiceContext. Concurrency via ThreadingHTTPServer.

治理（九輪）：所有 /api/* 請求先解析服務端 Principal（policy.py），角色
上限由身份綁定（HERMES_API_KEYS）而非請求體聲明；每條路由帶最低角色，
臨床類端點（match/differential/formula/mistreatment/deep-research…）與
/api/tool 走同一策略層。/livez 與 /readyz 分離（假健康防護）。
"""
from __future__ import annotations

import inspect
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

from . import policy
from .service import ServiceContext, get_service

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 256 * 1024        # JSON request bodies are tiny; cap hard
MAX_RESPONSE_BYTES = 2_000_000     # 響應上限：超限回錯誤+trace_id，不靜默截斷
# 速率限制（每 IP 每分鐘；0=關閉。生產部署建議 HTTPS 反代 + IP allowlist）
RATE_LIMIT_PER_MIN = int(os.environ.get("HERMES_RATE_LIMIT", "0"))
_RATE_BUCKET: dict = {}
# optional bearer-token auth for non-localhost deployments:
#   HERMES_SERVER_TOKEN=... python3 -m hermes_shanghan serve --host 0.0.0.0
AUTH_TOKEN = os.environ.get("HERMES_SERVER_TOKEN", "")
# role-bound API keys（token:role[:subject] 逗號分隔）——配置後角色上限由
# 服務端身份決定，請求體 role 只能降級不可提權
API_KEYS = policy.parse_api_keys(os.environ.get("HERMES_API_KEYS", ""))
# 免鑒權探針路徑（負載均衡/監控需要）
OPEN_PATHS = ("/api/health", "/livez", "/readyz")


import threading

_RATE_LOCK = threading.Lock()

# 請求體整數參數的統一上下限（計算型 DoS 防護：算完才檢查響應大小為時已晚）
_INT_CAPS = {"top_k": (1, 50), "rounds": (1, 6), "max_steps": (1, 12),
             "n": (1, 200), "limit": (1, 100), "per_book": (1, 10),
             "offset": (0, 100000)}


def _clamp_body(body: Dict) -> None:
    for key, (lo, hi) in _INT_CAPS.items():
        if key in body:
            try:
                body[key] = max(lo, min(hi, int(body[key])))
            except (TypeError, ValueError):
                body[key] = lo


def _json_body(handler: BaseHTTPRequestHandler) -> Dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length <= 0:
        return {}
    if length > MAX_BODY_BYTES:
        raise ValueError("body_too_large")
    raw = handler.rfile.read(length)
    try:
        out = json.loads(raw.decode("utf-8"))
    except Exception:
        raise ValueError("invalid_json")     # 400，不再靜默轉成 {}
    return out if isinstance(out, dict) else {}


# route table: (method, regex, handler, min_role, wants_principal)
ROUTES: list = []


def route(method: str, pattern: str, min_role: str = "patient"):
    rx = re.compile(f"^{pattern}$")

    def deco(fn):
        wants = "ctx" in inspect.signature(fn).parameters
        ROUTES.append((method, rx, fn, min_role, wants))
        return fn
    return deco


# --------------------------------------------------------------------------
@route("GET", r"/api/health")
def _health(svc, body, m, q, ctx=None):
    return {"ok": True, "ready": svc.ready(), "backend": svc.llm.backend}


@route("GET", r"/api/stats")
def _stats(svc, body, m, q, ctx=None):
    return svc.stats()


@route("GET", r"/api/llm/status")
def _llm_status(svc, body, m, q, ctx=None):
    return svc.llm_status()


@route("GET", r"/api/formulas")
def _formulas(svc, body, m, q, ctx=None):
    return svc.list_formulas()


@route("GET", r"/api/channels")
def _channels(svc, body, m, q, ctx=None):
    return svc.channels()


@route("GET", r"/api/skills")
def _skills(svc, body, m, q, ctx=None):
    return svc.skills()


@route("POST", r"/api/search")
def _search(svc, body, m, q, ctx=None):
    return svc.search(body.get("query", ""), top_k=int(body.get("top_k", 8)),
                      six_channel=body.get("six_channel"), formula=body.get("formula"),
                      field=body.get("field"), expand=bool(body.get("expand")))


@route("GET", r"/api/clause/([^/]+)")
def _clause(svc, body, m, q, ctx=None):
    # 角色只來自 RequestContext（十一輪 P0-2）：患者主體即便不傳
    # role 參數也拿不到 student 默認值
    return svc.explain_clause(m.group(1), role=ctx.role_or("student"))


@route("POST", r"/api/explain")
def _explain(svc, body, m, q, ctx=None):
    return svc.explain_clause(body.get("ref"), role=ctx.role_or("student"))


@route("POST", r"/api/match", min_role="student")
def _match(svc, body, m, q, ctx=None):
    return svc.match(body.get("symptoms", []), pulse=body.get("pulse", []),
                     six_channel=body.get("six_channel"), top_k=int(body.get("top_k", 5)))


@route("POST", r"/api/differential", min_role="student")
def _diff(svc, body, m, q, ctx=None):
    return svc.differential(body.get("formulas", []),
                            use_llm=bool(body.get("use_llm", True)))


@route("POST", r"/api/teach", min_role="student")
def _teach(svc, body, m, q, ctx=None):
    return svc.teach(body.get("channel", "太陽病"))


@route("POST", r"/api/mistreatment", min_role="student")
def _mistreat(svc, body, m, q, ctx=None):
    return svc.mistreatment(body.get("query"))


@route("POST", r"/api/formula", min_role="student")
def _formula(svc, body, m, q, ctx=None):
    return svc.formula_rule(body.get("formula", ""))


@route("POST", r"/api/research", min_role="researcher")
def _research(svc, body, m, q, ctx=None):
    return svc.research(body.get("topic", ""), outputs=body.get("outputs"))


@route("POST", r"/api/paper", min_role="researcher")
def _paper(svc, body, m, q, ctx=None):
    return svc.paper(body.get("type", "formula_pattern"), topic=body.get("topic", ""),
                     use_llm=body.get("use_llm", True))


@route("POST", r"/api/complex")
def _complex(svc, body, m, q, ctx=None):
    return svc.complex(body.get("question", ""), role=ctx.effective_role)


@route("POST", r"/api/chat")
def _chat(svc, body, m, q, ctx=None):
    # session 以服務端主體命名空間隔離（防 fixation/串話）；未帶 session_id
    # 時服務端生成並隨響應回傳
    return svc.chat(body.get("question", ""),
                    session_id=str(body.get("session_id", "") or ""),
                    role=ctx.effective_role,
                    subject=(ctx.principal_id if ctx else "anonymous"))


@route("POST", r"/api/deep-research", min_role="researcher")
def _deep_research(svc, body, m, q, ctx=None):
    return svc.deep_research(body.get("topic", ""),
                             rounds=int(body.get("rounds", 3)))


@route("POST", r"/api/patient")
def _patient(svc, body, m, q, ctx=None):
    return svc.patient(body.get("question", ""))


@route("POST", r"/api/agent")
def _agent(svc, body, m, q, ctx=None):
    return svc.agent(body.get("question", ""), role=ctx.effective_role,
                     max_steps=int(body.get("max_steps", 5)))


@route("POST", r"/api/council")
def _council(svc, body, m, q, ctx=None):
    return svc.council(body.get("question", ""), role=ctx.effective_role)


@route("POST", r"/api/tool")
def _tool(svc, body, m, q, ctx=None):
    return svc.tool_call(body.get("name", ""), body.get("arguments", {}),
                         role=(ctx.effective_role or ""),
                         subject=(ctx.principal_id if ctx else ""))


@route("POST", r"/api/trace")
def _trace(svc, body, m, q, ctx=None):
    return svc.trace(body.get("type", body.get("query_type", "text")),
                     body.get("ref", ""),
                     synthesize=bool(body.get("synthesize", True)))


@route("GET", r"/api/tools")
def _tools(svc, body, m, q, ctx=None):
    return svc.tools()


@route("POST", r"/api/gold-sample", min_role="student")
def _gold_sample(svc, body, m, q, ctx=None):
    return svc.gold_sample(n=int(body.get("n", 20)),
                           stratify=bool(body.get("stratify", True)))


@route("POST", r"/api/gold-eval", min_role="student")
def _gold_eval(svc, body, m, q, ctx=None):
    return svc.gold_eval(body.get("rows", []))


@route("POST", r"/api/herb", min_role="student")
def _herb(svc, body, m, q, ctx=None):
    return svc.herb(body.get("name", body.get("herb", "")))


@route("POST", r"/api/trace/passages", min_role="student")
def _trace_passages(svc, body, m, q, ctx=None):
    return svc.trace_passages(str(body.get("book_dir", ""))[:80],
                              body.get("clause_ids", []),
                              offset=int(body.get("offset", 0) or 0),
                              limit=int(body.get("limit", 8) or 8))


@route("POST", r"/api/source/passage", min_role="student")
def _source_passage(svc, body, m, q, ctx=None):
    return svc.source_passage(body.get("book", ""), body.get("ref", ""))


@route("POST", r"/api/trace/mentions", min_role="student")
def _trace_mentions(svc, body, m, q, ctx=None):
    return svc.trace_mentions(str(body.get("name", ""))[:40],
                              str(body.get("book_dir", ""))[:60],
                              offset=int(body.get("offset", 0) or 0),
                              limit=int(body.get("limit", 6) or 6))


@route("POST", r"/api/errata")
def _errata_submit(svc, body, m, q, ctx=None):
    return svc.errata_submit(body.get("clause_ref", ""),
                             body.get("quote", ""),
                             body.get("suggestion", ""),
                             note=body.get("note", ""),
                             subject=(ctx.principal_id if ctx else "anonymous"))


@route("GET", r"/api/errata", min_role="doctor")
def _errata_list(svc, body, m, q, ctx=None):
    limit = _qint(q, "limit", 50, 1, 200)
    return svc.errata_list(limit=limit or 50)


@route("POST", r"/api/trace/term-passages", min_role="student")
def _term_passages(svc, body, m, q, ctx=None):
    return svc.term_passages(str(body.get("term", ""))[:40],
                             str(body.get("book", ""))[:60],
                             offset=int(body.get("offset", 0) or 0),
                             limit=int(body.get("limit", 6) or 6))


@route("POST", r"/api/quiz", min_role="student")
def _quiz(svc, body, m, q, ctx=None):
    return svc.quiz(channel=str(body.get("channel", ""))[:12],
                    n=int(body.get("n", 8) or 8),
                    seed=int(body.get("seed", 1) or 1),
                    use_llm=bool(body.get("use_llm", False)))


@route("GET", r"/api/charmap")
def _charmap(svc, body, m, q, ctx=None):
    return svc.charmap()


@route("POST", r"/api/intake")
def _intake(svc, body, m, q, ctx=None):
    return svc.intake(body.get("text", ""),
                      use_llm=bool(body.get("use_llm", True)))


@route("POST", r"/api/adjudicate", min_role="student")
def _adjudicate(svc, body, m, q, ctx=None):
    return svc.adjudicate(body.get("symptoms", []),
                          pulse=body.get("pulse", []),
                          six_channel=body.get("six_channel", ""),
                          use_llm=bool(body.get("use_llm", True)))


# -- 運行中心 / 評測 / Artifact / 治理（十二輪新控制面）---------------------
@route("GET", r"/api/whoami")
def _whoami(svc, body, m, q, ctx=None):
    # 前端角色選擇只是請求；真正的上限與生效角色由服務端裁定並回顯
    return {"principal_id": ctx.principal_id, "tenant_id": ctx.tenant_id,
            "role_ceiling": ctx.role_ceiling,
            "effective_role": ctx.effective_role,
            "request_id": ctx.request_id}


def _qint(q, key, default, lo, hi):
    """查詢串整數容錯：非數字/越界 → 400 級錯誤對象（不拋 500）。"""
    raw = (q.get(key, [str(default)]))[0]
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return None


@route("GET", r"/api/runs", min_role="student")
def _runs(svc, body, m, q, ctx=None):
    limit = _qint(q, "limit", 30, 1, 100)
    if limit is None:
        return {"error": "limit 必須是 1-100 的整數", "_status": 400}
    return svc.runs_list(limit=limit)


@route("GET", r"/api/runs/([A-Za-z0-9_\-]+)/spans", min_role="student")
def _run_spans(svc, body, m, q, ctx=None):
    offset = _qint(q, "offset", 0, 0, 100000)
    limit = _qint(q, "limit", 60, 1, 200)
    if offset is None or limit is None:
        return {"error": "offset/limit 必須是整數", "_status": 400}
    return svc.run_spans(m.group(1), offset=offset, limit=limit)


@route("GET", r"/api/runs/([A-Za-z0-9_\-]+)/output/([A-Za-z0-9_\-]+)",
       min_role="student")
def _run_output(svc, body, m, q, ctx=None):
    return svc.run_node_output(m.group(1), m.group(2))


@route("GET", r"/api/runs/([A-Za-z0-9_\-]+)/evidence", min_role="student")
def _run_evidence(svc, body, m, q, ctx=None):
    offset = _qint(q, "offset", 0, 0, 100000)
    limit = _qint(q, "limit", 100, 1, 400)
    if offset is None or limit is None:
        return {"error": "offset/limit 必須是整數", "_status": 400}
    return svc.run_evidence(m.group(1), offset=offset, limit=limit)


@route("GET", r"/api/runs/([A-Za-z0-9_\-]+)", min_role="student")
def _run_detail(svc, body, m, q, ctx=None):
    return svc.run_detail(m.group(1))


@route("POST", r"/api/runs", min_role="student")
def _run_start(svc, body, m, q, ctx=None):
    # 運行角色受服務端上限鉗制：ctx.effective_role 優先；全權主體可指定
    role = ctx.effective_role or body.get("run_role") or "researcher"
    return svc.run_start(body.get("query", ""),
                         mode=body.get("mode", "agent"), role=role,
                         max_steps=int(body.get("max_steps", 6)),
                         max_tool_calls=int(body.get("max_tool_calls", 12)))


@route("POST", r"/api/runs/([A-Za-z0-9_\-]+)/(approve|reject|resume|cancel|replay|export)",
       min_role="doctor")
def _run_action(svc, body, m, q, ctx=None):
    return svc.run_action(m.group(1), m.group(2),
                          approver=str(body.get("approver", ""))
                          or ctx.principal_id,
                          reason=str(body.get("reason", ""))[:400],
                          trigger=str(body.get("trigger", ""))[:60])


@route("POST", r"/api/eval/trajectory", min_role="researcher")
def _eval_traj(svc, body, m, q, ctx=None):
    return svc.eval_trajectory()


@route("POST", r"/api/eval/perturbation", min_role="researcher")
def _eval_pert(svc, body, m, q, ctx=None):
    return svc.eval_perturbation()


@route("GET", r"/api/artifacts", min_role="student")
def _artifacts(svc, body, m, q, ctx=None):
    return svc.artifacts()


@route("GET", r"/api/artifact", min_role="student")
def _artifact(svc, body, m, q, ctx=None):
    return svc.artifact_read((q.get("path", [""]))[0])


@route("GET", r"/api/artifact/meta", min_role="student")
def _artifact_meta(svc, body, m, q, ctx=None):
    return svc.artifact_meta((q.get("path", [""]))[0])


@route("GET", r"/api/artifact/download", min_role="student")
def _artifact_download(svc, body, m, q, ctx=None):
    out = svc.artifact_download((q.get("path", [""]))[0])
    if out is None:
        return {"error": "路徑不合法/文件不存在/超過 8MB 下載上限",
                "_status": 404}
    filename, mime, data = out
    return {"_file": {"filename": filename, "mime": mime}, "_bytes": data}


@route("GET", r"/api/sessions", min_role="student")
def _sessions(svc, body, m, q, ctx=None):
    return svc.sessions_list(ctx.principal_id)


@route("GET", r"/api/sessions/([A-Za-z0-9_\-]+)", min_role="student")
def _session_turns(svc, body, m, q, ctx=None):
    return svc.session_turns(ctx.principal_id, m.group(1))


@route("POST", r"/api/sessions/([A-Za-z0-9_\-]+)/delete", min_role="student")
def _session_delete(svc, body, m, q, ctx=None):
    return svc.session_delete(ctx.principal_id, m.group(1))


@route("GET", r"/api/governance", min_role="student")
def _governance(svc, body, m, q, ctx=None):
    return svc.governance()


@route("POST", r"/api/formula-explain", min_role="student")
def _formula_explain(svc, body, m, q, ctx=None):
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
            # open CORS 只在完全無憑證的本地開發模式（任一鑒權配置在場即關）
            if not AUTH_TOKEN and not API_KEYS:
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
            # 健康探針：/livez 只回進程存活，/readyz 校驗數據能力（假健康防護）
            if path == "/livez" and method == "GET":
                from ..health import livez
                self._send(200, livez())
                return
            if path == "/readyz" and method == "GET":
                from ..health import readyz
                out = readyz()
                self._send(200 if out["ready"] else 503, out)
                return
            if not path.startswith("/api/"):
                if method == "GET":
                    self._serve_static(path)
                else:
                    self._send(404, {"error": "not found"})
                return
            supplied = (self.headers.get("Authorization", "")
                        .removeprefix("Bearer ").strip()
                        or self.headers.get("X-Auth-Token", ""))
            if (AUTH_TOKEN or API_KEYS) and path in OPEN_PATHS:
                principal = policy.PrincipalContext(
                    subject_id="probe", role_ceiling="patient",
                    auth_level="none")
            else:
                principal = policy.resolve_principal(supplied, API_KEYS,
                                                     AUTH_TOKEN)
                if principal is None:
                    self._send(401, {"error": "unauthorized"})
                    return
            if RATE_LIMIT_PER_MIN:
                import time as _time
                ip = self.client_address[0]
                window = int(_time.time() // 60)
                key = (ip, window)
                with _RATE_LOCK:      # ThreadingHTTPServer 下的共享計數
                    _RATE_BUCKET.setdefault(key, 0)
                    _RATE_BUCKET[key] += 1
                    if len(_RATE_BUCKET) > 4096:    # 防字典無界增長
                        for k in [k for k in _RATE_BUCKET if k[1] < window]:
                            _RATE_BUCKET.pop(k, None)
                    limited = _RATE_BUCKET[key] > RATE_LIMIT_PER_MIN
                if limited:
                    self._send(429, {"error": "rate limited"})
                    return
            try:
                body = _json_body(self) if method == "POST" else {}
            except ValueError as ve:
                if str(ve) == "invalid_json":       # 非法 JSON → 400，不再靜默 {}
                    self._send(400, {"error": "invalid JSON body"})
                else:
                    self._send(413, {"error": "request body too large"})
                return
            _clamp_body(body)          # top_k/rounds/max_steps 等統一上下限
            for rmethod, rx, fn, min_role, wants_ctx in ROUTES:
                if rmethod != method:
                    continue
                mt = rx.match(path)
                if mt:
                    # 端點能力矩陣：主體上限低於端點最低角色 → 403
                    if not policy.allow_min_role(principal, min_role):
                        print(f"[policy-denied] {method} {path} "
                              f"subject={principal.subject_id} "
                              f"ceiling={principal.role_ceiling} "
                              f"required={min_role}")
                        self._send(403, {"error": "policy_denied",
                                         "required_role": min_role,
                                         "your_ceiling": principal.role_ceiling})
                        return
                    # 生效角色只此一處裁定；業務路由不得再讀 body/query role
                    import uuid as _uuid
                    try:
                        requested = body.pop("role", None) or \
                            (query.get("role", [None])[0])
                        eff = policy.effective_role(principal, requested)
                    except policy.PolicyDenied as pd:
                        print(f"[policy-denied] {method} {path} "
                              f"subject={principal.subject_id} {pd.reason}")
                        self._send(403, {"error": "policy_denied",
                                         "reason": pd.reason,
                                         "requested_role": pd.requested,
                                         "your_ceiling": pd.ceiling})
                        return
                    ctx = policy.RequestContext(
                        principal_id=principal.subject_id,
                        tenant_id=principal.tenant_id,
                        role_ceiling=principal.role_ceiling,
                        effective_role=eff,
                        request_id=_uuid.uuid4().hex[:12])
                    try:
                        kwargs = {"ctx": ctx} if wants_ctx else {}
                        result = fn(service, body, mt, query, **kwargs)
                        # 文件下載：附件頭 + 原始字節（十三輪 十三）
                        if isinstance(result, dict) and "_file" in result:
                            meta = result["_file"]
                            self.send_response(200)
                            self.send_header("Content-Type", meta["mime"])
                            self.send_header(
                                "Content-Disposition",
                                f'attachment; filename="{meta["filename"]}"')
                            data = result["_bytes"]
                            self.send_header("Content-Length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            return
                        status = 200
                        if isinstance(result, dict) and "_status" in result:
                            status = int(result.pop("_status"))
                        # 患者投影在序列化出口再次執行（不只依賴業務函數）
                        result = policy.project_for_role(result,
                                                         ctx.effective_role)
                        blob = json.dumps(result, ensure_ascii=False,
                                          default=str)
                        if len(blob.encode("utf-8")) > MAX_RESPONSE_BYTES:
                            tid = _uuid.uuid4().hex[:12]
                            print(f"[response-too-large trace_id={tid}] "
                                  f"{method} {path}")
                            self._send(500, {"error": "response too large",
                                             "trace_id": tid,
                                             "hint": "縮小 top_k/limit 或分頁"})
                            return
                        self._send(status, result)
                    except Exception as exc:
                        tid = _uuid.uuid4().hex[:12]
                        print(f"[error trace_id={tid}] {method} {path}")
                        traceback.print_exc()   # full detail server-side only
                        self._send(500, {"error": type(exc).__name__,
                                         "trace_id": tid})
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
