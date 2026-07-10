"""Minimal Model Context Protocol (MCP) server over stdio.

Implements just enough JSON-RPC 2.0 for Claude Code / Claude Desktop / any MCP
client to discover and call the grounded Shanghan tools — no third-party
dependencies. Speaks newline-delimited JSON-RPC on stdin/stdout.

Register in Claude Code, e.g.:
  claude mcp add shanghan -- python3 -m hermes_shanghan serve-mcp

Tools exposed are the read-only, evidence-returning ToolRegistry tools plus a
`shanghan_ask` tool that runs the full agent (citation-guarded answer).
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional

from .. import config
from ..agent.tools import get_registry

# 版本協商：客戶端請求的版本在支持列表內則回顯，否則回退到基線
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[-1]
from .._version import __version__
SERVER_INFO = {"name": "hermes-shanghanlun", "version": __version__}

# ---------------------------------------------------------------------------
# 實驗性長任務（tasks/*，對齊 MCP 2025-11-25 experimental tasks 形態）：
# 全庫掃描/深研類長調用 submit 後立即返回 task_id，工作線程執行；
# status/result/cancel/list 輪詢管理。誠實邊界：取消是協作式的——只讀
# 工具無中斷點，cancel 生效於「結果丟棄不返回」；stdio 單線程請求循環
# 不推送進度通知（progress notification 需雙向流，見 HARNESS.md 差距表）。
# ---------------------------------------------------------------------------
_TASKS: Dict[str, Dict] = {}
_TASK_LOCK = threading.Lock()
MAX_TASKS = 64
TASK_TTL_S = 900          # 終態任務 15 分鐘後回收（in-memory，非 durable）


def _prune_tasks() -> None:
    """調用方須持有 _TASK_LOCK。過期終態任務回收（TTL）。"""
    now = time.time()
    for k in [k for k, v in _TASKS.items()
              if v["status"] in ("completed", "failed", "cancelled")
              and now - v["started_at"] > TASK_TTL_S]:
        _TASKS.pop(k, None)


def _tasks_submit(params: Dict) -> Dict:
    name = params.get("name", "")
    args = params.get("arguments") or {}
    if name not in get_registry().names():
        raise KeyError(f"unknown tool: {name}")
    task_id = uuid.uuid4().hex[:16]
    with _TASK_LOCK:
        _prune_tasks()
        if len(_TASKS) >= MAX_TASKS:      # 只回收已終態的舊任務
            for k in [k for k, v in _TASKS.items()
                      if v["status"] in ("completed", "failed", "cancelled")]:
                _TASKS.pop(k)
                if len(_TASKS) < MAX_TASKS:
                    break
        _TASKS[task_id] = {"task_id": task_id, "tool": name,
                           "status": "running", "started_at": time.time(),
                           "cancelled": False, "result": None, "error": None}

    def _work():
        try:
            out = get_registry().call(name, args)
        except Exception as exc:
            out, err = None, f"{type(exc).__name__}: {exc}"
        else:
            err = out.get("error") if isinstance(out, dict) else None
        with _TASK_LOCK:
            t = _TASKS.get(task_id)
            if t is None:
                return
            if t["cancelled"]:
                t["status"] = "cancelled"     # 結果丟棄
            elif err:
                t["status"] = "failed"
                t["error"] = err
            else:
                t["status"] = "completed"
                t["result"] = out

    threading.Thread(target=_work, daemon=True).start()
    return {"task_id": task_id, "status": "running", "tool": name}


def _task_view(t: Dict) -> Dict:
    return {"task_id": t["task_id"], "tool": t["tool"], "status": t["status"],
            "elapsed_ms": int((time.time() - t["started_at"]) * 1000),
            "error": t["error"],
            "durability": "in-memory experimental（服務重啟即失，TTL 15min；"
                          "非 MCP durable tasks 完整實現）"}


def _tasks_status(params: Dict) -> Dict:
    with _TASK_LOCK:
        t = _TASKS.get(params.get("task_id", ""))
        if t is None:
            raise KeyError(f"unknown task: {params.get('task_id')}")
        return _task_view(t)


def _tasks_result(params: Dict) -> Dict:
    with _TASK_LOCK:
        t = _TASKS.get(params.get("task_id", ""))
        if t is None:
            raise KeyError(f"unknown task: {params.get('task_id')}")
        if t["status"] != "completed":
            raise KeyError(f"task not finished: status={t['status']}")
        return _content(t["result"])


def _tasks_cancel(params: Dict) -> Dict:
    with _TASK_LOCK:
        t = _TASKS.get(params.get("task_id", ""))
        if t is None:
            raise KeyError(f"unknown task: {params.get('task_id')}")
        if t["status"] == "running":
            t["cancelled"] = True
            note = "協作式取消：工具無中斷點，完成後結果丟棄"
        else:
            note = f"任務已終態 {t['status']}，取消無效果"
        return {"task_id": t["task_id"], "cancelled": t["cancelled"],
                "note": note}


def _tasks_list() -> Dict:
    with _TASK_LOCK:
        return {"tasks": [_task_view(t) for t in _TASKS.values()]}

# ---------------------------------------------------------------------------
# Resources：把核心資產暴露為 URI（MCP resources/list + resources/read）
# ---------------------------------------------------------------------------
MAX_RESOURCE_BYTES = 524_288

def _resource_catalog():
    items = [
        ("shanghan://clauses", "傷寒論條文全集（681 條，A 層）",
         config.CLAUSE_DIR / "clauses.jsonl", "application/jsonl"),
        ("shanghan://rules/formula", "方證規則（113 方，D 層/證據錨定 A）",
         config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl", "application/jsonl"),
        ("shanghan://trace/citation-network", "歷代引文計量網絡",
         config.TRACE_DIR / "citation_network.json", "application/json"),
        ("shanghan://trace/claims", "方證觀點庫（ClaimID，證據分級）",
         config.TRACE_DIR / "claims.json", "application/json"),
        ("shanghan://trace/schools", "學派註冊表（posthoc_induction）",
         config.TRACE_DIR / "schools.json", "application/json"),
        ("shanghan://trace/id-registry", "統一知識標識註冊表",
         config.TRACE_DIR / "id_registry.json", "application/json"),
        ("shanghan://manifest", "語料 manifest（57 部書，sha256）",
         config.MANIFEST_DIR / "corpus_manifest.json", "application/json"),
    ]
    lib_cat = config.LIBRARY_DIR / "catalog.json"
    if lib_cat.exists():
        items.append(("shanghan://library/catalog",
                      "中醫笈成全庫編目（803 部，P 層旁證）",
                      lib_cat, "application/json"))
    return items


def _resources_list() -> Dict:
    return {"resources": [
        {"uri": uri, "name": uri.split("//")[1], "description": desc,
         "mimeType": mime} for uri, desc, path, mime in _resource_catalog()
        if path.exists()]}


def _resources_read(uri: str) -> Dict:
    for u, desc, path, mime in _resource_catalog():
        if u == uri and path.exists():
            raw = path.read_bytes()
            truncated = len(raw) > MAX_RESOURCE_BYTES
            text = raw[:MAX_RESOURCE_BYTES].decode("utf-8", errors="replace")
            if truncated:
                text += f"\n…（截斷：完整 {len(raw)} bytes，僅返回前 {MAX_RESOURCE_BYTES}）"
            return {"contents": [{"uri": uri, "mimeType": mime, "text": text}]}
    raise KeyError(f"unknown resource: {uri}")


# ---------------------------------------------------------------------------
# Prompts：把核心工作流暴露為可選模板（MCP prompts/list + prompts/get）
# ---------------------------------------------------------------------------
PROMPTS = {
    "formula-differential": {
        "description": "方證鑒別：兩至三個經方的多軸對比與關鍵鑒別點（回源條文）",
        "arguments": [{"name": "formulas", "description": "方名，頓號分隔",
                       "required": True}],
        "template": "請用 shanghan_differential 對比 {formulas}，逐軸列出差異，"
                    "每個關鍵鑒別點附 clause_id；結論不得超出工具證據。",
    },
    "provenance-trace": {
        "description": "深度溯源：條文/方劑/術語的歷代傳播鏈與計量",
        "arguments": [{"name": "target", "description": "條文號/方名/術語",
                       "required": True}],
        "template": "請用 shanghan_trace 與 shanghan_citation_network 追溯"
                    "「{target}」：首見出處、注家解釋時間線、歷代引用計量；"
                    "「最早」一律表述為「在庫首現」。",
    },
    "misquote-review": {
        "description": "誤引審查：一段引文能否作為《傷寒論》原文直引",
        "arguments": [{"name": "quote", "description": "待審引文",
                       "required": True}],
        "template": "請用 shanghan_trace(query_type=quote) 審查「{quote}」，"
                    "逐片段判定原文逐字/後世歸納語，給出可否直引結論與改寫建議。",
    },
    "patient-intake": {
        "description": "患者安全問診：就診信息整理（不診斷不處方）",
        "arguments": [{"name": "narrative", "description": "患者自然敘述",
                       "required": True}],
        "template": "請用 shanghan_intake 整理患者敘述「{narrative}」為結構化"
                    "四診表，列出缺失關鍵信息與追問建議；不得給出診斷、方劑或劑量。",
    },
}


def _prompts_list() -> Dict:
    return {"prompts": [{"name": k, "description": v["description"],
                         "arguments": v["arguments"]}
                        for k, v in sorted(PROMPTS.items())]}


def _prompts_get(name: str, arguments: Dict) -> Dict:
    p = PROMPTS.get(name)
    if p is None:
        raise KeyError(f"unknown prompt: {name}")
    text = p["template"]
    for a in p["arguments"]:
        text = text.replace("{" + a["name"] + "}",
                            str((arguments or {}).get(a["name"], "")))
    return {"description": p["description"],
            "messages": [{"role": "user",
                          "content": {"type": "text", "text": text}}]}


def _result(id_: Any, result: Dict) -> Dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_: Any, code: int, message: str) -> Dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _tool_list() -> Dict:
    tools = []
    for t in get_registry().specs():
        fn = t["function"]
        tools.append({"name": fn["name"], "description": fn["description"],
                      "inputSchema": fn["parameters"]})
    # agent tool
    tools.append({
        "name": "shanghan_ask",
        "description": "用《傷寒論》智能體回答問題：自動取證、回源 clause_id、安全治理。",
        "inputSchema": {"type": "object", "properties": {
            "question": {"type": "string"},
            "role": {"type": "string", "enum": ["doctor", "researcher", "student", "patient"]}},
            "required": ["question"]}})
    return {"tools": tools}


def _content(obj: Any) -> Dict:
    return {"content": [{"type": "text",
                         "text": json.dumps(obj, ensure_ascii=False, indent=1)}]}


def handle(request: Dict) -> Optional[Dict]:
    method = request.get("method")
    id_ = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        asked = (params.get("protocolVersion") or "")
        version = asked if asked in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
        return _result(id_, {"protocolVersion": version,
                             "capabilities": {"tools": {"listChanged": False},
                                              "resources": {"listChanged": False},
                                              "prompts": {"listChanged": False},
                                              "experimental": {"tasks": {
                                                  "cancel": "cooperative"}}},
                             "serverInfo": SERVER_INFO})
    if method in ("notifications/initialized", "initialized"):
        return None  # notification, no response
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, _tool_list())
    if method == "resources/list":
        return _result(id_, _resources_list())
    if method == "resources/read":
        try:
            return _result(id_, _resources_read(params.get("uri", "")))
        except KeyError as exc:
            return _error(id_, -32602, str(exc))
    if method == "prompts/list":
        return _result(id_, _prompts_list())
    if method == "prompts/get":
        try:
            return _result(id_, _prompts_get(params.get("name", ""),
                                             params.get("arguments") or {}))
        except KeyError as exc:
            return _error(id_, -32602, str(exc))
    if method in ("tasks/submit", "tasks/status", "tasks/result",
                  "tasks/cancel", "tasks/list"):
        try:
            fn = {"tasks/submit": _tasks_submit, "tasks/status": _tasks_status,
                  "tasks/result": _tasks_result, "tasks/cancel": _tasks_cancel,
                  }.get(method)
            out = _tasks_list() if method == "tasks/list" else fn(params)
            return _result(id_, out)
        except KeyError as exc:
            return _error(id_, -32602, str(exc))
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "shanghan_ask":
                from ..agent.agent import ShanghanAgent
                out = ShanghanAgent().ask(args.get("question", ""), args.get("role"))
            else:
                out = get_registry().call(name, args)
            payload = _content(out)
            # registry-level failures (unknown tool, bad channel/formula…)
            # come back as {"error": ...} — mark them per MCP spec
            if isinstance(out, dict) and out.get("error"):
                payload["isError"] = True
            return _result(id_, payload)
        except Exception as exc:  # surface as tool error, keep server alive
            return _result(id_, {"content": [{"type": "text",
                                              "text": f"tool error: {type(exc).__name__}: {exc}"}],
                                 "isError": True})
    if id_ is not None:
        return _error(id_, -32601, f"method not found: {method}")
    return None


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_error(None, -32700, "parse error"),
                                    ensure_ascii=False) + "\n")
            stdout.flush()
            continue
        try:
            response = handle(request)
        except Exception as exc:
            response = _error(request.get("id"), -32603, f"internal error: {exc}")
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
