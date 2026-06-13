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
from typing import Any, Dict, Optional

from ..agent.tools import get_registry

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "hermes-shanghanlun", "version": "0.1.0"}


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
        return _result(id_, {"protocolVersion": PROTOCOL_VERSION,
                             "capabilities": {"tools": {}},
                             "serverInfo": SERVER_INFO})
    if method in ("notifications/initialized", "initialized"):
        return None  # notification, no response
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, _tool_list())
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "shanghan_ask":
                from ..agent.agent import ShanghanAgent
                out = ShanghanAgent().ask(args.get("question", ""), args.get("role"))
            else:
                out = get_registry().call(name, args)
            return _result(id_, _content(out))
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
