"""OpenAI-compatible tool specs + dispatcher.

Usable by Codex CLI, OpenCode/openclaw, the OpenAI/Anthropic SDKs, or any
function-calling loop. `openai_tool_specs()` returns the `tools=[...]` array;
`dispatch(name, arguments)` executes a call against the grounded registry and
returns JSON-serializable results (with clause_id evidence).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..agent.tools import get_registry


def openai_tool_specs() -> List[Dict]:
    """OpenAI/Anthropic function-calling tool array."""
    return get_registry().specs()


def anthropic_tool_specs() -> List[Dict]:
    """Anthropic Messages API tool format (name/description/input_schema)."""
    specs = []
    for t in get_registry().specs():
        fn = t["function"]
        specs.append({"name": fn["name"], "description": fn["description"],
                      "input_schema": fn["parameters"]})
    return specs


def dispatch(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return get_registry().call(name, arguments or {})


def export_specs(out_path: Path) -> Path:
    """Write a portable spec bundle (OpenAI + Anthropic + MCP-style) to disk."""
    from ..agent.tools import TOOLS_VERSION, get_registry
    bundle = {
        "name": "hermes-shanghanlun",
        "description": "《傷寒論》自主規則挖掘與證據回源工具集（只讀、回源 clause_id）。",
        "tool_spec_version": TOOLS_VERSION,
        "openai_tools": openai_tool_specs(),
        "anthropic_tools": anthropic_tool_specs(),
        "contracts": get_registry().contracts(),
        "invocation": {
            "cli": "python3 -m hermes_shanghan tool-call <name> --args '<json>'",
            "mcp": "python3 -m hermes_shanghan serve-mcp",
        },
        "safety": "患者語境禁止診斷/處方/劑量；所有結論回源條文編號。",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path
