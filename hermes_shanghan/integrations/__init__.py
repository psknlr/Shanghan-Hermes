"""Harness integrations.

Hermes-Shanghanlun exposes its grounded tools through three standard surfaces
so external agent runtimes can drive it:

  tool_specs.py  OpenAI-compatible function specs + dispatcher
                 → Codex CLI, OpenCode/openclaw, any function-calling LLM
  mcp_server.py  Model Context Protocol server over stdio
                 → Claude Code, Claude Desktop, any MCP client
  AGENTS.md      discovery doc for agent harnesses

All three share the single ToolRegistry capability surface, so behaviour
(and the evidence/safety guarantees) are identical across harnesses.
"""
from .tool_specs import dispatch, export_specs, openai_tool_specs

__all__ = ["openai_tool_specs", "dispatch", "export_specs"]
