"""Agentic layer: tool registry, citation guard, and a provider-agnostic
tool-calling agent that keeps every answer leashed to clause evidence."""
from .agent import ShanghanAgent
from .tools import ToolRegistry, get_registry

__all__ = ["ShanghanAgent", "ToolRegistry", "get_registry"]
