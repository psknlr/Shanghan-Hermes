"""LLM integration layer for Hermes-Shanghanlun.

Design contract (neuro-symbolic, evidence-leashed):
  * The deterministic rule base is the trustworthy substrate; the LLM is an
    additive layer for fluency, reach and semantic critique.
  * Every LLM claim is verified against clauses (citation guard / evidence
    verifier) BEFORE it reaches a user — the LLM cannot bypass
    「無證據鏈，不成回答」.
  * The layer degrades gracefully: with no `litellm` and no API key it runs a
    deterministic `local` backend, so the whole system works offline and is
    fully testable without a network.

Backends:
  litellm   real models via LiteLLM (100+ providers: Anthropic/OpenAI/…)
  local     deterministic, rule-derived responses (default offline fallback)
  scripted  queued responses for tests
"""
from .client import ChatResult, LLMClient, ToolCall, get_client
from .config import LLMSettings, load_settings

__all__ = ["LLMClient", "ChatResult", "ToolCall", "get_client",
           "LLMSettings", "load_settings"]
