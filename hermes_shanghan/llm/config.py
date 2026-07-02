"""LLM settings, resolved from environment with safe offline defaults.

Environment variables
  HERMES_LLM_PROVIDER   litellm | local | scripted | auto   (default auto)
  HERMES_LLM_MODEL      litellm model id (default anthropic/claude-opus-4-8)
  HERMES_LLM_TEMPERATURE
  HERMES_LLM_MAX_TOKENS
  HERMES_LLM_CACHE      1/0 — disk-cache responses (default 1)
  HERMES_LLM_TIMEOUT    seconds (default 60)
  HERMES_LLM_FALLBACK   local | none — fallback when a real call fails
  plus the usual provider keys: ANTHROPIC_API_KEY / OPENAI_API_KEY /
  AZURE_API_KEY(+AZURE_API_BASE/AZURE_API_VERSION) / POE_API_KEY /
  MINIMAX_API_KEY(+MINIMAX_API_BASE) / …

`auto` resolves to `litellm` only when the `litellm` package is importable
*and* a usable API key is present; otherwise it resolves to `local`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# Recommended Claude model ids (litellm-prefixed). Anthropic is the default
# for this system; any litellm-supported model id also works.
DEFAULT_MODEL = "anthropic/claude-opus-4-8"
RECOMMENDED_MODELS = {
    "anthropic/claude-opus-4-8": "Claude Opus 4.8 (推薦，深度推理)",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6 (均衡)",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku 4.5 (快速/批量)",
    "openai/gpt-4.1": "OpenAI GPT-4.1 (via litellm)",
    "openai/o4-mini": "OpenAI o4-mini (via litellm)",
    "azure/<deployment>": "Azure OpenAI（AZURE_API_KEY/AZURE_API_BASE/AZURE_API_VERSION）",
    "poe/Claude-Sonnet-4.5": "Poe（POE_API_KEY，OpenAI 兼容端點）",
    "minimax/MiniMax-M2": "MiniMax（MINIMAX_API_KEY，可用 MINIMAX_API_BASE 切換國內/國際站）",
}

# Env vars that, if present, indicate a usable provider key.
PROVIDER_KEY_ENV = [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY", "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY", "COHERE_API_KEY", "XAI_API_KEY",
    "POE_API_KEY", "MINIMAX_API_KEY",
]

# OpenAI-compatible gateways routed by model-id prefix. litellm has no native
# adapter for these, so we rewrite "<prefix>/<model>" → "openai/<model>" and
# point api_base/api_key at the gateway.
OPENAI_COMPATIBLE_ROUTES = {
    "poe": {"api_base": "https://api.poe.com/v1", "key_env": "POE_API_KEY",
            "base_env": "POE_API_BASE"},
    "minimax": {"api_base": "https://api.minimax.io/v1", "key_env": "MINIMAX_API_KEY",
                "base_env": "MINIMAX_API_BASE"},
}

# Per-task max_tokens floors. Long-form drafting (papers/reports) needs far
# more room than rule extraction; the floor is max()-ed with the user's
# HERMES_LLM_MAX_TOKENS so an explicit higher setting always wins.
TASK_MAX_TOKENS_FLOOR = {
    "paper": 8192,
    "report": 8192,
    "synthesize": 4096,
    "extract_rule": 2048,
    "critic": 2048,
}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def litellm_importable() -> bool:
    import importlib.util
    return importlib.util.find_spec("litellm") is not None


def has_provider_key() -> bool:
    return any(os.environ.get(k) for k in PROVIDER_KEY_ENV)


@dataclass
class LLMSettings:
    provider: str = "auto"
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 1536
    cache: bool = True
    timeout: int = 60
    fallback: str = "local"
    api_base: Optional[str] = None
    extra_keys_present: List[str] = field(default_factory=list)

    def resolve_backend(self) -> str:
        """Concrete backend after resolving `auto`."""
        if self.provider in ("litellm", "local", "scripted"):
            return self.provider
        # auto
        if litellm_importable() and has_provider_key():
            return "litellm"
        return "local"

    def max_tokens_for(self, task: Optional[str]) -> int:
        """Task-tiered token budget: floor per task, user setting wins if higher."""
        return max(self.max_tokens, TASK_MAX_TOKENS_FLOOR.get(task or "", 0))

    @property
    def reason(self) -> str:
        backend = self.resolve_backend()
        if backend == "litellm":
            return f"litellm 可用且檢測到 API key，使用模型 {self.model}"
        if self.provider == "local":
            return "顯式選擇 local 確定性後端"
        if not litellm_importable():
            return "未安裝 litellm，回退到 local 確定性後端（pip install litellm 以啟用真實模型）"
        if not has_provider_key():
            return "未檢測到任何 API key，回退到 local 確定性後端（設置 ANTHROPIC_API_KEY 等以啟用）"
        return "回退到 local 確定性後端"


def load_settings() -> LLMSettings:
    return LLMSettings(
        provider=os.environ.get("HERMES_LLM_PROVIDER", "auto").strip().lower(),
        model=os.environ.get("HERMES_LLM_MODEL", DEFAULT_MODEL).strip(),
        temperature=float(os.environ.get("HERMES_LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("HERMES_LLM_MAX_TOKENS", "1536")),
        cache=_env_bool("HERMES_LLM_CACHE", True),
        timeout=int(os.environ.get("HERMES_LLM_TIMEOUT", "60")),
        fallback=os.environ.get("HERMES_LLM_FALLBACK", "local").strip().lower(),
        api_base=os.environ.get("HERMES_LLM_API_BASE") or None,
        extra_keys_present=[k for k in PROVIDER_KEY_ENV if os.environ.get(k)],
    )
