"""Unified LLM client: provider selection, disk cache, usage tracking,
graceful fallback, and JSON/text convenience methods.

Never raises on a missing backend — `available` reports capability and every
method degrades to the deterministic `local` provider.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import cache as cache_mod
from .config import LLMSettings, load_settings
from .prompts import (critic_system_prompt, critic_user_prompt,
                      extract_system_prompt, extract_user_prompt,
                      paper_system_prompt, paper_user_prompt,
                      synth_system_prompt, synth_user_prompt)
from .providers import (ChatResult, LiteLLMProvider, LocalProvider,
                        ScriptedProvider, ToolCall)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


class LLMClient:
    def __init__(self, settings: Optional[LLMSettings] = None, provider=None):
        self.settings = settings or load_settings()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "cache_hits": 0, "errors": 0}
        if provider is not None:
            self._provider = provider
            self._backend = getattr(provider, "name", "scripted")
        else:
            self._backend = self.settings.resolve_backend()
            self._provider = self._build_provider(self._backend)

    def _build_provider(self, backend: str):
        if backend == "litellm":
            try:
                return LiteLLMProvider(self.settings)
            except Exception:
                self._backend = "local"
                return LocalProvider(self.settings)
        if backend == "scripted":
            return ScriptedProvider()
        return LocalProvider(self.settings)

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def available(self) -> bool:
        """True when a real (non-deterministic) model backs this client."""
        return self._backend == "litellm"

    @property
    def provider(self):
        return self._provider

    def status(self) -> Dict[str, Any]:
        return {"backend": self._backend, "model": self.settings.model,
                "available": self.available, "reason": self.settings.reason,
                "cache": self.settings.cache, "usage": dict(self.usage),
                "keys_present": self.settings.extra_keys_present}

    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
             temperature: Optional[float] = None, json_mode: bool = False,
             task: Optional[str] = None, context: Optional[Dict] = None,
             use_cache: bool = True) -> ChatResult:
        temp = self.settings.temperature if temperature is None else temperature
        # Batch mining (extract_rule/critic) is the most expensive path, so
        # task-based calls are cached too — the clause/rule content is fully
        # present in `messages`, which the key hashes.
        cacheable = (use_cache and self.settings.cache and self._backend == "litellm"
                     and temp == 0.0)
        key = None
        if cacheable:
            key = cache_mod.cache_key(self.settings.model, messages, tools, temp,
                                      task=task, json_mode=json_mode)
            hit = cache_mod.load(key)
            if hit is not None:
                self.usage["cache_hits"] += 1
                tcs = [ToolCall(t["id"], t["name"], t["arguments"])
                       for t in hit.get("tool_calls", [])]
                return ChatResult(content=hit.get("content", ""), tool_calls=tcs,
                                  usage=hit.get("usage", {}), backend=self._backend)
        try:
            res = self._provider.chat(messages, tools=tools, temperature=temp,
                                      json_mode=json_mode, task=task, context=context)
        except Exception as exc:  # real-model failure → graceful fallback
            self.usage["errors"] += 1
            if self.settings.fallback == "local" and self._backend == "litellm":
                res = LocalProvider(self.settings).chat(
                    messages, tools=tools, temperature=temp, json_mode=json_mode,
                    task=task, context=context)
                res.content = (res.content + f"\n\n（注：大模型調用失敗已回退 local："
                               f"{type(exc).__name__}）").strip()
            else:
                raise
        self.usage["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.usage[k] += res.usage.get(k, 0)
        if cacheable and key and not res.tool_calls:
            cache_mod.store(key, {"content": res.content, "tool_calls": [],
                                  "usage": res.usage})
        return res

    # -- convenience ----------------------------------------------------
    def complete(self, system: str, user: str, **kw) -> str:
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}], **kw).content

    def json_complete(self, system: str, user: str, *, task: Optional[str] = None,
                      context: Optional[Dict] = None) -> Dict[str, Any]:
        res = self.chat([{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        json_mode=True, task=task, context=context)
        return _extract_json(res.content)

    # -- high-level task helpers ---------------------------------------
    def extract_rules(self, clause, formula_names=None) -> Dict[str, Any]:
        from ..schemas import ShanghanClause
        cd = clause.to_dict() if isinstance(clause, ShanghanClause) else clause
        return self.json_complete(
            extract_system_prompt(),
            extract_user_prompt(cd["clause_id"], cd.get("chapter", ""),
                                cd.get("six_channel", ""), cd.get("clean_text", "")),
            task="extract_rule",
            context={"clause": cd, "formula_names": formula_names})

    def critic_review(self, clause, rule) -> Dict[str, Any]:
        from ..schemas import InitialRule, ShanghanClause
        cd = clause.to_dict() if isinstance(clause, ShanghanClause) else clause
        rd = rule.to_dict() if isinstance(rule, InitialRule) else rule
        return self.json_complete(
            critic_system_prompt(),
            critic_user_prompt(cd.get("clean_text", ""),
                               json.dumps(rd, ensure_ascii=False)),
            task="critic", context={"clause": cd, "rule": rd})

    def synthesize(self, question: str, evidence: List[Dict], role: str = "doctor",
                   max_span: int = 500) -> str:
        # full clause text (deduped by clause_id) — a truncated span starves
        # the model of the very evidence it must reason over
        seen: set = set()
        rows: List[str] = []
        for e in evidence:
            cid = e.get("clause_id", "?")
            if cid in seen:
                continue
            seen.add(cid)
            rows.append(f"- [{cid}|{e.get('layer_label','A 原文直述')}] "
                        f"{e.get('text', e.get('clean_text',''))[:max_span]}")
        block = "\n".join(rows)
        return self.chat(
            [{"role": "system", "content": synth_system_prompt(role)},
             {"role": "user", "content": synth_user_prompt(question, block)}],
            task="synthesize", context={"question": question, "evidence": evidence}).content

    def draft_paper(self, paper_type: str, title_root: str, topic: str,
                    digest: Dict[str, Any]) -> Dict[str, Any]:
        """引言/計量結果解讀/討論/結論 drafted from the research digest.

        Returns {} on empty/unparseable output; the writer falls back to its
        template sections. All prose must cite clause_ids and is citation-
        guarded by the caller.
        """
        return self.json_complete(
            paper_system_prompt(),
            paper_user_prompt(paper_type, title_root, topic, digest),
            task="paper",
            context={"paper_type": paper_type, "title_root": title_root,
                     "topic": topic, "digest": digest})


# module-level singleton ----------------------------------------------------
_CLIENT: Optional[LLMClient] = None


def get_client(force_reload: bool = False) -> LLMClient:
    global _CLIENT
    if _CLIENT is None or force_reload:
        _CLIENT = LLMClient()
    return _CLIENT


def set_client(client: LLMClient) -> None:
    """Inject a client (e.g. a scripted one in tests)."""
    global _CLIENT
    _CLIENT = client
