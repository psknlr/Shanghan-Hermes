"""LLM providers, all returning a normalized ChatResult.

  LiteLLMProvider  real models (Anthropic/OpenAI/… via litellm)
  LocalProvider    deterministic, rule-derived; drives the SAME tool-calling
                   loop as a real model so agent code is provider-agnostic
  ScriptedProvider queued responses for tests

The LocalProvider is what makes the system run offline: it picks a tool from
the question, then synthesizes a grounded answer from the tool results — a
real two-step ReAct, just with a deterministic "brain".
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    backend: str = ""
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
class LiteLLMProvider:
    name = "litellm"

    def __init__(self, settings):
        import litellm  # noqa: imported lazily; only when this backend is chosen
        self._litellm = litellm
        self.settings = settings
        litellm.drop_params = True  # tolerate provider-specific param gaps

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
             temperature: float = 0.0, json_mode: bool = False,
             task: Optional[str] = None, context: Optional[Dict] = None) -> ChatResult:
        kwargs: Dict[str, Any] = dict(
            model=self.settings.model, messages=messages,
            temperature=temperature, max_tokens=self.settings.max_tokens,
            timeout=self.settings.timeout)
        if self.settings.api_base:
            kwargs["api_base"] = self.settings.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._litellm.completion(**kwargs)
        msg = resp.choices[0].message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            tool_calls.append(ToolCall(id=tc.id or str(uuid.uuid4()),
                                       name=tc.function.name, arguments=args))
        usage = {}
        if getattr(resp, "usage", None):
            usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                     "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                     "total_tokens": getattr(resp.usage, "total_tokens", 0)}
        return ChatResult(content=msg.content or "", tool_calls=tool_calls,
                          usage=usage, backend="litellm", raw=resp)


# ---------------------------------------------------------------------------
class ScriptedProvider:
    """Returns queued ChatResults; for tests. Queue items may be ChatResult,
    a dict (→content/tool_calls) or a str (→content)."""
    name = "scripted"

    def __init__(self, queue: Optional[List[Any]] = None):
        self.queue: List[Any] = list(queue or [])
        self.calls: List[Dict] = []

    def push(self, item: Any):
        self.queue.append(item)

    def chat(self, messages, tools=None, temperature=0.0, json_mode=False,
             task=None, context=None) -> ChatResult:
        self.calls.append({"messages": messages, "tools": bool(tools), "task": task})
        if not self.queue:
            return ChatResult(content="", backend="scripted")
        item = self.queue.pop(0)
        if isinstance(item, ChatResult):
            item.backend = "scripted"
            return item
        if isinstance(item, dict):
            tcs = [ToolCall(tc.get("id", str(uuid.uuid4())), tc["name"],
                            tc.get("arguments", {})) for tc in item.get("tool_calls", [])]
            return ChatResult(content=item.get("content", ""), tool_calls=tcs,
                              backend="scripted")
        return ChatResult(content=str(item), backend="scripted")


# ---------------------------------------------------------------------------
_SIX_CHANNELS = ["太陽病", "陽明病", "少陽病", "太陰病", "少陰病", "厥陰病",
                 "霍亂病", "陰陽易差後勞復病"]


class LocalProvider:
    """Deterministic, rule-derived 'brain'. No network. Drives tool calls and
    synthesizes grounded answers so the agent loop runs identically offline."""
    name = "local"

    def __init__(self, settings=None):
        self.settings = settings

    # -- entry point ----------------------------------------------------
    def chat(self, messages, tools=None, temperature=0.0, json_mode=False,
             task=None, context=None) -> ChatResult:
        if task == "extract_rule":
            return ChatResult(content=json.dumps(self._extract(context or {}),
                                                 ensure_ascii=False), backend="local")
        if task == "critic":
            return ChatResult(content=json.dumps(self._critic(context or {}),
                                                 ensure_ascii=False), backend="local")
        if task == "synthesize":
            return ChatResult(content=self._synthesize(context or {}, messages),
                              backend="local")
        # agent tool-calling loop
        if tools:
            if not any(m.get("role") == "tool" for m in messages):
                return self._route_tool(messages, tools)
            return ChatResult(content=self._synthesize_from_tools(messages),
                              backend="local")
        # plain text fallback
        return ChatResult(content=self._synthesize({}, messages), backend="local")

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _last_user(messages) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        return ""

    def _route_tool(self, messages, tools) -> ChatResult:
        from ..textutil import normalize_query
        from .. import lexicon
        q_raw = self._last_user(messages)
        q = normalize_query(q_raw)
        available = {t["function"]["name"] for t in tools if t.get("function")}

        def call(name, args):
            return ChatResult(tool_calls=[ToolCall(str(uuid.uuid4()), name, args)],
                              backend="local")

        formulas = [n for n in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True)
                    if n in q][:3]
        m_num = re.search(r"(\d{1,3})", q)
        channel = next((c for c in _SIX_CHANNELS if c in q or c[:-1] in q), None)

        if len(formulas) >= 2 and "shanghan_differential" in available and \
                re.search(r"(鑒別|區別|不同|對比|vs|區分)", q):
            return call("shanghan_differential", {"formulas": formulas})
        if re.search(r"(誤治|誤下|誤汗|誤吐|火逆|壞病|變證|傳變)", q) and \
                "shanghan_mistreatment" in available:
            return call("shanghan_mistreatment", {"query": q_raw})
        if (re.search(r"第?\d{1,3}條", q) or re.search(r"SHL_SONGBEN", q_raw)) and \
                "shanghan_get_clause" in available and m_num:
            return call("shanghan_get_clause", {"ref": m_num.group(1)})
        if channel and "shanghan_six_channel" in available and \
                re.search(r"(六經|提綱|綱領|內部結構|主方|亞型|" + channel + ")", q):
            return call("shanghan_six_channel", {"channel": channel})
        if formulas and "shanghan_formula_rule" in available and \
                re.search(r"(方證|組成|加減|主治|要點|" + formulas[0] + ")", q):
            return call("shanghan_formula_rule", {"formula": formulas[0]})
        if re.search(r"(惡寒|發熱|無汗|汗出|脈|身疼|嘔|下利|口苦)", q) and \
                "shanghan_match_formula" in available:
            from ..extract.entities import EntityExtractor
            ex = EntityExtractor()
            res = ex.extract(q)
            if res.symptoms or res.pulse:
                return call("shanghan_match_formula",
                            {"symptoms": res.symptoms, "pulse": res.pulse})
        return call("shanghan_search", {"query": q_raw, "top_k": 6})

    def _synthesize_from_tools(self, messages) -> str:
        tool_payloads = []
        for m in messages:
            if m.get("role") == "tool":
                try:
                    tool_payloads.append(json.loads(m.get("content", "{}")))
                except Exception:
                    pass
        question = self._last_user(messages)
        return self._compose_answer(question, tool_payloads)

    def _synthesize(self, context: Dict, messages) -> str:
        question = context.get("question") or self._last_user(messages)
        evidence = context.get("evidence") or []
        if evidence:
            return self._compose_answer(question, [{"hits": evidence}])
        return ("（local 確定性後端）已根據規則庫與檢索作答；如需更自然的語言"
                "與深入推理，請配置 litellm 與 API key 後重試。")

    @staticmethod
    def _compose_answer(question: str, payloads: List[Dict]) -> str:
        lines: List[str] = ["（local 確定性後端：以下結論均回源條文，未調用外部大模型）", ""]
        cited = 0
        for p in payloads:
            if isinstance(p, dict) and p.get("matched_formula_patterns") is not None:
                lines.append("依方證匹配（僅供醫師參考，不替代臨床判斷）：")
                for m in p["matched_formula_patterns"][:3]:
                    ev = "、".join(e["clause_id"] for e in m.get("evidence", [])[:3])
                    lines.append(f"- {m['formula']}（{m.get('six_channel','')}，匹配度"
                                 f"{m.get('match_score')}）：{m.get('core_reason','')} 證據：{ev}")
                    cited += len(m.get("evidence", []))
            elif isinstance(p, dict) and p.get("differential") is not None:
                d = p["differential"]
                lines.append(f"鑒別：{' vs '.join(d.get('formulas', []))}")
                for disc in d.get("key_discriminators", [])[:5]:
                    lines.append(f"- {disc}")
                lines.append(f"證據條文：{'、'.join(d.get('supporting_clauses', [])[:5])}")
                cited += len(d.get("supporting_clauses", []))
            elif isinstance(p, dict) and p.get("tool") == "shanghan_formula_rule" \
                    and p.get("formula"):
                lines.append(f"【{p['formula']}方證】核心證："
                             f"{'、'.join(p.get('core_symptoms', [])[:6]) or '—'}；"
                             f"核心脈：{'、'.join(p.get('core_pulse', [])[:3]) or '—'}")
                if p.get("composition"):
                    herbs = "、".join(c["herb"] for c in p["composition"])
                    lines.append(f"組成（A 原文直述）：{herbs}")
                if p.get("modification_relations"):
                    lines.append("加減方：")
                    for m in p["modification_relations"][:8]:
                        lines.append(f"- {m.get('modified_formula')}："
                                     f"加 {m.get('added_herbs') or '—'}；減 {m.get('removed_herbs') or '—'}")
                lines.append(f"支持條文：{'、'.join(p.get('supporting_clauses', [])[:5])}")
                cited += len(p.get("supporting_clauses", []))
            elif isinstance(p, dict) and p.get("six_channel"):
                lines.append(f"【{p['six_channel']}】{p.get('summary','')}")
                lines.append(f"提綱：{p.get('outline_text','')}（{p.get('outline_clause_id','')}，A 原文直述）")
                if p.get("main_formulas"):
                    fs = "、".join(f["formula"] for f in p["main_formulas"][:6])
                    lines.append(f"主要方劑：{fs}")
                cited += 1
            elif isinstance(p, dict) and p.get("hits") is not None:
                lines.append("檢索到的相關條文（A 原文直述）：")
                for h in p["hits"][:5]:
                    lines.append(f"- [{h.get('clause_id')}] {h.get('text','')[:50]}…")
                    cited += 1
            elif isinstance(p, dict) and p.get("clause"):
                c = p["clause"]
                lines.append(f"[{c.get('clause_id')}] {c.get('clean_text','')}")
                cited += 1
            elif isinstance(p, dict) and p.get("paths") is not None:
                lines.append("誤治傳變路徑：")
                for path in p["paths"][:5]:
                    lines.append(f"- {path.get('mistreatment')}→{path.get('resulting_pattern')}"
                                 f"→{'、'.join(path.get('rescue_formulas', [])[:2])}"
                                 f"（{'、'.join(path.get('clauses', [])[:2])}）")
                    cited += len(path.get("clauses", []))
        if cited == 0:
            lines.append("（未檢索到充分的條文證據，無法作答。）")
        return "\n".join(lines)

    # -- deterministic 'LLM' extraction / critique ----------------------
    def _extract(self, context: Dict) -> Dict:
        """Rule-derived rules in the LLM output schema (then evidence-verified
        downstream — demonstrates the guard even on a 'dumb' model output)."""
        clause = context.get("clause")
        if clause is None:
            return {"rules": []}
        from ..extract.entities import EntityExtractor
        from ..extract.initial_rules import InitialRuleExtractor
        from ..schemas import ShanghanClause
        if isinstance(clause, dict):
            clause = ShanghanClause.from_dict(clause)
        ex = EntityExtractor(context.get("formula_names"))
        irs = InitialRuleExtractor(ex).extract_clause_rules(clause)
        out = []
        for r in irs:
            out.append({
                "rule_type": r.rule_type,
                "if_conditions": r.if_conditions,
                "then_conclusions": r.then_conclusions,
                "prescription_strength": r.prescription_strength,
                "evidence_span": r.evidence_span,
                "interpretation": r.interpretation,
                "interpretation_level": r.interpretation_level,
                "model_confidence": r.model_confidence,
            })
        return {"rules": out}

    def _critic(self, context: Dict) -> Dict:
        from ..review import critic as critic_mod
        from ..schemas import InitialRule, ShanghanClause
        rule = context.get("rule")
        clause = context.get("clause")
        if rule is None or clause is None:
            return {"verdict": "warn", "flags": ["local:missing_context"],
                    "rationale": "", "suggested_fix": ""}
        if isinstance(rule, dict):
            rule = InitialRule.from_dict(rule)
        if isinstance(clause, dict):
            clause = ShanghanClause.from_dict(clause)
        verdict, flags = critic_mod.criticize(rule, {clause.clause_id: clause})
        return {"verdict": verdict, "flags": flags,
                "rationale": "（local 規則批評器裁定）",
                "suggested_fix": ""}
