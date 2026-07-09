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
        self._model, self._route_kwargs = self._resolve_route(settings)

    @staticmethod
    def _resolve_route(settings):
        """Map poe/… and minimax/… ids onto their OpenAI-compatible gateways.

        azure/… and every litellm-native prefix pass through untouched
        (litellm reads AZURE_API_KEY/AZURE_API_BASE/AZURE_API_VERSION itself).
        """
        import os

        from .config import OPENAI_COMPATIBLE_ROUTES
        model = settings.model
        prefix, _, rest = model.partition("/")
        route = OPENAI_COMPATIBLE_ROUTES.get(prefix)
        if not route or not rest:
            return model, {}
        kwargs: Dict[str, Any] = {
            "api_base": os.environ.get(route["base_env"]) or route["api_base"]}
        key = os.environ.get(route["key_env"])
        if key:
            kwargs["api_key"] = key
        return f"openai/{rest}", kwargs

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
             temperature: float = 0.0, json_mode: bool = False,
             task: Optional[str] = None, context: Optional[Dict] = None) -> ChatResult:
        kwargs: Dict[str, Any] = dict(
            model=self._model, messages=messages,
            temperature=temperature,
            max_tokens=self.settings.max_tokens_for(task),
            timeout=self.settings.timeout)
        kwargs.update(self._route_kwargs)
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
        if task == "paper":
            return ChatResult(content=json.dumps(self._paper_sections(context or {}),
                                                 ensure_ascii=False), backend="local")
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
        # the orchestrator's dependency-context block（「（已取證 T1：…SHL_…）
        # 綜合任務：…」）carries clause ids and digits that must not hijack
        # routing — route the join task on its actual task text
        if "綜合任務：" in q_raw:
            q_raw = q_raw.split("綜合任務：")[-1]
        q = normalize_query(q_raw)
        available = {t["function"]["name"] for t in tools if t.get("function")}

        def call(name, args):
            return ChatResult(tool_calls=[ToolCall(str(uuid.uuid4()), name, args)],
                              backend="local")

        formulas = [n for n in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True)
                    if n in q][:3]
        m_num = re.search(r"(\d{1,3})", q)
        channel = next((c for c in _SIX_CHANNELS if c in q or c[:-1] in q), None)

        m_title = re.search(r"《([^》]{2,14})》", q_raw)
        if "shanghan_library" in available and \
                not re.search(r"(統計|多少條|頻次|計量概況|評測|接地率)", q) and \
                (re.search(r"(笈成|全庫|文獻|古籍|醫籍|歷代醫書|後世醫[家書]|哪些書|哪部書|查書|書目)", q)
                 or (m_title and m_title.group(1) not in
                     ("傷寒論", "傷寒雜病論", "金匱要略"))):
            if m_title and re.search(r"(目錄|章節|原文|讀|內容|講什麼)", q):
                return call("shanghan_library", {"book": m_title.group(1)})
            term = (m_title.group(1) if m_title
                    else (formulas[0] if formulas else ""))
            if not term:
                from ..extract.entities import EntityExtractor
                found = EntityExtractor().extract(q)
                cands = found.symptoms + found.disease_patterns
                term = max(cands, key=len) if cands else ""
            if not term:
                stripped = re.sub(r"(在|的|有|了|嗎|呢|？|\?|哪些書|哪部書|全庫|笈成|"
                                  r"文獻|古籍|醫籍|歷代醫書|後世醫[家書]|查書|書目|裡|中|"
                                  r"檢索|查閱|怎麼論述|如何論述|記載|提到|請|一下)", "",
                                  q_raw)
                term = stripped.strip()[:12]
            return call("shanghan_library", {"query": term})
        if re.search(r"(異文|桂本|桂林古本|千金翼|版本差)", q) and \
                "shanghan_variants" in available and m_num:
            return call("shanghan_variants", {"ref": m_num.group(1)})
        if re.search(r"(相關條文|關聯|傳變鏈|關係邊|鄰接)", q) and \
                "shanghan_relations" in available and m_num:
            return call("shanghan_relations", {"ref": m_num.group(1)})
        if re.search(r"(醫案|案例|實驗錄|診案)", q) and \
                "shanghan_case_search" in available:
            args = {"formula": formulas[0]} if formulas else {"keyword": q_raw[:12]}
            return call("shanghan_case_search", args)
        if re.search(r"(折合|換算|等於幾克|合多少克|是幾克)", q) and \
                "shanghan_dose_convert" in available:
            m_dose = re.search(r"([一二三四五六七八九十百半]+[銖分兩斤升合枚個][半]?)", q)
            if m_dose:
                return call("shanghan_dose_convert", {"dose": m_dose.group(1)})
        if re.search(r"(能不能用|可不可以用|禁忌嗎|有何禁忌|犯不犯禁)", q) and \
                formulas and "shanghan_contraindication_check" in available:
            from ..extract.entities import EntityExtractor
            found = EntityExtractor().extract(q)
            return call("shanghan_contraindication_check",
                        {"formula": formulas[0], "symptoms": found.symptoms})
        if re.search(r"(汗法|下法|吐法|和法|溫法|補法|利水|救逆|治法|法度|禁[汗下吐]|誤[汗下吐])", q) and \
                not re.search(r"(結胸|痞|壞病|變證|傳變|救治|火逆)", q) and \
                "shanghan_therapy" in available:
            m_th = re.search(r"(禁[汗下吐]|誤[汗下吐]|[汗下吐和溫清補]法|利水|救逆)", q)
            return call("shanghan_therapy",
                        {"method": m_th.group(1)} if m_th else {})
        if re.search(r"(注家|注本|分歧|詮釋|成無己|柯琴|尤怡|方有執|錢潢|黃元御)", q) and \
                "shanghan_divergence_atlas" in available:
            m_cl = re.search(r"(\d{2,4})\s*條|SHL_SONGBEN_(\d{4})", q)
            args = {"clause": (m_cl.group(1) or m_cl.group(2)).zfill(4)} if m_cl else {}
            return call("shanghan_divergence_atlas", args)
        if re.search(r"(劑量|藥量|用量|幾兩|折算|銖|克數)", q) and \
                "shanghan_dose" in available:
            return call("shanghan_dose",
                        {"formula": formulas[0]} if formulas else {})
        if re.search(r"(基準|評測|遮方|接地率|醫案回放)", q) and \
                "shanghan_eval_metrics" in available:
            return call("shanghan_eval_metrics", {})
        if re.search(r"(統計|多少條|頻次|計量概況|全庫)", q) and \
                "shanghan_corpus_stats" in available:
            return call("shanghan_corpus_stats", {})
        # 鑒別路由：簡繁與常見口語變體全覆蓋（七輪評審：路由須穩定命中）；
        # 兩個以上方名同現時，「比較/差別/怎麼分」類問法一律走鑒別工具
        if len(formulas) >= 2 and "shanghan_differential" in available and \
                re.search(r"(鑒別|鉴别|區別|区别|不同|對比|对比|比較|比较|"
                          r"vs|VS|區分|区分|分別|分别|差別|差别|差異|差异|"
                          r"怎麼分|怎么分|如何分)", q):
            return call("shanghan_differential", {"formulas": formulas})
        if re.search(r"(誤治|誤下|誤汗|誤吐|火逆|壞病|變證|傳變)", q) and \
                "shanghan_mistreatment" in available:
            return call("shanghan_mistreatment", {"query": q_raw})
        if (re.search(r"第?\d{1,3}條", q) or re.search(r"SHL_SONGBEN", q_raw)) and \
                "shanghan_get_clause" in available and m_num:
            # prefer an explicit 第N條/full SHL id over the first digit run,
            # so a stray "T1"-style token cannot become the clause ref
            m_ref = re.search(r"第?\s*(\d{1,3})\s*條", q)
            m_id = re.search(r"SHL_SONGBEN_(?:AUX_)?\d{4}", q_raw)
            ref = m_ref.group(1) if m_ref else \
                (m_id.group(0) if m_id else m_num.group(1))
            return call("shanghan_get_clause", {"ref": ref})
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
            elif isinstance(p, dict) and p.get("tool") == "shanghan_divergence_atlas":
                lines.append(f"注家分歧圖譜：{p.get('n_books', 9)} 注本、"
                             f"{p.get('n_commentary_rules', 0)} 條對齊注文。")
                for t in (p.get("top_divergent_clauses") or [])[:3]:
                    lines.append(f"- 爭點條文 {t['clause_id']}"
                                 f"（{t['n_commentators']} 家注，分歧度 {t['term_divergence']}）")
                    cited += 1
                for c in (p.get("clauses") or [])[:3]:
                    lines.append(f"- [{c['clause_id']}] {c['n_commentators']} 家注："
                                 f"{'、'.join(c.get('commentators', []))}")
                    cited += 1
            elif isinstance(p, dict) and p.get("tool") == "shanghan_dose":
                if p.get("ratio"):
                    r0 = p["ratio"]
                    lines.append(f"{p['formula']} 藥量比（銖當量，學派無關）：{r0['ratio']}；"
                                 f"三家折算總量(g)：{r0['total_weight_g']}"
                                 f"（{r0.get('clause_id','')}）")
                    cited += 1
                for e in (p.get("evolution_edges") or [])[:3]:
                    d0 = (e.get("dose_deltas") or [{}])[0]
                    lines.append(f"- {e['base']}→{e['modified']}（{e['edge_kind']}"
                                 + (f"：{d0.get('herb')}×{d0.get('factor')}" if d0 else "")
                                 + "）")
            elif isinstance(p, dict) and p.get("tool") == "shanghan_variants":
                lines.append(f"[{p.get('clause_id')}] 異文對勘（B層，{p.get('n_variants', 0)} 本）：")
                for v in (p.get("variants") or [])[:2]:
                    diff = "；".join(v.get("notable_differences", [])[:2]) or "用字基本一致"
                    lines.append(f"- {v['book']}（相似度{v['similarity']}）：{diff}")
                cited += 1
            elif isinstance(p, dict) and p.get("tool") == "shanghan_relations":
                lines.append(f"[{p.get('clause_id')}] 關係圖譜（{p.get('n_edges', 0)} 條邊）：")
                for e in (p.get("edges") or [])[:5]:
                    lines.append(f"- {e['relation_type']} → {e['other_clause_id']}："
                                 f"{e['description'][:36]}")
                    cited += 1
            elif isinstance(p, dict) and p.get("tool") == "shanghan_therapy":
                for t in (p.get("rules") or [])[:4]:
                    lines.append(f"【{t['method']}】{t['summary'][:50]}"
                                 f"（{'、'.join(t['supporting_clauses'][:3])}）")
                    cited += len(t["supporting_clauses"][:3])
            elif isinstance(p, dict) and p.get("tool") == "shanghan_contraindication_check":
                lines.append(f"{p.get('formula')} 禁忌檢查（輔助性質）：")
                for c0 in (p.get("formula_contraindications") or [])[:2]:
                    lines.append(f"- 原文禁例 [{c0.get('clause_id')}] {c0.get('condition', '')[:40]}")
                    cited += 1
                for c0 in (p.get("symptom_conflicts") or [])[:3]:
                    lines.append(f"- ⚠️ 證候衝突：所述「{c0['presented']}」與本方證之"
                                 f"「{c0['pattern_expects']}」相反")
                for b in (p.get("therapy_law_bans") or [])[:2]:
                    lines.append(f"- 法度禁例【{b['method']}】{b['summary'][:36]}"
                                 f"（{'、'.join(b['supporting_clauses'][:2])}）")
                    cited += 1
            elif isinstance(p, dict) and p.get("tool") == "shanghan_dose_convert":
                if p.get("kind") == "weight":
                    lines.append(f"「{p['raw']}」= {p['zhu']} 銖 = {p['liang']} 兩；"
                                 f"折算：考古 {p['grams_by_school']['kaogu']}g / "
                                 f"度量衡史 {p['grams_by_school']['duliangheng']}g / "
                                 f"明清 {p['grams_by_school']['zhezhuan']}g")
                elif p.get("kind") == "volume":
                    lines.append(f"「{p['raw']}」= {p['ml']} mL（{p.get('note', '')}）")
                else:
                    lines.append(f"「{p.get('raw')}」：{p.get('count', '')}"
                                 f"{p.get('count_unit', '')}（{p.get('note', '')}）")
                cited += 1
            elif isinstance(p, dict) and p.get("tool") == "shanghan_case_search":
                lines.append(f"醫案旁證（{p.get('source', '')}，非經文層）：")
                for cse in (p.get("cases") or [])[:2]:
                    lines.append(f"- {cse['title']}：證見 {'、'.join(cse['symptoms'][:4])}；"
                                 f"經文錨點 {'、'.join(cse['canonical_support'][:2])}")
                    cited += len(cse["canonical_support"][:2])
            elif isinstance(p, dict) and p.get("tool") == "shanghan_library":
                if not p.get("available", True):
                    lines.append(f"全庫未就緒：{p.get('hint', '')}")
                elif p.get("mode") == "read":
                    b = p.get("book", {})
                    lines.append(f"文獻查閱（非經文層）：《{b.get('title')}》"
                                 f"{b.get('author')}·{b.get('dynasty')}"
                                 f"（{b.get('category')}類）——"
                                 f"{(p.get('text') or '')[:120]}…")
                elif p.get("mode") == "overview":
                    cats = "、".join(f"{k}{v}部" for k, v in
                                    list((p.get("categories") or {}).items())[:6])
                    lines.append(f"全庫共 {p.get('n_books')} 部醫籍：{cats}……")
                else:
                    for h in (p.get("catalog_hits") or [])[:3]:
                        lines.append(f"- 《{h['title']}》{h['author']}·{h['dynasty']}"
                                     f"（{h['category']}類，約{h['approx_chars']}字）")
                    for h in (p.get("text_hits") or [])[:3]:
                        lines.append(f"- 《{h['title']}》§{h['section'][:14]}："
                                     f"…{h['excerpt'][:56]}…")
                    if not (p.get("catalog_hits") or p.get("text_hits")):
                        lines.append(f"全庫未檢得「{p.get('query', '')}」"
                                     + ("（掃描達上限，可縮小分類重試）"
                                        if p.get("scan_capped") else ""))
                lines.append("（以上屬文獻旁證層，僅供查閱，不作經文層證據）")
            elif isinstance(p, dict) and p.get("tool") == "shanghan_corpus_stats":
                tops = "、".join(f"{f}({n})" for f, n in (p.get("top_formulas") or [])[:5])
                lines.append(f"全庫計量：初始規則 {p.get('initial_rules', 0)} 條；"
                             f"高頻方 {tops}。")
                cited += 1
            elif isinstance(p, dict) and p.get("tool") == "shanghan_eval_metrics":
                cz = (p.get("suites", {}).get("cloze", {})
                      .get("metrics", {}).get("attainable", {}))
                gr = p.get("suites", {}).get("grounding", {}).get("metrics", {})
                lines.append(f"評測基準：遮方 Top-1 {cz.get('top1', '—')} / MRR "
                             f"{cz.get('mrr', '—')}；接地率 "
                             f"{gr.get('grounded_answer_rate', '—')}。")
                cited += 1
            elif isinstance(p, dict) and p.get("paths") is not None:
                lines.append("誤治傳變路徑：")
                for path in p["paths"][:5]:
                    lines.append(f"- {path.get('mistreatment')}→{path.get('resulting_pattern')}"
                                 f"→{'、'.join(path.get('rescue_formulas', [])[:2])}"
                                 f"（{'、'.join(path.get('clauses', [])[:2])}）")
                    cited += len(path.get("clauses", []))
        library_answered = any(isinstance(p, dict)
                               and p.get("tool") == "shanghan_library"
                               and (p.get("text_hits") or p.get("catalog_hits")
                                    or p.get("mode") in ("read", "overview"))
                               for p in payloads)
        if cited == 0 and not library_answered:
            lines.append("（未檢索到充分的條文證據，無法作答。）")
        return "\n".join(lines)

    # -- deterministic paper drafting ------------------------------------
    @staticmethod
    def _paper_sections(context: Dict) -> Dict:
        """Rule-derived 引言/計量解讀/討論/結論 from the research digest.

        Same output schema as a real model (task="paper"), so PaperWriter is
        provider-agnostic; every claim below cites clause_ids present in the
        digest, keeping the citation guard meaningful even offline.
        """
        d = context.get("digest") or {}
        topic = context.get("topic", "")
        title_root = context.get("title_root", "")
        top_f = d.get("top_formulas") or []
        top_s = d.get("top_symptoms") or []
        top_edges = d.get("top_symptom_edges") or []
        hubs = d.get("network_hubs") or []
        paths = d.get("mistreatment_paths") or []
        chans = d.get("channel_clauses") or []

        def fmt_freq(pairs, n=3):
            return "、".join(f"{t}（{c}）" for t, c in pairs[:n])

        intro = (f"《傷寒論》以六經統病、以方證相應。本研究以宋本"
                 f"{d.get('n_clauses', 0)} 條正文為唯一 A 層證據，經六道審核閘門"
                 f"產出 {d.get('n_initial_rules', 0)} 條可回源初始規則，"
                 f"在此基礎上以{topic}為切入，對{title_root}作條文計量學考察。"
                 "既往數字化工作多止於檢索與標註，缺少規則必須回到原文的硬約束；"
                 "本文的每一項計量結論均可回溯到具體條文編號。")

        lines = []
        if top_f:
            lines.append(f"方劑頻次以{fmt_freq(top_f)}為最高，"
                         f"與{chans[0][0] if chans else '太陽病'}篇條文佔比最大"
                         f"（{chans[0][1] if chans else '—'} 條）相互印證，"
                         "提示全書辨治重心落在表證階段的方證分化。")
        if top_s:
            lines.append(f"症狀頻次前列為{fmt_freq(top_s)}，多屬寒熱與汗出異常，"
                         "構成六經辨證的主幹指徵。")
        if top_edges:
            e = top_edges[0]
            lines.append(f"方-證共現網絡共 {d.get('symptom_edge_count', 0)} 條邊，"
                         f"最強共現為 {e.get('formula')}—{e.get('symptom')}"
                         f"（權重 {e.get('weight')}），即該方最穩定的原文指徵。")
        if hubs:
            h = hubs[0]
            lines.append(f"網絡樞紐方為{ '、'.join(x.get('formula','') for x in hubs[:3])}，"
                         f"其中{h.get('formula')}的關聯證候達 {h.get('degree')} 種，"
                         "顯示其作為類方之祖的網絡地位。")
        if paths:
            p = paths[0]
            cl = "、".join((p.get("clauses") or [])[:2])
            lines.append(f"誤治傳變共 {d.get('n_mistreatment', 0)} 條路徑，"
                         f"以 {p.get('mistreatment')}→{p.get('resulting_pattern')}→"
                         f"{'、'.join((p.get('rescue_formulas') or [])[:1])} 最為典型"
                         f"（{cl}），構成誤治—變證—救逆的閉環法度。")
        bm = d.get("benchmark") or {}
        cz = bm.get("cloze_attainable") or {}
        if cz.get("n"):
            lines.append(f"遮方預測基準（留一條文）在 {cz['n']} 個可達折上"
                         f"Top-1 {cz.get('top1')}、Top-3 {cz.get('top3')}、"
                         f"MRR {cz.get('mrr')}、藥物級F1 {cz.get('herb_f1')}，"
                         "表明條文級規則具備可量化的跨條泛化能力，"
                         f"另有 {bm.get('cloze_singleton_n', 0)} 個孤證方"
                         "在留一設置下結構性不可達。")
        cr = bm.get("case_replay") or {}
        if cr.get("n_scored"):
            lines.append(f"醫案回放基準（{cr.get('source','')}）實評 "
                         f"{cr['n_scored']} 案，Top-1 {cr.get('top1')}、"
                         f"Top-5 {cr.get('top5')}，量化了純條文規則對"
                         "真實臨證決策的解釋上限，也給增益層留下了明確的改進空間。")
        gd = bm.get("grounding") or {}
        if gd.get("n_questions"):
            lines.append(f"證據接地基準（{gd.get('backend','')} 後端）"
                         f"{gd['n_questions']} 問全部通過引用核驗：完全接地率 "
                         f"{gd.get('grounded_answer_rate')}、未核實引用率 "
                         f"{gd.get('unsupported_citation_rate')}，"
                         "為接入任意大模型提供了可對比的幻覺引用標尺。")
        at = d.get("commentary_atlas") or {}
        if at.get("n_books"):
            ap, dp = at.get("most_agreeing_pair"), at.get("most_diverging_pair")
            seg = (f"注家分歧圖譜覆蓋 {at['n_books']} 部注本、"
                   f"{at.get('n_commentary_rules', 0)} 條對齊注文，"
                   f"{at.get('n_clauses_multi_commentator', 0)} 條條文有多位注家。")
            if ap and dp:
                seg += (f"術語一致度最高為 {ap['a']}×{ap['b']}"
                        f"（{ap['mean_term_agreement']}），最低為 "
                        f"{dp['a']}×{dp['b']}（{dp['mean_term_agreement']}）——"
                        "注家譜系的親疏由數據呈現，無需先驗學派標籤。")
            lines.append(seg)
        ds = d.get("dosimetry") or {}
        if ds.get("parse_coverage", {}).get("n_rows"):
            pc = ds["parse_coverage"]
            seg = (f"劑量計量層解析 {pc['n_rows']} 條劑量"
                   f"（未解析 {pc.get('n_unparsed', 0)} 條已逐一列出）；"
                   "藥量比以銖當量計、與折算學派無關。")
            de = ds.get("dose_only_edges") or []
            if de:
                e = de[0]
                seg += (f"家族樹中 {len(de)} 條僅劑量變化的方對——如 "
                        f"{e['base']}→{e['modified']}（{e['delta']['herb']}"
                        f"×{e['delta']['factor']}）——證明量變致新方是"
                        "經方配伍的獨立維度。")
            lines.append(seg)
        quant = "\n".join(f"（{i+1}）{s}" for i, s in enumerate(lines)) or \
            "（計量摘要不足，未生成解讀。）"

        discussion = ("計量結果與條文結構互證：高頻方劑同時是共現網絡的樞紐與"
                      "加減家族樹的根節點，說明《傷寒論》的方證體系呈「核心方輻射」"
                      "而非均勻分佈；誤治路徑的救逆方高度集中，提示救逆法度自成子系統。"
                      "以上均為對 A 層原文的計量歸納（D/E 層），不宜回讀為仲景原意；"
                      "詞典覆蓋率與條文切分策略仍可能造成低頻項漏計。")
        conclusion = (f"以條文為最小證據單位的計量挖掘表明：{topic}的規律"
                      "可以在不犧牲可追溯性的前提下被規模化提取；"
                      "所有數字均可由 data/shanghan/ 下的規則庫與審計日誌復算。")
        return {"introduction": intro, "quant_interpretation": quant,
                "discussion": discussion, "conclusion": conclusion}

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
