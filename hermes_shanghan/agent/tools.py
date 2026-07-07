"""ToolRegistry — the single capability surface shared by the agent, the MCP
server and the OpenAI-compatible tool specs.

All tools are read-only and evidence-returning: each result carries clause_id
references so any downstream answer can be citation-checked. Patient-unsafe
operations are simply not exposed as tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .. import config
from ..schemas import read_jsonl


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]      # JSON schema
    func: Callable[..., Dict]

    def spec(self) -> Dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


class ToolRegistry:
    """Lazy-loads pipeline artifacts once, exposes 8 grounded tools."""

    def __init__(self):
        self._art = None
        self._clause_rag = None
        self._matcher = None
        self._tools: Dict[str, Tool] = {}
        self._register_all()

    # -- lazy resources -------------------------------------------------
    @property
    def art(self):
        if self._art is None:
            from ..orchestrator import Artifacts
            self._art = Artifacts()
        return self._art

    @property
    def clause_rag(self):
        if self._clause_rag is None:
            from ..rag.clause_rag import ClauseRAG
            self._clause_rag = ClauseRAG.load()
        return self._clause_rag

    @property
    def matcher(self):
        if self._matcher is None:
            from ..apps.doctor import FormulaMatcher
            self._matcher = FormulaMatcher(self.art.formula_rules, self.art.clause_store())
        return self._matcher

    # -- registration ---------------------------------------------------
    def _add(self, name, description, parameters, func):
        self._tools[name] = Tool(name, description, parameters, func)

    def _register_all(self):
        self._add(
            "shanghan_search",
            "檢索《傷寒論》原文條文（BM25+結構化過濾+關係擴展）。返回帶 clause_id 的條文命中。",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "症狀/方名/脈象/治法等檢索詞"},
                "top_k": {"type": "integer", "default": 6},
                "six_channel": {"type": "string", "description": "可選六經過濾，如 太陽病"},
                "formula": {"type": "string", "description": "可選方劑過濾"},
                "expand": {"type": "boolean", "default": False, "description": "關係圖譜擴展"}},
             "required": ["query"]},
            self._t_search)
        self._add(
            "shanghan_get_clause",
            "按條文號(1-398)或 clause_id 取條文全息：原文、實體標註、初始規則、條文關係。",
            {"type": "object", "properties": {
                "ref": {"type": "string", "description": "條文號或 SHL_SONGBEN_xxxx"}},
             "required": ["ref"]},
            self._t_get_clause)
        self._add(
            "shanghan_match_formula",
            "醫師端方證匹配：依症狀/脈象返回候選方證規則與原文證據（輔助性質，不替代臨床）。",
            {"type": "object", "properties": {
                "symptoms": {"type": "array", "items": {"type": "string"}},
                "pulse": {"type": "array", "items": {"type": "string"}},
                "six_channel": {"type": "string"},
                "top_k": {"type": "integer", "default": 5}},
             "required": ["symptoms"]},
            self._t_match)
        self._add(
            "shanghan_differential",
            "方證鑒別：給定 2-3 個方劑，返回多軸對比表與關鍵鑒別點及條文。",
            {"type": "object", "properties": {
                "formulas": {"type": "array", "items": {"type": "string"}}},
             "required": ["formulas"]},
            self._t_differential)
        self._add(
            "shanghan_six_channel",
            "六經規則：返回某經提綱、總括、亞型、主方、欲解時與禁忌/誤治條文。",
            {"type": "object", "properties": {
                "channel": {"type": "string", "description": "太陽病/陽明病/少陽病/太陰病/少陰病/厥陰病"}},
             "required": ["channel"]},
            self._t_six_channel)
        self._add(
            "shanghan_formula_rule",
            "方證規則：返回某方的核心證/兼證/脈象/組成/加減方/禁忌與支持條文。",
            {"type": "object", "properties": {
                "formula": {"type": "string"}},
             "required": ["formula"]},
            self._t_formula_rule)
        self._add(
            "shanghan_mistreatment",
            "誤治傳變圖譜：返回(誤治→變證→救治方→條文)路徑，可按關鍵詞過濾。",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "可選，如 誤下/結胸/火逆"}}},
            self._t_mistreatment)
        self._add(
            "shanghan_list_formulas",
            "列出規則庫中可用的方劑名稱（用於消歧或選擇）。",
            {"type": "object", "properties": {}},
            self._t_list_formulas)
        self._add(
            "shanghan_divergence_atlas",
            "注家分歧圖譜：9 部注本的對齊覆蓋、爭點條文榜、注家一致度矩陣與指紋；"
            "可按 clause_id 片段取單條的多注家記錄。",
            {"type": "object", "properties": {
                "clause": {"type": "string", "description": "可選 clause_id 片段，如 0012"}}},
            self._t_divergence)
        self._add(
            "shanghan_dose",
            "劑量計量層：某方的銖當量藥量比（學派無關）、三家折算總量與家族劑量演化邊；"
            "不給方名則返回全庫劑量摘要。",
            {"type": "object", "properties": {
                "formula": {"type": "string", "description": "可選方名，如 桂枝加芍藥湯"}}},
            self._t_dose)
        self._add(
            "shanghan_corpus_stats",
            "規則庫計量統計：條文/規則/關係/方證頻次/六經分佈等全庫數字（科研引用用）。",
            {"type": "object", "properties": {}},
            self._t_corpus_stats)
        self._add(
            "shanghan_eval_metrics",
            "客觀評測結果：遮方預測(LOCO)、醫案回放、證據接地率三大基準的當前指標與消融。",
            {"type": "object", "properties": {}},
            self._t_eval_metrics)

    # -- research-layer helpers -----------------------------------------
    @staticmethod
    def _research_json(name):
        import json
        p = config.RESEARCH_DIR / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _t_divergence(self, clause=None):
        a = self._research_json("commentary_divergence.json")
        if a is None:
            return {"tool": "shanghan_divergence_atlas",
                    "error": "分歧圖譜未生成：請先運行 pipeline"}
        if clause:
            rows = [r for r in a["clauses"] if clause in r["clause_id"]]
            return {"tool": "shanghan_divergence_atlas", "clause_filter": clause,
                    "book_coverage": a["book_coverage"], "clauses": rows[:10]}
        return {"tool": "shanghan_divergence_atlas",
                **{k: a[k] for k in ("n_books", "n_commentary_rules",
                                     "n_clauses_multi_commentator",
                                     "mean_term_divergence", "book_coverage",
                                     "agreement_matrix",
                                     "commentator_fingerprints")},
                "top_divergent_clauses": a["top_divergent_clauses"][:8]}

    def _t_dose(self, formula=None):
        ratios = self._research_json("dose_ratios.json")
        evo = self._research_json("dose_family_evolution.json")
        if ratios is None or evo is None:
            return {"tool": "shanghan_dose", "error": "劑量資產未生成：請先運行 pipeline"}
        if formula:
            f = next((x for x in ratios["formulas"] if x["formula"] == formula), None)
            edges = [e for e in evo["edges"]
                     if formula in (e["base"], e["modified"])]
            if f is None and not edges:
                return {"tool": "shanghan_dose", "error": f"無劑量數據：{formula}"}
            return {"tool": "shanghan_dose", "formula": formula,
                    "ratio": f, "evolution_edges": edges}
        summ = self._research_json("dose_summary.json") or {}
        return {"tool": "shanghan_dose", "note": ratios.get("note", ""),
                "summary": summ,
                "n_dose_only_edges": evo.get("n_dose_only_edges", 0)}

    def _t_corpus_stats(self):
        from collections import Counter
        rules = read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
        levels = Counter(r["autonomous_review"]["release_level"] for r in rules)
        formula_freq: Counter = Counter()
        channel: Counter = Counter()
        for c in self.art.clauses:
            if c.text_type != "original_clause":
                continue
            formula_freq.update(c.formula_names)
            if c.six_channel:
                channel[c.six_channel] += 1
        return {"tool": "shanghan_corpus_stats",
                "initial_rules": len(rules),
                "release_levels": dict(levels),
                "formula_pattern_rules": len(self.art.formula_rules),
                "differential_rules": len(self.art.differential_rules),
                "mistreatment_rules": len(self.art.mistreatment_rules),
                "variant_rules": len(self.art.variant_rules),
                "commentary_rules": len(self.art.commentary_rules),
                "top_formulas": formula_freq.most_common(12),
                "channel_clauses": channel.most_common()}

    def _t_eval_metrics(self):
        import json
        p = config.SHANGHAN_DIR / "eval" / "eval_summary.json"
        if not p.exists():
            return {"tool": "shanghan_eval_metrics",
                    "error": "評測未運行：請先執行 evaluate"}
        return {"tool": "shanghan_eval_metrics",
                **json.loads(p.read_text(encoding="utf-8"))}

    # -- tool implementations ------------------------------------------
    def _t_search(self, query, top_k=6, six_channel=None, formula=None, expand=False):
        hits = self.clause_rag.search(query, top_k=top_k, six_channel=six_channel,
                                      formula=formula, expand_relations=expand)
        return {"tool": "shanghan_search", "query": query, "hits": hits}

    def _t_get_clause(self, ref):
        c = self.clause_rag.get_clause(ref)
        if c is None:
            return {"tool": "shanghan_get_clause", "error": f"未找到條文 {ref}"}
        rules = [r for r in read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
                 if r["clause_id"] == c.clause_id]
        return {"tool": "shanghan_get_clause",
                "clause": {"clause_id": c.clause_id, "clause_number": c.clause_number,
                           "chapter": c.chapter, "six_channel": c.six_channel,
                           "clean_text": c.clean_text, "layer_label": "A 原文直述",
                           "symptoms": c.symptoms, "pulse": c.pulse,
                           "formulas": c.formula_names},
                "initial_rules": [{"id": r["initial_rule_id"], "type": r["rule_type"],
                                   "release": r["autonomous_review"]["release_level"]}
                                  for r in rules],
                "relations": self.clause_rag.related(c.clause_id, limit=6)}

    def _t_match(self, symptoms, pulse=None, six_channel=None, top_k=5):
        return self.matcher.match(symptoms=symptoms, pulse=pulse or [],
                                  six_channel=six_channel, top_k=top_k)

    def _t_differential(self, formulas):
        from ..textutil import normalize_query
        names = [normalize_query(f) for f in formulas]
        cands = [d for d in self.art.differential_rules if set(names) <= set(d.formulas)]
        if not cands:
            cands = [d for d in self.art.differential_rules
                     if len(set(names) & set(d.formulas)) >= 2]
        if not cands:
            from ..induce.differential import DifferentialInducer
            one = DifferentialInducer(self.art.formula_rules)._build_one(names, 999)
            cands = [one] if one else []
        if not cands:
            return {"tool": "shanghan_differential", "error": "無法構建該鑒別對",
                    "available_hint": "確認方名是否在規則庫中"}
        return {"tool": "shanghan_differential", "differential": cands[0].to_dict()}

    def _t_six_channel(self, channel):
        from ..textutil import normalize_query
        channel = normalize_query(channel)
        if not channel.endswith("病"):
            channel += "病"
        scr = next((r for r in self.art.six_channel_rules if r.six_channel == channel), None)
        if scr is None:
            return {"tool": "shanghan_six_channel", "error": f"未找到 {channel}",
                    "available": [r.six_channel for r in self.art.six_channel_rules]}
        d = scr.to_dict()
        d["tool"] = "shanghan_six_channel"
        return d

    def _t_formula_rule(self, formula):
        from ..textutil import normalize_query
        from ..lexicon import canonical_formula
        name = canonical_formula(normalize_query(formula))
        fpr = next((r for r in self.art.formula_rules if r.formula == name), None)
        if fpr is None:
            return {"tool": "shanghan_formula_rule", "error": f"未找到 {name} 的方證規則"}
        d = fpr.to_dict()
        d["tool"] = "shanghan_formula_rule"
        return d

    def _t_mistreatment(self, query=None):
        from ..textutil import normalize_query
        paths = self.art.mistreatment_rules
        if query:
            q = normalize_query(query)
            paths = [m for m in paths if q in m.mistreatment_type
                     or q in m.resulting_pattern
                     or any(q in f for f in m.rescue_formulas)] or paths
        return {"tool": "shanghan_mistreatment",
                "paths": [{"mistreatment": m.mistreatment_type,
                           "resulting_pattern": m.resulting_pattern,
                           "manifestations": m.manifestations[:6],
                           "rescue_formulas": m.rescue_formulas,
                           "clauses": m.supporting_clauses[:4],
                           "release_level": m.release_level} for m in paths[:12]]}

    def _t_list_formulas(self):
        return {"tool": "shanghan_list_formulas",
                "formulas": sorted(r.formula for r in self.art.formula_rules)}

    # -- access ---------------------------------------------------------
    def specs(self) -> List[Dict]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools)

    def call(self, name: str, arguments: Dict) -> Dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}", "available": self.names()}
        try:
            return tool.func(**(arguments or {}))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # never crash the agent on a tool error
            return {"error": f"tool {name} failed: {type(exc).__name__}: {exc}"}


class ScopedRegistry:
    """Least-privilege view of a registry: a dispatched subagent sees only
    the tools its subtask needs — smaller decision space for the model,
    smaller blast radius for a confused one."""

    def __init__(self, base: ToolRegistry, allowed: List[str]):
        self._base = base
        self._allowed = [n for n in allowed if n in base.names()]

    @property
    def art(self):
        return self._base.art

    def names(self) -> List[str]:
        return list(self._allowed)

    def specs(self) -> List[Dict]:
        return [s for s in self._base.specs()
                if s["function"]["name"] in self._allowed]

    def call(self, name: str, arguments: Dict) -> Dict:
        if name not in self._allowed:
            return {"error": f"tool out of scope: {name}",
                    "available": self.names()}
        return self._base.call(name, arguments)


_REGISTRY: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolRegistry()
    return _REGISTRY
