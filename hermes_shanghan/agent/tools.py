"""ToolRegistry — the single capability surface shared by the agent, the MCP
server and the OpenAI-compatible tool specs.

All tools are read-only and evidence-returning: each result carries clause_id
references so any downstream answer can be citation-checked. Patient-unsafe
operations are simply not exposed as tools.
"""
from __future__ import annotations

import copy
import json
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


# —— uniform result envelope ————————————————————————————————————
# Every successful tool result is stamped with its dominant evidence layer
# (A 原文直述／B 版本異文／C 注家解釋／D 後世歸納／E 模型推理；旁證=非經文層)
# and, where relevant, standing limitations — so binders/critics downstream
# never have to guess whether a payload is 原文 or 歸納.
TOOL_META: Dict[str, Dict] = {
    "shanghan_search": {"evidence_level": "A"},
    "shanghan_get_clause": {"evidence_level": "A"},
    "shanghan_match_formula": {
        "evidence_level": "D",
        "limitations": ["匹配分數為規則歸納（D層），證據錨定 A 層條文；不替代臨床判斷"]},
    "shanghan_differential": {
        "evidence_level": "D",
        "limitations": ["鑒別軸為跨條歸納（D層），關鍵鑒別點須回源 supporting_clauses"]},
    "shanghan_six_channel": {"evidence_level": "D",
                             "limitations": ["篇章級歸納，提綱原文屬 A 層"]},
    "shanghan_formula_rule": {"evidence_level": "D",
                              "limitations": ["方證規則為跨條歸納，組成/服法屬 A 層原文"]},
    "shanghan_mistreatment": {"evidence_level": "D"},
    "shanghan_list_formulas": {"evidence_level": "A"},
    "shanghan_divergence_atlas": {"evidence_level": "C"},
    "shanghan_dose": {"evidence_level": "A",
                      "limitations": ["藥量比為銖當量原文換算；折算克數依三家學派假設"]},
    "shanghan_corpus_stats": {"evidence_level": "D"},
    "shanghan_eval_metrics": {"evidence_level": "D"},
    "shanghan_variants": {"evidence_level": "B"},
    "shanghan_relations": {"evidence_level": "D"},
    "shanghan_therapy": {"evidence_level": "D"},
    "shanghan_contraindication_check": {"evidence_level": "D"},
    "shanghan_dose_convert": {"evidence_level": "A"},
    "shanghan_case_search": {"evidence_level": "旁證"},
    "shanghan_library": {"evidence_level": "旁證"},
    "shanghan_hypotheses": {
        "evidence_level": "D",
        "limitations": ["多假設分析為規則歸納（D層），置信度為啟發式評分；不替代臨床判斷"]},
    "shanghan_trace": {
        "evidence_level": "mixed",
        "limitations": ["溯源鏈混合 A 原文/B 異文/C 注家/D 歸納/引文邊/計量，"
                        "整體標 mixed，逐節層級見 section_evidence_levels；"
                        "學派歸屬與方證觀點命題屬後世歸納（posthoc_induction）"]},
    "shanghan_citation_network": {
        "evidence_level": "D",
        "limitations": ["計量指標由逐字引文邊確定性推導；語料最晚傳播層為民國，"
                        "現代引用需經 modern 接口導入"]},
    "shanghan_herb_profile": {
        "evidence_level": "A-derived",
        "limitations": ["原始事實取自 A 層（組成/條文/劑量寫法），配伍共現與"
                        "頻次排序屬確定性派生統計，非原文直述；"
                        "藥性功效解釋屬本草層未隨庫，不編造"]},
    "shanghan_formula_explain": {
        "evidence_level": "mixed",
        "limitations": ["一站式檔案混合 A/C/D 層與引文邊，逐節層級見 "
                        "section_evidence_levels；四層症狀口徑見 symptom_layers.note"]},
    "shanghan_intake": {
        "evidence_level": "D",
        "limitations": ["僅為就診信息整理（確定性詞表抽取），不構成診斷；"
                        "現代口語映射表透明可審"]},
    "shanghan_adjudicate": {
        "evidence_level": "D",
        "limitations": ["三態裁決為確定性規則（評分差距+反證+禁忌），核心是"
                        "說明「為什麼還不能定方」；不替代臨床判斷"]},
    "shanghan_conflict_audit": {
        "evidence_level": "D",
        "limitations": ["衝突判定基於互斥證對與方證規則，條文可回源；"
                        "改判候選僅為定位提示，不構成處方建議"]},
    "shanghan_mistreatment_simulate": {
        "evidence_level": "D",
        "limitations": ["單步路徑逐條錨定原文；多步鏈為組合視圖（假設路徑），"
                        "非原文連續敘述"]},
}

_RELEASE_CONFIDENCE = {"gold": 0.9, "silver": 0.75, "bronze": 0.6}

# patient-mode hard isolation: only reading/explaining the classics is
# exposed — no formula matching, no composition/dose, no therapy selection.
# This is registry-level enforcement, independent of prompts and redaction.
PATIENT_SAFE_TOOLS: List[str] = [
    "shanghan_search", "shanghan_get_clause", "shanghan_six_channel",
    "shanghan_relations", "shanghan_variants", "shanghan_divergence_atlas",
    "shanghan_corpus_stats", "shanghan_eval_metrics", "shanghan_library",
    "shanghan_intake",   # 就診信息整理：無方/無劑量/無診斷，患者端安全
]


class ToolRegistry:
    """Lazy-loads pipeline artifacts once, exposes the grounded tool surface
    (see `_register_all`; every result carries clause_id evidence)."""

    def __init__(self, cache_size: int = 256):
        self._art = None
        self._clause_rag = None
        self._matcher = None
        self._tools: Dict[str, Tool] = {}
        # (tool, canonical-args) → result cache: repeated retrieval within a
        # session/orchestration is free and reproducible
        self._cache: Dict[str, Dict] = {}
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
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
        self._add(
            "shanghan_variants",
            "版本異文（B層）：某條文在桂林古本/千金翼方版的對齊異文與用字差異。",
            {"type": "object", "properties": {
                "ref": {"type": "string", "description": "條文號或 clause_id"}},
             "required": ["ref"]},
            self._t_variants)
        self._add(
            "shanghan_relations",
            "條文關係圖譜遍歷：某條文的鄰接邊（同方族/鑒別/誤治傳變/禁忌/傳變/次序），"
            "支持按關係類型過濾——用於多跳推理與傳變鏈追蹤。",
            {"type": "object", "properties": {
                "ref": {"type": "string", "description": "條文號或 clause_id"},
                "relation_type": {"type": "string",
                                  "description": "可選：same_formula_family/differential/"
                                                 "mistreatment_transformation/transmission/"
                                                 "contraindication/sequence"}},
             "required": ["ref"]},
            self._t_relations)
        self._add(
            "shanghan_therapy",
            "治法規則：汗/吐/下/和/溫/補/救逆/利水的適應指徵、代表方、禁例與誤施之變。",
            {"type": "object", "properties": {
                "method": {"type": "string",
                           "description": "可選，如 汗法/下法/禁汗/誤下；不填返回總覽"}}},
            self._t_therapy)
        self._add(
            "shanghan_contraindication_check",
            "禁忌檢查（複合推理）：給定方劑與病人證候，返回該方原文禁忌、證候與方證的"
            "衝突（如無汗 vs 桂枝湯）及相關治法禁例——輔助性質，不替代臨床判斷。",
            {"type": "object", "properties": {
                "formula": {"type": "string"},
                "symptoms": {"type": "array", "items": {"type": "string"}}},
             "required": ["formula"]},
            self._t_contra_check)
        self._add(
            "shanghan_dose_convert",
            "漢制劑量換算計算器（確定性）：解析「三兩」「一兩十六銖」「半升」等劑量，"
            "返回銖當量與三家折算克數/毫升——避免模型心算錯誤。",
            {"type": "object", "properties": {
                "dose": {"type": "string", "description": "如 三兩 / 一兩半 / 半升 / 十二枚"}},
             "required": ["dose"]},
            self._t_dose_convert)
        self._add(
            "shanghan_case_search",
            "醫案檢索：《經方實驗錄》(1937 曹穎甫) 真實診案，按方名或關鍵詞查找；"
            "醫案屬旁證（非經文層），結果自動附該方的經文支持條文作錨點。",
            {"type": "object", "properties": {
                "formula": {"type": "string", "description": "可選方名"},
                "keyword": {"type": "string", "description": "可選關鍵詞（症狀/敘述）"},
                "top_k": {"type": "integer", "default": 3}},
             "required": []},
            self._t_case_search)
        self._add(
            "shanghan_hypotheses",
            "多假設方證分析（醫師/教學端）：依症狀脈象返回並列候選方證假設，"
            "每個假設帶支持證據/反證/缺失關鍵證，並生成鑒別追問；"
            "證據不足時建議先補充四診信息而非給單一答案。",
            {"type": "object", "properties": {
                "symptoms": {"type": "array", "items": {"type": "string"}},
                "pulse": {"type": "array", "items": {"type": "string"}},
                "six_channel": {"type": "string"},
                "top_k": {"type": "integer", "default": 4}},
             "required": ["symptoms"]},
            self._t_hypotheses)
        self._add(
            "shanghan_library",
            "中醫笈成全庫快速查閱（800+ 部醫籍，文獻旁證層/非經文層）：按書名/作者/"
            "朝代/分類檢索編目；按原文詞句全文檢索（返回書·章節定位的摘錄）；"
            "或按書名+章節閱讀原書。庫未下載時提示 `library fetch`。",
            {"type": "object", "properties": {
                "query": {"type": "string",
                          "description": "檢索詞：書名/作者（編目）或原文詞句（全文）"},
                "book": {"type": "string", "description": "可選書名——直接閱讀該書"},
                "section": {"type": "string", "description": "可選章節名（配合 book）"},
                "category": {"type": "string",
                             "description": "可選分類過濾：醫案/方書/本草/溫病/傷寒…"},
                "top_k": {"type": "integer", "default": 5}},
             "required": []},
            self._t_library)
        self._add(
            "shanghan_trace",
            "深度溯源鏈：條文（原文→異文→注家→歷代引用→計量）、方劑（首見→組成"
            "→類方劑量演化→方名傳播）、方證觀點（原文直述檢驗→注家首倡時間線→"
            "學派立場）、注家（學派/指紋/被轉引樞紐度）、學派（範式/一致度）、"
            "任意文本回源。多觀點並存，證據分級標註。",
            {"type": "object", "properties": {
                "query_type": {"type": "string",
                               "enum": ["clause", "formula", "claim",
                                        "school", "commentator", "text",
                                        "quote", "term", "dispute", "compare"],
                               "description": "溯源對象類型；quote=誤引檢測；dispute=注家爭議結構化（條文號）；compare=學派/注家比較（A vs B）"},
                "ref": {"type": "string",
                        "description": "條文號/方名/觀點關鍵詞/注家名/學派名/原文片段"}},
             "required": ["query_type", "ref"]},
            self._t_trace)
        self._add(
            "shanghan_citation_network",
            "學術計量網絡（確定性科學計量）：歷代著作→條文引文邊的引用網絡、"
            "被引最多條文、共引條文對、著作文獻耦合、朝代時間切片、突現分析、"
            "主路徑。可選 target（條文號或方名）返回該對象的傳播計量；"
            "scope 控制被引榜範圍（canonical=正文398條[默認]/auxiliary=輔助篇章/all）。",
            {"type": "object", "properties": {
                "target": {"type": "string",
                           "description": "可選：條文號（如 12）或方名（如 桂枝湯）"},
                "scope": {"type": "string",
                          "enum": ["canonical", "auxiliary", "all"],
                          "default": "canonical",
                          "description": "被引榜範圍：正文/輔助篇章/混排"},
                "top_k": {"type": "integer", "default": 8}},
             "required": []},
            self._t_citation_network)
        self._add(
            "shanghan_herb_profile",
            "藥證檔案（藥解）：單味藥的出現方劑、條文、劑量寫法、配伍共現"
            "網絡（同方共現計數）。只含可計算事實，不編造藥性/功效解釋。",
            {"type": "object", "properties": {
                "herb": {"type": "string", "description": "藥名，如 桂枝"}},
             "required": ["herb"]},
            self._t_herb_profile)
        self._add(
            "shanghan_formula_explain",
            "方解檔案（一站式）：首見條文、三層症狀口徑（首見方證/全書聚合/"
            "特殊上下文）、組成劑量比、煎服法、禁忌、類方鑒別、方名傳播、"
            "方證觀點分級。",
            {"type": "object", "properties": {
                "formula": {"type": "string", "description": "方名，如 桂枝湯"}},
             "required": ["formula"]},
            self._t_formula_explain)
        self._add(
            "shanghan_intake",
            "四診信息採集：把患者自然敘述整理為結構化四診表（主訴/病程/寒熱/"
            "汗/渴飲/二便/胸脅腹/痛/眠/舌/脈/誤治史/藥後反應）+ 缺失關鍵信息 "
            "+ 追問建議。只整理信息，不做匹配不做診斷（患者端安全）。",
            {"type": "object", "properties": {
                "text": {"type": "string", "description": "患者的自然語言敘述"}},
             "required": ["text"]},
            self._t_intake)
        self._add(
            "shanghan_adjudicate",
            "方證多假設裁決（醫師/教學端）：候選方證各附支持證/反證/缺失證/"
            "禁忌衝突，輸出三態裁決（傾向A/傾向B/不能裁決）+「為什麼還不能"
            "定方」+ 三個關鍵追問。",
            {"type": "object", "properties": {
                "symptoms": {"type": "array", "items": {"type": "string"}},
                "pulse": {"type": "array", "items": {"type": "string"}},
                "six_channel": {"type": "string"}},
             "required": ["symptoms"]},
            self._t_adjudicate)
        self._add(
            "shanghan_conflict_audit",
            "方證衝突審計（醫師端）：候選方 × 呈現表現 → 衝突項（核心證/兼證"
            "衝突）/衝突條文/是否禁忌/改判候選/應補問。比 top-k 匹配更安全的"
            "定位方式。",
            {"type": "object", "properties": {
                "formula": {"type": "string"},
                "symptoms": {"type": "array", "items": {"type": "string"}},
                "pulse": {"type": "array", "items": {"type": "string"}}},
             "required": ["formula", "symptoms"]},
            self._t_conflict_audit)
        self._add(
            "shanghan_mistreatment_simulate",
            "誤治傳變路徑模擬：某經 × 某誤治 → 變證分支 → 救逆方 → 條文依據；"
            "多步鏈為組合視圖並如實標註（每步錨定原文，鏈非原文連續敘述）。",
            {"type": "object", "properties": {
                "channel": {"type": "string", "default": "太陽病"},
                "mistreatment": {"type": "string",
                                 "description": "誤汗/誤下/誤吐/火逆；留空列全部"},
                "steps": {"type": "integer", "default": 1}},
             "required": []},
            self._t_mistreatment_simulate)

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
            res = self.resolve_formula(formula)
            dose_names = {x["formula"] for x in ratios["formulas"]} \
                | {n for e in evo["edges"] for n in (e["base"], e["modified"])}
            if res["resolved"]:
                formula = res["resolved"]
            else:
                # the dose layer covers formulas that may lack a pattern
                # rule — an exact dose-layer name must not be blocked by
                # rule-inventory disambiguation
                from .. import lexicon
                from ..textutil import normalize_query
                exact = lexicon.canonical_formula(normalize_query(formula))
                if exact in dose_names:
                    formula = exact
                else:
                    return self._ambiguous_payload("shanghan_dose", res)
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

    def _t_variants(self, ref):
        c = self.clause_rag.get_clause(ref)
        if c is None:
            return {"tool": "shanghan_variants", "error": f"未找到條文 {ref}"}
        rows = [{"book": v.variant_book, "similarity": v.similarity,
                 "variant_text": v.variant_text[:200],
                 "notable_differences": v.notable_differences}
                for v in self.art.variant_rules if v.clause_id == c.clause_id]
        return {"tool": "shanghan_variants", "clause_id": c.clause_id,
                "base_text": c.clean_text, "n_variants": len(rows),
                "variants": rows}

    def _relations_all(self):
        if not hasattr(self, "_rel_cache"):
            self._rel_cache = read_jsonl(config.RELATION_DIR / "clause_relations.jsonl")
        return self._rel_cache

    def _t_relations(self, ref, relation_type=None):
        c = self.clause_rag.get_clause(ref)
        if c is None:
            return {"tool": "shanghan_relations", "error": f"未找到條文 {ref}"}
        edges = []
        for r in self._relations_all():
            if r["relation_type"] in ("variant", "commentary_support"):
                continue        # B/C 層各有專用工具
            if relation_type and r["relation_type"] != relation_type:
                continue
            if c.clause_id in (r["source_clause_id"], r["target_clause_id"]):
                other = r["target_clause_id"] if r["source_clause_id"] == c.clause_id \
                    else r["source_clause_id"]
                oc = self.art.clause_store().get(other)
                edges.append({"relation_type": r["relation_type"],
                              "other_clause_id": other,
                              "other_text": oc.clean_text[:60] if oc else "",
                              "description": r["description"]})
        return {"tool": "shanghan_relations", "clause_id": c.clause_id,
                "n_edges": len(edges), "edges": edges[:15]}

    def _t_therapy(self, method=None):
        rules = self.art.therapy_rules
        if method:
            rows = [t for t in rules if method in t.therapy_method]
            if not rows:
                return {"tool": "shanghan_therapy",
                        "error": f"無此治法：{method}",
                        "available": sorted({t.therapy_method for t in rules})}
        else:
            rows = rules
        return {"tool": "shanghan_therapy", "n_rules": len(rows),
                "rules": [{"method": t.therapy_method, "polarity": t.polarity,
                           "summary": t.summary,
                           "indications": t.indications[:8],
                           "representative_formulas": t.representative_formulas[:6],
                           "supporting_clauses": t.supporting_clauses[:6]}
                          for t in rows[:12]]}

    def _t_contra_check(self, formula, symptoms=None):
        from .. import lexicon
        from ..textutil import normalize_query
        res = self.resolve_formula(formula)
        if not res["resolved"]:
            return self._ambiguous_payload("shanghan_contraindication_check", res)
        formula = res["resolved"]
        rule = next((r for r in self.art.formula_rules if r.formula == formula), None)
        if rule is None:
            return {"tool": "shanghan_contraindication_check",
                    "error": f"無方證規則：{formula}"}
        symptoms = [normalize_query(s) for s in (symptoms or []) if s.strip()]
        pattern = rule.core_symptoms + rule.associated_symptoms
        conflicts = []
        for s in symptoms:
            for a, b in lexicon.CONTRADICTORY_SYMPTOMS:
                if (s == a and b in pattern) or (s == b and a in pattern):
                    conflicts.append({"presented": s,
                                      "pattern_expects": b if s == a else a})
        therapy_bans, seen_methods = [], set()
        for t in self.art.therapy_rules:
            if t.polarity != "contraindicated" or t.therapy_method in seen_methods:
                continue
            base = t.therapy_method.lstrip("禁")          # 禁汗 → 汗
            indicated = next((x for x in self.art.therapy_rules
                              if x.therapy_method.startswith(base)
                              and x.polarity == "indicated"), None)
            if indicated and formula in indicated.representative_formulas:
                seen_methods.add(t.therapy_method)
                therapy_bans.append({"method": t.therapy_method,
                                     "summary": t.summary,
                                     "supporting_clauses": t.supporting_clauses[:4]})
        return {"tool": "shanghan_contraindication_check",
                "formula": formula,
                "formula_contraindications": rule.contraindications[:5],
                "symptom_conflicts": conflicts,
                "therapy_law_bans": therapy_bans,
                "notice": "僅為古籍禁忌法度輔助檢查，不能替代醫師臨床判斷。"}

    def _t_dose_convert(self, dose):
        from ..apps.dosimetry import SCHOOLS, SHENG_ML, parse_dose
        p = parse_dose(dose)
        if p["kind"] == "none":
            return {"tool": "shanghan_dose_convert", "raw": dose,
                    "error": "無法解析劑量表達式（支持 兩/銖/分/斤/升/合/枚/個 等漢制單位）"}
        out = {"tool": "shanghan_dose_convert", "raw": dose, "kind": p["kind"]}
        if p["kind"] == "weight":
            out["zhu"] = p["zhu"]
            out["liang"] = round(p["zhu"] / 24, 4)
            out["grams_by_school"] = p["grams"]
            out["schools"] = {k: v["label"] for k, v in SCHOOLS.items()}
        elif p["kind"] == "volume":
            out["ge"] = p["ge"]
            out["ml"] = p["ml"]
            out["note"] = f"1升≈{SHENG_ML}mL（漢代量器實測）"
        elif p["kind"] == "count":
            out["count"] = p["count"]
            out["count_unit"] = p.get("count_unit", "")
            out["note"] = "計數類不經未考證的單枚質量假設換算"
        return out

    def _cases_all(self):
        if not hasattr(self, "_case_cache"):
            from ..eval.cases import parse_cases
            from ..extract.entities import EntityExtractor
            try:
                self._case_cache, _ = parse_cases(EntityExtractor())
            except FileNotFoundError:
                self._case_cache = []
        return self._case_cache

    def _t_case_search(self, formula=None, keyword=None, top_k=3):
        from .. import lexicon
        from ..textutil import normalize_query
        cases = self._cases_all()
        if not cases:
            return {"tool": "shanghan_case_search", "error": "醫案語料不可用"}
        if formula:
            formula = lexicon.canonical_formula(normalize_query(formula))
            cases = [c for c in cases if c["gold"] == formula]
        if keyword:
            kw = normalize_query(keyword)
            cases = [c for c in cases
                     if kw in normalize_query(c["title"])
                     or kw in "、".join(c["symptoms"])]
        rows = []
        for c in cases[:top_k]:
            anchor = next((r.supporting_clauses[:3] for r in self.art.formula_rules
                           if r.formula == c["gold"]), [])
            rows.append({"title": c["title"], "formula": c["gold"],
                         "symptoms": c["symptoms"][:8], "pulse": c["pulse"][:3],
                         "canonical_support": anchor})
        return {"tool": "shanghan_case_search",
                "source": "經方實驗錄（1937，曹穎甫）",
                "evidence_layer": "醫案旁證（非經文層；經文錨點見 canonical_support）",
                "n_matched": len(cases), "cases": rows}

    def _t_library(self, query=None, book=None, section=None,
                   category=None, top_k=5):
        from ..corpus import library
        if not library.ensure_available(verbose=False):
            return {"tool": "shanghan_library", "available": False,
                    "hint": "全庫未下載：運行 `python3 -m hermes_shanghan "
                            "library fetch`（約 69MB，自動校驗/解壓/建索引），"
                            "或設 HERMES_LIBRARY_AUTOFETCH=1 由首次調用自動獲取"}
        lib = library.Library()
        note = "文獻旁證層（非經文層）：出處僅供文獻查閱，不進入證據閘門"
        if book:
            out = lib.read(book, section=section or "", max_chars=2400)
            if "error" in out:
                return {"tool": "shanghan_library", "available": True,
                        "evidence_layer": note, **out}
            return {"tool": "shanghan_library", "available": True,
                    "evidence_layer": note, "mode": "read", **out,
                    "toc": [t["title"] for t in lib.toc(book)][:30]}
        q = (query or "").strip()
        if not q:
            return {"tool": "shanghan_library", "available": True,
                    "mode": "overview", "categories": lib.categories(),
                    "n_books": lib.catalog["n_books"]}
        catalog_hits = lib.search(q, category=category or "", limit=top_k)
        text = lib.grep(q, category=category or "", limit=top_k) \
            if len("".join(q.split())) >= 2 else {}
        return {"tool": "shanghan_library", "available": True,
                "evidence_layer": note, "mode": "search", "query": q,
                "catalog_hits": catalog_hits,
                "text_hits": text.get("hits", []),
                "n_text_hits": text.get("n_hits", 0),
                "scan_capped": text.get("scan_capped", False)}

    def _t_herb_profile(self, herb):
        from ..apps.herbal import herb_profile
        return {"tool": "shanghan_herb_profile", **herb_profile(herb)}

    def _t_intake(self, text):
        from ..apps.bianzheng import intake_parse
        return {"tool": "shanghan_intake", **intake_parse(text)}

    def _t_adjudicate(self, symptoms, pulse=None, six_channel=None):
        from ..apps.bianzheng import adjudicate
        return {"tool": "shanghan_adjudicate",
                **adjudicate(symptoms, pulse=pulse,
                             six_channel=six_channel or "", registry=self)}

    def _t_conflict_audit(self, formula, symptoms, pulse=None):
        from ..apps.bianzheng import conflict_audit
        return {"tool": "shanghan_conflict_audit",
                **conflict_audit(formula, symptoms, pulse=pulse, registry=self)}

    def _t_mistreatment_simulate(self, channel="太陽病", mistreatment="", steps=1):
        from ..apps.bianzheng import mistreatment_simulate
        return {"tool": "shanghan_mistreatment_simulate",
                **mistreatment_simulate(channel, mistreatment, steps)}

    def _t_formula_explain(self, formula):
        from ..trace.chains import formula_explain
        return {"tool": "shanghan_formula_explain", **formula_explain(formula)}

    def _t_trace(self, query_type, ref):
        from ..trace.chains import trace_dispatch
        res = trace_dispatch(query_type, ref)
        if "error" in res:
            return {"tool": "shanghan_trace", **res}
        return {"tool": "shanghan_trace", "trace": res}

    def _t_citation_network(self, target=None, top_k=8, scope="canonical"):
        from ..textutil import fold_variants, normalize_query
        from ..trace import builder as trace_builder
        net = trace_builder.load_network()
        # scope 一致性契約（方案 A）：時間切片/共引/突現/主路徑全部按 scope
        # 過濾，canonical 輸出中不出現任何 AUX 條文（trace-audit-scope 可驗）
        slice_key = {"canonical": "top_canonical", "auxiliary": "top_auxiliary",
                     "all": "top_clauses"}.get(scope, "top_canonical")
        slices = [{"dynasty": s["dynasty"], "n_works": s["n_works"],
                   "n_edges": s.get("n_edges_" + scope, s["n_edges"]),
                   "top_clauses": s.get(slice_key, s.get("top_clauses", []))}
                  for s in net["time_slices"]]
        out = {"tool": "shanghan_citation_network",
               "overview": net["overview"],
               "scope": scope,
               "scope_note": "scope 貫穿全部計量字段（被引榜/時間切片/共引/"
                             "突現/主路徑）；overview 為全域總量統計。",
               "time_slices": slices,
               "note": net.get("note", "")}
        if target:
            from ..trace.chains import (_citations_by_dynasty, _clauses,
                                        _main_path, _resolve_clause)
            c = _resolve_clause(target, _clauses())
            if c is not None:
                cid = c["clause_id"]
                out["target"] = {
                    "kind": "clause", "clause_id": cid,
                    "citations": _citations_by_dynasty([cid]),
                    "main_path": _main_path(cid),
                    "bursts": [b for b in net.get("bursts", [])
                               if b["clause_id"] == cid]}
                return out
            q = normalize_query(target)
            fm = next((f for f in trace_builder.load_formula_mentions()
                       .get("formulas", [])
                       if fold_variants(f.get("formula", "")) == q), None)
            if fm is not None:
                out["target"] = {"kind": "formula", "formula": fm["formula"],
                                 "total_mentions": fm["total_mentions"],
                                 "n_books": fm["n_books"],
                                 "by_book": fm["by_book"][:top_k]}
                return out
            out["target"] = {"kind": "unknown",
                             "note": f"未識別 target {target}（可用條文號或方名）"}
            return out
        ranking_key = {"canonical": "top_cited_canonical",
                       "auxiliary": "top_cited_auxiliary",
                       "all": "top_cited_clauses"}.get(scope, "top_cited_canonical")
        out["top_cited_clauses"] = net.get(ranking_key,
                                           net["top_cited_clauses"])[:top_k]
        out["ranking_note"] = net.get("ranking_note", "")
        scoped = net.get("scoped", {}).get(scope, {})
        out["cocitation_pairs"] = scoped.get(
            "cocitation_pairs", net["cocitation_pairs"])[:top_k]
        out["bursts"] = scoped.get("bursts", net.get("bursts", []))[:top_k]
        out["main_paths"] = scoped.get(
            "main_paths", net.get("main_paths", []))[:3]
        # 文獻耦合也按 scope（著作條文集先過濾再算 Jaccard——書對字段不含
        # clause_id，審計器掃不到，故靠逐 scope 重算 + 單元測試保證）
        out["bibliographic_coupling"] = scoped.get(
            "bibliographic_coupling", net["bibliographic_coupling"])[:top_k]
        return out

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
        names, unresolved = [], []
        for f in formulas:
            res = self.resolve_formula(f)
            if res["resolved"]:
                names.append(res["resolved"])
            else:
                unresolved.append(res)
        if unresolved:
            out = self._ambiguous_payload("shanghan_differential", unresolved[0])
            out["resolved_formulas"] = names
            return out
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

    # -- formula-name disambiguation --------------------------------------
    def _formula_inventory(self) -> List[str]:
        return [r.formula for r in self.art.formula_rules]

    def resolve_formula(self, formula: str) -> Dict:
        """Normalize + canonicalize + fuzzy-resolve a formula name against
        the rule inventory. See lexicon.disambiguate_formula."""
        from .. import lexicon
        from ..textutil import normalize_query
        return lexicon.disambiguate_formula(normalize_query(formula),
                                            self._formula_inventory())

    @staticmethod
    def _ambiguous_payload(tool: str, res: Dict) -> Dict:
        return {"tool": tool,
                "error": (f"方名「{res['input']}」無法唯一定位"
                          if res["candidates"] else
                          f"未找到方名「{res['input']}」的規則"),
                "ambiguous": res["ambiguous"],
                "candidates": res["candidates"],
                "hint": ("請從 candidates 中選定一個方名重試；"
                         "如需完整清單可調 shanghan_list_formulas"
                         if res["candidates"] else
                         "可調 shanghan_list_formulas 查看可用方名")}

    def _t_formula_rule(self, formula):
        res = self.resolve_formula(formula)
        if not res["resolved"]:
            return self._ambiguous_payload("shanghan_formula_rule", res)
        fpr = next((r for r in self.art.formula_rules
                    if r.formula == res["resolved"]), None)
        if fpr is None:
            return {"tool": "shanghan_formula_rule",
                    "error": f"未找到 {res['resolved']} 的方證規則"}
        d = fpr.to_dict()
        d["tool"] = "shanghan_formula_rule"
        if res["resolved"] != res["input"]:
            d["resolved_from"] = res["input"]
        return d

    def _t_hypotheses(self, symptoms, pulse=None, six_channel=None, top_k=4):
        from .hypothesis import HypothesisManager
        return HypothesisManager(self).analyze(
            symptoms=symptoms, pulse=pulse or [],
            six_channel=six_channel, top_k=top_k)

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

    def for_role(self, role: Optional[str]) -> "ToolRegistry":
        """Hard role isolation at the capability surface: patient sessions
        get a registry that simply does not contain prescription-adjacent
        tools — 不是提示詞約束，而是能力面裁剪."""
        if role == "patient":
            return ScopedRegistry(self, PATIENT_SAFE_TOOLS)
        return self

    def call(self, name: str, arguments: Dict) -> Dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}", "available": self.names()}
        arguments = self._coerce_args(tool, dict(arguments or {}))
        problem = self._validate_args(tool, arguments)
        if problem:
            return {"tool": name, "error": f"參數校驗失敗：{problem}",
                    "expected_schema": tool.parameters}
        key = name + "::" + json.dumps(arguments, ensure_ascii=False,
                                       sort_keys=True, default=str)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            out = copy.deepcopy(cached)
            out["cache_hit"] = True
            return out
        self.cache_misses += 1
        try:
            result = tool.func(**arguments)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # never crash the agent on a tool error
            return {"error": f"tool {name} failed: {type(exc).__name__}: {exc}"}
        result = self._stamp(name, result)
        if isinstance(result, dict) and "error" not in result:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = copy.deepcopy(result)
        return result

    # -- envelope helpers -------------------------------------------------
    @staticmethod
    def _coerce_args(tool: Tool, arguments: Dict) -> Dict:
        """Repair common LLM slips (top_k="6", symptoms="惡寒") instead of
        failing the call."""
        props = tool.parameters.get("properties", {})
        for k, v in list(arguments.items()):
            want = props.get(k, {}).get("type")
            if want == "integer" and isinstance(v, str) and v.strip().isdigit():
                arguments[k] = int(v)
            elif want == "array" and isinstance(v, str):
                arguments[k] = [s for s in
                                (x.strip() for x in
                                 v.replace("，", ",").replace("、", ",").split(","))
                                if s]
            elif want == "boolean" and isinstance(v, str):
                arguments[k] = v.strip().lower() in ("true", "1", "yes", "是")
        return arguments

    @staticmethod
    def _validate_args(tool: Tool, arguments: Dict) -> Optional[str]:
        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        # an explicitly-passed empty list is a legal value（pulse-only 方證
        # 匹配傳 symptoms=[]）——only absent/None/"" count as missing
        missing = [r for r in required
                   if r not in arguments or arguments.get(r) in (None, "")]
        if missing:
            return f"缺少必填參數 {'、'.join(missing)}"
        unknown = [k for k in arguments if k not in props]
        if unknown:
            return f"未知參數 {'、'.join(unknown)}（可用：{'、'.join(props)}）"
        type_map = {"string": str, "integer": int, "boolean": bool,
                    "array": list, "object": dict}
        for k, v in arguments.items():
            want = props.get(k, {}).get("type")
            py = type_map.get(want)
            if py and v is not None and not isinstance(v, py):
                return f"參數 {k} 應為 {want}"
        return None

    def _stamp(self, name: str, result: Any) -> Any:
        meta = TOOL_META.get(name)
        if not (meta and isinstance(result, dict)) or "error" in result:
            return result
        result.setdefault("evidence_level", meta["evidence_level"])
        if meta.get("limitations"):
            result.setdefault("limitations", list(meta["limitations"]))
        result.setdefault("confidence", self._result_confidence(name, result))
        return result

    @staticmethod
    def _result_confidence(name: str, result: Dict) -> float:
        """Deterministic, honest confidence: derived from match scores /
        release levels / hit presence — not a model's self-assessment."""
        if name in ("shanghan_match_formula", "shanghan_hypotheses"):
            m = (result.get("matched_formula_patterns")
                 or result.get("hypotheses") or [])
            top = (m[0].get("match_score") or m[0].get("score", 0)) if m else 0
            return round(min(0.95, float(top or 0)), 2) if m else 0.1
        if name == "shanghan_differential":
            d = result.get("differential") or {}
            return _RELEASE_CONFIDENCE.get(d.get("release_level"), 0.7)
        if name == "shanghan_formula_rule":
            return _RELEASE_CONFIDENCE.get(result.get("release_level"), 0.7)
        if name == "shanghan_search":
            return 0.9 if result.get("hits") else 0.2
        if name in ("shanghan_get_clause", "shanghan_dose_convert",
                    "shanghan_corpus_stats", "shanghan_eval_metrics",
                    "shanghan_list_formulas"):
            return 0.95        # deterministic lookup / computation
        if name in ("shanghan_trace", "shanghan_citation_network"):
            return 0.9         # deterministic derivation over verbatim edges
        return 0.8


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

    def for_role(self, role: Optional[str]) -> "ScopedRegistry":
        if role == "patient":
            return ScopedRegistry(self._base,
                                  [n for n in self._allowed
                                   if n in PATIENT_SAFE_TOOLS])
        return self

    def resolve_formula(self, formula: str) -> Dict:
        return self._base.resolve_formula(formula)


_REGISTRY: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolRegistry()
    return _REGISTRY
