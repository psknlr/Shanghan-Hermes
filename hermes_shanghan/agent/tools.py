"""ToolRegistry — the single capability surface shared by the agent, the MCP
server and the OpenAI-compatible tool specs.

All tools are read-only and evidence-returning: each result carries clause_id
references so any downstream answer can be citation-checked. Patient-unsafe
operations are simply not exposed as tools.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import config
from ..schemas import read_jsonl


TOOLS_VERSION = "1.2.0"             # 工具面語義版本（契約隨規格導出）
MAX_RESULT_BYTES = 262_144          # 單工具結果上限（外部 harness 穩定集成）
TOOL_TIMEOUT_S = 30.0               # 契約 timeout_s（HERMES_TOOL_TIMEOUT 可調，0=關）


class ToolTimeout(Exception):
    pass


def _tool_timeout() -> float:
    raw = os.environ.get("HERMES_TOOL_TIMEOUT", "")
    if raw == "":
        return TOOL_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        return TOOL_TIMEOUT_S


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

    def contract(self) -> Dict:
        """機器可讀工具契約（評審第 6 條）：版本/權限/副作用/證據層/
        冪等性/大小限制/schema 指紋。純標準庫實現（無 Pydantic）。
        ``enforced`` 節如實聲明哪些條款由 call() 管道真正執行（九輪
        P0-8：契約不得只是聲明），哪些屬構造保證。"""
        import hashlib
        meta = TOOL_META.get(self.name, {})
        return {
            "name": self.name,
            "version": TOOLS_VERSION,
            "permission_level": ("patient_safe" if self.name in PATIENT_SAFE_TOOLS
                                 else "doctor_researcher"),
            "evidence_level": meta.get("evidence_level", ""),
            "limitations": meta.get("limitations", []),
            "side_effect": "read",          # 全部工具只讀（設計不變式）
            "timeout_s": int(TOOL_TIMEOUT_S),
            "cacheable": True,
            "idempotent": True,
            "max_result_bytes": MAX_RESULT_BYTES,
            "error_schema": {"error": "string", "hint": "string?"},
            "schema_hash": hashlib.sha256(
                json.dumps(self.parameters, sort_keys=True,
                           ensure_ascii=False).encode()).hexdigest()[:16],
            # call() 管道逐項執行狀態（不是願望清單）：
            "enforced": {
                "args_schema": "runtime（_validate_args：必填/未知參數/類型）",
                "timeout_s": "runtime（工作線程 + join(timeout)；超時回錯誤"
                             "信封，守護線程結果丟棄；HERMES_TOOL_TIMEOUT=0 關閉）",
                "max_result_bytes": "runtime（超限報錯不靜默截斷）",
                "output_shape": "runtime（非 dict 輸出視為契約違例）",
                "cacheable": "runtime（緩存鍵含 tools_version+語料指紋）",
                "side_effect_read": "by-construction（工具函數無寫路徑；"
                                    "非運行時攔截，見 tests 只讀不變式）",
                "idempotent": "by-construction（只讀派生，重試安全）",
                "audit": "runtime（環形審計日誌 audit_tail()；harness 下"
                         "另有 span 級軌跡）",
            },
        }


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
    # —— classics 全庫工具族（十五輪 P0-2：P 層是一等證據，不再是
    # 「證據系統之外的文本」；逐條 verbatim+座標+quote_hash 可重驗）——
    "classics_search_passages": {
        "evidence_level": "P",
        "limitations": ["文獻旁證層：可回源重驗，不進入 A 層經文閘門；"
                        "scan_capped 時零命中≠全庫不存在"]},
    "classics_read_passage": {"evidence_level": "P"},
    "classics_compare_witnesses": {
        "evidence_level": "P",
        "limitations": ["傳本歸組按折疊書名，同名異書需人工消歧"]},
    "classics_trace_citation": {
        "evidence_level": "P",
        "limitations": ["在庫首現≠歷史首現；反證搜索的部分匹配候選需人工核驗"]},
    "classics_resolve_term": {
        "evidence_level": "P",
        "limitations": ["僅異體折疊與出現概況；通假/古今詞/同義映射屬規劃層"]},
    "classics_concept_drift": {
        "evidence_level": "P",
        "limitations": ["頻次漂移≠語義漂移；計數受 per_book/max_scan 封頂"]},
    "classics_library_stats": {
        "evidence_level": "P",
        "limitations": ["統計對象是笈成全庫書目，非傷寒論規則庫"]},
    "classics_export_evidence_packet": {"evidence_level": "P"},
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
    # classics 全庫查閱族：只讀文獻層，與 shanghan_library 同等暴露口徑
    "classics_search_passages", "classics_read_passage",
    "classics_compare_witnesses", "classics_trace_citation",
    "classics_resolve_term", "classics_concept_drift",
    "classics_library_stats", "classics_export_evidence_packet",
]


class ToolRegistry:
    """Lazy-loads pipeline artifacts once, exposes the grounded tool surface
    (see `_register_all`; every result carries clause_id evidence)."""

    def __init__(self, cache_size: int = 256):
        self._art = None
        self._clause_rag = None
        self._matcher = None
        self._tools: Dict[str, Tool] = {}
        # (tool, canonical-args, tools_version, corpus_fp) → result cache:
        # repeated retrieval within a session/orchestration is free and
        # reproducible; 語料指紋入鍵——語料換版緩存自然失效（九輪 P0-8）
        self._cache: Dict[str, Dict] = {}
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self._corpus_fp = self._corpus_fingerprint()
        # 環形審計日誌：每次調用的 {tool, ok, ms, cache_hit}（輕量常駐；
        # harness 運行下另有 span 級 JSONL 軌跡）
        self.audit_log: deque = deque(maxlen=256)
        # ThreadingHTTPServer 下的共享狀態鎖（十一輪 九：緩存/計數並發安全）
        self._lock = threading.Lock()
        # 超時熔斷：超時的工作線程仍在後台運行（無法強殺），滯留過多時
        # 熔斷新調用而不是無限堆線程
        self._zombie_threads: List[threading.Thread] = []
        self._register_all()

    @staticmethod
    def _corpus_fingerprint() -> str:
        import hashlib
        m = config.MANIFEST_DIR / "corpus_manifest.json"
        try:
            return hashlib.sha256(m.read_bytes()).hexdigest()[:12]
        except OSError:
            return "no-manifest"

    @staticmethod
    def _library_fp() -> str:
        """笈成全庫指紋（編目路徑+mtime）：換庫/重建編目即變。"""
        cat = config.LIBRARY_DIR / "catalog.json"
        try:
            return f"{cat}@{cat.stat().st_mtime_ns}"
        except OSError:
            return f"{cat}@absent"

    def audit_tail(self, n: int = 20) -> List[Dict]:
        return list(self.audit_log)[-n:]

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
            "傷寒論規則庫統計（領域統計，**非**笈成全庫統計）：條文/規則/"
            "關係/方證頻次/六經分佈等數字（科研引用用）。全庫書目統計請用 "
            "classics_library_stats。",
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
        # 十五輪 P0-1：classics_* 工具族（全量古籍平台層，獨立於傷寒論
        # 領域）——同一 Registry 註冊，自動獲得 Broker 台賬/軌跡/預算/契約
        from ..classics.tools import register_classics_tools
        register_classics_tools(self._add)

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
                "semantic": "傷寒論規則庫（領域）統計；笈成全庫書目統計見 "
                            "classics_library_stats——兩者語義嚴格分離",
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
        from ..trace.evidence import records_for_hits
        return {"tool": "shanghan_search", "query": query, "hits": hits,
                # 逐證據來源對象（十輪 六.1）：命中帶檢索上下文的溯源記錄
                "evidence_records": records_for_hits(
                    hits, self.art.clause_store(), query)}

    def _t_get_clause(self, ref):
        c = self.clause_rag.get_clause(ref)
        if c is None:
            return {"tool": "shanghan_get_clause", "error": f"未找到條文 {ref}"}
        rules = [r for r in read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
                 if r["clause_id"] == c.clause_id]
        from ..trace.evidence import evidence_record
        return {"tool": "shanghan_get_clause",
                "clause": {"clause_id": c.clause_id, "clause_number": c.clause_number,
                           "chapter": c.chapter, "six_channel": c.six_channel,
                           "clean_text": c.clean_text, "layer_label": "A 原文直述",
                           "symptoms": c.symptoms, "pulse": c.pulse,
                           "formulas": c.formula_names},
                "evidence_record": evidence_record(c),
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

    MAX_ZOMBIE_THREADS = 8

    def _run_with_timeout(self, func: Callable, kwargs: Dict, timeout: float):
        """契約 timeout_s 的運行時執行：工作線程 + join(timeout)。超時拋
        ToolTimeout（守護線程繼續完成但結果丟棄——只讀工具無副作用可棄）。
        滯留線程超閾值時熔斷（circuit breaker）：連續超時不再無限堆線程。
        timeout<=0 時內聯執行（調試/剖析用）。"""
        if timeout <= 0:
            return func(**kwargs)
        with self._lock:
            self._zombie_threads = [t for t in self._zombie_threads
                                    if t.is_alive()]
            if len(self._zombie_threads) >= self.MAX_ZOMBIE_THREADS:
                raise ToolTimeout(
                    f"circuit open：{len(self._zombie_threads)} 個超時工具"
                    "線程仍在運行，熔斷新調用（等待滯留線程結束）")
        box: Dict[str, Any] = {}

        def _target():
            try:
                box["result"] = func(**kwargs)
            except BaseException as exc:      # 傳回主線程統一走錯誤信封
                box["exc"] = exc

        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            with self._lock:
                self._zombie_threads.append(th)
            raise ToolTimeout(f"timeout after {timeout}s")
        if "exc" in box:
            raise box["exc"]
        return box.get("result")

    # 這些工具以 supporting_clauses 等 id 列表作核心證據——把條文正文
    # 摘錄一併放進結果（十四輪 P0-四：模型必須真的讀到正文，證據才算
    # 返回；純導航類工具如 relations/trace 不附，保持 id_mention_only）
    EXCERPT_TOOLS = frozenset({
        "shanghan_formula_rule", "shanghan_differential",
        "shanghan_six_channel", "shanghan_mistreatment", "shanghan_therapy",
        "shanghan_match_formula", "shanghan_hypotheses",
        "shanghan_adjudicate", "shanghan_conflict_audit"})

    def _attach_excerpts(self, name: str, result: Dict) -> Dict:
        if name not in self.EXCERPT_TOOLS or not isinstance(result, dict) \
                or "error" in result:
            return result
        import re as _re
        rx = _re.compile(r"SHL_SONGBEN_(?:AUX_)?\d{4}")
        blob = json.dumps(result, ensure_ascii=False, default=str)
        ids = list(dict.fromkeys(rx.findall(blob)))[:12]
        if not ids:
            return result
        try:
            store = self.art.clause_store()
        except Exception:
            return result
        excerpts = []
        for cid in ids:
            c = store.get(cid)
            text = getattr(c, "clean_text", "") if c else ""
            if text and text[:12] not in blob:
                excerpts.append({"clause_id": cid, "text": text[:100]})
        if excerpts:
            result["evidence_excerpts"] = excerpts
            result.setdefault("evidence_excerpts_note",
                              "支持條文正文摘錄（隨結果進入模型上下文，"
                              "使引用可按 primary_text_returned 核驗）")
        return result

    @staticmethod
    def _safe_exc(exc: BaseException) -> str:
        """異常信息進錯誤信封前脫敏：去倉庫絕對路徑 + 截斷（信封可能
        原樣返回給外部調用方）。"""
        msg = str(exc).replace(str(config.REPO_ROOT), "<repo>")
        return f"{type(exc).__name__}: {msg[:200]}"

    def call(self, name: str, arguments: Dict) -> Dict:
        """Capability-Broker 管道（九輪 P0-8）：
        默認拒絕（未知工具）→ 參數修復/校驗 → 緩存（版本化鍵）→
        超時執行（熔斷保護）→ 輸出形狀/大小校驗 → 分層蓋章 → 審計。"""
        t0 = time.time()
        repairs: List[Dict] = []

        def _audit(out: Dict, cache_hit: bool = False) -> Dict:
            entry = {"tool": name, "ok": "error" not in out,
                     "cache_hit": cache_hit,
                     "ms": int((time.time() - t0) * 1000),
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if repairs:
                entry["repairs"] = repairs
            self.audit_log.append(entry)
            return out

        tool = self._tools.get(name)
        if tool is None:      # 默認拒絕：不在註冊表=不可調用
            return _audit({"error": f"unknown tool: {name}",
                           "available": self.names()})
        arguments, repairs = self._coerce_args(tool, dict(arguments or {}))
        problem = self._validate_args(tool, arguments)
        if problem:
            return _audit({"tool": name, "error": f"參數校驗失敗：{problem}",
                           "expected_schema": tool.parameters})
        # 十五輪：全庫類工具緩存鍵含**庫指紋**（庫換版/測試換庫緩存自然
        # 失效——傷寒論語料指紋管不到笈成全庫）
        fp = self._corpus_fp
        if name.startswith("classics_") or name == "shanghan_library":
            fp = f"{fp}+{self._library_fp()}"
        key = "::".join([name, TOOLS_VERSION, fp,
                         json.dumps(arguments, ensure_ascii=False,
                                    sort_keys=True, default=str)])
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                out = copy.deepcopy(cached)
            else:
                self.cache_misses += 1
                out = None
        if out is not None:
            out["cache_hit"] = True
            return _audit(out, cache_hit=True)
        timeout = _tool_timeout()
        try:
            result = self._run_with_timeout(tool.func, arguments, timeout)
        except ToolTimeout as exc:
            return _audit({"tool": name,
                           "error": f"tool {name} timeout：{exc}"
                                    f"（契約 timeout_s={timeout:g}s）",
                           "hint": "縮小參數範圍；長任務走 MCP tasks/submit"})
        except TypeError as exc:
            return _audit({"error": f"bad arguments for {name}: "
                                    f"{self._safe_exc(exc)}"})
        except Exception as exc:  # never crash the agent on a tool error
            return _audit({"error": f"tool {name} failed: "
                                    f"{self._safe_exc(exc)}"})
        # 輸出形狀契約：工具必須返回 dict（非 dict 視為契約違例）
        if not isinstance(result, dict):
            return _audit({"tool": name,
                           "error": f"tool {name} 輸出契約違例：期望 dict，"
                                    f"得到 {type(result).__name__}"})
        result = self._stamp(name, result)
        result = self._attach_excerpts(name, result)
        # 輸出大小護欄（工具契約 max_result_bytes）：超限如實報錯而非靜默截斷
        blob = json.dumps(result, ensure_ascii=False, default=str)
        if len(blob.encode("utf-8")) > MAX_RESULT_BYTES:
            return _audit({"tool": name,
                           "error": f"結果超過契約上限 {MAX_RESULT_BYTES} bytes",
                           "hint": "縮小 top_k/limit 參數或分頁調用"})
        if "error" not in result:
            with self._lock:
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = copy.deepcopy(result)
        return _audit(result)

    def contracts(self) -> List[Dict]:
        """全部工具的機器可讀契約（隨 tool_specs.json 導出）。"""
        return [self._tools[n].contract() for n in sorted(self._tools)]

    # -- envelope helpers -------------------------------------------------
    @staticmethod
    def _coerce_args(tool: Tool, arguments: Dict) -> Tuple[Dict, List[Dict]]:
        """Repair common LLM slips (top_k="6", symptoms="惡寒") instead of
        failing the call. 每次修復記錄 {arg, from, to, reason} 進審計
        （十一輪 八.2）；布爾只接受明確真/假詞——"banana" 不再被靜默
        修成 False，交由校驗報錯。"""
        props = tool.parameters.get("properties", {})
        repairs: List[Dict] = []
        for k, v in list(arguments.items()):
            want = props.get(k, {}).get("type")
            if want == "integer" and isinstance(v, str) and v.strip().isdigit():
                arguments[k] = int(v)
                repairs.append({"arg": k, "from": v, "to": arguments[k],
                                "reason": "string→integer"})
            elif want == "array" and isinstance(v, str):
                arguments[k] = [s for s in
                                (x.strip() for x in
                                 v.replace("，", ",").replace("、", ",").split(","))
                                if s]
                repairs.append({"arg": k, "from": v, "to": arguments[k],
                                "reason": "string→array（頓號/逗號切分）"})
            elif want == "boolean" and isinstance(v, str):
                low = v.strip().lower()
                if low in ("true", "1", "yes", "是"):
                    arguments[k] = True
                elif low in ("false", "0", "no", "否"):
                    arguments[k] = False
                else:
                    continue      # 不明字符串不猜——留給類型校驗報錯
                repairs.append({"arg": k, "from": v, "to": arguments[k],
                                "reason": "string→boolean（明確真假詞）"})
        return arguments, repairs

    @staticmethod
    def _validate_args(tool: Tool, arguments: Dict) -> Optional[str]:
        """JSON-Schema 子集深校驗（十一輪 八.2）：必填/未知/類型 +
        enum / minimum / maximum / maxItems / 數組元素類型 / pattern /
        maxLength。"""
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
                    "array": list, "object": dict, "number": (int, float)}
        import re as _re
        for k, v in arguments.items():
            spec = props.get(k, {})
            want = spec.get("type")
            py = type_map.get(want)
            if py and v is not None and not isinstance(v, py):
                return f"參數 {k} 應為 {want}"
            if v is None:
                continue
            if "enum" in spec and v not in spec["enum"]:
                return f"參數 {k}={v!r} 不在枚舉 {spec['enum']}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if "minimum" in spec and v < spec["minimum"]:
                    return f"參數 {k}={v} 低於下限 {spec['minimum']}"
                if "maximum" in spec and v > spec["maximum"]:
                    return f"參數 {k}={v} 超過上限 {spec['maximum']}"
            if isinstance(v, list):
                if "maxItems" in spec and len(v) > spec["maxItems"]:
                    return f"參數 {k} 元素數 {len(v)} 超過 {spec['maxItems']}"
                item_t = type_map.get((spec.get("items") or {}).get("type"))
                if item_t and any(not isinstance(x, item_t) for x in v):
                    return f"參數 {k} 的元素應為 {spec['items']['type']}"
            if isinstance(v, str):
                if "maxLength" in spec and len(v) > spec["maxLength"]:
                    return f"參數 {k} 長度 {len(v)} 超過 {spec['maxLength']}"
                if "pattern" in spec and not _re.search(spec["pattern"], v):
                    return f"參數 {k} 不匹配 pattern {spec['pattern']}"
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
        # 資產缺失時響亮失敗而非空運行（九輪 P0-7 假健康）：pip wheel 不含
        # 語料，獨立安裝後首次構建註冊表即在此被攔截並給出修復指引
        from ..health import assert_ready
        assert_ready(context="ToolRegistry")
        _REGISTRY = ToolRegistry()
    return _REGISTRY
