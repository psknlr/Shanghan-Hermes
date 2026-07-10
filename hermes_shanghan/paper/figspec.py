"""聲明式圖形規範（十五輪 十四：Figure Factory，不再逐個硬編碼字符串）。

- JournalProfile：物理尺寸（mm）與字號下限——「720px 清晰」不等於
  「單欄 89mm 下仍清晰」，渲染按期刊剖面執行。
- FigureSpec / PanelSpec / FigureLegend：每張圖聲明它服務的科學問題、
  主信息、面板、圖例要素（n/數據源/誤差定義/證據層/Source Data 指向）。
- FIGURE_PLANS：**每種論文類型有自己的圖組**（P0-2：八種論文不再共用
  同一套圖）；圖組順序即正文 Fig 編號順序。
- stable_id：sha256 穩定 ID（P0-1：根除 ``abs(hash())`` 的進程隨機性，
  資產字節級可復現）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def stable_id(namespace: str, value: str) -> str:
    digest = hashlib.sha256(
        f"{namespace}\0{value}".encode("utf-8")).hexdigest()[:12]
    return f"{namespace}_{digest}"


@dataclass(frozen=True)
class JournalProfile:
    name: str = "nature-like"
    single_column_mm: float = 89.0
    double_column_mm: float = 183.0
    body_font_pt: float = 7.0
    panel_label_pt: float = 8.0
    min_line_pt: float = 0.5
    min_font_px: int = 9            # 720px 畫布上的字號下限（QA 執行）


PROFILE = JournalProfile()


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str                   # a / b / c …
    chart_type: str                 # hbar / heatmap / interval / graph-source
    data_query: str                 # 數據從哪來（人可讀，入 manifest）
    x: str = ""
    y: str = ""
    color: str = ""
    uncertainty: str = ""           # 誤差/區間口徑；空=點估計（如實聲明）


@dataclass(frozen=True)
class FigureLegend:
    title: str
    panels: Dict[str, str]          # panel_id → 一句話描述
    n: str                          # 樣本量口徑（分母定義）
    data_source: str
    error_definition: str           # 「無誤差條（描述性統計）」也要寫明
    evidence_level: str
    abbreviations: str = ""


@dataclass(frozen=True)
class FigureSpec:
    key: str                        # 內部鍵（emitter 選擇）
    scientific_question: str
    main_message: str
    target_width: str               # single | double
    panels: Tuple[PanelSpec, ...]
    legend: FigureLegend
    renderer: str                   # hermes-stdlib-svg/1 | mermaid-source |
                                    # dot-source+graphml

    def width_mm(self, profile: JournalProfile = PROFILE) -> float:
        return (profile.single_column_mm if self.target_width == "single"
                else profile.double_column_mm)


def _leg(title, panels, n, src, err, lvl, abbr="") -> FigureLegend:
    return FigureLegend(title, panels, n, src, err, lvl, abbr)


FIGURE_SPECS: Dict[str, FigureSpec] = {
    "channel_formula": FigureSpec(
        "channel_formula",
        "六經各自以哪些方劑為主治骨幹？",
        "六經—主方的隸屬結構呈強分化：太陽方最多、厥陰最少",
        "double",
        (PanelSpec("a", "graph-source", "six_channel_rules.main_formulas"),),
        _leg("六經—方劑分佈圖",
             {"a": "每經列其主方（按載方條文數，前 4）"},
             "n=六經規則全部主方邊（見 source data）",
             "six_channel_rules（D 層歸納，錨定 A 層條文）",
             "無誤差條（確定性計數）", "D（錨定 A）"),
        "mermaid-source"),
    "mistreatment_paths": FigureSpec(
        "mistreatment_paths",
        "誤治如何驅動證候轉變、以何方救逆？",
        "誤治→變證→救逆方構成有向網絡，匯於少數樞紐變證",
        "double",
        (PanelSpec("a", "graph-source", "mistreatment_rules"),),
        _leg("誤治—變證路徑圖",
             {"a": "實線=誤治致變；虛線=救逆方"},
             "n=全部誤治傳變規則（見 source data）",
             "mistreatment_rules（D 層，逐條錨定條文）",
             "無誤差條（規則圖）", "D（錨定 A）"),
        "dot-source+graphml"),
    "formula_family": FigureSpec(
        "formula_family",
        "經方加減關係構成怎樣的家族樹？",
        "桂枝湯/麻黃湯等基方輻射出多層加減家族",
        "double",
        (PanelSpec("a", "graph-source", "formula_rules.modification_relations"),),
        _leg("方劑家族樹",
             {"a": "有向邊：基方→加減方"},
             "n=全部加減關係邊（見 source data）",
             "formula_rules（A 層組成派生）",
             "無誤差條（確定性關係）", "A-derived"),
        "dot-source+graphml"),
    "clause_topics": FigureSpec(
        "clause_topics",
        "條文主題在六經內如何分佈？",
        "方證條文為主體，誤治/禁忌/脈證/預後構成次級主題",
        "double",
        (PanelSpec("a", "graph-source", "clauses(six_channel×theme)"),),
        _leg("條文主題聚類圖",
             {"a": "六經 × 六類主題的條文分桶（樣例條文號）"},
             "n=398 條正文（每條入唯一主題桶）",
             "宋本正文條文（A 層直計）",
             "無誤差條（確定性分桶）", "A-derived"),
        "mermaid-source"),
    "formula_freq": FigureSpec(
        "formula_freq",
        "哪些方劑在宋本正文中載方條文最多？",
        "桂枝湯類獨大；前 10 方覆蓋大部分載方條文",
        "single",
        (PanelSpec("a", "hbar", "formula_freq.most_common(10)",
                   x="載方條文數", y="方劑"),),
        _leg("宋本正文中各方劑的載方條文數（前 10）",
             {"a": "橫條=載方條文數；一條條文含多方時分別計數"},
             "n=398 條正文（分母：全部原文條文）",
             "宋本正文 A 層直計",
             "無誤差條（全量計數，非抽樣）", "A"),
        "hermes-stdlib-svg/1"),
    "commentator_agreement": FigureSpec(
        "commentator_agreement",
        "九注家對相同條文的術語詮釋有多一致？",
        "注家兩兩一致度分層明顯；共注條數各對不同（n 顯示於格內）",
        "double",
        (PanelSpec("a", "heatmap", "commentary_divergence.agreement_matrix",
                   uncertainty="每格附 n=共注條數；均值無 CI（詞彙級指標）"),),
        _leg("注家術語一致度矩陣",
             {"a": "色階=平均術語一致度；格內括號=共注條文數 n"},
             "n=每對注家的共注條文數（逐格標注，各不相同）",
             "commentary_divergence.json（C/D 層）",
             "均值點估計；未做 bootstrap CI（術語 Jaccard 為詞彙級下界，"
             "不等於觀點一致）", "C/D",
             abbr="共注 n 過小（<5）的格值不穩定"),
        "hermes-stdlib-svg/1"),
    "dose_totals": FigureSpec(
        "dose_totals",
        "不同折算假設下全方總重量差多少？",
        "三家折算是**學術假設情景區間**，非三個測量值",
        "double",
        (PanelSpec("a", "interval", "dose_ratios.total_weight_g",
                   uncertainty="區間=三家折算假設（考古實測/度量衡史/"
                               "明清折算）的情景範圍"),),
        _leg("全方總重量的折算假設區間（前 6 重方）",
             {"a": "橫線=三家假設的範圍；點=各家取值"},
             "n=每方一組三家折算（僅計重量類藥）",
             "dose_ratios.json（A 層藥量比 + D/E 層折算假設）",
             "區間為折算學派差異，非測量誤差；**不構成臨床劑量建議**",
             "A + D/E"),
        "hermes-stdlib-svg/1"),
    "benchmark": FigureSpec(
        "benchmark",
        "規則系統在遮方/醫案/接地三類任務上表現如何？",
        "不同任務指標語義不同，分組呈現；均為單次評測點估計",
        "single",
        (PanelSpec("a", "hbar", "eval/*.json metrics",
                   x="指標值（0–1）",
                   uncertainty="單次評測點估計；無 bootstrap CI/基線對比"
                               "（如實聲明，屬改進路線）"),),
        _leg("客觀評測基準",
             {"a": "各任務指標（Top-k/MRR/接地率），0–1 刻度軸"},
             "n=各任務樣本量見 source data（口徑不同，不可跨行比較）",
             "eval/ 持久化結果",
             "點估計，無置信區間（單次運行）——跨任務不可直接比大小", "D"),
        "hermes-stdlib-svg/1"),
}

# 每種論文類型的圖組（P0-2）：服務論文核心問題，非「庫裡有什麼畫什麼」
FIGURE_PLANS: Dict[str, List[str]] = {
    "formula_pattern": ["channel_formula", "formula_family", "formula_freq",
                        "clause_topics"],
    "six_channel_kg": ["channel_formula", "clause_topics", "formula_freq"],
    "mistreatment": ["mistreatment_paths", "clause_topics", "formula_freq"],
    "network_pharmacology": ["formula_family", "formula_freq", "dose_totals"],
    "commentary_compare": ["commentator_agreement", "clause_topics"],
    "methodology": ["channel_formula", "mistreatment_paths", "benchmark"],
    "benchmark": ["benchmark", "formula_freq"],
    "provenance": ["formula_family", "commentator_agreement"],
}
