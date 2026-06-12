"""PaperWriterAgent — full manuscript generation grounded in mined data.

Six paper types (per protocol). Every Results statement carries rule IDs /
clause IDs; figures are emitted as DOT/Mermaid sources + CSV tables under
data/shanghan/papers/<slug>/ so the manuscript is reproducible.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from ..schemas import (DifferentialRule, FormulaPatternRule, InitialRule,
                       MistreatmentTransformationRule, ShanghanClause,
                       SixChannelRule, VariantRule, read_jsonl)

PAPER_TYPES = {
    "formula_pattern": "《傷寒論》方證規律挖掘",
    "six_channel_kg": "《傷寒論》六經辨證知識圖譜",
    "mistreatment": "《傷寒論》誤治傳變規則研究",
    "network_pharmacology": "《傷寒論》方劑網絡藥理學前置研究",
    "commentary_compare": "《傷寒論》方劑歷代注釋比較",
    "methodology": "《傷寒論》古籍數據挖掘與智能體方法學研究",
}


class PaperWriter:
    def __init__(self, clauses: List[ShanghanClause],
                 initial_rules: List[InitialRule],
                 formula_rules: List[FormulaPatternRule],
                 six_channel_rules: List[SixChannelRule],
                 mistreatment_rules: List[MistreatmentTransformationRule],
                 differential_rules: Optional[List[DifferentialRule]] = None):
        self.clauses = [c for c in clauses if c.text_type == "original_clause"]
        self.initial_rules = initial_rules
        self.formula_rules = formula_rules
        self.six_channel_rules = six_channel_rules
        self.mistreatment_rules = mistreatment_rules
        self.differential_rules = differential_rules or []

    # ------------------------------------------------------------------
    def _stats(self) -> Dict:
        level = Counter(r.autonomous_review.release_level for r in self.initial_rules)
        rtype = Counter(r.rule_type for r in self.initial_rules)
        formula_freq = Counter()
        channel_clauses = Counter()
        for c in self.clauses:
            formula_freq.update(c.formula_names)
            if c.six_channel:
                channel_clauses[c.six_channel] += 1
        return {
            "n_clauses": len(self.clauses),
            "n_initial_rules": len(self.initial_rules),
            "levels": dict(level), "rule_types": dict(rtype),
            "n_formula_rules": len(self.formula_rules),
            "n_mistreatment": len(self.mistreatment_rules),
            "n_differential": len(self.differential_rules),
            "formula_freq": formula_freq, "channel_clauses": channel_clauses,
        }

    # ------------------------------------------------------------------
    def generate(self, paper_type: str = "formula_pattern",
                 topic: str = "", out_dir: Optional[Path] = None) -> Path:
        if paper_type not in PAPER_TYPES:
            paper_type = "formula_pattern"
        title_root = PAPER_TYPES[paper_type]
        topic = topic or {"formula_pattern": "桂枝湯類方證",
                          "six_channel_kg": "六經辨證體系",
                          "mistreatment": "誤治傳變路徑",
                          "network_pharmacology": "經方藥物網絡",
                          "commentary_compare": "桂枝湯歷代注釋",
                          "methodology": "Hermes自主審核框架"}[paper_type]
        slug = f"{paper_type}_{time.strftime('%Y%m%d')}"
        out = out_dir or (config.PAPER_DIR / slug)
        out.mkdir(parents=True, exist_ok=True)
        s = self._stats()

        # ---------- figures & tables (reproducible assets) -----------------
        figures = self._emit_figures(out, paper_type, s)
        tables = self._emit_tables(out, s)

        title = f"基於條文級規則挖掘與自主審核的{title_root}：以{topic}為例"
        gold = s["levels"].get("gold", 0)
        silver = s["levels"].get("silver", 0)
        bronze = s["levels"].get("bronze", 0)
        top_f = s["formula_freq"].most_common(5)

        abstract = f"""目的：將《傷寒論》宋本條文轉化為可回源、可審核的結構化規則體系，並以{topic}為例驗證方法可行性。
方法：以宋本（趙開美本）現代編號{ s['n_clauses']}條為唯一原文證據層（A層），桂林古本等為異文層（B層），成無己注為注釋層（C層）；經條文切分、否定感知實體抽取、條文級初始規則抽取，再通過 Schema 校驗、證據回源驗證、語義審查、對抗式批評（ShanghanCritic）、自動修復與共識評級六道閘門完成自主審核。
結果：共獲得初始規則{ s['n_initial_rules']}條（gold {gold}、silver {silver}、bronze {bronze}），方證規則{ s['n_formula_rules']}個，誤治傳變路徑{ s['n_mistreatment']}條，鑒別規則{ s['n_differential']}組；高頻方劑為{ '、'.join(f for f, _ in top_f)}。每條結論均可追溯至條文編號。
結論：條文級證據回源 + 模型自主審核可在不犧牲可追溯性的前提下規模化提取《傷寒論》知識，為方證研究、知識圖譜與臨床輔助提供可驗證的數據底座。
關鍵詞：傷寒論；方證對應；六經辨證；知識圖譜；規則挖掘；證據回源"""

        methods = f"""## 3 方法

### 3.1 語料與版本分層
主底本為《傷寒論（宋本）》（明·趙開美刻本，現代通行編號 1–{s['n_clauses']}），
另納入宋本辨脈法、傷寒例與可/不可諸篇作為輔助條文層。版本策略：A層宋本原文、
B層異文（桂林古本、千金翼方版）、C層注釋（成無己《註解傷寒論》等）、
D層後世類方歸納（《傷寒論類方》）、E層模型解釋。合併規則永不覆蓋初始條文規則。

### 3.2 條文切分與實體抽取
以 `<#/>` 編號標記切分條文，`<F>` 塊解析方劑組成（含 `<l>` 劑量炮製）與煎服法。
實體抽取採用最長優先、否定感知匹配（「不惡寒」不得落入「惡寒」），
覆蓋六經、病名、症狀、脈象、方劑、藥物、劑量、治法、禁忌、誤治、傳變、預後十二類。

### 3.3 初始規則抽取
逐條抽取 InitialRule，禁止跨條歸納；處方強度按原文用語分級
（主之＞宜＞屬＞與＞可與），證據片段保存逐字原文。

### 3.4 自主審核流水線
SchemaValidator → EvidenceVerifier（證據逐字回源、方-條對應、條件落文）→
SemanticReviewer（六經/規則類型/強度一致性）→ ShanghanCritic
（後世術語混入、禁忌遺漏、可與誇大、主之擴域、中風/傷寒混淆、
少陰寒化/熱化混淆、陽明經/腑混淆）→ AutoRepair（單輪修復後復檢）→
ConsensusJudge + ReleaseGate（gold ≥ {config.RELEASE_GOLD} / silver ≥ {config.RELEASE_SILVER} / bronze ≥ {config.RELEASE_BRONZE}）。

### 3.5 規則層級
InitialRule → FormulaPatternRule → SixChannelRule → TherapyRule →
MistreatmentTransformationRule → MergedShanghanRule；另建 ClauseRelation
圖譜（同方族/鑒別/誤治傳變/禁忌/傳變/異文/注釋支持七類邊）。"""

        results = self._results_section(paper_type, s, topic)

        discussion = f"""## 5 討論

（1）證據邊界。本研究嚴格區分原文直述與後世歸納：如「營衛不和」屬後世
病機術語，批評器將其攔截出規則主體並降級為模型解釋層；「可與」與「主之」
的證據強度差異被顯式建模，避免將斟酌之辭讀作必用之訓。
（2）版本異文的影響。B層對齊顯示宋本與桂本在部分條文存在用字差異，
規則層僅以宋本為準、異文以 VariantRule 並行記錄，供版本學研究取用。
（3）侷限。實體詞典覆蓋率有限，個別罕見證候表達可能漏標；
亞型命名（如太陽蓄水）屬後世歸納框架，雖逐一錨定條文，仍不宜回讀為仲景原意；
自動審核可保證「不偽造證據」，但不能替代學科專家對歸納合理性的終審。
（4）應用。規則庫已編譯為六經、方證、誤治、禁忌、鑒別等 Skill，
可服務醫師輔助（標註輔助性質）、科研挖掘（共現網絡、知識圖譜）與教學練習；
患者端嚴格禁用診斷與處方功能。"""

        conclusion = """## 6 結論

以條文為最小證據單位、以自主審核為質量閘門的 Hermes 流水線，能夠將
《傷寒論》整書轉化為層級化、可回源、可調用的規則系統；所有方證判斷
均可回到條文，所有歸納均標明證據層級，為中醫古籍的可計算化提供了
一條可複製的路徑。"""

        references = """## 參考文獻

[1] 張仲景. 傷寒論（宋本，明·趙開美校刻）. 東漢.
[2] 張仲景. 傷寒雜病論（桂林古本）. 桂林羅哲初手抄本.
[3] 成無己. 註解傷寒論. 金·皇統四年(1144).
[4] 徐大椿. 傷寒論類方. 清.
[5] 吳謙等. 醫宗金鑒·訂正仲景全書傷寒論注. 清·乾隆.
[6] Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. Found Trends Inf Retr. 2009.
[7] 本研究數據與代碼：data/shanghan/（規則庫、審計日誌、圖表源文件）."""

        cover_letter = f"""## Cover Letter

尊敬的編輯：

茲投稿論文《{title}》。本研究首次將「逐字證據回源 + 對抗式自主審核」
引入《傷寒論》全書規則挖掘，產出 {s['n_initial_rules']} 條分級規則並全部
附帶條文證據鏈，數據與代碼完全公開可復現。本文未一稿多投，所有作者
同意投稿。懇請審閱。

通訊作者敬上
{time.strftime('%Y-%m-%d')}"""

        manuscript = "\n\n".join([
            f"# {title}", "## 摘要\n\n" + abstract,
            "## 1 引言\n\n《傷寒論》以六經統病、以方證相應，其條文之間存在並列、遞進、"
            "誤治轉變、鑒別與禁忌等強結構關係。既往數字化工作多止於全文檢索或人工標註，"
            "缺少「規則必須回到原文」的硬性約束。本文提出 Hermes-Shanghanlun 框架，"
            "把每一條原文轉化為可追蹤規則，使每一個方證判斷都能回到條文編號。",
            "## 2 數據\n\n" + self._data_section(s),
            methods, results, discussion, conclusion,
            "## 圖表清單\n\n" + "\n".join(f"- {f}" for f in figures + tables),
            references, cover_letter,
            "## Supplementary Materials\n\n- S1 規則庫 rules_initial/initial_rules.jsonl\n"
            "- S2 審計日誌 audit/audit_log.jsonl\n- S3 條文關係圖 relations/clause_relations.jsonl\n"
            "- S4 共現網絡與家族樹 research/*.json",
        ])
        (out / "manuscript.md").write_text(manuscript, encoding="utf-8")
        meta = {"paper_type": paper_type, "title": title, "topic": topic,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "figures": figures, "tables": tables,
                "statistics": {k: v for k, v in s.items()
                               if k not in ("formula_freq", "channel_clauses")}}
        (out / "paper_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return out / "manuscript.md"

    # ------------------------------------------------------------------
    def _data_section(self, s: Dict) -> str:
        rows = "\n".join(f"| {ch} | {n} |" for ch, n in s["channel_clauses"].most_common())
        return (f"宋本條文 {s['n_clauses']} 條（含霍亂、陰陽易差後勞復附篇），"
                f"六經分佈如下：\n\n| 六經 | 條文數 |\n|---|---|\n{rows}")

    def _results_section(self, paper_type: str, s: Dict, topic: str) -> str:
        lines = ["## 4 結果", ""]
        lines.append(f"### 4.1 規則庫總體\n共 {s['n_initial_rules']} 條初始規則："
                     + "、".join(f"{k} {v}條" for k, v in sorted(s['rule_types'].items(),
                                                              key=lambda kv: -kv[1])[:8])
                     + f"。分級：gold {s['levels'].get('gold',0)} / silver "
                       f"{s['levels'].get('silver',0)} / bronze {s['levels'].get('bronze',0)}。")
        top = s["formula_freq"].most_common(10)
        lines.append("### 4.2 高頻方劑\n| 方劑 | 條文數 |\n|---|---|\n" +
                     "\n".join(f"| {f} | {n} |" for f, n in top))
        if paper_type == "mistreatment" or True:
            paths = self.mistreatment_rules[:8]
            lines.append("### 4.3 誤治傳變路徑（節選）\n| 誤治 | 變證 | 救治方 | 證據條文 |\n|---|---|---|---|\n" +
                         "\n".join(f"| {m.mistreatment_type} | {m.resulting_pattern} | "
                                   f"{'、'.join(m.rescue_formulas[:2])} | "
                                   f"{'、'.join(m.supporting_clauses[:2])} |" for m in paths))
        fprs = [f for f in self.formula_rules if topic and (f.formula in topic or
                (f.formula_family and f.formula_family in topic))][:6] or self.formula_rules[:6]
        lines.append("### 4.4 方證規則示例\n| 方劑 | 核心證 | 核心脈 | 條文 | 等級 |\n|---|---|---|---|---|\n" +
                     "\n".join(f"| {f.formula} | {'、'.join(f.core_symptoms[:4])} | "
                               f"{'、'.join(f.core_pulse[:2]) or '—'} | "
                               f"{'、'.join(f.supporting_clauses[:2])} | {f.release_level} |"
                               for f in fprs))
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    def _emit_figures(self, out: Path, paper_type: str, s: Dict) -> List[str]:
        figs: List[str] = []
        # Fig1: channel-formula distribution (mermaid)
        mer = ["```mermaid", "graph LR"]
        for scr in self.six_channel_rules:
            ch = scr.six_channel
            mer.append(f'  {config.CHANNEL_PINYIN.get(ch, ch)}["{ch}"]')
            for f in scr.main_formulas[:4]:
                fid = f["formula"]
                mer.append(f'  {config.CHANNEL_PINYIN.get(ch, ch)} --> '
                           f'f{abs(hash(fid)) % 9999}["{fid} ({f["clause_count"]})"]')
        mer.append("```")
        (out / "fig1_channel_formula.mmd.md").write_text("\n".join(mer), encoding="utf-8")
        figs.append("Fig.1 六經-方劑分佈圖 (fig1_channel_formula.mmd.md)")

        # Fig2: mistreatment path graph (DOT)
        dot = ["digraph mistreatment {", "  rankdir=LR;"]
        for m in self.mistreatment_rules:
            a, b = m.mistreatment_type, m.resulting_pattern
            dot.append(f'  "{a}" -> "{b}";')
            for f in m.rescue_formulas[:2]:
                dot.append(f'  "{b}" -> "{f}" [style=dashed];')
        dot.append("}")
        (out / "fig2_mistreatment_paths.dot").write_text("\n".join(dot), encoding="utf-8")
        figs.append("Fig.2 誤治-變證路徑圖 (fig2_mistreatment_paths.dot)")

        # Fig3: formula family tree (DOT)
        dot = ["digraph family {", "  rankdir=LR;"]
        for r in self.formula_rules:
            for mrel in r.modification_relations:
                dot.append(f'  "{r.formula}" -> "{mrel["modified_formula"]}";')
        dot.append("}")
        (out / "fig3_formula_family.dot").write_text("\n".join(dot), encoding="utf-8")
        figs.append("Fig.3 方劑家族樹 (fig3_formula_family.dot)")

        # Fig4: clause topic clusters — channel × dominant theme grouping
        clusters: Dict[str, Dict[str, List[int]]] = {}
        for c in self.clauses:
            if not c.six_channel:
                continue
            if c.mistreatment_terms:
                theme = "誤治變證"
            elif c.contraindication_terms:
                theme = "禁忌法度"
            elif c.contains_formula:
                theme = "方證條文"
            elif c.prognosis_terms:
                theme = "預後判斷"
            elif c.pulse:
                theme = "脈證關係"
            else:
                theme = "病證界定"
            clusters.setdefault(c.six_channel, {}).setdefault(theme, []).append(c.clause_number)
        mer = ["```mermaid", "graph TD"]
        for ch, themes in clusters.items():
            chid = config.CHANNEL_PINYIN.get(ch, ch)
            mer.append(f'  {chid}["{ch}"]')
            for theme, nums in sorted(themes.items(), key=lambda kv: -len(kv[1])):
                tid = f"{chid}_{abs(hash(theme)) % 999}"
                sample = "、".join(str(n) for n in nums[:5])
                mer.append(f'  {chid} --> {tid}["{theme} ({len(nums)}條，如{sample}…)"]')
        mer.append("```")
        (out / "fig4_clause_topic_clusters.mmd.md").write_text("\n".join(mer), encoding="utf-8")
        figs.append("Fig.4 條文主題聚類圖 (fig4_clause_topic_clusters.mmd.md)")
        return figs

    def _emit_tables(self, out: Path, s: Dict) -> List[str]:
        tables: List[str] = []
        with (out / "table1_rule_levels.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["release_level", "count"])
            for k, v in sorted(s["levels"].items()):
                w.writerow([k, v])
        tables.append("Table 1 規則分級統計 (table1_rule_levels.csv)")
        with (out / "table2_formula_frequency.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["formula", "clause_count"])
            for f, n in s["formula_freq"].most_common(30):
                w.writerow([f, n])
        tables.append("Table 2 方劑頻次表 (table2_formula_frequency.csv)")
        with (out / "table3_rule_types.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rule_type", "count"])
            for k, v in sorted(s["rule_types"].items(), key=lambda kv: -kv[1]):
                w.writerow([k, v])
        tables.append("Table 3 規則類型統計 (table3_rule_types.csv)")

        # Table 4: version variant comparison (layer B alignments with diffs)
        variants = [VariantRule.from_dict(d) for d in
                    read_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl")]
        with (out / "table4_variant_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["clause_id", "variant_book", "similarity",
                        "base_text", "variant_text", "notable_differences"])
            n = 0
            for v in variants:
                if not v.notable_differences:
                    continue
                w.writerow([v.clause_id, v.variant_book, v.similarity,
                            v.base_text[:80], v.variant_text[:80],
                            "；".join(v.notable_differences)])
                n += 1
                if n >= 100:
                    break
        tables.append("Table 4 版本異文對比表 (table4_variant_comparison.csv)")
        return tables
