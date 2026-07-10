"""PaperWriterAgent — full manuscript generation grounded in mined data.

Six paper types (per protocol). Every Results statement carries rule IDs /
clause IDs; figures are emitted as DOT/Mermaid sources + CSV tables under
data/shanghan/papers/<slug>/ so the manuscript is reproducible.

Trusted-base + augmentation-layer split: templates own the structure and the
data tables (deterministic, recomputable); the LLM reads the quantitative
research assets (frequency tables, co-occurrence networks, mistreatment
paths) and drafts 引言/計量結果解讀/討論/結論. Its prose passes the
CitationGuard before landing in the manuscript — 無證據鏈，不成回答 holds
for machine-written papers too. Offline, the `local` backend produces
deterministic sections through the same code path.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from ..schemas import (CommentaryRule, DifferentialRule, FormulaPatternRule,
                       InitialRule, MistreatmentTransformationRule,
                       ShanghanClause, SixChannelRule, VariantRule, read_jsonl)

PAPER_TYPES = {
    "formula_pattern": "《傷寒論》方證規律挖掘",
    "six_channel_kg": "《傷寒論》六經辨證知識圖譜",
    "mistreatment": "《傷寒論》誤治傳變規則研究",
    # 十五輪 十三.1：不冒充網絡藥理學（無化合物/靶點/PPI/富集）——如實
    # 命名為共現網絡研究，定位為網絡藥理學的古籍證據前置層
    "network_pharmacology": "《傷寒論》方劑—證候共現網絡研究（網絡藥理學前置）",
    "commentary_compare": "《傷寒論》方劑歷代注釋比較",
    "methodology": "《傷寒論》古籍數據挖掘與智能體方法學研究",
    "benchmark": "《傷寒論》規則系統客觀評測（遮方預測/醫案回放/證據接地）",
    "provenance": "《傷寒論》學術溯源研究（深度研究循環自動生成）",
}


class PaperWriter:
    def __init__(self, clauses: List[ShanghanClause],
                 initial_rules: List[InitialRule],
                 formula_rules: List[FormulaPatternRule],
                 six_channel_rules: List[SixChannelRule],
                 mistreatment_rules: List[MistreatmentTransformationRule],
                 differential_rules: Optional[List[DifferentialRule]] = None,
                 commentary_rules: Optional[List[CommentaryRule]] = None,
                 llm_client=None):
        self.all_clauses = list(clauses)          # incl. AUX — citation store
        self.clauses = [c for c in clauses if c.text_type == "original_clause"]
        self.initial_rules = initial_rules
        self.formula_rules = formula_rules
        self.six_channel_rules = six_channel_rules
        self.mistreatment_rules = mistreatment_rules
        self.differential_rules = differential_rules or []
        self.commentary_rules = commentary_rules
        self._llm_client = llm_client

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
    def _research_digest(self, s: Dict, topic: str) -> Dict:
        """Compact digest of the quantitative research assets for the LLM.

        Reads data/shanghan/research/ when present (the pipeline's canonical
        outputs); otherwise recomputes the same statistics in-memory. Only
        clause_ids listed here may be cited by the drafted prose.
        """
        rd = config.RESEARCH_DIR

        def _load_json(name):
            p = rd / name
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return None
            return None

        sym_net = _load_json("formula_symptom_network.json")
        paths = _load_json("mistreatment_paths.json")
        tree = _load_json("formula_family_tree.json")
        if sym_net is None or paths is None or tree is None:
            from ..apps.research import ResearchMiner
            miner = ResearchMiner(self.clauses, self.formula_rules,
                                  self.mistreatment_rules)
            sym_net = sym_net or miner.cooccurrence("symptom")
            paths = paths or miner.mistreatment_paths()
            tree = tree or miner.family_tree()

        edges = sym_net.get("edges", [])
        degree: Counter = Counter()
        for e in edges:
            degree[e["formula"]] += 1
        sym_freq: Counter = Counter()
        pulse_freq: Counter = Counter()
        for c in self.clauses:
            sym_freq.update(c.symptoms)
            pulse_freq.update(c.pulse)

        topic_rules = [f for f in self.formula_rules
                       if topic and (f.formula in topic or
                                     (f.formula_family and f.formula_family in topic))]
        sample_rules = (topic_rules or
                        sorted(self.formula_rules,
                               key=lambda f: -len(f.supporting_clauses)))[:6]
        families = sorted(tree.get("families", []),
                          key=lambda fam: -len(fam.get("modifications", [])))
        return {
            "n_clauses": s["n_clauses"],
            "n_initial_rules": s["n_initial_rules"],
            "release_levels": s["levels"],
            "n_formula_rules": s["n_formula_rules"],
            "n_mistreatment": s["n_mistreatment"],
            "n_differential": s["n_differential"],
            "channel_clauses": s["channel_clauses"].most_common(),
            "top_formulas": s["formula_freq"].most_common(10),
            "top_symptoms": sym_freq.most_common(10),
            "top_pulses": pulse_freq.most_common(8),
            "symptom_edge_count": len(edges),
            "top_symptom_edges": edges[:10],
            "network_hubs": [{"formula": f, "degree": d}
                             for f, d in degree.most_common(8)],
            "top_families": [{"base": fam["base"],
                              "n_modifications": len(fam["modifications"])}
                             for fam in families[:6]],
            "mistreatment_paths": [{
                "mistreatment": p["mistreatment"],
                "resulting_pattern": p["resulting_pattern"],
                "rescue_formulas": p.get("rescue_formulas", [])[:3],
                "clauses": p.get("clauses", [])[:3],
            } for p in paths[:8]],
            "sample_formula_rules": [{
                "formula": f.formula,
                "core_symptoms": f.core_symptoms[:4],
                "core_pulse": f.core_pulse[:2],
                "supporting_clauses": f.supporting_clauses[:3],
            } for f in sample_rules],
            "benchmark": self._benchmark_digest(),
            "commentary_atlas": self._atlas_digest(),
            "dosimetry": self._dose_digest(),
            "deep_research": ({
                "n_rounds": self._dossier["n_rounds"],
                "coverage": self._dossier["coverage"],
                "findings": [{"dimension": f["dimension"],
                              "summary": f["summary"],
                              "verified_clause_ids": f.get("verified_clause_ids", [])[:3]}
                             for f in self._dossier["findings"]],
            } if getattr(self, "_dossier", None) else {}),
        }

    def _atlas_digest(self) -> Dict:
        a = self._load_research("commentary_divergence.json")
        if not a:
            return {}
        ag = sorted(a["agreement_matrix"], key=lambda x: -x["mean_term_agreement"])
        return {"n_books": a["n_books"],
                "n_commentary_rules": a["n_commentary_rules"],
                "n_clauses_multi_commentator": a["n_clauses_multi_commentator"],
                "book_coverage": {b: c["n_aligned_clauses"]
                                  for b, c in a["book_coverage"].items()},
                "most_agreeing_pair": ag[0] if ag else None,
                "most_diverging_pair": ag[-1] if ag else None,
                "top_divergent_clauses": [
                    {"clause_id": t["clause_id"],
                     "n_commentators": t["n_commentators"],
                     "term_divergence": t["term_divergence"]}
                    for t in a["top_divergent_clauses"][:3]]}

    def _dose_digest(self) -> Dict:
        summ = self._load_research("dose_summary.json")
        evo = self._load_research("dose_family_evolution.json")
        if not summ:
            return {}
        dose_only = [e for e in evo.get("edges", [])
                     if e["dose_deltas"] and not e["added_herbs"]
                     and not e["removed_herbs"]]
        return {"parse_coverage": summ.get("parse_coverage", {}),
                "heaviest_formulas": summ.get("heaviest_formulas_kaogu_g", [])[:3],
                "n_dose_delta_edges": evo.get("n_with_dose_delta", 0),
                "dose_only_edges": [
                    {"base": e["base"], "modified": e["modified"],
                     "delta": e["dose_deltas"][0]} for e in dose_only[:3]]}

    def _benchmark_digest(self) -> Dict:
        """Compact evaluation metrics for the drafting layer ({} if not run)."""
        ev = self._load_eval()
        out: Dict = {}
        cz = ev.get("cloze", {}).get("metrics", {})
        if cz.get("attainable"):
            out["cloze_attainable"] = cz["attainable"]
            out["cloze_singleton_n"] = cz.get("singleton_unattainable", {}).get("n", 0)
        cs = ev.get("cases", {})
        if cs:
            out["case_replay"] = {**cs.get("metrics", {}),
                                  "n_out_of_scope": cs.get("n_out_of_scope", 0),
                                  "source": cs.get("source", "")}
        gr = ev.get("grounding", {})
        if gr:
            out["grounding"] = {**gr.get("metrics", {}),
                                "backend": gr.get("backend", "")}
        return out

    def _draft_sections(self, paper_type: str, title_root: str, topic: str,
                        digest: Dict) -> Dict:
        """LLM (or offline-deterministic) 引言/計量解讀/討論/結論 + guard report."""
        from ..agent.citation_guard import CitationGuard
        from ..llm.client import get_client
        client = self._llm_client or get_client()
        try:
            draft = client.draft_paper(paper_type, title_root, topic, digest) or {}
        except Exception:
            draft = {}
        sections = {k: v.strip() for k, v in draft.items()
                    if k in ("introduction", "quant_interpretation",
                             "discussion", "conclusion")
                    and isinstance(v, str) and v.strip()}
        report = None
        if sections:
            guard = CitationGuard({c.clause_id: c for c in self.all_clauses})
            report = guard.check("\n".join(sections.values()))
        return {"sections": sections, "citation_report": report,
                "backend": client.backend}

    # ------------------------------------------------------------------
    def generate(self, paper_type: str = "formula_pattern",
                 topic: str = "", out_dir: Optional[Path] = None,
                 use_llm: bool = True) -> Path:
        if paper_type not in PAPER_TYPES:
            # 十五輪 十三.2：fail-fast——不得靜默退回默認類型生成另一篇論文
            raise ValueError(f"未知論文類型 {paper_type!r}"
                             f"（可用：{sorted(PAPER_TYPES)}）")
        title_root = PAPER_TYPES[paper_type]
        topic = topic or {"formula_pattern": "桂枝湯類方證",
                          "six_channel_kg": "六經辨證體系",
                          "mistreatment": "誤治傳變路徑",
                          "network_pharmacology": "經方藥物網絡",
                          "commentary_compare": "桂枝湯歷代注釋",
                          "methodology": "Hermes自主審核框架",
                          "benchmark": "遮方預測與醫案回放基準",
                          "provenance": "桂枝湯類方源流"}[paper_type]
        s = self._stats()
        # 十五輪 P0-3：版本化輸出——修訂目錄按**內容指紋**尋址（同輸入
        # 同目錄=冪等重生成；輸入變化=新修訂目錄），不再按日期覆蓋。
        # revisions.json 追加式記錄修訂史（審稿修改/語料換版可回溯）。
        revision = self._revision_id(paper_type, topic, s)
        if out_dir is None:
            base = config.PAPER_DIR / paper_type
            out = base / f"rev_{revision}"
            out.mkdir(parents=True, exist_ok=True)
            self._record_revision(base, revision, topic)
        else:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)

        # provenance papers run the deep-research loop first; its dossier
        # becomes both a results section and drafting-layer input
        self._dossier = None
        if paper_type == "provenance":
            from ..agent.research_loop import DeepResearcher
            self._dossier = DeepResearcher().run(topic)

        # ---------- figures & tables (reproducible assets) -----------------
        fig_out = self._emit_figures(out, paper_type, s)
        figures = fig_out["figs"]
        fig_labels = fig_out["fig_labels"]
        tables = self._emit_tables(out, s)

        # ---------- augmentation layer: LLM drafts from the research digest
        digest = self._research_digest(s, topic)
        draft = (self._draft_sections(paper_type, title_root, topic, digest)
                 if use_llm else {"sections": {}, "citation_report": None,
                                  "backend": "disabled"})
        drafted = draft["sections"]

        title = f"基於條文級規則挖掘與自主審核的{title_root}：以{topic}為例"
        gold = s["levels"].get("gold", 0)
        silver = s["levels"].get("silver", 0)
        bronze = s["levels"].get("bronze", 0)
        top_f = s["formula_freq"].most_common(5)

        abstract = f"""目的：將《傷寒論》宋本條文轉化為可回源、可審核的結構化規則體系，並以{topic}為例驗證方法可行性。
方法：以宋本（趙開美本）現代編號{ s['n_clauses']}條為唯一原文證據層（A層），桂林古本等為異文層（B層），成無己注為注釋層（C層）；經條文切分、否定感知實體抽取、條文級初始規則抽取，再通過 Schema 校驗、證據回源驗證、語義審查、對抗式批評（ShanghanCritic）、自動修復與共識評級六道閘門完成自主審核。
結果：共獲得初始規則{ s['n_initial_rules']}條（gold {gold}、silver {silver}、bronze {bronze}），方證規則{ s['n_formula_rules']}個，誤治傳變路徑{ s['n_mistreatment']}條，鑒別規則{ s['n_differential']}組；高頻方劑為{ '、'.join(f for f, _ in top_f)}。證據分層聲明：原文事實逐條錨定條文編號（A層）；聚合統計、注家一致度與模型評測屬派生指標，錨定其數據資產與代碼指紋（D/E層），不以單條條文冒充依據。
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

        results = self._results_section(paper_type, s, topic, digest,
                                        fig_labels=fig_labels)

        intro_body = drafted.get("introduction") or (
            "《傷寒論》以六經統病、以方證相應，其條文之間存在並列、遞進、"
            "誤治轉變、鑒別與禁忌等強結構關係。既往數字化工作多止於全文檢索或人工標註，"
            "缺少「規則必須回到原文」的硬性約束。本文提出 Hermes-Shanghanlun 框架，"
            "把每一條原文轉化為可追蹤規則，使每一個方證判斷都能回到條文編號。")

        quant_body = drafted.get("quant_interpretation", "")
        quant_section = ""
        if quant_body:
            quant_section = ("## 5 計量結果解讀\n\n"
                             "（本節由增益層基於 research/ 計量資產撰寫，屬 D/E 層歸納，"
                             "引用已過 CitationGuard 核驗。）\n\n" + quant_body)

        discussion = "## 6 討論\n\n" + (drafted.get("discussion") or
            """（1）證據邊界。本研究嚴格區分原文直述與後世歸納：如「營衛不和」屬後世
病機術語，批評器將其攔截出規則主體並降級為模型解釋層；「可與」與「主之」
的證據強度差異被顯式建模，避免將斟酌之辭讀作必用之訓。
（2）版本異文的影響。B層對齊顯示宋本與桂本在部分條文存在用字差異，
規則層僅以宋本為準、異文以 VariantRule 並行記錄，供版本學研究取用。
（3）侷限。實體詞典覆蓋率有限，個別罕見證候表達可能漏標；
亞型命名（如太陽蓄水）屬後世歸納框架，雖逐一錨定條文，仍不宜回讀為仲景原意；
自動審核可保證「不偽造證據」，但不能替代學科專家對歸納合理性的終審。
（4）應用。規則庫已編譯為六經、方證、誤治、禁忌、鑒別等 Skill，
可服務醫師輔助（標註輔助性質）、科研挖掘（共現網絡、知識圖譜）與教學練習；
患者端嚴格禁用診斷與處方功能。""")

        conclusion = "## 7 結論\n\n" + (drafted.get("conclusion") or
            """以條文為最小證據單位、以自主審核為質量閘門的 Hermes 流水線，能夠將
《傷寒論》整書轉化為層級化、可回源、可調用的規則系統；所有方證判斷
均可回到條文，所有歸納均標明證據層級，為中醫古籍的可計算化提供了
一條可複製的路徑。""")

        # citation-verification footer for the machine-drafted prose
        report = draft.get("citation_report")
        if report is not None:
            notes = ["", "—" * 12, "【增益層引用核驗】"]
            if report.verified_ids:
                notes.append("已核實條文：" + "、".join(report.verified_ids))
            if report.unsupported_ids:
                notes.append("⚠️ 未能核實的條文編號（請勿採信）："
                             + "、".join(report.unsupported_ids))
            if report.quote_mismatches:
                notes.append("⚠️ 有引文未能在所引條文中逐字核對。")
            if not report.has_any_citation:
                notes.append("⚠️ 增益層文本未包含可核驗的條文編號，僅供參考。")
            conclusion += "\n" + "\n".join(notes)

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
（稿件修訂 rev_{revision}；語料指紋見 paper_meta.json——日期於投稿時填寫）"""

        parts = [
            f"# {title}", "## 摘要\n\n" + abstract,
            "## 1 引言\n\n" + intro_body,
            "## 2 數據\n\n" + self._data_section(s),
            methods, results,
        ]
        if quant_section:
            parts.append(quant_section)
        parts += [
            discussion, conclusion,
            self._legends_section(fig_out["legends"]),
            "## 圖表清單\n\n" + "\n".join(f"- {f}" for f in figures + tables),
            references, cover_letter,
            "## Supplementary Materials\n\n- S1 規則庫 rules_initial/initial_rules.jsonl\n"
            "- S2 審計日誌 audit/audit_log.jsonl\n- S3 條文關係圖 relations/clause_relations.jsonl\n"
            "- S4 共現網絡與家族樹 research/*.json\n"
            "- S5 Source Data source_data/（逐 panel CSV，見 figures_manifest.json）\n"
            "- S6 Figure QA figure_qa/qa_report.json（含 CVD 模擬 ΔE 實測）",
        ]
        manuscript = "\n\n".join(parts)
        (out / "manuscript.md").write_text(manuscript, encoding="utf-8")
        (out / "figure_legends.json").write_text(
            json.dumps(fig_out["legends"], ensure_ascii=False, indent=1),
            encoding="utf-8")
        # 十五輪 十七：Figure QA 是發布門禁——硬違例即失敗，不靜默出廠
        from .figure_qa import run_qa
        qa = run_qa(out, fig_out["emitted"], manuscript, fig_out["legends"])
        if not qa["ok"]:
            raise RuntimeError("Figure QA 硬違例："
                               + "；".join(qa["hard_violations"]))
        meta = {"paper_type": paper_type, "title": title, "topic": topic,
                "revision": f"rev_{revision}",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "figures": figures, "tables": tables,
                "figure_plan": list(fig_labels),
                "figure_qa": {"ok": qa["ok"],
                              "soft_issues": qa["soft_issues"],
                              "palette_cvd_ok": qa["palette_cvd"]["ok"]},
                "llm_backend": draft.get("backend", "disabled"),
                "llm_sections": sorted(drafted.keys()),
                "citation_report": (report.to_dict() if report is not None else None),
                "statistics": {k: v for k, v in s.items()
                               if k not in ("formula_freq", "channel_clauses")}}
        (out / "paper_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return out / "manuscript.md"

    # ------------------------------------------------------------------
    def _revision_id(self, paper_type: str, topic: str, s: Dict) -> str:
        """修訂指紋：語料 manifest + 論文類型 + 主題 + 統計快照——同輸入
        冪等，變輸入開新目錄（P0-3：不覆蓋、可回溯、可比對）。"""
        import hashlib
        m = config.MANIFEST_DIR / "corpus_manifest.json"
        corpus = hashlib.sha256(m.read_bytes()).hexdigest()[:12] \
            if m.exists() else "no-manifest"
        stats_blob = json.dumps(
            {k: v for k, v in sorted(s.items())
             if k not in ("formula_freq", "channel_clauses")},
            ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(
            f"{corpus}|{paper_type}|{topic}|{stats_blob}".encode()).hexdigest()[:12]

    @staticmethod
    def _record_revision(base: Path, revision: str, topic: str) -> None:
        path = base / "revisions.json"
        entries = []
        if path.exists():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                entries = []
        if not any(e.get("revision") == f"rev_{revision}" for e in entries):
            entries.append({"revision": f"rev_{revision}", "topic": topic,
                            "first_generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            path.write_text(json.dumps(entries, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    @staticmethod
    def _legends_section(legends: Dict[str, Dict]) -> str:
        """正式 Figure Legends（十五輪 P0-6）：每圖標題/panel/n/數據源/
        誤差定義/證據層/Source Data 指向。"""
        if not legends:
            return "## 圖例（Figure Legends）\n\n（本論文類型無圖組。）"
        blocks = ["## 圖例（Figure Legends）", ""]
        for leg in legends.values():
            panels = "；".join(f"({p}) {d}" for p, d in leg["panels"].items())
            blocks.append(
                f"**{leg['fig_no']}｜{leg['title']}** {panels}。"
                f"n：{leg['n']}。數據來源：{leg['data_source']}。"
                f"誤差/統計定義：{leg['error_definition']}。"
                f"證據層級：{leg['evidence_level']}。"
                f"Source Data：{'、'.join('source_data/' + f for f in leg['source_data'])}。"
                + (f"備註：{leg['abbreviations']}。" if leg["abbreviations"] else ""))
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    def _data_section(self, s: Dict) -> str:
        rows = "\n".join(f"| {ch} | {n} |" for ch, n in s["channel_clauses"].most_common())
        return (f"宋本條文 {s['n_clauses']} 條（含霍亂、陰陽易差後勞復附篇），"
                f"六經分佈如下：\n\n| 六經 | 條文數 |\n|---|---|\n{rows}")

    def _results_section(self, paper_type: str, s: Dict, topic: str,
                         digest: Dict,
                         fig_labels: Optional[Dict[str, str]] = None) -> str:
        fig_labels = fig_labels or {}
        lines = ["## 4 結果", ""]
        # 十五輪 P0-5：圖組進入論證鏈——按編號順序給出每張圖回答的科學
        # 問題與主信息（正文引用，不是文末附件清單）
        if fig_labels:
            from .figspec import FIGURE_SPECS
            nav = []
            for key, label in fig_labels.items():
                sp = FIGURE_SPECS[key]
                nav.append(f"如 {label} 所示，{sp.main_message}"
                           f"（回答：{sp.scientific_question}）")
            lines.append("### 4.0 圖組導覽\n" + "；".join(nav) + "。")
        lines.append(f"### 4.1 規則庫總體\n共 {s['n_initial_rules']} 條初始規則："
                     + "、".join(f"{k} {v}條" for k, v in sorted(s['rule_types'].items(),
                                                              key=lambda kv: -kv[1])[:8])
                     + f"。分級：gold {s['levels'].get('gold',0)} / silver "
                       f"{s['levels'].get('silver',0)} / bronze {s['levels'].get('bronze',0)}。")
        top = s["formula_freq"].most_common(10)
        lines.append("### 4.2 高頻方劑\n| 方劑 | 條文數 |\n|---|---|\n" +
                     "\n".join(f"| {f} | {n} |" for f, n in top) +
                     (f"\n\n（分佈與計數口徑見 {fig_labels['formula_freq']} "
                      "及其圖例。）" if "formula_freq" in fig_labels else ""))
        n_sub = 2

        if paper_type in ("mistreatment", "methodology"):
            n_sub += 1
            paths = self.mistreatment_rules[:8]
            ref = (f"（完整路徑網絡見 {fig_labels['mistreatment_paths']}。）"
                   if "mistreatment_paths" in fig_labels else "")
            lines.append(f"### 4.{n_sub} 誤治傳變路徑（節選）{ref}\n| 誤治 | 變證 | 救治方 | 證據條文 |\n|---|---|---|---|\n" +
                         "\n".join(f"| {m.mistreatment_type} | {m.resulting_pattern} | "
                                   f"{'、'.join(m.rescue_formulas[:2])} | "
                                   f"{'、'.join(m.supporting_clauses[:2])} |" for m in paths))

        if paper_type == "network_pharmacology":
            n_sub += 1
            hubs = digest.get("network_hubs", [])
            edges = digest.get("top_symptom_edges", [])
            lines.append(f"### 4.{n_sub} 方-證共現網絡\n"
                         f"共 {digest.get('symptom_edge_count', 0)} 條方-症共現邊。"
                         "樞紐方（按關聯證候數）：\n\n| 方劑 | 關聯證候數 |\n|---|---|\n" +
                         "\n".join(f"| {h['formula']} | {h['degree']} |" for h in hubs[:8]) +
                         "\n\n最強共現邊：\n\n| 方劑 | 症狀 | 權重 |\n|---|---|---|\n" +
                         "\n".join(f"| {e['formula']} | {e['symptom']} | {e['weight']} |"
                                   for e in edges[:8]))
            n_sub += 1
            herb_freq: Counter = Counter()
            for c in self.clauses:
                herb_freq.update(c.herbs)
            lines.append(f"### 4.{n_sub} 高頻藥物（網絡藥理學靶點篩選候選）\n"
                         "| 藥物 | 條文數 |\n|---|---|\n" +
                         "\n".join(f"| {h} | {n} |" for h, n in herb_freq.most_common(12)))

        if paper_type == "commentary_compare":
            n_sub += 1
            lines.append(f"### 4.{n_sub} 多注家對齊示例\n" +
                         self._commentary_table(topic))
            n_sub += 1
            ref = (f"（一致度矩陣與共注 n 見 {fig_labels['commentator_agreement']}。）"
                   if "commentator_agreement" in fig_labels else "")
            lines.append(f"### 4.{n_sub} 注家分歧圖譜{ref}\n" + self._atlas_tables())

        if paper_type == "network_pharmacology":
            n_sub += 1
            ref = (f"（折算假設區間見 {fig_labels['dose_totals']}。）"
                   if "dose_totals" in fig_labels else "")
            lines.append(f"### 4.{n_sub} 劑量計量層{ref}\n" + self._dose_tables())
            n_sub += 1
            lines.append(f"### 4.{n_sub} 研究定位聲明\n本研究是**方劑—證候"
                         "共現網絡**分析（古籍證據層），不含化合物/靶點/"
                         "蛋白互作/富集分析與分子驗證——不冒充傳統意義的"
                         "網絡藥理學；其價值在為後續網絡藥理學研究提供"
                         "可回源的經典證據前置圖譜。")

        if paper_type == "methodology":
            n_sub += 1
            lines.append(f"### 4.{n_sub} 審核閘門通過情況\n" + self._audit_table(s))

        if paper_type == "benchmark":
            n_sub += 1
            ref = (f"（指標總覽見 {fig_labels['benchmark']}；各任務樣本量"
                   "見其 Source Data。）" if "benchmark" in fig_labels else "")
            lines.append(f"### 4.{n_sub} 客觀評測結果{ref}\n"
                         + self._benchmark_tables())

        if paper_type == "provenance" and self._dossier:
            n_sub += 1
            lines.append(f"### 4.{n_sub} 深度研究循環溯源發現\n"
                         + self._provenance_tables(self._dossier))

        if paper_type in ("formula_pattern", "six_channel_kg", "mistreatment"):
            n_sub += 1
            fprs = [f for f in self.formula_rules if topic and (f.formula in topic or
                    (f.formula_family and f.formula_family in topic))][:6] or self.formula_rules[:6]
            lines.append(f"### 4.{n_sub} 方證規則示例\n| 方劑 | 核心證 | 核心脈 | 條文 | 等級 |\n|---|---|---|---|---|\n" +
                         "\n".join(f"| {f.formula} | {'、'.join(f.core_symptoms[:4])} | "
                                   f"{'、'.join(f.core_pulse[:2]) or '—'} | "
                                   f"{'、'.join(f.supporting_clauses[:2])} | {f.release_level} |"
                                   for f in fprs))
        return "\n\n".join(lines)

    @staticmethod
    def _cell(text: str, limit: int) -> str:
        """Markdown-table-safe cell: strip newlines/pipes before truncating."""
        return (text or "").replace("\n", "").replace("|", "／")[:limit]

    def _commentary_table(self, topic: str) -> str:
        rules = self.commentary_rules
        if rules is None:
            rules = [CommentaryRule.from_dict(d) for d in read_jsonl(
                config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")]
        store = {c.clause_id: c for c in self.all_clauses}
        # prefer commentaries on clauses that mention the topic formula
        def relevant(r):
            c = store.get(r.clause_id)
            return bool(c and topic and any(f in topic for f in c.formula_names))
        picked = [r for r in rules if relevant(r)][:6] or rules[:6]
        if not picked:
            return "（無注文對齊數據。）"
        rows = []
        for r in picked:
            base = store.get(r.clause_id)
            base_text = self._cell(base.clean_text, 40) + "…" if base else "—"
            rows.append(f"| {r.clause_id} | {r.commentator} | {base_text} | "
                        f"{self._cell(r.commentary_text, 40)}… | "
                        f"{r.alignment_similarity:.2f} |")
        return ("| 條文 | 注家 | 原文（A層） | 注文（C層） | 對齊相似度 |\n"
                "|---|---|---|---|---|\n" + "\n".join(rows))

    @staticmethod
    def _load_research(name: str) -> Dict:
        p = config.RESEARCH_DIR / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _atlas_tables(self) -> str:
        a = self._load_research("commentary_divergence.json")
        if not a:
            return "（尚無分歧圖譜：請先運行 pipeline。）"
        cov = "\n".join(
            f"| {b} | {c['commentator']} | {c['n_aligned_clauses']} | {c['mean_similarity']} |"
            for b, c in a["book_coverage"].items())
        top = "\n".join(
            f"| {t['clause_id']} | {t['n_commentators']} | {t['term_divergence']} | "
            f"{self._cell(t['clause_text'], 28)}… |"
            for t in a["top_divergent_clauses"][:8])
        ag = sorted(a["agreement_matrix"], key=lambda x: -x["mean_term_agreement"])
        pairs = ag[:3] + ag[-3:]
        agr = "\n".join(f"| {p['a']} × {p['b']} | {p['mean_term_agreement']} | "
                        f"{p['n_shared_clauses']} |" for p in pairs)
        return (f"九注本條文級對齊共 {a['n_commentary_rules']} 條注文，"
                f"{a['n_clauses_multi_commentator']} 條條文有 ≥2 位注家。\n\n"
                f"**各注本對齊覆蓋（低覆蓋為結構性事實，不填充）**\n\n"
                f"| 注本 | 注家 | 對齊條數 | 平均相似度 |\n|---|---|---|---|\n{cov}\n\n"
                f"**分歧最大條文（術語剖面 Jaccard 距離）**\n\n"
                f"| 條文 | 注家數 | 分歧度 | 原文 |\n|---|---|---|---|\n{top}\n\n"
                f"**注家一致度矩陣（最相近3對 / 最分歧3對）**\n\n"
                f"| 注家對 | 術語一致度 | 共注條數 |\n|---|---|---|\n{agr}")

    def _dose_tables(self) -> str:
        ratios = self._load_research("dose_ratios.json")
        evo = self._load_research("dose_family_evolution.json")
        summ = self._load_research("dose_summary.json")
        if not ratios:
            return "（尚無劑量資產：請先運行 pipeline。）"
        rt = "\n".join(f"| {f['formula']} | {f['ratio'][:40]} | "
                       f"{f['total_weight_g']['kaogu']} / "
                       f"{f['total_weight_g']['duliangheng']} / "
                       f"{f['total_weight_g']['zhezhuan']} |"
                       for f in ratios["formulas"][:10])
        rows = []
        for e in evo.get("edges", []):
            for d in e["dose_deltas"][:1]:
                rows.append(f"| {e['base']} → {e['modified']} | {e['edge_kind']} | "
                            f"{d['herb']}：{d['base_raw'][:8]}→{d['mod_raw'][:8]}"
                            f"（×{d['factor']}） |")
        cov = summ.get("parse_coverage", {})
        return (f"劑量比例以銖當量計、與折算學派無關；絕對質量按三家折算並存"
                f"（考古實測/度量衡史/明清折算，D/E 層標註）。解析 "
                f"{cov.get('n_rows','—')} 條劑量，未解析 {cov.get('n_unparsed','—')} "
                f"條（逐一列於 dose_table.json）。\n\n"
                f"**方內藥量比（前10方；總量 g：考古/度量衡史/折算）**\n\n"
                f"| 方 | 銖當量比 | 總量(g) |\n|---|---|---|\n{rt}\n\n"
                f"**家族樹劑量演化（加味≠增量；dose-only 邊 "
                f"{evo.get('n_dose_only_edges','—')} 條）**\n\n"
                f"| 方對 | 邊類型 | 劑量變化 |\n|---|---|---|\n" + "\n".join(rows[:10]))

    def _provenance_tables(self, dossier: Dict) -> str:
        cov = "、".join(f"{d}×{n}" for d, n in dossier["coverage"].items())
        rounds = "\n".join(
            f"| {r['round']} | " + "；".join(
                f"{t['module']}（{t['reason'][:14]}）" for t in r["tasks"]) + " |"
            for r in dossier["rounds"])
        finds = "\n".join(
            f"| {f['dimension']} | {f['module']} | "
            f"{self._cell(f['summary'], 64)} | "
            f"{'、'.join(f.get('verified_clause_ids', [])[:3]) or '—'} |"
            for f in dossier["findings"])
        return (f"研究循環共 {dossier['n_rounds']} 輪（後端 {dossier['backend']}），"
                f"五維度覆蓋：{cov}；證據條文 "
                f"{len(dossier['evidence_clause_ids'])} 條全部核驗。\n\n"
                f"**循環軌跡（規劃器逐輪選調模塊）**\n\n"
                f"| 輪 | 調用（理由） |\n|---|---|\n{rounds}\n\n"
                f"**溯源發現（子代理產出，逐條引用核驗）**\n\n"
                f"| 維度 | 模塊 | 發現 | 已核實條文 |\n|---|---|---|---|\n{finds}")

    @staticmethod
    def _load_eval() -> Dict:
        """Persisted evaluation results (run `evaluate` first); {} if absent."""
        out: Dict = {}
        d = config.SHANGHAN_DIR / "eval"
        for key, name in (("cloze", "cloze_results.json"),
                          ("ablations", "cloze_ablations.json"),
                          ("cases", "case_results.json"),
                          ("grounding", "grounding_results.json")):
            p = d / name
            if p.exists():
                try:
                    out[key] = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return out

    def _benchmark_tables(self) -> str:
        ev = self._load_eval()
        if not ev:
            return "（尚未運行評測：請先執行 `python3 -m hermes_shanghan evaluate`。）"
        parts: List[str] = []
        cz = ev.get("cloze", {}).get("metrics", {})
        if cz:
            rows = []
            for split, label in (("all", "全部折"), ("attainable", "可達折（金方仍在庫）"),
                                 ("singleton_unattainable", "孤證折（不可達）"),
                                 ("attainable_zhuzhi_only", "可達·僅主之條")):
                m = cz.get(split, {})
                if m.get("n"):
                    rows.append(f"| {label} | {m['n']} | {m.get('top1','—')} | "
                                f"{m.get('top3','—')} | {m.get('top5','—')} | "
                                f"{m.get('mrr','—')} | {m.get('herb_f1','—')} |")
            parts.append("**遮方預測（留一條文，自監督）**\n\n"
                         "| 子集 | n | Top-1 | Top-3 | Top-5 | MRR | 藥物F1 |\n"
                         "|---|---|---|---|---|---|---|\n" + "\n".join(rows))
        ab = ev.get("ablations", {})
        if ab:
            rows = [f"| {k} | {v.get('top1','—')} | {v.get('top3','—')} | {v.get('mrr','—')} |"
                    for k, v in ab.items()]
            parts.append("**匹配器消融（可達折）**\n\n| 配置 | Top-1 | Top-3 | MRR |\n"
                         "|---|---|---|---|\n" + "\n".join(rows))
        cs = ev.get("cases", {})
        if cs:
            m = cs.get("metrics", {})
            parts.append(f"**醫案回放（{cs.get('source','')}，1937 曹穎甫實案）**\n\n"
                         f"解析 {cs.get('n_cases_parsed', 0)} 案（另有病名案 "
                         f"{cs.get('n_non_formula_titles', 0)}）；界外方（多屬金匱）"
                         f"{cs.get('n_out_of_scope', 0)}、證候不足 "
                         f"{cs.get('n_insufficient_findings', 0)}，實評 "
                         f"{m.get('n_scored', 0)} 案：Top-1 {m.get('top1','—')} / "
                         f"Top-3 {m.get('top3','—')} / Top-5 {m.get('top5','—')} / "
                         f"MRR {m.get('mrr','—')}。")
        gr = ev.get("grounding", {})
        if gr:
            m = gr.get("metrics", {})
            parts.append(f"**證據接地（後端：{gr.get('backend','—')}）**\n\n"
                         f"{m.get('n_questions', 0)} 問：完全接地率 "
                         f"{m.get('grounded_answer_rate','—')}、未核實引用率 "
                         f"{m.get('unsupported_citation_rate','—')}、"
                         f"篇均已核實引用 {m.get('mean_verified_per_answer','—')} 條。")
        return "\n\n".join(parts)

    def _audit_table(self, s: Dict) -> str:
        stage: Counter = Counter()
        path = config.AUDIT_DIR / "audit_log.jsonl"
        if path.exists():
            for d in read_jsonl(path):
                stage[d.get("stage", "?")] += 1
        levels = s["levels"]
        rows = "\n".join(f"| {k} | {v} |" for k, v in stage.most_common())
        return (f"六道閘門審計記錄共 {sum(stage.values())} 條：\n\n"
                f"| 閘門階段 | 記錄數 |\n|---|---|\n{rows}\n\n"
                f"發佈分級：gold {levels.get('gold',0)} / silver {levels.get('silver',0)} / "
                f"bronze {levels.get('bronze',0)} / rejected {levels.get('rejected',0)}。")

    # ------------------------------------------------------------------
    # Figure Factory（十五輪 十四）：FIGURE_PLANS 驅動——每種論文類型有
    # 自己的圖組；每張圖帶結構化圖例、逐 panel Source Data、manifest；
    # 網絡圖同時輸出 GraphML + 邊表 CSV + 確定性 layout.json（投稿級
    # 渲染交給 Graphviz/Gephi，本層保證數據與佈局可復現）。
    # ------------------------------------------------------------------
    @staticmethod
    def _write_csv(path: Path, header: List[str], rows: List[List]) -> str:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return path.name

    @staticmethod
    def _write_graphml(path: Path, nodes: List[str],
                       edges: List[Dict]) -> None:
        from xml.sax.saxutils import escape
        from .figspec import stable_id
        lines = ["<?xml version='1.0' encoding='UTF-8'?>",
                 "<graphml xmlns='http://graphml.graphdrawing.org/xmlns'>",
                 "<key id='label' for='node' attr.name='label' attr.type='string'/>",
                 "<key id='kind' for='edge' attr.name='kind' attr.type='string'/>",
                 "<graph edgedefault='directed'>"]
        for n in nodes:
            lines.append(f"<node id='{stable_id('n', n)}'>"
                         f"<data key='label'>{escape(n)}</data></node>")
        for e in edges:
            lines.append(f"<edge source='{stable_id('n', e['src'])}' "
                         f"target='{stable_id('n', e['dst'])}'>"
                         f"<data key='kind'>{escape(e.get('kind', ''))}</data>"
                         f"</edge>")
        lines += ["</graph>", "</graphml>"]
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_layout(path: Path, nodes: List[str]) -> None:
        """確定性環形佈局座標（固定 renderer 前的座標凍結基線）。"""
        import math
        from .figspec import stable_id
        n = max(1, len(nodes))
        coords = {stable_id("n", name): {
            "label": name,
            "x": round(math.cos(2 * math.pi * i / n), 4),
            "y": round(math.sin(2 * math.pi * i / n), 4)}
            for i, name in enumerate(nodes)}
        path.write_text(json.dumps({"layout": "deterministic-circular",
                                    "nodes": coords},
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")

    def _emit_figures(self, out: Path, paper_type: str, s: Dict) -> Dict:
        from .figspec import FIGURE_PLANS, FIGURE_SPECS, stable_id
        sd = out / "source_data"
        sd.mkdir(exist_ok=True)
        plan = FIGURE_PLANS[paper_type]
        emitted: List[Dict] = []
        legends: Dict[str, Dict] = {}
        fig_labels: Dict[str, str] = {}
        figs: List[str] = []
        manifest: List[Dict] = []
        n_fig = 0
        for key in plan:
            spec = FIGURE_SPECS[key]
            fig_no = f"Fig.{n_fig + 1}"
            result = getattr(self, f"_fig_{key}")(out, sd, s, fig_no, spec)
            if result.get("skipped"):
                manifest.append({"key": key, "skipped": True,
                                 "skip_reason": result["skip_reason"]})
                emitted.append({"fig_no": "", "key": key, "file": "",
                                "skipped": True,
                                "skip_reason": result["skip_reason"]})
                continue
            n_fig += 1
            fig_labels[key] = fig_no
            figs.append(f"{fig_no} {spec.legend.title} ({result['file']})")
            leg = spec.legend
            legends[key] = {"fig_no": fig_no, "title": leg.title,
                            "panels": leg.panels, "n": leg.n,
                            "data_source": leg.data_source,
                            "error_definition": leg.error_definition,
                            "evidence_level": leg.evidence_level,
                            "abbreviations": leg.abbreviations,
                            "source_data": result["source_data"]}
            import hashlib as _hl
            data_hash = _hl.sha256(b"".join(
                (sd / f).read_bytes() for f in result["source_data"])).hexdigest()[:16]
            manifest.append({
                "figure_id": fig_no, "key": key, "file": result["file"],
                "title": leg.title,
                "scientific_question": spec.scientific_question,
                "main_message": spec.main_message,
                "renderer": spec.renderer,
                "width_mm": spec.width_mm(),
                "panels": [p.panel_id for p in spec.panels],
                "uncertainty": [p.uncertainty for p in spec.panels],
                "data_sha256": data_hash,
                "spec_sha256": stable_id("spec", repr(spec))[5:],
                "source_data": result["source_data"],
                "extra_assets": result.get("extra", []),
            })
            emitted.append({"fig_no": fig_no, "key": key,
                            "file": result["file"], "skipped": False})
        (out / "figures_manifest.json").write_text(
            json.dumps({"paper_type": paper_type, "plan": plan,
                        "figures": manifest,
                        "note": "網絡圖投稿級渲染：用隨附 GraphML+layout.json"
                                " 經 Graphviz/Gephi/Cytoscape 導出 SVG/PDF"
                                "（stdlib 不做低質量佈局冒充投稿圖）"},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        return {"figs": figs, "emitted": emitted, "legends": legends,
                "fig_labels": fig_labels}

    # —— per-figure emitters ————————————————————————————————
    def _fig_channel_formula(self, out, sd, s, fig_no, spec) -> Dict:
        from .figspec import stable_id
        rows, mer = [], ["```mermaid", "graph LR"]
        for scr in self.six_channel_rules:
            ch = scr.six_channel
            mer.append(f'  {config.CHANNEL_PINYIN.get(ch, ch)}["{ch}"]')
            for f in scr.main_formulas[:4]:
                fid = f["formula"]
                mer.append(f'  {config.CHANNEL_PINYIN.get(ch, ch)} --> '
                           f'{stable_id("f", fid)}["{fid} ({f["clause_count"]})"]')
                rows.append([ch, fid, f["clause_count"]])
        mer.append("```")
        name = "fig_channel_formula.mmd.md"
        (out / name).write_text("\n".join(mer), encoding="utf-8")
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_channel_formula.csv",
                              ["six_channel", "formula", "clause_count"], rows)
        return {"file": name, "source_data": [src]}

    def _fig_mistreatment_paths(self, out, sd, s, fig_no, spec) -> Dict:
        dot = ["digraph mistreatment {", "  rankdir=LR;"]
        nodes, edges, rows = [], [], []
        for m in self.mistreatment_rules:
            a, b = m.mistreatment_type, m.resulting_pattern
            dot.append(f'  "{a}" -> "{b}";')
            edges.append({"src": a, "dst": b, "kind": "transforms"})
            rows.append([a, b, "", "transforms"])
            nodes += [a, b]
            for f in m.rescue_formulas[:2]:
                dot.append(f'  "{b}" -> "{f}" [style=dashed];')
                edges.append({"src": b, "dst": f, "kind": "rescued_by"})
                rows.append([b, f, "", "rescued_by"])
                nodes.append(f)
        dot.append("}")
        name = "fig_mistreatment_paths.dot"
        (out / name).write_text("\n".join(dot), encoding="utf-8")
        nodes = list(dict.fromkeys(nodes))
        self._write_graphml(out / "fig_mistreatment_paths.graphml", nodes, edges)
        self._write_layout(out / "fig_mistreatment_paths.layout.json", nodes)
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_mistreatment_edges.csv",
                              ["source", "target", "weight", "kind"], rows)
        return {"file": name, "source_data": [src],
                "extra": ["fig_mistreatment_paths.graphml",
                          "fig_mistreatment_paths.layout.json"]}

    def _fig_formula_family(self, out, sd, s, fig_no, spec) -> Dict:
        dot = ["digraph family {", "  rankdir=LR;"]
        nodes, edges, rows = [], [], []
        for r in self.formula_rules:
            for mrel in r.modification_relations:
                dot.append(f'  "{r.formula}" -> "{mrel["modified_formula"]}";')
                edges.append({"src": r.formula,
                              "dst": mrel["modified_formula"],
                              "kind": "modification"})
                rows.append([r.formula, mrel["modified_formula"]])
                nodes += [r.formula, mrel["modified_formula"]]
        dot.append("}")
        name = "fig_formula_family.dot"
        (out / name).write_text("\n".join(dot), encoding="utf-8")
        nodes = list(dict.fromkeys(nodes))
        self._write_graphml(out / "fig_formula_family.graphml", nodes, edges)
        self._write_layout(out / "fig_formula_family.layout.json", nodes)
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_family_edges.csv",
                              ["base_formula", "modified_formula"], rows)
        return {"file": name, "source_data": [src],
                "extra": ["fig_formula_family.graphml",
                          "fig_formula_family.layout.json"]}

    def _fig_clause_topics(self, out, sd, s, fig_no, spec) -> Dict:
        from .figspec import stable_id
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
            clusters.setdefault(c.six_channel, {}).setdefault(theme, []) \
                .append(c.clause_number)
        mer = ["```mermaid", "graph TD"]
        rows = []
        for ch, themes in clusters.items():
            chid = config.CHANNEL_PINYIN.get(ch, ch)
            mer.append(f'  {chid}["{ch}"]')
            for theme, nums in sorted(themes.items(), key=lambda kv: -len(kv[1])):
                tid = stable_id("t", f"{ch}|{theme}")
                sample = "、".join(str(n) for n in nums[:5])
                mer.append(f'  {chid} --> {tid}["{theme} ({len(nums)}條，'
                           f'如{sample}…)"]')
                rows.append([ch, theme, len(nums),
                             "、".join(str(n) for n in nums[:8])])
        mer.append("```")
        name = "fig_clause_topics.mmd.md"
        (out / name).write_text("\n".join(mer), encoding="utf-8")
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_clause_topics.csv",
                              ["six_channel", "theme", "n_clauses",
                               "example_clause_numbers"], rows)
        return {"file": name, "source_data": [src]}

    def _fig_formula_freq(self, out, sd, s, fig_no, spec) -> Dict:
        from .charts import hbar_chart
        pairs = s["formula_freq"].most_common(10)
        name = "fig_formula_frequency.svg"
        (out / name).write_text(
            hbar_chart(pairs, "宋本正文中各方劑的載方條文數（前 10）",
                       "n=398 條正文；一條條文含多方時分別計數",
                       width_mm=spec.width_mm(),
                       desc=spec.main_message,
                       x_label="載方條文數（A 層全量計數）"),
            encoding="utf-8")
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_formula_frequency.csv",
                              ["formula", "clause_count"],
                              [[f, n] for f, n in pairs])
        return {"file": name, "source_data": [src]}

    def _fig_commentator_agreement(self, out, sd, s, fig_no, spec) -> Dict:
        from .charts import heatmap
        atlas = self._load_research("commentary_divergence.json")
        if not atlas.get("agreement_matrix"):
            return {"skipped": True,
                    "skip_reason": "commentary_divergence.json 未生成"
                                   "（先運行 pipeline）"}
        comms = sorted({p[k] for p in atlas["agreement_matrix"]
                        for k in ("a", "b")})
        vals = {(p["a"], p["b"]): p["mean_term_agreement"]
                for p in atlas["agreement_matrix"]}
        ns = {(p["a"], p["b"]): p["n_shared_clauses"]
              for p in atlas["agreement_matrix"]}
        name = "fig_commentator_agreement.svg"
        (out / name).write_text(
            heatmap(comms, vals, "注家術語一致度矩陣",
                    "色階=平均術語一致度；格內 n=共注條文數（各對不同）；"
                    "詞彙級下界，不等於觀點一致",
                    width_mm=spec.width_mm(), desc=spec.main_message,
                    cell_n=ns),
            encoding="utf-8")
        srcs = [self._write_csv(
                    sd / f"{fig_no.replace('.', '')}a_agreement.csv",
                    ["commentator_a", "commentator_b", "mean_term_agreement"],
                    [[a, b, v] for (a, b), v in sorted(vals.items())]),
                self._write_csv(
                    sd / f"{fig_no.replace('.', '')}b_shared_clause_counts.csv",
                    ["commentator_a", "commentator_b", "n_shared_clauses"],
                    [[a, b, v] for (a, b), v in sorted(ns.items())])]
        return {"file": name, "source_data": srcs}

    def _fig_dose_totals(self, out, sd, s, fig_no, spec) -> Dict:
        from .charts import interval_chart
        ratios = self._load_research("dose_ratios.json")
        if not ratios.get("formulas"):
            return {"skipped": True,
                    "skip_reason": "dose_ratios.json 未生成（先運行 pipeline）"}
        top = sorted(ratios["formulas"],
                     key=lambda f: -f["total_weight_g"]["kaogu"])[:6]
        rows = [(f["formula"], [f["total_weight_g"]["kaogu"],
                                f["total_weight_g"]["duliangheng"],
                                f["total_weight_g"]["zhezhuan"]]) for f in top]
        name = "fig_dose_totals.svg"
        (out / name).write_text(
            interval_chart(rows, ["考古實測", "度量衡史", "明清折算"],
                           "全方總重量的折算假設區間（g，僅計重量類藥）",
                           "三家折算=學術假設情景，非測量值；"
                           "不構成臨床劑量建議",
                           width_mm=spec.width_mm(), desc=spec.main_message,
                           x_label="總重量（g；區間=折算學派差異）"),
            encoding="utf-8")
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_dose_scenarios.csv",
                              ["formula", "school", "total_weight_g"],
                              [[f["formula"], sch, f["total_weight_g"][k]]
                               for f in top
                               for sch, k in (("考古實測", "kaogu"),
                                              ("度量衡史", "duliangheng"),
                                              ("明清折算", "zhezhuan"))])
        return {"file": name, "source_data": [src]}

    def _fig_benchmark(self, out, sd, s, fig_no, spec) -> Dict:
        from .charts import hbar_chart
        ev = self._load_eval()
        cz = ev.get("cloze", {}).get("metrics", {}).get("attainable", {})
        if not cz.get("n"):
            return {"skipped": True,
                    "skip_reason": "評測未運行（先執行 evaluate）"}
        cs = ev.get("cases", {}).get("metrics", {})
        gr = ev.get("grounding", {}).get("metrics", {})
        pairs = [("遮方 Top-1", cz.get("top1", 0)),
                 ("遮方 Top-3", cz.get("top3", 0)),
                 ("遮方 Top-5", cz.get("top5", 0)),
                 ("遮方 MRR", cz.get("mrr", 0)),
                 ("醫案 Top-1", cs.get("top1", 0)),
                 ("醫案 MRR", cs.get("mrr", 0)),
                 ("接地率", gr.get("grounded_answer_rate", 0))]
        name = "fig_benchmark.svg"
        (out / name).write_text(
            hbar_chart(pairs, "客觀評測基準",
                       "各任務指標語義與樣本量不同，不可跨行比大小；"
                       "點估計（單次評測，無 CI）",
                       value_fmt="{:.2f}", width_mm=spec.width_mm(),
                       desc=spec.main_message,
                       x_label="指標值（0–1；遮方=LOCO 可達折，醫案=經方實驗錄）"),
            encoding="utf-8")
        src = self._write_csv(sd / f"{fig_no.replace('.', '')}a_benchmark_metrics.csv",
                              ["metric", "value", "task_n"],
                              [["遮方 Top-1", cz.get("top1", 0), cz.get("n", "")],
                               ["遮方 Top-3", cz.get("top3", 0), cz.get("n", "")],
                               ["遮方 Top-5", cz.get("top5", 0), cz.get("n", "")],
                               ["遮方 MRR", cz.get("mrr", 0), cz.get("n", "")],
                               ["醫案 Top-1", cs.get("top1", 0),
                                cs.get("n_scored", "")],
                               ["醫案 MRR", cs.get("mrr", 0),
                                cs.get("n_scored", "")],
                               ["接地率", gr.get("grounded_answer_rate", 0),
                                gr.get("n_questions", "")]])
        return {"file": name, "source_data": [src]}

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
