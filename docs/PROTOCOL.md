# Hermes-Shanghanlun Protocol（《傷寒論》自主規則挖掘與 Skill 生成系統協議）

本文檔是系統的設計協議（規範層），代碼實現見 `hermes_shanghan/`，與協議條目的對應關係在文末。

## 一、系統定位

Hermes-Shanghanlun 是專門面向《傷寒論》的古籍智能體系統。核心任務不是總結條文，而是把《傷寒論》轉化為**可回源、可推理、可比較、可教學、可寫作、可調用**的規則系統：

```text
《傷寒論》原文自動解析 → 條文級規則挖掘 → 六經體系歸納 → 方證規則生成
→ 誤治傳變規則生成 → 禁忌法度規則生成 → 多版本/注本比較
→ Hermes Skill 編譯 → 醫師、科研、教學、患者教育多端調用
```

核心原則（硬約束）：

```text
無原文，不成規則。
無條文編號，不成證據。
無證據鏈，不成回答。
合併規則不能覆蓋初始條文規則。
方證歸納必須區分原文直述、後世歸納、模型解釋。
患者端禁止自動診斷、自動處方和劑量建議。
```

## 二、知識結構

1. **六經主軸**：太陽/陽明/少陽/太陰/少陰/厥陰 各成獨立 Skill（`hermes.shanghan.taiyang` …），支持橫向比較；霍亂、陰陽易差後勞復作附篇處理。
2. **治法主軸**：汗/吐/下/和/溫/清/補/救逆/利水 + 禁汗/禁下/禁吐 + 誤汗/誤下/誤吐/火逆。
3. **方證主軸**：方證對應而非方劑列表；首批覆蓋桂枝湯、麻黃湯、葛根湯、大小青龍、大小柴胡、白虎（加人參）、三承氣、四逆、真武、理中、五苓、豬苓、三瀉心、黃連阿膠、烏梅丸、當歸四逆等（實際已覆蓋全書 109 方）。
4. **條文邏輯主軸**：並列/遞進/轉變/誤治後變/鑒別/禁忌/方後注/煎服法/預後 —— 以 ClauseRelation 圖譜建模。

## 三、版本策略

主底本：**傷寒論（宋本）**（趙開美本，現代通行 398 條編號，由「條文版」承載）。

證據分層（任何輸出必須標註層級）：

```text
A 層：宋本原文（條文版 398 條 + 宋本輔助篇章）
B 層：版本異文（桂林古本、千金翼方版）
C 層：歷代注釋（成無己《註解傷寒論》逐條對齊等）
D 層：後世類方歸納（《傷寒論類方》、跨條歸納規則）
E 層：模型現代解釋（interpretation_level 強制標註）
```

## 四、核心數據對象

`ShanghanClause`（條文單元，sha256 指紋）→ `InitialRule`（單條抽取，含 autonomous_review 審核塊）→ `ClauseRelation`（七類關係邊）→ `FormulaPatternRule` / `SixChannelRule` / `TherapyRule` / `MistreatmentTransformationRule` / `DifferentialRule` / `VariantRule` / `CommentaryRule` → `MergedShanghanRule`（僅引用、附證據鏈、衝突顯式記錄）。字段定義見 `hermes_shanghan/schemas.py`。

## 五、規則類型體系（17 種）

```text
01 six_channel_definition_rule  六經綱領    02 disease_pattern_rule     病證定義
03 formula_pattern_rule         方證對應    04 pulse_symptom_rule       脈證關係
05 therapy_selection_rule       治法選擇    06 contraindication_rule    禁忌
07 mistreatment_rule            誤治        08 transformation_rule      傳變
09 prognosis_rule               預後        10 administration_rule      煎服法
11 formula_composition_rule     方藥組成    12 dosage_processing_rule   劑量炮製
13 differential_rule            鑒別        14 rescue_reverse_rule      救逆
15 recurrence_rule              復發/勞復   16 variant_rule             版本異文
17 commentary_rule              注家解釋
```

## 六、Agent 架構（16 個）與代碼對應

| # | Agent | 實現 |
|---|---|---|
| 1 | ShanghanDownloaderAgent | `corpus/downloader.py`（版本 manifest + sha256） |
| 2 | ShanghanCatalogAgent | `corpus/catalog.py`（書名/版本/篇章/六經映射） |
| 3 | ClauseSegmenterAgent | `corpus/segmenter.py`（條文/方劑塊/方後注/煎服法/邏輯詞） |
| 4 | ClassicalTextRAGAgent | `rag/clause_rag.py`（條文號/方名/症狀/脈象/治法/禁忌檢索） |
| 5 | EntityExtractorAgent | `extract/entities.py`（12 類實體，否定感知） |
| 6 | InitialRuleExtractorAgent | `extract/initial_rules.py`（逐條抽取，禁止跨條歸納） |
| 7 | EvidenceVerifierAgent | `review/validators.py::verify_evidence`（逐字回源） |
| 8 | ShanghanCriticAgent | `review/critic.py`（對抗式錯誤清單） |
| 9 | ConsensusJudgeAgent | `review/pipeline.py`（gold/silver/bronze/rejected） |
| 10 | FormulaPatternAgent | `induce/formula_patterns.py`（方證譜系/加減方） |
| 11 | SixChannelInducerAgent | `induce/six_channels.py` |
| 12 | MistreatmentTransformationAgent | `induce/mistreatment.py`（傳變圖譜） |
| 13 | DifferentialDiagnosisAgent | `induce/differential.py` |
| 14 | SkillBuilderAgent | `skills/builder.py`（SKILL.md/rules.jsonl/examples.jsonl） |
| 15 | PaperWriterAgent | `paper/writer.py`（6 類論文 + 圖表 + Cover Letter） |
| 16 | SafetyGovernanceAgent | `safety.py`（角色化治理 + 意圖守衛 + 劑量脫敏） |

（另：AutoRepairAgent `review/repair.py`、RelationBuilder `induce/relations.py`、MergedRuleBuilder `induce/merged.py`。）

## 七、五大 Workflow

1. **條文級規則挖掘**：文本 → 切分 → 實體 → InitialRule → 回源驗證 → 對抗審核 → 修復 → 分級發布（`rules_initial/` `audit/` `rejected/`）。
2. **六經體系構建**：章節映射 → 提綱/主證/變證/方劑/禁忌/誤治 → SixChannelRule → 六經 Skill。
3. **方證體系構建**：方名歸一 → 組成抽取 → 方證聚類 → 主證/兼證/禁忌 → 加減方識別 → 方證 Skill。
4. **誤治傳變圖譜**：誤治關鍵詞 → 變證 → 救治方 → 原文證據（模板 + 自動發現雙通道）。
5. **方證鑒別**：典型鑒別對 + 自動發現（共享核心證 ≥3），輸出六經/症狀/脈象/寒熱虛實/裏實/汗/禁忌/組成差異多軸對比。

入口：`orchestrator.py::run_pipeline`，CLI `python3 -m hermes_shanghan pipeline`。

## 八、審核標準

每條規則必過：SchemaValidator → EvidenceVerifier → SemanticReviewer → ShanghanCritic → ConsensusJudge → ReleaseGate。

**拒絕條件**：evidence_span 不在原文 / 方劑與條文不對應 / 症狀由模型自行補充且不可修復 / 把注釋當原文 / Schema 違規 / 把患者端問題轉成處方建議。
**降級條件**：治法為後世歸納、病機解釋非原文直述、支持條文不足、版本異文存在、方證邊界不清、經過自動修復（分數封頂 0.92）。

## 九、RAG 與 Memory

原文 RAG：BM25（字 n-gram）+ 結構化過濾（六經/方劑/字段）+ 條文關係圖譜擴展；Skill RAG：角色判斷 → 意圖路由 → Skill → 規則 → 回源 → 安全審查。Memory 七模塊見 `memory/store.py`。

## 十、安全與多端

醫師端標註輔助性質；患者端僅做術語通俗解釋、症狀整理、風險提醒，意圖守衛拒絕診斷/處方/劑量並脫敏劑量文本；科研端強制證據分層；教學端輸出綱領/亞型/主方/誤治/禁忌/條文/練習題。

## 最終定位

> 一個以《傷寒論》條文為最小證據單位，以六經辨證和方證對應為核心規則體系，以模型自主審核和原文回源為質量保障，以 Skill RAG 為調用方式，服務醫師、科研人員、學生和患者教育的《傷寒論》智能體操作系統。
> 讓《傷寒論》的每一條原文都能轉化為可追蹤規則，讓每一個方證判斷都能回到條文，讓每一個科研問題都能自動生成證據鏈。
