# 智能體路線圖（評審建議的採納狀態與規劃）

外部評審提出 14 個智能體建議（A1–A5 證據溯源類、B6–B9 方證辨證類、
C10–C12 方藥知識類、D13–D14 注家學派類）。本文檔記錄逐項處置：
**A 組與 C10/C11 已落地**；B/D 組屬臨床交互與裁決類產品能力，體量與
安全審查要求高，列入規劃並標明現有部分能力，不倉促上線。

## 已落地（本輪）

| # | 建議 | 落點 |
|---|---|---|
| A1 | Scope Consistency Auditor | `trace-audit-scope`：三 scope 輸出全文遞歸掃描違例，CI 可跑（`scientometrics.audit_scope_consistency`） |
| A2 | Citation Evidence Auditor | `trace-audit-citation --book X --clause N`：逐邊給出模式/最長片段/覆蓋率/歸屬歧義/套語邊界/斷章風險提示/轉引標記 + 確定性可靠性分級 |
| A3 | Quotation Gold-standard Builder | `trace-gold-sample --n 50 --out gold.csv`（確定性等距抽樣+算法預測列）→ 人工標註 → `trace-gold-eval --file gold.csv`（P/R/F1 + 模式一致率 + 分歧樣本） |
| A4 | Misquotation Detection | `trace "營衛不和，桂枝湯主之" -t quote`：逐片段判定原文逐字/後世歸納語，關聯方證觀點庫與 A 層相關表述，整句給直引可否結論 |
| A5 | Claim Lineage | claims.json 增補 `first_proponent`（最早可見注家，以在庫注本為限）與 `term_first_use`（各術語首現注家/朝代） |
| C10 | 藥解 | `herb 桂枝`：出現方劑/條文/劑量寫法/配伍共現網絡；**不編造藥性解釋**（本草層未隨庫，如實聲明） |
| C11 | 方解 | `formula-explain 桂枝湯`：首見/方證/組成劑量比/煎服/禁忌/類方鑒別/方名傳播/觀點分級一站式 |

## 第四輪落地補充

| # | 建議 | 落點 |
|---|---|---|
| 術語譜系智能體 | 「營衛不和最早何時出現？」 | `trace 營衛不和 -t term`：A 層逐字檢驗 → 在庫首現注家（柯琴·清《來蘇集》）→ 學派分佈 → 關聯方證觀點與原文相關表達 → 現代回聲；「胃家實」正確判原文逐字（180 條） |
| herb/formula-explain 工具化 | 產品落差：CLI 有而智能體/MCP/Web 無 | **採納**：`shanghan_herb_profile` + `shanghan_formula_explain` 入 ToolRegistry（22→24，規格自動導出 MCP/OpenAI/Anthropic），`POST /api/herb`、`/api/formula-explain`；患者模式不暴露（含劑量）。Web UI 卡片列入下輪（SPA 模塊改造獨立做） |
| 方解三層症狀口徑 | core_symptoms 過寬風險 | **採納**：`symptom_layers` 三層分列——首見方證（第 12 條直載）/全書聚合（含頻次）/特殊上下文（誤治/禁忌/傳變單列，如 15 條誤下後氣上衝），附「不得徑作標準核心證」提示 |
| 金標準分層抽樣 | 等距抽樣可能被大部頭主導 | **採納**：`--stratify`（朝代×預測模式分層，含負例層；比例配額、每層≥1、層內等距、零隨機可復現）；按預測模式分層是評測慣例，文檔注明最終評測建議雙人標註 |
| 轉引語義細分 | self vs relay | **採納**：`trace-audit-citation` 區分「本書注文（self_commentary）」與「轉引注文（relay_commentary）」——來蘇集×12 命中柯琴注判本書注文，張卿子本×12 命中成無己注判後世轉引 |
| 耦合逐 scope | bibliographic_coupling 全域 | **採納**：著作條文集先按域過濾再算 Jaccard，入 `scoped`；合成邊單元測試驗證語義（共享 12 正文+12 輔助 → canonical/auxiliary 各 12、all 24） |

## 第五輪落地：B 組方證辨證閉環（`apps/bianzheng.py`，四工具入註冊表）

| # | 建議 | 落點 |
|---|---|---|
| B6 | 四診信息採集 | `intake` / `shanghan_intake`（**患者白名單唯一辨證類工具**）：自然敘述→結構化四診表（主訴/病程/寒熱/汗/渴飲/二便/胸脅腹/痛/眠/舌/脈/誤治史/藥後反應）+ 缺失關鍵信息 + 追問建議；現代口語→古籍術語映射表透明可審（怕冷→惡寒）；只整理信息不做匹配 |
| B7 | 多假設裁決 | `adjudicate` / `shanghan_adjudicate`：基於 HypothesisManager，每候選附支持/反證/缺失/禁忌衝突，三態裁決（傾向A/傾向B/不能裁決）+「為什麼還不能定方」+ 3 個關鍵追問 |
| B8 | 方證衝突審計 | `conflict-check` / `shanghan_conflict_audit`：互斥證對×方證規則→核心證/兼證衝突分級 + 衝突條文回源 + 觸發禁例 + 改判候選（無汗→麻黃湯類）+ 應補問；固有禁例不虛升嚴重度 |
| B9 | 誤治傳變模擬 | `simulate-mistreatment` / `shanghan_mistreatment_simulate`：經×誤治→變證分支→救逆方→條文依據（60 條規則逐條錨定）；多步鏈為組合視圖並如實標註「非原文連續敘述」 |

另落地：文檔同步守衛 `tests/test_docs_sync.py`（README/TEST_REPORT/
LLM_AGENT 的測試數與工具數必須等於運行時實測，漂移即紅——回應
「TEST_REPORT 同步智能體」建議，以 CI 守衛而非另一個生成器實現）。

## 第六輪落地（P2 研究平台 + P3 方藥 + 前端）

| 建議 | 落點 |
|---|---|
| 注家爭議結構化（不裁決） | `trace -t dispute`：貼近原文程度/後世術語密度/分歧類型提示/論文寫法建議 |
| 學派比較 | `trace -t compare "柯琴 vs 尤怡"`：範式/指紋/實測一致度/高分歧條文 |
| 誤引檢測 Web 工作台 | Web「溯源工作台」模塊：粘貼即逐片段標注（分色）+ 證據層級摺疊面板 |
| 本草證據層接入 | herb_profile.bencao_layer：全庫本草類原文摘錄，嚴格旁證分層 |
| 方劑異名歸並 | trace/aliases.py 編輯性對照表 + 組成比對；異名與正名分列計量 |
| 移動端 UI | ≤820px 側欄轉頂部橫向導航、柵格塌縮、抽屜全寬 |
| 研發來源標識 | README/Web 頁腳/Colab 頁首：醫哲未來人工智能研究院 |

## 規劃中（D 組與深化項）

| # | 建議 | 現有部分能力 | 評估與規劃 |
|---|---|---|---|
| B6 | 四診信息採集 | 患者端意圖守衛與就診信息整理（`apps/patient.py`）；實體抽取器可識別症狀/脈象/病程 | **合理，優先級最高**。需新增：結構化四診 schema（主訴/病程時間線/寒熱汗渴二便腹證舌脈/誤治史）、缺失信息追問生成。患者端嚴格限「就診信息整理」，方證匹配僅醫師端——沿用既有硬隔離 |
| B7 | 方證多假設裁決 | `shanghan_hypotheses` 已有並列假設+支持/反證/缺失關鍵證+鑒別追問 | **合理**。差距在「裁決層」：傾向 A/B/不能裁決三態輸出 + 3 個關鍵補問。可基於 `agent/hypothesis.py` 擴展，複用 `consensus.py` 裁決機制 |
| B8 | 方證衝突審計 | `shanghan_contraindication_check`（方+證候→衝突/禁例）已覆蓋大半；匹配器對 negated_findings 有懲罰 | **合理**。差距：衝突強度分級與「改判建議」。宜作為 contraindication_check 的增強版而非新智能體，避免能力面重複 |
| B9 | 誤治傳變路徑模擬 | `shanghan_mistreatment` 有 60 條誤治→變證→救治路徑（含條文依據） | **合理**。差距：多步動態模擬（誤治鏈式傳變）與圖形化。規則已成圖（ClauseRelation mistreatment_transformation 邊），需路徑搜索 + Web UI 呈現 |
| C12 | 煎服法智能體 | FormulaBlock 已結構化 preparation/administration/post_notes；方證規則含 administration_notes | **合理**。差距：服後觀察/中病即止/調護的規則化抽取 + 患者端脫敏解釋（「現代不可直接執行」提示涉醫療安全，需審慎措辭） |
| D13 | 注家爭議裁決 | 分歧圖譜有爭點條文榜/一致度矩陣/指紋；`commentator_chain` 有學派歸屬與被轉引樞紐度 | **部分合理**。「裁決」措辭與「多觀點並存不裁決」原則衝突——改為「爭議結構化呈現」：分歧類型標註（訓詁/方證/病機/治法）可由注文術語剖面確定性分類；「貼近原文程度」可用注文與條文的逐字重合率計算。列入下輪 |
| D14 | 學派比較 | `school_chain` + 一致度矩陣 + 指紋 + 引文網絡差異均已可查 | **大半已具備**。差距：兩注家/兩學派的對照報告模板（把現有六類資產拼成一份對比文檔），適合作 `paper --type school_compare` 論文類型 |

## 新增規劃項（第四輪評審提出）

| 建議 | 現狀與評估 |
|---|---|
| 本草證據層接入（神農本草經/別錄/綱目…） | **合理，全庫已含這些書**（`library fetch` 後可檢索）。差距：本草條目與藥名的結構化對齊 + 嚴格分層（傷寒 A 層事實 / 本草 C/D 層藥性 / 後世方論 / 模型解釋）。作為 herb_profile 的擴展層規劃 |
| 方劑異名歸並（桂枝湯/陽旦湯…） | **合理且是方名計量的已聲明邊界**。需異名對照表（編輯性元數據，逐條注明依據）+ 組成比對驗證「是否同方」；不可僅憑名稱合併 |
| 誤引檢測 Web 工作台 | 後端已就緒（`-t quote` 逐片段判定）；差距僅前端粘貼-標紅工作台，隨 Web UI 卡片一起做 |
| Rule Diff Agent（流水線重跑差異報告） | **合理**：字節級可復現使 diff 天然可行（git diff data/shanghan/ 已是下限）；規則級語義 diff（新增/刪除/分級變化/核心證變化）列入工程迭代 |
| Data Lineage Agent | 部分已具備（句→clause_id→sha256→manifest 鏈條各環節都在）；差距是一鍵串起來的報告器 |
| Evaluation Dashboard | `evaluate` + `paper --type benchmark` 已覆蓋大半；差距是聚合看板（含引文識別 P/R/F1 與金標準結果） |
| Human-in-the-loop 標註閉環 | **Web UI 已落地（七輪）**：工具台「金標準標註閉環」面板——分層抽樣→瀏覽器內標註→即時 P/R/F1（rows 進 rows 出不落盤）；剩餘差距：多標註者一致率（Cohen's κ）與分歧仲裁流程 |
| 多智能體專家獨立性 | 現狀：合議專家各自綁定自身工具證據（evidence packet 雛形），但判斷邏輯是規則工具視角。規劃：每專家獨立 LLM 上下文 + 獨立 evidence packet + 交叉 critique 輪 + ConsensusJudge 仲裁；離線模式保留規則專家作 fallback（同構可測原則） |

## 設計約束（所有後續智能體一體遵守）

1. 無證據鏈不成回答：新智能體輸出必須攜帶可核驗 clause_id；
2. 原文直述/後世歸納分層：D13/D14 的任何「裁決」只呈現證據結構，不判對錯；
3. 患者端硬隔離：B6 四診採集在患者端僅作信息整理，禁入方證匹配；
4. 確定性優先：能用規則與計量實現的不依賴 LLM；LLM 僅作增益層且過引用核驗。
