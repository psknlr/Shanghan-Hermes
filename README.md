# Hermes-Shanghanlun（傷寒-赫爾墨斯）

**《傷寒論》自主規則挖掘與 Skill 生成系統** —— 把《傷寒論》轉化為一個可回源、可推理、可比較、可教學、可寫作、可調用的規則系統。

> 研發：**醫哲未來人工智能研究院**（YiZhe Future AI Research Institute）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Shanghan-Hermes/blob/main/notebooks/Hermes_Shanghanlun_Colab.ipynb)
一鍵 Colab 全功能演示（[`notebooks/Hermes_Shanghanlun_Colab.ipynb`](notebooks/Hermes_Shanghanlun_Colab.ipynb)）：
流水線→檢索/匹配/鑒別/教學→注家圖譜/劑量層→三大評測→反思/編排/會話/研究循環
智能體→溯源論文+SVG圖表→Web控制台 iframe——全程離線可跑，可選接入
Anthropic/OpenAI/Azure/Poe/MiniMax。

```text
《傷寒論》原文自動解析 → 條文級規則挖掘 → 六經體系歸納 → 方證規則生成
→ 誤治傳變規則生成 → 禁忌法度規則生成 → 多版本/注本比較
→ Hermes Skill 編譯 → 醫師、科研、教學、患者教育多端調用
```

## 核心原則

> 無原文，不成規則。無條文編號，不成證據。無證據鏈，不成回答。
> 合併規則不能覆蓋初始條文規則。
> 方證歸納必須區分原文直述、後世歸納、模型解釋。
> 患者端禁止自動診斷、自動處方和劑量建議。

這些不是口號，而是流水線中的硬性閘門：每條規則的 `evidence_span` 必須逐字
存在於對應條文；證據回源失敗的規則直接進入 `rejected/`；對抗性測試
（`tests/test_review.py`）注入偽造證據並斷言其被拒絕。

## Web 控制台（一站集成全部功能 + 多智能體）

```bash
python3 -m hermes_shanghan pipeline     # 首次生成規則庫
python3 -m hermes_shanghan serve        # 打開 http://127.0.0.1:8765/
# 非本機部署：設 HERMES_SERVER_TOKEN=… 開啟 Bearer 鑒權（同時關閉開放 CORS）；
# 請求體上限 256KB，異常只回錯誤類型不回內部細節
```

純標準庫實現（`http.server` + 原生 JS 單頁應用，無構建、無 CDN、離線可用；
**移動端自適應**，頁腳標注研發來源）。15 個模塊：總覽 · **智能體（單/多智能體
合議）** · 原文檢索 · 方證匹配 · 方證鑒別 · 六經教學 · 誤治傳變 · 科研挖掘 ·
**溯源工作台**（誤引檢測/文本回源/術語譜系/方劑源流/注家爭議/學派比較）·
**方藥檔案**（藥證+本草旁證層/方解四層口徑）· **辨證閉環**（四診採集→裁決→
衝突審計）· **工具台**（28 工具通用調用/笈成全庫/深度研究/評測看板/
金標準標註閉環）· 論文生成 · Skill 庫 · 接入。證據優先：答案中的
`clause_id` 可點擊展開條文全息（A/B/C/D/E 五層色標）；多智能體合議把「規劃→取證→
方證/鑒別/六經/誤治專家→批評→綜合」可視化為時間線，每步附證據與引用核驗；
接入真實大模型時每位專家對自身工具證據附一句合議評述（引用同樣過核驗）。
詳見 [`docs/WEB_UI.md`](docs/WEB_UI.md)；能力成熟度與表述邊界見
[`docs/MATURITY.md`](docs/MATURITY.md)。

## 快速開始

純 Python 標準庫實現，無任何第三方依賴（Python ≥ 3.9）。

```bash
# 一鍵全量流水線（語料 → 條文 → 規則 → 審核 → 歸納 → Skill → 科研資產）
python3 -m hermes_shanghan pipeline

# 規則庫統計
python3 -m hermes_shanghan stats

# 醫師端：方證匹配（簡繁與異體字[脇/鞕/欬/濇]皆可輸入；核心證/兼證/
# 提綱證[如口苦→少陽]/近似證分級計分）
python3 -m hermes_shanghan match --symptoms "恶寒,发热,无汗,身疼痛" --pulse "浮紧"

# 患者端（自動角色推斷 + 意圖守衛）
python3 -m hermes_shanghan ask "医生说我是太阳表证，这是什么意思？"

# 教學端：六經學習（綱領/亞型/主方/誤治/禁忌/練習題）
python3 -m hermes_shanghan teach 太陽病

# 條文全息解釋（原文A/異文B/九注本C/規則/關係圖譜）
python3 -m hermes_shanghan explain-clause 12

# 注家分歧圖譜（9 注本對齊/爭點條文榜/一致度矩陣/注家指紋）
python3 -m hermes_shanghan divergence
python3 -m hermes_shanghan divergence --clause 0012

# 劑量計量層（銖當量藥量比[學派無關]/三家折算/家族劑量演化）
python3 -m hermes_shanghan dose 桂枝加芍藥湯

# 原文 RAG 檢索（BM25 + 結構化過濾 + 關係擴展）
python3 -m hermes_shanghan search "往來寒熱 胸脅苦滿" --expand
python3 -m hermes_shanghan search "第38條"

# 方證鑒別
python3 -m hermes_shanghan differential 桂枝湯 麻黃湯
python3 -m hermes_shanghan differential 半夏瀉心湯 生薑瀉心湯 甘草瀉心湯

# 科研端：共現網絡 / 頻次 / 家族樹 / 論文大綱
python3 -m hermes_shanghan research "桂枝湯類方證演化"

# 自動論文生成（8 種論文類型；模板管結構+數據表格+SVG統計圖表[CVD校驗調色板]，
# 增益層起草引言/計量結果解讀/討論/結論，全部引用過 CitationGuard 核驗）
python3 -m hermes_shanghan paper --type mistreatment --topic 誤治傳變路徑
python3 -m hermes_shanghan paper --type network_pharmacology --no-llm   # 純模板

# 列出已編譯 Skill
python3 -m hermes_shanghan skills

# Web 控制台（集成全部功能 + 多智能體）
python3 -m hermes_shanghan serve                 # http://127.0.0.1:8765/

# 客觀評測（遮方預測LOCO / 醫案回放 / 證據接地率；--ablations 消融）
python3 -m hermes_shanghan evaluate --suite all --ablations

# 深度研究循環（規劃→子代理→批評家迭代收斂）+ 一鍵學術溯源論文
python3 -m hermes_shanghan deep-research "桂枝湯類方的劑量演化與注家詮釋"
python3 -m hermes_shanghan paper --type provenance --topic 桂枝湯類方源流

# 深度溯源鏈（任意文本回源 / 條文 / 方劑 / 方證觀點 / 注家 / 學派）
python3 -m hermes_shanghan trace "观其脉证，知犯何逆，随证治之"
python3 -m hermes_shanghan trace 桂枝湯 -t formula
python3 -m hermes_shanghan trace 營衛不和 -t claim
python3 -m hermes_shanghan trace 成無己 -t commentator

# 學術計量網絡（歷代引文/共引/文獻耦合/朝代切片/突現/主路徑；
# --scope canonical|auxiliary|all 正文/輔助篇章分榜）
python3 -m hermes_shanghan trace-network --target 桂枝湯
python3 -m hermes_shanghan trace-network --scope auxiliary

# 全庫引文掃描（803 部醫籍→傷寒條文，漢→共和國；需先 library fetch）
python3 -m hermes_shanghan trace-scan-library --category 醫案
python3 -m hermes_shanghan trace-scan-library --out /tmp/library_edges.jsonl

# 誤引檢測 / 術語譜系 / 注家爭議結構化（不裁決）/ 學派比較
python3 -m hermes_shanghan trace "营卫不和，桂枝汤主之" -t quote
python3 -m hermes_shanghan trace 營衛不和 -t term
python3 -m hermes_shanghan trace 12 -t dispute
python3 -m hermes_shanghan trace "柯琴 vs 尤怡" -t compare

# 引文邊審計 / Scope 一致性審計 / 金標準標註閉環
python3 -m hermes_shanghan trace-audit-citation --book 傷寒來蘇集 --clause 12
python3 -m hermes_shanghan trace-audit-scope
python3 -m hermes_shanghan trace-gold-sample --n 50 --out gold.csv --stratify   # 分層抽樣；標註後：
python3 -m hermes_shanghan trace-gold-eval --file gold.csv           # P/R/F1

# 藥證檔案 · 方解一站式（四層症狀口徑 + 煎服法警示）
python3 -m hermes_shanghan herb 桂枝
python3 -m hermes_shanghan formula-explain 桂枝湯

# 方證辨證閉環（B 組；智能體路線圖見 docs/AGENT_ROADMAP.md）
python3 -m hermes_shanghan intake "发热，怕冷，出汗，头痛，服退烧药后腹泻"
python3 -m hermes_shanghan adjudicate --symptoms 發熱,惡寒,無汗,身疼痛 --pulse 浮緊
python3 -m hermes_shanghan conflict-check --formula 桂枝湯 --symptoms 無汗,發熱
python3 -m hermes_shanghan simulate-mistreatment --channel 太陽病 --type 誤下

# Harness 運行（顯式節點圖+checkpoint+span軌跡+發布閘門；docs/HARNESS.md）
python3 -m hermes_shanghan run "桂枝湯與麻黃湯如何鑒別？" --mode agent --role doctor
python3 -m hermes_shanghan run-resume <run_id> --approve --approver 張醫師   # 人工審核放行
python3 -m hermes_shanghan run-replay <run_id>   # local 後端重放指紋必一致
python3 -m hermes_shanghan run-export <run_id> --format md

# 智能體問答（工具取證 + 回源核驗 + 反思自糾；離線可用）
python3 -m hermes_shanghan agent "少陰病寒化與熱化怎麼區分？" --role student

# 複合問題編排（任務分解→作用域子代理→綜合再核驗）
python3 -m hermes_shanghan solve "桂枝湯與麻黃湯如何鑒別？各自劑量比是多少？注家有何分歧？"
python3 -m hermes_shanghan llm-status            # 查看 LLM 後端

# 測試（391 項：對抗性審核 + 智能體架構 + 28 工具 + 評測 + 七維研究循環 + 全庫接入 + 治理探針
#       + 可復現性/證據鏈硬化 + 溯源層（引文識別/計量網絡/五類溯源鏈）+ Colab守衛）
python3 -m unittest discover -s tests

# 中醫笈成全庫（800+ 部醫籍）：配置完成後一條命令自動下載（69MB，
# sha256 校驗 → 解壓 → 編目 → 字符索引），落於 data/library/（不入庫）
python3 -m hermes_shanghan library fetch
python3 -m hermes_shanghan library search 金匱          # 編目檢索（書名/作者/朝代/分類）
python3 -m hermes_shanghan library grep 奔豚 --category 醫案   # 全文檢索（書·章節定位）
python3 -m hermes_shanghan library read 傷寒來蘇集 --section 傷寒總論
```

## LLM 接入與智能體（神經符號增益層）

系統把確定性規則庫作為**可信底座**，LLM 作為**增益層**——但 LLM 產出的每一句話
都要先過「引用核驗」才能到達用戶，即使接入大模型，`無證據鏈，不成回答` 依然成立。

```bash
# 啟用真實大模型（可選；不裝則自動用 local 確定性後端，離線可跑）
pip install "litellm>=1.40"
export ANTHROPIC_API_KEY=sk-...                       # 或 OPENAI_API_KEY 等
export HERMES_LLM_MODEL=anthropic/claude-opus-4-8     # 經 LiteLLM，支持 100+ provider

# 也支持 Azure / Poe / MiniMax：
export AZURE_API_KEY=... AZURE_API_BASE=... AZURE_API_VERSION=...
export HERMES_LLM_MODEL=azure/<deployment>            # litellm 原生
export POE_API_KEY=...     HERMES_LLM_MODEL=poe/Claude-Sonnet-4.5      # OpenAI 兼容端點
export MINIMAX_API_KEY=... HERMES_LLM_MODEL=minimax/MiniMax-M2
export MINIMAX_API_BASE=https://api.minimaxi.com/v1   # 國內站可選覆蓋

# 智能體：自動取證、回源 clause_id、安全治理
python3 -m hermes_shanghan agent "病人往來寒熱、胸脅苦滿、口苦，考慮什麼方？" --role doctor

# LLM 增強的規則挖掘（候選規則仍過全部審核閘門；響應按內容磁盤緩存，重跑免費）
python3 -m hermes_shanghan pipeline --llm-extract --llm-critic
python3 -m hermes_shanghan llm-extract 12

# LLM 起草論文：讀入 research/ 計量資產（頻次/共現網絡/家族樹/誤治路徑），
# 撰寫引言/計量結果解讀/討論/結論；max_tokens 按任務分級（論文 ≥8192），
# 產出引用逐一過 CitationGuard，未核實編號在文末顯式標記「請勿採信」
python3 -m hermes_shanghan paper --type formula_pattern --topic 桂枝湯類方證

# 直接調用工具 / 導出工具規格
python3 -m hermes_shanghan tool-call shanghan_differential --args '{"formulas":["桂枝湯","麻黃湯"]}'
python3 -m hermes_shanghan export-tools --out tools.json
```

**接入智能體框架**（28 個只讀回源工具 + 1 個智能體工具，三種 harness 共用同一能力面；
模型經 function-calling **自主選擇調用**）。除檢索/匹配/鑒別/六經/誤治外，還包括：
分歧圖譜 · 劑量計量 · 全庫統計 · 評測指標 · **異文對勘**（B層）· **關係圖譜遍歷**
（多跳推理）· **治法法度** · **禁忌檢查**（複合推理：方+證候→衝突/法度禁例）·
**劑量換算計算器**（確定性，免模型心算）· **醫案檢索**（實驗錄旁證+經文錨點）·
**全庫文獻查閱**（中醫笈成 800+ 部：編目/全文/按章閱讀，文獻旁證層）·
**深度溯源鏈**（條文/方劑/方證觀點/注家/學派五類鏈）·
**學術計量網絡**（歷代引文/共引/耦合/切片/突現/主路徑，scope 貫穿全字段）·
**藥證檔案** · **方解一站式**（三層症狀口徑）：

| Harness | 接入方式 |
|---|---|
| Claude Code / Desktop | `claude mcp add shanghan -- python3 -m hermes_shanghan serve-mcp`（MCP stdio） |
| Codex CLI / OpenCode / openclaw | `export-tools` 導出 OpenAI/Anthropic 工具規格；`tool-call` 作分發目標 |
| 任意 LiteLLM 智能體 | `from hermes_shanghan.agent import ShanghanAgent` |

四項保證跨 harness 一致：**證據回源**（answer 引用 clause_id，guard 核驗）、
**層級標註**（A/B/C/D/E）、**患者安全**（診斷/處方/劑量上游攔截）、
**優雅降級**（無 litellm/key 自動用 local 後端）。詳見 [`docs/LLM_AGENT.md`](docs/LLM_AGENT.md)。

**智能體架構四層**（在線/離線同構，全部可測）：
- **反思自糾環**：引用核驗不通過（偽造編號/無引用/**引用未綁定本輪證據**）
  → 裁決回饋給模型、允許補充取證後重答（有界輪數）——核驗器從標注器升級為
  閉環控制器。嚴格 RAG 接地：所引 clause_id 必須出現在**本輪工具證據**中，
  「庫裡存在但本輪未檢索到」會被標記 `outside_evidence` 並觸發重答；
- **複合任務編排**（`solve` / `POST /api/complex`）：分解複合問題→按類型
  派遣**工具域受限**（ScopedRegistry 最小權限）的子代理→綜合後整體再核驗；
  research 型子任務自動派遣深度研究循環；
- **會話記憶**（`POST /api/chat`，按 session_id 隔離）：方名錨點 + 證據
  台賬跨輪累積，「它的劑量比呢？」自動指代消解；
- **深度研究循環**（`deep-research`）：見下文專節。

## 數據與版本分層

| 層 | 含義 | 底本 |
|---|---|---|
| A | 原文直述 | 傷寒論（宋本，趙開美本）：條文版 398 條編號 + 宋本輔助篇章（辨脈法/傷寒例/痙濕暍/可與不可諸篇） |
| B | 版本異文 | 傷寒雜病論（桂林古本）、傷寒論（千金翼方版）—— 條文級自動對齊 |
| C | 注家解釋 | 九部注本逐條對齊：成無己/方有執/柯琴（來蘇集+論注）/尤怡/錢潢/張卿子/黃元御/丹波元簡 |
| D | 後世類方歸納 | 《傷寒論類方》及跨條文歸納規則 |
| E | 模型推理 | 流水線生成的解釋（強制標註 `interpretation_level`） |

語料庫隨庫提交 **57 部**傷寒/金匱類古籍（`data/corpus_raw/`，含 sha256 manifest）。
原始歸檔清單共列 69 部，其中 12 部（金匱類 9 部、傷寒類 3 部：重訂通俗傷寒論、
類證活人書×2）未隨倉庫提交；差額在 `corpus_manifest.json` 的
`vendor_missing_books` 中逐一記錄並由測試核驗（缺失書目均不參與任何流水線層），
不做靜默計數。

**全庫擴展（文獻旁證層）**：`library fetch` 可自動下載中醫笈成全庫歸檔
（[jicheng.tw](https://jicheng.tw)，book-20180111.7z，803 部醫籍 / 843 個文本單元，
sha256 固定校驗）。解析器完整覆蓋其全部版式：`<book>` 元數據（含分類/參本/備考等
全字段）、單檔書、多卷書（`2-15`、`2-0.3` 卷-章-節混合編號）、嵌套子書
（如《醫宗金鑑》15 部子書）與 menu 導航頁排除。快速調用機制：毫秒級編目檢索 +
字符倒排索引剪枝的全文檢索（候選集可證完備，掃描達上限時顯式標記 `scan_capped`）+
章節目錄/按節閱讀；經 CLI（`library`）、智能體工具（`shanghan_library`）與 MCP
三種入口共用。全庫屬**文獻旁證層**：出處（書·作者·朝代·章節）僅供查閱，
不進入經文層證據閘門。設 `HERMES_LIBRARY_AUTOFETCH=1` 可由首次調用自動獲取。

## 規則層級（合併規則永不覆蓋初始規則）

```text
ShanghanClause (398 條正文 + 283 條輔助 + <F>方劑塊)
  └─ InitialRule         1,501 條（逐條抽取，禁止跨條歸納；一條多方分支各成規則；15+2 種規則類型）
       └─ FormulaPatternRule    113 個方證規則（核心證[主之條優先]/兼證/組成/煎服/加減/禁忌）
       └─ SixChannelRule          8 個六經規則（提綱/亞型/主方/欲解時）
       └─ TherapyRule            23 個治法規則（汗吐下和溫清補救逆 + 禁/誤）
       └─ MistreatmentRule       60 條誤治傳變路徑（誤治→變證→救治方）
       └─ DifferentialRule       64 組方證鑒別（多軸對比表，含自動發現）
            └─ MergedShanghanRule 121 條合併規則（僅引用下層 ID + 證據鏈）
另：ClauseRelation 4,286 條關係邊 ｜ VariantRule 616 條異文 ｜ CommentaryRule 2,958 條注文（9 注本）
```

## 自主審核流水線（每條規則 6 道閘門）

```text
SchemaValidator → EvidenceVerifier → SemanticReviewer → ShanghanCritic
→ AutoRepair（單輪修復後復檢）→ ConsensusJudge + ReleaseGate
                                   gold ≥0.90 / silver ≥0.78 / bronze ≥0.62 / rejected
```

ShanghanCritic 專門攔截協議列舉的錯誤類型：後世術語（營衛不和等）混入規則主體、
忽略同條禁忌、「可與」誇大為「主之」、「主之」擴域、太陽中風/傷寒混淆、
少陰寒化/熱化混淆、陽明經證/腑證混淆、否定陷阱（「不惡寒」誤標「惡寒」）。
全部 7,569 條審計記錄落盤於 `data/shanghan/audit/`。語義閘另含兩條硬約束：
禁忌類規則所在條文若無「不可/勿/禁/忌」禁例標記即硬性失敗（防 LLM 抽取虛構）；
長條文（>120 字）中證據跨度覆蓋全條 90% 以上者不得評 gold（過寬證據看似有據、
實不能證明「條件→結論」的具體綁定，一律降級）。

**可復現性防線**：`ingest`/`pipeline` 帶前置校驗——語料發現為空或缺少
傷寒論_條文版/宋本等關鍵書目時直接報錯拒絕覆蓋現有 manifest（不靜默清零）；
正文切分必須恰為 398 條，否則中止；manifest 原子寫入（tmp+replace）。
語料目錄名兼容 `#Uxxxx` 轉義（p7zip/unzip 在 C locale 下解壓會轉義中文名，
發現與讀取均按解碼名匹配）。pip 安裝等數據不隨包場景可用
`HERMES_SHANGHAN_DATA=/path/to/data` 直接指定數據根（或 `HERMES_SHANGHAN_ROOT`
指定倉庫根）。

**字節級可復現**：所有集合派生字段落盤前均按確定性次序排序（最長優先、同長按
字典序；對齊候選同分按段落序），任意 `PYTHONHASHSEED` 下重跑
`python3 -m hermes_shanghan pipeline`，`data/shanghan/` 與 `data/skills/` 產物
逐字節一致（`memory/` 含更新時間戳、`papers/` 含生成日期，二者除外）。

## 客觀評測（`evaluate`，四大基準 · 零人工標註 · 全確定性）

| 基準 | 設計 | 當前基線 |
|---|---|---|
| **遮方預測**（自監督） | Ithaca/Aeneas 式遮蔽的臨床決策版：遮住條文所載方，以該條分支段的證候作查詢，**留一條文**（該條全部規則從知識庫剔除後再預測，杜絕泄漏） | 可達折 n=140：Top-1 0.19 / Top-3 0.39 / Top-5 0.47 / MRR 0.30 / 藥物級F1 0.41；孤證方 31 折結構性不可達（單獨報告，不靜默剔除） |
| **醫案回放**（外部效度） | 《經方實驗錄》（1937，曹穎甫）百年實案：案題即金標準方，查詢僅取**首個處方標記之前**的敘述（藥量行/疏方語/方名三重切分防泄漏） | 75 方劑案（另 23 病名案）；界外方 16（多屬金匱）、證候不足 19，實評 40 案：Top-1 0.175 / MRR 0.21 —— 純條文規則對真實臨證的解釋上限被誠實量化 |
| **證據接地率**（幻覺標尺） | 由規則庫確定性生成 30 問，經智能體作答後統計 CitationGuard 核驗結果；2025 年文獻報告 LLM 偽造引用率 18–94%，本系統為任意後端提供可對比刻度 | local 後端：完全接地率 1.00、未核實引用率 0.00、篇均已核實引用 4.57 條（確定性下限；真實模型後端據此對標） |
| **智能體基準**（行為回歸） | 路由（問題→工具選擇）、回答級接地（越界引用/句級 claim 綁定率）、鑒別軸覆蓋（桂枝湯vs麻黃湯須含汗出/無汗軸）、患者安全（拒答/劑量泄漏/越權工具/過度拒答） | local 後端：路由準確率 1.00、越界引用率 0.00、句級接地 0.79、鑒別軸覆蓋 1.00、安全通過率 1.00 |

`--ablations` 對匹配器各評分組件做消融（近似證匹配 +0.7pp Top-1；提綱證加權
對遮方任務無增益、服務於主訴式查詢——兩類任務的分工由數據呈現）。全部結果
落盤 `data/shanghan/eval/`（無時間戳，納入字節級可復現保證）；
`paper --type benchmark` 自動把評測表格與增益層解讀寫成方法學論文。

## 深度研究循環（`deep-research`，loop engineering + 子代理）

一鍵自主學術溯源：**規劃器**逐輪選擇調用模塊（真模型經 JSON 規劃自主選調；
local 後端按覆蓋缺口確定性規劃，同一代碼路徑離線可跑）→ **子代理**逐任務
取證並產出引用核驗的研究發現 → **批評家**檢查七大溯源維度（原文源流/異文
注家/方證計量/劑量計量/客觀評測/醫案例證/引文傳播）缺口，未覆蓋維度進入
下一輪 → 收斂或達最大輪數。溯源檔案（dossier）驅動第 8 種論文類型 `provenance`：循環軌跡表
+ 逐條核驗的溯源發現表 + 增益層綜合。

所有論文自動附帶 **SVG 統計圖表**（純標準庫生成，CVD 校驗調色板、全直接
標注、隨附 CSV 表格視圖）：高頻方劑條形圖、注家一致度熱圖、劑量三家折算
圖、評測基準圖。

## 注家分歧圖譜（`divergence`，C 層的可計算化）

九部注本經「引文—注文」感知對齊（來蘇集同段引注切分、條辨序號剝離、成注
校勘小注剝離、引文樣段落守衛防止把下一條原文錯收為注文）：**2,958 條注文
對齊，395 條條文有 ≥2 位注家**（平均相似度 0.82–0.93；懸解因無標點重排僅
對齊 48 條——低覆蓋是結構性事實，如實報告）。在對齊之上計算：

- **爭點條文榜**：按注家術語剖面的 Jaccard 距離排出分歧最大條文（367/278/
  296 條等），學術史爭點由數據浮現；
- **一致度矩陣**：最高的一對是**張卿子×成無己（0.897）**——《張卿子傷寒論》
  本以成注為底本，算法在無先驗下重新發現了這層文獻承襲；最低段全是
  跨學派對（柯琴×黃元御 0.073、方有執×錢潢 0.0842），
  恰為錯簡重訂與維護舊論之分野；
- **注家指紋**：各注家超比例使用的分析詞彙（丹波元簡考據詞、尤怡治法詞、
  張卿子承成注營衛框架）。

## 劑量計量層（`dose`，經方本源劑量的可計算化）

536 條 `<F>` 塊劑量結構化：重量 378 / 計數 96 / 容量 42 / 長度 1，
另 14 條為純炮製語（本無劑量）、5 條不可解析（一錢匕/如雞子大等，逐一列出）。
複合與後綴寫法（一兩十六銖、一兩半、大者一枚、三百枚）均正確解析。三個設計原則：**藥量比以銖當量計、與折算學派無關**；絕對質量
按三家折算並存（考古實測 1兩=15.625g / 吳承洛 13.92g / 明清折算 3g，D/E 層
標註）；計數類（枚/個/把）不經未經考證的單枚質量假設換算。核心產出：

- **家族樹劑量演化**：加味 ≠ 增量——桂枝湯→桂枝加芍藥湯是**芍藥三兩→六兩
  （×2.0）的純劑量邊**（藥味集合完全相同），桂枝加桂湯同理（桂枝×1.67）；
  「量變致新方」首次成為圖譜上可查詢的一等關係；
- 方內藥量比、各藥常用量眾數、全方總量排行（分學派）——
  `research/dose_*.json` 四件套，`network_pharmacology` 論文自動引用。

## 深度溯源與學術計量層（`trace`，從「條文回源」到「文獻級知識譜系」）

在 A/B/C/D/E 五層之上新增**跨書引文層**：把隨庫 57 部傷寒/金匱著作（宋→
民國，歷時約 1800 年）對宋本條文的引用逐字檢測為 **40,660 條引文邊**
（聚合為 13,405 個著作×條文對），按「引述標記 × 覆蓋率」判定七種引用模式
（明引/節引/暗引/化用/改寫/**轉引注文**/存疑引用——有標記而庫內無匹配者
只計存疑、不猜出處）。在引文邊之上做確定性科學計量：被引榜、共引條文對、
著作文獻耦合、朝代時間切片、突現分析、主路徑（如第 12 條：東漢→宋補亡論
→金成注→明條辨→清來蘇集）。配套：

- **統一知識標識**（WorkID/EditionID/FormulaID/MethodID/SyndromeID/SchoolID/
  ClaimID/CitationEdgeID），與 clause_id 兼容並存；
- **學派註冊表**（10 派，posthoc_induction 標註）：範式/成員/適用邊界/對立
  學派，並回填分歧圖譜實測一致度（錯簡×以法類證 0.0842 的分野有數據面）；
- **結構化方證觀點**（7 個 ClaimID）：證據等級由逐字檢驗判定——「營衛不和」
  自動識別出第 53/54 條「榮氣和/衛氣不和」的原文直述成分，注家首倡時間線
  按朝代浮現（金成無己→清柯琴「調和營衛」）；
- **五類溯源鏈**（`trace` 命令 / `shanghan_trace` 工具 / `POST /api/trace`）：
  原文、方劑、方證觀點、注家（含被轉引樞紐度：成無己注文被 46 部後世著作
  轉引，張卿子本居首——文獻承襲的又一數據重現）、學派；
- **現代引用接口**：現代論文/教材引用經 `modern_citations.jsonl` 導入、
  逐字回源並做九類引用功能分類——**不隨庫分發、不憑空生成**；
- **全庫掃描**（`trace-scan-library`）：引用方可擴展到中醫笈成 803 部醫籍
  （實測 63,906 條邊、541 部書含傷寒引文，漢→唐→宋金元明清→民國→
  共和國），全庫層屬旁證、按需重建不入庫；`trace` 文本回源在傷寒論內
  無匹配時自動退到全庫檢索候選出處；
- **引文識別自檢基準**：合成明引/節引/暗引 檢出率 1.00、改寫 1.00（≤45 字
  條文）、負例誤報 0.00（確定性下限標尺）。被引榜正文/輔助篇章分榜
  （`--scope`），溯源報告逐節標註證據層（`section_evidence_levels`）。
  詳見 [`docs/TRACE.md`](docs/TRACE.md)；測試環境與耗時見
  [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md)；拒絕機制實證見
  [`docs/REJECTION_CASES.md`](docs/REJECTION_CASES.md)。

## Skill 目錄（139 個 Skill，每個含 SKILL.md + rules.jsonl + examples.jsonl）

```text
data/skills/shanghanlun/
├─ hermes.shanghan.catalog/                 目錄與版本總覽
├─ hermes.shanghan.six_channels/{taiyang,yangming,shaoyang,taiyin,shaoyin,jueyin,huoluan,laofu}/
├─ hermes.shanghan.formula_patterns/        113 個方證 Skill（guizhi_tang, mahuang_tang,
│                                           xiaochaihu_tang, dachengqi_tang, wumei_wan…）
├─ hermes.shanghan.mistreatment/            誤治傳變圖譜
├─ hermes.shanghan.contraindications/       禁忌法度（含宋本可/不可專篇）
├─ hermes.shanghan.therapy/{sweating,purgation,harmonization,…}/  治法規則（8 個子Skill）
├─ hermes.shanghan.transformation/          傳變規則
├─ hermes.shanghan.differential/            方證鑒別
├─ hermes.shanghan.clause_explainer/        條文解釋
├─ hermes.shanghan.variants/                版本異文
├─ hermes.shanghan.paper_writer/            論文寫作
└─ hermes.shanghan.patient_education/       患者教育（硬性安全邊界）
```

Skill RAG（`hermes_shanghan/rag/skill_rag.py`）按
`用戶問題 → 角色判斷 → Skill 檢索 → 規則調用 → 原文回源 → 安全審查`
路由；處方/劑量/診斷意圖在角色不明時一律按患者模式保守處理。

## Memory 模塊（7 個，`data/shanghan/memory/`）

`clause_memory`（條文處理狀態）、`formula_memory`（別名/組成/加減方）、
`six_channel_memory`、`mistreatment_memory`、`critic_memory`（高頻錯誤模式）、
`skill_memory`（構建歷史）、`paper_memory`（論文數據沉澱）。

## 安全治理

| 端 | 策略 |
|---|---|
| 醫師端 | 每個結果標註「僅為古籍方證輔助匹配，不能替代醫師臨床判斷」 |
| 患者端 | 意圖守衛拒絕診斷/處方/劑量請求；劑量文本自動脫敏；輸出剝離方劑推薦字段；提供術語通俗解釋、就診清單整理、風險信號提醒 |
| 科研端 | 強制標註 A/B/C/D/E 五個證據層級 |
| 教學端 | 標註教學輔助性質 |

## 項目結構

```text
hermes_shanghan/
├─ config.py / lexicon.py / textutil.py / schemas.py / safety.py
├─ corpus/      downloader（版本manifest）· library（笈成全庫：自動下載+編目+全文索引）
│               · catalog（篇章）· segmenter（條文切分）
├─ extract/     entities（否定感知實體抽取）· initial_rules（條文級規則）
├─ review/      validators · critic（對抗審核）· repair · pipeline（六道閘門）
├─ induce/      relations · formula_patterns · six_channels · therapy
│               · mistreatment · differential · merged
├─ rag/         bm25 · clause_rag（原文RAG）· skill_rag（技能路由）
├─ eval/        cloze（遮方LOCO）· cases（醫案回放）· grounding（接地率）
│               · agent_bench（智能體基準：路由/接地/鑒別覆蓋/安全）· runner
├─ apps/        doctor · research · teaching · patient · dosimetry（劑量層）
│               · commentary_atlas（注家分歧圖譜）
├─ trace/       quotation（七模式引文識別+自檢基準）· scientometrics（共引/
│               耦合/切片/突現/主路徑）· ids（統一標識）· schools（學派）
│               · claims（方證觀點）· chains（五類溯源鏈）· modern（現代接口）
├─ skills/      builder（Skill編譯）· pinyin
├─ paper/       writer（8 類論文 + LLM 增益層）· charts（純標準庫 SVG 統計圖）
├─ memory/      store（9 個記憶模塊，含 correction/project）
├─ llm/         config · cache · prompts · providers(litellm/local/scripted) · client
├─ agent/       tools(28 工具+ScopedRegistry+患者白名單+結果緩存) · citation_guard
│               · agent(ReAct+反思自糾+工具預算) · planner(任務圖規劃)
│               · evidence_binder(句級 claim→證據綁定) · hypothesis(多假設+鑒別追問)
│               · complex_agent(任務圖編排) · session(會話記憶+糾錯記憶)
│               · multi_agent(議會) · consensus(共識/分歧裁決)
│               · research_loop（深度研究循環：問題細化→子代理→缺口報告）
├─ integrations/ tool_specs(OpenAI/Anthropic) · mcp_server(Claude Code) · AGENTS.md
├─ server/      service(API面) · http_server(stdlib) · static(SPA: index/css/js)
├─ orchestrator.py（五大 Workflow 總調度，可選 --llm-extract/--llm-critic）· cli.py
tests/          391 項測試 ｜ notebooks/ Colab 全功能演示（守衛測試保證與代碼同步）
data/corpus_raw/   69 部古籍語料（含 manifest）
data/library/      中醫笈成全庫（803 部，`library fetch` 自動下載，不入庫）
data/shanghan/     全部生成資產（規則庫/審計/關係/科研/溯源/論文）
data/skills/       139 個編譯後 Skill
docs/PROTOCOL.md   完整協議文本
```

## MVP 路線達成情況

- ✅ MVP-1 宋本條文解析：398 條 + clause_id + 原文檢索 + 方/證/脈抽取
- ✅ MVP-2 太陽病 Skill：taiyang + guizhi_tang + mahuang_tang + gegen_tang + 誤治
- ✅ MVP-3 方證系統：桂枝/麻黃/柴胡/承氣/瀉心/四逆六大類方全覆蓋（113 方）
- ✅ MVP-4 六經全覆蓋：太陽/陽明/少陽/太陰/少陰/厥陰（+霍亂/勞復附篇）
- ✅ MVP-5 科研與 Paper Writer：方證知識圖譜/六經規則/誤治傳變三類論文自動生成

## 免責聲明

本系統為古籍知識工程研究工具。所有輸出基於《傷寒論》原文的結構化轉寫，
僅供學術研究、教學與醫師參考，不構成醫療建議；臨床決策請遵專業醫師判斷。
