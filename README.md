# Hermes-Shanghanlun（傷寒-赫爾墨斯）

**《傷寒論》自主規則挖掘與 Skill 生成系統** —— 把《傷寒論》轉化為一個可回源、可推理、可比較、可教學、可寫作、可調用的規則系統。

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
```

純標準庫實現（`http.server` + 原生 JS 單頁應用，無構建、無 CDN、離線可用）。
11 個模塊：總覽 · **智能體（單/多智能體合議）** · 原文檢索 · 方證匹配 · 方證鑒別 ·
六經教學 · 誤治傳變 · 科研挖掘 · 論文生成 · Skill 庫 · 接入。證據優先：答案中的
`clause_id` 可點擊展開條文全息（A/B/C/D/E 五層色標）；多智能體合議把「規劃→取證→
方證/鑒別/六經/誤治專家→批評→綜合」可視化為時間線，每步附證據與引用核驗；
接入真實大模型時每位專家對自身工具證據附一句合議評述（引用同樣過核驗）。
詳見 [`docs/WEB_UI.md`](docs/WEB_UI.md)。

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

# 條文全息解釋（原文A/異文B/成無己注C/規則/關係圖譜）
python3 -m hermes_shanghan explain-clause 12

# 原文 RAG 檢索（BM25 + 結構化過濾 + 關係擴展）
python3 -m hermes_shanghan search "往來寒熱 胸脅苦滿" --expand
python3 -m hermes_shanghan search "第38條"

# 方證鑒別
python3 -m hermes_shanghan differential 桂枝湯 麻黃湯
python3 -m hermes_shanghan differential 半夏瀉心湯 生薑瀉心湯 甘草瀉心湯

# 科研端：共現網絡 / 頻次 / 家族樹 / 論文大綱
python3 -m hermes_shanghan research "桂枝湯類方證演化"

# 自動論文生成（6 種論文類型；模板管結構與數據表格，增益層基於 research/
# 計量資產起草引言/計量結果解讀/討論/結論，全部引用過 CitationGuard 核驗）
python3 -m hermes_shanghan paper --type mistreatment --topic 誤治傳變路徑
python3 -m hermes_shanghan paper --type network_pharmacology --no-llm   # 純模板

# 列出已編譯 Skill
python3 -m hermes_shanghan skills

# Web 控制台（集成全部功能 + 多智能體）
python3 -m hermes_shanghan serve                 # http://127.0.0.1:8765/

# 智能體問答（工具取證 + 回源核驗 + 安全治理；離線可用）
python3 -m hermes_shanghan agent "少陰病寒化與熱化怎麼區分？" --role student
python3 -m hermes_shanghan llm-status            # 查看 LLM 後端

# 測試（97 項：對抗性審核 + LLM/智能體/多智能體 + Web/HTTP + 論文增益層 + 異體字/匹配調優）
python3 -m unittest discover -s tests
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

**接入智能體框架**（8 個只讀回源工具 + 1 個智能體工具，三種 harness 共用同一能力面）：

| Harness | 接入方式 |
|---|---|
| Claude Code / Desktop | `claude mcp add shanghan -- python3 -m hermes_shanghan serve-mcp`（MCP stdio） |
| Codex CLI / OpenCode / openclaw | `export-tools` 導出 OpenAI/Anthropic 工具規格；`tool-call` 作分發目標 |
| 任意 LiteLLM 智能體 | `from hermes_shanghan.agent import ShanghanAgent` |

四項保證跨 harness 一致：**證據回源**（answer 引用 clause_id，guard 核驗）、
**層級標註**（A/B/C/D/E）、**患者安全**（診斷/處方/劑量上游攔截）、
**優雅降級**（無 litellm/key 自動用 local 後端）。詳見 [`docs/LLM_AGENT.md`](docs/LLM_AGENT.md)。

## 數據與版本分層

| 層 | 含義 | 底本 |
|---|---|---|
| A | 原文直述 | 傷寒論（宋本，趙開美本）：條文版 398 條編號 + 宋本輔助篇章（辨脈法/傷寒例/痙濕暍/可與不可諸篇） |
| B | 版本異文 | 傷寒雜病論（桂林古本）、傷寒論（千金翼方版）—— 條文級自動對齊 |
| C | 注家解釋 | 成無己《註解傷寒論》（逐條對齊）、傷寒論條辨、傷寒來蘇集等 |
| D | 後世類方歸納 | 《傷寒論類方》及跨條文歸納規則 |
| E | 模型推理 | 流水線生成的解釋（強制標註 `interpretation_level`） |

語料庫隨庫提交 **57 部**傷寒/金匱類古籍（`data/corpus_raw/`，含 sha256 manifest）。
原始歸檔清單共列 69 部，其中 12 部（金匱類 9 部、傷寒類 3 部：重訂通俗傷寒論、
類證活人書×2）未隨倉庫提交；差額在 `corpus_manifest.json` 的
`vendor_missing_books` 中逐一記錄並由測試核驗（缺失書目均不參與任何流水線層），
不做靜默計數。

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
另：ClauseRelation 1,711 條關係邊 ｜ VariantRule 616 條異文 ｜ CommentaryRule 383 條成注
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
全部 7,569 條審計記錄落盤於 `data/shanghan/audit/`。

**字節級可復現**：所有集合派生字段落盤前均按確定性次序排序（最長優先、同長按
字典序；對齊候選同分按段落序），任意 `PYTHONHASHSEED` 下重跑
`python3 -m hermes_shanghan pipeline`，`data/shanghan/` 與 `data/skills/` 產物
逐字節一致（`memory/` 含更新時間戳、`papers/` 含生成日期，二者除外）。

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
├─ corpus/      downloader（版本manifest）· catalog（篇章）· segmenter（條文切分）
├─ extract/     entities（否定感知實體抽取）· initial_rules（條文級規則）
├─ review/      validators · critic（對抗審核）· repair · pipeline（六道閘門）
├─ induce/      relations · formula_patterns · six_channels · therapy
│               · mistreatment · differential · merged
├─ rag/         bm25 · clause_rag（原文RAG）· skill_rag（技能路由）
├─ apps/        doctor · research · teaching · patient
├─ skills/      builder（Skill編譯）· pinyin
├─ paper/       writer（6 類論文 + 圖表資產 + LLM 計量解讀增益層，引用過核驗）
├─ memory/      store（7 個記憶模塊）
├─ llm/         config · cache · prompts · providers(litellm/local/scripted) · client
├─ agent/       tools(8 個回源工具) · citation_guard · agent(ReAct) · multi_agent(議會)
├─ integrations/ tool_specs(OpenAI/Anthropic) · mcp_server(Claude Code) · AGENTS.md
├─ server/      service(API面) · http_server(stdlib) · static(SPA: index/css/js)
├─ orchestrator.py（五大 Workflow 總調度，可選 --llm-extract/--llm-critic）· cli.py
tests/          97 項測試（對抗性審核 + LLM/智能體/多智能體 + Web/HTTP + 論文增益層等）
data/corpus_raw/   69 部古籍語料（含 manifest）
data/shanghan/     全部生成資產（規則庫/審計/關係/科研/論文）
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
