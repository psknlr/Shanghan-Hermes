# 平台化藍圖（十二輪評審採納：從傷寒論系統到通用古籍智能體平台）

評審判斷成立：當前系統是**高質量功能演示 + 部分平台能力**，尚非平台級。
本文檔把「平台化重構」落為可執行的分層映射與遷移計劃——並如實標注
哪些已是平台件、哪些仍強耦合傷寒論。

## 一、現狀分層審計（誠實清單）

| 目標層 | 已是平台件（領域無關，可直接複用） | 仍強耦合傷寒論 |
|---|---|---|
| Core | `trace/evidence.py`（EvidenceRecord 結構）、`corpus/worktype.py`（work_type/證據層裁定）、`corpus/library.py`（供應鏈+全庫檢索）、`health.py`（readyz 骨架） | 條文切分器（398 條規則）、實體詞表（lexicon）、A–E 層書目註冊（config） |
| Agent | ReAct/Council/Complex/DeepResearch 控制流、CitationGuard/EvidenceBinder 機制 | 工具面 28 個 `shanghan_*`、提示詞、路由詞表 |
| Harness | **全部**：RunSpec/狀態圖/TriageDecision 分支/RunBudget/Broker 台賬/發布閘門/ApprovalRequest/span 軌跡/replay 指紋/run.lock 心跳 | 觸發詞（候選方=方劑工具集合）屬領域配置 |
| 治理 | **全部**：policy（Principal/RequestContext/投影）、API keys、審計、readyz、MCP 協議層 | 患者投影字段清單屬領域配置 |
| UI | `/console.html` 運行中心（運行/會話/評測/Artifact/治理——全部走平台 API，不含傷寒論業務邏輯） | `/` 經典 UI 的 15 業務模塊 |

結論：**Harness 與治理層已經是平台**；耦合集中在 Core 的語料處理與
Agent 的工具面/詞表——這正是 DomainSpec 插件要吸收的部分。

## 二、領域插件 seam（已落地第一步）

`hermes_shanghan/domains.py`：DomainSpec（domain_id/書目/工具前綴/語料
類目/狀態）。當前註冊 shanghan（active）、jingui/neijing（planned，
如實標注「待建」而非佔位假實現）。

## 三、遷移計劃（不在現結構上繼續疊功能）

| 階段 | 動作 | 驗收 |
|---|---|---|
| M1 | 把 config 中書目/層註冊、lexicon 詞表、工具註冊搬進 DomainSpec 的掛載點（平台代碼不 import 領域常量） | 平台包 0 處 import 傷寒論詞表 |
| M2 | 切分器接口化：`DomainSpec.parser`（輸入原書文本 → PassageRecord 流）；398 條切分器成為 shanghan 插件的 parser | 金匱 parser 以同接口接入並產 manifest |
| M3 | 工具面按前綴命名空間掛載（`<domain>_search` 等），Registry 聚合多領域 | /api/tools 按領域分組 |
| M4 | UI 業務模塊按領域配置生成（導航/文案來自 DomainSpec） | 新領域零 UI 代碼上線基礎四件套（檢索/條文/溯源/運行） |

## 四、雙 UI 策略（本輪落地）

- `/`（經典 UI）：15 業務模塊**原樣保留**，僅補 token 認證頭與運行中心
  入口——啟用 HERMES_API_KEYS/SERVER_TOKEN 後不再全 401；
- `/console.html`（運行中心，新）：統一控制面——總覽（版本/readyz）、
  運行中心（任務列表/節點軌跡/工具調用/證據台賬/審批/重放/導出）、
  會話（session_id 續接 + 結構化指代解析）、評測（軌跡/故障注入，
  口徑標注）、標註（金標準閉環）、Artifact（防穿越下載）、治理
  （角色策略/審計尾部）。token 存 localStorage，兩 UI 共享。

## 五、如實差距（不宣傳為已有）

- 異步任務系統：run 啟動已異步（線程+輪詢）；隊列/取消/多 worker 未做；
- 多用戶生產：ThreadingHTTPServer 定位開發服務，生產走 ASGI + OIDC；
- 標註仲裁：單標註閉環在，雙人+Cohen's κ 在路線圖；
- Artifact 版本管理：目前是文件列表+讀取，無版本化存儲；
- 領域插件：seam 已立，M1–M4 未動工——**現在仍是傷寒論系統**。
