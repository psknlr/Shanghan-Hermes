# Web 控制台

把全部功能集成到一個瀏覽器控制台，並疊加 LLM 與多智能體系統。與整個項目一致：
**零第三方依賴**（純標準庫 `http.server` + 原生 JS 單頁應用，無構建、無 CDN），
離線可用。

## 啟動

```bash
python3 -m hermes_shanghan pipeline      # 首次：生成規則庫
python3 -m hermes_shanghan serve         # 啟動，瀏覽器打開 http://127.0.0.1:8765/
# 可選：--host 0.0.0.0 --port 8800
```

啟用真實大模型（可選）後，控制台頂部後端徽章會由 `local` 變為 `litellm`：

```bash
pip install litellm && export ANTHROPIC_API_KEY=sk-...
export HERMES_LLM_MODEL=anthropic/claude-opus-4-8
python3 -m hermes_shanghan serve
```

## 11 個功能模塊（左側導航）

| 模塊 | 能力 |
|---|---|
| **總覽** | 條文/規則/Skill 計量、規則分級條形圖、LLM 後端狀態 |
| **智能體** | 對話式問答，可切換**單智能體 / 多智能體合議**；展示工具調用/專家協作時間線、證據卡片、引用核驗橫幅；角色（醫師/科研/學生/患者） |
| **原文檢索** | BM25 + 六經過濾 + 關係擴展；命中可點開條文抽屜 |
| **方證匹配** | 症狀/脈象 chip 輸入 → 候選方證 + 原文證據（醫師輔助） |
| **方證鑒別** | 選 2–3 方 → 多軸對比表 + 關鍵鑒別點 |
| **六經教學** | 提綱/亞型/主方/誤治/禁忌/條文/練習題 |
| **誤治傳變** | 誤治→變證→救治方→條文 路徑表 |
| **科研挖掘** | 高頻方劑/症狀、共現網絡、論文大綱 |
| **論文生成** | 6 類論文，渲染手稿 + 圖表資產清單 |
| **Skill 庫** | 135 個編譯後 Skill 瀏覽 |
| **接入/關於** | LLM 配置、Claude Code / Codex / OpenCode 接入命令 |

## 設計要點

- **證據優先**：答案中的 `clause_id` 渲染為可點擊的青色徽章，點擊在右側抽屜
  打開條文全息（原文 A / 異文 B / 注釋 C / 規則 / 關係）。證據層以
  A/B/C/D/E 五色標籤標註。
- **多智能體可視化**：合議模式把「調度規劃師 → 原文取證師 → 方證/鑒別/六經/
  誤治專家 → 安全治理官 → 合議綜合官」逐步呈現為時間線，每步附其證據條文。
- **引用核驗橫幅**：每條回答下方顯示綠色「✓ 證據核驗通過」或紅色
  「⚠ 未能核實」——即使接入大模型，無法回源的條文號會被當場標記。
- **安全治理**：患者模式涉及診斷/處方/劑量的提問顯示紅色攔截卡片，
  不調用任何模型。

## API（供二次開發 / 其他前端）

`GET /api/stats`、`/api/llm/status`、`/api/skills`、`/api/formulas`、`/api/channels`、
`GET /api/clause/<ref>`；`POST /api/search|match|differential|teach|mistreatment|`
`research|paper|agent|council|patient|tool|trace|herb|formula-explain|`
`gold-sample|gold-eval`；`GET /api/tools`（28 工具規格，工具台數據源）。全部返回 JSON，結構與 CLI 一致。
`POST /api/trace {"type": "clause|formula|claim|school|commentator|text",
"ref": "…"}` 返回五類溯源鏈報告（詳見 [`docs/TRACE.md`](TRACE.md)）。

## 架構

```
server/
├─ service.py       ServiceContext：框架無關 API 面（可單測）
├─ http_server.py   http.server 處理器 + 路由 + 靜態服務
└─ static/          index.html · app.css · app.js（自包含 SPA）
agent/multi_agent.py  Council 多智能體編排（規劃/取證/專家/批評/綜合）
```
