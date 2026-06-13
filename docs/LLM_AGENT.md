# LLM 接入、智能體與 Harness 集成

本文檔說明 Hermes-Shanghanlun 的神經符號（neuro-symbolic）增益層：如何接入大
語言模型、智能體如何在保持「證據回源」鐵律的前提下自主取證作答，以及如何被
Claude Code / Codex / OpenCode（openclaw）等智能體框架調用。

## 設計哲學：LLM 只做增益，絕不繞過證據閘門

```text
┌─────────────────────────────────────────────────────────────┐
│  可信底座（確定性）                                          │
│  條文 681 · 規則 1471 · 審核閘門 6 道 · 安全治理 · BM25 RAG  │
└───────────────▲─────────────────────────────▲───────────────┘
                │ 取證(工具調用)               │ 證據核驗(citation guard)
┌───────────────┴─────────────────────────────┴───────────────┐
│  增益層（LLM，可選）                                         │
│  自然語言推理 · 更難的抽取 · 語義批評 · 多輪智能體           │
└─────────────────────────────────────────────────────────────┘
```

- LLM 產出的每一句話，回給用戶前都要過 **citation guard**：凡引用的 clause_id
  或原文引文無法在語料中核實，一律標記警告。
- LLM 抽取的每一條規則，都要過 **同一套審核閘門**（證據回源是安全網）。
- 患者語境：意圖守衛在任何模型/工具調用**之前**攔截診斷/處方/劑量請求。
- **優雅降級**：未安裝 litellm 或無 API key 時，自動使用 `local` 確定性後端，
  全系統離線可用、可測試，代碼路徑與在線完全一致。

## 啟用真實大模型

```bash
pip install "litellm>=1.40"          # 或 pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...       # 或 OPENAI_API_KEY 等任一 provider key
export HERMES_LLM_MODEL=anthropic/claude-opus-4-8   # 可選，默認即此
python3 -m hermes_shanghan llm-status              # 確認後端
```

支持的後端（經 LiteLLM，100+ provider）：Anthropic Claude、OpenAI、Azure、
Gemini、Groq、Mistral、DeepSeek、OpenRouter、本地 Ollama 等。

| 環境變量 | 作用 | 默認 |
|---|---|---|
| `HERMES_LLM_PROVIDER` | `auto`/`litellm`/`local`/`scripted` | auto |
| `HERMES_LLM_MODEL` | litellm 模型 id | anthropic/claude-opus-4-8 |
| `HERMES_LLM_TEMPERATURE` | 採樣溫度 | 0.0 |
| `HERMES_LLM_MAX_TOKENS` | 最大輸出 | 1536 |
| `HERMES_LLM_CACHE` | 磁盤緩存響應（可復現） | 1 |
| `HERMES_LLM_FALLBACK` | 調用失敗回退 `local`/`none` | local |

`auto` 僅在「litellm 已安裝 **且** 檢測到 API key」時選用真實模型，否則 `local`。

## 智能體問答

```bash
# 自動推斷角色 + 工具取證 + 回源核驗 + 安全治理
python3 -m hermes_shanghan agent "少陰病寒化與熱化怎麼區分？" --role student
python3 -m hermes_shanghan agent "病人往來寒熱、胸脅苦滿、口苦，考慮什麼方？" --role doctor --answer-only
python3 -m hermes_shanghan agent "给我开个方" --role patient   # 被意圖守衛拒絕
```

智能體循環（在線/離線同構）：
```
system(角色契約) → user(問題) → [tool_call → tool_result]* → answer
                                          ↓
                          citation guard（核驗每個 clause_id/引文）
                                          ↓
                          safety.governed（角色化安全治理）
```

返回結構包含 `tools_used`、`evidence_clause_ids`、`citation_report`、
`agent_trace`（每一步工具調用與裁決），完全可審計。

## LLM 增強的規則挖掘

```bash
# 單條：LLM 抽取候選規則 → 過全部審核閘門
python3 -m hermes_shanghan llm-extract 12

# 全量：LLM 抽取增強 + LLM 對抗式批評器（候選仍受證據閘門約束）
python3 -m hermes_shanghan pipeline --llm-extract --llm-critic
```

- `--llm-extract`：LLM 候選規則與確定性規則合併去重後，**統一過審核**。
  在 `local` 後端，LLM 鏡像規則引擎，增量為 0；真實模型才會擴大召回。
- `--llm-critic`：LLM 對抗式批評器作為**附加閘門**，僅能下調等級（advisory），
  不能把證據不實的規則提升放行——硬證據閘門始終優先。

## 8 個可調用工具（智能體 / harness 共用同一能力面）

`shanghan_search`、`shanghan_get_clause`、`shanghan_match_formula`、
`shanghan_differential`、`shanghan_six_channel`、`shanghan_formula_rule`、
`shanghan_mistreatment`、`shanghan_list_formulas`。全部只讀、回源 clause_id。

```bash
python3 -m hermes_shanghan tool-call shanghan_differential --args '{"formulas":["桂枝湯","麻黃湯"]}'
python3 -m hermes_shanghan export-tools --out tools.json   # OpenAI+Anthropic 規格
```

## 接入智能體框架

### Claude Code / Claude Desktop（MCP）
```bash
claude mcp add shanghan -- python3 -m hermes_shanghan serve-mcp
```
暴露上述 8 個工具 + `shanghan_ask`（完整智能體）。MCP 服務器為純標準庫實現的
JSON-RPC over stdio，無第三方依賴。

### Codex CLI / OpenCode / openclaw（OpenAI 兼容工具）
```bash
python3 -m hermes_shanghan export-tools --out tools.json
python3 -m hermes_shanghan tool-call shanghan_search --args '{"query":"結胸"}'
```
或在 Python 函數調用循環中：
```python
from hermes_shanghan.integrations import openai_tool_specs, dispatch
tools = openai_tool_specs()
dispatch("shanghan_six_channel", {"channel": "太陽病"})
```

### 任意 LiteLLM 智能體
```python
from hermes_shanghan.agent import ShanghanAgent
print(ShanghanAgent().ask("桂枝湯與麻黃湯如何鑒別？", role="doctor")["answer"])
```

詳見 `hermes_shanghan/integrations/AGENTS.md`。

## 模塊一覽

```text
hermes_shanghan/llm/         config · cache · prompts · providers(litellm/local/scripted) · client
hermes_shanghan/agent/       tools(8) · citation_guard · agent(ReAct)
hermes_shanghan/extract/     llm_extractor（LLM 抽取，過審核閘門）
hermes_shanghan/review/      llm_critic（可選附加閘門）
hermes_shanghan/integrations/ tool_specs(OpenAI/Anthropic) · mcp_server · AGENTS.md
```
