# Agent 執行 Harness（狀態圖 · 可恢復 · 可觀測 · 可審計）

回應「頂級 harness」評審（十二方向）。目標架構（已按此落地/規劃）：

```text
Hermes Harness =
  RunSpec + StateGraph + ToolContract + EvidenceLedger
+ CitationGuard + SafetyGate + HumanReviewGate
+ TraceStore + EvalRunner
+ MCP Resources/Prompts/Tools + CorpusLifecycle
```

## 一、已落地（v1，純標準庫）

| 評審方向 | 落點 |
|---|---|
| 1. 統一 RunSpec/RunState | `agent/harness/state.py`：RunSpec（含 corpus_version/tool_spec_version 指紋）、NodeSpec（retry/fallback/evidence_requirement/release_condition）、NodeResult、RunState（evidence_ledger/tool_calls/guardrail_events） |
| 2. 顯式節點圖 | v1 四節點圖（intake→execute→evidence_audit→release_gate），節點帶重試/降級策略、依賴跳過；模式引擎（agent/council/deep-research/solve）作為 execute 節點掛入 |
| 3. checkpoint/resume/replay | `runs/<run_id>/state.json`（原子寫）+ `events.jsonl`；`run` / `run-list` / `run-resume [--approve]` / `run-replay`（local 後端全確定，對比回答指紋）/ `run-export --format md\|json` |
| 4. span 級軌跡 | `harness/tracing.py`：OTel 風格 JSONL span（trace/span/parent、時長、輸入輸出哈希、error、evidence_ids；token/cost 在 local 後端如實記 null）；TracedRegistry 使每次工具調用自動成 span |
| 5. MCP 完整性 | 版本協商（2024-11-05/2025-03-26/2025-06-18）+ **resources**（條文/方證規則/引文網絡/觀點庫/學派/ID註冊表/manifest/全庫編目 8 個 URI）+ **prompts**（方證鑒別/深度溯源/誤引審查/患者安全問診 4 模板） |
| 6. 工具契約 | `Tool.contract()`：version/permission_level/side_effect(read 不變式)/evidence_level/timeout/cacheable/idempotent/max_result_bytes/error_schema/schema_hash；隨 `tool_specs.json` 導出（`contracts` 節）；輸出超限報錯不靜默截斷 |
| 7. 軌跡級評測 | `eval/trajectory.py`：tool_name_accuracy / trajectory_validity_rate / refusal_precision + **故障注入**（工具拋異常/空結果）計 recovery_success_rate（實測 1.0：優雅降級不崩潰） |
| 8. Human-in-the-loop 發布閘門 | `harness/release_gate.py` 五道（evidence/safety/role/uncertainty/human-review）；醫師端候選方、引用核驗失敗、方證未決、論文生成觸發 needs_human_review → run 轉 paused → `run-resume --approve --approver X`（審批人記錄於 guardrail_events） |
| 10. 語料生命週期 | `corpus/source_registry.py`：source_id/sha256/license/parser_version/質檢/證據層歸屬；新增 **P 層**（旁證）正式入 LAYER_LABEL，與 A–E 嚴格分離 |
| 12. API 治理 | `/api/tool` 按角色限權（patient 硬裁剪）；響應大小上限（超限回錯誤+trace_id）；每 IP 速率限制（`HERMES_RATE_LIMIT` 選裝）；異常響應附 trace_id（詳情僅留服務端日誌） |

## 二、規劃中（如實列差距）

| 方向 | 差距與計劃 |
|---|---|
| 2+. 圖原生細粒度編排 | v1 把模式引擎整體作為 execute 節點；下一步把檢索/專家/批評/綜合拆成獨立節點（complex_agent 的任務圖已是雛形），失敗恢復到子節點粒度 |
| 5+. MCP progress/cancellation/sampling | 長任務（全庫掃描/深研）的進度通知與取消需要雙向流；stdio 單線程實現需要任務線程池，列入下輪 |
| 7+. redteam / 多標註者 κ | 對抗提示集與 Cohen's κ 一致率待建（goldset 已有單標註閉環） |
| 9. 專家獨立 evidence packet | 見 AGENT_ROADMAP「多智能體專家獨立性」設計 |
| 12+. 後台任務隊列 | 全庫掃描/深研改 background job + progress + cancel；當前為同步阻塞（CLI 可 Ctrl-C，run 狀態可恢復） |
| Pydantic/OTel SDK | 零依賴約束下不引入；契約/span 為兼容結構，外部可直接轉譯 |

## 三、使用

```bash
python3 -m hermes_shanghan run "桂枝湯與麻黃湯如何鑒別？" --mode agent --role doctor
# → status: paused（醫師端候選方觸發人工審核）
python3 -m hermes_shanghan run-list
python3 -m hermes_shanghan run-resume <run_id> --approve --approver 張醫師
python3 -m hermes_shanghan run-replay <run_id>     # local 後端指紋必一致
python3 -m hermes_shanghan run-export <run_id> --format md
```

運行目錄 `data/shanghan/runs/<run_id>/`（state.json + events.jsonl，
含時間戳故 gitignore，不影響流水線字節級可復現保證）。
