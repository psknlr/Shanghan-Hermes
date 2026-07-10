# Agent 執行 Harness（狀態圖 · 可恢復 · 可觀測 · 可審計）

回應「頂級 harness」評審（八輪十二方向 + 九輪動態審計）。目標架構：

```text
Hermes Harness =
  RunSpec + StateGraph + ToolContract + EvidenceLedger
+ CitationGuard + SafetyGate + HumanReviewGate
+ TraceStore + EvalRunner
+ MCP Resources/Prompts/Tools + CorpusLifecycle
+ Policy/Principal + RunBudget + Readyz          ← 九輪新增
```

## 一、已落地（v2，純標準庫）

| 評審方向 | 落點 |
|---|---|
| 1. 統一 RunSpec/RunState | `agent/harness/state.py`：RunSpec 含**環境指紋**（corpus/tool_spec/python/backend/git HEAD）、NodeSpec（retry/fallback/evidence_requirement/release_condition）、RunState（evidence_ledger/tool_calls/guardrail_events/approval_requests/budget_snapshot） |
| 2. 顯式節點圖 | v1 四節點圖（intake→execute→evidence_audit→release_gate），重試/降級/依賴跳過；模式引擎作為 execute 節點掛入 |
| 3. checkpoint/resume/replay | `runs/<run_id>/state.json`（原子寫）+ `events.jsonl` + **run.lock 單寫者鎖**；trace_id **跨 resume 延續**；replay 先對比環境指紋（不一致如實標 comparable=False），再對比回答指紋 |
| 4. span 級軌跡 | OTel 風格 JSONL span；TracedRegistry 使每次工具調用自動成 span（含 cache_hit/budget_denied/backend 元數據）；**異常入軌跡前脫敏**（去絕對路徑+截斷） |
| 5. MCP | 版本協商（3 版本）+ resources（8 URI）+ prompts（4 模板）+ **實驗性 tasks**（submit/status/result/cancel/list，長任務不再同步阻塞；取消為協作式——結果丟棄，如實聲明） |
| 6. 工具契約 | `Tool.contract()` 帶 **enforced 節**：逐條款聲明執行方式。運行時真執行：參數校驗、**超時**（工作線程+join，超時回錯誤信封）、輸出形狀（必須 dict）、大小上限（報錯不截斷）、**版本化緩存鍵**（tools_version+語料指紋入鍵）、環形審計日誌 |
| 7. 軌跡級評測 | tool_name_accuracy / trajectory_validity_rate / refusal_precision + 故障注入 recovery_success_rate——**均為閉集回歸護欄，非能力宣傳指標**（口徑表見 MATURITY.md） |
| 8. Human-in-the-loop 發布閘門 | **五態 fail-closed**：pass / pass_with_warning / review_required / **blocked** / **failed_closed**。citation_report 缺失→failed_closed；偽造引用/患者端方藥指令→blocked（**人工批准不可放行**）；候選方檢測用結構化信號（match/hypotheses/adjudicate 在台賬）非「湯」字關鍵詞；review_required 生成 ApprovalRequest（審什麼/證據指紋/時間/審批人）；`run-resume --approve` **重新執行下游閘門**後才放行（pass_after_human_review），`--reject` 駁回 |
| 9-. 統一預算 | `RunBudget`：Harness 控制器持有、TracedRegistry **原子扣減**（跨 for_role 副本共享），批量 tool_calls 逐個檢查，超限回 BUDGET_EXHAUSTED 不執行；agent 內部 `_react` 同樣逐調用檢查（模型單輪返回 N 個調用不能突破預算） |
| 10. 語料生命週期 | `corpus/source_registry.py` + P 層；**readyz**（`/livez` `/readyz` 分離 + CLI `readyz`）：manifest/398 條/規則庫/工具規格逐項校驗；資產缺失時 ToolRegistry 構建**響亮失敗**（assert_ready），拒絕 wheel 假健康空運行（數據部署二選一見 pyproject 說明） |
| 11. 依賴注入 | execute 節點統一注入 TracedRegistry；**solve 模式（ComplexAgent）與子代理不再自行 get_registry()**——複雜任務的工具調用進台賬與 span 樹 |
| 12. API 治理 | **服務端 Principal**（`server/policy.py`）：HERMES_API_KEYS 綁定 token→角色上限，請求體 role 只可降級、自提權 403（可審計）；全部臨床端點（match/differential/formula/mistreatment/deep-research…）帶最低角色過同一策略層；session 以主體命名空間隔離（無 id 不共用 default，服務端生成回傳；TTL+容量上限）；糾正記憶帶來源與 unverified 信任級 |
| 13. 規劃編譯 | `planner.compile_plan`：唯一 ID/依賴存在/無環/類型合法/預算——LLM 計劃編譯失敗先回饋修復一次，仍失敗 **fail-closed 回退確定性規劃器**；execution_order 遇環直接拋錯，不再靜默按序執行；max_subtasks 不再被 max(...,5) 覆蓋 |
| 14. 研究覆蓋狀態 | 深研發現帶狀態 FAILED/EMPTY/DATA_FOUND/EVIDENCE_FOUND/VERIFIED：工具報錯或空手而歸**不算覆蓋**；無引用 finding 不再 citation_ok=True；聚合統計模塊如實標 DATA_FOUND 不冒充 VERIFIED；harness 回答納入全部發現（不截前 4） |

## 二、規劃中（如實列差距）

| 方向 | 差距與計劃 |
|---|---|
| 2+. 圖原生細粒度編排 | v2 仍把模式引擎整體作為 execute 節點；把檢索/專家/批評/綜合拆成獨立 typed 節點（input/output schema、節點級預算/緩存/取消）列下輪 |
| 3+. durable execution | 現為單進程 JSON checkpoint + 文件鎖：無 lease/心跳/exactly-once/DLQ；副作用工具（現全只讀）加入前必須先補 |
| 5+. MCP progress notification | tasks 已可輪詢；服務端主動 progress 推送需雙向流改造 |
| 8+. 身份聯邦 | Principal 已服務端化；JWT/OIDC/反代映射屬部署層，接口留在 policy.resolve_principal |
| 9. 專家獨立 evidence packet | 見 AGENT_ROADMAP「多智能體專家獨立性」設計（分層檢索隔離+匿名 claim 評審+主動反證） |
| P0-5+. 語義蘊含核驗 | EvidenceBinder 已輸出結構化 claim/evidence_links，verifier 如實標 lexical_overlap_v1（詞彙級下界）；supports/contradicts 級 entailment 需模型後端 |
| 7+. redteam / 多標註者 κ | 對抗提示集與 Cohen's κ 一致率待建（goldset 已有單標註閉環） |
| Pydantic/OTel SDK | 零依賴約束下不引入；契約/span 為兼容結構，外部可直接轉譯 |

## 三、十層目標架構映射（十輪評審 七）

評審給出的十層目標架構逐層對照現狀（落點 / 差距，均如實）：

| # | 目標層 | 現狀落點 | 差距 |
|---|---|---|---|
| 1 | Identity & Tenant Gateway | `server/policy.py`：Principal（subject/角色上限/auth_level），HERMES_API_KEYS 綁定 | 多租戶 tenant_id 僅佔位；JWT/OIDC 屬部署層 |
| 2 | Policy Engine | 端點最低角色矩陣 + 請求體 role 只降不升 + 工具面 ScopedRegistry 硬裁剪 + 紅旗分診/意圖守衛 | 目的限制（purpose_of_use）與逐工具二次審批矩陣未細化 |
| 3 | Durable Run Controller | 四節點狀態圖 + 原子 checkpoint + run.lock + RunBudget 原子扣減 + resume/replay（環境指紋） | typed DAG 細粒度節點、事務語義、節點級取消在路線 |
| 4 | Model Gateway | `llm/client.py` 統一後端路由（local 確定性/litellm 增益層），backend 進 RunSpec 指紋 | token/cost 真實計量僅真模型後端有；模型版本 pin 未強制 |
| 5 | Capability Broker | `ToolRegistry.call()` 管道：默認拒絕→參數校驗→版本化緩存→超時→輸出校驗→大小→審計（契約 enforced 節如實聲明） | idempotency-key 級去重未做（工具全只讀，重試天然安全） |
| 6 | Evidence Plane | A/B/C/D/E/P 分層 + **EvidenceRecord 逐證據來源對象**（版本指紋/quote_hash/檢索上下文，缺失字段記 null）+ **work_type 分類**（未登記書目 fail-closed 到 P，證據層不由目錄名決定）+ 引文邊質量信號（coverage/modes） | 字符級偏移未保留（切分管道限制，如實記 null）；Support/Contradict 語義關係為詞表級 |
| 7 | Independent Specialist Agents | Council 多視角 + **argument 論證鏈**（支持/反證/異文分叉/注家共同與爭議/隱含假設/不可裁決七段分層） | 專家獨立 evidence packet（分層檢索隔離+匿名評審）在路線 |
| 8 | Verification & Guardrail Pipeline | schema（參數/輸出）→ CitationGuard（編號+本輪取證+引文逐字）→ EvidenceBinder（句級詞彙下界）→ 安全治理 → 質量警示 | entailment 級語義核驗需模型後端（verifier 如實標 lexical） |
| 9 | Human Approval | ApprovalRequest（digest/時間/審批人）+ approve 重跑下游閘門 + reject + blocked 不可批准 | action 級（逐工具調用）批准與有效期（expires_at）未做 |
| 10 | Release + Observability | 五態發布決策 + span 軌跡（JSONL，OTel 兼容）+ 審計環 + 閉集回歸指標（口徑表見 MATURITY.md） | OTel exporter/retention policy 屬部署層 |

**語料供應鏈**（十輪 六.4）：`library.fetch` 全鏈加固——URL allowlist
（自定義源須 `HERMES_LIBRARY_ALLOW_CUSTOM=1` + 顯式 SHA-256，fail-closed）、
下載超時與 200MB 上限、成員名審查（絕對路徑/`..`）、解壓後樹審查
（symlink/設備文件/3 萬文件/2GB/壓縮比 60×）、臨時目錄解壓 + 結構校驗後
原子切換、`provenance.json` 全程記錄。

## 四、使用

```bash
python3 -m hermes_shanghan run "惡寒發熱，汗出，脈浮緩，用什麼方？" --mode agent --role doctor
# → status: paused（結構化候選方信號觸發人工審核，附 ApprovalRequest）
python3 -m hermes_shanghan run-list
python3 -m hermes_shanghan run-resume <run_id> --approve --approver 張醫師   # 重跑下游閘門後放行
python3 -m hermes_shanghan run-resume <run_id> --reject  --approver 張醫師   # 駁回
python3 -m hermes_shanghan run-replay <run_id>     # 指紋一致+local 後端 → 回答指紋必一致
python3 -m hermes_shanghan run-export <run_id> --format md
python3 -m hermes_shanghan readyz --runtime        # 就緒探針（exit 2=未就緒）
```

治理部署示例：

```bash
# 角色綁定 API key：patient key 無法自稱 doctor（403 policy_denied）
HERMES_API_KEYS="tokA:patient:alice,tokB:doctor:drwang" \
  python3 -m hermes_shanghan serve --host 0.0.0.0
# 公網匿名演示：整個匿名面裁到患者安全層
HERMES_ANON_ROLE=patient python3 -m hermes_shanghan serve --host 0.0.0.0
```

運行目錄 `data/shanghan/runs/<run_id>/`（state.json + events.jsonl +
run.lock，含時間戳故 gitignore，不影響流水線字節級可復現保證）。
