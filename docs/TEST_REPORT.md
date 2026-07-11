# 測試運行報告（環境 · 耗時 · 外部依賴 · 已知告警）

本報告回應評審意見「全量測試需要明確報告」。以下數據在本倉庫當前提交上
實測（測試本身確定性，耗時隨機器浮動）。

## 一、參考環境與總量

| 項 | 值 |
|---|---|
| Python | 3.11.15（倉庫最低要求 ≥ 3.9） |
| 操作系統 | Linux 6.18（x86_64 容器） |
| 第三方依賴 | 無（純標準庫；`litellm`/`py7zr` 均為可選） |
| 測試總數 | 500 項 / 29 個模塊（實測值由 tests/test_docs_sync.py 守衛強制與文檔一致） |
| 全量耗時 | `python3 -m unittest discover -s tests`：**約 20–40 秒**（見下） |
| 網絡需求 | **零**（默認全部離線） |
| 7z 需求 | **零**（僅 `library fetch` 可選功能需要 `py7zr` 或系統 p7zip） |
| 自動下載 | **無**：測試不觸發 69MB 全庫下載（全庫掃描測試用合成夾具；`HERMES_LIBRARY_AUTOFETCH` 默認關閉） |

## 二、逐模塊耗時（單獨運行實測）

| 模塊 | 測試數 | 耗時 |
|---|---|---|
| test_trace | 26 | **11.1s**（最慢：構建引文索引 + 自檢基準 + 研究循環） |
| test_paper_llm | 15 | 1.8s |
| test_server | 13 | 1.4s |
| test_hardening | 19 | 1.3s |
| test_agent_enhancements | 30 | 1.3s |
| test_eval | 8 | 1.1s |
| test_deep_research | 9 | 0.9s |
| 其餘 11 個模塊 | 120 | 各 ≤ 0.6s |

## 三、為什麼有的環境跑超過 120 秒（排查順序）

1. **設置了真實 LLM 環境變量**（最常見）。若環境中存在
   `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` 等且裝有 litellm，智能體類測試會
   走真實模型調用（網絡往返 × 多輪反思）。確定性離線運行請：
   ```bash
   HERMES_LLM_PROVIDER=local python3 -m unittest discover -s tests
   ```
2. **流水線資產缺失**。`data/shanghan/` 產物隨庫提交，測試直接讀取；
   若被刪除，首個用到的測試會觸發 `run_pipeline()` 重建（本機約 10s，
   慢盤/低配機器上更久）。溯源資產缺失同理（`trace-build` 約 6s）。
3. **低配置環境**（如免費 Colab 單核）：全量以 2–4 倍係數放大。

## 四、已知告警狀態

| 告警 | 狀態 |
|---|---|
| `ResourceWarning: unclosed socket`（test_server） | **已修復**：`tearDownClass` 補 `server_close()` + `thread.join()` |
| `ResourceWarning: unclosed socket`（test_hardening，第二處） | **已修復**：鑒權測試補 `server_close()` + `th.join()`，兩處 `HTTPError` 響應對象顯式 `close()`（HTTPError 本身持有 socket） |
| `ResourceWarning: unclosed file clauses.jsonl`（test_refinements） | **已修復**：改用 `with open(...)` 上下文管理 |
| test_server 論文測試向 `data/shanghan/papers/` 洩漏產物 | **已修復**：測試內把 `PAPER_DIR` 重定向到臨時目錄，倉庫資產區不再被測試污染 |
| 其他 ResourceWarning / DeprecationWarning | Python 3.11 環境 `-W default` 全量運行未觀察到；不同 Python 版本（如 3.13）GC 時機不同，仍可能暴露新的告警——不影響功能與結果，發現後按上述模式收斂（`python3 -W default -X tracemalloc=5 …` 可定位分配點） |

## 五、測試分層一覽

| 類別 | 模塊 | 要點 |
|---|---|---|
| 對抗性審核 | test_review | 注入偽造證據/幻覺條文/錯方並斷言拒絕或修復（見 [`REJECTION_CASES.md`](REJECTION_CASES.md)） |
| 語料與可復現 | test_corpus, test_hardening | 398 條切分不變式、manifest 原子性、字節級復現 |
| 規則與應用 | test_entities, test_apps_and_rag, test_atlas_dosimetry, test_refinements | 抽取/匹配/圖譜/劑量 |
| 智能體 | test_agent_architecture, test_agent_enhancements, test_llm_agent, test_new_tools, test_deep_research | 反思環/編排/會話/路由/22 工具 |
| 評測 | test_eval | 遮方 LOCO/醫案回放/接地率 |
| 溯源層 | test_trace | 七模式引文識別/自檢基準/計量網絡/五類鏈/全庫掃描夾具/字節級重建 |
| 服務端 | test_server | HTTP API + 鑒權 + 限額 |
| 治理探針 | test_governance, test_harness | 鏡像九輪動態探針：發布閘門 fail-closed/blocked 不可批准、批量調用不破預算、solve 進台賬、角色自提權拒絕、session 隔離、readyz 假健康、契約超時/版本化緩存鍵、planner 圖編譯、研究覆蓋狀態 |
| 不變量 | test_invariants | 十三輪：外層獨立複核（偽造自報不過閘+分歧記錄）、審批不可覆蓋技術失敗（failed_closed）與失敗節點合法重試對照、queued 先落盤、取消、心跳線程、台賬角色分類（僅編號≠正文）、指代控制工具參數與答案主實體、400 參數校驗、分頁、下載附件頭、會話持久化往返、Colab 守衛（19工具變體/全庫下載opt-in/readiness raise/cell id/Harness 演示）；十四輪：僅編號證據不入允許集（strict_round 不放行）、citation_failure 不可審批（無證據鏈審批豁免根除）、逐 trigger 審批（一次批一項+陳舊 digest 拒絕）、tool 模式經 Broker+閘門全鏈路、五模式逐一有 dispatch、worker 崩潰落盤 failed、隊列背壓 429、終態不可取消、損壞 run/殘行 trace 不斷列表、會話重啟語義恢復、並發持久化無丟失、長主題文件名不碰撞、工具規格內容哈希指紋、code_tree 指紋、runs Artifact 用創建時指紋 |
| 控制面 | test_console | 十二輪：運行中心 API（異步啟動/輪詢/審批/導出）、Artifact 防穿越、評測端點、雙 UI 在位與認證頭、結構化指代解析（主語優先防偽成功）、Colab P0 守衛（固定版本/ensure_server/冪等克隆/零硬編碼統計） |
| 證據完整性 | test_evidence_integrity | 十一輪對抗回歸：零檢索猜編號不過閘、台賬 Broker 強不變量、患者 key 全鏈路（clause 不回退 student+出口投影）、intake 圖分支、結構化臨床動作、引文歸屬綁定、無引用不放行、參數深校驗、版本單源 |
| 來源治理 | test_provenance | 十輪：work_type 分類（未登記書目 fail-closed 到 P，證據層不由目錄名決定）、全庫供應鏈（URL allowlist/強制哈希/路徑穿越/symlink/壓縮比）、EvidenceRecord 逐證據來源對象、方證論證結構（反證條文/隱含假設/不可裁決） |
| 全量古籍智能體 | test_classics | 十五輪：遞歸編目（三層嵌套/隔代繼承/父子正文不重計）、Passage 穩定 ID（跨 PYTHONHASHSEED 進程）、分層檢索（布爾/座標/計數/逐層解釋/誠實封頂）、P 層證據逐字重驗（篡改即抓）、引文溯源時間有序+反證搜索、統計語義分離（全庫書目≠規則庫）、ClassicsAgent 研究留痕、harness classics 模式全鏈路（Broker P 台賬+偽造 psg 阻斷+「最早」結論政策閘）、DomainPlugin 可執行、深研全庫維度誠實跳過、驗收審計、UI 在位 |
| 論文圖形工廠 | test_paper_figs | 十五輪：sha256 穩定 ID（跨進程字節級可復現）、逐論文類型圖組（兩兩不同）、版本化輸出（內容指紋修訂目錄+冪等）、正文引用圖+正式圖例、逐 panel Source Data+數據指紋、SVG 物理尺寸/無障礙/字號下限/標籤不截斷、GraphML+凍結佈局、CVD 檢查真實落盤、fail-fast 非法類型、劑量假設區間圖、熱圖色階+共注 n、摘要不過度宣稱 |
| 文檔同步 | test_notebook | Colab 守衛（nbformat/可編譯/API 存在/溯源節在冊） |
| 全庫 | test_library | 編目/索引/檢索（合成夾具，不下載） |

> 本報告數據更新於溯源層合入時；後續大改動請重測並同步本表。
