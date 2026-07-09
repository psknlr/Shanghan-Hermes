# 中醫古籍深度溯源與學術計量追溯層（trace）

本文檔對照「中醫古籍深度溯源與學術計量追溯智能體」設計方案（二十一節），
逐項說明：哪些能力在本庫**原已具備**、哪些由 trace 層**新建**、哪些設計點
**經評估後調整**（附理由）。全層純標準庫、確定性、字節級可復現，延續
「無證據鏈，不成回答」的硬約束。

## 一、對照設計方案的實現狀態

| 設計方案章節 | 狀態 | 落點 |
|---|---|---|
| §2 原文證據錨點 / 多觀點並存 | 原已具備並沿用 | CitationGuard／多智能體合議；trace 層全部輸出綁定 clause_id |
| §3 溯源對象（條文/方/藥/證/法/觀點/注家/學派/引用） | 新建統一建模 | `trace/ids.py` + `chains.py` 六類入口 |
| §4.1–4.4 原典/注本/方書/醫案庫 | 原已具備 | A/B/C/D 四層 + 經方實驗錄 |
| §4.5 現代學術文獻庫 | 調整（見下） | `trace/modern.py` 導入接口 |
| §5 統一知識標識 | 新建 | `id_registry.json`（WorkID/EditionID/FormulaID/MethodID/SyndromeID；SchoolID/ClaimID/CitationEdgeID 見各資產） |
| §6 古籍引用模式識別 | 新建（核心增量） | `trace/quotation.py`：明引/節引/暗引/化用/改寫/轉引注文/存疑引用 |
| §7 條文深度溯源 | 新建（整合既有層） | `chains.clause_chain`：原文→異文→上下文→注家→歷代引用→計量→現代 |
| §8 方劑源流追溯 | 新建（整合劑量層） | `chains.formula_chain`：首見→組成→類方劑量演化→方名傳播 |
| §9 方證觀點演化 | 新建 | `trace/claims.py`：7 個結構化 ClaimID，證據分級全由數據判定 |
| §10 學派觀點建模 | 新建 | `trace/schools.py`：10 個 SchoolID，回填分歧圖譜一致度實測 |
| §11 現代引用與功能分類 | 調整實現 | `modern.py`：導入+回源+九類功能分類規則 |
| §12 科學計量 | 新建 | `trace/scientometrics.py`：引文網絡/共引/文獻耦合/時間切片/突現/主路徑 |
| §13 雙層知識圖譜 | 新建（第二層） | 層1=既有 ClauseRelation；層2=引文邊網絡；橋接=clause_id/方名/ClaimID |
| §14 智能體模塊 | 接線 | 工具 `shanghan_trace`/`shanghan_citation_network`；研究循環第 7 維「引文傳播」 |
| §15 核心算法 | 新建 | 逐字 8-gram 對角線合併 + 覆蓋率分類 + 稀有二元組剪枝 Dice |
| §16 五類溯源鏈 | 新建 | `trace/chains.py`（原文/方劑/方證觀點/注家/學派） |
| §17 輸出模板 | 新建 | 溯源鏈結構化報告（evidence_grade + warnings 必附） |
| §18 評價體系 | 部分新建 | `quotation.selfcheck()` 合成基準 + 24 項單元測試；原有四大基準沿用 |
| §19 應用場景 | 覆蓋 | 見下「使用」 |
| §20 證據約束生成 | 原已具備並擴展 | 引文邊逐字回源；存疑引用只計數不猜出處 |

## 二、經評估後調整的設計點（及理由）

1. **現代論文/教材庫不隨庫分發、不憑空生成。** 現代文獻受版權與獲取限制，
   打包即意味著要麼侵權要麼編造。實現為 `modern_citations.jsonl` 導入接口：
   研究者自備記錄，導入時過與古籍層相同的逐字回源匹配器並做引用功能分類，
   與歷代切片同網絡可比。語料自帶的最晚傳播層為民國（1937《經方實驗錄》），
   歷時跨度東漢→民國約 1800 年，計量結論如實限定在此範圍。

2. **「誤引檢測」改為「存疑引用」統計。** 有引述標記（某曰/某云/《書名》）
   而在傷寒論內無可回源匹配的段落，多數是引《內經》《難經》等他書或佚文，
   斷為「誤引」需要他書全文庫佐證。系統只作存疑計數並保留樣例
   （`citation_book_stats.json`），不猜出處——寧可少斷言，不可錯斷言。

3. **深度改寫（無標記且無逐字片段的意譯）不偽裝成確定性能力。** 語義向量
   匹配屬 LLM 增益層（可選接入），確定性核心只承諾：逐字片段（≥8 字）、
   帶標記的近似改寫（稀有二元組剪枝 + Dice ≥ 0.45）。`selfcheck()` 合成基準
   如實給出各模式檢出率下限。

4. **學派歸屬標為 posthoc_induction 且回填實測證據。** 學派是學術史歸納而
   非文本事實。註冊表僅收語料在庫著者，並把注家分歧圖譜的一致度矩陣回填：
   例如錯簡重訂（方有執）× 以法類證（錢潢）實測一致度 0.0842，為「學派
   分野」提供可檢驗的數據面，而非單純貼標籤。

5. **方證觀點的證據等級由機器判定而非種子預設。** 種子只給命題與解釋性
   術語；「營衛不和」類命題若其術語逐字見於 A 層（如第 53 條「榮氣和」、
   54 條「衛氣不和」）則判「原文直述成分」，僅見於注文則判「後世歸納」——
   協議「方證歸納必須區分原文直述、後世歸納」在觀點層的延伸。

6. **段落級全量引文邊（~4 萬條）不作為提交資產。** 可由掃描器在數秒內
   確定性重建（`trace-scan-full --out`），倉庫只提交 (著作,條文) 級聚合
   （~1.3 萬行）與計量網絡摘要，避免倉庫膨脹且保持字節級可復現。

## 三、引用模式判定規則

| 模式 | 判定（折疊異體字、剝離標點後） |
|---|---|
| 明引 | 引述標記 + 條文覆蓋率 ≥ 0.7 |
| 節引 | 引述標記 + 逐字片段 ≥ 8 字 + 覆蓋率 < 0.7 |
| 暗引 | 無標記 + 覆蓋率 ≥ 0.7 |
| 化用 | 無標記 + 逐字片段 ≥ 8 字 |
| 改寫 | 有標記、無逐字片段、Dice ≥ 0.45 |
| 轉引注文 | 逐字片段（≥12 字）命中注家注文而非經文（記 via_book/via_commentator） |
| 存疑引用 | 有標記而庫內無匹配（只計數 + 樣例，不猜出處） |

防護：套語 8-gram（見於 >20 條文）從索引剔除；同一片段可歸屬 >3 條文則
放棄計邊；對話體（問曰/答曰/師曰）不算引述標記；注本自引/同注家自引不算
轉引。

## 四、資產清單（`data/shanghan/trace/`）

| 文件 | 內容 |
|---|---|
| `id_registry.json` | 統一 ID：57 WorkID、4 EditionID、113 FormulaID、23 MethodID、8 SyndromeID + 朝代補注（逐條註明依據） |
| `citation_edges_agg.jsonl` | (著作,條文) 級引文邊聚合（模式計數/最大覆蓋率/最長片段） |
| `citation_relay_agg.jsonl` | (著作,經由注本) 級轉引聚合——注本樞紐度 |
| `citation_network.json` | 計量網絡：被引榜/共引對/文獻耦合/朝代切片/突現/主路徑 |
| `citation_book_stats.json` | 逐書掃描統計 + 掃描參數 + 存疑標記樣例 |
| `formula_mentions.json` | 方名源流（長名優先遮蔽防子串誤計） |
| `schools.json` | 10 學派：範式/成員/適用邊界/對立學派/一致度證據 |
| `claims.json` | 7 方證觀點：原文逐字檢驗/注家首倡時間線/學派立場/爭議 |

## 五、使用

```bash
# 任意文本回源（原文溯源鏈；簡繁/異體字皆可）
python3 -m hermes_shanghan trace "观其脉证，知犯何逆，随证治之"

# 條文 / 方劑 / 方證觀點 / 注家 / 學派溯源鏈
python3 -m hermes_shanghan trace 12 -t clause
python3 -m hermes_shanghan trace 桂枝湯 -t formula
python3 -m hermes_shanghan trace 營衛不和 -t claim
python3 -m hermes_shanghan trace 成無己 -t commentator
python3 -m hermes_shanghan trace 錯簡重訂 -t school

# 學術計量網絡（總覽或指定對象）
python3 -m hermes_shanghan trace-network
python3 -m hermes_shanghan trace-network --target 桂枝湯

# 重建溯源資產 / 導出全量段落級引文邊
python3 -m hermes_shanghan trace-build
python3 -m hermes_shanghan trace-scan-full --out /tmp/edges.jsonl
```

服務端 `POST /api/trace {"type": "formula", "ref": "桂枝湯"}`；
智能體/MCP 經 `shanghan_trace`、`shanghan_citation_network` 兩工具自動可用
（規格自動導出）。患者模式**不暴露**溯源工具（方劑鏈含組成/劑量）。

## 六、全庫掃描（引用方擴展到 803 部醫籍）

`trace-scan-library` 把引文掃描的**引用方**從隨庫 57 部擴展到中醫笈成
全庫（需先 `library fetch`）。實測（本容器）：843 個文本單元掃描 55 秒，
**63,906 條引文邊、541 部書含傷寒引文**，時間跨度漢→唐（千金/外台）→
宋金元→明（普濟方/證治準繩）→清（醫宗金鑑）→民國→**中華人民共和國**
（972 條邊）——「現代傳播層」在全庫尺度上真實存在。三點誠實約束：
全庫邊屬文獻旁證層（layer=旁證，不入經文閘門）；漢代單元是仲景書自身
的庫內版本（版本見證而非後世引用，按朝代過濾）；全庫不隨倉庫分發，
故其邊不作為提交資產，由命令按需重建。`text_trace` 在傷寒論內無匹配時
自動退到全庫檢索候選出處（如「肾主骨」→《內經》類書目定位），
只報「書·章節」不臆斷首出。

## 七、二輪評審意見採納記錄

| 意見 | 處置 |
|---|---|
| trace 整體標 C 層不準確 | **採納**：`evidence_level: "mixed"`，五類鏈逐節附 `section_evidence_levels`（A/B/C/D/引文邊/計量/現代導入逐節標註） |
| 輔助篇章主導被引榜 | **採納並證實**（混排榜前 7 名均為 AUX）：計量網絡分三榜（canonical/auxiliary/all），工具與 CLI 加 `scope` 參數，默認 canonical；主路徑改基於正文榜；資產附 `ranking_note` |
| 全量測試需明確報告 | **採納**：[`TEST_REPORT.md`](TEST_REPORT.md)（環境/總量/逐模塊耗時/慢因排查/告警狀態）；test_server 的 socket ResourceWarning 已修復 |
| rejected=0 與 README 張力 | **採納（文檔而非演示數據）**：[`REJECTION_CASES.md`](REJECTION_CASES.md) 給出審計實測（evidence·fail 10 例、repair 32 例）與五個對抗案例的真實閘門輸出；不往 `data/` 放手工樣例以免破壞字節級可復現 |

## 七b、三輪評審意見採納記錄

| 意見 | 處置 |
|---|---|
| scope 只過濾被引榜，time_slices 等仍混 AUX | **採納方案 A（嚴格全字段過濾）**：時間切片焦點/共引/突現/主路徑逐 scope 重算（非事後過濾，杜絕「過濾後榜單失真」），工具端全字段按 scope 組裝；新增 `trace-audit-scope`（A1）對輸出全文遞歸掃描違例，實測三 scope 全部 0 違例 |
| TEST_REPORT 與實測不一致（3.13 有 ResourceWarning） | **採納**：修復 test_refinements 未關閉的 clauses.jsonl、test_hardening 第二處服務器 socket（`server_close`+`HTTPError.close()`）；文檔表述改為版本相關的誠實措辭 |
| earliest_source 實為支持條文全集且混入 AUX | **採納**：改為 `first_attestation`（單一首見條文，注明「首見=宋本條文序，非跨書史源判定」）+ `supporting_clauses.{canonical,auxiliary}` 分列 |
| A1–A5 證據溯源智能體 | **全部落地**；B/D 組臨床與裁決類列入 [`AGENT_ROADMAP.md`](AGENT_ROADMAP.md)（含現有能力映射與設計約束） |

新增能力速覽：`trace X -t quote`（誤引檢測）· `trace-audit-citation`
（逐邊可靠性審計）· `trace-gold-sample`/`trace-gold-eval`（人工金標準閉環）·
claims 增 `first_proponent`/`term_first_use`（觀點譜系）·
`herb`（藥證檔案）· `formula-explain`（方解一站式）。

## 七c、四輪評審意見採納記錄

全部六項採納（詳表見 [`AGENT_ROADMAP.md`](AGENT_ROADMAP.md) 第四輪落地補充）：
文獻耦合逐 scope 重算（合成邊單元測試驗證語義）· `trace-audit-citation`
區分本書注文（self_commentary）與後世轉引（relay_commentary）·
藥解/方解升級為註冊表工具（24 工具，規格自動導出）+ `POST /api/herb`、
`/api/formula-explain` · 方解三層症狀口徑（首見/全書聚合/特殊上下文）·
金標準 `--stratify` 分層抽樣（朝代×預測模式，零隨機）· 術語譜系鏈
（`-t term`：A 層逐字檢驗→在庫首現注家→學派分佈）· memory 文檔同步
（Core 7 + 2 agent-layer）。

## 七d、六輪落地（研究平台增強 + 前端）

- **注家爭議結構化**（`trace 12 -t dispute`）：各家觀點按朝代排列，附
  貼近原文程度（字面重合率）、後世術語密度、分歧類型提示（訓詁/方證/
  病機/治法/劑量，E 層啟發式標注）與論文寫法建議；呈現證據結構不裁決。
- **學派比較**（`trace "柯琴 vs 尤怡" -t compare`）：範式/指紋術語/實測
  一致度/高分歧條文對照。
- **方劑異名歸並**（`trace/aliases.py`）：編輯性對照表（陽旦湯/人參湯/
  回逆湯…）+ 組成比對結論；可安全歸並者參與方名計量但**與正名分列**，
  歧義名（柴胡湯）標不可合併。
- **本草證據層**（herb_profile.bencao_layer）：全庫本草類書原文摘錄
  （書·章節定位），嚴格分層為旁證，不入經文閘門。
- **Web 溯源工作台**：誤引檢測粘貼即標注（原文直引/後世歸納語分色）、
  文本回源、術語譜系、方劑源流、注家爭議、學派比較六模式；另有方藥
  檔案與辨證閉環模塊；移動端自適應；頁腳標注研發來源
  （醫哲未來人工智能研究院）。
- **Colab**：新增辨證閉環/溯源工作台演示與 ngrok 公網映射一節。

## 八、已知邊界（如實聲明）

- 引文檢測以宋本條文（含輔助篇章）為靶集；《金匱》條文、方後注不在靶集。
- 方名計量不含異名歸併（如「陽旦湯」之於桂枝湯）；屬後續工作。
- 突現分析在 50 餘部著作粒度上為粗粒度信號，樣本量隨切片如實報告。
- `selfcheck()` 為合成基準（算法下限標尺），非人工標註金標準；與古籍文獻
  專家的一致性評價需另行人工評測。
- 全庫掃描邊未納入隨庫提交的計量網絡資產（全庫不隨倉庫分發）；
  需要全庫尺度計量時先 `library fetch` 再 `trace-scan-library --out`。
