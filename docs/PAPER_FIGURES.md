# 論文 Figure Factory（十五輪：從「有什麼畫什麼」到聲明式圖形系統）

## 已落地（tests/test_paper_figs.py 守衛）

| 評審項 | 落地 |
|---|---|
| P0-1 跨進程不可復現 | `abs(hash())` 全部替換為 sha256 `stable_id()`——同庫版本下 Mermaid/DOT 節點 ID 與整份資產**字節級一致**（測試跨 PYTHONHASHSEED 子進程驗證） |
| P0-2 八種論文共用一套圖 | `figspec.FIGURE_PLANS`：每種論文類型聲明自己的圖組（兩兩不同，測試守衛）；每張圖聲明科學問題與主信息（FigureSpec） |
| P0-3 同日覆蓋 | 輸出目錄按**內容指紋**尋址：`papers/<type>/rev_<hash12>/`（語料 manifest+類型+主題+統計快照）——同輸入冪等、變輸入開新修訂；`revisions.json` 追加式修訂史 |
| P0-4 DOT/Mermaid 非投稿圖 | 網絡圖同時輸出 **GraphML + 邊表 CSV + 凍結佈局 layout.json**（確定性環形座標）；投稿級渲染交 Graphviz/Gephi/Cytoscape（manifest 註明——stdlib 不做低質量佈局冒充投稿圖） |
| P0-5 正文不引用圖 | 結果節「4.0 圖組導覽」按編號給出每張圖的科學問題與主信息；高頻方劑/誤治/一致度/劑量/評測小節內聯引用對應 Fig.N；**QA 硬門禁：未被正文引用的圖使生成失敗** |
| P0-6 無正式圖例 | `## 圖例（Figure Legends）`：逐圖 標題/panel 描述/n（分母定義）/數據來源/誤差與統計定義/證據層級/Source Data 指向 + figure_legends.json |
| 九.1 物理尺寸 | SVG width/height 以 **mm** 標注（JournalProfile：單欄 89 / 雙欄 183）；viewBox 保留 px 畫布 |
| 九.3 靜默截斷 | 標籤列寬按最長標籤動態計算，`[:12]`/`[:5]` 類切片全部移除；熱圖列標籤旋轉全名呈現 |
| 九.4 刻度與量綱 | 條形/區間圖帶 0–100% 刻度軸 + 軸標題（單位與口徑） |
| 九.5 無障礙 | `<title>`/`<desc>`/`role="img"`/`aria-labelledby` 全圖在場（QA 檢查） |
| 十.Fig6 | 熱圖附色階圖例、每格 `(n=共注條數)`、對角「—」顯式區分自身與缺失 |
| 十.Fig7 | 劑量圖改**情景假設區間圖**（橫線=三家折算範圍、彩點=各家取值），標注「學術假設，非臨床劑量」——不再畫成三根確定性柱子 |
| 十.Fig8 | 評測圖標注「各任務指標語義與樣本量不同，不可跨行比大小；點估計無 CI（單次評測）」，Source Data 帶各任務 n |
| 十一 Source Data | `source_data/FigNa_*.csv` 逐 panel 綁定 + figures_manifest.json 記 data_sha256（openpyxl 不在 stdlib——以逐 panel CSV + manifest 對應 Nature 的 per-sheet 要求，xlsx 屬可選增強） |
| 十二 CVD 驗證 | `figure_qa.palette_cvd_report()`：Machado(2009) severity=1.0 線性 RGB 模擬 protan/deutan/tritan → CIE76 ΔE(Lab) 逐相鄰色對 + 灰度 ΔL，結果落盤 `figure_qa/qa_report.json`——「已驗證」從註釋變成可復核資產 |
| 十三.1 命名 | network_pharmacology 標題改「方劑—證候共現網絡研究（網絡藥理學前置）」+ 結果節「研究定位聲明」（無化合物/靶點/PPI/富集，不冒充） |
| 十三.2 | 非法論文類型 fail-fast（ValueError），不靜默退回默認類型 |
| 十三.3 | 摘要過強結論改分層表述：原文事實錨定條文編號；聚合統計/一致度/評測屬派生指標，錨定數據資產指紋 |
| 十七 QA 門禁 | run_qa()：XML 合法/title+desc+role/物理尺寸/字號下限/圖被正文引用/圖例在場/Source Data 在場/調色板 CVD——硬違例使 generate() 拋錯 |

## 如實差距（不宣傳為已有）

- **統計嚴謹性**：bootstrap CI、效應量（log OR/MI/信息增益）、置換檢驗、
  多次運行方差未實現——當前圖如實標注「點估計、無 CI」，不畫假誤差條；
- **投稿格式**：PDF/EPS/TIFF、DOCX/LaTeX、行號、Reporting Summary 未實現
  ——SVG 為權威產物，300/600dpi 位圖轉換屬外部渲染步驟；
- **字體凍結**：聲明 font-family 但未嵌字體文件（字體文件不可隨庫分發）；
  manifest 記錄聲明值，最終渲染環境須自行固定字體版本；
- **視覺回歸**：approved-vs-new 像素/結構 diff 未實現（輸出哈希穩定性已由
  字節級復現測試覆蓋）；
- **依賴策略**（評審十五「paper extra」建議的改進採納）：核心保持零依賴；
  `pip install .[paper]`（matplotlib/pandas/openpyxl/cairosvg）**暫不寫入
  pyproject**——聲明未被代碼使用的依賴是另一種「願望清單」；待多面板
  統計圖真的改用 matplotlib 渲染時與代碼同步加入。
