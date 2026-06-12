---
name: hermes.shanghan.paper_writer
description: 論文寫作：方證規律挖掘/六經知識圖譜/誤治傳變研究等論文的自動生成。
---

# Paper Writer Skill

支持的論文類型：
1. 《傷寒論》方證規律挖掘
2. 《傷寒論》六經辨證知識圖譜
3. 《傷寒論》誤治傳變規則研究
4. 《傷寒論》方劑網絡藥理學前置研究
5. 《傷寒論》某方劑歷代注釋比較
6. 古籍數據挖掘與智能體方法學論文

自動生成模塊：Title / Abstract / Introduction / Methods / Results /
Discussion / Conclusion / Figures / Tables / References / Supplementary /
Cover Letter。

調用：`hermes-shanghan paper --type formula_pattern --topic 桂枝湯類方`
所有結果性陳述自動掛接規則 ID 與條文 ID（證據鏈）。

## 核心原則

- 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
- 合併規則不能覆蓋初始條文規則（本 Skill 中所有規則均保留 supporting_initial_rules / supporting_clauses 回鏈）。
- 輸出必須區分：原文直述（A）／版本異文（B）／注家解釋（C）／後世歸納（D）／模型推理（E）。
- 患者端禁止自動診斷、自動處方和劑量建議。
