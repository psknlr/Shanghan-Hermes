---
name: hermes.shanghan.clause_explainer
description: 條文解釋：按條文號回源原文，附實體標註、初始規則、條文關係、異文與注釋。
---

# 條文解釋 Skill

輸入條文號（1–398）或 clause_id，輸出：
1. 原文（A層，verbatim）+ 篇章 + 六經歸屬；
2. 實體標註：症狀/脈象/方劑/治法/禁忌/誤治/預後；
3. 本條抽取的 InitialRules（含審核等級）；
4. 條文關係：同方族/鑒別/誤治傳變/禁忌/傳變；
5. 版本異文（B層）與成無己注（C層）；
6. 模型解讀（E層，明確標註）。

調用：`hermes-shanghan explain-clause 12`

## 核心原則

- 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
- 合併規則不能覆蓋初始條文規則（本 Skill 中所有規則均保留 supporting_initial_rules / supporting_clauses 回鏈）。
- 輸出必須區分：原文直述（A）／版本異文（B）／注家解釋（C）／後世歸納（D）／模型推理（E）。
- 患者端禁止自動診斷、自動處方和劑量建議。
