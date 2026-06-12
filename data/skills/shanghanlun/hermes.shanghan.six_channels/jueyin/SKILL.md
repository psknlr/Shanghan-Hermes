---
name: hermes.shanghan.jueyin
description: 厥陰病六經規則：提綱、亞型、主方、禁忌、誤治與條文證據。
six_channel: 厥陰病
release_level: silver
---

# 厥陰病 Skill

## 提綱（原文直述）
> 厥陰之為病，消渴，氣上撞心，心中疼熱，飢而不欲食，食則吐蚘。下之利不止。
（SHL_SONGBEN_0326）

## 總括（chapter_level_induction，模型歸納）
厥陰病提綱為消渴、氣上撞心、心中疼熱、飢而不欲食；寒熱錯雜，烏梅丸為代表；厥熱勝復判預後，熱利白頭翁湯，寒厥當歸四逆湯。

欲解時：從丑至卯上

## 內部結構（亞型名稱為後世歸納，證據條文為原文）
- **厥陰寒熱錯雜**（錨定方：烏梅丸、乾薑黃芩黃連人參湯、麻黃升麻湯；證據條文：SHL_SONGBEN_0326）
- **厥陰寒證**（錨定方：當歸四逆湯、吳茱萸湯；證據條文：SHL_SONGBEN_0351）
- **厥陰熱利**（錨定方：白頭翁湯；證據條文：SHL_SONGBEN_0371）
- **厥逆辨治**（錨定方：四逆湯、瓜蒂散；證據條文：SHL_SONGBEN_0326、SHL_SONGBEN_0327、SHL_SONGBEN_0328）

## 主要方劑（按條文頻次）
| 方劑 | 條文數 |
|---|---|
| 四逆湯 | 4 |
| 當歸四逆湯 | 2 |
| 白頭翁湯 | 2 |
| 黃芩湯 | 1 |
| 烏梅丸 | 1 |
| 白虎湯 | 1 |
| 當歸四逆加吳茱萸生薑湯 | 1 |
| 瓜蒂散 | 1 |
| 茯苓甘草湯 | 1 |
| 麻黃升麻湯 | 1 |

## 禁忌條文
SHL_SONGBEN_0330、SHL_SONGBEN_0347、SHL_SONGBEN_0364

## 誤治相關條文
SHL_SONGBEN_0326、SHL_SONGBEN_0330、SHL_SONGBEN_0335、SHL_SONGBEN_0347、SHL_SONGBEN_0349、SHL_SONGBEN_0357、SHL_SONGBEN_0359、SHL_SONGBEN_0362、SHL_SONGBEN_0380

## 核心原則

- 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
- 合併規則不能覆蓋初始條文規則（本 Skill 中所有規則均保留 supporting_initial_rules / supporting_clauses 回鏈）。
- 輸出必須區分：原文直述（A）／版本異文（B）／注家解釋（C）／後世歸納（D）／模型推理（E）。
- 患者端禁止自動診斷、自動處方和劑量建議。
