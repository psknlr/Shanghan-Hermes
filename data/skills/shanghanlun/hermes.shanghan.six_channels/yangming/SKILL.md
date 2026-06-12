---
name: hermes.shanghan.yangming
description: 陽明病六經規則：提綱、亞型、主方、禁忌、誤治與條文證據。
six_channel: 陽明病
release_level: silver
---

# 陽明病 Skill

## 提綱（原文直述）
> 陽明之為病，胃家實是也。
（SHL_SONGBEN_0180）

## 總括（chapter_level_induction，模型歸納）
陽明病提綱為胃家實；熱在經者主以白虎輩清之，實在腑者以三承氣輩下之，兼有發黃、脾約諸證，辨潮熱、譫語、燥屎為下法之指徵。

欲解時：從申至戌上

## 內部結構（亞型名稱為後世歸納，證據條文為原文）
- **陽明熱證（經表之熱）**（錨定方：白虎湯、白虎加人參湯；證據條文：SHL_SONGBEN_0222）
- **陽明腑實**（錨定方：大承氣湯、小承氣湯、調胃承氣湯；證據條文：SHL_SONGBEN_0201、SHL_SONGBEN_0208、SHL_SONGBEN_0209）
- **陽明發黃**（錨定方：茵陳蒿湯、梔子檗皮湯、麻黃連軺赤小豆湯；證據條文：SHL_SONGBEN_0187、SHL_SONGBEN_0199、SHL_SONGBEN_0200）
- **脾約**（錨定方：麻子仁丸；證據條文：SHL_SONGBEN_0179）

## 主要方劑（按條文頻次）
| 方劑 | 條文數 |
|---|---|
| 大承氣湯 | 16 |
| 小承氣湯 | 6 |
| 調胃承氣湯 | 3 |
| 小柴胡湯 | 3 |
| 梔子豉湯 | 2 |
| 豬苓湯 | 2 |
| 麻黃湯 | 2 |
| 桂枝湯 | 2 |
| 茵陳蒿湯 | 2 |
| 抵當湯 | 2 |

## 禁忌條文
SHL_SONGBEN_0204、SHL_SONGBEN_0205、SHL_SONGBEN_0206、SHL_SONGBEN_0209、SHL_SONGBEN_0214、SHL_SONGBEN_0224、SHL_SONGBEN_0233、SHL_SONGBEN_0238、SHL_SONGBEN_0259

## 誤治相關條文
SHL_SONGBEN_0189、SHL_SONGBEN_0195、SHL_SONGBEN_0200、SHL_SONGBEN_0203、SHL_SONGBEN_0211、SHL_SONGBEN_0212、SHL_SONGBEN_0215、SHL_SONGBEN_0217、SHL_SONGBEN_0219、SHL_SONGBEN_0220、SHL_SONGBEN_0221、SHL_SONGBEN_0228

## 核心原則

- 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
- 合併規則不能覆蓋初始條文規則（本 Skill 中所有規則均保留 supporting_initial_rules / supporting_clauses 回鏈）。
- 輸出必須區分：原文直述（A）／版本異文（B）／注家解釋（C）／後世歸納（D）／模型推理（E）。
- 患者端禁止自動診斷、自動處方和劑量建議。
