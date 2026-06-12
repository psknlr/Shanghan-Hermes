---
name: hermes.shanghan.taiyang
description: 太陽病六經規則：提綱、亞型、主方、禁忌、誤治與條文證據。
six_channel: 太陽病
release_level: silver
---

# 太陽病 Skill

## 提綱（原文直述）
> 太陽之為病，脈浮，頭項強痛而惡寒。
（SHL_SONGBEN_0001）

## 總括（chapter_level_induction，模型歸納）
太陽病以表證為主，提綱為脈浮、頭項強痛而惡寒；分中風（表虛有汗）與傷寒（表實無汗），另有蓄水、蓄血及誤治後諸變證，方以桂枝湯、麻黃湯為兩大主軸。

欲解時：從巳至未上

## 內部結構（亞型名稱為後世歸納，證據條文為原文）
- **太陽中風（表虛）**（錨定方：桂枝湯；證據條文：SHL_SONGBEN_0002、SHL_SONGBEN_0006、SHL_SONGBEN_0012）
- **太陽傷寒（表實）**（錨定方：麻黃湯；證據條文：SHL_SONGBEN_0003、SHL_SONGBEN_0004、SHL_SONGBEN_0005）
- **太陽蓄水**（錨定方：五苓散；證據條文：SHL_SONGBEN_0006、SHL_SONGBEN_0028、SHL_SONGBEN_0040）
- **太陽蓄血**（錨定方：桃核承氣湯、抵當湯、抵當丸；證據條文：SHL_SONGBEN_0106、SHL_SONGBEN_0124、SHL_SONGBEN_0125）
- **太陽變證（誤治壞病）**（錨定方：桂枝加附子湯、梔子豉湯、半夏瀉心湯、大陷胸湯；證據條文：SHL_SONGBEN_0030、SHL_SONGBEN_0112、SHL_SONGBEN_0124）

## 主要方劑（按條文頻次）
| 方劑 | 條文數 |
|---|---|
| 桂枝湯 | 21 |
| 小柴胡湯 | 13 |
| 麻黃湯 | 7 |
| 五苓散 | 6 |
| 調胃承氣湯 | 5 |
| 大陷胸湯 | 5 |
| 白虎加人參湯 | 4 |
| 四逆湯 | 3 |
| 梔子豉湯 | 3 |
| 大柴胡湯 | 3 |

## 禁忌條文
SHL_SONGBEN_0016、SHL_SONGBEN_0017、SHL_SONGBEN_0023、SHL_SONGBEN_0027、SHL_SONGBEN_0036、SHL_SONGBEN_0038、SHL_SONGBEN_0044、SHL_SONGBEN_0048、SHL_SONGBEN_0049、SHL_SONGBEN_0050

## 誤治相關條文
SHL_SONGBEN_0006、SHL_SONGBEN_0015、SHL_SONGBEN_0016、SHL_SONGBEN_0021、SHL_SONGBEN_0028、SHL_SONGBEN_0029、SHL_SONGBEN_0034、SHL_SONGBEN_0043、SHL_SONGBEN_0044、SHL_SONGBEN_0045、SHL_SONGBEN_0048、SHL_SONGBEN_0049

## 核心原則

- 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
- 合併規則不能覆蓋初始條文規則（本 Skill 中所有規則均保留 supporting_initial_rules / supporting_clauses 回鏈）。
- 輸出必須區分：原文直述（A）／版本異文（B）／注家解釋（C）／後世歸納（D）／模型推理（E）。
- 患者端禁止自動診斷、自動處方和劑量建議。
