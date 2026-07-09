"""方劑異名歸並（編輯性對照表 + 組成比對驗證）。

方名傳播計量的已聲明邊界：同方異名（桂枝湯/陽旦湯）不歸並會低估傳播。
本表為編輯性元數據（posthoc_induction），逐條注明依據與「是否同方」的
組成比對結論；**不可僅憑名稱合併**——凡組成/劑量存疑者標 not_mergeable
並給出理由。異名的跨書出現計量與正名分列，不混入正名計數。
"""
from __future__ import annotations

from typing import Dict, List

FORMULA_ALIASES: List[Dict] = [
    {"canonical": "桂枝湯", "alias": "陽旦湯",
     "source": "《傷寒論》第 30 條「證象陽旦」；《金匱·婦人產後》陽旦湯即桂枝湯（林億等注）",
     "same_formula": True,
     "composition_diff": "無（歷代主流意見同方；《千金》陽旦湯加黃芩屬另一系）",
     "evidence_grade": "後世訓釋（成注/林億校語），非宋本正名",
     "note": "《千金方》別有加黃芩之陽旦湯，合併時須區分兩系"},
    {"canonical": "理中丸", "alias": "人參湯",
     "source": "《金匱·胸痹》人參湯與理中丸藥味全同（人參/甘草/白朮/乾薑各三兩）",
     "same_formula": True,
     "composition_diff": "劑型不同（丸/湯），藥味與比例同",
     "evidence_grade": "組成比對可證（A 層事實）",
     "note": "丸湯異劑：主治語境不同（霍亂 vs 胸痹），計量合併須標注劑型"},
    {"canonical": "小柴胡湯", "alias": "柴胡湯",
     "source": "後世方書常以「柴胡湯」簡稱小柴胡湯",
     "same_formula": False,
     "composition_diff": "「柴胡湯」為簡稱/類稱，亦可指大柴胡湯等柴胡類方",
     "evidence_grade": "名稱歧義，不可自動歸並",
     "note": "簡稱在無上下文時歧義，計量不合併（防止大/小柴胡混計）"},
    {"canonical": "四逆湯", "alias": "回逆湯",
     "source": "部分傳本/注本避諱或異寫作「回逆」",
     "same_formula": True,
     "composition_diff": "無",
     "evidence_grade": "版本異寫",
     "note": "出現頻次極低，計量影響可忽略"},
    {"canonical": "調胃承氣湯", "alias": "小承氣湯（誤稱）",
     "source": "後世文獻偶有承氣三方互誤",
     "same_formula": False,
     "composition_diff": "三承氣組成不同（芒硝/枳實/厚朴之有無）",
     "evidence_grade": "不可合併（組成不同）",
     "note": "列入以警示：承氣類名稱相近但組成證治俱異"},
]


def alias_names() -> List[str]:
    """可安全參與計量的異名（same_formula=True 且無歧義）。"""
    return [a["alias"] for a in FORMULA_ALIASES
            if a["same_formula"] and "（" not in a["alias"]]


def aliases_for(formula: str) -> List[Dict]:
    return [a for a in FORMULA_ALIASES if a["canonical"] == formula]
