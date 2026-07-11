"""Text utilities for classical Chinese processing.

Includes markup stripping for the corpus wiki-format, a char-bigram
tokenizer used by BM25 and similarity scoring, and a small
simplified→traditional mapping so that user queries typed in simplified
Chinese still hit the traditional-character corpus.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Corpus markup
# ---------------------------------------------------------------------------
RE_J_NOTE = re.compile(r"<j>(.*?)</j>", re.S)          # inline collation note
RE_L_NOTE = re.compile(r"<l>(.*?)</l>", re.S)          # dose / processing note
RE_TAG = re.compile(r"</?[A-Za-z#][^>]*>")             # any other tag
RE_BOLD = re.compile(r"\*\*(.*?)\*\*")
RE_WIKILINK = re.compile(r"\[\[[^\]]*\]\]")
RE_UNDERLINE = re.compile(r"__(.*?)__")


def strip_markup(text: str, keep_notes: bool = False) -> str:
    """Remove wiki markup; optionally keep <j> note text in brackets."""
    if keep_notes:
        text = RE_J_NOTE.sub(lambda m: "（注：" + m.group(1) + "）", text)
    else:
        text = RE_J_NOTE.sub("", text)
    text = RE_L_NOTE.sub(lambda m: "（" + m.group(1) + "）", text)
    text = RE_BOLD.sub(r"\1", text)
    text = RE_UNDERLINE.sub(r"\1", text)
    text = RE_WIKILINK.sub("", text)
    text = RE_TAG.sub("", text)
    text = text.replace("\\\\", "")
    return text.strip()


def extract_j_notes(text: str) -> List[str]:
    return [m.strip() for m in RE_J_NOTE.findall(text)]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sentence segmentation (classical punctuation)
# ---------------------------------------------------------------------------
SENT_SPLIT = re.compile(r"(?<=[。！？；])")
SUB_SPLIT = re.compile(r"[，、：]")


def split_sentences(text: str) -> List[str]:
    return [s for s in (p.strip() for p in SENT_SPLIT.split(text)) if s]


def split_subclauses(text: str) -> List[str]:
    out: List[str] = []
    for sent in split_sentences(text):
        for sub in SUB_SPLIT.split(sent):
            sub = sub.strip("。！？；，、： ")
            if sub:
                out.append(sub)
    return out


# ---------------------------------------------------------------------------
# Tokenization for retrieval: char unigrams + bigrams (CJK only)
# ---------------------------------------------------------------------------
RE_CJK = re.compile(r"[㐀-鿿]")


def cjk_chars(text: str) -> List[str]:
    return RE_CJK.findall(text)


def tokenize(text: str) -> List[str]:
    chars = cjk_chars(text)
    tokens = list(chars)
    tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


def bigram_set(text: str) -> set:
    chars = cjk_chars(text)
    if len(chars) < 2:
        return set(chars)
    return {a + b for a, b in zip(chars, chars[1:])}


def similarity(a: str, b: str) -> float:
    """Dice coefficient over character bigrams - robust for clause alignment."""
    sa, sb = bigram_set(a), bigram_set(b)
    if not sa or not sb:
        return 0.0
    return 2 * len(sa & sb) / (len(sa) + len(sb))


# ---------------------------------------------------------------------------
# Graph-variant folding (異體字校勘層)
# ---------------------------------------------------------------------------
# The Songben corpus writes several characters in variant glyphs that the
# lexicon records in canonical form (e.g. 「胸脇苦滿」 vs 詞典「胸脅苦滿」,
# 「心下痞鞕」 vs 「痞硬」). Folding maps variant → canonical so term matching
# and evidence containment work; 「逐字」 is preserved modulo 異體字 — the
# standard collation convention — and clause clean_text keeps original glyphs.
# Single-char translate preserves string length, so match offsets stay valid.
#（十九輪）幾→几：宋本作「項背強几几」，簡體輸入「几几」經 s2t 誤升為
# 「幾幾」而失配——兩側折疊統一到「几」，簡繁輸入皆可命中；真「幾」字
#（幾日等）兩側同折，比較仍一致。
_VARIANT_MAP = str.maketrans({"脇": "脅", "鞕": "硬", "欬": "咳", "濇": "澀",
                              "幾": "几"})


def fold_variants(text: str) -> str:
    return (text or "").translate(_VARIANT_MAP)


def contains_verbatim(haystack: str, needle: str) -> bool:
    """Verbatim containment ignoring whitespace and graph-variant glyphs."""
    norm = lambda s: fold_variants(re.sub(r"\s+", "", s))
    return norm(needle) in norm(haystack)


def find_all(text: str, term: str) -> List[int]:
    out, start = [], 0
    while True:
        i = text.find(term, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


# ---------------------------------------------------------------------------
# Simplified → Traditional normalization (domain character map)
# ---------------------------------------------------------------------------
# Only characters that occur in Shanghan Lun domain vocabulary are mapped.
_S2T_PAIRS = (
    "恶惡 风風 发發 热熱 无無 呕嘔 干乾 烦煩 谵譫 语語 满滿 胁脅 头頭 项項 强強 "
    "脉脈 紧緊 缓緩 数數 细細 汤湯 黄黃 龙龍 参參 调調 气氣 阳陽 阴陰 泻瀉 连連 "
    "胶膠 乌烏 当當 归歸 猪豬 姜薑 与與 后後 误誤 证證 经經 体體 实實 虚虛 师師 "
    "沉沈 里裏 表表 临臨 床牀 药藥 剂劑 医醫 论論 伤傷 寒寒 杂雜 张張 机機 "
    "条條 辨辨 删刪 难難 红紅 紫紫 觉覺 转轉 输輸 阐闡 释釋 减減 协協 闷悶 呃呃 "
    "哕噦 衄衄 疼疼 痛痛 痞痞 厥厥 利利 秘祕 结結 胸胸 烧燒 针針 灸灸 熏熏 熨熨 "
    "悸悸 眩眩 冒冒 渴渴 饮飲 食食 谷穀 溏溏 脓膿 血血 尿尿 溲溲 汗汗 吐吐 下下 "
    "温溫 清清 补補 救救 逆逆 传傳 变變 愈癒 死死 生生 长長 短短 迟遲 疾疾 滑滑 "
    "涩澀 弦弦 微微 弱弱 洪洪 大大 芤芤 革革 动動 促促 代代 牢牢 濡濡 散散 伏伏 "
    "国國 学學 书書 读讀 万萬 亿億 历歷 历曆 复復 复複 见見 观觀 视視 听聽 闻聞 "
    "问問 诊診 络絡 腑腑 脏臟 肾腎 脾脾 肺肺 肝肝 胆膽 肠腸 胃胃 膀膀 胱胱 焦焦 "
    "营營 卫衛 荣榮 个個 们們 这這 对對 时時 将將 应應 须須 须鬚 单單 双雙 几幾 "
    "儿兒 处處 内內 两兩 仅僅 从從 众眾 优優 会會 伞傘 备備 储儲 兰蘭 关關 兴興 "
    "兹茲 养養 兼兼 决決 况況 净淨 准準 凉涼 凄淒 减减 凑湊 亏虧 云雲 互互 "
    "井井 亚亞 些些 交交 亥亥 亦亦 产產 享享 亲親 仁仁 仆僕 介介 仍仍 仓倉 "
    "仔仔 他他 仗仗 付付 仙仙 代代 令令 以以 仪儀 件件 价價 任任 份份 仿彷 "
    "瓜瓜 瓣瓣 甘甘 甚甚 甜甜 椒椒 茱茱 萸萸 苓苓 术朮 桂桂 枝枝 芍芍 "
    "草草 麻麻 杏杏 仁仁 石石 膏膏 知知 母母 粳粳 米米 葛葛 根根 柴柴 胡胡 "
    "芩芩 夏夏 枣棗 蛔蚘 栀梔 豉豉 翁翁 柏柏 皮皮 茵茵 陈陳 蒿蒿 泽澤 "
    "胆膽 矾礬 蜜蜜 煎煎 导導 赤赤 滑滑 蛤蛤 文文 灶灶 中中 "
    "极極 标標 准准 确確 诉訴 销銷 镇鎮 错錯 钱錢 铁鐵 铃鈴 银銀 镜鏡 闭閉 "
    "问问 间間 闰閏 闲閒 闹鬧 阅閱 阵陣 阶階 际際 陆陸 陈陈 降降 限限 院院 "
    "页頁 顶頂 顷頃 项项 顺順 颂頌 预預 领領 频頻 颗顆 题題 颜顏 额額 风风 "
    "饥飢 饱飽 饮饮 饴飴 饼餅 馆館 首首 香香 马馬 驱驅 验驗 骨骨 高高 鬼鬼 "
    "鱼魚 鸟鳥 鸡雞 麦麥 麻麻 黑黑 默默 鼓鼓 鼻鼻 齐齊 齿齒 龈齦 龟龜 "
)
S2T = {}
for pair in _S2T_PAIRS.split():
    if len(pair) == 2 and pair[0] != pair[1]:
        S2T[pair[0]] = pair[1]


def s2t(text: str) -> str:
    """Best-effort simplified→traditional for domain queries."""
    return "".join(S2T.get(ch, ch) for ch in text)


# 繁→簡（顯示層，十八輪）：由 S2T 反轉派生——反向映射天然無歧義
# （歷/曆→历、復/複→复 多對一在此方向是安全的），另補少量高頻字。
_T2S_EXTRA = {"裡": "里", "係": "系", "堅": "坚", "髒": "脏", "灣": "湾",
              "億": "亿", "點": "点", "團": "团", "戰": "战", "邊": "边",
              "隨": "随", "證": "证", "與": "与", "為": "为", "後": "后",
              # 界面常用字（十八輪簡體顯示補充）
              "總": "总", "覽": "览", "檢": "检", "檔": "档", "閉": "闭",
              "環": "环", "庫": "库", "於": "于", "運": "运", "鑒": "鉴",
              "練": "练", "習": "习", "題": "题", "擴": "扩", "濾": "滤",
              "譜": "谱", "關": "关", "讀": "读", "體": "体", "類": "类",
              "維": "维", "評": "评", "測": "测", "標": "标", "註": "注",
              "議": "议", "錄": "录", "選": "选", "擇": "择", "鍵": "键",
              "統": "统", "計": "计", "網": "网", "絡": "络", "圖": "图",
              "層": "层", "廣": "广", "節": "节", "術": "术", "語": "语",
              "識": "识", "別": "别", "來": "来", "現": "现", "詢": "询",
              "務": "务", "動": "动", "詞": "词", "書": "书", "籍": "籍",
              "義": "义", "釋": "释", "問": "问", "答": "答", "門": "门",
              "間": "间", "頁": "页", "顯": "显", "示": "示", "轉": "转",
              "換": "换", "載": "载", "續": "续", "鏈": "链", "驗": "验"}
T2S = {t: s for s, t in S2T.items() if s != t}
T2S.update(_T2S_EXTRA)


def t2s(text: str) -> str:
    """Best-effort traditional→simplified；僅供顯示層，原文以繁體為準。"""
    return "".join(T2S.get(ch, ch) for ch in text)


def normalize_query(text: str) -> str:
    return fold_variants(s2t(text.strip()))


def dedupe_keep_order(items: Iterable) -> List:
    seen, out = set(), []
    for it in items:
        key = it if isinstance(it, (str, int, tuple)) else repr(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
