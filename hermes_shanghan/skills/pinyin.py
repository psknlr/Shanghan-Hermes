"""Deterministic romanization for formula names → skill directory slugs."""
from __future__ import annotations

import re

CHAR_PINYIN = {
    "桂": "gui", "枝": "zhi", "湯": "tang", "麻": "ma", "黃": "huang", "葛": "ge",
    "根": "gen", "加": "jia", "半": "ban", "夏": "xia", "芩": "qin", "連": "lian",
    "大": "da", "小": "xiao", "柴": "chai", "胡": "hu", "白": "bai", "虎": "hu",
    "人": "ren", "參": "shen", "調": "tiao", "胃": "wei", "承": "cheng", "氣": "qi",
    "四": "si", "逆": "ni", "真": "zhen", "武": "wu", "理": "li", "中": "zhong",
    "丸": "wan", "五": "wu", "苓": "ling", "散": "san", "豬": "zhu", "瀉": "xie",
    "心": "xin", "生": "sheng", "薑": "jiang", "甘": "gan", "草": "cao",
    "阿": "e", "膠": "jiao", "烏": "wu", "梅": "mei", "當": "dang", "歸": "gui",
    "吳": "wu", "茱": "zhu", "萸": "yu", "石": "shi", "膏": "gao", "知": "zhi",
    "母": "mu", "粳": "jing", "米": "mi", "附": "fu", "子": "zi", "乾": "gan",
    "芍": "shao", "藥": "yao", "蜀": "shu", "漆": "qi", "牡": "mu", "蠣": "li",
    "龍": "long", "骨": "gu", "救": "jiu", "茯": "fu", "朮": "zhu", "棗": "zao",
    "厚": "hou", "朴": "po", "杏": "xing", "仁": "ren", "陷": "xian", "胸": "xiong",
    "文": "wen", "蛤": "ge", "梔": "zhi", "豉": "chi", "檗": "bo", "皮": "pi",
    "茵": "yin", "陳": "chen", "蒿": "hao", "膚": "fu", "桔": "jie", "梗": "geng",
    "苦": "ku", "酒": "jiu", "通": "tong", "脈": "mai", "頭": "tou", "翁": "weng",
    "赤": "chi", "脂": "zhi", "禹": "yu", "餘": "yu", "糧": "liang", "旋": "xuan",
    "覆": "fu", "代": "dai", "赭": "zhe", "枳": "zhi", "實": "shi", "燒": "shao",
    "褌": "kun", "澤": "ze", "竹": "zhu", "葉": "ye", "麥": "mai", "門": "men",
    "冬": "dong", "升": "sheng", "細": "xi", "辛": "xin", "桃": "tao", "花": "hua",
    "土": "tu", "瓜": "gua", "蜜": "mi", "煎": "jian", "導": "dao", "膽": "dan",
    "汁": "zhi", "新": "xin", "雞": "ji", "屎": "shi", "十": "shi", "蒂": "di",
    "抵": "di", "芒": "mang", "消": "xiao", "硝": "xiao", "核": "he", "去": "qu",
    "青": "qing", "越": "yue", "婢": "bi", "一": "yi", "二": "er", "三": "san",
    "各": "ge", "兩": "liang", "建": "jian", "炙": "zhi", "及": "ji", "黑": "hei",
    "雄": "xiong", "豆": "dou", "連軺": "lianyao", "軺": "yao", "蔥": "cong",
    "豭": "jia", "鼠": "shu", "礬": "fan", "滑": "hua", "代赭": "daizhe",
    "鉛": "qian", "丹": "dan", "蜘": "zhi", "蛛": "zhu", "蛇": "she", "床": "chuang",
    "敗": "bai", "醬": "jiang", "薏": "yi", "苡": "yi", "葦": "wei", "莖": "jing",
}


def formula_slug(name: str) -> str:
    out = []
    for ch in name:
        out.append(CHAR_PINYIN.get(ch, ""))
    slug = "_".join(p for p in out if p)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "formula_" + str(abs(hash(name)) % 10_000)
    # compress: join syllables of the stem, keep tang/wan/san suffix separate
    parts = slug.split("_")
    if len(parts) >= 2 and parts[-1] in ("tang", "wan", "san"):
        return "".join(parts[:-1]) + "_" + parts[-1]
    return "".join(parts)
