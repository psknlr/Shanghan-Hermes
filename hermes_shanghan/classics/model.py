"""通用古籍 Passage/Span 身份模型（十五輪 P1-1）。

全庫文本的知識單元不再只是「路徑+文件名+標題」：

    Corpus（笈成全庫，archive_sha256 指紋）
    └─ Work（抽象著作：以折疊書名為 base，如「傷寒論」）
       └─ Witness（具體傳本 = 一個編目單元，如「傷寒論_宋本」；
                   版本字段來自 <book> 元數據）
          └─ File（卷冊文件，讀序由 ordered_files 決定）
             └─ Passage（章節切分段：以 ======標題====== 分段的扁平化正文）
                └─ Span（字符座標區間，可逐字重驗）

Passage ID 是 sha256 穩定 ID（``psg_<12hex>``），與進程無關、與掃描順序
無關——同一庫版本下永遠相同，可入論文穩定引用。《傷寒論》第 12 條是
Passage 的一個領域投影，不再是全平台唯一知識單元。

誠實邊界：影印頁對應、頁碼/行號座標、跨卷段識別需要底本掃描件對齊，
本層只保證「轉錄文本內」的字符級座標與逐字重驗。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..corpus import library as _lib
from ..textutil import fold_variants

RE_PASSAGE_ID = re.compile(r"psg_[0-9a-f]{12}")

# 朝代排序（時間有序檢索/首現候選用）；未知朝代排最後、標 unranked
_DYNASTY_SEQ = (
    "周", "秦", "西漢", "東漢", "漢", "三國", "西晉", "東晉", "晉",
    "南北朝", "南朝", "北朝", "隋", "唐", "五代", "北宋", "南宋", "宋",
    "金", "元", "明", "清", "民國", "日本", "朝鮮",
)
DYNASTY_ORDER: Dict[str, int] = {d: i for i, d in enumerate(_DYNASTY_SEQ)}
UNRANKED = len(_DYNASTY_SEQ) + 10


def stable_id(namespace: str, value: str) -> str:
    """跨進程穩定 ID：sha256（不用內置 hash()——帶進程隨機種子）。"""
    digest = hashlib.sha256(
        f"{namespace}\0{value}".encode("utf-8")).hexdigest()[:12]
    return f"{namespace}_{digest}"


def dynasty_rank(dynasty: str) -> int:
    d = (dynasty or "").strip()
    if d in DYNASTY_ORDER:
        return DYNASTY_ORDER[d]
    for name, rank in DYNASTY_ORDER.items():   # 「明末清初」→ 取最早匹配
        if name in d:
            return rank
    return UNRANKED


def work_base_title(title: str) -> str:
    """Witness 標題 → 抽象著作名：「傷寒論_宋本」→「傷寒論」。"""
    return fold_variants((title or "").split("_")[0].split("·")[0].strip())


@dataclass(frozen=True)
class Passage:
    passage_id: str
    work_id: str            # 編目單元 id（= Witness）
    file: str
    seq: int                # 文件內章節段序號（0 起）
    section: str            # 所屬章節標題（可為空）
    flat_text: str          # 扁平化正文：去行內空白、保留原字（未折疊）

    def locator(self) -> Dict:
        return {"passage_id": self.passage_id, "work_id": self.work_id,
                "file": self.file, "seq": self.seq, "section": self.section,
                "n_chars": len(self.flat_text)}


def passage_uid(work_id: str, file: str, seq: int) -> str:
    return stable_id("psg", f"{work_id}/{file}#{seq}")


def segment_file(text: str) -> List[Tuple[str, str]]:
    """把一個卷冊文件切成 (章節標題, 扁平化正文) 段。

    與全文檢索同一套切分（換行硬折行在段內展平——語料在句中折行），
    <book> 元數據塊先剔除。fold_variants 是 1:1 字符映射，故折疊文本
    座標與未折疊扁平文本座標一一對應。
    """
    text = _lib.RE_BOOK_META.sub("", text)
    section, buf = "", []
    segments: List[Tuple[str, str]] = []
    for line in text.splitlines():
        m = _lib.RE_HEADING.match(line.strip())
        if m:
            if buf:
                segments.append((section, "".join(buf)))
                buf = []
            section = m.group(2)
        else:
            buf.append("".join(line.split()))
    if buf:
        segments.append((section, "".join(buf)))
    return [(s, t) for s, t in segments if t]


def passages_of_unit(lib: "_lib.Library", unit: Dict) -> List[Passage]:
    """一個編目單元的全部 Passage（讀序穩定 → passage_id 穩定）。"""
    out: List[Passage] = []
    for name in unit["files"]:
        raw = (_lib.books_dir(lib.root) / unit["id"] / name).read_text(
            encoding="utf-8", errors="replace")
        for seq, (section, flat) in enumerate(segment_file(raw)):
            out.append(Passage(passage_uid(unit["id"], name, seq),
                               unit["id"], name, seq, section, flat))
    return out


class PassageIndex:
    """按需構建、按單元緩存的 Passage 視圖（純內存，不落盤）。"""

    def __init__(self, lib: "_lib.Library"):
        self.lib = lib
        self._cache: Dict[str, List[Passage]] = {}
        self._by_id: Dict[str, Passage] = {}

    def unit_passages(self, unit: Dict) -> List[Passage]:
        uid = unit["id"]
        if uid not in self._cache:
            ps = passages_of_unit(self.lib, unit)
            self._cache[uid] = ps
            for p in ps:
                self._by_id[p.passage_id] = p
        return self._cache[uid]

    def get(self, passage_id: str, work: str = "",
            max_scan_units: int = 400) -> Optional[Passage]:
        """按 passage_id 取段。帶 work 提示時只掃該單元；否則按編目序
        掃描至多 ``max_scan_units`` 個單元——掃描封頂如實返回 None，
        調用方須把「未在掃描範圍找到」與「不存在」區分開。"""
        if passage_id in self._by_id:
            return self._by_id[passage_id]
        units = self.lib.units
        if work:
            u = self.lib._resolve(work)
            units = [u] if u else []
        for unit in units[:max_scan_units]:
            for p in self.unit_passages(unit):
                if p.passage_id == passage_id:
                    return p
        return None
