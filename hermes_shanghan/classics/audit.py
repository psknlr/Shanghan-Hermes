"""全庫驗收審計（十五輪 六）：把「803 部大概沒問題」變成可查證的報告。

    python3 -m hermes_shanghan library audit [--sample N]

逐單元讀取正文，統計：解析成功/空文本/編碼異常（替換字符）/缺書名/
缺作者/缺朝代/缺分類/多層嵌套/重複文本（sha256）/目錄識別率/不可讀
文件清單/最大文件/最深層級/庫指紋。``--sample N`` 另生成分層抽樣金標準
清單（朝代×分類 分層，確定性抽取），供人工復核。
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ..corpus import library as _libmod


def acceptance_report(root: Optional[Path] = None, sample: int = 0) -> Dict:
    root = _libmod.library_root(root)
    if not _libmod.is_available(root):
        return {"available": False,
                "hint": "python3 -m hermes_shanghan library fetch"}
    cat = _libmod.load_catalog(root)
    units = cat["units"]
    books = _libmod.books_dir(root)

    n_empty = n_encoding = n_parsed = 0
    unreadable: List[str] = []
    text_hashes: Dict[str, List[str]] = {}
    largest = {"unit": "", "file": "", "bytes": 0}
    toc_units = 0
    for u in units:
        text_all = []
        for name in u["files"]:
            p = books / u["id"] / name
            try:
                t = p.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                n_encoding += 1
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                unreadable.append(f"{u['id']}/{name}")
                continue
            if "�" in t:
                n_encoding += 1
            text_all.append(t)
            size = p.stat().st_size
            if size > largest["bytes"]:
                largest = {"unit": u["id"], "file": name, "bytes": size}
        body = _libmod.RE_BOOK_META.sub("", "\n".join(text_all)).strip()
        if u["files"]:
            n_parsed += 1
        if not body:
            n_empty += 1
        else:
            h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            text_hashes.setdefault(h, []).append(u["id"])
        if any(_libmod.RE_HEADING.match(ln.strip())
               for t in text_all for ln in t.splitlines()):
            toc_units += 1

    duplicates = {h: ids for h, ids in text_hashes.items() if len(ids) > 1}
    depth = Counter(u.get("depth", u["id"].count("/") + 1) for u in units)
    report = {
        "available": True,
        "library_fingerprint": cat.get("archive_sha256", ""),
        "n_books": cat["n_books"], "n_units": cat["n_units"],
        "n_parsed": n_parsed, "n_empty_text": n_empty,
        "n_encoding_anomalies": n_encoding,
        "n_missing_title": sum(1 for u in units
                               if u["title"] == u["id"].split("/")[-1]
                               and not u["files"]),
        "n_missing_author": sum(1 for u in units if not u["author"]),
        "n_missing_dynasty": sum(1 for u in units if not u["dynasty"]),
        "n_missing_category": sum(1 for u in units if not u["category"]),
        "depth_histogram": dict(sorted(depth.items())),
        "max_depth": cat.get("max_depth", max(depth) if depth else 0),
        "n_duplicate_text_groups": len(duplicates),
        "duplicate_text_groups": dict(list(duplicates.items())[:20]),
        "toc_recognition_rate": round(toc_units / max(1, len(units)), 3),
        "unreadable_files": unreadable[:50],
        "largest_file": largest,
    }
    if sample > 0:
        report["gold_sample"] = stratified_sample(units, sample)
    return report


def stratified_sample(units: List[Dict], n: int) -> List[Dict]:
    """朝代×分類 分層確定性抽樣（sha256 排序，無隨機種子依賴）。"""
    tops = [u for u in units if not u["parent"]]
    strata: Dict[str, List[Dict]] = {}
    for u in tops:
        key = f"{u['dynasty'] or '無朝代'}|{u['category'] or '無分類'}"
        strata.setdefault(key, []).append(u)
    for members in strata.values():
        members.sort(key=lambda u: hashlib.sha256(
            u["id"].encode("utf-8")).hexdigest())
    out: List[Dict] = []
    while len(out) < min(n, len(tops)):
        progressed = False
        for key in sorted(strata):
            if strata[key] and len(out) < n:
                u = strata[key].pop(0)
                out.append({"id": u["id"], "title": u["title"],
                            "dynasty": u["dynasty"], "category": u["category"],
                            "n_files": len(u["files"]),
                            "stratum": key,
                            "check": "人工復核：書名/作者/朝代/章節切分/"
                                     "正文完整性/編碼"})
                progressed = True
        if not progressed:
            break
    return out
