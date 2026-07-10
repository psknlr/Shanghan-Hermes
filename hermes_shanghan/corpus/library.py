"""JichengLibrary — 中醫笈成全庫接入與快速查閱層。

把 https://jicheng.tw 發佈的中醫古籍全庫（book-*.7z，約 69MB 壓縮 /
311MB 展開，800+ 部醫籍）納入 Hermes 的文獻旁證層（非經文層）：

- **自動獲取**：``fetch()`` 下載 → sha256 校驗 → 解壓（py7zr 或系統 7z，
  均缺失時給出明確安裝指引）→ 建目錄編目 + 全庫字符索引，全程冪等。
- **完整解析**：兼容全庫的所有版式——``<book>`` 元數據塊（書名/作者/朝代/
  年份/分類/品質/版本/參本/備考/作者描述/地域……全字段保留）、單檔書
  （正文在 index.txt）、多卷書（1.txt…n.txt、"2-15"、"2-0.3" 等卷-章-節
  混合編號）、嵌套子書（如《醫宗金鑑》15 部子書各帶自己的元數據）、
  menu.txt 導航頁（排除出正文）。
- **快速調用**：``Library`` 提供毫秒級編目檢索（書名/作者/朝代/分類，
  異體字折疊）、全文檢索（稀字倒排索引先剪枝候選書，再逐書驗證原文，
  返回帶書名/章節定位的摘錄）、章節目錄與按節閱讀。

全庫屬文獻旁證層：檢索結果標注出處（書名·作者·朝代·章節），但不進入
規則庫的證據閘門——經文層證據仍只認宋本條文。庫體積大，不隨倉庫分發，
``data/library/`` 已列入 .gitignore，配置完成後一條命令自動下載：

    python3 -m hermes_shanghan library fetch
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .. import config
from ..textutil import fold_variants

RE_BOOK_META = re.compile(r"<book>(.*?)</book>", re.S)
RE_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
# volume-chapter-section stems: "3", "2-15", "2-0.3" — each dash part may
# carry one decimal point (prefaces are numbered 0.1, 0.2, …)
RE_NUM_STEM = re.compile(r"^\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)*$")

BOOKS_SUBDIR = "books"
CATALOG_NAME = "catalog.json"
CHARINDEX_NAME = "charindex.json"


# ---------------------------------------------------------------------------
# Paths & availability
# ---------------------------------------------------------------------------
def library_root(root: Optional[Path] = None) -> Path:
    return Path(root) if root else config.LIBRARY_DIR


def books_dir(root: Optional[Path] = None) -> Path:
    return library_root(root) / BOOKS_SUBDIR


def is_available(root: Optional[Path] = None) -> bool:
    return (library_root(root) / CATALOG_NAME).exists()


def ensure_available(root: Optional[Path] = None, auto: Optional[bool] = None,
                     verbose: bool = True) -> bool:
    """Return True when the library is usable; optionally auto-fetch.

    ``auto=None`` defers to the ``HERMES_LIBRARY_AUTOFETCH`` env var, so a
    configured deployment can make the first lookup pull the corpus in.
    """
    if is_available(root):
        return True
    if auto is None:
        auto = os.environ.get("HERMES_LIBRARY_AUTOFETCH", "") in ("1", "true", "yes")
    if not auto:
        return False
    fetch(root=root, verbose=verbose)
    return is_available(root)


# ---------------------------------------------------------------------------
# Acquisition: download → verify → extract → index
# 供應鏈安全（十輪 六.4）：URL allowlist、強制 SHA-256、超時、大小上限、
# 成員枚舉、路徑穿越/symlink/設備文件拒絕、展開體積與文件數上限、
# 壓縮比上限、臨時目錄解壓 + 結構校驗後原子切換、全程 provenance 記錄。
# ---------------------------------------------------------------------------
DOWNLOAD_TIMEOUT_S = 30            # 連接/讀取超時
MAX_ARCHIVE_BYTES = 200 << 20      # 壓縮包上限 200MB（官方檔 ~69MB）
MAX_MEMBERS = 30_000               # 解壓文件數上限
MAX_EXTRACTED_BYTES = 2 << 30      # 展開體積上限 2GB（官方 ~311MB）
MAX_COMPRESSION_RATIO = 60         # 展開/壓縮比上限（zip-bomb 防護）
MIN_BOOK_DIRS = 10                 # 結構校驗：至少多少書目目錄才算合法庫


class SupplyChainError(RuntimeError):
    pass


def _download(url: str, dest: Path, verbose: bool = True) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-shanghan/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, \
                tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            if total > MAX_ARCHIVE_BYTES:
                raise SupplyChainError(
                    f"壓縮包聲明體積 {total} 超過上限 {MAX_ARCHIVE_BYTES}")
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if done > MAX_ARCHIVE_BYTES:
                    raise SupplyChainError(
                        f"下載超過上限 {MAX_ARCHIVE_BYTES} bytes，已中止")
                if verbose and total and done % (8 << 20) < (1 << 20):
                    print(f"  下載中 {done / (1 << 20):.0f}/{total / (1 << 20):.0f} MB",
                          file=sys.stderr)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_member_names(names: Iterable[str]) -> None:
    """解壓前成員名審查：拒絕絕對路徑與路徑穿越。"""
    for name in names:
        p = name.replace("\\", "/")
        if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
            raise SupplyChainError(f"壓縮包成員為絕對路徑：{name}")
        if ".." in p.split("/"):
            raise SupplyChainError(f"壓縮包成員含路徑穿越：{name}")


def validate_extracted_tree(target: Path) -> Dict:
    """解壓後樹審查：symlink/設備文件/越界/文件數/展開體積。
    返回統計（進 provenance）。"""
    base = target.resolve()
    n_files = 0
    total = 0
    for p in target.rglob("*"):
        if p.is_symlink():
            raise SupplyChainError(f"壓縮包含符號鏈接：{p.name}")
        if not p.resolve().is_relative_to(base):
            raise SupplyChainError(f"解壓越界：{p}")
        if p.is_file():
            import stat as _stat
            if not _stat.S_ISREG(p.stat().st_mode):     # regular files only
                raise SupplyChainError(f"非常規文件（設備/管道）：{p.name}")
            n_files += 1
            total += p.stat().st_size
            if n_files > MAX_MEMBERS:
                raise SupplyChainError(f"文件數超過上限 {MAX_MEMBERS}")
            if total > MAX_EXTRACTED_BYTES:
                raise SupplyChainError(f"展開體積超過上限 {MAX_EXTRACTED_BYTES}")
    return {"n_files": n_files, "extracted_bytes": total}


def _extract_7z(archive: Path, target: Path) -> str:
    """Extract with py7zr if importable (成員名先審查), else system 7z
    （解壓後樹審查兜底——兩條路徑都必須過 validate_extracted_tree）。"""
    try:
        import py7zr  # type: ignore
        with py7zr.SevenZipFile(str(archive)) as z:
            validate_member_names(f.filename for f in z.list())
            z.extractall(path=str(target))
        return "py7zr"
    except ImportError:
        pass
    for exe in ("7z", "7za", "7zr"):
        binary = shutil.which(exe)
        if binary:
            subprocess.run([binary, "x", "-y", f"-o{target}", str(archive)],
                           check=True, stdout=subprocess.DEVNULL)
            return exe
    raise RuntimeError(
        "無法解壓 7z 檔：請安裝 `pip install py7zr` 或系統 p7zip"
        "（Debian/Ubuntu: apt install p7zip-full；macOS: brew install p7zip）")


def resolve_source(url: Optional[str], sha256: str = "") -> Tuple[str, str]:
    """來源裁定（fail-closed）：默認 allowlist URL 用固定哈希；自定義 URL
    必須 (a) 顯式提供 sha256 且 (b) 設 HERMES_LIBRARY_ALLOW_CUSTOM=1。
    無哈希的來源一律拒絕——不存在「下載了再說」。"""
    url = url or config.LIBRARY_URL
    if url == config.LIBRARY_URL:
        return url, config.LIBRARY_SHA256
    if os.environ.get("HERMES_LIBRARY_ALLOW_CUSTOM") != "1":
        raise SupplyChainError(
            f"非默認庫源 {url} 被拒：自定義來源須設 "
            "HERMES_LIBRARY_ALLOW_CUSTOM=1 並顯式提供 sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        raise SupplyChainError("自定義庫源必須提供 64 位十六進制 SHA-256")
    return url, sha256


def fetch(url: Optional[str] = None, root: Optional[Path] = None,
          force: bool = False, keep_archive: bool = False,
          verbose: bool = True, sha256: str = "") -> Path:
    """Download + verify + extract + index the full library. Idempotent.

    供應鏈保證：所有來源必須有 SHA-256（見 resolve_source）；解壓進臨時
    目錄，成員/樹審查 + 結構校驗通過後才原子切換到 books/；全程寫
    provenance.json（url/哈希/體積/文件數/校驗清單/時間）。
    """
    url, pinned = resolve_source(url, sha256)
    root = library_root(root)
    if is_available(root) and not force:
        if verbose:
            print(f"全庫已就緒：{root}", file=sys.stderr)
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / url.rsplit("/", 1)[-1]

    if archive.exists() and _sha256_file(archive) != pinned:
        archive.unlink()
    if not archive.exists():
        if verbose:
            print(f"下載 {url} …", file=sys.stderr)
        _download(url, archive, verbose=verbose)
    digest = _sha256_file(archive)
    if digest != pinned:
        raise SupplyChainError(f"sha256 校驗失敗：{digest} ≠ {pinned}（{archive}）")

    # 臨時目錄解壓 → 審查 → 原子切換（books/ 不會出現半解壓狀態）
    staging = root / (BOOKS_SUBDIR + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if verbose:
        print("解壓中（臨時目錄+審查後原子切換）…", file=sys.stderr)
    try:
        extractor = _extract_7z(archive, staging)
        stats = validate_extracted_tree(staging)
        if stats["extracted_bytes"] > archive.stat().st_size * MAX_COMPRESSION_RATIO:
            raise SupplyChainError(
                f"壓縮比異常：展開 {stats['extracted_bytes']} / 壓縮 "
                f"{archive.stat().st_size} 超過 {MAX_COMPRESSION_RATIO}×")
        n_dirs = sum(1 for p in staging.iterdir() if p.is_dir())
        if n_dirs < MIN_BOOK_DIRS:
            raise SupplyChainError(
                f"結構校驗失敗：僅 {n_dirs} 個書目目錄（<{MIN_BOOK_DIRS}），"
                "不似合法全庫，已拒絕切換")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    target = books_dir(root)
    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)

    import time as _time
    (root / "provenance.json").write_text(json.dumps({
        "source_url": url, "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size, **stats,
        "extractor": extractor,
        "fetched_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validations": ["sha256_pinned", "member_names_or_tree",
                        "no_symlink_no_device", "size_and_count_caps",
                        f"compression_ratio<= {MAX_COMPRESSION_RATIO}",
                        "structure_min_dirs", "atomic_switch"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print("建編目與字符索引…", file=sys.stderr)
    catalog = build_catalog(root, archive_sha256=digest, source_url=url,
                            extractor=extractor)
    build_char_index(root, catalog)
    if not keep_archive:
        archive.unlink()
    if verbose:
        print(f"完成：{catalog['n_units']} 個文本單元 / "
              f"{catalog['n_books']} 部書 → {root}", file=sys.stderr)
    return root


# ---------------------------------------------------------------------------
# Parsing: metadata, reading order, headings
# ---------------------------------------------------------------------------
def parse_meta(index_text: str) -> Dict[str, str]:
    """Parse a <book> metadata block, keeping every field verbatim."""
    m = RE_BOOK_META.search(index_text)
    meta: Dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def _stem_key(stem: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in stem.split("-"))


def ordered_files(book_dir: Path) -> List[str]:
    """Reading order: index.txt, then numeric stems; menu.txt excluded."""
    names: List[str] = []
    if (book_dir / "index.txt").exists():
        names.append("index.txt")
    nums = sorted((_stem_key(p.stem), p.name) for p in book_dir.glob("*.txt")
                  if RE_NUM_STEM.match(p.stem))
    names.extend(n for _, n in nums)
    return names


def read_unit_text(unit_dir: Path) -> str:
    parts = [(unit_dir / n).read_text(encoding="utf-8", errors="replace")
             for n in ordered_files(unit_dir)]
    return "\n".join(parts)


def _unit_entry(unit_dir: Path, unit_id: str, root: Path,
                parent: str = "") -> Dict:
    index = unit_dir / "index.txt"
    meta = parse_meta(index.read_text(encoding="utf-8", errors="replace")) \
        if index.exists() else {}
    files = ordered_files(unit_dir)
    n_chars = sum((unit_dir / n).stat().st_size for n in files) // 3  # ≈UTF-8 CJK
    return {
        "id": unit_id,
        "title": meta.get("書名", unit_dir.name),
        "author": meta.get("作者", ""),
        "dynasty": meta.get("朝代", "").strip(),
        "year": meta.get("年份", ""),
        "category": meta.get("分類", "").strip(),
        "quality": meta.get("品質", ""),
        "edition": meta.get("版本", ""),
        "parent": parent,
        "extra": {k: v for k, v in sorted(meta.items())
                  if k not in ("書名", "作者", "朝代", "年份", "分類",
                               "品質", "版本")},
        "files": files,
        "approx_chars": n_chars,
    }


def build_catalog(root: Optional[Path] = None, archive_sha256: str = "",
                  source_url: str = "", extractor: str = "") -> Dict:
    """Walk the extracted flat layout into one entry per text unit.

    A *unit* is a directory holding readable text: a top-level book, or a
    nested sub-book (e.g. 醫宗金鑑/訂正仲景全書傷寒論註) with its own
    metadata. Sub-books inherit missing 分類/朝代 from their parent.
    """
    root = library_root(root)
    base = books_dir(root)
    units: List[Dict] = []
    for book in sorted(p for p in base.iterdir() if p.is_dir()):
        entry = _unit_entry(book, book.name, root)
        subs = sorted(c for c in book.iterdir() if c.is_dir())
        entry["sub_books"] = [f"{book.name}/{c.name}" for c in subs]
        units.append(entry)
        for c in subs:
            sub = _unit_entry(c, f"{book.name}/{c.name}", root,
                              parent=book.name)
            sub["sub_books"] = []
            for field in ("category", "dynasty", "author"):
                sub[field] = sub[field] or entry[field]
            units.append(sub)
    units.sort(key=lambda u: u["id"])
    from collections import Counter
    cats = Counter(u["category"] for u in units if not u["parent"])
    catalog = {
        "source_url": source_url,
        "archive_sha256": archive_sha256,
        "extractor": extractor,
        "n_books": sum(1 for u in units if not u["parent"]),
        "n_units": len(units),
        "categories": dict(sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))),
        "units": units,
    }
    (root / CATALOG_NAME).write_text(
        json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    return catalog


def build_char_index(root: Optional[Path] = None,
                     catalog: Optional[Dict] = None) -> Dict:
    """Character inverted index: char → sorted unit ordinals (exact).

    Full-text search intersects the posting lists of a query's rarest
    characters, so the candidate set provably contains every true match —
    pure stdlib, no external search engine. Every CJK character is indexed
    (the library's big compilations make even niche characters appear in
    hundreds of units, so a df cutoff would blind the index).
    """
    root = library_root(root)
    catalog = catalog or load_catalog(root)
    postings: Dict[str, List[int]] = {}
    for i, u in enumerate(catalog["units"]):
        if not u["files"]:
            continue
        chars = set(fold_variants(read_unit_text(books_dir(root) / u["id"])))
        for ch in chars:
            if ch.isspace() or ord(ch) < 128:
                continue
            postings.setdefault(ch, []).append(i)
    index = {"chars": dict(sorted(postings.items()))}
    (root / CHARINDEX_NAME).write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    return index


def load_catalog(root: Optional[Path] = None) -> Dict:
    path = library_root(root) / CATALOG_NAME
    if not path.exists():
        raise FileNotFoundError(
            "全庫未就緒：請先運行 `python3 -m hermes_shanghan library fetch`")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fast consultation layer
# ---------------------------------------------------------------------------
class Library:
    """毫秒級編目檢索 + 稀字剪枝全文檢索 + 章節閱讀。"""

    def __init__(self, root: Optional[Path] = None):
        self.root = library_root(root)
        self.catalog = load_catalog(self.root)
        self.units: List[Dict] = self.catalog["units"]
        self._by_id = {u["id"]: u for u in self.units}
        self._title_index = [
            (fold_variants(u["title"]), fold_variants(u["author"]),
             u["dynasty"], u["category"], u) for u in self.units]
        self._charindex: Optional[Dict] = None

    # -- catalog search ---------------------------------------------------
    def search(self, query: str, category: str = "",
               limit: int = 20) -> List[Dict]:
        """Search 書名/作者/朝代/分類 (variant-folded). Ranked: title>author."""
        q = fold_variants(query.strip())
        hits: List[Tuple[int, Dict]] = []
        for title, author, dynasty, cat, u in self._title_index:
            if category and category not in cat:
                continue
            if q and q in title:
                score = 3 if not u["parent"] else 2
            elif q and (q in author or q == dynasty or q in cat):
                score = 1
            elif not q:
                score = 1
            else:
                continue
            hits.append((score, u))
        hits.sort(key=lambda h: (-h[0], -h[1]["approx_chars"], h[1]["id"]))
        return [self._brief(u) for _, u in hits[:limit]]

    def categories(self) -> Dict[str, int]:
        return self.catalog["categories"]

    def info(self, book_id: str) -> Optional[Dict]:
        u = self._resolve(book_id)
        return dict(u) if u else None

    # -- reading ----------------------------------------------------------
    def toc(self, book_id: str) -> List[Dict]:
        """Section headings (======…======) in reading order."""
        u = self._resolve(book_id)
        if u is None:
            return []
        out = []
        for name in u["files"]:
            text = (books_dir(self.root) / u["id"] / name).read_text(
                encoding="utf-8", errors="replace")
            for line in text.splitlines():
                m = RE_HEADING.match(line.strip())
                if m:
                    out.append({"level": 7 - len(m.group(1)),
                                "title": m.group(2), "file": name})
        return out

    def read(self, book_id: str, section: str = "",
             max_chars: int = 4000) -> Dict:
        """Read a book (or one section of it, matched on the heading)."""
        u = self._resolve(book_id)
        if u is None:
            return {"error": f"全庫查無此書：{book_id}"}
        text = RE_BOOK_META.sub("", read_unit_text(books_dir(self.root) / u["id"]))
        if section:
            sec = fold_variants(section)
            lines = text.splitlines()
            start = next((i for i, ln in enumerate(lines)
                          if (m := RE_HEADING.match(ln.strip()))
                          and sec in fold_variants(m.group(2))), None)
            if start is None:
                return {"error": f"《{u['title']}》查無章節：{section}",
                        "toc": [t["title"] for t in self.toc(book_id)][:40]}
            level = len(RE_HEADING.match(lines[start].strip()).group(1))
            end = next((j for j in range(start + 1, len(lines))
                        if (m := RE_HEADING.match(lines[j].strip()))
                        and len(m.group(1)) >= level), len(lines))
            text = "\n".join(lines[start:end])
        truncated = len(text) > max_chars
        return {"book": self._brief(u), "section": section,
                "text": text[:max_chars], "truncated": truncated,
                "total_chars": len(text)}

    # -- full-text search ---------------------------------------------------
    def grep(self, query: str, category: str = "", limit: int = 12,
             per_book: int = 3, max_scan: int = 200) -> Dict:
        """Verbatim full-text search across the whole library.

        1) prune candidate units via the char inverted index (exact — the
           candidate set contains every true match),
        2) stream-scan only those units, tracking the enclosing section,
        3) return locator-stamped excerpts（書·章節·摘錄）.

        ``scan_capped`` is True when candidates exceeded ``max_scan`` before
        ``limit`` hits were found — 0 hits then means "not in the first
        max_scan candidate books", not "absent from the library".
        """
        q = fold_variants("".join(query.split()))
        if len(q) < 2:
            return {"error": "全文檢索詞至少 2 字"}
        cands = self._candidates(q)
        matches: List[Dict] = []
        scanned = 0
        capped = False
        for i in cands:
            u = self.units[i]
            if category and category not in u["category"]:
                continue
            if len(matches) >= limit:
                break
            if scanned >= max_scan:
                capped = True
                break
            scanned += 1
            found = self._scan_unit(u, q, per_book)
            matches.extend(found[:max(0, limit - len(matches))])
        return {"query": query, "n_hits": len(matches),
                "n_books_scanned": scanned, "scan_capped": capped,
                "n_candidate_books": len(cands), "hits": matches}

    # -- internals ----------------------------------------------------------
    def _resolve(self, book_id: str) -> Optional[Dict]:
        u = self._by_id.get(book_id)
        if u is None:
            q = fold_variants(book_id.strip().strip("《》"))
            u = next((x for x in self.units
                      if fold_variants(x["title"]) == q
                      or x["id"].split("/")[-1] == q), None)
            if u is None:
                u = next((x for x in self.units
                          if q and q in fold_variants(x["title"])), None)
        return u

    def _load_charindex(self) -> Dict:
        if self._charindex is None:
            path = self.root / CHARINDEX_NAME
            self._charindex = json.loads(path.read_text(encoding="utf-8")) \
                if path.exists() else {"chars": {}}
        return self._charindex

    def _candidates(self, q: str) -> List[int]:
        chars = self._load_charindex()["chars"]
        if not chars:                           # no index → scan everything
            return list(range(len(self.units)))
        cjk = [ch for ch in set(q) if ord(ch) >= 128]
        if any(ch not in chars for ch in cjk):
            return []                           # char absent from library
        rare = sorted((len(chars[ch]), ch) for ch in cjk)
        if not rare:                            # ASCII-only query
            return list(range(len(self.units)))
        result = set(chars[rare[0][1]])
        for _, ch in rare[1:4]:
            result &= set(chars[ch])
        return sorted(result)

    def _scan_unit(self, u: Dict, q: str, per_book: int) -> List[Dict]:
        out: List[Dict] = []
        for name in u["files"]:
            if len(out) >= per_book:
                break
            text = (books_dir(self.root) / u["id"] / name).read_text(
                encoding="utf-8", errors="replace")
            # segment the file at heading lines so every excerpt carries
            # its enclosing 章節; hard line-wraps inside a segment are
            # unwrapped before matching (the corpus wraps mid-sentence)
            section, buf = "", []
            segments: List[Tuple[str, str]] = []
            for line in text.splitlines():
                m = RE_HEADING.match(line.strip())
                if m:
                    if buf:
                        segments.append((section, "".join(buf)))
                        buf = []
                    section = m.group(2)
                else:
                    buf.append("".join(line.split()))
            if buf:
                segments.append((section, "".join(buf)))
            for section, flat in segments:
                pos = fold_variants(flat).find(q)
                if pos < 0:
                    continue
                lo, hi = max(0, pos - 40), pos + len(q) + 40
                out.append({"book_id": u["id"], "title": u["title"],
                            "author": u["author"], "dynasty": u["dynasty"],
                            "category": u["category"], "file": name,
                            "section": section,
                            "excerpt": flat[lo:hi]})
                if len(out) >= per_book:
                    break
        return out

    @staticmethod
    def _brief(u: Dict) -> Dict:
        return {k: u[k] for k in ("id", "title", "author", "dynasty", "year",
                                  "category", "quality", "approx_chars")} | \
            {"n_files": len(u["files"]), "sub_books": u["sub_books"]}


def status(root: Optional[Path] = None) -> Dict:
    root = library_root(root)
    if not is_available(root):
        return {"available": False, "root": str(root),
                "hint": "python3 -m hermes_shanghan library fetch"}
    cat = load_catalog(root)
    return {"available": True, "root": str(root),
            "n_books": cat["n_books"], "n_units": cat["n_units"],
            "archive_sha256": cat.get("archive_sha256", ""),
            "source_url": cat.get("source_url", ""),
            "categories": cat["categories"]}
