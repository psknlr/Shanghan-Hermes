"""溯源資產構建器：一鍵產出全部 trace 層資產（字節級可復現）。

產物（``data/shanghan/trace/``）：

- ``id_registry.json``        統一知識標識註冊表
- ``citation_edges_agg.jsonl`` (著作, 條文) 級引文邊聚合（提交用緊湊形態）
- ``citation_relay_agg.jsonl`` (著作, 經由注本) 級轉引聚合
- ``citation_network.json``   科學計量網絡（共引/耦合/切片/突現/主路徑）
- ``citation_book_stats.json`` 逐書掃描統計 + 掃描參數 + 存疑標記樣例
- ``formula_mentions.json``   方名源流計量
- ``schools.json``            學派註冊表（回填一致度證據）
- ``claims.json``             結構化方證觀點庫

段落級全量邊（~4 萬條）體積大、可由掃描器在數秒內確定性重建，
故不作為提交資產；需要時用 ``trace scan-full --out`` 導出。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from ..schemas import read_jsonl, write_jsonl

TRACE_DIR_NAME = "trace"

_MATCHER = None
_CACHE: Dict[str, object] = {}


def trace_dir() -> Path:
    return getattr(config, "TRACE_DIR", config.SHANGHAN_DIR / TRACE_DIR_NAME)


def _clause_texts() -> Dict[str, str]:
    return {c["clause_id"]: c["clean_text"]
            for c in read_jsonl(config.CLAUSE_DIR / "clauses.jsonl")
            if c.get("text_type") in ("original_clause", "auxiliary_clause")}


def get_matcher():
    """條文回源匹配器單例（供工具、溯源鏈與現代文獻接口共用）。"""
    global _MATCHER
    if _MATCHER is None:
        from .quotation import ClauseIndex, ClauseMatcher
        _MATCHER = ClauseMatcher(ClauseIndex(_clause_texts()))
    return _MATCHER


def build_all(verbose: bool = False) -> Dict[str, int]:
    """構建並落盤全部 trace 資產，返回統計。"""
    from .claims import build_claims
    from .ids import build_registry
    from .quotation import formula_mentions, scan_corpus
    from .schools import build_school_registry
    from .scientometrics import aggregate_edges, aggregate_relay, build_network

    out = trace_dir()
    out.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}

    def _write_json(name: str, obj: Dict) -> None:
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    # 1. 統一 ID 註冊表
    registry = build_registry()
    _write_json("id_registry.json", registry)
    stats["works"] = registry["counts"]["works"]
    stats["formulas"] = registry["counts"]["formulas"]

    # 2. 引文掃描（段落級 → 聚合落盤）
    commentary_rules = read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")
    scan = scan_corpus(_clause_texts(), commentary_rules, verbose=verbose)
    agg = aggregate_edges(scan["edges"])
    relay = aggregate_relay(scan["edges"])
    write_jsonl(out / "citation_edges_agg.jsonl", agg)
    write_jsonl(out / "citation_relay_agg.jsonl", relay)
    _write_json("citation_book_stats.json",
                {"params": scan["params"], "books": scan["book_stats"]})
    stats["citation_edges"] = len(scan["edges"])
    stats["citation_pairs"] = len(agg)

    # 3. 科學計量網絡
    network = build_network(scan["edges"], scan["book_stats"])
    _write_json("citation_network.json", network)
    stats["cited_clauses"] = network["overview"]["n_clauses_cited"]

    # 4. 方名源流（含可安全歸並的異名，如陽旦湯；異名與正名分列計量）
    from .aliases import alias_names
    formula_rules = read_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")
    names = [r.get("formula", "") for r in formula_rules] + alias_names()
    mentions = formula_mentions(names, verbose=verbose)
    _write_json("formula_mentions.json", mentions)
    stats["formulas_mentioned"] = mentions["n_formulas_mentioned"]

    # 5. 學派註冊表
    schools = build_school_registry()
    _write_json("schools.json", schools)
    stats["schools"] = schools["n_schools"]

    # 6. 方證觀點庫（注本朝代元數據來自註冊表）
    book_meta = {}
    for w in registry["works"]:
        book_meta[w["book_dir"]] = w
        book_meta[w["title"]] = w
    claims = build_claims(commentary_books_meta=book_meta,
                          commentator_school=schools["commentator_school"])
    _write_json("claims.json", claims)
    stats["claims"] = claims["n_claims"]

    if verbose:
        print(f"    [trace] 資產已落盤 {out}: {stats}")
    _CACHE.clear()
    return stats


# ---------------------------------------------------------------------------
# 讀取器（含缺失時自動構建）
# ---------------------------------------------------------------------------
def ensure_built() -> None:
    if not (trace_dir() / "citation_network.json").exists():
        build_all()


def _load_json(name: str) -> Dict:
    if name not in _CACHE:
        ensure_built()
        _CACHE[name] = json.loads((trace_dir() / name).read_text(encoding="utf-8"))
    return _CACHE[name]  # type: ignore[return-value]


def load_registry() -> Dict:
    return _load_json("id_registry.json")


def load_network() -> Dict:
    return _load_json("citation_network.json")


def load_book_stats() -> Dict:
    return _load_json("citation_book_stats.json")


def load_schools() -> Dict:
    return _load_json("schools.json")


def load_claims() -> Dict:
    return _load_json("claims.json")


def load_formula_mentions() -> Dict:
    return _load_json("formula_mentions.json")


def load_agg_edges() -> List[Dict]:
    key = "citation_edges_agg.jsonl"
    if key not in _CACHE:
        ensure_built()
        _CACHE[key] = read_jsonl(trace_dir() / key)
    return _CACHE[key]  # type: ignore[return-value]


def load_relay_edges() -> List[Dict]:
    key = "citation_relay_agg.jsonl"
    if key not in _CACHE:
        ensure_built()
        _CACHE[key] = read_jsonl(trace_dir() / key)
    return _CACHE[key]  # type: ignore[return-value]
