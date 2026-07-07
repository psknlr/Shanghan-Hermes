"""Evaluation runner — three suites + matcher ablations, persisted under
data/shanghan/eval/ as deterministic JSON (no timestamps, so the artifacts
join the byte-reproducibility guarantee).

Suites
  cloze      遮方預測（LOCO 自監督，Ithaca 式遮蔽但遮的是臨床決策）
  cases      醫案回放（經方實驗錄，百年名醫處方作 ground truth）
  grounding  證據接地率（引用核驗指標，衡量任何後端的幻覺引用）

Ablations re-run the cloze suite with each matcher feature disabled so every
scoring component's contribution is a measured number, not a claim.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from .. import config
from .cases import CaseBenchmark
from .cloze import ClozeBenchmark
from .grounding import GroundingBenchmark, build_question_bank

EVAL_DIR_NAME = "eval"


def _eval_dir():
    d = config.SHANGHAN_DIR / EVAL_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(name: str, payload: Dict) -> str:
    p = _eval_dir() / name
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return str(p)


def run_suites(suites=("cloze", "cases", "grounding", "agent"),
               ablations: bool = False,
               limit: Optional[int] = None, verbose: bool = True) -> Dict:
    from ..orchestrator import Artifacts
    art = Artifacts()
    store = art.clause_store()
    summary: Dict = {"suites": {}}

    def log(msg):
        if verbose:
            print(msg)

    if "cloze" in suites:
        log("▶ 遮方預測基準（LOCO）…")
        bench = ClozeBenchmark(art.clauses, art.initial_rules)
        res = bench.run(limit=limit)
        res_slim = {k: v for k, v in res.items() if k != "records"}
        _write("cloze_results.json", res)
        summary["suites"]["cloze"] = res_slim
        if ablations:
            abl: Dict = {}
            for flag in ("use_outline_boost", "use_near_match"):
                b = ClozeBenchmark(art.clauses, art.initial_rules,
                                   **{flag: False})
                r = b.run(limit=limit)
                abl[f"without_{flag[4:]}"] = r["metrics"]["attainable"]
                log(f"  · ablation {flag}=False done")
            abl["full"] = res["metrics"]["attainable"]
            _write("cloze_ablations.json", abl)
            summary["suites"]["cloze_ablations"] = abl

    if "cases" in suites:
        log("▶ 醫案回放基準（經方實驗錄）…")
        try:
            bench = CaseBenchmark(art.formula_rules, store)
            res = bench.run(limit=limit)
            _write("case_results.json", res)
            summary["suites"]["cases"] = {k: v for k, v in res.items()
                                          if k != "records"}
        except FileNotFoundError:
            log("  · 醫案語料缺失，跳過（已記錄）")
            summary["suites"]["cases"] = {"error": "case corpus not found"}

    if "grounding" in suites:
        log("▶ 證據接地率基準…")
        qs = build_question_bank(art.formula_rules, art.six_channel_rules)
        res = GroundingBenchmark().run(qs, limit=limit)
        _write("grounding_results.json", res)
        summary["suites"]["grounding"] = {k: v for k, v in res.items()
                                          if k != "records"}

    if "agent" in suites:
        log("▶ 智能體基準（路由/接地/鑒別覆蓋/安全）…")
        from .agent_bench import run_agent_benchmarks
        res = run_agent_benchmarks(limit=limit)
        _write("agent_bench_results.json", res)
        summary["suites"]["agent"] = {
            "headline": res["headline"],
            "benchmarks": {k: {"metrics": v["metrics"]}
                           for k, v in res["benchmarks"].items()}}

    _write("eval_summary.json", summary)
    return summary
