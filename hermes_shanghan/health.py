"""就緒探針（九輪 P0-7：拒絕「假健康」）。

pip wheel 只含代碼不含語料（見 pyproject 說明）；獨立安裝後進程能啟動、
API 能返回 200，但規則庫為 0——這是危險的假健康。因此把健康檢查拆分：

  livez()   進程是否活着（永遠廉價，不觸碰數據）
  readyz()  數據能力是否完整：manifest 哈希可讀、398 條核心編號完整、
            規則庫非空、tool_specs 與運行時一致、溯源資產在位

並提供 ``assert_ready()``：ToolRegistry 首次構建前調用，資產缺失時
**響亮報錯**（MissingAssetsError，附修復指引），而不是靜默空運行。
``HERMES_ALLOW_DEGRADED=1`` 可顯式豁免（明知故犯要寫在環境裡）。

數據部署二選一（均以 readyz 為門）：
  1. git clone 本倉庫（數據隨庫提交）；
  2. pip 安裝 wheel + 設 ``HERMES_SHANGHAN_DATA`` 指向獨立分發的數據資產
     目錄（其 manifest 哈希即版本指紋）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from . import config

CANONICAL_CLAUSES = 398


class MissingAssetsError(RuntimeError):
    pass


def livez() -> Dict:
    return {"ok": True, "pid": os.getpid()}


def _check(name: str, ok: bool, detail: str) -> Dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def readyz(include_runtime: bool = False) -> Dict:
    """逐項校驗數據能力。include_runtime=True 時額外對比運行時工具註冊表
    與提交的 tool_specs.json（需要構建註冊表，較重）。"""
    checks: List[Dict] = []

    manifest_p = config.MANIFEST_DIR / "corpus_manifest.json"
    manifest = None
    if manifest_p.exists():
        try:
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            checks.append(_check("manifest", True,
                                 f"{len(manifest.get('books', manifest))} 項"))
        except Exception as exc:
            checks.append(_check("manifest", False, f"損壞：{exc}"))
    else:
        checks.append(_check("manifest", False, f"缺失 {manifest_p.name}"))

    clause_p = config.CLAUSE_DIR / "clauses.jsonl"
    if clause_p.exists():
        canonical = total = 0
        try:
            with clause_p.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    total += 1
                    if json.loads(line).get("text_type") == "original_clause":
                        canonical += 1
            ok = canonical == CANONICAL_CLAUSES
            checks.append(_check(
                "clauses", ok,
                f"共 {total} 條，核心 {canonical}/{CANONICAL_CLAUSES}"))
        except Exception as exc:
            checks.append(_check("clauses", False, f"損壞：{exc}"))
    else:
        checks.append(_check("clauses", False, "缺失 clauses.jsonl"))

    for name, path in (
            ("initial_rules", config.RULES_INITIAL_DIR / "initial_rules.jsonl"),
            ("formula_rules",
             config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")):
        if path.exists() and path.stat().st_size > 0:
            n = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip())
            checks.append(_check(name, n > 0, f"{n} 條"))
        else:
            checks.append(_check(name, False, f"缺失/為空 {path.name}"))

    # 檢索索引由 ClauseRAG 啟動時在內存構建（無盤上索引文件），
    # 其就緒性由 clauses 檢查覆蓋——不做空目錄假檢查

    spec_p = config.SHANGHAN_DIR / "tool_specs.json"
    if spec_p.exists():
        try:
            spec = json.loads(spec_p.read_text(encoding="utf-8"))
            n_tools = len(spec.get("openai_tools", []))
            n_contracts = len(spec.get("contracts", []))
            ok = n_tools > 0 and n_tools == n_contracts
            detail = f"{n_tools} 工具 / {n_contracts} 契約"
            if include_runtime:
                from .agent.tools import get_registry
                n_rt = len(get_registry().names())
                ok = ok and n_rt == n_tools
                detail += f" / 運行時 {n_rt}"
            checks.append(_check("tool_specs", ok, detail))
        except Exception as exc:
            checks.append(_check("tool_specs", False, f"損壞：{exc}"))
    else:
        checks.append(_check("tool_specs", False, "缺失 tool_specs.json"))

    trace_p = config.TRACE_DIR / "claims.json"
    checks.append(_check("trace_assets", trace_p.exists(),
                         "claims.json " + ("在位" if trace_p.exists() else "缺失")))

    ready = all(c["ok"] for c in checks)
    return {"ready": ready, "checks": checks,
            "data_dir": str(config.DATA_DIR),
            "hint": "" if ready else
            "數據資產不完整：git clone 倉庫（數據隨庫提交）並運行 "
            "`python3 -m hermes_shanghan pipeline`，或設 HERMES_SHANGHAN_DATA "
            "指向數據資產目錄。pip wheel 只含代碼不含語料。"}


def assert_ready(context: str = "") -> None:
    """核心資產缺失時響亮失敗（除非 HERMES_ALLOW_DEGRADED=1 顯式豁免）。
    只做文件級檢查，廉價且無循環依賴。"""
    if os.environ.get("HERMES_ALLOW_DEGRADED") == "1":
        return
    missing = [str(p.name) for p in (
        config.CLAUSE_DIR / "clauses.jsonl",
        config.RULES_INITIAL_DIR / "initial_rules.jsonl",
        config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl",
    ) if not p.exists()]
    if missing:
        raise MissingAssetsError(
            f"{context or '系統'}啟動被拒：數據資產缺失 {missing}（拒絕假健康"
            f"空運行）。當前數據根 {config.DATA_DIR}。修復：git clone 倉庫並"
            "運行 `python3 -m hermes_shanghan pipeline`，或設 "
            "HERMES_SHANGHAN_DATA 指向數據資產；確要空跑設 "
            "HERMES_ALLOW_DEGRADED=1。")
