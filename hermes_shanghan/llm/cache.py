"""Disk cache for LLM responses — makes real-model runs reproducible & cheap.

Keyed by sha256(model + normalized messages + tools + temperature). Cached
entries are plain JSON under data/shanghan/llm_cache/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config


def _cache_dir() -> Path:
    d = config.SHANGHAN_DIR / "llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(model: str, messages: List[Dict], tools: Optional[List[Dict]],
              temperature: float, task: Optional[str] = None,
              json_mode: bool = False) -> str:
    payload = json.dumps({"model": model, "messages": messages,
                          "tools": tools or [], "temperature": temperature,
                          "task": task or "", "json_mode": json_mode},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load(key: str) -> Optional[Dict[str, Any]]:
    p = _cache_dir() / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def store(key: str, value: Dict[str, Any]) -> None:
    p = _cache_dir() / f"{key}.json"
    try:
        p.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
