"""Memory modules for Hermes-Shanghanlun.

Seven JSON-backed stores, per the protocol:
  clause_memory        — per-clause processing state
  formula_memory       — aliases, composition, modification family, clauses
  six_channel_memory   — channel rules, outline & variant clauses
  mistreatment_memory  — mistreatment types, paths, rescue formulas
  critic_memory        — recurring model error patterns + examples
  skill_memory         — per-skill build history & usage notes
  paper_memory         — data/figures/conclusions used by generated papers

Each store is a dict persisted at data/shanghan/memory/<name>.json with a
small update API; stores are append-friendly and human-inspectable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config

MEMORY_NAMES = [
    "clause_memory", "formula_memory", "six_channel_memory",
    "mistreatment_memory", "critic_memory", "skill_memory", "paper_memory",
    # agent-layer stores: user corrections（「不是X而是Y」）and long-running
    # research-project state (topics, key formulas, draft outputs)
    "correction_memory", "project_memory",
]


class MemoryStore:
    def __init__(self, name: str, root: Optional[Path] = None):
        if name not in MEMORY_NAMES:
            raise ValueError(f"unknown memory module: {name}")
        self.name = name
        self.path = (root or config.MEMORY_DIR) / f"{name}.json"
        self.data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def update(self, key: str, **fields):
        entry = self.data.setdefault(key, {})
        entry.update(fields)
        entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def append(self, key: str, item: Any, max_items: int = 200):
        lst: List = self.data.setdefault(key, [])
        lst.append(item)
        if len(lst) > max_items:
            del lst[: len(lst) - max_items]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


class MemoryHub:
    """Convenience accessor for all seven stores."""

    def __init__(self, root: Optional[Path] = None):
        self._root = root
        self._stores: Dict[str, MemoryStore] = {}

    def __getattr__(self, name: str) -> MemoryStore:
        if name in MEMORY_NAMES:
            if name not in self._stores:
                self._stores[name] = MemoryStore(name, self._root)
            return self._stores[name]
        raise AttributeError(name)

    def save_all(self):
        for s in self._stores.values():
            s.save()
