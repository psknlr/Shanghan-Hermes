"""Central configuration for Hermes-Shanghanlun.

All paths are resolved relative to the repository root so the system can be
run from any working directory with ``python -m hermes_shanghan``.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("HERMES_SHANGHAN_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = REPO_ROOT / "data"
CORPUS_RAW_DIR = DATA_DIR / "corpus_raw"
SHANGHAN_DIR = DATA_DIR / "shanghan"

MANIFEST_DIR = SHANGHAN_DIR / "manifest"
CLAUSE_DIR = SHANGHAN_DIR / "clauses"
RELATION_DIR = SHANGHAN_DIR / "relations"
RULES_INITIAL_DIR = SHANGHAN_DIR / "rules_initial"
AUDIT_DIR = SHANGHAN_DIR / "audit"
REJECTED_DIR = SHANGHAN_DIR / "rejected"
RULES_FORMULA_DIR = SHANGHAN_DIR / "rules_formula"
RULES_SIX_CHANNEL_DIR = SHANGHAN_DIR / "rules_six_channel"
RULES_THERAPY_DIR = SHANGHAN_DIR / "rules_therapy"
RULES_MISTREATMENT_DIR = SHANGHAN_DIR / "rules_mistreatment"
RULES_DIFFERENTIAL_DIR = SHANGHAN_DIR / "rules_differential"
RULES_MERGED_DIR = SHANGHAN_DIR / "rules_merged"
RULES_VARIANT_DIR = SHANGHAN_DIR / "rules_variant"
RULES_COMMENTARY_DIR = SHANGHAN_DIR / "rules_commentary"
MEMORY_DIR = SHANGHAN_DIR / "memory"
INDEX_DIR = SHANGHAN_DIR / "index"
RESEARCH_DIR = SHANGHAN_DIR / "research"
PAPER_DIR = SHANGHAN_DIR / "papers"

SKILLS_DIR = DATA_DIR / "skills" / "shanghanlun"

ALL_OUTPUT_DIRS = [
    MANIFEST_DIR, CLAUSE_DIR, RELATION_DIR, RULES_INITIAL_DIR, AUDIT_DIR,
    REJECTED_DIR, RULES_FORMULA_DIR, RULES_SIX_CHANNEL_DIR, RULES_THERAPY_DIR,
    RULES_MISTREATMENT_DIR, RULES_DIFFERENTIAL_DIR, RULES_MERGED_DIR,
    RULES_VARIANT_DIR, RULES_COMMENTARY_DIR, MEMORY_DIR, INDEX_DIR,
    RESEARCH_DIR, PAPER_DIR, SKILLS_DIR,
]

# ---------------------------------------------------------------------------
# Canonical source books (Hermes evidence layers A-E)
# ---------------------------------------------------------------------------
# Layer A: canonical original clauses (Songben with modern numbering).
# Layer B: version variants. Layer C: commentaries. Layer D: latter-day
# formula-family induction. Layer E: model interpretation (generated).
PRIMARY_BOOK = "傷寒論_條文版"          # Songben text with the standard 1-398 numbering
SONGBEN_FULL_BOOK = "傷寒論_宋本"       # full Songben incl. auxiliary chapters
VARIANT_BOOKS = ["傷寒雜病論_桂本", "傷寒論_千金翼方版"]
COMMENTARY_ALIGN_BOOK = "註解傷寒論"    # Cheng Wuji's annotated edition, aligned clause by clause
COMMENTARY_BOOKS = [
    "註解傷寒論", "傷寒論條辨", "傷寒來蘇集", "傷寒貫珠集", "傷寒溯源集",
    "張卿子傷寒論", "傷寒懸解", "傷寒論注", "傷寒論輯義",
]
FORMULA_FAMILY_BOOKS = ["傷寒論類方"]

LAYER_OF_BOOK = {
    PRIMARY_BOOK: "A",
    SONGBEN_FULL_BOOK: "A",
    **{b: "B" for b in VARIANT_BOOKS},
    **{b: "C" for b in COMMENTARY_BOOKS},
    **{b: "D" for b in FORMULA_FAMILY_BOOKS},
}

LAYER_LABEL = {
    "A": "原文直述",
    "B": "版本異文",
    "C": "注家解釋",
    "D": "後世類方歸納",
    "E": "模型推理",
}

# ---------------------------------------------------------------------------
# Six-channel chapter mapping for the canonical numbered edition
# ---------------------------------------------------------------------------
CHAPTER_TO_CHANNEL = {
    "辨太陽病脈證並治上": "太陽病",
    "辨太陽病脈證並治中": "太陽病",
    "辨太陽病脈證並治下": "太陽病",
    "辨陽明病脈證並治": "陽明病",
    "辨少陽病脈證並治": "少陽病",
    "辨太陰病脈證並治": "太陰病",
    "辨少陰病脈證並治": "少陰病",
    "辨厥陰病脈證並治": "厥陰病",
    "辨霍亂病脈證並治": "霍亂病",
    "辨陰陽易差後勞復病脈證並治": "陰陽易差後勞復病",
}

SIX_CHANNELS = ["太陽病", "陽明病", "少陽病", "太陰病", "少陰病", "厥陰病"]
EXTRA_CHANNELS = ["霍亂病", "陰陽易差後勞復病"]

CHANNEL_PINYIN = {
    "太陽病": "taiyang", "陽明病": "yangming", "少陽病": "shaoyang",
    "太陰病": "taiyin", "少陰病": "shaoyin", "厥陰病": "jueyin",
    "霍亂病": "huoluan", "陰陽易差後勞復病": "laofu",
}

# Canonical chapter (提綱) clause numbers per channel.
CHANNEL_OUTLINE_CLAUSE = {
    "太陽病": 1, "陽明病": 180, "少陽病": 263,
    "太陰病": 273, "少陰病": 281, "厥陰病": 326,
}

# ---------------------------------------------------------------------------
# Release gate thresholds (ConsensusJudge / ReleaseGate)
# ---------------------------------------------------------------------------
RELEASE_GOLD = 0.90
RELEASE_SILVER = 0.78
RELEASE_BRONZE = 0.62

ID_PREFIX_CLAUSE = "SHL_SONGBEN_"
ID_PREFIX_AUX = "SHL_SONGBEN_AUX_"


def ensure_dirs() -> None:
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
