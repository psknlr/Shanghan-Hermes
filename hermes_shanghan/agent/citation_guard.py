"""Citation guard — enforces 「無條文編號，不成證據」 on LLM output.

Scans an answer for clause references (SHL_SONGBEN_xxxx or 第N條), verifies
each against the clause store, and checks that any quoted classical text near
a citation actually matches the cited clause. Unsupported citations and
fabricated quotes are reported so the agent can flag or strip them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..textutil import similarity

RE_CLAUSE_ID = re.compile(r"SHL_SONGBEN_(?:AUX_)?\d{4}")
RE_CLAUSE_NUM = re.compile(r"第\s*(\d{1,3})\s*條")
RE_QUOTE = re.compile(r"[「『\"]([^」』\"]{6,80})[」』\"]")


@dataclass
class CitationReport:
    cited_ids: List[str] = field(default_factory=list)
    verified_ids: List[str] = field(default_factory=list)
    unsupported_ids: List[str] = field(default_factory=list)
    # verified against the corpus but NOT present in this round's tool
    # evidence — exists, yet the agent never retrieved it (嚴格 RAG 接地)
    outside_evidence_ids: List[str] = field(default_factory=list)
    quote_mismatches: List[Dict] = field(default_factory=list)
    has_any_citation: bool = False

    @property
    def ok(self) -> bool:
        return not self.unsupported_ids and not self.quote_mismatches \
            and not self.outside_evidence_ids

    def to_dict(self) -> Dict:
        return {"cited": self.cited_ids, "verified": self.verified_ids,
                "unsupported": self.unsupported_ids,
                "outside_evidence": self.outside_evidence_ids,
                "quote_mismatches": self.quote_mismatches,
                "has_any_citation": self.has_any_citation, "ok": self.ok}


class CitationGuard:
    def __init__(self, clause_store: Dict):
        self.store = clause_store          # clause_id -> ShanghanClause
        self._by_number = {c.clause_number: c for c in clause_store.values()
                           if getattr(c, "clause_number", 0)}

    def _resolve(self, ref: str):
        if ref in self.store:
            return self.store[ref]
        return None

    def check(self, answer: str,
              allowed_ids: Optional[List[str]] = None) -> CitationReport:
        """Verify citations against the corpus and, when ``allowed_ids`` is
        given, against this round's tool evidence: a clause that exists but
        was never retrieved is flagged ``outside_evidence`` — 引用必須綁定
        本輪取證，不能只是「庫裡存在」."""
        rep = CitationReport()
        ids = list(dict.fromkeys(RE_CLAUSE_ID.findall(answer)))
        for m in RE_CLAUSE_NUM.findall(answer):
            c = self._by_number.get(int(m))
            if c:
                ids.append(c.clause_id)
        ids = list(dict.fromkeys(ids))
        rep.cited_ids = ids
        rep.has_any_citation = bool(ids)
        allowed = set(allowed_ids) if allowed_ids is not None else None
        for cid in ids:
            c = self._resolve(cid)
            if c is None:
                rep.unsupported_ids.append(cid)
            elif allowed is not None and cid not in allowed:
                rep.outside_evidence_ids.append(cid)
                rep.verified_ids.append(cid)
            else:
                rep.verified_ids.append(cid)

        # verify quoted classical text against ALL cited clauses（任一已引
        # 條文含該引文即通過；不做「最近引用位置」配對——句級歸屬與詞彙
        # 對齊見 EvidenceBinder，此處只攔「引文在所有引用條文中都不存在」
        # 的偽造）
        quotes = RE_QUOTE.findall(answer)
        if quotes and rep.verified_ids:
            corpus = {cid: self.store[cid].clean_text for cid in rep.verified_ids}
            for q in quotes:
                if any(q in t or similarity(q, t) >= 0.6 for t in corpus.values()):
                    continue
                # quote not found in any cited clause → possible fabrication
                best = max(corpus.values(), key=lambda t: similarity(q, t), default="")
                if similarity(q, best) < 0.45:
                    rep.quote_mismatches.append({"quote": q, "matched": False})
        return rep

    def annotate(self, answer: str, rep: CitationReport) -> str:
        """Append a verification footer; warn on unsupported citations."""
        footer = ["", "—" * 12, "【證據核驗】"]
        if rep.verified_ids:
            footer.append(f"已核實條文：{'、'.join(rep.verified_ids)}（A 原文直述，可回源）")
        if rep.unsupported_ids:
            footer.append(f"⚠️ 未能核實的條文編號（請勿採信）：{'、'.join(rep.unsupported_ids)}")
        if rep.outside_evidence_ids:
            footer.append("⚠️ 以下條文雖存在於語料，但未出現在本輪檢索證據中"
                          f"（引用未接地）：{'、'.join(rep.outside_evidence_ids)}")
        if rep.quote_mismatches:
            qs = "；".join(m["quote"][:20] for m in rep.quote_mismatches)
            footer.append(f"⚠️ 以下引文未能在所引條文中逐字核對：{qs}")
        if not rep.has_any_citation:
            footer.append("⚠️ 本回答未包含可核驗的條文編號，按本系統規則僅供參考。")
        return answer + "\n" + "\n".join(footer)
