"""ClassicalTextRAGAgent — retrieval over original clauses.

Supports the protocol's retrieval modes:
  * 條文號 (clause number / clause_id) direct lookup;
  * 方名 / 症狀 / 脈象 / 治法 / 禁忌 structured field filters;
  * BM25 lexical search (char n-grams) with structured-field boosting;
  * clause-relation graph expansion (related clauses appended).

Every hit returns the verbatim clause with book/chapter metadata so answers
can always cite their source (無條文編號，不成證據).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import config, lexicon
from ..schemas import ClauseRelation, ShanghanClause, read_jsonl
from ..textutil import normalize_query
from .bm25 import BM25Index

RE_CLAUSE_NUM_QUERY = re.compile(r"第?(\d{1,3})[條条]")


class ClauseRAG:
    def __init__(self, clauses: List[ShanghanClause],
                 relations: Optional[List[ClauseRelation]] = None):
        self.clauses = clauses
        self.by_id: Dict[str, ShanghanClause] = {c.clause_id: c for c in clauses}
        self.by_number: Dict[int, ShanghanClause] = {
            c.clause_number: c for c in clauses
            if c.text_type == "original_clause" and c.clause_number}
        self.relations = relations or []
        self._rel_by_src: Dict[str, List[ClauseRelation]] = {}
        for r in self.relations:
            self._rel_by_src.setdefault(r.source_clause_id, []).append(r)
            self._rel_by_src.setdefault(r.target_clause_id, []).append(r)
        self.index = BM25Index()
        for c in clauses:
            blocks = "\n".join(fb.raw_text for fb in c.formula_blocks)
            self.index.add(c.clause_id, c.clean_text + "\n" + blocks)
        self.index.finalize()

    @classmethod
    def load(cls) -> "ClauseRAG":
        clause_dicts = read_jsonl(config.CLAUSE_DIR / "clauses.jsonl")
        clauses = [ShanghanClause.from_dict(d) for d in clause_dicts]
        rel_dicts = read_jsonl(config.RELATION_DIR / "clause_relations.jsonl")
        relations = [ClauseRelation.from_dict(d) for d in rel_dicts]
        return cls(clauses, relations)

    # ------------------------------------------------------------------
    def get_clause(self, ref) -> Optional[ShanghanClause]:
        if isinstance(ref, int):
            return self.by_number.get(ref)
        ref = str(ref)
        if ref.isdigit():
            return self.by_number.get(int(ref))
        return self.by_id.get(ref)

    def related(self, clause_id: str, limit: int = 8) -> List[Dict]:
        out = []
        for r in self._rel_by_src.get(clause_id, [])[:limit * 2]:
            other = r.target_clause_id if r.source_clause_id == clause_id else r.source_clause_id
            out.append({"relation_type": r.relation_type, "clause_id": other,
                        "description": r.description, "confidence": r.confidence})
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 8,
               six_channel: Optional[str] = None,
               formula: Optional[str] = None,
               field: Optional[str] = None,
               expand_relations: bool = False) -> List[Dict]:
        query = normalize_query(query)

        # direct clause-number reference
        m = RE_CLAUSE_NUM_QUERY.search(query)
        if m:
            c = self.by_number.get(int(m.group(1)))
            if c:
                return [self._hit(c, 99.0, "clause_number")]

        # structured filter candidates
        def passes(c: ShanghanClause) -> bool:
            if six_channel and c.six_channel != six_channel:
                return False
            if formula:
                f = lexicon.canonical_formula(normalize_query(formula))
                if f not in c.formula_names:
                    return False
            if field:
                fields = {
                    "symptom": c.symptoms, "pulse": c.pulse,
                    "therapy": c.therapy_terms,
                    "contraindication": c.contraindication_terms,
                    "mistreatment": c.mistreatment_terms,
                    "formula": c.formula_names,
                    "disease": c.disease_patterns,
                }
                vals = fields.get(field, [])
                if not any(query.strip() in v or v in query for v in vals):
                    return False
            return True

        scored = self.index.search(query, top_k=top_k * 5)
        results = []
        for cid, score in scored:
            c = self.by_id[cid]
            if not passes(c):
                continue
            boost = 0.0
            # structured boosting
            fq = lexicon.canonical_formula(query)
            if fq in c.formula_names:
                boost += 3.0
            if any(s in query for s in c.symptoms):
                boost += 0.6
            if any(p in query for p in c.pulse):
                boost += 0.6
            if c.text_type == "original_clause":
                boost += 0.4
            results.append(self._hit(c, score + boost, "bm25"))
            if len(results) >= top_k:
                break
        results.sort(key=lambda h: -h["score"])

        if expand_relations and results:
            seen = {h["clause_id"] for h in results}
            for h in list(results[:3]):
                for rel in self.related(h["clause_id"], limit=3):
                    rid = rel["clause_id"]
                    if rid in seen or rid not in self.by_id:
                        continue
                    seen.add(rid)
                    extra = self._hit(self.by_id[rid], 0.1, f"relation:{rel['relation_type']}")
                    results.append(extra)
        return results

    def _hit(self, c: ShanghanClause, score: float, source: str) -> Dict:
        return {
            "clause_id": c.clause_id,
            "clause_number": c.clause_number,
            "book": c.book_title,
            "chapter": c.chapter,
            "six_channel": c.six_channel,
            "text": c.clean_text,
            "text_type": c.text_type,
            "layer": c.layer,
            "layer_label": config.LAYER_LABEL.get(c.layer, ""),
            "formulas": c.formula_names,
            "score": round(score, 3),
            "match_source": source,
        }
