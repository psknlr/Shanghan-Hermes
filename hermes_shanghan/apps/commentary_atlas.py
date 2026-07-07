"""注家分歧圖譜 (Commentary Divergence Atlas) — C 層的可計算化.

With every quote-then-comment 注本 aligned clause-by-clause, the
interpretation history of each clause becomes computable:

  book coverage        per-book alignment count & mean similarity —
                       philologically honest: books that paraphrase or
                       reorganize score low and are reported as such
  term profiles        each commentary's analytic vocabulary (病機/八綱/
                       治法 terms from the shared lexicon — the same closed
                       list the critic uses, so D-layer terminology is
                       detected, not invented)
  clause divergence    per clause with ≥2 distinct commentators: mean
                       pairwise Jaccard distance between term profiles
                       (+ lexical bigram distance as a style-robust check);
                       the ranking surfaces 學術爭點條文 from data alone
  agreement matrix     commentator × commentator mean profile similarity
                       over co-annotated clauses — school clusters emerge
                       from the numbers without hand-assigned labels
  fingerprints         per commentator: terms used disproportionately vs
                       the全體 average (distinctive vocabulary)

Everything here is D/E-layer induction over C-layer aligned text; each
record carries clause_ids so every claim remains back-traceable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Set

from .. import lexicon
from ..schemas import CommentaryRule, ShanghanClause
from ..textutil import similarity

# analytic vocabulary: the critic's posthoc-term list + channel names +
# multi-char therapy terms — a CLOSED lexicon shared with the review gates
ANALYTIC_EXTRA = [
    "表裏", "表證", "裏證", "半表半裏", "寒熱", "虛實", "陰陽", "榮衛",
    "津液", "胃氣", "正氣", "邪氣", "經絡", "三焦", "腠理", "汗解",
    "傳經", "直中", "合病", "併病", "壞病", "標本", "從治", "逆治",
]


def analytic_terms() -> List[str]:
    terms: Set[str] = set(lexicon.POSTHOC_TERMS) | set(ANALYTIC_EXTRA)
    terms |= set(lexicon.CHANNEL_IN_TEXT.values())
    for v in lexicon.THERAPY_METHODS.values():
        terms |= {t for t in v if len(t) >= 2}
    return sorted(terms, key=lambda t: (-len(t), t))


class CommentaryAtlas:
    def __init__(self, commentary_rules: List[CommentaryRule],
                 clause_store: Dict[str, ShanghanClause]):
        self.rules = commentary_rules
        self.clauses = clause_store
        self.terms = analytic_terms()

    # ------------------------------------------------------------------
    def _profile(self, text: str) -> Set[str]:
        return {t for t in self.terms if t in text}

    @staticmethod
    def _jaccard_distance(a: Set[str], b: Set[str]) -> Optional[float]:
        if not a and not b:
            return None
        return 1.0 - len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    def build(self, top_n: int = 20) -> Dict:
        # book coverage -------------------------------------------------
        by_book: Dict[str, List[CommentaryRule]] = defaultdict(list)
        for r in self.rules:
            by_book[r.book].append(r)
        coverage = {book: {
            "commentator": rs[0].commentator,
            "n_aligned_clauses": len({r.clause_id for r in rs}),
            "mean_similarity": round(sum(r.alignment_similarity for r in rs)
                                     / len(rs), 3),
        } for book, rs in sorted(by_book.items())}

        # per clause: best rule per DISTINCT commentator; ties (柯琴's two
        # books often align identically) break on rule id so the atlas is
        # invariant to input rule order
        per_clause: Dict[str, Dict[str, CommentaryRule]] = defaultdict(dict)
        for r in self.rules:
            cur = per_clause[r.clause_id].get(r.commentator)
            if cur is None or \
                    (r.alignment_similarity, cur.commentary_rule_id) > \
                    (cur.alignment_similarity, r.commentary_rule_id):
                per_clause[r.clause_id][r.commentator] = r

        # divergence per clause ------------------------------------------
        clause_rows: List[Dict] = []
        pair_sims: Dict[frozenset, List[float]] = defaultdict(list)
        term_usage: Dict[str, Counter] = defaultdict(Counter)
        n_annotated: Counter = Counter()
        for cid, by_comm in per_clause.items():
            profiles = {comm: self._profile(r.commentary_text)
                        for comm, r in by_comm.items()}
            for comm, prof in profiles.items():
                n_annotated[comm] += 1
                term_usage[comm].update(prof)
            if len(by_comm) < 2:
                continue
            term_ds, lex_ds = [], []
            for a, b in combinations(sorted(by_comm), 2):
                d = self._jaccard_distance(profiles[a], profiles[b])
                if d is not None:
                    term_ds.append(d)
                    pair_sims[frozenset((a, b))].append(1.0 - d)
                lex_ds.append(1.0 - similarity(by_comm[a].commentary_text,
                                               by_comm[b].commentary_text))
            union = set().union(*profiles.values())
            distinctive = {comm: sorted(prof - set().union(
                *(p for c2, p in profiles.items() if c2 != comm)))[:6]
                for comm, prof in profiles.items() if prof}
            clause_rows.append({
                "clause_id": cid,
                "n_commentators": len(by_comm),
                "commentators": sorted(by_comm),
                "term_divergence": round(sum(term_ds) / len(term_ds), 4)
                    if term_ds else None,
                "lexical_divergence": round(sum(lex_ds) / len(lex_ds), 4),
                "n_analytic_terms": len(union),
                "distinctive_terms": {k: v for k, v in distinctive.items() if v},
            })
        clause_rows.sort(key=lambda r: (-(r["term_divergence"] or 0),
                                        -r["lexical_divergence"],
                                        r["clause_id"]))

        # agreement matrix ------------------------------------------------
        agreement = []
        for pair, sims in sorted(pair_sims.items(),
                                 key=lambda kv: tuple(sorted(kv[0]))):
            a, b = sorted(pair)
            agreement.append({"a": a, "b": b, "n_shared_clauses": len(sims),
                              "mean_term_agreement": round(sum(sims) / len(sims), 4)})

        # fingerprints: leave-one-out usage-rate lift — the commentator's
        # own usage is excluded from the reference rate, otherwise dominant
        # commentators compress toward 1 while sparse ones get inflated
        # ceilings; +1 pseudo-count keeps exclusive terms finite
        global_rate: Counter = Counter()
        for comm, cnt in term_usage.items():
            for t, n in cnt.items():
                global_rate[t] += n
        total_annotated = sum(n_annotated.values()) or 1
        fingerprints = {}
        for comm, cnt in sorted(term_usage.items()):
            rest_n = max(1, total_annotated - n_annotated[comm])
            rows = []
            for t, n in cnt.items():
                if n < 3:
                    continue
                own = n / n_annotated[comm]
                rest = max(1, global_rate[t] - n) / rest_n
                rows.append((round(own / rest, 2), n, t))
            rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
            fingerprints[comm] = [{"term": t, "lift": lift, "n": n}
                                  for lift, n, t in rows[:8]]

        top = []
        for r in clause_rows[:top_n]:
            c = self.clauses.get(r["clause_id"])
            snippet = c.clean_text.replace("\n", "")[:60] + "…" if c else ""
            top.append({**r, "clause_text": snippet})
        multi = [r for r in clause_rows if r["n_commentators"] >= 2]
        return {
            "interpretation_level": "posthoc_induction(D/E) over aligned C-layer",
            "n_commentary_rules": len(self.rules),
            "n_books": len(by_book),
            "book_coverage": coverage,
            "n_clauses_multi_commentator": len(multi),
            "mean_term_divergence": round(
                sum(r["term_divergence"] for r in multi
                    if r["term_divergence"] is not None)
                / max(1, sum(1 for r in multi
                             if r["term_divergence"] is not None)), 4),
            "top_divergent_clauses": top,
            "agreement_matrix": agreement,
            "commentator_fingerprints": fingerprints,
            "clauses": clause_rows,
        }
