"""Pure-python BM25 (Okapi) over char unigram+bigram tokens.

No external dependencies; fast enough for the full corpus (≤ tens of
thousands of short classical paragraphs).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from ..textutil import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.tf: List[Counter] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    def add(self, doc_id: str, text: str):
        toks = tokenize(text)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(toks))
        counts = Counter(toks)
        self.tf.append(counts)
        for t in counts:
            self.df[t] += 1

    def finalize(self):
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        # inverted index for speed
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for t, c in counts.items():
                self._postings[t].append((i, c))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.doc_ids:
            return []
        q_toks = set(tokenize(query))
        n_docs = len(self.doc_ids)
        scores: Dict[int, float] = defaultdict(float)
        for t in q_toks:
            postings = self._postings.get(t)
            if not postings:
                continue
            idf = math.log(1 + (n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in postings:
                dl = self.doc_len[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.doc_ids[i], round(s, 4)) for i, s in ranked]
