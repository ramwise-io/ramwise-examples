"""A tiny vector-space search engine, built from scratch.

Inverted index + TF-IDF + cosine ranking, plus a minimal boolean layer.
Companion code for: https://ramwise.dev/blog/i-built-a-search-engine-by-hand/

The point is not to compete with a real search library. It is to make the two
lessons from the post concrete:

  1. An inverted index is the right *shape* for the data. You never store the
     zeros. A dense document-term matrix stores a cell for every term in every
     document; this stores only the postings that exist.

  2. Scoring only touches the documents a query term points to -- you walk the
     postings lists for the query's terms and score those, never the whole
     corpus. That, not the cosine formula, is why inverted indexes exist.

Pure standard library. No dependencies.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TinySearch:
    def __init__(self) -> None:
        self.docs: dict[str, str] = {}
        # the inverted index: term -> {doc_id: term_frequency}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._idf: dict[str, float] = {}
        self._doc_norm: dict[str, float] = {}
        self._indexed = False

    def add(self, doc_id: str, text: str) -> None:
        self.docs[doc_id] = text
        self._indexed = False

    def index(self) -> None:
        self.postings.clear()
        for doc_id, text in self.docs.items():
            tf: dict[str, int] = defaultdict(int)
            for tok in tokenize(text):
                tf[tok] += 1
            for term, freq in tf.items():
                self.postings[term][doc_id] = freq

        n = len(self.docs) or 1
        # idf = log(N / df). No smoothing -- exactly the choice the post flags as
        # fine on a toy corpus and quietly distorting on a real one.
        self._idf = {t: math.log(n / len(p)) for t, p in self.postings.items()}

        norms: dict[str, float] = defaultdict(float)
        for term, plist in self.postings.items():
            idf = self._idf[term]
            for doc_id, freq in plist.items():
                w = freq * idf
                norms[doc_id] += w * w
        self._doc_norm = {d: math.sqrt(s) or 1.0 for d, s in norms.items()}
        self._indexed = True

    def scores(self, query: str) -> dict[str, float]:
        """Cosine scores over ONLY the docs that contain a query term."""
        if not self._indexed:
            self.index()
        qtf: dict[str, int] = defaultdict(int)
        for tok in tokenize(query):
            qtf[tok] += 1
        qweights = {t: qtf[t] * self._idf.get(t, 0.0) for t in qtf}
        qnorm = math.sqrt(sum(w * w for w in qweights.values())) or 1.0

        dot: dict[str, float] = defaultdict(float)
        for term, qw in qweights.items():
            if qw == 0.0:
                continue
            # This loop is the whole point: we only visit documents in the
            # query terms' postings lists, never the full corpus.
            for doc_id, freq in self.postings.get(term, {}).items():
                dot[doc_id] += qw * (freq * self._idf[term])
        return {d: s / (self._doc_norm[d] * qnorm) for d, s in dot.items()}

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]


# --- a minimal boolean layer over score dicts (AND=min, OR=max, NOT=difference) ---
def _term_docs(engine: TinySearch, term: str) -> set[str]:
    return set(engine.postings.get(term, {}))


def boolean(engine: TinySearch, a: str, op: str, b: str) -> set[str]:
    """Set semantics over single terms: 'AND' | 'OR' | 'NOT' (a AND NOT b)."""
    da, db = _term_docs(engine, a), _term_docs(engine, b)
    if op == "AND":
        return da & db
    if op == "OR":
        return da | db
    if op == "NOT":
        return da - db
    raise ValueError(f"unknown op: {op!r}")


if __name__ == "__main__":
    eng = TinySearch()
    eng.add("d1", "international travel guide: flights, visas and journeys abroad")
    eng.add("d2", "budget travel tips for cheap flights")
    eng.add("d3", "home gardening in winter")
    eng.add("d4", "visa requirements for international students")
    for doc_id, score in eng.search("international travel", k=3):
        print(f"{score:.3f}  {doc_id}  {eng.docs[doc_id]}")
    print("travel AND visa ->", boolean(eng, "travel", "AND", "visa"))
