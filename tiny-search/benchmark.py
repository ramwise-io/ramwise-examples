"""The demo-vs-benchmark lesson, made runnable.

Companion to: https://ramwise.dev/blog/i-built-a-search-engine-by-hand/

By eye, the engine looks great -- type a query, get relevant-looking results.
Then you benchmark it against relevance judgments and the number can be humbling.
This script shows two plausible-looking changes and what a benchmark reveals:

  * aggressive query expansion (synonym flooding) drops PRECISION, because the
    extra terms pull in off-topic documents that happen to share a synonym;
  * a hard score cutoff drops RECALL, because it throws away relevant documents
    that legitimately scored low.

The demo looks the same through all of it. The benchmark does not.

Run:  python benchmark.py
"""
from __future__ import annotations

from tiny_search import TinySearch, tokenize

CORPUS = {
    # relevant, on-topic documents
    "d1": "international travel guide flights visas journeys abroad",
    "d2": "budget travel cheap flights hostels",
    "d4": "visa requirements international students exchange",
    "d6": "travel insurance trips abroad",
    "d8": "flight booking airline miles cheap deals",
    # plainly unrelated documents
    "d3": "home gardening winter frost protection",
    "d7": "cooking pasta home quick recipes",
    # "trap" documents: they match only the EXPANSION synonyms, not the query
    "t1": "daily commute to the office morning routine move around the city",
    "t2": "global foreign exchange markets outside investors world economy",
}
QUERIES = {
    "international travel": {"d1", "d4", "d6"},
    "cheap flights": {"d2", "d8"},
}

# A deliberately noisy synonym table -- the WordNet-style expansion the post
# blames for burying the signal.
EXPANSION = {
    "travel": ["journey", "trip", "move", "commute", "go"],
    "international": ["external", "outside", "foreign", "global"],
    "cheap": ["low", "inexpensive", "budget", "discount"],
}


def build() -> TinySearch:
    eng = TinySearch()
    for doc_id, text in CORPUS.items():
        eng.add(doc_id, text)
    eng.index()
    return eng


def expand(query: str) -> str:
    out = list(tokenize(query))
    for tok in tokenize(query):
        out.extend(EXPANSION.get(tok, []))
    return " ".join(out)


def mean_precision_at_k(eng: TinySearch, *, expand_queries: bool, k: int = 3) -> float:
    vals = []
    for query, relevant in QUERIES.items():
        q = expand(query) if expand_queries else query
        top = [doc for doc, _ in eng.search(q, k=k)]
        vals.append(sum(1 for d in top if d in relevant) / len(top) if top else 0.0)
    return sum(vals) / len(vals)


def mean_recall(eng: TinySearch, *, threshold: float) -> float:
    vals = []
    for query, relevant in QUERIES.items():
        retrieved = {d for d, s in eng.search(query, k=len(CORPUS)) if s >= threshold}
        vals.append(len(retrieved & relevant) / len(relevant))
    return sum(vals) / len(vals)


if __name__ == "__main__":
    eng = build()

    print("What the demo shows (top 3 for 'international travel'):")
    for doc, score in eng.search("international travel", k=3):
        print(f"  {score:.3f}  {doc}  {CORPUS[doc]}")

    p_plain = mean_precision_at_k(eng, expand_queries=False)
    p_expanded = mean_precision_at_k(eng, expand_queries=True)
    r_plain = mean_recall(eng, threshold=0.0)
    r_cutoff = mean_recall(eng, threshold=0.20)

    print("\nWhat the benchmark shows:")
    print(f"  mean precision@3, plain queries ......... {p_plain:.2f}")
    print(f"  mean precision@3, + query expansion ..... {p_expanded:.2f}   <- traps sneak in; precision falls")
    print(f"  mean recall, no cutoff .................. {r_plain:.2f}")
    print(f"  mean recall, + hard score cutoff ....... {r_cutoff:.2f}   <- relevant low-scorers dropped")
    print("\nThe demo looked the same the whole time. The benchmark did not.")
