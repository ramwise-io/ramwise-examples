"""Weight isn't agreement: an evidence layer's two core moves, made runnable.

Companion code for:
https://ramwise.dev/blog/weight-isnt-agreement/

Two ideas, concrete:

1. compute_belief() returns a claim's WEIGHT (how much independent support it
   has) and its CONTESTATION (whether sources disagree) as SEPARATE values,
   never collapsed into one number. Weight consumes supporting evidence only,
   after collapsing correlated sources so five copies of one article count as
   about one voice -- volume can't buy truth, only independent corroboration
   can. Opposition never lowers the weight; it flips the contestation label.

2. classify_pair() routes two observations that disagree about the same thing
   through a small truth table -- instead of "latest wins," which would throw
   the disagreement away.

Pure standard library (math). A teaching distillation, not the real system.

Run:  python evidence.py
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

K = 1.2  # saturation constant: ~3 independent supporting sources -> ~0.85 weight


@dataclass(frozen=True)
class Observation:
    subject: str
    predicate: str
    obj: str
    source: str
    supports: bool = True   # True = asserts the claim, False = opposes it
    valid_from: int = 0     # a coarse "when this was true" clock
    text: str = ""          # snippet; identical text = a duplicate, not a witness


def independent_mass(observations) -> float:
    """Collapse correlated observations into ~one 'voice' each.

    Observations from the same source, or with identical snippet text, are
    echoes of one another rather than independent corroboration. Each distinct
    group counts as one full voice plus a heavily discounted remainder, so a
    wire story reposted five times counts as roughly one -- not five.
    """
    groups: dict[str, int] = {}
    for o in observations:
        key = o.text.strip().lower() or o.source   # same text OR same source = one voice
        groups[key] = groups.get(key, 0) + 1
    return sum(1.0 + 0.15 * (n - 1) for n in groups.values())


def compute_belief(observations):
    """Return (weight, contestation) -- computed on different axes, returned together."""
    support = [o for o in observations if o.supports]
    oppose = [o for o in observations if not o.supports]

    # WEIGHT: supporting mass only. Opposition does not enter this number.
    weight = round(1.0 - exp(-K * independent_mass(support)), 3)

    # CONTESTATION: a separate label about whether anyone independently disagrees.
    if oppose:
        contestation = "contested"
    elif independent_mass(support) >= 2:
        contestation = "consensus"
    else:
        contestation = "singular"

    return weight, contestation


def classify_pair(a: Observation, b: Observation) -> str:
    """Route two observations about the same (subject, predicate). No latest-wins."""
    if (a.subject, a.predicate) != (b.subject, b.predicate):
        return "unrelated"
    if a.obj == b.obj and a.supports == b.supports:
        return "reinforcement"                 # two voices, same claim
    if a.source == b.source:
        # one source changing its story over time is a correction, not a fight
        return "self_correction" if a.valid_from != b.valid_from else "internal_conflict"
    if a.valid_from == b.valid_from:
        return "contestation"                  # different sources, same time, disagree
    # different sources at different times: which supersedes which is unknowable
    # here, so preserve both rather than silently overwrite one with the other
    return "coexist"


if __name__ == "__main__":
    S, P = "eiffel_tower", "located_in"

    three_independent = [
        Observation(S, P, "Paris", source="atlas", text="the tower stands in Paris"),
        Observation(S, P, "Paris", source="encyclopedia", text="located in Paris, France"),
        Observation(S, P, "Paris", source="survey_office", text="Parisian landmark, 7th arr."),
    ]
    five_copies = [
        Observation(S, P, "Paris", source=f"blog_{i}", text="COPY-PASTE: it is in Paris")
        for i in range(5)
    ]

    w3, c3 = compute_belief(three_independent)
    w5, c5 = compute_belief(five_copies)
    naive5 = round(1.0 - exp(-K * len(five_copies)), 3)  # if we'd counted all 5 as voices
    print("Volume can't buy truth (independence discounting):")
    print(f"  3 independent sources        -> weight {w3}  ({c3})")
    print(f"  5 copies of one blog         -> weight {w5}  ({c5})   [naive count would say {naive5}]")
    print()

    contested = three_independent + [
        Observation(S, P, "Berlin", source="rumor_mill", supports=False, text="actually Berlin?"),
    ]
    wc, cc = compute_belief(contested)
    print("Weight is not agreement (adding opposition):")
    print(f"  before opposition            -> weight {w3}  ({c3})")
    print(f"  after 1 opposing source      -> weight {wc}  ({cc})   <- weight unchanged, label flipped")
    print()

    print("Two observations disagree -> route, don't pick a winner:")
    a = Observation(S, P, "Paris", source="atlas", valid_from=2000)
    b = Observation(S, P, "Berlin", source="gazette", valid_from=2000)
    c = Observation(S, P, "Berlin", source="atlas", valid_from=2020)
    print(f"  diff sources, same time      -> {classify_pair(a, b)}")
    print(f"  same source, later time      -> {classify_pair(a, c)}")
