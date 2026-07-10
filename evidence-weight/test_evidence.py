"""Tests for evidence.py. Run with `python test_evidence.py` or `pytest`."""
from __future__ import annotations

from evidence import Observation, classify_pair, compute_belief, independent_mass

S, P = "eiffel_tower", "located_in"


def _support(source, text):
    return Observation(S, P, "Paris", source=source, text=text)


def test_weight_uses_only_support():
    support = [_support("a", "one"), _support("b", "two")]
    oppose = Observation(S, P, "Berlin", source="c", supports=False, text="no")
    w_support, _ = compute_belief(support)
    w_with_oppose, label = compute_belief(support + [oppose])
    assert w_with_oppose == w_support        # opposition never lowers the weight
    assert label == "contested"              # it only flips the label


def test_independence_discount_beats_volume():
    three_independent = [_support(f"src{i}", f"text {i}") for i in range(3)]
    five_copies = [_support(f"blog{i}", "identical text") for i in range(5)]
    w3, _ = compute_belief(three_independent)
    w5, _ = compute_belief(five_copies)
    assert w3 > w5                            # 3 real voices beat 5 echoes
    # five copies collapse to roughly one voice, nowhere near five
    assert independent_mass(five_copies) < 2.0
    assert independent_mass(three_independent) >= 3.0


def test_contestation_labels():
    assert compute_belief([_support("a", "one")])[1] == "singular"
    assert compute_belief([_support("a", "one"), _support("b", "two")])[1] == "consensus"
    contested = [_support("a", "one"),
                 Observation(S, P, "Berlin", source="b", supports=False)]
    assert compute_belief(contested)[1] == "contested"


def test_classify_pair_routes_instead_of_latest_wins():
    paris_2000 = Observation(S, P, "Paris", source="atlas", valid_from=2000)
    berlin_2000 = Observation(S, P, "Berlin", source="gazette", valid_from=2000)
    berlin_2020_same = Observation(S, P, "Berlin", source="atlas", valid_from=2020)
    paris_again = Observation(S, P, "Paris", source="almanac", valid_from=2000)

    assert classify_pair(paris_2000, berlin_2000) == "contestation"     # not "latest wins"
    assert classify_pair(paris_2000, berlin_2020_same) == "self_correction"
    assert classify_pair(paris_2000, paris_again) == "reinforcement"
    assert classify_pair(paris_2000, Observation("x", "y", "z", "s")) == "unrelated"


def test_conflicting_claim_can_still_be_high_weight():
    # three strong supporters + one opposer: high weight AND contested, together
    obs = [_support(f"s{i}", f"t{i}") for i in range(3)]
    obs.append(Observation(S, P, "Berlin", source="z", supports=False))
    weight, label = compute_belief(obs)
    assert weight > 0.9 and label == "contested"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
