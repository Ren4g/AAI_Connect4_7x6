from scripts.arena_checkpoints import promotion_decision


def test_promotion_requires_opening_count_score_and_lower_ci():
    passing = {"unique_openings": 100, "score": 0.58,
               "score_bootstrap_95": (0.51, 0.64)}
    assert promotion_decision(passing)
    assert not promotion_decision({**passing, "score_bootstrap_95": (0.49, 0.64)})
    assert not promotion_decision({**passing, "score": 0.5})
    assert not promotion_decision({**passing, "unique_openings": 99})
