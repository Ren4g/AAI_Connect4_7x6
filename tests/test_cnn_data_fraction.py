import pytest

import numpy as np

from connect4.train_cnn import policy_targets, select_train_indices


def test_data_fraction_uses_nested_whole_source_games():
    rows = [
        {"source_game": f"game-{game}", "split": "train"}
        for game in range(20)
        for _ in range(2)
    ] + [
        {"source_game": "validation-game", "split": "validation"}
    ]

    small, small_games = select_train_indices(rows, 0.1, 55)
    medium, medium_games = select_train_indices(rows, 0.3, 55)
    full, full_games = select_train_indices(rows, 1.0, 55)

    assert len(small) == small_games * 2 == 4
    assert len(medium) == medium_games * 2 == 12
    assert len(full) == full_games * 2 == 40
    assert set(small) < set(medium) < set(full)
    assert all(index < 40 for index in full)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_data_fraction_rejects_out_of_range_values(fraction):
    with pytest.raises(ValueError, match="data fraction"):
        select_train_indices([], fraction, 1)


def test_hard_policy_target_is_one_hot_and_uses_an_exact_optimal_action():
    policy = np.full((2, 7), 1 / 7, dtype=np.float32)
    optimal = np.zeros((2, 7), dtype=bool)
    optimal[0, [1, 4]] = True
    optimal[1, 6] = True

    targets = policy_targets(policy, optimal, "hard_best")

    assert np.all(targets.sum(axis=1) == 1)
    assert targets[0, 4] == 1.0
    assert targets[1, 6] == 1.0
    assert np.all(targets[~optimal] == 0.0)


def test_dataset_policy_target_preserves_soft_distribution():
    policy = np.asarray([[0.2, 0.8, 0, 0, 0, 0, 0]], dtype=np.float32)
    optimal = np.asarray([[False, True, False, False, False, False, False]])

    targets = policy_targets(policy, optimal, "dataset")

    np.testing.assert_array_equal(targets, policy)
