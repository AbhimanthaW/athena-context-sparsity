import pytest

from src.occupancy import (
    count_histories,
    calculate_occupancy_stats,
)


def test_occupancy_stats():
    """
    For n=3:

        k = 2

    Block 1:
        [1,2,3,4]

        histories:
            (1,2)
            (2,3)

    Block 2:
        [1,2,3,5]

        histories:
            (1,2)
            (2,3)

    Block 3:
        [5,6,7]

        history:
            (5,6)
    """
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [5, 6, 7],
    ]

    k = 2

    stats = calculate_occupancy_stats(
        sequences,
        k,
    )

    # ---------------------------------------------------------------
    # C(1,2) = 2
    # C(2,3) = 2
    # C(5,6) = 1
    # ---------------------------------------------------------------

    # Total history occurrences.
    assert stats["N_k"] == 5

    # Distinct histories.
    assert stats["T_k"] == 3

    # Only (5,6) occurs exactly once.
    assert stats["f_1_k"] == 1

    # Singleton-history rate.
    assert stats["S_k"] == pytest.approx(
        1 / 5
    )

    # Distinct-history rate.
    assert stats["D_k"] == pytest.approx(
        3 / 5
    )


def test_history_counts_match_manual_values():
    """
    Test the raw history-counting layer separately from the derived
    occupancy statistics.
    """
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [5, 6, 7],
    ]

    counts = count_histories(
        sequences,
        k=2,
    )

    assert counts[(1, 2)] == 2
    assert counts[(2, 3)] == 2
    assert counts[(5, 6)] == 1

    assert len(counts) == 3


def test_no_cross_block_history_counts():
    """
    Occupancy statistics must obey the same artificial-boundary rule
    as NGramModel.fit().

    These blocks must not create history (2,) followed by a target in
    the second block.
    """
    sequences = [
        [1, 2],
        [3, 4],
    ]

    counts = count_histories(
        sequences,
        k=1,
    )

    # Only histories that have targets within their own blocks.
    assert counts[(1,)] == 1
    assert counts[(3,)] == 1

    assert counts.get(
        (2,),
        0,
    ) == 0


def test_singleton_example():
    """
    Manual Stage-5 example.

    History tokens before targets:

        1 1 1 1 1 2 2 3 4 5

    Counts:
        1 -> 5
        2 -> 2
        3 -> 1
        4 -> 1
        5 -> 1

    Therefore:

        N_1 = 10
        T_1 = 5
        f_1 = 3
        S_1 = 0.3
        D_1 = 0.5
    """
    sequences = [
        [1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 99],
    ]

    stats = calculate_occupancy_stats(
        sequences,
        k=1,
    )

    assert stats["N_k"] == 10
    assert stats["T_k"] == 5
    assert stats["f_1_k"] == 3

    assert stats["S_k"] == pytest.approx(
        0.3
    )

    assert stats["D_k"] == pytest.approx(
        0.5
    )


def test_invalid_history_length_rejected():
    """
    Negative history lengths are meaningless.
    """
    with pytest.raises(ValueError):
        calculate_occupancy_stats(
            [[1, 2, 3]],
            k=-1,
        )