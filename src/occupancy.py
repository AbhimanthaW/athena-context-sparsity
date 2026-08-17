from __future__ import annotations

from collections import Counter


def count_histories(
    sequences: list[list[int]],
    k: int,
) -> Counter:
    """
    Count length-k training histories.

    Only histories that actually precede a target token are counted.

    This makes occupancy counts directly consistent with the histories
    used during n-gram training.

    Example for k = 2:

        sequence:
            a b c d

        counted histories:
            (a,b) -> target c
            (b,c) -> target d

        The final pair (c,d) is NOT counted because no target follows it.
    """
    if k < 0:
        raise ValueError(
            "history length k must be non-negative."
        )

    history_counts = Counter()

    for sequence in sequences:

        # target_idx identifies the target following the history.
        for target_idx in range(k, len(sequence)):

            if k == 0:
                history = ()
            else:
                history = tuple(
                    sequence[
                        target_idx - k:
                        target_idx
                    ]
                )

            history_counts[history] += 1

    return history_counts


def calculate_occupancy_stats(
    sequences: list[list[int]],
    k: int,
) -> dict:
    """
    Calculate the training-only occupancy statistics used in the paper.

    N_k:
        Total number of history occurrences.

    T_k:
        Number of distinct observed histories.

    f_1^(k):
        Number of histories observed exactly once.

    S_k:
        Singleton-history rate.

            S_k = f_1^(k) / N_k

    D_k:
        Distinct-history rate.

            D_k = T_k / N_k

    S_k is our primary Good-Turing-inspired sparsity diagnostic.
    """

    history_counts = count_histories(
        sequences,
        k,
    )

    # ---------------------------------------------------------------
    # Total history occurrences
    # ---------------------------------------------------------------
    n_k = sum(
        history_counts.values()
    )

    # ---------------------------------------------------------------
    # Number of distinct histories
    # ---------------------------------------------------------------
    t_k = len(history_counts)

    # ---------------------------------------------------------------
    # Number of singleton histories
    # ---------------------------------------------------------------
    f_1_k = sum(
        1
        for count in history_counts.values()
        if count == 1
    )

    # Empty / too-short input protection.
    if n_k == 0:
        return {
            "N_k": 0,
            "T_k": 0,
            "f_1_k": 0,
            "S_k": 0.0,
            "D_k": 0.0,
        }

    # ---------------------------------------------------------------
    # Singleton-history rate
    # ---------------------------------------------------------------
    s_k = f_1_k / n_k

    # ---------------------------------------------------------------
    # Distinct-history rate
    # ---------------------------------------------------------------
    d_k = t_k / n_k

    return {
        "N_k": n_k,
        "T_k": t_k,
        "f_1_k": f_1_k,
        "S_k": s_k,
        "D_k": d_k,
    }