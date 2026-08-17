from __future__ import annotations

import math

def evaluate_model(
    model,
    eval_sequences: list[list[int]],
    max_history_length: int = 5,
) -> dict:
    """
    Evaluate an n-gram model on held-out data.

    CRITICAL EXPERIMENTAL RULE
    --------------------------
    Every model order must be evaluated on EXACTLY THE SAME target tokens.

    Because the largest model is a 6-gram:

        n_max = 6
        k_max = 5

    every model begins prediction at position 5.

    Therefore:

        H_1, H_2, ..., H_6

    are directly comparable and:

        Delta H_n = H_n - H_(n-1)

    cannot be contaminated by different target sets.
    """

    k = model.n - 1

    if max_history_length < 0:
        raise ValueError(
            "max_history_length must be non-negative."
        )

    if k > max_history_length:
        raise ValueError(
            f"Model requires history length {k}, "
            f"but max_history_length is "
            f"{max_history_length}."
        )

    total_predictions = 0

    # Sum of log q(w|h) over all held-out prediction events.
    sum_log_probability = 0.0

    # Counts how many held-out history OCCURRENCES were completely absent
    # from training.
    unseen_history_occurrences = 0

    for sequence in eval_sequences:

        # ---------------------------------------------------------------
        # Same-target evaluation
        # ---------------------------------------------------------------
        #
        # Even a unigram starts at max_history_length.
        #
        # Example:
        #   max_history_length = 5
        #
        # target indices:
        #   5, 6, 7, ...
        #
        # for EVERY n.
        # ---------------------------------------------------------------
        for target_idx in range(
            max_history_length,
            len(sequence),
        ):

            target = sequence[target_idx]

            # Model uses only the amount of context appropriate for its
            # own order.
            if k == 0:
                history = ()
            else:
                history = tuple(
                    sequence[
                        target_idx - k:
                        target_idx
                    ]
                )

            # Natural logarithm -> nats/token.
            log_probability = model.log_probability(
                history,
                target,
            )

            sum_log_probability += log_probability
            total_predictions += 1

            # -----------------------------------------------------------
            # Actual held-out unseen-history occurrence
            # -----------------------------------------------------------
            #
            # This counts OCCURRENCES rather than merely unique unseen
            # history types.
            #
            # If the same unseen history appears 20 times in test,
            # it contributes 20 unseen prediction events.
            # -----------------------------------------------------------
            if history not in model.history_counts:
                unseen_history_occurrences += 1

    # -------------------------------------------------------------------
    # Guard against empty or extremely short evaluation sequences
    # -------------------------------------------------------------------
    if total_predictions == 0:
        return {
            "cross_entropy": float("inf"),
            "perplexity": float("inf"),
            "U_k": 0.0,
            "N": 0,
        }

    # -------------------------------------------------------------------
    # Empirical held-out cross-entropy
    #
    #          1
    # H = - ------- sum log q(w_t | h_t)
    #          N
    #
    # Units:
    #   nats / token
    # -------------------------------------------------------------------
    cross_entropy = -(
        sum_log_probability
        / total_predictions
    )

    # -------------------------------------------------------------------
    # Perplexity
    #
    # PP = exp(H)
    # -------------------------------------------------------------------
    try:
        perplexity = math.exp(
            cross_entropy
        )
    except OverflowError:
        perplexity = float("inf")

    # -------------------------------------------------------------------
    # Actual held-out unseen-history rate
    #
    #         unseen history occurrences
    # U_k = -------------------------------
    #          held-out prediction events
    # -------------------------------------------------------------------
    u_k = (
        unseen_history_occurrences
        / total_predictions
    )

    return {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "U_k": u_k,
        "N": total_predictions,
    }

def evaluate_per_target_losses(
    model,
    eval_sequences,
    max_history_length=5,
):
    """
    Return per-target negative log-probabilities for aligned evaluation.

    The target positions are identical to evaluate_model().
    """

    k = model.n - 1

    if max_history_length < 0:
        raise ValueError(
            "max_history_length must be non-negative."
        )

    if k > max_history_length:
        raise ValueError(
            "Model history length exceeds max_history_length."
        )

    losses = []

    for sequence in eval_sequences:

        for target_idx in range(
            max_history_length,
            len(sequence),
        ):

            target = sequence[target_idx]

            if k == 0:
                history = ()
            else:
                history = tuple(
                    sequence[
                        target_idx - k:
                        target_idx
                    ]
                )

            loss = -model.log_probability(
                history,
                target,
            )

            losses.append(
                loss
            )

    return losses

def evaluate_per_target_losses(
    model,
    eval_sequences: list[list[int]],
    max_history_length: int = 5,
) -> list[float]:
    """
    Return one negative log-probability for every aligned held-out target.

    Target positions are identical to evaluate_model(), allowing exact
    paired comparisons between different model orders.
    """

    k = model.n - 1

    if max_history_length < 0:
        raise ValueError(
            "max_history_length must be non-negative."
        )

    if k > max_history_length:
        raise ValueError(
            f"Model requires history length {k}, "
            f"but max_history_length is "
            f"{max_history_length}."
        )

    losses = []

    for sequence in eval_sequences:

        for target_idx in range(
            max_history_length,
            len(sequence),
        ):

            target = sequence[target_idx]

            if k == 0:
                history = ()
            else:
                history = tuple(
                    sequence[
                        target_idx - k:
                        target_idx
                    ]
                )

            loss = -model.log_probability(
                history,
                target,
            )

            losses.append(
                loss
            )

    return losses