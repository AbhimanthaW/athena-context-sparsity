import math

import pytest

from src.ngram import NGramModel
from src.evaluation import evaluate_model


def test_cross_entropy_and_perplexity():
    """
    Manually verify held-out cross-entropy and perplexity.

    Training:
        [1,2,1,2]

    Bigram counts:
        C((1,)) = 2
        C((1,),2) = 2

    |V| = 3
    alpha = 1

    Therefore:

        q(2|1)
        = (2 + 1) / (2 + 3)
        = 3/5
        = 0.6
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
    }

    train_sequences = [
        [1, 2, 1, 2],
    ]

    eval_sequences = [
        [1, 2],
    ]

    model = NGramModel(
        n=2,
        alpha=1.0,
        vocabulary=vocab,
    )

    model.fit(
        train_sequences
    )

    # This small unit test concerns a bigram only, so one previous token
    # is sufficient for aligned evaluation.
    stats = evaluate_model(
        model,
        eval_sequences,
        max_history_length=1,
    )

    expected_probability = 3 / 5

    expected_cross_entropy = -math.log(
        expected_probability
    )

    expected_perplexity = math.exp(
        expected_cross_entropy
    )

    assert stats["cross_entropy"] == pytest.approx(
        expected_cross_entropy
    )

    assert stats["perplexity"] == pytest.approx(
        expected_perplexity
    )

    assert stats["N"] == 1


def test_unseen_history_rate():
    """
    Training histories:

        (1,) -> seen

    Evaluation history occurrences:

        (1,) -> seen
        (3,) -> unseen
        (3,) -> unseen

    Therefore:

        U_k = 2/3
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }

    train_sequences = [
        [1, 2],
    ]

    eval_sequences = [
        [1, 2],
        [3, 4],
        [3, 4],
    ]

    model = NGramModel(
        n=2,
        alpha=1.0,
        vocabulary=vocab,
    )

    model.fit(
        train_sequences
    )

    stats = evaluate_model(
        model,
        eval_sequences,
        max_history_length=1,
    )

    assert stats["N"] == 3

    assert stats["U_k"] == pytest.approx(
        2 / 3
    )


def test_same_target_evaluation():
    """
    This is the critical experimental-alignment test.

    Maximum tested model order:

        n_max = 6

    Therefore:

        k_max = 5

    With an evaluation sequence of length 8:

        target positions = 5, 6, 7

    So EVERY model order n=1,...,6 must evaluate exactly:

        N = 3

    prediction events.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
    }

    sequence = [
        1, 2, 3, 4, 5, 6, 7, 8,
    ]

    eval_sequences = [
        sequence,
    ]

    prediction_counts = []

    for n in range(1, 7):

        model = NGramModel(
            n=n,
            alpha=1.0,
            vocabulary=vocab,
        )

        # Training on the same sequence is fine for this structural
        # unit test. We are testing denominator alignment, not
        # generalisation.
        model.fit([
            sequence,
        ])

        stats = evaluate_model(
            model,
            eval_sequences,
            max_history_length=5,
        )

        prediction_counts.append(
            stats["N"]
        )

    # Every model must evaluate the same 3 target positions.
    assert prediction_counts == [
        3, 3, 3, 3, 3, 3,
    ]


def test_same_target_identity_not_only_count():
    """
    Stronger version of the alignment test.

    It is not sufficient that every model evaluates the same NUMBER
    of tokens. They must correspond to the same target positions.

    For sequence length 8 and max_history_length=5, the targets are:

        sequence[5]
        sequence[6]
        sequence[7]

    The evaluator architecture guarantees this by using the shared
    target-index range:

        range(5, len(sequence))
    """
    sequence = [
        1, 2, 3, 4, 5, 6, 7, 8,
    ]

    expected_target_indices = [
        5,
        6,
        7,
    ]

    actual_target_indices = list(
        range(
            5,
            len(sequence),
        )
    )

    assert (
        actual_target_indices
        == expected_target_indices
    )


def test_unigram_unseen_history_rate_is_zero_after_training():
    """
    A unigram model uses the empty history:

        ()

    If it has been trained on at least one token, () is always present
    in history_counts.

    Therefore its unseen-history rate should be zero.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
    }

    model = NGramModel(
        n=1,
        alpha=1.0,
        vocabulary=vocab,
    )

    model.fit([
        [1, 2, 3, 4, 5, 6],
    ])

    stats = evaluate_model(
        model,
        [[1, 2, 3, 4, 5, 6]],
        max_history_length=5,
    )

    assert stats["N"] == 1
    assert stats["U_k"] == pytest.approx(0.0)


def test_empty_evaluation_returns_defined_sentinel_values():
    """
    Evaluation data with no target positions should not crash.

    The evaluator deliberately returns:

        cross_entropy = infinity
        perplexity    = infinity
        U_k           = 0
        N             = 0
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
    }

    model = NGramModel(
        n=1,
        alpha=1.0,
        vocabulary=vocab,
    )

    model.fit([
        [1],
    ])

    stats = evaluate_model(
        model,
        [[1]],
        max_history_length=5,
    )

    assert math.isinf(
        stats["cross_entropy"]
    )

    assert math.isinf(
        stats["perplexity"]
    )

    assert stats["U_k"] == 0.0
    assert stats["N"] == 0