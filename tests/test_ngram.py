import pytest

from src.ngram import NGramModel


def test_no_cross_block_ngrams():
    """
    Separate training blocks must never generate artificial n-grams
    across their boundaries.

    Blocks:
        [1, 2]
        [3, 4]

    Valid bigrams:
        1 -> 2
        3 -> 4

    Invalid artificial bigram:
        2 -> 3
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }

    sequences = [
        [1, 2],
        [3, 4],
    ]

    model = NGramModel(
        n=2,
        alpha=1.0,
        vocabulary=vocab,
    )

    model.fit(sequences)

    assert model.ngram_counts[((1,), 2)] == 1
    assert model.ngram_counts[((3,), 4)] == 1

    # Critical boundary check.
    assert model.ngram_counts.get(
        ((2,), 3),
        0,
    ) == 0


def test_hand_calculated_counts():
    """
    Verify trigram counts against manually calculated values.

    n = 3
    k = 2
    """
    vocab = {
        "<UNK>": 0,
        "the": 1,
        "king": 2,
        "was": 3,
        "tired": 4,
    }

    sequences = [
        [1, 2, 3, 4],
        [2, 3, 2],
    ]

    # alpha must be strictly positive in the final model.
    # Smoothing does not affect raw count collection.
    model = NGramModel(
        n=3,
        alpha=0.1,
        vocabulary=vocab,
    )

    model.fit(sequences)

    # ---------------------------------------------------------------
    # Sequence 1:
    #
    # (1,2) -> 3
    # (2,3) -> 4
    #
    # Sequence 2:
    #
    # (2,3) -> 2
    # ---------------------------------------------------------------

    assert model.history_counts[(1, 2)] == 1
    assert model.history_counts[(2, 3)] == 2

    assert model.ngram_counts[((1, 2), 3)] == 1
    assert model.ngram_counts[((2, 3), 4)] == 1
    assert model.ngram_counts[((2, 3), 2)] == 1

    assert model.ngram_counts.get(
        ((2, 3), 1),
        0,
    ) == 0

    # Three total prediction events.
    assert model.total_prediction_events == 3


def test_fit_resets_existing_counts():
    """
    Calling fit() twice must replace model state rather than silently
    doubling all counts.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
    }

    sequences = [
        [1, 2, 1],
    ]

    model = NGramModel(
        n=2,
        alpha=1.0,
        vocabulary=vocab,
    )

    model.fit(sequences)

    first_history_counts = model.history_counts.copy()
    first_ngram_counts = model.ngram_counts.copy()

    # Fit the same data again.
    model.fit(sequences)

    assert model.history_counts == first_history_counts
    assert model.ngram_counts == first_ngram_counts


def test_probability_normalization():
    """
    For any valid history:

        sum_w q_alpha(w | h) = 1

    Test both seen and completely unseen histories.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
    }

    sequences = [
        [1, 2, 1],
    ]

    model = NGramModel(
        n=2,
        alpha=0.5,
        vocabulary=vocab,
    )

    model.fit(sequences)

    seen_history_a = (1,)
    seen_history_b = (2,)

    # Token ID 0 never occurred as a training history.
    unseen_history = (0,)

    for history in [
        seen_history_a,
        seen_history_b,
        unseen_history,
    ]:
        prob_sum = sum(
            model.probability(
                history,
                token_id,
            )
            for token_id in vocab.values()
        )

        assert prob_sum == pytest.approx(
            1.0,
            rel=1e-12,
            abs=1e-12,
        )


def test_unseen_history_is_uniform():
    """
    Under positive add-alpha smoothing, if C(h)=0:

        q_alpha(w|h) = 1 / |V|

    for every vocabulary item.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
        "b": 2,
    }

    model = NGramModel(
        n=2,
        alpha=0.5,
        vocabulary=vocab,
    )

    model.fit([
        [1, 2],
    ])

    unseen_history = (0,)

    expected = 1 / len(vocab)

    for token_id in vocab.values():
        assert model.probability(
            unseen_history,
            token_id,
        ) == pytest.approx(expected)


def test_invalid_alpha_rejected():
    """
    alpha <= 0 is deliberately unsupported in this experiment.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
    }

    with pytest.raises(ValueError):
        NGramModel(
            n=2,
            alpha=0.0,
            vocabulary=vocab,
        )


def test_invalid_order_rejected():
    """
    Model order must satisfy n >= 1.
    """
    vocab = {
        "<UNK>": 0,
        "a": 1,
    }

    with pytest.raises(ValueError):
        NGramModel(
            n=0,
            alpha=1.0,
            vocabulary=vocab,
        )