import pytest

from src.preprocess import (
    tokenize,
    build_vocabulary,
    apply_vocabulary,
    encode_tokens,
    make_blocks,
    shuffle_blocks,
    calculate_subset_block_counts,
    get_training_subset,
)


def test_tokenisation():
    """
    Tokenisation should:
    - lowercase text
    - retain internal apostrophes
    - remove ordinary punctuation
    """
    text = "Thou art the King's friend! Don't leave."

    expected = [
        "thou",
        "art",
        "the",
        "king's",
        "friend",
        "don't",
        "leave",
    ]

    assert tokenize(text) == expected


def test_vocabulary_leakage():
    """
    Vocabulary must be constructed using training data only.

    Words appearing only in validation/test data must not enter the
    frozen vocabulary.

    Words below the minimum frequency threshold in training must also
    map to <UNK>.
    """
    d_train = [
        "apple",
        "apple",
        "banana",
    ]

    d_val = [
        "cherry",
        "apple",
    ]

    d_test = [
        "date",
        "banana",
    ]

    vocab = build_vocabulary(
        d_train,
        min_frequency=2,
    )

    # ---------------------------------------------------------------
    # Vocabulary contents
    # ---------------------------------------------------------------
    # apple appears twice -> included
    # banana appears once -> excluded
    # cherry/date never appear in training -> excluded
    # ---------------------------------------------------------------
    assert "apple" in vocab
    assert "banana" not in vocab
    assert "cherry" not in vocab
    assert "date" not in vocab

    assert "<UNK>" in vocab
    assert vocab["<UNK>"] == 0

    # ---------------------------------------------------------------
    # Apply fixed vocabulary
    # ---------------------------------------------------------------
    # NOTE:
    # apply_vocabulary() now returns readable token strings rather
    # than integer IDs.
    # ---------------------------------------------------------------
    val_tokens = apply_vocabulary(
        d_val,
        vocab,
    )

    test_tokens = apply_vocabulary(
        d_test,
        vocab,
    )

    assert val_tokens == [
        "<UNK>",
        "apple",
    ]

    assert test_tokens == [
        "<UNK>",
        "<UNK>",
    ]

    # ---------------------------------------------------------------
    # Encoding happens separately.
    # ---------------------------------------------------------------
    val_ids = encode_tokens(
        val_tokens,
        vocab,
    )

    test_ids = encode_tokens(
        test_tokens,
        vocab,
    )

    assert val_ids == [
        vocab["<UNK>"],
        vocab["apple"],
    ]

    assert test_ids == [
        vocab["<UNK>"],
        vocab["<UNK>"],
    ]


def test_block_creation_preserves_boundaries():
    """
    make_blocks() should split a token sequence into separate blocks
    without flattening them back together.
    """
    tokens = list(range(10))

    blocks = make_blocks(
        tokens,
        block_size=4,
    )

    assert blocks == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_shuffle_is_deterministic():
    """
    A fixed seed must always produce the same shuffled block order.

    shuffle_blocks() uses its own local RNG and therefore should not
    depend on global random state.
    """
    blocks = [
        [i]
        for i in range(100)
    ]

    first = shuffle_blocks(
        blocks,
        seed=42,
    )

    second = shuffle_blocks(
        blocks,
        seed=42,
    )

    assert first == second

    # The function must not modify the original list.
    assert blocks == [
        [i]
        for i in range(100)
    ]


def test_nested_training_subsets():
    """
    Nested training subsets are represented by:

        one shuffled block sequence
        +
        block counts for each fraction.

    Therefore:

        D_5% ⊆ D_10% ⊆ ... ⊆ D_100%
    """
    total_blocks = 100

    blocks = [
        [i]
        for i in range(total_blocks)
    ]

    shuffled_blocks = shuffle_blocks(
        blocks,
        seed=42,
    )

    fractions = [
        0.05,
        0.10,
        0.20,
        0.40,
        0.80,
        1.00,
    ]

    subset_counts = calculate_subset_block_counts(
        total_blocks=len(shuffled_blocks),
        fractions=fractions,
    )

    dataset_payload = {
        "train_blocks": shuffled_blocks,
        "subset_block_counts": subset_counts,
    }

    d_5 = get_training_subset(
        dataset_payload,
        "5%",
    )

    d_10 = get_training_subset(
        dataset_payload,
        "10%",
    )

    d_20 = get_training_subset(
        dataset_payload,
        "20%",
    )

    d_40 = get_training_subset(
        dataset_payload,
        "40%",
    )

    d_80 = get_training_subset(
        dataset_payload,
        "80%",
    )

    d_100 = get_training_subset(
        dataset_payload,
        "100%",
    )

    # ---------------------------------------------------------------
    # Exact expected sizes for 100 blocks
    # ---------------------------------------------------------------
    assert len(d_5) == 5
    assert len(d_10) == 10
    assert len(d_20) == 20
    assert len(d_40) == 40
    assert len(d_80) == 80
    assert len(d_100) == 100

    # ---------------------------------------------------------------
    # Stronger nesting test:
    #
    # Because each subset is a prefix of the same shuffled block list,
    # compare prefixes directly rather than using `b in larger_set`.
    # ---------------------------------------------------------------
    assert d_10[:len(d_5)] == d_5
    assert d_20[:len(d_10)] == d_10
    assert d_40[:len(d_20)] == d_20
    assert d_80[:len(d_40)] == d_40
    assert d_100[:len(d_80)] == d_80