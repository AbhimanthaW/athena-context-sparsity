from __future__ import annotations

import json
from pathlib import Path

from src.preprocess import get_training_subset
from src.ngram import NGramModel
from src.occupancy import calculate_occupancy_stats
from src.evaluation import evaluate_model


# ---------------------------------------------------------------------------
# Frozen smoke-test condition
# ---------------------------------------------------------------------------
# This is NOT a hyperparameter search and NOT a final experimental result.
#
# Its only purpose is to verify that the complete real-data pipeline works:
#
# processed Shakespeare
#       ↓
# 20% training subset
#       ↓
# trigram model
#       ↓
# occupancy statistics
#       ↓
# validation evaluation
#
TRAINING_SUBSET = "20%"
N = 3
ALPHA = 0.1

# All model orders in the real experiment will be evaluated on targets
# beginning at position 5 because the largest tested history length is 5.
MAX_HISTORY_LENGTH = 5


def get_project_root() -> Path:
    """
    scripts/smoke_test.py lives at:

        <project_root>/scripts/smoke_test.py

    so parents[1] is the repository root.
    """
    return Path(__file__).resolve().parents[1]


def load_json(path: Path):
    """
    Load a JSON file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def invert_vocabulary(vocabulary: dict[str, int]) -> dict[int, str]:
    """
    Convert:

        token -> integer ID

    into:

        integer ID -> token

    so that model histories can be inspected as real Shakespeare words.
    """
    return {
        token_id: token
        for token, token_id in vocabulary.items()
    }


def decode_history(
    history: tuple[int, ...],
    id_to_token: dict[int, str],
) -> str:
    """
    Convert a tuple of integer token IDs into readable text.

    Example:

        (51, 812)
            ↓
        "the king"
    """
    return " ".join(
        id_to_token.get(token_id, "<UNKNOWN_ID>")
        for token_id in history
    )


def inspect_common_histories(
    model: NGramModel,
    id_to_token: dict[int, str],
    n_histories: int = 3,
    n_successors: int = 5,
) -> None:
    """
    Print several common training histories and their most common successors.

    This is a manual sanity check.

    We want to see plausible language statistics rather than malformed
    histories, accidental cross-block events, or mysterious integer soup.
    """
    print("\n--- Manual History Inspection ---")

    most_common_histories = model.history_counts.most_common(
        n_histories
    )

    for history, history_count in most_common_histories:
        decoded_history = decode_history(
            history,
            id_to_token,
        )

        print(
        f'\nHistory: "{decoded_history}" '
        f"(count={history_count})"
        )

        # Collect all target tokens observed after this history.
        successors = []

        for (stored_history, token_id), count in model.ngram_counts.items():

            if stored_history == history:
                successors.append(
                    (
                        token_id,
                        count,
                    )
                )

        # Sort successor words by descending count.
        successors.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        print("Top successors:")

        for token_id, count in successors[:n_successors]:

            token = id_to_token.get(
                token_id,
                "<UNKNOWN_ID>",
            )

            print(
                f"  {token:<15} {count}"
            )


def main() -> None:
    project_root = get_project_root()

    processed_dir = (
        project_root
        / "data"
        / "processed"
    )

    dataset_path = (
        processed_dir
        / "dataset_blocks.json"
    )

    vocab_path = (
        processed_dir
        / "vocab.json"
    )

    # ---------------------------------------------------------------------
    # 1. Load processed dataset + frozen vocabulary
    # ---------------------------------------------------------------------
    print("Loading processed dataset...")

    dataset = load_json(
        dataset_path
    )

    vocabulary = load_json(
        vocab_path
    )

    id_to_token = invert_vocabulary(
        vocabulary
    )

    # ---------------------------------------------------------------------
    # 2. Reconstruct frozen 20% nested training subset
    # ---------------------------------------------------------------------
    training_sequences = get_training_subset(
        dataset,
        TRAINING_SUBSET,
    )

    validation_sequences = dataset[
        "val_sequences"
    ]

    training_token_count = sum(
        len(sequence)
        for sequence in training_sequences
    )

    validation_token_count = sum(
        len(sequence)
        for sequence in validation_sequences
    )

    print(
        f"Training subset: {TRAINING_SUBSET}"
    )

    print(
        f"Training tokens: {training_token_count:,}"
    )

    print(
        f"Validation tokens: {validation_token_count:,}"
    )

    # ---------------------------------------------------------------------
    # 3. Train the fixed smoke-test model
    # ---------------------------------------------------------------------
    print(
        f"\nTraining {N}-gram model "
        f"with alpha={ALPHA}..."
    )

    model = NGramModel(
        n=N,
        alpha=ALPHA,
        vocabulary=vocabulary,
    )

    model.fit(
        training_sequences
    )

    # n = 3 -> k = 2
    k = N - 1

    # ---------------------------------------------------------------------
    # 4. Compute TRAINING-ONLY occupancy statistics
    # ---------------------------------------------------------------------
    occupancy = calculate_occupancy_stats(
        training_sequences,
        k=k,
    )

    # ---------------------------------------------------------------------
    # 5. Evaluate on VALIDATION only
    # ---------------------------------------------------------------------
    #
    # We deliberately do not touch the test set.
    #
    # max_history_length=5 ensures the same evaluation target positions
    # will later be used for n=1,...,6.
    # ---------------------------------------------------------------------
    validation_stats = evaluate_model(
        model,
        validation_sequences,
        max_history_length=MAX_HISTORY_LENGTH,
    )

    # ---------------------------------------------------------------------
    # 6. Print smoke-test result
    # ---------------------------------------------------------------------
    print("\n--- Smoke Test Results ---")

    print(
        f"Model order n:              {N}"
    )

    print(
        f"History length k:           {k}"
    )

    print(
        f"Alpha:                      {ALPHA}"
    )

    print(
        f"Training tokens:            "
        f"{training_token_count:,}"
    )

    print()

    print(
        f"N_k history occurrences:    "
        f"{occupancy['N_k']:,}"
    )

    print(
        f"T_k distinct histories:     "
        f"{occupancy['T_k']:,}"
    )

    print(
        f"f_1 singleton histories:    "
        f"{occupancy['f_1_k']:,}"
    )

    print(
        f"S_k singleton rate:         "
        f"{occupancy['S_k']:.6f}"
    )

    print(
        f"D_k distinct-history rate: "
        f"{occupancy['D_k']:.6f}"
    )

    print()

    print(
        f"Validation cross-entropy:    "
        f"{validation_stats['cross_entropy']:.6f} "
        f"nats/token"
    )

    print(
        f"Validation perplexity:       "
        f"{validation_stats['perplexity']:.6f}"
    )

    print(
        f"Validation unseen rate U_k: "
        f"{validation_stats['U_k']:.6f}"
    )

    print(
        f"Evaluation targets N:        "
        f"{validation_stats['N']:,}"
    )

    # ---------------------------------------------------------------------
    # 7. Basic invariant checks
    # ---------------------------------------------------------------------
    #
    # These do not replace unit tests.
    # They simply catch impossible real-data outputs immediately.
    # ---------------------------------------------------------------------
    assert 0.0 <= occupancy["S_k"] <= 1.0
    assert 0.0 <= occupancy["D_k"] <= 1.0
    assert 0.0 <= validation_stats["U_k"] <= 1.0

    assert validation_stats["cross_entropy"] > 0
    assert validation_stats["perplexity"] > 0

    # With one validation sequence of 98,844 tokens and shared target
    # offset 5:
    #
    # expected N = 98,844 - 5 = 98,839
    expected_targets = sum(
        max(
            0,
            len(sequence) - MAX_HISTORY_LENGTH,
        )
        for sequence in validation_sequences
    )

    assert (
        validation_stats["N"]
        == expected_targets
    ), (
        f"Expected {expected_targets:,} evaluation targets, "
        f"got {validation_stats['N']:,}."
    )

    print(
        "\nAll real-data smoke-test invariants passed."
    )

    # ---------------------------------------------------------------------
    # 8. Human-readable history inspection
    # ---------------------------------------------------------------------
    inspect_common_histories(
        model,
        id_to_token,
        n_histories=3,
        n_successors=5,
    )


if __name__ == "__main__":
    main()