from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.preprocess import get_training_subset
from src.ngram import NGramModel
from src.occupancy import calculate_occupancy_stats
from src.evaluation import evaluate_model


# ===========================================================================
# PROJECT PATHS
# ===========================================================================

def get_project_root() -> Path:
    """
    Return the repository root.

    experiment.py lives at:

        <project_root>/src/experiment.py

    so parents[1] is the repository root.
    """
    return Path(__file__).resolve().parents[1]


# ===========================================================================
# FILE LOADING
# ===========================================================================

def load_json(path: Path):
    """
    Load a JSON file using UTF-8 encoding.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ===========================================================================
# CONFIGURATION VALIDATION
# ===========================================================================

def validate_experiment_config(config: dict) -> None:
    """
    Validate the fields required by the validation experiment.

    We fail before running any models if the frozen configuration is
    incomplete or mathematically invalid.
    """
    required_fields = {
        "training_subset_fractions",
        "orders",
        "alphas",
        "max_history_length",
    }

    missing = required_fields - config.keys()

    if missing:
        raise ValueError(
            f"Missing experiment config fields: {sorted(missing)}"
        )

    # ----------------------------------------------------------------------
    # Training fractions
    # ----------------------------------------------------------------------
    for fraction in config["training_subset_fractions"]:
        if not 0 < fraction <= 1:
            raise ValueError(
                "Every training subset fraction must lie in (0, 1]."
            )

    # ----------------------------------------------------------------------
    # Model orders
    # ----------------------------------------------------------------------
    for n in config["orders"]:
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                "Every n-gram order must be a positive integer."
            )

    # ----------------------------------------------------------------------
    # Smoothing strengths
    # ----------------------------------------------------------------------
    for alpha in config["alphas"]:
        if alpha <= 0:
            raise ValueError(
                "Every alpha must be strictly positive."
            )

    # ----------------------------------------------------------------------
    # Evaluation alignment
    # ----------------------------------------------------------------------
    max_history_length = config["max_history_length"]

    largest_required_history = max(config["orders"]) - 1

    if max_history_length < largest_required_history:
        raise ValueError(
            "max_history_length is too small for the largest model order."
        )


# ===========================================================================
# TRAINING-SUBSET LABELS
# ===========================================================================

def fraction_to_label(fraction: float) -> str:
    """
    Convert a numeric fraction into the labels used by preprocessing.

    Example:

        0.05 -> "5%"
        0.20 -> "20%"
        1.00 -> "100%"
    """
    return f"{int(round(fraction * 100))}%"


# ===========================================================================
# CSV OUTPUT
# ===========================================================================

def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """
    Write rows to CSV using an explicit, stable column order.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# RESULT SANITY CHECKS
# ===========================================================================

def validate_result_row(row: dict) -> None:
    """
    Check invariants that must hold for every validation result.

    An impossible result should stop the experiment immediately rather
    than quietly entering the final CSV.
    """

    # Cross-entropy must be finite.
    if not math.isfinite(
        row["validation_cross_entropy"]
    ):
        raise RuntimeError(
            "Non-finite validation cross-entropy detected."
        )

    # Perplexity must also be finite and positive.
    if not math.isfinite(
        row["validation_perplexity"]
    ):
        raise RuntimeError(
            "Non-finite validation perplexity detected."
        )

    if row["validation_perplexity"] <= 0:
        raise RuntimeError(
            "Validation perplexity must be positive."
        )

    # Occupancy / coverage rates are probabilities or proportions.
    for field in [
        "S_k",
        "D_k",
        "validation_U_k",
    ]:
        value = row[field]

        if not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"{field} must lie in [0,1], got {value}."
            )

    # Training history events should exist.
    if row["N_k"] <= 0:
        raise RuntimeError(
            "N_k must be positive."
        )

    # Evaluation must contain targets.
    if row["validation_N"] <= 0:
        raise RuntimeError(
            "Validation evaluation contained no targets."
        )


# ===========================================================================
# HYPERPARAMETER SELECTION
# ===========================================================================

def select_best_alphas(
    validation_rows: list[dict],
) -> list[dict]:
    """
    Select alpha independently for every (training size, n) condition.

    Selection rule:

        alpha*(m,n)
        =
        argmin_alpha H_val(m,n,alpha)

    IMPORTANT:
    Test data plays no role here.

    Exact ties are resolved deterministically by choosing the smaller alpha.
    This tie-break rule is arbitrary but fixed before test evaluation.
    """

    grouped = defaultdict(list)

    for row in validation_rows:

        key = (
            row["training_fraction"],
            row["training_subset"],
            row["n"],
        )

        grouped[key].append(row)

    selected_rows = []

    for key, candidates in grouped.items():

        # Primary criterion:
        #   lowest validation cross-entropy
        #
        # Secondary deterministic tie-break:
        #   smaller alpha
        best = min(
            candidates,
            key=lambda row: (
                row["validation_cross_entropy"],
                row["alpha"],
            ),
        )

        selected_rows.append(
            {
                "training_fraction":
                    best["training_fraction"],

                "training_subset":
                    best["training_subset"],

                "training_tokens":
                    best["training_tokens"],

                "n":
                    best["n"],

                "k":
                    best["k"],

                "selected_alpha":
                    best["alpha"],

                "validation_cross_entropy":
                    best["validation_cross_entropy"],

                "validation_perplexity":
                    best["validation_perplexity"],

                "validation_U_k":
                    best["validation_U_k"],

                "validation_N":
                    best["validation_N"],

                # Occupancy values do not depend on alpha,
                # but retaining them here makes this file useful
                # for auditing each final selected condition.
                "N_k":
                    best["N_k"],

                "T_k":
                    best["T_k"],

                "f_1_k":
                    best["f_1_k"],

                "S_k":
                    best["S_k"],

                "D_k":
                    best["D_k"],
            }
        )

    # Stable scientifically meaningful ordering.
    selected_rows.sort(
        key=lambda row: (
            row["training_fraction"],
            row["n"],
        )
    )

    return selected_rows


# ===========================================================================
# MAIN VALIDATION SWEEP
# ===========================================================================

def main() -> None:
    project_root = get_project_root()

    # ----------------------------------------------------------------------
    # Input paths
    # ----------------------------------------------------------------------
    config_path = (
        project_root
        / "experiments"
        / "config.json"
    )

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "dataset_blocks.json"
    )

    vocab_path = (
        project_root
        / "data"
        / "processed"
        / "vocab.json"
    )

    # ----------------------------------------------------------------------
    # Output paths
    # ----------------------------------------------------------------------
    results_dir = (
        project_root
        / "results"
    )

    validation_results_path = (
        results_dir
        / "validation_results.csv"
    )

    selected_hyperparameters_path = (
        results_dir
        / "selected_hyperparameters.csv"
    )

    # ----------------------------------------------------------------------
    # Load frozen experiment state
    # ----------------------------------------------------------------------
    print("Loading experiment configuration and processed corpus...")

    config = load_json(
        config_path
    )

    validate_experiment_config(
        config
    )

    dataset = load_json(
        dataset_path
    )

    vocabulary = load_json(
        vocab_path
    )

    # ----------------------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally load ONLY validation data.
    #
    # dataset["test_sequences"] is never read anywhere in this script.
    # ----------------------------------------------------------------------
    validation_sequences = dataset[
        "val_sequences"
    ]

    training_fractions = config[
        "training_subset_fractions"
    ]

    orders = config[
        "orders"
    ]

    alphas = config[
        "alphas"
    ]

    max_history_length = config[
        "max_history_length"
    ]

    expected_conditions = (
        len(training_fractions)
        * len(orders)
        * len(alphas)
    )

    print(
        f"Validation conditions to run: "
        f"{expected_conditions}"
    )

    print(
        f"Training sizes: {len(training_fractions)}"
    )

    print(
        f"Model orders: {len(orders)}"
    )

    print(
        f"Alpha values: {len(alphas)}"
    )

    print()

    # ----------------------------------------------------------------------
    # Results accumulated in memory before writing.
    # ----------------------------------------------------------------------
    validation_rows = []

    # This will be set by the first model and then checked for all later
    # models to guarantee same-target evaluation across the entire sweep.
    shared_validation_n = None

    condition_number = 0

    # ======================================================================
    # LOOP 1: TRAINING SIZE
    # ======================================================================
    for training_fraction in training_fractions:

        subset_label = fraction_to_label(
            training_fraction
        )

        training_sequences = get_training_subset(
            dataset,
            subset_label,
        )

        training_tokens = sum(
            len(sequence)
            for sequence in training_sequences
        )

        print(
            f"\n=== Training subset {subset_label} "
            f"({training_tokens:,} tokens) ==="
        )

        # ==================================================================
        # LOOP 2: MODEL ORDER
        # ==================================================================
        for n in orders:

            k = n - 1

            # --------------------------------------------------------------
            # Training occupancy depends only on:
            #
            #   training subset
            #   history length
            #
            # It does NOT depend on alpha.
            #
            # Therefore calculate it once per (m,n), not four times.
            # --------------------------------------------------------------
            occupancy = calculate_occupancy_stats(
                training_sequences,
                k=k,
            )

            print(
                f"\n  n={n}, k={k} | "
                f"S_k={occupancy['S_k']:.6f} | "
                f"D_k={occupancy['D_k']:.6f}"
            )

            # ==============================================================
            # LOOP 3: SMOOTHING STRENGTH
            # ==============================================================
            for alpha in alphas:

                condition_number += 1

                print(
                    f"    "
                    f"[{condition_number:>3}/"
                    f"{expected_conditions}] "
                    f"alpha={alpha}"
                )

                # ----------------------------------------------------------
                # Train a fresh model for this condition.
                #
                # Although counts are mathematically independent of alpha,
                # refitting keeps each condition self-contained and avoids
                # mutating model state between hyperparameter evaluations.
                #
                # For this corpus size, the additional computation is
                # acceptable and methodological clarity is preferable.
                # ----------------------------------------------------------
                model = NGramModel(
                    n=n,
                    alpha=alpha,
                    vocabulary=vocabulary,
                )

                model.fit(
                    training_sequences
                )

                # ----------------------------------------------------------
                # Internal consistency:
                #
                # NGramModel.fit() and occupancy counting are supposed to
                # represent exactly the same history-target events.
                # ----------------------------------------------------------
                if (
                    model.total_prediction_events
                    != occupancy["N_k"]
                ):
                    raise RuntimeError(
                        "Mismatch between model training events "
                        "and occupancy N_k."
                    )

                # ----------------------------------------------------------
                # VALIDATION ONLY.
                #
                # Test data remains untouched.
                # ----------------------------------------------------------
                validation_stats = evaluate_model(
                    model,
                    validation_sequences,
                    max_history_length=max_history_length,
                )

                # ----------------------------------------------------------
                # Verify all 144 conditions use exactly the same number
                # of validation targets.
                # ----------------------------------------------------------
                if shared_validation_n is None:
                    shared_validation_n = (
                        validation_stats["N"]
                    )

                elif (
                    validation_stats["N"]
                    != shared_validation_n
                ):
                    raise RuntimeError(
                        "Validation target count changed across "
                        "experimental conditions."
                    )

                row = {
                    # ------------------------------------------------------
                    # Independent variables
                    # ------------------------------------------------------
                    "training_fraction":
                        training_fraction,

                    "training_subset":
                        subset_label,

                    "training_tokens":
                        training_tokens,

                    "n":
                        n,

                    "k":
                        k,

                    "alpha":
                        alpha,

                    # ------------------------------------------------------
                    # TRAINING-ONLY sparsity statistics
                    # ------------------------------------------------------
                    "N_k":
                        occupancy["N_k"],

                    "T_k":
                        occupancy["T_k"],

                    "f_1_k":
                        occupancy["f_1_k"],

                    "S_k":
                        occupancy["S_k"],

                    "D_k":
                        occupancy["D_k"],

                    # ------------------------------------------------------
                    # VALIDATION outcomes
                    # ------------------------------------------------------
                    "validation_cross_entropy":
                        validation_stats[
                            "cross_entropy"
                        ],

                    "validation_perplexity":
                        validation_stats[
                            "perplexity"
                        ],

                    "validation_U_k":
                        validation_stats[
                            "U_k"
                        ],

                    "validation_N":
                        validation_stats[
                            "N"
                        ],
                }

                validate_result_row(
                    row
                )

                validation_rows.append(
                    row
                )

    # ======================================================================
    # COMPLETENESS CHECK
    # ======================================================================
    if len(validation_rows) != expected_conditions:
        raise RuntimeError(
            f"Expected {expected_conditions} validation rows, "
            f"but produced {len(validation_rows)}."
        )

    # ======================================================================
    # WRITE ALL 144 VALIDATION RESULTS
    # ======================================================================
    validation_fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "n",
        "k",
        "alpha",
        "N_k",
        "T_k",
        "f_1_k",
        "S_k",
        "D_k",
        "validation_cross_entropy",
        "validation_perplexity",
        "validation_U_k",
        "validation_N",
    ]

    write_csv(
        validation_results_path,
        validation_rows,
        validation_fieldnames,
    )

    # ======================================================================
    # SELECT ONE ALPHA FOR EACH (m,n)
    # ======================================================================
    selected_rows = select_best_alphas(
        validation_rows
    )

    expected_selected_conditions = (
        len(training_fractions)
        * len(orders)
    )

    if (
        len(selected_rows)
        != expected_selected_conditions
    ):
        raise RuntimeError(
            f"Expected {expected_selected_conditions} "
            f"selected conditions, "
            f"but produced {len(selected_rows)}."
        )

    selected_fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "n",
        "k",
        "selected_alpha",
        "validation_cross_entropy",
        "validation_perplexity",
        "validation_U_k",
        "validation_N",
        "N_k",
        "T_k",
        "f_1_k",
        "S_k",
        "D_k",
    ]

    write_csv(
        selected_hyperparameters_path,
        selected_rows,
        selected_fieldnames,
    )

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    print("\n============================================")
    print("VALIDATION SWEEP COMPLETE")
    print("============================================")

    print(
        f"Validation rows: "
        f"{len(validation_rows)}"
    )

    print(
        f"Selected (m,n) conditions: "
        f"{len(selected_rows)}"
    )

    print(
        f"Shared validation targets per model: "
        f"{shared_validation_n:,}"
    )

    print(
        "\nFull validation results:"
    )

    print(
        validation_results_path
    )

    print(
        "\nFrozen hyperparameter selections:"
    )

    print(
        selected_hyperparameters_path
    )

    print(
        "\nIMPORTANT: "
        "The test set has not been evaluated by this script."
    )


if __name__ == "__main__":
    main()