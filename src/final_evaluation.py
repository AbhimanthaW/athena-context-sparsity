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
    final_evaluation.py lives at:

        <project_root>/src/final_evaluation.py

    so parents[1] is the repository root.
    """
    return Path(__file__).resolve().parents[1]


# ===========================================================================
# GENERIC FILE HELPERS
# ===========================================================================

def load_json(path: Path):
    """
    Load a UTF-8 JSON file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """
    Write rows to a CSV file using a fixed column order.
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
# TRAINING-SUBSET LABELS
# ===========================================================================

def fraction_to_label(fraction: float) -> str:
    """
    Convert:

        0.05 -> "5%"
        0.20 -> "20%"
        1.00 -> "100%"
    """
    return f"{int(round(fraction * 100))}%"


# ===========================================================================
# LOAD FROZEN HYPERPARAMETERS
# ===========================================================================

def load_selected_hyperparameters(
    path: Path,
) -> list[dict]:
    """
    Read the frozen validation-selected hyperparameters.

    This file is the ONLY source of smoothing choices during final
    test evaluation.

    There is deliberately no hyperparameter-selection logic in this script.
    """

    required_fields = {
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
    }

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "selected_hyperparameters.csv has no header."
            )

        missing = required_fields - set(
            reader.fieldnames
        )

        if missing:
            raise ValueError(
                "selected_hyperparameters.csv is missing "
                f"fields: {sorted(missing)}"
            )

        for raw in reader:

            rows.append(
                {
                    "training_fraction":
                        float(raw["training_fraction"]),

                    "training_subset":
                        raw["training_subset"],

                    "training_tokens":
                        int(raw["training_tokens"]),

                    "n":
                        int(raw["n"]),

                    "k":
                        int(raw["k"]),

                    "selected_alpha":
                        float(raw["selected_alpha"]),

                    # Validation values are retained only for provenance.
                    # They are NOT used to make any new choices.
                    "validation_cross_entropy":
                        float(raw["validation_cross_entropy"]),

                    "validation_perplexity":
                        float(raw["validation_perplexity"]),

                    "validation_U_k":
                        float(raw["validation_U_k"]),

                    "validation_N":
                        int(raw["validation_N"]),

                    "N_k":
                        int(raw["N_k"]),

                    "T_k":
                        int(raw["T_k"]),

                    "f_1_k":
                        int(raw["f_1_k"]),

                    "S_k":
                        float(raw["S_k"]),

                    "D_k":
                        float(raw["D_k"]),
                }
            )

    return rows


# ===========================================================================
# VALIDATE THE FROZEN SELECTION FILE
# ===========================================================================

def validate_selected_hyperparameters(
    rows: list[dict],
    config: dict,
) -> None:
    """
    Verify that the frozen hyperparameter file contains exactly one
    selection for every preregistered (training size, model order).

    This prevents the test script from silently skipping, duplicating,
    or inventing experimental conditions.
    """

    fractions = config[
        "training_subset_fractions"
    ]

    orders = config[
        "orders"
    ]

    alphas = config[
        "alphas"
    ]

    expected_pairs = {
        (
            fraction_to_label(fraction),
            n,
        )
        for fraction in fractions
        for n in orders
    }

    actual_pairs = [
        (
            row["training_subset"],
            row["n"],
        )
        for row in rows
    ]

    # Exactly 36 conditions should exist.
    if len(actual_pairs) != len(expected_pairs):
        raise ValueError(
            f"Expected {len(expected_pairs)} frozen conditions, "
            f"found {len(actual_pairs)}."
        )

    # Duplicates would mean one condition appears more than once.
    if len(set(actual_pairs)) != len(actual_pairs):
        raise ValueError(
            "Duplicate (training subset, n) conditions detected."
        )

    if set(actual_pairs) != expected_pairs:
        raise ValueError(
            "Frozen hyperparameter conditions do not match "
            "the configured experimental grid."
        )

    for row in rows:

        # History length must be consistent with model order.
        if row["k"] != row["n"] - 1:
            raise ValueError(
                f"Invalid k for n={row['n']}."
            )

        # Selected alpha must come from the frozen validation grid.
        alpha_is_valid = any(
            math.isclose(
                row["selected_alpha"],
                allowed_alpha,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for allowed_alpha in alphas
        )

        if not alpha_is_valid:
            raise ValueError(
                f"Selected alpha {row['selected_alpha']} "
                "is not in the frozen alpha grid."
            )


# ===========================================================================
# OCCUPANCY CONSISTENCY
# ===========================================================================

def validate_frozen_occupancy(
    selected_row: dict,
    occupancy: dict,
) -> None:
    """
    Recompute the training-only occupancy statistics during final evaluation
    and verify that they exactly agree with the frozen validation-stage file.

    This is an important reproducibility check:

    if the training data or preprocessing somehow changed after validation,
    this function should stop the test run.
    """

    integer_fields = [
        "N_k",
        "T_k",
        "f_1_k",
    ]

    for field in integer_fields:
        if occupancy[field] != selected_row[field]:
            raise RuntimeError(
                f"Frozen {field} mismatch for "
                f"{selected_row['training_subset']}, "
                f"n={selected_row['n']}: "
                f"expected {selected_row[field]}, "
                f"got {occupancy[field]}."
            )

    float_fields = [
        "S_k",
        "D_k",
    ]

    for field in float_fields:
        if not math.isclose(
            occupancy[field],
            selected_row[field],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Frozen {field} mismatch for "
                f"{selected_row['training_subset']}, "
                f"n={selected_row['n']}."
            )


# ===========================================================================
# TEST RESULT VALIDATION
# ===========================================================================

def validate_test_row(row: dict) -> None:
    """
    Reject mathematically impossible or structurally invalid final results.
    """

    if not math.isfinite(
        row["test_cross_entropy"]
    ):
        raise RuntimeError(
            "Non-finite test cross-entropy detected."
        )

    if not math.isfinite(
        row["test_perplexity"]
    ):
        raise RuntimeError(
            "Non-finite test perplexity detected."
        )

    if row["test_cross_entropy"] < 0:
        raise RuntimeError(
            "Cross-entropy cannot be negative."
        )

    if row["test_perplexity"] <= 0:
        raise RuntimeError(
            "Perplexity must be positive."
        )

    for field in [
        "S_k",
        "D_k",
        "test_U_k",
    ]:
        if not 0.0 <= row[field] <= 1.0:
            raise RuntimeError(
                f"{field} must lie in [0,1]."
            )

    if row["N_k"] <= 0:
        raise RuntimeError(
            "N_k must be positive."
        )

    if row["test_N"] <= 0:
        raise RuntimeError(
            "Test evaluation contains no target events."
        )


# ===========================================================================
# TRANSITION TABLE
# ===========================================================================

def build_transitions(
    main_rows: list[dict],
) -> list[dict]:
    """
    Convert the 36 model-level final results into the 30 order-transition
    observations used by the main research question.

    For each fixed training size:

        Delta H_n
        =
        H_n - H_(n-1)

    for:

        n = 2,3,4,5,6.

    Interpretation:

        Delta H_n < 0
            increasing context improved held-out performance

        Delta H_n > 0
            increasing context hurt held-out performance

    The sparsity statistic paired with Delta H_n is:

        S_(n-1)

    which is exactly the S_k value of the higher-order model where:

        k = n - 1.
    """

    grouped = defaultdict(list)

    for row in main_rows:
        grouped[
            row["training_subset"]
        ].append(row)

    transitions = []

    for training_subset, rows in grouped.items():

        ordered = sorted(
            rows,
            key=lambda row: row["n"],
        )

        # The primary experiment expects n = 1,...,6.
        observed_orders = [
            row["n"]
            for row in ordered
        ]

        expected_orders = list(
            range(
                min(observed_orders),
                max(observed_orders) + 1,
            )
        )

        if observed_orders != expected_orders:
            raise ValueError(
                f"Non-consecutive model orders for "
                f"{training_subset}: {observed_orders}"
            )

        # Compare each model with the immediately lower order.
        for index in range(
            1,
            len(ordered),
        ):

            previous = ordered[
                index - 1
            ]

            current = ordered[
                index
            ]

            if current["n"] != previous["n"] + 1:
                raise ValueError(
                    "Transitions must compare consecutive model orders."
                )

            delta_h = (
                current["test_cross_entropy"]
                - previous["test_cross_entropy"]
            )

            transitions.append(
                {
                    "training_fraction":
                        current["training_fraction"],

                    "training_subset":
                        training_subset,

                    "training_tokens":
                        current["training_tokens"],

                    "from_order":
                        previous["n"],

                    "to_order":
                        current["n"],

                    # k of the higher-order model.
                    #
                    # This is the history whose occupancy is being used
                    # to predict the value of increasing model order.
                    "history_length":
                        current["k"],

                    "selected_alpha_previous":
                        previous["selected_alpha"],

                    "selected_alpha_current":
                        current["selected_alpha"],

                    # Primary training-only predictor.
                    "S_k":
                        current["S_k"],

                    # Cheap alternative sparsity baseline.
                    "D_k":
                        current["D_k"],

                    # Actual held-out coverage measure.
                    "U_k":
                        current["test_U_k"],

                    # Cross-entropies defining the transition.
                    "H_previous":
                        previous["test_cross_entropy"],

                    "H_current":
                        current["test_cross_entropy"],

                    # CENTRAL OUTCOME.
                    "delta_H":
                        delta_h,
                }
            )

    transitions.sort(
        key=lambda row: (
            row["training_fraction"],
            row["to_order"],
        )
    )

    return transitions


# ===========================================================================
# MAIN FINAL TEST EVALUATION
# ===========================================================================

def main() -> None:
    project_root = get_project_root()

    # ----------------------------------------------------------------------
    # Frozen inputs
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

    selected_path = (
        project_root
        / "results"
        / "selected_hyperparameters.csv"
    )

    # ----------------------------------------------------------------------
    # Final outputs
    # ----------------------------------------------------------------------
    results_dir = (
        project_root
        / "results"
    )

    main_results_path = (
        results_dir
        / "main_results.csv"
    )

    transitions_path = (
        results_dir
        / "transitions.csv"
    )

    # ----------------------------------------------------------------------
    # Load experiment state
    # ----------------------------------------------------------------------
    print(
        "Loading frozen experiment state..."
    )

    config = load_json(
        config_path
    )

    dataset = load_json(
        dataset_path
    )

    vocabulary = load_json(
        vocab_path
    )

    selected_rows = (
        load_selected_hyperparameters(
            selected_path
        )
    )

    validate_selected_hyperparameters(
        selected_rows,
        config,
    )

    # ----------------------------------------------------------------------
    # THE TEST SET IS UNLOCKED HERE.
    #
    # No smoothing search occurs below this point.
    # Every alpha was already frozen using validation data.
    # ----------------------------------------------------------------------
    test_sequences = dataset[
        "test_sequences"
    ]

    max_history_length = config[
        "max_history_length"
    ]

    # Number of common target positions that SHOULD be evaluated.
    expected_test_targets = sum(
        max(
            0,
            len(sequence) - max_history_length,
        )
        for sequence in test_sequences
    )

    print(
        f"Frozen model conditions: "
        f"{len(selected_rows)}"
    )

    print(
        f"Expected shared test targets: "
        f"{expected_test_targets:,}"
    )

    print(
        "\nBeginning locked test-set evaluation..."
    )

    # Stable execution order.
    selected_rows.sort(
        key=lambda row: (
            row["training_fraction"],
            row["n"],
        )
    )

    main_rows = []

    shared_test_n = None

    total_conditions = len(
        selected_rows
    )

    # ======================================================================
    # EXACTLY ONE TEST MODEL PER FROZEN CONDITION
    # ======================================================================
    for condition_number, selected in enumerate(
        selected_rows,
        start=1,
    ):

        training_subset = selected[
            "training_subset"
        ]

        n = selected["n"]
        k = selected["k"]

        alpha = selected[
            "selected_alpha"
        ]

        # ------------------------------------------------------------------
        # Reconstruct the same training subset used during validation.
        # ------------------------------------------------------------------
        training_sequences = get_training_subset(
            dataset,
            training_subset,
        )

        training_tokens = sum(
            len(sequence)
            for sequence in training_sequences
        )

        if (
            training_tokens
            != selected["training_tokens"]
        ):
            raise RuntimeError(
                f"Training-token mismatch for "
                f"{training_subset}, n={n}."
            )

        # ------------------------------------------------------------------
        # Recompute TRAINING-ONLY occupancy.
        # ------------------------------------------------------------------
        occupancy = calculate_occupancy_stats(
            training_sequences,
            k=k,
        )

        # Ensure training data are bit-for-bit structurally consistent with
        # the validation-stage experiment.
        validate_frozen_occupancy(
            selected,
            occupancy,
        )

        print(
            f"[{condition_number:>2}/{total_conditions}] "
            f"{training_subset:>4} | "
            f"n={n} | "
            f"alpha={alpha}"
        )

        # ------------------------------------------------------------------
        # Train exactly ONE model using the frozen alpha.
        # ------------------------------------------------------------------
        model = NGramModel(
            n=n,
            alpha=alpha,
            vocabulary=vocabulary,
        )

        model.fit(
            training_sequences
        )

        # The model and occupancy layer must count the same training events.
        if (
            model.total_prediction_events
            != occupancy["N_k"]
        ):
            raise RuntimeError(
                "Model-event count does not match occupancy N_k."
            )

        # ------------------------------------------------------------------
        # FINAL TEST EVALUATION.
        #
        # This is the first use of the test partition.
        # ------------------------------------------------------------------
        test_stats = evaluate_model(
            model,
            test_sequences,
            max_history_length=max_history_length,
        )

        # ------------------------------------------------------------------
        # Same-target check across all 36 final models.
        # ------------------------------------------------------------------
        if shared_test_n is None:
            shared_test_n = test_stats[
                "N"
            ]

        elif (
            test_stats["N"]
            != shared_test_n
        ):
            raise RuntimeError(
                "Test target count changed across model conditions."
            )

        if (
            test_stats["N"]
            != expected_test_targets
        ):
            raise RuntimeError(
                f"Expected {expected_test_targets} test targets, "
                f"got {test_stats['N']}."
            )

        row = {
            # --------------------------------------------------------------
            # Experimental condition
            # --------------------------------------------------------------
            "training_fraction":
                selected["training_fraction"],

            "training_subset":
                training_subset,

            "training_tokens":
                training_tokens,

            "n":
                n,

            "k":
                k,

            "selected_alpha":
                alpha,

            # --------------------------------------------------------------
            # Frozen validation metric retained for provenance only.
            # --------------------------------------------------------------
            "validation_cross_entropy":
                selected["validation_cross_entropy"],

            # --------------------------------------------------------------
            # Training-only occupancy statistics
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # FINAL TEST METRICS
            # --------------------------------------------------------------
            "test_cross_entropy":
                test_stats[
                    "cross_entropy"
                ],

            "test_perplexity":
                test_stats[
                    "perplexity"
                ],

            "test_U_k":
                test_stats[
                    "U_k"
                ],

            "test_N":
                test_stats[
                    "N"
                ],
        }

        validate_test_row(
            row
        )

        main_rows.append(
            row
        )

    # ======================================================================
    # COMPLETENESS CHECK
    # ======================================================================
    expected_model_conditions = (
        len(
            config["training_subset_fractions"]
        )
        * len(
            config["orders"]
        )
    )

    if (
        len(main_rows)
        != expected_model_conditions
    ):
        raise RuntimeError(
            f"Expected {expected_model_conditions} final models, "
            f"got {len(main_rows)}."
        )

    # ======================================================================
    # CREATE 30 ORDER TRANSITIONS
    # ======================================================================
    transitions = build_transitions(
        main_rows
    )

    expected_transitions = (
        len(
            config["training_subset_fractions"]
        )
        * (
            len(config["orders"]) - 1
        )
    )

    if (
        len(transitions)
        != expected_transitions
    ):
        raise RuntimeError(
            f"Expected {expected_transitions} transitions, "
            f"got {len(transitions)}."
        )

    # ======================================================================
    # WRITE 36 FINAL MODEL RESULTS
    # ======================================================================
    main_fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "n",
        "k",
        "selected_alpha",
        "validation_cross_entropy",
        "N_k",
        "T_k",
        "f_1_k",
        "S_k",
        "D_k",
        "test_cross_entropy",
        "test_perplexity",
        "test_U_k",
        "test_N",
    ]

    write_csv(
        main_results_path,
        main_rows,
        main_fieldnames,
    )

    # ======================================================================
    # WRITE 30 TRANSITION OBSERVATIONS
    # ======================================================================
    transition_fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "history_length",
        "selected_alpha_previous",
        "selected_alpha_current",
        "S_k",
        "D_k",
        "U_k",
        "H_previous",
        "H_current",
        "delta_H",
    ]

    write_csv(
        transitions_path,
        transitions,
        transition_fieldnames,
    )

    # ======================================================================
    # FINAL STRUCTURAL SUMMARY
    # ======================================================================
    print(
        "\n============================================"
    )

    print(
        "LOCKED TEST EVALUATION COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        f"Final model rows: "
        f"{len(main_rows)}"
    )

    print(
        f"Transition rows: "
        f"{len(transitions)}"
    )

    print(
        f"Shared test targets per model: "
        f"{shared_test_n:,}"
    )

    print(
        "\nFinal model results:"
    )

    print(
        main_results_path
    )

    print(
        "\nOrder-transition results:"
    )

    print(
        transitions_path
    )

    print(
        "\nNo test-driven hyperparameter changes were performed."
    )


if __name__ == "__main__":
    main()