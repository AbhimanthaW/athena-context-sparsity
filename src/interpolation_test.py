from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import evaluate_per_target_losses
from src.final_evaluation import (
    load_selected_hyperparameters,
    validate_selected_hyperparameters,
)
from src.interpolation import (
    interpolated_losses,
)
from src.interpolation_validation import (
    validate_interpolation_config,
)
from src.ngram import NGramModel
from src.preprocess import get_training_subset


# ===========================================================================
# FROZEN ARTIFACT IDENTITIES
# ===========================================================================


EXPECTED_HYPERPARAMETER_SHA256 = (
    "a04a261d18fa30c35e5291759bc751802"
    "ccddb75d260c22e639b71d5ef7a1d19"
)

EXPECTED_INTERPOLATION_SHA256 = (
    "2e7fc62042dab3a024194555ff2f7ebb9"
    "1a1efd81c5cf3b8ad43676f1029587c"
)


# ===========================================================================
# HELPERS
# ===========================================================================


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
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

        writer.writerows(
            rows
        )


# ===========================================================================
# LOAD FROZEN INTERPOLATION SELECTIONS
# ===========================================================================


def load_interpolation_selections(
    path: Path,
) -> list[dict]:
    """
    Load the 30 validation-selected interpolation weights.
    """

    required_fields = {
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "selected_alpha_previous",
        "selected_alpha_current",
        "selected_lambda_high",
        "validation_cross_entropy_previous",
        "validation_cross_entropy_current",
        "selected_interpolation_validation_cross_entropy",
        "selected_interpolation_validation_perplexity",
        "delta_vs_lower_validation",
        "improvement_vs_original_high_validation",
        "validation_N",
    }

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Interpolation selection file has no header."
            )

        missing = (
            required_fields
            - set(
                reader.fieldnames
            )
        )

        if missing:
            raise ValueError(
                "Interpolation selection file is missing fields: "
                f"{sorted(missing)}"
            )

        for raw in reader:

            rows.append(
                {
                    "training_fraction":
                        float(
                            raw[
                                "training_fraction"
                            ]
                        ),

                    "training_subset":
                        raw[
                            "training_subset"
                        ],

                    "training_tokens":
                        int(
                            raw[
                                "training_tokens"
                            ]
                        ),

                    "from_order":
                        int(
                            raw[
                                "from_order"
                            ]
                        ),

                    "to_order":
                        int(
                            raw[
                                "to_order"
                            ]
                        ),

                    "selected_alpha_previous":
                        float(
                            raw[
                                "selected_alpha_previous"
                            ]
                        ),

                    "selected_alpha_current":
                        float(
                            raw[
                                "selected_alpha_current"
                            ]
                        ),

                    "selected_lambda_high":
                        float(
                            raw[
                                "selected_lambda_high"
                            ]
                        ),

                    "validation_cross_entropy_previous":
                        float(
                            raw[
                                "validation_cross_entropy_previous"
                            ]
                        ),

                    "validation_cross_entropy_current":
                        float(
                            raw[
                                "validation_cross_entropy_current"
                            ]
                        ),

                    "selected_interpolation_validation_cross_entropy":
                        float(
                            raw[
                                "selected_interpolation_validation_cross_entropy"
                            ]
                        ),

                    "selected_interpolation_validation_perplexity":
                        float(
                            raw[
                                "selected_interpolation_validation_perplexity"
                            ]
                        ),

                    "delta_vs_lower_validation":
                        float(
                            raw[
                                "delta_vs_lower_validation"
                            ]
                        ),

                    "improvement_vs_original_high_validation":
                        float(
                            raw[
                                "improvement_vs_original_high_validation"
                            ]
                        ),

                    "validation_N":
                        int(
                            raw[
                                "validation_N"
                            ]
                        ),
                }
            )

    return rows


# ===========================================================================
# VALIDATE FROZEN SELECTIONS
# ===========================================================================


def validate_interpolation_selections(
    selections: list[dict],
    config: dict,
    selected_hyperparameters: list[dict],
) -> None:
    """
    Verify that the robustness-test selections contain exactly one frozen
    lambda for each of the 30 intended order transitions.
    """

    lambdas = validate_interpolation_config(
        config
    )

    expected_pairs = {
        (
            f"{int(round(fraction * 100))}%",
            from_order,
            from_order + 1,
        )
        for fraction in config[
            "training_subset_fractions"
        ]
        for from_order in range(
            1,
            6,
        )
    }

    actual_pairs = [
        (
            row[
                "training_subset"
            ],
            row[
                "from_order"
            ],
            row[
                "to_order"
            ],
        )
        for row in selections
    ]

    if len(
        selections
    ) != 30:
        raise ValueError(
            f"Expected 30 interpolation selections, "
            f"found {len(selections)}."
        )

    if len(
        set(
            actual_pairs
        )
    ) != len(
        actual_pairs
    ):
        raise ValueError(
            "Duplicate interpolation transitions detected."
        )

    if set(
        actual_pairs
    ) != expected_pairs:
        raise ValueError(
            "Interpolation transitions do not match "
            "the configured experiment."
        )

    frozen_lookup = {
        (
            row[
                "training_subset"
            ],
            row[
                "n"
            ],
        ): row
        for row in selected_hyperparameters
    }

    for row in selections:

        from_order = row[
            "from_order"
        ]

        to_order = row[
            "to_order"
        ]

        if (
            to_order
            != from_order + 1
        ):
            raise ValueError(
                "Interpolation must compare consecutive orders."
            )

        lambda_high = row[
            "selected_lambda_high"
        ]

        lambda_valid = any(
            math.isclose(
                lambda_high,
                allowed,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for allowed in lambdas
        )

        if not lambda_valid:
            raise ValueError(
                f"Selected lambda {lambda_high} "
                "does not belong to frozen lambda grid."
            )

        subset = row[
            "training_subset"
        ]

        previous_frozen = frozen_lookup[
            (
                subset,
                from_order,
            )
        ]

        current_frozen = frozen_lookup[
            (
                subset,
                to_order,
            )
        ]

        if not math.isclose(
            row[
                "selected_alpha_previous"
            ],
            previous_frozen[
                "selected_alpha"
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Previous alpha mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )

        if not math.isclose(
            row[
                "selected_alpha_current"
            ],
            current_frozen[
                "selected_alpha"
            ],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Current alpha mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )


# ===========================================================================
# FROZEN PRIMARY RESULT LOOKUPS
# ===========================================================================


def get_main_result_row(
    main_results: pd.DataFrame,
    training_subset: str,
    n: int,
) -> pd.Series:
    rows = main_results[
        (
            main_results[
                "training_subset"
            ]
            == training_subset
        )
        & (
            main_results[
                "n"
            ]
            == n
        )
    ]

    if len(
        rows
    ) != 1:
        raise ValueError(
            f"Expected one frozen main result for "
            f"{training_subset}, n={n}."
        )

    return rows.iloc[0]


def get_original_transition_row(
    transitions: pd.DataFrame,
    training_subset: str,
    from_order: int,
    to_order: int,
) -> pd.Series:
    rows = transitions[
        (
            transitions[
                "training_subset"
            ]
            == training_subset
        )
        & (
            transitions[
                "from_order"
            ]
            == from_order
        )
        & (
            transitions[
                "to_order"
            ]
            == to_order
        )
    ]

    if len(
        rows
    ) != 1:
        raise ValueError(
            f"Expected one original transition for "
            f"{training_subset}, "
            f"{from_order}->{to_order}."
        )

    return rows.iloc[0]


# ===========================================================================
# MAIN LOCKED ROBUSTNESS TEST
# ===========================================================================


def main() -> None:
    project_root = (
        get_project_root()
    )

    # ----------------------------------------------------------------------
    # Inputs
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

    vocabulary_path = (
        project_root
        / "data"
        / "processed"
        / "vocab.json"
    )

    selected_hyperparameters_path = (
        project_root
        / "results"
        / "selected_hyperparameters.csv"
    )

    interpolation_selected_path = (
        project_root
        / "results"
        / "interpolation_selected.csv"
    )

    main_results_path = (
        project_root
        / "results"
        / "main_results.csv"
    )

    transitions_path = (
        project_root
        / "results"
        / "transitions.csv"
    )

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------
    output_path = (
        project_root
        / "results"
        / "interpolation_test.csv"
    )

    print(
        "Loading frozen interpolation robustness state..."
    )

    config = load_json(
        config_path
    )

    dataset = load_json(
        dataset_path
    )

    vocabulary = load_json(
        vocabulary_path
    )

    selected_hyperparameters = (
        load_selected_hyperparameters(
            selected_hyperparameters_path
        )
    )

    validate_selected_hyperparameters(
        selected_hyperparameters,
        config,
    )

    interpolation_selections = (
        load_interpolation_selections(
            interpolation_selected_path
        )
    )

    validate_interpolation_selections(
        interpolation_selections,
        config,
        selected_hyperparameters,
    )

    # ----------------------------------------------------------------------
    # Verify both frozen artifacts cryptographically.
    # ----------------------------------------------------------------------
    hyperparameter_hash = sha256_file(
        selected_hyperparameters_path
    )

    interpolation_hash = sha256_file(
        interpolation_selected_path
    )

    print(
        f"Frozen hyperparameter SHA-256: "
        f"{hyperparameter_hash}"
    )

    print(
        f"Frozen interpolation SHA-256: "
        f"{interpolation_hash}"
    )

    if (
        hyperparameter_hash
        != EXPECTED_HYPERPARAMETER_SHA256
    ):
        raise RuntimeError(
            "Primary hyperparameter artifact changed."
        )

    if (
        interpolation_hash
        != EXPECTED_INTERPOLATION_SHA256
    ):
        raise RuntimeError(
            "Interpolation selection artifact changed."
        )

    main_results = pd.read_csv(
        main_results_path
    )

    original_transitions = pd.read_csv(
        transitions_path
    )

    test_sequences = dataset[
        "test_sequences"
    ]

    max_history_length = config[
        "max_history_length"
    ]

    expected_test_targets = sum(
        max(
            0,
            len(sequence)
            - max_history_length,
        )
        for sequence in test_sequences
    )

    if (
        expected_test_targets
        != 98840
    ):
        raise RuntimeError(
            f"Expected 98,840 test targets, "
            f"found {expected_test_targets}."
        )

    selected_lookup = {
        (
            row[
                "training_subset"
            ],
            row[
                "n"
            ],
        ): row
        for row in selected_hyperparameters
    }

    interpolation_lookup = {
        (
            row[
                "training_subset"
            ],
            row[
                "from_order"
            ],
            row[
                "to_order"
            ],
        ): row
        for row in interpolation_selections
    }

    output_rows = []

    # ======================================================================
    # TRAINING-SIZE LOOP
    # ======================================================================
    for fraction in sorted(
        config[
            "training_subset_fractions"
        ]
    ):

        subset_label = (
            f"{int(round(fraction * 100))}%"
        )

        training_sequences = (
            get_training_subset(
                dataset,
                subset_label,
            )
        )

        training_tokens = sum(
            len(sequence)
            for sequence in training_sequences
        )

        print(
            f"\n=== {subset_label} "
            f"({training_tokens:,} training tokens) ==="
        )

        losses_by_order = {}

        # ------------------------------------------------------------------
        # Reconstruct all six frozen component models.
        # ------------------------------------------------------------------
        for n in config[
            "orders"
        ]:

            selected = selected_lookup[
                (
                    subset_label,
                    n,
                )
            ]

            alpha = selected[
                "selected_alpha"
            ]

            print(
                f"  reconstructing n={n}, "
                f"alpha={alpha}"
            )

            model = NGramModel(
                n=n,
                alpha=alpha,
                vocabulary=vocabulary,
            )

            model.fit(
                training_sequences
            )

            losses = np.asarray(
                evaluate_per_target_losses(
                    model,
                    test_sequences,
                    max_history_length=(
                        max_history_length
                    ),
                ),
                dtype=float,
            )

            if (
                len(losses)
                != expected_test_targets
            ):
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "test target count mismatch."
                )

            if not np.isfinite(
                losses
            ).all():
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "non-finite test losses."
                )

            recomputed_ce = float(
                losses.mean()
            )

            frozen_main = (
                get_main_result_row(
                    main_results,
                    subset_label,
                    n,
                )
            )

            frozen_ce = float(
                frozen_main[
                    "test_cross_entropy"
                ]
            )

            if not math.isclose(
                recomputed_ce,
                frozen_ce,
                rel_tol=1e-11,
                abs_tol=1e-11,
            ):
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "reconstructed test CE does not "
                    "match frozen primary result."
                )

            losses_by_order[
                n
            ] = losses

        # ==================================================================
        # FROZEN TRANSITION LOOP
        # ==================================================================
        for to_order in range(
            2,
            7,
        ):

            from_order = (
                to_order - 1
            )

            interpolation_selection = (
                interpolation_lookup[
                    (
                        subset_label,
                        from_order,
                        to_order,
                    )
                ]
            )

            lambda_high = (
                interpolation_selection[
                    "selected_lambda_high"
                ]
            )

            lower_losses = losses_by_order[
                from_order
            ]

            higher_losses = losses_by_order[
                to_order
            ]

            mixed_losses = (
                interpolated_losses(
                    lower_losses,
                    higher_losses,
                    lambda_high=(
                        lambda_high
                    ),
                )
            )

            if (
                len(mixed_losses)
                != expected_test_targets
            ):
                raise RuntimeError(
                    "Interpolated test target count mismatch."
                )

            if not np.isfinite(
                mixed_losses
            ).all():
                raise RuntimeError(
                    "Non-finite interpolated test losses."
                )

            lower_ce = float(
                lower_losses.mean()
            )

            higher_ce = float(
                higher_losses.mean()
            )

            interpolation_ce = float(
                mixed_losses.mean()
            )

            try:
                interpolation_pp = math.exp(
                    interpolation_ce
                )
            except OverflowError:
                interpolation_pp = float(
                    "inf"
                )

            original_delta = (
                higher_ce
                - lower_ce
            )

            interpolated_delta = (
                interpolation_ce
                - lower_ce
            )

            improvement_vs_original_high = (
                higher_ce
                - interpolation_ce
            )

            # --------------------------------------------------------------
            # Verify original Delta H against frozen primary transition.
            # --------------------------------------------------------------
            frozen_transition = (
                get_original_transition_row(
                    original_transitions,
                    subset_label,
                    from_order,
                    to_order,
                )
            )

            if not math.isclose(
                original_delta,
                float(
                    frozen_transition[
                        "delta_H"
                    ]
                ),
                rel_tol=1e-11,
                abs_tol=1e-11,
            ):
                raise RuntimeError(
                    f"{subset_label}, "
                    f"{from_order}->{to_order}: "
                    "original Delta H mismatch."
                )

            print(
                f"  {from_order}->{to_order} | "
                f"lambda={lambda_high:.1f} | "
                f"original DeltaH={original_delta:.6f} | "
                f"interpolated DeltaH={interpolated_delta:.6f}"
            )

            output_rows.append(
                {
                    "training_fraction":
                        fraction,

                    "training_subset":
                        subset_label,

                    "training_tokens":
                        training_tokens,

                    "from_order":
                        from_order,

                    "to_order":
                        to_order,

                    "selected_alpha_previous":
                        interpolation_selection[
                            "selected_alpha_previous"
                        ],

                    "selected_alpha_current":
                        interpolation_selection[
                            "selected_alpha_current"
                        ],

                    "selected_lambda_high":
                        lambda_high,

                    "test_cross_entropy_previous":
                        lower_ce,

                    "test_cross_entropy_current":
                        higher_ce,

                    "interpolation_test_cross_entropy":
                        interpolation_ce,

                    "interpolation_test_perplexity":
                        interpolation_pp,

                    "original_delta_H":
                        original_delta,

                    "interpolated_delta_H":
                        interpolated_delta,

                    "improvement_vs_original_high_test":
                        improvement_vs_original_high,

                    "test_N":
                        expected_test_targets,
                }
            )

    # ======================================================================
    # COMPLETENESS
    # ======================================================================
    if len(
        output_rows
    ) != 30:
        raise RuntimeError(
            f"Expected 30 interpolation-test rows, "
            f"produced {len(output_rows)}."
        )

    fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "selected_alpha_previous",
        "selected_alpha_current",
        "selected_lambda_high",
        "test_cross_entropy_previous",
        "test_cross_entropy_current",
        "interpolation_test_cross_entropy",
        "interpolation_test_perplexity",
        "original_delta_H",
        "interpolated_delta_H",
        "improvement_vs_original_high_test",
        "test_N",
    ]

    write_csv(
        output_path,
        output_rows,
        fieldnames,
    )

    # ======================================================================
    # DESCRIPTIVE SUMMARY
    # ======================================================================
    tolerance = 1e-12

    positive = sum(
        row[
            "interpolated_delta_H"
        ]
        > tolerance
        for row in output_rows
    )

    negative = sum(
        row[
            "interpolated_delta_H"
        ]
        < -tolerance
        for row in output_rows
    )

    approximately_zero = (
        len(output_rows)
        - positive
        - negative
    )

    improved_vs_original_high = sum(
        row[
            "improvement_vs_original_high_test"
        ]
        > tolerance
        for row in output_rows
    )

    print(
        "\n============================================"
    )

    print(
        "LOCKED INTERPOLATION ROBUSTNESS TEST COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        f"Transitions evaluated: "
        f"{len(output_rows)}"
    )

    print(
        "\nInterpolated Delta H relative to lower-order model:"
    )

    print(
        f"  Delta H < 0: "
        f"{negative}/30"
    )

    print(
        f"  Delta H ~= 0: "
        f"{approximately_zero}/30"
    )

    print(
        f"  Delta H > 0: "
        f"{positive}/30"
    )

    print(
        "\nInterpolation improved on original higher-order model:"
    )

    print(
        f"  {improved_vs_original_high}/30"
    )

    print(
        "\nResults:"
    )

    print(
        output_path
    )

    print(
        "\nNo interpolation weights were selected using test data."
    )


if __name__ == "__main__":
    main()