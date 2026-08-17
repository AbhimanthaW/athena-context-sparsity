from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_per_target_losses
from src.final_evaluation import (
    load_selected_hyperparameters,
    validate_selected_hyperparameters,
)
from src.interpolation import (
    interpolated_cross_entropy,
)
from src.ngram import NGramModel
from src.preprocess import get_training_subset


# ===========================================================================
# FROZEN PRIMARY EXPERIMENT IDENTITY
# ===========================================================================


EXPECTED_HYPERPARAMETER_SHA256 = (
    "a04a261d18fa30c35e5291759bc751802"
    "ccddb75d260c22e639b71d5ef7a1d19"
)


# ===========================================================================
# GENERAL HELPERS
# ===========================================================================


def get_project_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[1]


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
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
# CONFIG
# ===========================================================================


def validate_interpolation_config(
    config: dict,
) -> list[float]:
    """
    Validate and return the frozen lambda grid.
    """

    if "interpolation" not in config:
        raise ValueError(
            "Missing interpolation configuration."
        )

    interpolation = config[
        "interpolation"
    ]

    if "lambdas" not in interpolation:
        raise ValueError(
            "Missing interpolation lambda grid."
        )

    lambdas = interpolation[
        "lambdas"
    ]

    if not isinstance(
        lambdas,
        list,
    ):
        raise ValueError(
            "Interpolation lambdas must be a list."
        )

    if len(lambdas) < 2:
        raise ValueError(
            "Interpolation grid requires at least two values."
        )

    parsed = []

    for value in lambdas:

        value = float(
            value
        )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Every interpolation lambda must lie in [0,1]."
            )

        parsed.append(
            value
        )

    if len(
        set(parsed)
    ) != len(
        parsed
    ):
        raise ValueError(
            "Interpolation lambda grid contains duplicates."
        )

    if parsed != sorted(
        parsed
    ):
        raise ValueError(
            "Interpolation lambda grid must be sorted."
        )

    if 0.0 not in parsed:
        raise ValueError(
            "Interpolation grid must include lambda=0."
        )

    if 1.0 not in parsed:
        raise ValueError(
            "Interpolation grid must include lambda=1."
        )

    return parsed


# ===========================================================================
# SELECTION
# ===========================================================================


def select_best_lambdas(
    validation_rows: list[dict],
) -> list[dict]:
    """
    Select one lambda for each:

        (training subset, order transition)

    using validation cross-entropy only.

    Tie-break rule:
        smaller lambda_high wins an exact tie.

    The conservative tie-break gives less weight to the more complex
    higher-order model when validation performance is identical.
    """

    grouped: dict[
        tuple[str, int, int],
        list[dict],
    ] = {}

    for row in validation_rows:

        key = (
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

        grouped.setdefault(
            key,
            [],
        ).append(
            row
        )

    selected = []

    for key in sorted(
        grouped.keys(),
        key=lambda item: (
            float(
                grouped[
                    item
                ][0][
                    "training_fraction"
                ]
            ),
            item[2],
        ),
    ):

        candidates = grouped[
            key
        ]

        best = min(
            candidates,
            key=lambda row: (
                row[
                    "interpolation_validation_cross_entropy"
                ],
                row[
                    "lambda_high"
                ],
            ),
        )

        selected.append(
            {
                "training_fraction":
                    best[
                        "training_fraction"
                    ],

                "training_subset":
                    best[
                        "training_subset"
                    ],

                "training_tokens":
                    best[
                        "training_tokens"
                    ],

                "from_order":
                    best[
                        "from_order"
                    ],

                "to_order":
                    best[
                        "to_order"
                    ],

                "selected_alpha_previous":
                    best[
                        "selected_alpha_previous"
                    ],

                "selected_alpha_current":
                    best[
                        "selected_alpha_current"
                    ],

                "selected_lambda_high":
                    best[
                        "lambda_high"
                    ],

                "validation_cross_entropy_previous":
                    best[
                        "validation_cross_entropy_previous"
                    ],

                "validation_cross_entropy_current":
                    best[
                        "validation_cross_entropy_current"
                    ],

                "selected_interpolation_validation_cross_entropy":
                    best[
                        "interpolation_validation_cross_entropy"
                    ],

                "selected_interpolation_validation_perplexity":
                    best[
                        "interpolation_validation_perplexity"
                    ],

                # Negative means interpolated model beats
                # the lower-order model.
                "delta_vs_lower_validation":
                    (
                        best[
                            "interpolation_validation_cross_entropy"
                        ]
                        - best[
                            "validation_cross_entropy_previous"
                        ]
                    ),

                # Positive means interpolation improved on the
                # original higher-order model.
                "improvement_vs_original_high_validation":
                    (
                        best[
                            "validation_cross_entropy_current"
                        ]
                        - best[
                            "interpolation_validation_cross_entropy"
                        ]
                    ),

                "validation_N":
                    best[
                        "validation_N"
                    ],
            }
        )

    return selected


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    project_root = (
        get_project_root()
    )

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

    validation_output_path = (
        project_root
        / "results"
        / "interpolation_validation.csv"
    )

    selected_output_path = (
        project_root
        / "results"
        / "interpolation_selected.csv"
    )

    print(
        "Loading frozen interpolation-validation inputs..."
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

    lambdas = validate_interpolation_config(
        config
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

    # ----------------------------------------------------------------------
    # Verify that the component models use the exact same hyperparameters
    # as the locked primary experiment.
    # ----------------------------------------------------------------------
    actual_hash = sha256_file(
        selected_hyperparameters_path
    )

    print(
        f"Frozen hyperparameter SHA-256: {actual_hash}"
    )

    if (
        actual_hash
        != EXPECTED_HYPERPARAMETER_SHA256
    ):
        raise RuntimeError(
            "Frozen hyperparameter file does not match "
            "the primary experiment."
        )

    validation_sequences = dataset[
        "val_sequences"
    ]

    max_history_length = config[
        "max_history_length"
    ]

    expected_validation_targets = sum(
        max(
            0,
            len(sequence)
            - max_history_length,
        )
        for sequence in validation_sequences
    )

    if expected_validation_targets != 98839:
        raise RuntimeError(
            f"Expected 98,839 validation targets, "
            f"found {expected_validation_targets}."
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

    print(
        f"Lambda values: {len(lambdas)}"
    )

    print(
        f"Transitions per training size: 5"
    )

    print(
        f"Expected validation conditions: "
        f"{6 * 5 * len(lambdas)}"
    )

    validation_rows = []

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
        # Reconstruct all six already-frozen component models.
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
                    validation_sequences,
                    max_history_length=(
                        max_history_length
                    ),
                ),
                dtype=float,
            )

            if (
                len(losses)
                != expected_validation_targets
            ):
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "validation target-count mismatch."
                )

            if not np.isfinite(
                losses
            ).all():
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "non-finite validation losses."
                )

            # --------------------------------------------------------------
            # The per-target losses must reproduce the already frozen
            # aggregate validation result.
            # --------------------------------------------------------------
            recomputed_ce = float(
                losses.mean()
            )

            frozen_ce = float(
                selected[
                    "validation_cross_entropy"
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
                    "validation loss reconstruction "
                    "does not match frozen CE."
                )

            losses_by_order[
                n
            ] = losses

        # ==================================================================
        # TRANSITION LOOP
        # ==================================================================
        for to_order in range(
            2,
            7,
        ):

            from_order = (
                to_order - 1
            )

            previous_selected = (
                selected_lookup[
                    (
                        subset_label,
                        from_order,
                    )
                ]
            )

            current_selected = (
                selected_lookup[
                    (
                        subset_label,
                        to_order,
                    )
                ]
            )

            lower_losses = losses_by_order[
                from_order
            ]

            higher_losses = losses_by_order[
                to_order
            ]

            lower_ce = float(
                lower_losses.mean()
            )

            higher_ce = float(
                higher_losses.mean()
            )

            print(
                f"  {from_order}->{to_order}: "
                f"H_lower={lower_ce:.6f}, "
                f"H_high={higher_ce:.6f}"
            )

            # ==============================================================
            # LAMBDA GRID
            # ==============================================================
            for lambda_high in lambdas:

                interpolation_ce = (
                    interpolated_cross_entropy(
                        lower_losses,
                        higher_losses,
                        lambda_high,
                    )
                )

                interpolation_pp = math.exp(
                    interpolation_ce
                )

                validation_rows.append(
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
                            previous_selected[
                                "selected_alpha"
                            ],

                        "selected_alpha_current":
                            current_selected[
                                "selected_alpha"
                            ],

                        "lambda_high":
                            lambda_high,

                        "validation_cross_entropy_previous":
                            lower_ce,

                        "validation_cross_entropy_current":
                            higher_ce,

                        "interpolation_validation_cross_entropy":
                            interpolation_ce,

                        "interpolation_validation_perplexity":
                            interpolation_pp,

                        "validation_N":
                            expected_validation_targets,
                    }
                )

    # ======================================================================
    # COMPLETENESS
    # ======================================================================
    expected_rows = (
        len(
            config[
                "training_subset_fractions"
            ]
        )
        * 5
        * len(
            lambdas
        )
    )

    if (
        len(validation_rows)
        != expected_rows
    ):
        raise RuntimeError(
            f"Expected {expected_rows} interpolation validation rows, "
            f"produced {len(validation_rows)}."
        )

    selected_rows = (
        select_best_lambdas(
            validation_rows
        )
    )

    if len(
        selected_rows
    ) != 30:
        raise RuntimeError(
            f"Expected 30 frozen interpolation selections, "
            f"produced {len(selected_rows)}."
        )

    validation_fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "selected_alpha_previous",
        "selected_alpha_current",
        "lambda_high",
        "validation_cross_entropy_previous",
        "validation_cross_entropy_current",
        "interpolation_validation_cross_entropy",
        "interpolation_validation_perplexity",
        "validation_N",
    ]

    selection_fieldnames = [
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
    ]

    write_csv(
        validation_output_path,
        validation_rows,
        validation_fieldnames,
    )

    write_csv(
        selected_output_path,
        selected_rows,
        selection_fieldnames,
    )

    # ======================================================================
    # SUMMARY
    # ======================================================================
    selected_lambdas = [
        row[
            "selected_lambda_high"
        ]
        for row in selected_rows
    ]

    count_zero = sum(
        value == 0.0
        for value in selected_lambdas
    )

    count_one = sum(
        value == 1.0
        for value in selected_lambdas
    )

    count_interior = (
        len(selected_lambdas)
        - count_zero
        - count_one
    )

    print(
        "\n============================================"
    )

    print(
        "INTERPOLATION VALIDATION COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        f"Validation rows: "
        f"{len(validation_rows)}"
    )

    print(
        f"Selected transitions: "
        f"{len(selected_rows)}"
    )

    print(
        f"Shared validation targets: "
        f"{expected_validation_targets:,}"
    )

    print(
        "\nSelected lambda distribution:"
    )

    print(
        f"  lambda = 0: "
        f"{count_zero}/30"
    )

    print(
        f"  0 < lambda < 1: "
        f"{count_interior}/30"
    )

    print(
        f"  lambda = 1: "
        f"{count_one}/30"
    )

    print(
        "\nFull interpolation validation:"
    )

    print(
        validation_output_path
    )

    print(
        "\nFrozen interpolation selections:"
    )

    print(
        selected_output_path
    )

    print(
        "\nIMPORTANT: Test data were not evaluated."
    )


if __name__ == "__main__":
    main()