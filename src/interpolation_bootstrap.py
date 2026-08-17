from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.bootstrap_analysis import (
    circular_block_bootstrap_means,
    percentile_interval,
)
from src.evaluation import evaluate_per_target_losses
from src.final_evaluation import (
    load_selected_hyperparameters,
    validate_selected_hyperparameters,
)
from src.interpolation import interpolated_losses
from src.interpolation_test import (
    load_interpolation_selections,
    validate_interpolation_selections,
)
from src.ngram import NGramModel
from src.preprocess import get_training_subset


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


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

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
        writer.writerows(rows)


def validate_bootstrap_config(
    config: dict,
) -> dict:

    if "bootstrap" not in config:
        raise ValueError(
            "Missing bootstrap configuration."
        )

    bootstrap = config[
        "bootstrap"
    ]

    required = {
        "seed",
        "replicates",
        "primary_block_length",
        "sensitivity_block_lengths",
        "confidence_level",
    }

    missing = (
        required
        - set(
            bootstrap.keys()
        )
    )

    if missing:
        raise ValueError(
            f"Missing bootstrap fields: "
            f"{sorted(missing)}"
        )

    if (
        not isinstance(
            bootstrap["replicates"],
            int,
        )
        or bootstrap["replicates"] < 1
    ):
        raise ValueError(
            "Bootstrap replicates must be a positive integer."
        )

    if not (
        0.0
        < bootstrap[
            "confidence_level"
        ]
        < 1.0
    ):
        raise ValueError(
            "confidence_level must lie in (0,1)."
        )

    block_lengths = [
        bootstrap[
            "primary_block_length"
        ],
        *bootstrap[
            "sensitivity_block_lengths"
        ],
    ]

    if any(
        (
            not isinstance(
                length,
                int,
            )
            or length < 1
        )
        for length in block_lengths
    ):
        raise ValueError(
            "All bootstrap block lengths must "
            "be positive integers."
        )

    if len(
        block_lengths
    ) != len(
        set(block_lengths)
    ):
        raise ValueError(
            "Bootstrap block lengths must be unique."
        )

    return bootstrap


# ===========================================================================
# FROZEN TEST RESULT LOOKUP
# ===========================================================================


def get_interpolation_test_row(
    results: pd.DataFrame,
    training_subset: str,
    from_order: int,
    to_order: int,
) -> pd.Series:

    rows = results[
        (
            results[
                "training_subset"
            ]
            == training_subset
        )
        & (
            results[
                "from_order"
            ]
            == from_order
        )
        & (
            results[
                "to_order"
            ]
            == to_order
        )
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one interpolation-test row for "
            f"{training_subset}, "
            f"{from_order}->{to_order}; "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    project_root = get_project_root()

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

    hyperparameter_path = (
        project_root
        / "results"
        / "selected_hyperparameters.csv"
    )

    interpolation_selected_path = (
        project_root
        / "results"
        / "interpolation_selected.csv"
    )

    interpolation_test_path = (
        project_root
        / "results"
        / "interpolation_test.csv"
    )

    output_all_path = (
        project_root
        / "results"
        / "analysis"
        / "interpolation_bootstrap.csv"
    )

    output_primary_path = (
        project_root
        / "results"
        / "analysis"
        / "interpolation_bootstrap_primary.csv"
    )

    print(
        "Loading frozen interpolation-bootstrap state..."
    )

    config = load_json(
        config_path
    )

    bootstrap = validate_bootstrap_config(
        config
    )

    dataset = load_json(
        dataset_path
    )

    vocabulary = load_json(
        vocabulary_path
    )

    selected_hyperparameters = (
        load_selected_hyperparameters(
            hyperparameter_path
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
    # Verify frozen hashes
    # ----------------------------------------------------------------------
    hyperparameter_hash = sha256_file(
        hyperparameter_path
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
            "Frozen primary hyperparameter artifact changed."
        )

    if (
        interpolation_hash
        != EXPECTED_INTERPOLATION_SHA256
    ):
        raise RuntimeError(
            "Frozen interpolation artifact changed."
        )

    interpolation_test = pd.read_csv(
        interpolation_test_path
    )

    if len(
        interpolation_test
    ) != 30:
        raise RuntimeError(
            "Expected 30 frozen interpolation-test rows."
        )

    # ----------------------------------------------------------------------
    # Test target alignment
    # ----------------------------------------------------------------------
    test_sequences = dataset[
        "test_sequences"
    ]

    max_history_length = config[
        "max_history_length"
    ]

    expected_targets = sum(
        max(
            0,
            len(sequence)
            - max_history_length,
        )
        for sequence in test_sequences
    )

    if expected_targets != 98840:
        raise RuntimeError(
            f"Expected 98,840 test targets, "
            f"found {expected_targets}."
        )

    selected_lookup = {
        (
            row[
                "training_subset"
            ],
            row["n"],
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

    seed = int(
        bootstrap["seed"]
    )

    replicates = int(
        bootstrap[
            "replicates"
        ]
    )

    primary_block_length = int(
        bootstrap[
            "primary_block_length"
        ]
    )

    sensitivity_lengths = [
        int(value)
        for value in bootstrap[
            "sensitivity_block_lengths"
        ]
    ]

    confidence_level = float(
        bootstrap[
            "confidence_level"
        ]
    )

    block_lengths = [
        primary_block_length,
        *sensitivity_lengths,
    ]

    output_rows = []

    # ======================================================================
    # TRAINING SIZE
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
        # Reconstruct all frozen models
        # ------------------------------------------------------------------
        for n in config[
            "orders"
        ]:

            frozen = selected_lookup[
                (
                    subset_label,
                    n,
                )
            ]

            alpha = frozen[
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

            if len(
                losses
            ) != expected_targets:
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "test-target alignment failure."
                )

            if not np.isfinite(
                losses
            ).all():
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "non-finite test losses."
                )

            losses_by_order[
                n
            ] = losses

        # ==================================================================
        # TRANSITIONS
        # ==================================================================
        for to_order in range(
            2,
            7,
        ):

            from_order = (
                to_order - 1
            )

            selection = (
                interpolation_lookup[
                    (
                        subset_label,
                        from_order,
                        to_order,
                    )
                ]
            )

            lambda_high = float(
                selection[
                    "selected_lambda_high"
                ]
            )

            lower_losses = (
                losses_by_order[
                    from_order
                ]
            )

            higher_losses = (
                losses_by_order[
                    to_order
                ]
            )

            mixed_losses = (
                interpolated_losses(
                    lower_losses,
                    higher_losses,
                    lambda_high,
                )
            )

            # --------------------------------------------------------------
            # Paired quantity:
            #
            # mixed loss - lower-order loss
            #
            # Negative means interpolation improves.
            # --------------------------------------------------------------
            paired_differences = (
                mixed_losses
                - lower_losses
            )

            observed_delta = float(
                paired_differences.mean()
            )

            frozen_row = (
                get_interpolation_test_row(
                    interpolation_test,
                    subset_label,
                    from_order,
                    to_order,
                )
            )

            frozen_delta = float(
                frozen_row[
                    "interpolated_delta_H"
                ]
            )

            if not math.isclose(
                observed_delta,
                frozen_delta,
                rel_tol=1e-11,
                abs_tol=1e-11,
            ):
                raise RuntimeError(
                    f"{subset_label}, "
                    f"{from_order}->{to_order}: "
                    "reconstructed interpolated Delta H "
                    "does not match frozen test result."
                )

            print(
                f"  {from_order}->{to_order} | "
                f"lambda={lambda_high:.1f} | "
                f"DeltaH={observed_delta:.6f}"
            )

            # ==============================================================
            # BOOTSTRAP
            # ==============================================================
            for block_length in block_lengths:

                # Stable independent RNG stream for each condition.
                seed_sequence = (
                    np.random.SeedSequence(
                        [
                            seed,
                            808,
                            int(
                                round(
                                    fraction
                                    * 100
                                )
                            ),
                            to_order,
                            block_length,
                        ]
                    )
                )

                rng = np.random.default_rng(
                    seed_sequence
                )

                estimates = (
                    circular_block_bootstrap_means(
                        differences=(
                            paired_differences
                        ),
                        block_length=(
                            block_length
                        ),
                        replicates=(
                            replicates
                        ),
                        rng=rng,
                    )
                )

                ci_lower, ci_upper = (
                    percentile_interval(
                        estimates,
                        confidence_level,
                    )
                )

                bootstrap_mean = float(
                    estimates.mean()
                )

                bootstrap_std = float(
                    estimates.std(
                        ddof=1
                    )
                )

                contains_zero = bool(
                    ci_lower
                    <= 0.0
                    <= ci_upper
                )

                fraction_negative = float(
                    np.mean(
                        estimates < 0.0
                    )
                )

                fraction_positive = float(
                    np.mean(
                        estimates > 0.0
                    )
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

                        "selected_lambda_high":
                            lambda_high,

                        "interpolated_delta_H":
                            observed_delta,

                        "block_length":
                            block_length,

                        "bootstrap_replicates":
                            replicates,

                        "confidence_level":
                            confidence_level,

                        "bootstrap_mean":
                            bootstrap_mean,

                        "bootstrap_std":
                            bootstrap_std,

                        "ci_lower":
                            ci_lower,

                        "ci_upper":
                            ci_upper,

                        "contains_zero":
                            contains_zero,

                        "fraction_bootstrap_negative":
                            fraction_negative,

                        "fraction_bootstrap_positive":
                            fraction_positive,
                    }
                )

    # ======================================================================
    # COMPLETENESS
    # ======================================================================
    expected_rows = (
        30
        * len(
            block_lengths
        )
    )

    if len(
        output_rows
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"produced {len(output_rows)}."
        )

    fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "selected_lambda_high",
        "interpolated_delta_H",
        "block_length",
        "bootstrap_replicates",
        "confidence_level",
        "bootstrap_mean",
        "bootstrap_std",
        "ci_lower",
        "ci_upper",
        "contains_zero",
        "fraction_bootstrap_negative",
        "fraction_bootstrap_positive",
    ]

    write_csv(
        output_all_path,
        output_rows,
        fieldnames,
    )

    dataframe = pd.DataFrame(
        output_rows
    )

    primary = (
        dataframe[
            dataframe[
                "block_length"
            ]
            == primary_block_length
        ]
        .sort_values(
            [
                "training_fraction",
                "to_order",
            ]
        )
    )

    primary.to_csv(
        output_primary_path,
        index=False,
    )

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print(
        "\n============================================"
    )

    print(
        "INTERPOLATION BOOTSTRAP COMPLETE"
    )

    print(
        "============================================"
    )

    for block_length in block_lengths:

        subset = dataframe[
            dataframe[
                "block_length"
            ]
            == block_length
        ]

        below_zero = int(
            (
                subset[
                    "ci_upper"
                ]
                < 0.0
            ).sum()
        )

        above_zero = int(
            (
                subset[
                    "ci_lower"
                ]
                > 0.0
            ).sum()
        )

        contains_zero = int(
            subset[
                "contains_zero"
            ].sum()
        )

        print(
            f"\nBlock length {block_length}:"
        )

        print(
            f"  CI entirely < 0: "
            f"{below_zero}/30"
        )

        print(
            f"  CI contains 0:   "
            f"{contains_zero}/30"
        )

        print(
            f"  CI entirely > 0: "
            f"{above_zero}/30"
        )

    print(
        "\nPrimary results:"
    )

    print(
        output_primary_path
    )

    print(
        "\nAll sensitivity results:"
    )

    print(
        output_all_path
    )


if __name__ == "__main__":
    main()