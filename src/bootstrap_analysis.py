from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import evaluate_per_target_losses
from src.final_evaluation import (
    load_selected_hyperparameters,
    validate_selected_hyperparameters,
)
from src.ngram import NGramModel
from src.preprocess import get_training_subset


# ===========================================================================
# FROZEN EXPERIMENT IDENTITY
# ===========================================================================

# SHA-256 recorded immediately before the locked primary test run.
#
# The bootstrap must use exactly the same validation-selected
# hyperparameters.
EXPECTED_HYPERPARAMETER_SHA256 = (
    "a04a261d18fa30c35e5291759bc751802"
    "ccddb75d260c22e639b71d5ef7a1d19"
)


# ===========================================================================
# PATHS / FILE HELPERS
# ===========================================================================


def get_project_root() -> Path:
    """
    bootstrap_analysis.py lives at:

        <project_root>/src/bootstrap_analysis.py

    therefore parents[1] is the repository root.
    """
    return Path(__file__).resolve().parents[1]


def load_json(path: Path):
    """
    Load UTF-8 JSON.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    """
    Compute SHA-256 of a file.

    This verifies that the bootstrap uses the exact frozen
    hyperparameter-selection artifact from the primary experiment.
    """
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
    """
    Write rows using a stable column order.
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
        writer.writerows(
            rows
        )


def save_json(
    path: Path,
    payload: dict,
) -> None:
    """
    Save deterministic JSON.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )


# ===========================================================================
# BOOTSTRAP CONFIG VALIDATION
# ===========================================================================


def validate_bootstrap_config(
    config: dict,
) -> dict:
    """
    Validate and return the frozen bootstrap configuration.
    """

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
        - bootstrap.keys()
    )

    if missing:
        raise ValueError(
            f"Missing bootstrap fields: "
            f"{sorted(missing)}"
        )

    seed = bootstrap[
        "seed"
    ]

    replicates = bootstrap[
        "replicates"
    ]

    primary_block_length = bootstrap[
        "primary_block_length"
    ]

    sensitivity = bootstrap[
        "sensitivity_block_lengths"
    ]

    confidence = bootstrap[
        "confidence_level"
    ]

    if not isinstance(
        seed,
        int,
    ):
        raise ValueError(
            "Bootstrap seed must be an integer."
        )

    if (
        not isinstance(
            replicates,
            int,
        )
        or replicates < 1
    ):
        raise ValueError(
            "Bootstrap replicates must be a positive integer."
        )

    if (
        not isinstance(
            primary_block_length,
            int,
        )
        or primary_block_length < 1
    ):
        raise ValueError(
            "Primary block length must be a positive integer."
        )

    if not isinstance(
        sensitivity,
        list,
    ):
        raise ValueError(
            "sensitivity_block_lengths must be a list."
        )

    for block_length in sensitivity:

        if (
            not isinstance(
                block_length,
                int,
            )
            or block_length < 1
        ):
            raise ValueError(
                "Every sensitivity block length "
                "must be a positive integer."
            )

    if not (
        0.0
        < confidence
        < 1.0
    ):
        raise ValueError(
            "confidence_level must lie in (0,1)."
        )

    all_lengths = [
        primary_block_length,
        *sensitivity,
    ]

    if len(
        set(all_lengths)
    ) != len(
        all_lengths
    ):
        raise ValueError(
            "Bootstrap block lengths must be unique."
        )

    return bootstrap


# ===========================================================================
# FROZEN RESULT LOOKUPS
# ===========================================================================


def get_main_result_row(
    main_results: pd.DataFrame,
    training_subset: str,
    n: int,
) -> pd.Series:
    """
    Retrieve exactly one frozen main-result row.
    """

    rows = main_results[
        (
            main_results[
                "training_subset"
            ]
            == training_subset
        )
        & (
            main_results["n"]
            == n
        )
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one main result for "
            f"{training_subset}, n={n}; "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


def get_transition_row(
    transitions: pd.DataFrame,
    training_subset: str,
    from_order: int,
    to_order: int,
) -> pd.Series:
    """
    Retrieve exactly one frozen transition row.
    """

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

    if len(rows) != 1:
        raise ValueError(
            f"Expected one transition for "
            f"{training_subset}, "
            f"{from_order}->{to_order}; "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


# ===========================================================================
# CIRCULAR MOVING-BLOCK BOOTSTRAP
# ===========================================================================


def circular_window_sums(
    values: np.ndarray,
    window_length: int,
) -> np.ndarray:
    """
    Compute the sum of every possible circular contiguous window.

    If values has length N, this returns N window sums.

    Example:

        values = [a, b, c, d]
        L = 3

    windows are:

        [a,b,c]
        [b,c,d]
        [c,d,a]
        [d,a,b]

    This allows a circular moving-block bootstrap while preserving
    local ordering within each sampled block.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    if n == 0:
        raise ValueError(
            "Cannot bootstrap an empty array."
        )

    if not (
        1
        <= window_length
        <= n
    ):
        raise ValueError(
            f"window_length must lie in [1,{n}], "
            f"got {window_length}."
        )

    if window_length == 1:
        return values.copy()

    extension = np.concatenate(
        [
            values,
            values[
                : window_length - 1
            ],
        ]
    )

    cumulative = np.concatenate(
        [
            np.array(
                [0.0]
            ),
            np.cumsum(
                extension,
                dtype=float,
            ),
        ]
    )

    sums = (
        cumulative[
            window_length:
        ]
        - cumulative[
            :-window_length
        ]
    )

    # There should be exactly one circular window starting
    # at each original target position.
    if len(sums) != n:
        raise RuntimeError(
            "Circular-window construction produced "
            "an unexpected number of windows."
        )

    return sums


def circular_block_bootstrap_means(
    differences: np.ndarray,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate paired circular moving-block bootstrap estimates of mean
    per-target loss difference.

    differences[t] is:

        loss_high_order[t] - loss_low_order[t]

    so every block automatically preserves model pairing.

    Each bootstrap replicate contains exactly N target observations.

    Rather than materializing N losses for every replicate, this function
    resamples precomputed block sums, making 5000 x 30 x 3 bootstrap
    analyses practical.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    n = len(
        differences
    )

    if n == 0:
        raise ValueError(
            "Cannot bootstrap an empty difference array."
        )

    if not (
        1
        <= block_length
        <= n
    ):
        raise ValueError(
            f"block_length must lie in [1,{n}], "
            f"got {block_length}."
        )

    if replicates < 1:
        raise ValueError(
            "replicates must be positive."
        )

    full_block_count = (
        n // block_length
    )

    remainder = (
        n % block_length
    )

    full_block_sums = (
        circular_window_sums(
            differences,
            block_length,
        )
    )

    # ---------------------------------------------------------------
    # Sample starting positions for each complete block.
    #
    # Shape:
    #
    #   (bootstrap replicates, full blocks per replicate)
    # ---------------------------------------------------------------
    full_starts = rng.integers(
        low=0,
        high=n,
        size=(
            replicates,
            full_block_count,
        ),
    )

    replicate_sums = (
        full_block_sums[
            full_starts
        ]
        .sum(
            axis=1
        )
    )

    # ---------------------------------------------------------------
    # If N is not exactly divisible by L, add one circular block of
    # the remaining length.
    #
    # Example:
    #
    # N = 98,840
    # L = 1,000
    #
    # 98 complete blocks + one 840-target block.
    # ---------------------------------------------------------------
    if remainder > 0:

        remainder_sums = (
            circular_window_sums(
                differences,
                remainder,
            )
        )

        remainder_starts = (
            rng.integers(
                low=0,
                high=n,
                size=replicates,
            )
        )

        replicate_sums += (
            remainder_sums[
                remainder_starts
            ]
        )

    bootstrap_means = (
        replicate_sums
        / n
    )

    return bootstrap_means


def percentile_interval(
    bootstrap_values: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Return a two-sided percentile bootstrap confidence interval.
    """

    alpha = (
        1.0
        - confidence_level
    )

    lower_probability = (
        alpha / 2.0
    )

    upper_probability = (
        1.0
        - alpha / 2.0
    )

    lower, upper = np.quantile(
        bootstrap_values,
        [
            lower_probability,
            upper_probability,
        ],
    )

    return (
        float(lower),
        float(upper),
    )


# ===========================================================================
# FIGURE 4
# ===========================================================================


def plot_primary_bootstrap_intervals(
    bootstrap_results: pd.DataFrame,
    primary_block_length: int,
    output_path: Path,
) -> None:
    """
    Figure 4:
        Observed Delta H with paired block-bootstrap 95% intervals.

    Only the preregistered PRIMARY block length is plotted.
    Sensitivity block lengths remain in the result table.
    """

    primary = bootstrap_results[
        bootstrap_results[
            "block_length"
        ]
        == primary_block_length
    ].copy()

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            6.0,
        )
    )

    transitions = sorted(
        {
            (
                int(row.from_order),
                int(row.to_order),
            )
            for row in primary.itertuples()
        }
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
    ]

    # Small x offsets keep confidence intervals from sitting directly
    # on top of one another at each training percentage.
    offsets = np.linspace(
        -1.2,
        1.2,
        len(transitions),
    )

    for index, (
        from_order,
        to_order,
    ) in enumerate(
        transitions
    ):

        group = primary[
            (
                primary[
                    "from_order"
                ]
                == from_order
            )
            & (
                primary[
                    "to_order"
                ]
                == to_order
            )
        ].sort_values(
            "training_fraction"
        )

        x = (
            group[
                "training_fraction"
            ].to_numpy()
            * 100.0
            + offsets[index]
        )

        observed = group[
            "delta_H_observed"
        ].to_numpy()

        lower = group[
            "ci_lower"
        ].to_numpy()

        upper = group[
            "ci_upper"
        ].to_numpy()

        y_error = np.vstack(
            [
                observed - lower,
                upper - observed,
            ]
        )

        axis.errorbar(
            x,
            observed,
            yerr=y_error,
            marker=markers[index],
            capsize=3,
            label=(
                f"{from_order}"
                r"$\rightarrow$"
                f"{to_order}"
            ),
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_title(
        "Marginal test loss with paired block-bootstrap intervals"
    )

    axis.set_xlabel(
        "Training fraction (%)"
    )

    axis.set_ylabel(
        r"$\Delta H_n$ (nats/token)"
    )

    axis.set_xticks(
        [
            5,
            10,
            20,
            40,
            80,
            100,
        ]
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        title="Order transition",
        frameon=False,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path.with_suffix(
            ".png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    figure.savefig(
        output_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    project_root = (
        get_project_root()
    )

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

    selected_path = (
        project_root
        / "results"
        / "selected_hyperparameters.csv"
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
    # Output paths
    # ----------------------------------------------------------------------
    analysis_dir = (
        project_root
        / "results"
        / "analysis"
    )

    bootstrap_results_path = (
        analysis_dir
        / "bootstrap_transitions.csv"
    )

    bootstrap_primary_path = (
        analysis_dir
        / "bootstrap_primary.csv"
    )

    bootstrap_summary_path = (
        analysis_dir
        / "bootstrap_summary.json"
    )

    figure_path = (
        project_root
        / "figures"
        / "figure4_bootstrap_delta_h"
    )

    # ----------------------------------------------------------------------
    # Load frozen experiment artifacts.
    # ----------------------------------------------------------------------
    print(
        "Loading frozen bootstrap inputs..."
    )

    config = load_json(
        config_path
    )

    bootstrap_config = (
        validate_bootstrap_config(
            config
        )
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

    main_results = pd.read_csv(
        main_results_path
    )

    transitions = pd.read_csv(
        transitions_path
    )

    # ----------------------------------------------------------------------
    # Frozen-selection hash verification.
    # ----------------------------------------------------------------------
    actual_hash = sha256_file(
        selected_path
    )

    print(
        f"Frozen hyperparameter SHA-256: "
        f"{actual_hash}"
    )

    if (
        actual_hash
        != EXPECTED_HYPERPARAMETER_SHA256
    ):
        raise RuntimeError(
            "selected_hyperparameters.csv does not match "
            "the hash recorded before the primary test run."
        )

    # ----------------------------------------------------------------------
    # Test data are already consumed by the primary experiment.
    #
    # We are now performing uncertainty analysis on the same frozen
    # evaluation, not selecting models or changing methodology.
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
            f"Expected 98,840 aligned test targets, "
            f"found {expected_targets}."
        )

    seed = bootstrap_config[
        "seed"
    ]

    replicates = bootstrap_config[
        "replicates"
    ]

    primary_block_length = (
        bootstrap_config[
            "primary_block_length"
        ]
    )

    sensitivity_lengths = (
        bootstrap_config[
            "sensitivity_block_lengths"
        ]
    )

    confidence_level = (
        bootstrap_config[
            "confidence_level"
        ]
    )

    block_lengths = [
        primary_block_length,
        *sensitivity_lengths,
    ]

    for block_length in block_lengths:

        if block_length > expected_targets:
            raise ValueError(
                f"Block length {block_length} exceeds "
                f"test target count {expected_targets}."
            )

    # ----------------------------------------------------------------------
    # Stable lookup for frozen alpha values.
    # ----------------------------------------------------------------------
    selected_lookup = {
        (
            row[
                "training_subset"
            ],
            row[
                "n"
            ],
        ): row
        for row in selected_rows
    }

    bootstrap_rows = []

    training_fractions = (
        sorted(
            config[
                "training_subset_fractions"
            ]
        )
    )

    print(
        f"Bootstrap replicates per condition: "
        f"{replicates:,}"
    )

    print(
        f"Block lengths: "
        f"{block_lengths}"
    )

    print(
        f"Confidence level: "
        f"{confidence_level:.2%}"
    )

    # ======================================================================
    # TRAINING SUBSET LOOP
    # ======================================================================
    for fraction in training_fractions:

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

        # --------------------------------------------------------------
        # Reconstruct all six frozen models for this training subset and
        # store aligned per-target losses.
        #
        # Only six arrays are kept in memory at once.
        # --------------------------------------------------------------
        losses_by_order = {}

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

            # ----------------------------------------------------------
            # Exact target alignment.
            # ----------------------------------------------------------
            if len(
                losses
            ) != expected_targets:
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "per-target loss count mismatch."
                )

            if not np.isfinite(
                losses
            ).all():
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "non-finite per-target losses detected."
                )

            # ----------------------------------------------------------
            # Reproduce the already-frozen test cross-entropy before
            # doing any bootstrap work.
            # ----------------------------------------------------------
            main_row = (
                get_main_result_row(
                    main_results,
                    subset_label,
                    n,
                )
            )

            recomputed_cross_entropy = (
                float(
                    losses.mean()
                )
            )

            frozen_cross_entropy = float(
                main_row[
                    "test_cross_entropy"
                ]
            )

            if not math.isclose(
                recomputed_cross_entropy,
                frozen_cross_entropy,
                rel_tol=1e-11,
                abs_tol=1e-11,
            ):
                raise RuntimeError(
                    f"{subset_label}, n={n}: "
                    "per-target losses do not reproduce "
                    "the frozen test cross-entropy."
                )

            losses_by_order[
                n
            ] = losses

        # ==============================================================
        # TRANSITION LOOP
        # ==============================================================
        for to_order in range(
            2,
            7,
        ):

            from_order = (
                to_order - 1
            )

            previous_losses = (
                losses_by_order[
                    from_order
                ]
            )

            current_losses = (
                losses_by_order[
                    to_order
                ]
            )

            if len(
                previous_losses
            ) != len(
                current_losses
            ):
                raise RuntimeError(
                    "Paired loss arrays have "
                    "different lengths."
                )

            paired_differences = (
                current_losses
                - previous_losses
            )

            observed_delta = float(
                paired_differences.mean()
            )

            # ----------------------------------------------------------
            # Reproduce the frozen transition exactly.
            # ----------------------------------------------------------
            frozen_transition = (
                get_transition_row(
                    transitions,
                    subset_label,
                    from_order,
                    to_order,
                )
            )

            frozen_delta = float(
                frozen_transition[
                    "delta_H"
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
                    "paired per-target differences do not "
                    "reproduce frozen delta_H."
                )

            print(
                f"  {from_order}->{to_order}: "
                f"Delta H={observed_delta:.6f}"
            )

            # ==========================================================
            # BLOCK-LENGTH LOOP
            # ==========================================================
            for block_length in block_lengths:

                # ------------------------------------------------------
                # Stable independent RNG stream for each condition.
                #
                # This makes the result invariant to loop ordering.
                # ------------------------------------------------------
                seed_sequence = (
                    np.random.SeedSequence(
                        [
                            seed,
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

                rng = (
                    np.random.default_rng(
                        seed_sequence
                    )
                )

                bootstrap_estimates = (
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

                (
                    ci_lower,
                    ci_upper,
                ) = percentile_interval(
                    bootstrap_estimates,
                    confidence_level,
                )

                bootstrap_mean = float(
                    bootstrap_estimates.mean()
                )

                bootstrap_std = float(
                    bootstrap_estimates.std(
                        ddof=1
                    )
                )

                contains_zero = bool(
                    ci_lower
                    <= 0.0
                    <= ci_upper
                )

                fraction_positive = float(
                    np.mean(
                        bootstrap_estimates
                        > 0.0
                    )
                )

                bootstrap_rows.append(
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

                        "delta_H_observed":
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

                        "fraction_bootstrap_positive":
                            fraction_positive,
                    }
                )

    # ======================================================================
    # COMPLETENESS
    # ======================================================================
    expected_rows = (
        6
        * 5
        * len(
            block_lengths
        )
    )

    if len(
        bootstrap_rows
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} bootstrap rows, "
            f"produced {len(bootstrap_rows)}."
        )

    fieldnames = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "delta_H_observed",
        "block_length",
        "bootstrap_replicates",
        "confidence_level",
        "bootstrap_mean",
        "bootstrap_std",
        "ci_lower",
        "ci_upper",
        "contains_zero",
        "fraction_bootstrap_positive",
    ]

    write_csv(
        bootstrap_results_path,
        bootstrap_rows,
        fieldnames,
    )

    bootstrap_dataframe = (
        pd.DataFrame(
            bootstrap_rows
        )
    )

    # ----------------------------------------------------------------------
    # Convenience file containing only the preregistered PRIMARY block
    # length.
    # ----------------------------------------------------------------------
    primary_results = (
        bootstrap_dataframe[
            bootstrap_dataframe[
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

    primary_results.to_csv(
        bootstrap_primary_path,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Summaries by block length.
    # ----------------------------------------------------------------------
    sensitivity_summary = {}

    for block_length in block_lengths:

        subset = bootstrap_dataframe[
            bootstrap_dataframe[
                "block_length"
            ]
            == block_length
        ]

        entirely_positive = int(
            (
                subset[
                    "ci_lower"
                ]
                > 0.0
            ).sum()
        )

        entirely_negative = int(
            (
                subset[
                    "ci_upper"
                ]
                < 0.0
            ).sum()
        )

        overlapping_zero = int(
            subset[
                "contains_zero"
            ].sum()
        )

        sensitivity_summary[
            str(
                block_length
            )
        ] = {
            "n_transitions":
                int(
                    len(subset)
                ),

            "ci_entirely_above_zero":
                entirely_positive,

            "ci_contains_zero":
                overlapping_zero,

            "ci_entirely_below_zero":
                entirely_negative,
        }

    summary = {
        "method":
            "paired circular moving-block bootstrap",

        "paired_quantity":
            (
                "per-target loss difference "
                "loss_n - loss_n_minus_1"
            ),

        "frozen_hyperparameter_sha256":
            actual_hash,

        "bootstrap_seed":
            seed,

        "bootstrap_replicates":
            replicates,

        "primary_block_length":
            primary_block_length,

        "sensitivity_block_lengths":
            sensitivity_lengths,

        "confidence_level":
            confidence_level,

        "test_targets":
            expected_targets,

        "sensitivity_summary":
            sensitivity_summary,

        "interpretation_limit":
            (
                "These intervals quantify uncertainty associated with "
                "resampling contiguous held-out target blocks. They do "
                "not include uncertainty from alternative training "
                "samples, corpus choice, preprocessing choices, or "
                "hyperparameter selection."
            ),
    }

    save_json(
        bootstrap_summary_path,
        summary,
    )

    # ----------------------------------------------------------------------
    # Figure 4.
    # ----------------------------------------------------------------------
    plot_primary_bootstrap_intervals(
        bootstrap_dataframe,
        primary_block_length,
        figure_path,
    )

    # ======================================================================
    # CONSOLE SUMMARY
    # ======================================================================
    print(
        "\n============================================"
    )

    print(
        "PAIRED BLOCK-BOOTSTRAP COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        f"Bootstrap result rows: "
        f"{len(bootstrap_rows)}"
    )

    print(
        f"Primary transition rows: "
        f"{len(primary_results)}"
    )

    for block_length in block_lengths:

        result = (
            sensitivity_summary[
                str(
                    block_length
                )
            ]
        )

        print(
            f"\nBlock length {block_length}:"
        )

        print(
            "  CI entirely > 0: "
            f"{result['ci_entirely_above_zero']}/30"
        )

        print(
            "  CI contains 0:   "
            f"{result['ci_contains_zero']}/30"
        )

        print(
            "  CI entirely < 0: "
            f"{result['ci_entirely_below_zero']}/30"
        )

    print(
        "\nPrimary bootstrap results:"
    )

    print(
        bootstrap_primary_path
    )

    print(
        "\nAll sensitivity results:"
    )

    print(
        bootstrap_results_path
    )

    print(
        "\nBootstrap summary:"
    )

    print(
        bootstrap_summary_path
    )

    print(
        "\nFigure 4:"
    )

    print(
        figure_path.with_suffix(
            ".png"
        )
    )


if __name__ == "__main__":
    main()