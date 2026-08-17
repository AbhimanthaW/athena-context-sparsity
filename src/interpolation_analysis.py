from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_BLOCK_LENGTH = 1000
EXPECTED_BLOCK_LENGTHS = {
    500,
    1000,
    2000,
}

ZERO_TOLERANCE = 1e-12


# ===========================================================================
# PATHS
# ===========================================================================


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_inputs(
    interpolation_test: pd.DataFrame,
    primary_bootstrap: pd.DataFrame,
    all_bootstrap: pd.DataFrame,
) -> None:
    """
    Validate the frozen interpolation robustness artifacts before
    summarizing them.
    """

    if len(interpolation_test) != 30:
        raise ValueError(
            f"Expected 30 interpolation-test rows, "
            f"found {len(interpolation_test)}."
        )

    if len(primary_bootstrap) != 30:
        raise ValueError(
            f"Expected 30 primary bootstrap rows, "
            f"found {len(primary_bootstrap)}."
        )

    if len(all_bootstrap) != 90:
        raise ValueError(
            f"Expected 90 bootstrap sensitivity rows, "
            f"found {len(all_bootstrap)}."
        )

    observed_lengths = set(
        all_bootstrap[
            "block_length"
        ].astype(int)
    )

    if observed_lengths != EXPECTED_BLOCK_LENGTHS:
        raise ValueError(
            f"Unexpected block lengths: "
            f"{sorted(observed_lengths)}."
        )

    if not (
        primary_bootstrap[
            "block_length"
        ]
        == PRIMARY_BLOCK_LENGTH
    ).all():
        raise ValueError(
            "Primary bootstrap file contains "
            "unexpected block lengths."
        )

    # ----------------------------------------------------------------------
    # Unique experimental keys
    # ----------------------------------------------------------------------
    key_columns = [
        "training_subset",
        "from_order",
        "to_order",
    ]

    if interpolation_test.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Duplicate interpolation-test condition."
        )

    if primary_bootstrap.duplicated(
        subset=key_columns
    ).any():
        raise ValueError(
            "Duplicate primary-bootstrap condition."
        )

    # ----------------------------------------------------------------------
    # Every primary bootstrap Delta H must reproduce the frozen
    # interpolation-test Delta H.
    # ----------------------------------------------------------------------
    merged = interpolation_test.merge(
        primary_bootstrap,
        on=key_columns,
        suffixes=(
            "_test",
            "_bootstrap",
        ),
        validate="one_to_one",
    )

    if len(merged) != 30:
        raise ValueError(
            "Interpolation-test and primary-bootstrap "
            "condition grids do not align."
        )

    if not np.allclose(
        merged[
            "interpolated_delta_H_test"
        ].to_numpy(dtype=float),
        merged[
            "interpolated_delta_H_bootstrap"
        ].to_numpy(dtype=float),
        rtol=1e-11,
        atol=1e-11,
    ):
        raise ValueError(
            "Primary bootstrap Delta H does not reproduce "
            "the frozen interpolation-test result."
        )

    # ----------------------------------------------------------------------
    # Lambda=0 must imply exact identity with lower-order model.
    # ----------------------------------------------------------------------
    zero_lambda = interpolation_test[
        np.isclose(
            interpolation_test[
                "selected_lambda_high"
            ],
            0.0,
            rtol=0.0,
            atol=ZERO_TOLERANCE,
        )
    ]

    for _, row in zero_lambda.iterrows():

        if not math.isclose(
            float(
                row[
                    "interpolated_delta_H"
                ]
            ),
            0.0,
            rel_tol=0.0,
            abs_tol=ZERO_TOLERANCE,
        ):
            raise ValueError(
                "lambda=0 interpolation does not have "
                "Delta H=0."
            )

    print(
        "Interpolation robustness structural validation passed."
    )


# ===========================================================================
# CLASSIFICATION
# ===========================================================================


def classify_primary_row(
    row: pd.Series,
) -> str:
    """
    Classify one primary interpolation result.

    Categories:

    clear_benefit
        95% CI entirely below zero.

    exact_fallback
        lambda=0 and Delta H=0 by construction.

    uncertain
        nonzero observed interpolation whose interval includes zero.

    clear_harm
        95% CI entirely above zero.
    """

    lambda_high = float(
        row[
            "selected_lambda_high"
        ]
    )

    observed = float(
        row[
            "interpolated_delta_H"
        ]
    )

    lower = float(
        row[
            "ci_lower"
        ]
    )

    upper = float(
        row[
            "ci_upper"
        ]
    )

    if (
        math.isclose(
            lambda_high,
            0.0,
            rel_tol=0.0,
            abs_tol=ZERO_TOLERANCE,
        )
        and math.isclose(
            observed,
            0.0,
            rel_tol=0.0,
            abs_tol=ZERO_TOLERANCE,
        )
    ):
        return "exact_fallback"

    if upper < 0.0:
        return "clear_benefit"

    if lower > 0.0:
        return "clear_harm"

    return "uncertain"


# ===========================================================================
# SUMMARY
# ===========================================================================


def build_primary_summary(
    interpolation_test: pd.DataFrame,
    primary_bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Combine frozen test and primary-bootstrap results.
    """

    key_columns = [
        "training_fraction",
        "training_subset",
        "training_tokens",
        "from_order",
        "to_order",
        "selected_lambda_high",
    ]

    merged = interpolation_test.merge(
        primary_bootstrap,
        on=key_columns,
        suffixes=(
            "_test",
            "_bootstrap",
        ),
        validate="one_to_one",
    )

    # These values were independently checked during validate_inputs().
    merged[
        "interpolated_delta_H"
    ] = merged[
        "interpolated_delta_H_test"
    ]

    merged[
        "classification"
    ] = merged.apply(
        classify_primary_row,
        axis=1,
    )

    # ----------------------------------------------------------------------
    # Reduction in marginal loss relative to original fixed-order model.
    #
    # Positive means interpolation improved over the original transition.
    # ----------------------------------------------------------------------
    merged[
        "delta_H_reduction"
    ] = (
        merged[
            "original_delta_H"
        ]
        - merged[
            "interpolated_delta_H"
        ]
    )

    counts = (
        merged[
            "classification"
        ]
        .value_counts()
        .to_dict()
    )

    nonzero_lambda = merged[
        merged[
            "selected_lambda_high"
        ]
        > ZERO_TOLERANCE
    ]

    summary = {
        "n_transitions":
            int(
                len(merged)
            ),

        "clear_benefit":
            int(
                counts.get(
                    "clear_benefit",
                    0,
                )
            ),

        "exact_fallback":
            int(
                counts.get(
                    "exact_fallback",
                    0,
                )
            ),

        "uncertain":
            int(
                counts.get(
                    "uncertain",
                    0,
                )
            ),

        "clear_harm":
            int(
                counts.get(
                    "clear_harm",
                    0,
                )
            ),

        "nonzero_lambda_conditions":
            int(
                len(nonzero_lambda)
            ),

        "clear_benefit_among_nonzero_lambda":
            int(
                (
                    nonzero_lambda[
                        "classification"
                    ]
                    == "clear_benefit"
                ).sum()
            ),

        "interpolation_improved_over_original_high_order":
            int(
                (
                    merged[
                        "improvement_vs_original_high_test"
                    ]
                    > ZERO_TOLERANCE
                ).sum()
            ),

        "median_original_delta_H":
            float(
                merged[
                    "original_delta_H"
                ].median()
            ),

        "median_interpolated_delta_H":
            float(
                merged[
                    "interpolated_delta_H"
                ].median()
            ),
    }

    return (
        merged,
        summary,
    )


def build_sensitivity_summary(
    all_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize interval classifications for each tested block length.
    """

    rows = []

    for block_length, group in all_bootstrap.groupby(
        "block_length"
    ):

        exact_fallback = (
            np.isclose(
                group[
                    "selected_lambda_high"
                ],
                0.0,
                rtol=0.0,
                atol=ZERO_TOLERANCE,
            )
            & np.isclose(
                group[
                    "interpolated_delta_H"
                ],
                0.0,
                rtol=0.0,
                atol=ZERO_TOLERANCE,
            )
        )

        clear_benefit = (
            group[
                "ci_upper"
            ]
            < 0.0
        )

        clear_harm = (
            group[
                "ci_lower"
            ]
            > 0.0
        )

        uncertain_nonzero = (
            group[
                "contains_zero"
            ]
            & ~exact_fallback
        )

        rows.append(
            {
                "block_length":
                    int(
                        block_length
                    ),

                "clear_benefit":
                    int(
                        clear_benefit.sum()
                    ),

                "exact_fallback":
                    int(
                        exact_fallback.sum()
                    ),

                "uncertain_nonzero":
                    int(
                        uncertain_nonzero.sum()
                    ),

                "clear_harm":
                    int(
                        clear_harm.sum()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "block_length"
        )
    )


def build_transition_summary(
    primary_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize robustness behavior separately for each order transition.
    """

    rows = []

    grouped = primary_results.groupby(
        [
            "from_order",
            "to_order",
        ]
    )

    for (
        from_order,
        to_order,
    ), group in grouped:

        rows.append(
            {
                "from_order":
                    int(
                        from_order
                    ),

                "to_order":
                    int(
                        to_order
                    ),

                "n_training_sizes":
                    int(
                        len(group)
                    ),

                "median_lambda_high":
                    float(
                        group[
                            "selected_lambda_high"
                        ].median()
                    ),

                "median_original_delta_H":
                    float(
                        group[
                            "original_delta_H"
                        ].median()
                    ),

                "median_interpolated_delta_H":
                    float(
                        group[
                            "interpolated_delta_H"
                        ].median()
                    ),

                "clear_benefit":
                    int(
                        (
                            group[
                                "classification"
                            ]
                            == "clear_benefit"
                        ).sum()
                    ),

                "exact_fallback":
                    int(
                        (
                            group[
                                "classification"
                            ]
                            == "exact_fallback"
                        ).sum()
                    ),

                "uncertain":
                    int(
                        (
                            group[
                                "classification"
                            ]
                            == "uncertain"
                        ).sum()
                    ),

                "clear_harm":
                    int(
                        (
                            group[
                                "classification"
                            ]
                            == "clear_harm"
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ===========================================================================
# FIGURE 5
# ===========================================================================


def plot_robustness_comparison(
    primary_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Figure 5:

    Compare original fixed-order Delta H with validation-selected
    interpolated Delta H across training sizes.

    Each panel represents one model-order transition.

    Interpolated values include primary 95% paired block-bootstrap
    intervals.
    """

    transitions = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(
            11,
            11,
        ),
        sharex=True,
    )

    axes_flat = axes.flatten()

    for index, (
        from_order,
        to_order,
    ) in enumerate(
        transitions
    ):

        axis = axes_flat[
            index
        ]

        group = primary_results[
            (
                primary_results[
                    "from_order"
                ]
                == from_order
            )
            & (
                primary_results[
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
            ].to_numpy(dtype=float)
            * 100.0
        )

        original = group[
            "original_delta_H"
        ].to_numpy(dtype=float)

        interpolated = group[
            "interpolated_delta_H"
        ].to_numpy(dtype=float)

        lower = group[
            "ci_lower"
        ].to_numpy(dtype=float)

        upper = group[
            "ci_upper"
        ].to_numpy(dtype=float)

        error = np.vstack(
            [
                interpolated - lower,
                upper - interpolated,
            ]
        )

        axis.plot(
            x,
            original,
            marker="o",
            label="Fixed-order",
        )

        axis.errorbar(
            x,
            interpolated,
            yerr=error,
            marker="s",
            capsize=3,
            label="Interpolated",
        )

        axis.axhline(
            0.0,
            linestyle="--",
            linewidth=1,
        )

        axis.set_title(
            f"{from_order}"
            r"$\rightarrow$"
            f"{to_order}"
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

    # Remove unused sixth panel.
    figure.delaxes(
        axes_flat[-1]
    )

    figure.supxlabel(
        "Training fraction (%)"
    )

    figure.supylabel(
        r"$\Delta H$ (nats/token)"
    )

    figure.suptitle(
        "Fixed-order and interpolated marginal test loss",
        y=0.995,
    )

    handles, labels = (
        axes_flat[0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.965,
        ),
    )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.94,
        )
    )

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
    project_root = get_project_root()

    interpolation_test_path = (
        project_root
        / "results"
        / "interpolation_test.csv"
    )

    primary_bootstrap_path = (
        project_root
        / "results"
        / "analysis"
        / "interpolation_bootstrap_primary.csv"
    )

    all_bootstrap_path = (
        project_root
        / "results"
        / "analysis"
        / "interpolation_bootstrap.csv"
    )

    analysis_dir = (
        project_root
        / "results"
        / "analysis"
    )

    figure_path = (
        project_root
        / "figures"
        / "figure5_interpolation_robustness"
    )

    interpolation_test = pd.read_csv(
        interpolation_test_path
    )

    primary_bootstrap = pd.read_csv(
        primary_bootstrap_path
    )

    all_bootstrap = pd.read_csv(
        all_bootstrap_path
    )

    validate_inputs(
        interpolation_test,
        primary_bootstrap,
        all_bootstrap,
    )

    (
        primary_results,
        primary_summary,
    ) = build_primary_summary(
        interpolation_test,
        primary_bootstrap,
    )

    sensitivity_summary = (
        build_sensitivity_summary(
            all_bootstrap
        )
    )

    transition_summary = (
        build_transition_summary(
            primary_results
        )
    )

    primary_results.to_csv(
        analysis_dir
        / "interpolation_robustness_primary.csv",
        index=False,
    )

    sensitivity_summary.to_csv(
        analysis_dir
        / "interpolation_robustness_sensitivity.csv",
        index=False,
    )

    transition_summary.to_csv(
        analysis_dir
        / "interpolation_robustness_by_transition.csv",
        index=False,
    )

    with (
        analysis_dir
        / "interpolation_robustness_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            primary_summary,
            file,
            indent=2,
            sort_keys=True,
        )

    plot_robustness_comparison(
        primary_results,
        figure_path,
    )

    print(
        "\n============================================"
    )

    print(
        "INTERPOLATION ROBUSTNESS SUMMARY"
    )

    print(
        "============================================"
    )

    print(
        f"Clear benefit: "
        f"{primary_summary['clear_benefit']}/30"
    )

    print(
        f"Exact lower-order fallback: "
        f"{primary_summary['exact_fallback']}/30"
    )

    print(
        f"Uncertain nonzero effect: "
        f"{primary_summary['uncertain']}/30"
    )

    print(
        f"Clear harm: "
        f"{primary_summary['clear_harm']}/30"
    )

    print(
        "\nAmong nonzero-lambda conditions:"
    )

    print(
        f"Clear benefit: "
        f"{primary_summary['clear_benefit_among_nonzero_lambda']}/"
        f"{primary_summary['nonzero_lambda_conditions']}"
    )

    print(
        "\nInterpolation improved over the original "
        "fixed-order higher-order model:"
    )

    print(
        f"{primary_summary['interpolation_improved_over_original_high_order']}/30"
    )

    print(
        "\nBlock-length sensitivity:"
    )

    print(
        sensitivity_summary.to_string(
            index=False
        )
    )

    print(
        "\nFigure:"
    )

    print(
        figure_path.with_suffix(
            ".png"
        )
    )


if __name__ == "__main__":
    main()