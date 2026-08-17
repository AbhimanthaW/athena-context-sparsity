from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

# Allow analysis to run without an interactive display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ===========================================================================
# PATHS
# ===========================================================================


def get_project_root() -> Path:
    """
    analysis.py lives at:

        <project_root>/src/analysis.py

    so parents[1] is the repository root.
    """
    return Path(__file__).resolve().parents[1]


# ===========================================================================
# FILE HELPERS
# ===========================================================================


def load_results(
    main_results_path: Path,
    transitions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the frozen primary experiment outputs.

    This script deliberately analyses only already-frozen result tables.
    It does not read raw corpus data or rerun language models.
    """
    main = pd.read_csv(main_results_path)
    transitions = pd.read_csv(transitions_path)

    return main, transitions


def save_json(
    path: Path,
    payload: dict,
) -> None:
    """
    Save a JSON object with deterministic formatting.
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
# NUMERIC HELPERS
# ===========================================================================


def safe_spearman(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> float:
    """
    Compute a Spearman rank correlation safely.

    No inferential p-value is reported because observations across training
    sizes are derived from nested training subsets and therefore are not
    independent replicates.

    Returns NaN if:
    - fewer than two observations exist;
    - x is constant;
    - y is constant.
    """
    x_array = np.asarray(
        x,
        dtype=float,
    )

    y_array = np.asarray(
        y,
        dtype=float,
    )

    if len(x_array) < 2:
        return math.nan

    if np.allclose(
        x_array,
        x_array[0],
    ):
        return math.nan

    if np.allclose(
        y_array,
        y_array[0],
    ):
        return math.nan

    result = spearmanr(
        x_array,
        y_array,
    )

    return float(
        result.statistic
    )


def is_nondecreasing(
    values: list[float],
    tolerance: float = 1e-12,
) -> bool:
    """
    Return True if:

        x_1 <= x_2 <= ... <= x_n

    up to floating-point tolerance.
    """
    return all(
        later >= earlier - tolerance
        for earlier, later in zip(
            values,
            values[1:],
        )
    )


def is_nonincreasing(
    values: list[float],
    tolerance: float = 1e-12,
) -> bool:
    """
    Return True if:

        x_1 >= x_2 >= ... >= x_n

    up to floating-point tolerance.
    """
    return all(
        later <= earlier + tolerance
        for earlier, later in zip(
            values,
            values[1:],
        )
    )


def mean_absolute_error(
    prediction: pd.Series,
    target: pd.Series,
) -> float:
    """
    Compute mean absolute error.
    """
    return float(
        np.mean(
            np.abs(
                prediction.to_numpy(dtype=float)
                - target.to_numpy(dtype=float)
            )
        )
    )


def root_mean_squared_error(
    prediction: pd.Series,
    target: pd.Series,
) -> float:
    """
    Compute root mean squared error.
    """
    differences = (
        prediction.to_numpy(dtype=float)
        - target.to_numpy(dtype=float)
    )

    return float(
        np.sqrt(
            np.mean(
                differences ** 2
            )
        )
    )


# ===========================================================================
# STRUCTURAL VALIDATION
# ===========================================================================


def validate_primary_results(
    main: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    """
    Validate the frozen analysis inputs before calculating any scientific
    summaries.

    Checks include:
    - expected row counts;
    - duplicate conditions;
    - complete n=1,...,6 grids;
    - identical test denominators;
    - finite losses;
    - valid rates;
    - exact Delta H reconstruction;
    - transition/history alignment;
    - agreement between transitions.csv and main_results.csv.
    """

    # ----------------------------------------------------------------------
    # Expected row counts
    # ----------------------------------------------------------------------
    if len(main) != 36:
        raise ValueError(
            f"Expected 36 main-result rows, found {len(main)}."
        )

    if len(transitions) != 30:
        raise ValueError(
            f"Expected 30 transition rows, found {len(transitions)}."
        )

    # ----------------------------------------------------------------------
    # Duplicate conditions
    # ----------------------------------------------------------------------
    main_duplicates = main.duplicated(
        subset=[
            "training_subset",
            "n",
        ]
    )

    if main_duplicates.any():
        raise ValueError(
            "Duplicate (training_subset, n) rows "
            "found in main_results.csv."
        )

    transition_duplicates = transitions.duplicated(
        subset=[
            "training_subset",
            "from_order",
            "to_order",
        ]
    )

    if transition_duplicates.any():
        raise ValueError(
            "Duplicate transition rows found in transitions.csv."
        )

    # ----------------------------------------------------------------------
    # Every training subset must contain n=1,...,6 exactly once.
    # ----------------------------------------------------------------------
    expected_orders = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]

    for subset, group in main.groupby(
        "training_subset"
    ):
        observed_orders = sorted(
            group["n"].tolist()
        )

        if observed_orders != expected_orders:
            raise ValueError(
                f"{subset} has model orders {observed_orders}; "
                f"expected {expected_orders}."
            )

    # ----------------------------------------------------------------------
    # Every training subset should yield exactly five transitions.
    # ----------------------------------------------------------------------
    for subset, group in transitions.groupby(
        "training_subset"
    ):
        if len(group) != 5:
            raise ValueError(
                f"{subset} has {len(group)} transitions; "
                "expected exactly 5."
            )

    # ----------------------------------------------------------------------
    # Shared test target denominator
    # ----------------------------------------------------------------------
    unique_test_n = main[
        "test_N"
    ].unique()

    if len(unique_test_n) != 1:
        raise ValueError(
            "Models were not evaluated on the same "
            "number of test targets."
        )

    if int(unique_test_n[0]) != 98840:
        raise ValueError(
            f"Expected test_N=98,840, found {unique_test_n[0]}."
        )

    # ----------------------------------------------------------------------
    # Probability/rate columns must lie in [0,1].
    # ----------------------------------------------------------------------
    for column in [
        "S_k",
        "D_k",
        "test_U_k",
    ]:
        if not main[column].between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"{column} contains values outside [0,1]."
            )

    for column in [
        "S_k",
        "D_k",
        "U_k",
    ]:
        if not transitions[column].between(
            0.0,
            1.0,
        ).all():
            raise ValueError(
                f"Transition column {column} "
                "contains values outside [0,1]."
            )

    # ----------------------------------------------------------------------
    # Final losses must be finite.
    # ----------------------------------------------------------------------
    for column in [
        "test_cross_entropy",
        "test_perplexity",
    ]:
        if not np.isfinite(
            main[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

    if (
        main[
            "test_cross_entropy"
        ]
        < 0
    ).any():
        raise ValueError(
            "Negative cross-entropy detected."
        )

    if (
        main[
            "test_perplexity"
        ]
        <= 0
    ).any():
        raise ValueError(
            "Non-positive perplexity detected."
        )

    # ----------------------------------------------------------------------
    # Independently reconstruct Delta H.
    # ----------------------------------------------------------------------
    recomputed_delta = (
        transitions["H_current"]
        - transitions["H_previous"]
    )

    if not np.allclose(
        recomputed_delta.to_numpy(dtype=float),
        transitions[
            "delta_H"
        ].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Stored delta_H does not match "
            "H_current - H_previous."
        )

    # ----------------------------------------------------------------------
    # Transition history length must equal n-1 of the higher-order model.
    # ----------------------------------------------------------------------
    expected_history_length = (
        transitions["to_order"]
        - 1
    )

    if not np.array_equal(
        expected_history_length.to_numpy(),
        transitions[
            "history_length"
        ].to_numpy(),
    ):
        raise ValueError(
            "Transition history lengths are misaligned."
        )

    # ----------------------------------------------------------------------
    # Strong cross-check:
    #
    # Every transition row must exactly agree with the corresponding
    # adjacent rows from main_results.csv.
    # ----------------------------------------------------------------------
    for _, transition in transitions.iterrows():

        subset = transition[
            "training_subset"
        ]

        from_order = int(
            transition["from_order"]
        )

        to_order = int(
            transition["to_order"]
        )

        previous_rows = main[
            (
                main["training_subset"]
                == subset
            )
            & (
                main["n"]
                == from_order
            )
        ]

        current_rows = main[
            (
                main["training_subset"]
                == subset
            )
            & (
                main["n"]
                == to_order
            )
        ]

        if len(previous_rows) != 1:
            raise ValueError(
                f"Could not uniquely locate previous model "
                f"for {subset}, n={from_order}."
            )

        if len(current_rows) != 1:
            raise ValueError(
                f"Could not uniquely locate current model "
                f"for {subset}, n={to_order}."
            )

        previous = previous_rows.iloc[0]
        current = current_rows.iloc[0]

        # --------------------------------------------------------------
        # Cross-entropy agreement
        # --------------------------------------------------------------
        if not math.isclose(
            float(
                transition[
                    "H_previous"
                ]
            ),
            float(
                previous[
                    "test_cross_entropy"
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"H_previous mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )

        if not math.isclose(
            float(
                transition[
                    "H_current"
                ]
            ),
            float(
                current[
                    "test_cross_entropy"
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"H_current mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )

        # --------------------------------------------------------------
        # Predictor values must come from the HIGHER-order model.
        # --------------------------------------------------------------
        comparisons = [
            (
                "S_k",
                "S_k",
            ),
            (
                "D_k",
                "D_k",
            ),
            (
                "U_k",
                "test_U_k",
            ),
        ]

        for transition_field, main_field in comparisons:

            if not math.isclose(
                float(
                    transition[
                        transition_field
                    ]
                ),
                float(
                    current[
                        main_field
                    ]
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{transition_field} mismatch for "
                    f"{subset}, {from_order}->{to_order}."
                )

        # --------------------------------------------------------------
        # Alpha alignment
        # --------------------------------------------------------------
        if not math.isclose(
            float(
                transition[
                    "selected_alpha_previous"
                ]
            ),
            float(
                previous[
                    "selected_alpha"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Previous alpha mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )

        if not math.isclose(
            float(
                transition[
                    "selected_alpha_current"
                ]
            ),
            float(
                current[
                    "selected_alpha"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Current alpha mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )

        # --------------------------------------------------------------
        # Training-size alignment
        # --------------------------------------------------------------
        if (
            int(
                transition[
                    "training_tokens"
                ]
            )
            != int(
                current[
                    "training_tokens"
                ]
            )
        ):
            raise ValueError(
                f"Training-token mismatch for "
                f"{subset}, {from_order}->{to_order}."
            )

    print(
        "Primary-result structural validation passed."
    )


# ===========================================================================
# H1 — GENERALISATION REVERSALS
# ===========================================================================


def analyze_h1(
    transitions: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    H1:
        At least one increase in model order harms held-out performance.

    Delta H_n > 0:
        higher order performs worse.

    Delta H_n < 0:
        higher order performs better.
    """

    tolerance = 1e-12

    positive = int(
        (
            transitions[
                "delta_H"
            ]
            > tolerance
        ).sum()
    )

    negative = int(
        (
            transitions[
                "delta_H"
            ]
            < -tolerance
        ).sum()
    )

    approximately_zero = int(
        len(transitions)
        - positive
        - negative
    )

    by_transition = (
        transitions
        .groupby(
            [
                "from_order",
                "to_order",
            ],
            as_index=False,
        )
        .agg(
            n_observations=(
                "delta_H",
                "size",
            ),
            min_delta_H=(
                "delta_H",
                "min",
            ),
            median_delta_H=(
                "delta_H",
                "median",
            ),
            max_delta_H=(
                "delta_H",
                "max",
            ),
        )
    )

    summary = {
        "n_transitions":
            int(
                len(transitions)
            ),

        "positive_delta_H":
            positive,

        "negative_delta_H":
            negative,

        "approximately_zero_delta_H":
            approximately_zero,

        "fraction_positive":
            float(
                positive
                / len(transitions)
            ),

        "minimum_delta_H":
            float(
                transitions[
                    "delta_H"
                ].min()
            ),

        "maximum_delta_H":
            float(
                transitions[
                    "delta_H"
                ].max()
            ),
    }

    return (
        summary,
        by_transition,
    )


# ===========================================================================
# H2 — SPARSITY STRUCTURE
# ===========================================================================


def analyze_h2(
    main: pd.DataFrame,
) -> pd.DataFrame:
    """
    H2 examines how singleton-history rate changes with:

    1. history length k at fixed training size;
    2. training size at fixed nontrivial history length k >= 1.

    k=0 is excluded from the training-size analysis because:

        S_0 = 0

    identically, so it provides no meaningful evidence about sparsity.
    """

    rows = []

    # ----------------------------------------------------------------------
    # H2a:
    # S_k versus history length for each fixed training subset.
    # ----------------------------------------------------------------------
    for subset, group in main.groupby(
        "training_subset"
    ):

        ordered = group.sort_values(
            "k"
        )

        k_values = ordered[
            "k"
        ].tolist()

        s_values = ordered[
            "S_k"
        ].tolist()

        rho = safe_spearman(
            ordered["k"],
            ordered["S_k"],
        )

        rows.append(
            {
                "analysis":
                    "S_k_vs_history_length",

                "fixed_variable":
                    "training_subset",

                "fixed_value":
                    subset,

                "n_observations":
                    int(
                        len(ordered)
                    ),

                "expected_direction":
                    "nondecreasing",

                "monotonic":
                    is_nondecreasing(
                        s_values
                    ),

                "spearman_rho":
                    rho,

                "x_values":
                    ",".join(
                        str(value)
                        for value
                        in k_values
                    ),

                "y_values":
                    ",".join(
                        f"{value:.12g}"
                        for value
                        in s_values
                    ),
            }
        )

    # ----------------------------------------------------------------------
    # H2b:
    # S_k versus training size for each fixed k >= 1.
    # ----------------------------------------------------------------------
    nontrivial = main[
        main["k"] >= 1
    ].copy()

    for k, group in nontrivial.groupby(
        "k"
    ):

        ordered = group.sort_values(
            "training_fraction"
        )

        training_values = ordered[
            "training_tokens"
        ].tolist()

        s_values = ordered[
            "S_k"
        ].tolist()

        rho = safe_spearman(
            ordered[
                "training_tokens"
            ],
            ordered[
                "S_k"
            ],
        )

        rows.append(
            {
                "analysis":
                    "S_k_vs_training_size",

                "fixed_variable":
                    "history_length",

                "fixed_value":
                    str(
                        int(k)
                    ),

                "n_observations":
                    int(
                        len(ordered)
                    ),

                "expected_direction":
                    "nonincreasing",

                "monotonic":
                    is_nonincreasing(
                        s_values
                    ),

                "spearman_rho":
                    rho,

                "x_values":
                    ",".join(
                        str(value)
                        for value
                        in training_values
                    ),

                "y_values":
                    ",".join(
                        f"{value:.12g}"
                        for value
                        in s_values
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ===========================================================================
# H3 — TRAINING SPARSITY VS HELD-OUT COVERAGE
# ===========================================================================


def analyze_h3(
    main: pd.DataFrame,
) -> pd.DataFrame:
    """
    H3:
        Does training-only singleton-history rate S_k track actual held-out
        unseen-history rate U_k?

    Primary analysis:
        within each fixed history length k=1,...,5 across training sizes.

    Secondary:
        pooled across all k=1,...,5.

    D_k is included as a simple competing training-only baseline.

    In addition to rank association, MAE and RMSE measure numerical
    calibration against held-out unseen-history rate.
    """

    rows = []

    relevant = main[
        main["k"] >= 1
    ].copy()

    # ----------------------------------------------------------------------
    # History-length-specific results
    # ----------------------------------------------------------------------
    for k, group in relevant.groupby(
        "k"
    ):

        ordered = group.sort_values(
            "training_fraction"
        )

        rho_s = safe_spearman(
            ordered["S_k"],
            ordered["test_U_k"],
        )

        rho_d = safe_spearman(
            ordered["D_k"],
            ordered["test_U_k"],
        )

        mae_s = mean_absolute_error(
            ordered["S_k"],
            ordered["test_U_k"],
        )

        mae_d = mean_absolute_error(
            ordered["D_k"],
            ordered["test_U_k"],
        )

        rmse_s = root_mean_squared_error(
            ordered["S_k"],
            ordered["test_U_k"],
        )

        rmse_d = root_mean_squared_error(
            ordered["D_k"],
            ordered["test_U_k"],
        )

        rows.append(
            {
                "scope":
                    "fixed_history_length",

                "history_length":
                    int(k),

                "n_observations":
                    int(
                        len(ordered)
                    ),

                "rho_S_vs_U":
                    rho_s,

                "rho_D_vs_U":
                    rho_d,

                "MAE_S_vs_U":
                    mae_s,

                "MAE_D_vs_U":
                    mae_d,

                "RMSE_S_vs_U":
                    rmse_s,

                "RMSE_D_vs_U":
                    rmse_d,
            }
        )

    # ----------------------------------------------------------------------
    # Secondary pooled result
    # ----------------------------------------------------------------------
    rho_s_pool = safe_spearman(
        relevant["S_k"],
        relevant["test_U_k"],
    )

    rho_d_pool = safe_spearman(
        relevant["D_k"],
        relevant["test_U_k"],
    )

    pooled_mae_s = mean_absolute_error(
        relevant["S_k"],
        relevant["test_U_k"],
    )

    pooled_mae_d = mean_absolute_error(
        relevant["D_k"],
        relevant["test_U_k"],
    )

    pooled_rmse_s = root_mean_squared_error(
        relevant["S_k"],
        relevant["test_U_k"],
    )

    pooled_rmse_d = root_mean_squared_error(
        relevant["D_k"],
        relevant["test_U_k"],
    )

    rows.append(
        {
            "scope":
                "pooled_secondary",

            "history_length":
                np.nan,

            "n_observations":
                int(
                    len(relevant)
                ),

            "rho_S_vs_U":
                rho_s_pool,

            "rho_D_vs_U":
                rho_d_pool,

            "MAE_S_vs_U":
                pooled_mae_s,

            "MAE_D_vs_U":
                pooled_mae_d,

            "RMSE_S_vs_U":
                pooled_rmse_s,

            "RMSE_D_vs_U":
                pooled_rmse_d,
        }
    )

    return pd.DataFrame(
        rows
    )


# ===========================================================================
# H4 — SPARSITY VS MARGINAL CONTEXT VALUE
# ===========================================================================


def analyze_h4(
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    H4:
        Within each fixed order transition, does larger training-only
        singleton-history rate correspond to larger Delta H?

    Primary:
        transition-specific Spearman correlations across training sizes.

    Also report:
        D_k versus Delta H;
        U_k versus Delta H;
        training size versus Delta H.

    The training-size comparison is critical because S_k varies strongly
    with training size. Therefore this experiment cannot establish that
    S_k provides independent predictive information beyond training size.
    """

    rows = []

    grouped = transitions.groupby(
        [
            "from_order",
            "to_order",
        ]
    )

    for (
        from_order,
        to_order,
    ), group in grouped:

        ordered = group.sort_values(
            "training_fraction"
        )

        rho_s = safe_spearman(
            ordered["S_k"],
            ordered["delta_H"],
        )

        rho_d = safe_spearman(
            ordered["D_k"],
            ordered["delta_H"],
        )

        rho_u = safe_spearman(
            ordered["U_k"],
            ordered["delta_H"],
        )

        rho_training = safe_spearman(
            ordered[
                "training_tokens"
            ],
            ordered[
                "delta_H"
            ],
        )

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

                "history_length":
                    int(
                        ordered[
                            "history_length"
                        ].iloc[0]
                    ),

                "n_observations":
                    int(
                        len(ordered)
                    ),

                "rho_S_vs_delta_H":
                    rho_s,

                "rho_D_vs_delta_H":
                    rho_d,

                "rho_U_vs_delta_H":
                    rho_u,

                "rho_training_size_vs_delta_H":
                    rho_training,
            }
        )

    result = pd.DataFrame(
        rows
    )

    median_rho_s = float(
        result[
            "rho_S_vs_delta_H"
        ].median()
    )

    # ----------------------------------------------------------------------
    # Secondary pooled summaries.
    #
    # These are not primary because pooling mixes fundamentally different
    # order transitions.
    # ----------------------------------------------------------------------
    rho_s_pool = safe_spearman(
        transitions["S_k"],
        transitions["delta_H"],
    )

    rho_d_pool = safe_spearman(
        transitions["D_k"],
        transitions["delta_H"],
    )

    rho_u_pool = safe_spearman(
        transitions["U_k"],
        transitions["delta_H"],
    )

    summary = {
        "median_transition_rho_S_vs_delta_H":
            median_rho_s,

        "pooled_secondary_rho_S_vs_delta_H":
            rho_s_pool,

        "pooled_secondary_rho_D_vs_delta_H":
            rho_d_pool,

        "pooled_secondary_rho_U_vs_delta_H":
            rho_u_pool,

        "important_interpretation":
            (
                "Transition-specific correlations are primary. "
                "Training size and S_k are strongly coupled, so these "
                "results do not identify predictive information from S_k "
                "independent of training size."
            ),
    }

    return (
        result,
        summary,
    )


# ===========================================================================
# FIGURE HELPERS
# ===========================================================================


def save_figure(
    figure: plt.Figure,
    base_path: Path,
) -> None:
    """
    Save figures as both PNG and PDF.

    PNG:
        convenient for quick inspection and GitHub.

    PDF:
        preferable for eventual paper inclusion.
    """
    base_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        base_path.with_suffix(
            ".png"
        ),
        dpi=220,
        bbox_inches="tight",
    )

    figure.savefig(
        base_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ===========================================================================
# FIGURE 1
# ===========================================================================


def plot_test_cross_entropy_by_order(
    main: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Figure 1:
        Test cross-entropy versus n-gram order.

    One line is shown for each training fraction.
    """

    figure, axis = plt.subplots(
        figsize=(
            8.5,
            5.5,
        )
    )

    ordered_fractions = sorted(
        main[
            "training_fraction"
        ].unique()
    )

    for fraction in ordered_fractions:

        group = (
            main[
                main[
                    "training_fraction"
                ]
                == fraction
            ]
            .sort_values(
                "n"
            )
        )

        label = (
            f"{int(round(fraction * 100))}%"
        )

        axis.plot(
            group["n"],
            group[
                "test_cross_entropy"
            ],
            marker="o",
            label=label,
        )

    axis.set_title(
        "Test cross-entropy by n-gram order"
    )

    axis.set_xlabel(
        "Model order n"
    )

    axis.set_ylabel(
        "Test cross-entropy (nats/token)"
    )

    axis.set_xticks(
        [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        title="Training fraction",
        frameon=False,
        ncols=2,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_path,
    )


# ===========================================================================
# FIGURE 2
# ===========================================================================


def plot_s_vs_u(
    main: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Figure 2:
        Training singleton-history rate S_k
        versus
        held-out unseen-history rate U_k.

    k=0 is omitted because both quantities are trivially zero for unigrams.

    The y=x line is a calibration reference, not a fitted model.
    """

    relevant = main[
        main["k"] >= 1
    ].copy()

    figure, axis = plt.subplots(
        figsize=(
            7.0,
            6.0,
        )
    )

    for k in sorted(
        relevant[
            "k"
        ].unique()
    ):

        group = (
            relevant[
                relevant["k"]
                == k
            ]
            .sort_values(
                "training_fraction"
            )
        )

        axis.plot(
            group["S_k"],
            group[
                "test_U_k"
            ],
            marker="o",
            label=f"k={int(k)}",
        )

    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        linewidth=1,
        label="y = x",
    )

    axis.set_title(
        "Training singleton-history rate vs test unseen-history rate"
    )

    axis.set_xlabel(
        r"Singleton-history rate $S_k$"
    )

    axis.set_ylabel(
        r"Test unseen-history rate $U_k$"
    )

    axis.set_xlim(
        0,
        1.02,
    )

    axis.set_ylim(
        0,
        1.02,
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        title="History length",
        frameon=False,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_path,
    )


# ===========================================================================
# FIGURE 3
# ===========================================================================


def plot_s_vs_delta_h(
    transitions: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Figure 3:
        Singleton-history rate versus marginal test loss.

    Each line represents one fixed order transition.

    Points are connected in training-size order.

    No pooled regression line is shown because pooling order transitions
    would mix different model-comparison regimes.
    """

    figure, axis = plt.subplots(
        figsize=(
            8.0,
            6.0,
        )
    )

    grouped = transitions.groupby(
        [
            "from_order",
            "to_order",
        ]
    )

    for (
        from_order,
        to_order,
    ), group in grouped:

        ordered = group.sort_values(
            "training_fraction"
        )

        axis.plot(
            ordered["S_k"],
            ordered["delta_H"],
            marker="o",
            label=(
                f"{int(from_order)}"
                r"$\rightarrow$"
                f"{int(to_order)}"
            ),
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    axis.set_title(
        "Singleton-history rate vs marginal test loss"
    )

    axis.set_xlabel(
        r"Singleton-history rate $S_{n-1}$"
    )

    axis.set_ylabel(
        r"$\Delta H_n = H_n - H_{n-1}$ (nats/token)"
    )

    axis.set_xlim(
        0,
        1.02,
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        title="Order transition",
        frameon=False,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_path,
    )


# ===========================================================================
# CONSOLE SUMMARY
# ===========================================================================


def print_analysis_summary(
    h1: dict,
    h2: pd.DataFrame,
    h3: pd.DataFrame,
    h4: pd.DataFrame,
    h4_summary: dict,
) -> None:
    """
    Print a concise human-readable summary of the frozen primary analysis.
    """

    print(
        "\n============================================"
    )

    print(
        "PRIMARY ANALYSIS SUMMARY"
    )

    print(
        "============================================"
    )

    # ----------------------------------------------------------------------
    # H1
    # ----------------------------------------------------------------------
    print(
        "\nH1 — Generalisation reversals"
    )

    print(
        f"Delta H > 0: "
        f"{h1['positive_delta_H']}/"
        f"{h1['n_transitions']}"
    )

    print(
        f"Delta H < 0: "
        f"{h1['negative_delta_H']}/"
        f"{h1['n_transitions']}"
    )

    print(
        f"Minimum Delta H: "
        f"{h1['minimum_delta_H']:.6f}"
    )

    print(
        f"Maximum Delta H: "
        f"{h1['maximum_delta_H']:.6f}"
    )

    # ----------------------------------------------------------------------
    # H2
    # ----------------------------------------------------------------------
    print(
        "\nH2 — Singleton-history sparsity structure"
    )

    h2_history = h2[
        h2["analysis"]
        == "S_k_vs_history_length"
    ]

    h2_training = h2[
        h2["analysis"]
        == "S_k_vs_training_size"
    ]

    print(
        "S_k nondecreasing with k: "
        f"{int(h2_history['monotonic'].sum())}/"
        f"{len(h2_history)} training sizes"
    )

    print(
        "S_k nonincreasing with training size: "
        f"{int(h2_training['monotonic'].sum())}/"
        f"{len(h2_training)} nontrivial history lengths"
    )

    # ----------------------------------------------------------------------
    # H3
    # ----------------------------------------------------------------------
    print(
        "\nH3 — S_k versus held-out U_k"
    )

    fixed_h3 = h3[
        h3["scope"]
        == "fixed_history_length"
    ]

    for _, row in fixed_h3.iterrows():

        print(
            f"k={int(row['history_length'])}: "
            f"rho(S,U)="
            f"{row['rho_S_vs_U']:.6f} | "
            f"MAE(S,U)="
            f"{row['MAE_S_vs_U']:.6f} | "
            f"MAE(D,U)="
            f"{row['MAE_D_vs_U']:.6f}"
        )

    pooled_h3 = h3[
        h3["scope"]
        == "pooled_secondary"
    ].iloc[0]

    print(
        "Pooled secondary rho(S,U): "
        f"{pooled_h3['rho_S_vs_U']:.6f}"
    )

    print(
        "Pooled MAE(S,U): "
        f"{pooled_h3['MAE_S_vs_U']:.6f}"
    )

    print(
        "Pooled MAE(D,U): "
        f"{pooled_h3['MAE_D_vs_U']:.6f}"
    )

    print(
        "Pooled RMSE(S,U): "
        f"{pooled_h3['RMSE_S_vs_U']:.6f}"
    )

    print(
        "Pooled RMSE(D,U): "
        f"{pooled_h3['RMSE_D_vs_U']:.6f}"
    )

    # ----------------------------------------------------------------------
    # H4
    # ----------------------------------------------------------------------
    print(
        "\nH4 — S_k versus Delta H"
    )

    for _, row in h4.iterrows():

        print(
            f"{int(row['from_order'])}"
            f"->{int(row['to_order'])}: "
            f"rho(S,DeltaH)="
            f"{row['rho_S_vs_delta_H']:.6f} | "
            f"rho(training_size,DeltaH)="
            f"{row['rho_training_size_vs_delta_H']:.6f}"
        )

    print(
        "Median transition-specific rho(S,DeltaH): "
        f"{h4_summary['median_transition_rho_S_vs_delta_H']:.6f}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "Training size and S_k are strongly coupled. "
        "H4 therefore measures association across nested "
        "training sizes, not predictive information from S_k "
        "independent of training size."
    )

    print(
        "Naive rank-correlation p-values are intentionally "
        "not reported because the nested subsets are dependent."
    )


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    project_root = get_project_root()

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

    analysis_dir = (
        project_root
        / "results"
        / "analysis"
    )

    figures_dir = (
        project_root
        / "figures"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Load frozen primary outputs.
    # ----------------------------------------------------------------------
    main_results, transitions = load_results(
        main_results_path,
        transitions_path,
    )

    # ----------------------------------------------------------------------
    # Validate the two frozen result tables before analysing them.
    # ----------------------------------------------------------------------
    validate_primary_results(
        main_results,
        transitions,
    )

    # ----------------------------------------------------------------------
    # H1
    # ----------------------------------------------------------------------
    (
        h1_summary,
        h1_by_transition,
    ) = analyze_h1(
        transitions
    )

    h1_by_transition.to_csv(
        analysis_dir
        / "h1_generalisation_reversals.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # H2
    # ----------------------------------------------------------------------
    h2_results = analyze_h2(
        main_results
    )

    h2_results.to_csv(
        analysis_dir
        / "h2_sparsity_structure.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # H3
    # ----------------------------------------------------------------------
    h3_results = analyze_h3(
        main_results
    )

    h3_results.to_csv(
        analysis_dir
        / "h3_s_vs_unseen_rate.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # H4
    # ----------------------------------------------------------------------
    (
        h4_results,
        h4_summary,
    ) = analyze_h4(
        transitions
    )

    h4_results.to_csv(
        analysis_dir
        / "h4_s_vs_delta_h.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # Extract pooled H3 row once.
    # ----------------------------------------------------------------------
    pooled_h3 = h3_results[
        h3_results["scope"]
        == "pooled_secondary"
    ].iloc[0]

    # ----------------------------------------------------------------------
    # Machine-readable summary.
    # ----------------------------------------------------------------------
    summary = {
        "H1": h1_summary,

        "H2": {
            "all_training_sizes_nondecreasing_with_k":
                bool(
                    h2_results[
                        h2_results[
                            "analysis"
                        ]
                        == "S_k_vs_history_length"
                    ][
                        "monotonic"
                    ].all()
                ),

            "all_nontrivial_history_lengths_nonincreasing_with_training_size":
                bool(
                    h2_results[
                        h2_results[
                            "analysis"
                        ]
                        == "S_k_vs_training_size"
                    ][
                        "monotonic"
                    ].all()
                ),
        },

        "H3": {
            "pooled_secondary_rho_S_vs_U":
                float(
                    pooled_h3[
                        "rho_S_vs_U"
                    ]
                ),

            "pooled_secondary_rho_D_vs_U":
                float(
                    pooled_h3[
                        "rho_D_vs_U"
                    ]
                ),

            "pooled_MAE_S_vs_U":
                float(
                    pooled_h3[
                        "MAE_S_vs_U"
                    ]
                ),

            "pooled_MAE_D_vs_U":
                float(
                    pooled_h3[
                        "MAE_D_vs_U"
                    ]
                ),

            "pooled_RMSE_S_vs_U":
                float(
                    pooled_h3[
                        "RMSE_S_vs_U"
                    ]
                ),

            "pooled_RMSE_D_vs_U":
                float(
                    pooled_h3[
                        "RMSE_D_vs_U"
                    ]
                ),
        },

        "H4": h4_summary,

        "statistical_note": (
            "Naive Spearman p-values are not reported because "
            "observations across training sizes are derived from "
            "nested training subsets and therefore are dependent."
        ),

        "methodological_caveat": (
            "S_k varies systematically with training size. "
            "The current experiment measures association between "
            "S_k and Delta H within fixed order transitions across "
            "training sizes, but does not identify predictive "
            "information from S_k independent of training size."
        ),
    }

    save_json(
        analysis_dir
        / "summary.json",
        summary,
    )

    # ----------------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------------
    plot_test_cross_entropy_by_order(
        main_results,
        figures_dir
        / "figure1_test_cross_entropy_by_order",
    )

    plot_s_vs_u(
        main_results,
        figures_dir
        / "figure2_singleton_vs_unseen_rate",
    )

    plot_s_vs_delta_h(
        transitions,
        figures_dir
        / "figure3_singleton_vs_delta_h",
    )

    # ----------------------------------------------------------------------
    # Console output
    # ----------------------------------------------------------------------
    print_analysis_summary(
        h1_summary,
        h2_results,
        h3_results,
        h4_results,
        h4_summary,
    )

    print(
        "\nAnalysis tables written to:"
    )

    print(
        analysis_dir
    )

    print(
        "\nFigures written to:"
    )

    print(
        figures_dir
    )


if __name__ == "__main__":
    main()