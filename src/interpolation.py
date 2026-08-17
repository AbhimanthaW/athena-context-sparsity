from __future__ import annotations

import numpy as np


def validate_lambda(
    lambda_high: float,
) -> None:
    """
    Validate interpolation weight.

    lambda_high is the probability weight assigned to the HIGHER-order model.

        lambda_high = 0
            lower-order model only

        lambda_high = 1
            higher-order model only
    """

    if not 0.0 <= lambda_high <= 1.0:
        raise ValueError(
            "lambda_high must lie in [0, 1]."
        )


def interpolated_losses(
    lower_losses,
    higher_losses,
    lambda_high: float,
) -> np.ndarray:
    """
    Compute per-target negative log-probabilities for the interpolation

        q_mix
        =
        (1 - lambda_high) q_lower
        +
        lambda_high q_higher.

    Inputs are per-target negative log-probabilities:

        lower_losses[t] = -log q_lower,t
        higher_losses[t] = -log q_higher,t

    We work in log-space using np.logaddexp for numerical stability.
    """

    validate_lambda(
        lambda_high
    )

    lower = np.asarray(
        lower_losses,
        dtype=float,
    )

    higher = np.asarray(
        higher_losses,
        dtype=float,
    )

    if lower.ndim != 1:
        raise ValueError(
            "lower_losses must be one-dimensional."
        )

    if higher.ndim != 1:
        raise ValueError(
            "higher_losses must be one-dimensional."
        )

    if len(lower) != len(higher):
        raise ValueError(
            "Lower- and higher-order loss arrays "
            "must have identical lengths."
        )

    if len(lower) == 0:
        raise ValueError(
            "Cannot interpolate empty loss arrays."
        )

    if not np.isfinite(
        lower
    ).all():
        raise ValueError(
            "lower_losses contains non-finite values."
        )

    if not np.isfinite(
        higher
    ).all():
        raise ValueError(
            "higher_losses contains non-finite values."
        )

    # Exact endpoint behavior is useful both scientifically
    # and for unit testing.
    if lambda_high == 0.0:
        return lower.copy()

    if lambda_high == 1.0:
        return higher.copy()

    # Since:
    #
    # loss = -log(q)
    #
    # then:
    #
    # log(q) = -loss
    #
    log_q_lower = -lower
    log_q_higher = -higher

    log_weight_lower = np.log1p(
        -lambda_high
    )

    log_weight_higher = np.log(
        lambda_high
    )

    log_mixture_probability = np.logaddexp(
        log_weight_lower
        + log_q_lower,
        log_weight_higher
        + log_q_higher,
    )

    return -log_mixture_probability


def interpolated_cross_entropy(
    lower_losses,
    higher_losses,
    lambda_high: float,
) -> float:
    """
    Return mean interpolated negative log-probability.
    """

    losses = interpolated_losses(
        lower_losses,
        higher_losses,
        lambda_high,
    )

    return float(
        losses.mean()
    )