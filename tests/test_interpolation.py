import math

import numpy as np
import pytest

from src.interpolation import (
    interpolated_cross_entropy,
    interpolated_losses,
)


def test_interpolation_endpoints():
    lower = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    higher = np.array(
        [
            4.0,
            5.0,
            6.0,
        ]
    )

    result_lower = interpolated_losses(
        lower,
        higher,
        lambda_high=0.0,
    )

    result_higher = interpolated_losses(
        lower,
        higher,
        lambda_high=1.0,
    )

    assert np.array_equal(
        result_lower,
        lower,
    )

    assert np.array_equal(
        result_higher,
        higher,
    )


def test_interpolation_matches_manual_probability_mixture():
    """
    Lower model:

        q = [0.8, 0.2]

    Higher model:

        q = [0.6, 0.4]

    With lambda_high = 0.25:

        q_mix
        =
        0.75 q_lower
        +
        0.25 q_higher

        =
        [0.75, 0.25]
    """

    lower_probabilities = np.array(
        [
            0.8,
            0.2,
        ]
    )

    higher_probabilities = np.array(
        [
            0.6,
            0.4,
        ]
    )

    lower_losses = -np.log(
        lower_probabilities
    )

    higher_losses = -np.log(
        higher_probabilities
    )

    result = interpolated_losses(
        lower_losses,
        higher_losses,
        lambda_high=0.25,
    )

    expected_probabilities = np.array(
        [
            0.75,
            0.25,
        ]
    )

    expected_losses = -np.log(
        expected_probabilities
    )

    assert np.allclose(
        result,
        expected_losses,
    )


def test_interpolated_cross_entropy_is_mean_loss():
    lower = np.array(
        [
            1.0,
            1.5,
            2.0,
        ]
    )

    higher = np.array(
        [
            2.0,
            1.0,
            2.5,
        ]
    )

    losses = interpolated_losses(
        lower,
        higher,
        lambda_high=0.4,
    )

    cross_entropy = interpolated_cross_entropy(
        lower,
        higher,
        lambda_high=0.4,
    )

    assert cross_entropy == pytest.approx(
        float(
            losses.mean()
        )
    )


def test_invalid_interpolation_inputs_rejected():
    with pytest.raises(
        ValueError
    ):
        interpolated_losses(
            [1.0],
            [1.0],
            lambda_high=-0.1,
        )

    with pytest.raises(
        ValueError
    ):
        interpolated_losses(
            [1.0],
            [1.0],
            lambda_high=1.1,
        )

    with pytest.raises(
        ValueError
    ):
        interpolated_losses(
            [1.0, 2.0],
            [1.0],
            lambda_high=0.5,
        )