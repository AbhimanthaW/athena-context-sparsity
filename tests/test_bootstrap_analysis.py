import numpy as np
import pytest

from src.bootstrap_analysis import (
    circular_block_bootstrap_means,
    circular_window_sums,
    percentile_interval,
)


def test_circular_window_sums():
    values = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    result = circular_window_sums(
        values,
        window_length=3,
    )

    # Circular windows:
    #
    # 1+2+3 = 6
    # 2+3+4 = 9
    # 3+4+1 = 8
    # 4+1+2 = 7
    expected = np.array(
        [
            6.0,
            9.0,
            8.0,
            7.0,
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_constant_differences_bootstrap_exactly():
    """
    If every paired loss difference is exactly 0.25, every possible
    block-bootstrap replicate must also have mean 0.25.
    """

    differences = np.full(
        100,
        0.25,
    )

    rng = np.random.default_rng(
        42
    )

    estimates = (
        circular_block_bootstrap_means(
            differences=differences,
            block_length=10,
            replicates=100,
            rng=rng,
        )
    )

    assert np.allclose(
        estimates,
        0.25,
    )


def test_bootstrap_is_reproducible():
    differences = np.arange(
        100,
        dtype=float,
    )

    rng_a = np.random.default_rng(
        42
    )

    rng_b = np.random.default_rng(
        42
    )

    result_a = (
        circular_block_bootstrap_means(
            differences=differences,
            block_length=10,
            replicates=50,
            rng=rng_a,
        )
    )

    result_b = (
        circular_block_bootstrap_means(
            differences=differences,
            block_length=10,
            replicates=50,
            rng=rng_b,
        )
    )

    assert np.array_equal(
        result_a,
        result_b,
    )


def test_percentile_interval():
    values = np.arange(
        100,
        dtype=float,
    )

    lower, upper = percentile_interval(
        values,
        confidence_level=0.95,
    )

    assert lower == pytest.approx(
        np.quantile(
            values,
            0.025,
        )
    )

    assert upper == pytest.approx(
        np.quantile(
            values,
            0.975,
        )
    )