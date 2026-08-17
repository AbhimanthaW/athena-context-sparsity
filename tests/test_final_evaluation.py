import pytest

from src.final_evaluation import build_transitions


def make_row(
    n,
    cross_entropy,
    s_k,
    d_k,
    u_k,
):
    """
    Construct a minimal synthetic final-result row for transition tests.
    """
    return {
        "training_fraction": 0.2,
        "training_subset": "20%",
        "training_tokens": 158000,
        "n": n,
        "k": n - 1,
        "selected_alpha": 0.01,
        "S_k": s_k,
        "D_k": d_k,
        "test_U_k": u_k,
        "test_cross_entropy": cross_entropy,
    }


def test_transition_delta_h_and_sparsity_alignment():
    """
    Verify:

        Delta H_n = H_n - H_(n-1)

    and that the sparsity statistic paired with the transition comes from
    the HIGHER-order model.

    For transition 1 -> 2:

        H_1 = 2.0
        H_2 = 1.8

        Delta H_2 = -0.2

    and S_1 should come from the n=2 row.
    """
    rows = [
        make_row(
            n=1,
            cross_entropy=2.0,
            s_k=0.0,
            d_k=0.0,
            u_k=0.0,
        ),
        make_row(
            n=2,
            cross_entropy=1.8,
            s_k=0.1,
            d_k=0.2,
            u_k=0.15,
        ),
        make_row(
            n=3,
            cross_entropy=2.1,
            s_k=0.5,
            d_k=0.6,
            u_k=0.55,
        ),
    ]

    transitions = build_transitions(
        rows
    )

    assert len(transitions) == 2

    first = transitions[0]

    assert first["from_order"] == 1
    assert first["to_order"] == 2

    assert first["history_length"] == 1

    assert first["delta_H"] == pytest.approx(
        -0.2
    )

    # Critical:
    # S_k must come from the higher-order n=2 model.
    assert first["S_k"] == pytest.approx(
        0.1
    )

    assert first["U_k"] == pytest.approx(
        0.15
    )

    second = transitions[1]

    assert second["from_order"] == 2
    assert second["to_order"] == 3

    assert second["history_length"] == 2

    assert second["delta_H"] == pytest.approx(
        0.3
    )

    assert second["S_k"] == pytest.approx(
        0.5
    )


def test_transition_builder_rejects_missing_order():
    """
    A transition table must not silently compare n=1 directly with n=3.

    Missing orders indicate an incomplete main experiment.
    """
    rows = [
        make_row(
            n=1,
            cross_entropy=2.0,
            s_k=0.0,
            d_k=0.0,
            u_k=0.0,
        ),
        make_row(
            n=3,
            cross_entropy=2.1,
            s_k=0.5,
            d_k=0.6,
            u_k=0.55,
        ),
    ]

    with pytest.raises(ValueError):
        build_transitions(
            rows
        )