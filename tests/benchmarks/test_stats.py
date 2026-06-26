"""Tests for the overlapping-variance equivalence statistic (pure numpy)."""

import numpy as np
import pytest

from benchmarks.stats import equivalence


def test_known_overlap_is_equivalent_with_positive_margin():
    # Same centre, generous spread -> spreads overlap -> equivalent, margin > 0.
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    candidate = np.array([1.2, 2.1, 3.3, 3.9, 5.1])
    verdict = equivalence(reference, candidate)
    assert verdict["equivalent"] is True
    assert verdict["margin"] > 0.0
    # margin == threshold - delta_mean, exactly.
    np.testing.assert_allclose(verdict["margin"], verdict["threshold"] - verdict["delta_mean"])


def test_known_non_overlap_is_not_equivalent_with_negative_margin():
    # Far-apart means, tiny spreads -> disjoint -> not equivalent, margin < 0.
    reference = np.array([0.0010, 0.0020, 0.0015, 0.0012, 0.0018])
    candidate = np.array([10.0, 10.1, 9.9, 10.05, 9.95])
    verdict = equivalence(reference, candidate)
    assert verdict["equivalent"] is False
    assert verdict["margin"] < 0.0


def test_boundary_delta_equals_threshold_is_strictly_not_equivalent():
    # Exactly delta_mean == threshold locks the strict-< rule: reference mean 0
    # std 1, candidate mean 1 std 1 -> delta 1 == threshold 1 -> margin 0.
    reference = np.array([-1.0, 1.0])
    candidate = np.array([0.0, 2.0])
    verdict = equivalence(reference, candidate)
    np.testing.assert_allclose(verdict["delta_mean"], verdict["threshold"])
    np.testing.assert_allclose(verdict["margin"], 0.0)
    assert verdict["equivalent"] is False  # strict < => boundary is NOT equivalent


def test_identical_zero_variance_constants_are_not_equivalent():
    # Degenerate edge: both libraries produce the same constant for every seed
    # (e.g. ZDT4 hypervolume == 0 for all three). delta == threshold == 0; the
    # strict-< rule reports NOT equivalent (documented degenerate case).
    verdict = equivalence(np.full(5, 0.0), np.full(5, 0.0))
    assert verdict["equivalent"] is False
    np.testing.assert_allclose(verdict["margin"], 0.0)


def test_threshold_uses_larger_std_and_verdict_is_order_symmetric():
    reference = np.array([0.0, 0.0, 0.0, 0.0])  # std 0
    candidate = np.array([-2.0, -1.0, 1.0, 2.0])  # larger std
    forward = equivalence(reference, candidate)
    backward = equivalence(candidate, reference)
    np.testing.assert_allclose(forward["threshold"], np.std(candidate))
    # Swapping reference/candidate cannot change the verdict (threshold + |delta|
    # are both order-independent).
    assert forward["equivalent"] == backward["equivalent"]
    np.testing.assert_allclose(forward["margin"], backward["margin"])


def test_serialisable_python_scalars():
    verdict = equivalence(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert isinstance(verdict["equivalent"], bool)
    for key in ("mean_reference", "std_reference", "delta_mean", "threshold", "margin"):
        assert isinstance(verdict[key], float)


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (np.array([]), np.array([1.0])),
        (np.array([1.0]), np.array([])),
        (np.array([[1.0, 2.0]]), np.array([1.0, 2.0])),
    ],
)
def test_invalid_shapes_raise(reference, candidate):
    with pytest.raises(ValueError):
        equivalence(reference, candidate)
