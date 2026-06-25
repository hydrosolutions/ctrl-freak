"""Tests for benchmark metrics (single- and multi-objective)."""

import numpy as np
import pytest

pytest.importorskip("pymoo")

from benchmarks.metrics import (  # noqa: E402  (after importorskip guard)
    gd,
    hypervolume,
    igd_plus,
    is_success,
    objective_error,
    solution_distance,
    success_rate,
)

# --- Single-objective metrics -------------------------------------------------


def test_objective_error_basic():
    np.testing.assert_allclose(objective_error(2.5, 2.0), 0.5)
    np.testing.assert_allclose(objective_error(2.0, 2.0), 0.0)
    np.testing.assert_allclose(objective_error(-1.0, 1.0), 2.0)


def test_solution_distance_euclidean():
    d = solution_distance(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(d, 5.0)
    d0 = solution_distance(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(d0, 0.0)


def test_solution_distance_shape_mismatch_raises():
    with pytest.raises(ValueError):
        solution_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_is_success_strict_boundary():
    # exactly-representable values: abs/subtraction are exact, so these are
    # deterministic cross-platform (no float +tiny-tiny rounding traps).
    # strictly below epsilon -> success
    assert is_success(1.25, 1.0, 0.5) is True
    # error == epsilon exactly -> NOT a success (strict <)
    assert is_success(1.5, 1.0, 0.5) is False
    # above epsilon -> not a success
    assert is_success(2.0, 1.0, 0.5) is False


def test_success_rate_fraction():
    values = np.array([1.0, 1.5, 1.0, 1.0])
    np.testing.assert_allclose(success_rate(values, 1.0, 1e-6), 0.75)


def test_success_rate_strict_boundary():
    # errors [0.0, 0.5] vs epsilon 0.5: exactly one is at epsilon and must NOT
    # count (strict <). Exactly-representable values keep this deterministic.
    values = np.array([1.0, 1.5])
    np.testing.assert_allclose(success_rate(values, 1.0, 0.5), 0.5)


def test_success_rate_empty_raises():
    with pytest.raises(ValueError):
        success_rate(np.array([]), 1.0, 1e-6)


# --- Multi-objective metrics: self-consistency --------------------------------


def test_igd_plus_self_is_zero():
    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(igd_plus(front, front), 0.0, atol=1e-12)


def test_gd_self_is_zero():
    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(gd(front, front), 0.0, atol=1e-12)


# --- Multi-objective metrics: agreement with pymoo directly -------------------


def test_igd_plus_matches_pymoo():
    from pymoo.indicators.igd_plus import IGDPlus

    true = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    approx = np.array([[0.1, 1.0], [1.0, 0.1]])
    expected = float(IGDPlus(true)(approx))
    np.testing.assert_allclose(igd_plus(approx, true), expected)


def test_gd_matches_pymoo():
    from pymoo.indicators.gd import GD

    true = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    approx = np.array([[0.1, 1.0], [1.0, 0.1]])
    expected = float(GD(true)(approx))
    np.testing.assert_allclose(gd(approx, true), expected)


def test_hypervolume_matches_pymoo():
    from pymoo.indicators.hv import HV

    objs = np.array([[0.2, 0.8], [0.8, 0.2]])
    ref = np.array([1.1, 1.1])
    expected = float(HV(ref_point=ref)(objs))
    np.testing.assert_allclose(hypervolume(objs, ref_point=ref), expected)


def test_hypervolume_default_ref_point():
    objs = np.array([[0.2, 0.8], [0.8, 0.2]])
    np.testing.assert_allclose(hypervolume(objs), 0.45, atol=1e-2)


def test_mo_metrics_known_offset_value():
    # true = unit-axis points, approx = scaled out by 1.0 along each axis
    true = np.array([[0.0, 1.0], [1.0, 0.0]])
    approx = np.array([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_allclose(igd_plus(approx, true), 1.0)
    np.testing.assert_allclose(gd(approx, true), 1.0)


# --- Multi-objective metrics: validation guards -------------------------------


def test_igd_plus_empty_raises():
    with pytest.raises(ValueError):
        igd_plus(np.empty((0, 2)), np.array([[0.0, 1.0]]))


def test_gd_empty_raises():
    with pytest.raises(ValueError):
        gd(np.array([[0.0, 1.0]]), np.empty((0, 2)))


def test_igd_plus_non_2d_raises():
    with pytest.raises(ValueError):
        igd_plus(np.array([0.0, 1.0]), np.array([[0.0, 1.0]]))


def test_igd_plus_objective_mismatch_raises():
    with pytest.raises(ValueError):
        igd_plus(np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 1.0]]))


def test_hypervolume_empty_raises():
    with pytest.raises(ValueError):
        hypervolume(np.array([]))


def test_hypervolume_non_2d_raises():
    with pytest.raises(ValueError):
        hypervolume(np.array([0.2, 0.8]))
