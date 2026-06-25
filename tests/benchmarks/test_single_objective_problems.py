"""Tests for the six single-objective benchmark problems and their registry."""

import numpy as np
import pytest

from benchmarks.problems.single_objective import (
    DIM,
    SO_PROBLEMS,
    SOProblem,
    ackley,
    griewank,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
)

EXPECTED_NAMES = {"sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel"}

# Per-problem absolute tolerance for f(x*) == f*. Schwefel's offset constant and
# optimum are both rounded to 4 d.p., leaving an O(1e-4) residual at 10D; the
# rest are exact-to-machine (Ackley carries only a ~1e-16 exp(1)-vs-e residue).
OPTIMUM_ATOL = {
    "sphere": 1e-10,
    "rosenbrock": 1e-10,
    "rastrigin": 1e-10,
    "ackley": 1e-10,
    "griewank": 1e-10,
    "schwefel": 1e-3,
}


def test_registry_has_exactly_the_six_problems():
    assert set(SO_PROBLEMS) == EXPECTED_NAMES
    assert len(SO_PROBLEMS) == 6
    for name, problem in SO_PROBLEMS.items():
        assert problem.name == name


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_value_at_known_optimum(name):
    p = SO_PROBLEMS[name]
    np.testing.assert_allclose(p.func(p.x_star), p.f_star, atol=OPTIMUM_ATOL[name])


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_dim_is_ten_and_x_star_shape(name):
    p = SO_PROBLEMS[name]
    assert p.dim == DIM == 10
    assert p.x_star.shape == (10,)


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_bounds_shape_order_and_contain_optimum(name):
    p = SO_PROBLEMS[name]
    assert len(p.bounds) == 2
    low, high = p.bounds
    assert low < high
    assert np.all(p.x_star >= low)
    assert np.all(p.x_star <= high)


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_x_star_is_read_only(name):
    # Shared metadata: s5 reads x_star to compute ||x - x*||; lock guards it.
    assert not SO_PROBLEMS[name].x_star.flags.writeable


def test_reference_values():
    # A second known point per function, independent of the optimum.
    np.testing.assert_allclose(sphere(np.ones(DIM)), 10.0, atol=1e-12)
    np.testing.assert_allclose(rosenbrock(np.zeros(DIM)), 9.0, atol=1e-12)
    np.testing.assert_allclose(rastrigin(np.ones(DIM)), 10.0, atol=1e-9)
    # ackley(1*1) = 20*(1 - e^-0.2): exp(cos(2*pi*1)) = exp(1) = e cancels +e.
    np.testing.assert_allclose(ackley(np.ones(DIM)), 3.625384938440364, rtol=1e-9)
    # griewank(pi*sqrt(i)) = 55*pi^2/4000: each cos(x_i/sqrt(i)) = cos(pi) = -1,
    # product over 10 dims = 1.
    x_collapse = np.pi * np.sqrt(np.arange(1, DIM + 1))
    np.testing.assert_allclose(griewank(x_collapse), 55.0 * np.pi**2 / 4000.0, rtol=1e-9)
    # schwefel(0) = 418.9829*d - 0.
    np.testing.assert_allclose(schwefel(np.zeros(DIM)), 4189.829, rtol=1e-9)


def test_so_problem_rejects_invalid_fields():
    with pytest.raises(ValueError):
        SOProblem("bad_shape", sphere, np.zeros(3), 0.0, (-1.0, 1.0), 10, 1e-2)
    with pytest.raises(ValueError):
        SOProblem("bad_bounds", sphere, np.zeros(10), 0.0, (1.0, -1.0), 10, 1e-2)
    with pytest.raises(ValueError):
        SOProblem("bad_epsilon", sphere, np.zeros(10), 0.0, (-1.0, 1.0), 10, 0.0)
    with pytest.raises(ValueError):
        SOProblem("bad_dim", sphere, np.zeros(10), 0.0, (-1.0, 1.0), 0, 1e-2)
