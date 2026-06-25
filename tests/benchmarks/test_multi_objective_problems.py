"""Validate multi-objective benchmark problems against pymoo references.

Guarded with ``pytest.importorskip("pymoo")`` so the numpy-floor CI job skips
(rather than errors) when pymoo does not resolve. Validation is done against pymoo
directly — never via ``benchmarks.metrics`` — to keep this module independent of
the s3 metric layer.
"""

import numpy as np
import pytest

pytest.importorskip("pymoo")

from pymoo.indicators.igd_plus import IGDPlus
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

from benchmarks.problems import multi_objective as mo

# Tolerance for IGD+ between two finite samplings of the SAME continuous ZDT
# curve (density-bounded, not machine-zero); observed max ~1.2e-3.
ZDT_IGDPLUS_ATOL = 5e-3

ZDT_NAMES = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")


def _pymoo_zdt(name: str):
    """Construct the pymoo ZDT problem at this suite's 30 decision variables."""
    return get_problem(name, n_var=mo.ZDT_N_VAR)


def _sample_within_bounds(problem: mo.MOProblem, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` decision vectors uniformly within ``problem.bounds``."""
    rng = np.random.default_rng(seed)
    xl, xu = problem.bounds
    return xl + rng.random((n, problem.n_var)) * (xu - xl)


# --- Registry / metadata shape ------------------------------------------------
def test_registry_keys():
    assert sorted(mo.MO_PROBLEMS) == ["dtlz2", "zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]


@pytest.mark.parametrize("name", ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz2"])
def test_metadata_dimensions(name):
    p = mo.MO_PROBLEMS[name]
    xl, xu = p.bounds
    assert p.name == name
    assert xl.shape == (p.n_var,)
    assert xu.shape == (p.n_var,)
    assert np.all(xu >= xl)
    assert p.ref_point.shape == (p.n_obj,)


def test_zdt_dimensions():
    for name in ZDT_NAMES:
        p = mo.MO_PROBLEMS[name]
        assert (p.n_var, p.n_obj) == (30, 2)
        np.testing.assert_array_equal(p.ref_point, [1.1, 1.1])


def test_dtlz2_dimensions():
    p = mo.MO_PROBLEMS["dtlz2"]
    assert (p.n_var, p.n_obj) == (12, 3)
    np.testing.assert_array_equal(p.ref_point, [1.1, 1.1, 1.1])


def test_zdt4_heterogeneous_bounds():
    xl, xu = mo.MO_PROBLEMS["zdt4"].bounds
    np.testing.assert_array_equal(xl, np.concatenate([[0.0], np.full(29, -5.0)]))
    np.testing.assert_array_equal(xu, np.concatenate([[1.0], np.full(29, 5.0)]))


# --- Objective functions match pymoo exactly ----------------------------------
@pytest.mark.parametrize("name", ZDT_NAMES)
def test_zdt_objective_matches_pymoo(name):
    problem = mo.MO_PROBLEMS[name]
    X = _sample_within_bounds(problem, n=25, seed=0)
    mine = np.array([problem.func(x) for x in X])
    ref = _pymoo_zdt(name).evaluate(X)
    np.testing.assert_allclose(mine, ref, atol=1e-10)


def test_dtlz2_objective_matches_pymoo():
    problem = mo.MO_PROBLEMS["dtlz2"]
    X = _sample_within_bounds(problem, n=25, seed=1)
    mine = np.array([problem.func(x) for x in X])
    ref = get_problem("dtlz2", n_var=mo.DTLZ2_N_VAR, n_obj=mo.DTLZ2_N_OBJ).evaluate(X)
    np.testing.assert_allclose(mine, ref, atol=1e-12)


# --- Front samplers: exact analytical relation (machine precision) ------------
def test_zdt1_front_relation():
    front = mo.zdt1_front()
    np.testing.assert_allclose(front[:, 1], 1.0 - np.sqrt(front[:, 0]))


def test_zdt2_front_relation():
    front = mo.zdt2_front()
    np.testing.assert_allclose(front[:, 1], 1.0 - front[:, 0] ** 2)


def test_zdt3_front_relation_and_regions():
    front = mo.zdt3_front()
    f1, f2 = front[:, 0], front[:, 1]
    np.testing.assert_allclose(f2, 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1))
    # every f1 lies inside one of the five canonical regions
    in_region = np.zeros(len(f1), dtype=bool)
    for lo, hi in mo.ZDT3_REGIONS:
        in_region |= (f1 >= lo - 1e-9) & (f1 <= hi + 1e-9)
    assert in_region.all()


def test_zdt4_front_relation():
    front = mo.zdt4_front()
    np.testing.assert_allclose(front[:, 1], 1.0 - np.sqrt(front[:, 0]))


def test_zdt6_front_relation_and_range():
    front = mo.zdt6_front()
    f1, f2 = front[:, 0], front[:, 1]
    np.testing.assert_allclose(f2, 1.0 - f1**2)
    assert f1.min() >= mo.ZDT6_F1_MIN - 1e-12
    assert f1.max() <= 1.0 + 1e-12
    np.testing.assert_allclose(f1.min(), mo.ZDT6_F1_MIN)


def test_dtlz2_front_on_unit_sphere():
    front = mo.dtlz2_front()
    assert front.shape == (1275, 3)
    np.testing.assert_allclose((front**2).sum(axis=1), 1.0)
    assert np.all(front >= 0.0)  # first octant


# --- Front samplers vs pymoo pareto_front (IGD+) ------------------------------
@pytest.mark.parametrize("name", ZDT_NAMES)
def test_zdt_front_igdplus_vs_pymoo(name):
    mine = mo.MO_PROBLEMS[name].true_front_sampler()
    pymoo_pf = _pymoo_zdt(name).pareto_front()
    # density-bounded (two samplings of the same curve), not machine-zero
    assert IGDPlus(pymoo_pf)(mine) < ZDT_IGDPLUS_ATOL


def test_dtlz2_front_igdplus_vs_pymoo():
    mine = mo.dtlz2_front()
    ref_dirs = get_reference_directions("das-dennis", mo.DTLZ2_N_OBJ, n_partitions=mo.DTLZ2_N_PARTITIONS)
    pymoo_pf = get_problem("dtlz2", n_var=mo.DTLZ2_N_VAR, n_obj=mo.DTLZ2_N_OBJ).pareto_front(ref_dirs)
    # identical generic-sphere construction => exactly zero
    assert IGDPlus(pymoo_pf)(mine) < 1e-9
