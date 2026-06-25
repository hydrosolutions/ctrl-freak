"""Tests for the ported ctrl-freak SBX shared operators (``benchmarks/harness/operators.py``).

The crux of the Option-C fairness design is that the port reproduces ctrl-freak's
SBX bit-for-bit, so every harness runs the identical crossover. The core/port and
DEAP-mate tests are pure (ctrl-freak + numpy) and run on the numpy floor; the pymoo
``Crossover`` test guards pymoo locally.
"""

import numpy as np
import pytest

from benchmarks.harness.operators import (
    ctrl_freak_sbx_child,
    deap_ctrl_freak_sbx,
    make_pymoo_ctrl_freak_sbx,
)
from ctrl_freak import sbx_crossover

ETA = 15.0
BOUNDS = (-5.12, 5.12)


@pytest.mark.parametrize("seed", [0, 1, 7, 1234])
def test_sbx_port_is_bit_identical_to_ctrl_freak(seed):
    """The ported core reproduces ctrl-freak's sbx_crossover exactly (same RNG seed)."""
    gen = np.random.default_rng(seed)
    p1 = gen.uniform(*BOUNDS, size=10)
    p2 = gen.uniform(*BOUNDS, size=10)
    native = sbx_crossover(eta=ETA, bounds=BOUNDS, seed=seed)(p1, p2)
    port = ctrl_freak_sbx_child(p1, p2, ETA, BOUNDS[0], BOUNDS[1], np.random.default_rng(seed))
    np.testing.assert_array_equal(native, port)


def test_sbx_port_bit_identical_across_sequential_calls():
    """Sequential calls share the generator's advancement, matching ctrl-freak."""
    native_op = sbx_crossover(eta=ETA, bounds=BOUNDS, seed=99)
    port_rng = np.random.default_rng(99)
    parents = np.random.default_rng(3)
    for _ in range(5):
        p1 = parents.uniform(*BOUNDS, size=8)
        p2 = parents.uniform(*BOUNDS, size=8)
        np.testing.assert_array_equal(
            native_op(p1, p2),
            ctrl_freak_sbx_child(p1, p2, ETA, BOUNDS[0], BOUNDS[1], port_rng),
        )


def test_single_child_structure():
    """ctrl_freak_sbx_child returns ONE child of the parent shape, within bounds."""
    child = ctrl_freak_sbx_child(np.zeros(6), np.ones(6), ETA, 0.0, 1.0, np.random.default_rng(0))
    assert child.shape == (6,)
    assert np.all((child >= 0.0) & (child <= 1.0))


def test_recombination_fraction_is_one():
    """Every eligible variable is recombined (fraction 1.0); non-eligible keep parent 1."""
    p1 = np.zeros(10)
    p2 = p1.copy()
    p2[[0, 2, 4, 6, 8]] = 3.0  # only these are eligible
    child = ctrl_freak_sbx_child(p1, p2, ETA, -5.12, 5.12, np.random.default_rng(0))
    eligible = np.abs(p1 - p2) > 1e-14
    np.testing.assert_array_equal(child[~eligible], p1[~eligible])
    assert np.count_nonzero(child[eligible] != p1[eligible]) == int(eligible.sum())


def test_deap_mate_two_independent_calls():
    """The DEAP mate edits both individuals in place via two independent core calls."""
    a, b = [0.2, 0.4, 0.6], [0.8, 0.1, 0.9]
    c1, c2 = deap_ctrl_freak_sbx(a, b, ETA, 0.0, 1.0, np.random.default_rng(0))
    assert c1 is a and c2 is b
    assert len(c1) == 3 and len(c2) == 3
    rng2 = np.random.default_rng(0)
    exp1 = ctrl_freak_sbx_child(np.array([0.2, 0.4, 0.6]), np.array([0.8, 0.1, 0.9]), ETA, 0.0, 1.0, rng2)
    exp2 = ctrl_freak_sbx_child(np.array([0.8, 0.1, 0.9]), np.array([0.2, 0.4, 0.6]), ETA, 0.0, 1.0, rng2)
    np.testing.assert_array_equal(np.asarray(c1), exp1)
    np.testing.assert_array_equal(np.asarray(c2), exp2)


def test_pymoo_crossover_produces_two_children_via_core():
    """The pymoo Crossover yields 2 offspring/mating matching two independent core calls."""
    pytest.importorskip("pymoo")
    from pymoo.core.problem import ElementwiseProblem

    crossover = make_pymoo_ctrl_freak_sbx(eta=ETA, prob=1.0)
    assert (crossover.n_parents, crossover.n_offsprings) == (2, 2)

    class _P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=4, n_obj=1, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = float(np.sum(x))

    gen = np.random.default_rng(0)
    p1 = gen.uniform(0, 1, 4)
    p2 = gen.uniform(0, 1, 4)
    X = np.stack([p1, p2])[:, None, :]  # (2 parents, 1 mating, 4 vars)
    Q = crossover._do(_P(), X, random_state=np.random.default_rng(0))
    assert Q.shape == (2, 1, 4)
    rng2 = np.random.default_rng(0)
    exp0 = ctrl_freak_sbx_child(p1, p2, ETA, 0.0, 1.0, rng2)
    exp1 = ctrl_freak_sbx_child(p2, p1, ETA, 0.0, 1.0, rng2)
    np.testing.assert_array_equal(Q[0, 0], exp0)
    np.testing.assert_array_equal(Q[1, 0], exp1)
