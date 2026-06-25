"""Tests for the fair single-objective harness (ctrl-freak vs pymoo vs DEAP).

These are tiny smoke tests (small pop/gen) plus isolated survival-equivalence
unit tests. The full 12x3x30 sweep lives in s5's runner, never here. pymoo and
DEAP are guarded with ``importorskip`` so the numpy-floor CI job skips this
module instead of erroring.
"""

import numpy as np
import pytest

pytest.importorskip("pymoo")
pytest.importorskip("deap")

from benchmarks import config
from benchmarks.harness.single_objective import (
    SORunResult,
    _elitist_survivor_indices,
    _make_pymoo_elitist_survival,
    run_ctrl_freak,
    run_deap,
    run_pymoo,
    run_so,
)
from benchmarks.problems.single_objective import SO_PROBLEMS, SOProblem
from ctrl_freak.population import Population
from ctrl_freak.survival.elitist import elitist_survival

ADAPTERS = {"ctrl-freak": run_ctrl_freak, "pymoo": run_pymoo, "deap": run_deap}
SMALL_POP = 10  # even (ctrl-freak requires even pop_size)
SMALL_GEN = 3
SMALL_BUDGET = SMALL_POP + SMALL_GEN * SMALL_POP  # 40

# Synthetic survival cases; parents-first. Includes deliberate ties to lock the
# stable tie-break (first kept among tied parents; stable order among tied offspring).
SURVIVAL_CASES = [
    ([5.0, 2.0, 9.0, 7.0], [1.0, 8.0, 0.5, 6.0]),
    ([2.0, 1.0, 1.0, 4.0], [0.5, 3.0, 0.5, 1.5]),
    ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ([0.1, 5.0, 6.0, 7.0], [2.0, 3.0, 4.0, 8.0]),
]


def _reference_survivor_indices(parent_f, offspring_f):
    """ctrl-freak elitist_survival(1) survivor indices on a parents-first pop."""
    combined = np.array(parent_f + offspring_f, dtype=float)
    n = len(parent_f)
    pop = Population(x=np.zeros((len(combined), 1)), objectives=combined.reshape(-1, 1))
    indices, _ = elitist_survival(elite_count=1)(pop, n, parent_size=n)
    return indices


def test_eval_count_identical_across_libraries():
    """All three libraries call evaluate() exactly pop + n_gen*pop times (R2 / invariant)."""
    problem = SO_PROBLEMS["sphere"]
    reported = {}
    for name, adapter in ADAPTERS.items():
        calls = {"n": 0}

        def counted(x, _calls=calls, _func=problem.func):
            _calls["n"] += 1
            return _func(x)

        counting_problem = SOProblem(
            name="counting",
            func=counted,
            x_star=problem.x_star,
            f_star=problem.f_star,
            bounds=problem.bounds,
            dim=problem.dim,
            epsilon=problem.epsilon,
        )
        result = adapter(counting_problem, seed=0, pop_size=SMALL_POP, n_generations=SMALL_GEN)
        assert result.n_evaluations == SMALL_BUDGET, name
        # independent func-call counter proves the reported count is real
        assert calls["n"] == SMALL_BUDGET, name
        assert len(result.history) == SMALL_GEN + 1, name
        reported[name] = result.n_evaluations
    # equal eval count is necessary, not sufficient: assert identical across libs
    assert len(set(reported.values())) == 1


def test_run_so_uses_config_budget():
    """run_so binds the committed 100/200 budget => 20100 evaluations and converges."""
    result = run_so(SO_PROBLEMS["sphere"], "ctrl-freak", seed=0)
    expected = config.SO_POP_SIZE + config.SO_N_GENERATIONS * config.SO_POP_SIZE
    assert result.n_evaluations == expected == 20100
    assert len(result.history) == config.SO_N_GENERATIONS + 1
    assert result.best_f < 0.1  # sphere converges well below 0.1


def test_run_so_rejects_unknown_library():
    with pytest.raises(ValueError):
        run_so(SO_PROBLEMS["sphere"], "scipy", seed=0)


@pytest.mark.parametrize("parent_f, offspring_f", SURVIVAL_CASES)
def test_elitist_survivor_indices_matches_reference(parent_f, offspring_f):
    """Shared survival core (used by pymoo + DEAP) replicates elitist_survival(1)."""
    n = len(parent_f)
    combined = np.array(parent_f + offspring_f, dtype=float)
    got = _elitist_survivor_indices(combined, n_survive=n, n_elite=1)
    ref = _reference_survivor_indices(parent_f, offspring_f)
    np.testing.assert_array_equal(got, ref)


def test_pymoo_survival_replicates_elitist():
    """The pymoo custom Survival, in isolation, matches elitist_survival(1) (R3)."""
    from pymoo.core.population import Population as PymooPopulation

    parent_f = [2.0, 1.0, 1.0, 4.0]
    offspring_f = [0.5, 3.0, 0.5, 1.5]
    n = len(parent_f)
    combined = np.array(parent_f + offspring_f, dtype=float)
    # encode X = combined index so we can read back which originals survived
    pop = PymooPopulation.new(
        "X",
        np.arange(len(combined), dtype=float).reshape(-1, 1),
        "F",
        combined.reshape(-1, 1),
    )
    survivors = _make_pymoo_elitist_survival(n_elite=1).do(None, pop, n_survive=n)
    got = survivors.get("X")[:, 0].astype(int)
    ref = _reference_survivor_indices(parent_f, offspring_f)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("name", list(ADAPTERS))
def test_history_monotone_and_best_consistent(name):
    """Elitist survival => non-increasing history; best_f == history[-1]; minimization direction."""
    problem = SO_PROBLEMS["sphere"]
    result = ADAPTERS[name](problem, seed=1, pop_size=SMALL_POP, n_generations=SMALL_GEN)
    history = np.asarray(result.history)
    assert len(history) == SMALL_GEN + 1
    assert np.all(np.diff(history) <= 1e-9)
    assert result.best_f == pytest.approx(result.history[-1])
    assert result.best_f <= result.history[0]
    assert result.best_x.shape == (problem.dim,)


@pytest.mark.parametrize("name", list(ADAPTERS))
def test_reproducible(name):
    """Same seed reproduces best_x, best_f, and history exactly."""
    problem = SO_PROBLEMS["sphere"]
    a = ADAPTERS[name](problem, seed=3, pop_size=SMALL_POP, n_generations=SMALL_GEN)
    b = ADAPTERS[name](problem, seed=3, pop_size=SMALL_POP, n_generations=SMALL_GEN)
    assert a.best_f == b.best_f
    np.testing.assert_array_equal(a.best_x, b.best_x)
    assert a.history == b.history


@pytest.mark.parametrize("name", list(ADAPTERS))
@pytest.mark.parametrize("problem_name", ["sphere", "rastrigin"])
def test_smoke_returns_valid_result(name, problem_name):
    """Each adapter returns a finite, in-bounds SORunResult at the right eval count."""
    problem = SO_PROBLEMS[problem_name]
    result = ADAPTERS[name](problem, seed=0, pop_size=SMALL_POP, n_generations=SMALL_GEN)
    assert isinstance(result, SORunResult)
    assert np.isfinite(result.best_f)
    assert result.best_x.shape == (problem.dim,)
    assert result.n_evaluations == SMALL_BUDGET
    lo, hi = problem.bounds
    assert np.all(result.best_x >= lo) and np.all(result.best_x <= hi)
