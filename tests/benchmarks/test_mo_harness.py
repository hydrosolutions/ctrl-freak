"""Smoke + invariant tests for the fair MO comparison harness.

CI-collected and intentionally tiny (small pop/gen). Asserts the parity
invariants -- equal eval counts, the eval-count formula, identical pop/gen, and
non-dominated-only front extraction (incl. the ZDT3 clean-front case) -- without
running the full 25 100-eval sweep (that is s5). Guarded with ``importorskip`` so
the numpy-floor CI job skips (not errors) when pymoo/DEAP do not resolve.
"""

import numpy as np
import pytest

pytest.importorskip("pymoo")
pytest.importorskip("deap")

from benchmarks.harness.multi_objective import LIBRARIES, _non_dominated_mask, run_mo  # noqa: E402
from benchmarks.problems.multi_objective import MO_PROBLEMS  # noqa: E402
from ctrl_freak.primitives.pareto import non_dominated_sort  # noqa: E402

# Tiny smoke budget: 20 * (1 + 5) = 120 evals. pop_size is even AND a multiple of
# 4 (DEAP selTournamentDCD requires k divisible by four).
SMOKE = {"pop_size": 20, "n_generations": 5}


def _is_non_dominated_only(objectives: np.ndarray) -> bool:
    return bool((non_dominated_sort(objectives) == 0).all())


def test_non_dominated_mask_drops_dominated():
    objs = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 2.0]])
    assert _non_dominated_mask(objs).tolist() == [True, True, True, False]


def test_eval_counts_equal_and_match_formula():
    results = {lib: run_mo(MO_PROBLEMS["zdt1"], lib, seed=0, **SMOKE) for lib in LIBRARIES}
    expected = SMOKE["pop_size"] * (1 + SMOKE["n_generations"])
    # equal eval count across all three libraries, equal to the budget formula
    assert {results[lib].n_evaluations for lib in LIBRARIES} == {expected}
    # identical pop_size / n_generations across all three libraries
    assert len({(results[lib].pop_size, results[lib].n_generations) for lib in LIBRARIES}) == 1
    # formula holds from each run's own recorded budget
    for lib in LIBRARIES:
        r = results[lib]
        assert r.n_evaluations == r.pop_size * (1 + r.n_generations)


def test_default_eval_counts_equal_25100():
    results = {lib: run_mo(MO_PROBLEMS["zdt1"], lib, seed=0) for lib in LIBRARIES}
    assert {results[lib].n_evaluations for lib in LIBRARIES} == {25_100}


@pytest.mark.parametrize("problem", ["zdt1", "zdt3", "zdt4", "dtlz2"])
@pytest.mark.parametrize("library", LIBRARIES)
def test_front_is_non_dominated_only(problem, library):
    r = run_mo(MO_PROBLEMS[problem], library, seed=0, **SMOKE)
    assert _is_non_dominated_only(r.front_objectives)
    assert r.front_x.shape[0] == r.front_objectives.shape[0]
    assert r.front_objectives.shape[1] == MO_PROBLEMS[problem].n_obj
    assert r.front_x.shape[1] == MO_PROBLEMS[problem].n_var


@pytest.mark.parametrize("library", LIBRARIES)
def test_zdt3_front_is_clean(library):
    # N6: the legacy ZDT3 "scatter" -- the reported front must be non-dominated-only.
    r = run_mo(MO_PROBLEMS["zdt3"], library, seed=0, **SMOKE)
    assert _is_non_dominated_only(r.front_objectives)
    assert r.front_objectives.shape[0] <= r.pop_size


@pytest.mark.parametrize("library", LIBRARIES)
def test_history_shapes(library):
    r = run_mo(MO_PROBLEMS["zdt1"], library, seed=0, **SMOKE)
    assert len(r.history) == r.n_generations + 1
    for snap in r.history:
        assert snap.ndim == 2 and snap.shape[1] == 2
        assert _is_non_dominated_only(snap)


def test_invalid_library_raises():
    with pytest.raises(ValueError, match="library"):
        run_mo(MO_PROBLEMS["zdt1"], "scipy", seed=0, **SMOKE)
