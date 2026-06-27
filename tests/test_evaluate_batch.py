"""Tests for the optional evaluate_batch hook on ga() and nsga2().

For BOTH algorithms these tests prove:

1. Back-compat: evaluate_batch=None reproduces the default per-individual path.
2. Equivalence: supplying evaluate_batch (a vectorized form of the same
   per-individual evaluate) yields results IDENTICAL to the per-individual path
   on the same seed / pop_size.
3. Wiring: the batch callable provably receives the full (n, n_params) matrix
   and the per-individual lift loop is NOT entered (the per-individual evaluate
   is never called).
"""

from typing import Any

import numpy as np

from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2


def _init2(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=2)


def _evaluate_ga(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _evaluate_nsga2(x: np.ndarray) -> np.ndarray:
    return np.array([np.sum(x), np.sum((1.0 - x) ** 2)])


def _crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0


def _mutate(x: np.ndarray) -> np.ndarray:
    return x.copy()


# ---------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------
def test_ga_evaluate_batch_matches_per_individual():
    """evaluate_batch yields ga() results IDENTICAL to the per-individual path."""

    def evaluate_batch(pop: np.ndarray) -> np.ndarray:
        # Row-wise application of the SAME per-individual evaluate.
        return np.array([_evaluate_ga(row) for row in pop])

    common: dict[str, Any] = {
        "init": _init2,
        "crossover": _crossover,
        "mutate": _mutate,
        "pop_size": 10,
        "n_generations": 5,
        "seed": 123,
    }
    reference = ga(evaluate=_evaluate_ga, **common)
    batched = ga(evaluate=_evaluate_ga, evaluate_batch=evaluate_batch, **common)

    np.testing.assert_array_equal(batched.population.x, reference.population.x)
    np.testing.assert_array_equal(batched.population.objectives, reference.population.objectives)
    np.testing.assert_array_equal(batched.fitness, reference.fitness)
    assert batched.best_idx == reference.best_idx
    assert batched.generations == reference.generations
    assert batched.evaluations == reference.evaluations


def test_ga_evaluate_batch_receives_full_matrix_and_bypasses_loop():
    """evaluate_batch sees the (pop_size, n_params) matrix; per-individual evaluate is never called."""
    pop_size = 8
    n_params = 2
    seen_shapes: list[tuple[int, ...]] = []

    def evaluate_batch(pop: np.ndarray) -> np.ndarray:
        seen_shapes.append(pop.shape)
        return np.sum(pop**2, axis=1)

    def forbidden_evaluate(x: np.ndarray) -> float:
        raise AssertionError("per-individual evaluate must not be called when evaluate_batch is supplied")

    result = ga(
        init=_init2,
        evaluate=forbidden_evaluate,
        evaluate_batch=evaluate_batch,
        crossover=_crossover,
        mutate=_mutate,
        pop_size=pop_size,
        n_generations=3,
        seed=7,
    )

    assert result.population.x.shape == (pop_size, n_params)
    assert seen_shapes  # at least the initial-population evaluation happened
    for shape in seen_shapes:
        assert shape == (pop_size, n_params)


def test_ga_evaluate_batch_none_is_unchanged():
    """evaluate_batch=None reproduces the default per-individual ga() exactly (back-compat)."""
    common: dict[str, Any] = {
        "init": _init2,
        "evaluate": _evaluate_ga,
        "crossover": _crossover,
        "mutate": _mutate,
        "pop_size": 10,
        "n_generations": 4,
        "seed": 99,
    }
    default = ga(**common)
    explicit_none = ga(evaluate_batch=None, **common)

    np.testing.assert_array_equal(default.population.x, explicit_none.population.x)
    np.testing.assert_array_equal(default.population.objectives, explicit_none.population.objectives)
    np.testing.assert_array_equal(default.fitness, explicit_none.fitness)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
def test_nsga2_evaluate_batch_matches_per_individual():
    """evaluate_batch yields nsga2() results IDENTICAL to the per-individual path."""

    def evaluate_batch(pop: np.ndarray) -> np.ndarray:
        return np.stack([_evaluate_nsga2(row) for row in pop])

    common: dict[str, Any] = {
        "init": _init2,
        "crossover": _crossover,
        "mutate": _mutate,
        "pop_size": 10,
        "n_generations": 5,
        "seed": 321,
    }
    reference = nsga2(evaluate=_evaluate_nsga2, **common)
    batched = nsga2(evaluate=_evaluate_nsga2, evaluate_batch=evaluate_batch, **common)

    np.testing.assert_array_equal(batched.population.x, reference.population.x)
    np.testing.assert_array_equal(batched.population.objectives, reference.population.objectives)
    np.testing.assert_array_equal(batched.rank, reference.rank)
    np.testing.assert_array_equal(batched.crowding_distance, reference.crowding_distance)
    assert batched.generations == reference.generations
    assert batched.evaluations == reference.evaluations


def test_nsga2_evaluate_batch_receives_full_matrix_and_bypasses_loop():
    """evaluate_batch sees the (pop_size, n_params) matrix; per-individual evaluate is never called."""
    pop_size = 8
    n_params = 2
    seen_shapes: list[tuple[int, ...]] = []

    def evaluate_batch(pop: np.ndarray) -> np.ndarray:
        seen_shapes.append(pop.shape)
        return np.stack([pop.sum(axis=1), (1.0 - pop).sum(axis=1)], axis=1)

    def forbidden_evaluate(x: np.ndarray) -> np.ndarray:
        raise AssertionError("per-individual evaluate must not be called when evaluate_batch is supplied")

    result = nsga2(
        init=_init2,
        evaluate=forbidden_evaluate,
        evaluate_batch=evaluate_batch,
        crossover=_crossover,
        mutate=_mutate,
        pop_size=pop_size,
        n_generations=3,
        seed=7,
    )

    assert result.population.x.shape == (pop_size, n_params)
    assert result.population.objectives is not None
    assert result.population.objectives.shape == (pop_size, 2)
    assert seen_shapes
    for shape in seen_shapes:
        assert shape == (pop_size, n_params)


def test_nsga2_evaluate_batch_none_is_unchanged():
    """evaluate_batch=None reproduces the default per-individual nsga2() exactly (back-compat)."""
    common: dict[str, Any] = {
        "init": _init2,
        "evaluate": _evaluate_nsga2,
        "crossover": _crossover,
        "mutate": _mutate,
        "pop_size": 10,
        "n_generations": 4,
        "seed": 99,
    }
    default = nsga2(**common)
    explicit_none = nsga2(evaluate_batch=None, **common)

    np.testing.assert_array_equal(default.population.x, explicit_none.population.x)
    np.testing.assert_array_equal(default.population.objectives, explicit_none.population.objectives)
    np.testing.assert_array_equal(default.rank, explicit_none.rank)
    np.testing.assert_array_equal(default.crowding_distance, explicit_none.crowding_distance)
