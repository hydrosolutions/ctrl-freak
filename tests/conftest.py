"""Shared test fixtures for ctrl-freak tests.

This module provides common fixtures used across test modules:
- rng: Seeded random number generator
- simple_population: Basic population with rank/crowding computed
- Crossover and mutation operator fixtures
"""

import numpy as np
import pytest

from ctrl_freak import Population, crowding_distance, non_dominated_sort


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded random number generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_population() -> Population:
    """Create a simple population with objectives, rank, and crowding distance.

    This population has 4 individuals forming a simple Pareto front structure.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])

    ranks = non_dominated_sort(objectives)
    cd = np.zeros(len(objectives), dtype=np.float64)
    for r in range(int(ranks.max()) + 1):
        mask = ranks == r
        cd[mask] = crowding_distance(objectives[mask])

    return Population(x=x, objectives=objectives, rank=ranks, crowding_distance=cd)


@pytest.fixture
def identity_crossover():
    """Crossover that returns the first parent unchanged."""

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return p1.copy()

    return crossover


@pytest.fixture
def averaging_crossover():
    """Crossover that returns the average of both parents."""

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return (p1 + p2) / 2

    return crossover


@pytest.fixture
def identity_mutate():
    """Mutation that returns the individual unchanged."""

    def mutate(x: np.ndarray) -> np.ndarray:
        return x.copy()

    return mutate


@pytest.fixture
def small_perturbation_mutate():
    """Mutation that adds small random perturbation."""

    def mutate(x: np.ndarray) -> np.ndarray:
        return x + 0.01 * np.random.randn(len(x))

    return mutate


@pytest.fixture
def tracking_crossover():
    """Crossover that tracks all calls for verification.

    Returns a tuple of (crossover_fn, call_log).
    """
    call_log: list[tuple[np.ndarray, np.ndarray]] = []

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        call_log.append((p1.copy(), p2.copy()))
        return (p1 + p2) / 2

    return crossover, call_log


@pytest.fixture
def tracking_mutate():
    """Mutation that tracks all calls for verification.

    Returns a tuple of (mutate_fn, call_log).
    """
    call_log: list[np.ndarray] = []

    def mutate(x: np.ndarray) -> np.ndarray:
        call_log.append(x.copy())
        return x.copy()

    return mutate, call_log


@pytest.fixture
def zdt1_problem():
    """ZDT1 multi-objective test problem.

    ZDT1 is a standard benchmark for multi-objective optimization with:
    - n_vars decision variables in [0, 1]
    - 2 objectives (minimize both)
    - Convex Pareto front

    Returns:
        Dict with init, evaluate, crossover, and mutate functions.
    """
    n_vars = 5

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0, 1, size=n_vars)

    def evaluate(x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1 + 9 * np.mean(x[1:])
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return (p1 + p2) / 2

    def mutate(x: np.ndarray) -> np.ndarray:
        return np.clip(x + 0.01 * np.random.randn(len(x)), 0, 1)

    return {"init": init, "evaluate": evaluate, "crossover": crossover, "mutate": mutate}


@pytest.fixture
def simple_biobj_problem():
    """Simple bi-objective problem for unit tests.

    A simple problem where objectives are just sums:
    - f1 = sum(x)
    - f2 = sum(1 - x)

    These objectives trade off naturally (minimizing one increases the other).

    Returns:
        Dict with init, evaluate, crossover, and mutate functions.
    """
    n_vars = 3

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0, 1, size=n_vars)

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.array([x.sum(), (1 - x).sum()])

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return (p1 + p2) / 2

    def mutate(x: np.ndarray) -> np.ndarray:
        return np.clip(x + 0.01 * np.random.randn(len(x)), 0, 1)

    return {"init": init, "evaluate": evaluate, "crossover": crossover, "mutate": mutate}
