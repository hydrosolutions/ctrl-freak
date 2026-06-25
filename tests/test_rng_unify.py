"""Regression tests for master-seed unification."""

import numpy as np

from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2
from ctrl_freak.operators import polynomial_mutation, sbx_crossover
from ctrl_freak.population import Population
from ctrl_freak.survival.nsga2 import nsga2_survival


def test_single_seed_controls_operators_in_ga():
    """A single seed controls entropy-seeded GA operators."""

    def make_kwargs():
        crossover = sbx_crossover(eta=15.0, bounds=(-5.12, 5.12))
        mutate = polynomial_mutation(eta=20.0, bounds=(-5.12, 5.12))
        return {
            "init": lambda rng: rng.uniform(-5.12, 5.12, size=5),
            "evaluate": lambda x: float(np.sum(x**2)),
            "crossover": crossover,
            "mutate": mutate,
            "pop_size": 20,
            "n_generations": 10,
            "seed": 12345,
        }

    r1 = ga(**make_kwargs())
    r2 = ga(**make_kwargs())
    np.testing.assert_array_equal(r1.population.x, r2.population.x)
    np.testing.assert_array_equal(r1.fitness, r2.fitness)


def test_single_seed_controls_full_default_stack_nsga2():
    """A single seed controls NSGA-II init, selection, crossover, and mutation."""

    def make_kwargs():
        crossover = sbx_crossover(eta=15.0, bounds=(0.0, 1.0))
        mutate = polynomial_mutation(eta=20.0, bounds=(0.0, 1.0))
        return {
            "init": lambda rng: rng.uniform(0, 1, size=5),
            "evaluate": lambda x: np.array([x.sum(), (1 - x).sum()]),
            "crossover": crossover,
            "mutate": mutate,
            "pop_size": 20,
            "n_generations": 10,
            "seed": 7,
            "select": "crowded",
            "survive": "nsga2",
        }

    a = nsga2(**make_kwargs())
    b = nsga2(**make_kwargs())
    np.testing.assert_array_equal(a.population.x, b.population.x)
    np.testing.assert_array_equal(a.rank, b.rank)
    np.testing.assert_array_equal(a.crowding_distance, b.crowding_distance)


def test_ga_n_workers_1_vs_2_identical_pure_evaluate():
    """Sequential and parallel GA evaluation match for a pure objective."""

    def kw():
        return {
            "init": lambda rng: rng.uniform(0, 1, size=4),
            "evaluate": lambda x: float(np.sum(x**2)),
            "crossover": lambda p1, p2: (p1 + p2) / 2,
            "mutate": lambda x: x.copy(),
            "pop_size": 10,
            "n_generations": 5,
            "seed": 42,
        }

    a = ga(**kw(), n_workers=1)
    b = ga(**kw(), n_workers=2)
    np.testing.assert_array_equal(a.population.x, b.population.x)
    np.testing.assert_array_equal(a.fitness, b.fitness)


def test_nsga2_survival_critical_front_crowding_is_subset_recompute():
    """Pin critical-front crowding recomputed on the selected subset."""
    obj = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    pop = Population(x=np.zeros((5, 1)), objectives=obj)
    selector = nsga2_survival()
    indices, state = selector(pop, n_survivors=4)

    np.testing.assert_array_equal(indices, np.array([4, 0, 3, 2], dtype=np.intp))
    np.testing.assert_array_equal(state["rank"], np.array([0, 0, 0, 0]))
    np.testing.assert_array_equal(state["crowding_distance"], np.array([np.inf, np.inf, 1.0, 1.5]))
