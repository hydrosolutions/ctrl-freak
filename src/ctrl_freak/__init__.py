"""ctrl-freak: Extensible Genetic Algorithm Framework.

A pure numpy implementation of genetic algorithms including NSGA-II for
multi-objective optimization and standard GA for single-objective optimization.

Example (multi-objective with NSGA-II):
    >>> from ctrl_freak import nsga2, Population
    >>> import numpy as np
    >>> def init(rng): return rng.uniform(0, 1, size=3)
    >>> def evaluate(x): return np.array([x.sum(), (1 - x).sum()])
    >>> def crossover(p1, p2): return (p1 + p2) / 2
    >>> def mutate(x): return np.clip(x + 0.01, 0, 1)
    >>> result = nsga2(init, evaluate, crossover, mutate, pop_size=10, n_generations=5, seed=42)
    >>> len(result)
    10

Example (single-objective with GA):
    >>> from ctrl_freak import ga
    >>> import numpy as np
    >>> def init(rng): return rng.uniform(0, 1, size=3)
    >>> def evaluate(x): return x.sum()  # Single objective
    >>> def crossover(p1, p2): return (p1 + p2) / 2
    >>> def mutate(x): return np.clip(x + 0.01, 0, 1)
    >>> result = ga(init, evaluate, crossover, mutate, pop_size=10, n_generations=5, seed=42)
    >>> len(result)
    10
"""

from ctrl_freak.algorithm import survivor_selection
from ctrl_freak.algorithms import ga, nsga2
from ctrl_freak.operators import (
    create_offspring,
    lift,
    lift_parallel,
    polynomial_mutation,
    sbx_crossover,
    select_parents,
)
from ctrl_freak.population import IndividualView, Population
from ctrl_freak.primitives import (
    crowding_distance,
    dominates,
    dominates_matrix,
    non_dominated_sort,
)
from ctrl_freak.registry import (
    SelectionRegistry,
    SurvivalRegistry,
    list_selections,
    list_survivals,
)
from ctrl_freak.results import GAResult, NSGA2Result
from ctrl_freak.selection import crowded_tournament, fitness_tournament, roulette_wheel
from ctrl_freak.survival import elitist_survival, nsga2_survival, truncation_survival

__all__ = [
    # Algorithms
    "nsga2",
    "ga",
    # Selection strategies
    "crowded_tournament",
    "fitness_tournament",
    "roulette_wheel",
    # Survival strategies
    "nsga2_survival",
    "truncation_survival",
    "elitist_survival",
    # Genetic operators
    "lift",
    "lift_parallel",
    "select_parents",
    "create_offspring",
    "sbx_crossover",
    "polynomial_mutation",
    # Primitives
    "dominates",
    "dominates_matrix",
    "non_dominated_sort",
    "crowding_distance",
    # Registry system
    "SelectionRegistry",
    "SurvivalRegistry",
    "list_selections",
    "list_survivals",
    # Data structures
    "Population",
    "IndividualView",
    # Result types
    "NSGA2Result",
    "GAResult",
    # Legacy
    "survivor_selection",
]
