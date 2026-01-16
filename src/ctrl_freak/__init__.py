"""ctrl-freak: NSGA-II Implementation.

A pure numpy implementation of the NSGA-II multi-objective genetic algorithm.

Example:
    >>> from ctrl_freak import nsga2, Population
    >>> import numpy as np
    >>> def init(rng): return rng.uniform(0, 1, size=3)
    >>> def evaluate(x): return np.array([x.sum(), (1 - x).sum()])
    >>> def crossover(p1, p2): return (p1 + p2) / 2
    >>> def mutate(x): return np.clip(x + 0.01, 0, 1)
    >>> result = nsga2(init, evaluate, crossover, mutate, pop_size=10, n_generations=5, seed=42)
    >>> len(result)
    10
"""

from ctrl_freak.algorithm import nsga2, survivor_selection
from ctrl_freak.operators import create_offspring, lift, polynomial_mutation, sbx_crossover, select_parents
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

__all__ = [
    # Main algorithm
    "nsga2",
    "survivor_selection",
    # Selection utilities
    "lift",
    "select_parents",
    "create_offspring",
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
    # Standard operators
    "sbx_crossover",
    "polynomial_mutation",
    # Data structures
    "Population",
    "IndividualView",
    # Result types
    "NSGA2Result",
    "GAResult",
]
