"""Genetic operators for NSGA-II.

This module provides the core genetic operators for NSGA-II:

- lift: Decorator to apply per-individual functions to entire populations
- select_parents: Binary tournament selection using crowded comparison
- create_offspring: Create offspring via selection, crossover, and mutation
"""

from collections.abc import Callable

import numpy as np

from ctrl_freak.population import Population


def lift(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Lift a per-individual function to work on a population.

    This utility allows users to write simple per-individual functions
    while the framework handles batching/vectorization.

    Args:
        fn: Function that operates on a single individual.
            Signature: (n_vars,) -> (n_out,)

    Returns:
        A function that operates on a population.
        Signature: (n, n_vars) -> (n, n_out)

    Example:
        >>> def evaluate_one(x: np.ndarray) -> np.ndarray:
        ...     return np.array([x.sum(), x.prod()])
        >>> evaluate = lift(evaluate_one)
        >>> pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> evaluate(pop_x)
        array([[ 3.,  2.],
               [ 7., 12.]])
    """

    def lifted(x: np.ndarray) -> np.ndarray:
        return np.stack([fn(x[i]) for i in range(x.shape[0])])

    return lifted


def select_parents(pop: Population, n_parents: int, rng: np.random.Generator) -> np.ndarray:
    """Select parents using binary tournament selection (vectorized).

    Uses the crowded comparison operator: lower rank wins, ties broken by
    higher crowding distance.

    Args:
        pop: Population with rank and crowding_distance computed.
        n_parents: Number of parents to select.
        rng: Random number generator for reproducibility.

    Returns:
        Array of shape (n_parents,) containing indices into population.

    Raises:
        ValueError: If pop.rank or pop.crowding_distance is None.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> parents = select_parents(pop, n_parents=10, rng=rng)
        >>> parents.shape
        (10,)
    """
    if pop.rank is None:
        raise ValueError("Population must have rank computed for parent selection")
    if pop.crowding_distance is None:
        raise ValueError("Population must have crowding_distance computed for parent selection")

    n = len(pop.x)
    candidates = rng.integers(0, n, size=(n_parents, 2))

    rank_a = pop.rank[candidates[:, 0]]
    rank_b = pop.rank[candidates[:, 1]]
    cd_a = pop.crowding_distance[candidates[:, 0]]
    cd_b = pop.crowding_distance[candidates[:, 1]]

    # a wins if: lower rank OR (same rank AND higher or equal crowding distance)
    a_wins = (rank_a < rank_b) | ((rank_a == rank_b) & (cd_a >= cd_b))

    return np.where(a_wins, candidates[:, 0], candidates[:, 1])


def create_offspring(
    pop: Population,
    n_offspring: int,
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create offspring via selection, crossover, and mutation.

    Selects 2*n_offspring parents using binary tournament, crosses them
    in pairs, and applies mutation to all offspring.

    Args:
        pop: Parent population with rank and crowding_distance computed.
        n_offspring: Number of offspring to create.
        crossover: User's crossover function.
            Signature: (n_vars,), (n_vars,) -> (n_vars,)
        mutate: User's mutation function.
            Signature: (n_vars,) -> (n_vars,)
        rng: Random number generator for reproducibility.

    Returns:
        Array of shape (n_offspring, n_vars) containing offspring decision
        variables (unevaluated).

    Example:
        >>> def simple_crossover(p1, p2):
        ...     return (p1 + p2) / 2
        >>> def simple_mutate(x):
        ...     return x + 0.01 * np.random.randn(len(x))
        >>> offspring = create_offspring(pop, 50, simple_crossover, simple_mutate, rng)
        >>> offspring.shape
        (50, n_vars)
    """
    parent_idx = select_parents(pop, n_offspring * 2, rng)

    # Crossover pairs (2i, 2i+1) to get n_offspring children
    offspring_x = np.stack(
        [crossover(pop.x[parent_idx[2 * i]], pop.x[parent_idx[2 * i + 1]]) for i in range(n_offspring)]
    )

    # Mutate all offspring
    offspring_x = np.stack([mutate(x) for x in offspring_x])

    return offspring_x
