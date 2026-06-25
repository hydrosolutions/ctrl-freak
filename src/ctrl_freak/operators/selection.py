"""Selection and offspring creation operators for NSGA-II.

This module provides:
- select_parents: Binary tournament selection using crowded comparison
- create_offspring: Create offspring via selection, crossover, and mutation
"""

from collections.abc import Callable

import numpy as np

from ctrl_freak.population import Population


def select_parents(
    pop: Population,
    n_parents: int,
    rng: np.random.Generator,
    rank: np.ndarray,
    crowding_distance: np.ndarray,
) -> np.ndarray:
    """Select parents using binary tournament selection (vectorized).

    Uses the crowded comparison operator: lower rank wins, ties broken by
    higher crowding distance.

    Parameters
    ----------
    pop : Population
        Population used for its size.
    n_parents : int
        Number of parents to select.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    rank : numpy.ndarray
        Pareto front ranks for all individuals. Shape is ``(n,)``.
    crowding_distance : numpy.ndarray
        Crowding distances for all individuals. Shape is ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_parents,)`` containing indices into ``pop``.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.operators.selection import select_parents
    >>> pop = Population(x=np.zeros((4, 2)), objectives=np.zeros((4, 2)))
    >>> rng = np.random.default_rng(0)
    >>> parents = select_parents(
    ...     pop,
    ...     n_parents=10,
    ...     rng=rng,
    ...     rank=np.array([0, 0, 1, 1]),
    ...     crowding_distance=np.array([1.0, 2.0, 1.0, 2.0]),
    ... )
    >>> parents.shape
    (10,)
    """
    n = len(pop.x)
    candidates = rng.integers(0, n, size=(n_parents, 2))

    rank_a = rank[candidates[:, 0]]
    rank_b = rank[candidates[:, 1]]
    cd_a = crowding_distance[candidates[:, 0]]
    cd_b = crowding_distance[candidates[:, 1]]

    # a wins if: lower rank OR (same rank AND higher or equal crowding distance)
    a_wins = (rank_a < rank_b) | ((rank_a == rank_b) & (cd_a >= cd_b))

    return np.where(a_wins, candidates[:, 0], candidates[:, 1])


def create_offspring(
    pop: Population,
    n_offspring: int,
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    rank: np.ndarray,
    crowding_distance: np.ndarray,
) -> np.ndarray:
    """Create offspring via selection, crossover, and mutation.

    Selects 2*n_offspring parents using binary tournament, crosses them
    in pairs, and applies mutation to all offspring.

    Parameters
    ----------
    pop : Population
        Parent population.
    n_offspring : int
        Number of offspring to create.
    crossover : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
        Crossover function with signature ``(n_vars,), (n_vars,) -> (n_vars,)``.
    mutate : Callable[[numpy.ndarray], numpy.ndarray]
        Mutation function with signature ``(n_vars,) -> (n_vars,)``.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    rank : numpy.ndarray
        Pareto front ranks for all individuals. Shape is ``(n,)``.
    crowding_distance : numpy.ndarray
        Crowding distances for all individuals. Shape is ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_offspring, n_vars)`` containing unevaluated
        offspring decision variables.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.operators.selection import create_offspring
    >>> pop = Population(x=np.arange(8.0).reshape(4, 2), objectives=np.zeros((4, 2)))
    >>> rng = np.random.default_rng(0)
    >>> crossover = lambda p1, p2: (p1 + p2) / 2
    >>> mutate = lambda x: x.copy()
    >>> offspring = create_offspring(
    ...     pop,
    ...     n_offspring=3,
    ...     crossover=crossover,
    ...     mutate=mutate,
    ...     rng=rng,
    ...     rank=np.array([0, 0, 1, 1]),
    ...     crowding_distance=np.array([1.0, 2.0, 1.0, 2.0]),
    ... )
    >>> offspring.shape
    (3, 2)
    """
    parent_idx = select_parents(pop, n_offspring * 2, rng, rank=rank, crowding_distance=crowding_distance)

    # Crossover pairs (2i, 2i+1) to get n_offspring children
    offspring_x = np.stack(
        [crossover(pop.x[parent_idx[2 * i]], pop.x[parent_idx[2 * i + 1]]) for i in range(n_offspring)]
    )

    # Mutate all offspring
    offspring_x = np.stack([mutate(x) for x in offspring_x])

    return offspring_x
