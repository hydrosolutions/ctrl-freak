"""Protocol definitions for selection strategies in evolutionary algorithms.

This module defines the core protocols (interfaces) for selection strategies used
in evolutionary algorithms like NSGA-II and genetic algorithms. These protocols
enable a pluggable architecture where different selection strategies can be
swapped without changing the algorithm implementation.

The two main selection phases in evolutionary algorithms are:

1. **Parent Selection**: Choosing individuals from the current population to
   create offspring. Strategies include tournament selection, roulette wheel,
   and crowded tournament selection (NSGA-II).

2. **Survivor Selection**: Determining which individuals from a combined
   parent+offspring population survive to the next generation. Strategies
   include NSGA-II non-dominated sorting with crowding, truncation selection,
   and elitist strategies.

Examples
--------
Parent and survivor selectors are usually passed into algorithms as callables::

    def my_algorithm(parent_selector, survivor_selector, pop, rng, state):
        parent_indices = parent_selector(pop, n_parents=100, rng=rng, **state)
        survivor_indices, new_state = survivor_selector(
            pop,
            n_survivors=100,
            parent_size=100,
        )
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ctrl_freak.population import Population


@runtime_checkable
class ParentSelector(Protocol):
    """Protocol for parent selection strategies.

    Parent selectors choose which individuals from a population will be used
    as parents for creating offspring. Different strategies (tournament,
    roulette wheel, crowded selection) implement this protocol.

    The selector is called with the population, number of parents to select,
    a random number generator, and optional keyword arguments containing
    algorithm-specific data (e.g., rank, crowding_distance for NSGA-II).

    Parameters
    ----------
    pop : Population
        The current population to select parents from.
    n_parents : int
        Number of parent indices to return. The same individual may be selected
        multiple times depending on the strategy.
    rng : numpy.random.Generator
        Random number generator for reproducible stochastic selection.
    **kwargs : numpy.ndarray
        Algorithm-specific state data. NSGA-II selectors typically receive
        ``rank`` and ``crowding_distance`` arrays. Single-objective selectors
        may receive ``fitness`` values.

    Returns
    -------
    numpy.ndarray
        Indices into the population. Shape is ``(n_parents,)`` with values in
        ``[0, len(pop))``.

    Example implementations:
        - Tournament selection: Compare k random individuals, select best
        - Roulette wheel: Probability proportional to fitness
        - Crowded tournament (NSGA-II): Compare by rank, then crowding distance
        - Random selection: Uniform random choice

    Examples
    --------
    A selector implementation is a callable with the protocol signature::

        def tournament_selector(pop, n_parents, rng, **kwargs):
            fitness = kwargs["fitness"]
            indices = []
            for _ in range(n_parents):
                candidates = rng.choice(len(pop), size=2, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                indices.append(winner)
            return np.array(indices)
    """

    def __call__(
        self,
        pop: Population,
        n_parents: int,
        rng: np.random.Generator,
        **kwargs: np.ndarray,
    ) -> np.ndarray:
        """Select parent indices from the population.

        Parameters
        ----------
        pop : Population
            The current population to select parents from.
        n_parents : int
            Number of parent indices to return.
        rng : numpy.random.Generator
            Random number generator for reproducibility.
        **kwargs : numpy.ndarray
            Algorithm-specific state, such as ``rank`` or
            ``crowding_distance``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_parents,)`` containing selected parent indices.
        """
        ...


@runtime_checkable
class SurvivorSelector(Protocol):
    """Protocol for survivor selection strategies.

    Survivor selectors determine which individuals survive to the next generation
    from a combined parent+offspring population. Different strategies (NSGA-II,
    truncation, elitist) implement this protocol.

    The selector is called with the population, number of survivors to select,
    and optional keyword arguments containing algorithm-specific data.

    Parameters
    ----------
    pop : Population
        Combined population, typically parents plus offspring.
    n_survivors : int
        Number of individuals to select for the next generation.
    **kwargs : Any
        Algorithm-specific input data. Values may include arrays, pre-computed
        metrics, or non-array controls such as the integer ``parent_size``.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, numpy.ndarray]]
        Selected survivor indices and algorithm-specific state for the next
        generation.

    Example implementations:
        - NSGA-II: Non-dominated sorting + crowding distance truncation
        - Truncation: Keep top n_survivors by fitness
        - Elitist: Always keep best individual, random for rest
        - Age-based: Prefer younger individuals

    Examples
    --------
    A selector implementation returns survivor indices and state::

        def truncation_selector(pop, n_survivors, **kwargs):
            fitness = pop.objectives[:, 0]
            survivor_indices = np.argsort(fitness)[:n_survivors]
            return survivor_indices, {"fitness": fitness[survivor_indices]}
    """

    def __call__(
        self,
        pop: Population,
        n_survivors: int,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivor indices from the population.

        Parameters
        ----------
        pop : Population
            The combined population to select survivors from.
        n_survivors : int
            Number of survivors to select.
        **kwargs : Any
            Algorithm-specific input data, including non-array values such as
            integer ``parent_size``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, numpy.ndarray]]
            Selected survivor indices and state data for the next generation.
        """
        ...
