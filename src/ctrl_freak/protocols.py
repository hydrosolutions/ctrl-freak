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

Example usage:
    ```python
    def my_algorithm(
        parent_selector: ParentSelector,
        survivor_selector: SurvivorSelector,
        ...
    ):
        # Select parents for mating
        parent_indices = parent_selector(pop, n_parents=100, rng=rng, **state)

        # After creating offspring and combining populations
        survivor_indices, new_state = survivor_selector(combined_pop, n_survivors=100)
    ```
"""

from typing import Protocol, runtime_checkable

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

    Parameters:
        pop: The current population to select parents from.
        n_parents: Number of parent indices to return. May select the same
            individual multiple times depending on the strategy.
        rng: NumPy random number generator for reproducible stochastic selection.
        **kwargs: Algorithm-specific state data. For NSGA-II, this typically
            includes 'rank' and 'crowding_distance' arrays. For single-objective
            GA, this might include 'fitness' values.

    Returns:
        Array of indices into the population indicating selected parents.
        Shape is (n_parents,) with values in range [0, len(pop)).

    Example implementations:
        - Tournament selection: Compare k random individuals, select best
        - Roulette wheel: Probability proportional to fitness
        - Crowded tournament (NSGA-II): Compare by rank, then crowding distance
        - Random selection: Uniform random choice

    Example:
        ```python
        def tournament_selector(
            pop: Population,
            n_parents: int,
            rng: np.random.Generator,
            **kwargs
        ) -> np.ndarray:
            fitness = kwargs['fitness']
            indices = []
            for _ in range(n_parents):
                candidates = rng.choice(len(pop), size=2, replace=False)
                winner = candidates[np.argmax(fitness[candidates])]
                indices.append(winner)
            return np.array(indices)
        ```
    """

    def __call__(
        self,
        pop: Population,
        n_parents: int,
        rng: np.random.Generator,
        **kwargs: np.ndarray,
    ) -> np.ndarray:
        """Select parent indices from the population.

        Args:
            pop: The current population to select parents from.
            n_parents: Number of parent indices to return.
            rng: NumPy random number generator for reproducibility.
            **kwargs: Algorithm-specific state (e.g., rank, crowding_distance).

        Returns:
            Array of shape (n_parents,) containing indices of selected parents.
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

    Parameters:
        pop: The combined population (typically parents + offspring) to select
            survivors from.
        n_survivors: Number of individuals to select for the next generation.
            Usually equals the original population size.
        **kwargs: Algorithm-specific input data. May include pre-computed
            metrics from previous generations.

    Returns:
        A tuple of:
        - indices: Array of indices into the population for selected survivors.
            Shape is (n_survivors,) with unique values in range [0, len(pop)).
        - state: Dictionary containing algorithm-specific state that should be
            passed to the next generation's parent selection. For NSGA-II,
            this includes {'rank': ..., 'crowding_distance': ...}. For GA,
            this might be {'fitness': ...}. Keys are strings, values are
            numpy arrays aligned with the selected survivors.

    Example implementations:
        - NSGA-II: Non-dominated sorting + crowding distance truncation
        - Truncation: Keep top n_survivors by fitness
        - Elitist: Always keep best individual, random for rest
        - Age-based: Prefer younger individuals

    Example:
        ```python
        def truncation_selector(
            pop: Population,
            n_survivors: int,
            **kwargs
        ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
            # Assuming single-objective minimization
            fitness = pop.objectives[:, 0]
            sorted_indices = np.argsort(fitness)
            survivor_indices = sorted_indices[:n_survivors]
            return survivor_indices, {'fitness': fitness[survivor_indices]}
        ```
    """

    def __call__(
        self,
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivor indices from the population.

        Args:
            pop: The combined population to select survivors from.
            n_survivors: Number of survivors to select.
            **kwargs: Algorithm-specific input data.

        Returns:
            Tuple of (indices, state) where indices is an array of selected
            survivor indices and state is a dictionary of algorithm-specific
            data for the next generation.
        """
        ...
