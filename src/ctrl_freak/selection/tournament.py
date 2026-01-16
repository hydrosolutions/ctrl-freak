"""Fitness tournament selection for single-objective optimization."""

import numpy as np

from ctrl_freak.population import Population


def fitness_tournament(tournament_size: int = 2):
    """Create a fitness-based tournament parent selector for single-objective optimization.

    Args:
        tournament_size: Number of individuals competing in each tournament (default: 2).

    Returns:
        A ParentSelector callable that selects parent indices based on fitness.

    Example:
        >>> selector = fitness_tournament(tournament_size=3)
        >>> # Use with explicit fitness
        >>> parents = selector(pop, n_parents=20, rng=rng, fitness=fitness_array)
        >>> # Or with single-objective population (extracts from objectives)
        >>> parents = selector(pop, n_parents=20, rng=rng)
    """
    def selector(
        pop: Population,
        n_parents: int,
        rng: np.random.Generator,
        **kwargs: np.ndarray,
    ) -> np.ndarray:
        """Select parents using fitness tournament selection.

        Args:
            pop: Population to select from.
            n_parents: Number of parents to select.
            rng: Random number generator for reproducibility.
            **kwargs: May include 'fitness' array (1D). If not provided, extracts from
                pop.objectives if it has exactly one column.

        Returns:
            Array of selected parent indices.

        Raises:
            ValueError: If no fitness source is available (no 'fitness' kwarg and
                either objectives is None or has multiple columns).
        """
        # Get fitness array
        if "fitness" in kwargs:
            fitness = kwargs["fitness"]
        elif pop.objectives is not None and pop.objectives.shape[1] == 1:
            # Extract from single-column objectives
            fitness = pop.objectives[:, 0]
        else:
            raise ValueError(
                "fitness tournament selection requires 'fitness' in kwargs or "
                "single-column objectives in population"
            )

        pop_size = len(pop)

        # Select n_parents winners via tournament
        selected = np.empty(n_parents, dtype=np.intp)

        for i in range(n_parents):
            # Pick tournament_size random individuals
            candidates = rng.integers(0, pop_size, size=tournament_size)

            # Find winner: prefer lower fitness (minimization)
            best_idx = candidates[0]
            for c in candidates[1:]:
                if fitness[c] < fitness[best_idx]:
                    best_idx = c

            selected[i] = best_idx

        return selected

    return selector
