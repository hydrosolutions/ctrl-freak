"""Crowded tournament selection for multi-objective optimization."""

import numpy as np

from ctrl_freak.population import Population


def crowded_tournament(tournament_size: int = 2):
    """Create a crowded tournament parent selector.

    In crowded tournament selection, individuals are compared by:
    1. Pareto rank (lower is better)
    2. If ranks are equal, crowding distance (higher is better for diversity)

    Args:
        tournament_size: Number of individuals in each tournament. Default 2.

    Returns:
        A ParentSelector callable that selects parent indices.

    Example:
        >>> selector = crowded_tournament(tournament_size=2)
        >>> parents = selector(pop, n_parents=20, rng=rng, rank=rank, crowding_distance=cd)
    """
    def selector(
        pop: Population,
        n_parents: int,
        rng: np.random.Generator,
        **kwargs: np.ndarray,
    ) -> np.ndarray:
        """Select parents using crowded tournament selection.

        Args:
            pop: Population to select from.
            n_parents: Number of parents to select.
            rng: Random number generator for reproducibility.
            **kwargs: Must include 'rank' and 'crowding_distance' arrays.

        Returns:
            Array of selected parent indices.

        Raises:
            ValueError: If 'rank' or 'crowding_distance' not in kwargs.
        """
        # Validate required kwargs
        if "rank" not in kwargs:
            raise ValueError("crowded tournament selection requires 'rank' in kwargs")
        if "crowding_distance" not in kwargs:
            raise ValueError("crowded tournament selection requires 'crowding_distance' in kwargs")

        rank = kwargs["rank"]
        crowding_distance = kwargs["crowding_distance"]
        pop_size = len(pop)

        # Select n_parents winners via tournament
        selected = np.empty(n_parents, dtype=np.intp)

        for i in range(n_parents):
            # Pick tournament_size random individuals
            candidates = rng.integers(0, pop_size, size=tournament_size)

            # Find winner: prefer lower rank, break ties with higher crowding distance
            best_idx = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best_idx] or rank[c] == rank[best_idx] and crowding_distance[c] > crowding_distance[best_idx]:
                    best_idx = c

            selected[i] = best_idx

        return selected

    return selector
