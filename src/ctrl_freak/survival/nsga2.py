"""NSGA-II survivor selection strategy.

This module implements the NSGA-II survivor selection strategy which uses
Pareto ranking and crowding distance to select individuals for the next
generation. This is the core survival mechanism of the NSGA-II algorithm.
"""

import numpy as np

from ctrl_freak.population import Population
from ctrl_freak.primitives import crowding_distance, non_dominated_sort


def nsga2_survival():
    """Create NSGA-II survivor selector.

    The NSGA-II survival strategy implements elitist selection by:
    1. Computing Pareto ranks using non-dominated sorting
    2. Filling survivors front-by-front in rank order
    3. When a front only partially fits, selecting individuals with highest
       crowding distance (most isolated in objective space)

    This preserves both convergence (keeping better fronts) and diversity
    (preferring isolated individuals within a front).

    Returns:
        A SurvivorSelector callable that selects survivor indices and returns
        state with 'rank' and 'crowding_distance' arrays.

    Example:
        >>> selector = nsga2_survival()
        >>> survivors, state = selector(combined_pop, n_survivors=100)
        >>> rank = state['rank']
        >>> crowding_distance = state['crowding_distance']
    """

    def selector(
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivors using NSGA-II crowded selection.

        Args:
            pop: Combined population (typically parents + offspring) to select from.
            n_survivors: Number of survivors to select for the next generation.
            **kwargs: Unused. NSGA-II computes all metrics internally.

        Returns:
            Tuple of (indices, state) where:
            - indices: Array of shape (n_survivors,) containing indices of selected
              survivors from the input population.
            - state: Dictionary with keys:
                - 'rank': Pareto front ranks for selected survivors. Shape (n_survivors,).
                - 'crowding_distance': Crowding distances for selected survivors.
                  Shape (n_survivors,). Boundary individuals get infinite distance.

        Raises:
            ValueError: If population has no objectives, n_survivors is invalid,
                or n_survivors exceeds population size.

        Example:
            >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> obj = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
            >>> pop = Population(x=x, objectives=obj)
            >>> selector = nsga2_survival()
            >>> indices, state = selector(pop, n_survivors=2)
            >>> len(indices)
            2
        """
        if pop.objectives is None:
            raise ValueError("Population must have objectives computed for survivor selection")
        if n_survivors <= 0:
            raise ValueError(f"n_survivors must be positive, got {n_survivors}")
        if n_survivors > len(pop):
            raise ValueError(f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})")

        # Compute ranks for entire population
        all_ranks = non_dominated_sort(pop.objectives)

        # Fill survivors front-by-front
        selected: list[int] = []
        current_rank = 0

        while len(selected) < n_survivors:
            # Get indices of individuals in current front
            front_idx = np.where(all_ranks == current_rank)[0]

            if len(selected) + len(front_idx) <= n_survivors:
                # Whole front fits - add all
                selected.extend(front_idx.tolist())
            else:
                # Critical front - select by highest crowding distance
                remaining = n_survivors - len(selected)
                cd = crowding_distance(pop.objectives[front_idx])
                # Sort by crowding distance descending, take top 'remaining'
                top_cd_indices = np.argsort(cd)[::-1][:remaining]
                selected.extend(front_idx[top_cd_indices].tolist())

            current_rank += 1

        selected_arr = np.array(selected, dtype=np.intp)

        # Compute crowding distance for all fronts in selected survivors
        selected_ranks = all_ranks[selected_arr]
        selected_cd = np.zeros(n_survivors, dtype=np.float64)

        for r in range(int(selected_ranks.max()) + 1):
            mask = selected_ranks == r
            if np.any(mask):
                # Get indices in selected_arr that belong to this front
                front_indices = np.where(mask)[0]
                # Get original population indices for this front
                original_indices = selected_arr[front_indices]
                # Compute crowding distance for this front's objectives
                selected_cd[front_indices] = crowding_distance(pop.objectives[original_indices])

        return selected_arr, {
            "rank": selected_ranks,
            "crowding_distance": selected_cd,
        }

    return selector
