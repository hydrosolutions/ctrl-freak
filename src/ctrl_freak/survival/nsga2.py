"""NSGA-II survivor selection strategy.

Examples
--------
>>> import numpy as np
>>> from ctrl_freak.population import Population
>>> from ctrl_freak.survival.nsga2 import nsga2_survival
>>> obj = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
>>> pop = Population(x=np.zeros((4, 1)), objectives=obj)
>>> selector = nsga2_survival()
>>> indices, state = selector(pop, n_survivors=2)
>>> indices.shape
(2,)
>>> sorted(state)
['crowding_distance', 'rank']
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

    Returns
    -------
    callable
        Survivor selector that returns selected indices and rank/crowding state.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.survival.nsga2 import nsga2_survival
    >>> obj = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    >>> pop = Population(x=np.zeros((4, 1)), objectives=obj)
    >>> selector = nsga2_survival()
    >>> indices, state = selector(pop, n_survivors=2)
    >>> indices.shape
    (2,)
    >>> sorted(state)
    ['crowding_distance', 'rank']
    """

    def selector(
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivors using NSGA-II crowded selection.

        Parameters
        ----------
        pop
            Population to select from.
        n_survivors
            Number of survivors to select.
        **kwargs
            Unused keyword arguments.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, numpy.ndarray]]
            Selected indices and state containing ``rank`` and
            ``crowding_distance`` arrays.

        Raises
        ------
        ValueError
            If objectives are missing or ``n_survivors`` is invalid.

        Examples
        --------
        >>> import numpy as np
        >>> from ctrl_freak.population import Population
        >>> from ctrl_freak.survival.nsga2 import nsga2_survival
        >>> obj = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        >>> pop = Population(x=np.zeros((3, 1)), objectives=obj)
        >>> indices, state = nsga2_survival()(pop, n_survivors=2)
        >>> indices.shape
        (2,)
        >>> state["rank"].shape
        (2,)
        """
        if pop.objectives is None:
            raise ValueError("Population must have objectives computed for survivor selection")
        if n_survivors <= 0:
            raise ValueError(f"n_survivors must be positive, got {n_survivors}")
        if n_survivors > len(pop):
            raise ValueError(f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})")

        # Compute ranks for entire population
        all_ranks = non_dominated_sort(pop.objectives)

        # Fill survivors front-by-front, caching crowding distance as we go.
        selected: list[int] = []
        selected_cd_parts: list[np.ndarray] = []
        current_rank = 0

        while len(selected) < n_survivors:
            front_idx = np.where(all_ranks == current_rank)[0]

            if len(selected) + len(front_idx) <= n_survivors:
                # Whole front fits: crowding over the full front equals crowding over the
                # selected subset, so cache it directly.
                selected.extend(front_idx.tolist())
                selected_cd_parts.append(crowding_distance(pop.objectives[front_idx]))
            else:
                # Critical front: select by full-front crowding, then recompute over the
                # selected subset to match the prior returned-state semantics.
                remaining = n_survivors - len(selected)
                cd = crowding_distance(pop.objectives[front_idx])
                top_cd_indices = np.argsort(cd)[::-1][:remaining]
                chosen = front_idx[top_cd_indices]
                selected.extend(chosen.tolist())
                selected_cd_parts.append(crowding_distance(pop.objectives[chosen]))

            current_rank += 1

        selected_arr = np.array(selected, dtype=np.intp)
        selected_ranks = all_ranks[selected_arr]
        selected_cd = np.concatenate(selected_cd_parts).astype(np.float64)

        return selected_arr, {
            "rank": selected_ranks,
            "crowding_distance": selected_cd,
        }

    return selector
