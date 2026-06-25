"""Fitness tournament selection for single-objective optimization."""

import numpy as np

from ctrl_freak.population import Population


def fitness_tournament(tournament_size: int = 2):
    """Create a fitness-based tournament parent selector for single-objective optimization.

    Parameters
    ----------
    tournament_size
        Number of individuals competing in each tournament.

    Returns
    -------
    callable
        Parent selector that returns selected parent indices.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.selection.tournament import fitness_tournament
    >>> pop = Population(x=np.zeros((4, 2)), objectives=np.array([[3.0], [1.0], [2.0], [4.0]]))
    >>> selector = fitness_tournament(tournament_size=2)
    >>> parents = selector(pop, 5, np.random.default_rng(0))
    >>> parents.shape
    (5,)
    """

    def selector(
        pop: Population,
        n_parents: int,
        rng: np.random.Generator,
        **kwargs: np.ndarray,
    ) -> np.ndarray:
        """Select parents using fitness tournament selection.

        Parameters
        ----------
        pop
            Population to select from.
        n_parents
            Number of parents to select.
        rng
            Random number generator.
        **kwargs
            Optional ``fitness`` array. If omitted, fitness is extracted from a
            single-objective population.

        Returns
        -------
        numpy.ndarray
            Selected parent indices.

        Raises
        ------
        ValueError
            If no valid fitness source is available.

        Examples
        --------
        >>> import numpy as np
        >>> from ctrl_freak.population import Population
        >>> from ctrl_freak.selection.tournament import fitness_tournament
        >>> pop = Population(x=np.zeros((3, 1)), objectives=np.array([[2.0], [1.0], [3.0]]))
        >>> selector = fitness_tournament()
        >>> out = selector(pop, 4, np.random.default_rng(2))
        >>> out.shape
        (4,)
        """
        # Get fitness array
        if "fitness" in kwargs:
            fitness = kwargs["fitness"]
        elif pop.objectives is not None and pop.objectives.shape[1] == 1:
            # Extract from single-column objectives
            fitness = pop.objectives[:, 0]
        else:
            raise ValueError(
                "fitness tournament selection requires 'fitness' in kwargs or single-column objectives in population"
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
