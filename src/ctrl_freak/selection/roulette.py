"""Roulette wheel (fitness-proportionate) selection for single-objective optimization."""

import numpy as np

from ctrl_freak.population import Population


def roulette_wheel():
    """Create a roulette wheel (fitness-proportionate) parent selector.

    Selection probability is inversely proportional to fitness because
    lower fitness is better (minimization). Uses max_fitness - fitness
    to convert minimization to maximization probabilities.

    For fitness values f_i, the selection probability p_i is computed as:
        weights_i = max_fitness - f_i + ε
        p_i = weights_i / Σ(weights_j)

    where ε is a small constant to ensure the worst individual has non-zero probability.

    Returns
    -------
    callable
        Parent selector that performs fitness-proportionate selection.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.selection.roulette import roulette_wheel
    >>> pop = Population(x=np.zeros((4, 2)), objectives=np.array([[4.0], [1.0], [2.0], [3.0]]))
    >>> selector = roulette_wheel()
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
        """Select parents using fitness-proportionate (roulette wheel) selection.

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
            Selected parent indices with shape ``(n_parents,)`` and dtype
            ``np.intp``.

        Raises
        ------
        ValueError
            If no valid fitness source is available.

        Examples
        --------
        >>> import numpy as np
        >>> from ctrl_freak.population import Population
        >>> from ctrl_freak.selection.roulette import roulette_wheel
        >>> pop = Population(x=np.zeros((3, 1)), objectives=np.array([[3.0], [1.0], [2.0]]))
        >>> selector = roulette_wheel()
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
                "roulette wheel selection requires 'fitness' in kwargs or single-column objectives in population"
            )

        pop_size = len(pop)

        # Handle edge case: all equal fitness -> uniform selection
        if np.all(fitness == fitness[0]):
            # All individuals have equal fitness, select uniformly
            return rng.choice(pop_size, size=n_parents, replace=True).astype(np.intp)

        # Convert minimization to maximization: lower fitness -> higher selection probability
        # Use max - fitness approach to avoid division by near-zero values
        max_fitness = np.max(fitness)
        epsilon = 1e-10  # Small constant to handle max fitness case

        # Compute selection weights: (max_fitness - fitness + epsilon)
        # This ensures the best individual (lowest fitness) gets highest weight
        weights = max_fitness - fitness + epsilon

        # Normalize to probabilities
        probs = weights / weights.sum()

        # Select n_parents indices with replacement using roulette wheel
        selected = rng.choice(pop_size, size=n_parents, replace=True, p=probs)

        return selected.astype(np.intp)

    return selector
