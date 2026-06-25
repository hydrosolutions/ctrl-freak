"""Elitist survival selection for single-objective optimization."""

from __future__ import annotations

import numpy as np

from ctrl_freak.population import Population


def elitist_survival(elite_count: int = 1):
    """Create elitist survivor selector.

    Elitist survival always keeps the best `elite_count` individuals from the
    parent population, filling remaining slots with the best offspring.

    Parameters
    ----------
    elite_count
        Number of elite parents to preserve.

    Returns
    -------
    callable
        Survivor selector function.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.survival.elitist import elitist_survival
    >>> obj = np.array([[2.0], [1.0], [4.0], [0.5]])
    >>> pop = Population(x=np.zeros((4, 1)), objectives=obj)
    >>> indices, state = elitist_survival(elite_count=1)(pop, 2, parent_size=2)
    >>> indices.shape
    (2,)
    >>> state["fitness"].shape
    (2,)
    """
    if elite_count <= 0:
        raise ValueError(f"elite_count must be positive, got {elite_count}")

    def selector(
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivors using elitist selection.

        Parameters
        ----------
        pop
            Combined parent and offspring population.
        n_survivors
            Number of survivors to select.
        **kwargs
            Must include ``parent_size``. May include explicit ``fitness``.

        Returns
        -------
        tuple[numpy.ndarray, dict[str, numpy.ndarray]]
            Selected indices and state containing the selected ``fitness`` values.

        Raises
        ------
        ValueError
            If inputs are invalid or no valid fitness source is available.

        Examples
        --------
        >>> import numpy as np
        >>> from ctrl_freak.population import Population
        >>> from ctrl_freak.survival.elitist import elitist_survival
        >>> obj = np.array([[2.0], [1.0], [4.0], [3.0], [0.5], [1.5]])
        >>> pop = Population(x=np.zeros((6, 1)), objectives=obj)
        >>> indices, state = elitist_survival(elite_count=1)(pop, 3, parent_size=3)
        >>> indices
        array([1, 4, 5])
        >>> state["fitness"].shape
        (3,)
        """
        # Validate n_survivors
        if n_survivors <= 0:
            raise ValueError(f"n_survivors must be positive, got {n_survivors}")
        if n_survivors > len(pop):
            raise ValueError(f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})")

        # Validate parent_size kwarg
        parent_size = kwargs.get("parent_size")
        if parent_size is None:
            raise ValueError(
                "elitist survival requires 'parent_size' kwarg to distinguish "
                "parents from offspring in the combined population"
            )
        if not isinstance(parent_size, (int, np.integer)):
            raise ValueError(f"parent_size must be an integer, got {type(parent_size)}")
        if parent_size <= 0:
            raise ValueError(f"parent_size must be positive, got {parent_size}")
        if parent_size > len(pop):
            raise ValueError(f"parent_size ({parent_size}) cannot exceed population size ({len(pop)})")

        # Validate elite_count constraints
        if elite_count > parent_size:
            raise ValueError(f"elite_count ({elite_count}) cannot exceed parent_size ({parent_size})")
        if elite_count > n_survivors:
            raise ValueError(f"elite_count ({elite_count}) cannot exceed n_survivors ({n_survivors})")

        # Get fitness values
        fitness = kwargs.get("fitness")
        if fitness is None:
            if pop.objectives is None:
                raise ValueError("Population must have objectives computed for survivor selection")
            if pop.objectives.shape[1] != 1:
                raise ValueError(
                    f"elitist survival requires single-objective optimization "
                    f"(got {pop.objectives.shape[1]} objectives). "
                    "Pass explicit 'fitness' kwarg for multi-objective."
                )
            fitness = pop.objectives[:, 0]

        # Split population into parents and offspring
        parent_fitness = fitness[:parent_size]
        offspring_fitness = fitness[parent_size:]

        # Select elite parents (best elite_count from parents)
        # Use stable sort for deterministic tie-breaking
        parent_sorted_indices = np.argsort(parent_fitness, kind="stable")
        elite_indices = parent_sorted_indices[:elite_count].astype(np.intp)

        # Select best offspring to fill remaining slots
        n_offspring_needed = n_survivors - elite_count
        if n_offspring_needed > 0:
            # Use stable sort for deterministic tie-breaking
            offspring_sorted_indices = np.argsort(offspring_fitness, kind="stable")
            # Map back to original indices in combined population
            best_offspring_indices = (offspring_sorted_indices[:n_offspring_needed] + parent_size).astype(np.intp)

            # Combine elite parents and best offspring
            survivor_indices = np.concatenate([elite_indices, best_offspring_indices])
        else:
            survivor_indices = elite_indices

        # Compute state for survivors
        survivor_fitness = fitness[survivor_indices].copy()

        return survivor_indices, {"fitness": survivor_fitness}

    return selector
