"""Truncation survival selection for single-objective optimization."""

from __future__ import annotations

import numpy as np

from ctrl_freak.population import Population


def truncation_survival():
    """Create truncation survivor selector.

    Truncation survival keeps the k best individuals by fitness value.
    Lower fitness is better (minimization).

    Returns
    -------
    callable
        Survivor selector callable.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.population import Population
    >>> from ctrl_freak.survival.truncation import truncation_survival
    >>> obj = np.array([[4.0], [2.0], [3.0], [1.0]])
    >>> pop = Population(x=np.zeros((4, 1)), objectives=obj)
    >>> indices, state = truncation_survival()(pop, n_survivors=2)
    >>> indices
    array([3, 1])
    >>> state["fitness"].shape
    (2,)
    """

    def selector(
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivors using truncation selection.

        Parameters
        ----------
        pop
            Population to select from.
        n_survivors
            Number of survivors to select.
        **kwargs
            Optional ``fitness`` array. If omitted, fitness is extracted from a
            single-objective population.

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
        >>> from ctrl_freak.survival.truncation import truncation_survival
        >>> obj = np.array([[4.0], [2.0], [3.0], [1.0]])
        >>> pop = Population(x=np.zeros((4, 1)), objectives=obj)
        >>> indices, state = truncation_survival()(pop, n_survivors=2)
        >>> indices
        array([3, 1])
        >>> state["fitness"]
        array([1., 2.])
        """
        # Validate n_survivors
        if n_survivors <= 0:
            raise ValueError(f"n_survivors must be positive, got {n_survivors}")
        if n_survivors > len(pop):
            raise ValueError(f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})")

        # Get fitness values
        fitness = kwargs.get("fitness")
        if fitness is None:
            if pop.objectives is None:
                raise ValueError("Population must have objectives computed for survivor selection")
            if pop.objectives.shape[1] != 1:
                raise ValueError(
                    f"truncation requires single-objective optimization "
                    f"(got {pop.objectives.shape[1]} objectives). "
                    "Pass explicit 'fitness' kwarg for multi-objective."
                )
            fitness = pop.objectives[:, 0]

        # Sort by fitness (ascending = best first for minimization)
        # Use stable sort for deterministic tie-breaking
        sorted_indices = np.argsort(fitness, kind="stable")
        survivor_indices = sorted_indices[:n_survivors].astype(np.intp)

        # Compute state for survivors
        survivor_fitness = fitness[survivor_indices].copy()

        return survivor_indices, {"fitness": survivor_fitness}

    return selector
