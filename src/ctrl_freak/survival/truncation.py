"""Truncation survival selection for single-objective optimization."""

from __future__ import annotations

import numpy as np

from ctrl_freak.population import Population


def truncation_survival():
    """Create truncation survivor selector.

    Truncation survival keeps the k best individuals by fitness value.
    Lower fitness is better (minimization).

    Returns:
        A SurvivorSelector callable.

    Example:
        >>> selector = truncation_survival()
        >>> indices, state = selector(pop, n_survivors=10)
        >>> state['fitness']  # fitness values of survivors
    """

    def selector(
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivors using truncation selection.

        Args:
            pop: Population to select survivors from.
            n_survivors: Number of survivors to select for the next generation.
            **kwargs: Optional keyword arguments. If 'fitness' is provided, it is used
                directly. Otherwise, fitness is extracted from pop.objectives (which
                must be single-objective).

        Returns:
            Tuple of (indices, state) where:
            - indices: Array of shape (n_survivors,) containing indices of selected
              survivors from the input population, ordered by fitness (best first).
            - state: Dictionary with key 'fitness' containing fitness values of
              selected survivors. Shape (n_survivors,).

        Raises:
            ValueError: If population has no objectives and no fitness kwarg provided,
                if n_survivors is invalid, if n_survivors exceeds population size,
                or if population has multiple objectives without explicit fitness kwarg.

        Example:
            >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> obj = np.array([[4.0], [2.0], [3.0], [1.0]])
            >>> pop = Population(x=x, objectives=obj)
            >>> selector = truncation_survival()
            >>> indices, state = selector(pop, n_survivors=2)
            >>> indices  # [3, 1] - individuals with fitness 1.0 and 2.0
            >>> state['fitness']  # [1.0, 2.0]
        """
        # Validate n_survivors
        if n_survivors <= 0:
            raise ValueError(f"n_survivors must be positive, got {n_survivors}")
        if n_survivors > len(pop):
            raise ValueError(
                f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})"
            )

        # Get fitness values
        fitness = kwargs.get("fitness")
        if fitness is None:
            if pop.objectives is None:
                raise ValueError(
                    "Population must have objectives computed for survivor selection"
                )
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
