"""Elitist survival selection for single-objective optimization."""

from __future__ import annotations

import numpy as np

from ctrl_freak.population import Population


def elitist_survival(elite_count: int = 1):
    """Create elitist survivor selector.

    Elitist survival always keeps the best `elite_count` individuals from the
    parent population, filling remaining slots with the best offspring.

    Args:
        elite_count: Number of elite parents to preserve. Default is 1.

    Returns:
        A survivor selector function.

    Example:
        >>> selector = elitist_survival(elite_count=2)
        >>> indices, state = selector(combined_pop, n_survivors=10, parent_size=5)
    """
    if elite_count <= 0:
        raise ValueError(f"elite_count must be positive, got {elite_count}")

    def selector(
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Select survivors using elitist selection.

        Args:
            pop: Combined population (parents + offspring) to select survivors from.
            n_survivors: Number of survivors to select for the next generation.
            **kwargs: Required keyword arguments:
                - parent_size: Number of parent individuals in the combined population.
                Optional keyword arguments:
                - fitness: Explicit fitness array. If not provided, extracted from
                  pop.objectives (which must be single-objective).

        Returns:
            Tuple of (indices, state) where:
            - indices: Array of shape (n_survivors,) containing indices of selected
              survivors from the input population. Elite parents come first, followed
              by best offspring.
            - state: Dictionary with key 'fitness' containing fitness values of
              selected survivors. Shape (n_survivors,).

        Raises:
            ValueError: If population has no objectives and no fitness kwarg provided,
                if n_survivors is invalid, if n_survivors exceeds population size,
                if population has multiple objectives without explicit fitness kwarg,
                if parent_size kwarg is not provided, if elite_count exceeds parent_size,
                or if elite_count exceeds n_survivors.

        Example:
            >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
            >>> obj = np.array([[2.0], [1.0], [4.0], [3.0], [0.5], [1.5]])
            >>> pop = Population(x=x, objectives=obj)  # 3 parents + 3 offspring
            >>> selector = elitist_survival(elite_count=1)
            >>> indices, state = selector(pop, n_survivors=3, parent_size=3)
            >>> # Should select: best parent (idx 1), then best 2 offspring (idx 4, 5)
        """
        # Validate n_survivors
        if n_survivors <= 0:
            raise ValueError(f"n_survivors must be positive, got {n_survivors}")
        if n_survivors > len(pop):
            raise ValueError(
                f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})"
            )

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
            raise ValueError(
                f"parent_size ({parent_size}) cannot exceed population size ({len(pop)})"
            )

        # Validate elite_count constraints
        if elite_count > parent_size:
            raise ValueError(
                f"elite_count ({elite_count}) cannot exceed parent_size ({parent_size})"
            )
        if elite_count > n_survivors:
            raise ValueError(
                f"elite_count ({elite_count}) cannot exceed n_survivors ({n_survivors})"
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
            best_offspring_indices = (
                offspring_sorted_indices[:n_offspring_needed] + parent_size
            ).astype(np.intp)

            # Combine elite parents and best offspring
            survivor_indices = np.concatenate([elite_indices, best_offspring_indices])
        else:
            survivor_indices = elite_indices

        # Compute state for survivors
        survivor_fitness = fitness[survivor_indices].copy()

        return survivor_indices, {"fitness": survivor_fitness}

    return selector
