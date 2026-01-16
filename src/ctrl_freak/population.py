"""Population data structures for multi-objective optimization.

This module provides the core data structures for representing populations
of individuals in multi-objective optimization algorithms:

- Population: A struct-of-arrays representation of multiple individuals
- IndividualView: A read-only view of a single individual

Both classes are immutable (frozen dataclasses) to enforce functional style.
Population is algorithm-agnostic - it only stores decision variables and objectives.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndividualView:
    """Read-only view of a single individual in a population.

    This class provides a convenient way to access the data for a single
    individual, returned by Population.__getitem__.

    Attributes:
        x: Decision variables for this individual, shape (n_vars,).
        objectives: Objective values for this individual, shape (n_obj,), or None.

    Example:
        >>> pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> individual = pop[0]
        >>> individual.x
        array([1., 2.])
    """

    x: np.ndarray
    objectives: np.ndarray | None


@dataclass(frozen=True)
class Population:
    """Immutable struct-of-arrays representation of a population.

    This class stores multiple individuals using a struct-of-arrays layout
    for efficient vectorized operations. All arrays are copied on construction
    to ensure immutability.

    Population is algorithm-agnostic - it only stores decision variables and
    objective values. Algorithm-specific data (like Pareto ranks or crowding
    distances) should be managed separately by the algorithm implementation.

    Attributes:
        x: Decision variables for all individuals, shape (n, n_vars).
        objectives: Objective values, shape (n, n_obj), or None if not evaluated.

    Example:
        >>> x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> obj = np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])
        >>> pop = Population(x=x, objectives=obj)
        >>> len(pop)
        3
        >>> pop.n_vars
        2
        >>> pop.n_obj
        2
    """

    x: np.ndarray
    objectives: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate shapes and copy arrays for immutability.

        Raises:
            TypeError: If x is not a numpy array.
            ValueError: If array shapes are inconsistent or invalid.
        """
        # Validate x
        if not isinstance(self.x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(self.x).__name__}")
        if self.x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {self.x.shape}")

        n = self.x.shape[0]

        # Copy x for immutability (use object.__setattr__ for frozen dataclass)
        object.__setattr__(self, "x", self.x.copy())

        # Validate and copy objectives
        if self.objectives is not None:
            if not isinstance(self.objectives, np.ndarray):
                raise TypeError(f"objectives must be a numpy array, got {type(self.objectives).__name__}")
            if self.objectives.ndim != 2:
                raise ValueError(f"objectives must be 2D, got shape {self.objectives.shape}")
            if self.objectives.shape[0] != n:
                raise ValueError(f"objectives has {self.objectives.shape[0]} individuals, expected {n} to match x")
            object.__setattr__(self, "objectives", self.objectives.copy())

    def __len__(self) -> int:
        """Return the number of individuals in the population.

        Returns:
            Number of individuals (rows in x).
        """
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> IndividualView:
        """Get a read-only view of a single individual.

        Args:
            idx: Index of the individual (supports negative indexing).

        Returns:
            IndividualView containing the data for the specified individual.

        Raises:
            TypeError: If idx is not an integer.
            IndexError: If idx is out of bounds.

        Example:
            >>> pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
            >>> pop[0].x
            array([1., 2.])
            >>> pop[-1].x
            array([3., 4.])
        """
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"indices must be integers, got {type(idx).__name__}")

        n = len(self)
        original_idx = idx
        # Handle negative indexing
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            raise IndexError(f"index {original_idx} is out of bounds for population with {n} individuals")

        return IndividualView(
            x=self.x[idx],
            objectives=self.objectives[idx] if self.objectives is not None else None,
        )

    @property
    def n_individuals(self) -> int:
        """Return the number of individuals in the population.

        Returns:
            Number of individuals (same as len(self)).
        """
        return self.x.shape[0]

    @property
    def n_vars(self) -> int:
        """Return the number of decision variables per individual.

        Returns:
            Number of decision variables (columns in x).
        """
        return self.x.shape[1]

    @property
    def n_obj(self) -> int | None:
        """Return the number of objectives, or None if not evaluated.

        Returns:
            Number of objectives (columns in objectives), or None if objectives is None.
        """
        if self.objectives is None:
            return None
        return self.objectives.shape[1]
