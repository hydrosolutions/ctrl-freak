"""Algorithm-specific result types for genetic algorithms.

This module provides result dataclasses that encapsulate algorithm-specific
metadata along with the final population:

- NSGA2Result: Results from NSGA-II multi-objective optimization
- GAResult: Results from single-objective genetic algorithms

Both classes are immutable (frozen dataclasses) to enforce functional style.
All numpy arrays are copied on construction to ensure immutability.
"""

from dataclasses import dataclass

import numpy as np

from ctrl_freak.population import Population


@dataclass(frozen=True)
class NSGA2Result:
    """Results from NSGA-II multi-objective optimization algorithm.

    This class encapsulates the final population along with algorithm-specific
    metadata like Pareto ranks and crowding distances. All arrays are copied
    on construction to ensure immutability.

    Attributes:
        population: The final population after optimization.
        rank: Pareto rank for each individual, shape (n,). Rank 0 indicates
            individuals on the Pareto front (non-dominated).
        crowding_distance: Crowding distance for each individual, shape (n,).
            Used for diversity preservation. Individuals at extremes have infinite
            crowding distance.
        generations: Number of generations completed during optimization.
        evaluations: Total number of objective function evaluations performed.

    Example:
        >>> x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> obj = np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])
        >>> pop = Population(x=x, objectives=obj)
        >>> rank = np.array([0, 0, 1])
        >>> cd = np.array([np.inf, np.inf, 0.5])
        >>> result = NSGA2Result(
        ...     population=pop,
        ...     rank=rank,
        ...     crowding_distance=cd,
        ...     generations=100,
        ...     evaluations=5000
        ... )
        >>> front = result.pareto_front
        >>> len(front)
        2
    """

    population: Population
    rank: np.ndarray
    crowding_distance: np.ndarray
    generations: int
    evaluations: int

    def __post_init__(self) -> None:
        """Validate shapes and copy arrays for immutability.

        Raises:
            TypeError: If rank or crowding_distance are not numpy arrays.
            ValueError: If array shapes are inconsistent.
        """
        n = len(self.population)

        # Validate rank
        if not isinstance(self.rank, np.ndarray):
            raise TypeError(f"rank must be a numpy array, got {type(self.rank).__name__}")
        if self.rank.ndim != 1:
            raise ValueError(f"rank must be 1D, got shape {self.rank.shape}")
        if self.rank.shape[0] != n:
            raise ValueError(f"rank has {self.rank.shape[0]} elements, expected {n} to match population size")

        # Validate crowding_distance
        if not isinstance(self.crowding_distance, np.ndarray):
            raise TypeError(f"crowding_distance must be a numpy array, got {type(self.crowding_distance).__name__}")
        if self.crowding_distance.ndim != 1:
            raise ValueError(f"crowding_distance must be 1D, got shape {self.crowding_distance.shape}")
        if self.crowding_distance.shape[0] != n:
            raise ValueError(f"crowding_distance has {self.crowding_distance.shape[0]} elements, expected {n} to match population size")

        # Copy arrays for immutability (use object.__setattr__ for frozen dataclass)
        object.__setattr__(self, "rank", self.rank.copy())
        object.__setattr__(self, "crowding_distance", self.crowding_distance.copy())

    @property
    def pareto_front(self) -> Population:
        """Extract the Pareto front (rank-0 individuals) as a new Population.

        Returns:
            A new Population containing only individuals with rank 0.

        Example:
            >>> # Assuming result is an NSGA2Result instance
            >>> front = result.pareto_front
            >>> front.n_individuals  # Number of individuals on Pareto front
        """
        # Find indices of rank-0 individuals
        rank_0_mask = self.rank == 0
        rank_0_indices = np.where(rank_0_mask)[0]

        # Extract rank-0 individuals
        x_front = self.population.x[rank_0_indices]
        objectives_front = (
            self.population.objectives[rank_0_indices]
            if self.population.objectives is not None
            else None
        )

        return Population(x=x_front, objectives=objectives_front)


@dataclass(frozen=True)
class GAResult:
    """Results from single-objective genetic algorithm optimization.

    This class encapsulates the final population along with fitness values
    for each individual. In minimization problems, lower fitness is better.
    All arrays are copied on construction to ensure immutability.

    Attributes:
        population: The final population after optimization.
        fitness: Fitness values for each individual, shape (n,). Lower values
            indicate better fitness for minimization problems.
        best_idx: Index of the best individual (lowest fitness for minimization).
        generations: Number of generations completed during optimization.
        evaluations: Total number of objective function evaluations performed.

    Example:
        >>> x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> pop = Population(x=x)
        >>> fitness = np.array([0.5, 0.3, 0.7])
        >>> result = GAResult(
        ...     population=pop,
        ...     fitness=fitness,
        ...     best_idx=1,
        ...     generations=50,
        ...     evaluations=2500
        ... )
        >>> best_x, best_fitness = result.best
        >>> best_x
        array([3., 4.])
        >>> best_fitness
        0.3
    """

    population: Population
    fitness: np.ndarray
    best_idx: int
    generations: int
    evaluations: int

    def __post_init__(self) -> None:
        """Validate shapes and copy arrays for immutability.

        Raises:
            TypeError: If fitness is not a numpy array or best_idx is not an integer.
            ValueError: If array shapes are inconsistent or best_idx is out of bounds.
        """
        n = len(self.population)

        # Validate fitness
        if not isinstance(self.fitness, np.ndarray):
            raise TypeError(f"fitness must be a numpy array, got {type(self.fitness).__name__}")
        if self.fitness.ndim != 1:
            raise ValueError(f"fitness must be 1D, got shape {self.fitness.shape}")
        if self.fitness.shape[0] != n:
            raise ValueError(f"fitness has {self.fitness.shape[0]} elements, expected {n} to match population size")

        # Validate best_idx
        if not isinstance(self.best_idx, (int, np.integer)):
            raise TypeError(f"best_idx must be an integer, got {type(self.best_idx).__name__}")
        if self.best_idx < 0 or self.best_idx >= n:
            raise ValueError(f"best_idx {self.best_idx} is out of bounds for population with {n} individuals")

        # Copy fitness for immutability (use object.__setattr__ for frozen dataclass)
        object.__setattr__(self, "fitness", self.fitness.copy())

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Extract the best individual and its fitness value.

        Returns:
            Tuple of (x, fitness) where x is the decision variables of the best
            individual (shape (n_vars,)) and fitness is its fitness value.

        Example:
            >>> # Assuming result is a GAResult instance
            >>> best_x, best_fitness = result.best
            >>> print(f"Best solution: {best_x} with fitness {best_fitness}")
        """
        best_x = self.population.x[self.best_idx]
        best_fitness = self.fitness[self.best_idx]
        return (best_x, float(best_fitness))
