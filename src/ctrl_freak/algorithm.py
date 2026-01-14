"""NSGA-II main algorithm implementation.

This module provides the core NSGA-II algorithm:
- survivor_selection: Select N survivors from combined P + Q population
- nsga2: Main optimization loop
"""

from collections.abc import Callable

import numpy as np

from ctrl_freak.operators import create_offspring, lift
from ctrl_freak.population import Population
from ctrl_freak.primitives import crowding_distance, non_dominated_sort


def _compute_crowding_for_all_fronts(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Compute crowding distance for all individuals across all fronts.

    Helper function that iterates over each Pareto front and computes
    crowding distance for individuals within that front.

    Args:
        objectives: Objective values for all individuals. Shape (n, n_obj).
        ranks: Pareto front ranks for all individuals. Shape (n,).

    Returns:
        Array of shape (n,) containing crowding distances.
    """
    cd = np.zeros(len(objectives), dtype=np.float64)
    for r in range(int(ranks.max()) + 1):
        mask = ranks == r
        cd[mask] = crowding_distance(objectives[mask])
    return cd


def survivor_selection(pop: Population, n_survivors: int) -> Population:
    """Select survivors using NSGA-II crowded selection.

    Implements the (mu + lambda) selection of NSGA-II:
    1. Compute Pareto ranks for combined population
    2. Fill survivors front-by-front until capacity
    3. For critical front (partial fit), select by highest crowding distance
    4. Recompute rank and crowding distance for survivors

    Args:
        pop: Combined P + Q population with objectives computed (2N individuals).
        n_survivors: Target size (N).

    Returns:
        New Population of size n_survivors with rank and crowding_distance
        computed, ready for next generation's selection.

    Raises:
        ValueError: If population has no objectives or n_survivors is invalid.

    Example:
        >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> obj = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        >>> pop = Population(x=x, objectives=obj)
        >>> survivors = survivor_selection(pop, 2)
        >>> len(survivors)
        2
    """
    if pop.objectives is None:
        raise ValueError("Population must have objectives computed for survivor selection")
    if n_survivors <= 0:
        raise ValueError(f"n_survivors must be positive, got {n_survivors}")
    if n_survivors > len(pop):
        raise ValueError(f"n_survivors ({n_survivors}) cannot exceed population size ({len(pop)})")

    # Compute ranks for combined population
    ranks = non_dominated_sort(pop.objectives)

    selected: list[int] = []
    current_rank = 0

    while len(selected) < n_survivors:
        # Get indices of individuals in current front
        front_idx = np.where(ranks == current_rank)[0]

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

    selected_arr = np.array(selected)

    # Extract selected individuals
    new_x = pop.x[selected_arr]
    new_obj = pop.objectives[selected_arr]

    # Recompute rank and crowding distance for survivors
    new_ranks = non_dominated_sort(new_obj)
    new_cd = _compute_crowding_for_all_fronts(new_obj, new_ranks)

    return Population(x=new_x, objectives=new_obj, rank=new_ranks, crowding_distance=new_cd)


def nsga2(
    init: Callable[[np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    pop_size: int,
    n_generations: int,
    seed: int | None = None,
    callback: Callable[[Population, int], bool] | None = None,
) -> Population:
    """Run NSGA-II multi-objective optimization.

    Implements the NSGA-II algorithm for multi-objective optimization with:
    - Binary tournament selection using crowded comparison
    - User-defined crossover and mutation operators
    - (mu + lambda) survivor selection with Pareto ranking

    Args:
        init: Initialize one individual.
            Signature: (rng,) -> (n_vars,)
        evaluate: Evaluate one individual.
            Signature: (n_vars,) -> (n_obj,)
        crossover: Cross two parents to produce one child.
            Signature: (n_vars,), (n_vars,) -> (n_vars,)
        mutate: Mutate one individual.
            Signature: (n_vars,) -> (n_vars,)
        pop_size: Population size N.
        n_generations: Number of generations to run.
        seed: Random seed for reproducibility. If None, uses system entropy.
        callback: Optional callback called each generation.
            Signature: (pop, gen) -> stop?
            If callback returns True, optimization stops early.

    Returns:
        Final population with rank and crowding_distance computed.
        Pareto-optimal solutions can be extracted via: pop.x[pop.rank == 0]

    Raises:
        ValueError: If pop_size or n_generations is not positive.

    Example:
        >>> def init(rng):
        ...     return rng.uniform(0, 1, size=3)
        >>> def evaluate(x):
        ...     return np.array([x.sum(), (1 - x).sum()])
        >>> def crossover(p1, p2):
        ...     return (p1 + p2) / 2
        >>> def mutate(x):
        ...     return np.clip(x + 0.01, 0, 1)
        >>> result = nsga2(init, evaluate, crossover, mutate, pop_size=10, n_generations=5, seed=42)
        >>> len(result)
        10
    """
    if pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {pop_size}")
    if n_generations < 0:
        raise ValueError(f"n_generations must be non-negative, got {n_generations}")

    rng = np.random.default_rng(seed)

    # Initialize population
    init_x = np.stack([init(rng) for _ in range(pop_size)])
    init_obj = lift(evaluate)(init_x)

    # Compute initial ranks and crowding distances
    ranks = non_dominated_sort(init_obj)
    cd = _compute_crowding_for_all_fronts(init_obj, ranks)

    pop = Population(x=init_x, objectives=init_obj, rank=ranks, crowding_distance=cd)

    # Main evolutionary loop
    for gen in range(n_generations):
        # Check callback for early stopping
        if callback is not None and callback(pop, gen):
            break

        # Create offspring
        offspring_x = create_offspring(pop, pop_size, crossover, mutate, rng)
        offspring_obj = lift(evaluate)(offspring_x)

        # Combine parent and offspring populations
        combined = Population(
            x=np.concatenate([pop.x, offspring_x]),
            objectives=np.concatenate([pop.objectives, offspring_obj]),
            rank=None,
            crowding_distance=None,
        )

        # Select survivors for next generation
        pop = survivor_selection(combined, pop_size)

    return pop
