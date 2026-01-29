"""Refactored NSGA-II algorithm with pluggable selection and survival strategies.

This module provides the refactored nsga2() function that uses the registry system
for flexible selection and survival strategy configuration. The function maintains
a simple functional API while enabling strategy customization through either
string-based registry lookups or direct callable injection.

Key Features:
- Pluggable parent selection (default: "crowded" tournament)
- Pluggable survivor selection (default: "nsga2" non-dominated sorting)
- Returns NSGA2Result with full algorithm state (rank, crowding distance)
- Updated callback signature that receives NSGA2Result instead of Population
- Thread-safe random number generation with optional seeding

Example:
    >>> from ctrl_freak.algorithms.nsga2 import nsga2
    >>>
    >>> def init(rng):
    ...     return rng.uniform(0, 1, size=3)
    >>>
    >>> def evaluate(x):
    ...     return np.array([x.sum(), (1 - x).sum()])
    >>>
    >>> def crossover(p1, p2):
    ...     return (p1 + p2) / 2
    >>>
    >>> def mutate(x):
    ...     return np.clip(x + 0.01 * rng.standard_normal(len(x)), 0, 1)
    >>>
    >>> # Using default strategies (string-based)
    >>> result = nsga2(
    ...     init=init,
    ...     evaluate=evaluate,
    ...     crossover=crossover,
    ...     mutate=mutate,
    ...     pop_size=100,
    ...     n_generations=50,
    ...     seed=42
    ... )
    >>>
    >>> # Access Pareto front
    >>> pareto_front = result.pareto_front
    >>> print(f"Found {len(pareto_front)} solutions on Pareto front")
    >>>
    >>> # Using custom strategies (string-based)
    >>> result = nsga2(
    ...     init=init,
    ...     evaluate=evaluate,
    ...     crossover=crossover,
    ...     mutate=mutate,
    ...     pop_size=100,
    ...     n_generations=50,
    ...     select="crowded",
    ...     survive="truncation",
    ...     seed=42
    ... )
    >>>
    >>> # Using callable strategies (advanced)
    >>> from ctrl_freak.selection.crowded import crowded_tournament
    >>> from ctrl_freak.survival.nsga2 import nsga2_survival
    >>>
    >>> custom_selector = crowded_tournament(tournament_size=3)
    >>> custom_survivor = nsga2_survival()
    >>>
    >>> result = nsga2(
    ...     init=init,
    ...     evaluate=evaluate,
    ...     crossover=crossover,
    ...     mutate=mutate,
    ...     pop_size=100,
    ...     n_generations=50,
    ...     select=custom_selector,
    ...     survive=custom_survivor,
    ...     seed=42
    ... )
"""

from collections.abc import Callable

import numpy as np

# Import selection and survival modules to trigger strategy registration
import ctrl_freak.selection  # noqa: F401
import ctrl_freak.survival  # noqa: F401
from ctrl_freak.operators import lift, lift_parallel
from ctrl_freak.population import Population
from ctrl_freak.protocols import ParentSelector, SurvivorSelector
from ctrl_freak.registry import SelectionRegistry, SurvivalRegistry
from ctrl_freak.results import NSGA2Result


def nsga2(
    init: Callable[[np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    pop_size: int,
    n_generations: int,
    seed: int | None = None,
    callback: Callable[[NSGA2Result, int], bool] | None = None,
    select: str | ParentSelector = "crowded",
    survive: str | SurvivorSelector = "nsga2",
    n_workers: int = 1,
) -> NSGA2Result:
    """Run NSGA-II multi-objective optimization with pluggable strategies.

    This is the refactored NSGA-II implementation that uses the registry system
    for flexible selection and survival strategy configuration. The function
    maintains a simple functional API while enabling strategy customization.

    Args:
        init: Initialize one individual.
            Signature: (rng,) -> (n_vars,)
        evaluate: Evaluate one individual.
            Signature: (n_vars,) -> (n_obj,)
        crossover: Cross two parents to produce one child.
            Signature: (n_vars,), (n_vars,) -> (n_vars,)
        mutate: Mutate one individual.
            Signature: (n_vars,) -> (n_vars,)
        pop_size: Population size N. Must be even for proper parent pairing.
        n_generations: Number of generations to run.
        seed: Random seed for reproducibility. If None, uses system entropy.
        callback: Optional callback called at the start of each generation.
            Signature: (result: NSGA2Result, generation: int) -> bool
            If callback returns True, optimization stops early.
            Note: This is a breaking change from the old callback signature
            which received (Population, int).
        select: Parent selection strategy. Can be:
            - String: Name of registered strategy (e.g., "crowded", "roulette")
            - ParentSelector: Direct callable following the ParentSelector protocol
        survive: Survivor selection strategy. Can be:
            - String: Name of registered strategy (e.g., "nsga2", "truncation", "elitist")
            - SurvivorSelector: Direct callable following the SurvivorSelector protocol
        n_workers: Number of parallel workers for evaluation. Use 1 for sequential
            execution (default), -1 for all CPU cores, or any positive integer.
            Note: evaluate function must be picklable for parallel execution.

    Returns:
        NSGA2Result containing:
        - population: Final population with x and objectives
        - rank: Pareto ranks for each individual (0 = Pareto front)
        - crowding_distance: Crowding distances for diversity
        - generations: Number of generations completed
        - evaluations: Total number of function evaluations

    Raises:
        ValueError: If pop_size is not positive, if pop_size is odd (required for
            parent pairing), if n_generations is negative, or if n_workers is
            invalid (must be positive or -1).
        KeyError: If string strategy names are not found in registries.

    Algorithm Flow:
        1. Resolve selection and survival strategies from registry or use callables
        2. Initialize population and evaluate
        3. Apply survival strategy to get initial rank and crowding distance
        4. For each generation:
           a. Call callback with current NSGA2Result (early stopping if returns True)
           b. Select parents using parent_selector with current state
           c. Create offspring via crossover and mutation
           d. Evaluate offspring population
           e. Combine parents + offspring
           f. Apply survival selection to get survivors and updated state
           g. Update population and state for next generation
        5. Return final NSGA2Result with full state

    Example with string-based strategies:
        >>> def init(rng):
        ...     return rng.uniform(0, 1, size=3)
        >>>
        >>> def evaluate(x):
        ...     return np.array([x.sum(), (1 - x).sum()])
        >>>
        >>> def crossover(p1, p2):
        ...     return (p1 + p2) / 2
        >>>
        >>> def mutate(x):
        ...     return np.clip(x + 0.01, 0, 1)
        >>>
        >>> result = nsga2(
        ...     init=init,
        ...     evaluate=evaluate,
        ...     crossover=crossover,
        ...     mutate=mutate,
        ...     pop_size=100,
        ...     n_generations=50,
        ...     seed=42,
        ...     select="crowded",
        ...     survive="nsga2"
        ... )
        >>>
        >>> # Extract Pareto front
        >>> front = result.pareto_front
        >>> print(f"Pareto front has {len(front)} solutions")

    Example with callable strategies:
        >>> from ctrl_freak.selection.crowded import crowded_tournament
        >>> from ctrl_freak.survival.nsga2 import nsga2_survival
        >>>
        >>> # Create custom configured strategies
        >>> my_selector = crowded_tournament(tournament_size=3)
        >>> my_survivor = nsga2_survival()
        >>>
        >>> result = nsga2(
        ...     init=init,
        ...     evaluate=evaluate,
        ...     crossover=crossover,
        ...     mutate=mutate,
        ...     pop_size=100,
        ...     n_generations=50,
        ...     select=my_selector,
        ...     survive=my_survivor,
        ...     seed=42
        ... )

    Example with callback for early stopping:
        >>> def early_stop(result: NSGA2Result, gen: int) -> bool:
        ...     # Stop if we have 10 solutions on Pareto front
        ...     pareto_count = np.sum(result.rank == 0)
        ...     if pareto_count >= 10:
        ...         print(f"Early stopping at generation {gen}")
        ...         return True
        ...     return False
        >>>
        >>> result = nsga2(
        ...     init=init,
        ...     evaluate=evaluate,
        ...     crossover=crossover,
        ...     mutate=mutate,
        ...     pop_size=100,
        ...     n_generations=1000,
        ...     callback=early_stop,
        ...     seed=42
        ... )
    """
    # Validate inputs
    if pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {pop_size}")
    if pop_size % 2 != 0:
        raise ValueError(f"pop_size must be even for proper parent pairing, got {pop_size}")
    if n_generations < 0:
        raise ValueError(f"n_generations must be non-negative, got {n_generations}")
    if n_workers < 1 and n_workers != -1:
        raise ValueError(f"n_workers must be positive or -1 (all cores), got {n_workers}")

    # Resolve selection and survival strategies
    parent_selector = SelectionRegistry.get(select) if isinstance(select, str) else select

    survivor_selector = SurvivalRegistry.get(survive) if isinstance(survive, str) else survive

    # Create evaluator (parallel or sequential)
    lifted_evaluate = lift_parallel(evaluate, n_workers) if n_workers != 1 else lift(evaluate)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Initialize population
    init_x = np.stack([init(rng) for _ in range(pop_size)])
    init_obj = lifted_evaluate(init_x)
    pop = Population(x=init_x, objectives=init_obj)

    # Compute initial state via survival strategy
    # This gives us initial rank and crowding distance
    np.arange(pop_size)
    _, state = survivor_selector(pop, pop_size)

    # Track evaluations: initial population
    total_evaluations = pop_size
    generations_completed = 0

    # Main evolutionary loop
    for gen in range(n_generations):
        # Create current NSGA2Result for callback
        current_result = NSGA2Result(
            population=pop,
            rank=state["rank"],
            crowding_distance=state["crowding_distance"],
            generations=generations_completed,
            evaluations=total_evaluations,
        )

        # Check callback for early stopping
        if callback is not None and callback(current_result, gen):
            break

        # Select parents using current state
        parent_indices = parent_selector(pop, pop_size, rng, **state)

        # Create offspring via crossover and mutation
        offspring_x = np.empty_like(pop.x)
        for i in range(0, pop_size, 2):
            p1_idx, p2_idx = parent_indices[i], parent_indices[i + 1]
            child1 = crossover(pop.x[p1_idx], pop.x[p2_idx])
            child2 = crossover(pop.x[p2_idx], pop.x[p1_idx])
            offspring_x[i] = mutate(child1)
            offspring_x[i + 1] = mutate(child2)

        # Evaluate offspring
        offspring_obj = lifted_evaluate(offspring_x)
        total_evaluations += pop_size

        # Combine parent and offspring populations
        assert pop.objectives is not None  # Guaranteed by initialization
        combined = Population(
            x=np.concatenate([pop.x, offspring_x]),
            objectives=np.concatenate([pop.objectives, offspring_obj]),
        )

        # Select survivors for next generation
        survivor_indices, state = survivor_selector(combined, pop_size)

        # Update population
        pop = Population(
            x=combined.x[survivor_indices],
            objectives=combined.objectives[survivor_indices],
        )

        generations_completed += 1

    # Return final result
    final_result = NSGA2Result(
        population=pop,
        rank=state["rank"],
        crowding_distance=state["crowding_distance"],
        generations=generations_completed,
        evaluations=total_evaluations,
    )

    return final_result
