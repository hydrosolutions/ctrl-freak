"""Standard genetic algorithm with pluggable selection and survival strategies.

This module provides the standard single-objective genetic algorithm (GA) that uses
the registry system for flexible selection and survival strategy configuration. The
function maintains a simple functional API while enabling strategy customization
through either string-based registry lookups or direct callable injection.

Key Features:
- Pluggable parent selection (default: "tournament")
- Pluggable survivor selection (default: "elitist")
- Returns GAResult with fitness values and best individual
- Callback signature receives GAResult instead of Population
- Thread-safe random number generation with optional seeding

Example:
    >>> from ctrl_freak.algorithms.ga import ga
    >>>
    >>> def init(rng):
    ...     return rng.uniform(0, 1, size=3)
    >>>
    >>> def evaluate(x):
    ...     return x.sum()  # Returns scalar for single-objective
    >>>
    >>> def crossover(p1, p2):
    ...     return (p1 + p2) / 2
    >>>
    >>> def mutate(x):
    ...     return np.clip(x + 0.01 * rng.standard_normal(len(x)), 0, 1)
    >>>
    >>> # Using default strategies (string-based)
    >>> result = ga(
    ...     init=init,
    ...     evaluate=evaluate,
    ...     crossover=crossover,
    ...     mutate=mutate,
    ...     pop_size=100,
    ...     n_generations=50,
    ...     seed=42
    ... )
    >>>
    >>> # Access best solution
    >>> best_x, best_fitness = result.best
    >>> print(f"Best solution: {best_x} with fitness {best_fitness}")
    >>>
    >>> # Using custom strategies (string-based)
    >>> result = ga(
    ...     init=init,
    ...     evaluate=evaluate,
    ...     crossover=crossover,
    ...     mutate=mutate,
    ...     pop_size=100,
    ...     n_generations=50,
    ...     select="tournament",
    ...     survive="elitist",
    ...     seed=42
    ... )
    >>>
    >>> # Using callable strategies (advanced)
    >>> from ctrl_freak.selection.tournament import fitness_tournament
    >>> from ctrl_freak.survival.elitist import elitist_survival
    >>>
    >>> custom_selector = fitness_tournament(tournament_size=3)
    >>> custom_survivor = elitist_survival(elite_count=2)
    >>>
    >>> result = ga(
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
from ctrl_freak.population import Population
from ctrl_freak.protocols import ParentSelector, SurvivorSelector
from ctrl_freak.registry import SelectionRegistry, SurvivalRegistry
from ctrl_freak.results import GAResult


def ga(
    init: Callable[[np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], float],
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    pop_size: int,
    n_generations: int,
    seed: int | None = None,
    callback: Callable[[GAResult, int], bool] | None = None,
    select: str | ParentSelector = "tournament",
    survive: str | SurvivorSelector = "elitist",
) -> GAResult:
    """Run standard single-objective genetic algorithm with pluggable strategies.

    This is the standard GA implementation that uses the registry system for
    flexible selection and survival strategy configuration. The function maintains
    a simple functional API while enabling strategy customization.

    Args:
        init: Initialize one individual.
            Signature: (rng,) -> (n_vars,)
        evaluate: Evaluate one individual, returning scalar fitness value.
            Signature: (n_vars,) -> float
            Note: For minimization problems, lower values are better.
        crossover: Cross two parents to produce one child.
            Signature: (n_vars,), (n_vars,) -> (n_vars,)
        mutate: Mutate one individual.
            Signature: (n_vars,) -> (n_vars,)
        pop_size: Population size N. Must be even for proper parent pairing.
        n_generations: Number of generations to run.
        seed: Random seed for reproducibility. If None, uses system entropy.
        callback: Optional callback called at the start of each generation.
            Signature: (result: GAResult, generation: int) -> bool
            If callback returns True, optimization stops early.
        select: Parent selection strategy. Can be:
            - String: Name of registered strategy (e.g., "tournament", "roulette")
            - ParentSelector: Direct callable following the ParentSelector protocol
        survive: Survivor selection strategy. Can be:
            - String: Name of registered strategy (e.g., "elitist", "truncation")
            - SurvivorSelector: Direct callable following the SurvivorSelector protocol

    Returns:
        GAResult containing:
        - population: Final population with x and objectives
        - fitness: Fitness values for each individual (same as objectives[:, 0])
        - best_idx: Index of the best individual (lowest fitness)
        - generations: Number of generations completed
        - evaluations: Total number of function evaluations

    Raises:
        ValueError: If pop_size is not positive, if pop_size is odd (required for
            parent pairing), or if n_generations is negative.
        KeyError: If string strategy names are not found in registries.

    Algorithm Flow:
        1. Resolve selection and survival strategies from registry or use callables
        2. Initialize population and evaluate
        3. Apply survival strategy to get initial fitness state
        4. For each generation:
           a. Call callback with current GAResult (early stopping if returns True)
           b. Select parents using parent_selector with current state
           c. Create offspring via crossover and mutation
           d. Evaluate offspring population
           e. Combine parents + offspring
           f. Apply survival selection to get survivors and updated state
           g. Update population and state for next generation
        5. Return final GAResult with best individual

    Example with string-based strategies:
        >>> def init(rng):
        ...     return rng.uniform(0, 1, size=3)
        >>>
        >>> def evaluate(x):
        ...     return x.sum()  # Scalar fitness
        >>>
        >>> def crossover(p1, p2):
        ...     return (p1 + p2) / 2
        >>>
        >>> def mutate(x):
        ...     return np.clip(x + 0.01, 0, 1)
        >>>
        >>> result = ga(
        ...     init=init,
        ...     evaluate=evaluate,
        ...     crossover=crossover,
        ...     mutate=mutate,
        ...     pop_size=100,
        ...     n_generations=50,
        ...     seed=42,
        ...     select="tournament",
        ...     survive="elitist"
        ... )
        >>>
        >>> # Extract best solution
        >>> best_x, best_fitness = result.best
        >>> print(f"Best: {best_x} with fitness {best_fitness}")

    Example with callable strategies:
        >>> from ctrl_freak.selection.tournament import fitness_tournament
        >>> from ctrl_freak.survival.elitist import elitist_survival
        >>>
        >>> # Create custom configured strategies
        >>> my_selector = fitness_tournament(tournament_size=5)
        >>> my_survivor = elitist_survival(elite_count=2)
        >>>
        >>> result = ga(
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
        >>> def early_stop(result: GAResult, gen: int) -> bool:
        ...     # Stop if best fitness is below threshold
        ...     _, best_fitness = result.best
        ...     if best_fitness < 1e-6:
        ...         print(f"Early stopping at generation {gen}")
        ...         return True
        ...     return False
        >>>
        >>> result = ga(
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

    # Resolve selection and survival strategies
    parent_selector = SelectionRegistry.get(select) if isinstance(select, str) else select
    survivor_selector = SurvivalRegistry.get(survive) if isinstance(survive, str) else survive

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Initialize population
    init_x = np.stack([init(rng) for _ in range(pop_size)])
    # Apply per-individual evaluation (lift expects array->array, so we evaluate manually)
    init_obj = np.array([evaluate(init_x[i]) for i in range(pop_size)])

    # Ensure objectives are shape (n, 1) for single-objective
    if init_obj.ndim == 1:
        init_obj = init_obj.reshape(-1, 1)

    pop = Population(x=init_x, objectives=init_obj)

    # Compute initial fitness state
    # Extract fitness from single-objective population
    assert pop.objectives is not None  # Guaranteed by initialization above
    state: dict[str, np.ndarray] = {"fitness": pop.objectives[:, 0].copy()}

    # Track evaluations: initial population
    total_evaluations = pop_size
    generations_completed = 0

    # Main evolutionary loop
    for gen in range(n_generations):
        # Extract fitness from state for current GAResult
        fitness = state["fitness"]
        best_idx = int(np.argmin(fitness))

        # Create current GAResult for callback
        current_result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=best_idx,
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
        offspring_obj = np.array([evaluate(offspring_x[i]) for i in range(pop_size)])
        # Ensure shape (n, 1)
        if offspring_obj.ndim == 1:
            offspring_obj = offspring_obj.reshape(-1, 1)
        total_evaluations += pop_size

        # Combine parent and offspring populations
        assert pop.objectives is not None  # Guaranteed by initialization
        combined = Population(
            x=np.concatenate([pop.x, offspring_x]),
            objectives=np.concatenate([pop.objectives, offspring_obj]),
        )

        # Select survivors for next generation
        # For elitist survival, we need to pass parent_size kwarg
        # Note: parent_size is int but protocol expects np.ndarray; implementations accept both
        survivor_indices, state = survivor_selector(
            combined,
            pop_size,
            parent_size=pop_size,  # type: ignore[arg-type]
        )

        # Update population
        assert combined.objectives is not None  # Guaranteed by Population construction above
        pop = Population(
            x=combined.x[survivor_indices],
            objectives=combined.objectives[survivor_indices],
        )

        generations_completed += 1

    # Return final result
    fitness = state["fitness"]
    best_idx = int(np.argmin(fitness))

    final_result = GAResult(
        population=pop,
        fitness=fitness,
        best_idx=best_idx,
        generations=generations_completed,
        evaluations=total_evaluations,
    )

    return final_result
