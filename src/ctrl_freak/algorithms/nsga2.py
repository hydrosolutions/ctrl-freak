"""NSGA-II algorithm with pluggable strategies.

Examples
--------
>>> import numpy as np
>>> from ctrl_freak.algorithms.nsga2 import nsga2
>>> def init(rng):
...     return rng.uniform(0.0, 1.0, size=3)
>>> def evaluate(x):
...     return np.array([np.sum(x), np.sum(1.0 - x)])
>>> result = nsga2(
...     init=init,
...     evaluate=evaluate,
...     crossover=lambda p1, p2: (p1 + p2) / 2,
...     mutate=lambda x: np.clip(x + 0.01, 0.0, 1.0),
...     pop_size=10,
...     n_generations=2,
...     seed=42,
... )
>>> result.population.x.shape
(10, 3)
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
    """Run NSGA-II multi-objective optimization.

    Parameters
    ----------
    init
        Callable that initializes one individual from a random generator.
    evaluate
        Callable that evaluates one individual and returns objective values.
    crossover
        Callable that crosses two parents to produce one child.
    mutate
        Callable that mutates one individual.
    pop_size
        Population size. Must be positive and even.
    n_generations
        Number of generations to run.
    seed
        Master random seed. If ``None``, system entropy is used.
    callback
        Optional callback called before each generation. Return ``True`` to stop.
    select
        Parent selection strategy name or callable.
    survive
        Survivor selection strategy name or callable.
    n_workers
        Number of workers for objective evaluation. Parallel evaluation is
        deterministic only when ``evaluate`` is pure.

    Returns
    -------
    NSGA2Result
        Final population, ranks, crowding distances, generation count, and
        evaluation count.

    Raises
    ------
    ValueError
        If size, generation, or worker arguments are invalid.
    KeyError
        If a named strategy is not registered.

    Examples
    --------
    >>> import numpy as np
    >>> from ctrl_freak.algorithms.nsga2 import nsga2
    >>> def init(rng):
    ...     return rng.uniform(0.0, 1.0, size=2)
    >>> def evaluate(x):
    ...     return np.array([np.sum(x), np.sum(1.0 - x)])
    >>> result = nsga2(
    ...     init=init,
    ...     evaluate=evaluate,
    ...     crossover=lambda p1, p2: (p1 + p2) / 2,
    ...     mutate=lambda x: x.copy(),
    ...     pop_size=10,
    ...     n_generations=2,
    ...     seed=1,
    ... )
    >>> result.generations
    2
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

    # Derive independent per-phase RNG streams from the single master seed so one seed
    # reproduces init + parent selection + crossover + mutation bit-identically.
    # Child order is the reproducibility contract: [init, select, crossover, mutate]. Never reorder.
    init_ss, select_ss, crossover_ss, mutate_ss = np.random.SeedSequence(seed).spawn(4)
    rng = np.random.default_rng(init_ss)
    select_rng = np.random.default_rng(select_ss)
    set_crossover_rng = getattr(crossover, "set_rng", None)
    if callable(set_crossover_rng):
        set_crossover_rng(np.random.default_rng(crossover_ss))
    set_mutate_rng = getattr(mutate, "set_rng", None)
    if callable(set_mutate_rng):
        set_mutate_rng(np.random.default_rng(mutate_ss))

    # Initialize population
    init_x = np.stack([init(rng) for _ in range(pop_size)])
    init_obj = lifted_evaluate(init_x)
    pop = Population(x=init_x, objectives=init_obj)

    # Compute initial state via survival strategy
    # This gives us initial rank and crowding distance
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
        parent_indices = parent_selector(pop, pop_size, select_rng, **state)

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
        assert combined.objectives is not None  # Guaranteed by Population construction above
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
