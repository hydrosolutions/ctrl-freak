"""Fair multi-objective comparison harness: ctrl-freak vs pymoo vs DEAP NSGA-II.

This module runs NSGA-II from three libraries -- ctrl-freak's :func:`nsga2`,
``pymoo``'s ``NSGA2``, and a hand-rolled DEAP NSGA-II loop -- on a shared
:class:`~benchmarks.problems.multi_objective.MOProblem` with **aligned operators
and an identical evaluation budget**, returning *raw* output only: the extracted
non-dominated front (objectives + decision vectors), a per-generation
non-dominated convergence history, and the objective-evaluation count. It computes
**no metrics** (IGD+/GD/HV live in :mod:`benchmarks.metrics` and are applied by the
s5 runner); the harness exists so that any residual library difference is an
implementation difference, not a configuration difference.

Alignment (all read from :mod:`benchmarks.config`):

* **Crossover -- IDENTICAL across all three.** ctrl-freak's exact single-child,
  every-eligible-variable SBX (``eta = SBX_ETA = 15.0``) is ported via
  :mod:`benchmarks.harness.operators` into a custom pymoo ``Crossover`` and a
  custom DEAP mate (both call the same ``ctrl_freak_sbx_child`` core twice per
  mating); ctrl-freak's ``nsga2`` uses its native ``sbx_crossover`` (the same
  operator). Only the per-framework RNG source differs. This is the human-gated
  "Option C" decision -- pymoo's stock ``SBX`` (``prob_var`` 0.5) and DEAP's
  ``cxSimulatedBinaryBounded`` (per-variable 0.5) recombine only ~half the
  variables, which put ctrl-freak ~70% above both baselines on convex-ZDT IGD+ (a
  config difference that would fail s5's equivalence test). Porting eliminates it,
  isolating the NSGA-II loop (selection, survival, non-dominated sort, crowding) as
  the sole subject of the comparison.
* **Mutation** -- polynomial mutation, ``eta = PM_ETA (20.0)``, applied to *every*
  offspring at a *per-variable* rate ``1 / n_var``. For ``pymoo`` this requires
  ``PM(prob=1.0, prob_var=1/n_var)``: pymoo's ``prob`` is the per-*individual*
  application probability and ``prob_var`` is the per-*variable* rate. (The legacy
  ``PM(prob=1/n_var)`` mutates only ~3% of individuals and silently cripples
  pymoo -- see the ZDT3 note below.)
* **Parent selection** -- crowded binary tournament (rank, then crowding):
  ctrl-freak ``crowded``, pymoo NSGA2 default, DEAP ``selTournamentDCD``.
* **Survival** -- NSGA-II (non-dominated sort + crowding), (mu + lambda):
  ctrl-freak ``nsga2``, pymoo default ``RankAndCrowdingSurvival``, DEAP
  ``selNSGA2``.
* **Budget** -- ``pop_size`` initial evaluations plus ``n_generations`` offspring
  batches of ``pop_size``: ``pop_size * (1 + n_generations)`` evaluations,
  identical across all three libraries (pymoo is driven via its object-oriented
  interface with ``NoTermination`` so the count is exact, not off by one).

Front extraction uses a single shared extractor (:func:`_non_dominated_mask`,
ctrl-freak's ``non_dominated_sort == 0``) applied to each library's full final
population, so every adapter reports identical "non-dominated front" semantics
(verified to coincide with each library's native extraction).

ZDT3 "visible scatter" root cause (N6)
--------------------------------------
The legacy benchmark README captioned a ZDT3 figure "Pymoo shows visible scatter".
That was **not** a plotting/extraction artifact (at the full budget every
library's final population is already 100% non-dominated) and **not** operator
unfairness in the algorithms. It was the legacy harness's silent mutation
misalignment: ``PM(prob=1/n_var)`` mutated only ~3% of individuals (~0.03 gene
flips each) versus ctrl-freak/DEAP's ~1.0 -- a ~34x under-mutation that slowed
pymoo's convergence and left a few non-dominated points above ZDT3's disconnected
front. With the correct ``PM(prob=1.0, prob_var=1/n_var)`` used here the scatter
disappears, and with the ported SBX above all three libraries converge to nearly
identical fronts on the standard ZDT/DTLZ2 problems (e.g. ZDT1 IGD+ 0.0060 /
0.0060 / 0.0059, ZDT3 0.0032 / 0.0033 / 0.0030 for ctrl-freak / pymoo / DEAP).

Examples
--------
>>> import numpy as np
>>> from benchmarks.problems.multi_objective import MO_PROBLEMS
>>> res = run_mo(MO_PROBLEMS["zdt1"], "ctrl-freak", seed=0, pop_size=8, n_generations=2)
>>> res.front_objectives.shape[1]
2
>>> res.n_evaluations
24
>>> len(res.history) == res.n_generations + 1
True
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from benchmarks import config
from benchmarks.harness.operators import deap_ctrl_freak_sbx, make_pymoo_ctrl_freak_sbx
from benchmarks.problems.multi_objective import MOProblem
from ctrl_freak.primitives.pareto import non_dominated_sort

LIBRARIES: tuple[str, ...] = ("ctrl-freak", "pymoo", "deap")
"""The three NSGA-II implementations the harness can run."""


@dataclass(frozen=True)
class MORunResult:
    """Raw, metric-free result of one NSGA-II run.

    Attributes
    ----------
    library : str
        One of :data:`LIBRARIES`.
    problem : str
        The :class:`~benchmarks.problems.multi_objective.MOProblem` name.
    seed : int
        Master seed applied to the run.
    pop_size : int
        Population size used.
    n_generations : int
        Number of offspring generations (excludes the initial population).
    n_evaluations : int
        Objective-function evaluation count, ``pop_size * (1 + n_generations)``.
    front_objectives : numpy.ndarray
        Non-dominated front objectives, shape ``(n_nd, n_obj)``.
    front_x : numpy.ndarray
        Decision vectors of the non-dominated front, shape ``(n_nd, n_var)``.
    history : list of numpy.ndarray
        Per-generation non-dominated front objectives; length
        ``n_generations + 1`` (snapshots after ``0, 1, ..., n_generations``
        completed generations). Each element has shape ``(k_g, n_obj)``. Raw
        objectives only -- the s5 runner turns these into convergence metrics.

    Examples
    --------
    >>> import numpy as np
    >>> r = MORunResult(
    ...     library="ctrl-freak", problem="zdt1", seed=0, pop_size=4,
    ...     n_generations=1, n_evaluations=8,
    ...     front_objectives=np.array([[0.0, 1.0], [1.0, 0.0]]),
    ...     front_x=np.zeros((2, 3)),
    ...     history=[np.array([[0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])],
    ... )
    >>> r.n_evaluations, r.front_objectives.shape
    (8, (2, 2))
    """

    library: str
    problem: str
    seed: int
    pop_size: int
    n_generations: int
    n_evaluations: int
    front_objectives: np.ndarray
    front_x: np.ndarray
    history: list[np.ndarray]


def _non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated (rank-0) rows of ``objectives``.

    Uses ctrl-freak's :func:`~ctrl_freak.primitives.pareto.non_dominated_sort`
    so every adapter extracts its reported front with identical semantics.

    Parameters
    ----------
    objectives : numpy.ndarray
        Objective values, shape ``(n, n_obj)``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n,)``; ``True`` where the row is non-dominated.

    Examples
    --------
    >>> objs = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    >>> _non_dominated_mask(objs).tolist()
    [True, True, False]
    """
    if objectives.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return non_dominated_sort(objectives) == 0


def _extract_front(objectives: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the non-dominated ``(objectives, x)`` rows of a population.

    Parameters
    ----------
    objectives : numpy.ndarray
        Population objectives, shape ``(n, n_obj)``.
    x : numpy.ndarray
        Population decision vectors, shape ``(n, n_var)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(front_objectives, front_x)`` restricted to non-dominated rows.

    Examples
    --------
    >>> objs = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    >>> xs = np.array([[0.0], [1.0], [2.0]])
    >>> fo, fx = _extract_front(objs, xs)
    >>> fo.shape, fx.shape
    ((2, 2), (2, 1))
    """
    mask = _non_dominated_mask(objectives)
    return objectives[mask], x[mask]


def _run_ctrl_freak(
    problem: MOProblem, seed: int, pop_size: int, n_generations: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    """Run ctrl-freak ``nsga2``; return ``(front_obj, front_x, history, n_eval)``."""
    from ctrl_freak import nsga2, polynomial_mutation, sbx_crossover

    xl, xu = problem.bounds
    bounds = (xl, xu)
    crossover = sbx_crossover(eta=config.SBX_ETA, bounds=bounds, seed=seed)
    mutate = polynomial_mutation(eta=config.PM_ETA, prob=config.mutation_prob(problem.n_var), bounds=bounds, seed=seed)

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(xl, xu)

    history: list[np.ndarray] = []

    def callback(result, gen: int) -> bool:
        objectives = result.population.objectives
        history.append(objectives[_non_dominated_mask(objectives)].copy())
        return False

    result = nsga2(
        init=init,
        evaluate=problem.func,
        crossover=crossover,
        mutate=mutate,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        callback=callback,
        select="crowded",
        survive="nsga2",
    )
    final_obj = result.population.objectives
    history.append(final_obj[_non_dominated_mask(final_obj)].copy())
    front_obj, front_x = _extract_front(final_obj, result.population.x)
    return front_obj, front_x, history, result.evaluations


def _run_pymoo(
    problem: MOProblem, seed: int, pop_size: int, n_generations: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    """Run pymoo ``NSGA2`` (ported SBX); return ``(front_obj, front_x, history, n_eval)``."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    xl, xu = problem.bounds
    func = problem.func
    n_var = problem.n_var
    n_obj = problem.n_obj

    class _Problem(PymooProblem):
        def __init__(self) -> None:
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs) -> None:
            out["F"] = np.array([func(xi) for xi in X])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=make_pymoo_ctrl_freak_sbx(eta=config.SBX_ETA, prob=config.CROSSOVER_PROB),
        mutation=PM(eta=config.PM_ETA, prob=1.0, prob_var=config.mutation_prob(n_var)),
        eliminate_duplicates=False,
    )
    algorithm.setup(_Problem(), termination=NoTermination(), seed=seed, verbose=False)

    history: list[np.ndarray] = []
    algorithm.next()  # first step evaluates the initial population
    objectives = algorithm.pop.get("F")
    history.append(objectives[_non_dominated_mask(objectives)].copy())
    for _ in range(n_generations):
        algorithm.next()
        objectives = algorithm.pop.get("F")
        history.append(objectives[_non_dominated_mask(objectives)].copy())

    final_obj = algorithm.pop.get("F")
    final_x = algorithm.pop.get("X")
    front_obj, front_x = _extract_front(final_obj, final_x)
    return front_obj, front_x, history, int(algorithm.evaluator.n_eval)


def _run_deap(
    problem: MOProblem, seed: int, pop_size: int, n_generations: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    """Run a DEAP NSGA-II loop (ported SBX); return ``(front_obj, front_x, history, n_eval)``."""
    import random

    from deap import base, creator, tools

    xl, xu = problem.bounds
    func = problem.func
    n_var = problem.n_var
    n_obj = problem.n_obj

    # Distinct names from the s4a SO harness (FitnessMin / Individual), recreated
    # every call because n_obj varies across problems (ZDT=2, DTLZ2=3) -- an
    # idempotent ``hasattr`` guard would reuse stale weights on the next problem.
    if hasattr(creator, "FitnessMinMO"):
        del creator.FitnessMinMO
    if hasattr(creator, "IndividualMO"):
        del creator.IndividualMO
    creator.create("FitnessMinMO", base.Fitness, weights=tuple([-1.0] * n_obj))
    creator.create("IndividualMO", list, fitness=creator.FitnessMinMO)

    random.seed(seed)
    np.random.seed(seed)
    cx_rng = np.random.default_rng(seed)  # ported SBX is numpy-driven

    toolbox = base.Toolbox()

    def init_individual():
        return creator.IndividualMO(np.random.uniform(xl, xu, size=n_var).tolist())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual) -> tuple[float, ...]:
        return tuple(func(np.asarray(individual)))

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", deap_ctrl_freak_sbx, eta=config.SBX_ETA, low=xl, up=xu, rng=cx_rng)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=config.PM_ETA,
        low=list(xl),
        up=list(xu),
        indpb=config.mutation_prob(n_var),
    )
    toolbox.register("select", tools.selNSGA2)

    def snapshot(pop) -> np.ndarray:
        objectives = np.array([ind.fitness.values for ind in pop])
        return objectives[_non_dominated_mask(objectives)].copy()

    n_eval = 0
    pop = toolbox.population(n=pop_size)
    for ind, fit in zip(pop, map(toolbox.evaluate, pop), strict=True):
        ind.fitness.values = fit
    n_eval += len(pop)
    pop = toolbox.select(pop, len(pop))  # assign crowding distance; no re-evaluation

    history: list[np.ndarray] = [snapshot(pop)]

    for _ in range(n_generations):
        offspring = [toolbox.clone(ind) for ind in tools.selTournamentDCD(pop, len(pop))]
        for i in range(0, len(offspring), 2):
            toolbox.mate(offspring[i], offspring[i + 1])
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid), strict=True):
            ind.fitness.values = fit
        n_eval += len(invalid)
        pop = toolbox.select(pop + offspring, pop_size)
        history.append(snapshot(pop))

    final_obj = np.array([ind.fitness.values for ind in pop])
    final_x = np.array([list(ind) for ind in pop])
    front_obj, front_x = _extract_front(final_obj, final_x)
    return front_obj, front_x, history, n_eval


_ADAPTERS: dict[str, Callable[[MOProblem, int, int, int], tuple]] = {
    "ctrl-freak": _run_ctrl_freak,
    "pymoo": _run_pymoo,
    "deap": _run_deap,
}


def run_mo(
    problem: MOProblem,
    library: str,
    seed: int,
    *,
    pop_size: int = config.MO_POP_SIZE,
    n_generations: int = config.MO_N_GENERATIONS,
) -> MORunResult:
    """Run one NSGA-II optimization and return its raw (metric-free) result.

    Parameters
    ----------
    problem : MOProblem
        The multi-objective problem (objective function, bounds, dimensions).
    library : {"ctrl-freak", "pymoo", "deap"}
        Which NSGA-II implementation to run.
    seed : int
        Master seed. The same integer seeds all three libraries for paired
        comparison (RNG mechanics differ per library, so runs are reproducible
        within a library but not bit-identical across libraries).
    pop_size : int, optional
        Population size (default :data:`benchmarks.config.MO_POP_SIZE` = 100).
        Must be even and a multiple of 4 (DEAP's ``selTournamentDCD``). Small
        values are for smoke tests; the committed sweep uses the default.
    n_generations : int, optional
        Number of offspring generations (default
        :data:`benchmarks.config.MO_N_GENERATIONS` = 250).

    Returns
    -------
    MORunResult
        Raw output: non-dominated front, per-generation history, evaluation count.

    Raises
    ------
    ValueError
        If ``library`` is not one of :data:`LIBRARIES`.

    Examples
    --------
    >>> from benchmarks.problems.multi_objective import MO_PROBLEMS
    >>> from ctrl_freak.primitives.pareto import non_dominated_sort
    >>> res = run_mo(MO_PROBLEMS["zdt1"], "ctrl-freak", seed=0, pop_size=8, n_generations=2)
    >>> res.library, res.problem, res.n_evaluations
    ('ctrl-freak', 'zdt1', 24)
    >>> bool((non_dominated_sort(res.front_objectives) == 0).all())
    True
    """
    if library not in _ADAPTERS:
        raise ValueError(f"library must be one of {LIBRARIES}, got {library!r}")
    front_obj, front_x, history, n_eval = _ADAPTERS[library](problem, seed, pop_size, n_generations)
    return MORunResult(
        library=library,
        problem=problem.name,
        seed=seed,
        pop_size=pop_size,
        n_generations=n_generations,
        n_evaluations=n_eval,
        front_objectives=front_obj,
        front_x=front_x,
        history=history,
    )
