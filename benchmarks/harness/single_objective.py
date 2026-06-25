"""Fair single-objective optimization harness: ctrl-freak vs pymoo vs DEAP.

This module runs the **identical** real-coded genetic algorithm across three
libraries so that any difference in their results in the s5 sweep reflects a
ctrl-freak implementation bug, not a configuration artifact. Each adapter uses
the same operators (simulated binary crossover ``eta=15``, polynomial mutation
``eta=20`` with per-variable probability ``1/dim``), the same parent selection
(binary tournament by fitness), the same survival
(``elitist_survival(elite_count=1)`` -- one best parent plus the best
``pop_size - 1`` offspring), the same evaluation budget
(``pop_size + n_generations * pop_size``), and per-run seeding. Only the library
implementation differs.

The harness returns **raw output only** -- best decision vector, best objective,
per-generation convergence history, and the evaluation count -- via
:class:`SORunResult`. It computes no metrics; success rate, ``|f - f*|``, and
``||x - x*||`` belong to the metrics layer.

Public entry point :func:`run_so` binds the committed budget from
:mod:`benchmarks.config`; the three adapters :func:`run_ctrl_freak`,
:func:`run_pymoo`, and :func:`run_deap` take ``pop_size`` and ``n_generations``
explicitly so tests can exercise small budgets.

Notes
-----
Alignment of the three implementations (the only intended difference is the
library):

================= ============================== ============================== ==============================
Aspect            ctrl-freak (reference)         pymoo mirror                   DEAP mirror
================= ============================== ============================== ==============================
Initialization    ``rng.uniform(lo, hi, dim)``   ``FloatRandomSampling()``      ``random.uniform(lo, hi)``
Crossover         ``sbx_crossover(eta=15)``      ported core (custom Crossover) ported core (custom mate)
  per mating      every mating (prob == 1.0)     ``prob=1.0``                   ``cxpb=1.0`` (every pair)
  per variable    all eligible (1.0)             all eligible (1.0, ported)     all eligible (1.0, ported)
Mutation          ``polynomial_mutation(eta=20)``  ``PM(eta=20)``               ``mutPolynomialBounded(eta=20)``
  per individual  every offspring (1.0)          ``prob=1.0``                   ``mutpb=1.0`` (every individual)
  per variable    ``prob=1/dim``                 ``prob_var=1/dim``             ``indpb=1/dim``
Parent selection  ``fitness_tournament(size=2)`` ``TournamentSelection(p=2)``   ``selTournament(tournsize=2)``
Survival          ``elitist_survival(1)``        custom ``Survival`` (split)    ``_elitist_survivor_indices``
Eval budget       ``pop + n_gen*pop``            ``("n_gen", n_gen+1)``         init + invalid offspring
Seeding           ``SeedSequence(seed)``         ``minimize(seed=seed)``        ``random.seed`` + ``np.random``
================= ============================== ============================== ==============================

SBX is **identical** across all three: ctrl-freak's exact single-child,
every-eligible-variable SBX is ported into a custom pymoo ``Crossover`` and a
custom DEAP mate (both call the same core twice per pair --
``benchmarks.harness.operators``); only the per-framework RNG source differs. This
isolates the GA loop (selection, survival, sorting) as the sole subject of the
comparison. (Earlier Option A left pymoo/DEAP at the idiomatic per-variable 0.5,
which put ctrl-freak ~70% above both baselines on convex ZDT IGD+ -- a config
difference that would fail s5's equivalence test; Option C eliminates it.)

Examples
--------
>>> from benchmarks.problems.single_objective import SO_PROBLEMS
>>> from benchmarks.harness.single_objective import run_ctrl_freak
>>> result = run_ctrl_freak(SO_PROBLEMS["sphere"], seed=0, pop_size=10, n_generations=3)
>>> result.n_evaluations
40
>>> len(result.history)
4
>>> result.best_x.shape
(10,)
>>> bool(result.best_f >= 0.0)
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks import config
from benchmarks.harness.operators import deap_ctrl_freak_sbx, make_pymoo_ctrl_freak_sbx
from benchmarks.problems.single_objective import SOProblem

__all__ = [
    "SORunResult",
    "run_so",
    "run_ctrl_freak",
    "run_pymoo",
    "run_deap",
]


@dataclass
class SORunResult:
    """Raw, normalized output of one single-objective optimization run.

    Parameters
    ----------
    best_x : numpy.ndarray
        Best decision vector found, shape ``(dim,)``.
    best_f : float
        Best (minimum) objective value found.
    history : list of float
        Per-generation best objective, ``history[k]`` being the population best
        after ``k`` completed generations (``history[0]`` is the initial-population
        best). Length ``n_generations + 1``. Monotone non-increasing because
        elitist survival preserves the best individual.
    n_evaluations : int
        Total number of objective evaluations,
        ``pop_size + n_generations * pop_size``.
    """

    best_x: np.ndarray
    best_f: float
    history: list[float]
    n_evaluations: int


def _elitist_survivor_indices(combined_fitness: np.ndarray, n_survive: int, n_elite: int = 1) -> np.ndarray:
    """Replicate ``elitist_survival(elite_count=n_elite)`` as pure index logic.

    Given a **parents-first** concatenated fitness vector -- the first
    ``n_survive`` entries are parents, the remainder are offspring, minimization
    -- return the survivor indices into that combined array: the best
    ``n_elite`` parents followed by the best ``n_survive - n_elite`` offspring,
    using a stable sort for tie-breaking exactly as
    ``ctrl_freak.survival.elitist`` does.

    Parameters
    ----------
    combined_fitness : numpy.ndarray
        1-D fitness of ``[parents..., offspring...]`` (parents first).
    n_survive : int
        Number of survivors; equals ``parent_size`` for this harness.
    n_elite : int, default=1
        Number of elite parents to preserve.

    Returns
    -------
    numpy.ndarray
        Survivor indices (``numpy.intp``): elite parents then best offspring.

    Examples
    --------
    >>> import numpy as np
    >>> _elitist_survivor_indices(np.array([2.0, 1.0, 4.0, 3.0, 0.5, 1.5]), 3)
    array([1, 4, 5])
    """
    fitness = np.asarray(combined_fitness, dtype=float)
    parent_fitness = fitness[:n_survive]
    offspring_fitness = fitness[n_survive:]
    elite = np.argsort(parent_fitness, kind="stable")[:n_elite]
    n_offspring_needed = n_survive - n_elite
    best_offspring = np.argsort(offspring_fitness, kind="stable")[:n_offspring_needed] + n_survive
    return np.concatenate([elite, best_offspring]).astype(np.intp)


def _compare_by_fitness(pop, P, random_state=None, **kwargs):
    """pymoo tournament comparator mirroring ``fitness_tournament(size=2)``.

    Lower objective wins; ties resolve to the first competitor (mirroring
    ctrl-freak's first-on-tie tournament). ``P`` has shape ``(n_tournaments, 2)``.

    Parameters
    ----------
    pop : pymoo.core.population.Population
        The population the tournament indices refer to.
    P : numpy.ndarray
        Tournament index pairs, shape ``(n_tournaments, 2)``.
    random_state : optional
        Accepted for the pymoo calling convention; unused (deterministic).

    Returns
    -------
    numpy.ndarray
        Winner index per tournament, shape ``(n_tournaments, 1)``.
    """
    winners = np.empty(P.shape[0], dtype=int)
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        winners[i] = a if pop[a].F[0] <= pop[b].F[0] else b
    return winners[:, None]


def _make_pymoo_elitist_survival(n_elite: int = 1):
    """Build the custom pymoo ``Survival`` replicating ``elitist_survival(1)``.

    The stock ``GeneticAlgorithm._advance`` merges parents-first then calls
    ``survival.do(..., n_survive=pop_size)``; this survival splits that merged
    population at ``n_survive`` and keeps ``[best n_elite parents] + [best
    (n_survive - n_elite) offspring]`` via :func:`_elitist_survivor_indices`.
    Returned as a factory so ``pymoo`` is imported lazily and the survival can be
    unit-tested in isolation.

    Parameters
    ----------
    n_elite : int, default=1
        Number of elite parents to preserve.

    Returns
    -------
    pymoo.core.survival.Survival
        The configured elitist survival instance.
    """
    from pymoo.core.survival import Survival

    class _ElitistSurvival(Survival):
        def __init__(self) -> None:
            super().__init__(filter_infeasible=False)

        def _do(self, problem, pop, *args, n_survive=None, random_state=None, **kwargs):
            assert n_survive is not None  # stock GA always passes n_survive=pop_size
            indices = _elitist_survivor_indices(pop.get("F")[:, 0], n_survive, n_elite)
            return pop[indices]

    return _ElitistSurvival()


def run_ctrl_freak(problem: SOProblem, seed: int, pop_size: int, n_generations: int) -> SORunResult:
    """Run ctrl-freak ``ga`` on a single-objective problem (the reference run).

    Parameters
    ----------
    problem : SOProblem
        Problem providing ``func``, scalar ``bounds``, and ``dim``.
    seed : int
        Master seed; one seed reproduces the run bit-identically.
    pop_size : int
        Population size (must be even, per ctrl-freak's ``ga``).
    n_generations : int
        Number of generations.

    Returns
    -------
    SORunResult
        Raw output: best vector/objective, convergence history, eval count.

    Examples
    --------
    >>> from benchmarks.problems.single_objective import SO_PROBLEMS
    >>> result = run_ctrl_freak(SO_PROBLEMS["sphere"], seed=0, pop_size=10, n_generations=3)
    >>> result.n_evaluations
    40
    >>> len(result.history)
    4
    >>> result.best_x.shape
    (10,)
    """
    from ctrl_freak import ga, polynomial_mutation, sbx_crossover

    lo, hi = problem.bounds
    dim = problem.dim

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(lo, hi, size=dim)

    history: list[float] = []

    def callback(result, generation) -> bool:
        history.append(float(result.fitness[result.best_idx]))
        return False

    result = ga(
        init=init,
        evaluate=problem.func,
        crossover=sbx_crossover(eta=config.SBX_ETA, bounds=problem.bounds),
        mutate=polynomial_mutation(eta=config.PM_ETA, prob=config.mutation_prob(dim), bounds=problem.bounds),
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        select="tournament",
        survive="elitist",
        callback=callback,
    )
    best_x, best_f = result.best
    history.append(float(best_f))
    return SORunResult(
        best_x=np.asarray(best_x, dtype=float),
        best_f=float(best_f),
        history=history,
        n_evaluations=int(result.evaluations),
    )


def run_pymoo(problem: SOProblem, seed: int, pop_size: int, n_generations: int) -> SORunResult:
    """Run pymoo ``GA`` aligned to ctrl-freak's algorithm.

    Uses ``eliminate_duplicates=False``, ``n_offsprings=pop_size``,
    ``termination=("n_gen", n_generations + 1)`` (pymoo counts the initial
    population as generation 1), a first-on-tie binary tournament, the custom
    elitist survival from :func:`_make_pymoo_elitist_survival`, and the ported
    ctrl-freak SBX from :func:`~benchmarks.harness.operators.make_pymoo_ctrl_freak_sbx`
    (identical crossover, not pymoo's stock ``SBX``). Mutation uses ``prob=1.0``
    (every individual) with ``prob_var=1/dim`` (per variable); ``prob=1/dim`` would
    silently weaken mutation.

    Parameters
    ----------
    problem : SOProblem
        Problem providing ``func``, scalar ``bounds``, and ``dim``.
    seed : int
        Random seed passed to ``minimize``.
    pop_size : int
        Population size and offspring count per generation.
    n_generations : int
        Number of offspring batches (generations) to evolve.

    Returns
    -------
    SORunResult
        Raw output: best vector/objective, convergence history, eval count.

    Examples
    --------
    >>> from benchmarks.problems.single_objective import SO_PROBLEMS
    >>> result = run_pymoo(SO_PROBLEMS["sphere"], seed=0, pop_size=10, n_generations=3)
    >>> result.n_evaluations
    40
    >>> len(result.history)
    4
    >>> result.best_x.shape
    (10,)
    """
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.callback import Callback
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.selection.tournament import TournamentSelection
    from pymoo.optimize import minimize

    lo, hi = problem.bounds
    dim = problem.dim
    func = problem.func

    class _SOProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=dim, n_obj=1, xl=lo, xu=hi)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = func(x)

    class _HistoryCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.history: list[float] = []

        def notify(self, algorithm):
            self.history.append(float(algorithm.pop.get("F").min()))

    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        selection=TournamentSelection(func_comp=_compare_by_fitness, pressure=2),
        crossover=make_pymoo_ctrl_freak_sbx(eta=config.SBX_ETA, prob=config.CROSSOVER_PROB),
        mutation=PM(eta=config.PM_ETA, prob=1.0, prob_var=config.mutation_prob(dim)),
        survival=_make_pymoo_elitist_survival(n_elite=1),
        n_offsprings=pop_size,
        eliminate_duplicates=False,
    )
    callback = _HistoryCallback()
    result = minimize(
        _SOProblem(),
        algorithm,
        termination=("n_gen", n_generations + 1),
        seed=seed,
        callback=callback,
        verbose=False,
    )
    return SORunResult(
        best_x=np.asarray(result.X, dtype=float),
        best_f=float(np.ravel(result.F)[0]),
        history=callback.history,
        n_evaluations=int(result.algorithm.evaluator.n_eval),
    )


def run_deap(problem: SOProblem, seed: int, pop_size: int, n_generations: int) -> SORunResult:
    """Run a DEAP GA aligned to ctrl-freak via a custom varAnd loop.

    The loop selects ``pop_size`` parents by binary tournament, applies the ported
    ctrl-freak SBX (:func:`~benchmarks.harness.operators.deap_ctrl_freak_sbx`, driven
    by a numpy ``Generator``) to every consecutive pair (``cxpb=1.0``) and mutation
    to every individual (``mutpb=1.0``; the per-gene ``indpb=1/dim`` does the
    masking), evaluates only invalid-fitness offspring, and survives via the same
    elitist rule (:func:`_elitist_survivor_indices` over
    ``[current population, offspring]``). This is **not** ``eaMuPlusLambda``/``varOr``.
    ``mutpb=1/dim`` would silently weaken mutation -- the per-individual probability
    stays ``1.0``.

    Parameters
    ----------
    problem : SOProblem
        Problem providing ``func``, scalar ``bounds``, and ``dim``.
    seed : int
        Seed for python ``random`` (and numpy) for reproducibility.
    pop_size : int
        Population size (even).
    n_generations : int
        Number of generations.

    Returns
    -------
    SORunResult
        Raw output: best vector/objective, convergence history, eval count.

    Examples
    --------
    >>> from benchmarks.problems.single_objective import SO_PROBLEMS
    >>> result = run_deap(SO_PROBLEMS["sphere"], seed=0, pop_size=10, n_generations=3)
    >>> result.n_evaluations
    40
    >>> len(result.history)
    4
    >>> result.best_x.shape
    (10,)
    """
    import random

    from deap import base, creator, tools

    lo, hi = problem.bounds
    dim = problem.dim
    func = problem.func

    random.seed(seed)
    np.random.seed(seed)
    cx_rng = np.random.default_rng(seed)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)  # ty: ignore[unresolved-attribute]

    toolbox: Any = base.Toolbox()  # Any: DEAP registers attributes dynamically
    toolbox.register("attr_float", random.uniform, lo, hi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dim)  # ty: ignore[unresolved-attribute]
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", deap_ctrl_freak_sbx, eta=config.SBX_ETA, low=lo, up=hi, rng=cx_rng)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=config.PM_ETA,
        low=lo,
        up=hi,
        indpb=config.mutation_prob(dim),
    )
    toolbox.register("select", tools.selTournament, tournsize=2)

    def evaluate(individual) -> tuple[float]:
        return (func(np.asarray(individual, dtype=float)),)

    population = toolbox.population(n=pop_size)
    for ind in population:
        ind.fitness.values = evaluate(ind)
    n_evaluations = pop_size

    history: list[float] = [min(ind.fitness.values[0] for ind in population)]

    for _ in range(n_generations):
        parents = toolbox.select(population, pop_size)
        offspring = [toolbox.clone(ind) for ind in parents]
        for i in range(0, pop_size, 2):
            toolbox.mate(offspring[i], offspring[i + 1])
        for ind in offspring:
            toolbox.mutate(ind)
        for ind in offspring:
            del ind.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = evaluate(ind)
        n_evaluations += len(invalid)

        parent_fitness = np.array([ind.fitness.values[0] for ind in population])
        offspring_fitness = np.array([ind.fitness.values[0] for ind in offspring])
        indices = _elitist_survivor_indices(np.concatenate([parent_fitness, offspring_fitness]), pop_size)
        population = [population[i] if i < pop_size else offspring[i - pop_size] for i in indices]
        history.append(min(ind.fitness.values[0] for ind in population))

    best = min(population, key=lambda ind: ind.fitness.values[0])
    return SORunResult(
        best_x=np.asarray(best, dtype=float),
        best_f=float(best.fitness.values[0]),
        history=history,
        n_evaluations=n_evaluations,
    )


_ADAPTERS = {
    "ctrl-freak": run_ctrl_freak,
    "pymoo": run_pymoo,
    "deap": run_deap,
}


def run_so(problem: SOProblem, library: str, seed: int) -> SORunResult:
    """Run one single-objective optimization at the committed config budget.

    Dispatches to the ctrl-freak, pymoo, or DEAP adapter using
    ``config.SO_POP_SIZE`` and ``config.SO_N_GENERATIONS``.

    Parameters
    ----------
    problem : SOProblem
        Problem from ``benchmarks.problems.single_objective``.
    library : {"ctrl-freak", "pymoo", "deap"}
        Which implementation to run.
    seed : int
        Per-run seed.

    Returns
    -------
    SORunResult
        Raw output for the chosen library.

    Raises
    ------
    ValueError
        If ``library`` is not one of the three supported names.

    Examples
    --------
    >>> from benchmarks.problems.single_objective import SO_PROBLEMS
    >>> from benchmarks.harness.single_objective import run_so
    >>> result = run_so(SO_PROBLEMS["sphere"], "ctrl-freak", seed=0)
    >>> result.n_evaluations
    20100
    >>> bool(result.best_f < 0.1)
    True
    >>> len(result.history)
    201
    """
    if library not in _ADAPTERS:
        raise ValueError(f"unknown library {library!r}; expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[library](problem, seed, config.SO_POP_SIZE, config.SO_N_GENERATIONS)
