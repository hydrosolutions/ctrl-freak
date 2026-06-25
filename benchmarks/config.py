"""Shared run-configuration constants for the ctrl-freak benchmark suite.

This module holds *pure configuration only*: the committed random-seed list, the
single- and multi-objective evolutionary-algorithm budgets, and the genetic
operator hyper-parameters shared by every fair-comparison harness. It imports
nothing beyond the standard library, so it loads without pulling in ``pymoo``,
``deap``, ``matplotlib``, or even ``numpy`` -- the numpy-floor CI job imports it
freely. Every harness sub-module reads its budgets and operator settings from
here, so a single edit re-parameterises the entire sweep reproducibly.

Constants
---------
SEEDS : list[int]
    The committed list of 30 master seeds, ``0`` .. ``29``. Each integer seeds a
    full ctrl-freak run bit-identically (via ``SeedSequence(seed).spawn(4)``) and
    is shared across all three libraries for paired comparison.
N_SEEDS : int
    Convenience length of ``SEEDS`` (30); used to size the committed sweep.
SO_POP_SIZE, SO_N_GENERATIONS : int
    Single-objective GA budget: population 100, 200 generations.
MO_POP_SIZE, MO_N_GENERATIONS : int
    Multi-objective NSGA-II budget: population 100, 250 generations.
SBX_ETA : float
    Distribution index for simulated binary crossover (15.0).
PM_ETA : float
    Distribution index for polynomial mutation (20.0).
CROSSOVER_PROB : float
    Probability that a mating produces a crossover child (1.0; ctrl-freak applies
    crossover to every mating, so baselines must mirror ``prob = 1.0``).

Notes
-----
The per-variable polynomial-mutation probability is intentionally *not* a fixed
constant: it is ``1 / n_vars`` (one expected gene flip per genome), matching
ctrl-freak's ``polynomial_mutation`` default. Harnesses obtain it per problem via
``mutation_prob(n_vars)`` rather than hard-coding a number.

Examples
--------
>>> from benchmarks import config
>>> len(config.SEEDS)
30
>>> config.N_SEEDS
30
>>> (config.SEEDS[0], config.SEEDS[-1])
(0, 29)
>>> (config.SO_POP_SIZE, config.SO_N_GENERATIONS)
(100, 200)
>>> (config.MO_POP_SIZE, config.MO_N_GENERATIONS)
(100, 250)
>>> (config.SBX_ETA, config.PM_ETA, config.CROSSOVER_PROB)
(15.0, 20.0, 1.0)
>>> config.mutation_prob(30)
0.03333333333333333
"""

# Committed master seeds, shared across all libraries for paired comparison.
SEEDS: list[int] = list(range(30))
N_SEEDS: int = len(SEEDS)

# Single-objective GA budget (ctrl-freak `ga` vs pymoo `GA` vs DEAP).
SO_POP_SIZE: int = 100
SO_N_GENERATIONS: int = 200

# Multi-objective NSGA-II budget (ctrl-freak `nsga2` vs pymoo `NSGA2` vs DEAP).
MO_POP_SIZE: int = 100
MO_N_GENERATIONS: int = 250

# Genetic operator hyper-parameters shared by every harness.
SBX_ETA: float = 15.0
PM_ETA: float = 20.0
CROSSOVER_PROB: float = 1.0


def mutation_prob(n_vars: int) -> float:
    """Return the per-variable polynomial-mutation probability for a genome.

    Encodes the ``1 / n_vars`` convention (one expected gene flip per genome) that
    ctrl-freak's ``polynomial_mutation`` uses by default, so every harness reads
    the same value instead of hard-coding it.

    Parameters
    ----------
    n_vars : int
        Number of decision variables in the genome. Must be positive.

    Returns
    -------
    float
        The per-variable mutation probability, ``1 / n_vars``.

    Raises
    ------
    ValueError
        If `n_vars` is not a positive integer.

    Examples
    --------
    >>> mutation_prob(10)
    0.1
    >>> mutation_prob(30)
    0.03333333333333333
    """
    if n_vars <= 0:
        raise ValueError(f"n_vars must be positive, got {n_vars}")
    return 1.0 / n_vars
