"""Ported ctrl-freak SBX crossover, shared by the fair single- and multi-objective harnesses.

The fairness comparison must isolate the GA/NSGA-II *loop* (selection, survival,
non-dominated sort, crowding) from the *operators*. Because ctrl-freak, pymoo, and
DEAP implement simulated binary crossover (SBX) with irreconcilable per-variable
semantics -- ctrl-freak recombines **every** eligible variable and returns a
**single** child via per-variable mixing of the two SBX child-values, while pymoo
(``prob_var`` default 0.5) and DEAP (per-variable 0.5, hardcoded) recombine only
about half -- this module **ports ctrl-freak's exact SBX** into custom pymoo and
DEAP operators so all three libraries run the *identical* crossover. Only the RNG
source differs per framework.

The single source of truth is :func:`ctrl_freak_sbx_child`, a bit-identical port of
``ctrl_freak.operators.standard._SBXCrossover.__call__``: eligibility is
``|p1 - p2| > 1e-14`` and ``xl < xu``; child-values come from the same ``betaq``
math; a per-variable fair coin mixes the two child-values; non-eligible variables
keep the first parent; the result is clipped to bounds. Two children per mating are
produced by **two independent calls** -- ``ctrl_freak_sbx_child(p1, p2, ...)`` and
``ctrl_freak_sbx_child(p2, p1, ...)`` -- mirroring ctrl-freak's algorithm layer,
where ``crossover(p1, p2)`` and ``crossover(p2, p1)`` consume the RNG independently
(not a conjugate pair).

Public API (also imported by the multi-objective harness)
--------------------------------------------------------
ctrl_freak_sbx_child
    The core single-child operator (numpy ``Generator`` RNG).
make_pymoo_ctrl_freak_sbx
    Factory returning a pymoo ``Crossover`` (2 parents, 2 offspring) that calls the
    core twice per mating. Lazily imports ``pymoo`` so this module loads without it.
deap_ctrl_freak_sbx
    A DEAP-compatible in-place mate calling the core twice per pair.

Examples
--------
The port reproduces ctrl-freak's operator bit-for-bit given the same RNG and parents:

>>> import numpy as np
>>> from ctrl_freak import sbx_crossover
>>> p1 = np.array([0.2, 0.4, 0.6])
>>> p2 = np.array([0.8, 0.1, 0.9])
>>> native = sbx_crossover(eta=15.0, bounds=(0.0, 1.0), seed=5)(p1, p2)
>>> port = ctrl_freak_sbx_child(p1, p2, 15.0, 0.0, 1.0, np.random.default_rng(5))
>>> bool(np.array_equal(native, port))
True
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "ctrl_freak_sbx_child",
    "make_pymoo_ctrl_freak_sbx",
    "deap_ctrl_freak_sbx",
]


def ctrl_freak_sbx_child(
    p1: np.ndarray,
    p2: np.ndarray,
    eta: float,
    xl: float | np.ndarray,
    xu: float | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Produce one SBX child, a bit-identical port of ctrl-freak's ``_SBXCrossover``.

    Recombines **every** eligible variable (``|p1 - p2| > 1e-14`` and ``xl < xu``)
    and returns a single child: per variable, the two SBX child-values are mixed by
    a fair coin; non-eligible variables keep ``p1``; the result is clipped to
    ``[xl, xu]``. Consumes ``rng`` in the same order as ctrl-freak (one draw of size
    ``len(p1)`` for ``betaq``, then one for the mixing coin), so the same seed and
    parents reproduce ctrl-freak's operator exactly.

    Parameters
    ----------
    p1, p2 : numpy.ndarray
        Parent decision vectors, shape ``(n_var,)``. ``p1`` is the parent retained
        where a variable is not eligible for recombination.
    eta : float
        SBX distribution index (15.0 in this suite).
    xl, xu : float or numpy.ndarray
        Lower and upper bounds; scalars broadcast to all variables.
    rng : numpy.random.Generator
        Random generator; advanced by two draws of size ``n_var``.

    Returns
    -------
    numpy.ndarray
        One child decision vector, shape ``(n_var,)``, clipped to bounds.

    Examples
    --------
    >>> import numpy as np
    >>> child = ctrl_freak_sbx_child(
    ...     np.array([0.2, 0.4, 0.6]), np.array([0.3, 0.5, 0.7]),
    ...     eta=15.0, xl=0.0, xu=1.0, rng=np.random.default_rng(0),
    ... )
    >>> child.shape
    (3,)
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    eps = 1e-14
    eta = float(eta)
    exponent = 1.0 / (eta + 1.0)

    xl_arr = np.broadcast_to(np.asarray(xl, dtype=float), p1.shape)
    xu_arr = np.broadcast_to(np.asarray(xu, dtype=float), p1.shape)

    sm = p1 < p2
    y1 = np.where(sm, p1, p2)
    y2 = np.where(sm, p2, p1)

    eligible = (np.abs(p1 - p2) > eps) & (xl_arr < xu_arr)
    delta = np.where(eligible, y2 - y1, 1.0)

    rand = rng.random(p1.shape[0])

    def calc_betaq(beta: np.ndarray) -> np.ndarray:
        alpha = 2.0 - np.power(beta, -(eta + 1.0))
        mask = rand <= (1.0 / alpha)
        return np.where(
            mask,
            np.power(rand * alpha, exponent),
            np.power(1.0 / (2.0 - rand * alpha), exponent),
        )

    beta1 = 1.0 + (2.0 * (y1 - xl_arr) / delta)
    c1 = 0.5 * ((y1 + y2) - calc_betaq(beta1) * delta)

    beta2 = 1.0 + (2.0 * (xu_arr - y2) / delta)
    c2 = 0.5 * ((y1 + y2) + calc_betaq(beta2) * delta)

    child_for_p1 = np.where(sm, c1, c2)
    child_for_p2 = np.where(sm, c2, c1)
    child = np.where(rng.random(p1.shape[0]) < 0.5, child_for_p1, child_for_p2)
    child = np.where(eligible, child, p1)
    return np.clip(child, xl_arr, xu_arr)


def make_pymoo_ctrl_freak_sbx(eta: float, prob: float = 1.0):
    """Build a pymoo ``Crossover`` that runs ctrl-freak's exact SBX.

    The returned operator declares 2 parents and 2 offspring; its ``_do`` produces
    the two children of each mating via two independent
    :func:`ctrl_freak_sbx_child` calls (``(p1, p2)`` then ``(p2, p1)``), driven by
    pymoo's threaded ``random_state``. ``prob`` is the per-mating crossover
    probability (1.0 -> every mating crosses, matching ctrl-freak). ``pymoo`` is
    imported lazily so this module loads without it.

    Parameters
    ----------
    eta : float
        SBX distribution index.
    prob : float, default=1.0
        Per-mating crossover probability.

    Returns
    -------
    pymoo.core.crossover.Crossover
        The configured crossover operator.

    Examples
    --------
    >>> cx = make_pymoo_ctrl_freak_sbx(eta=15.0)
    >>> (cx.n_parents, cx.n_offsprings)
    (2, 2)
    """
    from pymoo.core.crossover import Crossover

    class _CtrlFreakSBX(Crossover):
        def __init__(self) -> None:
            super().__init__(n_parents=2, n_offsprings=2, prob=prob)
            self.eta = float(eta)

        def _do(self, problem, X, *args, random_state=None, **kwargs):
            assert random_state is not None  # pymoo always threads a Generator
            _, n_matings, n_var = X.shape
            xl, xu = problem.xl, problem.xu
            Q = np.empty((2, n_matings, n_var), dtype=float)
            for m in range(n_matings):
                Q[0, m] = ctrl_freak_sbx_child(X[0, m], X[1, m], self.eta, xl, xu, random_state)
                Q[1, m] = ctrl_freak_sbx_child(X[1, m], X[0, m], self.eta, xl, xu, random_state)
            return Q

    return _CtrlFreakSBX()


def deap_ctrl_freak_sbx(
    ind1: Any,
    ind2: Any,
    eta: float,
    low: float | np.ndarray,
    up: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[Any, Any]:
    """DEAP-compatible mate running ctrl-freak's exact SBX, modifying in place.

    Produces the two children via two independent :func:`ctrl_freak_sbx_child`
    calls (``(ind1, ind2)`` then ``(ind2, ind1)``) and writes them back into the
    DEAP individuals. Register with a numpy ``Generator`` so DEAP's crossover uses
    the same operator as the other libraries.

    Parameters
    ----------
    ind1, ind2 : Sequence[float]
        DEAP individuals (mutable sequences), modified in place.
    eta : float
        SBX distribution index.
    low, up : float or numpy.ndarray
        Lower and upper bounds; scalars broadcast to all variables.
    rng : numpy.random.Generator
        Random generator driving the crossover.

    Returns
    -------
    tuple
        ``(ind1, ind2)``, the two mutated individuals (DEAP mate convention).

    Examples
    --------
    >>> import numpy as np
    >>> a, b = [0.2, 0.4, 0.6], [0.8, 0.1, 0.9]
    >>> c1, c2 = deap_ctrl_freak_sbx(a, b, 15.0, 0.0, 1.0, np.random.default_rng(0))
    >>> (len(c1), len(c2), c1 is a)
    (3, 3, True)
    """
    p1 = np.asarray(ind1, dtype=float)
    p2 = np.asarray(ind2, dtype=float)
    child1 = ctrl_freak_sbx_child(p1, p2, eta, low, up, rng)
    child2 = ctrl_freak_sbx_child(p2, p1, eta, low, up, rng)
    ind1[:] = child1.tolist()
    ind2[:] = child2.tolist()
    return ind1, ind2
