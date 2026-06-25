"""Multi-objective benchmark problems with analytical Pareto fronts.

This module defines the multi-objective half of the ctrl-freak validation
benchmark: the ZDT1, ZDT2, ZDT3, ZDT4, and ZDT6 problems (each with 30 decision
variables and 2 objectives) and DTLZ2 (3 objectives, 12 decision variables). Each
problem provides three things — an *objective function* mapping one decision
vector to its objective vector, an *analytical true-Pareto-front sampler*
returning points that lie exactly on the known optimal front, and *metadata*
(decision-variable bounds, ``n_var``, ``n_obj``, and a hypervolume reference
point) bundled in :class:`MOProblem` and collected in the :data:`MO_PROBLEMS`
registry.

The objective formulas reproduce the standard ZDT / DTLZ definitions and match
``pymoo``'s problem evaluations to machine precision; the front samplers reproduce
``pymoo``'s analytical ``pareto_front()`` references (validated in
``tests/benchmarks/test_multi_objective_problems.py``).

Constants
---------
ZDT_N_VAR : int
    Decision-variable count for every ZDT problem (30).
DTLZ2_N_VAR, DTLZ2_N_OBJ, DTLZ2_K : int
    DTLZ2 dimensions: ``n_var = n_obj - 1 + k = 2 + 10 = 12``, ``n_obj = 3``,
    ``k = 10``.
DTLZ2_N_PARTITIONS : int
    das-dennis partition count for the DTLZ2 front (49 ⇒ 1275 reference points).
ZDT6_F1_MIN : float
    Analytical lower bound of ``f1`` on the ZDT6 Pareto front (0.2807753191).
ZDT3_REGIONS : tuple[tuple[float, float], ...]
    The five disconnected ``f1`` intervals that make up the (non-dominated) ZDT3
    front.

Notes
-----
**Decision-variable count.** The canonical Zitzler-Deb-Thiele paper defines ZDT4
and ZDT6 with ``n = 10``; this suite uses ``n = 30`` for all five ZDT problems
(per the project benchmark spec). This changes only the search difficulty in
decision space, **not** the analytical front: every ZDT front lies at ``g = 1``,
which is attained independently of ``n``, so the front samplers take no ``n``
argument and are unaffected.

Examples
--------
>>> sorted(MO_PROBLEMS)
['dtlz2', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']
>>> p = MO_PROBLEMS["dtlz2"]
>>> (p.n_var, p.n_obj)
(12, 3)
>>> p.ref_point.tolist()
[1.1, 1.1, 1.1]
>>> z = MO_PROBLEMS["zdt4"]
>>> (z.n_var, z.n_obj)
(30, 2)
>>> z.ref_point.tolist()
[1.1, 1.1]
>>> [z.bounds[0].shape, z.bounds[1].shape]
[(30,), (30,)]
>>> (float(z.bounds[0][1]), float(z.bounds[1][1]))  # ZDT4 var-2 bounds are [-5, 5]
(-5.0, 5.0)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# --- Shared problem configuration ---------------------------------------------
ZDT_N_VAR: int = 30
DTLZ2_N_OBJ: int = 3
DTLZ2_K: int = 10
DTLZ2_N_VAR: int = DTLZ2_N_OBJ - 1 + DTLZ2_K  # 2 + 10 = 12
DTLZ2_N_PARTITIONS: int = 49  # das-dennis -> (50 * 51) / 2 = 1275 reference points

# Analytical lower bound of f1 on the ZDT6 Pareto front (max of
# exp(-4 x1) * sin^6(6 pi x1) over x1 in [0, 1]); matches pymoo's reference.
ZDT6_F1_MIN: float = 0.2807753191

# The five disconnected non-dominated f1 intervals of the ZDT3 front. These are
# pymoo's exact region constants, so the sampled front aligns with pymoo's
# pareto_front() under IGD+.
ZDT3_REGIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0830015349),
    (0.182228780, 0.2577623634),
    (0.4093136748, 0.4538821041),
    (0.6183967944, 0.6525117038),
    (0.8233317983, 0.8518328654),
)


# --- Objective functions (one decision vector -> objective vector) ------------
def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 objectives: convex Pareto front.

    Parameters
    ----------
    x : numpy.ndarray
        Decision vector of shape ``(30,)`` with values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The two objectives ``[f1, f2]`` to minimise.

    Examples
    --------
    >>> zdt1(np.zeros(30))
    array([0., 1.])
    """
    n = x.shape[0]
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g)
    return np.array([f1, g * h])


def zdt2(x: np.ndarray) -> np.ndarray:
    """ZDT2 objectives: concave (non-convex) Pareto front.

    Parameters
    ----------
    x : numpy.ndarray
        Decision vector of shape ``(30,)`` with values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The two objectives ``[f1, f2]`` to minimise.

    Examples
    --------
    >>> zdt2(np.zeros(30))
    array([0., 1.])
    """
    n = x.shape[0]
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - (f1 / g) ** 2
    return np.array([f1, g * h])


def zdt3(x: np.ndarray) -> np.ndarray:
    """ZDT3 objectives: discontinuous Pareto front (five disconnected segments).

    Parameters
    ----------
    x : numpy.ndarray
        Decision vector of shape ``(30,)`` with values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The two objectives ``[f1, f2]`` to minimise.

    Examples
    --------
    >>> zdt3(np.zeros(30))
    array([0., 1.])
    """
    n = x.shape[0]
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
    return np.array([f1, g * h])


def zdt4(x: np.ndarray) -> np.ndarray:
    """ZDT4 objectives: convex front, strongly multimodal decision space.

    Variable ``x[0]`` lies in ``[0, 1]``; the remaining variables lie in
    ``[-5, 5]`` (see :data:`MO_PROBLEMS`). The Pareto front is identical to ZDT1
    (``f2 = 1 - sqrt(f1)``) because ``g = 1`` at the optimum for any ``n``.

    Parameters
    ----------
    x : numpy.ndarray
        Decision vector of shape ``(30,)``: ``x[0]`` in ``[0, 1]``, ``x[1:]`` in
        ``[-5, 5]``.

    Returns
    -------
    numpy.ndarray
        The two objectives ``[f1, f2]`` to minimise.

    Examples
    --------
    >>> zdt4(np.zeros(30))
    array([0., 1.])
    """
    n = x.shape[0]
    f1 = x[0]
    g = 1.0 + 10.0 * (n - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
    h = 1.0 - np.sqrt(f1 / g)
    return np.array([f1, g * h])


def zdt6(x: np.ndarray) -> np.ndarray:
    """ZDT6 objectives: non-uniform, biased, concave Pareto front.

    The front spans ``f1 in [ZDT6_F1_MIN, 1]`` with ``f2 = 1 - f1**2``.

    Parameters
    ----------
    x : numpy.ndarray
        Decision vector of shape ``(30,)`` with values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The two objectives ``[f1, f2]`` to minimise.

    Examples
    --------
    >>> zdt6(np.zeros(30))
    array([1., 0.])
    """
    n = x.shape[0]
    f1 = 1.0 - np.exp(-4.0 * x[0]) * np.sin(6.0 * np.pi * x[0]) ** 6
    g = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1)) ** 0.25
    h = 1.0 - (f1 / g) ** 2
    return np.array([f1, g * h])


def dtlz2(x: np.ndarray) -> np.ndarray:
    """DTLZ2 objectives (3 objectives): first-octant unit-sphere Pareto front.

    The first ``n_obj - 1 = 2`` variables parametrise the front; the last
    ``k = 10`` variables drive ``g`` (zero at the optimum, where they all equal
    ``0.5``). At the optimum the objectives satisfy ``f1**2 + f2**2 + f3**2 = 1``.

    Parameters
    ----------
    x : numpy.ndarray
        Decision vector of shape ``(12,)`` with values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        The three objectives ``[f1, f2, f3]`` to minimise.

    Examples
    --------
    >>> x = np.concatenate([np.zeros(2), np.full(10, 0.5)])
    >>> dtlz2(x)
    array([1., 0., 0.])
    """
    g = np.sum((x[DTLZ2_N_OBJ - 1 :] - 0.5) ** 2)
    theta = x[: DTLZ2_N_OBJ - 1] * (np.pi / 2.0)
    f1 = (1.0 + g) * np.cos(theta[0]) * np.cos(theta[1])
    f2 = (1.0 + g) * np.cos(theta[0]) * np.sin(theta[1])
    f3 = (1.0 + g) * np.sin(theta[0])
    return np.array([f1, f2, f3])


# --- Analytical true-Pareto-front samplers ------------------------------------
def zdt1_front(n_points: int = 200) -> np.ndarray:
    """Sample the analytical ZDT1 Pareto front (convex, ``f2 = 1 - sqrt(f1)``).

    Parameters
    ----------
    n_points : int, optional
        Number of evenly spaced front points to return (default 200).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, 2)`` of points on the true front.

    Examples
    --------
    >>> front = zdt1_front(5)
    >>> front.shape
    (5, 2)
    >>> front[0].tolist()
    [0.0, 1.0]
    >>> front[-1].tolist()
    [1.0, 0.0]
    """
    f1 = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


def zdt2_front(n_points: int = 200) -> np.ndarray:
    """Sample the analytical ZDT2 Pareto front (concave, ``f2 = 1 - f1**2``).

    Parameters
    ----------
    n_points : int, optional
        Number of evenly spaced front points to return (default 200).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, 2)`` of points on the true front.

    Examples
    --------
    >>> front = zdt2_front(5)
    >>> front.shape
    (5, 2)
    >>> front[0].tolist()
    [0.0, 1.0]
    >>> front[-1].tolist()
    [1.0, 0.0]
    """
    f1 = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([f1, 1.0 - f1**2])


def zdt3_front(n_points_per_segment: int = 40) -> np.ndarray:
    """Sample the analytical ZDT3 Pareto front (five disconnected segments).

    Only the non-dominated portions are sampled, by restricting ``f1`` to the
    canonical :data:`ZDT3_REGIONS` intervals; on each segment
    ``f2 = 1 - sqrt(f1) - f1 * sin(10 * pi * f1)``.

    Parameters
    ----------
    n_points_per_segment : int, optional
        Number of evenly spaced points sampled within each of the five segments
        (default 40 ⇒ 200 points total).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(5 * n_points_per_segment, 2)`` of points on the true
        front.

    Examples
    --------
    >>> front = zdt3_front(8)
    >>> front.shape
    (40, 2)
    >>> bool(np.all(front[:, 0] <= 0.8518328654 + 1e-9))
    True
    """
    segments = []
    for lo, hi in ZDT3_REGIONS:
        f1 = np.linspace(lo, hi, n_points_per_segment)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        segments.append(np.column_stack([f1, f2]))
    return np.vstack(segments)


def zdt4_front(n_points: int = 200) -> np.ndarray:
    """Sample the analytical ZDT4 Pareto front (identical to ZDT1).

    Parameters
    ----------
    n_points : int, optional
        Number of evenly spaced front points to return (default 200).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, 2)`` of points on the true front
        (``f2 = 1 - sqrt(f1)``).

    Examples
    --------
    >>> front = zdt4_front(5)
    >>> front.shape
    (5, 2)
    >>> front[-1].tolist()
    [1.0, 0.0]
    """
    f1 = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


def zdt6_front(n_points: int = 200) -> np.ndarray:
    """Sample the analytical ZDT6 Pareto front (``f2 = 1 - f1**2``, biased range).

    ``f1`` ranges over ``[ZDT6_F1_MIN, 1]`` (the front does not reach ``f1 = 0``).

    Parameters
    ----------
    n_points : int, optional
        Number of evenly spaced front points to return (default 200).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, 2)`` of points on the true front.

    Examples
    --------
    >>> front = zdt6_front(5)
    >>> front.shape
    (5, 2)
    >>> round(float(front[0, 0]), 10)
    0.2807753191
    >>> front[-1].tolist()
    [1.0, 0.0]
    """
    f1 = np.linspace(ZDT6_F1_MIN, 1.0, n_points)
    return np.column_stack([f1, 1.0 - f1**2])


def dtlz2_front(n_partitions: int = DTLZ2_N_PARTITIONS) -> np.ndarray:
    """Sample the analytical DTLZ2 Pareto front (first-octant unit sphere).

    Generates a das-dennis reference-direction lattice on the 3-objective simplex
    and normalises each direction to the unit sphere, yielding an even spread of
    points satisfying ``f1**2 + f2**2 + f3**2 = 1`` with all ``f_i >= 0``. With
    the default ``n_partitions`` this returns 1275 points and reproduces pymoo's
    ``DTLZ2.pareto_front(ref_dirs)`` exactly.

    Parameters
    ----------
    n_partitions : int, optional
        das-dennis partition count (default :data:`DTLZ2_N_PARTITIONS` = 49 ⇒
        ``(n_partitions + 1)(n_partitions + 2) / 2`` points).

    Returns
    -------
    numpy.ndarray
        Array of shape ``((n_partitions + 1)(n_partitions + 2) / 2, 3)`` of points
        on the true front.

    Examples
    --------
    >>> front = dtlz2_front(n_partitions=2)
    >>> front.shape
    (6, 3)
    >>> bool(np.allclose((front**2).sum(axis=1), 1.0))
    True
    """
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("das-dennis", DTLZ2_N_OBJ, n_partitions=n_partitions)
    return ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)


# --- Per-problem metadata + registry ------------------------------------------
@dataclass(frozen=True, eq=False)
class MOProblem:
    """Metadata bundle for one multi-objective benchmark problem.

    ``eq=False`` because the numpy-array fields make the auto-generated ``__eq__``
    ambiguous; instances are identity-compared (and never compared in this suite).

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"zdt1"``).
    func : Callable[[numpy.ndarray], numpy.ndarray]
        Objective function mapping one decision vector to its objective vector.
    true_front_sampler : Callable[..., numpy.ndarray]
        Returns points on the analytical true Pareto front, shape
        ``(n_points, n_obj)``.
    bounds : tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper decision-variable bounds ``(xl, xu)``, each of shape
        ``(n_var,)``.
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    ref_point : numpy.ndarray
        Hypervolume reference point of shape ``(n_obj,)`` (dominated by the whole
        front).

    Examples
    --------
    >>> p = MO_PROBLEMS["zdt1"]
    >>> isinstance(p, MOProblem)
    True
    >>> p.func(np.zeros(p.n_var)).shape
    (2,)
    >>> p.true_front_sampler(5).shape
    (5, 2)
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    true_front_sampler: Callable[..., np.ndarray]
    bounds: tuple[np.ndarray, np.ndarray]
    n_var: int
    n_obj: int
    ref_point: np.ndarray


def _uniform_bounds(n_var: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xl, xu)`` arrays of shape ``(n_var,)`` filled with ``lo``/``hi``.

    Parameters
    ----------
    n_var : int
        Number of decision variables.
    lo, hi : float
        Lower and upper bound applied to every variable.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The ``(xl, xu)`` bound arrays.

    Examples
    --------
    >>> xl, xu = _uniform_bounds(3, 0.0, 1.0)
    >>> (xl.tolist(), xu.tolist())
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    """
    return np.full(n_var, lo), np.full(n_var, hi)


def _zdt4_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return ZDT4 bounds: ``x[0]`` in ``[0, 1]``, ``x[1:]`` in ``[-5, 5]``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The ``(xl, xu)`` bound arrays of shape ``(30,)``.

    Examples
    --------
    >>> xl, xu = _zdt4_bounds()
    >>> (float(xl[0]), float(xu[0]), float(xl[1]), float(xu[1]))
    (0.0, 1.0, -5.0, 5.0)
    """
    xl = np.full(ZDT_N_VAR, -5.0)
    xu = np.full(ZDT_N_VAR, 5.0)
    xl[0] = 0.0
    xu[0] = 1.0
    return xl, xu


_ZDT_REF_POINT = np.array([1.1, 1.1])
_DTLZ2_REF_POINT = np.array([1.1, 1.1, 1.1])

MO_PROBLEMS: dict[str, MOProblem] = {
    "zdt1": MOProblem("zdt1", zdt1, zdt1_front, _uniform_bounds(ZDT_N_VAR, 0.0, 1.0), ZDT_N_VAR, 2, _ZDT_REF_POINT),
    "zdt2": MOProblem("zdt2", zdt2, zdt2_front, _uniform_bounds(ZDT_N_VAR, 0.0, 1.0), ZDT_N_VAR, 2, _ZDT_REF_POINT),
    "zdt3": MOProblem("zdt3", zdt3, zdt3_front, _uniform_bounds(ZDT_N_VAR, 0.0, 1.0), ZDT_N_VAR, 2, _ZDT_REF_POINT),
    "zdt4": MOProblem("zdt4", zdt4, zdt4_front, _zdt4_bounds(), ZDT_N_VAR, 2, _ZDT_REF_POINT),
    "zdt6": MOProblem("zdt6", zdt6, zdt6_front, _uniform_bounds(ZDT_N_VAR, 0.0, 1.0), ZDT_N_VAR, 2, _ZDT_REF_POINT),
    "dtlz2": MOProblem(
        "dtlz2", dtlz2, dtlz2_front, _uniform_bounds(DTLZ2_N_VAR, 0.0, 1.0), DTLZ2_N_VAR, DTLZ2_N_OBJ, _DTLZ2_REF_POINT
    ),
}
