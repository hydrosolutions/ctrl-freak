"""Performance metrics for benchmark optimization runs.

This module provides metrics for evaluating optimization results against known
optima. Single-objective metrics compare a found solution to the analytical
optimum (`f*`, `x*`) and report success within a tolerance `epsilon`.
Multi-objective metrics measure convergence of a Pareto-front approximation
against an analytical true front using pymoo indicators (IGD+, GD) plus the
hypervolume indicator.

All functions are pure: they take values (objective values, solutions,
reference fronts, `f*`, `x*`, `epsilon`) as explicit arguments and never import
problem definitions.

Examples
--------
>>> objective_error(0.5, 0.0)
0.5
>>> objs = np.array([[0.2, 0.8], [0.8, 0.2]])
>>> round(hypervolume(objs), 2)
0.45
"""

import numpy as np
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus


def objective_error(f_found: float, f_star: float) -> float:
    """Absolute objective-value error to the known optimum.

    Parameters
    ----------
    f_found : float
        Best objective value found by the optimizer (minimization).
    f_star : float
        Known global optimum objective value.

    Returns
    -------
    float
        ``abs(f_found - f_star)``.

    Examples
    --------
    >>> objective_error(2.5, 2.0)
    0.5
    >>> objective_error(0.0, 0.0)
    0.0
    """
    return abs(float(f_found) - float(f_star))


def solution_distance(x_found: np.ndarray, x_star: np.ndarray) -> float:
    """Euclidean distance from the found solution to the known optimizer.

    Parameters
    ----------
    x_found : numpy.ndarray
        Decision vector found by the optimizer, shape ``(n_vars,)``.
    x_star : numpy.ndarray
        Known global optimizer ``x*``, shape ``(n_vars,)``.

    Returns
    -------
    float
        The Euclidean norm ``||x_found - x_star||_2``.

    Raises
    ------
    ValueError
        If ``x_found`` and ``x_star`` do not have the same shape.

    Examples
    --------
    >>> solution_distance(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
    5.0
    >>> solution_distance(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    0.0
    """
    a = np.asarray(x_found, dtype=float)
    b = np.asarray(x_star, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"x_found and x_star must have the same shape, got {a.shape} and {b.shape}")
    return float(np.linalg.norm(a - b))


def is_success(f_found: float, f_star: float, epsilon: float) -> bool:
    """Whether a run reached within ``epsilon`` of the known optimum.

    Uses a strict comparison: a result exactly at ``epsilon`` is **not** a
    success.

    Parameters
    ----------
    f_found : float
        Best objective value found by the optimizer (minimization).
    f_star : float
        Known global optimum objective value.
    epsilon : float
        Success tolerance; success requires ``abs(f_found - f_star) < epsilon``.

    Returns
    -------
    bool
        ``True`` if the absolute objective error is strictly below
        ``epsilon``, else ``False``.

    Examples
    --------
    >>> is_success(1.25, 1.0, 0.5)   # error 0.25 < 0.5
    True
    >>> is_success(1.5, 1.0, 0.5)    # error 0.5 == 0.5, strict < => not success
    False
    >>> is_success(2.0, 1.0, 0.5)    # error 1.0 > 0.5
    False
    """
    return bool(objective_error(f_found, f_star) < epsilon)


def success_rate(f_found_values: np.ndarray, f_star: float, epsilon: float) -> float:
    """Fraction of runs that reached within ``epsilon`` of the optimum.

    Aggregates per-seed best objective values into a success fraction, using
    the same strict ``< epsilon`` rule as :func:`is_success`.

    Parameters
    ----------
    f_found_values : numpy.ndarray
        Best objective values across seeds, shape ``(n_seeds,)``.
    f_star : float
        Known global optimum objective value.
    epsilon : float
        Success tolerance; a seed counts as a success when its absolute
        objective error is ``< epsilon``.

    Returns
    -------
    float
        Fraction of seeds with ``abs(f_found - f_star) < epsilon`` in
        ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``f_found_values`` is empty.

    Examples
    --------
    >>> round(success_rate(np.array([1.0, 1.5, 1.0]), 1.0, 1e-6), 4)
    0.6667
    >>> success_rate(np.array([1.0, 1.0]), 1.0, 1e-6)
    1.0
    """
    values = np.asarray(f_found_values, dtype=float)
    if values.size == 0:
        raise ValueError("f_found_values array cannot be empty")
    errors = np.abs(values - float(f_star))
    return float(np.mean(errors < epsilon))


def igd_plus(approx_front: np.ndarray, true_front: np.ndarray) -> float:
    """Inverted Generational Distance plus (IGD+) to the analytical front.

    IGD+ measures, for each point of the true Pareto front, the modified
    distance to the nearest approximation point (penalizing only the dominated
    components). Lower is better; ``0`` means every true-front point is matched
    or dominated by the approximation. Computed via ``pymoo`` as
    ``IGDPlus(true_front)(approx_front)``.

    Parameters
    ----------
    approx_front : numpy.ndarray
        Objective values of the Pareto-front approximation, shape
        ``(n, n_obj)``.
    true_front : numpy.ndarray
        Analytical true Pareto front (reference set), shape ``(m, n_obj)``.

    Returns
    -------
    float
        The IGD+ value. Lower is better.

    Raises
    ------
    ValueError
        If either array is empty, not two-dimensional, or the two fronts have a
        differing number of objectives.

    Examples
    --------
    >>> true = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> igd_plus(true, true)
    0.0
    >>> approx = np.array([[0.0, 2.0], [2.0, 0.0]])
    >>> igd_plus(approx, true)
    1.0
    """
    approx, true = _validate_fronts(approx_front, true_front)
    return float(IGDPlus(true)(approx))


def gd(approx_front: np.ndarray, true_front: np.ndarray) -> float:
    """Generational Distance (GD) to the analytical front.

    GD measures the average distance from each approximation point to the
    nearest point of the true Pareto front. Lower is better; ``0`` means every
    approximation point lies on the true front. Computed via ``pymoo`` as
    ``GD(true_front)(approx_front)``.

    Parameters
    ----------
    approx_front : numpy.ndarray
        Objective values of the Pareto-front approximation, shape
        ``(n, n_obj)``.
    true_front : numpy.ndarray
        Analytical true Pareto front (reference set), shape ``(m, n_obj)``.

    Returns
    -------
    float
        The GD value. Lower is better.

    Raises
    ------
    ValueError
        If either array is empty, not two-dimensional, or the two fronts have a
        differing number of objectives.

    Examples
    --------
    >>> true = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> gd(true, true)
    0.0
    >>> approx = np.array([[0.0, 2.0], [2.0, 0.0]])
    >>> gd(approx, true)
    1.0
    """
    approx, true = _validate_fronts(approx_front, true_front)
    return float(GD(true)(approx))


def hypervolume(objectives: np.ndarray, ref_point: np.ndarray | None = None) -> float:
    """Compute hypervolume indicator.

    The hypervolume (or S-metric) measures the volume of objective space
    dominated by the Pareto front approximation and bounded by a reference point.
    Higher values indicate better convergence and diversity.

    Parameters
    ----------
    objectives : numpy.ndarray
        Objective values of the Pareto front approximation with shape
        ``(n, n_obj)``.
    ref_point : numpy.ndarray, optional
        Reference point with one component per objective. Defaults to
        ``[1.1, 1.1]``, valid only for **2-objective** ZDT problems. For an
        n-objective problem (e.g. the 3-objective DTLZ2) the caller MUST pass a
        dimension-correct ``ref_point`` (length ``n_obj``) taken from the
        problem's s2 metadata; the default is never appropriate beyond 2-D.

    Returns
    -------
    float
        Hypervolume value. Higher is better for minimization problems.

    Raises
    ------
    ValueError
        If ``objectives`` is empty or is not two-dimensional.

    Examples
    --------
    >>> objs = np.array([[0.2, 0.8], [0.8, 0.2]])
    >>> round(hypervolume(objs, ref_point=np.array([1.1, 1.1])), 2)
    0.45
    """
    if objectives.size == 0:
        raise ValueError("objectives array cannot be empty")

    if objectives.ndim != 2:
        raise ValueError(f"objectives must be 2D array, got shape {objectives.shape}")

    if ref_point is None:
        ref_point = np.array([1.1, 1.1])

    indicator = HV(ref_point=ref_point)
    return float(indicator(objectives))


def _validate_fronts(approx_front: np.ndarray, true_front: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a pair of objective-space fronts.

    Shared guard for the IGD+/GD wrappers, mirroring the empty/2-D checks used
    by :func:`hypervolume` and additionally requiring a matching number of
    objectives.

    Parameters
    ----------
    approx_front : numpy.ndarray
        Approximation front, expected shape ``(n, n_obj)``.
    true_front : numpy.ndarray
        True/reference front, expected shape ``(m, n_obj)``.

    Returns
    -------
    tuple of numpy.ndarray
        The two fronts as ``float`` arrays ``(approx, true)``.

    Raises
    ------
    ValueError
        If either array is empty, not two-dimensional, or the fronts have a
        differing number of objectives.

    Examples
    --------
    >>> a, b = _validate_fronts(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]))
    >>> a.shape, b.shape
    ((1, 2), (1, 2))
    """
    approx = np.asarray(approx_front, dtype=float)
    true = np.asarray(true_front, dtype=float)
    if approx.size == 0 or true.size == 0:
        raise ValueError("front arrays cannot be empty")
    if approx.ndim != 2 or true.ndim != 2:
        raise ValueError(f"fronts must be 2D arrays, got shapes {approx.shape} and {true.shape}")
    if approx.shape[1] != true.shape[1]:
        raise ValueError(f"fronts must have the same number of objectives, got {approx.shape[1]} and {true.shape[1]}")
    return approx, true
