"""Performance metrics for multi-objective optimization.

This module provides metrics for evaluating the quality of Pareto front
approximations, primarily using the hypervolume indicator.
"""

import numpy as np
from pymoo.indicators.hv import HV


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
        Reference point. Defaults to ``[1.1, 1.1]`` for ZDT problems.

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
