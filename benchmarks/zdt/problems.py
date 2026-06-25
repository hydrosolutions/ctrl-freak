"""ZDT test problems for multi-objective optimization benchmarking.

The ZDT (Zitzler-Deb-Thiele) test suite is a standard benchmark for
multi-objective evolutionary algorithms. All problems have:
- n decision variables in [0, 1]
- 2 objectives to minimize
- Known Pareto-optimal fronts for validation

References
----------
Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
evolutionary algorithms: Empirical results. Evolutionary Computation, 8(2),
173-195.
"""

from collections.abc import Callable

import numpy as np

# Problem configuration
N_VARS: int = 30
BOUNDS: tuple[float, float] = (0.0, 1.0)


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1: Convex Pareto front.

    The Pareto-optimal front is formed by x_i = 0 for i > 1,
    resulting in f2 = 1 - sqrt(f1).

    Parameters
    ----------
    x : numpy.ndarray
        Decision variables with shape ``(n_vars,)`` and values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Two objectives to minimize.

    Examples
    --------
    >>> zdt1(np.zeros(N_VARS))
    array([0., 1.])
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    """ZDT2: Non-convex (concave) Pareto front.

    The Pareto-optimal front is formed by x_i = 0 for i > 1,
    resulting in f2 = 1 - (f1)^2.

    Parameters
    ----------
    x : numpy.ndarray
        Decision variables with shape ``(n_vars,)`` and values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Two objectives to minimize.

    Examples
    --------
    >>> zdt2(np.zeros(N_VARS))
    array([0., 1.])
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - (f1 / g) ** 2
    f2 = g * h
    return np.array([f1, f2])


def zdt3(x: np.ndarray) -> np.ndarray:
    """ZDT3: Discontinuous Pareto front.

    The Pareto front consists of several disconnected convex parts
    due to the sine term in the h function.

    Parameters
    ----------
    x : numpy.ndarray
        Decision variables with shape ``(n_vars,)`` and values in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Two objectives to minimize.

    Examples
    --------
    >>> zdt3(np.zeros(N_VARS))
    array([0., 1.])
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return np.array([f1, f2])


# Registry of all ZDT problems
PROBLEMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "zdt1": zdt1,
    "zdt2": zdt2,
    "zdt3": zdt3,
}
