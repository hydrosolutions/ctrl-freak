"""Base genetic operators.

This module provides the core lift decorator for genetic operators.
"""

from collections.abc import Callable

import numpy as np


def lift(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Lift a per-individual function to work on a population.

    This utility allows users to write simple per-individual functions
    while the framework handles batching/vectorization.

    Args:
        fn: Function that operates on a single individual.
            Signature: (n_vars,) -> (n_out,)

    Returns:
        A function that operates on a population.
        Signature: (n, n_vars) -> (n, n_out)

    Example:
        >>> def evaluate_one(x: np.ndarray) -> np.ndarray:
        ...     return np.array([x.sum(), x.prod()])
        >>> evaluate = lift(evaluate_one)
        >>> pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> evaluate(pop_x)
        array([[ 3.,  2.],
               [ 7., 12.]])
    """

    def lifted(x: np.ndarray) -> np.ndarray:
        return np.stack([fn(x[i]) for i in range(x.shape[0])])

    return lifted


def lift_parallel(
    fn: Callable[[np.ndarray], np.ndarray], n_workers: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Lift a per-individual function to work on a population with parallel execution.

    This utility allows users to write simple per-individual functions
    while the framework handles batching/vectorization with parallel workers.

    Args:
        fn: Function that operates on a single individual.
            Signature: (n_vars,) -> (n_out,)
            Must be picklable for multiprocessing.
        n_workers: Number of parallel workers. Use -1 for all CPU cores.

    Returns:
        A function that operates on a population in parallel.
        Signature: (n, n_vars) -> (n, n_out)

    Example:
        >>> def evaluate_one(x: np.ndarray) -> np.ndarray:
        ...     return np.array([x.sum(), x.prod()])
        >>> evaluate = lift_parallel(evaluate_one, n_workers=4)
        >>> pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> evaluate(pop_x)
        array([[ 3.,  2.],
               [ 7., 12.]])
    """
    from joblib import Parallel, delayed

    def lifted(x: np.ndarray) -> np.ndarray:
        results: list[np.ndarray] = Parallel(n_jobs=n_workers)(  # type: ignore[assignment]
            delayed(fn)(x[i]) for i in range(x.shape[0])
        )
        return np.stack(results)

    return lifted
