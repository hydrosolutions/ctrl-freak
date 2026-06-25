"""Pareto-based ranking and diversity primitives.

Examples
--------
>>> import numpy as np
>>> ranks = non_dominated_sort(np.array([[1.0, 1.0], [2.0, 2.0]]))
>>> ranks
array([0, 1])
"""

from ctrl_freak.primitives.pareto import (
    crowding_distance,
    dominates,
    dominates_matrix,
    non_dominated_sort,
)

__all__ = [
    "dominates",
    "dominates_matrix",
    "non_dominated_sort",
    "crowding_distance",
]
