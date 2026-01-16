"""NSGA-II primitives for Pareto-based ranking and diversity.

This package provides core pure functions for multi-objective optimization.
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
