"""NSGA-II primitives for Pareto-based ranking and diversity.

This module provides the core pure functions for NSGA-II:
- dominates: scalar Pareto dominance check
- dominates_matrix: vectorized pairwise dominance
- non_dominated_sort: Deb's fast non-dominated sorting algorithm
- crowding_distance: diversity metric for solutions in a Pareto front
"""

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if solution a Pareto-dominates solution b (minimization).

    A solution a dominates b if and only if:
      - a[i] <= b[i] for ALL objectives (a is at least as good everywhere)
      - a[i] < b[i] for AT LEAST ONE objective (a is strictly better somewhere)

    Args:
        a: Objective values for solution a. Shape (n_obj,).
        b: Objective values for solution b. Shape (n_obj,).

    Returns:
        True if a dominates b, False otherwise.

    Examples:
        >>> dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        True
        >>> dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
        False
    """
    return bool(np.all(a <= b) and np.any(a < b))


def dominates_matrix(objectives: np.ndarray) -> np.ndarray:
    """Compute pairwise dominance for all individuals (vectorized).

    Uses broadcasting to efficiently compute whether individual i dominates
    individual j for all pairs (i, j).

    Args:
        objectives: Objective values for all individuals. Shape (n, n_obj).

    Returns:
        Boolean array of shape (n, n) where result[i, j] = True iff
        individual i dominates individual j.

    Examples:
        >>> objs = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
        >>> dom = dominates_matrix(objs)
        >>> dom[0, 1]  # Does [1,1] dominate [2,2]?
        True
        >>> dom[0, 2]  # Does [1,1] dominate [1,2]?
        True
    """
    # Reshape for broadcasting: (n, 1, n_obj) vs (1, n, n_obj)
    a = objectives[:, np.newaxis, :]  # (n, 1, n_obj)
    b = objectives[np.newaxis, :, :]  # (1, n, n_obj)

    # a[i] <= b[j] for all objectives
    all_leq = np.all(a <= b, axis=2)  # (n, n)

    # a[i] < b[j] for at least one objective
    any_lt = np.any(a < b, axis=2)  # (n, n)

    return all_leq & any_lt


def non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
    """Assign each individual to a Pareto front using Deb's fast algorithm.

    Implements the fast non-dominated sorting algorithm from NSGA-II.
    Time complexity: O(M * N^2) where M = number of objectives, N = population size.

    Args:
        objectives: Objective values for all individuals. Shape (n, n_obj).

    Returns:
        Integer array of shape (n,) where rank[i] is the front index for
        individual i. Rank 0 = Pareto optimal (first front), rank 1 = second
        front, etc.

    Examples:
        >>> objs = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        >>> non_dominated_sort(objs)
        array([0, 1, 2])
    """
    n = objectives.shape[0]

    if n == 0:
        return np.array([], dtype=np.int64)

    # Compute pairwise dominance matrix
    dom_matrix = dominates_matrix(objectives)

    # domination_count[i] = number of individuals that dominate i
    domination_count = dom_matrix.sum(axis=0)

    # Initialize ranks array
    ranks = np.full(n, -1, dtype=np.int64)

    current_rank = 0
    remaining = np.arange(n)

    while len(remaining) > 0:
        # Find individuals with zero domination count (current front)
        front_mask = domination_count[remaining] == 0
        front = remaining[front_mask]

        if len(front) == 0:
            # Safety check: if no front found but individuals remain,
            # assign remaining to current rank (shouldn't happen with valid data)
            ranks[remaining] = current_rank
            break

        # Assign rank to front members
        ranks[front] = current_rank

        # Remove front members from remaining
        remaining = remaining[~front_mask]

        # Update domination counts: subtract dominance from front members
        for idx in front:
            # For each remaining individual, if this front member dominated them,
            # decrement their count
            dominated_by_idx = dom_matrix[idx, remaining]
            domination_count[remaining] -= dominated_by_idx.astype(np.int64)

        current_rank += 1

    return ranks


def crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for individuals in a single Pareto front.

    Crowding distance measures how isolated a solution is in objective space.
    Higher values indicate more isolated solutions (preferred for diversity).

    Boundary solutions (with min/max values for any objective) receive
    infinite distance. Interior solutions receive the sum of normalized
    neighbor distances across all objectives.

    Args:
        front_objectives: Objective values for individuals in ONE front only.
            Shape (n_front, n_obj).

    Returns:
        Array of shape (n_front,) containing crowding distances.
        Higher values indicate more isolated (preferred) solutions.

    Examples:
        >>> objs = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        >>> cd = crowding_distance(objs)
        >>> np.isinf(cd[0]) and np.isinf(cd[-1])  # Boundary points
        True
    """
    n_front = front_objectives.shape[0]

    if n_front == 0:
        return np.array([], dtype=np.float64)

    if n_front == 1:
        # Single individual gets infinite distance (it's both min and max)
        return np.array([np.inf])

    if n_front == 2:
        # Two individuals are both boundary points
        return np.array([np.inf, np.inf])

    n_obj = front_objectives.shape[1]
    distances = np.zeros(n_front, dtype=np.float64)

    for m in range(n_obj):
        # Sort indices by objective m
        sorted_indices = np.argsort(front_objectives[:, m])

        # Objective range for normalization
        obj_min = front_objectives[sorted_indices[0], m]
        obj_max = front_objectives[sorted_indices[-1], m]
        obj_range = obj_max - obj_min

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Interior points: add normalized neighbor distance
        if obj_range > 0:
            for i in range(1, n_front - 1):
                prev_idx = sorted_indices[i - 1]
                curr_idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]

                neighbor_dist = front_objectives[next_idx, m] - front_objectives[prev_idx, m]
                distances[curr_idx] += neighbor_dist / obj_range

    return distances
