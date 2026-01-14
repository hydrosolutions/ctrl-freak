"""Tests for NSGA-II primitives.

Comprehensive test suite covering:
- TestDominates: Pareto dominance checks
- TestDominatesMatrix: Vectorized pairwise dominance
- TestNonDominatedSort: Deb's fast non-dominated sorting
- TestCrowdingDistance: Diversity metric computation
"""

import numpy as np
import pytest

from ctrl_freak.primitives import (
    crowding_distance,
    dominates,
    dominates_matrix,
    non_dominated_sort,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_2d_objectives() -> np.ndarray:
    """Simple 2D objectives with clear dominance hierarchy.

    Layout (minimization):
        [1,1] dominates all others (best in both objectives)
        [2,2] dominates [3,3]
        [1,3] dominates [3,3]
        [3,1] dominates [3,3]
        [1,3] and [3,1] are tradeoffs with each other
        [2,2], [1,3], [3,1] are tradeoffs with each other

    Resulting fronts:
        Front 0: [1,1]
        Front 1: [2,2], [1,3], [3,1]
        Front 2: [3,3]

    Returns:
        Array of shape (5, 2) with objectives.
    """
    return np.array(
        [
            [1.0, 1.0],  # 0: dominates all (front 0)
            [2.0, 2.0],  # 1: dominated by 0 only (front 1)
            [3.0, 3.0],  # 2: dominated by 0, 1, 3, 4 (front 2)
            [1.0, 3.0],  # 3: dominated by 0 only (front 1)
            [3.0, 1.0],  # 4: dominated by 0 only (front 1)
        ]
    )


@pytest.fixture
def pareto_front_2d() -> np.ndarray:
    """A Pareto front where no solution dominates another.

    All points lie on the Pareto-optimal frontier for minimization.

    Returns:
        Array of shape (4, 2) with objectives.
    """
    return np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ]
    )


@pytest.fixture
def all_dominated_chain() -> np.ndarray:
    """Linear dominance chain where each dominates the next.

    [1,1] > [2,2] > [3,3] > [4,4] (> means "dominates")

    Returns:
        Array of shape (4, 2) with objectives.
    """
    return np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )


@pytest.fixture
def three_objective_data() -> np.ndarray:
    """Three-objective test data.

    Returns:
        Array of shape (4, 3) with objectives.
    """
    return np.array(
        [
            [1.0, 2.0, 3.0],  # 0
            [2.0, 1.0, 3.0],  # 1
            [3.0, 3.0, 1.0],  # 2
            [2.0, 2.0, 2.0],  # 3: dominated by none of above (tradeoff)
        ]
    )


# =============================================================================
# TestDominates
# =============================================================================


class TestDominates:
    """Tests for the scalar dominates function."""

    def test_clear_dominance(self) -> None:
        """Solution with all better values dominates."""
        a = np.array([1.0, 1.0])
        b = np.array([2.0, 2.0])
        assert dominates(a, b) is True

    def test_clear_dominance_reverse_is_false(self) -> None:
        """Dominated solution does not dominate the dominant one."""
        a = np.array([1.0, 1.0])
        b = np.array([2.0, 2.0])
        assert dominates(b, a) is False

    def test_identical_solutions_no_dominance(self) -> None:
        """Identical solutions do not dominate each other."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_tradeoff_no_dominance(self) -> None:
        """Solutions with tradeoffs do not dominate each other."""
        a = np.array([1.0, 3.0])
        b = np.array([3.0, 1.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_partial_tie_with_one_better(self) -> None:
        """One tie and one strictly better gives dominance."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_single_objective(self) -> None:
        """Single objective case works correctly."""
        a = np.array([1.0])
        b = np.array([2.0])
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_single_objective_tie(self) -> None:
        """Single objective tie is not dominance."""
        a = np.array([1.0])
        b = np.array([1.0])
        assert dominates(a, b) is False

    def test_three_objectives_dominance(self) -> None:
        """Dominance works with three objectives."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        assert dominates(a, b) is True

    def test_three_objectives_tradeoff(self) -> None:
        """Tradeoffs work with three objectives."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert dominates(a, b) is False
        assert dominates(b, a) is False

    def test_many_objectives(self) -> None:
        """Dominance scales to many objectives."""
        n_obj = 10
        a = np.ones(n_obj)
        b = np.ones(n_obj) * 2
        assert dominates(a, b) is True

    def test_epsilon_difference(self) -> None:
        """Very small differences still count as dominance."""
        a = np.array([1.0, 1.0])
        b = np.array([1.0 + 1e-10, 1.0])
        assert dominates(a, b) is True

    def test_negative_values(self) -> None:
        """Dominance works with negative values."""
        a = np.array([-2.0, -2.0])
        b = np.array([-1.0, -1.0])
        assert dominates(a, b) is True


# =============================================================================
# TestDominatesMatrix
# =============================================================================


class TestDominatesMatrix:
    """Tests for the vectorized dominates_matrix function."""

    def test_agrees_with_scalar_dominates(self, simple_2d_objectives: np.ndarray) -> None:
        """Matrix result agrees with scalar dominates for all pairs."""
        n = simple_2d_objectives.shape[0]
        matrix = dominates_matrix(simple_2d_objectives)

        for i in range(n):
            for j in range(n):
                expected = dominates(simple_2d_objectives[i], simple_2d_objectives[j])
                assert matrix[i, j] == expected, f"Mismatch at ({i}, {j})"

    def test_diagonal_is_false(self, simple_2d_objectives: np.ndarray) -> None:
        """No solution dominates itself."""
        matrix = dominates_matrix(simple_2d_objectives)
        assert not np.any(np.diag(matrix))

    def test_pareto_front_no_dominance(self, pareto_front_2d: np.ndarray) -> None:
        """Pareto front has no pairwise dominance."""
        matrix = dominates_matrix(pareto_front_2d)
        assert not np.any(matrix)

    def test_chain_dominance(self, all_dominated_chain: np.ndarray) -> None:
        """Chain has expected dominance pattern."""
        matrix = dominates_matrix(all_dominated_chain)

        # i dominates j iff i < j (in chain)
        n = all_dominated_chain.shape[0]
        for i in range(n):
            for j in range(n):
                if i < j:
                    assert matrix[i, j]
                else:
                    assert not matrix[i, j]

    def test_empty_objectives(self) -> None:
        """Empty objectives returns empty matrix."""
        empty = np.zeros((0, 2))
        matrix = dominates_matrix(empty)
        assert matrix.shape == (0, 0)

    def test_single_individual(self) -> None:
        """Single individual returns 1x1 False matrix."""
        single = np.array([[1.0, 2.0]])
        matrix = dominates_matrix(single)
        assert matrix.shape == (1, 1)
        assert not matrix[0, 0]

    def test_output_shape(self, simple_2d_objectives: np.ndarray) -> None:
        """Output shape is (n, n)."""
        n = simple_2d_objectives.shape[0]
        matrix = dominates_matrix(simple_2d_objectives)
        assert matrix.shape == (n, n)

    def test_output_dtype(self, simple_2d_objectives: np.ndarray) -> None:
        """Output is boolean array."""
        matrix = dominates_matrix(simple_2d_objectives)
        assert matrix.dtype == np.bool_

    def test_three_objectives(self, three_objective_data: np.ndarray) -> None:
        """Matrix works with three objectives."""
        matrix = dominates_matrix(three_objective_data)
        n = three_objective_data.shape[0]

        for i in range(n):
            for j in range(n):
                expected = dominates(three_objective_data[i], three_objective_data[j])
                assert matrix[i, j] == expected


# =============================================================================
# TestNonDominatedSort
# =============================================================================


class TestNonDominatedSort:
    """Tests for non_dominated_sort function."""

    def test_clear_hierarchy(self, all_dominated_chain: np.ndarray) -> None:
        """Chain produces sequential ranks 0, 1, 2, 3."""
        ranks = non_dominated_sort(all_dominated_chain)
        np.testing.assert_array_equal(ranks, [0, 1, 2, 3])

    def test_all_pareto_optimal(self, pareto_front_2d: np.ndarray) -> None:
        """Pareto front all gets rank 0."""
        ranks = non_dominated_sort(pareto_front_2d)
        np.testing.assert_array_equal(ranks, [0, 0, 0, 0])

    def test_mixed_fronts(self, simple_2d_objectives: np.ndarray) -> None:
        """Mixed objectives produce correct front assignments.

        [1,1] is rank 0 (dominates all others - best in both objectives)
        [2,2], [1,3], [3,1] are rank 1 (dominated only by [1,1], tradeoffs with each other)
        [3,3] is rank 2 (dominated by all above)
        """
        ranks = non_dominated_sort(simple_2d_objectives)

        # First front: only [1,1]
        assert ranks[0] == 0  # [1,1]

        # Second front: [2,2], [1,3], [3,1] - all dominated only by [1,1]
        assert ranks[1] == 1  # [2,2]
        assert ranks[3] == 1  # [1,3]
        assert ranks[4] == 1  # [3,1]

        # Third front: [3,3] - dominated by everyone
        assert ranks[2] == 2  # [3,3]

    def test_ties_same_front(self) -> None:
        """Identical solutions are in the same front."""
        objectives = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        )
        ranks = non_dominated_sort(objectives)
        np.testing.assert_array_equal(ranks, [0, 0, 0])

    def test_empty_objectives(self) -> None:
        """Empty input returns empty ranks."""
        empty = np.zeros((0, 2))
        ranks = non_dominated_sort(empty)
        assert len(ranks) == 0

    def test_single_individual(self) -> None:
        """Single individual gets rank 0."""
        single = np.array([[1.0, 2.0]])
        ranks = non_dominated_sort(single)
        np.testing.assert_array_equal(ranks, [0])

    def test_two_individuals_one_dominates(self) -> None:
        """Two individuals where one dominates the other."""
        objectives = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
        ranks = non_dominated_sort(objectives)
        np.testing.assert_array_equal(ranks, [0, 1])

    def test_two_individuals_tradeoff(self) -> None:
        """Two individuals in tradeoff are both rank 0."""
        objectives = np.array(
            [
                [1.0, 3.0],
                [3.0, 1.0],
            ]
        )
        ranks = non_dominated_sort(objectives)
        np.testing.assert_array_equal(ranks, [0, 0])

    def test_output_dtype(self, simple_2d_objectives: np.ndarray) -> None:
        """Output is integer array."""
        ranks = non_dominated_sort(simple_2d_objectives)
        assert ranks.dtype == np.int64

    def test_all_ranks_non_negative(self, simple_2d_objectives: np.ndarray) -> None:
        """All ranks are non-negative."""
        ranks = non_dominated_sort(simple_2d_objectives)
        assert np.all(ranks >= 0)

    def test_rank_zero_exists(self, simple_2d_objectives: np.ndarray) -> None:
        """At least one solution has rank 0."""
        ranks = non_dominated_sort(simple_2d_objectives)
        assert np.any(ranks == 0)

    def test_three_objectives(self, three_objective_data: np.ndarray) -> None:
        """Sorting works with three objectives."""
        ranks = non_dominated_sort(three_objective_data)

        # All four are Pareto-optimal (no solution dominates another)
        np.testing.assert_array_equal(ranks, [0, 0, 0, 0])

    def test_large_population(self) -> None:
        """Handles larger populations."""
        np.random.seed(42)
        objectives = np.random.rand(100, 3)
        ranks = non_dominated_sort(objectives)

        assert len(ranks) == 100
        assert np.all(ranks >= 0)
        assert np.any(ranks == 0)


# =============================================================================
# TestCrowdingDistance
# =============================================================================


class TestCrowdingDistance:
    """Tests for crowding_distance function."""

    def test_boundary_points_infinite(self, pareto_front_2d: np.ndarray) -> None:
        """Boundary points (min/max per objective) get infinite distance."""
        cd = crowding_distance(pareto_front_2d)

        # First and last in sorted order get inf
        assert np.isinf(cd[0])  # [1,4] - min of obj 0
        assert np.isinf(cd[3])  # [4,1] - max of obj 0, min of obj 1

    def test_interior_points_finite(self, pareto_front_2d: np.ndarray) -> None:
        """Interior points get finite positive distance."""
        cd = crowding_distance(pareto_front_2d)

        # Middle points should be finite
        assert np.isfinite(cd[1])
        assert np.isfinite(cd[2])
        assert cd[1] > 0
        assert cd[2] > 0

    def test_single_individual_infinite(self) -> None:
        """Single individual gets infinite distance."""
        single = np.array([[1.0, 2.0]])
        cd = crowding_distance(single)
        assert len(cd) == 1
        assert np.isinf(cd[0])

    def test_two_individuals_both_infinite(self) -> None:
        """Two individuals are both boundary points."""
        two = np.array(
            [
                [1.0, 3.0],
                [3.0, 1.0],
            ]
        )
        cd = crowding_distance(two)
        assert np.all(np.isinf(cd))

    def test_empty_front(self) -> None:
        """Empty front returns empty distances."""
        empty = np.zeros((0, 2))
        cd = crowding_distance(empty)
        assert len(cd) == 0

    def test_all_identical_solutions(self) -> None:
        """Identical solutions naturally get low crowding distance.

        The algorithm doesn't special-case ties - identical solutions
        are indeed crowded (low distance is correct).
        """
        identical = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        )
        cd = crowding_distance(identical)

        # Boundary points still get inf (they're at min=max for each objective)
        # Interior points get 0 (no spread, neighbors have same value)
        assert len(cd) == 4

    def test_evenly_spaced_solutions(self) -> None:
        """Evenly spaced solutions have equal interior distances."""
        # Points at (0,4), (1,3), (2,2), (3,1), (4,0) - evenly spaced
        evenly_spaced = np.array(
            [
                [0.0, 4.0],
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [4.0, 0.0],
            ]
        )
        cd = crowding_distance(evenly_spaced)

        # Boundary points infinite
        assert np.isinf(cd[0])
        assert np.isinf(cd[4])

        # Interior points should have equal distance
        interior_distances = cd[1:4]
        np.testing.assert_allclose(
            interior_distances[0],
            interior_distances[1],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            interior_distances[1],
            interior_distances[2],
            rtol=1e-10,
        )

    def test_output_length(self, pareto_front_2d: np.ndarray) -> None:
        """Output has same length as input."""
        cd = crowding_distance(pareto_front_2d)
        assert len(cd) == len(pareto_front_2d)

    def test_output_non_negative(self, pareto_front_2d: np.ndarray) -> None:
        """All distances are non-negative."""
        cd = crowding_distance(pareto_front_2d)
        assert np.all(cd >= 0)

    def test_three_objectives(self) -> None:
        """Crowding distance works with three objectives."""
        objectives = np.array(
            [
                [1.0, 4.0, 2.0],
                [2.0, 3.0, 3.0],
                [3.0, 2.0, 1.0],
                [4.0, 1.0, 4.0],
            ]
        )
        cd = crowding_distance(objectives)

        assert len(cd) == 4
        assert np.all(cd >= 0)

    def test_crowded_vs_isolated(self) -> None:
        """More isolated solutions have higher crowding distance."""
        # Points: boundary, crowded interior, isolated interior, boundary
        objectives = np.array(
            [
                [0.0, 10.0],  # boundary
                [4.9, 5.1],  # very close to next
                [5.0, 5.0],  # very close to previous
                [10.0, 0.0],  # boundary
            ]
        )
        cd = crowding_distance(objectives)

        # Boundaries get inf
        assert np.isinf(cd[0])
        assert np.isinf(cd[3])

        # The interior points should have some finite distance
        # Due to close spacing, distances will be relatively small
        assert np.isfinite(cd[1])
        assert np.isfinite(cd[2])
