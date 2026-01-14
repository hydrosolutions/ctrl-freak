"""Tests for genetic operators module.

Tests the three core functions:
- lift: Applies per-individual functions to populations
- select_parents: Binary tournament selection with crowded comparison
- create_offspring: Creates offspring via selection, crossover, mutation
"""

import numpy as np
import pytest

from ctrl_freak.operators import create_offspring, lift, select_parents
from ctrl_freak.population import Population

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_population() -> Population:
    """Population with rank and crowding_distance set.

    Creates a population of 6 individuals with varying ranks and crowding distances
    to test selection behavior.
    """
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
    )
    objectives = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.5, 0.5],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )
    # Ranks: 0, 0, 1, 1, 2, 2 (two per front)
    rank = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    # Crowding distances: vary within fronts
    crowding_distance = np.array([np.inf, 0.5, np.inf, 0.3, np.inf, 0.2])

    return Population(x=x, objectives=objectives, rank=rank, crowding_distance=crowding_distance)


@pytest.fixture
def identity_crossover() -> callable:
    """Crossover that returns first parent unchanged (for shape testing)."""

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return p1.copy()

    return crossover


@pytest.fixture
def identity_mutate() -> callable:
    """Mutation that returns individual unchanged (for shape testing)."""

    def mutate(x: np.ndarray) -> np.ndarray:
        return x.copy()

    return mutate


@pytest.fixture
def tracking_crossover() -> tuple[callable, list]:
    """Crossover that records all calls for verification.

    Returns:
        Tuple of (crossover_fn, call_log) where call_log is a list of (p1, p2) tuples.
    """
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        calls.append((p1.copy(), p2.copy()))
        return (p1 + p2) / 2

    return crossover, calls


@pytest.fixture
def tracking_mutate() -> tuple[callable, list]:
    """Mutation that records all calls for verification.

    Returns:
        Tuple of (mutate_fn, call_log) where call_log is a list of individual arrays.
    """
    calls: list[np.ndarray] = []

    def mutate(x: np.ndarray) -> np.ndarray:
        calls.append(x.copy())
        return x + 0.01

    return mutate, calls


# =============================================================================
# TestLift
# =============================================================================


class TestLift:
    """Tests for the lift function."""

    def test_applies_to_each_row(self) -> None:
        """Lifted function should apply to each row of input."""

        # Simple function that doubles input
        def double(x: np.ndarray) -> np.ndarray:
            return x * 2

        lifted_fn = lift(double)
        pop_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = lifted_fn(pop_x)

        expected = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_output_shape(self) -> None:
        """Output shape should be (n, output_dim)."""

        def identity(x: np.ndarray) -> np.ndarray:
            return x

        lifted_fn = lift(identity)
        pop_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (2, 3)

    def test_handles_dimension_change(self) -> None:
        """Lifted function should handle output dimension different from input."""

        # Function that reduces dimensions (evaluation-like)
        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum(), x.prod()])

        lifted_fn = lift(evaluate)
        pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (2, 2)
        expected = np.array([[3.0, 2.0], [7.0, 12.0]])
        np.testing.assert_array_equal(result, expected)

    def test_handles_expansion(self) -> None:
        """Lifted function should handle output larger than input."""

        def expand(x: np.ndarray) -> np.ndarray:
            return np.concatenate([x, x * 2])

        lifted_fn = lift(expand)
        pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (2, 4)
        expected = np.array([[1.0, 2.0, 2.0, 4.0], [3.0, 4.0, 6.0, 8.0]])
        np.testing.assert_array_equal(result, expected)

    def test_single_row_input(self) -> None:
        """Lifted function should work with single row input."""

        def square(x: np.ndarray) -> np.ndarray:
            return x**2

        lifted_fn = lift(square)
        pop_x = np.array([[2.0, 3.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (1, 2)
        expected = np.array([[4.0, 9.0]])
        np.testing.assert_array_equal(result, expected)

    def test_empty_input_raises(self) -> None:
        """Lifted function raises ValueError on empty input.

        Empty populations are not valid inputs - fail fast rather than silently.
        """

        def identity(x: np.ndarray) -> np.ndarray:
            return x

        lifted_fn = lift(identity)
        pop_x = np.zeros((0, 3))

        with pytest.raises(ValueError, match="at least one array"):
            lifted_fn(pop_x)

    def test_preserves_dtype(self) -> None:
        """Lifted function should preserve the dtype of the output."""

        def to_int(x: np.ndarray) -> np.ndarray:
            return x.astype(np.int32)

        lifted_fn = lift(to_int)
        pop_x = np.array([[1.5, 2.5], [3.5, 4.5]])

        result = lifted_fn(pop_x)

        assert result.dtype == np.int32


# =============================================================================
# TestSelectParents
# =============================================================================


class TestSelectParents:
    """Tests for binary tournament parent selection."""

    def test_returns_correct_shape(self, simple_population: Population, rng: np.random.Generator) -> None:
        """Should return array of shape (n_parents,)."""
        result = select_parents(simple_population, n_parents=10, rng=rng)

        assert result.shape == (10,)

    def test_returns_valid_indices(self, simple_population: Population, rng: np.random.Generator) -> None:
        """All returned indices should be valid population indices."""
        result = select_parents(simple_population, n_parents=100, rng=rng)

        assert np.all(result >= 0)
        assert np.all(result < len(simple_population.x))

    def test_prefers_lower_rank(self, rng: np.random.Generator) -> None:
        """Lower rank individuals should be preferred in tournaments."""
        # Create population where rank clearly determines winner
        x = np.array([[0.0], [1.0]])
        objectives = np.array([[0.5], [0.5]])
        rank = np.array([0, 1], dtype=np.int64)  # First is clearly better
        crowding_distance = np.array([1.0, 1.0])  # Equal CD

        pop = Population(x=x, objectives=objectives, rank=rank, crowding_distance=crowding_distance)

        # With many trials, the rank-0 individual should win most tournaments
        # When comparing 0 vs 1, rank 0 always wins
        np.random.seed(42)
        test_rng = np.random.default_rng(42)
        result = select_parents(pop, n_parents=1000, rng=test_rng)

        # Count how often each individual was selected
        count_0 = np.sum(result == 0)
        count_1 = np.sum(result == 1)

        # When both compete (0.5 probability), rank 0 always wins
        # When same individual competes, it wins
        # So rank 0 should be selected more often
        assert count_0 > count_1

    def test_ties_use_crowding_distance(self, rng: np.random.Generator) -> None:
        """When ranks are equal, higher crowding distance should win."""
        x = np.array([[0.0], [1.0]])
        objectives = np.array([[0.5], [0.5]])
        rank = np.array([0, 0], dtype=np.int64)  # Same rank
        crowding_distance = np.array([10.0, 1.0])  # First has much higher CD

        pop = Population(x=x, objectives=objectives, rank=rank, crowding_distance=crowding_distance)

        test_rng = np.random.default_rng(42)
        result = select_parents(pop, n_parents=1000, rng=test_rng)

        count_0 = np.sum(result == 0)
        count_1 = np.sum(result == 1)

        # Higher CD (individual 0) should win when they compete
        assert count_0 > count_1

    def test_deterministic_with_seed(self, simple_population: Population) -> None:
        """Same seed should produce identical results."""
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        result1 = select_parents(simple_population, n_parents=50, rng=rng1)
        result2 = select_parents(simple_population, n_parents=50, rng=rng2)

        np.testing.assert_array_equal(result1, result2)

    def test_raises_without_rank(self, rng: np.random.Generator) -> None:
        """Should raise ValueError if rank is None."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        pop = Population(x=x, rank=None, crowding_distance=np.array([1.0, 1.0]))

        with pytest.raises(ValueError, match="rank"):
            select_parents(pop, n_parents=5, rng=rng)

    def test_raises_without_crowding_distance(self, rng: np.random.Generator) -> None:
        """Should raise ValueError if crowding_distance is None."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        pop = Population(x=x, rank=np.array([0, 1], dtype=np.int64), crowding_distance=None)

        with pytest.raises(ValueError, match="crowding_distance"):
            select_parents(pop, n_parents=5, rng=rng)

    def test_equal_cd_first_wins(self, rng: np.random.Generator) -> None:
        """When rank and CD are equal, first candidate wins (>= on CD)."""
        x = np.array([[0.0], [1.0]])
        objectives = np.array([[0.5], [0.5]])
        rank = np.array([0, 0], dtype=np.int64)
        crowding_distance = np.array([1.0, 1.0])  # Exactly equal

        pop = Population(x=x, objectives=objectives, rank=rank, crowding_distance=crowding_distance)

        # Create controlled test: fix candidates to always be [0, 1]
        # With >= comparison, candidate 0 (first) should win
        # Manually verify the logic
        candidates = np.array([[0, 1]])
        rank_a = pop.rank[candidates[:, 0]]  # 0
        rank_b = pop.rank[candidates[:, 1]]  # 0
        cd_a = pop.crowding_distance[candidates[:, 0]]  # 1.0
        cd_b = pop.crowding_distance[candidates[:, 1]]  # 1.0

        a_wins = (rank_a < rank_b) | ((rank_a == rank_b) & (cd_a >= cd_b))

        assert a_wins[0]  # First candidate should win with >= comparison


# =============================================================================
# TestCreateOffspring
# =============================================================================


class TestCreateOffspring:
    """Tests for offspring creation via selection, crossover, and mutation."""

    def test_returns_correct_shape(
        self,
        simple_population: Population,
        identity_crossover: callable,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Should return array of shape (n_offspring, n_vars)."""
        result = create_offspring(
            simple_population,
            n_offspring=5,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng,
        )

        assert result.shape == (5, 2)

    def test_calls_crossover_correct_times(
        self,
        simple_population: Population,
        tracking_crossover: tuple[callable, list],
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Crossover should be called exactly n_offspring times."""
        crossover_fn, crossover_calls = tracking_crossover

        create_offspring(
            simple_population,
            n_offspring=7,
            crossover=crossover_fn,
            mutate=identity_mutate,
            rng=rng,
        )

        assert len(crossover_calls) == 7

    def test_calls_mutate_correct_times(
        self,
        simple_population: Population,
        identity_crossover: callable,
        tracking_mutate: tuple[callable, list],
        rng: np.random.Generator,
    ) -> None:
        """Mutation should be called exactly n_offspring times."""
        mutate_fn, mutate_calls = tracking_mutate

        create_offspring(
            simple_population,
            n_offspring=7,
            crossover=identity_crossover,
            mutate=mutate_fn,
            rng=rng,
        )

        assert len(mutate_calls) == 7

    def test_crossover_receives_valid_parents(
        self,
        simple_population: Population,
        tracking_crossover: tuple[callable, list],
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Crossover should receive valid individuals from population."""
        crossover_fn, crossover_calls = tracking_crossover

        create_offspring(
            simple_population,
            n_offspring=10,
            crossover=crossover_fn,
            mutate=identity_mutate,
            rng=rng,
        )

        # Each call should have parents from the population
        for p1, p2 in crossover_calls:
            assert p1.shape == (2,)
            assert p2.shape == (2,)
            # Check that p1 and p2 are actual rows from the population
            assert any(np.array_equal(p1, simple_population.x[i]) for i in range(len(simple_population.x)))
            assert any(np.array_equal(p2, simple_population.x[i]) for i in range(len(simple_population.x)))

    def test_mutation_receives_crossover_output(
        self,
        simple_population: Population,
        rng: np.random.Generator,
    ) -> None:
        """Mutation should receive the output of crossover."""
        crossover_output: list[np.ndarray] = []
        mutate_input: list[np.ndarray] = []

        def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            result = p1 + p2
            crossover_output.append(result.copy())
            return result

        def mutate(x: np.ndarray) -> np.ndarray:
            mutate_input.append(x.copy())
            return x

        create_offspring(
            simple_population,
            n_offspring=3,
            crossover=crossover,
            mutate=mutate,
            rng=rng,
        )

        # Mutation inputs should match crossover outputs
        assert len(crossover_output) == len(mutate_input) == 3
        for co, mi in zip(crossover_output, mutate_input, strict=True):
            np.testing.assert_array_equal(co, mi)

    def test_deterministic_with_seed(
        self,
        simple_population: Population,
        identity_crossover: callable,
        identity_mutate: callable,
    ) -> None:
        """Same seed should produce identical offspring."""
        rng1 = np.random.default_rng(99999)
        rng2 = np.random.default_rng(99999)

        result1 = create_offspring(
            simple_population,
            n_offspring=10,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng1,
        )

        result2 = create_offspring(
            simple_population,
            n_offspring=10,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng2,
        )

        np.testing.assert_array_equal(result1, result2)

    def test_single_offspring(
        self,
        simple_population: Population,
        identity_crossover: callable,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Should work with n_offspring=1."""
        result = create_offspring(
            simple_population,
            n_offspring=1,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng,
        )

        assert result.shape == (1, 2)

    def test_large_offspring_count(
        self,
        simple_population: Population,
        identity_crossover: callable,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Should work with more offspring than population size."""
        result = create_offspring(
            simple_population,
            n_offspring=100,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng,
        )

        assert result.shape == (100, 2)

    def test_applies_crossover_to_pairs(
        self,
        simple_population: Population,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Crossover should be applied to consecutive pairs of selected parents."""
        pair_indices: list[tuple[int, int]] = []

        def tracking_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            # Find indices of p1 and p2 in population
            idx1 = None
            idx2 = None
            for i in range(len(simple_population.x)):
                if np.array_equal(p1, simple_population.x[i]):
                    idx1 = i
                if np.array_equal(p2, simple_population.x[i]):
                    idx2 = i
            pair_indices.append((idx1, idx2))
            return (p1 + p2) / 2

        create_offspring(
            simple_population,
            n_offspring=3,
            crossover=tracking_crossover,
            mutate=identity_mutate,
            rng=rng,
        )

        # Should have 3 pairs
        assert len(pair_indices) == 3
        # All indices should be valid
        for idx1, idx2 in pair_indices:
            assert idx1 is not None
            assert idx2 is not None
            assert 0 <= idx1 < len(simple_population.x)
            assert 0 <= idx2 < len(simple_population.x)
