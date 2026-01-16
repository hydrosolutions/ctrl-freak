"""Tests for standard genetic operators module.

Tests the factory functions:
- sbx_crossover: Simulated Binary Crossover operator
- polynomial_mutation: Polynomial mutation operator
"""

import numpy as np
import pytest

from ctrl_freak.standard_operators import polynomial_mutation, sbx_crossover

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_bounds() -> tuple[float, float]:
    """Default bounds for testing."""
    return (0.0, 1.0)


@pytest.fixture
def custom_bounds() -> tuple[float, float]:
    """Custom bounds for testing."""
    return (-5.0, 5.0)


# =============================================================================
# TestSBXCrossover
# =============================================================================


class TestSBXCrossover:
    """Tests for SBX crossover operator."""

    def test_returns_callable(self, default_bounds: tuple[float, float]) -> None:
        """sbx_crossover should return a callable function."""
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds)

        assert callable(crossover)

    def test_returns_correct_shape(self, default_bounds: tuple[float, float]) -> None:
        """Crossover should return array with same shape as parents."""
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds, seed=42)
        p1 = np.array([0.2, 0.4, 0.6])
        p2 = np.array([0.3, 0.5, 0.7])

        child = crossover(p1, p2)

        assert child.shape == p1.shape
        assert child.shape == (3,)

    def test_child_within_bounds(self, default_bounds: tuple[float, float]) -> None:
        """Children should always be within specified bounds."""
        lower, upper = default_bounds
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds, seed=42)

        p1 = np.array([0.1, 0.9, 0.5])
        p2 = np.array([0.8, 0.2, 0.5])

        for _ in range(100):
            child = crossover(p1, p2)
            assert np.all(child >= lower), f"Child below lower bound: {child}"
            assert np.all(child <= upper), f"Child above upper bound: {child}"

    def test_child_within_custom_bounds(self, custom_bounds: tuple[float, float]) -> None:
        """Children should respect custom bounds."""
        lower, upper = custom_bounds
        crossover = sbx_crossover(eta=15.0, bounds=custom_bounds, seed=42)

        p1 = np.array([-4.0, 0.0, 4.0])
        p2 = np.array([4.0, 0.0, -4.0])

        for _ in range(100):
            child = crossover(p1, p2)
            assert np.all(child >= lower), f"Child below lower bound: {child}"
            assert np.all(child <= upper), f"Child above upper bound: {child}"

    def test_high_eta_produces_closer_children(self) -> None:
        """Higher eta should produce children closer to parents."""
        bounds = (0.0, 1.0)
        n_samples = 500
        seed = 42

        p1 = np.array([0.3, 0.5, 0.7])
        p2 = np.array([0.4, 0.6, 0.8])
        parent_midpoint = (p1 + p2) / 2

        # Low eta - more exploration
        low_eta_crossover = sbx_crossover(eta=2.0, bounds=bounds, seed=seed)
        low_eta_distances = []
        for _ in range(n_samples):
            child = low_eta_crossover(p1, p2)
            low_eta_distances.append(np.linalg.norm(child - parent_midpoint))

        # High eta - children closer to parents
        high_eta_crossover = sbx_crossover(eta=50.0, bounds=bounds, seed=seed)
        high_eta_distances = []
        for _ in range(n_samples):
            child = high_eta_crossover(p1, p2)
            high_eta_distances.append(np.linalg.norm(child - parent_midpoint))

        # High eta should produce smaller average distance from parent midpoint
        avg_low_eta = np.mean(low_eta_distances)
        avg_high_eta = np.mean(high_eta_distances)

        assert avg_high_eta < avg_low_eta, (
            f"High eta should produce closer children. Low eta avg: {avg_low_eta:.4f}, High eta avg: {avg_high_eta:.4f}"
        )

    def test_deterministic_with_seed(self, default_bounds: tuple[float, float]) -> None:
        """Same seed should produce identical results."""
        p1 = np.array([0.2, 0.4, 0.6])
        p2 = np.array([0.3, 0.5, 0.7])

        crossover1 = sbx_crossover(eta=15.0, bounds=default_bounds, seed=12345)
        crossover2 = sbx_crossover(eta=15.0, bounds=default_bounds, seed=12345)

        children1 = [crossover1(p1, p2) for _ in range(10)]
        children2 = [crossover2(p1, p2) for _ in range(10)]

        for c1, c2 in zip(children1, children2, strict=True):
            np.testing.assert_array_equal(c1, c2)

    def test_different_seeds_produce_different_results(self, default_bounds: tuple[float, float]) -> None:
        """Different seeds should produce different results."""
        p1 = np.array([0.2, 0.4, 0.6])
        p2 = np.array([0.3, 0.5, 0.7])

        crossover1 = sbx_crossover(eta=15.0, bounds=default_bounds, seed=111)
        crossover2 = sbx_crossover(eta=15.0, bounds=default_bounds, seed=222)

        child1 = crossover1(p1, p2)
        child2 = crossover2(p1, p2)

        # Should be different
        assert not np.allclose(child1, child2)

    def test_identical_parents(self, default_bounds: tuple[float, float]) -> None:
        """When parents are identical, child should be close to parents."""
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds, seed=42)
        p = np.array([0.5, 0.5, 0.5])

        # With high eta and identical parents, child should be close to parent
        for _ in range(50):
            child = crossover(p, p)
            # Child may drift slightly due to SBX formula, but should be bounded
            assert np.all(np.abs(child - p) < 0.5)

    def test_single_variable(self, default_bounds: tuple[float, float]) -> None:
        """Should work with single-variable individuals."""
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds, seed=42)
        p1 = np.array([0.2])
        p2 = np.array([0.8])

        child = crossover(p1, p2)

        assert child.shape == (1,)
        assert 0.0 <= child[0] <= 1.0

    def test_extreme_parents_at_bounds(self, default_bounds: tuple[float, float]) -> None:
        """Should handle parents at boundary values."""
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds, seed=42)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 1.0])

        for _ in range(100):
            child = crossover(p1, p2)
            assert np.all(child >= 0.0)
            assert np.all(child <= 1.0)


# =============================================================================
# TestPolynomialMutation
# =============================================================================


class TestPolynomialMutation:
    """Tests for polynomial mutation operator."""

    def test_returns_callable(self, default_bounds: tuple[float, float]) -> None:
        """polynomial_mutation should return a callable function."""
        mutate = polynomial_mutation(eta=20.0, bounds=default_bounds)

        assert callable(mutate)

    def test_returns_correct_shape(self, default_bounds: tuple[float, float]) -> None:
        """Mutation should return array with same shape as input."""
        mutate = polynomial_mutation(eta=20.0, bounds=default_bounds, seed=42)
        x = np.array([0.2, 0.4, 0.6])

        mutated = mutate(x)

        assert mutated.shape == x.shape
        assert mutated.shape == (3,)

    def test_mutated_within_bounds(self, default_bounds: tuple[float, float]) -> None:
        """Mutated values should always be within bounds."""
        lower, upper = default_bounds
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=42)

        x = np.array([0.1, 0.5, 0.9])

        for _ in range(100):
            mutated = mutate(x)
            assert np.all(mutated >= lower), f"Below lower bound: {mutated}"
            assert np.all(mutated <= upper), f"Above upper bound: {mutated}"

    def test_mutated_within_custom_bounds(self, custom_bounds: tuple[float, float]) -> None:
        """Mutated values should respect custom bounds."""
        lower, upper = custom_bounds
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=custom_bounds, seed=42)

        x = np.array([-3.0, 0.0, 3.0])

        for _ in range(100):
            mutated = mutate(x)
            assert np.all(mutated >= lower), f"Below lower bound: {mutated}"
            assert np.all(mutated <= upper), f"Above upper bound: {mutated}"

    def test_default_probability_is_one_over_n(self) -> None:
        """Default mutation probability should be 1/n_vars."""
        n_vars = 10
        n_trials = 1000
        seed = 42

        mutate = polynomial_mutation(eta=20.0, prob=None, bounds=(0.0, 1.0), seed=seed)
        x = np.full(n_vars, 0.5)

        # Count how many variables change per mutation
        total_changes = 0
        for _ in range(n_trials):
            mutated = mutate(x.copy())
            total_changes += np.sum(mutated != x)

        # Expected: 1/n_vars * n_vars = 1 mutation per trial on average
        actual_avg = total_changes / n_trials

        # Should be close to 1.0 mutation per trial on average
        # Allow some variance
        assert 0.5 < actual_avg < 2.0, f"Expected ~1.0 mutations, got {actual_avg:.2f}"

    def test_high_probability_mutates_all(self) -> None:
        """With prob=1.0, all variables should be mutated."""
        n_vars = 5
        n_trials = 50
        seed = 42

        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=(0.0, 1.0), seed=seed)
        x = np.full(n_vars, 0.5)

        # With prob=1.0, at least some change should occur in most cases
        change_counts = []
        for _ in range(n_trials):
            mutated = mutate(x)
            changes = np.sum(mutated != x)
            change_counts.append(changes)

        # All variables should change in most trials
        avg_changes = np.mean(change_counts)
        assert avg_changes > 0.9 * n_vars, f"Expected ~{n_vars} changes, got {avg_changes:.1f}"

    def test_zero_probability_no_changes(self) -> None:
        """With prob=0.0, no variables should be mutated."""
        mutate = polynomial_mutation(eta=20.0, prob=0.0, bounds=(0.0, 1.0), seed=42)
        x = np.array([0.2, 0.4, 0.6, 0.8])

        for _ in range(50):
            mutated = mutate(x)
            np.testing.assert_array_equal(mutated, x)

    def test_high_eta_produces_smaller_mutations(self) -> None:
        """Higher eta should produce smaller perturbations."""
        bounds = (0.0, 1.0)
        n_samples = 500
        x = np.array([0.5, 0.5, 0.5])

        # Low eta - larger mutations
        low_eta_mutate = polynomial_mutation(eta=5.0, prob=1.0, bounds=bounds, seed=42)
        low_eta_distances = []
        for _ in range(n_samples):
            mutated = low_eta_mutate(x)
            low_eta_distances.append(np.linalg.norm(mutated - x))

        # High eta - smaller mutations
        high_eta_mutate = polynomial_mutation(eta=100.0, prob=1.0, bounds=bounds, seed=42)
        high_eta_distances = []
        for _ in range(n_samples):
            mutated = high_eta_mutate(x)
            high_eta_distances.append(np.linalg.norm(mutated - x))

        avg_low_eta = np.mean(low_eta_distances)
        avg_high_eta = np.mean(high_eta_distances)

        assert avg_high_eta < avg_low_eta, (
            f"High eta should produce smaller mutations. "
            f"Low eta avg: {avg_low_eta:.4f}, High eta avg: {avg_high_eta:.4f}"
        )

    def test_deterministic_with_seed(self, default_bounds: tuple[float, float]) -> None:
        """Same seed should produce identical results."""
        x = np.array([0.2, 0.4, 0.6])

        mutate1 = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=12345)
        mutate2 = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=12345)

        results1 = [mutate1(x) for _ in range(10)]
        results2 = [mutate2(x) for _ in range(10)]

        for r1, r2 in zip(results1, results2, strict=True):
            np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_produce_different_results(self, default_bounds: tuple[float, float]) -> None:
        """Different seeds should produce different results."""
        x = np.array([0.2, 0.4, 0.6])

        mutate1 = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=111)
        mutate2 = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=222)

        result1 = mutate1(x)
        result2 = mutate2(x)

        assert not np.allclose(result1, result2)

    def test_does_not_modify_input(self, default_bounds: tuple[float, float]) -> None:
        """Mutation should not modify the input array."""
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=42)
        x = np.array([0.3, 0.5, 0.7])
        x_original = x.copy()

        _ = mutate(x)

        np.testing.assert_array_equal(x, x_original)

    def test_single_variable(self, default_bounds: tuple[float, float]) -> None:
        """Should work with single-variable individuals."""
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=42)
        x = np.array([0.5])

        mutated = mutate(x)

        assert mutated.shape == (1,)
        assert 0.0 <= mutated[0] <= 1.0

    def test_at_boundary_values(self, default_bounds: tuple[float, float]) -> None:
        """Should handle values at boundary correctly."""
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=default_bounds, seed=42)

        x_lower = np.array([0.0, 0.0, 0.0])
        x_upper = np.array([1.0, 1.0, 1.0])

        for _ in range(50):
            mutated_lower = mutate(x_lower)
            mutated_upper = mutate(x_upper)

            assert np.all(mutated_lower >= 0.0)
            assert np.all(mutated_lower <= 1.0)
            assert np.all(mutated_upper >= 0.0)
            assert np.all(mutated_upper <= 1.0)


# =============================================================================
# TestIntegration
# =============================================================================


class TestIntegration:
    """Integration tests for standard operators with nsga2."""

    def test_sbx_compatible_with_nsga2(self) -> None:
        """SBX crossover should work with nsga2."""
        from ctrl_freak import nsga2

        n_vars = 5
        bounds = (0.0, 1.0)

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(bounds[0], bounds[1], size=n_vars)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum(), (1 - x).sum()])

        crossover = sbx_crossover(eta=15.0, bounds=bounds, seed=42)

        def mutate(x: np.ndarray) -> np.ndarray:
            return np.clip(x + 0.01, bounds[0], bounds[1])

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
        )

        assert result.x.shape == (20, n_vars)
        assert result.objectives.shape == (20, 2)
        assert np.all(result.x >= bounds[0])
        assert np.all(result.x <= bounds[1])

    def test_polynomial_mutation_compatible_with_nsga2(self) -> None:
        """Polynomial mutation should work with nsga2."""
        from ctrl_freak import nsga2

        n_vars = 5
        bounds = (0.0, 1.0)

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(bounds[0], bounds[1], size=n_vars)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum(), (1 - x).sum()])

        def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        mutate = polynomial_mutation(eta=20.0, bounds=bounds, seed=42)

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
        )

        assert result.x.shape == (20, n_vars)
        assert result.objectives.shape == (20, 2)
        assert np.all(result.x >= bounds[0])
        assert np.all(result.x <= bounds[1])

    def test_both_operators_together(self) -> None:
        """Both standard operators should work together in nsga2."""
        from ctrl_freak import nsga2

        n_vars = 10
        bounds = (0.0, 1.0)

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(bounds[0], bounds[1], size=n_vars)

        def evaluate(x: np.ndarray) -> np.ndarray:
            # ZDT1-like objective
            f1 = x[0]
            g = 1 + 9 * np.mean(x[1:])
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.array([f1, f2])

        crossover = sbx_crossover(eta=15.0, bounds=bounds, seed=100)
        mutate = polynomial_mutation(eta=20.0, bounds=bounds, seed=200)

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=50,
            n_generations=20,
            seed=42,
        )

        # Check basic properties
        assert result.x.shape == (50, n_vars)
        assert result.objectives.shape == (50, 2)
        assert np.all(result.x >= bounds[0])
        assert np.all(result.x <= bounds[1])

        # Should have some Pareto-optimal solutions
        from ctrl_freak import non_dominated_sort

        ranks = non_dominated_sort(result.objectives)
        assert np.sum(ranks == 0) > 0
