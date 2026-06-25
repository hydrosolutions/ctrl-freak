"""Tests for genetic operators module.

Tests the three core functions:
- lift: Applies per-individual functions to populations
- select_parents: Binary tournament selection with crowded comparison
- create_offspring: Creates offspring via selection, crossover, mutation
"""

import numpy as np
import pytest

from ctrl_freak import crowding_distance as compute_crowding_distance
from ctrl_freak.operators import create_offspring, lift, lift_parallel, select_parents
from ctrl_freak.operators.standard import polynomial_mutation, sbx_crossover
from ctrl_freak.population import Population

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def default_bounds() -> tuple[float, float]:
    """Default bounds for testing."""
    return (0.0, 1.0)


@pytest.fixture
def custom_bounds() -> tuple[float, float]:
    """Custom bounds for testing."""
    return (-5.0, 5.0)


@pytest.fixture
def per_variable_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Per-variable bounds for 3 decision variables."""
    lower = np.array([0.0, -10.0, 100.0])
    upper = np.array([1.0, 10.0, 200.0])
    return (lower, upper)


def _compute_crowding_for_all_fronts(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Compute crowding distance for all individuals across all fronts."""
    cd = np.zeros(len(objectives), dtype=np.float64)
    for r in range(int(ranks.max()) + 1):
        mask = ranks == r
        cd[mask] = compute_crowding_distance(objectives[mask])
    return cd


@pytest.fixture
def simple_population() -> tuple[Population, np.ndarray, np.ndarray]:
    """Population with rank and crowding_distance computed separately.

    Creates a population of 6 individuals with varying ranks and crowding distances
    to test selection behavior.

    Returns:
        Tuple of (population, rank, crowding_distance).
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

    pop = Population(x=x, objectives=objectives)
    return pop, rank, crowding_distance


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
# TestPolynomialMutationDegenerateBounds
# =============================================================================


class TestPolynomialMutationDegenerateBounds:
    """Regression tests for fixed variables in polynomial mutation."""

    def test_lower_equals_upper_no_nan(self) -> None:
        """A fixed variable (lower==upper) must never produce NaN via the real mutate path."""
        lower = np.array([0.0, 5.0])
        upper = np.array([1.0, 5.0])
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=(lower, upper), seed=42)
        x = np.array([0.5, 5.0])
        out = mutate(x)
        assert not np.any(np.isnan(out)), f"NaN produced: {out}"
        np.testing.assert_array_equal(out[1], 5.0)
        assert lower[0] <= out[0] <= upper[0]


def test_legacy_shim_removed() -> None:
    """The deprecated ctrl_freak.standard_operators shim is removed."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ctrl_freak.standard_operators")


class TestSeedInjection:
    """Contract tests for post-construction Generator injection."""

    def test_set_rng_overrides_seed(self) -> None:
        """Injected rng should replace the constructor seed for SBX."""
        p1 = np.array([0.2, 0.4, 0.6])
        p2 = np.array([0.3, 0.5, 0.7])
        seeded = sbx_crossover(seed=1)
        injected = sbx_crossover(seed=999)
        injected.set_rng(np.random.default_rng(1))

        np.testing.assert_array_equal(seeded(p1, p2), injected(p1, p2))

    def test_set_rng_present_on_both_operators(self) -> None:
        """Both standard operators expose the seed-injection hook."""
        assert hasattr(sbx_crossover(), "set_rng")
        assert hasattr(polynomial_mutation(), "set_rng")

    def test_standalone_seed_path_preserved(self) -> None:
        """Two standalone operators with the same seed produce the same sequence."""
        p1 = np.array([0.2, 0.4, 0.6])
        p2 = np.array([0.3, 0.5, 0.7])
        crossover1 = sbx_crossover(seed=7)
        crossover2 = sbx_crossover(seed=7)

        children1 = [crossover1(p1, p2) for _ in range(5)]
        children2 = [crossover2(p1, p2) for _ in range(5)]

        for child1, child2 in zip(children1, children2, strict=True):
            np.testing.assert_array_equal(child1, child2)

    def test_injected_generator_advances(self) -> None:
        """Injected generators advance across calls."""
        p1 = np.array([0.2, 0.4, 0.6])
        p2 = np.array([0.3, 0.5, 0.7])
        crossover = sbx_crossover(seed=999)
        crossover.set_rng(np.random.default_rng(1))

        child1 = crossover(p1, p2)
        child2 = crossover(p1, p2)

        assert not np.allclose(child1, child2)

    def test_set_rng_overrides_seed_mutation(self) -> None:
        """Injected rng should replace the constructor seed for mutation."""
        x = np.array([0.2, 0.4, 0.6])
        seeded = polynomial_mutation(prob=1.0, seed=1)
        injected = polynomial_mutation(prob=1.0, seed=999)
        injected.set_rng(np.random.default_rng(1))

        np.testing.assert_array_equal(seeded(x), injected(x))


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

        low_eta_crossover = sbx_crossover(eta=2.0, bounds=bounds, seed=seed)
        low_eta_distances = [np.linalg.norm(low_eta_crossover(p1, p2) - parent_midpoint) for _ in range(n_samples)]

        high_eta_crossover = sbx_crossover(eta=50.0, bounds=bounds, seed=seed)
        high_eta_distances = [np.linalg.norm(high_eta_crossover(p1, p2) - parent_midpoint) for _ in range(n_samples)]

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

        assert not np.allclose(crossover1(p1, p2), crossover2(p1, p2))

    def test_identical_parents(self, default_bounds: tuple[float, float]) -> None:
        """When parents are identical, child should exactly equal the parent."""
        crossover = sbx_crossover(eta=15.0, bounds=default_bounds, seed=42)
        p = np.array([0.5, 0.5, 0.5])

        for _ in range(50):
            child = crossover(p, p)
            np.testing.assert_array_equal(child, p)

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

    def test_child_within_per_variable_bounds(self, per_variable_bounds: tuple[np.ndarray, np.ndarray]) -> None:
        """Children should respect element-wise per-variable bounds."""
        lower, upper = per_variable_bounds
        crossover = sbx_crossover(eta=15.0, bounds=per_variable_bounds, seed=42)

        p1 = np.array([0.5, 0.0, 150.0])
        p2 = np.array([0.8, -5.0, 180.0])

        for _ in range(100):
            child = crossover(p1, p2)
            assert np.all(child >= lower), f"Child below lower bound: {child}"
            assert np.all(child <= upper), f"Child above upper bound: {child}"

    def test_per_variable_bounds_children_use_full_range(
        self, per_variable_bounds: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """With low eta and extreme parents, children should span most of each variable's range."""
        lower, upper = per_variable_bounds
        crossover = sbx_crossover(eta=2.0, bounds=per_variable_bounds, seed=42)

        p1 = lower.copy()
        p2 = upper.copy()
        children = np.array([crossover(p1, p2) for _ in range(500)])

        for i in range(len(lower)):
            child_range = children[:, i].max() - children[:, i].min()
            variable_range = upper[i] - lower[i]
            assert child_range > 0.5 * variable_range, (
                f"Variable {i}: children span {child_range:.4f} but variable range is {variable_range:.4f}"
            )

    def test_per_variable_bounds_deterministic_with_seed(
        self, per_variable_bounds: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Same seed should produce identical children with array bounds."""
        p1 = np.array([0.5, 0.0, 150.0])
        p2 = np.array([0.8, -5.0, 180.0])

        crossover1 = sbx_crossover(eta=15.0, bounds=per_variable_bounds, seed=12345)
        crossover2 = sbx_crossover(eta=15.0, bounds=per_variable_bounds, seed=12345)

        children1 = [crossover1(p1, p2) for _ in range(10)]
        children2 = [crossover2(p1, p2) for _ in range(10)]

        for c1, c2 in zip(children1, children2, strict=True):
            np.testing.assert_array_equal(c1, c2)


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
        mutate = polynomial_mutation(eta=20.0, prob=None, bounds=(0.0, 1.0), seed=42)
        x = np.full(n_vars, 0.5)

        total_changes = 0
        for _ in range(n_trials):
            mutated = mutate(x.copy())
            total_changes += np.sum(mutated != x)

        actual_avg = total_changes / n_trials
        assert 0.5 < actual_avg < 2.0, f"Expected ~1.0 mutations, got {actual_avg:.2f}"

    def test_high_probability_mutates_all(self) -> None:
        """With prob=1.0, all variables should be mutated."""
        n_vars = 5
        n_trials = 50
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=(0.0, 1.0), seed=42)
        x = np.full(n_vars, 0.5)

        change_counts = []
        for _ in range(n_trials):
            mutated = mutate(x)
            change_counts.append(np.sum(mutated != x))

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

        low_eta_mutate = polynomial_mutation(eta=5.0, prob=1.0, bounds=bounds, seed=42)
        low_eta_distances = [np.linalg.norm(low_eta_mutate(x) - x) for _ in range(n_samples)]

        high_eta_mutate = polynomial_mutation(eta=100.0, prob=1.0, bounds=bounds, seed=42)
        high_eta_distances = [np.linalg.norm(high_eta_mutate(x) - x) for _ in range(n_samples)]

        avg_low_eta = np.mean(low_eta_distances)
        avg_high_eta = np.mean(high_eta_distances)

        assert avg_high_eta < avg_low_eta, (
            f"High eta should produce smaller mutations. Low eta avg: {avg_low_eta:.4f}, "
            f"High eta avg: {avg_high_eta:.4f}"
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

        assert not np.allclose(mutate1(x), mutate2(x))

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

    def test_mutated_within_per_variable_bounds(self, per_variable_bounds: tuple[np.ndarray, np.ndarray]) -> None:
        """Mutated values should respect element-wise per-variable bounds."""
        lower, upper = per_variable_bounds
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=per_variable_bounds, seed=42)

        x = np.array([0.5, 0.0, 150.0])

        for _ in range(100):
            mutated = mutate(x)
            assert np.all(mutated >= lower), f"Below lower bound: {mutated}"
            assert np.all(mutated <= upper), f"Above upper bound: {mutated}"

    def test_per_variable_bounds_mutation_magnitude_scales(
        self, per_variable_bounds: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Variable with range 100 should have >10x larger avg mutation than variable with range 1."""
        mutate = polynomial_mutation(eta=20.0, prob=1.0, bounds=per_variable_bounds, seed=42)

        x = np.array([0.5, 0.0, 150.0])
        deltas = np.array([np.abs(mutate(x) - x) for _ in range(1000)])
        avg_deltas = deltas.mean(axis=0)

        assert avg_deltas[2] > 10.0 * avg_deltas[0], (
            f"Expected variable 2 (range 100) to have >10x larger mutations than "
            f"variable 0 (range 1). Got avg_deltas={avg_deltas}"
        )

    def test_per_variable_bounds_deterministic_with_seed(
        self, per_variable_bounds: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Same seed should produce identical mutations with array bounds."""
        x = np.array([0.5, 0.0, 150.0])

        mutate1 = polynomial_mutation(eta=20.0, prob=1.0, bounds=per_variable_bounds, seed=12345)
        mutate2 = polynomial_mutation(eta=20.0, prob=1.0, bounds=per_variable_bounds, seed=12345)

        results1 = [mutate1(x) for _ in range(10)]
        results2 = [mutate2(x) for _ in range(10)]

        for r1, r2 in zip(results1, results2, strict=True):
            np.testing.assert_array_equal(r1, r2)


class TestStandardOperatorIntegration:
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

        assert result.population.x.shape == (20, n_vars)
        assert result.population.objectives.shape == (20, 2)
        assert np.all(result.population.x >= bounds[0])
        assert np.all(result.population.x <= bounds[1])

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

        assert result.population.x.shape == (20, n_vars)
        assert result.population.objectives.shape == (20, 2)
        assert np.all(result.population.x >= bounds[0])
        assert np.all(result.population.x <= bounds[1])

    def test_both_operators_together(self) -> None:
        """Both standard operators should work together in nsga2."""
        from ctrl_freak import non_dominated_sort, nsga2

        n_vars = 10
        bounds = (0.0, 1.0)

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(bounds[0], bounds[1], size=n_vars)

        def evaluate(x: np.ndarray) -> np.ndarray:
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

        assert result.population.x.shape == (50, n_vars)
        assert result.population.objectives.shape == (50, 2)
        assert np.all(result.population.x >= bounds[0])
        assert np.all(result.population.x <= bounds[1])

        ranks = non_dominated_sort(result.population.objectives)
        assert np.sum(ranks == 0) > 0

    def test_both_operators_with_per_variable_bounds(self) -> None:
        """Full NSGA-II run with per-variable array bounds should keep all individuals in bounds."""
        from ctrl_freak import nsga2

        lower = np.array([0.0, -10.0, 100.0])
        upper = np.array([1.0, 10.0, 200.0])
        bounds = (lower, upper)
        n_vars = len(lower)

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(lower, upper)

        def evaluate(x: np.ndarray) -> np.ndarray:
            f1 = np.sum((x - lower) / (upper - lower))
            f2 = np.sum((upper - x) / (upper - lower))
            return np.array([f1, f2])

        crossover = sbx_crossover(eta=15.0, bounds=bounds, seed=100)
        mutate = polynomial_mutation(eta=20.0, bounds=bounds, seed=200)

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=10,
            seed=42,
        )

        assert result.population.x.shape == (20, n_vars)
        assert np.all(result.population.x >= lower), "Some individuals below lower bounds"
        assert np.all(result.population.x <= upper), "Some individuals above upper bounds"


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


class TestLiftParallel:
    """Tests for the lift_parallel function."""

    def test_applies_to_each_row(self) -> None:
        """Lifted parallel function should apply to each row of input."""

        def double(x: np.ndarray) -> np.ndarray:
            return x * 2

        lifted_fn = lift_parallel(double, n_workers=2)
        pop_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = lifted_fn(pop_x)

        expected = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_output_shape(self) -> None:
        """Output shape should be (n, output_dim)."""

        def identity(x: np.ndarray) -> np.ndarray:
            return x

        lifted_fn = lift_parallel(identity, n_workers=2)
        pop_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (2, 3)

    def test_handles_dimension_change(self) -> None:
        """Lifted function should handle output dimension different from input."""

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum(), x.prod()])

        lifted_fn = lift_parallel(evaluate, n_workers=2)
        pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (2, 2)
        expected = np.array([[3.0, 2.0], [7.0, 12.0]])
        np.testing.assert_array_equal(result, expected)

    def test_equivalence_with_serial_lift(self) -> None:
        """Parallel lift should produce same results as serial lift."""

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum(), x.mean(), x.max()])

        pop_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        serial_result = lift(evaluate)(pop_x)
        parallel_result = lift_parallel(evaluate, n_workers=2)(pop_x)

        np.testing.assert_array_equal(serial_result, parallel_result)

    def test_single_row_input(self) -> None:
        """Lifted function should work with single row input."""

        def square(x: np.ndarray) -> np.ndarray:
            return x**2

        lifted_fn = lift_parallel(square, n_workers=2)
        pop_x = np.array([[2.0, 3.0]])

        result = lifted_fn(pop_x)

        assert result.shape == (1, 2)
        expected = np.array([[4.0, 9.0]])
        np.testing.assert_array_equal(result, expected)

    def test_n_workers_all_cores(self) -> None:
        """n_workers=-1 should work (uses all CPU cores)."""

        def double(x: np.ndarray) -> np.ndarray:
            return x * 2

        lifted_fn = lift_parallel(double, n_workers=-1)
        pop_x = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = lifted_fn(pop_x)

        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_dtype(self) -> None:
        """Lifted function should preserve the dtype of the output."""

        def to_int(x: np.ndarray) -> np.ndarray:
            return x.astype(np.int32)

        lifted_fn = lift_parallel(to_int, n_workers=2)
        pop_x = np.array([[1.5, 2.5], [3.5, 4.5]])

        result = lifted_fn(pop_x)

        assert result.dtype == np.int32


# =============================================================================
# TestSelectParents
# =============================================================================


class TestSelectParents:
    """Tests for binary tournament parent selection."""

    def test_returns_correct_shape(
        self, simple_population: tuple[Population, np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """Should return array of shape (n_parents,)."""
        pop, rank, crowding_distance = simple_population
        result = select_parents(pop, n_parents=10, rng=rng, rank=rank, crowding_distance=crowding_distance)

        assert result.shape == (10,)

    def test_returns_valid_indices(
        self, simple_population: tuple[Population, np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> None:
        """All returned indices should be valid population indices."""
        pop, rank, crowding_distance = simple_population
        result = select_parents(pop, n_parents=100, rng=rng, rank=rank, crowding_distance=crowding_distance)

        assert np.all(result >= 0)
        assert np.all(result < len(pop.x))

    def test_prefers_lower_rank(self, rng: np.random.Generator) -> None:
        """Lower rank individuals should be preferred in tournaments."""
        # Create population where rank clearly determines winner
        x = np.array([[0.0], [1.0]])
        objectives = np.array([[0.5], [0.5]])
        rank = np.array([0, 1], dtype=np.int64)  # First is clearly better
        crowding_distance = np.array([1.0, 1.0])  # Equal CD

        pop = Population(x=x, objectives=objectives)

        # With many trials, the rank-0 individual should win most tournaments
        # When comparing 0 vs 1, rank 0 always wins
        np.random.seed(42)
        test_rng = np.random.default_rng(42)
        result = select_parents(pop, n_parents=1000, rng=test_rng, rank=rank, crowding_distance=crowding_distance)

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

        pop = Population(x=x, objectives=objectives)

        test_rng = np.random.default_rng(42)
        result = select_parents(pop, n_parents=1000, rng=test_rng, rank=rank, crowding_distance=crowding_distance)

        count_0 = np.sum(result == 0)
        count_1 = np.sum(result == 1)

        # Higher CD (individual 0) should win when they compete
        assert count_0 > count_1

    def test_deterministic_with_seed(self, simple_population: tuple[Population, np.ndarray, np.ndarray]) -> None:
        """Same seed should produce identical results."""
        pop, rank, crowding_distance = simple_population
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        result1 = select_parents(pop, n_parents=50, rng=rng1, rank=rank, crowding_distance=crowding_distance)
        result2 = select_parents(pop, n_parents=50, rng=rng2, rank=rank, crowding_distance=crowding_distance)

        np.testing.assert_array_equal(result1, result2)

    def test_equal_cd_first_wins(self) -> None:
        """When rank and CD are equal, first candidate wins (>= on CD)."""
        rank = np.array([0, 0], dtype=np.int64)
        crowding_distance = np.array([1.0, 1.0])  # Exactly equal

        # Create controlled test: fix candidates to always be [0, 1]
        # With >= comparison, candidate 0 (first) should win
        # Manually verify the logic
        candidates = np.array([[0, 1]])
        rank_a = rank[candidates[:, 0]]  # 0
        rank_b = rank[candidates[:, 1]]  # 0
        cd_a = crowding_distance[candidates[:, 0]]  # 1.0
        cd_b = crowding_distance[candidates[:, 1]]  # 1.0

        a_wins = (rank_a < rank_b) | ((rank_a == rank_b) & (cd_a >= cd_b))

        assert a_wins[0]  # First candidate should win with >= comparison


# =============================================================================
# TestCreateOffspring
# =============================================================================


class TestCreateOffspring:
    """Tests for offspring creation via selection, crossover, and mutation."""

    def test_returns_correct_shape(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        identity_crossover: callable,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Should return array of shape (n_offspring, n_vars)."""
        pop, rank, crowding_distance = simple_population
        result = create_offspring(
            pop,
            n_offspring=5,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert result.shape == (5, 2)

    def test_calls_crossover_correct_times(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        tracking_crossover: tuple[callable, list],
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Crossover should be called exactly n_offspring times."""
        pop, rank, crowding_distance = simple_population
        crossover_fn, crossover_calls = tracking_crossover

        create_offspring(
            pop,
            n_offspring=7,
            crossover=crossover_fn,
            mutate=identity_mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert len(crossover_calls) == 7

    def test_calls_mutate_correct_times(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        identity_crossover: callable,
        tracking_mutate: tuple[callable, list],
        rng: np.random.Generator,
    ) -> None:
        """Mutation should be called exactly n_offspring times."""
        pop, rank, crowding_distance = simple_population
        mutate_fn, mutate_calls = tracking_mutate

        create_offspring(
            pop,
            n_offspring=7,
            crossover=identity_crossover,
            mutate=mutate_fn,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert len(mutate_calls) == 7

    def test_crossover_receives_valid_parents(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        tracking_crossover: tuple[callable, list],
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Crossover should receive valid individuals from population."""
        pop, rank, crowding_distance = simple_population
        crossover_fn, crossover_calls = tracking_crossover

        create_offspring(
            pop,
            n_offspring=10,
            crossover=crossover_fn,
            mutate=identity_mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Each call should have parents from the population
        for p1, p2 in crossover_calls:
            assert p1.shape == (2,)
            assert p2.shape == (2,)
            # Check that p1 and p2 are actual rows from the population
            assert any(np.array_equal(p1, pop.x[i]) for i in range(len(pop.x)))
            assert any(np.array_equal(p2, pop.x[i]) for i in range(len(pop.x)))

    def test_mutation_receives_crossover_output(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Mutation should receive the output of crossover."""
        pop, rank, crowding_distance = simple_population
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
            pop,
            n_offspring=3,
            crossover=crossover,
            mutate=mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Mutation inputs should match crossover outputs
        assert len(crossover_output) == len(mutate_input) == 3
        for co, mi in zip(crossover_output, mutate_input, strict=True):
            np.testing.assert_array_equal(co, mi)

    def test_deterministic_with_seed(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        identity_crossover: callable,
        identity_mutate: callable,
    ) -> None:
        """Same seed should produce identical offspring."""
        pop, rank, crowding_distance = simple_population
        rng1 = np.random.default_rng(99999)
        rng2 = np.random.default_rng(99999)

        result1 = create_offspring(
            pop,
            n_offspring=10,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng1,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        result2 = create_offspring(
            pop,
            n_offspring=10,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng2,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        np.testing.assert_array_equal(result1, result2)

    def test_single_offspring(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        identity_crossover: callable,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Should work with n_offspring=1."""
        pop, rank, crowding_distance = simple_population
        result = create_offspring(
            pop,
            n_offspring=1,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert result.shape == (1, 2)

    def test_large_offspring_count(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        identity_crossover: callable,
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Should work with more offspring than population size."""
        pop, rank, crowding_distance = simple_population
        result = create_offspring(
            pop,
            n_offspring=100,
            crossover=identity_crossover,
            mutate=identity_mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert result.shape == (100, 2)

    def test_applies_crossover_to_pairs(
        self,
        simple_population: tuple[Population, np.ndarray, np.ndarray],
        identity_mutate: callable,
        rng: np.random.Generator,
    ) -> None:
        """Crossover should be applied to consecutive pairs of selected parents."""
        pop, rank, crowding_distance = simple_population
        pair_indices: list[tuple[int, int]] = []

        def tracking_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            # Find indices of p1 and p2 in population
            idx1 = None
            idx2 = None
            for i in range(len(pop.x)):
                if np.array_equal(p1, pop.x[i]):
                    idx1 = i
                if np.array_equal(p2, pop.x[i]):
                    idx2 = i
            pair_indices.append((idx1, idx2))
            return (p1 + p2) / 2

        create_offspring(
            pop,
            n_offspring=3,
            crossover=tracking_crossover,
            mutate=identity_mutate,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Should have 3 pairs
        assert len(pair_indices) == 3
        # All indices should be valid
        for idx1, idx2 in pair_indices:
            assert idx1 is not None
            assert idx2 is not None
            assert 0 <= idx1 < len(pop.x)
            assert 0 <= idx2 < len(pop.x)
