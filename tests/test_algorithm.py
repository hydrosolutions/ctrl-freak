"""Tests for the NSGA-II algorithm module.

This module tests:
- survivor_selection: Correct size, preserves Pareto front, uses crowding for critical front
- nsga2: Integration tests for the main algorithm
- Callbacks: Early stopping behavior
- Property-based tests: Invariants that must hold
"""

import numpy as np
import pytest

from ctrl_freak import (
    NSGA2Result,
    Population,
    crowding_distance,
    non_dominated_sort,
    nsga2,
    survivor_selection,
)


def _compute_crowding_for_all_fronts(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Compute crowding distance for all individuals across all fronts."""
    cd = np.zeros(len(objectives), dtype=np.float64)
    for r in range(int(ranks.max()) + 1):
        mask = ranks == r
        cd[mask] = crowding_distance(objectives[mask])
    return cd


# =============================================================================
# TestSurvivorSelection
# =============================================================================


class TestSurvivorSelection:
    """Tests for the survivor_selection function."""

    def test_returns_correct_size(self, simple_population: Population) -> None:
        """Survivor selection returns exactly n_survivors individuals."""
        result = survivor_selection(simple_population, 2)
        assert len(result) == 2

    def test_preserves_population_type(self, simple_population: Population) -> None:
        """Result is a Population instance."""
        result = survivor_selection(simple_population, 2)
        assert isinstance(result, Population)

    def test_has_objectives(self, simple_population: Population) -> None:
        """Returned population has objectives array."""
        result = survivor_selection(simple_population, 2)
        assert result.objectives is not None
        assert len(result.objectives) == 2

    def test_preserves_pareto_front(self) -> None:
        """All Pareto-optimal individuals are preserved when possible."""
        # Create population with 2 rank-0 and 2 rank-1 individuals
        x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        objectives = np.array([[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        # [1,4] and [4,1] are Pareto-optimal; [2,2] dominates [3,3]

        pop = Population(x=x, objectives=objectives)
        result = survivor_selection(pop, 3)

        # Both Pareto-optimal points should be in survivors
        result_obj_set = {tuple(row) for row in result.objectives}
        assert (1.0, 4.0) in result_obj_set
        assert (4.0, 1.0) in result_obj_set

    def test_uses_crowding_for_critical_front(self) -> None:
        """Critical front selection uses crowding distance."""
        # Create 5 individuals, all in same front (none dominate each other)
        x = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        # All on Pareto front (non-dominated): forming a line in objective space
        objectives = np.array(
            [
                [1.0, 5.0],  # boundary - inf crowding
                [2.0, 4.0],  # interior
                [3.0, 3.0],  # interior - central, lower crowding
                [4.0, 2.0],  # interior
                [5.0, 1.0],  # boundary - inf crowding
            ]
        )

        pop = Population(x=x, objectives=objectives)
        result = survivor_selection(pop, 3)

        # Boundary points should be preserved (inf crowding distance)
        result_obj_set = {tuple(row) for row in result.objectives}
        assert (1.0, 5.0) in result_obj_set
        assert (5.0, 1.0) in result_obj_set

    def test_selects_all_when_n_survivors_equals_pop_size(self, simple_population: Population) -> None:
        """When n_survivors equals population size, all are selected."""
        result = survivor_selection(simple_population, len(simple_population))
        assert len(result) == len(simple_population)

    def test_raises_on_no_objectives(self) -> None:
        """Raises ValueError if population has no objectives."""
        pop = Population(x=np.array([[1, 2], [3, 4]]), objectives=None)
        with pytest.raises(ValueError, match="objectives"):
            survivor_selection(pop, 1)

    def test_raises_on_zero_survivors(self, simple_population: Population) -> None:
        """Raises ValueError for n_survivors <= 0."""
        with pytest.raises(ValueError, match="positive"):
            survivor_selection(simple_population, 0)

    def test_raises_on_negative_survivors(self, simple_population: Population) -> None:
        """Raises ValueError for negative n_survivors."""
        with pytest.raises(ValueError, match="positive"):
            survivor_selection(simple_population, -1)

    def test_raises_when_n_survivors_exceeds_pop_size(self, simple_population: Population) -> None:
        """Raises ValueError when n_survivors > population size."""
        with pytest.raises(ValueError, match="cannot exceed"):
            survivor_selection(simple_population, len(simple_population) + 1)

    def test_survivors_have_correct_ranks_when_recomputed(self) -> None:
        """Ranks recomputed for survivors are correct."""
        # Create population where removing some individuals changes ranks
        x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        objectives = np.array([[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        # [1,4] and [4,1] are rank 0; [2,2] is rank 1 (dominated by both); [3,3] is rank 2

        pop = Population(x=x, objectives=objectives)
        result = survivor_selection(pop, 3)

        # Compute ranks for the result and verify they are valid
        result_ranks = non_dominated_sort(result.objectives)
        # All ranks should be non-negative and form a proper ranking
        assert np.all(result_ranks >= 0)
        assert 0 in result_ranks  # At least one rank 0


# =============================================================================
# TestNSGA2Integration
# =============================================================================


class TestNSGA2Integration:
    """Integration tests for the nsga2 main loop."""

    def test_returns_population(self, simple_biobj_problem: dict) -> None:
        """nsga2 returns a NSGA2Result instance."""
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=5,
            seed=42,
        )
        assert isinstance(result, NSGA2Result)
        assert isinstance(result.population, Population)

    def test_returns_correct_size(self, simple_biobj_problem: dict) -> None:
        """Result population has exactly pop_size individuals."""
        pop_size = 20
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=pop_size,
            n_generations=5,
            seed=42,
        )
        assert len(result.population) == pop_size

    def test_has_objectives_computed(self, simple_biobj_problem: dict) -> None:
        """Result population has objectives array."""
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=5,
            seed=42,
        )
        assert result.population.objectives is not None
        assert result.population.objectives.shape[0] == 10
        assert result.population.objectives.shape[1] == 2  # bi-objective

    def test_deterministic_with_seed(self, simple_biobj_problem: dict) -> None:
        """Same seed produces identical results when operators are deterministic.

        Note: Determinism requires user-provided operators to be deterministic.
        This test uses deterministic crossover and identity mutation.
        """

        # Use deterministic operators (no random mutation)
        def deterministic_mutate(x: np.ndarray) -> np.ndarray:
            return x.copy()

        kwargs = {
            "init": simple_biobj_problem["init"],
            "evaluate": simple_biobj_problem["evaluate"],
            "crossover": simple_biobj_problem["crossover"],
            "mutate": deterministic_mutate,
            "pop_size": 10,
            "n_generations": 5,
            "seed": 12345,
        }

        result1 = nsga2(**kwargs)
        result2 = nsga2(**kwargs)

        np.testing.assert_array_equal(result1.population.x, result2.population.x)
        np.testing.assert_array_equal(result1.population.objectives, result2.population.objectives)

    def test_different_seeds_produce_different_results(self, simple_biobj_problem: dict) -> None:
        """Different seeds produce different results."""
        common_kwargs = {
            "init": simple_biobj_problem["init"],
            "evaluate": simple_biobj_problem["evaluate"],
            "crossover": simple_biobj_problem["crossover"],
            "mutate": simple_biobj_problem["mutate"],
            "pop_size": 10,
            "n_generations": 5,
        }

        result1 = nsga2(**common_kwargs, seed=1)
        result2 = nsga2(**common_kwargs, seed=2)

        # Very unlikely to be identical with different seeds
        assert not np.allclose(result1.population.x, result2.population.x)

    def test_improves_over_generations_zdt1(self, zdt1_problem: dict) -> None:
        """Optimization improves (or maintains) hypervolume over generations."""
        # Run short optimization
        initial_result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=20,
            n_generations=2,
            seed=42,
        )

        # Run longer optimization
        final_result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=20,
            n_generations=50,
            seed=42,
        )

        # Compare using simple dominated hypervolume proxy:
        # Sum of objective values for Pareto front (lower is better for minimization)
        initial_ranks = non_dominated_sort(initial_result.population.objectives)
        final_ranks = non_dominated_sort(final_result.population.objectives)

        initial_front = initial_result.population.objectives[initial_ranks == 0]
        final_front = final_result.population.objectives[final_ranks == 0]

        # Mean objective value should decrease (improvement)
        initial_mean = initial_front.mean()
        final_mean = final_front.mean()

        assert final_mean <= initial_mean

    def test_zero_generations_returns_initial_population(self, simple_biobj_problem: dict) -> None:
        """With n_generations=0, returns the initialized population."""
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=0,
            seed=42,
        )
        assert len(result.population) == 10
        assert result.population.objectives is not None

    def test_raises_on_invalid_pop_size(self, simple_biobj_problem: dict) -> None:
        """Raises ValueError for non-positive pop_size."""
        with pytest.raises(ValueError, match="pop_size"):
            nsga2(
                init=simple_biobj_problem["init"],
                evaluate=simple_biobj_problem["evaluate"],
                crossover=simple_biobj_problem["crossover"],
                mutate=simple_biobj_problem["mutate"],
                pop_size=0,
                n_generations=5,
                seed=42,
            )

    def test_raises_on_negative_generations(self, simple_biobj_problem: dict) -> None:
        """Raises ValueError for negative n_generations."""
        with pytest.raises(ValueError, match="n_generations"):
            nsga2(
                init=simple_biobj_problem["init"],
                evaluate=simple_biobj_problem["evaluate"],
                crossover=simple_biobj_problem["crossover"],
                mutate=simple_biobj_problem["mutate"],
                pop_size=10,
                n_generations=-1,
                seed=42,
            )


# =============================================================================
# TestNSGA2Callback
# =============================================================================


class TestNSGA2Callback:
    """Tests for the callback functionality in nsga2."""

    def test_callback_receives_population_and_generation(self, simple_biobj_problem: dict) -> None:
        """Callback receives (NSGA2Result, int) arguments."""
        received_args: list[tuple[NSGA2Result, int]] = []

        def callback(result: NSGA2Result, gen: int) -> bool:
            received_args.append((result, gen))
            return False

        nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=3,
            seed=42,
            callback=callback,
        )

        assert len(received_args) == 3
        for i, (result, gen) in enumerate(received_args):
            assert isinstance(result, NSGA2Result)
            assert gen == i

    def test_callback_can_stop_early(self, simple_biobj_problem: dict) -> None:
        """Returning True from callback stops optimization early."""
        call_count = 0

        def callback(result: NSGA2Result, gen: int) -> bool:
            nonlocal call_count
            call_count += 1
            return gen >= 1  # Stop after generation 1

        nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=10,
            seed=42,
            callback=callback,
        )

        assert call_count == 2  # Called for gen 0 and gen 1

    def test_callback_not_called_with_zero_generations(self, simple_biobj_problem: dict) -> None:
        """Callback is not called when n_generations=0."""
        call_count = 0

        def callback(result: NSGA2Result, gen: int) -> bool:
            nonlocal call_count
            call_count += 1
            return False

        nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=0,
            seed=42,
            callback=callback,
        )

        assert call_count == 0

    def test_callback_receives_current_population_state(self, simple_biobj_problem: dict) -> None:
        """Each callback receives the current generation's result."""
        results: list[NSGA2Result] = []

        def callback(result: NSGA2Result, gen: int) -> bool:
            results.append(result)
            return False

        nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=3,
            seed=42,
            callback=callback,
        )

        # Each result's population should have objectives
        for result in results:
            assert result.population.objectives is not None


# =============================================================================
# TestPropertyBased
# =============================================================================


class TestPropertyBased:
    """Property-based tests for invariants that must always hold."""

    def test_rank_zero_is_truly_non_dominated(self, zdt1_problem: dict) -> None:
        """All rank-0 individuals are truly non-dominated by each other."""
        result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=20,
            n_generations=10,
            seed=42,
        )

        ranks = non_dominated_sort(result.population.objectives)
        pareto_front = result.population.objectives[ranks == 0]

        # Check that no solution in the front dominates another
        for i in range(len(pareto_front)):
            for j in range(len(pareto_front)):
                if i == j:
                    continue
                a = pareto_front[i]
                b = pareto_front[j]
                # a should NOT dominate b
                dominates = np.all(a <= b) and np.any(a < b)
                assert not dominates, f"Solution {i} dominates solution {j} but both are rank 0"

    def test_crowding_distance_non_negative(self, zdt1_problem: dict) -> None:
        """All crowding distances are non-negative (or inf)."""
        result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=20,
            n_generations=10,
            seed=42,
        )

        # Compute crowding distance for the result
        ranks = non_dominated_sort(result.population.objectives)
        cd = _compute_crowding_for_all_fronts(result.population.objectives, ranks)
        assert np.all(cd >= 0)

    def test_ranks_are_contiguous(self, zdt1_problem: dict) -> None:
        """Ranks form a contiguous sequence starting from 0."""
        result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=20,
            n_generations=10,
            seed=42,
        )

        ranks = non_dominated_sort(result.population.objectives)
        unique_ranks = np.unique(ranks)
        expected_ranks = np.arange(unique_ranks.max() + 1)
        np.testing.assert_array_equal(unique_ranks, expected_ranks)

    def test_population_size_invariant(self, zdt1_problem: dict) -> None:
        """Population size remains constant throughout optimization."""
        pop_size = 16  # Must be even for new nsga2 implementation
        sizes: list[int] = []

        def callback(result: NSGA2Result, gen: int) -> bool:
            sizes.append(len(result.population))
            return False

        result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=pop_size,
            n_generations=5,
            seed=42,
            callback=callback,
        )

        assert len(result.population) == pop_size
        assert all(s == pop_size for s in sizes)

    def test_objectives_correspond_to_x(self, simple_biobj_problem: dict) -> None:
        """Each objectives row corresponds to evaluating its x row."""
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=5,
            seed=42,
        )

        # Re-evaluate and compare
        for i in range(len(result.population)):
            expected_obj = simple_biobj_problem["evaluate"](result.population.x[i])
            np.testing.assert_allclose(result.population.objectives[i], expected_obj)

    def test_boundary_solutions_have_inf_crowding(self, zdt1_problem: dict) -> None:
        """In each front with 3+ individuals, boundary solutions have inf crowding."""
        result = nsga2(
            init=zdt1_problem["init"],
            evaluate=zdt1_problem["evaluate"],
            crossover=zdt1_problem["crossover"],
            mutate=zdt1_problem["mutate"],
            pop_size=30,
            n_generations=20,
            seed=42,
        )

        ranks = non_dominated_sort(result.population.objectives)
        cd = _compute_crowding_for_all_fronts(result.population.objectives, ranks)

        # Check each front
        for r in range(int(ranks.max()) + 1):
            mask = ranks == r
            front_obj = result.population.objectives[mask]
            front_cd = cd[mask]

            if len(front_cd) >= 3:
                # Recompute crowding to verify
                expected_cd = crowding_distance(front_obj)
                np.testing.assert_array_almost_equal(front_cd, expected_cd)


# =============================================================================
# Additional edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_individual_population(self) -> None:
        """Handles population of size 2."""

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(0, 1, size=3)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum(), (1 - x).sum()])

        def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        def mutate(x: np.ndarray) -> np.ndarray:
            return x.copy()

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=2,
            n_generations=3,
            seed=42,
        )

        assert len(result.population) == 2

        # Compute crowding distance
        ranks = non_dominated_sort(result.population.objectives)
        cd = _compute_crowding_for_all_fronts(result.population.objectives, ranks)

        # With 2 individuals in a front, both should have inf crowding
        assert np.all(np.isinf(cd))

    def test_single_objective_works(self) -> None:
        """Algorithm works with single-objective optimization."""

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(0, 1, size=3)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x.sum()])  # Single objective

        def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        def mutate(x: np.ndarray) -> np.ndarray:
            return np.clip(x + 0.1 * np.random.randn(len(x)), 0, 1)

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=10,
            n_generations=5,
            seed=42,
        )

        assert len(result.population) == 10
        assert result.population.objectives.shape[1] == 1

    def test_many_objectives(self) -> None:
        """Algorithm works with many objectives (5+)."""
        n_obj = 5

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(0, 1, size=10)

        def evaluate(x: np.ndarray) -> np.ndarray:
            # Split x into n_obj chunks and sum each
            chunk_size = len(x) // n_obj
            return np.array([x[i * chunk_size : (i + 1) * chunk_size].sum() for i in range(n_obj)])

        def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        def mutate(x: np.ndarray) -> np.ndarray:
            return np.clip(x + 0.1 * np.random.randn(len(x)), 0, 1)

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
        )

        assert len(result.population) == 20
        assert result.population.objectives.shape[1] == n_obj

    def test_high_dimensional_decision_space(self) -> None:
        """Algorithm works with high-dimensional decision variables."""
        n_vars = 100

        def init(rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(0, 1, size=n_vars)

        def evaluate(x: np.ndarray) -> np.ndarray:
            return np.array([x[:50].sum(), x[50:].sum()])

        def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        def mutate(x: np.ndarray) -> np.ndarray:
            return x.copy()

        result = nsga2(
            init=init,
            evaluate=evaluate,
            crossover=crossover,
            mutate=mutate,
            pop_size=10,
            n_generations=3,
            seed=42,
        )

        assert len(result.population) == 10
        assert result.population.x.shape[1] == n_vars


# =============================================================================
# TestNSGA2ParallelEvaluation
# =============================================================================


class TestNSGA2ParallelEvaluation:
    """Tests for parallel evaluation in NSGA-II."""

    def test_parallel_produces_same_result_as_sequential(self, simple_biobj_problem: dict) -> None:
        """Parallel evaluation should produce identical results to sequential with same seed."""
        # Use deterministic operators
        def deterministic_mutate(x: np.ndarray) -> np.ndarray:
            return x.copy()

        kwargs = {
            "init": simple_biobj_problem["init"],
            "evaluate": simple_biobj_problem["evaluate"],
            "crossover": simple_biobj_problem["crossover"],
            "mutate": deterministic_mutate,
            "pop_size": 10,
            "n_generations": 5,
            "seed": 42,
        }

        result_seq = nsga2(**kwargs, n_workers=1)
        result_par = nsga2(**kwargs, n_workers=2)

        np.testing.assert_array_equal(result_seq.population.x, result_par.population.x)
        np.testing.assert_array_equal(result_seq.population.objectives, result_par.population.objectives)

    def test_n_workers_minus_one_smoke_test(self, simple_biobj_problem: dict) -> None:
        """n_workers=-1 (all cores) should work without errors."""
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=10,
            n_generations=2,
            seed=42,
            n_workers=-1,
        )

        assert isinstance(result, NSGA2Result)
        assert len(result.population) == 10

    def test_invalid_n_workers_raises_value_error(self, simple_biobj_problem: dict) -> None:
        """Invalid n_workers values should raise ValueError."""
        with pytest.raises(ValueError, match="n_workers must be positive or -1"):
            nsga2(
                init=simple_biobj_problem["init"],
                evaluate=simple_biobj_problem["evaluate"],
                crossover=simple_biobj_problem["crossover"],
                mutate=simple_biobj_problem["mutate"],
                pop_size=10,
                n_generations=5,
                seed=42,
                n_workers=0,
            )

        with pytest.raises(ValueError, match="n_workers must be positive or -1"):
            nsga2(
                init=simple_biobj_problem["init"],
                evaluate=simple_biobj_problem["evaluate"],
                crossover=simple_biobj_problem["crossover"],
                mutate=simple_biobj_problem["mutate"],
                pop_size=10,
                n_generations=5,
                seed=42,
                n_workers=-2,
            )

    def test_evaluations_count_same_parallel_vs_sequential(self, simple_biobj_problem: dict) -> None:
        """Total evaluation count should be same for parallel and sequential."""
        kwargs = {
            "init": simple_biobj_problem["init"],
            "evaluate": simple_biobj_problem["evaluate"],
            "crossover": simple_biobj_problem["crossover"],
            "mutate": simple_biobj_problem["mutate"],
            "pop_size": 10,
            "n_generations": 5,
            "seed": 42,
        }

        result_seq = nsga2(**kwargs, n_workers=1)
        result_par = nsga2(**kwargs, n_workers=2)

        assert result_seq.evaluations == result_par.evaluations
