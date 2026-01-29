"""Tests for the standard single-objective genetic algorithm module.

This module tests:
- ga(): Core functionality, strategy resolution, callbacks, edge cases
- GAResult: Construction and properties
- Integration tests with sphere function
"""

import numpy as np
import pytest

from ctrl_freak import GAResult, Population
from ctrl_freak.algorithms.ga import ga
from ctrl_freak.operators import polynomial_mutation, sbx_crossover
from ctrl_freak.selection.tournament import fitness_tournament
from ctrl_freak.survival.elitist import elitist_survival

# =============================================================================
# TestGA - Core Functionality
# =============================================================================


class TestGA:
    """Tests for the ga() function core functionality."""

    def test_population_size_invariant(self, sphere_problem: dict) -> None:
        """Population size remains constant across generations."""
        pop_size = 20
        sizes: list[int] = []

        def callback(result: GAResult, gen: int) -> bool:
            sizes.append(len(result.population))
            return False

        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=pop_size,
            n_generations=5,
            seed=42,
            callback=callback,
        )

        # Final population should be correct size
        assert len(result.population) == pop_size
        # All intermediate generations should be correct size
        assert all(s == pop_size for s in sizes)

    def test_fitness_improvement(self, sphere_problem: dict) -> None:
        """Final fitness should be better (lower) than initial fitness."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Store initial fitness
        initial_best: float | None = None

        def callback(result: GAResult, gen: int) -> bool:
            nonlocal initial_best
            if gen == 0:
                _, fitness = result.best
                initial_best = fitness
            return False

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=30,
            n_generations=20,
            seed=42,
            callback=callback,
        )

        _, final_best = result.best
        assert initial_best is not None
        # Should improve (lower fitness is better)
        assert final_best <= initial_best

    def test_determinism_with_same_seed(self, sphere_problem: dict) -> None:
        """Same seed produces identical results."""

        # Use deterministic operators for perfect reproducibility
        def deterministic_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        def deterministic_mutate(x: np.ndarray) -> np.ndarray:
            return x.copy()

        kwargs = {
            "init": sphere_problem["init"],
            "evaluate": sphere_problem["evaluate"],
            "crossover": deterministic_crossover,
            "mutate": deterministic_mutate,
            "pop_size": 20,
            "n_generations": 5,
            "seed": 12345,
        }

        result1 = ga(**kwargs)
        result2 = ga(**kwargs)

        np.testing.assert_array_equal(result1.population.x, result2.population.x)
        np.testing.assert_array_equal(result1.fitness, result2.fitness)
        assert result1.best_idx == result2.best_idx

    # -------------------------------------------------------------------------
    # Strategy Resolution Tests
    # -------------------------------------------------------------------------

    def test_resolves_tournament_selection(self, sphere_problem: dict) -> None:
        """String 'tournament' is resolved to tournament selection."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Should not raise - 'tournament' is registered
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select="tournament",  # String strategy
            survive="elitist",
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 20

    def test_resolves_roulette_selection(self, sphere_problem: dict) -> None:
        """String 'roulette' is resolved to roulette wheel selection."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Should not raise - 'roulette' is registered
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select="roulette",  # String strategy
            survive="elitist",
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 20

    def test_resolves_elitist_survival(self, sphere_problem: dict) -> None:
        """String 'elitist' is resolved to elitist survival."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Should not raise - 'elitist' is registered
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select="tournament",
            survive="elitist",  # String strategy
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 20

    def test_resolves_truncation_survival(self, sphere_problem: dict) -> None:
        """String 'truncation' is resolved to truncation survival."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Should not raise - 'truncation' is registered
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select="tournament",
            survive="truncation",  # String strategy
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 20

    def test_accepts_custom_selection_callable(self, sphere_problem: dict) -> None:
        """Custom ParentSelector callable is accepted."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Create custom selector
        custom_selector = fitness_tournament(tournament_size=3)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select=custom_selector,  # Callable strategy
            survive="elitist",
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 20

    def test_accepts_custom_survival_callable(self, sphere_problem: dict) -> None:
        """Custom SurvivorSelector callable is accepted."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Create custom survivor
        custom_survivor = elitist_survival(elite_count=2)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select="tournament",
            survive=custom_survivor,  # Callable strategy
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 20

    # -------------------------------------------------------------------------
    # GAResult Tests
    # -------------------------------------------------------------------------

    def test_result_fields_populated(self, sphere_problem: dict) -> None:
        """GAResult has all fields populated correctly."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        pop_size = 20
        n_generations = 10

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=pop_size,
            n_generations=n_generations,
            seed=42,
        )

        # Check all fields are populated
        assert isinstance(result.population, Population)
        assert result.population.x is not None
        assert result.population.objectives is not None
        assert isinstance(result.fitness, np.ndarray)
        assert isinstance(result.best_idx, (int, np.integer))
        assert isinstance(result.generations, int)
        assert isinstance(result.evaluations, int)

        # Check shapes and values
        assert len(result.population) == pop_size
        assert len(result.fitness) == pop_size
        assert 0 <= result.best_idx < pop_size
        assert result.generations == n_generations
        # Evaluations: initial pop + n_generations * pop_size offspring
        assert result.evaluations == pop_size + n_generations * pop_size

    def test_best_property_returns_correct_values(self, sphere_problem: dict) -> None:
        """GAResult.best property returns correct (x, fitness) tuple."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=10,
            seed=42,
        )

        best_x, best_fitness = result.best

        # Check types
        assert isinstance(best_x, np.ndarray)
        assert isinstance(best_fitness, float)

        # Check correctness
        expected_best_idx = np.argmin(result.fitness)
        assert result.best_idx == expected_best_idx
        np.testing.assert_array_equal(best_x, result.population.x[expected_best_idx])
        assert best_fitness == result.fitness[expected_best_idx]

        # Verify best_fitness matches re-evaluation
        re_evaluated = sphere_problem["evaluate"](best_x)
        assert abs(best_fitness - re_evaluated) < 1e-10

    # -------------------------------------------------------------------------
    # Callback Tests
    # -------------------------------------------------------------------------

    def test_callback_receives_correct_types(self, sphere_problem: dict) -> None:
        """Callback receives (GAResult, int) arguments."""
        received_args: list[tuple[type, type]] = []

        def callback(result: GAResult, gen: int) -> bool:
            received_args.append((type(result), type(gen)))
            return False

        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=10,
            n_generations=3,
            seed=42,
            callback=callback,
        )

        assert len(received_args) == 3
        for result_type, gen_type in received_args:
            assert result_type is GAResult
            assert gen_type is int

    def test_callback_can_stop_early(self, sphere_problem: dict) -> None:
        """Returning True from callback stops optimization early."""
        call_count = 0

        def callback(result: GAResult, gen: int) -> bool:
            nonlocal call_count
            call_count += 1
            return gen >= 2  # Stop after generation 2

        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=10,
            n_generations=100,  # Request many generations
            seed=42,
            callback=callback,
        )

        # Should stop early
        assert call_count == 3  # Called for gen 0, 1, 2
        assert result.generations == 2  # Only 2 generations completed

    # -------------------------------------------------------------------------
    # Edge Cases and Validation
    # -------------------------------------------------------------------------

    def test_zero_generations(self, sphere_problem: dict) -> None:
        """Zero generations returns initial population."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=0,
            seed=42,
        )

        assert len(result.population) == 20
        assert result.generations == 0
        # Only initial population evaluated
        assert result.evaluations == 20

    def test_callback_not_called_with_zero_generations(self, sphere_problem: dict) -> None:
        """Callback is not called when n_generations=0."""
        call_count = 0

        def callback(result: GAResult, gen: int) -> bool:
            nonlocal call_count
            call_count += 1
            return False

        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=10,
            n_generations=0,
            seed=42,
            callback=callback,
        )

        assert call_count == 0

    def test_raises_on_negative_pop_size(self, sphere_problem: dict) -> None:
        """Raises ValueError for negative pop_size."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(ValueError, match="pop_size must be positive"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=-10,
                n_generations=5,
                seed=42,
            )

    def test_raises_on_zero_pop_size(self, sphere_problem: dict) -> None:
        """Raises ValueError for zero pop_size."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(ValueError, match="pop_size must be positive"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=0,
                n_generations=5,
                seed=42,
            )

    def test_raises_on_odd_pop_size(self, sphere_problem: dict) -> None:
        """Raises ValueError for odd pop_size (required for parent pairing)."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(ValueError, match="pop_size must be even"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=11,  # Odd number
                n_generations=5,
                seed=42,
            )

    def test_raises_on_negative_generations(self, sphere_problem: dict) -> None:
        """Raises ValueError for negative n_generations."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(ValueError, match="n_generations must be non-negative"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=20,
                n_generations=-5,
                seed=42,
            )

    def test_raises_on_unregistered_selection_strategy(self, sphere_problem: dict) -> None:
        """Raises KeyError for unregistered selection strategy string."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(KeyError):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=20,
                n_generations=5,
                seed=42,
                select="nonexistent_strategy",
                survive="elitist",
            )

    def test_raises_on_unregistered_survival_strategy(self, sphere_problem: dict) -> None:
        """Raises KeyError for unregistered survival strategy string."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(KeyError):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=20,
                n_generations=5,
                seed=42,
                select="tournament",
                survive="nonexistent_strategy",
            )

    def test_elitist_survival_uses_elite_count_default(self, sphere_problem: dict) -> None:
        """Elitist survival uses elite_count=1 by default when passed as string."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        # Should not raise - uses default elite_count=1
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=20,
            n_generations=5,
            seed=42,
            select="tournament",
            survive="elitist",
        )

        assert isinstance(result, GAResult)


# =============================================================================
# TestGASphereConvergence - Integration Test
# =============================================================================


class TestGASphereConvergence:
    """Integration test for convergence on sphere function."""

    def test_converges_on_sphere(self, sphere_problem: dict) -> None:
        """GA converges toward global optimum on sphere function."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(
            eta=20.0,
            prob=0.1,  # 10% mutation probability per variable
            bounds=sphere_problem["bounds"],
            seed=43,
        )

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=50,
            n_generations=100,
            seed=42,
            select="tournament",
            survive="elitist",
        )

        best_x, best_fitness = result.best

        # Should converge close to known optimum (0.0)
        # Sphere function: f(x) = sum(x_i^2), minimum at x=0 with f(0)=0
        assert best_fitness < 1.0, f"Expected fitness < 1.0, got {best_fitness}"

        # Best solution should be close to origin
        assert np.linalg.norm(best_x) < 2.0, f"Expected solution near origin, got norm {np.linalg.norm(best_x)}"

    def test_tracks_improvement_over_generations(self, sphere_problem: dict) -> None:
        """Best fitness improves (or stays same) over generations."""
        fitness_history: list[float] = []

        def callback(result: GAResult, gen: int) -> bool:
            _, fitness = result.best
            fitness_history.append(fitness)
            return False

        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=30,
            n_generations=20,
            seed=42,
            callback=callback,
        )

        # Check that fitness is monotonically non-increasing (with elitism)
        for i in range(1, len(fitness_history)):
            assert fitness_history[i] <= fitness_history[i - 1], (
                f"Fitness increased at generation {i}: {fitness_history[i - 1]} -> {fitness_history[i]}"
            )

    def test_different_selection_strategies_converge(self, sphere_problem: dict) -> None:
        """Different selection strategies all achieve convergence."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        strategies = ["tournament", "roulette"]
        results = []

        for strategy in strategies:
            result = ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=30,
                n_generations=50,
                seed=42,
                select=strategy,
                survive="elitist",
            )
            results.append(result)

        # All strategies should achieve reasonable convergence
        for i, result in enumerate(results):
            _, fitness = result.best
            assert fitness < 5.0, f"Strategy {strategies[i]} did not converge well: fitness={fitness}"

    def test_different_survival_strategies_work(self, sphere_problem: dict) -> None:
        """Different survival strategies produce valid results."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        strategies = ["elitist", "truncation"]
        results = []

        for strategy in strategies:
            result = ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=30,
                n_generations=50,
                seed=42,
                select="tournament",
                survive=strategy,
            )
            results.append(result)

        # All strategies should produce valid results
        for i, result in enumerate(results):
            assert len(result.population) == 30
            assert result.generations == 50
            _, fitness = result.best
            # Should achieve some level of optimization
            assert fitness < 50.0, f"Strategy {strategies[i]} did not optimize: fitness={fitness}"


# =============================================================================
# TestGAParallelEvaluation
# =============================================================================


class TestGAParallelEvaluation:
    """Tests for parallel evaluation in GA."""

    def test_parallel_produces_same_result_as_sequential(self, sphere_problem: dict) -> None:
        """Parallel evaluation should produce identical results to sequential with same seed."""
        # Use deterministic operators
        def deterministic_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            return (p1 + p2) / 2

        def deterministic_mutate(x: np.ndarray) -> np.ndarray:
            return x.copy()

        kwargs = {
            "init": sphere_problem["init"],
            "evaluate": sphere_problem["evaluate"],
            "crossover": deterministic_crossover,
            "mutate": deterministic_mutate,
            "pop_size": 10,
            "n_generations": 5,
            "seed": 42,
        }

        result_seq = ga(**kwargs, n_workers=1)
        result_par = ga(**kwargs, n_workers=2)

        np.testing.assert_array_equal(result_seq.population.x, result_par.population.x)
        np.testing.assert_array_equal(result_seq.fitness, result_par.fitness)
        assert result_seq.best_idx == result_par.best_idx

    def test_n_workers_minus_one_smoke_test(self, sphere_problem: dict) -> None:
        """n_workers=-1 (all cores) should work without errors."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=crossover,
            mutate=mutate,
            pop_size=10,
            n_generations=2,
            seed=42,
            n_workers=-1,
        )

        assert isinstance(result, GAResult)
        assert len(result.population) == 10

    def test_invalid_n_workers_raises_value_error(self, sphere_problem: dict) -> None:
        """Invalid n_workers values should raise ValueError."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        with pytest.raises(ValueError, match="n_workers must be positive or -1"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=10,
                n_generations=5,
                seed=42,
                n_workers=0,
            )

        with pytest.raises(ValueError, match="n_workers must be positive or -1"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=crossover,
                mutate=mutate,
                pop_size=10,
                n_generations=5,
                seed=42,
                n_workers=-2,
            )

    def test_evaluations_count_same_parallel_vs_sequential(self, sphere_problem: dict) -> None:
        """Total evaluation count should be same for parallel and sequential."""
        crossover = sbx_crossover(eta=15.0, bounds=sphere_problem["bounds"], seed=42)
        mutate = polynomial_mutation(eta=20.0, bounds=sphere_problem["bounds"], seed=43)

        kwargs = {
            "init": sphere_problem["init"],
            "evaluate": sphere_problem["evaluate"],
            "crossover": crossover,
            "mutate": mutate,
            "pop_size": 10,
            "n_generations": 5,
            "seed": 42,
        }

        result_seq = ga(**kwargs, n_workers=1)
        result_par = ga(**kwargs, n_workers=2)

        assert result_seq.evaluations == result_par.evaluations
