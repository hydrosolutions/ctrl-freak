"""Tests for algorithm-specific result types.

This module tests NSGA2Result and GAResult dataclasses,
following the testing philosophy from CLAUDE.md:
- Test behavior, not implementation
- Each test should fail for one reason
- Assert both exception type and message fragment for error tests
"""

import numpy as np
import pytest

from ctrl_freak import Population
from ctrl_freak.results import GAResult, NSGA2Result


class TestNSGA2ResultConstruction:
    """Tests for NSGA2Result construction and validation."""

    def test_constructs_with_valid_data(self) -> None:
        """NSGA2Result can be constructed with valid inputs."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        obj = np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])
        pop = Population(x=x, objectives=obj)
        rank = np.array([0, 0, 1])
        cd = np.array([np.inf, 1.0, 0.5])

        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=100,
            evaluations=5000,
        )

        assert result.population == pop
        assert result.generations == 100
        assert result.evaluations == 5000
        np.testing.assert_array_equal(result.rank, rank)
        np.testing.assert_array_equal(result.crowding_distance, cd)

    def test_rejects_non_array_rank(self) -> None:
        """NSGA2Result rejects rank that is not a numpy array."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        cd = np.array([1.0])
        with pytest.raises(TypeError, match="rank must be a numpy array"):
            NSGA2Result(
                population=pop,
                rank=[0],
                crowding_distance=cd,
                generations=10,
                evaluations=100,
            )

    def test_rejects_non_array_crowding_distance(self) -> None:
        """NSGA2Result rejects crowding_distance that is not a numpy array."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        rank = np.array([0])
        with pytest.raises(TypeError, match="crowding_distance must be a numpy array"):
            NSGA2Result(
                population=pop,
                rank=rank,
                crowding_distance=[1.0],
                generations=10,
                evaluations=100,
            )

    def test_rejects_2d_rank(self) -> None:
        """NSGA2Result rejects 2D rank array."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        rank = np.array([[0]])  # 2D instead of 1D
        cd = np.array([1.0])
        with pytest.raises(ValueError, match="rank must be 1D"):
            NSGA2Result(
                population=pop,
                rank=rank,
                crowding_distance=cd,
                generations=10,
                evaluations=100,
            )

    def test_rejects_2d_crowding_distance(self) -> None:
        """NSGA2Result rejects 2D crowding_distance array."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        rank = np.array([0])
        cd = np.array([[1.0]])  # 2D instead of 1D
        with pytest.raises(ValueError, match="crowding_distance must be 1D"):
            NSGA2Result(
                population=pop,
                rank=rank,
                crowding_distance=cd,
                generations=10,
                evaluations=100,
            )

    def test_rejects_mismatched_rank_size(self) -> None:
        """NSGA2Result rejects rank with size different from population."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        rank = np.array([0])  # Only 1 element, pop has 2
        cd = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="rank has 1 elements, expected 2"):
            NSGA2Result(
                population=pop,
                rank=rank,
                crowding_distance=cd,
                generations=10,
                evaluations=100,
            )

    def test_rejects_mismatched_crowding_distance_size(self) -> None:
        """NSGA2Result rejects crowding_distance with size different from population."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        rank = np.array([0, 1])
        cd = np.array([1.0])  # Only 1 element, pop has 2
        with pytest.raises(ValueError, match="crowding_distance has 1 elements, expected 2"):
            NSGA2Result(
                population=pop,
                rank=rank,
                crowding_distance=cd,
                generations=10,
                evaluations=100,
            )


class TestNSGA2ResultImmutability:
    """Tests for NSGA2Result immutability guarantees."""

    def test_frozen_dataclass_rejects_attribute_assignment(self) -> None:
        """NSGA2Result is frozen and rejects attribute assignment."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        result = NSGA2Result(
            population=pop,
            rank=np.array([0]),
            crowding_distance=np.array([1.0]),
            generations=10,
            evaluations=100,
        )
        with pytest.raises(AttributeError):
            result.generations = 999

    def test_rank_is_copied_on_construction(self) -> None:
        """Modifying original rank array does not affect result."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        rank = np.array([0, 1])
        cd = np.array([1.0, 2.0])
        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        # Modify original
        rank[0] = 999

        # Result should be unaffected
        assert result.rank[0] == 0

    def test_crowding_distance_is_copied_on_construction(self) -> None:
        """Modifying original crowding_distance array does not affect result."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        rank = np.array([0, 1])
        cd = np.array([1.0, 2.0])
        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        # Modify original
        cd[0] = 999.0

        # Result should be unaffected
        assert result.crowding_distance[0] == 1.0


class TestNSGA2ResultParetoFront:
    """Tests for NSGA2Result.pareto_front property."""

    def test_pareto_front_returns_population(self) -> None:
        """pareto_front returns a Population instance."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        rank = np.array([0, 1])
        cd = np.array([1.0, 2.0])
        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        front = result.pareto_front
        assert isinstance(front, Population)

    def test_pareto_front_includes_only_rank_zero(self) -> None:
        """pareto_front includes only individuals with rank 0."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        obj = np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        pop = Population(x=x, objectives=obj)
        rank = np.array([0, 1, 0, 1])
        cd = np.array([np.inf, 0.5, np.inf, 0.3])

        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        front = result.pareto_front

        # Should have 2 individuals (indices 0 and 2)
        assert len(front) == 2
        np.testing.assert_array_equal(front.x[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(front.x[1], np.array([5.0, 6.0]))
        np.testing.assert_array_equal(front.objectives[0], np.array([0.5, 0.5]))
        np.testing.assert_array_equal(front.objectives[1], np.array([0.4, 0.6]))

    def test_pareto_front_when_all_rank_zero(self) -> None:
        """pareto_front returns entire population when all have rank 0."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)
        rank = np.array([0, 0, 0])
        cd = np.array([1.0, 2.0, 3.0])

        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        front = result.pareto_front
        assert len(front) == 3

    def test_pareto_front_when_none_rank_zero(self) -> None:
        """pareto_front returns empty population when no rank 0 individuals."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop = Population(x=x)
        rank = np.array([1, 2])
        cd = np.array([1.0, 2.0])

        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        front = result.pareto_front
        assert len(front) == 0

    def test_pareto_front_handles_none_objectives(self) -> None:
        """pareto_front works when population has no objectives."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)  # No objectives
        rank = np.array([0, 1, 0])
        cd = np.array([1.0, 2.0, 3.0])

        result = NSGA2Result(
            population=pop,
            rank=rank,
            crowding_distance=cd,
            generations=10,
            evaluations=100,
        )

        front = result.pareto_front
        assert len(front) == 2
        assert front.objectives is None


class TestGAResultConstruction:
    """Tests for GAResult construction and validation."""

    def test_constructs_with_valid_data(self) -> None:
        """GAResult can be constructed with valid inputs."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)
        fitness = np.array([0.5, 0.3, 0.7])
        best_idx = 1

        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=best_idx,
            generations=50,
            evaluations=2500,
        )

        assert result.population == pop
        assert result.best_idx == 1
        assert result.generations == 50
        assert result.evaluations == 2500
        np.testing.assert_array_equal(result.fitness, fitness)

    def test_rejects_non_array_fitness(self) -> None:
        """GAResult rejects fitness that is not a numpy array."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        with pytest.raises(TypeError, match="fitness must be a numpy array"):
            GAResult(
                population=pop,
                fitness=[0.5],
                best_idx=0,
                generations=10,
                evaluations=100,
            )

    def test_rejects_2d_fitness(self) -> None:
        """GAResult rejects 2D fitness array."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        fitness = np.array([[0.5]])  # 2D instead of 1D
        with pytest.raises(ValueError, match="fitness must be 1D"):
            GAResult(
                population=pop,
                fitness=fitness,
                best_idx=0,
                generations=10,
                evaluations=100,
            )

    def test_rejects_mismatched_fitness_size(self) -> None:
        """GAResult rejects fitness with size different from population."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        fitness = np.array([0.5])  # Only 1 element, pop has 2
        with pytest.raises(ValueError, match="fitness has 1 elements, expected 2"):
            GAResult(
                population=pop,
                fitness=fitness,
                best_idx=0,
                generations=10,
                evaluations=100,
            )

    def test_rejects_non_integer_best_idx(self) -> None:
        """GAResult rejects best_idx that is not an integer."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        fitness = np.array([0.5])
        with pytest.raises(TypeError, match="best_idx must be an integer"):
            GAResult(
                population=pop,
                fitness=fitness,
                best_idx=0.5,
                generations=10,
                evaluations=100,
            )

    def test_rejects_negative_best_idx(self) -> None:
        """GAResult rejects negative best_idx."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        fitness = np.array([0.5])
        with pytest.raises(ValueError, match="best_idx -1 is out of bounds"):
            GAResult(
                population=pop,
                fitness=fitness,
                best_idx=-1,
                generations=10,
                evaluations=100,
            )

    def test_rejects_out_of_bounds_best_idx(self) -> None:
        """GAResult rejects best_idx that exceeds population size."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        fitness = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="best_idx 2 is out of bounds"):
            GAResult(
                population=pop,
                fitness=fitness,
                best_idx=2,
                generations=10,
                evaluations=100,
            )

    def test_accepts_numpy_integer_best_idx(self) -> None:
        """GAResult accepts numpy integer types for best_idx."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        fitness = np.array([0.5, 0.3])
        best_idx = np.int64(1)

        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=best_idx,
            generations=10,
            evaluations=100,
        )

        assert result.best_idx == 1


class TestGAResultImmutability:
    """Tests for GAResult immutability guarantees."""

    def test_frozen_dataclass_rejects_attribute_assignment(self) -> None:
        """GAResult is frozen and rejects attribute assignment."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        result = GAResult(
            population=pop,
            fitness=np.array([0.5]),
            best_idx=0,
            generations=10,
            evaluations=100,
        )
        with pytest.raises(AttributeError):
            result.generations = 999

    def test_fitness_is_copied_on_construction(self) -> None:
        """Modifying original fitness array does not affect result."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        fitness = np.array([0.5, 0.3])
        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=1,
            generations=10,
            evaluations=100,
        )

        # Modify original
        fitness[0] = 999.0

        # Result should be unaffected
        assert result.fitness[0] == 0.5


class TestGAResultBest:
    """Tests for GAResult.best property."""

    def test_best_returns_tuple(self) -> None:
        """best returns a tuple of (x, fitness)."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        fitness = np.array([0.5, 0.3])
        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=1,
            generations=10,
            evaluations=100,
        )

        best = result.best
        assert isinstance(best, tuple)
        assert len(best) == 2

    def test_best_returns_correct_individual(self) -> None:
        """best returns the decision variables and fitness of best individual."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)
        fitness = np.array([0.5, 0.3, 0.7])
        best_idx = 1

        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=best_idx,
            generations=10,
            evaluations=100,
        )

        best_x, best_fitness = result.best

        np.testing.assert_array_equal(best_x, np.array([3.0, 4.0]))
        assert best_fitness == 0.3

    def test_best_returns_float_fitness(self) -> None:
        """best returns fitness as a Python float, not numpy type."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        fitness = np.array([0.5])
        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=0,
            generations=10,
            evaluations=100,
        )

        _, best_fitness = result.best
        assert isinstance(best_fitness, float)
        assert not isinstance(best_fitness, np.floating)

    def test_best_at_first_index(self) -> None:
        """best works correctly when best individual is at index 0."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop = Population(x=x)
        fitness = np.array([0.2, 0.5])

        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=0,
            generations=10,
            evaluations=100,
        )

        best_x, best_fitness = result.best
        np.testing.assert_array_equal(best_x, np.array([1.0, 2.0]))
        assert best_fitness == 0.2

    def test_best_at_last_index(self) -> None:
        """best works correctly when best individual is at last index."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)
        fitness = np.array([0.5, 0.4, 0.1])

        result = GAResult(
            population=pop,
            fitness=fitness,
            best_idx=2,
            generations=10,
            evaluations=100,
        )

        best_x, best_fitness = result.best
        np.testing.assert_array_equal(best_x, np.array([5.0, 6.0]))
        assert best_fitness == 0.1
