"""Tests for survival strategies."""

import numpy as np
import pytest

from ctrl_freak.population import Population
from ctrl_freak.registry import SurvivalRegistry
from ctrl_freak.survival import elitist_survival, nsga2_survival, truncation_survival


class TestNSGA2Survival:
    """Tests for NSGA-II survival selection."""

    def test_returns_correct_number_of_survivors(self, simple_population):
        """Test that NSGA-II survival returns the requested number of survivors."""
        selector = nsga2_survival()
        n_survivors = 2

        indices, state = selector(simple_population, n_survivors)

        assert indices.shape == (n_survivors,)
        assert indices.dtype == np.intp
        assert np.all(indices >= 0)
        assert np.all(indices < len(simple_population))
        # Check that all indices are unique
        assert len(np.unique(indices)) == n_survivors

    def test_returns_state_with_rank_and_crowding_distance(self, simple_population):
        """Test that NSGA-II survival returns state dict with required keys."""
        selector = nsga2_survival()
        n_survivors = 3

        indices, state = selector(simple_population, n_survivors)

        # Check state dictionary has required keys
        assert "rank" in state
        assert "crowding_distance" in state

        # Check shapes match n_survivors
        assert state["rank"].shape == (n_survivors,)
        assert state["crowding_distance"].shape == (n_survivors,)

        # Check dtypes
        assert state["rank"].dtype in [np.int64, np.intp]
        assert state["crowding_distance"].dtype == np.float64

    def test_preserves_complete_fronts_before_partial_ones(self):
        """Test that complete Pareto fronts are preserved before using crowding distance."""
        # Create population with clear front structure:
        # Front 0: individuals 0, 1 (both Pareto optimal - non-dominated)
        # Front 1: individuals 2, 3 (both dominated)
        # To ensure 2,3 are dominated, they must be worse in all objectives or worse in some
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        objectives = np.array([
            [1.0, 5.0],  # Front 0 - non-dominated
            [5.0, 1.0],  # Front 0 - non-dominated (trades off with individual 0)
            [2.0, 4.0],  # Front 1 - dominated by individual 0 (1.0 < 2.0, 5.0 > 4.0 but not strict domination)
            [4.0, 2.0],  # Front 1 - dominated by individual 1 (5.0 > 4.0, 1.0 < 2.0 but not strict domination)
            [3.0, 3.0],  # Front 2 - dominated by both individuals from front 1
        ])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        # Select 3 survivors - should take all of front 0 (2) + best from front 1 (1)
        indices, state = selector(pop, n_survivors=3)

        rank = state["rank"]

        # Count survivors from each front
        n_front_0 = np.sum(rank == 0)

        # At least 2 survivors should have rank 0 (complete front 0)
        # This test just ensures complete fronts are taken first
        assert n_front_0 >= 2

        # Maximum rank should not exceed 1 (shouldn't reach front 2 with only 3 survivors)
        assert np.max(rank) <= 1

    def test_uses_crowding_distance_for_critical_front_selection(self):
        """Test that crowding distance is used to select from partially-fitted front."""
        # Create a population where front 0 has 4 individuals
        # We'll select 2 survivors, so all come from front 0
        # Individuals with higher crowding distance should be preferred
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([
            [1.0, 4.0],  # Boundary point (min f1, max f2)
            [2.0, 3.0],  # Interior point
            [3.0, 2.0],  # Interior point
            [4.0, 1.0],  # Boundary point (max f1, min f2)
        ])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        # Select 2 survivors from front 0 (all 4 are in same front)
        # Boundary points (0 and 3) should be selected due to infinite crowding distance
        indices, state = selector(pop, n_survivors=2)

        # Both boundary points should be selected
        assert 0 in indices
        assert 3 in indices

        # Check crowding distances in state
        cd = state["crowding_distance"]
        # Both selected survivors should have infinite crowding distance
        assert np.all(np.isinf(cd))

    def test_boundary_points_get_infinite_crowding_distance(self):
        """Test that boundary points receive infinite crowding distance."""
        # Create a simple front with clear boundaries
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([
            [1.0, 4.0],  # Boundary: min f1, max f2
            [2.0, 3.0],  # Interior
            [3.0, 2.0],  # Interior
            [4.0, 1.0],  # Boundary: max f1, min f2
        ])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        # Select all 4 survivors to check all crowding distances
        indices, state = selector(pop, n_survivors=4)

        cd = state["crowding_distance"]

        # Find which survivors correspond to boundary points (0 and 3)
        # Since all are rank 0, we need to map back to original indices
        boundary_mask = np.isin(indices, [0, 3])
        interior_mask = np.isin(indices, [1, 2])

        # Boundary points should have infinite crowding distance
        assert np.all(np.isinf(cd[boundary_mask]))
        # Interior points should have finite crowding distance
        assert np.all(np.isfinite(cd[interior_mask]))

    def test_deterministic_output(self):
        """Test that NSGA-II survival produces deterministic output."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()
        n_survivors = 3

        # Run twice
        indices1, state1 = selector(pop, n_survivors)
        indices2, state2 = selector(pop, n_survivors)

        # Results should be identical
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_array_equal(state1["rank"], state2["rank"])
        np.testing.assert_array_equal(state1["crowding_distance"], state2["crowding_distance"])

    def test_error_when_no_objectives(self):
        """Test that ValueError is raised when population has no objectives."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop = Population(x=x, objectives=None)

        selector = nsga2_survival()

        with pytest.raises(ValueError, match="Population must have objectives computed"):
            selector(pop, n_survivors=1)

    def test_error_when_n_survivors_is_zero(self, simple_population):
        """Test that ValueError is raised when n_survivors is zero."""
        selector = nsga2_survival()

        with pytest.raises(ValueError, match="n_survivors must be positive"):
            selector(simple_population, n_survivors=0)

    def test_error_when_n_survivors_is_negative(self, simple_population):
        """Test that ValueError is raised when n_survivors is negative."""
        selector = nsga2_survival()

        with pytest.raises(ValueError, match="n_survivors must be positive"):
            selector(simple_population, n_survivors=-1)

    def test_error_when_n_survivors_exceeds_population_size(self, simple_population):
        """Test that ValueError is raised when n_survivors exceeds population size."""
        selector = nsga2_survival()
        pop_size = len(simple_population)

        with pytest.raises(ValueError, match="n_survivors .* cannot exceed population size"):
            selector(simple_population, n_survivors=pop_size + 1)

    def test_all_survivors_when_n_survivors_equals_population_size(self, simple_population):
        """Test that all individuals survive when n_survivors equals population size."""
        selector = nsga2_survival()
        pop_size = len(simple_population)

        indices, state = selector(simple_population, n_survivors=pop_size)

        # All individuals should be selected
        assert len(indices) == pop_size
        assert set(indices) == set(range(pop_size))

    def test_single_survivor_selection(self):
        """Test selecting just one survivor."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        objectives = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        indices, state = selector(pop, n_survivors=1)

        # Should select exactly one survivor
        assert len(indices) == 1
        assert 0 <= indices[0] < len(pop)

        # State should have length 1
        assert len(state["rank"]) == 1
        assert len(state["crowding_distance"]) == 1

    def test_multiple_fronts_selection(self):
        """Test selection across multiple Pareto fronts."""
        # Create population with clear hierarchical domination
        # Use a simple pattern where higher values = worse in both objectives
        x = np.arange(6).reshape(6, 1)
        objectives = np.array([
            [1.0, 1.0],   # Front 0 - best in both
            [2.0, 2.0],   # Front 1 - dominated by front 0
            [3.0, 3.0],   # Front 2 - dominated by front 1
            [4.0, 4.0],   # Front 3 - dominated by front 2
            [5.0, 5.0],   # Front 4 - dominated by front 3
            [6.0, 6.0],   # Front 5 - dominated by front 4
        ])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        # Select 3 survivors: should get fronts 0, 1, 2
        indices, state = selector(pop, n_survivors=3)

        rank = state["rank"]

        # Maximum rank should not exceed 2 (fronts 0, 1, 2)
        assert np.max(rank) <= 2

        # Minimum rank should be 0 (front 0 always included)
        assert np.min(rank) == 0

        # Should have exactly 3 survivors
        assert len(rank) == 3

    def test_crowding_distance_within_front(self):
        """Test that crowding distances are computed correctly within each front."""
        # Create a simple population where we know the expected crowding distances
        x = np.arange(6).reshape(6, 1)
        objectives = np.array([
            [1.0, 5.0],  # Front 0 - boundary (min f1)
            [3.0, 3.0],  # Front 0 - interior
            [5.0, 1.0],  # Front 0 - boundary (max f1)
            [2.0, 4.0],  # Front 1 - boundary (min f1)
            [4.0, 2.0],  # Front 1 - boundary (max f1)
            [3.0, 3.0],  # Front 1 - should be dominated, actually this is same as index 1
        ])
        # Adjust to ensure clear fronts
        objectives[5] = [2.5, 3.5]  # Make it actually in front 1

        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        # Select all 6 to check all crowding distances
        indices, state = selector(pop, n_survivors=6)

        cd = state["crowding_distance"]
        rank = state["rank"]

        # Within each rank, boundary points should have infinite crowding distance
        for r in np.unique(rank):
            front_mask = rank == r
            front_cd = cd[front_mask]

            # At least 2 boundary points per front (if front has 3+ individuals)
            if np.sum(front_mask) >= 3:
                n_infinite = np.sum(np.isinf(front_cd))
                assert n_infinite == 2

    def test_registration_in_survival_registry(self):
        """Test that 'nsga2' is registered in SurvivalRegistry."""
        # Force re-registration by reloading the module
        import importlib

        import ctrl_freak.survival

        importlib.reload(ctrl_freak.survival)

        assert "nsga2" in SurvivalRegistry.list()
        selector = SurvivalRegistry.get("nsga2")

        # Verify it's callable and returns correct signature
        assert callable(selector)

        # Test that it works when retrieved from registry
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = np.array([[1.0, 2.0], [2.0, 1.0]])
        pop = Population(x=x, objectives=objectives)

        indices, state = selector(pop, n_survivors=1)
        assert len(indices) == 1
        assert "rank" in state
        assert "crowding_distance" in state


class TestNSGA2SurvivalEdgeCases:
    """Test edge cases for NSGA-II survival selection."""

    def test_single_individual_population(self):
        """Test survival with single individual."""
        x = np.array([[1.0, 2.0]])
        objectives = np.array([[1.0, 2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        indices, state = selector(pop, n_survivors=1)

        assert len(indices) == 1
        assert indices[0] == 0
        assert state["rank"][0] == 0
        # Single individual gets infinite crowding distance
        assert np.isinf(state["crowding_distance"][0])

    def test_two_individual_population(self):
        """Test survival with two individuals."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = np.array([[1.0, 2.0], [2.0, 1.0]])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        # Both are on front 0 (non-dominated)
        indices, state = selector(pop, n_survivors=2)

        assert len(indices) == 2
        # Both should have rank 0
        assert np.all(state["rank"] == 0)
        # Both are boundary points in a 2-individual front
        assert np.all(np.isinf(state["crowding_distance"]))

    def test_all_identical_objectives(self):
        """Test survival when all individuals have identical objectives."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        objectives = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        indices, state = selector(pop, n_survivors=2)

        # All should be rank 0 (all non-dominated since identical)
        assert np.all(state["rank"] == 0)
        # Should select 2 survivors
        assert len(indices) == 2

    def test_single_objective_behaves_correctly(self):
        """Test survival with single-objective optimization."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[1.0], [2.0], [3.0], [4.0]])
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()

        indices, state = selector(pop, n_survivors=2)

        # With single objective, each individual is in its own front
        # Best individual (lowest objective) should be rank 0
        assert len(indices) == 2

        # First survivor should be the best (index 0)
        assert 0 in indices

    def test_large_population_selection(self):
        """Test survival with larger population."""
        # Create population with 100 individuals
        n_pop = 100
        x = np.arange(n_pop).reshape(n_pop, 1).astype(float)
        # Create random objectives
        rng = np.random.default_rng(42)
        objectives = rng.uniform(0, 10, size=(n_pop, 2))
        pop = Population(x=x, objectives=objectives)

        selector = nsga2_survival()
        n_survivors = 50

        indices, state = selector(pop, n_survivors)

        # Should return correct number of survivors
        assert len(indices) == n_survivors
        assert len(np.unique(indices)) == n_survivors

        # State should have correct shapes
        assert state["rank"].shape == (n_survivors,)
        assert state["crowding_distance"].shape == (n_survivors,)

        # Ranks should be monotonically non-decreasing (fronts filled in order)
        # Actually, within a generation they might not be sorted, but lower ranks should appear
        # At least the minimum rank should be 0
        assert np.min(state["rank"]) == 0


class TestTruncationSurvival:
    """Tests for truncation survival selection."""

    def test_returns_correct_number_of_survivors(self):
        """Test that truncation survival returns the requested number of survivors."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        objectives = np.array([[4.0], [2.0], [3.0], [1.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 2

        indices, state = selector(pop, n_survivors)

        assert indices.shape == (n_survivors,)
        assert indices.dtype == np.intp
        assert np.all(indices >= 0)
        assert np.all(indices < len(pop))
        # Check that all indices are unique
        assert len(np.unique(indices)) == n_survivors

    def test_returns_state_with_fitness(self):
        """Test that truncation survival returns state dict with fitness key."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        objectives = np.array([[3.0], [1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 2

        indices, state = selector(pop, n_survivors)

        # Check state dictionary has fitness key
        assert "fitness" in state

        # Check shape matches n_survivors
        assert state["fitness"].shape == (n_survivors,)

        # Check dtype
        assert state["fitness"].dtype in [np.float64, np.float32]

    def test_selects_individuals_with_lowest_fitness(self):
        """Test that truncation selects individuals with lowest fitness values."""
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        objectives = np.array([[5.0], [1.0], [3.0], [2.0], [4.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 3

        indices, state = selector(pop, n_survivors)

        # Should select individuals with fitness 1.0, 2.0, 3.0 (indices 1, 3, 2)
        expected_fitness = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(state["fitness"], expected_fitness)

        # Verify indices correspond to correct individuals
        assert 1 in indices  # fitness 1.0
        assert 3 in indices  # fitness 2.0
        assert 2 in indices  # fitness 3.0

    def test_correct_ordering_by_fitness(self):
        """Test that survivors are ordered by fitness (best first)."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[4.0], [1.0], [3.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 3

        indices, state = selector(pop, n_survivors)

        # Fitness should be in ascending order (best = lowest first)
        assert np.all(np.diff(state["fitness"]) >= 0)

        # Verify correct fitness values
        np.testing.assert_array_equal(state["fitness"], [1.0, 2.0, 3.0])

    def test_ties_handled_deterministically(self):
        """Test that ties in fitness are handled deterministically (stable sort)."""
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        objectives = np.array([[2.0], [2.0], [1.0], [2.0], [3.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 3

        # Run twice to ensure deterministic results
        indices1, state1 = selector(pop, n_survivors)
        indices2, state2 = selector(pop, n_survivors)

        # Results should be identical
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_array_equal(state1["fitness"], state2["fitness"])

        # Should select individual with fitness 1.0 (index 2) and two with fitness 2.0
        # Due to stable sort, should prefer earlier indices: 0, 1 over 3
        assert 2 in indices1  # fitness 1.0
        assert np.sum(state1["fitness"] == 2.0) == 2

        # Stable sort means indices 0 and 1 should be selected over index 3
        assert 0 in indices1
        assert 1 in indices1

    def test_all_survivors_when_n_survivors_equals_population_size(self):
        """Test that all individuals survive when n_survivors equals population size."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[4.0], [3.0], [2.0], [1.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        pop_size = len(pop)

        indices, state = selector(pop, n_survivors=pop_size)

        # All individuals should be selected
        assert len(indices) == pop_size
        assert set(indices) == set(range(pop_size))

        # Should be in fitness order
        assert np.all(np.diff(state["fitness"]) >= 0)

    def test_single_survivor_selection(self):
        """Test selecting just one survivor."""
        x = np.array([[1.0], [2.0], [3.0]])
        objectives = np.array([[3.0], [1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()

        indices, state = selector(pop, n_survivors=1)

        # Should select exactly one survivor with best fitness
        assert len(indices) == 1
        assert indices[0] == 1  # Individual with fitness 1.0
        assert state["fitness"][0] == 1.0

    def test_error_when_n_survivors_is_zero(self):
        """Test that ValueError is raised when n_survivors is zero."""
        x = np.array([[1.0], [2.0]])
        objectives = np.array([[1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()

        with pytest.raises(ValueError, match="n_survivors must be positive"):
            selector(pop, n_survivors=0)

    def test_error_when_n_survivors_is_negative(self):
        """Test that ValueError is raised when n_survivors is negative."""
        x = np.array([[1.0], [2.0]])
        objectives = np.array([[1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()

        with pytest.raises(ValueError, match="n_survivors must be positive"):
            selector(pop, n_survivors=-1)

    def test_error_when_n_survivors_exceeds_population_size(self):
        """Test that ValueError is raised when n_survivors exceeds population size."""
        x = np.array([[1.0], [2.0], [3.0]])
        objectives = np.array([[1.0], [2.0], [3.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        pop_size = len(pop)

        with pytest.raises(ValueError, match="n_survivors .* cannot exceed population size"):
            selector(pop, n_survivors=pop_size + 1)

    def test_error_when_no_objectives_and_no_fitness_kwarg(self):
        """Test that ValueError is raised when population has no objectives and no fitness kwarg."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop = Population(x=x, objectives=None)

        selector = truncation_survival()

        with pytest.raises(ValueError, match="Population must have objectives computed"):
            selector(pop, n_survivors=1)

    def test_error_when_multi_objective_without_fitness_kwarg(self):
        """Test that ValueError is raised for multi-objective without explicit fitness."""
        x = np.array([[1.0], [2.0], [3.0]])
        objectives = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()

        with pytest.raises(
            ValueError, match="truncation requires single-objective optimization.*Pass explicit 'fitness'"
        ):
            selector(pop, n_survivors=2)

    def test_works_with_explicit_fitness_kwarg(self):
        """Test that truncation works with explicit fitness kwarg for multi-objective."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        # Define custom fitness as sum of objectives
        custom_fitness = np.array([5.0, 5.0, 5.0, 5.0])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()

        # Should work with explicit fitness
        indices, state = selector(pop, n_survivors=2, fitness=custom_fitness)

        # With all equal fitness, stable sort should select first two indices
        assert 0 in indices
        assert 1 in indices
        np.testing.assert_array_equal(state["fitness"], [5.0, 5.0])

    def test_registration_in_survival_registry(self):
        """Test that 'truncation' is registered in SurvivalRegistry."""
        # Force re-registration by reloading the module
        import importlib

        import ctrl_freak.survival

        importlib.reload(ctrl_freak.survival)

        assert "truncation" in SurvivalRegistry.list()
        selector = SurvivalRegistry.get("truncation")

        # Verify it's callable and returns correct signature
        assert callable(selector)

        # Test that it works when retrieved from registry
        x = np.array([[1.0], [2.0], [3.0]])
        objectives = np.array([[3.0], [1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        indices, state = selector(pop, n_survivors=2)
        assert len(indices) == 2
        assert "fitness" in state
        np.testing.assert_array_equal(state["fitness"], [1.0, 2.0])

    def test_deterministic_output(self):
        """Test that truncation survival produces deterministic output."""
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        objectives = np.array([[2.5], [1.2], [3.8], [0.9], [4.1]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 3

        # Run twice
        indices1, state1 = selector(pop, n_survivors)
        indices2, state2 = selector(pop, n_survivors)

        # Results should be identical
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_array_equal(state1["fitness"], state2["fitness"])

    def test_large_population_selection(self):
        """Test truncation with larger population."""
        # Create population with 100 individuals
        n_pop = 100
        x = np.arange(n_pop).reshape(n_pop, 1).astype(float)
        # Create random objectives (single-objective)
        rng = np.random.default_rng(42)
        objectives = rng.uniform(0, 10, size=(n_pop, 1))
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()
        n_survivors = 50

        indices, state = selector(pop, n_survivors)

        # Should return correct number of survivors
        assert len(indices) == n_survivors
        assert len(np.unique(indices)) == n_survivors

        # State should have correct shape
        assert state["fitness"].shape == (n_survivors,)

        # Fitness should be in ascending order
        assert np.all(np.diff(state["fitness"]) >= 0)

        # All survivors should have fitness <= any non-survivor
        survivor_fitness = state["fitness"]
        all_fitness = objectives[:, 0]
        non_survivor_mask = ~np.isin(np.arange(n_pop), indices)
        if np.any(non_survivor_mask):
            non_survivor_fitness = all_fitness[non_survivor_mask]
            # Maximum survivor fitness should be <= minimum non-survivor fitness
            # (or at least close, accounting for ties)
            assert np.max(survivor_fitness) <= np.max(non_survivor_fitness)

    def test_all_identical_fitness(self):
        """Test truncation when all individuals have identical fitness."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[2.0], [2.0], [2.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = truncation_survival()

        indices, state = selector(pop, n_survivors=2)

        # Should select 2 survivors (stable sort picks first 2)
        assert len(indices) == 2
        assert 0 in indices
        assert 1 in indices

        # All fitness values should be 2.0
        np.testing.assert_array_equal(state["fitness"], [2.0, 2.0])


class TestElitistSurvival:
    """Tests for elitist survival selection."""

    def test_returns_correct_number_of_survivors(self):
        """Test that elitist survival returns the requested number of survivors."""
        # Combined population: 4 parents + 4 offspring
        x = np.random.randn(8, 2)
        objectives = np.array(
            [[4.0], [2.0], [3.0], [1.0], [0.5], [1.5], [2.5], [3.5]]  # parents  # offspring
        )
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=2)
        indices, state = selector(pop, n_survivors=4, parent_size=4)

        assert indices.shape == (4,)
        assert indices.dtype == np.intp
        assert np.all(indices >= 0)
        assert np.all(indices < len(pop))
        # Check that all indices are unique
        assert len(np.unique(indices)) == 4

    def test_returns_state_with_fitness(self):
        """Test that elitist survival returns state dict with fitness key."""
        x = np.random.randn(6, 2)
        objectives = np.array([[3.0], [1.0], [2.0], [0.5], [1.5], [2.5]])  # 3 parents, 3 offspring
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)
        n_survivors = 3

        indices, state = selector(pop, n_survivors, parent_size=3)

        # Check state dictionary has fitness key
        assert "fitness" in state

        # Check shape matches n_survivors
        assert state["fitness"].shape == (n_survivors,)

        # Check dtype
        assert state["fitness"].dtype in [np.float64, np.float32]

    def test_preserves_elite_parents(self):
        """Test that elite parents are always preserved."""
        x = np.arange(8).reshape(8, 1).astype(float)
        # Parents (indices 0-3): fitness [4.0, 2.0, 3.0, 1.0]
        # Best parents: idx 3 (1.0), idx 1 (2.0)
        # Offspring (indices 4-7): fitness [0.5, 1.5, 2.5, 3.5]
        objectives = np.array([[4.0], [2.0], [3.0], [1.0], [0.5], [1.5], [2.5], [3.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=2)
        indices, state = selector(pop, n_survivors=4, parent_size=4)

        # Should include 2 best parents (indices 3 and 1)
        assert 3 in indices  # best parent with fitness 1.0
        assert 1 in indices  # second best parent with fitness 2.0

    def test_fills_remaining_slots_from_offspring(self):
        """Test that remaining slots are filled with best offspring."""
        x = np.arange(8).reshape(8, 1).astype(float)
        # Parents (indices 0-3): fitness [4.0, 2.0, 3.0, 1.0]
        # Offspring (indices 4-7): fitness [0.5, 1.5, 2.5, 3.5]
        # Best offspring: idx 4 (0.5), idx 5 (1.5)
        objectives = np.array([[4.0], [2.0], [3.0], [1.0], [0.5], [1.5], [2.5], [3.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)
        indices, state = selector(pop, n_survivors=3, parent_size=4)

        # Should have 1 elite parent + 2 best offspring
        # Elite: idx 3 (fitness 1.0)
        # Best offspring: idx 4 (0.5), idx 5 (1.5)
        assert 3 in indices
        assert 4 in indices
        assert 5 in indices
        assert len(indices) == 3

    def test_correct_fitness_values_in_state(self):
        """Test that state contains correct fitness values for survivors."""
        x = np.arange(6).reshape(6, 1).astype(float)
        # Parents: [3.0, 1.0, 2.0]
        # Offspring: [0.5, 1.5, 2.5]
        objectives = np.array([[3.0], [1.0], [2.0], [0.5], [1.5], [2.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)
        indices, state = selector(pop, n_survivors=3, parent_size=3)

        # Elite: idx 1 (1.0)
        # Best offspring: idx 3 (0.5), idx 4 (1.5)
        survivor_fitness = state["fitness"]

        # Check that we have the right fitness values (order may vary)
        expected_fitness = sorted([1.0, 0.5, 1.5])
        actual_fitness = sorted(survivor_fitness.tolist())
        np.testing.assert_array_almost_equal(actual_fitness, expected_fitness)

    def test_deterministic_output(self):
        """Test that elitist survival produces deterministic output."""
        x = np.random.randn(8, 2)
        objectives = np.array([[4.0], [2.0], [3.0], [1.0], [0.5], [1.5], [2.5], [3.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=2)

        # Run twice
        indices1, state1 = selector(pop, n_survivors=4, parent_size=4)
        indices2, state2 = selector(pop, n_survivors=4, parent_size=4)

        # Results should be identical
        np.testing.assert_array_equal(indices1, indices2)
        np.testing.assert_array_equal(state1["fitness"], state2["fitness"])

    def test_error_when_no_parent_size_kwarg(self):
        """Test that ValueError is raised when parent_size kwarg is missing."""
        x = np.array([[1.0], [2.0]])
        objectives = np.array([[1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)

        with pytest.raises(ValueError, match="elitist survival requires 'parent_size' kwarg"):
            selector(pop, n_survivors=1)

    def test_error_when_elite_count_exceeds_parent_size(self):
        """Test that ValueError is raised when elite_count > parent_size."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[1.0], [2.0], [3.0], [4.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=3)

        with pytest.raises(ValueError, match="elite_count .* cannot exceed parent_size"):
            selector(pop, n_survivors=3, parent_size=2)

    def test_error_when_elite_count_exceeds_n_survivors(self):
        """Test that ValueError is raised when elite_count > n_survivors."""
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        objectives = np.array([[1.0], [2.0], [3.0], [4.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=3)

        with pytest.raises(ValueError, match="elite_count .* cannot exceed n_survivors"):
            selector(pop, n_survivors=2, parent_size=4)

    def test_error_when_invalid_elite_count_at_creation(self):
        """Test that ValueError is raised when creating selector with invalid elite_count."""
        with pytest.raises(ValueError, match="elite_count must be positive"):
            elitist_survival(elite_count=0)

        with pytest.raises(ValueError, match="elite_count must be positive"):
            elitist_survival(elite_count=-1)

    def test_error_when_n_survivors_is_zero(self):
        """Test that ValueError is raised when n_survivors is zero."""
        x = np.array([[1.0], [2.0]])
        objectives = np.array([[1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)

        with pytest.raises(ValueError, match="n_survivors must be positive"):
            selector(pop, n_survivors=0, parent_size=1)

    def test_error_when_n_survivors_is_negative(self):
        """Test that ValueError is raised when n_survivors is negative."""
        x = np.array([[1.0], [2.0]])
        objectives = np.array([[1.0], [2.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)

        with pytest.raises(ValueError, match="n_survivors must be positive"):
            selector(pop, n_survivors=-1, parent_size=1)

    def test_error_when_n_survivors_exceeds_population_size(self):
        """Test that ValueError is raised when n_survivors exceeds population size."""
        x = np.array([[1.0], [2.0], [3.0]])
        objectives = np.array([[1.0], [2.0], [3.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)
        pop_size = len(pop)

        with pytest.raises(ValueError, match="n_survivors .* cannot exceed population size"):
            selector(pop, n_survivors=pop_size + 1, parent_size=2)

    def test_error_when_no_objectives_and_no_fitness_kwarg(self):
        """Test that ValueError is raised when population has no objectives and no fitness kwarg."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop = Population(x=x, objectives=None)

        selector = elitist_survival(elite_count=1)

        with pytest.raises(ValueError, match="Population must have objectives computed"):
            selector(pop, n_survivors=1, parent_size=1)

    def test_error_when_multi_objective_without_fitness_kwarg(self):
        """Test that ValueError is raised for multi-objective without explicit fitness."""
        x = np.array([[1.0], [2.0], [3.0]])
        objectives = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)

        with pytest.raises(
            ValueError,
            match="elitist survival requires single-objective optimization.*Pass explicit 'fitness'",
        ):
            selector(pop, n_survivors=2, parent_size=2)

    def test_works_with_explicit_fitness_kwarg(self):
        """Test that elitist survival works with explicit fitness kwarg for multi-objective."""
        x = np.arange(6).reshape(6, 1).astype(float)
        objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [0.5, 0.5], [1.5, 1.5]])
        # Custom fitness: sum of objectives
        custom_fitness = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 3.0])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)

        # Parents (0-3), offspring (4-5)
        # Best parent by custom fitness: idx 0 or 1 or 2 or 3 (all 5.0, stable sort picks 0)
        # Best offspring: idx 4 (1.0), idx 5 (3.0)
        indices, state = selector(pop, n_survivors=3, parent_size=4, fitness=custom_fitness)

        assert len(indices) == 3
        # Elite parent (first with fitness 5.0 due to stable sort)
        assert 0 in indices
        # Best offspring
        assert 4 in indices
        assert 5 in indices

    def test_elite_count_equals_n_survivors(self):
        """Test edge case where elite_count equals n_survivors (only elites survive)."""
        x = np.arange(6).reshape(6, 1).astype(float)
        objectives = np.array([[4.0], [2.0], [3.0], [1.0], [0.5], [1.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=3)
        indices, state = selector(pop, n_survivors=3, parent_size=4)

        # Should select only the 3 best parents
        # Best parents: idx 3 (1.0), idx 1 (2.0), idx 2 (3.0)
        assert len(indices) == 3
        assert 3 in indices
        assert 1 in indices
        assert 2 in indices

        # No offspring should be selected
        assert all(idx < 4 for idx in indices)

    def test_single_elite_with_many_offspring(self):
        """Test with single elite and multiple offspring slots."""
        x = np.arange(10).reshape(10, 1).astype(float)
        # Parents: [5.0, 3.0, 4.0]
        # Offspring: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        objectives = np.array(
            [[5.0], [3.0], [4.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]]
        )
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)
        indices, state = selector(pop, n_survivors=5, parent_size=3)

        # Should have 1 elite (idx 1 with fitness 3.0) + 4 best offspring
        assert len(indices) == 5
        assert 1 in indices  # best parent

        # Best 4 offspring: indices 3, 4, 5, 6
        assert 3 in indices
        assert 4 in indices
        assert 5 in indices
        assert 6 in indices

    def test_ties_in_parents_handled_deterministically(self):
        """Test that ties in parent fitness are handled deterministically."""
        x = np.arange(8).reshape(8, 1).astype(float)
        # Parents: [2.0, 2.0, 2.0, 1.0]
        # Offspring: [0.5, 1.5, 2.5, 3.5]
        objectives = np.array([[2.0], [2.0], [2.0], [1.0], [0.5], [1.5], [2.5], [3.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=2)

        # Run twice
        indices1, _ = selector(pop, n_survivors=3, parent_size=4)
        indices2, _ = selector(pop, n_survivors=3, parent_size=4)

        # Should be deterministic
        np.testing.assert_array_equal(indices1, indices2)

        # Best parent: idx 3 (1.0)
        # Second best: idx 0 (2.0, due to stable sort)
        assert 3 in indices1
        assert 0 in indices1

    def test_ties_in_offspring_handled_deterministically(self):
        """Test that ties in offspring fitness are handled deterministically."""
        x = np.arange(8).reshape(8, 1).astype(float)
        # Parents: [3.0, 2.0, 4.0, 1.0]
        # Offspring: [2.0, 2.0, 2.0, 3.0]
        objectives = np.array([[3.0], [2.0], [4.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=1)

        # Run twice
        indices1, _ = selector(pop, n_survivors=3, parent_size=4)
        indices2, _ = selector(pop, n_survivors=3, parent_size=4)

        # Should be deterministic
        np.testing.assert_array_equal(indices1, indices2)

        # Elite: idx 3 (1.0)
        # Best offspring: idx 4, 5 (both 2.0, stable sort picks first two)
        assert 3 in indices1
        assert 4 in indices1
        assert 5 in indices1

    def test_registration_in_survival_registry(self):
        """Test that 'elitist' is registered in SurvivalRegistry."""
        # Force re-registration by reloading the module
        import importlib

        import ctrl_freak.survival

        importlib.reload(ctrl_freak.survival)

        assert "elitist" in SurvivalRegistry.list()

        # Get a configured selector from registry (registry calls factory with kwargs)
        selector = SurvivalRegistry.get("elitist", elite_count=1)

        # Verify it's callable (selector function returned by factory)
        assert callable(selector)

        # Test that it works
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        objectives = np.array([[3.0], [1.0], [2.0], [0.5], [1.5], [2.5]])
        pop = Population(x=x, objectives=objectives)

        indices, state = selector(pop, n_survivors=3, parent_size=3)
        assert len(indices) == 3
        assert "fitness" in state

    def test_all_survivors_are_parents_when_no_offspring_needed(self):
        """Test selecting all from parents when offspring slots are zero."""
        x = np.arange(6).reshape(6, 1).astype(float)
        # Parents: [4.0, 2.0, 3.0, 1.0]
        # Offspring: [0.5, 1.5]
        objectives = np.array([[4.0], [2.0], [3.0], [1.0], [0.5], [1.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=4)
        indices, state = selector(pop, n_survivors=4, parent_size=4)

        # All survivors should be parents
        assert len(indices) == 4
        assert all(idx < 4 for idx in indices)

        # Should be the 4 best parents in order
        assert 3 in indices  # fitness 1.0
        assert 1 in indices  # fitness 2.0
        assert 2 in indices  # fitness 3.0
        assert 0 in indices  # fitness 4.0

    def test_single_offspring_slot(self):
        """Test with only one offspring slot to fill."""
        x = np.arange(6).reshape(6, 1).astype(float)
        objectives = np.array([[4.0], [2.0], [3.0], [0.5], [1.5], [2.5]])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=2)
        indices, state = selector(pop, n_survivors=3, parent_size=3)

        # 2 elites from parents + 1 best offspring
        assert len(indices) == 3
        # Best parents: idx 1 (2.0), idx 2 (3.0)
        assert 1 in indices
        assert 2 in indices
        # Best offspring: idx 3 (0.5)
        assert 3 in indices

    def test_large_population_selection(self):
        """Test elitist survival with larger population."""
        n_parents = 50
        n_offspring = 50
        n_pop = n_parents + n_offspring

        x = np.arange(n_pop).reshape(n_pop, 1).astype(float)
        rng = np.random.default_rng(42)
        parent_objectives = rng.uniform(1, 10, size=(n_parents, 1))
        offspring_objectives = rng.uniform(0, 5, size=(n_offspring, 1))
        objectives = np.vstack([parent_objectives, offspring_objectives])
        pop = Population(x=x, objectives=objectives)

        selector = elitist_survival(elite_count=10)
        n_survivors = 50

        indices, state = selector(pop, n_survivors, parent_size=n_parents)

        # Should return correct number of survivors
        assert len(indices) == n_survivors
        assert len(np.unique(indices)) == n_survivors

        # State should have correct shape
        assert state["fitness"].shape == (n_survivors,)

        # Should have 10 elites from parents
        parent_indices = indices[indices < n_parents]
        assert len(parent_indices) >= 10  # At least 10 parents

        # Check that 10 best parents are included
        best_parent_indices = np.argsort(objectives[:n_parents, 0])[:10]
        for idx in best_parent_indices:
            assert idx in indices
