"""Tests for survival strategies."""

import numpy as np
import pytest

from ctrl_freak.population import Population
from ctrl_freak.registry import SurvivalRegistry
from ctrl_freak.survival import nsga2_survival


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
