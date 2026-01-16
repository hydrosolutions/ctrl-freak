"""Tests for selection strategies."""

import numpy as np
import pytest

from ctrl_freak.population import Population
from ctrl_freak.registry import SelectionRegistry
from ctrl_freak.selection import crowded_tournament, fitness_tournament


class TestCrowdedTournament:
    """Tests for crowded tournament selection."""

    def test_basic_selection_returns_correct_shape(self, simple_population, rng):
        """Test that basic tournament selection returns the requested number of parents."""
        selector = crowded_tournament(tournament_size=2)
        n_parents = 10
        pop_size = len(simple_population)

        # Create dummy rank and crowding distance
        rank = np.zeros(pop_size, dtype=np.intp)
        crowding_distance = np.ones(pop_size)

        parents = selector(
            simple_population,
            n_parents,
            rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert parents.shape == (n_parents,)
        assert parents.dtype == np.intp
        assert np.all(parents >= 0)
        assert np.all(parents < pop_size)

    def test_rank_preference_lower_rank_wins(self, simple_population, rng):
        """Test that lower rank always wins over higher rank."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)

        # Create scenario: individual 0 has rank 0, all others have rank 1
        rank = np.ones(pop_size, dtype=np.intp)
        rank[0] = 0
        crowding_distance = np.ones(pop_size)  # All equal crowding

        # With many selections, if rank preference works, individual 0 should be selected often
        n_parents = 100
        parents = selector(
            simple_population,
            n_parents,
            rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Individual 0 should appear frequently (much more than 10% of the time)
        # With tournament size 2 and 1 individual at rank 0, probability is:
        # P(selecting 0) = 1 - P(not selecting 0) = 1 - (pop_size-1)/pop_size * (pop_size-1)/pop_size
        selection_count = np.sum(parents == 0)
        assert selection_count > 20  # Should be much more than random (10%)

    def test_crowding_tie_breaking_higher_crowding_wins(self, simple_population, rng):
        """Test that when ranks are equal, higher crowding distance wins."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)

        # All individuals have same rank, but individual 0 has much higher crowding
        rank = np.zeros(pop_size, dtype=np.intp)
        crowding_distance = np.ones(pop_size)
        crowding_distance[0] = 100.0  # Much higher crowding distance

        n_parents = 100
        parents = selector(
            simple_population,
            n_parents,
            rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Individual 0 should be selected frequently due to high crowding distance
        selection_count = np.sum(parents == 0)
        assert selection_count > 20  # Should be much more than random (10%)

    def test_missing_rank_raises_value_error(self, simple_population, rng):
        """Test that missing 'rank' in kwargs raises ValueError."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)
        crowding_distance = np.ones(pop_size)

        with pytest.raises(ValueError, match="crowded tournament selection requires 'rank' in kwargs"):
            selector(
                simple_population,
                10,
                rng,
                crowding_distance=crowding_distance,
            )

    def test_missing_crowding_distance_raises_value_error(self, simple_population, rng):
        """Test that missing 'crowding_distance' in kwargs raises ValueError."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)
        rank = np.zeros(pop_size, dtype=np.intp)

        with pytest.raises(ValueError, match="crowded tournament selection requires 'crowding_distance' in kwargs"):
            selector(
                simple_population,
                10,
                rng,
                rank=rank,
            )

    def test_determinism_same_seed_same_results(self, simple_population):
        """Test that same RNG seed produces same results."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)
        n_parents = 20

        rank = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        crowding_distance = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

        # Run twice with same seed
        rng1 = np.random.default_rng(42)
        parents1 = selector(
            simple_population,
            n_parents,
            rng1,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        rng2 = np.random.default_rng(42)
        parents2 = selector(
            simple_population,
            n_parents,
            rng2,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        np.testing.assert_array_equal(parents1, parents2)

    def test_tournament_size_effect_larger_more_selective(self, simple_population):
        """Test that larger tournament size is more selective."""
        pop_size = len(simple_population)

        # Create scenario: individual 0 has best rank and crowding
        rank = np.ones(pop_size, dtype=np.intp)
        rank[0] = 0
        crowding_distance = np.ones(pop_size)
        crowding_distance[0] = 10.0

        n_parents = 100

        # Test with tournament size 2
        selector_2 = crowded_tournament(tournament_size=2)
        rng_2 = np.random.default_rng(123)
        parents_2 = selector_2(
            simple_population,
            n_parents,
            rng_2,
            rank=rank,
            crowding_distance=crowding_distance,
        )
        count_2 = np.sum(parents_2 == 0)

        # Test with tournament size 4
        selector_4 = crowded_tournament(tournament_size=4)
        rng_4 = np.random.default_rng(123)
        parents_4 = selector_4(
            simple_population,
            n_parents,
            rng_4,
            rank=rank,
            crowding_distance=crowding_distance,
        )
        count_4 = np.sum(parents_4 == 0)

        # Larger tournament should select the best individual more often
        assert count_4 > count_2

    def test_registration_in_selection_registry(self):
        """Test that 'crowded' is registered in SelectionRegistry."""
        # Force re-registration by reloading the module
        # (other tests may clear registries with autouse fixtures)
        import importlib

        import ctrl_freak.selection
        importlib.reload(ctrl_freak.selection)

        assert "crowded" in SelectionRegistry.list()
        selector = SelectionRegistry.get("crowded", tournament_size=3)

        # Verify it's callable and returns the right signature
        assert callable(selector)


class TestCrowdedTournamentEdgeCases:
    """Test edge cases for crowded tournament selection."""

    def test_single_parent_selection(self, simple_population, rng):
        """Test selecting just one parent."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)

        rank = np.zeros(pop_size, dtype=np.intp)
        crowding_distance = np.ones(pop_size)

        parents = selector(
            simple_population,
            n_parents=1,
            rng=rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        assert parents.shape == (1,)
        assert 0 <= parents[0] < pop_size

    def test_all_same_rank_and_crowding(self, simple_population, rng):
        """Test selection when all individuals are identical in rank and crowding."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)

        # All identical
        rank = np.zeros(pop_size, dtype=np.intp)
        crowding_distance = np.ones(pop_size)

        n_parents = 50
        parents = selector(
            simple_population,
            n_parents,
            rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Should still return valid indices
        assert parents.shape == (n_parents,)
        assert np.all(parents >= 0)
        assert np.all(parents < pop_size)
        # Should have some variety in selection (not all the same)
        assert len(np.unique(parents)) > 1

    def test_infinite_crowding_distance_handled(self, simple_population, rng):
        """Test that infinite crowding distance is handled correctly."""
        selector = crowded_tournament(tournament_size=2)
        pop_size = len(simple_population)

        rank = np.zeros(pop_size, dtype=np.intp)
        crowding_distance = np.ones(pop_size)
        crowding_distance[0] = np.inf  # Infinite crowding distance

        n_parents = 50
        parents = selector(
            simple_population,
            n_parents,
            rng,
            rank=rank,
            crowding_distance=crowding_distance,
        )

        # Individual 0 with infinite crowding should be selected frequently
        selection_count = np.sum(parents == 0)
        assert selection_count > 10  # Should be selected often


class TestFitnessTournament:
    """Tests for fitness tournament selection."""

    def test_basic_selection_returns_correct_shape(self, simple_population, rng):
        """Test that basic tournament selection returns the requested number of parents."""
        selector = fitness_tournament(tournament_size=2)
        n_parents = 10
        pop_size = len(simple_population)

        # Create dummy fitness array (single-objective)
        fitness = np.array([1.0, 2.0, 3.0, 4.0])

        parents = selector(
            simple_population,
            n_parents,
            rng,
            fitness=fitness,
        )

        assert parents.shape == (n_parents,)
        assert parents.dtype == np.intp
        assert np.all(parents >= 0)
        assert np.all(parents < pop_size)

    def test_fitness_preference_lower_wins(self, simple_population, rng):
        """Test that lower fitness always wins (minimization)."""
        selector = fitness_tournament(tournament_size=2)
        pop_size = len(simple_population)

        # Create scenario: individual 0 has much lower fitness than all others
        fitness = np.array([0.1, 10.0, 10.0, 10.0])

        # With many selections, if fitness preference works, individual 0 should be selected often
        n_parents = 100
        parents = selector(
            simple_population,
            n_parents,
            rng,
            fitness=fitness,
        )

        # Individual 0 should appear frequently (much more than 25% of the time)
        selection_count = np.sum(parents == 0)
        assert selection_count > 40  # Should be selected most of the time

    def test_tournament_size_effect(self, simple_population):
        """Test that larger tournament size increases selection pressure."""
        pop_size = len(simple_population)

        # Create scenario: individual 0 has best fitness
        fitness = np.array([1.0, 10.0, 10.0, 10.0])

        n_parents = 100

        # Test with tournament size 2
        selector_2 = fitness_tournament(tournament_size=2)
        rng_2 = np.random.default_rng(123)
        parents_2 = selector_2(
            simple_population,
            n_parents,
            rng_2,
            fitness=fitness,
        )
        count_2 = np.sum(parents_2 == 0)

        # Test with tournament size 4
        selector_4 = fitness_tournament(tournament_size=4)
        rng_4 = np.random.default_rng(123)
        parents_4 = selector_4(
            simple_population,
            n_parents,
            rng_4,
            fitness=fitness,
        )
        count_4 = np.sum(parents_4 == 0)

        # Larger tournament should select the best individual more often
        assert count_4 > count_2

    def test_determinism_same_seed(self, simple_population):
        """Test that same RNG seed produces same results."""
        selector = fitness_tournament(tournament_size=2)
        n_parents = 20

        fitness = np.array([1.0, 2.0, 3.0, 4.0])

        # Run twice with same seed
        rng1 = np.random.default_rng(42)
        parents1 = selector(
            simple_population,
            n_parents,
            rng1,
            fitness=fitness,
        )

        rng2 = np.random.default_rng(42)
        parents2 = selector(
            simple_population,
            n_parents,
            rng2,
            fitness=fitness,
        )

        np.testing.assert_array_equal(parents1, parents2)

    def test_extracts_from_single_objective(self, rng):
        """Test that fitness is extracted from single-column objectives when fitness kwarg not provided."""
        # Create population with single-objective (1 column)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        objectives = np.array([[0.5], [1.5], [2.5], [3.5]])  # Single column
        pop = Population(x=x, objectives=objectives)

        selector = fitness_tournament(tournament_size=2)
        n_parents = 100

        parents = selector(pop, n_parents, rng)

        # Should work without explicit fitness kwarg
        assert parents.shape == (n_parents,)
        # Individual 0 has lowest objective, should be selected most often
        assert np.sum(parents == 0) > 25

    def test_fitness_kwarg_overrides_objectives(self, rng):
        """Test that explicit fitness kwarg takes precedence over objectives."""
        # Create population with single-objective
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        objectives = np.array([[10.0], [20.0], [30.0], [40.0]])  # Individual 0 has lowest
        pop = Population(x=x, objectives=objectives)

        # But provide fitness where individual 3 has lowest
        fitness = np.array([40.0, 30.0, 20.0, 10.0])

        selector = fitness_tournament(tournament_size=2)
        n_parents = 100

        parents = selector(pop, n_parents, rng, fitness=fitness)

        # Should use fitness kwarg, so individual 3 should be selected most often
        assert np.sum(parents == 3) > np.sum(parents == 0)

    def test_error_when_no_fitness_source(self, simple_population, rng):
        """Test that ValueError is raised when no fitness source is available."""
        selector = fitness_tournament(tournament_size=2)

        # simple_population has 2-column objectives (multi-objective)
        with pytest.raises(
            ValueError,
            match="fitness tournament selection requires 'fitness' in kwargs or single-column objectives in population"
        ):
            selector(simple_population, 10, rng)

    def test_error_when_objectives_is_none(self, rng):
        """Test that ValueError is raised when objectives is None and no fitness kwarg."""
        # Create population without objectives
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        pop = Population(x=x, objectives=None)

        selector = fitness_tournament(tournament_size=2)

        with pytest.raises(
            ValueError,
            match="fitness tournament selection requires 'fitness' in kwargs or single-column objectives in population"
        ):
            selector(pop, 10, rng)

    def test_registration_in_selection_registry(self):
        """Test that 'tournament' is registered in SelectionRegistry."""
        # Force re-registration by reloading the module
        import importlib

        import ctrl_freak.selection
        importlib.reload(ctrl_freak.selection)

        assert "tournament" in SelectionRegistry.list()
        selector = SelectionRegistry.get("tournament", tournament_size=3)

        # Verify it's callable
        assert callable(selector)
