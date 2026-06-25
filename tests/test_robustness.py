"""Cross-cutting robustness and validation contract tests."""

from typing import Any, cast

import numpy as np
import pytest

from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2
from ctrl_freak.operators import polynomial_mutation, sbx_crossover
from ctrl_freak.population import Population
from ctrl_freak.selection.crowded import crowded_tournament
from ctrl_freak.selection.roulette import roulette_wheel
from ctrl_freak.selection.tournament import fitness_tournament
from ctrl_freak.survival.elitist import elitist_survival
from ctrl_freak.survival.nsga2 import nsga2_survival
from ctrl_freak.survival.truncation import truncation_survival


class TestSeedDeterminism:
    """Regression proof: conftest mutate fixtures ignore global RNG state."""

    def test_small_perturbation_mutate_is_global_state_independent(self, small_perturbation_mutate):
        np.random.seed(424242)
        x = np.array([1.0, 2.0])
        out = small_perturbation_mutate(x)
        expected = x + 0.01 * np.random.default_rng(0xC0FFEE).standard_normal(2)
        np.testing.assert_allclose(out, expected)
        np.testing.assert_allclose(out, np.array([1.01006644, 1.99679809]), rtol=0, atol=1e-8)

    def test_simple_biobj_mutate_is_global_state_independent(self, simple_biobj_problem):
        np.random.seed(987654321)
        mutate = simple_biobj_problem["mutate"]
        x = np.array([0.5, 0.5, 0.5])
        out = mutate(x)
        expected = np.clip(x + 0.01 * np.random.default_rng(0xBADC0DE).standard_normal(3), 0, 1)
        np.testing.assert_allclose(out, expected)
        np.testing.assert_allclose(
            out,
            np.array([0.49084587, 0.48137351, 0.50984759]),
            rtol=0,
            atol=1e-8,
        )


class TestPopulationBoundaries:
    def test_empty_population_constructs(self):
        pop = Population(x=np.zeros((0, 2)), objectives=np.zeros((0, 2)))

        assert len(pop) == 0
        assert pop.n_vars == 2
        assert pop.n_obj == 2

    def test_single_individual_population(self):
        pop = Population(x=np.zeros((1, 2)), objectives=np.zeros((1, 2)))

        assert len(pop) == 1

    def test_population_rejects_1d_x(self):
        with pytest.raises(ValueError, match="x must be 2D"):
            Population(x=np.zeros(3))

    def test_population_rejects_non_array_x(self):
        with pytest.raises(TypeError, match="x must be a numpy array"):
            Population(x=cast(Any, [[1.0, 2.0]]))

    def test_population_rejects_mismatched_objectives(self):
        with pytest.raises(ValueError, match="objectives has 3 individuals, expected 2"):
            Population(x=np.zeros((2, 1)), objectives=np.zeros((3, 1)))

    def test_population_getitem_out_of_bounds(self):
        pop = Population(x=np.zeros((2, 2)))

        with pytest.raises(IndexError, match="out of bounds"):
            pop[10]


class TestAlgorithmBoundaries:
    def test_ga_smallest_even_pop_size(self, sphere_problem):
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=lambda p1, p2: (p1 + p2) / 2,
            mutate=lambda x: x.copy(),
            pop_size=2,
            n_generations=3,
            seed=1,
        )

        assert len(result.population) == 2
        assert result.generations == 3

    def test_ga_single_generation(self, sphere_problem):
        result = ga(
            init=sphere_problem["init"],
            evaluate=sphere_problem["evaluate"],
            crossover=lambda p1, p2: (p1 + p2) / 2,
            mutate=lambda x: x.copy(),
            pop_size=4,
            n_generations=1,
            seed=1,
        )

        assert result.generations == 1
        assert result.evaluations == 8

    def test_nsga2_smallest_even_pop_size(self, simple_biobj_problem):
        result = nsga2(
            init=simple_biobj_problem["init"],
            evaluate=simple_biobj_problem["evaluate"],
            crossover=simple_biobj_problem["crossover"],
            mutate=simple_biobj_problem["mutate"],
            pop_size=2,
            n_generations=1,
            seed=1,
        )

        assert len(result.population) == 2
        assert result.population.objectives is not None
        assert np.isfinite(result.population.objectives).all()

    def test_ga_pop_size_one_rejected(self, sphere_problem):
        with pytest.raises(ValueError, match="pop_size must be even"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=lambda p1, p2: (p1 + p2) / 2,
                mutate=lambda x: x.copy(),
                pop_size=1,
                n_generations=3,
                seed=1,
            )

    def test_ga_float_pop_size_raises(self, sphere_problem):
        with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
            ga(
                init=sphere_problem["init"],
                evaluate=sphere_problem["evaluate"],
                crossover=lambda p1, p2: (p1 + p2) / 2,
                mutate=lambda x: x.copy(),
                pop_size=cast(Any, 2.0),
                n_generations=3,
                seed=1,
            )


class TestDegenerateBounds:
    def test_polynomial_mutation_degenerate_bounds_no_nan(self):
        m = polynomial_mutation(eta=20.0, bounds=(np.array([0.5, 0.5]), np.array([0.5, 0.5])), seed=1)

        out = m(np.array([0.5, 0.5]))

        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out, np.array([0.5, 0.5]))

    def test_sbx_crossover_degenerate_bounds_no_nan(self):
        c = sbx_crossover(eta=20.0, bounds=(np.array([0.5, 0.5]), np.array([0.5, 0.5])), seed=1)

        child = c(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

        assert np.isfinite(child).all()
        np.testing.assert_array_equal(child, np.array([0.5, 0.5]))

    def test_partial_degenerate_bounds_finite_and_within(self):
        lower = np.array([0.0, 0.5])
        upper = np.array([1.0, 0.5])
        m = polynomial_mutation(eta=20.0, bounds=(lower, upper), seed=1)
        c = sbx_crossover(eta=20.0, bounds=(lower, upper), seed=1)

        mo = m(np.array([0.3, 0.5]))
        co = c(np.array([0.3, 0.5]), np.array([0.7, 0.5]))

        assert np.isfinite(mo).all()
        assert np.isfinite(co).all()
        assert (mo >= lower).all()
        assert (mo <= upper).all()
        assert (co >= lower).all()
        assert (co <= upper).all()
        assert mo[1] == 0.5
        assert co[1] == 0.5


class TestSurvivorErrorContracts:
    def test_nsga2_survival_requires_objectives(self):
        with pytest.raises(ValueError, match="must have objectives"):
            nsga2_survival()(Population(x=np.zeros((2, 2))), 1)

    def test_nsga2_survival_n_survivors_nonpositive(self):
        with pytest.raises(ValueError, match="n_survivors must be positive"):
            nsga2_survival()(Population(x=np.zeros((2, 1)), objectives=np.zeros((2, 2))), 0)

    def test_nsga2_survival_n_survivors_exceeds_pop(self):
        match = r"n_survivors \(5\) cannot exceed population size \(2\)"

        with pytest.raises(ValueError, match=match):
            nsga2_survival()(Population(x=np.zeros((2, 1)), objectives=np.zeros((2, 2))), 5)

    def test_truncation_n_survivors_exceeds_pop(self):
        with pytest.raises(ValueError, match=r"cannot exceed population size \(2\)"):
            truncation_survival()(Population(x=np.zeros((2, 1)), objectives=np.zeros((2, 1))), 5)

    def test_elitist_requires_parent_size(self):
        with pytest.raises(ValueError, match="requires 'parent_size'"):
            elitist_survival(elite_count=1)(Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 1))), 2)

    def test_elitist_parent_size_must_be_int(self):
        with pytest.raises(ValueError, match="parent_size must be an integer"):
            elitist_survival(elite_count=1)(
                Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 1))),
                2,
                parent_size=2.5,
            )

    def test_elitist_single_objective_required_for_multiobj(self):
        with pytest.raises(ValueError, match="requires single-objective"):
            elitist_survival(elite_count=1)(
                Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 2))),
                2,
                parent_size=2,
            )


class TestSelectorErrorContracts:
    def test_crowded_tournament_requires_rank(self):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="requires 'rank'"):
            crowded_tournament()(Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 2))), 4, rng)

    def test_crowded_tournament_requires_crowding_distance(self):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="requires 'crowding_distance'"):
            crowded_tournament()(
                Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 2))),
                4,
                rng,
                rank=np.array([0, 0, 1, 1]),
            )

    def test_fitness_tournament_requires_fitness_source(self):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="requires 'fitness'"):
            fitness_tournament()(Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 2))), 4, rng)

    def test_roulette_requires_fitness_source(self):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="requires 'fitness'"):
            roulette_wheel()(Population(x=np.zeros((4, 1)), objectives=np.zeros((4, 2))), 4, rng)
