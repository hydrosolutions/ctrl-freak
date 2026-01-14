"""Tests for Population and IndividualView data structures.

This module tests the core data structures for NSGA-II optimization,
following the testing philosophy from CLAUDE.md:
- Test behavior, not implementation
- Each test should fail for one reason
- Assert both exception type and message fragment for error tests
"""

import numpy as np
import pytest

from ctrl_freak import IndividualView, Population


class TestPopulationConstruction:
    """Tests for Population construction and validation."""

    def test_constructs_with_x_only(self) -> None:
        """Population can be constructed with only decision variables."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)

        assert len(pop) == 3
        assert pop.n_vars == 2
        assert pop.objectives is None
        assert pop.rank is None
        assert pop.crowding_distance is None

    def test_constructs_with_all_fields(self) -> None:
        """Population can be constructed with all optional fields."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = np.array([[0.5, 0.5], [0.3, 0.7]])
        rank = np.array([0, 1], dtype=np.int64)
        crowding_distance = np.array([1.0, 2.0], dtype=np.float64)

        pop = Population(x=x, objectives=objectives, rank=rank, crowding_distance=crowding_distance)

        assert len(pop) == 2
        assert pop.n_vars == 2
        assert pop.n_obj == 2
        np.testing.assert_array_equal(pop.objectives, objectives)
        np.testing.assert_array_equal(pop.rank, rank)
        np.testing.assert_array_equal(pop.crowding_distance, crowding_distance)

    def test_rejects_non_array_x(self) -> None:
        """Population rejects x that is not a numpy array."""
        with pytest.raises(TypeError, match="x must be a numpy array"):
            Population(x=[[1.0, 2.0], [3.0, 4.0]])

    def test_rejects_1d_x(self) -> None:
        """Population rejects 1D x array."""
        with pytest.raises(ValueError, match="x must be 2D"):
            Population(x=np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d_x(self) -> None:
        """Population rejects 3D x array."""
        with pytest.raises(ValueError, match="x must be 2D"):
            Population(x=np.zeros((2, 3, 4)))

    def test_rejects_non_array_objectives(self) -> None:
        """Population rejects objectives that is not a numpy array."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(TypeError, match="objectives must be a numpy array"):
            Population(x=x, objectives=[[0.5, 0.5]])

    def test_rejects_1d_objectives(self) -> None:
        """Population rejects 1D objectives array."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="objectives must be 2D"):
            Population(x=x, objectives=np.array([0.5, 0.5]))

    def test_rejects_mismatched_objectives_rows(self) -> None:
        """Population rejects objectives with different number of individuals than x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = np.array([[0.5, 0.5]])  # Only 1 row, x has 2
        with pytest.raises(ValueError, match="objectives has 1 individuals, expected 2"):
            Population(x=x, objectives=objectives)

    def test_rejects_non_array_rank(self) -> None:
        """Population rejects rank that is not a numpy array."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(TypeError, match="rank must be a numpy array"):
            Population(x=x, rank=[0])

    def test_rejects_2d_rank(self) -> None:
        """Population rejects 2D rank array."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="rank must be 1D"):
            Population(x=x, rank=np.array([[0]]))

    def test_rejects_mismatched_rank_length(self) -> None:
        """Population rejects rank with different length than x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        rank = np.array([0, 1, 2], dtype=np.int64)  # 3 elements, x has 2 rows
        with pytest.raises(ValueError, match="rank has 3 elements, expected 2"):
            Population(x=x, rank=rank)

    def test_rejects_non_integer_rank(self) -> None:
        """Population rejects rank with non-integer dtype."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="rank must have integer dtype"):
            Population(x=x, rank=np.array([0.5]))

    def test_rejects_non_array_crowding_distance(self) -> None:
        """Population rejects crowding_distance that is not a numpy array."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(TypeError, match="crowding_distance must be a numpy array"):
            Population(x=x, crowding_distance=[1.0])

    def test_rejects_2d_crowding_distance(self) -> None:
        """Population rejects 2D crowding_distance array."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="crowding_distance must be 1D"):
            Population(x=x, crowding_distance=np.array([[1.0]]))

    def test_rejects_mismatched_crowding_distance_length(self) -> None:
        """Population rejects crowding_distance with different length than x."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        cd = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 3 elements, x has 2 rows
        with pytest.raises(ValueError, match="crowding_distance has 3 elements, expected 2"):
            Population(x=x, crowding_distance=cd)

    def test_rejects_non_float_crowding_distance(self) -> None:
        """Population rejects crowding_distance with non-float dtype."""
        x = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="crowding_distance must have float dtype"):
            Population(x=x, crowding_distance=np.array([1], dtype=np.int64))


class TestPopulationImmutability:
    """Tests for Population immutability guarantees."""

    def test_frozen_dataclass_rejects_attribute_assignment(self) -> None:
        """Population is frozen and rejects attribute assignment."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        with pytest.raises(AttributeError):
            pop.x = np.array([[9.0, 9.0]])

    def test_x_is_copied_on_construction(self) -> None:
        """Modifying original x array does not affect population."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pop = Population(x=x)

        # Modify original
        x[0, 0] = 999.0

        # Population should be unaffected
        assert pop.x[0, 0] == 1.0

    def test_objectives_is_copied_on_construction(self) -> None:
        """Modifying original objectives array does not affect population."""
        x = np.array([[1.0, 2.0]])
        objectives = np.array([[0.5, 0.5]])
        pop = Population(x=x, objectives=objectives)

        # Modify original
        objectives[0, 0] = 999.0

        # Population should be unaffected
        assert pop.objectives[0, 0] == 0.5

    def test_rank_is_copied_on_construction(self) -> None:
        """Modifying original rank array does not affect population."""
        x = np.array([[1.0, 2.0]])
        rank = np.array([0], dtype=np.int64)
        pop = Population(x=x, rank=rank)

        # Modify original
        rank[0] = 999

        # Population should be unaffected
        assert pop.rank[0] == 0

    def test_crowding_distance_is_copied_on_construction(self) -> None:
        """Modifying original crowding_distance array does not affect population."""
        x = np.array([[1.0, 2.0]])
        cd = np.array([1.0], dtype=np.float64)
        pop = Population(x=x, crowding_distance=cd)

        # Modify original
        cd[0] = 999.0

        # Population should be unaffected
        assert pop.crowding_distance[0] == 1.0


class TestPopulationGetItem:
    """Tests for Population.__getitem__ (indexing)."""

    def test_returns_individual_view(self) -> None:
        """__getitem__ returns an IndividualView instance."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        individual = pop[0]
        assert isinstance(individual, IndividualView)

    def test_getitem_returns_correct_x(self) -> None:
        """__getitem__ returns correct decision variables."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)

        np.testing.assert_array_equal(pop[0].x, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(pop[1].x, np.array([3.0, 4.0]))
        np.testing.assert_array_equal(pop[2].x, np.array([5.0, 6.0]))

    def test_getitem_returns_correct_objectives(self) -> None:
        """__getitem__ returns correct objectives."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = np.array([[0.5, 0.5], [0.3, 0.7]])
        pop = Population(x=x, objectives=objectives)

        np.testing.assert_array_equal(pop[0].objectives, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(pop[1].objectives, np.array([0.3, 0.7]))

    def test_getitem_returns_none_objectives_when_not_set(self) -> None:
        """__getitem__ returns None for objectives when not evaluated."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        assert pop[0].objectives is None

    def test_getitem_returns_correct_rank(self) -> None:
        """__getitem__ returns correct rank as int."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        rank = np.array([0, 1], dtype=np.int64)
        pop = Population(x=x, rank=rank)

        assert pop[0].rank == 0
        assert pop[1].rank == 1
        assert isinstance(pop[0].rank, int)

    def test_getitem_returns_none_rank_when_not_set(self) -> None:
        """__getitem__ returns None for rank when not sorted."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        assert pop[0].rank is None

    def test_getitem_returns_correct_crowding_distance(self) -> None:
        """__getitem__ returns correct crowding_distance as float."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        cd = np.array([1.5, 2.5], dtype=np.float64)
        pop = Population(x=x, crowding_distance=cd)

        assert pop[0].crowding_distance == 1.5
        assert pop[1].crowding_distance == 2.5
        assert isinstance(pop[0].crowding_distance, float)

    def test_getitem_returns_none_crowding_distance_when_not_set(self) -> None:
        """__getitem__ returns None for crowding_distance when not computed."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        assert pop[0].crowding_distance is None

    def test_negative_indexing(self) -> None:
        """__getitem__ supports negative indexing."""
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pop = Population(x=x)

        np.testing.assert_array_equal(pop[-1].x, np.array([5.0, 6.0]))
        np.testing.assert_array_equal(pop[-2].x, np.array([3.0, 4.0]))
        np.testing.assert_array_equal(pop[-3].x, np.array([1.0, 2.0]))

    def test_index_out_of_bounds_positive(self) -> None:
        """__getitem__ raises IndexError for out of bounds positive index."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(IndexError, match="index 2 is out of bounds"):
            pop[2]

    def test_index_out_of_bounds_negative(self) -> None:
        """__getitem__ raises IndexError for out of bounds negative index."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(IndexError, match="index -3 is out of bounds"):
            pop[-3]

    def test_rejects_non_integer_index(self) -> None:
        """__getitem__ rejects non-integer indices."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(TypeError, match="indices must be integers"):
            pop[0.5]

    def test_rejects_slice_index(self) -> None:
        """__getitem__ rejects slice indices (integer indexing only)."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(TypeError, match="indices must be integers"):
            pop[0:1]

    def test_accepts_numpy_integer(self) -> None:
        """__getitem__ accepts numpy integer types."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        idx = np.int64(1)
        individual = pop[idx]
        np.testing.assert_array_equal(individual.x, np.array([3.0, 4.0]))


class TestPopulationProperties:
    """Tests for Population properties."""

    def test_n_individuals(self) -> None:
        """n_individuals returns correct count."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert pop.n_individuals == 3

    def test_n_individuals_matches_len(self) -> None:
        """n_individuals matches len()."""
        pop = Population(x=np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert pop.n_individuals == len(pop)

    def test_n_vars(self) -> None:
        """n_vars returns correct count."""
        pop = Population(x=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert pop.n_vars == 3

    def test_n_obj_with_objectives(self) -> None:
        """n_obj returns correct count when objectives are set."""
        x = np.array([[1.0, 2.0]])
        objectives = np.array([[0.5, 0.5, 0.5]])
        pop = Population(x=x, objectives=objectives)
        assert pop.n_obj == 3

    def test_n_obj_without_objectives(self) -> None:
        """n_obj returns None when objectives are not set."""
        pop = Population(x=np.array([[1.0, 2.0]]))
        assert pop.n_obj is None


class TestIndividualView:
    """Tests for IndividualView data class."""

    def test_stores_all_fields(self) -> None:
        """IndividualView stores all provided fields."""
        x = np.array([1.0, 2.0])
        objectives = np.array([0.5, 0.5])
        view = IndividualView(x=x, objectives=objectives, rank=0, crowding_distance=1.5)

        np.testing.assert_array_equal(view.x, x)
        np.testing.assert_array_equal(view.objectives, objectives)
        assert view.rank == 0
        assert view.crowding_distance == 1.5

    def test_stores_none_values(self) -> None:
        """IndividualView stores None for optional fields."""
        x = np.array([1.0, 2.0])
        view = IndividualView(x=x, objectives=None, rank=None, crowding_distance=None)

        np.testing.assert_array_equal(view.x, x)
        assert view.objectives is None
        assert view.rank is None
        assert view.crowding_distance is None

    def test_is_frozen(self) -> None:
        """IndividualView is frozen and rejects attribute assignment."""
        view = IndividualView(
            x=np.array([1.0, 2.0]),
            objectives=None,
            rank=None,
            crowding_distance=None,
        )
        with pytest.raises(AttributeError):
            view.x = np.array([9.0, 9.0])

    def test_is_frozen_for_all_attributes(self) -> None:
        """IndividualView rejects assignment to all attributes."""
        view = IndividualView(
            x=np.array([1.0, 2.0]),
            objectives=np.array([0.5]),
            rank=0,
            crowding_distance=1.0,
        )

        with pytest.raises(AttributeError):
            view.objectives = np.array([0.9])

        with pytest.raises(AttributeError):
            view.rank = 1

        with pytest.raises(AttributeError):
            view.crowding_distance = 2.0
