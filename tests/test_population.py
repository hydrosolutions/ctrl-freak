"""Tests for Population and IndividualView data structures.

This module tests the core data structures for multi-objective optimization,
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

    def test_constructs_with_all_fields(self) -> None:
        """Population can be constructed with x and objectives."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = np.array([[0.5, 0.5], [0.3, 0.7]])

        pop = Population(x=x, objectives=objectives)

        assert len(pop) == 2
        assert pop.n_vars == 2
        assert pop.n_obj == 2
        np.testing.assert_array_equal(pop.objectives, objectives)

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
        view = IndividualView(x=x, objectives=objectives)

        np.testing.assert_array_equal(view.x, x)
        np.testing.assert_array_equal(view.objectives, objectives)

    def test_stores_none_values(self) -> None:
        """IndividualView stores None for optional fields."""
        x = np.array([1.0, 2.0])
        view = IndividualView(x=x, objectives=None)

        np.testing.assert_array_equal(view.x, x)
        assert view.objectives is None

    def test_is_frozen(self) -> None:
        """IndividualView is frozen and rejects attribute assignment."""
        view = IndividualView(
            x=np.array([1.0, 2.0]),
            objectives=None,
        )
        with pytest.raises(AttributeError):
            view.x = np.array([9.0, 9.0])

    def test_is_frozen_for_all_attributes(self) -> None:
        """IndividualView rejects assignment to all attributes."""
        view = IndividualView(
            x=np.array([1.0, 2.0]),
            objectives=np.array([0.5]),
        )

        with pytest.raises(AttributeError):
            view.objectives = np.array([0.9])
