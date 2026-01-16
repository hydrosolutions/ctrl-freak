"""Tests for registry system.

This module tests the registry pattern for selection and survival strategies,
following the testing philosophy from CLAUDE.md:
- Test behavior, not implementation
- Each test should fail for one reason
- Assert both exception type and message fragment for error tests
"""

import numpy as np
import pytest

from ctrl_freak import Population
from ctrl_freak.registry import (
    SelectionRegistry,
    SurvivalRegistry,
    list_selections,
    list_survivals,
)


@pytest.fixture(autouse=True)
def isolate_registries():
    """Save and restore registries to ensure test isolation."""
    # Save current state
    saved_selection = SelectionRegistry._registry.copy()
    saved_survival = SurvivalRegistry._registry.copy()
    # Clear for fresh test
    SelectionRegistry._registry = {}
    SurvivalRegistry._registry = {}
    yield
    # Restore original state
    SelectionRegistry._registry = saved_selection
    SurvivalRegistry._registry = saved_survival


class TestSelectionRegistry:
    """Tests for SelectionRegistry class."""

    def test_register_adds_factory_to_registry(self) -> None:
        """Registering a factory adds it to the registry."""

        def factory():
            return lambda pop, n, rng, **kw: np.array([0])

        SelectionRegistry.register("test", factory)
        assert "test" in SelectionRegistry.list()

    def test_register_overwrites_existing_strategy(self) -> None:
        """Registering with same name overwrites previous factory."""

        def factory1():
            return lambda pop, n, rng, **kw: np.array([0])

        def factory2():
            return lambda pop, n, rng, **kw: np.array([1])

        SelectionRegistry.register("test", factory1)
        SelectionRegistry.register("test", factory2)

        # Verify the factory was overwritten by checking list
        assert SelectionRegistry.list() == ["test"]
        # Verify we can get the strategy without error
        _ = SelectionRegistry.get("test")

    def test_get_returns_configured_selector(self) -> None:
        """Getting a strategy returns a configured selector callable."""

        def factory(size: int = 2):
            def selector(pop, n_parents, rng, **kwargs):
                return np.arange(n_parents) * size

            return selector

        SelectionRegistry.register("test", factory)
        selector = SelectionRegistry.get("test", size=3)

        # Create a dummy population
        x = np.array([[1.0, 2.0]] * 10)
        pop = Population(x=x)
        rng = np.random.default_rng(42)

        result = selector(pop, n_parents=5, rng=rng)
        expected = np.array([0, 3, 6, 9, 12])
        np.testing.assert_array_equal(result, expected)

    def test_get_with_default_kwargs(self) -> None:
        """Getting a strategy with no kwargs uses factory defaults."""

        def factory(size: int = 2):
            def selector(pop, n_parents, rng, **kwargs):
                return np.full(n_parents, size)

            return selector

        SelectionRegistry.register("test", factory)
        selector = SelectionRegistry.get("test")

        x = np.array([[1.0, 2.0]] * 10)
        pop = Population(x=x)
        rng = np.random.default_rng(42)

        result = selector(pop, n_parents=3, rng=rng)
        expected = np.array([2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_get_raises_keyerror_for_unknown_strategy(self) -> None:
        """Getting an unregistered strategy raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="Selection strategy 'unknown' not found"):
            SelectionRegistry.get("unknown")

    def test_keyerror_message_lists_available_strategies(self) -> None:
        """KeyError message includes list of available strategies."""
        SelectionRegistry.register("strategy1", lambda: lambda pop, n, rng, **kw: np.array([]))
        SelectionRegistry.register("strategy2", lambda: lambda pop, n, rng, **kw: np.array([]))

        with pytest.raises(KeyError, match="Available strategies: strategy1, strategy2"):
            SelectionRegistry.get("unknown")

    def test_keyerror_message_shows_none_when_empty(self) -> None:
        """KeyError message shows 'none' when no strategies registered."""
        with pytest.raises(KeyError, match="Available strategies: none"):
            SelectionRegistry.get("unknown")

    def test_list_returns_empty_for_new_registry(self) -> None:
        """List returns empty list when no strategies registered."""
        assert SelectionRegistry.list() == []

    def test_list_returns_sorted_strategy_names(self) -> None:
        """List returns all registered strategies in sorted order."""
        SelectionRegistry.register("zebra", lambda: lambda pop, n, rng, **kw: np.array([]))
        SelectionRegistry.register("alpha", lambda: lambda pop, n, rng, **kw: np.array([]))
        SelectionRegistry.register("beta", lambda: lambda pop, n, rng, **kw: np.array([]))

        assert SelectionRegistry.list() == ["alpha", "beta", "zebra"]

    def test_factory_receives_all_kwargs(self) -> None:
        """Factory receives all keyword arguments passed to get()."""
        received_kwargs = {}

        def factory(**kwargs):
            received_kwargs.update(kwargs)
            return lambda pop, n, rng, **kw: np.array([])

        SelectionRegistry.register("test", factory)
        SelectionRegistry.get("test", size=5, temperature=0.5, custom=True)

        assert received_kwargs == {"size": 5, "temperature": 0.5, "custom": True}


class TestSurvivalRegistry:
    """Tests for SurvivalRegistry class."""

    def test_register_adds_factory_to_registry(self) -> None:
        """Registering a factory adds it to the registry."""

        def factory():
            return lambda pop, n, **kw: (np.array([0]), {})

        SurvivalRegistry.register("test", factory)
        assert "test" in SurvivalRegistry.list()

    def test_register_overwrites_existing_strategy(self) -> None:
        """Registering with same name overwrites previous factory."""

        def factory1():
            return lambda pop, n, **kw: (np.array([0]), {})

        def factory2():
            return lambda pop, n, **kw: (np.array([1]), {})

        SurvivalRegistry.register("test", factory1)
        SurvivalRegistry.register("test", factory2)

        # Verify the factory was overwritten by checking list
        assert SurvivalRegistry.list() == ["test"]
        # Verify we can get the strategy without error
        _ = SurvivalRegistry.get("test")

    def test_get_returns_configured_selector(self) -> None:
        """Getting a strategy returns a configured selector callable."""

        def factory(elite_size: int = 1):
            def selector(pop, n_survivors, **kwargs):
                indices = np.arange(n_survivors)
                state = {"elite_size": np.full(n_survivors, elite_size)}
                return indices, state

            return selector

        SurvivalRegistry.register("test", factory)
        selector = SurvivalRegistry.get("test", elite_size=3)

        x = np.array([[1.0, 2.0]] * 10)
        pop = Population(x=x)

        indices, state = selector(pop, n_survivors=5)
        expected_indices = np.array([0, 1, 2, 3, 4])
        expected_state = {"elite_size": np.array([3, 3, 3, 3, 3])}

        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(state["elite_size"], expected_state["elite_size"])

    def test_get_with_default_kwargs(self) -> None:
        """Getting a strategy with no kwargs uses factory defaults."""

        def factory(elite_size: int = 1):
            def selector(pop, n_survivors, **kwargs):
                return np.arange(n_survivors), {"elite_size": elite_size}

            return selector

        SurvivalRegistry.register("test", factory)
        selector = SurvivalRegistry.get("test")

        x = np.array([[1.0, 2.0]] * 10)
        pop = Population(x=x)

        indices, state = selector(pop, n_survivors=3)
        assert state["elite_size"] == 1

    def test_get_raises_keyerror_for_unknown_strategy(self) -> None:
        """Getting an unregistered strategy raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="Survival strategy 'unknown' not found"):
            SurvivalRegistry.get("unknown")

    def test_keyerror_message_lists_available_strategies(self) -> None:
        """KeyError message includes list of available strategies."""
        SurvivalRegistry.register("strategy1", lambda: lambda pop, n, **kw: (np.array([]), {}))
        SurvivalRegistry.register("strategy2", lambda: lambda pop, n, **kw: (np.array([]), {}))

        with pytest.raises(KeyError, match="Available strategies: strategy1, strategy2"):
            SurvivalRegistry.get("unknown")

    def test_keyerror_message_shows_none_when_empty(self) -> None:
        """KeyError message shows 'none' when no strategies registered."""
        with pytest.raises(KeyError, match="Available strategies: none"):
            SurvivalRegistry.get("unknown")

    def test_list_returns_empty_for_new_registry(self) -> None:
        """List returns empty list when no strategies registered."""
        assert SurvivalRegistry.list() == []

    def test_list_returns_sorted_strategy_names(self) -> None:
        """List returns all registered strategies in sorted order."""
        SurvivalRegistry.register("zebra", lambda: lambda pop, n, **kw: (np.array([]), {}))
        SurvivalRegistry.register("alpha", lambda: lambda pop, n, **kw: (np.array([]), {}))
        SurvivalRegistry.register("beta", lambda: lambda pop, n, **kw: (np.array([]), {}))

        assert SurvivalRegistry.list() == ["alpha", "beta", "zebra"]

    def test_factory_receives_all_kwargs(self) -> None:
        """Factory receives all keyword arguments passed to get()."""
        received_kwargs = {}

        def factory(**kwargs):
            received_kwargs.update(kwargs)
            return lambda pop, n, **kw: (np.array([]), {})

        SurvivalRegistry.register("test", factory)
        SurvivalRegistry.get("test", elite_size=10, diversity=True, custom="value")

        assert received_kwargs == {"elite_size": 10, "diversity": True, "custom": "value"}

    def test_selector_returns_tuple_with_indices_and_state(self) -> None:
        """Survival selector returns tuple of (indices, state dict)."""

        def factory():
            def selector(pop, n_survivors, **kwargs):
                indices = np.array([1, 2, 3])
                state = {"rank": np.array([0, 0, 1]), "crowding": np.array([1.0, 2.0, 0.5])}
                return indices, state

            return selector

        SurvivalRegistry.register("test", factory)
        selector = SurvivalRegistry.get("test")

        x = np.array([[1.0, 2.0]] * 10)
        pop = Population(x=x)

        result = selector(pop, n_survivors=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

        indices, state = result
        np.testing.assert_array_equal(indices, np.array([1, 2, 3]))
        assert "rank" in state
        assert "crowding" in state


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_list_selections_returns_selection_registry_list(self) -> None:
        """list_selections() delegates to SelectionRegistry.list()."""
        SelectionRegistry.register("test1", lambda: lambda pop, n, rng, **kw: np.array([]))
        SelectionRegistry.register("test2", lambda: lambda pop, n, rng, **kw: np.array([]))

        assert list_selections() == SelectionRegistry.list()
        assert list_selections() == ["test1", "test2"]

    def test_list_survivals_returns_survival_registry_list(self) -> None:
        """list_survivals() delegates to SurvivalRegistry.list()."""
        SurvivalRegistry.register("test1", lambda: lambda pop, n, **kw: (np.array([]), {}))
        SurvivalRegistry.register("test2", lambda: lambda pop, n, **kw: (np.array([]), {}))

        assert list_survivals() == SurvivalRegistry.list()
        assert list_survivals() == ["test1", "test2"]

    def test_convenience_functions_return_empty_when_no_registrations(self) -> None:
        """Convenience functions return empty lists when no strategies registered."""
        assert list_selections() == []
        assert list_survivals() == []


class TestRegistryIsolation:
    """Tests for ensuring registries are independent."""

    def test_selection_and_survival_registries_are_independent(self) -> None:
        """SelectionRegistry and SurvivalRegistry maintain separate state."""
        SelectionRegistry.register("shared_name", lambda: lambda pop, n, rng, **kw: np.array([]))
        SurvivalRegistry.register("shared_name", lambda: lambda pop, n, **kw: (np.array([]), {}))

        assert list_selections() == ["shared_name"]
        assert list_survivals() == ["shared_name"]

        # Each should only have one entry
        assert len(list_selections()) == 1
        assert len(list_survivals()) == 1

    def test_clearing_one_registry_does_not_affect_other(self) -> None:
        """Clearing one registry does not affect the other."""
        SelectionRegistry.register("sel", lambda: lambda pop, n, rng, **kw: np.array([]))
        SurvivalRegistry.register("surv", lambda: lambda pop, n, **kw: (np.array([]), {}))

        SelectionRegistry._registry = {}

        assert list_selections() == []
        assert list_survivals() == ["surv"]
