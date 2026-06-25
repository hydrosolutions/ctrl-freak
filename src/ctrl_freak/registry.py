"""Registry system for selection and survival strategies.

This module provides a registry pattern for managing selection and survival
strategies in evolutionary algorithms. Instead of hardcoding strategy
implementations, users can register factories that create configured selectors
and retrieve them by name.

The registry pattern enables:
- **Pluggable strategies**: Swap selection methods without code changes
- **Configuration-driven experiments**: Select strategies by string name from config files
- **Discoverability**: List all available strategies programmatically
- **Factory pattern**: Register functions that create configured selectors

There are two independent registries:
1. **SelectionRegistry**: For parent selection strategies (ParentSelector protocol)
2. **SurvivalRegistry**: For survivor selection strategies (SurvivorSelector protocol)

Examples
--------
Strategies are registered as factories and retrieved by name::

    from ctrl_freak.registry import SelectionRegistry, list_selections

    def tournament_factory(size=2):
        def tournament_selector(pop, n_parents, rng, **kwargs):
            ...
        return tournament_selector

    SelectionRegistry.register("tournament", tournament_factory)
    selector = SelectionRegistry.get("tournament", size=3)
    available = list_selections()
"""

from collections.abc import Callable

from ctrl_freak.protocols import ParentSelector, SurvivorSelector


class SelectionRegistry:
    """Registry for parent selection strategies.

    This class provides a class-level registry for parent selection strategy
    factories. Strategies are registered by name and can be retrieved with
    custom configuration parameters.

    The registry stores factory functions that accept keyword arguments and
    return ParentSelector callables. This enables flexible configuration at
    retrieval time.

    Attributes
    ----------
    _registry : dict[str, Callable[..., ParentSelector]]
        Mapping from strategy names to factory functions.

    Examples
    --------
    A strategy factory returns a callable that satisfies ``ParentSelector``::

        def tournament_factory(size=2):
            def selector(pop, n_parents, rng, **kwargs):
                ...
            return selector

        SelectionRegistry.register("tournament", tournament_factory)
        selector = SelectionRegistry.get("tournament", size=5)
    """

    _registry: dict[str, Callable[..., ParentSelector]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., ParentSelector]) -> None:
        """Register a parent selection strategy factory.

        The factory is a callable that accepts keyword arguments and returns
        a ParentSelector. This enables strategies to be configured at retrieval
        time with custom parameters.

        Parameters
        ----------
        name : str
            Unique name for the strategy. Existing names are overwritten.
        factory : Callable[..., ParentSelector]
            Callable that returns a parent selector.

        Examples
        --------
        >>> from ctrl_freak.registry import SelectionRegistry
        >>> SelectionRegistry.register(
        ...     "__doc_random__",
        ...     lambda: lambda pop, n_parents, rng, **kw: rng.choice(len(pop), n_parents),
        ... )
        >>> "__doc_random__" in SelectionRegistry.list()
        True
        """
        cls._registry[name] = factory

    @classmethod
    def get(cls, name: str, **kwargs) -> ParentSelector:
        """Get a configured parent selector by name.

        Retrieves the factory for the given strategy name and calls it with
        the provided keyword arguments to create a configured selector.

        Parameters
        ----------
        name : str
            Name of the registered strategy.
        **kwargs
            Configuration parameters passed to the factory function.

        Returns
        -------
        ParentSelector
            A configured parent selector callable.

        Raises
        ------
        KeyError
            If the strategy name is not registered.

        Examples
        --------
        >>> from ctrl_freak.registry import SelectionRegistry
        >>> SelectionRegistry.register("__doc_empty__", lambda: lambda pop, n, rng, **kw: [])
        >>> selector = SelectionRegistry.get("__doc_empty__")
        >>> callable(selector)
        True
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "none"
            raise KeyError(f"Selection strategy '{name}' not found. Available strategies: {available}")
        factory = cls._registry[name]
        return factory(**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """Return list of registered strategy names.

        Returns
        -------
        list[str]
            Sorted list of all registered parent selection strategy names.

        Examples
        --------
        >>> from ctrl_freak.registry import SelectionRegistry
        >>> isinstance(SelectionRegistry.list(), list)
        True
        """
        return sorted(cls._registry.keys())


class SurvivalRegistry:
    """Registry for survivor selection strategies.

    This class provides a class-level registry for survivor selection strategy
    factories. Strategies are registered by name and can be retrieved with
    custom configuration parameters.

    The registry stores factory functions that accept keyword arguments and
    return SurvivorSelector callables. This enables flexible configuration at
    retrieval time.

    Attributes
    ----------
    _registry : dict[str, Callable[..., SurvivorSelector]]
        Mapping from strategy names to factory functions.

    Examples
    --------
    A strategy factory returns a callable that satisfies ``SurvivorSelector``::

        def nsga2_factory(preserve_diversity=True):
            def selector(pop, n_survivors, **kwargs):
                ...
            return selector

        SurvivalRegistry.register("nsga2", nsga2_factory)
        selector = SurvivalRegistry.get("nsga2", preserve_diversity=False)
    """

    _registry: dict[str, Callable[..., SurvivorSelector]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., SurvivorSelector]) -> None:
        """Register a survivor selection strategy factory.

        The factory is a callable that accepts keyword arguments and returns
        a SurvivorSelector. This enables strategies to be configured at retrieval
        time with custom parameters.

        Parameters
        ----------
        name : str
            Unique name for the strategy. Existing names are overwritten.
        factory : Callable[..., SurvivorSelector]
            Callable that returns a survivor selector.

        Examples
        --------
        >>> import numpy as np
        >>> from ctrl_freak.registry import SurvivalRegistry
        >>> SurvivalRegistry.register(
        ...     "__doc_truncation__",
        ...     lambda: lambda pop, n, **kw: (np.arange(n), {}),
        ... )
        >>> "__doc_truncation__" in SurvivalRegistry.list()
        True
        """
        cls._registry[name] = factory

    @classmethod
    def get(cls, name: str, **kwargs) -> SurvivorSelector:
        """Get a configured survivor selector by name.

        Retrieves the factory for the given strategy name and calls it with
        the provided keyword arguments to create a configured selector.

        Parameters
        ----------
        name : str
            Name of the registered strategy.
        **kwargs
            Configuration parameters passed to the factory function.

        Returns
        -------
        SurvivorSelector
            A configured survivor selector callable.

        Raises
        ------
        KeyError
            If the strategy name is not registered.

        Examples
        --------
        >>> from ctrl_freak.registry import SurvivalRegistry
        >>> SurvivalRegistry.register("__doc_survivor__", lambda: lambda pop, n, **kw: ([], {}))
        >>> selector = SurvivalRegistry.get("__doc_survivor__")
        >>> callable(selector)
        True
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "none"
            raise KeyError(f"Survival strategy '{name}' not found. Available strategies: {available}")
        factory = cls._registry[name]
        return factory(**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """Return list of registered strategy names.

        Returns
        -------
        list[str]
            Sorted list of all registered survivor selection strategy names.

        Examples
        --------
        >>> from ctrl_freak.registry import SurvivalRegistry
        >>> isinstance(SurvivalRegistry.list(), list)
        True
        """
        return sorted(cls._registry.keys())


def list_selections() -> list[str]:
    """List all registered parent selection strategies.

    Convenience function that returns SelectionRegistry.list().

    Returns
    -------
    list[str]
        Sorted list of all registered parent selection strategy names.

    Examples
    --------
    >>> from ctrl_freak.registry import list_selections
    >>> isinstance(list_selections(), list)
    True
    """
    return SelectionRegistry.list()


def list_survivals() -> list[str]:
    """List all registered survivor selection strategies.

    Convenience function that returns SurvivalRegistry.list().

    Returns
    -------
    list[str]
        Sorted list of all registered survivor selection strategy names.

    Examples
    --------
    >>> from ctrl_freak.registry import list_survivals
    >>> isinstance(list_survivals(), list)
    True
    """
    return SurvivalRegistry.list()
