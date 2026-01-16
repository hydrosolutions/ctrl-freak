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

Basic usage:
    ```python
    from ctrl_freak.registry import SelectionRegistry, list_selections

    # Register a strategy factory
    def tournament_factory(size: int = 2):
        def tournament_selector(pop, n_parents, rng, **kwargs):
            # Implementation here
            ...
        return tournament_selector

    SelectionRegistry.register("tournament", tournament_factory)

    # Get a configured selector
    selector = SelectionRegistry.get("tournament", size=3)

    # List available strategies
    available = list_selections()  # ["tournament", ...]
    ```

Advanced usage with lambdas:
    ```python
    from ctrl_freak.registry import SurvivalRegistry

    # Register with lambda factory
    SurvivalRegistry.register(
        "truncation",
        lambda minimize=True: truncation_selector(minimize)
    )

    # Get with custom configuration
    selector = SurvivalRegistry.get("truncation", minimize=False)
    ```
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

    Class Attributes:
        _registry: Dictionary mapping strategy names to factory functions.
            Keys are strategy names (str), values are callables that return
            ParentSelector instances.

    Example:
        ```python
        # Define a strategy factory
        def tournament_factory(size: int = 2):
            def selector(pop, n_parents, rng, **kwargs):
                fitness = kwargs.get('fitness')
                # Tournament selection implementation
                ...
            return selector

        # Register the factory
        SelectionRegistry.register("tournament", tournament_factory)

        # Retrieve configured selector
        selector = SelectionRegistry.get("tournament", size=5)

        # List all strategies
        strategies = SelectionRegistry.list()  # ["tournament"]
        ```
    """

    _registry: dict[str, Callable[..., ParentSelector]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., ParentSelector]) -> None:
        """Register a parent selection strategy factory.

        The factory is a callable that accepts keyword arguments and returns
        a ParentSelector. This enables strategies to be configured at retrieval
        time with custom parameters.

        Args:
            name: Unique name for the strategy. Will overwrite if already exists.
            factory: Callable that returns a ParentSelector. Should accept
                keyword arguments for configuration.

        Example:
            ```python
            # Register a simple factory
            SelectionRegistry.register(
                "random",
                lambda: lambda pop, n, rng, **kw: rng.choice(len(pop), n)
            )

            # Register a configurable factory
            def tournament_factory(size: int = 2):
                def selector(pop, n_parents, rng, **kwargs):
                    # Use 'size' parameter
                    ...
                return selector

            SelectionRegistry.register("tournament", tournament_factory)
            ```
        """
        cls._registry[name] = factory

    @classmethod
    def get(cls, name: str, **kwargs) -> ParentSelector:
        """Get a configured parent selector by name.

        Retrieves the factory for the given strategy name and calls it with
        the provided keyword arguments to create a configured selector.

        Args:
            name: Name of the registered strategy.
            **kwargs: Configuration parameters passed to the factory function.

        Returns:
            A configured ParentSelector callable.

        Raises:
            KeyError: If the strategy name is not registered. Error message
                includes list of available strategies.

        Example:
            ```python
            # Get with default configuration
            selector = SelectionRegistry.get("tournament")

            # Get with custom configuration
            selector = SelectionRegistry.get("tournament", size=5)

            # Use the selector
            parent_indices = selector(pop, n_parents=20, rng=rng, fitness=fitness)
            ```
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "none"
            raise KeyError(f"Selection strategy '{name}' not found. Available strategies: {available}")
        factory = cls._registry[name]
        return factory(**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """Return list of registered strategy names.

        Returns:
            Sorted list of all registered parent selection strategy names.

        Example:
            ```python
            strategies = SelectionRegistry.list()
            print(f"Available: {', '.join(strategies)}")
            ```
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

    Class Attributes:
        _registry: Dictionary mapping strategy names to factory functions.
            Keys are strategy names (str), values are callables that return
            SurvivorSelector instances.

    Example:
        ```python
        # Define a strategy factory
        def nsga2_factory(preserve_diversity: bool = True):
            def selector(pop, n_survivors, **kwargs):
                # NSGA-II implementation
                rank = non_dominated_sort(pop.objectives)
                crowding = compute_crowding(pop.objectives, rank)
                # Truncation logic
                ...
                return indices, {'rank': rank, 'crowding_distance': crowding}
            return selector

        # Register the factory
        SurvivalRegistry.register("nsga2", nsga2_factory)

        # Retrieve configured selector
        selector = SurvivalRegistry.get("nsga2", preserve_diversity=False)

        # List all strategies
        strategies = SurvivalRegistry.list()  # ["nsga2"]
        ```
    """

    _registry: dict[str, Callable[..., SurvivorSelector]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., SurvivorSelector]) -> None:
        """Register a survivor selection strategy factory.

        The factory is a callable that accepts keyword arguments and returns
        a SurvivorSelector. This enables strategies to be configured at retrieval
        time with custom parameters.

        Args:
            name: Unique name for the strategy. Will overwrite if already exists.
            factory: Callable that returns a SurvivorSelector. Should accept
                keyword arguments for configuration.

        Example:
            ```python
            # Register a simple factory
            SurvivalRegistry.register(
                "truncation",
                lambda: lambda pop, n, **kw: (np.argsort(pop.objectives[:, 0])[:n], {})
            )

            # Register a configurable factory
            def elitist_factory(elite_fraction: float = 0.1):
                def selector(pop, n_survivors, **kwargs):
                    n_elite = int(n_survivors * elite_fraction)
                    # Selection logic using elite_fraction
                    ...
                    return indices, state
                return selector

            SurvivalRegistry.register("elitist", elitist_factory)
            ```
        """
        cls._registry[name] = factory

    @classmethod
    def get(cls, name: str, **kwargs) -> SurvivorSelector:
        """Get a configured survivor selector by name.

        Retrieves the factory for the given strategy name and calls it with
        the provided keyword arguments to create a configured selector.

        Args:
            name: Name of the registered strategy.
            **kwargs: Configuration parameters passed to the factory function.

        Returns:
            A configured SurvivorSelector callable.

        Raises:
            KeyError: If the strategy name is not registered. Error message
                includes list of available strategies.

        Example:
            ```python
            # Get with default configuration
            selector = SurvivalRegistry.get("nsga2")

            # Get with custom configuration
            selector = SurvivalRegistry.get("elitist", elite_fraction=0.2)

            # Use the selector
            survivor_indices, state = selector(combined_pop, n_survivors=100)
            ```
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "none"
            raise KeyError(f"Survival strategy '{name}' not found. Available strategies: {available}")
        factory = cls._registry[name]
        return factory(**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """Return list of registered strategy names.

        Returns:
            Sorted list of all registered survivor selection strategy names.

        Example:
            ```python
            strategies = SurvivalRegistry.list()
            print(f"Available: {', '.join(strategies)}")
            ```
        """
        return sorted(cls._registry.keys())


def list_selections() -> list[str]:
    """List all registered parent selection strategies.

    Convenience function that returns SelectionRegistry.list().

    Returns:
        Sorted list of all registered parent selection strategy names.

    Example:
        ```python
        from ctrl_freak.registry import list_selections

        strategies = list_selections()
        for name in strategies:
            print(f"- {name}")
        ```
    """
    return SelectionRegistry.list()


def list_survivals() -> list[str]:
    """List all registered survivor selection strategies.

    Convenience function that returns SurvivalRegistry.list().

    Returns:
        Sorted list of all registered survivor selection strategy names.

    Example:
        ```python
        from ctrl_freak.registry import list_survivals

        strategies = list_survivals()
        for name in strategies:
            print(f"- {name}")
        ```
    """
    return SurvivalRegistry.list()
