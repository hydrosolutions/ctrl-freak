"""Selection strategies for genetic algorithms."""

from ctrl_freak.registry import SelectionRegistry
from ctrl_freak.selection.crowded import crowded_tournament
from ctrl_freak.selection.roulette import roulette_wheel
from ctrl_freak.selection.tournament import fitness_tournament

# Register built-in selection strategies
SelectionRegistry.register("crowded", crowded_tournament)
SelectionRegistry.register("tournament", fitness_tournament)
SelectionRegistry.register("roulette", roulette_wheel)

__all__ = ["crowded_tournament", "fitness_tournament", "roulette_wheel"]
