"""Selection strategies for genetic algorithms."""

from ctrl_freak.selection.crowded import crowded_tournament
from ctrl_freak.registry import SelectionRegistry

# Register built-in selection strategies
SelectionRegistry.register("crowded", crowded_tournament)

__all__ = ["crowded_tournament"]
