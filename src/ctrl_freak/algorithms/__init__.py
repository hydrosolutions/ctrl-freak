"""Evolutionary algorithm implementations.

This module provides pluggable algorithm implementations with configurable
selection and survival strategies.
"""

from ctrl_freak.algorithms.ga import ga
from ctrl_freak.algorithms.nsga2 import nsga2

__all__ = ["ga", "nsga2"]
