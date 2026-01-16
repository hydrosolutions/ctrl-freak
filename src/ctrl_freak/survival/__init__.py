"""Survival strategies for evolutionary algorithms."""

from ctrl_freak.registry import SurvivalRegistry
from ctrl_freak.survival.nsga2 import nsga2_survival

# Register built-in survival strategies
SurvivalRegistry.register("nsga2", nsga2_survival)

__all__ = ["nsga2_survival"]
