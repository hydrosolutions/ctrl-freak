"""Survival strategies for evolutionary algorithms."""

from ctrl_freak.registry import SurvivalRegistry
from ctrl_freak.survival.nsga2 import nsga2_survival
from ctrl_freak.survival.truncation import truncation_survival

# Register built-in survival strategies
SurvivalRegistry.register("nsga2", nsga2_survival)
SurvivalRegistry.register("truncation", truncation_survival)

__all__ = ["nsga2_survival", "truncation_survival"]
