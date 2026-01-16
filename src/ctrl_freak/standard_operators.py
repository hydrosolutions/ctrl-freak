"""Backward compatibility shim for standard_operators.

This module maintains backward compatibility by re-exporting from the new
operators.standard module.

DEPRECATED: Import from ctrl_freak.operators instead.
"""

from ctrl_freak.operators.standard import polynomial_mutation, sbx_crossover

__all__ = ["sbx_crossover", "polynomial_mutation"]
