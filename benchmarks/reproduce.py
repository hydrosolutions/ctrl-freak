"""Regenerate every benchmark artifact end to end: sweep, then tables + figures.

This is the single reproducibility entry point for the validation benchmark suite.
It runs the full fair sweep (:func:`benchmarks.run.main` -- 12 problems x 3
libraries x 30 seeds, writing ``benchmarks/results/benchmark_results.json``) and
then renders that JSON into the report tables and figures
(:func:`benchmarks.render.main`). One command rebuilds the committed artifact and
everything derived from it::

    uv run python benchmarks/reproduce.py

The sweep dominates the wall-clock (~20-35 min on a laptop); rendering is seconds.
Library versions are pinned via ``uv.lock`` and re-embedded in the JSON metadata on
every run, so the regenerated artifact stays self-describing.

Examples
--------
>>> from benchmarks import reproduce
>>> callable(reproduce.main)
True
"""

# ruff: noqa: E402  (sys.path bootstrap must precede the first-party imports)

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import render, run


def main() -> None:
    """Run the sweep, then render tables and figures from the fresh JSON.

    Examples
    --------
    >>> callable(main)
    True
    """
    run.main()
    render.main()


if __name__ == "__main__":
    main()
