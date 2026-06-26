"""Consolidated doctest gate for the whole ``benchmarks`` package (s5-owned).

A single CI-collected test that discovers every benchmark module at runtime (a
recursive glob, never a frozen list) and runs its doctests, so the cross-cutting
"runnable doctests" discipline is enforced in CI for the entire suite -- including
modules later steps add (s6's ``render.py`` / ``reproduce.py``) without editing
this file. Modules whose doctests need an optional heavy dependency
(``pymoo`` / ``deap`` / ``matplotlib``) are skipped, not errored, when that
dependency is absent, so the numpy-floor CI job stays green. The legacy
``benchmarks/zdt/`` tree is excluded (s6 removes it).
"""

import doctest
import importlib
import importlib.util
from pathlib import Path

import pytest

# Recursive runtime discovery root: <repo>/benchmarks.
_BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"

# Heavy optional deps each module's *doctests* need to execute. Discovery is
# dynamic; this map only annotates the guard. Modules absent from the map
# (anything a future step adds) default to ``_DEFAULT_REQUIRES`` -- requiring all
# three -- so they run under the full-dev CI job and skip under numpy-floor,
# never erroring.
_REQUIRES: dict[str, tuple[str, ...]] = {
    "benchmarks.config": (),
    "benchmarks.stats": (),
    "benchmarks.run": ("pymoo", "deap"),
    "benchmarks.metrics": ("pymoo",),
    "benchmarks.problems.single_objective": (),
    "benchmarks.problems.multi_objective": ("pymoo",),
    "benchmarks.harness.operators": ("pymoo", "deap"),
    "benchmarks.harness.single_objective": ("pymoo", "deap"),
    "benchmarks.harness.multi_objective": ("pymoo", "deap"),
}
_DEFAULT_REQUIRES: tuple[str, ...] = ("pymoo", "deap", "matplotlib")


def _discover_modules() -> list[str]:
    """Return the dotted names of every benchmark module to doctest.

    Recursively globs ``benchmarks/**/*.py``, skipping the legacy ``zdt/`` tree
    and ``__init__.py`` files.
    """
    modules: list[str] = []
    for path in sorted(_BENCHMARKS_DIR.rglob("*.py")):
        relative = path.relative_to(_BENCHMARKS_DIR.parent)
        if "zdt" in relative.parts or path.name == "__init__.py":
            continue
        modules.append(".".join(relative.with_suffix("").parts))
    return modules


_MODULES = _discover_modules()


@pytest.mark.parametrize("module_name", _MODULES)
def test_benchmark_module_doctests(module_name: str) -> None:
    """Run one benchmark module's doctests, skipping if a heavy dep is absent."""
    for dependency in _REQUIRES.get(module_name, _DEFAULT_REQUIRES):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{module_name} doctests require {dependency!r}, which is not installed")
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, verbose=False)
    assert results.attempted > 0, f"{module_name} has no doctests"
    assert results.failed == 0, f"{module_name}: {results.failed} of {results.attempted} doctests failed"
