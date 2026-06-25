# Project Instructions

## 0. Project Overview

ctrl-freak is an extensible, pure-numpy genetic algorithm framework for single-objective
and multi-objective optimization. It provides:

- `ga()` — a single-objective genetic algorithm.
- `nsga2()` — the NSGA-II multi-objective genetic algorithm (non-dominated sorting +
  crowding-distance survival).

The framework is built on composable primitives and pluggable strategies: parent-selection
and survival operators (registered via lightweight registries), SBX crossover and polynomial
mutation operators, Pareto primitives (dominance, non-dominated sort, crowding distance), and
`Population`/result types. It depends on `numpy` and `joblib`; joblib backs parallel
evaluation via the `n_workers` parameter.

## 1. Python Environment

Use `uv` exclusively.

- Add dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Run commands: `uv run <command>`
- Run tests: `uv run pytest`

Do not use `pip`, `poetry`, `conda`, or `pip-tools` directly.

## 2. Code Style

Use `ruff` for formatting and linting, and `ty` for type checking.

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check
```

If `ty` is not installed yet:

```bash
uv add --dev ty
```

Use modern Python typing syntax:

- Prefer built-in generics: `list[str]`, `dict[str, int]`, `tuple[str, ...]`.
- Prefer `|` unions: `str | None`.
- Avoid importing legacy aliases from `typing` such as `List`, `Dict`, `Tuple`, or `Optional`.
- Import from `typing` only when needed for features with no built-in equivalent, such as `Protocol`, `Literal`, or `NewType`.

### Docstrings

Write all docstrings in **numpy style** (the NumPy/SciPy convention). Do not use Google style.

- Use the standard numpy sections — `Parameters`, `Returns`, `Raises`, `Yields`, `Notes`,
  `Examples` — each title underlined with hyphens.
- Every public function, method, class, and module docstring MUST include an `Examples`
  section, and that example MUST be a runnable doctest that passes under
  `uv run pytest --doctest-modules`. Define every name the example references (imports and
  setup) so it executes standalone.
- Use modern typing in the signature (`list[str]`, `str | None`) and refer to those types in
  the docstring prose; do not restate full type annotations redundantly in the description.

Example:

```python
def add(a: int, b: int) -> int:
    """Add two integers.

    Parameters
    ----------
    a : int
        First addend.
    b : int
        Second addend.

    Returns
    -------
    int
        The sum of `a` and `b`.

    Examples
    --------
    >>> add(2, 3)
    5
    """
    return a + b
```

## 3. Versioning and Releases

The version lives in a single place: the `version` field of `pyproject.toml`. The package
exposes it at runtime via `importlib.metadata`:

```python
# src/ctrl_freak/__init__.py
from importlib.metadata import version

__version__ = version("ctrl-freak")
```

Do NOT duplicate the version string anywhere else, and do NOT bump on every commit.

Bump the version only when preparing a release, using `uv`:

```bash
uv version --bump patch   # or: minor, major
```

Bump `minor` or `major` only when explicitly requested; otherwise `patch`.

Releases are published from GitHub, not from a developer machine. Cutting a GitHub Release
triggers the CI workflow that runs `uv build` and `uv publish` to PyPI via Trusted Publishing
(OIDC, no stored tokens). Do not run `uv publish` or `twine` locally, and do not create
release tags by hand — the GitHub Release creates the tag.

## 4. Testing Complex Data Objects

Prefer third-party testing utilities over manual element-wise assertions when comparing complex data objects.

Avoid manually checking lengths, schemas, coordinates, dimensions, shapes, dtypes, or element-wise equality when a library-specific assertion exists.

### NumPy

Use `numpy.testing`.

```python
import numpy as np

np.testing.assert_array_equal(result, expected)
np.testing.assert_allclose(result, expected)
```

### Xarray

Use `xarray.testing`.

```python
import xarray as xr

xr.testing.assert_equal(result, expected)
xr.testing.assert_identical(result, expected)
xr.testing.assert_allclose(result, expected)
```

### Polars

Use `polars.testing`.

```python
import polars.testing as pl_testing

pl_testing.assert_frame_equal(result_df, expected_df)
pl_testing.assert_series_equal(result_series, expected_series)
```
