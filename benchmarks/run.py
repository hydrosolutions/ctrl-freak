"""Validation sweep: 12 problems x 3 libraries x 30 seeds -> committed JSON.

This script is the executable heart of the validation benchmark suite. It runs
the fair single- and multi-objective harnesses (:mod:`benchmarks.harness`) across
every problem (:mod:`benchmarks.problems`), every library (ctrl-freak, pymoo,
DEAP), and the committed 30-seed list (:data:`benchmarks.config.SEEDS`), applies
the metrics (:mod:`benchmarks.metrics`), aggregates them, and adjudicates the
**parity** claim with the overlapping-variance statistic
(:func:`benchmarks.stats.equivalence`). The result is written to
``benchmarks/results/benchmark_results.json`` -- the committed artifact the
pydrology manuscript's numbers trace back to.

It is **not** a test: it lives in ``benchmarks/`` (outside ``testpaths=["tests"]``)
and is never named ``test_*``, so pytest never collects it. Run it explicitly::

    uv run python benchmarks/run.py

The committed JSON embeds the seeds, the pinned library versions (via
:mod:`importlib.metadata`), the budgets and operator settings, and a UTC
timestamp, so the artifact is self-describing and reproducible.

Convergence history (R6 decision)
---------------------------------
Persisting every per-seed per-generation Pareto front across all 1080 runs would
be tens of megabytes. Instead each run's harness ``history`` is reduced to a
single **scalar convergence metric per generation** -- best objective error for
SO, IGD+ for MO -- and then aggregated to a **mean and std curve per
(problem, library)** across the 30 seeds (not stored per-seed). For the MO
Pareto-front scatter figures, the final non-dominated front of a single
representative seed (the first committed seed) is stored verbatim per
(problem, library). Both feed s6's figures while keeping the JSON ~1-2 MB.

Examples
--------
>>> import numpy as np
>>> _aggregate(np.array([1.0, 2.0, 3.0]))["mean"]
2.0
>>> curve = _aggregate_curve([[2.0, 1.0], [4.0, 1.0]])
>>> (curve["mean"], curve["std"])
([3.0, 1.0], [1.0, 0.0])
"""

# ruff: noqa: E402  (sys.path bootstrap must precede the first-party imports)

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

# Make ``import benchmarks`` work under a plain ``python benchmarks/run.py``
# invocation (where sys.path[0] is benchmarks/, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import config, stats
from benchmarks.harness.multi_objective import run_mo
from benchmarks.harness.single_objective import run_so
from benchmarks.metrics import gd, hypervolume, igd_plus, objective_error, solution_distance, success_rate
from benchmarks.problems.multi_objective import MO_PROBLEMS
from benchmarks.problems.single_objective import SO_PROBLEMS

LIBRARIES: tuple[str, ...] = ("ctrl-freak", "pymoo", "deap")
"""The reference library first, then the two baselines."""
BASELINES: tuple[str, ...] = ("pymoo", "deap")
"""Libraries ctrl-freak is adjudicated against."""
SO_METRICS: tuple[str, ...] = ("objective_error", "solution_distance")
"""Per-seed SO metrics carried into the equivalence verdict."""
MO_METRICS: tuple[str, ...] = ("igd_plus", "gd", "hypervolume")
"""Per-seed MO metrics carried into the equivalence verdict (HV secondary)."""
_PINNED: tuple[str, ...] = ("ctrl_freak", "pymoo", "deap", "numpy")
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "benchmark_results.json"
"""Committed artifact path."""
_STATISTIC = "overlapping_variance: equivalent iff |mean_cf - mean_lib| < max(std_cf, std_lib)"


def _aggregate(values: np.ndarray) -> dict[str, float]:
    """Population mean and std of a per-seed metric array.

    Parameters
    ----------
    values : numpy.ndarray
        Per-seed metric values, 1-D.

    Returns
    -------
    dict
        ``{"mean": float, "std": float}`` (``ddof=0``).

    Examples
    --------
    >>> r = _aggregate(np.array([1.0, 2.0, 3.0]))
    >>> (r["mean"], round(r["std"], 6))
    (2.0, 0.816497)
    """
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _metric_array(records: list[dict], problem: str, library: str, metric: str) -> np.ndarray:
    """Per-seed values of one metric for one (problem, library) from raw records.

    Parameters
    ----------
    records : list of dict
        Raw per-run records (each carries ``problem``, ``library`` and the metric
        keys).
    problem, library, metric : str
        Selectors.

    Returns
    -------
    numpy.ndarray
        The metric's per-seed values, ordered as in `records`.

    Examples
    --------
    >>> recs = [
    ...     {"problem": "sphere", "library": "ctrl-freak", "objective_error": 0.1},
    ...     {"problem": "sphere", "library": "pymoo", "objective_error": 0.2},
    ... ]
    >>> _metric_array(recs, "sphere", "ctrl-freak", "objective_error").tolist()
    [0.1]
    """
    return np.array(
        [r[metric] for r in records if r["problem"] == problem and r["library"] == library],
        dtype=float,
    )


def _aggregate_curve(curves: list[list[float]]) -> dict[str, list[float]]:
    """Mean and std (across seeds) of a stack of equal-length scalar curves.

    Parameters
    ----------
    curves : list of list of float
        One scalar convergence curve per seed; every curve has length
        ``n_generations + 1``.

    Returns
    -------
    dict
        ``{"mean": [...], "std": [...]}`` -- the per-generation mean and std.

    Examples
    --------
    >>> c = _aggregate_curve([[2.0, 1.0], [4.0, 1.0]])
    >>> (c["mean"], c["std"])
    ([3.0, 1.0], [1.0, 0.0])
    """
    mat = np.asarray(curves, dtype=float)
    return {"mean": np.mean(mat, axis=0).tolist(), "std": np.std(mat, axis=0).tolist()}


def _equivalence_section(
    records: list[dict], problems: tuple[str, ...], metrics: tuple[str, ...]
) -> dict[str, dict[str, dict[str, dict[str, float | bool]]]]:
    """Overlapping-variance verdicts for every problem x metric x baseline.

    Parameters
    ----------
    records : list of dict
        Raw per-run records for one modality (SO or MO).
    problems : tuple of str
        Problem names to adjudicate.
    metrics : tuple of str
        Metric keys to adjudicate.

    Returns
    -------
    dict
        ``section[problem][metric][baseline]`` -> the
        :func:`benchmarks.stats.equivalence` verdict (ctrl-freak as reference).

    Examples
    --------
    >>> recs = [
    ...     {"problem": "p", "library": "ctrl-freak", "m": 1.0},
    ...     {"problem": "p", "library": "ctrl-freak", "m": 1.2},
    ...     {"problem": "p", "library": "pymoo", "m": 1.0},
    ...     {"problem": "p", "library": "pymoo", "m": 1.4},
    ...     {"problem": "p", "library": "deap", "m": 5.0},
    ...     {"problem": "p", "library": "deap", "m": 5.2},
    ... ]
    >>> sec = _equivalence_section(recs, ("p",), ("m",))
    >>> sec["p"]["m"]["pymoo"]["equivalent"]
    True
    >>> sec["p"]["m"]["deap"]["equivalent"]
    False
    """
    section: dict[str, dict[str, dict[str, dict[str, float | bool]]]] = {}
    for problem in problems:
        section[problem] = {}
        for metric in metrics:
            cf = _metric_array(records, problem, "ctrl-freak", metric)
            section[problem][metric] = {
                base: stats.equivalence(cf, _metric_array(records, problem, base, metric)) for base in BASELINES
            }
    return section


def run_so_sweep(seeds: list[int]) -> tuple[list[dict], dict, dict]:
    """Run the single-objective sweep over all SO problems x libraries x seeds.

    Parameters
    ----------
    seeds : list of int
        Seeds to run (the committed sweep passes :data:`benchmarks.config.SEEDS`).

    Returns
    -------
    tuple
        ``(records, aggregated, convergence)`` where ``records`` is the flat
        per-run list, ``aggregated[problem][library]`` holds per-metric mean/std
        plus ``success_rate``, and ``convergence[problem][library]`` holds the
        mean/std objective-error curve.

    Examples
    --------
    >>> records, aggregated, convergence = run_so_sweep(config.SEEDS)  # doctest: +SKIP
    """
    records: list[dict] = []
    curves: dict[tuple[str, str], list[list[float]]] = {}
    for name, problem in SO_PROBLEMS.items():
        for library in LIBRARIES:
            seed_curves: list[list[float]] = []
            for seed in seeds:
                result = run_so(problem, library, seed)
                records.append(
                    {
                        "kind": "so",
                        "problem": name,
                        "library": library,
                        "seed": seed,
                        "objective_error": objective_error(result.best_f, problem.f_star),
                        "solution_distance": solution_distance(result.best_x, problem.x_star),
                        "best_f": result.best_f,
                        "n_evaluations": result.n_evaluations,
                    }
                )
                seed_curves.append([abs(value - problem.f_star) for value in result.history])
            curves[(name, library)] = seed_curves

    aggregated: dict[str, dict[str, dict]] = {}
    convergence: dict[str, dict[str, dict]] = {}
    for name, problem in SO_PROBLEMS.items():
        aggregated[name] = {}
        convergence[name] = {}
        for library in LIBRARIES:
            best_f = _metric_array(records, name, library, "best_f")
            aggregated[name][library] = {
                metric: _aggregate(_metric_array(records, name, library, metric)) for metric in SO_METRICS
            }
            aggregated[name][library]["success_rate"] = success_rate(best_f, problem.f_star, problem.epsilon)
            convergence[name][library] = _aggregate_curve(curves[(name, library)])
    return records, aggregated, convergence


def run_mo_sweep(seeds: list[int]) -> tuple[list[dict], dict, dict, dict]:
    """Run the multi-objective sweep over all MO problems x libraries x seeds.

    Parameters
    ----------
    seeds : list of int
        Seeds to run (the committed sweep passes :data:`benchmarks.config.SEEDS`).

    Returns
    -------
    tuple
        ``(records, aggregated, convergence, final_fronts)``. ``records`` is the
        flat per-run list; ``aggregated[problem][library]`` holds per-metric
        mean/std; ``convergence[problem][library]`` holds the mean/std IGD+ curve;
        ``final_fronts[problem][library]`` is the first seed's non-dominated front
        objectives (for s6's Pareto scatter).

    Examples
    --------
    >>> records, aggregated, convergence, fronts = run_mo_sweep(config.SEEDS)  # doctest: +SKIP
    """
    records: list[dict] = []
    curves: dict[tuple[str, str], list[list[float]]] = {}
    final_fronts: dict[str, dict[str, list]] = {}
    representative = seeds[0]
    for name, problem in MO_PROBLEMS.items():
        true_front = problem.true_front_sampler()
        final_fronts[name] = {}
        for library in LIBRARIES:
            seed_curves: list[list[float]] = []
            for seed in seeds:
                result = run_mo(problem, library, seed)
                records.append(
                    {
                        "kind": "mo",
                        "problem": name,
                        "library": library,
                        "seed": seed,
                        "igd_plus": igd_plus(result.front_objectives, true_front),
                        "gd": gd(result.front_objectives, true_front),
                        "hypervolume": hypervolume(result.front_objectives, ref_point=problem.ref_point),
                        "n_front_points": int(result.front_objectives.shape[0]),
                        "n_evaluations": result.n_evaluations,
                    }
                )
                seed_curves.append([igd_plus(snapshot, true_front) for snapshot in result.history])
                if seed == representative:
                    final_fronts[name][library] = result.front_objectives.tolist()
            curves[(name, library)] = seed_curves

    aggregated: dict[str, dict[str, dict]] = {}
    convergence: dict[str, dict[str, dict]] = {}
    for name in MO_PROBLEMS:
        aggregated[name] = {}
        convergence[name] = {}
        for library in LIBRARIES:
            aggregated[name][library] = {
                metric: _aggregate(_metric_array(records, name, library, metric)) for metric in MO_METRICS
            }
            convergence[name][library] = _aggregate_curve(curves[(name, library)])
    return records, aggregated, convergence, final_fronts


def build_results(seeds: list[int]) -> dict:
    """Assemble the full results document (sweep + aggregates + verdicts + meta).

    Parameters
    ----------
    seeds : list of int
        The committed seed list.

    Returns
    -------
    dict
        The JSON-serialisable results document (see the module docstring schema).

    Examples
    --------
    >>> doc = build_results(config.SEEDS)  # doctest: +SKIP
    """
    so_records, so_aggregated, so_convergence = run_so_sweep(seeds)
    mo_records, mo_aggregated, mo_convergence, mo_fronts = run_mo_sweep(seeds)

    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "libraries": list(LIBRARIES),
        "statistic": _STATISTIC,
        "versions": {pkg: version(pkg) for pkg in _PINNED},
        "budgets": {
            "so_pop_size": config.SO_POP_SIZE,
            "so_n_generations": config.SO_N_GENERATIONS,
            "mo_pop_size": config.MO_POP_SIZE,
            "mo_n_generations": config.MO_N_GENERATIONS,
            "sbx_eta": config.SBX_ETA,
            "pm_eta": config.PM_ETA,
            "crossover_prob": config.CROSSOVER_PROB,
        },
    }
    return {
        "metadata": metadata,
        "single_objective": {
            "raw": so_records,
            "aggregated": so_aggregated,
            "equivalence": _equivalence_section(so_records, tuple(SO_PROBLEMS), SO_METRICS),
            "convergence": so_convergence,
        },
        "multi_objective": {
            "raw": mo_records,
            "aggregated": mo_aggregated,
            "equivalence": _equivalence_section(mo_records, tuple(MO_PROBLEMS), MO_METRICS),
            "convergence": mo_convergence,
            "final_fronts": mo_fronts,
        },
    }


def main() -> None:
    """Run the committed 30-seed sweep and write the results JSON.

    Examples
    --------
    >>> callable(main)
    True
    """
    document = build_results(config.SEEDS)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(document, indent=2))
    print(f"wrote {RESULTS_PATH} ({RESULTS_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
