"""Render the committed benchmark results into report tables and figures.

This is the s6 reporting step's renderer. It reads the committed validation
artifact ``benchmarks/results/benchmark_results.json`` (produced by
:mod:`benchmarks.run`) and emits the two human-facing products the pydrology
manuscript cites:

* **Markdown tables** -- per-problem mean +/- std for every metric, each cell
  paired with the overlapping-variance equivalence verdict and margin. The tables
  are injected *in place* between ``<!-- BEGIN:id -->`` / ``<!-- END:id -->`` marker
  comments in ``benchmarks/README.md`` and ``docs/validation.md``, so the prose is
  hand-written but the numbers always trace to the committed JSON.
* **Figures** -- SO and MO convergence curves (seed-mean line + a +/-std band, one
  panel per problem, three libraries) and a final-front Pareto scatter against the
  analytical true fronts. Each figure is written to ``benchmarks/results/figures/``
  (for the GitHub-rendered report) and ``docs/assets/validation/`` (so
  ``mkdocs build --strict`` resolves the image links from inside ``docs/``).

The renderer never re-runs the sweep and never rewrites the JSON; it is a pure
JSON-to-artifacts pass. Run it via :mod:`benchmarks.reproduce` (sweep + render) or
directly::

    uv run python benchmarks/render.py

Examples
--------
>>> from benchmarks.render import _fmt
>>> _fmt(0.00321, 0.000386)
'3.21e-03 ± 3.86e-04'
"""

# ruff: noqa: E402  (sys.path bootstrap + matplotlib backend must precede imports)

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Make ``import benchmarks`` work under a plain ``python benchmarks/render.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.problems.multi_objective import MO_PROBLEMS

LIBRARIES: tuple[str, ...] = ("ctrl-freak", "pymoo", "deap")
"""Reference library first, then the two baselines (display order)."""
BASELINES: tuple[str, ...] = ("pymoo", "deap")
"""Libraries ctrl-freak is adjudicated against."""
SO_ORDER: tuple[str, ...] = ("sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel")
"""Single-objective problems in display order."""
MO_ORDER: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz2")
"""Multi-objective problems in display order (DTLZ2 last: the only 3-objective one)."""

_REPORT_CLAIMS: dict[str, dict[str, tuple[str, ...]]] = {
    # table_id -> {claim -> problems}. Encodes exactly the parity the report prose
    # asserts, so the renderer fails fast if a committed verdict contradicts it.
    "so_objective_error": {"indistinguishable": SO_ORDER},
    "so_solution_distance": {"indistinguishable": SO_ORDER},
    "mo_igd_plus": {"indistinguishable": ("zdt1", "zdt2", "zdt3", "dtlz2"), "at_least_as_good": ("zdt4", "zdt6")},
    "mo_gd": {"indistinguishable": ("zdt1", "zdt2", "zdt3", "dtlz2"), "at_least_as_good": ("zdt4", "zdt6")},
    "mo_hypervolume": {
        "indistinguishable": ("zdt1", "zdt2", "zdt3", "dtlz2"),
        "at_least_as_good": ("zdt4", "zdt6"),
    },
}
"""Parity claims the report prose makes, keyed by the table that backs each.

``indistinguishable`` problems must render an ``equivalent`` verdict against both
baselines; ``at_least_as_good`` problems (the documented ZDT4/ZDT6 exceptions) must
never render ``ctrl-freak worse``. :func:`check_report_consistency` enforces this.
"""

_BENCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = _BENCH_DIR / "results" / "benchmark_results.json"
FIGURES_DIR = _BENCH_DIR / "results" / "figures"
README_PATH = _BENCH_DIR / "README.md"
DOCS_FIGURES_DIR = _REPO_ROOT / "docs" / "assets" / "validation"
VALIDATION_PATH = _REPO_ROOT / "docs" / "validation.md"
_TINY = 1e-12  # log-scale floor for the lower edge of a +/-std band


def _fmt(mean: float, std: float) -> str:
    """Format a ``mean +/- std`` pair in two-significant-figure scientific form.

    Parameters
    ----------
    mean, std : float
        The aggregate mean and population standard deviation of one metric.

    Returns
    -------
    str
        e.g. ``'3.21e-03 ± 3.86e-04'``.

    Examples
    --------
    >>> _fmt(1.0, 0.5)
    '1.00e+00 ± 5.00e-01'
    """
    return f"{mean:.2e} ± {std:.2e}"


def _verdict_cell(verdict: dict, lower_is_better: bool) -> str:
    """Render one overlapping-variance verdict as a compact table cell.

    Parameters
    ----------
    verdict : dict
        An entry of the JSON ``equivalence`` section (keys ``equivalent``,
        ``margin``, ``delta_mean``, ``mean_reference``, ``mean_candidate``).
    lower_is_better : bool
        ``True`` for error metrics (objective_error, solution_distance, IGD+, GD)
        where smaller is better; ``False`` for hypervolume.

    Returns
    -------
    str
        ``'equivalent (margin +M)'`` when the spreads overlap; otherwise the
        direction relative to ctrl-freak (``'... ctrl-freak better/worse'``), or
        ``'... identical (degenerate)'`` for the zero-gap/zero-variance case
        (e.g. ZDT4 hypervolume ``0/0/0``).

    Examples
    --------
    >>> _verdict_cell({"equivalent": True, "margin": 0.0123}, lower_is_better=True)
    'equivalent (margin +1.2e-02)'
    >>> _verdict_cell(
    ...     {"equivalent": False, "delta_mean": 8.0, "mean_reference": 8.7, "mean_candidate": 19.2},
    ...     lower_is_better=True,
    ... )
    'not equivalent — ctrl-freak better'
    >>> _verdict_cell({"equivalent": False, "delta_mean": 0.0}, lower_is_better=False)
    'not equivalent — identical (degenerate)'
    """
    if verdict["equivalent"]:
        return f"equivalent (margin +{verdict['margin']:.1e})"
    if verdict["delta_mean"] == 0.0:
        return "not equivalent — identical (degenerate)"
    better = (
        verdict["mean_reference"] < verdict["mean_candidate"]
        if lower_is_better
        else verdict["mean_reference"] > verdict["mean_candidate"]
    )
    return "not equivalent — ctrl-freak " + ("better" if better else "worse")


def _metric_table(section: dict, order: tuple[str, ...], metric: str, lower_is_better: bool) -> str:
    """Build a per-problem table for one metric (mean +/- std per library + verdicts).

    Parameters
    ----------
    section : dict
        One modality block (``results["single_objective"]`` or
        ``results["multi_objective"]``); read for ``aggregated`` and
        ``equivalence``.
    order : tuple of str
        Problem display order.
    metric : str
        Metric key (e.g. ``"objective_error"``, ``"igd_plus"``).
    lower_is_better : bool
        Passed to :func:`_verdict_cell`.

    Returns
    -------
    str
        A GitHub/MkDocs markdown table.

    Examples
    --------
    >>> section = {
    ...     "aggregated": {"p": {
    ...         "ctrl-freak": {"m": {"mean": 1.0, "std": 0.1}},
    ...         "pymoo": {"m": {"mean": 1.1, "std": 0.1}},
    ...         "deap": {"m": {"mean": 1.2, "std": 0.1}},
    ...     }},
    ...     "equivalence": {"p": {"m": {
    ...         "pymoo": {"equivalent": True, "margin": 0.05},
    ...         "deap": {"equivalent": True, "margin": 0.02},
    ...     }}},
    ... }
    >>> print(_metric_table(section, ("p",), "m", True))  # doctest: +ELLIPSIS
    | Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |
    |---|---|---|---|---|---|
    | p | 1.00e+00 ± 1.00e-01 | ... | equivalent (margin +5.0e-02) | equivalent (margin +2.0e-02) |
    """
    agg = section["aggregated"]
    eq = section["equivalence"]
    lines = [
        "| Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |",
        "|---|---|---|---|---|---|",
    ]
    for p in order:
        cells = [p]
        for lib in LIBRARIES:
            m = agg[p][lib][metric]
            cells.append(_fmt(m["mean"], m["std"]))
        for b in BASELINES:
            cells.append(_verdict_cell(eq[p][metric][b], lower_is_better))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _success_table(section: dict, order: tuple[str, ...]) -> str:
    """Build the SO success-rate table (one library column each).

    Parameters
    ----------
    section : dict
        ``results["single_objective"]``.
    order : tuple of str
        Problem display order.

    Returns
    -------
    str
        A markdown table of per-library success rates.

    Examples
    --------
    >>> section = {"aggregated": {"p": {
    ...     "ctrl-freak": {"success_rate": 0.0},
    ...     "pymoo": {"success_rate": 0.0},
    ...     "deap": {"success_rate": 0.0},
    ... }}}
    >>> print(_success_table(section, ("p",)))
    | Problem | ctrl-freak | pymoo | deap |
    |---|---|---|---|
    | p | 0.00 | 0.00 | 0.00 |
    """
    agg = section["aggregated"]
    lines = ["| Problem | ctrl-freak | pymoo | deap |", "|---|---|---|---|"]
    for p in order:
        cells = [p] + [f"{agg[p][lib]['success_rate']:.2f}" for lib in LIBRARIES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _provenance_table(meta: dict) -> str:
    """Build the provenance table (pinned versions, budgets, seeds) from metadata.

    Parameters
    ----------
    meta : dict
        The ``metadata`` block of the results document.

    Returns
    -------
    str
        A two-column markdown table.

    Examples
    --------
    >>> meta = {
    ...     "n_seeds": 30,
    ...     "versions": {"ctrl_freak": "0.1.0", "pymoo": "0.6.1.6", "deap": "1.4.3", "numpy": "2.4.1"},
    ...     "budgets": {"so_pop_size": 100, "so_n_generations": 200, "mo_pop_size": 100,
    ...                 "mo_n_generations": 250, "sbx_eta": 15.0, "pm_eta": 20.0},
    ... }
    >>> print(_provenance_table(meta))  # doctest: +ELLIPSIS
    | Field | Value |
    |---|---|
    | Seeds | 30 (0–29) |
    | ctrl-freak | 0.1.0 |
    ...
    | SBX / PM eta | 15.0 / 20.0 |
    """
    v = meta["versions"]
    b = meta["budgets"]
    rows = [
        ("Seeds", f"{meta['n_seeds']} (0–{meta['n_seeds'] - 1})"),
        ("ctrl-freak", v["ctrl_freak"]),
        ("pymoo", v["pymoo"]),
        ("deap", v["deap"]),
        ("numpy", v["numpy"]),
        (
            "SO budget",
            f"pop {b['so_pop_size']} × {b['so_n_generations']} gen ({b['so_pop_size'] * (1 + b['so_n_generations'])} evals)",
        ),
        (
            "MO budget",
            f"pop {b['mo_pop_size']} × {b['mo_n_generations']} gen ({b['mo_pop_size'] * (1 + b['mo_n_generations'])} evals)",
        ),
        ("SBX / PM eta", f"{b['sbx_eta']} / {b['pm_eta']}"),
    ]
    lines = ["| Field | Value |", "|---|---|"]
    lines += [f"| {k} | {val} |" for k, val in rows]
    return "\n".join(lines)


def build_tables(results: dict) -> dict[str, str]:
    """Build every injectable table, keyed by its marker id.

    Parameters
    ----------
    results : dict
        The full results document.

    Returns
    -------
    dict
        Mapping ``marker_id -> markdown_table``.

    Examples
    --------
    >>> tables = build_tables(results)  # doctest: +SKIP
    """
    so = results["single_objective"]
    mo = results["multi_objective"]
    return {
        "provenance": _provenance_table(results["metadata"]),
        "so_objective_error": _metric_table(so, SO_ORDER, "objective_error", True),
        "so_solution_distance": _metric_table(so, SO_ORDER, "solution_distance", True),
        "so_success_rate": _success_table(so, SO_ORDER),
        "mo_igd_plus": _metric_table(mo, MO_ORDER, "igd_plus", True),
        "mo_gd": _metric_table(mo, MO_ORDER, "gd", True),
        "mo_hypervolume": _metric_table(mo, MO_ORDER, "hypervolume", False),
    }


def _verdict_rows(table: str) -> dict[str, tuple[str, str]]:
    """Parse a metric table into ``{problem: (vs_pymoo_cell, vs_deap_cell)}``.

    Parameters
    ----------
    table : str
        A markdown metric table built by :func:`_metric_table` (columns
        ``Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap``).

    Returns
    -------
    dict
        Problem name -> its two verdict cells (the final two columns).

    Examples
    --------
    >>> t = "| Problem | a | b | c | vs pymoo | vs deap |\\n|---|---|---|---|---|---|\\n| p | 1 | 1 | 1 | x | y |"
    >>> _verdict_rows(t)["p"]
    ('x', 'y')
    """
    rows: dict[str, tuple[str, str]] = {}
    for line in table.splitlines():
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) < 6 or parts[0] == "Problem" or set(parts[0]) <= set("-"):
            continue
        rows[parts[0]] = (parts[-2], parts[-1])
    return rows


def check_report_consistency(
    tables: dict[str, str], claims: dict[str, dict[str, tuple[str, ...]]] = _REPORT_CLAIMS
) -> None:
    """Raise if any injected verdict cell contradicts the report's prose claims.

    Scans the verdict cells of the freshly built tables -- the exact text injected
    into ``benchmarks/README.md`` and ``docs/validation.md`` -- and raises if a
    problem the prose declares *indistinguishable* renders ``not equivalent`` for
    either baseline, or a problem declared *at least as good* (the documented
    ZDT4/ZDT6 exceptions) renders ``ctrl-freak worse``. This makes it impossible to
    ship a citation page whose prose contradicts a committed verdict, and catches
    any future verdict flip.

    Parameters
    ----------
    tables : dict
        Output of :func:`build_tables` (marker id -> markdown table).
    claims : dict, optional
        The prose's parity claims (defaults to :data:`_REPORT_CLAIMS`).

    Raises
    ------
    ValueError
        Listing every (table, problem, baseline) whose verdict contradicts the
        claimed parity.

    Examples
    --------
    >>> ok = {"m": "| Problem | a | b | c | vs pymoo | vs deap |\\n|---|---|---|---|---|---|\\n| p | 1 | 1 | 1 | equivalent (margin +1.0e-01) | equivalent (margin +1.0e-01) |"}
    >>> check_report_consistency(ok, {"m": {"indistinguishable": ("p",)}}) is None
    True
    >>> bad = {"m": ok["m"].replace("equivalent (margin +1.0e-01) | equivalent", "not equivalent — ctrl-freak worse | equivalent", 1)}
    >>> try:
    ...     check_report_consistency(bad, {"m": {"indistinguishable": ("p",)}})
    ... except ValueError:
    ...     print("raised")
    raised
    """
    violations: list[str] = []
    for table_id, by_claim in claims.items():
        rows = _verdict_rows(tables[table_id])
        for claim, problems in by_claim.items():
            for p in problems:
                for base, cell in zip(BASELINES, rows[p], strict=True):
                    if claim == "indistinguishable" and cell.startswith("not equivalent"):
                        violations.append(f"{table_id}/{p} vs {base}: prose claims indistinguishable, cell is {cell!r}")
                    elif claim == "at_least_as_good" and "worse" in cell:
                        violations.append(f"{table_id}/{p} vs {base}: prose claims at-least-as-good, cell is {cell!r}")
    if violations:
        raise ValueError("report prose contradicts committed verdicts:\n" + "\n".join(violations))


def _replace_markers(text: str, tables: dict[str, str]) -> str:
    """Replace the body of every ``BEGIN/END`` marker pair found in `text`.

    Only ids present in `text` are replaced; ids absent from `text` are ignored, so
    each target file declares the subset of tables it wants by including the
    matching markers.

    Parameters
    ----------
    text : str
        Source markdown.
    tables : dict
        Mapping ``marker_id -> markdown_table``.

    Returns
    -------
    str
        Markdown with each marker body replaced by its table.

    Examples
    --------
    >>> _replace_markers("a <!-- BEGIN:x -->old<!-- END:x --> b", {"x": "NEW"})
    'a <!-- BEGIN:x -->\\n\\nNEW\\n\\n<!-- END:x --> b'
    """
    for tid, table in tables.items():
        pattern = re.compile(
            rf"(<!-- BEGIN:{re.escape(tid)} -->)(.*?)(<!-- END:{re.escape(tid)} -->)",
            re.DOTALL,
        )
        text = pattern.sub(rf"\1\n\n{table}\n\n\3", text)
    return text


def inject_tables(path: Path, tables: dict[str, str]) -> int:
    """Inject `tables` into the marker blocks of the markdown file at `path`.

    Parameters
    ----------
    path : pathlib.Path
        Target markdown file (rewritten in place).
    tables : dict
        Mapping ``marker_id -> markdown_table``.

    Returns
    -------
    int
        The number of marker blocks replaced (for a sanity print).

    Examples
    --------
    >>> inject_tables(README_PATH, build_tables(results))  # doctest: +SKIP
    """
    text = path.read_text()
    before = len(re.findall(r"<!-- BEGIN:", text))
    path.write_text(_replace_markers(text, tables))
    return before


def _plot_band(ax, x: np.ndarray, mean: list[float], std: list[float], label: str) -> None:
    """Plot a seed-mean line with a +/-std band on a log-y axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    x : numpy.ndarray
        Generation indices.
    mean, std : list of float
        The per-generation mean and std curve.
    label : str
        Legend label (library name).

    Examples
    --------
    >>> _plot_band(ax, x, mean, std, "ctrl-freak")  # doctest: +SKIP
    """
    mean_a = np.asarray(mean)
    std_a = np.asarray(std)
    lower = np.maximum(mean_a - std_a, _TINY)
    (line,) = ax.plot(x, mean_a, label=label, linewidth=1.3)
    ax.fill_between(x, lower, mean_a + std_a, alpha=0.2, color=line.get_color())


def figure_convergence(results: dict, modality: str, order: tuple[str, ...], ylabel: str, outputs: list[Path]) -> None:
    """Render a 2x3 grid of seed-mean +/-std convergence curves and save it.

    Parameters
    ----------
    results : dict
        The full results document.
    modality : str
        ``"single_objective"`` or ``"multi_objective"``.
    order : tuple of str
        Problem display order (six names).
    ylabel : str
        Y-axis label (``"best |f − f*|"`` for SO, ``"IGD+"`` for MO).
    outputs : list of pathlib.Path
        Destination PNG paths (both the report and the docs copy).

    Examples
    --------
    >>> figure_convergence(results, "single_objective", SO_ORDER, "best |f − f*|", outputs)  # doctest: +SKIP
    """
    conv = results[modality]["convergence"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, p in zip(axes.flat, order, strict=True):
        x = np.arange(len(conv[p]["ctrl-freak"]["mean"]))
        for lib in LIBRARIES:
            _plot_band(ax, x, conv[p][lib]["mean"], conv[p][lib]["std"], lib)
        ax.set_yscale("log")
        ax.set_title(p)
        ax.set_xlabel("generation")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, outputs)


def figure_mo_pareto(results: dict, outputs: list[Path]) -> None:
    """Render the final-front Pareto scatter (5 ZDT panels 2D, DTLZ2 panel 3D).

    Parameters
    ----------
    results : dict
        The full results document.
    outputs : list of pathlib.Path
        Destination PNG paths.

    Examples
    --------
    >>> figure_mo_pareto(results, outputs)  # doctest: +SKIP
    """
    fronts = results["multi_objective"]["final_fronts"]
    fig = plt.figure(figsize=(13, 7))
    for i, p in enumerate(MO_ORDER):
        problem = MO_PROBLEMS[p]
        true = problem.true_front_sampler()
        if problem.n_obj == 3:
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            ax.scatter(true[:, 0], true[:, 1], true[:, 2], s=4, c="lightgray", label="true front")
            cf = np.asarray(fronts[p]["ctrl-freak"])
            ax.scatter(cf[:, 0], cf[:, 1], cf[:, 2], s=6, label="ctrl-freak")
            ax.set_title(f"{p} (3-obj)")
        else:
            ax = fig.add_subplot(2, 3, i + 1)
            ax.plot(true[:, 0], true[:, 1], c="lightgray", linewidth=2, label="true front")
            for lib in LIBRARIES:
                f = np.asarray(fronts[p][lib])
                ax.scatter(f[:, 0], f[:, 1], s=5, label=lib)
            ax.set_title(p)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
        ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, outputs)


def _save(fig, outputs: list[Path]) -> None:
    """Save a figure to every destination path, creating parent dirs, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    outputs : list of pathlib.Path
        Destination PNG paths.

    Examples
    --------
    >>> _save(fig, [Path("a.png")])  # doctest: +SKIP
    """
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
    plt.close(fig)


def _figure_targets(name: str) -> list[Path]:
    """Return the report and docs destinations for a figure filename.

    Parameters
    ----------
    name : str
        Figure filename (e.g. ``"so_convergence.png"``).

    Returns
    -------
    list of pathlib.Path
        ``[benchmarks/results/figures/<name>, docs/assets/validation/<name>]``.

    Examples
    --------
    >>> [p.name for p in _figure_targets("x.png")]
    ['x.png', 'x.png']
    """
    return [FIGURES_DIR / name, DOCS_FIGURES_DIR / name]


def render(results_path: Path = RESULTS_PATH) -> None:
    """Inject all tables and (re)generate all figures from the committed JSON.

    Parameters
    ----------
    results_path : pathlib.Path, optional
        The committed results artifact (defaults to
        ``benchmarks/results/benchmark_results.json``).

    Examples
    --------
    >>> render()  # doctest: +SKIP
    """
    results = json.loads(Path(results_path).read_text())
    tables = build_tables(results)
    check_report_consistency(tables)  # fail fast: never write a report that contradicts a verdict
    for target in (README_PATH, VALIDATION_PATH):
        n = inject_tables(target, tables)
        print(f"injected {n} table block(s) into {target}")
    figure_convergence(results, "single_objective", SO_ORDER, "best |f − f*|", _figure_targets("so_convergence.png"))
    figure_convergence(results, "multi_objective", MO_ORDER, "IGD+", _figure_targets("mo_convergence.png"))
    figure_mo_pareto(results, _figure_targets("mo_pareto_fronts.png"))
    print(f"wrote figures to {FIGURES_DIR} and {DOCS_FIGURES_DIR}")


def main() -> None:
    """Render tables + figures from the committed results JSON.

    Examples
    --------
    >>> callable(main)
    True
    """
    render()


if __name__ == "__main__":
    main()
