"""Pure-numpy equivalence statistic for the validation benchmark suite.

The milestone substantiates a **parity** ("just as good") claim, which is an
*equivalence* statement, not a difference test. A non-significant t-test is
explicitly insufficient (absence of evidence is not evidence of absence), so this
module implements the chosen, margin-free **overlapping-variance** rule from
``BENCHMARK_PLAN.md`` instead.

For one ``problem x metric`` and a paired set of per-seed values from the
reference library (ctrl-freak) and one baseline (pymoo or DEAP), the two are
**equivalent** iff the gap between their means is smaller than the larger of the
two per-seed standard deviations::

    |mean_reference - mean_candidate| < max(std_reference, std_candidate)

The signed **margin** ``max(std_reference, std_candidate) - |delta_mean|`` is
reported alongside the boolean verdict: a positive margin means the spreads
overlap (equivalent); a negative margin quantifies how far apart the
distributions sit. The comparison is strict (``<``), matching the strict
``< epsilon`` convention used by :func:`benchmarks.metrics.is_success`, so the
degenerate case of two identical zero-variance constants (``delta = threshold =
0``) reports *not equivalent*.

The standard deviation is the population estimate (``numpy`` default,
``ddof=0``), consistent with the rest of the suite. No ``scipy``; ``numpy`` only.

Examples
--------
>>> verdict = equivalence([1.0, 2.0, 3.0], [1.1, 2.0, 2.9])
>>> verdict["equivalent"]
True
>>> round(verdict["margin"], 6) > 0
True
>>> equivalence([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])["equivalent"]
False
"""

import numpy as np

__all__ = ["equivalence"]


def equivalence(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    """Overlapping-variance equivalence verdict for one metric distribution pair.

    Parameters
    ----------
    reference : numpy.ndarray
        Per-seed metric values from the reference library (ctrl-freak), 1-D,
        non-empty.
    candidate : numpy.ndarray
        Per-seed metric values from the baseline library (pymoo or DEAP), 1-D,
        non-empty. Need not be the same length as `reference`.

    Returns
    -------
    dict
        Keys ``mean_reference``, ``std_reference``, ``mean_candidate``,
        ``std_candidate``, ``delta_mean`` (``|mean_reference - mean_candidate|``),
        ``threshold`` (``max(std_reference, std_candidate)``), ``margin``
        (``threshold - delta_mean``; positive iff equivalent), and ``equivalent``
        (``bool``, ``delta_mean < threshold``). All numeric values are Python
        floats so the dict is JSON-serialisable.

    Raises
    ------
    ValueError
        If either array is not 1-D or is empty.

    Examples
    --------
    >>> r = equivalence(np.array([0.10, 0.12, 0.11]), np.array([0.10, 0.11, 0.12]))
    >>> r["equivalent"]
    True
    >>> sorted(r)
    ['delta_mean', 'equivalent', 'margin', 'mean_candidate', 'mean_reference', 'std_candidate', 'std_reference', 'threshold']
    """
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.ndim != 1 or cand.ndim != 1:
        raise ValueError(f"reference and candidate must be 1-D, got {ref.ndim}-D and {cand.ndim}-D")
    if ref.size == 0 or cand.size == 0:
        raise ValueError("reference and candidate must be non-empty")

    mean_reference = float(np.mean(ref))
    std_reference = float(np.std(ref))
    mean_candidate = float(np.mean(cand))
    std_candidate = float(np.std(cand))
    delta_mean = abs(mean_reference - mean_candidate)
    threshold = max(std_reference, std_candidate)
    margin = threshold - delta_mean
    return {
        "mean_reference": mean_reference,
        "std_reference": std_reference,
        "mean_candidate": mean_candidate,
        "std_candidate": std_candidate,
        "delta_mean": delta_mean,
        "threshold": threshold,
        "margin": margin,
        "equivalent": bool(delta_mean < threshold),
    }
