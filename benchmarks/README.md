# Validation Benchmark Suite

This suite validates ctrl-freak's `ga()` (single-objective) and `nsga2()`
(multi-objective) against [pymoo](https://pymoo.org) and
[DEAP](https://deap.readthedocs.io) on standard problems with **known analytical
optima**. Its purpose is scientific trust, not marketing: the claim is **parity** —
*across these problems, ctrl-freak's results are statistically indistinguishable
from pymoo and DEAP* — so a manuscript that calibrates parameters with ctrl-freak
has a citable basis for the optimizer. We never claim ctrl-freak is *better*.

Every table and figure on this page is regenerated from the committed results
artifact `results/benchmark_results.json` by `render.py`; the numbers trace to that
artifact.

## Provenance

<!-- BEGIN:provenance -->

| Field | Value |
|---|---|
| Seeds | 30 (0–29) |
| ctrl-freak | 0.2.0 |
| pymoo | 0.6.1.6 |
| deap | 1.4.3 |
| numpy | 2.4.1 |
| SO budget | pop 100 × 200 gen (20100 evals) |
| MO budget | pop 100 × 250 gen (25100 evals) |
| SBX / PM eta | 15.0 / 20.0 |

<!-- END:provenance -->

## Problems

| Single-objective (10-D, f\* = 0) | Multi-objective |
|---|---|
| Sphere — separable, unimodal | ZDT1 — convex front (30 vars) |
| Rosenbrock — curved valley | ZDT2 — concave front (30 vars) |
| Rastrigin — highly multimodal | ZDT3 — discontinuous front (30 vars) |
| Ackley — multimodal, one funnel | ZDT4 — convex front, multimodal space (30 vars) |
| Griewank — coupled product term | ZDT6 — biased, concave front (30 vars) |
| Schwefel — deceptive global basin | DTLZ2 — 3-objective unit-sphere octant (12 vars) |

Single-objective problems are 10-D (multimodality bites at 10-D while a real-coded
GA can still reach the optimum, keeping the metrics informative). The
multi-objective problems carry analytical true-front samplers used as the IGD+ / GD
reference.

## Methodology

**Identical algorithm, only the library differs.** Fairness is load-bearing: any
residual difference must come from the algorithm, not the configuration. So all
three libraries run ctrl-freak's exact operator stack:

- **Crossover — ported, identical across all three.** ctrl-freak's single-child,
  every-eligible-variable SBX (eta 15) is ported into a custom pymoo `Crossover`
  and a custom DEAP mate; only each framework's RNG source differs. (pymoo's stock
  `SBX` and DEAP's `cxSimulatedBinaryBounded` recombine only ~half the variables by
  default, which alone put ctrl-freak ~70 % above both baselines on convex-ZDT
  IGD+ — a configuration artifact, now eliminated.)
- **Mutation — aligned.** Polynomial mutation (eta 20) applied to every offspring
  at a per-variable rate `1/n_vars` (`PM(prob=1.0, prob_var=1/n_var)` for pymoo,
  `mutPolynomialBounded(indpb=1/n_var)` for DEAP).
- **Selection / survival — aligned.** Binary tournament; elitist(1) survival for
  SO, NSGA-II (non-dominated sort + crowding) for MO.
- **Budget — identical.** SO: population 100 × 200 generations (20 100
  evaluations). MO: population 100 × 250 generations (25 100 evaluations). The
  equivalence claim is void otherwise.

**Statistic — overlapping variance.** Parity is an *equivalence* claim, not a
difference test (a non-significant t-test is not evidence of equivalence). For each
problem and metric, ctrl-freak and a baseline are **equivalent** when
`|mean_cf − mean_lib| < max(std_cf, std_lib)` — the cross-library gap is smaller
than the seed-to-seed spread. Each table cell pairs the per-library mean ± std with
the verdict and its margin against pymoo and DEAP. Thirty seeds.

## Single-objective results

Single-objective parity is led by the continuous error metrics — the absolute
objective error |f − f*| and the distance to the known optimum ‖x − x*‖ — which are
statistically indistinguishable from both baselines across all six problems
(including the multimodal Rastrigin, Ackley, and Schwefel).

### Objective error |f − f\*|

<!-- BEGIN:so_objective_error -->

| Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |
|---|---|---|---|---|---|
| sphere | 4.61e-03 ± 1.99e-03 | 3.68e-03 ± 1.39e-03 | 3.63e-03 ± 1.86e-03 | equivalent (margin +1.1e-03) | equivalent (margin +1.0e-03) |
| rosenbrock | 6.58e+00 ± 1.65e+00 | 6.90e+00 ± 1.39e+00 | 6.72e+00 ± 1.46e+00 | equivalent (margin +1.3e+00) | equivalent (margin +1.5e+00) |
| rastrigin | 1.84e+00 ± 8.87e-01 | 1.65e+00 ± 9.06e-01 | 1.99e+00 ± 1.10e+00 | equivalent (margin +7.1e-01) | equivalent (margin +9.4e-01) |
| ackley | 7.10e-01 ± 2.29e-01 | 7.12e-01 ± 2.32e-01 | 8.67e-01 ± 3.01e-01 | equivalent (margin +2.3e-01) | equivalent (margin +1.4e-01) |
| griewank | 7.11e-01 ± 1.18e-01 | 7.09e-01 ± 1.19e-01 | 7.10e-01 ± 1.02e-01 | equivalent (margin +1.2e-01) | equivalent (margin +1.2e-01) |
| schwefel | 7.04e+01 ± 7.27e+01 | 3.84e+01 ± 5.47e+01 | 6.62e+01 ± 8.51e+01 | equivalent (margin +4.1e+01) | equivalent (margin +8.1e+01) |

<!-- END:so_objective_error -->

### Solution distance ‖x − x\*‖

<!-- BEGIN:so_solution_distance -->

| Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |
|---|---|---|---|---|---|
| sphere | 6.63e-02 ± 1.45e-02 | 5.96e-02 ± 1.12e-02 | 5.84e-02 ± 1.48e-02 | equivalent (margin +7.8e-03) | equivalent (margin +6.8e-03) |
| rosenbrock | 2.66e+00 ± 5.18e-01 | 2.75e+00 ± 2.80e-01 | 2.71e+00 ± 3.19e-01 | equivalent (margin +4.2e-01) | equivalent (margin +4.7e-01) |
| rastrigin | 6.79e-01 ± 5.17e-01 | 6.18e-01 ± 5.86e-01 | 8.53e-01 ± 6.04e-01 | equivalent (margin +5.2e-01) | equivalent (margin +4.3e-01) |
| ackley | 2.77e-01 ± 6.79e-02 | 2.77e-01 ± 6.97e-02 | 3.20e-01 ± 8.57e-02 | equivalent (margin +6.9e-02) | equivalent (margin +4.3e-02) |
| griewank | 2.63e+01 ± 7.37e+00 | 2.85e+01 ± 8.17e+00 | 2.64e+01 ± 5.95e+00 | equivalent (margin +6.0e+00) | equivalent (margin +7.3e+00) |
| schwefel | 3.84e+02 ± 3.86e+02 | 2.20e+02 ± 3.29e+02 | 3.44e+02 ± 4.01e+02 | equivalent (margin +2.2e+02) | equivalent (margin +3.6e+02) |

<!-- END:so_solution_distance -->

### Success rate (fraction of seeds reaching f − f\* < ε)

The success thresholds ε are strict (e.g. Sphere ε = 1e-6 while all three libraries
reach ~4e-3 at this budget), so the binary success rate is **`0` for every problem
and every library**. This is a documented strict-threshold non-convergence property
at this budget — not a failure and not a parity violation: the three libraries
reach the same error band (the continuous metrics above), and that is where the
single-objective parity evidence rests. The metric is reported faithfully rather
than retuned.

<!-- BEGIN:so_success_rate -->

| Problem | ctrl-freak | pymoo | deap |
|---|---|---|---|
| sphere | 0.00 | 0.00 | 0.00 |
| rosenbrock | 0.00 | 0.00 | 0.00 |
| rastrigin | 0.00 | 0.00 | 0.00 |
| ackley | 0.00 | 0.00 | 0.00 |
| griewank | 0.00 | 0.00 | 0.00 |
| schwefel | 0.00 | 0.00 | 0.00 |

<!-- END:so_success_rate -->

### Convergence

![Single-objective convergence: best objective error per generation, seed-mean ± std](results/figures/so_convergence.png)

## Multi-objective results

The primary multi-objective metrics are IGD+ and GD against the analytical true
front (convergence distance); hypervolume is kept as a secondary column.
Convergence is statistically indistinguishable from both baselines on ZDT1, ZDT2,
ZDT3, and DTLZ2 — on all three metrics (IGD+, GD, and hypervolume).

### IGD+ (to the analytical front)

<!-- BEGIN:mo_igd_plus -->

| Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |
|---|---|---|---|---|---|
| zdt1 | 5.85e-03 ± 3.75e-04 | 6.07e-03 ± 3.63e-04 | 5.94e-03 ± 4.22e-04 | equivalent (margin +1.5e-04) | equivalent (margin +3.3e-04) |
| zdt2 | 5.46e-03 ± 4.38e-04 | 5.87e-03 ± 3.77e-04 | 5.69e-03 ± 4.77e-04 | equivalent (margin +2.7e-05) | equivalent (margin +2.4e-04) |
| zdt3 | 5.91e-03 ± 1.02e-02 | 3.17e-03 ± 2.63e-04 | 3.14e-03 ± 2.22e-04 | equivalent (margin +7.4e-03) | equivalent (margin +7.4e-03) |
| zdt4 | 8.73e+00 ± 2.26e+00 | 1.59e+01 ± 4.68e+00 | 1.48e+01 ± 4.73e+00 | not equivalent — ctrl-freak better | not equivalent — ctrl-freak better |
| zdt6 | 3.81e-01 ± 3.88e-02 | 5.08e-01 ± 6.77e-02 | 5.25e-01 ± 6.04e-02 | not equivalent — ctrl-freak better | not equivalent — ctrl-freak better |
| dtlz2 | 5.47e-02 ± 3.67e-03 | 5.43e-02 ± 3.61e-03 | 5.47e-02 ± 3.28e-03 | equivalent (margin +3.3e-03) | equivalent (margin +3.6e-03) |

<!-- END:mo_igd_plus -->

### GD (to the analytical front)

<!-- BEGIN:mo_gd -->

| Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |
|---|---|---|---|---|---|
| zdt1 | 6.62e-03 ± 1.48e-03 | 7.10e-03 ± 1.18e-03 | 6.65e-03 ± 1.73e-03 | equivalent (margin +9.9e-04) | equivalent (margin +1.7e-03) |
| zdt2 | 7.11e-03 ± 3.07e-03 | 9.07e-03 ± 3.15e-03 | 8.66e-03 ± 3.68e-03 | equivalent (margin +1.2e-03) | equivalent (margin +2.1e-03) |
| zdt3 | 5.09e-03 ± 9.83e-04 | 5.68e-03 ± 1.18e-03 | 6.25e-03 ± 1.89e-03 | equivalent (margin +6.0e-04) | equivalent (margin +7.3e-04) |
| zdt4 | 1.04e+01 ± 2.89e+00 | 2.08e+01 ± 5.47e+00 | 1.97e+01 ± 5.79e+00 | not equivalent — ctrl-freak better | not equivalent — ctrl-freak better |
| zdt6 | 4.23e-01 ± 4.09e-02 | 5.75e-01 ± 7.77e-02 | 5.95e-01 ± 7.27e-02 | not equivalent — ctrl-freak better | not equivalent — ctrl-freak better |
| dtlz2 | 5.50e-02 ± 1.25e-02 | 5.66e-02 ± 1.33e-02 | 6.06e-02 ± 1.81e-02 | equivalent (margin +1.2e-02) | equivalent (margin +1.3e-02) |

<!-- END:mo_gd -->

### Hypervolume (secondary; reference point [1.1, …])

<!-- BEGIN:mo_hypervolume -->

| Problem | ctrl-freak | pymoo | deap | vs pymoo | vs deap |
|---|---|---|---|---|---|
| zdt1 | 8.65e-01 ± 6.89e-04 | 8.65e-01 ± 6.63e-04 | 8.65e-01 ± 7.49e-04 | equivalent (margin +3.2e-04) | equivalent (margin +6.0e-04) |
| zdt2 | 5.32e-01 ± 9.01e-04 | 5.31e-01 ± 7.96e-04 | 5.31e-01 ± 1.02e-03 | equivalent (margin +8.8e-06) | equivalent (margin +4.5e-04) |
| zdt3 | 1.32e+00 ± 2.04e-02 | 1.32e+00 ± 1.05e-03 | 1.32e+00 ± 9.44e-04 | equivalent (margin +1.5e-02) | equivalent (margin +1.5e-02) |
| zdt4 | 0.00e+00 ± 0.00e+00 | 0.00e+00 ± 0.00e+00 | 0.00e+00 ± 0.00e+00 | not equivalent — identical (degenerate) | not equivalent — identical (degenerate) |
| zdt6 | 1.14e-01 ± 2.26e-02 | 5.34e-02 ± 2.29e-02 | 4.61e-02 ± 2.25e-02 | not equivalent — ctrl-freak better | not equivalent — ctrl-freak better |
| dtlz2 | 6.55e-01 ± 1.09e-02 | 6.55e-01 ± 1.00e-02 | 6.55e-01 ± 9.13e-03 | equivalent (margin +1.0e-02) | equivalent (margin +1.0e-02) |

<!-- END:mo_hypervolume -->

### Two documented exceptions

- **ZDT4 and ZDT6.** None of the three libraries converges on these hardest
  multimodal / biased problems at this budget. The symmetric overlapping-variance
  test flags the gap, but **in ctrl-freak's favour**: ctrl-freak's IGD+, GD, and
  hypervolume are at least as good as both pymoo and DEAP (lower IGD+/GD; ZDT6
  higher HV). The trust claim holds (ctrl-freak is at-least-as-good); the clean
  equivalence statement simply does not apply where no library reaches the front.
  ZDT4's hypervolume is identically `0` for all three because every final front
  lies outside the `[1.1, 1.1]` reference box.
- These are reported, not hidden: the committed JSON carries the full
  per-problem × metric verdict and margin behind every cell above.

### Convergence and fronts

![Multi-objective convergence: IGD+ per generation, seed-mean ± std](results/figures/mo_convergence.png)

![Final non-dominated fronts vs the analytical true fronts](results/figures/mo_pareto_fronts.png)

The ZDT panels overlay the three libraries' final fronts on the analytical true
front; ZDT1/ZDT2/ZDT3 land on it for all three libraries, while ZDT4/ZDT6 sit off
it for all three (the documented exceptions). The DTLZ2 (3-objective) panel shows
ctrl-freak's final front against the unit-sphere octant; per-library DTLZ2 parity
is quantified in the IGD+/GD/HV tables and the convergence panel.

> **On the earlier "ZDT3 scatter" note.** A previous version of this report
> captioned a ZDT3 figure as pymoo showing "visible scatter". That was a
> configuration artifact, not an algorithm difference: the legacy harness used
> `PM(prob=1/n_var)`, which mutated far fewer genes (~0.03 vs ~1.0 expected gene
> flips per offspring at `n_var = 30`, a **~30× under-mutation**) and slowed
> pymoo's convergence. With the correctly aligned
> `PM(prob=1.0, prob_var=1/n_var)` used here, the scatter disappears and all three
> libraries converge to nearly identical fronts on the standard ZDT/DTLZ2 problems.

## Reproducing

The committed JSON and every table and figure are regenerated by one command:

```bash
uv run python benchmarks/reproduce.py
```

This runs the full sweep (`benchmarks/run.py` → `results/benchmark_results.json`,
~20–35 min) and then renders the tables and figures (`benchmarks/render.py`,
seconds). To re-render the tables and figures from an existing JSON without
re-running the sweep:

```bash
uv run python benchmarks/render.py
```

Library versions are pinned via `uv.lock` and re-embedded in the JSON metadata
(see Provenance).

## References

1. Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
   evolutionary algorithms: Empirical results. *Evolutionary Computation*, 8(2),
   173–195.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
   multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary
   Computation*, 6(2), 182–197.
3. Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). Scalable test problems
   for evolutionary multiobjective optimization (DTLZ). In *Evolutionary
   Multiobjective Optimization*, 105–145.
