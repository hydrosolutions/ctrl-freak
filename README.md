# ctrl-freak

[![PyPI](https://img.shields.io/pypi/v/ctrl-freak.svg)](https://pypi.org/project/ctrl-freak/)
[![Python](https://img.shields.io/pypi/pyversions/ctrl-freak.svg)](https://pypi.org/project/ctrl-freak/)
[![License](https://img.shields.io/pypi/l/ctrl-freak.svg)](LICENSE)

An extensible genetic algorithm framework for single and multi-objective optimization, built on pure numpy.

## Installation

```bash
uv add ctrl-freak
```

or:

```bash
pip install ctrl-freak
```

## Quick Start

```python
import numpy as np
from ctrl_freak import nsga2, ga

# === Multi-Objective: NSGA-II ===
def init(rng):
    return rng.uniform(0, 1, size=5)

def evaluate_multi(x):
    f1 = x[0]
    f2 = 1 - np.sqrt(x[0]) + x[1:]@x[1:]
    return np.array([f1, f2])

def crossover(p1, p2):
    return (p1 + p2) / 2

def mutate(x):
    return np.clip(x + np.random.normal(0, 0.1, size=x.shape), 0, 1)

result = nsga2(
    init=init,
    evaluate=evaluate_multi,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=50,
    seed=42,
)

# Extract Pareto front
pareto_front = result.pareto_front
print(f"Found {len(pareto_front)} Pareto-optimal solutions")

# === Single-Objective: Standard GA ===
def evaluate_single(x):
    return float(np.sum(x ** 2))  # Sphere function

result = ga(
    init=init,
    evaluate=evaluate_single,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=100,
    seed=42,
)

print(f"Best fitness: {result.best[1]:.6f}")
```

## Features

- Single-objective optimization with `ga()`.
- Multi-objective optimization with `nsga2()`.
- Pluggable parent-selection and survival strategies.
- Pure numpy primitives for Pareto dominance, non-dominated sorting, and crowding distance.
- Parallel evaluation through `n_workers`.

See the [full documentation](https://hydrosolutions.github.io/ctrl-freak/) for API details, user contracts, examples, and extension points.

## Benchmarks

ctrl-freak ships a validation benchmark suite that checks `ga()` and `nsga2()`
against pymoo and DEAP on standard problems with known optima. With the genetic
algorithm held identical across all three libraries (ported SBX, aligned mutation
and selection, identical evaluation budget), ctrl-freak's results are statistically
indistinguishable from both baselines on the single-objective error metrics (all
six functions) and on multi-objective convergence (ZDT1, ZDT2, ZDT3, and DTLZ2). On
the two hardest problems (ZDT4, ZDT6) none of the three libraries converges at this
budget, and ctrl-freak is at least as good as both. The goal is parity, not
superiority.

See the canonical report in [benchmarks/README.md](benchmarks/README.md) and the
citable [Validation page](https://hydrosolutions.github.io/ctrl-freak/validation/).

## Links

- [Documentation](https://hydrosolutions.github.io/ctrl-freak/)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
