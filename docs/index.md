# ctrl-freak

An extensible, pure-numpy genetic algorithm framework for **single-objective (GA)** and **multi-objective (NSGA-II)** optimization.

## Install

```bash
uv add ctrl-freak
```

## Quick start

```python
import numpy as np
from ctrl_freak import nsga2

def init(rng):
    return rng.uniform(0, 1, size=5)

def evaluate(x):
    f1 = x[0]
    f2 = 1 - np.sqrt(x[0]) + x[1:] @ x[1:]
    return np.array([f1, f2])

def crossover(p1, p2):
    return (p1 + p2) / 2

def mutate(x):
    return np.clip(x + 0.01, 0, 1)

result = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=50,
    seed=42,
)

pareto_front = result.pareto_front
print(f"Found {len(pareto_front)} Pareto-optimal solutions")
```

## Where to next

- [Usage Guide](usage.md) - installation, examples, working with results, standard operators, custom strategies
- [User Contracts](contracts.md) - exact contracts for `init` / `evaluate` / `crossover` / `mutate` and the selection/survival protocols
- [API Reference](api.md) - auto-generated reference for the full public API

## Reproducibility

A single `seed=` argument deterministically reproduces an entire run - initialization, parent selection, crossover, and mutation are all derived from that one seed via `numpy.random.SeedSequence.spawn`.
