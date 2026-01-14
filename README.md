# ctrl-freak

A pure numpy implementation of the NSGA-II multi-objective genetic algorithm.

## Maintenance Status

:large_green_circle: **Active Development**

This repository is part of an ongoing project and actively maintained.

---

## Installation

```bash
uv add ctrl-freak
```

## Quick Start

```python
import numpy as np
from ctrl_freak import nsga2

# Define your problem
def init(rng):
    return rng.uniform(0, 1, size=5)

def evaluate(x):
    f1 = x[0]
    f2 = 1 - np.sqrt(x[0]) + x[1:]@x[1:]
    return np.array([f1, f2])

def crossover(p1, p2):
    return (p1 + p2) / 2

def mutate(x):
    return np.clip(x + np.random.normal(0, 0.1, size=x.shape), 0, 1)

# Run optimization
result = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=50,
    seed=42,
)

# Extract Pareto front
pareto_x = result.x[result.rank == 0]
pareto_obj = result.objectives[result.rank == 0]
```

## Documentation

- [API Usage Guide](docs/usage.md) — Installation, examples, working with results
- [User Contracts](docs/contracts.md) — Function signatures and responsibilities

---

## Design Philosophy

- **Pure numpy** for performance
- **Functional style** with immutable data structures
- **User thinks about individuals**, framework handles vectorization via `lift()`
- **Fail fast** with eager validation
- **Domain agnostic** — framework handles selection pressure, user handles constraints/bounds

## Architecture

```
┌─────────────────────────────────────────────┐
│  User Domain Layer                          │
│  evaluate(), crossover(), mutate()          │
│  (per-individual, user-defined)             │
└─────────────────────────────────────────────┘
                    │ lift()
                    ▼
┌─────────────────────────────────────────────┐
│  NSGA-II Core                               │
│  - main loop orchestration                  │
│  - binary tournament selection              │
│  - survivor selection (fronts + crowding)   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Primitives (pure functions)                │
│  - non_dominated_sort(objectives) → ranks   │
│  - crowding_distance(front) → distances     │
│  - dominates(a, b) → bool                   │
└─────────────────────────────────────────────┘
```

## API Reference

### Main Entry Point

```python
nsga2(init, evaluate, crossover, mutate, pop_size, n_generations, seed=None, callback=None) -> Population
```

### User Function Contracts

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(rng) -> (n_vars,)` | Initialize one random individual |
| `evaluate` | `(n_vars,) -> (n_obj,)` | Compute objectives (minimization) |
| `crossover` | `(n_vars,), (n_vars,) -> (n_vars,)` | Combine two parents into one child |
| `mutate` | `(n_vars,) -> (n_vars,)` | Perturb an individual |

### Data Structures

**Population** — Immutable collection of solutions:
- `x: np.ndarray` — Decision variables `(n, n_vars)`
- `objectives: np.ndarray` — Objective values `(n, n_obj)`
- `rank: np.ndarray` — Pareto front ranks `(n,)` where 0 = optimal
- `crowding_distance: np.ndarray` — Diversity measure `(n,)`

### Primitives

| Function | Description |
|----------|-------------|
| `dominates(a, b)` | Check if `a` Pareto-dominates `b` |
| `dominates_matrix(objectives)` | Pairwise dominance matrix |
| `non_dominated_sort(objectives)` | Assign Pareto front ranks |
| `crowding_distance(front)` | Compute crowding distances for one front |

---

## Future Work

- SBX crossover and polynomial mutation (standard operators for continuous optimization)
- Parallel evaluation via multiprocessing in `lift()`
