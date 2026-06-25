# ctrl-freak

An extensible genetic algorithm framework for single and multi-objective optimization, built on pure numpy.

## Maintenance Status

🟢 **Active Development**

This repository is part of an ongoing project and actively maintained.

---

## Installation

```bash
uv add ctrl-freak
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
- **Extensible** via pluggable selection and survival strategies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Domain Layer                                          │
│  init(), evaluate(), crossover(), mutate()                  │
│  (per-individual, user-defined)                             │
└─────────────────────────────────────────────────────────────┘
                          │ lift()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Algorithm Layer                                            │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │   nsga2()   │    │    ga()     │                         │
│  │ multi-obj   │    │ single-obj  │                         │
│  └─────────────┘    └─────────────┘                         │
│         │                  │                                │
│         └────────┬─────────┘                                │
│                  ▼                                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Pluggable Strategies                                 │  │
│  │  Selection: crowded, tournament, roulette             │  │
│  │  Survival: nsga2, truncation, elitist                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Primitives (pure functions)                                │
│  non_dominated_sort(), crowding_distance(), dominates()     │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Algorithms

```python
# Multi-objective optimization
nsga2(init, evaluate, crossover, mutate, pop_size, n_generations,
      seed=None, callback=None, select='crowded', survive='nsga2',
      n_workers=1) -> NSGA2Result

# Single-objective optimization
ga(init, evaluate, crossover, mutate, pop_size, n_generations,
   seed=None, callback=None, select='tournament', survive='elitist',
   n_workers=1) -> GAResult
```

### User Function Contracts

| Function | Signature | Description |
|----------|-----------|-------------|
| `init` | `(rng) -> (n_vars,)` | Initialize one random individual |
| `evaluate` | `(n_vars,) -> (n_obj,)` or `float` | Compute objectives (minimization) |
| `crossover` | `(n_vars,), (n_vars,) -> (n_vars,)` | Combine two parents into one child |
| `mutate` | `(n_vars,) -> (n_vars,)` | Perturb an individual |

### Result Types

**NSGA2Result** — Multi-objective optimization result:

- `population: Population` — Final population
- `rank: np.ndarray` — Pareto front ranks `(n,)` where 0 = optimal
- `crowding_distance: np.ndarray` — Diversity measure `(n,)`
- `pareto_front: Population` — Property returning rank-0 individuals
- `generations: int` — Generations completed
- `evaluations: int` — Total evaluations

**GAResult** — Single-objective optimization result:

- `population: Population` — Final population
- `fitness: np.ndarray` — Fitness values `(n,)`
- `best: tuple[np.ndarray, float]` — Property returning (best_x, best_fitness)
- `generations: int` — Generations completed
- `evaluations: int` — Total evaluations

### Data Structures

**Population** — Immutable collection of solutions:

- `x: np.ndarray` — Decision variables `(n, n_vars)`
- `objectives: np.ndarray | None` — Objective values `(n, n_obj)`

### Selection Strategies

| Name | Function | Use Case |
|------|----------|----------|
| `'crowded'` | `crowded_tournament()` | NSGA-II (rank + crowding) |
| `'tournament'` | `fitness_tournament()` | GA (fitness-based) |
| `'roulette'` | `roulette_wheel()` | GA (fitness-proportionate) |

### Survival Strategies

| Name | Function | Use Case |
|------|----------|----------|
| `'nsga2'` | `nsga2_survival()` | NSGA-II (fronts + crowding) |
| `'truncation'` | `truncation_survival()` | Keep best k |
| `'elitist'` | `elitist_survival()` | Preserve elite parents |

### Primitives

| Function | Description |
|----------|-------------|
| `dominates(a, b)` | Check if `a` Pareto-dominates `b` |
| `dominates_matrix(objectives)` | Pairwise dominance matrix |
| `non_dominated_sort(objectives)` | Assign Pareto front ranks |
| `crowding_distance(front)` | Compute crowding distances for one front |

---

## Benchmarks

Tested against Pymoo and DEAP on ZDT test problems (100 pop, 250 generations, 10 seeds):

| Problem | ctrl-freak | Pymoo | DEAP |
|---------|------------|-------|------|
| ZDT1 | 0.8653 ± 0.0011 | 0.8241 ± 0.0255 | **0.8698 ± 0.0002** |
| ZDT2 | 0.5320 ± 0.0017 | 0.4764 ± 0.0182 | **0.5363 ± 0.0002** |
| ZDT3 | 1.3224 ± 0.0008 | 1.2836 ± 0.0123 | **1.3275 ± 0.0002** |

ctrl-freak matches DEAP-level hypervolume on ZDT1-3 with low variance. See [full benchmark results](benchmarks/README.md).

---
