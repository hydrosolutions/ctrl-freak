# RNG and Reproducibility in ctrl-freak

## Overview

ctrl-freak uses NumPy's modern `Generator` API (`np.random.default_rng`) for all random operations. A single seed controls the entire optimization run.

## Seeding Mechanism

```python
from ctrl_freak import ga, nsga2

# Reproducible run
result = ga(..., seed=42)

# Non-reproducible (uses system entropy)
result = ga(..., seed=None)
```

The RNG is created once at algorithm start and passed through all stochastic operations:

```python
rng = np.random.default_rng(seed)
```

## RNG Flow Architecture

```
Algorithm (seed)
    │
    ├── init(rng) ──────────────── Population initialization
    │
    └── Generation loop
            │
            ├── ParentSelector(rng) ─── Tournament, roulette selection
            │
            └── create_offspring(rng) ── Crossover, mutation
```

## User-Defined Functions

### `init()` - Required RNG Usage

The `init` function **must** use the provided RNG:

```python
# Correct - reproducible
def init(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, 1, size=10)

# Wrong - breaks reproducibility
def init(rng: np.random.Generator) -> np.ndarray:
    return np.random.random(10)  # Uses global state
```

### Custom ParentSelector

Custom selection strategies receive RNG via the protocol:

```python
from ctrl_freak.protocols import ParentSelector

def custom_selector(
    pop: Population,
    n_parents: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    # Use rng for all random operations
    return rng.choice(len(pop.x), size=n_parents)
```

### `crossover()` and `mutate()` - Optional RNG

These functions don't receive RNG from the framework. For reproducibility:

1. Use the standard operators with explicit seeds:

```python
from ctrl_freak.operators.standard import sbx_crossover, polynomial_mutation

crossover = sbx_crossover(eta=15.0, seed=123)
mutate = polynomial_mutation(eta=20.0, seed=456)
```

1. Or make them deterministic:

```python
def crossover(p1, p2):
    return (p1 + p2) / 2  # No randomness
```

## Standard Operators

Built-in operators (`sbx_crossover`, `polynomial_mutation`) create their own seeded RNG:

```python
def sbx_crossover(eta=15.0, bounds=(0, 1), seed=None):
    rng = np.random.default_rng(seed)  # Operator-local RNG

    def crossover(p1, p2):
        u = rng.random(len(p1))  # Uses closure RNG
        ...
    return crossover
```

This isolates operator randomness from algorithm RNG.

## Reproducibility Guarantees

With a fixed seed and deterministic user functions:

```python
result1 = ga(init=init, evaluate=f, crossover=cx, mutate=mut, seed=42)
result2 = ga(init=init, evaluate=f, crossover=cx, mutate=mut, seed=42)

assert np.array_equal(result1.population.x, result2.population.x)
assert np.array_equal(result1.fitness, result2.fitness)
```

Parallel evaluation (`n_workers > 1`) preserves reproducibility since evaluation is deterministic on inputs.

## Summary

| Component | RNG Source | User Responsibility |
|-----------|------------|---------------------|
| `init()` | Framework-provided | Must use provided `rng` |
| `ParentSelector` | Framework-provided | Must use provided `rng` |
| `SurvivorSelector` | None (deterministic) | N/A |
| `crossover()` | User-managed | Use seeded operators or be deterministic |
| `mutate()` | User-managed | Use seeded operators or be deterministic |
