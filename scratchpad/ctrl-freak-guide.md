# ctrl-freak: A Natural Language Guide

## What is ctrl-freak?

**ctrl-freak** is an extensible genetic algorithm framework built on pure NumPy. It handles all the bookkeeping of genetic algorithms—managing populations, selecting parents, tracking generations, recording the best solution—while letting you focus on defining your specific problem.

The key philosophy is: **you think about one individual at a time; the library handles the rest**.

---

## The Four Functions You Provide

To use ctrl-freak, you define four functions that describe your optimization problem:

| Function | What it does | Signature |
|----------|--------------|-----------|
| `init` | Create one random starting solution | `(rng) → individual` |
| `evaluate` | Compute fitness (lower is better) | `(individual) → float` |
| `crossover` | Combine two parents into a child | `(parent1, parent2) → child` |
| `mutate` | Apply random variation to an individual | `(individual) → individual` |

The library then orchestrates these functions: it creates a population by calling `init` many times, evaluates everyone with `evaluate`, selects parents, creates offspring using `crossover` and `mutate`, and repeats for many generations.

---

## The Defaults (What Happens When You Don't Specify)

ctrl-freak has sensible defaults for **selection** and **survival** strategies:

### 1. Parent Selection: `"tournament"` (default)

- Binary tournament: randomly pick 2 individuals, the fitter one becomes a parent
- This is exactly the `tournament_select` function students learned earlier
- Provides moderate selection pressure without being too aggressive

### 2. Survivor Selection: `"elitist"` (default)

- Always preserves the single best individual from the parent generation
- Fills remaining slots with the best offspring
- Guarantees the best solution is never lost across generations

### No Defaults for Crossover and Mutation

**Important**: There are NO defaults for `crossover` and `mutate`—you must always provide these. This is by design: the library doesn't know what your decision variables represent, so it can't assume a sensible crossover or mutation strategy.

---

## Overriding the Defaults

Here's the key insight for students: **the functions they implemented earlier can be plugged directly into ctrl-freak**.

### The Functions Students Wrote in Class

```python
import numpy as np

def tournament_select(population, fitness, rng, tournament_size=2):
    """Select one parent using tournament selection."""
    candidates = rng.choice(len(population), size=tournament_size, replace=False)
    winner = candidates[np.argmin(fitness[candidates])]
    return population[winner]

def blend_crossover(parent1, parent2, rng, alpha=0.5):
    """Create a child by blending two parents."""
    weights = rng.uniform(-alpha, 1 + alpha, size=parent1.shape)
    return parent1 + weights * (parent2 - parent1)

def gaussian_mutate(individual, rng, scale=0.3):
    """Apply Gaussian mutation to an individual."""
    noise = rng.normal(0, scale, size=individual.shape)
    return individual + noise
```

### Using Custom Operators with ctrl-freak

For crossover and mutate, pass them directly:

```python
from ctrl_freak import ga

result = ga(
    init=init,
    evaluate=evaluate,
    crossover=lambda p1, p2: blend_crossover(p1, p2, rng, alpha=0.5),
    mutate=lambda x: gaussian_mutate(x, rng, scale=0.3),
    pop_size=50,
    n_generations=100,
    seed=42,
)
```

For selection strategies, you can either:

1. **Use a string name**: `select="tournament"` or `select="roulette"`
2. **Pass a configured callable** from the library's built-in factories

### Available Built-in Strategies

| Strategy Type | Name | Description |
|---------------|------|-------------|
| Selection | `"tournament"` | Binary tournament (default for GA) |
| Selection | `"roulette"` | Fitness-proportionate selection |
| Selection | `"crowded"` | Crowded tournament (for NSGA-II) |
| Survival | `"elitist"` | Preserve best individual (default for GA) |
| Survival | `"truncation"` | Keep top k by fitness |
| Survival | `"nsga2"` | Pareto ranking + crowding (for NSGA-II) |

---

## What the Library Handles

When you call `ga()`, ctrl-freak:

1. **Initializes** the population by calling your `init` function `pop_size` times
2. **Evaluates** everyone using your `evaluate` function
3. For each generation:
   - **Selects** parents using tournament selection (or your override)
   - **Creates offspring** by calling your `crossover` and `mutate`
   - **Evaluates** the offspring
   - **Selects survivors** using elitist selection (or your override)
   - **Calls your callback** (if provided) with the current state
4. Returns a `GAResult` with the final population and best solution

---

## The Callback Mechanism

To track progress during optimization, you can provide a callback:

```python
history = []

def capture(result, generation):
    """Callback to record population state at each generation."""
    history.append({
        'gen': generation,
        'x': result.population.x.copy(),
        'fitness': result.fitness.copy(),
    })
    return False  # Continue (return True to stop early)

result = ga(..., callback=capture)
```

The callback receives:

- `result`: A `GAResult` object with the current population
- `generation`: The generation number (0-indexed)

Return `True` to stop early, `False` to continue.

### Early Stopping Example

```python
def early_stop(result, gen):
    _, best_fitness = result.best
    if best_fitness < 1e-6:
        print(f"Converged at generation {gen}")
        return True  # Stop optimization
    return False  # Continue

result = ga(..., callback=early_stop)
```

---

## The GAResult Object

After optimization completes:

```python
result = ga(...)

# The best solution found
best_x, best_fitness = result.best

# The entire final population
all_individuals = result.population.x      # shape (pop_size, n_vars)
all_fitness = result.fitness               # shape (pop_size,)

# Metadata
print(f"Completed {result.generations} generations")
print(f"Performed {result.evaluations} total evaluations")
```

### GAResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `population` | `Population` | Final population after optimization |
| `fitness` | `np.ndarray` | Fitness values, shape `(pop_size,)` |
| `best_idx` | `int` | Index of best individual |
| `generations` | `int` | Number of generations completed |
| `evaluations` | `int` | Total fitness evaluations performed |
| `best` | property | Returns `(best_x, best_fitness)` tuple |

---

## Complete Example: Himmelblau with Custom Operators

This example shows students how to use the operators they implemented earlier with ctrl-freak:

```python
import numpy as np
from ctrl_freak import ga

# Problem definition
def init(rng: np.random.Generator) -> np.ndarray:
    """Create one random individual in [-5, 5]²."""
    return rng.uniform(-5, 5, size=2)

def evaluate(x: np.ndarray) -> float:
    """Compute the Himmelblau function."""
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)

# Use the operators we implemented earlier!
def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Blend crossover with α=0.5."""
    rng = np.random.default_rng()
    weights = rng.uniform(-0.5, 1.5, size=p1.shape)
    return p1 + weights * (p2 - p1)

def mutate(x: np.ndarray) -> np.ndarray:
    """Gaussian mutation with σ=0.3."""
    return x + np.random.normal(0, 0.3, size=x.shape)

# Track history for visualization
history = []

def capture(result, generation):
    """Record population state at each generation."""
    history.append({
        'gen': generation,
        'x': result.population.x.copy(),
        'fitness': result.fitness.copy(),
    })
    return False

# Run the GA - ctrl-freak handles the rest!
result = ga(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=50,
    n_generations=40,
    seed=42,
    callback=capture,
    # These are the defaults - we could override them if we wanted:
    # select="tournament",  # Uses binary tournament (like our tournament_select)
    # survive="elitist",    # Keeps the best individual each generation
)

best_x, best_fitness = result.best
print(f"Best solution: ({best_x[0]:.4f}, {best_x[1]:.4f})")
print(f"Best fitness: {best_fitness:.6f}")
```

---

## Key Teaching Points

1. **The operators students implemented by hand are exactly what ctrl-freak needs.** The library provides the orchestration; students provide the domain knowledge through their custom functions.

2. **Defaults exist for selection and survival, but not for crossover and mutation.** This reflects the reality that selection strategies are domain-agnostic, while genetic operators depend on how solutions are represented.

3. **You can always override the defaults.** Pass a string name for built-in strategies, or pass your own callable for complete control.

4. **The callback mechanism enables visualization and early stopping.** Students can track the evolution process and create animations like the one in the teaching material.

5. **ctrl-freak handles the boring parts.** Population management, random number generation, tracking the best solution—all handled by the library so students can focus on the interesting parts.

---

## API Quick Reference

### Main Function

```python
from ctrl_freak import ga

result = ga(
    init=init,                    # Required: (rng) → individual
    evaluate=evaluate,            # Required: (x) → float
    crossover=crossover,          # Required: (p1, p2) → child
    mutate=mutate,                # Required: (x) → x'
    pop_size=100,                 # Required: population size (must be even)
    n_generations=50,             # Required: number of generations
    seed=42,                      # Optional: for reproducibility
    callback=None,                # Optional: (result, gen) → bool
    select="tournament",          # Optional: parent selection strategy
    survive="elitist",            # Optional: survivor selection strategy
    n_workers=1,                  # Optional: parallel evaluation (-1 for all cores)
)
```

### Imports

```python
from ctrl_freak import (
    # Algorithms
    ga,                    # Single-objective GA
    nsga2,                 # Multi-objective NSGA-II

    # Selection strategies
    fitness_tournament,    # Binary tournament for GA
    roulette_wheel,        # Fitness-proportionate
    crowded_tournament,    # For NSGA-II

    # Survival strategies
    elitist_survival,      # Preserve elite
    truncation_survival,   # Keep top k
    nsga2_survival,        # Pareto + crowding

    # Built-in operators
    sbx_crossover,         # Simulated Binary Crossover
    polynomial_mutation,   # Polynomial mutation

    # Data structures
    Population,
    GAResult,
    NSGA2Result,
)
```
