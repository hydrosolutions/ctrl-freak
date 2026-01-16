# ctrl-freak: API Usage Guide

This guide covers the practical usage of ctrl-freak for genetic algorithms, including both multi-objective optimization (NSGA-II) and single-objective optimization (standard GA).

---

## Installation

```bash
uv add ctrl-freak
```

---

## Quick Start

A complete minimal example optimizing the ZDT1 benchmark (2 objectives, 30 decision variables):

```python
import numpy as np
from ctrl_freak import nsga2

# Problem definition
N_VARS = 30
BOUNDS = (0.0, 1.0)

def init(rng: np.random.Generator) -> np.ndarray:
    """Initialize one individual with random values in [0, 1]."""
    return rng.uniform(BOUNDS[0], BOUNDS[1], size=N_VARS)

def evaluate(x: np.ndarray) -> np.ndarray:
    """ZDT1: two objectives to minimize."""
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (N_VARS - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Uniform crossover: randomly select genes from each parent."""
    mask = np.random.random(N_VARS) < 0.5
    return np.where(mask, p1, p2)

def mutate(x: np.ndarray) -> np.ndarray:
    """Gaussian mutation with bounds enforcement."""
    mutated = x + np.random.normal(0, 0.1, size=N_VARS)
    return np.clip(mutated, BOUNDS[0], BOUNDS[1])

# Run optimization
result = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    seed=42,
)

# Extract Pareto front using result's pareto_front property
pareto_front = result.pareto_front
pareto_x = pareto_front.x           # Decision variables
pareto_obj = pareto_front.objectives # Objective values

print(f"Pareto front size: {pareto_x.shape[0]}")
```

---

## Core Concepts

### Population Dataclass

The `Population` dataclass holds all individuals and their core data:

```python
@dataclass(frozen=True)
class Population:
    x: np.ndarray                         # (n, n_vars) decision variables
    objectives: np.ndarray | None         # (n, n_obj) objective values
```

Key properties:

- **Immutable**: All operations return new `Population` instances
- **Validation on construction**: Shape mismatches raise errors immediately
- **Arrays are always 2D for `x` and `objectives`**: Even single individuals have shape `(1, n_vars)`

Accessing data:

```python
# Decision variables for all individuals
all_x = pop.x  # Shape: (pop_size, n_vars)

# Objectives for all individuals
all_obj = pop.objectives  # Shape: (pop_size, n_obj)
```

### Result Types

Algorithm-specific metadata (such as Pareto ranks and crowding distances) are returned in result objects, not stored on `Population`:

- **`NSGA2Result`**: Returned by `nsga2()`, includes `population`, `rank`, `crowding_distance`, and `pareto_front` property
- **`GAResult`**: Returned by `ga()`, includes `population`, `fitness`, and `best` property

### Extracting the Pareto Front

Use the `pareto_front` property from `NSGA2Result`:

```python
result = nsga2(...)

# Access Pareto front directly
pareto_front = result.pareto_front  # Population of rank-0 individuals

# Decision variables of Pareto front
pareto_x = pareto_front.x

# Objectives of Pareto front
pareto_obj = pareto_front.objectives

# Number of Pareto-optimal solutions
n_pareto = len(pareto_front.x)
```

You can also access rank data directly:

```python
# Boolean mask for Pareto-optimal individuals
pareto_mask = result.rank == 0

# Equivalent to result.pareto_front.x
pareto_x = result.population.x[pareto_mask]
```

### IndividualView

Access a single individual via indexing:

```python
individual = pop[0]  # Returns IndividualView

# IndividualView attributes (all 1D or scalar)
individual.x                  # (n_vars,) decision variables
individual.objectives         # (n_obj,) objectives
```

`IndividualView` is read-only and useful for inspection or logging. Note that algorithm-specific metadata (rank, crowding distance, fitness) is not on `IndividualView` - access it from the result object instead.

---

## Single-Objective Optimization with ga()

For single-objective optimization, use the `ga()` function which implements a standard genetic algorithm with customizable selection and survival strategies.

### Quick Start Example

```python
import numpy as np
from ctrl_freak import ga

def init(rng):
    return rng.uniform(-5.12, 5.12, size=10)

def evaluate(x):
    return float(np.sum(x**2))  # Sphere function, returns float

def crossover(p1, p2):
    alpha = np.random.random()
    return alpha * p1 + (1 - alpha) * p2

def mutate(x):
    return np.clip(x + np.random.normal(0, 0.5, size=len(x)), -5.12, 5.12)

result = ga(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    seed=42,
)

print(f"Best fitness: {result.best[1]}")
print(f"Best solution: {result.best[0]}")
```

### Function Signature

```python
def ga(
    init: Callable[[np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], float],
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    pop_size: int,
    n_generations: int,
    seed: int | None = None,
    callback: Callable[[GAResult, int], bool] | None = None,
    select: str | ParentSelector = 'tournament',
    survive: str | SurvivorSelector = 'elitist',
) -> GAResult
```

### GAResult Structure

The `GAResult` dataclass contains the final population and algorithm-specific metadata:

```python
@dataclass(frozen=True)
class GAResult:
    population: Population        # Final population
    fitness: np.ndarray          # (n,) fitness values (lower is better)

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Returns (x, fitness) of the best individual."""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `init` | `(rng) -> (n_vars,)` | Initialize one individual using the provided RNG |
| `evaluate` | `(n_vars,) -> float` | Compute fitness for one individual (lower is better) |
| `crossover` | `(n_vars,), (n_vars,) -> (n_vars,)` | Combine two parents into one child |
| `mutate` | `(n_vars,) -> (n_vars,)` | Mutate one individual |
| `pop_size` | `int` | Number of individuals in the population |
| `n_generations` | `int` | Maximum number of generations |
| `seed` | `int \| None` | Random seed for reproducibility |
| `callback` | `(result, gen) -> bool` | Optional function called each generation; return `True` to stop |
| `select` | `str \| ParentSelector` | Parent selection strategy (default: 'tournament') |
| `survive` | `str \| SurvivorSelector` | Survival selection strategy (default: 'elitist') |

### Example: Tracking Convergence

```python
best_history = []

def track_convergence(result: GAResult, gen: int) -> bool:
    best_fitness = result.best[1]
    best_history.append(best_fitness)
    print(f"Gen {gen}: best = {best_fitness:.6f}")
    return False  # Continue

result = ga(..., callback=track_convergence)

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(best_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()
```

---

## Running NSGA-II

### Function Signature

```python
def nsga2(
    init: Callable[[np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], np.ndarray],
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    pop_size: int,
    n_generations: int,
    seed: int | None = None,
    callback: Callable[[NSGA2Result, int], bool] | None = None,
    select: str | ParentSelector = 'crowded',
    survive: str | SurvivorSelector = 'nsga2',
) -> NSGA2Result
```

### NSGA2Result Structure

The `NSGA2Result` dataclass contains the final population and NSGA-II-specific metadata:

```python
@dataclass(frozen=True)
class NSGA2Result:
    population: Population           # Final population
    rank: np.ndarray                # (n,) Pareto front rank (0 = optimal)
    crowding_distance: np.ndarray   # (n,) diversity measure

    @property
    def pareto_front(self) -> Population:
        """Returns Population of rank-0 (Pareto-optimal) individuals."""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `init` | `(rng) -> (n_vars,)` | Initialize one individual using the provided RNG |
| `evaluate` | `(n_vars,) -> (n_obj,)` | Compute objectives for one individual |
| `crossover` | `(n_vars,), (n_vars,) -> (n_vars,)` | Combine two parents into one child |
| `mutate` | `(n_vars,) -> (n_vars,)` | Mutate one individual |
| `pop_size` | `int` | Number of individuals in the population |
| `n_generations` | `int` | Maximum number of generations |
| `seed` | `int \| None` | Random seed for reproducibility |
| `callback` | `(result, gen) -> bool` | Optional function called each generation; return `True` to stop |
| `select` | `str \| ParentSelector` | Parent selection strategy (default: 'crowded') |
| `survive` | `str \| SurvivorSelector` | Survival selection strategy (default: 'nsga2') |

### Using Callbacks

Callbacks enable logging, early stopping, and custom termination conditions.

**Logging progress:**

```python
def log_progress(result: NSGA2Result, gen: int) -> bool:
    n_pareto = len(result.pareto_front.x)
    best_obj = result.pareto_front.objectives.min(axis=0)
    print(f"Gen {gen}: {n_pareto} Pareto solutions, best: {best_obj}")
    return False  # Continue optimization

result = nsga2(..., callback=log_progress)
```

**Early stopping on convergence:**

```python
class ConvergenceChecker:
    def __init__(self, patience: int = 20, tol: float = 1e-6):
        self.patience = patience
        self.tol = tol
        self.best_hypervolume = -np.inf
        self.generations_without_improvement = 0

    def __call__(self, result: NSGA2Result, gen: int) -> bool:
        hv = compute_hypervolume(result.pareto_front.objectives)
        if hv > self.best_hypervolume + self.tol:
            self.best_hypervolume = hv
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        return self.generations_without_improvement >= self.patience

checker = ConvergenceChecker(patience=50)
result = nsga2(..., callback=checker)
```

**Early stopping on target:**

```python
def stop_on_target(result: NSGA2Result, gen: int) -> bool:
    # Stop when any solution achieves both objectives < 0.1
    pareto_obj = result.pareto_front.objectives
    return np.any(np.all(pareto_obj < 0.1, axis=1))

result = nsga2(..., callback=stop_on_target)
```

### Reproducibility

Use the `seed` parameter for reproducible results:

```python
# These two runs produce identical results
result1 = nsga2(..., seed=42)
result2 = nsga2(..., seed=42)

assert np.allclose(result1.population.x, result2.population.x)
assert np.allclose(result1.population.objectives, result2.population.objectives)
```

The seed controls:
- Initial population generation (via `init`)
- Parent selection
- Any randomness in crossover/mutate if they use the provided RNG

---

## Working with Results

### NSGA-II Results

**Accessing Decision Variables and Objectives:**

```python
result = nsga2(...)

# All solutions
all_x = result.population.x              # (pop_size, n_vars)
all_obj = result.population.objectives   # (pop_size, n_obj)

# Pareto-optimal solutions using pareto_front property
pareto_front = result.pareto_front
pareto_x = pareto_front.x
pareto_obj = pareto_front.objectives

# Or using rank directly
pareto_mask = result.rank == 0
pareto_x = result.population.x[pareto_mask]
pareto_obj = result.population.objectives[pareto_mask]
```

**Iterating Over Solutions:**

```python
# Iterate using IndividualView
for i in range(len(result.population.x)):
    ind = result.population[i]
    rank = result.rank[i]
    cd = result.crowding_distance[i]
    print(f"Solution {i}: rank={rank}, cd={cd}, obj={ind.objectives}")

# Iterate over Pareto front only
for i, ind in enumerate(result.pareto_front):
    print(f"Pareto solution {i}: x={ind.x}, obj={ind.objectives}")
```

**Filtering by Rank:**

```python
# Get solutions by front
front_0 = result.population.x[result.rank == 0]  # Pareto optimal
front_1 = result.population.x[result.rank == 1]  # Second front
front_2 = result.population.x[result.rank == 2]  # Third front

# Get all non-dominated solutions (just front 0)
non_dominated = result.pareto_front.x

# Get solutions within top 3 fronts
top_3_mask = result.rank <= 2
top_3_x = result.population.x[top_3_mask]
top_3_obj = result.population.objectives[top_3_mask]
```

**Finding Extreme Solutions:**

```python
pareto_obj = result.pareto_front.objectives
pareto_x = result.pareto_front.x

# Best for objective 0
idx_best_f0 = np.argmin(pareto_obj[:, 0])
best_f0_solution = pareto_x[idx_best_f0]

# Best for objective 1
idx_best_f1 = np.argmin(pareto_obj[:, 1])
best_f1_solution = pareto_x[idx_best_f1]

# Compromise solution (closest to ideal point)
ideal = pareto_obj.min(axis=0)
distances = np.linalg.norm(pareto_obj - ideal, axis=1)
idx_compromise = np.argmin(distances)
compromise_solution = pareto_x[idx_compromise]
```

### GA Results

**Accessing Best Solution:**

```python
result = ga(...)

# Best individual (x, fitness)
best_x, best_fitness = result.best
print(f"Best solution: {best_x}")
print(f"Best fitness: {best_fitness}")

# Access full population
all_x = result.population.x
all_fitness = result.fitness

# Find top-k solutions
top_k = 10
top_indices = np.argsort(result.fitness)[:top_k]
top_x = result.population.x[top_indices]
top_fitness = result.fitness[top_indices]
```

**Iterating Over Solutions:**

```python
# Iterate by fitness order
sorted_indices = np.argsort(result.fitness)
for i in sorted_indices[:10]:  # Top 10
    ind = result.population[i]
    fitness = result.fitness[i]
    print(f"Solution {i}: fitness={fitness:.6f}, x={ind.x}")
```

---

## Complete Example: Constrained Optimization

Handling constraints via penalty objectives:

```python
import numpy as np
from ctrl_freak import nsga2

# Constrained problem: minimize f1, f2 subject to g(x) >= 0
N_VARS = 2
BOUNDS = (-5.0, 5.0)

def init(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(BOUNDS[0], BOUNDS[1], size=N_VARS)

def evaluate(x: np.ndarray) -> np.ndarray:
    # Original objectives
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 1) ** 2 + (x[1] - 1) ** 2

    # Constraint: x[0] + x[1] >= 1
    g = x[0] + x[1] - 1
    violation = max(0, -g)  # Positive when violated

    # Add penalty as third objective
    return np.array([f1, f2, violation])

def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    alpha = np.random.random()
    return alpha * p1 + (1 - alpha) * p2

def mutate(x: np.ndarray) -> np.ndarray:
    mutated = x + np.random.normal(0, 0.5, size=N_VARS)
    return np.clip(mutated, BOUNDS[0], BOUNDS[1])

result = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=100,
    seed=42,
)

# Filter for feasible Pareto-optimal solutions
pareto_mask = result.rank == 0
feasible_mask = result.population.objectives[:, 2] == 0  # No constraint violation
valid_mask = pareto_mask & feasible_mask

feasible_pareto_x = result.population.x[valid_mask]
feasible_pareto_obj = result.population.objectives[valid_mask, :2]  # Original objectives only
```

---

## Standard Genetic Operators

ctrl-freak provides well-established genetic operators commonly used in evolutionary multi-objective optimization. These are available as factory functions that return operators compatible with `nsga2()`.

### SBX Crossover (Simulated Binary Crossover)

SBX simulates single-point crossover behavior for real-valued variables. It produces children whose distribution around the parent values is controlled by the distribution index `eta`.

```python
from ctrl_freak import sbx_crossover

# Create crossover operator with default settings
crossover = sbx_crossover(eta=15.0, bounds=(0.0, 1.0), seed=42)

# Use with two parents
p1 = np.array([0.2, 0.4, 0.6])
p2 = np.array([0.3, 0.5, 0.7])
child = crossover(p1, p2)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta` | `float` | 15.0 | Distribution index. Higher values produce children closer to parents; lower values allow more exploration. Typical range: 2-20. |
| `bounds` | `tuple[float, float]` | (0.0, 1.0) | Lower and upper bounds for decision variables. |
| `seed` | `int \| None` | None | Random seed for reproducibility. |

### Polynomial Mutation

Polynomial mutation applies a bounded perturbation to each variable with a controllable probability and spread.

```python
from ctrl_freak import polynomial_mutation

# Create mutation operator with default settings
mutate = polynomial_mutation(eta=20.0, prob=None, bounds=(0.0, 1.0), seed=42)

# Use with an individual
x = np.array([0.3, 0.5, 0.7])
mutated = mutate(x)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta` | `float` | 20.0 | Distribution index. Higher values produce smaller perturbations (more local search); lower values allow larger jumps. Typical range: 20-100. |
| `prob` | `float \| None` | None | Mutation probability per variable. If None, uses 1/n_vars. |
| `bounds` | `tuple[float, float]` | (0.0, 1.0) | Lower and upper bounds for decision variables. |
| `seed` | `int \| None` | None | Random seed for reproducibility. |

### Complete Example with Standard Operators

```python
import numpy as np
from ctrl_freak import nsga2, sbx_crossover, polynomial_mutation

# Problem configuration
N_VARS = 30
BOUNDS = (0.0, 1.0)

def init(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(BOUNDS[0], BOUNDS[1], size=N_VARS)

def evaluate(x: np.ndarray) -> np.ndarray:
    # ZDT1 benchmark
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (N_VARS - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

# Create standard operators
crossover = sbx_crossover(eta=15.0, bounds=BOUNDS, seed=100)
mutate = polynomial_mutation(eta=20.0, bounds=BOUNDS, seed=200)

# Run optimization
result = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    seed=42,
)

# Extract Pareto front
pareto_front = result.pareto_front
print(f"Found {len(pareto_front.x)} Pareto-optimal solutions")
```

### Choosing eta Values

The distribution index `eta` controls the exploration/exploitation trade-off:

**For SBX crossover:**
- Lower `eta` (2-5): More exploration, children can be far from parents
- Higher `eta` (15-20): More exploitation, children stay close to parents
- Very high `eta` (50+): Very local search, children very similar to parents

**For polynomial mutation:**
- Lower `eta` (5-20): Larger perturbations, more exploration
- Higher `eta` (20-100): Smaller perturbations, fine-tuning
- Very high `eta` (100+): Very small changes, local refinement

**Typical configurations:**
- Early exploration: `sbx_crossover(eta=5)`, `polynomial_mutation(eta=20)`
- Balanced search: `sbx_crossover(eta=15)`, `polynomial_mutation(eta=20)`
- Local refinement: `sbx_crossover(eta=30)`, `polynomial_mutation(eta=100)`

---


## Customizing Selection Strategies

ctrl-freak provides pluggable parent selection strategies. Use string names for built-in strategies or pass custom callables.

### Built-in Selection Strategies

| Name | Function | Use Case |
|------|----------|----------|
| `'crowded'` | `crowded_tournament()` | NSGA-II (uses rank + crowding distance) |
| `'tournament'` | `fitness_tournament()` | Standard GA (fitness-based) |
| `'roulette'` | `roulette_wheel()` | Standard GA (fitness-proportionate) |

### Using String Names

```python
# NSGA-II with default crowded tournament
result = nsga2(..., select='crowded')

# GA with roulette wheel selection
result = ga(..., select='roulette')

# GA with tournament selection (default)
result = ga(..., select='tournament')
```

### Using Factory Functions

```python
from ctrl_freak import fitness_tournament, roulette_wheel

# Tournament with larger size
result = ga(..., select=fitness_tournament(tournament_size=5))

# Roulette wheel with specific seed
result = ga(..., select=roulette_wheel(seed=42))
```

### Custom Selection Strategy

Implement the ParentSelector protocol:

```python
def my_selector(pop, n_parents, rng, **kwargs):
    """
    Custom parent selection strategy.

    Args:
        pop: Population to select from
        n_parents: Number of parents to select
        rng: Random number generator
        **kwargs: Algorithm-specific metadata (e.g., fitness, rank, crowding_distance)

    Returns:
        np.ndarray: Indices of selected parents
    """
    # Your selection logic here
    # Example: random selection
    indices = rng.choice(len(pop.x), size=n_parents, replace=True)
    return indices

result = ga(..., select=my_selector)
```

---

## Customizing Survival Strategies

Survival strategies determine which individuals survive to the next generation.

### Built-in Survival Strategies

| Name | Function | Use Case |
|------|----------|----------|
| `'nsga2'` | `nsga2_survival()` | NSGA-II (Pareto ranking + crowding) |
| `'truncation'` | `truncation_survival()` | Keep best k by fitness |
| `'elitist'` | `elitist_survival()` | Preserve elite parents + best offspring |

### Using String Names

```python
# GA with truncation (no elitism)
result = ga(..., survive='truncation')

# GA with elitist survival (default, preserves 1 elite)
result = ga(..., survive='elitist')

# NSGA-II with default NSGA-II survival
result = nsga2(..., survive='nsga2')
```

### Using Factory Functions

```python
from ctrl_freak import elitist_survival

# Elitist with 5 elites instead of 1
result = ga(..., survive=elitist_survival(elite_count=5))
```

### Custom Survival Strategy

Implement the SurvivorSelector protocol:

```python
def my_survival(pop, n_survivors, **kwargs):
    """
    Custom survival strategy.

    Args:
        pop: Combined parent + offspring population
        n_survivors: Number of individuals to keep
        **kwargs: Algorithm-specific data (e.g., fitness for GA)

    Returns:
        tuple: (survivor_indices, state_dict)
            - survivor_indices: np.ndarray of indices to keep
            - state_dict: dict with algorithm-specific metadata for survivors
    """
    # Your survival logic here
    # Example: keep first n_survivors (not useful in practice)
    indices = np.arange(n_survivors)

    # Return indices and updated metadata
    fitness = kwargs.get('fitness')
    return indices, {'fitness': fitness[indices]}

result = ga(..., survive=my_survival)
```
## Migration from Previous API

### Breaking Changes in v2.0

The v2.0 release introduced a new extensible framework with several breaking changes to support single-objective GA and customizable selection strategies.

#### 1. Population no longer has rank/crowding_distance fields

**Old API:**
```python
pop = nsga2(...)
pareto_mask = pop.rank == 0
pareto_cd = pop.crowding_distance[pareto_mask]
```

**New API:**
```python
result = nsga2(...)
pareto_mask = result.rank == 0
pareto_cd = result.crowding_distance[pareto_mask]
```

#### 2. nsga2() returns NSGA2Result, not Population

**Old API:**
```python
pop = nsga2(...)
all_x = pop.x
```

**New API:**
```python
result = nsga2(...)
all_x = result.population.x
```

#### 3. Callback signature changed

**Old API:**
```python
def callback(pop: Population, gen: int) -> bool:
    n_pareto = np.sum(pop.rank == 0)
    return False

final_pop = nsga2(..., callback=callback)
```

**New API:**
```python
def callback(result: NSGA2Result, gen: int) -> bool:
    n_pareto = len(result.pareto_front.x)
    return False

result = nsga2(..., callback=callback)
```

#### 4. Pareto front extraction

**Old API:**
```python
pop = nsga2(...)
pareto_mask = pop.rank == 0
pareto_x = pop.x[pareto_mask]
pareto_obj = pop.objectives[pareto_mask]
```

**New API:**
```python
result = nsga2(...)
pareto_front = result.pareto_front
pareto_x = pareto_front.x
pareto_obj = pareto_front.objectives
```

#### 5. IndividualView no longer has rank/crowding_distance

**Old API:**
```python
ind = pop[0]
rank = ind.rank
cd = ind.crowding_distance
```

**New API:**
```python
ind = result.population[0]
rank = result.rank[0]
cd = result.crowding_distance[0]
```

### Complete Migration Example

**Old code:**
```python
from ctrl_freak import nsga2, Population

def callback(pop: Population, gen: int) -> bool:
    pareto_mask = pop.rank == 0
    n_pareto = np.sum(pareto_mask)
    print(f"Gen {gen}: {n_pareto} solutions")
    return False

pop = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    callback=callback,
)

pareto_mask = pop.rank == 0
pareto_x = pop.x[pareto_mask]
pareto_obj = pop.objectives[pareto_mask]
best_f0 = pareto_obj[:, 0].min()
```

**New code:**
```python
from ctrl_freak import nsga2, NSGA2Result

def callback(result: NSGA2Result, gen: int) -> bool:
    n_pareto = len(result.pareto_front.x)
    print(f"Gen {gen}: {n_pareto} solutions")
    return False

result = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    callback=callback,
)

pareto_front = result.pareto_front
pareto_x = pareto_front.x
pareto_obj = pareto_front.objectives
best_f0 = pareto_obj[:, 0].min()
```

### New Features in v2.0

1. **Single-objective GA**: Use `ga()` for standard genetic algorithms
2. **Customizable selection**: Pass `select` parameter to choose parent selection strategy
3. **Customizable survival**: Pass `survive` parameter to choose survival selection strategy
4. **Result types**: Clear separation between population data and algorithm metadata

---
