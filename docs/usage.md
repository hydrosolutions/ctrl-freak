# ctrl-freak: API Usage Guide

This guide covers the practical usage of ctrl-freak for multi-objective optimization with NSGA-II.

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
from ctrl_freak import nsga2, Population

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
final_pop = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    seed=42,
)

# Extract Pareto front
pareto_mask = final_pop.rank == 0
pareto_x = final_pop.x[pareto_mask]           # Decision variables
pareto_obj = final_pop.objectives[pareto_mask] # Objective values

print(f"Pareto front size: {pareto_x.shape[0]}")
```

---

## Core Concepts

### Population Dataclass

The `Population` dataclass holds all individuals and their computed attributes:

```python
@dataclass(frozen=True)
class Population:
    x: np.ndarray                         # (n, n_vars) decision variables
    objectives: np.ndarray | None         # (n, n_obj) objective values
    rank: np.ndarray | None               # (n,) Pareto front rank (0 = optimal)
    crowding_distance: np.ndarray | None  # (n,) diversity measure
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

# Ranks (lower = better, 0 = Pareto optimal)
all_ranks = pop.rank  # Shape: (pop_size,)

# Crowding distances (higher = more isolated = preferred)
all_cd = pop.crowding_distance  # Shape: (pop_size,)
```

### Extracting the Pareto Front

The Pareto-optimal solutions have `rank == 0`:

```python
# Boolean mask for Pareto-optimal individuals
pareto_mask = pop.rank == 0

# Decision variables of Pareto front
pareto_x = pop.x[pareto_mask]

# Objectives of Pareto front
pareto_obj = pop.objectives[pareto_mask]

# Number of Pareto-optimal solutions
n_pareto = np.sum(pareto_mask)
```

### IndividualView

Access a single individual via indexing:

```python
individual = pop[0]  # Returns IndividualView

# IndividualView attributes (all 1D or scalar)
individual.x                  # (n_vars,) decision variables
individual.objectives         # (n_obj,) objectives
individual.rank               # int
individual.crowding_distance  # float
```

`IndividualView` is read-only and useful for inspection or logging.

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
    callback: Callable[[Population, int], bool] | None = None,
) -> Population
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
| `callback` | `(pop, gen) -> bool` | Optional function called each generation; return `True` to stop |

### Using Callbacks

Callbacks enable logging, early stopping, and custom termination conditions.

**Logging progress:**

```python
def log_progress(pop: Population, gen: int) -> bool:
    n_pareto = np.sum(pop.rank == 0)
    best_obj = pop.objectives[pop.rank == 0].min(axis=0)
    print(f"Gen {gen}: {n_pareto} Pareto solutions, best: {best_obj}")
    return False  # Continue optimization

final_pop = nsga2(..., callback=log_progress)
```

**Early stopping on convergence:**

```python
class ConvergenceChecker:
    def __init__(self, patience: int = 20, tol: float = 1e-6):
        self.patience = patience
        self.tol = tol
        self.best_hypervolume = -np.inf
        self.generations_without_improvement = 0

    def __call__(self, pop: Population, gen: int) -> bool:
        hv = compute_hypervolume(pop.objectives[pop.rank == 0])
        if hv > self.best_hypervolume + self.tol:
            self.best_hypervolume = hv
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        return self.generations_without_improvement >= self.patience

checker = ConvergenceChecker(patience=50)
final_pop = nsga2(..., callback=checker)
```

**Early stopping on target:**

```python
def stop_on_target(pop: Population, gen: int) -> bool:
    # Stop when any solution achieves both objectives < 0.1
    pareto_obj = pop.objectives[pop.rank == 0]
    return np.any(np.all(pareto_obj < 0.1, axis=1))

final_pop = nsga2(..., callback=stop_on_target)
```

### Reproducibility

Use the `seed` parameter for reproducible results:

```python
# These two runs produce identical results
pop1 = nsga2(..., seed=42)
pop2 = nsga2(..., seed=42)

assert np.allclose(pop1.x, pop2.x)
assert np.allclose(pop1.objectives, pop2.objectives)
```

The seed controls:
- Initial population generation (via `init`)
- Parent selection (tournament selection)
- Any randomness in crossover/mutate if they use the provided RNG

---

## Working with Results

### Accessing Decision Variables and Objectives

```python
final_pop = nsga2(...)

# All solutions
all_x = final_pop.x              # (pop_size, n_vars)
all_obj = final_pop.objectives   # (pop_size, n_obj)

# Pareto-optimal solutions only
pareto_mask = final_pop.rank == 0
pareto_x = final_pop.x[pareto_mask]
pareto_obj = final_pop.objectives[pareto_mask]
```

### Iterating Over Solutions

```python
# Iterate using IndividualView
for i in range(len(final_pop.x)):
    ind = final_pop[i]
    print(f"Solution {i}: rank={ind.rank}, obj={ind.objectives}")

# Iterate over Pareto front only
pareto_idx = np.where(final_pop.rank == 0)[0]
for i in pareto_idx:
    ind = final_pop[i]
    print(f"Pareto solution: x={ind.x}, obj={ind.objectives}")
```

### Filtering by Rank

```python
# Get solutions by front
front_0 = final_pop.x[final_pop.rank == 0]  # Pareto optimal
front_1 = final_pop.x[final_pop.rank == 1]  # Second front
front_2 = final_pop.x[final_pop.rank == 2]  # Third front

# Get all non-dominated solutions (just front 0)
non_dominated = final_pop.x[final_pop.rank == 0]

# Get solutions within top 3 fronts
top_3_mask = final_pop.rank <= 2
top_3_x = final_pop.x[top_3_mask]
top_3_obj = final_pop.objectives[top_3_mask]
```

### Finding Extreme Solutions

```python
pareto_obj = final_pop.objectives[final_pop.rank == 0]
pareto_x = final_pop.x[final_pop.rank == 0]

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

final_pop = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=100,
    seed=42,
)

# Filter for feasible Pareto-optimal solutions
pareto_mask = final_pop.rank == 0
feasible_mask = final_pop.objectives[:, 2] == 0  # No constraint violation
valid_mask = pareto_mask & feasible_mask

feasible_pareto_x = final_pop.x[valid_mask]
feasible_pareto_obj = final_pop.objectives[valid_mask, :2]  # Original objectives only
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
final_pop = nsga2(
    init=init,
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=200,
    seed=42,
)

# Extract Pareto front
pareto_mask = final_pop.rank == 0
pareto_obj = final_pop.objectives[pareto_mask]
print(f"Found {pareto_obj.shape[0]} Pareto-optimal solutions")
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
