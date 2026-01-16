# ctrl-freak: User Function Contracts

This document specifies the exact contracts for user-provided functions and strategy protocols in ctrl-freak.

---

## Philosophy

**User thinks about individuals. Framework handles populations.**

You write functions that operate on a single individual. The framework:

- Calls your functions in a loop over the population
- Handles parent selection via customizable selection strategies
- Manages algorithm-specific survival logic (Pareto ranking, fitness-based selection, etc.)
- Orchestrates the generational cycle

This separation keeps user code simple and domain-focused while the framework handles algorithmic complexity.

The framework supports both **single-objective** optimization (via `ga()`) and **multi-objective** optimization (via `nsga2()`), with extensible selection and survival strategies.

---

## Algorithm Selection

ctrl-freak provides two main algorithms:

| Algorithm | Function | Objectives | Default Selection | Default Survival |
|-----------|----------|------------|-------------------|------------------|
| NSGA-II | `nsga2()` | Multi-objective | Crowded tournament | NSGA-II (fronts + crowding) |
| Standard GA | `ga()` | Single-objective | Fitness tournament | Elitist |

Both algorithms accept the same user-defined functions (`init`, `evaluate`, `crossover`, `mutate`) but differ in how they handle selection and survival.

---

## Objective Direction

**All objectives are minimized.**

To maximize an objective, negate it:

```python
def evaluate(x: np.ndarray) -> np.ndarray:
    accuracy = compute_accuracy(x)      # Want to maximize
    latency = compute_latency(x)        # Want to minimize
    return np.array([-accuracy, latency])  # Negate accuracy
```

---

## Function Contracts

### `init(rng: np.random.Generator) -> np.ndarray`

Generates the decision variables for one individual.

| Aspect | Specification |
|--------|---------------|
| **Input** | `rng`: A seeded `numpy.random.Generator` instance |
| **Output** | `np.ndarray` with shape `(n_vars,)` and dtype `float64` |
| **User responsibilities** | Generate values within valid bounds; ensure returned array has correct shape |
| **Framework responsibilities** | Provides a seeded RNG for reproducibility; calls `init` once per individual |

**Example:**

```python
def init(rng: np.random.Generator) -> np.ndarray:
    # Continuous variables in [0, 1]
    continuous = rng.uniform(0, 1, size=5)

    # Integer variables in {0, 1, 2, 3, 4}
    integers = rng.integers(0, 5, size=3).astype(float)

    return np.concatenate([continuous, integers])
```

**Notes:**

- Always use the provided `rng` for randomness (not `np.random.random()`)
- The framework does not validate bounds; out-of-bounds values propagate silently
- All decision variables must be numeric (float64)

---

### `evaluate(x: np.ndarray) -> np.ndarray | float`

Computes objective values for one individual.

#### Multi-Objective Evaluation (for nsga2())

| Aspect | Specification |
|--------|---------------|
| **Input** | `x`: `np.ndarray` with shape `(n_vars,)` containing decision variables |
| **Output** | `np.ndarray` with shape `(n_obj,)` containing objective values to minimize |
| **User responsibilities** | Return finite values; handle constraints via penalties; negate objectives to maximize |
| **Framework responsibilities** | Automatic lifting to population via `lift()`; never modifies input |

**Example:**

```python
def evaluate(x: np.ndarray) -> np.ndarray:
    # Two objectives
    f1 = x[0] ** 2 + x[1] ** 2          # Sphere
    f2 = (x[0] - 1) ** 2 + x[1] ** 2    # Shifted sphere
    return np.array([f1, f2])
```

**Requirements:**

- Return values must be finite (`np.isfinite(obj).all()` must be `True`)
- All objectives must be returned in consistent order across calls
- Shape must be exactly `(n_obj,)`, not `(n_obj, 1)` or `(1, n_obj)`

**Invalid outputs:**

- `np.inf` or `-np.inf` (breaks crowding distance calculation)
- `np.nan` (breaks dominance comparisons)
- Wrong shape (causes stacking errors)

#### Single-Objective Evaluation (for ga())

When using the `ga()` function for single-objective optimization, `evaluate` returns a scalar:

| Aspect | Specification |
|--------|---------------|
| **Input** | `x`: `np.ndarray` with shape `(n_vars,)` |
| **Output** | `float` — single objective value to minimize |

**Example:**

```python
def evaluate(x: np.ndarray) -> float:
    """Sphere function: minimize sum of squares."""
    return float(np.sum(x ** 2))
```

---

### `crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray`

Combines two parent individuals to produce one child.

| Aspect | Specification |
|--------|---------------|
| **Input** | `p1`, `p2`: Each `np.ndarray` with shape `(n_vars,)` |
| **Output** | `np.ndarray` with shape `(n_vars,)` representing one child |
| **User responsibilities** | Preserve bounds if needed; preserve constraint invariants if needed |
| **Framework responsibilities** | Selects parents via binary tournament; provides two distinct parents |

**Example (SBX-like):**

```python
def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # Simulated binary crossover (simplified)
    eta = 20  # Distribution index
    u = np.random.random(len(p1))

    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1 / (eta + 1)),
        (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    )

    child = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    return np.clip(child, 0, 1)  # Enforce bounds
```

**Example (uniform):**

```python
def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mask = np.random.random(len(p1)) < 0.5
    return np.where(mask, p1, p2)
```

**Notes:**

- Always produces exactly one child (not two)
- Parents may be identical (handle gracefully)
- Bounds enforcement is user responsibility

---

### `mutate(x: np.ndarray) -> np.ndarray`

Mutates one individual.

| Aspect | Specification |
|--------|---------------|
| **Input** | `x`: `np.ndarray` with shape `(n_vars,)` |
| **Output** | `np.ndarray` with shape `(n_vars,)` representing mutated individual |
| **User responsibilities** | Enforce bounds (clip); repair constraint violations; control mutation rate |
| **Framework responsibilities** | None. Full control given to user. |

**Example (Gaussian):**

```python
def mutate(x: np.ndarray) -> np.ndarray:
    mutation_rate = 0.1
    sigma = 0.1

    # Mutate each gene with probability mutation_rate
    mask = np.random.random(len(x)) < mutation_rate
    noise = np.random.normal(0, sigma, len(x))
    mutated = x + mask * noise

    return np.clip(mutated, 0, 1)  # Enforce bounds
```

**Example (polynomial):**

```python
def mutate(x: np.ndarray) -> np.ndarray:
    eta = 20  # Distribution index
    lower, upper = 0.0, 1.0

    mutated = x.copy()
    for i in range(len(x)):
        if np.random.random() < 0.1:  # Mutation probability
            delta = min(x[i] - lower, upper - x[i]) / (upper - lower)
            u = np.random.random()

            if u < 0.5:
                deltaq = (2 * u + (1 - 2 * u) * (1 - delta) ** (eta + 1)) ** (1 / (eta + 1)) - 1
            else:
                deltaq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta + 1)) ** (1 / (eta + 1))

            mutated[i] = x[i] + deltaq * (upper - lower)

    return np.clip(mutated, lower, upper)
```

**Critical requirements:**

- Always return values within valid bounds
- The framework does not clip or validate output
- Out-of-bounds values propagate through the entire algorithm

---

## Strategy Protocols

The framework uses protocols to define extensible selection and survival strategies. These allow you to customize algorithm behavior while maintaining type safety.

### ParentSelector Protocol

Parent selectors choose which individuals become parents for reproduction.

```python
from typing import Protocol
import numpy as np
from ctrl_freak import Population

class ParentSelector(Protocol):
    def __call__(
        self,
        pop: Population,
        n_parents: int,
        rng: np.random.Generator,
        **kwargs: np.ndarray,
    ) -> np.ndarray:
        """
        Select parent indices from the population.

        Args:
            pop: Population to select from
            n_parents: Number of parents to select
            rng: Random number generator for reproducibility
            **kwargs: Algorithm-specific metadata (e.g., rank, crowding_distance, fitness)

        Returns:
            np.ndarray of shape (n_parents,) containing selected indices
        """
        ...
```

**Contract:**
- Must return exactly `n_parents` indices
- Indices must be valid (0 <= idx < len(pop))
- Must use provided `rng` for any randomness
- May select the same individual multiple times

**Built-in implementations:**
- `crowded_tournament(tournament_size=2)` — For NSGA-II, uses rank and crowding distance
- `fitness_tournament(tournament_size=2)` — For GA, uses fitness values
- `roulette_wheel()` — Fitness-proportionate selection

### SurvivorSelector Protocol

Survivor selectors determine which individuals survive to the next generation.

```python
class SurvivorSelector(Protocol):
    def __call__(
        self,
        pop: Population,
        n_survivors: int,
        **kwargs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Select survivors and compute algorithm-specific state.

        Args:
            pop: Combined population (parents + offspring)
            n_survivors: Number of individuals to keep
            **kwargs: Algorithm-specific inputs (e.g., parent_size, fitness)

        Returns:
            Tuple of:
            - np.ndarray of shape (n_survivors,) containing survivor indices
            - dict mapping state names to arrays (e.g., {'rank': ..., 'fitness': ...})
        """
        ...
```

**Contract:**
- Must return exactly `n_survivors` indices
- Indices must be valid (0 <= idx < len(pop))
- State dict keys and values are algorithm-specific
- For NSGA-II: returns `{'rank': ..., 'crowding_distance': ...}`
- For GA: returns `{'fitness': ...}`

**Built-in implementations:**
- `nsga2_survival()` — Non-dominated sorting with crowding distance
- `truncation_survival()` — Keep best k by fitness
- `elitist_survival(elite_count=1)` — Preserve elite parents, fill with best offspring

---

## Implementing Custom Strategies

### Custom Parent Selector

```python
import numpy as np
from ctrl_freak import Population

def rank_proportionate_selection(
    pop: Population,
    n_parents: int,
    rng: np.random.Generator,
    **kwargs: np.ndarray,
) -> np.ndarray:
    """Select parents with probability inversely proportional to rank."""
    rank = kwargs.get('rank')
    if rank is None:
        raise ValueError("rank_proportionate_selection requires 'rank' kwarg")

    # Lower rank = higher probability
    max_rank = rank.max()
    weights = (max_rank - rank + 1).astype(float)
    weights /= weights.sum()

    return rng.choice(len(pop), size=n_parents, p=weights).astype(np.intp)


# Usage
result = nsga2(..., select=rank_proportionate_selection)
```

### Custom Survivor Selector

```python
def age_based_survival(
    pop: Population,
    n_survivors: int,
    **kwargs: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Survival favoring younger individuals (lower index = older)."""
    fitness = kwargs.get('fitness')
    if fitness is None:
        if pop.objectives is None or pop.objectives.shape[1] != 1:
            raise ValueError("Requires fitness kwarg or single-objective population")
        fitness = pop.objectives[:, 0]

    # Favor younger individuals by adding age penalty
    age_penalty = np.arange(len(pop)) * 0.01
    adjusted_fitness = fitness + age_penalty

    indices = np.argsort(adjusted_fitness)[:n_survivors]
    return indices.astype(np.intp), {'fitness': fitness[indices]}


# Usage
result = ga(..., survive=age_based_survival)
```

---

## Contract Summary Table

### User Functions

| Function | Input Shape | Output Shape | User Enforces | Framework Provides |
|----------|-------------|--------------|---------------|-------------------|
| `init` | (seeded RNG) | `(n_vars,)` | Bounds | Seeded RNG |
| `evaluate` | `(n_vars,)` | `(n_obj,)` or `float` | Finite values, constraints | Lifting to population |
| `crossover` | `(n_vars,)`, `(n_vars,)` | `(n_vars,)` | Bounds preservation | Tournament-selected parents |
| `mutate` | `(n_vars,)` | `(n_vars,)` | Bounds (clip) | Nothing |

### Strategy Protocols

| Protocol | Input | Output | Built-in Implementations |
|----------|-------|--------|--------------------------|
| `ParentSelector` | pop, n_parents, rng, **kwargs | indices array | crowded_tournament, fitness_tournament, roulette_wheel |
| `SurvivorSelector` | pop, n_survivors, **kwargs | (indices, state_dict) | nsga2_survival, truncation_survival, elitist_survival |

---

## Constraint Handling Strategies

The framework is constraint-agnostic. Users implement constraint handling through their functions.

### Design Philosophy: Feasibility is User Responsibility

**All user functions must return feasible individuals.** This is a consistent contract across `init`, `crossover`, and `mutate`. The framework does not provide a `repair` parameter — if you need to repair solutions, compose your operators with a repair function:

```python
from ctrl_freak import sbx_crossover, polynomial_mutation

# Your domain-specific repair function
def repair(x: np.ndarray) -> np.ndarray:
    # Example: enforce x[0] + x[1] <= 1
    if x[0] + x[1] > 1:
        scale = 1 / (x[0] + x[1])
        x = x.copy()
        x[0] *= scale
        x[1] *= scale
    return x

# Compose repair with standard operators
sbx = sbx_crossover(eta=15.0, bounds=(0.0, 1.0), seed=42)
poly_mut = polynomial_mutation(eta=20.0, bounds=(0.0, 1.0), seed=42)

crossover = lambda p1, p2: repair(sbx(p1, p2))
mutate = lambda x: repair(poly_mut(x))

# Use composed operators
result = nsga2(
    init=lambda rng: repair(rng.uniform(0, 1, size=10)),
    evaluate=evaluate,
    crossover=crossover,
    mutate=mutate,
    pop_size=100,
    n_generations=50,
)
```

This approach keeps the framework simple and gives you full control over when and how repair is applied.

### Hard Constraints (must be satisfied)

**Strategy 1: Repair in mutation**

```python
def mutate(x: np.ndarray) -> np.ndarray:
    mutated = x + np.random.normal(0, 0.1, len(x))
    mutated = np.clip(mutated, lower_bounds, upper_bounds)  # Box constraints

    # Repair constraint: x[0] + x[1] <= 1
    if mutated[0] + mutated[1] > 1:
        scale = 1 / (mutated[0] + mutated[1])
        mutated[0] *= scale
        mutated[1] *= scale

    return mutated
```

**Strategy 2: Preserve in crossover**

```python
def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # Both parents satisfy constraint; convex combination also satisfies
    alpha = np.random.random()
    return alpha * p1 + (1 - alpha) * p2  # Preserves convex constraints
```

**Strategy 3: Rejection in init**

```python
def init(rng: np.random.Generator) -> np.ndarray:
    while True:
        x = rng.uniform(0, 1, size=N_VARS)
        if constraint_satisfied(x):
            return x
```

### Soft Constraints (penalty-based)

Add constraint violations as additional objectives:

```python
def evaluate(x: np.ndarray) -> np.ndarray:
    # Original objectives
    f1 = objective_1(x)
    f2 = objective_2(x)

    # Constraint violations (0 when satisfied, positive when violated)
    g1 = max(0, x[0] + x[1] - 1)           # x[0] + x[1] <= 1
    g2 = max(0, 0.5 - x[2])                 # x[2] >= 0.5

    # Option A: Sum all violations into one penalty objective
    return np.array([f1, f2, g1 + g2])

    # Option B: Each constraint as separate objective (higher dimensional Pareto front)
    return np.array([f1, f2, g1, g2])
```

### Mixed Strategy

Combine repair for simple constraints with penalties for complex ones:

```python
def mutate(x: np.ndarray) -> np.ndarray:
    mutated = x + np.random.normal(0, 0.1, len(x))
    # Repair simple box constraints
    mutated = np.clip(mutated, 0, 1)
    return mutated

def evaluate(x: np.ndarray) -> np.ndarray:
    f1 = compute_cost(x)
    f2 = compute_time(x)

    # Complex nonlinear constraint as penalty
    violation = max(0, nonlinear_constraint(x))
    return np.array([f1, f2, violation])
```

---

## Common Pitfalls

### Shape Errors

```python
# Wrong: returns (n_obj, 1) instead of (n_obj,)
def evaluate(x):
    return np.array([[f1], [f2]])  # Shape (2, 1)

# Correct
def evaluate(x):
    return np.array([f1, f2])      # Shape (2,)
```

### Non-finite Values

```python
# Wrong: can return inf
def evaluate(x):
    return np.array([1 / x[0], x[1]])  # inf when x[0] = 0

# Correct: handle edge cases
def evaluate(x):
    f1 = 1 / x[0] if x[0] != 0 else 1e10  # Large but finite
    return np.array([f1, x[1]])
```

### Ignoring the Provided RNG

```python
# Wrong: not reproducible
def init(rng):
    return np.random.random(N_VARS)  # Uses global state

# Correct
def init(rng):
    return rng.random(N_VARS)        # Uses provided generator
```

### Forgetting Bounds Enforcement

```python
# Wrong: mutation can escape bounds
def mutate(x):
    return x + np.random.normal(0, 0.1, len(x))

# Correct
def mutate(x):
    mutated = x + np.random.normal(0, 0.1, len(x))
    return np.clip(mutated, lower, upper)
```
