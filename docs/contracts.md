# ctrl-freak: User Function Contracts

This document specifies the exact contracts for user-provided functions in ctrl-freak NSGA-II.

---

## Philosophy

**User thinks about individuals. Framework handles populations.**

You write functions that operate on a single individual. The framework:

- Calls your functions in a loop over the population
- Handles parent selection via tournament
- Manages Pareto ranking and crowding distance
- Orchestrates the generational cycle

This separation keeps user code simple and domain-focused while the framework handles algorithmic complexity.

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

### `evaluate(x: np.ndarray) -> np.ndarray`

Computes objective values for one individual.

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

## Contract Summary Table

| Function | Input Shape | Output Shape | User Enforces | Framework Provides |
|----------|-------------|--------------|---------------|-------------------|
| `init` | (seeded RNG) | `(n_vars,)` | Bounds | Seeded RNG |
| `evaluate` | `(n_vars,)` | `(n_obj,)` | Finite values, constraints as penalties | Lifting to population |
| `crossover` | `(n_vars,)`, `(n_vars,)` | `(n_vars,)` | Bounds, constraint preservation | Tournament-selected parents |
| `mutate` | `(n_vars,)` | `(n_vars,)` | Bounds (clip), constraint repair | Nothing |

---

## Constraint Handling Strategies

The framework is constraint-agnostic. Users implement constraint handling through their functions.

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
