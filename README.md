# ctrl-freak: NSGA-II Implementation

A pure numpy implementation of the NSGA-II multi-objective genetic algorithm.

---

## Design Decisions

### Philosophy

- **Pure numpy** for performance
- **Functional style** with immutable data structures
- **User thinks about individuals**, framework handles vectorization via `lift()`
- **Fail fast** with eager validation
- **Domain agnostic** — framework handles selection pressure, user handles constraints/bounds

---

## Components

### 1. Population Representation ✅ DESIGNED

**Decision**: Struct of Arrays (not Array of Structs)

```python
@dataclass(frozen=True)
class Population:
    x: np.ndarray                         # (n, n_vars) - always present
    objectives: np.ndarray | None         # (n, n_obj)
    rank: np.ndarray | None               # (n,) int
    crowding_distance: np.ndarray | None  # (n,) float

    def __post_init__(self): ...          # Validate shapes, fail fast
    def __getitem__(self, idx) -> IndividualView: ...
```

**Key points**:

- Frozen (immutable) - operations return new Population
- Validation on construction
- `__getitem__` returns IndividualView for convenience

### 2. IndividualView ✅ DESIGNED

```python
@dataclass(frozen=True)
class IndividualView:
    x: np.ndarray                         # (n_vars,)
    objectives: np.ndarray | None         # (n_obj,)
    rank: int | None
    crowding_distance: float | None
```

Read-only view of a single individual. Used for `Population.__getitem__` and returning results.

### 3. User Contracts ✅ DESIGNED

User defines functions at the **individual level**. Framework lifts them.

| Function | Signature (per individual) |
|----------|---------------------------|
| evaluate | `(n_vars,) -> (n_obj,)` |
| crossover | `(n_vars,), (n_vars,) -> (n_vars,)` |
| mutate | `(n_vars,) -> (n_vars,)` |

**Note**: The framework is **constraint and bounds agnostic**. Users handle domain-specific logic:
- Bounds enforcement → clip/repair in `mutate`
- Hard constraints → preserve in `crossover`, repair in `mutate`
- Soft constraints → encode as penalty objectives in `evaluate`

### 4. Lifting Utility ✅ DESIGNED

```python
def lift(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Lift a per-individual function to work on a population."""
    def lifted(x: np.ndarray) -> np.ndarray:
        return np.stack([fn(x[i]) for i in range(x.shape[0])])
    return lifted
```

User writes simple code, framework handles batching.

### 5. NSGA-II Primitives ✅ DESIGNED

Three pure functions for Pareto-based ranking and diversity.

#### 5.1 `dominates` (scalar)

```python
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if solution a Pareto-dominates solution b (minimization).

    a dominates b iff:
      - a[i] <= b[i] for ALL objectives (at least as good)
      - a[i] < b[i] for AT LEAST ONE objective (strictly better)

    Args:
        a: Shape (n_obj,)
        b: Shape (n_obj,)

    Returns:
        True if a dominates b
    """
    return np.all(a <= b) and np.any(a < b)
```

#### 5.2 `dominates_matrix` (vectorized)

```python
def dominates_matrix(objectives: np.ndarray) -> np.ndarray:
    """Compute pairwise dominance for all individuals.

    Args:
        objectives: Shape (n, n_obj)

    Returns:
        Shape (n, n) bool array where result[i,j] = True iff i dominates j
    """
```

**Why both?** Scalar is the "spec" (clear, testable). Matrix is the "implementation" (fast, used by `non_dominated_sort`).

#### 5.3 `non_dominated_sort`

```python
def non_dominated_sort(objectives: np.ndarray) -> np.ndarray:
    """Assign each individual to a Pareto front (Deb's fast algorithm).

    Args:
        objectives: Shape (n, n_obj)

    Returns:
        ranks: Shape (n,) int array where rank[i] is the front index
               (0 = Pareto optimal, 1 = second front, etc.)
    """
```

**Implementation**: Uses `dominates_matrix` internally. O(MN²) time/space where M = objectives, N = population.

#### 5.4 `crowding_distance`

```python
def crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for individuals in a SINGLE front.

    Args:
        front_objectives: Shape (n_front, n_obj) - objectives for ONE front only

    Returns:
        distances: Shape (n_front,) - higher = more isolated = preferred
    """
```

**Key design decisions**:

| Question | Decision | Rationale |
|----------|----------|-----------|
| API scope | Single front only | Pure functional, single responsibility. Caller iterates fronts. |
| Ties (identical objectives) | No special handling | Algorithm naturally assigns low distance (correct - they ARE crowded) |
| Single individual in front | Returns `inf` | Falls out naturally from boundary rule (it's both min and max) |

**Caller composes**:
```python
distances = np.zeros(n)
for r in range(max_rank + 1):
    mask = (ranks == r)
    distances[mask] = crowding_distance(objectives[mask])
```

---

## NSGA-II Core (all designed)

### 6. Selection ✅ DESIGNED

Binary tournament selection using crowded comparison operator (vectorized).

```python
def select_parents(pop: Population, n_parents: int, rng: np.random.Generator) -> np.ndarray:
    """Select n_parents using binary tournament (vectorized).

    Args:
        pop: Population with rank and crowding_distance computed
        n_parents: Number of parents to select
        rng: Random generator

    Returns:
        Shape (n_parents,) indices into population
    """
    n = len(pop.x)
    candidates = rng.integers(0, n, size=(n_parents, 2))

    rank_a = pop.rank[candidates[:, 0]]
    rank_b = pop.rank[candidates[:, 1]]
    cd_a = pop.crowding_distance[candidates[:, 0]]
    cd_b = pop.crowding_distance[candidates[:, 1]]

    # a wins if: lower rank OR (same rank AND higher crowding distance)
    a_wins = (rank_a < rank_b) | ((rank_a == rank_b) & (cd_a >= cd_b))

    return np.where(a_wins, candidates[:, 0], candidates[:, 1])
```

**Key decisions**:
- Vectorized for performance (no Python loops)
- Explicit RNG injection for reproducibility
- Tie-break: first candidate wins (deterministic, `>=` on crowding distance)

### 7. Offspring Creation ✅ DESIGNED

Create offspring via selection, crossover, and mutation.

```python
def create_offspring(
    pop: Population,
    n_offspring: int,
    crossover: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mutate: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator
) -> np.ndarray:
    """Create offspring via selection, crossover, and mutation.

    Args:
        pop: Parent population (with rank/crowding computed)
        n_offspring: Number of offspring to create
        crossover: User's crossover function, (n_vars,), (n_vars,) -> (n_vars,)
        mutate: User's mutation function, (n_vars,) -> (n_vars,)
        rng: Random generator

    Returns:
        Shape (n_offspring, n_vars) - offspring decision variables (unevaluated)
    """
    parent_idx = select_parents(pop, n_offspring * 2, rng)

    # Crossover pairs
    offspring_x = np.stack([
        crossover(pop.x[parent_idx[2*i]], pop.x[parent_idx[2*i + 1]])
        for i in range(n_offspring)
    ])

    # Mutate all
    offspring_x = np.stack([mutate(x) for x in offspring_x])

    return offspring_x
```

**Key decisions**:
- Crossover produces 1 child per pair (simpler API)
- Loop-based application (simple, optimize later if profiling shows bottleneck)
- Returns unevaluated offspring — evaluation and P ∪ Q combination happen in main loop

### 8. Survivor Selection ✅ DESIGNED

Select N survivors from combined P ∪ Q (2N) using NSGA-II crowded selection.

```python
def survivor_selection(pop: Population, n_survivors: int) -> Population:
    """Select survivors using NSGA-II crowded selection.

    Args:
        pop: Combined P ∪ Q with objectives computed (2N individuals)
        n_survivors: Target size (N)

    Returns:
        Population of size n_survivors, with rank/crowding recomputed
    """
    ranks = non_dominated_sort(pop.objectives)

    selected = []
    current_rank = 0

    while len(selected) < n_survivors:
        front_idx = np.where(ranks == current_rank)[0]

        if len(selected) + len(front_idx) <= n_survivors:
            # Whole front fits
            selected.extend(front_idx)
        else:
            # Critical front — use crowding distance
            remaining = n_survivors - len(selected)
            cd = crowding_distance(pop.objectives[front_idx])
            top_cd = np.argsort(cd)[::-1][:remaining]
            selected.extend(front_idx[top_cd])

        current_rank += 1

    selected = np.array(selected)

    # Extract selected individuals
    new_x = pop.x[selected]
    new_obj = pop.objectives[selected]

    # Recompute rank/crowding for survivors (needed for next generation's selection)
    new_ranks = non_dominated_sort(new_obj)
    new_cd = np.zeros(n_survivors)
    for r in range(new_ranks.max() + 1):
        mask = (new_ranks == r)
        new_cd[mask] = crowding_distance(new_obj[mask])

    return Population(x=new_x, objectives=new_obj, rank=new_ranks, crowding_distance=new_cd)
```

**Key decisions**:
- Fill front-by-front until capacity reached
- Critical front uses crowding distance to select most diverse individuals
- Recompute rank/crowding for survivors — returned population is ready for next generation's selection

### 9. Main Loop ✅ DESIGNED

Orchestrates initialization, evolution, and termination.

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
) -> Population:
    """Run NSGA-II optimization.

    Args:
        init: Initialize one individual, (rng,) -> (n_vars,)
        evaluate: Evaluate one individual, (n_vars,) -> (n_obj,)
        crossover: Cross two parents, (n_vars,), (n_vars,) -> (n_vars,)
        mutate: Mutate one individual, (n_vars,) -> (n_vars,)
        pop_size: Population size N
        n_generations: Number of generations
        seed: Random seed for reproducibility
        callback: Optional (pop, gen) -> stop? Called each generation

    Returns:
        Final population with rank/crowding computed
    """
    rng = np.random.default_rng(seed)

    # Initialize
    init_x = np.stack([init(rng) for _ in range(pop_size)])
    init_obj = lift(evaluate)(init_x)
    ranks = non_dominated_sort(init_obj)
    cd = np.zeros(pop_size)
    for r in range(ranks.max() + 1):
        mask = (ranks == r)
        cd[mask] = crowding_distance(init_obj[mask])
    pop = Population(x=init_x, objectives=init_obj, rank=ranks, crowding_distance=cd)

    # Main loop
    for gen in range(n_generations):
        if callback and callback(pop, gen):
            break

        offspring_x = create_offspring(pop, pop_size, crossover, mutate, rng)
        offspring_obj = lift(evaluate)(offspring_x)

        combined = Population(
            x=np.concatenate([pop.x, offspring_x]),
            objectives=np.concatenate([pop.objectives, offspring_obj]),
            rank=None,
            crowding_distance=None,
        )

        pop = survivor_selection(combined, pop_size)

    return pop
```

**Key decisions**:
- User provides `init` function (framework is bounds-agnostic)
- Fixed `n_generations` with optional `callback` for early stopping
- Returns final population — user extracts Pareto front via `pop.x[pop.rank == 0]`

---

## Future Work

### Default Operators (optional)

Convenience functions for continuous optimization — deferred until core is implemented and tested.

- SBX crossover (standard for continuous)
- Polynomial mutation (standard for continuous)
- Random initialization within bounds

### Parallelization

Could `lift()` use multiprocessing for expensive evaluate functions? Deferred, but worth considering API implications.

---

## Architecture Layers

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

