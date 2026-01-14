# NSGA-II Benchmarks

Benchmark suite for evaluating ctrl-freak NSGA-II implementation against standard multi-objective optimization test problems.

## Test Problems

### ZDT Suite

The ZDT (Zitzler-Deb-Thiele) test problems are standard benchmarks for multi-objective evolutionary algorithms:

| Problem | Pareto Front | Characteristics |
|---------|--------------|-----------------|
| ZDT1 | Convex | Continuous, 30 variables |
| ZDT2 | Concave | Non-convex front, 30 variables |
| ZDT3 | Discontinuous | Multiple disconnected segments |

All problems have:

- Decision variables in [0, 1]
- Two objectives to minimize
- Known analytical Pareto-optimal fronts

## Methodology

*To be filled after running benchmarks*

### Algorithm Configuration

- Population size: TBD
- Number of generations: TBD
- Crossover: SBX (eta = TBD)
- Mutation: Polynomial (eta = TBD)
- Number of runs: TBD (for statistical significance)

### Metrics

- **Hypervolume (HV)**: Volume of objective space dominated by the Pareto front approximation
- Reference point: [1.1, 1.1] for all ZDT problems

## Results

*Placeholder - results tables will be added after running benchmarks*

### Hypervolume Comparison

| Problem | ctrl-freak HV | Reference HV | Gap |
|---------|---------------|--------------|-----|
| ZDT1 | - | - | - |
| ZDT2 | - | - | - |
| ZDT3 | - | - | - |

### Convergence Plots

*To be added*

## How to Run

```bash
# Run all ZDT benchmarks
uv run python benchmarks/run_zdt.py

# Run specific problem
uv run python benchmarks/run_zdt.py --problem zdt1

# Run with custom settings
uv run python benchmarks/run_zdt.py --pop-size 100 --generations 250 --runs 10
```

## References

1. Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: Empirical results. *Evolutionary computation*, 8(2), 173-195.

2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE transactions on evolutionary computation*, 6(2), 182-197.
