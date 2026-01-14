# NSGA-II Benchmarks

Benchmark suite comparing ctrl-freak against Pymoo and DEAP on standard multi-objective optimization test problems.

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

### Algorithm Configuration

All libraries configured with identical parameters:

| Parameter | Value |
|-----------|-------|
| Population size | 100 |
| Generations | 250 |
| Decision variables | 30 |
| Bounds | [0, 1] |
| Crossover | SBX (eta = 15) |
| Mutation | Polynomial (eta = 20, prob = 1/30) |
| Runs per config | 10 (seeds 0-9) |

### Metrics

- **Hypervolume (HV)**: Volume of objective space dominated by the Pareto front approximation (higher is better)
- **Reference point**: [1.1, 1.1] for all ZDT problems
- **Time**: Wall-clock seconds per run

## Results

### Hypervolume (mean +/- std)

| Problem | ctrl-freak | Pymoo | DEAP |
|---------|------------|-------|------|
| ZDT1 | 0.8688 +/- 0.0006 | 0.8241 +/- 0.0255 | **0.8698 +/- 0.0002** |
| ZDT2 | 0.5356 +/- 0.0004 | 0.4764 +/- 0.0182 | **0.5363 +/- 0.0002** |
| ZDT3 | 1.3261 +/- 0.0006 | 1.2836 +/- 0.0123 | **1.3275 +/- 0.0002** |

### Timing (mean seconds per run)

| Problem | ctrl-freak | Pymoo | DEAP |
|---------|------------|-------|------|
| ZDT1 | 0.74 | **0.34** | 1.85 |
| ZDT2 | 0.76 | **0.34** | 1.89 |
| ZDT3 | 0.77 | **0.35** | 1.88 |

### Key Findings

1. **Quality**: ctrl-freak matches DEAP's hypervolume within 0.1% across all problems
2. **Consistency**: ctrl-freak has very low variance (0.0004-0.0006), comparable to DEAP
3. **Speed**: ctrl-freak is ~60% faster than DEAP but ~2x slower than Pymoo
4. **Pymoo gap**: Pymoo shows lower HV and higher variance, likely due to differences in mutation parameter semantics (per-individual vs per-variable probability)

### Visual Comparison

<img src="zdt/results/zdt3_pareto_comparison.png" alt="ZDT3 Pareto Front Comparison" width="50%">

*ZDT3 discontinuous Pareto front: ctrl-freak and DEAP closely track the true front, while Pymoo shows visible scatter.*

## How to Run

```bash
# Run full benchmark suite
uv run python benchmarks/zdt/run_benchmark.py

# Results saved to benchmarks/zdt/results/benchmark_results.json
```

## References

1. Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: Empirical results. *Evolutionary computation*, 8(2), 173-195.

2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE transactions on evolutionary computation*, 6(2), 182-197.
