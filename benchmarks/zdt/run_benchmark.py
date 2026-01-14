"""Benchmark runner comparing ctrl-freak, Pymoo, and DEAP on ZDT problems.

This script runs NSGA-II on ZDT1-3 using three different libraries with
consistent parameters to enable fair comparison.

Usage:
    uv run python benchmarks/zdt/run_benchmark.py
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem as PymooProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from benchmarks.metrics import hypervolume
from benchmarks.zdt.problems import BOUNDS, N_VARS, PROBLEMS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Experiment parameters
POP_SIZE = 100
N_GENERATIONS = 250
SBX_ETA = 15.0
PM_ETA = 20.0
MUTATION_PROB = 1.0 / N_VARS
N_RUNS = 10
SEEDS = list(range(N_RUNS))


def run_ctrl_freak(problem_name: str, problem_fn: callable, seed: int) -> tuple[float, float]:
    """Run NSGA-II using ctrl-freak library.

    Args:
        problem_name: Name of the problem (for logging).
        problem_fn: The ZDT problem function.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (hypervolume, elapsed_time_seconds).
    """
    from ctrl_freak import nsga2, polynomial_mutation, sbx_crossover

    crossover = sbx_crossover(eta=SBX_ETA, bounds=BOUNDS, seed=seed)
    mutate = polynomial_mutation(eta=PM_ETA, prob=MUTATION_PROB, bounds=BOUNDS, seed=seed + 1000)

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(BOUNDS[0], BOUNDS[1], size=N_VARS)

    start_time = time.perf_counter()
    result = nsga2(
        init=init,
        evaluate=problem_fn,
        crossover=crossover,
        mutate=mutate,
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        seed=seed,
    )
    elapsed = time.perf_counter() - start_time

    hv = hypervolume(result.objectives)
    return hv, elapsed


class PymooZDTProblem(PymooProblem):
    """Wrapper to use ZDT functions with Pymoo."""

    def __init__(self, problem_fn: callable) -> None:
        super().__init__(n_var=N_VARS, n_obj=2, xl=BOUNDS[0], xu=BOUNDS[1])
        self._problem_fn = problem_fn

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        # Evaluate each individual
        f = np.array([self._problem_fn(xi) for xi in x])
        out["F"] = f


def run_pymoo(problem_name: str, problem_fn: callable, seed: int) -> tuple[float, float]:
    """Run NSGA-II using Pymoo library.

    Args:
        problem_name: Name of the problem (for logging).
        problem_fn: The ZDT problem function.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (hypervolume, elapsed_time_seconds).
    """
    problem = PymooZDTProblem(problem_fn)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=SBX_ETA, prob=1.0),
        mutation=PM(eta=PM_ETA, prob=MUTATION_PROB),
        eliminate_duplicates=False,
    )

    termination = get_termination("n_gen", N_GENERATIONS)

    start_time = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - start_time

    hv = hypervolume(result.pop.get("F"))
    return hv, elapsed


def _setup_deap() -> None:
    """Set up DEAP creator classes (handles cleanup for multiple runs)."""
    from deap import base, creator

    # Clean up any existing creator classes
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def run_deap(problem_name: str, problem_fn: callable, seed: int) -> tuple[float, float]:
    """Run NSGA-II using DEAP library.

    Args:
        problem_name: Name of the problem (for logging).
        problem_fn: The ZDT problem function.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (hypervolume, elapsed_time_seconds).
    """
    import random

    from deap import base, creator, tools

    _setup_deap()

    # Set up toolbox
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_float", random.random)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N_VARS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    def evaluate(individual: list) -> tuple[float, float]:
        x = np.array(individual)
        obj = problem_fn(x)
        return tuple(obj)

    toolbox.register("evaluate", evaluate)

    # SBX crossover
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=SBX_ETA, low=BOUNDS[0], up=BOUNDS[1])

    # Polynomial mutation
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=PM_ETA,
        low=BOUNDS[0],
        up=BOUNDS[1],
        indpb=MUTATION_PROB,
    )

    # Selection using NSGA-II
    toolbox.register("select", tools.selNSGA2)

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.perf_counter()

    # Initialize population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses, strict=True):
        ind.fitness.values = fit

    # Assign crowding distance for initial population (required for selTournamentDCD)
    pop = toolbox.select(pop, len(pop))

    # Run NSGA-II
    for _ in range(N_GENERATIONS):
        # Select parents using binary tournament with crowding distance
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring):
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses, strict=True):
            ind.fitness.values = fit

        # Select survivors from combined population (mu + lambda)
        pop = toolbox.select(pop + offspring, POP_SIZE)

    elapsed = time.perf_counter() - start_time

    # Extract objectives
    objectives = np.array([ind.fitness.values for ind in pop])
    hv = hypervolume(objectives)
    return hv, elapsed


def run_benchmark() -> dict:
    """Run the full benchmark suite.

    Returns:
        Dictionary containing metadata and results.
    """
    timestamp = datetime.now(UTC).isoformat()

    metadata = {
        "timestamp": timestamp,
        "parameters": {
            "pop_size": POP_SIZE,
            "n_generations": N_GENERATIONS,
            "n_vars": N_VARS,
            "bounds": list(BOUNDS),
            "sbx_eta": SBX_ETA,
            "pm_eta": PM_ETA,
            "mutation_prob": MUTATION_PROB,
            "n_runs": N_RUNS,
            "seeds": SEEDS,
        },
    }

    results = []

    runners = [
        ("ctrl-freak", run_ctrl_freak),
        ("pymoo", run_pymoo),
        ("deap", run_deap),
    ]

    total_runs = len(PROBLEMS) * len(runners) * N_RUNS
    current_run = 0

    for problem_name, problem_fn in PROBLEMS.items():
        for library_name, runner in runners:
            for seed in SEEDS:
                current_run += 1
                logger.info(
                    f"Running [{current_run}/{total_runs}]: {library_name} on {problem_name.upper()} (seed={seed})"
                )

                hv, elapsed = runner(problem_name, problem_fn, seed)

                results.append(
                    {
                        "library": library_name,
                        "problem": problem_name.upper(),
                        "seed": seed,
                        "hypervolume": hv,
                        "time_seconds": elapsed,
                    }
                )

                logger.info(f"  HV: {hv:.4f}, Time: {elapsed:.2f}s")

    return {"metadata": metadata, "results": results}


def print_summary(results: dict) -> None:
    """Print a summary table of the benchmark results.

    Args:
        results: The benchmark results dictionary.
    """
    # Organize results by problem and library
    from collections import defaultdict

    data = defaultdict(lambda: defaultdict(list))

    for r in results["results"]:
        data[r["problem"]][r["library"]].append(r["hypervolume"])

    problems = sorted(data.keys())
    libraries = ["ctrl-freak", "pymoo", "deap"]

    # Print header
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\nParameters: pop_size={POP_SIZE}, generations={N_GENERATIONS}, runs={N_RUNS}")
    print()

    # Print table header
    header = f"{'Problem':<10}"
    for lib in libraries:
        header += f"{lib:>22}"
    print(header)
    print("-" * 76)

    # Print rows
    for problem in problems:
        row = f"{problem:<10}"
        for lib in libraries:
            hvs = data[problem][lib]
            if hvs:
                mean_hv = np.mean(hvs)
                std_hv = np.std(hvs)
                row += f"{mean_hv:>14.4f} +/- {std_hv:.4f}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    print("-" * 76)

    # Print timing summary
    print("\nTiming (mean seconds per run):")
    time_data = defaultdict(lambda: defaultdict(list))
    for r in results["results"]:
        time_data[r["problem"]][r["library"]].append(r["time_seconds"])

    header = f"{'Problem':<10}"
    for lib in libraries:
        header += f"{lib:>15}"
    print(header)
    print("-" * 55)

    for problem in problems:
        row = f"{problem:<10}"
        for lib in libraries:
            times = time_data[problem][lib]
            if times:
                mean_time = np.mean(times)
                row += f"{mean_time:>15.2f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print()


def main() -> None:
    """Main entry point for the benchmark."""
    logger.info("Starting ZDT benchmark suite")
    logger.info(f"Parameters: pop_size={POP_SIZE}, generations={N_GENERATIONS}, runs={N_RUNS}")

    results = run_benchmark()

    # Save results to JSON
    output_path = Path(__file__).parent / "results" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
