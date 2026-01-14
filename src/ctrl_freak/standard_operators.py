"""Standard genetic operators for NSGA-II.

This module provides well-known genetic operators commonly used in evolutionary
multi-objective optimization:

- SBX (Simulated Binary Crossover): A crossover operator that simulates
  single-point crossover behavior for real-valued variables
- Polynomial Mutation: A bounded mutation operator with controllable spread

Both operators are implemented as factory functions that return operator
functions compatible with the nsga2() interface.
"""

from collections.abc import Callable

import numpy as np


def sbx_crossover(
    eta: float = 15.0,
    bounds: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a Simulated Binary Crossover (SBX) operator.

    SBX simulates single-point crossover behavior for real-valued variables.
    It produces children whose distribution around the parent values is
    controlled by the distribution index eta.

    Args:
        eta: Distribution index (default 15.0). Higher values produce children
            closer to parents; lower values allow more exploration.
            Typical range: 2-20.
        bounds: Lower and upper bounds for decision variables (default (0.0, 1.0)).
            Children are clipped to these bounds.
        seed: Random seed for reproducibility. If None, uses a random seed.

    Returns:
        A crossover function with signature (p1, p2) -> child that is
        compatible with nsga2()'s crossover parameter.

    Example:
        >>> crossover = sbx_crossover(eta=15.0, bounds=(0.0, 1.0), seed=42)
        >>> p1 = np.array([0.2, 0.4, 0.6])
        >>> p2 = np.array([0.3, 0.5, 0.7])
        >>> child = crossover(p1, p2)
        >>> child.shape
        (3,)

    References:
        Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for
        continuous search space. Complex Systems, 9(2), 115-148.
    """
    rng = np.random.default_rng(seed)

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Apply SBX crossover to two parents, returning one child."""
        n_vars = len(p1)
        child = np.empty(n_vars, dtype=np.float64)

        for i in range(n_vars):
            u = rng.random()

            beta = (2.0 * u) ** (1.0 / (eta + 1.0)) if u <= 0.5 else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

            # Generate two symmetric children and randomly select one
            c1 = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
            c2 = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
            child[i] = c1 if rng.random() < 0.5 else c2

        # Enforce bounds
        lower, upper = bounds
        return np.clip(child, lower, upper)

    return crossover


def polynomial_mutation(
    eta: float = 20.0,
    prob: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a polynomial mutation operator.

    Polynomial mutation applies a bounded perturbation to each variable
    with probability prob. The spread of the mutation is controlled by
    the distribution index eta.

    Args:
        eta: Distribution index (default 20.0). Higher values produce smaller
            perturbations (more local search); lower values allow larger jumps.
            Typical range: 20-100.
        prob: Mutation probability per variable (default None, which uses 1/n_vars).
            If specified, each variable is mutated with this probability.
        bounds: Lower and upper bounds for decision variables (default (0.0, 1.0)).
            Mutations respect these bounds.
        seed: Random seed for reproducibility. If None, uses a random seed.

    Returns:
        A mutation function with signature (x) -> x' that is compatible
        with nsga2()'s mutate parameter.

    Example:
        >>> mutate = polynomial_mutation(eta=20.0, prob=0.1, bounds=(0.0, 1.0), seed=42)
        >>> x = np.array([0.5, 0.5, 0.5])
        >>> mutated = mutate(x)
        >>> mutated.shape
        (3,)

    References:
        Deb, K., & Goyal, M. (1996). A combined genetic adaptive search (GeneAS)
        for engineering design. Computer Science and Informatics, 26(4), 30-45.
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    delta_max = upper - lower

    def mutate(x: np.ndarray) -> np.ndarray:
        """Apply polynomial mutation to an individual."""
        n_vars = len(x)
        mutation_prob = prob if prob is not None else 1.0 / n_vars
        mutated = x.copy()

        for i in range(n_vars):
            if rng.random() < mutation_prob:
                # Compute normalized distance to bounds
                delta_l = (mutated[i] - lower) / delta_max
                delta_r = (upper - mutated[i]) / delta_max

                u = rng.random()

                if u < 0.5:
                    # Mutation towards lower bound
                    xy = 1.0 - delta_l
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    # Mutation towards upper bound
                    xy = 1.0 - delta_r
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** (1.0 / (eta + 1.0))

                mutated[i] = mutated[i] + delta_q * delta_max

        # Ensure bounds are respected (numerical safety)
        return np.clip(mutated, lower, upper)

    return mutate
