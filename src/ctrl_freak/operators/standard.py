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

Bounds = tuple[float, float] | tuple[np.ndarray, np.ndarray]
"""Bounds for decision variables.

A scalar pair ``(lower, upper)`` applies the same bounds to all variables.
A pair of arrays ``(lower_array, upper_array)`` specifies per-variable bounds;
each array must have the same length as the decision vector.
"""


def sbx_crossover(
    eta: float = 15.0,
    bounds: Bounds = (0.0, 1.0),
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
            Children are clipped to these bounds. Can be either:
            - A scalar pair ``(lower, upper)`` applied uniformly to all variables.
            - A pair of arrays ``(lower_array, upper_array)`` for per-variable
              bounds, where each array has the same length as the decision vector.
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

        Per-variable bounds:

        >>> lower = np.array([0.0, -10.0, 100.0])
        >>> upper = np.array([1.0,  10.0, 200.0])
        >>> crossover = sbx_crossover(eta=15.0, bounds=(lower, upper), seed=42)

    References:
        Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for
        continuous search space. Complex Systems, 9(2), 115-148.
    """
    rng = np.random.default_rng(seed)

    def crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Apply SBX crossover to two parents, returning one child."""
        n_vars = len(p1)

        # Generate all random values at once
        u = rng.random(n_vars)

        # Compute beta using vectorized conditional
        exponent = 1.0 / (eta + 1.0)
        beta = np.where(u <= 0.5, (2.0 * u) ** exponent, (1.0 / (2.0 * (1.0 - u))) ** exponent)

        # Compute both symmetric children
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

        # Randomly select c1 or c2 for each variable
        child = np.where(rng.random(n_vars) < 0.5, c1, c2)

        # Enforce bounds
        lower, upper = bounds
        return np.clip(child, lower, upper)

    return crossover


def polynomial_mutation(
    eta: float = 20.0,
    prob: float | None = None,
    bounds: Bounds = (0.0, 1.0),
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
            Mutations respect these bounds. Can be either:
            - A scalar pair ``(lower, upper)`` applied uniformly to all variables.
            - A pair of arrays ``(lower_array, upper_array)`` for per-variable
              bounds, where each array has the same length as the decision vector.
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

        Per-variable bounds:

        >>> lower = np.array([0.0, -10.0, 100.0])
        >>> upper = np.array([1.0,  10.0, 200.0])
        >>> mutate = polynomial_mutation(eta=20.0, prob=0.1, bounds=(lower, upper), seed=42)

    References:
        Deb, K., & Goyal, M. (1996). A combined genetic adaptive search (GeneAS)
        for engineering design. Computer Science and Informatics, 26(4), 30-45.
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    delta_max = upper - lower

    def mutate(x: np.ndarray) -> np.ndarray:
        """Apply polynomial mutation to an individual (vectorized)."""
        n_vars = len(x)
        mutation_prob = prob if prob is not None else 1.0 / n_vars

        # Generate mutation mask for all variables at once
        mutation_mask = rng.random(n_vars) < mutation_prob

        # Early exit if no mutations needed
        if not np.any(mutation_mask):
            return x.copy()

        # Normalized distances to bounds
        delta_l = (x - lower) / delta_max
        delta_r = (upper - x) / delta_max

        # Random values for mutation direction
        u = rng.random(n_vars)

        # Compute delta_q for both branches (mutation towards lower/upper bound)
        xy_left = 1.0 - delta_l
        val_left = 2.0 * u + (1.0 - 2.0 * u) * (xy_left ** (eta + 1.0))
        delta_q_left = val_left ** (1.0 / (eta + 1.0)) - 1.0

        xy_right = 1.0 - delta_r
        val_right = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy_right ** (eta + 1.0))
        delta_q_right = 1.0 - val_right ** (1.0 / (eta + 1.0))

        # Select appropriate delta_q based on u value
        delta_q = np.where(u < 0.5, delta_q_left, delta_q_right)

        # Apply mutations only where mask is True
        mutated = np.where(mutation_mask, x + delta_q * delta_max, x)

        # Ensure bounds are respected (numerical safety)
        return np.clip(mutated, lower, upper)

    return mutate
