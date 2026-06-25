"""Standard bounded genetic operators for NSGA-II.

This module provides Simulated Binary Crossover (SBX) and polynomial mutation
operators for real-valued decision vectors.

Seed-injection contract (consumed by the algorithm layer): `sbx_crossover(...)`
and `polynomial_mutation(...)` return callable operator objects. Each exposes
`set_rng(rng: numpy.random.Generator) -> None`, which replaces the operator's
internal generator. Until `set_rng` is called, the operator uses the generator
built from its constructor `seed=`. The call signatures are unchanged:
`crossover(p1, p2) -> child`, `mutate(x) -> x'`. Algorithms unify
reproducibility by calling `set_rng` with a `SeedSequence.spawn`-derived
generator after constructing operators; standalone users may ignore `set_rng`
and rely on `seed=`.
"""

from collections.abc import Callable

import numpy as np

Bounds = tuple[float, float] | tuple[np.ndarray, np.ndarray]
"""Bounds for decision variables.

A scalar pair ``(lower, upper)`` applies the same bounds to all variables.
A pair of arrays ``(lower_array, upper_array)`` specifies per-variable bounds.
"""


class _SeededOperator:
    """Mixin providing an injectable numpy Generator for genetic operators.

    The operator owns a ``numpy.random.Generator`` built from ``seed`` at
    construction. An orchestrating algorithm may replace it post-construction
    via :meth:`set_rng` so a single master seed reproduces the entire run.
    """

    _rng: np.random.Generator

    def set_rng(self, rng: np.random.Generator) -> None:
        """Inject a numpy Generator, replacing the operator's internal RNG.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator to use for all subsequent draws. Replaces the generator
            built from the constructor ``seed``.
        """
        self._rng = rng


class _SBXCrossover(_SeededOperator):
    """Callable bounded Simulated Binary Crossover operator."""

    def __init__(self, eta: float, bounds: Bounds, seed: int | None) -> None:
        self.eta = float(eta)
        self.bounds = bounds
        self._rng = np.random.default_rng(seed)

    def __call__(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Apply bounded SBX to two parents and return one child.

        Parameters
        ----------
        p1 : numpy.ndarray
            First parent decision vector.
        p2 : numpy.ndarray
            Second parent decision vector.

        Returns
        -------
        numpy.ndarray
            One child decision vector with the same shape as ``p1``.
        """
        eps = 1e-14
        eta = self.eta
        exponent = 1.0 / (eta + 1.0)
        rng = self._rng

        lower, upper = self.bounds
        xl = np.broadcast_to(np.asarray(lower, dtype=float), p1.shape)
        xu = np.broadcast_to(np.asarray(upper, dtype=float), p1.shape)

        sm = p1 < p2
        y1 = np.where(sm, p1, p2)
        y2 = np.where(sm, p2, p1)

        eligible = (np.abs(p1 - p2) > eps) & (xl < xu)
        delta = np.where(eligible, y2 - y1, 1.0)

        rand = rng.random(p1.shape[0])

        def calc_betaq(beta: np.ndarray) -> np.ndarray:
            alpha = 2.0 - np.power(beta, -(eta + 1.0))
            mask = rand <= (1.0 / alpha)
            return np.where(
                mask,
                np.power(rand * alpha, exponent),
                np.power(1.0 / (2.0 - rand * alpha), exponent),
            )

        beta1 = 1.0 + (2.0 * (y1 - xl) / delta)
        c1 = 0.5 * ((y1 + y2) - calc_betaq(beta1) * delta)

        beta2 = 1.0 + (2.0 * (xu - y2) / delta)
        c2 = 0.5 * ((y1 + y2) + calc_betaq(beta2) * delta)

        child_for_p1 = np.where(sm, c1, c2)
        child_for_p2 = np.where(sm, c2, c1)
        child = np.where(rng.random(p1.shape[0]) < 0.5, child_for_p1, child_for_p2)
        child = np.where(eligible, child, p1)
        return np.clip(child, xl, xu)


class _PolynomialMutation(_SeededOperator):
    """Callable bounded polynomial mutation operator."""

    def __init__(self, eta: float, prob: float | None, bounds: Bounds, seed: int | None) -> None:
        self.eta = float(eta)
        self.prob = prob
        self.bounds = bounds
        self._rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply polynomial mutation to an individual.

        Parameters
        ----------
        x : numpy.ndarray
            Decision vector to mutate.

        Returns
        -------
        numpy.ndarray
            Mutated decision vector with the same shape as ``x``.
        """
        n_vars = len(x)
        mutation_prob = self.prob if self.prob is not None else 1.0 / n_vars
        mutation_mask = self._rng.random(n_vars) < mutation_prob

        if not np.any(mutation_mask):
            return x.copy()

        lower, upper = self.bounds
        lower_arr = np.broadcast_to(np.asarray(lower, dtype=float), x.shape)
        upper_arr = np.broadcast_to(np.asarray(upper, dtype=float), x.shape)
        delta_max = upper_arr - lower_arr
        degenerate = delta_max == 0
        safe_delta = np.where(degenerate, 1.0, delta_max)

        delta_l = (x - lower_arr) / safe_delta
        delta_r = (upper_arr - x) / safe_delta

        u = self._rng.random(n_vars)

        xy_left = 1.0 - delta_l
        val_left = 2.0 * u + (1.0 - 2.0 * u) * (xy_left ** (self.eta + 1.0))
        delta_q_left = val_left ** (1.0 / (self.eta + 1.0)) - 1.0

        xy_right = 1.0 - delta_r
        val_right = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy_right ** (self.eta + 1.0))
        delta_q_right = 1.0 - val_right ** (1.0 / (self.eta + 1.0))

        delta_q = np.where(u < 0.5, delta_q_left, delta_q_right)

        effective_mask = mutation_mask & ~degenerate
        mutated = np.where(effective_mask, x + delta_q * delta_max, x)
        return np.clip(mutated, lower_arr, upper_arr)


def sbx_crossover(
    eta: float = 15.0,
    bounds: Bounds = (0.0, 1.0),
    seed: int | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a bounded Simulated Binary Crossover operator.

    Parameters
    ----------
    eta : float, default=15.0
        Distribution index. Higher values produce children closer to parents.
    bounds : tuple, default=(0.0, 1.0)
        Lower and upper decision-variable bounds. Scalars apply to all
        variables; arrays provide per-variable bounds.
    seed : int, optional
        Random seed for the standalone generator used until ``set_rng`` is
        called.

    Returns
    -------
    collections.abc.Callable
        Callable with signature ``(p1, p2) -> child``. The returned object also
        exposes ``set_rng(rng)`` for algorithm-level seed injection.

    References
    ----------
    Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for
    continuous search space. Complex Systems, 9(2), 115-148.

    Examples
    --------
    >>> crossover = sbx_crossover(eta=15.0, bounds=(0.0, 1.0), seed=42)
    >>> p1 = np.array([0.2, 0.4, 0.6])
    >>> p2 = np.array([0.3, 0.5, 0.7])
    >>> child = crossover(p1, p2)
    >>> child.shape
    (3,)

    Per-variable bounds:

    >>> lower = np.array([0.0, -10.0, 100.0])
    >>> upper = np.array([1.0, 10.0, 200.0])
    >>> crossover = sbx_crossover(eta=15.0, bounds=(lower, upper), seed=42)
    >>> crossover(np.array([0.5, 0.0, 150.0]), np.array([0.8, -5.0, 180.0])).shape
    (3,)

    Seed injection:

    >>> cx = sbx_crossover(eta=15.0, bounds=(0.0, 1.0), seed=0)
    >>> cx.set_rng(np.random.default_rng(123))
    >>> child = cx(np.array([0.2, 0.4]), np.array([0.6, 0.8]))
    >>> child.shape
    (2,)
    """
    return _SBXCrossover(eta=eta, bounds=bounds, seed=seed)


def polynomial_mutation(
    eta: float = 20.0,
    prob: float | None = None,
    bounds: Bounds = (0.0, 1.0),
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a bounded polynomial mutation operator.

    Parameters
    ----------
    eta : float, default=20.0
        Distribution index. Higher values produce smaller perturbations.
    prob : float, optional
        Mutation probability per variable. If omitted, uses ``1 / n_vars``.
    bounds : tuple, default=(0.0, 1.0)
        Lower and upper decision-variable bounds. Scalars apply to all
        variables; arrays provide per-variable bounds.
    seed : int, optional
        Random seed for the standalone generator used until ``set_rng`` is
        called.

    Returns
    -------
    collections.abc.Callable
        Callable with signature ``x -> x'``. The returned object also exposes
        ``set_rng(rng)`` for algorithm-level seed injection.

    References
    ----------
    Deb, K., & Goyal, M. (1996). A combined genetic adaptive search (GeneAS)
    for engineering design. Computer Science and Informatics, 26(4), 30-45.

    Examples
    --------
    >>> mutate = polynomial_mutation(eta=20.0, prob=0.1, bounds=(0.0, 1.0), seed=42)
    >>> x = np.array([0.5, 0.5, 0.5])
    >>> mutated = mutate(x)
    >>> mutated.shape
    (3,)

    Per-variable bounds:

    >>> lower = np.array([0.0, -10.0, 100.0])
    >>> upper = np.array([1.0, 10.0, 200.0])
    >>> mutate = polynomial_mutation(eta=20.0, prob=0.1, bounds=(lower, upper), seed=42)
    >>> mutate(np.array([0.5, 0.0, 150.0])).shape
    (3,)
    """
    return _PolynomialMutation(eta=eta, prob=prob, bounds=bounds, seed=seed)
