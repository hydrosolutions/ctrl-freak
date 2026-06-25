"""Single-objective benchmark functions with known global optima.

This module defines the six standard single-objective test functions used to
validate ctrl-freak's ``ga()`` against pymoo and DEAP: Sphere, Rosenbrock,
Rastrigin, Ackley, Griewank, and Schwefel, all at ``DIM = 10`` dimensions. Each
function is pure numpy (returning a Python ``float``) and is paired, via the
``SO_PROBLEMS`` registry, with its known global optimum ``x*``, optimum value
``f*``, standard search bounds, dimensionality, and a success threshold ``ε``.

The optimum metadata (``x_star``, ``f_star``, ``epsilon``) is consumed by the
benchmark metrics layer (success rate, ``|f - f*|``, ``||x - x*||``) and the fair
cross-library harness (``bounds`` and ``dim`` parameterise the operators); the
functions themselves never reference it.

Constants
---------
DIM : int
    The shared problem dimensionality (10). Multimodality bites at 10D while a
    basic real-coded GA can still reach the global optimum, keeping success rate
    informative for the parity claim.
SO_PROBLEMS : dict[str, SOProblem]
    Registry mapping each lowercase problem name to its :class:`SOProblem`
    metadata bundle. Contains exactly the six names ``ackley``, ``griewank``,
    ``rastrigin``, ``rosenbrock``, ``schwefel``, ``sphere``.

Notes
-----
Standard definitions follow Jamil & Yang (2013) and the Virtual Library of
Simulation Experiments (Surjanovic & Bingham, SFU). Schwefel uses the canonical
offset ``418.9829 * d`` with per-dimension optimum ``x_i ≈ 420.9687``; because
both the offset and the optimum are rounded to four decimals, ``f(x*)`` carries a
~1.3e-4 residual at 10D rather than being exactly zero.

Examples
--------
>>> import numpy as np
>>> from benchmarks.problems.single_objective import SO_PROBLEMS, sphere
>>> sphere(np.zeros(10))
0.0
>>> sorted(SO_PROBLEMS)
['ackley', 'griewank', 'rastrigin', 'rosenbrock', 'schwefel', 'sphere']
>>> SO_PROBLEMS["sphere"].f_star
0.0
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DIM",
    "SOProblem",
    "SO_PROBLEMS",
    "ackley",
    "griewank",
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "sphere",
]

DIM: int = 10

# Schwefel literature constants (rounded to four decimals; see module Notes).
_SCHWEFEL_OFFSET: float = 418.9829
_SCHWEFEL_OPTIMUM: float = 420.9687


def sphere(x: np.ndarray) -> float:
    """Sphere function: the separable, unimodal sum of squares.

    Parameters
    ----------
    x : np.ndarray
        Decision vector, 1-D, any length. Global optimum at the zero vector.

    Returns
    -------
    float
        ``sum(x_i**2)``; minimum ``0.0`` at ``x = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import sphere
    >>> sphere(np.zeros(10))
    0.0
    >>> sphere(np.ones(10))
    10.0
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: the non-separable, curved-valley banana.

    Parameters
    ----------
    x : np.ndarray
        Decision vector, 1-D, length >= 2. Global optimum at the all-ones vector.

    Returns
    -------
    float
        ``sum(100*(x_{i+1} - x_i**2)**2 + (1 - x_i)**2)``; minimum ``0.0`` at
        ``x = 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import rosenbrock
    >>> rosenbrock(np.ones(10))
    0.0
    >>> rosenbrock(np.zeros(10))
    9.0
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function: highly multimodal with a regular lattice of minima.

    Parameters
    ----------
    x : np.ndarray
        Decision vector, 1-D, any length. Global optimum at the zero vector.

    Returns
    -------
    float
        ``10*d + sum(x_i**2 - 10*cos(2*pi*x_i))``; minimum ``0.0`` at ``x = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import rastrigin
    >>> rastrigin(np.zeros(10))
    0.0
    """
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley function: a multimodal landscape with one global funnel.

    Parameters
    ----------
    x : np.ndarray
        Decision vector, 1-D, any length. Global optimum at the zero vector.

    Returns
    -------
    float
        The standard Ackley value; minimum ``0.0`` at ``x = 0`` (with a ~1e-16
        floating-point residue from ``exp(1)`` vs ``e``).

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import ackley
    >>> bool(abs(ackley(np.zeros(10))) < 1e-9)
    True
    """
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20.0 + np.e)


def griewank(x: np.ndarray) -> float:
    """Griewank function: a product term couples the dimensions.

    Parameters
    ----------
    x : np.ndarray
        Decision vector, 1-D, any length. Global optimum at the zero vector.

    Returns
    -------
    float
        ``1 + sum(x_i**2)/4000 - prod(cos(x_i/sqrt(i)))``; minimum ``0.0`` at
        ``x = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import griewank
    >>> griewank(np.zeros(10))
    0.0
    """
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1)
    return float(1.0 + np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))))


def schwefel(x: np.ndarray) -> float:
    """Schwefel function (2.26): deceptive, global basin far from second-best.

    Parameters
    ----------
    x : np.ndarray
        Decision vector, 1-D, any length. Global optimum at ``x_i ~= 420.9687``.

    Returns
    -------
    float
        ``418.9829*d - sum(x_i*sin(sqrt(|x_i|)))``; minimum ~``0.0`` at the
        per-dimension optimum (a ~1.3e-4 residual at 10D from the rounded
        constants; see module Notes).

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import schwefel
    >>> round(schwefel(np.full(10, 420.9687)), 3)
    0.0
    """
    x = np.asarray(x, dtype=float)
    return float(_SCHWEFEL_OFFSET * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


@dataclass(frozen=True, eq=False)
class SOProblem:
    """Metadata bundle pairing a single-objective function with its known optimum.

    ``eq=False`` is set deliberately: a generated ``__eq__`` would compare the
    ``x_star`` numpy arrays element-wise and raise on the ambiguous truth value.
    Instances are looked up by name from ``SO_PROBLEMS``, never compared.

    Parameters
    ----------
    name : str
        Lowercase registry key (e.g. ``"sphere"``).
    func : Callable[[np.ndarray], float]
        The objective; takes a 1-D decision vector of length `dim`, returns a
        scalar to minimise.
    x_star : np.ndarray
        Known global optimum location, shape ``(dim,)``. Set read-only on
        construction so downstream ``||x - x*||`` computations cannot mutate it.
    f_star : float
        Known global optimum value (0.0 for all six; Schwefel within ~1.3e-4).
    bounds : tuple[float, float]
        Symmetric per-dimension search box ``(low, high)`` applied to every
        variable; directly consumable by ctrl-freak's SBX/PM operators.
    dim : int
        Problem dimensionality (10 for every problem in this suite).
    epsilon : float
        Success threshold: a run succeeds when ``f_found - f_star < epsilon``.

    Raises
    ------
    ValueError
        If `dim` is not positive, `x_star` does not have shape ``(dim,)``, the
        bounds are not strictly increasing, or `epsilon` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from benchmarks.problems.single_objective import SO_PROBLEMS
    >>> p = SO_PROBLEMS["sphere"]
    >>> p.name
    'sphere'
    >>> p.dim
    10
    >>> p.bounds
    (-5.12, 5.12)
    >>> p.func(p.x_star) == p.f_star
    True
    """

    name: str
    func: Callable[[np.ndarray], float]
    x_star: np.ndarray
    f_star: float
    bounds: tuple[float, float]
    dim: int
    epsilon: float

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.x_star.shape != (self.dim,):
            raise ValueError(f"x_star must have shape ({self.dim},), got {self.x_star.shape}")
        low, high = self.bounds
        if low >= high:
            raise ValueError(f"bounds must be (low, high) with low < high, got {self.bounds}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        self.x_star.flags.writeable = False


SO_PROBLEMS: dict[str, SOProblem] = {
    "sphere": SOProblem(
        name="sphere",
        func=sphere,
        x_star=np.zeros(DIM),
        f_star=0.0,
        bounds=(-5.12, 5.12),
        dim=DIM,
        epsilon=1e-6,
    ),
    "rosenbrock": SOProblem(
        name="rosenbrock",
        func=rosenbrock,
        x_star=np.ones(DIM),
        f_star=0.0,
        bounds=(-2.048, 2.048),
        dim=DIM,
        epsilon=1e-3,
    ),
    "rastrigin": SOProblem(
        name="rastrigin",
        func=rastrigin,
        x_star=np.zeros(DIM),
        f_star=0.0,
        bounds=(-5.12, 5.12),
        dim=DIM,
        epsilon=1e-2,
    ),
    "ackley": SOProblem(
        name="ackley",
        func=ackley,
        x_star=np.zeros(DIM),
        f_star=0.0,
        bounds=(-32.768, 32.768),
        dim=DIM,
        epsilon=1e-2,
    ),
    "griewank": SOProblem(
        name="griewank",
        func=griewank,
        x_star=np.zeros(DIM),
        f_star=0.0,
        bounds=(-600.0, 600.0),
        dim=DIM,
        epsilon=1e-2,
    ),
    "schwefel": SOProblem(
        name="schwefel",
        func=schwefel,
        x_star=np.full(DIM, _SCHWEFEL_OPTIMUM),
        f_star=0.0,
        bounds=(-500.0, 500.0),
        dim=DIM,
        epsilon=1e-2,
    ),
}
