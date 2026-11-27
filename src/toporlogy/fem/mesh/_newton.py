"""The Newton iteration on free nodal unknowns shared by the time-dependent mesh solvers (cure_slab, nonlinear_heat).

Each caller keeps its own residual, its stop scale and its rounding-noise model; this loop only fixes the iteration:
max-norm of the free residual, stop when ``rmax <= rtol * scale + noise``, refusal of a non-finite residual, refusal
after ``max_iter`` corrections, and the correction from a sparse LU of the free block of the Jacobian."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.fem.mesh.history import StepRejected

Payload = TypeVar("Payload")


@dataclass
class NewtonResult(Generic[Payload]):
    """Converged iterate ``x``, its full residual ``R``, the caller's ``payload`` of that last evaluation, the number
    of corrections made and the free-residual max-norm of every evaluation."""

    x: np.ndarray
    R: np.ndarray
    payload: Payload
    iterations: int
    history: list[float]


def newton_free(x: np.ndarray, free: np.ndarray,
                evaluate: Callable[[np.ndarray], tuple[np.ndarray, float, float, Payload]],
                jacobian: Callable[[np.ndarray], sp.csr_matrix], rtol: float, max_iter: int) -> NewtonResult[Payload]:
    """Newton on ``x[free]``. ``evaluate(x)`` returns (full residual, stop scale, rounding noise, payload);
    ``jacobian(x)`` the full Jacobian. ``x`` is not modified; raises :class:`StepRejected` without converging."""
    history: list[float] = []
    for it in range(max_iter + 1):
        R, scale, noise, payload = evaluate(x)
        Rf = R[free]
        rmax = float(np.abs(Rf).max()) if len(free) else 0.0
        history.append(rmax)
        if not math.isfinite(rmax):
            raise StepRejected("non-finite residual")
        if rmax <= rtol * scale + noise:
            return NewtonResult(x, R, payload, it, history)
        if it == max_iter:
            raise StepRejected(f"no convergence in {max_iter} Newton iterations (residual {rmax:.3e}, "
                               f"target {rtol * scale + noise:.3e})")
        J = jacobian(x)
        x = x.copy()
        x[free] += spla.splu(J[free][:, free].tocsc()).solve(-Rf)
    raise AssertionError("unreachable")  # pragma: no cover
