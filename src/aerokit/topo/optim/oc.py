"""Optimality-criteria update for one volume-type constraint (top88 lines 70-80; topo3d's OC loop).

``x_new = clip(x · sqrt(−∂f/∂x / (λ ∂g/∂x)), x ± move, [0, 1])`` with λ found by bisection so that the constraint,
evaluated on ``x_new`` through the *nonlinear* chain (``constraint_fn``), is just satisfied. The bisection is the
linear midpoint rule of the legacy voxel driver (regression oracle): λ ∈ [l1, l2] = [0, 1e9], stop when
(l2 − l1)/(l1 + l2) < ``bisection_tol`` (topo3d 1e-4, top88 1e-3), at most ``max_bisect`` halvings.

``dv_scale`` multiplies ∂g/∂x before the ratio. It changes only the λ value the bisection works with (and so the
interval [0, 1e9] in which it searches); the legacy driver's constraint is Σx̄ ≤ V*·n, ours is mean(x̄) ≤ V*, so
``dv_scale = n_active`` reproduces its bisection sequence exactly (regression test).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from aerokit.references import cite
from aerokit.topo.optim.base import Optimizer

__all__ = ["OC"]


@cite("andreassen_2011", note="top88 OC update, lines 70-80")
class OC(Optimizer):
    """Optimality criteria with move limit; exactly one constraint (volume-like, ∂g/∂x ≥ 0)."""

    name = "OC"
    max_constraints = 1

    def __init__(self, move: float = 0.2, l1: float = 0.0, l2: float = 1e9, bisection_tol: float = 1e-4,
                 max_bisect: int = 80, dv_scale: float = 1.0) -> None:
        if not 0 < move <= 1:
            raise ValueError("0 < move ≤ 1 is required")
        self.move, self.l1, self.l2 = float(move), float(l1), float(l2)
        self.bisection_tol, self.max_bisect, self.dv_scale = float(bisection_tol), int(max_bisect), float(dv_scale)
        self.last_lambda = float("nan")

    def step(self, x: np.ndarray, f0: float, df0: np.ndarray, g: np.ndarray, dg: np.ndarray,
             constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        dg = np.atleast_2d(dg)
        if dg.shape[0] != 1:
            raise ValueError("OC handles exactly one constraint; use MMA for more")
        if constraint_fn is None:
            raise ValueError("OC needs constraint_fn (the bisection evaluates the constraint on x_new)")
        dv = dg[0] * self.dv_scale
        l1, l2 = self.l1, self.l2
        xl, xu = np.maximum(0.0, x - self.move), np.minimum(1.0, x + self.move)
        ratio = np.sqrt(np.maximum(-df0, 0.0) / np.maximum(dv, 1e-30))
        xnew = x
        lm = 0.5 * (l1 + l2)
        for _ in range(self.max_bisect):
            lm = 0.5 * (l1 + l2)
            xnew = np.clip(x * ratio / math.sqrt(lm), xl, xu)
            if float(np.asarray(constraint_fn(xnew)).reshape(-1)[0]) > 0:
                l1 = lm
            else:
                l2 = lm
            if (l2 - l1) / (l1 + l2 + 1e-30) < self.bisection_tol:
                break
        self.last_lambda = lm
        return xnew
