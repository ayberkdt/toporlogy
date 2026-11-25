"""Optimizer interface (docs/mimari_plan.md §2).

Problem form: ``min f0(x)`` s.t. ``g_i(x) ≤ 0`` (i = 1…m), ``0 ≤ x ≤ 1``. The study passes plain NumPy arrays.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["Optimizer"]


class Optimizer:
    """Base class.

    ``step(x, f0, df0, g, dg, constraint_fn)`` returns the next design. ``x`` (n,), ``df0`` (n,), ``g`` (m,),
    ``dg`` (m, n). ``constraint_fn(x_new) -> g(x_new)`` (m,) evaluates the constraints without gradients (OC uses it
    in its bisection, SiMPL in its shift search; MMA and GCMMA ignore it). ``reset()`` is called when the problem
    changes discontinuously (a β step).
    """

    name = "optimizer"
    max_constraints: int | None = None
    needs_constraint_fn = False             # True: step() evaluates g(x_new) itself, so constraints must not need U
    # True: step() tests candidates with ``value_fn(x) -> (f0, g)`` (objective and constraints as passed to step, no
    # gradients; one state solve per call). Study.run sets ``value_fn`` before the loop and clears it afterwards.
    needs_value_fn = False
    value_fn: Callable[[np.ndarray], tuple[float, np.ndarray]] | None = None

    def step(self, x: np.ndarray, f0: float, df0: np.ndarray, g: np.ndarray, dg: np.ndarray,
             constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        """Forget history (asymptotes, previous iterates)."""

    def kkt_multipliers(self) -> np.ndarray | None:
        """Lagrange multipliers of the last step for ``objective_scale(f0) * f0 + mu^T g`` (g as passed to ``step``),
        or None when the optimizer does not provide them. Used for ``Result.kkt_residual``."""
        return None

    def objective_scale(self, f0: float) -> float:
        """Factor the optimizer applies to the objective in the problem its multipliers refer to."""
        return 1.0

    def restart(self) -> None:
        """Start of a new run (Study.run): forget everything a previous run left. Default: ``reset()``."""
        self.reset()
