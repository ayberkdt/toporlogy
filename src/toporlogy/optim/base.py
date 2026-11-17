"""Optimizer interface (docs/mimari_plan.md §3.1).

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
    in its bisection; MMA ignores it). ``reset()`` is called when the problem changes discontinuously (a β step).
    """

    name = "optimizer"
    max_constraints: int | None = None

    def step(self, x: np.ndarray, f0: float, df0: np.ndarray, g: np.ndarray, dg: np.ndarray,
             constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        """Forget history (asymptotes, previous iterates)."""

    def restart(self) -> None:
        """Start of a new run (Study.run): forget everything a previous run left. Default: ``reset()``."""
        self.reset()
