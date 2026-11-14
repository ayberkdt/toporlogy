"""Method of moving asymptotes for ``min f0`` s.t. ``g_i ≤ 0`` (i = 1…m, any m ≥ 1), ``0 ≤ x ≤ 1``.

No new derivation: the two validated pieces of the repository are reused unchanged,

* :func:`toporlogy.optim._mma_core.mma_asymptote` — asymptote update (k ≤ 2: x ± asyinit; then ×asyincr /
  ×asydecr on (non-)oscillation, clipped to [asymin, asymax] from x);
* :func:`toporlogy.optim._mma_core.mma_multi` — convex subproblem with m constraints, dual maximised by
  L-BFGS-B (artificial variables y_i with c = 1000, d = 1; albefa = 0.1, raa0 = 1e-5).

Defaults and why
----------------
``move = 0.2``
    the move limit of the legacy 2-D driver's MMA path (the oracle in tests/toporlogy) and of the OC
    update of the legacy voxel driver. Stress-constrained problems typically use 0.05-0.1 (and shrink it as 2/β
    at high projection sharpness); ``move`` is a plain attribute, so a schedule may change it during a run.
``asyinit, asydecr, asyincr = 0.5, 0.7, 1.2``
    the values every MMA call in the repository passes (``topopt``, ``topo3d.optimize_stress``, ``stress3d``),
    which are the coefficients of Svanberg's 2007 code as ported in mmapy (see the ``svanberg_1987`` reference note).
``normalise = "first"``
    the objective is divided by |f0| of the first step after each :meth:`MMA.reset` (``topopt`` divides by c0 of the
    first iteration; here the scale restarts at every β step, when the problem changes). ``"current"`` divides by
    max(|f0|, 1e-9) at every step (the convention of ``topo3d.optimize_stress``); ``None`` uses f0 as given.
    MMA is not scale-invariant (raa0 and the y penalty c are absolute), so f0 = O(1) keeps the defaults meaningful.

Constraints are passed through as the study gives them (``Constraint.g`` is already a relative violation for a
volume bound), so their scale is the caller's choice.

Limitation (plain MMA, not GCMMA; measured 13 Nov 2026, ``tests/toporlogy/test_mma.py``): the method is not globally
convergent. A variable whose optimum is interior and touched by no active constraint (its objective derivative → 0
there) can end in a 2-cycle around the optimum: on ``min (x − 0.2)²`` amplitude 0.0043, with the asymptotes held at
asymin = 0.01 from x. Compliance-type problems (∂c/∂x < 0 everywhere, volume constraint active) do not show this.

:meth:`MMA.reset` forgets the asymptotes, the two previous iterates, the iteration counter and the objective scale;
:class:`toporlogy.study.Study` calls it at every β step (the old drivers do the same, ``stress3d``: "the MMA
asymptotes are reset at every change").
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from toporlogy.optim._mma_core import mma_asymptote, mma_multi
from toporlogy.references import cite
from toporlogy.optim.base import Optimizer

__all__ = ["MMA"]

_NORMALISE = ("first", "current", None)


@cite("svanberg_1987", note="asymptotes (topology.mma_asymptote) and m-constraint subproblem "
                            "(stress_topo.mma_multi, dual by L-BFGS-B); coefficients of Svanberg's 2007 code via mmapy")
class MMA(Optimizer):
    """MMA with any number of constraints.

    Parameters
    ----------
    move : float
        Subproblem move limit (fraction of xmax − xmin = 1), 0 < move ≤ 1.
    asyinit, asydecr, asyincr, asymin, asymax : float
        Asymptote coefficients (:func:`toporlogy.optim._mma_core.mma_asymptote`).
    albefa, raa0, c, d : float
        Subproblem constants (:func:`toporlogy.optim._mma_core.mma_multi`).
    normalise : {"first", "current", None}
        Objective scaling (module notes).

    Attributes
    ----------
    k : int
        Steps since the last reset.
    last_lambda : numpy.ndarray
        Subproblem multipliers of the last step (m,).
    """

    name = "MMA"
    max_constraints = None

    def __init__(self, move: float = 0.2, asyinit: float = 0.5, asydecr: float = 0.7, asyincr: float = 1.2,
                 asymin: float = 0.01, asymax: float = 10.0, albefa: float = 0.1, raa0: float = 1e-5,
                 c: float = 1000.0, d: float = 1.0, normalise: str | None = "first") -> None:
        if not 0 < move <= 1:
            raise ValueError("0 < move ≤ 1 is required")
        if normalise not in _NORMALISE:
            raise ValueError(f"normalise: one of {_NORMALISE}")
        if not (0 < asydecr < 1 < asyincr and 0 < asymin < asyinit < asymax):
            raise ValueError("need 0 < asydecr < 1 < asyincr and 0 < asymin < asyinit < asymax")
        self.move = float(move)
        self.asyinit, self.asydecr, self.asyincr = float(asyinit), float(asydecr), float(asyincr)
        self.asymin, self.asymax = float(asymin), float(asymax)
        self.albefa, self.raa0, self.c, self.d = float(albefa), float(raa0), float(c), float(d)
        self.normalise = normalise
        self.n_resets = -1                                  # counts reset() calls after construction
        self.reset()

    def reset(self) -> None:
        """Forget asymptotes, previous iterates, the step counter and the objective scale."""
        self.k = 0
        self.xold1: np.ndarray | None = None
        self.xold2: np.ndarray | None = None
        self.low: np.ndarray | None = None
        self.upp: np.ndarray | None = None
        self.f0_scale: float | None = None
        self.last_lambda = np.zeros(0)
        self.n_resets += 1

    def _scale(self, f0: float) -> float:
        if self.normalise is None:
            return 1.0
        if self.normalise == "current":
            return 1.0 / max(abs(f0), 1e-9)
        if self.f0_scale is None:
            self.f0_scale = 1.0 / abs(f0) if abs(f0) > 1e-300 else 1.0
        return self.f0_scale

    def step(self, x: np.ndarray, f0: float, df0: np.ndarray, g: np.ndarray, dg: np.ndarray,
             constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        x = np.asarray(x, float)
        n = x.size
        df0 = np.asarray(df0, float).reshape(n)
        g = np.atleast_1d(np.asarray(g, float)).reshape(-1)
        dg = np.atleast_2d(np.asarray(dg, float))
        if g.size < 1:
            raise ValueError("MMA needs at least one constraint (pass e.g. a volume bound)")
        if dg.shape != (g.size, n):
            raise ValueError(f"dg has shape {dg.shape}, expected ({g.size}, {n})")
        if self.xold1 is not None and self.xold1.shape != x.shape:
            raise ValueError("design size changed without reset()")
        self.k += 1
        if self.xold1 is None or self.xold2 is None:
            self.xold1, self.xold2 = x.copy(), x.copy()
        xmin, xmax = np.zeros(n), np.ones(n)
        self.low, self.upp = mma_asymptote(self.k, x, self.xold1, self.xold2, self.low, self.upp, xmin, xmax,
                                           self.asyinit, self.asydecr, self.asyincr, self.asymin, self.asymax)
        xnew, lam = mma_multi(x, xmin, xmax, self.low, self.upp, df0 * self._scale(f0), g, dg, move=self.move,
                              albefa=self.albefa, raa0=self.raa0, c=self.c, d=self.d)
        self.xold2, self.xold1 = self.xold1, x.copy()
        self.last_lambda = np.asarray(lam, float)
        return np.clip(xnew, 0.0, 1.0)
