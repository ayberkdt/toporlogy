"""GCMMA: the globally convergent version of the method of moving asymptotes (Svanberg 2002, 2007).

Plain MMA (:class:`~toporlogy.optimizers.mma.MMA`) takes whatever the subproblem returns. GCMMA makes every
approximation *conservative* before it accepts a step: each outer iteration solves the subproblem, evaluates the true
objective and constraints at the candidate (no gradients), and, while a true value exceeds its approximation there,
raises that function's curvature parameter ρ_i and solves again (inner iterations). An accepted candidate satisfies

    f_i(x̂) ≤ f̃_i(x̂)   for the objective and every constraint,

so a feasible iterate stays feasible and the objective cannot increase from one feasible outer iterate to the next,
both up to the acceptance tolerance ``conservative_tol`` (checked in ``tests/toporlogy/test_gcmma.py``; measured
there: at an interior optimum the iterates settle into a cycle whose amplitude in f is that tolerance, 1e-9 of the
scaled objective, against 4e-3 in x for plain MMA). This removes the documented failure of plain MMA in this repository:
the 2-cycle around an interior optimum (``mma.py`` module notes).

Approximations (i = 0 objective, i ≥ 1 constraints; l, u the asymptotes of :func:`mma_asymptote`):

    f̃_i(x̂) = f_i(x) + Σ_j [ p_ij (1/(u_j − x̂_j) − 1/(u_j − x_j)) + q_ij (1/(x̂_j − l_j) − 1/(x_j − l_j)) ],
    p_ij = (u_j − x_j)² (1.001 ∂f_i⁺ + 0.001 ∂f_i⁻ + ρ_i / Δ_j),
    q_ij = (x_j − l_j)² (0.001 ∂f_i⁺ + 1.001 ∂f_i⁻ + ρ_i / Δ_j),          Δ_j = xmax_j − xmin_j = 1.

Start of an outer iteration: ``ρ_i = max(rho_min, (rho_init / n) Σ_j |∂f_i/∂x_j| Δ_j)``. Inner update for a function
that is not conservative: ``ρ_i ← min(1.1 (ρ_i + δ_i), 10 ρ_i)`` with ``δ_i = (f_i(x̂) − f̃_i(x̂)) / d(x̂)`` and
``d(x̂) = Σ_j (u_j − l_j)(x̂_j − x_j)² / ((u_j − x̂_j)(x̂_j − l_j) Δ_j)``.

Sources. These are the formulas of Svanberg's 2007 note as I know them and as the public ports implement them
(inspected: A. Deetman's GCMMA-MMA-Python, GPL — read for the constants only, no code taken). The note itself could
not be opened in this work (``docs/literatur_2026-11-21.md``). Independent of the source, the property that matters
— conservativeness of every accepted step — is enforced by the inner loop and tested directly.

Differences from Svanberg's code, on purpose: the subproblem is solved through its dual by L-BFGS-B as in
:func:`toporlogy.optimizers._mma_core.mma_multi` (not by the primal-dual interior-point method), and a ``move``
limit is kept as in the rest of this library (Svanberg's GCMMA has none; ``move = 1`` reproduces that).

Cost: each inner iteration is one evaluation without gradients (one state solve). ``n_inner_total`` counts them;
compare methods by total evaluations, not by outer iterations.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.optimize as so

from toporlogy.optimizers._mma_core import mma_asymptote
from toporlogy.optimizers.base import Optimizer
from toporlogy.references import cite

__all__ = ["GCMMA", "conservative_subproblem"]


def conservative_subproblem(x: np.ndarray, low: np.ndarray, upp: np.ndarray, df: np.ndarray, f: np.ndarray,
                            rho: np.ndarray, move: float = 1.0, albefa: float = 0.1, c: float = 1000.0,
                            d: float = 1.0) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], dict]:
    """One GCMMA subproblem on the box [0, 1]ⁿ.

    ``df`` (m + 1, n) gradients of objective (row 0) and constraints, ``f`` (m + 1,) their values, ``rho`` (m + 1,)
    curvature parameters. Returns the minimiser, the multipliers (m,), a function ``approx(x̂) -> f̃(x̂)`` (m + 1,)
    and the outcome of the dual solve (as :func:`~toporlogy.optimizers._mma_core.mma_multi`'s ``info``).
    """
    ux, xl = upp - x, x - low
    dfp, dfm = np.maximum(df, 0.0), np.maximum(-df, 0.0)
    P = (1.001 * dfp + 0.001 * dfm + rho[:, None]) * ux ** 2
    Q = (0.001 * dfp + 1.001 * dfm + rho[:, None]) * xl ** 2
    r = f - np.sum(P / ux + Q / xl, axis=1)                    # f̃_i(x̂) = r_i + Σ P/(u − x̂) + Q/(x̂ − l)
    alpha = np.maximum(np.maximum(low + albefa * xl, x - move), 0.0)
    beta = np.minimum(np.minimum(upp - albefa * ux, x + move), 1.0)
    p0, q0, pi, qi = P[0], Q[0], P[1:], Q[1:]
    m = pi.shape[0]

    def approx(xx: np.ndarray) -> np.ndarray:
        return r + np.sum(P / (upp - xx) + Q / (xx - low), axis=1)

    def x_of(lam: np.ndarray) -> np.ndarray:
        sP, sQ = np.sqrt(p0 + lam @ pi), np.sqrt(q0 + lam @ qi)
        return np.clip((low * sP + upp * sQ) / (sP + sQ), alpha, beta)

    def neg_dual(lam: np.ndarray) -> tuple[float, np.ndarray]:
        xx = x_of(lam)
        gi = r[1:] + np.sum(pi / (upp - xx) + qi / (xx - low), axis=1)
        y = np.maximum(0.0, (lam - c) / d)
        W = float(np.sum(p0 / (upp - xx) + q0 / (xx - low)) + lam @ (gi - y) + np.sum(c * y + 0.5 * d * y * y))
        return -W, -(gi - y)

    res = so.minimize(neg_dual, np.zeros(m), jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * m,
                      options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12})
    gl = -(np.asarray(res.jac, float).reshape(-1))
    pg = np.where(res.x > 0.0, gl, np.maximum(gl, 0.0))
    info = {"success": bool(res.success), "message": str(res.message), "nit": int(getattr(res, "nit", -1)),
            "residual": float(np.max(np.abs(pg))) if pg.size else 0.0}
    return x_of(res.x), np.asarray(res.x, float), approx, info


@cite("svanberg_2002", "svanberg_2007", note="conservative approximations and inner iterations; sources not read "
                                             "(module notes); dual solved by L-BFGS-B")
class GCMMA(Optimizer):
    """Globally convergent MMA (module notes). Needs ``value_fn`` (set by :class:`~toporlogy.optimization.study.Study`):
    ``value_fn(x) -> (f0, g)`` evaluates objective and constraints without gradients.

    Parameters
    ----------
    move : float
        Move limit (1 = none, Svanberg's GCMMA).
    max_inner : int
        Inner iterations per step at most; the last candidate is taken if none is conservative (counted in
        ``n_not_conservative``).
    rho_init, rho_min : float
        Start value factor and floor of the curvature parameters.
    conservative_tol : float
        Accept when ``f_i(x̂) ≤ f̃_i(x̂) + conservative_tol · (1 + |f_i(x̂)|)`` (round-off of the state solve).
    asyinit, asydecr, asyincr, asymin, asymax, albefa, c, d, normalise :
        As for :class:`~toporlogy.optimizers.mma.MMA`.

    Attributes
    ----------
    n_inner_total : int
        Evaluations without gradients made inside the steps of this run (one per candidate tested).
    last_inner : int
        Inner iterations of the last step.
    """

    name = "GCMMA"
    max_constraints = None
    needs_value_fn = True

    def __init__(self, move: float = 1.0, max_inner: int = 15, rho_init: float = 0.1, rho_min: float = 1e-6,
                 conservative_tol: float = 1e-9, asyinit: float = 0.5, asydecr: float = 0.7, asyincr: float = 1.2,
                 asymin: float = 0.01, asymax: float = 10.0, albefa: float = 0.1, c: float = 1000.0, d: float = 1.0,
                 normalise: str | None = "first") -> None:
        if not 0 < move <= 1:
            raise ValueError("0 < move ≤ 1 is required")
        if normalise not in ("first", None):
            raise ValueError("normalise: 'first' or None (a scale that changes inside a step would break the "
                             "conservativeness test)")
        if max_inner < 0 or not (rho_init > 0 and rho_min > 0):
            raise ValueError("need max_inner >= 0, rho_init > 0 and rho_min > 0")
        if not (0 < asydecr < 1 < asyincr and 0 < asymin < asyinit < asymax):
            raise ValueError("need 0 < asydecr < 1 < asyincr and 0 < asymin < asyinit < asymax")
        self.move, self.max_inner = float(move), int(max_inner)
        self.rho_init, self.rho_min, self.conservative_tol = float(rho_init), float(rho_min), float(conservative_tol)
        self.asyinit, self.asydecr, self.asyincr = float(asyinit), float(asydecr), float(asyincr)
        self.asymin, self.asymax = float(asymin), float(asymax)
        self.albefa, self.c, self.d = float(albefa), float(c), float(d)
        self.normalise = normalise
        self.n_inner_total = 0
        self.n_not_conservative = 0
        self.last_inner = 0
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.xold1: np.ndarray | None = None
        self.xold2: np.ndarray | None = None
        self.low: np.ndarray | None = None
        self.upp: np.ndarray | None = None
        self.f0_scale: float | None = None
        self.last_lambda: np.ndarray = np.zeros(0)

    def restart(self) -> None:
        self.reset()
        self.n_inner_total = 0
        self.n_not_conservative = 0

    def kkt_multipliers(self) -> np.ndarray | None:
        return self.last_lambda.copy() if self.last_lambda.size else None

    def objective_scale(self, f0: float) -> float:
        return float(self.f0_scale) if self.f0_scale is not None else 1.0

    def step(self, x: np.ndarray, f0: float, df0: np.ndarray, g: np.ndarray, dg: np.ndarray,
             constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        x = np.asarray(x, float)
        n = x.size
        g = np.atleast_1d(np.asarray(g, float)).reshape(-1)
        dg = np.atleast_2d(np.asarray(dg, float))
        if g.size < 1:
            raise ValueError("GCMMA needs at least one constraint (pass e.g. a volume bound)")
        if dg.shape != (g.size, n):
            raise ValueError(f"dg has shape {dg.shape}, expected ({g.size}, {n})")
        if self.value_fn is None and self.max_inner > 0:
            raise ValueError("GCMMA needs value_fn (Study sets it; standalone: opt.value_fn = lambda x: (f0, g))")
        if self.normalise == "first" and self.f0_scale is None:
            self.f0_scale = 1.0 / abs(f0) if abs(f0) > 1e-300 else 1.0
        s = self.f0_scale if self.f0_scale is not None else 1.0
        self.k += 1
        if self.xold1 is None or self.xold2 is None:
            self.xold1, self.xold2 = x.copy(), x.copy()
        xmin, xmax = np.zeros(n), np.ones(n)
        self.low, self.upp = mma_asymptote(self.k, x, self.xold1, self.xold2, self.low, self.upp, xmin, xmax,
                                           self.asyinit, self.asydecr, self.asyincr, self.asymin, self.asymax)
        low, upp = self.low, self.upp
        df = np.vstack([np.asarray(df0, float).reshape(1, n) * s, dg])
        f = np.concatenate([[f0 * s], g])
        rho = np.maximum(self.rho_min, self.rho_init / n * np.abs(df).sum(axis=1))
        inner = 0
        while True:
            xnew, lam, approx, _ = conservative_subproblem(x, low, upp, df, f, rho, self.move, self.albefa,
                                                           self.c, self.d)
            xnew = np.clip(xnew, 0.0, 1.0)
            if self.max_inner == 0 or self.value_fn is None:
                break
            f0n, gn = self.value_fn(xnew)
            fn = np.concatenate([[float(f0n) * s], np.atleast_1d(np.asarray(gn, float)).reshape(-1)])
            fa = approx(xnew)
            excess = fn - fa
            bad = excess > self.conservative_tol * (1.0 + np.abs(fn))
            if not bad.any():
                break
            if inner >= self.max_inner:
                self.n_not_conservative += 1
                break
            inner += 1
            dd = float(np.sum((upp - low) * (xnew - x) ** 2 / ((upp - xnew) * (xnew - low))))
            delta = excess / max(dd, 1e-300)
            rho = np.where(bad, np.minimum(1.1 * (rho + delta), 10.0 * rho), rho)
        self.last_inner = inner
        self.n_inner_total += inner + (1 if (self.max_inner > 0 and self.value_fn is not None) else 0)
        self.xold2, self.xold1 = self.xold1, x.copy()
        self.last_lambda = lam
        return xnew
