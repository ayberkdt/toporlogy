"""MMA building blocks (Svanberg 1987): asymptote update and the m-constraint subproblem solved through its dual.

Copied unchanged from the repository's validated 2-D drivers so the library has no dependency on them; the 2-D MMA
oracle test (tests/toporlogy/test_mma.py) reproduces the original driver to 2e-9.
"""
from __future__ import annotations

import numpy as np
import scipy.optimize as so

from toporlogy.references import cite

__all__ = ["mma_asymptote", "mma_multi"]


@cite("svanberg_1987", note="§3; coefficients as in the mmapy port of Svanberg's 2007 MATLAB code")
def mma_asymptote(k: int, x: np.ndarray, xold1: np.ndarray, xold2: np.ndarray, low: np.ndarray | None,
                  upp: np.ndarray | None, xmin: np.ndarray, xmax: np.ndarray, asyinit: float = 0.5,
                  asydecr: float = 0.7, asyincr: float = 1.2, asymin: float = 0.01,
                  asymax: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Moving asymptotes of MMA.

    Parameters
    ----------
    k : int
        Iteration number (1-based).
    x, xold1, xold2 : numpy.ndarray
        Current design and the two previous designs.
    low, upp : numpy.ndarray or None
        Previous asymptotes (ignored for k ≤ 2).
    xmin, xmax : numpy.ndarray
        Variable bounds.
    asyinit, asydecr, asyincr, asymin, asymax : float
        Asymptote coefficients.

    Returns
    -------
    low, upp : numpy.ndarray
        New lower and upper asymptotes.

    Notes
    -----
    k ≤ 2: l = x − asyinit (xmax − xmin), u = x + asyinit (xmax − xmin).
    k > 2: γ = asyincr if (x − x₁)(x₁ − x₂) > 0, asydecr if < 0, 1 if = 0;
    l = x − γ (x₁ − l), u = x + γ (u − x₁), clipped to [asymin, asymax]·(xmax − xmin) from x.
    The coefficients were read from A. Deetman's Python port (GCMMA-MMA-Python, ``mmapy/mma.py``)
    of Svanberg's 2007 MATLAB code; Svanberg's 2007 notes could not be read.
    """
    dx = xmax - xmin
    if k <= 2:
        return x - asyinit * dx, x + asyinit * dx
    if low is None or upp is None:
        raise ValueError("mma_asymptote: previous asymptotes low/upp are required for k > 2")
    z = (x - xold1) * (xold1 - xold2)
    g = np.ones_like(x)
    g[z > 0] = asyincr
    g[z < 0] = asydecr
    low = x - g * (xold1 - low)
    upp = x + g * (upp - xold1)
    low = np.minimum(np.maximum(low, x - asymax * dx), x - asymin * dx)
    upp = np.maximum(np.minimum(upp, x + asymax * dx), x + asymin * dx)
    return low, upp


@cite("svanberg_1987", note="MMA approximations as in topology.mma_subproblem; dual solved with L-BFGS-B")
def mma_multi(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, low: np.ndarray, upp: np.ndarray,
              df0: np.ndarray, g: np.ndarray, dg: np.ndarray, move: float = 0.1, albefa: float = 0.1,
              raa0: float = 1e-5, c: float = 1000.0, d: float = 1.0,
              info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MMA subproblem with m constraints g_i ≤ 0 (rows of ``dg``); returns the new design and the multipliers.

    ``info`` (optional dict) receives the dual solve's outcome: ``success``, ``message``, ``nit`` (L-BFGS-B) and
    ``residual`` = max |projected gradient| of the dual at the returned multipliers (the KKT residual of the box-
    constrained dual; 0 at the exact dual optimum). L-BFGS-B's ``success`` alone is not a convergence test here:
    measured on the two MBB runs of test_mma (249 solves, 17 Nov 2026), the 2 ``ABNORMAL_TERMINATION_IN_LNSRCH``
    stops had residual ≤ 1.14e-8, the 247 successful solves up to 4.06e-8."""
    dx = np.maximum(xmax - xmin, 1e-5)
    alpha = np.maximum(np.maximum(low + albefa * (x - low), x - move * (xmax - xmin)), xmin)
    beta = np.minimum(np.minimum(upp - albefa * (upp - x), x + move * (xmax - xmin)), xmax)
    ux, xl = upp - x, x - low

    def pq(df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_, q_ = np.maximum(df, 0.0), np.maximum(-df, 0.0)
        e = 0.001 * (p_ + q_) + raa0 / dx
        return (p_ + e) * ux ** 2, (q_ + e) * xl ** 2

    p0, q0 = pq(df0)
    pi, qi = pq(np.atleast_2d(dg))
    b = np.sum(pi / ux + qi / xl, axis=1) - np.atleast_1d(g)

    def x_of(lam: np.ndarray) -> np.ndarray:
        P, Q = p0 + lam @ pi, q0 + lam @ qi
        sP, sQ = np.sqrt(P), np.sqrt(Q)
        return np.clip((low * sP + upp * sQ) / (sP + sQ), alpha, beta)

    def neg_dual(lam: np.ndarray) -> tuple[float, np.ndarray]:
        xx = x_of(lam)
        gi = np.sum(pi / (upp - xx) + qi / (xx - low), axis=1) - b
        y = np.maximum(0.0, (lam - c) / d)
        W = float(np.sum(p0 / (upp - xx) + q0 / (xx - low)) + lam @ (gi - y) + np.sum(c * y + 0.5 * d * y * y))
        return -W, -(gi - y)

    m = pi.shape[0]
    res = so.minimize(neg_dual, np.zeros(m), jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * m,
                      options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12})
    if info is not None:
        gl = -(np.asarray(res.jac, float).reshape(-1))         # ascent direction of the dual (= g_i(x(λ)) − y_i)
        pg = np.where(res.x > 0.0, gl, np.maximum(gl, 0.0))     # projected on λ ≥ 0
        info.update(success=bool(res.success), message=str(res.message), nit=int(getattr(res, "nit", -1)),
                    residual=float(np.max(np.abs(pg))) if pg.size else 0.0)
    return x_of(res.x), res.x
