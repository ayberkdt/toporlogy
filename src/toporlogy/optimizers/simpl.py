"""SiMPL: sigmoidal mirror descent with a projected latent variable (Kim, Lazarov, Surowiec & Keith 2025).

A first-order method for ``min f(x)`` over ``0 ≤ x ≤ 1`` with one volume-type constraint. The design is carried by a
latent variable ψ with ``x = sigmoid(ψ)``, so the bounds hold by construction and no move limit is needed. One step:

    ψ½ = ψ − α ∇f(x)                       (mirror-descent step for the Fermi–Dirac entropy)
    ψ⁺ = ψ½ − c,   x⁺ = sigmoid(ψ⁺)          (uniform shift c ≥ 0: the smallest one with g(x⁺) ≤ 0)

Step size: ``α = 1/‖∇f‖∞`` at the first step, afterwards the generalised Barzilai–Borwein value
``α_BB = |⟨Δx, Δψ⟩ / ⟨Δx, Δ∇f⟩|``; the trial step is the geometric mean of α_BB and the previous accepted step, then
backtracking (halving) until one of

* ``"armijo"``:  f(x⁺) ≤ f(x) + c1 ⟨∇f, x⁺ − x⟩,
* ``"bregman"``: f(x⁺) ≤ f(x) + ⟨∇f, x⁺ − x⟩ + D(x⁺, x)/α, with the Bregman divergence of the entropy
  ``D(a, b) = Σ a ln(a/b) + (1 − a) ln((1 − a)/(1 − b))``,

holds; ``"none"`` takes the trial step. Each backtracking test is one evaluation without gradients.

Where this comes from, precisely. No paper text could be opened in this work (``docs/literatur_2026-11-21.md``).
The update, the initial step, the BB formula and the Armijo test were read from the authors' public code (MFEM
example 37 and the ``mmto-simpl`` repository, summarised by a research pass); the geometric-mean trial step is from a
search snippet of arXiv:2411.19421; the Bregman test is written from memory of that paper and from the standard
relative-smoothness condition of mirror descent. Treat the method here as *this library's implementation of the SiMPL
idea*, not as a verified reproduction of the paper's algorithm; ``tests/toporlogy/test_simpl.py`` checks it against
problems with known solutions and against OC on the MBB beam.

Adaptation to this library. In the paper the volume constraint is linear in the design variable and the shift is the
exact Bregman projection onto it. Here the constraint is whatever the study passes (the volume of the *physical*
density, after filter and projection), evaluated through the chain by ``constraint_fn`` as in
:class:`~toporlogy.optimizers.oc.OC`; the shift is found by bisection. For a constraint that is linear in x both
coincide. The constraint must not decrease when x increases (volume-like).

Not covered (open in the literature as far as the search went): general nonlinear constraints such as stress or
local volume. A 2026 preprint treats several linear volume constraints (multi-material).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from toporlogy.optimizers.base import Optimizer
from toporlogy.references import cite

__all__ = ["SiMPL", "sigmoid", "logit", "bregman_divergence"]

_LINE_SEARCH = ("armijo", "bregman", "none")


def sigmoid(psi: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(−ψ)) without overflow."""
    out = np.empty_like(psi, dtype=float)
    pos = psi >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-psi[pos]))
    e = np.exp(psi[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """ln(x / (1 − x)) with x clipped to [eps, 1 − eps]."""
    xc = np.clip(x, eps, 1.0 - eps)
    return np.log(xc) - np.log1p(-xc)


def bregman_divergence(a: np.ndarray, b: np.ndarray, eps: float = 1e-300) -> float:
    """``Σ a ln(a/b) + (1 − a) ln((1 − a)/(1 − b))`` ≥ 0 (binary relative entropy; 0 ln 0 = 0)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t1 = np.where(a > 0, a * (np.log(np.maximum(a, eps)) - np.log(np.maximum(b, eps))), 0.0)
    t2 = np.where(a < 1, (1 - a) * (np.log(np.maximum(1 - a, eps)) - np.log(np.maximum(1 - b, eps))), 0.0)
    return float(np.sum(t1 + t2))


@cite("kim_2025", "keith_2025", note="update, initial step, BB step and Armijo test as in the authors' public code; "
                                     "papers not read (module notes); constraint handled through the chain")
class SiMPL(Optimizer):
    """Sigmoidal mirror descent with a projected latent variable (module notes); one volume-type constraint.

    Parameters
    ----------
    line_search : {"armijo", "bregman", "none"}
    c1 : float
        Armijo constant.
    max_backtrack : int
        Halvings per step at most; the last trial is taken if none passes (counted in ``n_line_search_failures``).
    alpha_min, alpha_max : float
        Bounds of the step size.
    shift_tol : float
        Tolerance of the bisection on the shift c (absolute, in latent units).
    psi_max : float
        |ψ| is clipped to this value (x within about exp(−psi_max) of the bounds).

    Attributes
    ----------
    alpha : float
        Step size of the last accepted step.
    last_shift : float
        Shift c of the last step (0 when the constraint was inactive).
    n_inner_total : int
        Evaluations without gradients made by the line search in this run.
    """

    name = "SiMPL"
    max_constraints = 1
    needs_constraint_fn = True

    def __init__(self, line_search: str = "armijo", c1: float = 1e-4, max_backtrack: int = 10,
                 alpha_min: float = 1e-12, alpha_max: float = 1e12, shift_tol: float = 1e-10,
                 psi_max: float = 30.0) -> None:
        if line_search not in _LINE_SEARCH:
            raise ValueError(f"line_search: one of {_LINE_SEARCH}")
        if not (0 < c1 < 1 and max_backtrack >= 0 and 0 < alpha_min < alpha_max and shift_tol > 0 and psi_max > 0):
            raise ValueError("need 0 < c1 < 1, max_backtrack >= 0, 0 < alpha_min < alpha_max, shift_tol > 0, "
                             "psi_max > 0")
        self.line_search, self.c1, self.max_backtrack = line_search, float(c1), int(max_backtrack)
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.shift_tol, self.psi_max = float(shift_tol), float(psi_max)
        self.needs_value_fn = line_search != "none"
        self.n_inner_total = 0
        self.n_line_search_failures = 0
        self.reset()

    def reset(self) -> None:
        """Forget the Barzilai–Borwein history and the step size (the problem changed)."""
        self.k = 0
        self.alpha = float("nan")
        self.last_shift = 0.0
        self._prev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None     # x, ψ, ∇f of the previous step

    def restart(self) -> None:
        self.reset()
        self.n_inner_total = 0
        self.n_line_search_failures = 0

    # ------------------------------------------------------------------ pieces
    def _project(self, psi_half: np.ndarray, constraint_fn: Callable[[np.ndarray], np.ndarray]) -> tuple[
            np.ndarray, float]:
        """Smallest shift c ≥ 0 with g(x(c)) ≤ 0, x(c) = sigmoid(clip(ψ½ − c)), by bracketing and bisection. The
        clip comes AFTER the shift: clipping ψ½ first is not the projection (a variable pushed far beyond the bound
        would be pulled back by the whole shift) and was measured to break descent at large steps."""
        m = self.psi_max

        def x_of(c: float) -> np.ndarray:
            return sigmoid(np.clip(psi_half - c, -m, m))

        def gval(c: float) -> float:
            return float(np.asarray(constraint_fn(x_of(c))).reshape(-1)[0])

        if gval(0.0) <= 0.0:
            return x_of(0.0), 0.0
        lo, hi = 0.0, 1.0
        while gval(hi) > 0.0:
            lo, hi = hi, 2.0 * hi
            if hi > 2.0 * m + float(np.max(psi_half)) + 64.0:
                raise ValueError("SiMPL: the constraint cannot be met by lowering the design (is it volume-like, "
                                 "and is its bound above the volume of the frozen solid regions?)")
        while hi - lo > self.shift_tol:
            mid = 0.5 * (lo + hi)
            if gval(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return x_of(hi), hi

    def _trial_alpha(self, x: np.ndarray, psi: np.ndarray, grad: np.ndarray) -> float:
        if self._prev is None:
            gmax = float(np.max(np.abs(grad)))
            return 1.0 / gmax if gmax > 0 else 1.0
        x0, psi0, g0 = self._prev
        dx = x - x0
        den = float(dx @ (grad - g0))
        num = float(dx @ (psi - psi0))
        if den == 0.0 or num == 0.0 or not math.isfinite(num / den):
            return self.alpha
        return math.sqrt(abs(num / den) * self.alpha)

    # ------------------------------------------------------------------ step
    def step(self, x: np.ndarray, f0: float, df0: np.ndarray, g: np.ndarray, dg: np.ndarray,
             constraint_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        x = np.asarray(x, float)
        grad = np.asarray(df0, float).reshape(x.size)
        dg = np.atleast_2d(np.asarray(dg, float))
        if dg.shape[0] != 1:
            raise ValueError("SiMPL handles exactly one (volume-type) constraint")
        if constraint_fn is None:
            raise ValueError("SiMPL needs constraint_fn (the shift is found on the constraint through the chain)")
        if float(dg.min()) < -1e-12 * max(float(np.abs(dg).max()), 1e-300):
            raise ValueError("SiMPL needs a volume-like constraint (dg/dx >= 0 everywhere)")
        if self.needs_value_fn and self.value_fn is None:
            raise ValueError("SiMPL with a line search needs value_fn (Study sets it)")
        self.k += 1
        psi = np.clip(logit(x), -self.psi_max, self.psi_max)
        alpha = min(max(self._trial_alpha(x, psi, grad), self.alpha_min), self.alpha_max)
        tries = 0
        while True:
            psi_half = psi - alpha * grad
            xnew, shift = self._project(psi_half, constraint_fn)
            if self.line_search == "none" or self.value_fn is None:
                break
            fnew = float(self.value_fn(xnew)[0])
            self.n_inner_total += 1
            lin = float(grad @ (xnew - x))
            if self.line_search == "armijo":
                ok = fnew <= f0 + self.c1 * lin + 1e-14 * abs(f0)
            else:
                ok = fnew <= f0 + lin + bregman_divergence(xnew, sigmoid(psi)) / alpha + 1e-14 * abs(f0)
            if ok:
                break
            if tries >= self.max_backtrack or alpha <= self.alpha_min:
                self.n_line_search_failures += 1
                break
            tries += 1
            alpha = max(0.5 * alpha, self.alpha_min)
        self._prev = (x.copy(), psi, grad.copy())
        self.alpha, self.last_shift = alpha, shift
        return xnew
