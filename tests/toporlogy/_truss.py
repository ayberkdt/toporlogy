"""Shallow two-bar (von Mises) truss for the path-following tests (docs/olcumler/nonlineer_cozucu_onkayit.md, A).

Bars from the supports (∓a, 0) to the apex (0, h), axial stiffness EA, Green strain ε = (l² − L²)/(2L²) and axial
force N = EA ε (St. Venant–Kirchhoff bar). Unknowns: apex displacement x = (u, v); dead load λ P̂ downward. With
w = −v (downward apex displacement) the symmetric equilibrium path is, in closed form,

    P(w) = EA w (2h − w)(h − w) / L³ ,   L = √(a² + h²),

with limit points w₁,₂ = h(1 ∓ 1/√3), P(w₁) = −P(w₂) = P_max = 2 EA h³ / (3√3 L³), and the lateral stiffness
K_uu = EA (2a² + w² − 2hw)/L³ > 0 for h < √2 a (no bifurcation from the symmetric path). Derived for these tests
(Crisfield 1991 and Bažant & Cedolin 1991 treat this truss; neither was read in this session).
"""
from __future__ import annotations

import math

import numpy as np

from toporlogy.solvers.newton import LinearSolveFailure, Linearization


class TwoBarTruss:
    """R(x, λ) = f_int(x) − λ q on the apex DOFs (u, v); q = (0, −P̂)."""

    conservative = True

    def __init__(self, a: float = 1.0, h: float = 0.1, EA: float = 1.0, P_ref: float | None = None) -> None:
        self.a, self.h, self.EA = a, h, EA
        self.L = math.hypot(a, h)
        self.P_max = 2.0 * EA * h ** 3 / (3.0 * math.sqrt(3.0) * self.L ** 3)
        self.P_ref = self.P_max if P_ref is None else P_ref
        self.load = np.array([0.0, -self.P_ref])
        self.w1 = h * (1.0 - 1.0 / math.sqrt(3.0))
        self.w2 = h * (1.0 + 1.0 / math.sqrt(3.0))
        self._d0 = np.array([[a, h], [-a, h]])                   # apex minus each support

    def P(self, w):
        return self.EA * w * (2 * self.h - w) * (self.h - w) / self.L ** 3

    def dPdw(self, w):
        h = self.h
        return self.EA * (3 * w * w - 6 * h * w + 2 * h * h) / self.L ** 3

    def internal(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        f = np.zeros(2)
        K = np.zeros((2, 2))
        W = 0.0
        L = self.L
        for d0 in self._d0:
            d = d0 + x
            eps = (d @ d - L * L) / (2 * L * L)
            f += self.EA * eps * d / L
            K += self.EA / L ** 3 * np.outer(d, d) + self.EA * eps / L * np.eye(2)
            W += 0.5 * self.EA * L * eps * eps
        return f, K, W

    def linearize(self, x: np.ndarray, lam: float) -> Linearization:
        f, K, W = self.internal(x)
        return Linearization(f - lam * self.load, K, float(np.abs(K).sum(axis=0).max()),
                             W - lam * float(self.load @ x))

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        try:
            out = np.linalg.solve(lin.matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise LinearSolveFailure(f"LinAlgError: {e}") from e
        if not np.all(np.isfinite(out)):
            raise LinearSolveFailure("non-finite solution")
        return out

    def det_sign(self, lin: Linearization) -> int:
        return int(np.sign(np.linalg.det(lin.matrix)))


class TrussAtLoad:
    """The truss at a fixed load factor, for :func:`toporlogy.solvers.newton.newton` (load control)."""

    conservative = True

    def __init__(self, truss: TwoBarTruss, lam: float) -> None:
        self.truss, self.lam = truss, lam

    def linearize(self, x: np.ndarray) -> Linearization:
        return self.truss.linearize(x, self.lam)

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        return self.truss.solve(lin, rhs)
