"""Polynomial manufactured solutions for linear isotropic elasticity on the unit square / cube.

``g(t) = t²(1 − t)²``, ``ψ = Π_d g(x_d)``; g and g' vanish at t = 0 and 1, so ψ and ∇ψ vanish on the boundary.

* 2-D: ``u = curl ψ + c ∇ψ`` with ``curl ψ = (∂ψ/∂y, −∂ψ/∂x)``;
* 3-D: ``u = curl(ψ, ψ, ψ) + c ∇ψ``.

So u = 0 on the whole boundary, ``div u = c Δψ``, and the pressure ``p = −λ div u = −(λc) Δψ`` has zero mean
(``∫ Δψ = ∮ ∇ψ · n = 0``). The body force of ``σ = 2μ ε + λ (div u) I`` is
``f = −div σ = −μ Δu − (λ + μ) ∇(div u) = −μ Δu − (λc + μc) ∇Δψ``.

The product ``λc`` is stored separately (``lam_c``), so ``c = μ/λ`` keeps a finite pressure ``−μ Δψ`` as λ → ∞
(:meth:`Manufactured.incompressible_limit`) and λ = ∞ (ν = 0.5) is exact. Every derivative is a product of 1-D
polynomial derivatives (exact, :class:`numpy.polynomial.Polynomial`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

__all__ = ["Manufactured"]

_G = [Polynomial([0.0, 0.0, 1.0, -2.0, 1.0])]
for _ in range(6):
    _G.append(_G[-1].deriv())
_COEF = [np.asarray(g.coef, float) for g in _G]


def _table(x: np.ndarray) -> list[list[np.ndarray]]:
    """g^(a)(x_d) for a = 0 … 6 and every direction d (each evaluated once)."""
    out: list[list[np.ndarray]] = []
    for d in range(x.shape[1]):
        xd = np.ascontiguousarray(x[:, d])
        out.append([np.polynomial.polynomial.polyval(xd, c) for c in _COEF])
    return out


def _dpsi(x: np.ndarray | list[list[np.ndarray]], alpha: tuple[int, ...]) -> np.ndarray:
    """D^α ψ at points x (n, dim) (or at a precomputed :func:`_table`)."""
    T = _table(np.asarray(x, float)) if isinstance(x, np.ndarray) else x
    out: np.ndarray = np.array(T[0][alpha[0]], float)
    for d in range(1, len(alpha)):
        out *= T[d][alpha[d]]
    return out


def _e(dim: int, *idx: int) -> tuple[int, ...]:
    a = [0] * dim
    for i in idx:
        a[i] += 1
    return tuple(a)


@dataclass(frozen=True)
class Manufactured:
    """u = curl ψ + c ∇ψ, pressure −lam_c Δψ (see the module notes). ``c`` and ``lam_c`` = λ·c."""

    dim: int
    c: float = 0.0
    lam_c: float = 0.0

    @classmethod
    def incompressible_limit(cls, dim: int, mu: float, lam: float) -> Manufactured:
        """c = μ/λ: finite pressure −μ Δψ for every λ, including λ = ∞."""
        return cls(dim, 0.0 if math.isinf(lam) else mu / lam, mu)

    # u_i = Σ coef · D^α ψ
    def _terms(self) -> list[list[tuple[float, tuple[int, ...]]]]:
        d = self.dim
        if d == 2:
            return [[(1.0, _e(2, 1)), (self.c, _e(2, 0))], [(-1.0, _e(2, 0)), (self.c, _e(2, 1))]]
        return [[(1.0, _e(3, 1)), (-1.0, _e(3, 2)), (self.c, _e(3, 0))],
                [(1.0, _e(3, 2)), (-1.0, _e(3, 0)), (self.c, _e(3, 1))],
                [(1.0, _e(3, 0)), (-1.0, _e(3, 1)), (self.c, _e(3, 2))]]

    def _eval(self, x: np.ndarray, extra: tuple[int, ...]) -> np.ndarray:
        x = np.asarray(x, float).reshape(-1, self.dim)
        T = _table(x)
        return np.stack([sum(cf * _dpsi(T, tuple(a + b for a, b in zip(al, extra, strict=True)))
                             for cf, al in comp if cf != 0.0) + np.zeros(len(x))
                         for comp in self._terms()], axis=1)

    def u(self, x: np.ndarray) -> np.ndarray:
        """(n, dim) displacement."""
        return self._eval(x, (0,) * self.dim)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """(n, dim, dim) ∂u_i/∂x_j."""
        return np.stack([self._eval(x, _e(self.dim, j)) for j in range(self.dim)], axis=2)

    def laplacian(self, x: np.ndarray) -> np.ndarray:
        return np.sum([self._eval(x, _e(self.dim, j, j)) for j in range(self.dim)], axis=0)

    def lap_psi(self, x: np.ndarray) -> np.ndarray:
        T = _table(np.asarray(x, float).reshape(-1, self.dim))
        return np.sum([_dpsi(T, _e(self.dim, j, j)) for j in range(self.dim)], axis=0)

    def grad_lap_psi(self, x: np.ndarray) -> np.ndarray:
        T = _table(np.asarray(x, float).reshape(-1, self.dim))
        return np.stack([sum(_dpsi(T, _e(self.dim, j, j, i)) for j in range(self.dim)) for i in range(self.dim)],
                        axis=1)

    def divergence(self, x: np.ndarray) -> np.ndarray:
        return self.c * self.lap_psi(x)

    def pressure(self, x: np.ndarray) -> np.ndarray:
        """p = −λ div u = −lam_c Δψ (positive in compression)."""
        return -self.lam_c * self.lap_psi(x)

    def body_force(self, mu: float, x: np.ndarray) -> np.ndarray:
        """(n, dim) f = −μ Δu − (λc + μc) ∇Δψ."""
        return -mu * self.laplacian(x) - (self.lam_c + mu * self.c) * self.grad_lap_psi(x)
