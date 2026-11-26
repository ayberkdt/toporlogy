"""Linear structural dynamics on the regular grid by Newmark's average-acceleration method (β = 1/4, γ = 1/2).

    M ü + C u̇ + K u = f(t),   C = a M + b K (Rayleigh),

with the consistent mass M = Σ_e ρ_e (∫ N Nᵀ dV ⊗ I) and K = Σ_e E_e K_e⁰; fixed DOFs are zero. The displacement
form solves K_eff u_{n+1} = f_{n+1} + M(c0 u_n + c2 v_n + c3 a_n) + C(c1 u_n + c4 v_n + c5 a_n) with
K_eff = K + c0 M + c1 C; average acceleration is unconditionally stable, second order and conserves the energy of an
undamped, unloaded system. Every step is a :class:`~toporlogy.fem.grid_spd.GridSPD` solve, so responses
differentiate in E, ρ, the loads and the initial state (the discrete adjoint by torch's chain rule); K_eff is
factorised once while its coefficients do not change. Pre-registered checks: ``docs/olcumler/newmark_onkayit.md``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import D_iso, D_plane_stress, element_matrix_from_D
from toporlogy.fem.grid_spd import GridSPD, apply_elements
from toporlogy.fem.transient_heat import capacity_matrix

__all__ = ["Newmark", "NewmarkResult", "mass_matrix_element"]


def mass_matrix_element(dim: int, h: float, thickness: float = 1.0) -> np.ndarray:
    """Consistent unit-density mass of an element (node-major DOFs: node · dim + component)."""
    vol = h ** dim * (thickness if dim == 2 else 1.0)
    return np.kron(capacity_matrix(dim), np.eye(dim)) * vol


class NewmarkResult:
    def __init__(self, u: torch.Tensor, v: torch.Tensor, a: torch.Tensor, t: np.ndarray) -> None:
        self.u, self.v, self.a, self.t = u, v, a, t


class Newmark:
    """Average-acceleration Newmark on the active grid (Q4 in 2-D plane stress, H8 in 3-D, or a given ``D``)."""

    BETA, GAMMA = 0.25, 0.5

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, nu: float = 0.3,
                 D: np.ndarray | None = None, element: str | None = None, rayleigh: tuple[float, float] = (0.0, 0.0)
                 ) -> None:
        dim = domain.dim
        if D is None:
            D = D_plane_stress(nu) if dim == 2 else D_iso(nu)
        element = element or ("Q4" if dim == 2 else "H8")
        geom = domain.thickness if dim == 2 else domain.h
        self.domain, self.dim = domain, dim
        self.KE = element_matrix_from_D(dim, element, np.asarray(D, float)) * geom
        self.ME = mass_matrix_element(dim, domain.h, domain.thickness)
        self.edof = np.asarray(domain.edof)[domain.active_elements]
        self.fixed = np.asarray(fixed, np.int64)
        self.a_m, self.b_k = float(rayleigh[0]), float(rayleigh[1])
        self.effective = GridSPD(domain, np.stack([self.KE, self.ME]), self.edof, domain.ndof, self.fixed)
        self.mass = GridSPD(domain, self.ME, self.edof, domain.ndof, self.fixed)

    def _K(self, E: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return apply_elements(self.KE, E, u, self.edof, self.domain.ndof)

    def _M(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return apply_elements(self.ME, rho, u, self.edof, self.domain.ndof)

    def _free(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(self.domain.ndof, dtype=x.dtype)
        mask[torch.as_tensor(self.fixed)] = 0.0
        return x * mask

    def energy(self, E: torch.Tensor, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return 0.5 * (v @ self._M(rho, v)) + 0.5 * (u @ self._K(E, u))

    def run(self, E: torch.Tensor, rho: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor, dt: float, n_steps: int,
            force: Callable[[float], torch.Tensor] | None = None) -> NewmarkResult:
        """u, v, a (n_steps + 1, ndof) from u0, v0; ``force(t)`` the load vector at time t (default 0)."""
        if not dt > 0:
            raise ValueError(f"dt > 0 is required, got {dt}")
        n = self.domain.ndof
        u = torch.as_tensor(u0, dtype=E.dtype)
        v = torch.as_tensor(v0, dtype=E.dtype)
        if u.shape != (n,) or v.shape != (n,):
            raise ValueError(f"u0 and v0 must have {n} entries (DOFs), got {tuple(u.shape)} and {tuple(v.shape)}")
        zero = torch.zeros(n, dtype=E.dtype)
        f = (lambda t: zero) if force is None else force
        am, bk = self.a_m, self.b_k
        b, g = self.BETA, self.GAMMA
        c0, c1 = 1 / (b * dt * dt), g / (b * dt)
        c2, c3 = 1 / (b * dt), 1 / (2 * b) - 1
        c4, c5 = g / b - 1, dt * (g / (2 * b) - 1)

        def C(x: torch.Tensor) -> torch.Tensor:
            return am * self._M(rho, x) + bk * self._K(E, x)
        a = self.mass.solve(rho, self._free(f(0.0) - C(v) - self._K(E, u)))
        coef = torch.stack([(1 + c1 * bk) * E, (c0 + c1 * am) * rho])
        U, V, A = [u], [v], [a]
        for k in range(1, int(n_steps) + 1):
            t = k * dt
            rhs = f(t) + self._M(rho, c0 * u + c2 * v + c3 * a) + C(c1 * u + c4 * v + c5 * a)
            un = self.effective.solve(coef, self._free(rhs))
            an = c0 * (un - u) - c2 * v - c3 * a
            vn = v + dt * ((1 - g) * a + g * an)
            u, v, a = un, vn, an
            U.append(u)
            V.append(v)
            A.append(a)
        return NewmarkResult(torch.stack(U), torch.stack(V), torch.stack(A), dt * np.arange(int(n_steps) + 1))
