"""Linear structural dynamics on the regular grid by Newmark's average-acceleration method (β = 1/4, γ = 1/2).

    M ü + C u̇ + K u = f(t),   C = a M + b K (Rayleigh),

with the consistent mass M = Σ_e ρ_e (∫ N Nᵀ dV ⊗ I) and K = Σ_e E_e K_e⁰; fixed DOFs are zero. The displacement
form solves K_eff u_{n+1} = f_{n+1} + M(c0 u_n + c2 v_n + c3 a_n) + C(c1 u_n + c4 v_n + c5 a_n) with
K_eff = K + c0 M + c1 C; average acceleration is unconditionally stable, second order and conserves the energy of an
undamped, unloaded system. Every step is a :class:`~toporlogy.fem.grid_spd.GridSPD` solve, so responses
differentiate in E, ρ, the loads and the initial state (the discrete adjoint by torch's chain rule); K_eff is
factorised once while its coefficients do not change. Pre-registered checks: ``docs/olcumler/newmark_onkayit.md``.
:class:`NewmarkPhysics` is the Study physics (YAML ``physics: {type: dynamics_newmark, ...}``, response
``dynamic_compliance``; DS1-DS7 in ``docs/olcumler/gecici_study_onkayit.md``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import D_iso, D_plane_stress, element_matrix_from_D
from toporlogy.fem.grid_spd import GridSPD, apply_elements
from toporlogy.fem.transient_heat import capacity_matrix

__all__ = ["Newmark", "NewmarkPhysics", "NewmarkResult", "PROFILES", "mass_matrix_element"]


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


PROFILES = ("step", "half_sine", "sine")


class NewmarkPhysics:
    """Study physics: M(x̄) ü + C u̇ + K(E) u = s(t) F from rest (u₀ = v₀ = 0), ``n_steps`` Newmark steps of ``dt``.

    Mass ρ_e = ``rho0`` (ρ_min + (1 − ρ_min) x̄_e) · frac_e (ρ_min = ``rhomin_ratio`` > 0 keeps M non-singular on void
    regions); F the study's single load case. Profiles s(t): ``step`` (1 for t ≥ 0), ``half_sine`` (sin(π t / τ)
    for t ≤ τ = ``pulse_duration``, then 0), ``sine`` (sin(2π t / ``period``)). No static state: :meth:`solve`
    raises; responses call :meth:`history` (``needs = {"history"}``).
    """

    name = "dynamics_newmark"

    def __init__(self, nm: Newmark, dt: float, n_steps: int, profile: str = "step",
                 pulse_duration: float | None = None, period: float | None = None, rho0: float = 1.0,
                 rhomin_ratio: float = 1e-6) -> None:
        if profile not in PROFILES:
            raise ValueError(f"profile: {' | '.join(PROFILES)}, got {profile!r}")
        if profile == "half_sine" and not (pulse_duration is not None and float(pulse_duration) > 0):
            raise ValueError(f"profile half_sine needs pulse_duration > 0, got {pulse_duration}")
        if profile == "sine" and not (period is not None and float(period) > 0):
            raise ValueError(f"profile sine needs period > 0, got {period}")
        if not dt > 0:
            raise ValueError(f"dt > 0 is required, got {dt}")
        if int(n_steps) < 1 or int(n_steps) != n_steps:
            raise ValueError(f"n_steps must be an integer >= 1, got {n_steps}")
        if not float(rho0) > 0:
            raise ValueError(f"rho0 > 0 is required, got {rho0}")
        if not 0.0 < float(rhomin_ratio) <= 1.0:
            raise ValueError(f"rhomin_ratio must lie in (0, 1], got {rhomin_ratio}")
        self.nm, self.domain = nm, nm.domain
        self.dt, self.n_steps, self.profile = float(dt), int(n_steps), profile
        self.pulse_duration = None if pulse_duration is None else float(pulse_duration)
        self.period = None if period is None else float(period)
        self.rho0, self.rhomin_ratio = float(rho0), float(rhomin_ratio)
        self.elements = self.domain.active_elements
        self.frac = np.asarray(self.domain.frac, float).reshape(-1)[self.elements]
        self.n_runs = 0

    def solve(self, E_active: torch.Tensor, F: Any) -> torch.Tensor:
        raise ValueError("physics dynamics_newmark has no static state u (it integrates a displacement history); "
                         "use the response dynamic_compliance")

    def load_factor(self, t: float) -> float:
        """s(t)."""
        if self.profile == "step":
            return 1.0
        if self.profile == "half_sine":
            tau = cast(float, self.pulse_duration)
            return math.sin(math.pi * t / tau) if t <= tau else 0.0
        return math.sin(2.0 * math.pi * t / cast(float, self.period))

    def times(self) -> np.ndarray:
        return self.dt * np.arange(self.n_steps + 1)

    def densities(self, xphys: torch.Tensor) -> torch.Tensor:
        """ρ_e of the active elements from the physical density (domain.shape)."""
        x = xphys.reshape(-1)[torch.as_tensor(self.elements, dtype=torch.long)]
        r = self.rhomin_ratio
        return self.rho0 * (r + (1.0 - r) * x) * torch.as_tensor(self.frac, dtype=x.dtype)

    def history(self, E_active: torch.Tensor, xphys: torch.Tensor, F: Any) -> NewmarkResult:
        """u, v, a (n_steps + 1, ndof) under s(t) F, differentiable in E and x̄."""
        n = self.domain.ndof
        Ft = torch.as_tensor(np.asarray(F, float).reshape(-1), dtype=E_active.dtype)
        if Ft.shape != (n,):
            raise ValueError(f"F must have {n} entries (DOFs), got {tuple(Ft.shape)}")
        zero = torch.zeros(n, dtype=E_active.dtype)
        self.n_runs += 1
        return self.nm.run(E_active, self.densities(xphys), zero, zero, self.dt, self.n_steps,
                           lambda t: self.load_factor(t) * Ft)
