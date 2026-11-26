"""Transient heat conduction on the regular grid by the θ-method (05 §3).

    C(x) Ṫ + K(x) T = q,    (C/Δt + θ K) T_{n+1} = (C/Δt − (1 − θ) K) T_n + q,

with the consistent heat capacity C = Σ_e c_e ∫ N Nᵀ dV (2^dim Gauss points, exact for bilinear products) and the
conduction matrix K of :mod:`toporlogy.fem.heat`. θ = 1 is backward Euler (first order, L-stable), θ = 1/2
Crank–Nicolson (second order). Temperatures at Dirichlet nodes are prescribed (constant in time). Every step is a
:class:`~toporlogy.fem.grid_spd.GridSPD` solve, so a response of the temperatures differentiates in the element
capacities, conductivities, the initial field and the source by torch's chain rule (the discrete adjoint); the
factorisation is reused while the coefficients do not change. Pre-registered checks:
``docs/olcumler/gecici_isi_onkayit.md`` (TH1-TH6). :class:`TransientHeatPhysics` is the Study physics (YAML
``physics: {type: transient_heat, ...}``, response ``transient_thermal_compliance``; HS1-HS7 in
``docs/olcumler/gecici_study_onkayit.md``).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain, require_grid
from toporlogy.fem.grid_spd import GridSPD, apply_elements
from toporlogy.fem.heat import conduction_matrix, element_nodes

__all__ = ["TransientHeat", "TransientHeatPhysics", "capacity_matrix", "nodal_source"]


def capacity_matrix(dim: int) -> np.ndarray:
    """∫ N Nᵀ dV of the unit square / cube in the corner order of the domain (2^dim Gauss points: exact)."""
    g = 1.0 / np.sqrt(3.0)
    pts = np.array(np.meshgrid(*([[-g, g]] * dim), indexing="ij")).reshape(dim, -1).T
    corner = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) if dim == 2 else
              np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1],
                        [-1, 1, 1]]))
    M = np.zeros((len(corner), len(corner)))
    for p in pts:
        N = np.prod(1 + corner * p, axis=1) / 2 ** dim
        M += np.outer(N, N) / 2 ** dim
    return M


class TransientHeat:
    """θ-method on the active grid. ``fixed`` / ``values``: Dirichlet nodes and temperatures."""

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray = (), values: Any = 0.0,
                 theta: float = 0.5) -> None:
        require_grid(domain, "TransientHeat")
        if not 0.0 <= float(theta) <= 1.0:
            raise ValueError(f"theta must lie in [0, 1], got {theta}")
        self.domain, self.theta = domain, float(theta)
        dim = domain.dim
        geom = domain.thickness if dim == 2 else domain.h
        self.KK = conduction_matrix(dim) * geom                           # ∫ ∇Nᵀ∇N dV for unit conductivity
        self.CC = capacity_matrix(dim) * domain.h ** dim * (domain.thickness if dim == 2 else 1.0)
        self.enodes = element_nodes(domain, domain.active_elements)
        self.fixed = np.asarray(fixed, np.int64)
        self.values = values
        self.system = GridSPD(domain, np.stack([self.CC, self.KK]), self.enodes, domain.nn, self.fixed, values)

    def run(self, k: torch.Tensor, c: torch.Tensor, T0: torch.Tensor, dt: float, n_steps: int,
            q: torch.Tensor | None = None) -> torch.Tensor:
        """Temperatures (n_steps + 1, nn) from T0 (row 0) with element conductivities ``k`` and capacities ``c``."""
        if not dt > 0:
            raise ValueError(f"dt > 0 is required, got {dt}")
        nn = self.domain.nn
        T = torch.as_tensor(T0, dtype=k.dtype)
        if T.shape != (nn,):
            raise ValueError(f"T0 must have {nn} entries (nodes), got {tuple(T.shape)}")
        th = self.theta
        lhs = torch.stack([c / dt, th * k])
        explicit = torch.stack([c / dt, -(1.0 - th) * k])
        src = torch.zeros(nn, dtype=k.dtype) if q is None else torch.as_tensor(q, dtype=k.dtype)
        out = [T]
        for _ in range(int(n_steps)):
            rhs = apply_elements(self.system.KE, explicit, T, self.enodes, nn) + src
            T = self.system.solve(lhs, rhs)
            out.append(T)
        return torch.stack(out)

    def total_heat(self, c: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """1ᵀ C T for each row of T (n_rows, nn)."""
        Ts = T if T.dim() == 2 else T[None]
        return torch.stack([apply_elements(self.CC, c, t, self.enodes, self.domain.nn).sum() for t in Ts])


def nodal_source(domain: Domain, density: float) -> np.ndarray:
    """Consistent nodal heat input (nn,) of a uniform source ``density`` per unit volume over the active elements
    (∫ N_i dV = element volume / 2^dim for bilinear / trilinear shape functions), times ``frac``."""
    require_grid(domain, "nodal_source")
    vol = domain.h ** domain.dim * (domain.thickness if domain.dim == 2 else 1.0)
    el = domain.active_elements
    frac = np.asarray(domain.frac, float).reshape(-1)[el]
    enodes = element_nodes(domain, el)
    q = np.zeros(domain.nn)
    np.add.at(q, enodes.ravel(), np.repeat(float(density) * vol * frac / enodes.shape[1], enodes.shape[1]))
    return q


class TransientHeatPhysics:
    """Study physics: C(x̄) Ṫ + K(E) T = q by the θ-method over ``n_steps`` steps of ``dt``.

    Conductivity k_e = ``conductivity`` · E_e / ``E0`` (E from the study's interpolation), heat capacity
    c_e = ``capacity`` · (c_min + (1 − c_min) x̄_e) · frac_e with c_min = ``capacity_min_ratio`` (1: uniform),
    design-independent nodal heat input ``q`` (nn,), uniform initial temperature ``T0`` (the Dirichlet nodes of
    ``heat`` start at their prescribed values). There is no static state: :meth:`solve` raises; responses call
    :meth:`history` (``needs = {"history"}``).
    """

    name = "transient_heat"

    def __init__(self, heat: TransientHeat, q: Any, dt: float, n_steps: int, T0: float = 0.0,
                 conductivity: float = 1.0, E0: float = 1.0, capacity: float = 1.0,
                 capacity_min_ratio: float = 1e-3) -> None:
        d = heat.domain
        q = np.asarray(q, float).reshape(-1)
        if q.shape != (d.nn,):
            raise ValueError(f"q must have {d.nn} entries (nodes), got {q.shape}")
        if not np.any(q):
            raise ValueError("transient_heat needs a heat input (source_density or power): q is zero")
        if not dt > 0:
            raise ValueError(f"dt > 0 is required, got {dt}")
        if int(n_steps) < 1 or int(n_steps) != n_steps:
            raise ValueError(f"n_steps must be an integer >= 1, got {n_steps}")
        for key, v in (("conductivity", conductivity), ("E0", E0), ("capacity", capacity)):
            if not float(v) > 0:
                raise ValueError(f"{key} > 0 is required, got {v}")
        if not 0.0 < float(capacity_min_ratio) <= 1.0:
            raise ValueError(f"capacity_min_ratio must lie in (0, 1], got {capacity_min_ratio}")
        self.heat, self.domain, self.q = heat, d, q
        self.dt, self.n_steps = float(dt), int(n_steps)
        self.conductivity, self.E0 = float(conductivity), float(E0)
        self.capacity, self.capacity_min_ratio = float(capacity), float(capacity_min_ratio)
        self.elements = d.active_elements
        self.frac = np.asarray(d.frac, float).reshape(-1)[self.elements]
        T = np.full(d.nn, float(T0))
        T[heat.fixed] = np.broadcast_to(np.asarray(heat.values, float), heat.fixed.shape)
        self.T0 = T
        self.n_runs = 0

    def solve(self, E_active: torch.Tensor, F: Any) -> torch.Tensor:
        raise ValueError("physics transient_heat has no static state u (it integrates a temperature history); use "
                         "the response transient_thermal_compliance")

    def conductivities(self, E_active: torch.Tensor) -> torch.Tensor:
        return E_active * (self.conductivity / self.E0)

    def capacities(self, xphys: torch.Tensor) -> torch.Tensor:
        """c_e of the active elements from the physical density (domain.shape)."""
        x = xphys.reshape(-1)[torch.as_tensor(self.elements, dtype=torch.long)]
        r = self.capacity_min_ratio
        return self.capacity * (r + (1.0 - r) * x) * torch.as_tensor(self.frac, dtype=x.dtype)

    def history(self, E_active: torch.Tensor, xphys: torch.Tensor) -> torch.Tensor:
        """Temperatures (n_steps + 1, nn), differentiable in E and x̄."""
        self.n_runs += 1
        return self.heat.run(self.conductivities(E_active), self.capacities(xphys),
                             torch.as_tensor(self.T0, dtype=E_active.dtype), self.dt, self.n_steps,
                             torch.as_tensor(self.q, dtype=E_active.dtype))
