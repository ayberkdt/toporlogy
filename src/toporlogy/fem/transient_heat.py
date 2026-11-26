"""Transient heat conduction on the regular grid by the θ-method (05 §3).

    C(x) Ṫ + K(x) T = q,    (C/Δt + θ K) T_{n+1} = (C/Δt − (1 − θ) K) T_n + q,

with the consistent heat capacity C = Σ_e c_e ∫ N Nᵀ dV (2^dim Gauss points, exact for bilinear products) and the
conduction matrix K of :mod:`toporlogy.fem.heat`. θ = 1 is backward Euler (first order, L-stable), θ = 1/2
Crank–Nicolson (second order). Temperatures at Dirichlet nodes are prescribed (constant in time). Every step is a
:class:`~toporlogy.fem.grid_spd.GridSPD` solve, so a response of the temperatures differentiates in the element
capacities, conductivities, the initial field and the source by torch's chain rule (the discrete adjoint); the
factorisation is reused while the coefficients do not change. Pre-registered checks:
``docs/olcumler/gecici_isi_onkayit.md`` (TH1-TH6).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.grid_spd import GridSPD, apply_elements
from toporlogy.fem.heat import conduction_matrix, element_nodes

__all__ = ["TransientHeat", "capacity_matrix"]


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
