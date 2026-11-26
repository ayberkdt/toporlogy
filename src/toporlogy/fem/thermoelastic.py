"""One-way thermoelasticity on the regular grid: temperature → thermal load → displacement (05 §2).

Small strain, ε_el = ε − α ΔT and σ = E · C₀ ε_el with the unit-modulus elasticity C₀ (Voigt, engineering shear
strains; α as a symmetric tensor, α_v = [α11, α22, α33, 2α23, 2α13, 2α12]). The thermal load of an element is

    f_e = E_e ∫ Bᵀ β (N · ΔT_e) dV,        β = C₀ α_v,

with 2 × 2 (× 2) Gauss points and ΔT interpolated bilinearly from the nodes. In 2-D plane stress β = D_ps α_p
(condensing σzz = σyz = σxz = 0 leaves the thermal strain in its in-plane components); in plane strain β is the
in-plane part of C₀³ᴰ α³ᴰ (εzz = 0). Q4 and H8 only: an incompatible-mode element would need a thermal load on its
modes, so Q4IM / H8IM are refused.

:class:`ThermoElastic` couples a :class:`GridSPD` heat solve (−div(k ∇T) = q) and an elastic solve; both are torch
autograd functions with derivatives with respect to the element coefficients *and* the right-hand side, so a
response of the displacement differentiates through the design-dependent thermal load and through the temperature.
Pre-registered checks: ``docs/olcumler/termoelastisite_onkayit.md`` (TE1-TE9). Not here: two-way coupling, transient
heat, temperature-dependent material, the Study / YAML integration.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import (D_iso, D_plane_strain, D_plane_stress, _B_unit_h8, _B_unit_q4,
                                    element_matrix_from_D)
from toporlogy.fem.grid_spd import GridSPD
from toporlogy.fem.heat import conduction_matrix, element_nodes

__all__ = ["ThermoElastic", "thermal_beta", "thermal_load_matrix", "alpha_voigt"]

_IN_PLANE = [0, 1, 5]


def alpha_voigt(alpha: Any, dim: int) -> np.ndarray:
    """Engineering Voigt vector of a thermal expansion tensor (a number = isotropic; (3, 3) or (2, 2) symmetric)."""
    a = np.asarray(alpha, float)
    A = np.eye(3) * float(a) if a.ndim == 0 else a
    if A.shape == (2, 2):
        A = np.pad(A, ((0, 1), (0, 1)))
    if A.shape != (3, 3) or not np.allclose(A, A.T, rtol=0, atol=1e-15 * max(np.abs(A).max(), 1e-300)):
        raise ValueError(f"alpha must be a number or a symmetric 2 x 2 / 3 x 3 tensor, got shape {a.shape}")
    return np.array([A[0, 0], A[1, 1], A[2, 2], 2 * A[1, 2], 2 * A[0, 2], 2 * A[0, 1]])


def thermal_beta(alpha: Any, dim: int, nu: float = 0.3, D: np.ndarray | None = None, plane: str = "stress",
                 C3: np.ndarray | None = None) -> np.ndarray:
    """β = C₀ α (unit modulus) in the element's Voigt order. 2-D: ``plane`` stress (β = D_ps α_p, ``D`` the plane
    stress matrix) or strain (β = (C3 α)_in-plane with the 3-D unit-modulus ``C3``)."""
    av = alpha_voigt(alpha, dim)
    if dim == 3:
        C = D_iso(nu) if D is None else np.asarray(D, float)
        return C @ av
    if plane == "stress":
        Dp = D_plane_stress(nu) if D is None else np.asarray(D, float)
        return Dp @ av[_IN_PLANE]
    if plane == "strain":
        C = D_iso(nu) if C3 is None else np.asarray(C3, float)
        return (C @ av)[_IN_PLANE]
    raise ValueError(f'plane: "stress" or "strain", got {plane!r}')


def thermal_load_matrix(dim: int, beta: np.ndarray, h: float, thickness: float = 1.0) -> np.ndarray:
    """G (n_dof_el, n_node_el): f_e = E_e · G · ΔT_e for an element of size h (∫ Bᵀ β N dV, 2^dim Gauss points)."""
    g = 1.0 / np.sqrt(3.0)
    pts = np.array(np.meshgrid(*([[-g, g]] * dim), indexing="ij")).reshape(dim, -1).T
    corner = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) if dim == 2 else
              np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1],
                        [-1, 1, 1]]))
    B_of = _B_unit_q4 if dim == 2 else _B_unit_h8
    G = np.zeros((dim * len(corner), len(corner)))
    for p in pts:
        N = np.prod(1 + corner * p, axis=1) / 2 ** dim
        G += np.outer(B_of(p)[0].T @ beta, N) / 2 ** dim          # Gauss weight = element volume / 2^dim (unit)
    return G * h ** (dim - 1) * (thickness if dim == 2 else 1.0)


class ThermoElastic:
    """Heat conduction followed by elasticity with the thermal load of the temperature rise above ``T_ref``.

    ``heat_fixed`` / ``heat_values``: prescribed temperatures (node numbers); ``mech_fixed``: fixed displacement DOFs.
    ``element``: "Q4" (2-D) or "H8" (3-D). The modulus of element e is E_e (passed to :meth:`displacement`), its
    conductivity k_e.
    """

    def __init__(self, domain: Domain, mech_fixed: Sequence[int] | np.ndarray, alpha: Any, nu: float = 0.3,
                 heat_fixed: Sequence[int] | np.ndarray = (), heat_values: Any = 0.0, T_ref: float = 0.0,
                 element: str | None = None, plane: str = "stress", D: np.ndarray | None = None) -> None:
        dim = domain.dim
        element = element or ("Q4" if dim == 2 else "H8")
        if element not in ("Q4", "H8"):
            raise ValueError(f"thermoelasticity supports Q4 and H8 only, got {element!r} (an incompatible-mode "
                             f"element would need a thermal load on its modes)")
        if dim == 2 and D is None:
            D = D_plane_stress(nu) if plane == "stress" else D_plane_strain(nu)
        elif D is None:
            D = D_iso(nu)
        self.domain, self.dim, self.T_ref = domain, dim, float(T_ref)
        self.D = np.asarray(D, float)
        geom = domain.thickness if dim == 2 else domain.h
        el = domain.active_elements
        self.elements = el
        self.edof = np.asarray(domain.edof)[el]
        self.enodes = element_nodes(domain, el)
        self.mech = GridSPD(domain, element_matrix_from_D(dim, element, self.D) * geom, self.edof,
                            domain.ndof, mech_fixed)
        self.heat = GridSPD(domain, conduction_matrix(dim) * geom, self.enodes, domain.nn, heat_fixed, heat_values)
        self.beta = thermal_beta(alpha, dim, nu, D=self.D if (dim == 3 or plane == "stress") else None,
                                 plane=plane)
        self.G = thermal_load_matrix(dim, self.beta, domain.h, domain.thickness)

    def temperature(self, k: torch.Tensor, q: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
        """Nodal temperature for element conductivities ``k`` and nodal heat input ``q`` (default 0)."""
        rhs = torch.zeros(self.domain.nn, dtype=k.dtype) if q is None else torch.as_tensor(q, dtype=k.dtype)
        return self.heat.solve(k, rhs)

    def thermal_load(self, E: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Global thermal load (ndof,) of moduli ``E`` and nodal temperatures ``T`` (differentiable in both)."""
        T = torch.as_tensor(T, dtype=E.dtype)
        if T.shape != (self.domain.nn,):
            raise ValueError(f"T must have {self.domain.nn} entries (nodes), got {tuple(T.shape)}")
        dT = (T - self.T_ref)[torch.as_tensor(self.enodes)]                  # (n_el, nnode)
        fe = E[:, None] * (dT @ torch.as_tensor(self.G.T, dtype=E.dtype))   # (n_el, ndof_el)
        F = torch.zeros(self.domain.ndof, dtype=E.dtype)
        return F.index_add(0, torch.as_tensor(self.edof.ravel()), fe.reshape(-1))

    def displacement(self, E: torch.Tensor, T: torch.Tensor, F_mech: torch.Tensor | np.ndarray | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, F) with F = F_mech + thermal load."""
        F = self.thermal_load(E, T)
        if F_mech is not None:
            F = F + torch.as_tensor(F_mech, dtype=E.dtype)
        return self.mech.solve(E, F), F

    def element_stress(self, E: np.ndarray, u: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Centroid stress E (D B u − β ΔT) per element (n_el, k), unit element size h."""
        dim = self.dim
        B = (_B_unit_q4 if dim == 2 else _B_unit_h8)(np.zeros(dim))[0] / self.domain.h
        dT = (np.asarray(T, float) - self.T_ref)[self.enodes].mean(axis=1)
        eps = np.asarray(u, float)[self.edof] @ B.T
        return np.asarray(E, float)[:, None] * (eps @ self.D.T - np.outer(dT, self.beta))
