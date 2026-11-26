"""Mindlin–Reissner plate bending on the regular 2-D grid (bilinear Q4, three DOFs per node).

Per node (w, βx, βy): transverse deflection and the rotations of the normal, with the shear strain γ = ∇w − β and the
curvature κ = [∂βx/∂x, ∂βy/∂y, ∂βx/∂y + ∂βy/∂x]. Bending: M = D_b κ, D_b = E t³ / (12 (1 − ν²)) [[1, ν, 0], [ν, 1, 0],
[0, 0, (1 − ν)/2]], 2 × 2 Gauss points. Shear: Q = k_s G t γ (k_s = 5/6), one Gauss point by default (selective
reduced integration against shear locking; ``shear="full"`` integrates it with 2 × 2 points, the locking negative
control). The element matrix is for unit modulus; element e is scaled by E_e through :class:`GridSPD`, so the
deflection differentiates in the moduli (and in the load) with autograd.

:func:`navier_deflection` is the Navier series of the hard simply supported rectangle under uniform pressure (Mindlin,
or Kirchhoff with ``kirchhoff=True``), derived in ``docs/olcumler/plaka_onkayit.md``. Pre-registered checks: PL1-PL7
there. :class:`PlatePhysics` is the Study physics (PQ1-PQ4). Not here: YAML integration, membrane coupling
(shells), distorted elements.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.grid_spd import GridSPD
from toporlogy.fem.heat import element_nodes

__all__ = ["MindlinPlate", "PlatePhysics", "plate_element_matrix", "hard_simply_supported", "navier_deflection",
           "SHEAR_FACTOR"]

SHEAR_FACTOR = 5.0 / 6.0
_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])   # corner order of the grid elements
_G = 1.0 / np.sqrt(3.0)


def _check(t: float, nu: float) -> None:
    if not t > 0:
        raise ValueError(f"plate thickness t must be > 0, got {t}")
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio must be in (-1, 0.5), got {nu}")


def _shape(xi: float, eta: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """N (4,) and ∂N/∂(x, y) (2, 4) at (ξ, η) of a square element of edge h."""
    c = _CORNERS
    N = (1 + c[:, 0] * xi) * (1 + c[:, 1] * eta) / 4.0
    dxi = c[:, 0] * (1 + c[:, 1] * eta) / 4.0
    deta = c[:, 1] * (1 + c[:, 0] * xi) / 4.0
    return N, np.vstack([dxi, deta]) * (2.0 / h)


def _b_bending(dN: np.ndarray) -> np.ndarray:
    B = np.zeros((3, 12))
    B[0, 1::3] = dN[0]
    B[1, 2::3] = dN[1]
    B[2, 1::3] = dN[1]
    B[2, 2::3] = dN[0]
    return B


def _b_shear(N: np.ndarray, dN: np.ndarray) -> np.ndarray:
    B = np.zeros((2, 12))
    B[0, 0::3] = dN[0]
    B[0, 1::3] = -N
    B[1, 0::3] = dN[1]
    B[1, 2::3] = -N
    return B


def plate_element_matrix(h: float, t: float, nu: float, shear: str = "reduced") -> np.ndarray:
    """(12, 12) unit-modulus stiffness of a square element of edge ``h`` and thickness ``t``; DOFs (w, βx, βy) per
    corner in the grid's corner order. ``shear``: "reduced" (one point) or "full" (2 × 2)."""
    _check(t, nu)
    if shear not in ("reduced", "full"):
        raise ValueError(f'shear: "reduced" or "full", got {shear!r}')
    Db = t ** 3 / (12.0 * (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ds = SHEAR_FACTOR * t / (2.0 * (1 + nu)) * np.eye(2)
    detJ = h * h / 4.0
    gauss = [(a * _G, b * _G) for a in (-1, 1) for b in (-1, 1)]
    K = np.zeros((12, 12))
    for xi, eta in gauss:
        N, dN = _shape(xi, eta, h)
        Bb = _b_bending(dN)
        K += Bb.T @ Db @ Bb * detJ
        if shear == "full":
            Bs = _b_shear(N, dN)
            K += Bs.T @ Ds @ Bs * detJ
    if shear == "reduced":
        N, dN = _shape(0.0, 0.0, h)
        Bs = _b_shear(N, dN)
        K += Bs.T @ Ds @ Bs * 4.0 * detJ
    return 0.5 * (K + K.T)


def hard_simply_supported(domain: Domain) -> np.ndarray:
    """Fixed DOFs of a hard simple support on the whole boundary of the box: w = 0 everywhere on it, βy = 0 on the
    edges x = const and βx = 0 on the edges y = const (the tangential rotations)."""
    X = np.asarray(domain.node_coords(), float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    tol = 1e-9 * domain.h
    on_x = (np.abs(X[:, 0] - lo[0]) < tol) | (np.abs(X[:, 0] - hi[0]) < tol)
    on_y = (np.abs(X[:, 1] - lo[1]) < tol) | (np.abs(X[:, 1] - hi[1]) < tol)
    out = [3 * np.flatnonzero(on_x | on_y), 3 * np.flatnonzero(on_x) + 2, 3 * np.flatnonzero(on_y) + 1]
    return np.unique(np.concatenate(out)).astype(np.int64)


class MindlinPlate:
    """Plate bending of a 2-D box ``domain`` with thickness ``t`` and Poisson ratio ``nu``. ``fixed``: fixed DOFs
    (3 per node: w, βx, βy). :meth:`solve` gives the DOF vector for element moduli E (n_active,) and a load (3 nn,)."""

    def __init__(self, domain: Domain, t: float, nu: float = 0.3, fixed: Sequence[int] | np.ndarray = (),
                 shear: str = "reduced") -> None:
        if domain.dim != 2:
            raise ValueError("MindlinPlate needs a 2-D domain (the plate mid-surface)")
        _check(float(t), float(nu))
        self.domain, self.t, self.nu, self.shear = domain, float(t), float(nu), shear
        self.ndof = 3 * domain.nn
        nodes = element_nodes(domain, domain.active_elements)
        self.edof = (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(len(nodes), 12)
        self.KE = plate_element_matrix(domain.h, self.t, self.nu, shear)
        self.spd = GridSPD(domain, self.KE, self.edof, self.ndof, fixed)

    def pressure_load(self, q: float) -> np.ndarray:
        """Consistent nodal load (3 nn,) of a uniform pressure q on the active elements (q h² / 4 per corner)."""
        F = np.zeros(self.ndof)
        np.add.at(F, self.edof[:, 0::3].ravel(), q * self.domain.h ** 2 / 4.0)
        return F

    def solve(self, E: Any, F: Any) -> torch.Tensor:
        """DOF vector (3 nn,) for moduli ``E`` (n_active,) and load ``F`` (3 nn,); differentiable in both."""
        E = torch.as_tensor(E, dtype=torch.float64)
        return self.spd.solve(E, torch.as_tensor(np.asarray(F, float), dtype=E.dtype))


class PlatePhysics:
    """Study physics of a :class:`MindlinPlate`: ``solve(E, F)`` → U (3 nn, n_cases), differentiable in E.
    ``state_dofs`` tells :class:`~toporlogy.optimization.study.Study` the length of a load vector (3 per node)."""

    name = "plate"

    def __init__(self, plate: MindlinPlate, loads: Any = None) -> None:
        self.plate, self.domain = plate, plate.domain
        self.state_dofs = plate.ndof
        # the physics' own load (3 nn, 1) for a study built from YAML (configuration.build uses it)
        self.state_loads = None if loads is None else np.asarray(loads, float).reshape(plate.ndof, -1)
        self.n_solves = 0

    def solve(self, E: torch.Tensor, F: Any) -> torch.Tensor:
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.shape[0] != self.state_dofs:
            raise ValueError(f"plate loads have {F2.shape[0]} rows, the plate {self.state_dofs} DOFs (3 per node)")
        self.n_solves += 1
        return torch.stack([self.plate.solve(E, F2[:, c]) for c in range(F2.shape[1])], dim=1)


def navier_deflection(a: float, b: float, t: float, E: float, nu: float, q: float, x: float, y: float,
                      terms: int = 399, kirchhoff: bool = False) -> float:
    """Deflection at (x, y) of the hard simply supported a × b plate under uniform q: Σ over odd m, n ≤ ``terms`` of
    q_mn [1/(D λ²) + 1/(k_s G t λ)] sin(mπx/a) sin(nπy/b), q_mn = 16 q/(π² m n), λ = π²(m²/a² + n²/b²); the shear
    term is dropped with ``kirchhoff``."""
    _check(t, nu)
    D = E * t ** 3 / (12.0 * (1 - nu ** 2))
    G = E / (2.0 * (1 + nu))
    m = np.arange(1, terms + 1, 2, dtype=float)
    M, N = np.meshgrid(m, m, indexing="ij")
    lam = np.pi ** 2 * (M ** 2 / a ** 2 + N ** 2 / b ** 2)
    qmn = 16.0 * q / (np.pi ** 2 * M * N)
    coef = 1.0 / (D * lam ** 2) + (0.0 if kirchhoff else 1.0 / (SHEAR_FACTOR * G * t * lam))
    return float(np.sum(qmn * coef * np.sin(M * np.pi * x / a) * np.sin(N * np.pi * y / b)))
