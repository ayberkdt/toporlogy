"""Inverse homogenisation on a general periodic mesh: the material layout of one periodic cell that maximises an
effective property (``docs/foundations/prereg/IH_inverse_homogenization.md``).

x̃ = F x with the periodic density filter below, E_e = emin + x̃_e^p (1 − emin) times the unit-modulus ``D``, the
effective stiffness from :func:`toporlogy.fem.mesh.homogenization.homogenize_mesh_elasticity` (self-adjoint dC/dE_e),
the volume fraction Σ|Ω_e| x̃_e / Σ|Ω_e| ≤ ``volume``, MMA. Objective ``"bulk"``: the 2-D bulk modulus
κ = (C11 + C12 + C21 + C22)/4 (Voigt [xx, yy, xy]), maximised. For the SIMP material to stay inside the
Hashin–Shtrikman bounds the exponent must exceed
:func:`toporlogy.core.micromechanics.required_interpolation_parameters` (3.08 for ν = 0.3, plane stress).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from toporlogy.fem.mesh.homogenization import homogenize_mesh_elasticity
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import ElasticProblem

__all__ = ["PeriodicDensityFilter", "CellOptimizeResult", "cell_objective", "optimize_cell", "CELL_OBJECTIVES"]

CELL_OBJECTIVES = ("bulk",)


class PeriodicDensityFilter:
    """x̃ = F x: cone weights max(0, rmin − d_ij) |Ω_j| with d_ij the minimum-image distance of the cell centroids
    over the lattice translates n·a (n ∈ {−1, 0, 1}^dim), each row normalised by its sum. ``matrix`` is F."""

    def __init__(self, mesh: Mesh, lattice: Any, rmin: float, cell_volumes: np.ndarray | None = None) -> None:
        if not rmin > 0:
            raise ValueError(f"rmin must be > 0 (a length), got {rmin}")
        L = np.atleast_2d(np.asarray(lattice, float))
        if L.shape != (mesh.dim, mesh.dim):
            raise ValueError(f"lattice must be ({mesh.dim}, {mesh.dim}), got {L.shape}")
        c = mesh.cell_centroids()
        n = len(c)
        vol = np.ones(n) if cell_volumes is None else np.asarray(cell_volumes, float)
        if vol.shape != (n,) or not (vol > 0).all():
            raise ValueError("cell_volumes must be (n_cells,) and > 0")
        tree = cKDTree(c)
        rows, cols, vals = [], [], []
        for shift in itertools.product((-1, 0, 1), repeat=mesh.dim):
            t = np.asarray(shift, float) @ L
            for i, nb in enumerate(tree.query_ball_point(c - t, r=float(rmin))):
                nb = np.asarray(nb, np.int64)
                if not len(nb):
                    continue
                w = float(rmin) - np.linalg.norm(c[nb] + t - c[i], axis=1)
                keep = w > 0
                rows.append(np.full(int(keep.sum()), i))
                cols.append(nb[keep])
                vals.append(w[keep] * vol[nb[keep]])
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        H.sum_duplicates()
        self.matrix = (sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H).tocsr()
        self._T = self.matrix.T.tocsr()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ np.asarray(x, float))

    def backward(self, g: np.ndarray) -> np.ndarray:
        """Fᵀ g (the chain rule through the filter)."""
        return np.asarray(self._T @ np.asarray(g, float))


@dataclass
class CellOptimizeResult:
    """Design ``x``, filtered ``x_phys``, the objective value (κ for "bulk"), the effective ``C``, the volume
    fraction, iterations, the stop reason ("max_iter" or "kkt"), the last KKT residual and the history."""

    x: np.ndarray
    x_phys: np.ndarray
    objective: float
    C: np.ndarray
    volume: float
    iterations: int
    stop: str
    kkt: float | None
    history: dict[str, list[float]] = field(default_factory=dict)


def _bulk(C: np.ndarray) -> tuple[float, np.ndarray]:
    """κ and dκ/dC for 2-D Voigt C."""
    w = np.zeros((3, 3))
    w[:2, :2] = 0.25
    return float(np.sum(w * C)), w


def cell_objective(mesh: Mesh, lattice: Any, D: Any, x_phys: np.ndarray, penal: float, emin: float,
                   objective: str = "bulk") -> tuple[float, np.ndarray, np.ndarray]:
    """Objective value, its gradient with respect to x̃ and C, for the physical densities ``x_phys``."""
    if objective not in CELL_OBJECTIVES:
        raise ValueError(f"objective must be one of {CELL_OBJECTIVES}, got {objective!r}")
    if mesh.dim != 2:
        raise ValueError("the bulk objective is written for 2-D cells")
    E = emin + x_phys ** penal * (1.0 - emin)
    h = homogenize_mesh_elasticity(mesh, D, lattice, E, sensitivities=True)
    assert h.dC is not None
    val, w = _bulk(h.C)
    dE = np.einsum("ij,eij->e", w, h.dC)
    return val, dE * penal * x_phys ** (penal - 1.0) * (1.0 - emin), h.C


def optimize_cell(mesh: Mesh, lattice: Any, D: Any, volume: float, rmin: float, objective: str = "bulk",
                  penal: float = 3.5, emin: float = 1e-9, max_iter: int = 300, move: float = 0.2,
                  x0: np.ndarray | None = None, kkt_tol: float | None = None) -> CellOptimizeResult:
    """Maximise the effective ``objective`` of the periodic cell under a volume fraction (module notes)."""
    from toporlogy.optimization.diagnostics import kkt_estimate
    from toporlogy.optimizers.mma import MMA
    if not 0 < volume < 1:
        raise ValueError(f"volume must be in (0, 1), got {volume}")
    vol = ElasticProblem(mesh, np.asarray(D, float)).geom.dV.sum(axis=1)
    vw = vol / vol.sum()
    filt = PeriodicDensityFilter(mesh, lattice, rmin, vol)
    x = np.full(mesh.n_cells, float(volume)) if x0 is None else np.asarray(x0, float).copy()
    if x.shape != (mesh.n_cells,):
        raise ValueError(f"x0 must have {mesh.n_cells} entries")
    opt = MMA(move=move)
    hist: dict[str, list[float]] = {"objective": [], "volume": [], "change": []}
    f_ref: float | None = None
    stop, kkt, it = "max_iter", None, 0
    val, C, xp = 0.0, np.zeros((3, 3)), filt(x)
    while it < int(max_iter):
        xp = filt(x)
        val, dval, C = cell_objective(mesh, lattice, D, xp, penal, emin, objective)
        f_ref = abs(val) if f_ref is None else f_ref
        f0, df0 = -val / f_ref, -filt.backward(dval) / f_ref              # MMA minimises
        v = float(vw @ xp)
        g = np.array([v / volume - 1.0])
        dg = (filt.backward(vw) / volume)[None, :]
        hist["objective"].append(val)
        hist["volume"].append(v)
        if kkt_tol is not None:
            kkt = float(kkt_estimate(x, df0, g, dg)["residual"])
            if kkt <= kkt_tol:
                stop = "kkt"
                break
        it += 1
        xn = opt.step(x, f0, df0, g, dg)
        hist["change"].append(float(np.abs(xn - x).max()))
        x = xn
    else:
        xp = filt(x)
        val, _, C = cell_objective(mesh, lattice, D, xp, penal, emin, objective)
    return CellOptimizeResult(x, xp, val, C, float(vw @ xp), it, stop, kkt, hist)
