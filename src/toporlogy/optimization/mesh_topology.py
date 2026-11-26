"""Density topology optimisation on a general mesh (first slice; ``docs/olcumler/ag_topoloji_onkayit.md``, MS1-MS6).

:class:`~toporlogy.optimization.study.Study` and the design chain work on the regular grid. This module runs the
classic compliance problem on any :class:`~toporlogy.fem.mesh.Mesh` (distorted quadrilaterals, triangles, 3-D cells)
without them: a density filter on the cell centroids (:class:`MeshDensityFilter`), the compliance and its adjoint
sensitivity from the unit-modulus element matrices of an :class:`~toporlogy.fem.mesh.ElasticProblem`
(:class:`MeshCompliance`) and a SIMP / MMA loop (:func:`optimize_mesh`). On ``Mesh.from_domain`` of a grid the filter
and the compliance are those of the grid path (MS1, MS2). Not here: the YAML / Study integration, projection,
other responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from toporlogy.fem.mesh import ElasticProblem, Mesh
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.optimizers.mma import MMA

__all__ = ["MeshDensityFilter", "MeshCompliance", "MeshResult", "optimize_mesh"]


class MeshDensityFilter:
    """x̃ = H x / Hs over the cells of ``mesh``: cone weights max(0, rmin − |c_i − c_j|) between cell centroids
    (``rmin`` in length units), each row normalised by its sum. ``matrix`` is the normalised sparse matrix.
    ``mesh`` is a :class:`~toporlogy.fem.mesh.Mesh` or the centroids themselves, an (n, d) array (shell facets)."""

    def __init__(self, mesh: Mesh | np.ndarray, rmin: float) -> None:
        if not rmin > 0:
            raise ValueError(f"rmin must be > 0 (a length), got {rmin}")
        c = np.asarray(mesh.cell_centroids() if isinstance(mesh, Mesh) else mesh, float)
        if c.ndim != 2 or len(c) == 0 or not np.isfinite(c).all():
            raise ValueError(f"centroids must be a finite non-empty (n, d) array, got shape {c.shape}")
        tree = cKDTree(c)
        rows, cols, vals = [], [], []
        for i, nb in enumerate(tree.query_ball_point(c, r=float(rmin))):
            nb = np.asarray(nb, np.int64)
            w = float(rmin) - np.linalg.norm(c[nb] - c[i], axis=1)
            keep = w > 0
            rows.append(np.full(int(keep.sum()), i))
            cols.append(nb[keep])
            vals.append(w[keep])
        n = len(c)
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.matrix = (sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H).tocsr()
        self._T = self.matrix.T.tocsr()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ np.asarray(x, float))

    def backward(self, g: np.ndarray) -> np.ndarray:
        """Hᵀ g (the chain rule through the filter)."""
        return np.asarray(self._T @ np.asarray(g, float))


class MeshCompliance:
    """Compliance c = Fᵀu of K(E) u = F with K(E) = Σ_e E_e K_e (``problem``'s element matrices for unit modulus)
    and zero displacements on ``fixed``; :meth:`__call__` returns (c, dc/dE) with dc/dE_e = −u_eᵀ K_e u_e."""

    def __init__(self, problem: ElasticProblem, F: np.ndarray, fixed: np.ndarray) -> None:
        self.problem = problem
        self.F = np.asarray(F, float).reshape(-1)
        if self.F.shape != (problem.ndof,):
            raise ValueError(f"load must have {problem.ndof} entries, got {self.F.shape}")
        self.ke = np.asarray(problem.element_stiffness(), float)        # (ne, nloc, nloc), unit modulus
        self.cell_dofs = np.asarray(problem.cell_dofs)
        fixed = np.unique(np.asarray(fixed, np.int64))
        self.free = np.setdiff1d(problem.active, fixed)
        self.n_solves = 0

    def __call__(self, E: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(c, dc/dE (ne,), u (ndof,)) for cell moduli E (ne,)."""
        E = np.asarray(E, float).reshape(-1)
        if E.shape != (len(self.ke),):
            raise ValueError(f"E must have {len(self.ke)} entries (cells), got {E.shape}")
        K = self.problem.pattern.assemble(E[:, None, None] * self.ke)
        fr = self.free
        u = np.zeros(self.problem.ndof)
        u[fr] = np.asarray(Factorization(K[fr][:, fr], "spd").solve(self.F[fr])).reshape(-1)
        self.n_solves += 1
        ue = u[self.cell_dofs]
        dc = -np.einsum("ea,eab,eb->e", ue, self.ke, ue)
        return float(self.F @ u), dc, u


@dataclass
class MeshResult:
    x: np.ndarray                   # design variables
    xphys: np.ndarray               # filtered density
    objective: float                # compliance of the returned design
    volume: float                   # volume fraction of the returned design
    history: dict[str, list[float]] = field(default_factory=dict)


def optimize_mesh(mesh: Mesh, D: np.ndarray, F: np.ndarray, fixed: np.ndarray, volume: float, rmin: float,
                  penal: float = 3.0, emin_ratio: float = 1e-9, max_iter: int = 50, tol: float = 0.01,
                  thickness: float = 1.0, move: float = 0.2, x0: np.ndarray | None = None) -> MeshResult:
    """Minimise compliance subject to a volume fraction ≤ ``volume`` (weighted by the cell volumes) on ``mesh``:
    density filter of radius ``rmin`` (length), SIMP E = E_min + x̄^p (1 − E_min) times the unit-modulus ``D``, MMA.
    Stops at ``max_iter`` or when the largest design change is below ``tol``."""
    if not 0.0 < volume <= 1.0:
        raise ValueError(f"volume must be in (0, 1], got {volume}")
    prob = ElasticProblem(mesh, np.asarray(D, float), thickness=thickness)
    comp = MeshCompliance(prob, F, fixed)
    filt = MeshDensityFilter(mesh, rmin)
    vol = np.asarray(prob.geom.dV, float).sum(axis=1)
    w = vol / vol.sum()
    emin = float(emin_ratio)
    x = np.full(mesh.n_cells, float(volume)) if x0 is None else np.asarray(x0, float).copy()
    opt = MMA(move=move)
    hist: dict[str, list[float]] = {"f0": [], "volume": [], "change": []}
    c0 = None

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray, float, np.ndarray, Any]:
        xp = filt(x)
        E = emin + xp ** penal * (1.0 - emin)
        c, dcdE, _ = comp(E)
        dc = filt.backward(dcdE * penal * xp ** (penal - 1.0) * (1.0 - emin))
        v = float(w @ xp)
        dv = filt.backward(w)
        return c, dc, v, dv, xp

    for _ in range(int(max_iter)):
        c, dc, v, dv, _xp = evaluate(x)
        c0 = c if c0 is None else c0
        hist["f0"].append(c)
        hist["volume"].append(v)
        xn = opt.step(x, c / c0, dc / c0, np.array([v / volume - 1.0]), (dv / volume)[None, :])
        change = float(np.abs(xn - x).max())
        hist["change"].append(change)
        x = xn
        if change < tol:
            break
    c, _dc, v, _dv, xp = evaluate(x)
    return MeshResult(x, xp, c, v, hist)
