"""A general mesh inside :class:`~toporlogy.optimization.study.Study` (``docs/olcumler/ag_topoloji_onkayit.md``,
MD1-MD5; design: ``docs/olcumler/tasarim_genel_ag_study.md``).

``Study`` reads a small set of domain attributes (``shape``, ``active_elements``, ``n_design``, ``frac``, ``ndof``,
``uniform_start``). :class:`MeshDomain` offers them for the cells of a :class:`~toporlogy.fem.mesh.Mesh`, presented as
a 1-D "grid" of ``n_cells`` (all active and full, no keep or void cells), so ``Study`` runs unchanged with
:class:`~toporlogy.design.chain.Embed` and :class:`~toporlogy.design.chain.Passive` on it, plus:

* :class:`MeshFilterLink`: the centroid density filter of :mod:`~toporlogy.optimization.mesh_topology` as a chain link;
* :class:`MeshElasticity`: linear elasticity, ``solve(E, F)`` → U with an autograd backward (self-adjoint);
* :class:`MeshVolume`: volume fraction weighted by the cell volumes (a :class:`Volume`, so the uniform start applies).

:class:`MeshFilterLink` and :class:`MeshVolume` read only the domain's ``centroids`` and ``cell_volume``, so they also
serve the shell facets of :class:`~toporlogy.optimization.shell_study.ShellDomain`.

Grid-only links and responses (the grid filters, AM, casting, symmetry tying, multi-phase interpolations, plain
``Volume``, local volume, stress, stress recovery, connectivity, mass properties, robust realisations) refuse a
``MeshDomain`` with a ``ValueError`` naming the component and its mesh counterpart where one exists
(:func:`toporlogy.core.domain.require_grid`; ``docs/olcumler/ag_topoloji_onkayit.md``, RF1-RF3). No YAML yet.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from toporlogy.design.chain import _SparseApply
from toporlogy.fem.mesh import ElasticProblem, Mesh
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.optimization.mesh_topology import MeshDensityFilter
from toporlogy.optimization.responses import EvalContext, Volume

__all__ = ["MeshDomain", "MeshFilterLink", "MeshElasticity", "MeshVolume"]


class _Cells(Protocol):
    """What :class:`MeshFilterLink` and :class:`MeshVolume` read from a domain: centroids (n, d), measures (n,)."""

    centroids: np.ndarray
    cell_volume: np.ndarray


class MeshDomain:
    """The cells of ``mesh`` as the design domain of a Study (a 1-D grid of n_cells, every cell active and full).
    ``cell_volume``: measure of each cell (area in 2-D without thickness, volume in 3-D); ``centroids``: their
    centroids (n_cells, dim)."""

    def __init__(self, mesh: Mesh) -> None:
        n = int(mesh.n_cells)
        self.mesh, self.dim = mesh, int(mesh.dim)
        self.shape: tuple[int, ...] = (n,)
        self.ne = self.n_active = self.n_design = n
        self.active = np.ones(n, bool)
        self.design = np.ones(n, bool)
        self.keep = np.zeros(n, bool)
        self.void = np.zeros(n, bool)
        self.frac = np.ones(n)
        self.active_elements = np.arange(n, dtype=np.int64)
        self.design_elements = np.arange(n, dtype=np.int64)
        self.ndof = int(mesh.n_nodes) * self.dim
        k = 3 if self.dim == 2 else 6
        geo = ElasticProblem(mesh, np.eye(k)).geom                   # geometry only (D unused)
        self.cell_volume = np.asarray(geo.dV, float).sum(axis=1)
        self.centroids = np.asarray(mesh.cell_centroids(), float)

    def uniform_start(self, volume_fraction: float, weighted: bool = False) -> np.ndarray:
        return np.full(self.n_design, float(min(max(volume_fraction, 0.0), 1.0)))

    def __repr__(self) -> str:
        return f"MeshDomain(cells={self.n_design}, nodes={self.mesh.n_nodes}, dim={self.dim})"


class MeshFilterLink(torch.nn.Module):
    """Chain link x̃ = H x / Hs of :class:`~toporlogy.optimization.mesh_topology.MeshDensityFilter` on the domain's
    ``centroids`` (``rmin`` in length units); autograd backward Hᵀ g."""

    def __init__(self, domain: _Cells, rmin: float) -> None:
        super().__init__()
        self.filter = MeshDensityFilter(domain.centroids, rmin)
        self._H, self._Ht = self.filter.matrix, self.filter.matrix.T.tocsr()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = _SparseApply.apply(x, self._H, self._Ht)
        return out


class _MeshSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E: torch.Tensor, phys: MeshElasticity, F: np.ndarray) -> torch.Tensor:
        Ev = E.detach().cpu().double().numpy()
        U, fact = phys.solve_numpy(Ev, F)
        ctx.phys, ctx.fact, ctx.U, ctx.dtype = phys, fact, U, E.dtype
        return torch.as_tensor(U, dtype=E.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, None, None]:
        phys = ctx.phys
        G = g.detach().cpu().double().numpy().reshape(ctx.U.shape)
        lam = np.zeros_like(G)
        fr = phys.free
        lam[fr] = ctx.fact.solve(G[fr]).reshape(len(fr), -1)       # K λ = g (K symmetric)
        Ue, Le = ctx.U[phys.cell_dofs], lam[phys.cell_dofs]          # (ne, nloc, nc)
        dE = -np.einsum("eac,eab,ebc->e", Le, phys.ke, Ue)
        return torch.as_tensor(dE, dtype=ctx.dtype), None, None


class MeshElasticity:
    """Study physics: K(E) u = F on ``domain.mesh`` with K(E) = Σ_e E_e K_e (unit-modulus ``D``), zero displacements
    on ``fixed``. ``solve(E, F)`` → U (ndof, n_cases), differentiable in E."""

    name = "mesh_elasticity"

    def __init__(self, domain: MeshDomain, D: np.ndarray, fixed: Any, thickness: float = 1.0) -> None:
        self.domain = domain
        self.problem = ElasticProblem(domain.mesh, np.asarray(D, float), thickness=thickness)
        self.ke = np.asarray(self.problem.element_stiffness(), float)
        self.cell_dofs = np.asarray(self.problem.cell_dofs)
        self.free = np.setdiff1d(self.problem.active, np.unique(np.asarray(fixed, np.int64)))
        self.n_solves = 0

    def solve_numpy(self, E: np.ndarray, F: Any) -> tuple[np.ndarray, Factorization]:
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.shape[0] != self.problem.ndof:
            raise ValueError(f"loads have {F2.shape[0]} rows, the mesh {self.problem.ndof} DOFs")
        K = self.problem.pattern.assemble(np.asarray(E, float).reshape(-1)[:, None, None] * self.ke)
        fr = self.free
        fact = Factorization(K[fr][:, fr], "spd")
        U = np.zeros_like(F2)
        U[fr] = fact.solve(F2[fr]).reshape(len(fr), -1)
        self.n_solves += 1
        return U, fact

    def solve(self, E: torch.Tensor, F: Any) -> torch.Tensor:
        out: torch.Tensor = _MeshSolve.apply(E, self, np.asarray(F, float))
        return out


class MeshVolume(Volume):
    """Volume fraction Σ V_e x̄_e / Σ V_e with the domain's ``cell_volume`` (cell volumes of a :class:`MeshDomain`,
    facet areas of a shell domain)."""

    def __init__(self, domain: _Cells) -> None:
        super().__init__(weighted=True)
        v = np.asarray(domain.cell_volume, float)
        self._w = torch.as_tensor(v / v.sum(), dtype=torch.float64)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        x = ctx.xphys.reshape(-1)
        return (self._w.to(x.dtype) * x).sum()
