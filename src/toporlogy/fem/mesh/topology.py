"""Mesh validation, hanging-node detection and cell / node renumbering with field permutation.

:func:`validate_mesh` only reports; nothing is merged, flipped or repaired. Findings (all ids are 0-based):

* ``duplicate_nodes`` (m, 2): pairs of nodes closer than ``node_tol`` times the point-cloud diameter, and
  ``duplicate_node_same_piece`` (m,) telling whether both nodes belong to the same connected piece (a coincident pair
  in different pieces may be deliberate disconnected geometry; in one piece it is a defect);
* ``duplicate_cells`` (m, 2): (first, later) cells with the same node set (any node order);
* ``repeated_node_cells``: cells that name a node twice;
* ``nonmanifold_facets``: one (k, 2) array of (cell, local facet) per facet shared by more than two cells;
* ``unused_nodes``; ``negative_cells`` (detJ < 0 at a sample, by cell id) and ``degenerate_cells`` (|detJ| ≈ 0);
* ``hanging_nodes`` (m, 3) rows (cell, local facet, node), 2-D only: a vertex on the interior of a boundary facet,
  i.e. a non-conforming interface (no multi-point constraints exist to resolve it); ``None`` in 3-D (NOT CHECKED).

A node index outside the point list is refused by :class:`~toporlogy.fem.mesh.mesh.Mesh` itself.

:func:`renumber` returns the permuted mesh and a :class:`Renumbering` that maps node-major DOF vectors and per-cell
fields in both directions. New cell k is old cell ``cell_perm[k]``, new node j is old node ``node_perm[j]``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from toporlogy.fem.mesh.kernels import DETJ_TOL
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.quality import sampled_detj

__all__ = ["MeshValidationError", "MeshReport", "validate_mesh", "require_valid", "hanging_nodes", "Renumbering",
           "renumber"]


class MeshValidationError(ValueError):
    """The mesh has structural defects; ``report`` holds them with ids."""

    code = "invalid_mesh"

    def __init__(self, msg: str, report: MeshReport) -> None:
        super().__init__(msg)
        self.report = report


@dataclass(frozen=True)
class MeshReport:
    duplicate_nodes: np.ndarray
    duplicate_node_same_piece: np.ndarray
    duplicate_cells: np.ndarray
    repeated_node_cells: np.ndarray
    nonmanifold_facets: list[np.ndarray]
    unused_nodes: np.ndarray
    negative_cells: np.ndarray
    degenerate_cells: np.ndarray
    hanging_nodes: np.ndarray | None

    @property
    def ok(self) -> bool:
        return not self.summary()

    def summary(self) -> str:
        """Defect list with ids, empty if there is none."""
        out = []
        if len(self.duplicate_nodes):
            out.append(f"coincident nodes {self.duplicate_nodes[:10].tolist()}")
        if len(self.duplicate_cells):
            out.append(f"duplicate cells (first, later) {self.duplicate_cells[:10].tolist()}")
        if len(self.repeated_node_cells):
            out.append(f"cells with a repeated node {self.repeated_node_cells[:10].tolist()}")
        if self.nonmanifold_facets:
            out.append(f"{len(self.nonmanifold_facets)} facet(s) shared by more than two cells, first: "
                       f"{self.nonmanifold_facets[0].tolist()}")
        if len(self.unused_nodes):
            out.append(f"unused nodes {self.unused_nodes[:10].tolist()}")
        if len(self.negative_cells):
            out.append(f"negative orientation (detJ < 0) in cells {self.negative_cells[:10].tolist()}")
        if len(self.degenerate_cells):
            out.append(f"degenerate (detJ ≈ 0) cells {self.degenerate_cells[:10].tolist()}")
        if self.hanging_nodes is not None and len(self.hanging_nodes):
            out.append(f"hanging nodes (cell, facet, node) {self.hanging_nodes[:10].tolist()}")
        return "; ".join(out)


def _pieces(mesh: Mesh) -> np.ndarray:
    """Connected-piece label of every node (−1 for unused nodes)."""
    _, lab = mesh.components()
    out = np.full(mesh.n_nodes, -1, np.int64)
    out[mesh.cells.ravel()] = np.repeat(lab, mesh.cells.shape[1])
    return out


def hanging_nodes(mesh: Mesh, tol: float = 1e-9) -> np.ndarray:
    """(m, 3) rows (cell, local facet, node) of used vertices lying strictly inside a boundary facet (2-D meshes),
    within ``tol`` times the facet's vertex-to-vertex length."""
    if mesh.dim != 2:
        raise ValueError("hanging-node detection is implemented for 2-D meshes only")
    ref = mesh.ref
    b = mesh.boundary_facets()
    fn = mesh.facet_nodes(b)
    a, c = mesh.points[fn[:, 0]], mesh.points[fn[:, 1]]
    vert = np.unique(mesh.cells[:, :ref.n_vertices])
    P = mesh.points[vert]
    rows = []
    for k in range(len(b)):
        e = c[k] - a[k]
        L2 = float(e @ e)
        if L2 == 0.0:
            continue
        t = (P - a[k]) @ e / L2
        dist = np.linalg.norm(P - (a[k] + t[:, None] * e), axis=1)
        sel = (dist <= tol * np.sqrt(L2)) & (t > tol) & (t < 1.0 - tol)
        for nd in vert[sel]:
            rows.append((b[k, 0], b[k, 1], nd))
    return np.array(rows, np.int64).reshape(-1, 3)


def validate_mesh(mesh: Mesh, node_tol: float = 1e-9, hanging: bool = True) -> MeshReport:
    """Report structural defects of ``mesh`` (see the module docstring); never modifies the mesh."""
    P = mesh.points
    diam = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) if len(P) else 0.0
    pairs = np.array(sorted(cKDTree(P).query_pairs(node_tol * diam)), np.int64).reshape(-1, 2) if len(P) else \
        np.zeros((0, 2), np.int64)
    piece = _pieces(mesh)
    same = (piece[pairs[:, 0]] == piece[pairs[:, 1]]) & (piece[pairs[:, 0]] >= 0) if len(pairs) else \
        np.zeros(0, bool)
    key = np.sort(mesh.cells, axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    later = np.flatnonzero(first[inv] != np.arange(mesh.n_cells))
    dup_cells = np.column_stack([first[inv[later]], later]) if len(later) else np.zeros((0, 2), np.int64)
    rep = np.flatnonzero((key[:, 1:] == key[:, :-1]).any(axis=1))
    cf, fkey = mesh._all_facets()
    uk, finv, cnt = np.unique(fkey, axis=0, return_inverse=True, return_counts=True)
    finv = finv.ravel()
    nonman = [cf[finv == g] for g in np.flatnonzero(cnt > 2)]
    unused = np.setdiff1d(np.arange(mesh.n_nodes), mesh.used_nodes())
    dj, s = sampled_detj(mesh)
    rel = dj / s[:, None]
    neg = np.flatnonzero((rel < -DETJ_TOL).any(axis=1))
    deg = np.flatnonzero(((np.abs(rel) <= DETJ_TOL).any(axis=1)) & ~(rel < -DETJ_TOL).any(axis=1))
    hang = hanging_nodes(mesh) if (hanging and mesh.dim == 2) else None
    return MeshReport(pairs, same, dup_cells, rep, nonman, unused, neg, deg, hang)


def require_valid(mesh: Mesh, node_tol: float = 1e-9) -> MeshReport:
    """:func:`validate_mesh`, raising :class:`MeshValidationError` naming the ids if anything is found."""
    rep = validate_mesh(mesh, node_tol)
    if not rep.ok:
        raise MeshValidationError(f"invalid mesh: {rep.summary()}", rep)
    return rep


# ---------------------------------------------------------------------------------------------- renumbering
def _check_perm(p: np.ndarray | None, n: int, what: str) -> np.ndarray:
    if p is None:
        return np.arange(n)
    p = np.asarray(p, np.int64)
    if p.shape != (n,) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError(f"{what} must be a permutation of 0 … {n - 1}")
    return p


@dataclass(frozen=True)
class Renumbering:
    """New → old (``cell_perm``, ``node_perm``) and old → new (``cell_inv``, ``node_inv``) numbering."""

    cell_perm: np.ndarray
    node_perm: np.ndarray
    cell_inv: np.ndarray
    node_inv: np.ndarray

    def _nodal(self, values: Any, perm: np.ndarray, ncomp: int | None) -> np.ndarray:
        v = np.asarray(values)
        n = len(perm)
        k = 1 if ncomp is None else int(ncomp)
        if v.shape[0] != n * k:
            raise ValueError(f"nodal field must have {n * k} rows (n_nodes × ncomp), got {v.shape[0]}")
        return v.reshape(n, k, *v.shape[1:])[perm].reshape(v.shape)

    def to_new_nodal(self, values: Any, ncomp: int | None = None) -> np.ndarray:
        """Old-numbered node-major field (n_nodes·ncomp, ...) in the new numbering."""
        return self._nodal(values, self.node_perm, ncomp)

    def to_old_nodal(self, values: Any, ncomp: int | None = None) -> np.ndarray:
        """New-numbered node-major field (n_nodes·ncomp, ...) back in the old numbering."""
        return self._nodal(values, self.node_inv, ncomp)

    def to_new_cell(self, values: Any) -> np.ndarray:
        return np.asarray(values)[self.cell_perm]

    def to_old_cell(self, values: Any) -> np.ndarray:
        return np.asarray(values)[self.cell_inv]


def renumber(mesh: Mesh, cell_perm: np.ndarray | None = None,
             node_perm: np.ndarray | None = None) -> tuple[Mesh, Renumbering]:
    """The mesh with new cell k = old cell ``cell_perm[k]`` and new node j = old node ``node_perm[j]`` (defaults:
    identity), tags, facet sets and node sets carried along; local node and facet order of every cell is unchanged."""
    cp = _check_perm(cell_perm, mesh.n_cells, "cell_perm")
    npm = _check_perm(node_perm, mesh.n_nodes, "node_perm")
    ci = np.empty_like(cp)
    ci[cp] = np.arange(len(cp))
    ni = np.empty_like(npm)
    ni[npm] = np.arange(len(npm))
    out = replace(mesh, points=mesh.points[npm], cells=ni[mesh.cells[cp]], cell_tags=mesh.cell_tags[cp],
                  facet_sets={k: np.column_stack([ci[v[:, 0]], v[:, 1]]) if len(v) else v.copy()
                              for k, v in mesh.facet_sets.items()},
                  node_sets={k: ni[v] for k, v in mesh.node_sets.items()})
    return out, Renumbering(cp, npm, ci, ni)
