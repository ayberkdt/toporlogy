"""Point location by inverse isoparametric map, and interpolation at located points.

For physical points p: candidate cells from a KD-tree on padded per-cell bounding boxes (the box of the cell mapped
on a reference lattice, padded by 10 % of its size plus 1e-9 of the mesh diameter, so a curved cell that bulges
between lattice points is still a candidate), then Newton's method on ``r(ξ) = x(ξ) − p`` from the reference
centroid with step halving, for affine and non-affine cells alike. Let ``g(ξ) = min(ξ₁ … ξ_d, 1 − Σξ)`` (simplices)
or ``1 − max|ξ_i|`` (tensor cells); ``g > tol`` is interior, ``|g| ≤ tol`` on the boundary, ``g < −tol`` outside
(``tol`` in reference coordinates, default 1e-10).

Result per point (deterministic): exactly one interior cell → ``INSIDE``; two or more interior cells (overlapping
cells, an invalid mesh) → ``AMBIGUOUS``, lowest cell id; only boundary hits (a facet, edge or vertex shared by
cells) → ``BOUNDARY``, lowest cell id; no hit → ``OUTSIDE``, cell −1, ξ = NaN. :func:`interpolate` returns NaN
outside: there is no nearest-neighbour extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.reference import ReferenceElement, get

__all__ = ["INSIDE", "BOUNDARY", "OUTSIDE", "AMBIGUOUS", "Location", "inverse_map", "locate", "interpolate"]

INSIDE, BOUNDARY, OUTSIDE, AMBIGUOUS = 0, 1, 2, 3


@dataclass(frozen=True)
class Location:
    """cell (n,) int (−1 outside), xi (n, dim) reference coordinates (NaN outside), status (n,) int8,
    n_hits (n,) number of cells that contain the point (interior or boundary)."""

    cell: np.ndarray
    xi: np.ndarray
    status: np.ndarray
    n_hits: np.ndarray


def _map(ref: ReferenceElement, X: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x (m, dim) and J (m, dim, dim) of m (cell, ξ) pairs: X (m, nn, dim), xi (m, dim)."""
    return np.einsum("mn,mni->mi", ref.shape(xi), X), np.einsum("mni,mnj->mij", X, ref.grad(xi))


def _inside_measure(ref: ReferenceElement, xi: np.ndarray) -> np.ndarray:
    if ref.kind == "Q":
        return 1.0 - np.abs(xi).max(axis=1)
    return np.minimum(xi.min(axis=1), 1.0 - xi.sum(axis=1))


def inverse_map(ref: ReferenceElement, X: np.ndarray, p: np.ndarray,
                max_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """ξ with x(ξ) = p for m (cell, point) pairs; X (m, n_nodes, dim), p (m, dim). Newton from the reference centroid
    with step halving (the residual norm must not grow). Returns (ξ (m, dim), converged (m,)); converged means the
    last step was below 1e-13 and the residual below 1e-9 of the cell size."""
    X = np.asarray(X, float)
    p = np.asarray(p, float)
    m = len(p)
    xi = np.tile(ref.nodes[:ref.n_vertices].mean(axis=0), (m, 1))
    h = np.linalg.norm(X.max(axis=1) - X.min(axis=1), axis=1)
    step = np.full(m, np.inf)
    for _ in range(max_iter):
        x, J = _map(ref, X, xi)
        r = x - p
        det = np.linalg.det(J)
        ok = np.abs(det) > 0
        if not ok.any():
            break
        delta = np.zeros_like(xi)
        delta[ok] = np.linalg.solve(J[ok], r[ok][..., None])[..., 0]
        r0 = np.linalg.norm(r, axis=1)
        lam: np.ndarray = np.ones(m)
        new = xi - delta
        for _ in range(10):
            rn = np.linalg.norm(_map(ref, X, new)[0] - p, axis=1)
            bad = ok & (rn > r0 * (1 + 1e-12)) & (np.linalg.norm(delta, axis=1) > 1e-15)
            if not bad.any():
                break
            lam = np.where(bad, lam * 0.5, lam)
            new = np.where(bad[:, None], xi - lam[:, None] * delta, new)
        step = np.abs(new - xi).max(axis=1)
        xi = new
        if (step <= 1e-14).all():
            break
    r = np.linalg.norm(_map(ref, X, xi)[0] - p, axis=1)
    return xi, (step <= 1e-13) & (r <= 1e-9 * np.maximum(h, 1e-300))


def _cell_boxes(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    ref = mesh.ref
    d = ref.dim
    if ref.kind == "Q":
        g = np.linspace(-1.0, 1.0, 5)
        lat = np.array(list(product(g, repeat=d)))
    else:
        g = np.linspace(0.0, 1.0, 7)
        lat = np.array([c for c in product(g, repeat=d) if sum(c) <= 1.0 + 1e-12])
    x = np.einsum("qn,eni->eqi", ref.shape(lat), mesh.cell_points())
    lo, hi = x.min(axis=1), x.max(axis=1)
    pad = 0.1 * (hi - lo) + 1e-9 * float(np.linalg.norm(mesh.points.max(0) - mesh.points.min(0)))
    return lo - pad, hi + pad


def locate(mesh: Mesh, points: np.ndarray, tol: float = 1e-10) -> Location:
    """Cell and reference coordinates of physical ``points`` (n, dim); see the module docstring."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != mesh.dim:
        raise ValueError(f"points must be (n, {mesh.dim}), got {P.shape}")
    ref = mesh.ref
    n = len(P)
    lo, hi = _cell_boxes(mesh)
    tree = cKDTree(0.5 * (lo + hi))
    rad = float(np.linalg.norm(0.5 * (hi - lo), axis=1).max()) * (1 + 1e-12)
    lists = tree.query_ball_point(P, rad)
    pp: np.ndarray = np.array([(i, c) for i, ls in enumerate(lists) for c in ls], np.int64).reshape(-1, 2)
    inb = ((P[pp[:, 0]] >= lo[pp[:, 1]]) & (P[pp[:, 0]] <= hi[pp[:, 1]])).all(axis=1)
    pp = pp[inb]
    cell = np.full(n, -1, np.int64)
    xi_out = np.full((n, mesh.dim), np.nan)
    status = np.full(n, OUTSIDE, np.int8)
    hits: np.ndarray = np.zeros(n, np.int64)
    if len(pp):
        xi, conv = inverse_map(ref, mesh.cell_points(pp[:, 1]), P[pp[:, 0]])
        g = _inside_measure(ref, xi)
        hit = conv & (g >= -tol)
        interior = hit & (g > tol)
        order = np.lexsort((pp[:, 1], pp[:, 0]))             # by point, then by cell id
        hits = np.bincount(pp[hit, 0], minlength=n).astype(np.int64)
        # per point: choose interior hits first, then boundary hits, lowest cell id (order is ascending)
        n_int = np.bincount(pp[interior, 0], minlength=n)
        for k in order:
            if not hit[k]:
                continue
            i = int(pp[k, 0])
            want_int = n_int[i] > 0
            if cell[i] >= 0 or (want_int and not interior[k]):
                continue
            cell[i] = pp[k, 1]
            xi_out[i] = xi[k]
            status[i] = INSIDE if want_int else BOUNDARY
        status[n_int >= 2] = AMBIGUOUS
    return Location(cell, xi_out, status, hits)


def interpolate(mesh: Mesh, nodal: np.ndarray, loc: Location) -> np.ndarray:
    """Nodal values (n_nodes, ...) interpolated at the located points; NaN where the point is outside."""
    F = np.asarray(nodal, float)
    if F.shape[0] != mesh.n_nodes:
        raise ValueError(f"nodal field must have {mesh.n_nodes} rows, got {F.shape[0]}")
    ref = get(mesh.element)
    out = np.full((len(loc.cell), *F.shape[1:]), np.nan)
    ins = loc.cell >= 0
    if ins.any():
        N = ref.shape(loc.xi[ins])                                        # (m, n_nodes_per_cell)
        out[ins] = np.einsum("mn,mn...->m...", N, F[mesh.cells[loc.cell[ins]]])
    return out
