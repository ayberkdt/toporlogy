"""Isoparametric mapping, the detJ check, and element kernels (stiffness, mass, conduction, loads, facet terms).

Mapping: ``x(ξ) = Σ_n N_n(ξ) X_n``, ``J = ∂x/∂ξ`` (dim × dim), ``∂N/∂x = ∂N/∂ξ · J⁻¹``, ``dV = w · detJ``.
detJ is checked at every point of the rule in use and at the reference nodes (:func:`check_jacobian`); the sign is
kept, never its absolute value: an inverted cell raises :class:`InvertedElementError`, a collapsed one
:class:`DegenerateElementError`, both naming the cells.

Voigt order and strain convention of :mod:`toporlogy.core.tensor`: 3-D ``[εx, εy, εz, γyz, γxz, γxy]``, 2-D
``[εx, εy, γxy]`` (engineering shear). Element DOFs are node-major: (u, v[, w]) per node in local node order.
2-D integrals carry the thickness ``t`` (plane stress or plane strain is decided by the D the caller passes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.fem.mesh.quadrature import QuadratureRule, rule
from toporlogy.fem.mesh.reference import ReferenceElement, get

__all__ = ["InvertedElementError", "DegenerateElementError", "DETJ_TOL", "Geometry", "geometry", "check_jacobian",
           "element_scale", "strain_matrix", "stiffness", "mass", "lumped_mass_hrz", "conduction", "body_load",
           "FacetGeometry", "facet_geometry", "facet_load", "facet_mass", "facet_reference", "gradients",
           "stiffness_degree"]

DETJ_TOL = 1e-12
"""detJ < −DETJ_TOL · s_e is an inverted cell, |detJ| ≤ DETJ_TOL · s_e a degenerate one; s_e = (longest edge)^dim."""


class InvertedElementError(ValueError):
    """A cell with negative detJ somewhere (inverted or folded node order)."""

    code = "inverted_element"

    def __init__(self, msg: str, elements: np.ndarray) -> None:
        super().__init__(msg)
        self.elements = elements


class DegenerateElementError(ValueError):
    """A cell with detJ ≈ 0 somewhere (collapsed nodes, zero area/volume)."""

    code = "degenerate_element"

    def __init__(self, msg: str, elements: np.ndarray) -> None:
        super().__init__(msg)
        self.elements = elements


_CELL_EDGES = {"line": ((0, 1),), "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
               "hex": ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6),
                       (3, 7)),
               "tri": ((0, 1), (1, 2), (2, 0)), "tet": ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))}


def element_scale(ref: ReferenceElement, X: np.ndarray) -> np.ndarray:
    """(ne,) s_e = (longest straight vertex-to-vertex edge)^dim of each cell (X: (ne, n_nodes, dim))."""
    e = np.array(_CELL_EDGES[ref.cell])
    L = np.linalg.norm(X[:, e[:, 0]] - X[:, e[:, 1]], axis=-1).max(axis=1)
    return L ** ref.dim


def check_jacobian(detJ: np.ndarray, scale: np.ndarray, cells: np.ndarray | None = None, what: str = "") -> None:
    """Raise if any detJ (ne, npts) is negative or ≈ 0 relative to the cell scale (ne,). ``cells``: global cell
    numbers for the message (default 0 … ne − 1)."""
    ids = np.arange(len(detJ)) if cells is None else np.asarray(cells)
    rel = detJ / scale[:, None]
    neg = (rel < -DETJ_TOL).any(axis=1)
    if neg.any():
        bad = ids[neg]
        raise InvertedElementError(f"{int(neg.sum())} inverted cell(s){what} (detJ < 0 at a quadrature point or "
                                   f"node), first: {bad[:10].tolist()}; min detJ/s_e = {rel.min():.3e}", bad)
    deg = (np.abs(rel) <= DETJ_TOL).any(axis=1)
    if deg.any():
        bad = ids[deg]
        raise DegenerateElementError(f"{int(deg.sum())} degenerate cell(s){what} (|detJ| ≤ {DETJ_TOL:g} s_e), "
                                     f"first: {bad[:10].tolist()}", bad)


@dataclass(frozen=True)
class Geometry:
    """Mapped quadrature data of a set of cells: points x (ne, nq, dim), J (ne, nq, dim, dim), detJ (ne, nq),
    physical gradients dNdx (ne, nq, n_nodes, dim), N (nq, n_nodes) and dV = w detJ (ne, nq)."""

    ref: ReferenceElement
    rule: QuadratureRule
    x: np.ndarray
    J: np.ndarray
    detJ: np.ndarray
    dNdx: np.ndarray
    N: np.ndarray
    dV: np.ndarray


def _jacobian(X: np.ndarray, dN: np.ndarray) -> np.ndarray:
    """J[e, q, i, j] = Σ_n X[e, n, i] dN[q, n, j] (batched matrix products)."""
    return np.swapaxes(X, 1, 2)[:, None] @ dN[None]


def geometry(ref: ReferenceElement, X: np.ndarray, q: QuadratureRule, cells: np.ndarray | None = None,
             check: bool = True) -> Geometry:
    """Map rule ``q`` onto the cells with node coordinates X (ne, n_nodes, dim); detJ checked unless ``check`` is
    False (only for tests that need the raw values)."""
    X = np.asarray(X, float)
    if X.ndim != 3 or X.shape[1:] != (ref.n_nodes, ref.dim):
        raise ValueError(f"{ref.name}: node coordinates must be (ne, {ref.n_nodes}, {ref.dim}), got {X.shape}")
    N = ref.shape(q.points)
    dN = ref.grad(q.points)
    J = _jacobian(X, dN)
    detJ = np.linalg.det(J)
    if check:
        Jn = _jacobian(X, ref.grad(ref.nodes))
        check_jacobian(np.concatenate([detJ, np.linalg.det(Jn)], axis=1), element_scale(ref, X), cells)
    invJ = np.linalg.inv(J)
    dNdx = dN[None] @ invJ                                     # (ne, nq, n, dim)
    x = N[None] @ X                                            # (ne, nq, dim)
    return Geometry(ref, q, x, J, detJ, dNdx, N, q.weights[None, :] * detJ)


def stiffness_degree(ref: ReferenceElement) -> int:
    """Quadrature degree that integrates Bᵀ D B exactly on affine simplices / parallelograms."""
    return 2 * ref.degree if ref.kind == "Q" else 2 * (ref.degree - 1)


def strain_matrix(dNdx: np.ndarray) -> np.ndarray:
    """B (..., k, n_nodes · dim) from physical gradients (..., n_nodes, dim); k = 3 (2-D) or 6 (3-D)."""
    *lead, n, dim = dNdx.shape
    gx = dNdx[..., 0]
    gy = dNdx[..., 1]
    if dim == 2:
        B = np.zeros((*lead, 3, 2 * n))
        B[..., 0, 0::2] = gx
        B[..., 1, 1::2] = gy
        B[..., 2, 0::2] = gy
        B[..., 2, 1::2] = gx
        return B
    gz = dNdx[..., 2]
    B = np.zeros((*lead, 6, 3 * n))
    B[..., 0, 0::3] = gx
    B[..., 1, 1::3] = gy
    B[..., 2, 2::3] = gz
    B[..., 3, 1::3] = gz
    B[..., 3, 2::3] = gy
    B[..., 4, 0::3] = gz
    B[..., 4, 2::3] = gx
    B[..., 5, 0::3] = gy
    B[..., 5, 1::3] = gx
    return B


def _per_cell(D: np.ndarray, ne: int, k: int) -> np.ndarray:
    D = np.asarray(D, float)
    if D.shape == (k, k):
        return np.broadcast_to(D, (ne, k, k))
    if D.shape == (ne, k, k):
        return D
    raise ValueError(f"D must be ({k}, {k}) or ({ne}, {k}, {k}), got {D.shape}")


def stiffness(g: Geometry, D: np.ndarray, thickness: float = 1.0) -> np.ndarray:
    """Element stiffness (ne, nloc, nloc), ``Σ_q Bᵀ D B dV`` (· t in 2-D); D (k, k) or one per cell (ne, k, k)."""
    dim = g.ref.dim
    B = strain_matrix(g.dNdx)                                    # (ne, nq, k, nloc)
    Dc = _per_cell(D, B.shape[0], B.shape[2])
    w = g.dV * (thickness if dim == 2 else 1.0)
    K = np.einsum("eqki,ekl,eqlj,eq->eij", B, Dc, B, w, optimize=True)
    return 0.5 * (K + np.swapaxes(K, 1, 2))


def mass(g: Geometry, rho: float | np.ndarray = 1.0, ncomp: int | None = None, thickness: float = 1.0) -> np.ndarray:
    """Consistent mass (ne, nloc, nloc), ``ρ Σ_q N Nᵀ dV`` per component (node-major DOFs); ``ncomp`` defaults to
    the spatial dimension; ρ scalar or one per cell."""
    ncomp = g.ref.dim if ncomp is None else int(ncomp)
    r = np.broadcast_to(np.asarray(rho, float), (g.dV.shape[0],))
    w = g.dV * r[:, None] * (thickness if g.ref.dim == 2 else 1.0)
    m = np.einsum("qa,qb,eq->eab", g.N, g.N, w)
    return np.einsum("eab,ij->eaibj", m, np.eye(ncomp)).reshape(len(m), g.ref.n_nodes * ncomp, -1)


def lumped_mass_hrz(M: np.ndarray) -> np.ndarray:
    """Hinton–Rock–Zienkiewicz (1976) lumping of consistent element masses (ne, n, n): the diagonal, scaled so that
    each element keeps its total mass (sum of all entries). Positive for every Lagrange element, unlike row sums of
    T6 / T10, whose vertex rows sum to zero or less. Returns (ne, n)."""
    d = np.diagonal(M, axis1=1, axis2=2)
    return d * (M.sum(axis=(1, 2)) / d.sum(axis=1))[:, None]


def conduction(g: Geometry, k: float | np.ndarray = 1.0, thickness: float = 1.0) -> np.ndarray:
    """Element conduction (ne, n, n), ``Σ_q ∇N κ ∇Nᵀ dV``; κ scalar, one per cell (ne,), or a tensor (dim, dim) /
    (ne, dim, dim)."""
    ne, dim = g.dV.shape[0], g.ref.dim
    kk = np.asarray(k, float)
    if kk.ndim == 0 or kk.shape == (ne,):
        kt = np.broadcast_to(kk, (ne,))[:, None, None] * np.eye(dim)
    else:
        kt = _per_cell(kk, ne, dim)
    w = g.dV * (thickness if dim == 2 else 1.0)
    return np.einsum("eqai,eij,eqbj,eq->eab", g.dNdx, kt, g.dNdx, w, optimize=True)


def _values(f: Any, x: np.ndarray, ncomp: int) -> np.ndarray:
    """Evaluate a constant (ncomp,) or a callable x (..., dim) → (..., ncomp) at points x (ne, nq, dim)."""
    if callable(f):
        return np.asarray(f(x.reshape(-1, x.shape[-1])), float).reshape(x.shape[0], x.shape[1], ncomp)
    return np.broadcast_to(np.asarray(f, float).reshape(ncomp), (*x.shape[:2], ncomp))


def body_load(g: Geometry, b: Any, ncomp: int | None = None, thickness: float = 1.0) -> np.ndarray:
    """Consistent element loads (ne, nloc) of a body force b (constant vector or callable of x)."""
    ncomp = g.ref.dim if ncomp is None else int(ncomp)
    v = _values(b, g.x, ncomp)
    w = g.dV * (thickness if g.ref.dim == 2 else 1.0)
    return np.einsum("qa,eqi,eq->eai", g.N, v, w).reshape(len(w), -1)


def gradients(g: Geometry, Ue: np.ndarray) -> np.ndarray:
    """∇u at the quadrature points (ne, nq, ncomp, dim) of element nodal values Ue (ne, n_nodes, ncomp)."""
    return np.einsum("eni,eqnj->eqij", Ue, g.dNdx)


# ---------------------------------------------------------------------------------------------- facets
@dataclass(frozen=True)
class FacetGeometry:
    """Mapped quadrature data on facets: points x (nf, nq, dim), outward unit normals n (nf, nq, dim), surface
    measure dS = w |J_s| (nf, nq), facet shape functions N (nq, nfn)."""

    ref: ReferenceElement
    x: np.ndarray
    normal: np.ndarray
    dS: np.ndarray
    N: np.ndarray


def facet_geometry(facet_ref: ReferenceElement, Xf: np.ndarray, cell_centroid: np.ndarray,
                   degree: int) -> FacetGeometry:
    """Facet points (nf, nfn, dim) of a dim-D mesh, facet element ``facet_ref`` (dimension dim − 1); the normal is
    oriented away from ``cell_centroid`` (nf, dim) of the owning cell."""
    q = rule(facet_ref.cell, degree)
    N = facet_ref.shape(q.points)
    dN = facet_ref.grad(q.points)                              # (nq, nfn, dim - 1)
    T = np.einsum("fni,qnj->fqij", Xf, dN)                     # tangents (nf, nq, dim, dim - 1)
    dim = Xf.shape[2]
    if dim == 2:
        t = T[..., 0]
        nrm = np.stack([t[..., 1], -t[..., 0]], axis=-1)
    else:
        nrm = np.cross(T[..., 0], T[..., 1])
    js = np.linalg.norm(nrm, axis=-1)
    L = np.linalg.norm(Xf[:, :, None, :] - Xf[:, None, :, :], axis=-1).max(axis=(1, 2))
    if (js <= DETJ_TOL * (L ** (dim - 1))[:, None]).any():
        bad = np.flatnonzero((js <= DETJ_TOL * (L ** (dim - 1))[:, None]).any(axis=1))
        raise DegenerateElementError(f"{len(bad)} degenerate facet(s), first: {bad[:10].tolist()}", bad)
    n = nrm / js[..., None]
    x = np.einsum("qn,fni->fqi", N, Xf)
    side = np.einsum("fqi,fqi->fq", n, x - cell_centroid[:, None, :])
    n = np.where((side < 0)[..., None], -n, n)
    return FacetGeometry(facet_ref, x, n, q.weights[None, :] * js, N)


def facet_load(fg: FacetGeometry, traction: Any = None, pressure: Any = None, ncomp: int | None = None,
               thickness: float = 1.0) -> np.ndarray:
    """Consistent facet loads (nf, nfn · ncomp) of a traction vector t or a pressure p (t = −p n, positive
    pressure pushes into the body); constants or callables of x. A scalar flux uses ``traction`` with ncomp = 1."""
    dim = fg.x.shape[2]
    ncomp = dim if ncomp is None else int(ncomp)
    if (traction is None) == (pressure is None):
        raise ValueError("give exactly one of traction, pressure")
    if pressure is not None:
        if ncomp != dim:
            raise ValueError("pressure acts on a vector field (ncomp = dim)")
        p = _values(pressure, fg.x, 1)[..., 0]
        v = -p[..., None] * fg.normal
    else:
        v = _values(traction, fg.x, ncomp)
    w = fg.dS * (thickness if dim == 2 else 1.0)
    return np.einsum("qa,fqi,fq->fai", fg.N, v, w).reshape(len(w), -1)


def facet_mass(fg: FacetGeometry, coef: float = 1.0, thickness: float = 1.0) -> np.ndarray:
    """Scalar facet matrix ``coef Σ_q N Nᵀ dS`` (nf, nfn, nfn) (Robin / convection term h ∫ T v)."""
    w = fg.dS * coef * (thickness if fg.x.shape[2] == 2 else 1.0)
    return np.einsum("qa,qb,fq->fab", fg.N, fg.N, w)


def facet_reference(ref: ReferenceElement) -> ReferenceElement:
    """Reference element of the facets of ``ref``."""
    if not ref.facet_element:
        raise ValueError(f"{ref.name} has no facet element")
    return get(ref.facet_element)

