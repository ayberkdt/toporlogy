"""Periodic homogenisation of a representative volume element (RVE) on the regular grid.

The whole box :class:`~toporlogy.core.domain.Domain` is one periodic cell (every element active and full; the material
is the element modulus field, so a void is an element of small modulus). For a macroscopic strain ε̄ the displacement
is u = ε̄ x + ũ with a periodic fluctuation ũ: the nodes on opposite faces share their degrees of freedom (a
master-slave map P from all nodes to the ``prod(shape)`` independent ones), and one node is held to remove the
translation. With K the stiffness of the cell,

    Pᵀ K P ũ_r = −Pᵀ K u_aff,       u = u_aff + P ũ_r,       C_ij = (1/|Y|) u_iᵀ K u_j

for the unit strain cases i, j (Voigt order of the library, engineering shear strains: [xx, yy, xy] in 2-D,
[xx, yy, zz, yz, xz, xy] in 3-D). |Y| is the cell volume (2-D: area × thickness). The conductivity is the same with
the temperature gradient (k_ij = (1/|Y|) T_iᵀ K T_j). The energy form is self-adjoint, so the derivative with respect
to an element modulus is dC_ij/dE_e = (1/|Y|) u_i^eᵀ K_e⁰ u_j^e with the unit-modulus element matrix K_e⁰.

``bc="kubc"`` gives the kinematic bound instead (u = ε̄ x on the whole boundary): it is stiffer than the periodic
value in the Loewner order. Pre-registered checks: ``docs/olcumler/homojenlestirme_onkayit.md`` (H1-H9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import D_iso, D_plane_stress, element_matrix_from_D
from toporlogy.fem.heat import conduction_matrix, element_nodes

__all__ = ["Homogenized", "homogenize_elasticity", "homogenize_conduction", "periodic_map", "BCS"]

BCS = ("periodic", "kubc")


@dataclass
class Homogenized:
    """Effective tensor ``C`` (Voigt, or the conductivity matrix), the total fields of the unit cases (ndof, k) and
    the element sensitivities ``dC[e, i, j]`` = dC_ij / dE_e (None unless asked for)."""

    C: np.ndarray
    fields: np.ndarray
    volume: float
    bc: str
    dC: np.ndarray | None = None


def _check_cell(domain: Domain) -> None:
    full = domain.n_active == int(np.prod(domain.shape)) and not np.any(np.asarray(domain.void))
    if not full or np.any(np.asarray(domain.frac) != 1.0):
        raise ValueError("homogenisation needs the full box as the periodic cell: every element active and full "
                         "(no void region, no partial volume); give a small modulus to void elements instead")
    if min(domain.shape) < 2:
        raise ValueError(f"homogenisation needs at least 2 elements along every axis, got shape {domain.shape}")


def _grid_index(domain: Domain) -> np.ndarray:
    X = np.asarray(domain.node_coords(), float)
    return np.rint((X - X.min(axis=0)) / domain.h).astype(np.int64)


def periodic_map(domain: Domain) -> np.ndarray:
    """Independent node (0 … prod(shape)−1) of every grid node: the node at index i is the one at i mod shape."""
    idx = _grid_index(domain) % np.asarray(domain.shape)
    return np.ravel_multi_index(tuple(idx.T), tuple(domain.shape))


def _volume(domain: Domain) -> float:
    v = float(np.prod(domain.shape)) * domain.h ** domain.dim
    return v * (domain.thickness if domain.dim == 2 else 1.0)


def _assemble(domain: Domain, KE: np.ndarray, weights: np.ndarray, dofs_per_node: int) -> sp.csr_matrix:
    el = domain.active_elements
    edof = np.asarray(domain.edof)[el] if dofs_per_node == domain.dim else element_nodes(domain, el)
    n = domain.nn * dofs_per_node
    rows = np.repeat(edof, edof.shape[1], axis=1).ravel()
    cols = np.tile(edof, (1, edof.shape[1])).ravel()
    vals = (np.asarray(weights, float)[:, None] * KE.ravel()[None, :]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _unit_cases(domain: Domain, kind: str) -> np.ndarray:
    """Affine fields of the unit macroscopic strains (elasticity) or temperature gradients (conduction)."""
    X = np.asarray(domain.node_coords(), float)
    X = X - X.min(axis=0)
    dim = domain.dim
    if kind == "conduction":
        return X.copy()                                             # T = x_i for the unit gradient along i
    pairs = [(0, 0), (1, 1), (0, 1)] if dim == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    U = np.zeros((domain.nn * dim, len(pairs)))
    for c, (a, b) in enumerate(pairs):
        eps = np.zeros((dim, dim))
        if a == b:
            eps[a, a] = 1.0
        else:
            eps[a, b] = eps[b, a] = 0.5                             # engineering shear γ_ab = 1
        U[:, c] = (X @ eps.T).ravel()
    return U


def _solve(domain: Domain, K: sp.csr_matrix, U_aff: np.ndarray, dpn: int, bc: str) -> np.ndarray:
    n = domain.nn * dpn
    if bc == "periodic":
        master = periodic_map(domain)
        nind = int(np.prod(domain.shape)) * dpn
        rows = np.arange(n)
        cols = (master[:, None] * dpn + np.arange(dpn)[None, :]).ravel()
        P = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(n, nind))
        Kr = (P.T @ K @ P).tocsc()
        keep = np.arange(dpn, nind)                                 # independent node 0 held (translation)
        A = Kr[keep][:, keep]
        lu = spla.splu(A.tocsc())
        out = U_aff.copy()
        R = -(P.T @ (K @ U_aff))
        for c in range(U_aff.shape[1]):
            ur = np.zeros(nind)
            ur[keep] = lu.solve(R[keep, c])
            out[:, c] += P @ ur
        return out
    if bc == "kubc":
        idx = _grid_index(domain)
        on_b = np.any((idx == 0) | (idx == np.asarray(domain.shape)), axis=1)
        fixed = (np.flatnonzero(on_b)[:, None] * dpn + np.arange(dpn)[None, :]).ravel()
        free = np.setdiff1d(np.arange(n), fixed)
        Kc = K.tocsc()
        lu = spla.splu(Kc[free][:, free].tocsc())
        out = U_aff.copy()
        rhs = -(Kc[free][:, fixed] @ U_aff[fixed])
        for c in range(U_aff.shape[1]):
            out[free, c] = lu.solve(rhs[:, c])
        return out
    raise ValueError(f"bc: {' | '.join(BCS)}, got {bc!r}")


def _effective(domain: Domain, K: sp.csr_matrix, KE: np.ndarray, dpn: int, kind: str, bc: str,
               sensitivities: bool) -> Homogenized:
    _check_cell(domain)
    U = _solve(domain, K, _unit_cases(domain, kind), dpn, bc)
    vol = _volume(domain)
    C = (U.T @ (K @ U)) / vol
    C = 0.5 * (C + C.T)
    dC = None
    if sensitivities:
        el = domain.active_elements
        edof = np.asarray(domain.edof)[el] if dpn == domain.dim else element_nodes(domain, el)
        Ue = U[edof]                                                # (n_el, nloc, k)
        dC = np.einsum("eai,ab,ebj->eij", Ue, KE, Ue) / vol
    return Homogenized(C, U, vol, bc, dC)


def homogenize_elasticity(domain: Domain, E: np.ndarray, nu: float = 0.3, D: np.ndarray | None = None,
                          element: str | None = None, bc: str = "periodic",
                          sensitivities: bool = False) -> Homogenized:
    """Effective Voigt stiffness of the cell with element moduli ``E`` (n_active,) times the unit-modulus ``D``
    (default isotropic ``nu``; plane stress in 2-D). ``sensitivities``: also dC/dE per element. The element
    sensitivity carries the element's geometry factor (2-D thickness, 3-D h)."""
    dim = domain.dim
    if D is None:
        D = D_plane_stress(nu) if dim == 2 else D_iso(nu)
    element = element or ("Q4" if dim == 2 else "H8")
    geom = domain.thickness if dim == 2 else domain.h
    KE = element_matrix_from_D(dim, element, np.asarray(D, float)) * geom
    E = np.asarray(E, float).reshape(-1)
    if E.shape != (domain.n_active,):
        raise ValueError(f"E must have {domain.n_active} entries (active elements), got {E.shape}")
    K = _assemble(domain, KE, E, dim)
    return _effective(domain, K, KE, dim, "elasticity", bc, sensitivities)


def homogenize_conduction(domain: Domain, k: np.ndarray, bc: str = "periodic",
                          sensitivities: bool = False) -> Homogenized:
    """Effective conductivity matrix (dim × dim) of the cell with element conductivities ``k`` (n_active,)."""
    dim = domain.dim
    geom = domain.thickness if dim == 2 else domain.h
    KE = np.asarray(conduction_matrix(dim), float) * geom
    k = np.asarray(k, float).reshape(-1)
    if k.shape != (domain.n_active,):
        raise ValueError(f"k must have {domain.n_active} entries (active elements), got {k.shape}")
    K = _assemble(domain, KE, k, 1)
    return _effective(domain, K, KE, 1, "conduction", bc, sensitivities)
