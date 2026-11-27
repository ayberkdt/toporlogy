"""Periodic homogenisation of a cell given as a general mesh, with design sensitivities.

The whole mesh is one periodic cell with lattice vectors ``a_1 … a_d`` (any, skew allowed). Per cell the material is
``E_e · D`` (elasticity) or ``k_e · κ`` (conduction). For each unit macro strain ε̄ (Voigt order of the library,
engineering shear: [xx, yy, xy] in 2-D, [xx, yy, zz, yz, xz, xy] in 3-D) the displacement is u = H x + w with H the
symmetric strain tensor of ε̄ (H_xy = γ_xy / 2) and w periodic: the rows ``u(x + a) − u(x) = H a`` of
:func:`toporlogy.fem.mesh.constraints.periodic_lattice` plus one pinned node (the translation), solved by elimination
with the dependent corner rows dropped. Then

    C_ij = (1/|Y|) u_iᵀ K u_j,        dC_ij/dE_e = (1/|Y|) u_iᵉᵀ K_e(D) u_jᵉ

with |Y| = |det[a_1 … a_d]| (× thickness in 2-D). The energy form is self-adjoint (u_i minimises the energy under
its constraints), so the derivative needs no adjoint solve. Conduction: T = g·x + w and k_ij likewise. The regular
grid has its own implementation (:mod:`toporlogy.fem.homogenization`); the two are compared in
``tests/toporlogy/test_mesh_homogenization.py``. Preregistration: ``docs/foundations/prereg/MH_mesh_homogenization.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.constraints import LinearConstraints, periodic_lattice, solve_constrained
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import ElasticProblem, HeatProblem

__all__ = ["MeshHomogenized", "homogenize_mesh_elasticity", "homogenize_mesh_conduction"]


@dataclass
class MeshHomogenized:
    """Effective matrix ``C`` (k, k), its derivatives ``dC`` (n_cells, k, k) with respect to the cell factors (or
    None), the total unit-case fields ``u`` (ndof, k) and the cell measure ``volume`` (|Y|, × thickness in 2-D)."""

    C: np.ndarray
    dC: np.ndarray | None
    u: np.ndarray
    volume: float


def _lattice(mesh: Mesh, lattice: Any) -> np.ndarray:
    L = np.atleast_2d(np.asarray(lattice, float))
    if L.shape != (mesh.dim, mesh.dim):
        raise ValueError(f"lattice must be ({mesh.dim}, {mesh.dim}) (one vector per row), got {L.shape}")
    det = float(np.linalg.det(L))
    if not np.isfinite(det) or abs(det) <= 1e-12 * float(np.prod(np.linalg.norm(L, axis=1))):
        raise ValueError(f"the lattice vectors are linearly dependent (det {det:.3e})")
    return L


def _factors(x: Any, n: int, what: str) -> np.ndarray:
    if x is None:
        return np.ones(n)
    v = np.asarray(x, float).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{what} must have one value per cell ({n}), got {np.shape(x)}")
    if not (np.isfinite(v).all() and (v > 0).all()):
        raise ValueError(f"{what} must be finite and > 0")
    return v


def _unit_strains(dim: int) -> list[np.ndarray]:
    """Symmetric H of the unit Voigt strains (engineering shear: the tensor component is γ / 2)."""
    pairs = [(0, 0), (1, 1), (0, 1)] if dim == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    out = []
    for i, j in pairs:
        H = np.zeros((dim, dim))
        if i == j:
            H[i, i] = 1.0
        else:
            H[i, j] = H[j, i] = 0.5
        out.append(H)
    return out


def _periodic_cases(mesh: Mesh, L: np.ndarray, Hs: list[np.ndarray], ncomp: int, ndof: int) -> LinearConstraints:
    """One set of rows with one right-hand-side column per case (the rows do not depend on H), plus the pin."""
    parts = [periodic_lattice(mesh, L, H, ncomp=ncomp) for H in Hs]
    C = parts[0].C
    for p in parts[1:]:
        if (p.C != C).nnz:
            raise AssertionError("periodic rows differ between cases")  # pragma: no cover
    per = LinearConstraints(C, np.column_stack([p.d[:, 0] for p in parts]), parts[0].labels)
    pin = int(mesh.used_nodes()[0])
    return per + LinearConstraints.fix(ndof, ncomp * pin + np.arange(ncomp), np.zeros((ncomp, len(Hs))))


def _effective(problem: Any, Ke0: np.ndarray, cons: LinearConstraints, volume: float,
               sensitivities: bool) -> MeshHomogenized:
    sol = solve_constrained(problem, np.zeros((problem.ndof, cons.n_cases)), cons, redundant="drop")
    U = sol.u
    K = problem.K
    C = np.asarray(U.T @ (K @ U)) / volume
    C = 0.5 * (C + C.T)
    dC = None
    if sensitivities:
        Ue = U[problem.cell_dofs]                                    # (ne, nloc, k)
        dC = np.einsum("eai,eab,ebj->eij", Ue, Ke0, Ue, optimize=True) / volume
        dC = 0.5 * (dC + np.swapaxes(dC, 1, 2))
    return MeshHomogenized(C, dC, U, volume)


def homogenize_mesh_elasticity(mesh: Mesh, D: Any, lattice: Any, E: Any = None, thickness: float = 1.0,
                               sensitivities: bool = False, degree: int | None = None) -> MeshHomogenized:
    """Effective Voigt stiffness of the periodic cell ``mesh`` with lattice vectors ``lattice`` (dim, dim), one per
    row, and cell moduli ``E`` (n_cells,) times ``D`` (k, k; plane stress or strain as the caller builds it in
    2-D). ``sensitivities``: also dC/dE per cell. Raises ``ConstraintError`` when a lattice vector matches no node
    pair, ``ValueError`` for invalid inputs."""
    L = _lattice(mesh, lattice)
    Ev = _factors(E, mesh.n_cells, "E")
    D0 = np.asarray(D, float)
    k = 3 if mesh.dim == 2 else 6
    if D0.shape != (k, k):
        raise ValueError(f"D must be ({k}, {k}), got {D0.shape}")
    P = ElasticProblem(mesh, Ev[:, None, None] * D0, thickness=thickness, degree=degree)
    Ke0 = kn.stiffness(P.geom, np.broadcast_to(D0, (mesh.n_cells, k, k)), P.thickness)
    volume = abs(float(np.linalg.det(L))) * (P.thickness if mesh.dim == 2 else 1.0)
    cons = _periodic_cases(mesh, L, _unit_strains(mesh.dim), mesh.dim, P.ndof)
    return _effective(P, Ke0, cons, volume, sensitivities)


def homogenize_mesh_conduction(mesh: Mesh, kappa: Any, lattice: Any, k: Any = None, thickness: float = 1.0,
                               sensitivities: bool = False, degree: int | None = None) -> MeshHomogenized:
    """Effective conductivity (dim, dim) of the periodic cell with cell factors ``k`` (n_cells,) times ``kappa``
    (a scalar or a (dim, dim) tensor). ``sensitivities``: also dk_eff/dk_e per cell."""
    L = _lattice(mesh, lattice)
    kv = _factors(k, mesh.n_cells, "k")
    dim = mesh.dim
    kap = np.asarray(kappa, float)
    kt = kap * np.eye(dim) if kap.ndim == 0 else kap
    if kt.shape != (dim, dim):
        raise ValueError(f"kappa must be a scalar or ({dim}, {dim}), got {kap.shape}")
    P = HeatProblem(mesh, kv[:, None, None] * kt, thickness=thickness, degree=degree)
    Ke0 = kn.conduction(P.geom, np.broadcast_to(kt, (mesh.n_cells, dim, dim)), P.thickness)
    volume = abs(float(np.linalg.det(L))) * (P.thickness if dim == 2 else 1.0)
    gs = [np.eye(dim)[i][None, :] for i in range(dim)]               # T = g·x: "H" is the (1, dim) gradient row
    cons = _periodic_cases(mesh, L, gs, 1, P.ndof)
    return _effective(P, Ke0, cons, volume, sensitivities)

