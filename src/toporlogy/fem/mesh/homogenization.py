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

Bounds (``bc``, elasticity; ``docs/foundations/prereg/MB_mesh_bounds.md``): ``"kubc"`` prescribes u = H x at every
boundary node (stiffer than periodic), ``"subc"`` applies the uniform stress σ̄ through its consistent load
Σ_e ∫ Bᵀ σ̄ dV with the rigid motions held by the fewest DOFs, measures the mean strain (a column of the compliance)
and returns S⁻¹ (softer): C_SUBC ≤ C_PBC ≤ C_KUBC. For these two the lattice is not used and |Y| = Σ dV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla

from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.constraints import LinearConstraints, periodic_lattice, solve_constrained
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.problem import ElasticProblem, HeatProblem, rigid_modes

__all__ = ["MeshHomogenized", "homogenize_mesh_elasticity", "homogenize_mesh_conduction", "MESH_BCS"]

MESH_BCS = ("periodic", "kubc", "subc")


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


def _same_measure(L: np.ndarray, P: Any) -> None:
    """A periodic cell tiles with exactly its own measure: |det L| must equal the mesh measure Σ dV (a sub-period
    lattice still matches node pairs on a structured mesh and would give a silently wrong answer; MH9)."""
    det = abs(float(np.linalg.det(L)))
    meas = float(P.geom.dV.sum())
    if abs(det - meas) > 1e-9 * meas:
        raise ValueError(f"the lattice cell measure |det lattice| = {det:.12g} differs from the mesh measure "
                         f"{meas:.12g}: the lattice is not the period of this mesh")


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
                               sensitivities: bool = False, degree: int | None = None,
                               bc: str = "periodic") -> MeshHomogenized:
    """Effective Voigt stiffness of the periodic cell ``mesh`` with lattice vectors ``lattice`` (dim, dim), one per
    row, and cell moduli ``E`` (n_cells,) times ``D`` (k, k; plane stress or strain as the caller builds it in
    2-D). ``sensitivities``: also dC/dE per cell. Raises ``ConstraintError`` when a lattice vector matches no node
    pair, ``ValueError`` for invalid inputs. ``bc``: "periodic" (default), "kubc" or "subc" (module notes)."""
    L = _lattice(mesh, lattice) if bc == "periodic" else np.eye(mesh.dim)
    Ev = _factors(E, mesh.n_cells, "E")
    D0 = np.asarray(D, float)
    k = 3 if mesh.dim == 2 else 6
    if D0.shape != (k, k):
        raise ValueError(f"D must be ({k}, {k}), got {D0.shape}")
    if bc not in MESH_BCS:
        raise ValueError(f"bc must be one of {MESH_BCS}, got {bc!r}")
    if sensitivities and bc != "periodic":
        raise ValueError("sensitivities are implemented for bc='periodic' only")
    P = ElasticProblem(mesh, Ev[:, None, None] * D0, thickness=thickness, degree=degree)
    Hs = _unit_strains(mesh.dim)
    if bc == "subc":
        return _static(P, mesh, k)
    if bc == "kubc":
        nodes = mesh.boundary_nodes()
        dofs = (mesh.dim * nodes[:, None] + np.arange(mesh.dim)).ravel()
        vals = np.column_stack([(mesh.points[nodes] @ H.T).ravel() for H in Hs])
        volume = float(P.geom.dV.sum()) * (P.thickness if mesh.dim == 2 else 1.0)
        return _effective(P, np.zeros(0), LinearConstraints.fix(P.ndof, dofs, vals), volume, False)
    Ke0 = kn.stiffness(P.geom, np.broadcast_to(D0, (mesh.n_cells, k, k)), P.thickness)
    _same_measure(L, P)
    volume = abs(float(np.linalg.det(L))) * (P.thickness if mesh.dim == 2 else 1.0)
    cons = _periodic_cases(mesh, L, Hs, mesh.dim, P.ndof)
    return _effective(P, Ke0, cons, volume, sensitivities)


def _static(P: ElasticProblem, mesh: Mesh, k: int) -> MeshHomogenized:
    """SUBC: unit uniform stresses, fewest DOFs against rigid motion, mean strain = compliance columns."""
    t = P.thickness if mesh.dim == 2 else 1.0
    B = kn.strain_matrix(P.geom.dNdx)                                # (ne, nq, k, nloc)
    w = P.geom.dV * t
    volume = float(w.sum())
    fe = np.einsum("eqkj,eq->ejk", B, w)                             # ∫ Bᵀ e_k dV per cell, (ne, nloc, k)
    F = np.zeros((P.ndof, k))
    np.add.at(F, P.cell_dofs, fe)
    used = mesh.used_nodes()
    R = rigid_modes(mesh.points[used], mesh.dim)
    udofs = (mesh.dim * used[:, None] + np.arange(mesh.dim)).ravel()
    piv = sla.qr(R.T, pivoting=True, mode="r")[1][: R.shape[1]]      # the most independent DOFs
    hold = LinearConstraints.fix(P.ndof, udofs[piv], np.zeros((R.shape[1], k)))
    sol = solve_constrained(P, F, hold)
    Ue = sol.u[P.cell_dofs]                                          # (ne, nloc, k)
    S = np.einsum("eqkj,eq,ejl->kl", B, w, Ue) / volume              # mean strain of unit stress l
    S = 0.5 * (S + S.T)
    return MeshHomogenized(np.linalg.inv(S), None, sol.u, volume)


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
    if not np.isfinite(kt).all() or np.abs(kt - kt.T).max() > 1e-12 * np.abs(kt).max():
        raise ValueError("kappa must be finite and symmetric (MH10)")
    try:
        np.linalg.cholesky(kt)
    except np.linalg.LinAlgError:
        raise ValueError("kappa must be positive definite (MH10)") from None
    P = HeatProblem(mesh, kv[:, None, None] * kt, thickness=thickness, degree=degree)
    Ke0 = kn.conduction(P.geom, np.broadcast_to(kt, (mesh.n_cells, dim, dim)), P.thickness)
    _same_measure(L, P)
    volume = abs(float(np.linalg.det(L))) * (P.thickness if dim == 2 else 1.0)
    gs = [np.eye(dim)[i][None, :] for i in range(dim)]               # T = g·x: "H" is the (1, dim) gradient row
    cons = _periodic_cases(mesh, L, gs, 1, P.ndof)
    return _effective(P, Ke0, cons, volume, sensitivities)

