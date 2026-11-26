"""Re-analysis of a finished shape on staircase voxel meshes, with an independent assembly (scikit-fem).

The finished part (an implicit φ, φ < 0 inside: :class:`toporlogy.post.pipeline.Implicit` or any callable) is
sampled at the centres of a regular grid of spacing h; a voxel is solid where φ(centre) < 0. The hexahedral mesh of
the solid voxels is analysed with trilinear H8 elements twice: by toporlogy (:class:`toporlogy.fem.Elasticity`) and
by scikit-fem (``ElementVector(ElementHex1())``, its own quadrature and assembly), on the same nodes, supports and
nodal forces. Both integrate the trilinear stiffness exactly, so the two compliances agree to round-off; a difference
is an assembly or bookkeeping error in one of them.

What this is not: a body-fitted model. The boundary is a staircase whose volume and shape error is O(h) (the volume
of every mesh is reported next to the implicit volume), so a series over h mixes the geometry error with the
discretisation error and need not show an asymptotic order. It does not replace the T10 re-analysis of
:mod:`toporlogy.post.verify` (which needs gmsh). scikit-fem is optional; without it only the toporlogy solve runs.

Boundary conditions are callables in physical coordinates, as in :mod:`toporlogy.post.verify`:
``supports`` = sequence of ``select`` or ``(select, directions)`` applied to the nodes of the solid voxels;
``loads`` = sequence of ``(select, force (3,))``: each total force is spread over the boundary faces whose four
nodes are all selected, in proportion to the face area (consistent nodal forces of a uniform traction on bilinear
quadrilaterals: a quarter of the face area to each corner).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp

__all__ = ["voxel_solid", "face_weights", "solve_voxels", "staircase_series"]

Select = Callable[[np.ndarray], Any]


def voxel_solid(phi: Callable[[np.ndarray], np.ndarray], lo: Sequence[float], hi: Sequence[float],
             h: float) -> tuple[np.ndarray, np.ndarray]:
    """(solid (nx, ny, nz) bool, origin (3,)) of the voxels of spacing ``h`` covering [lo, hi] whose centre has
    φ < 0. The grid starts at ``lo`` exactly, so planes at lo + k·h are voxel faces."""
    lo_a = np.asarray(lo, float)
    shape = tuple(int(np.ceil((float(hi[k]) - lo_a[k]) / h - 1e-9)) for k in range(3))
    idx = np.indices(shape).reshape(3, -1).T
    c = lo_a + h * (idx + 0.5)
    return (np.asarray(phi(c)) < 0).reshape(shape), lo_a


def _node_index(solid: np.ndarray) -> np.ndarray:
    """(ne, 8) lattice node numbers (C order over shape + 1) of the solid voxels, local order of the H8 corners
    (x fastest within the voxel: 000, 100, 110, 010, 001, 101, 111, 011)."""
    ns = tuple(s + 1 for s in solid.shape)
    ijk = np.argwhere(solid)
    off = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    nodes = ijk[:, None, :] + off[None]
    return np.ravel_multi_index(tuple(nodes.reshape(-1, 3).T), ns).reshape(-1, 8)


def face_weights(solid: np.ndarray, h: float, selected: np.ndarray) -> np.ndarray:
    """Consistent nodal areas (lattice nodes, C order over shape + 1) of the boundary faces of the solid voxels whose
    four nodes are all ``selected`` (bool per lattice node): a quarter of h² to each corner of each such face."""
    ns = tuple(s + 1 for s in solid.shape)
    sel = np.asarray(selected, bool).reshape(ns)
    w = np.zeros(ns)
    pad = np.pad(solid, 1)
    for ax in range(3):
        for side in (-1, 1):
            nb = np.roll(pad, -side, axis=ax)[1:-1, 1:-1, 1:-1]
            faces = np.argwhere(solid & ~nb)                     # voxel ijk with a free face on this side
            if not len(faces):
                continue
            base = faces.copy()
            if side == 1:
                base[:, ax] += 1
            others = [a for a in range(3) if a != ax]
            corners = []
            for da in (0, 1):
                for db in (0, 1):
                    q = base.copy()
                    q[:, others[0]] += da
                    q[:, others[1]] += db
                    corners.append(q)
            cs = np.stack(corners, axis=1)                         # (nf, 4, 3)
            ok = sel[tuple(cs.reshape(-1, 3).T)].reshape(-1, 4).all(axis=1)
            for k in range(4):
                np.add.at(w, tuple(cs[ok, k].T), 0.25 * h * h)
    return w.ravel()


def _supports(items: Sequence[Any]) -> list[tuple[Select, tuple[int, ...]]]:
    out: list[tuple[Select, tuple[int, ...]]] = []
    for it in items:
        if callable(it):
            out.append((it, (0, 1, 2)))
        else:
            sel, dirs = it
            out.append((sel, tuple(int(d) for d in (dirs if np.ndim(dirs) else [dirs]))))
    return out


def solve_voxels(solid: np.ndarray, origin: Sequence[float] | np.ndarray, h: float, supports: Sequence[Any],
                 loads: Sequence[tuple[Select, Any]], E: float, nu: float = 0.3,
                 backends: Sequence[str] = ("toporlogy", "skfem"), solver: str = "direct",
                 tol: float = 1e-10) -> dict[str, Any]:
    """Linear static solve of the voxel mesh (module notes); per backend the compliance F·U, and for scikit-fem the
    sum of the reactions next to the applied force (equilibrium). ``solver`` of the toporlogy backend: "direct"
    (:class:`~toporlogy.fem.Elasticity`) or "gmg" (:class:`~toporlogy.fem.elastic_gmg.ElasticityGMG`, relative
    residual ``tol``, no direct fallback: large 3-D meshes, where the sparse LU of this machine is slow)."""
    from toporlogy.core.domain import Domain
    solid = np.asarray(solid, bool)
    if solid.ndim != 3 or not solid.any():
        raise ValueError("solid must be a non-empty 3-D bool array")
    org = np.asarray(origin, float)
    ns = tuple(s + 1 for s in solid.shape)
    conn = _node_index(solid)
    used = np.unique(conn)
    X = org + h * np.indices(ns).reshape(3, -1).T                 # lattice node coordinates
    on = np.zeros(int(np.prod(ns)), bool)
    on[used] = True
    fixed_l: list[np.ndarray] = []
    for sel, dirs in _supports(supports):
        nn = np.flatnonzero(on & np.asarray(sel(X), bool))
        if not len(nn):
            raise ValueError("a support selects no node of the solid voxels")
        fixed_l.append((3 * nn[:, None] + np.array(dirs)).ravel())
    fixed = np.unique(np.concatenate(fixed_l))
    F = np.zeros(3 * len(X))
    for sel, f in loads:
        mask = on & np.asarray(sel(X), bool)
        w = face_weights(solid, h, mask)
        if not w.sum() > 0:
            raise ValueError("a load selects no boundary face of the solid voxels")
        for k in range(3):
            F[3 * np.arange(len(X)) + k] += float(np.asarray(f, float)[k]) * w / w.sum()
    out: dict[str, Any] = {"h": float(h), "elements": int(len(conn)), "nodes": int(len(used)),
                           "dofs": int(3 * len(used)), "volume": float(len(conn)) * h ** 3,
                           "applied_force": F.reshape(-1, 3).sum(0).tolist()}
    if "toporlogy" in backends:
        t0 = time.perf_counter()
        d = Domain(solid.shape, h, origin=org, frac=solid.astype(float))
        if solver == "direct":
            from toporlogy.fem.elastic import Elasticity
            m: Any = Elasticity(d, fixed, nu=nu, element="H8")
        elif solver == "gmg":
            from toporlogy.fem.elastic_gmg import ElasticityGMG
            m = ElasticityGMG(d, fixed, nu=nu, element="H8", tol=tol, maxit=2000, fallback_max_elements=0,
                              direct_below=0)
        else:
            raise ValueError(f'solver: "direct" | "gmg", got {solver!r}')
        U = m.solve_numpy(np.full(d.n_active, float(E)), F)[:, 0]
        out["toporlogy"] = {"compliance": float(F @ U), "solver": solver, "seconds": time.perf_counter() - t0}
    if "skfem" in backends:
        out["skfem"] = _skfem(X, conn, used, fixed, F, E, nu, ns)
    return out


def _skfem(X: np.ndarray, conn: np.ndarray, used: np.ndarray, fixed: np.ndarray, F: np.ndarray, E: float,
           nu: float, node_shape: tuple[int, ...]) -> dict[str, Any]:
    """scikit-fem assembly on the same nodes: MeshHex of the used lattice nodes, its own element ordering."""
    from skfem import Basis, ElementHex1, ElementVector, MeshHex, asm
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    from toporlogy.solvers import DirectSolver
    from toporlogy.solvers.ordering import nested_dissection_dofs
    t0 = time.perf_counter()
    loc = np.full(len(X), -1, np.int64)
    loc[used] = np.arange(len(used))
    # scikit-fem's hexahedron numbering (read from MeshHex.init_tensor, scikit-fem 12.0.2): corners 000, 010, 100,
    # 001, 110, 011, 101, 111 of (x, y, z); our local order is 000, 100, 110, 010, 001, 101, 111, 011
    t = loc[conn][:, [0, 3, 1, 4, 2, 7, 5, 6]].T
    mesh = MeshHex(X[used].T, t)
    basis = Basis(mesh, ElementVector(ElementHex1()))
    K = asm(linear_elasticity(*lame_parameters(float(E), float(nu))), basis).tocsr()
    nd = basis.nodal_dofs                                       # (3, n_used): DOF of node n, component c
    perm = np.empty(3 * len(used), np.int64)                    # our DOF 3 n + c of a used node -> scikit-fem DOF
    perm[3 * np.arange(len(used))[:, None] + np.arange(3)] = nd.T
    Ko = K[perm][:, perm].tocsr()                               # scikit-fem's matrix in our DOF numbering
    Fo = F.reshape(-1, 3)[used].ravel()
    fixed_u = 3 * loc[fixed // 3] + fixed % 3
    free = np.setdiff1d(np.arange(Ko.shape[0]), fixed_u)
    # elimination order only (nested dissection of the lattice, as the toporlogy solve): SuperLU's own MMD order on
    # this matrix took 2180 s for 51 000 DOF, nested dissection seconds; the order does not change the solution
    lattice = (3 * used[:, None] + np.arange(3)).ravel()[free]
    order = nested_dissection_dofs(node_shape, 3, lattice)
    s = DirectSolver()
    s.factor(sp.triu(Ko[free][:, free], format="csr"), order=order)
    U = np.zeros(Ko.shape[0])
    U[free] = np.asarray(s.solve(Fo[free])).reshape(-1)
    s.release()
    R = (Ko @ U - Fo).reshape(-1, 3)
    on_fix = np.unique(fixed_u // 3)
    return {"compliance": float(Fo @ U), "reaction_sum": R[on_fix].sum(0).tolist(), "solver": s.method,
            "seconds": time.perf_counter() - t0}


def staircase_series(phi: Callable[[np.ndarray], np.ndarray], lo: Sequence[float], hi: Sequence[float],
                     sizes: Sequence[float], supports: Sequence[Any], loads: Sequence[tuple[Select, Any]], E: float,
                     nu: float = 0.3, backends: Sequence[str] = ("toporlogy",),
                     volume_reference: float | None = None, solver: str = "direct") -> dict[str, Any]:
    """:func:`solve_voxels` on the voxelisations of φ at every size, the voxel volume next to
    ``volume_reference`` (the implicit or surface volume: the geometry error of the staircase), and
    :func:`toporlogy.post.verify.assess_convergence` of the toporlogy compliance (module notes: mixes geometry and
    discretisation error). ``solver``: see :func:`solve_voxels`."""
    from toporlogy.post.verify import assess_convergence
    meshes = []
    for h in sorted((float(s) for s in sizes), reverse=True):
        solid, org = voxel_solid(phi, lo, hi, h)
        r = solve_voxels(solid, org, h, supports, loads, E, nu, backends, solver=solver)
        if volume_reference:
            r["volume_error"] = r["volume"] / float(volume_reference) - 1.0
        meshes.append(r)
    key = backends[0]
    verdict = assess_convergence([m["h"] for m in meshes], [m[key]["compliance"] for m in meshes])
    return {"meshes": meshes, "convergence": verdict, "backend": key,
            "note": "staircase boundary: geometry error O(h) is part of every difference (not body-fitted)"}
