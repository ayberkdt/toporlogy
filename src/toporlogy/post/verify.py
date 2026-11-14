"""Re-verification of the finished surface: gmsh 10-node tetrahedra + linear static T10 solve.

Quadratic tetrahedron (element, 4-point Keast rule exact for Bᵀ D B of the
quadratic tet, consistent nodal areas, nodal-averaged stress). Boundary conditions are plain callables in physical
coordinates, so a separate BC module can supply them later:

* ``supports``: sequence of ``select`` or ``(select, directions)``; ``select(X (n, 3)) -> bool (n,)`` is applied to
  the *boundary* nodes; selected nodes get zero displacement in ``directions`` (default all three);
* ``load_cases``: ``{name: [(select, force (3,)), ...]}`` or one list; each total force is spread over the selected
  boundary nodes in proportion to their consistent area (a uniform traction on the selected faces);
* ``exclusions``: sequence of ``select(X) -> bool`` over all nodes; von Mises is also reported outside them
  (support corners and load patches are singular in a linear model).

The voxel model on the optimisation grid (:func:`voxel_reference`) uses the same callables on the grid nodes (loads
spread equally over the selected nodes) so the change from the voxel model to the finished part is reported.
Smoothing changes stiffness, stress and feasibility (Swierstra, Gupta & Langelaar 2020; Subedi, Verma & Suresh 2020;
neither read in this session) — this module measures that change for the part at hand.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from toporlogy.references import cite

__all__ = ["TetMesh", "tet_mesh", "stiffness_t10", "solve_t10", "voxel_reference", "reverify", "d_iso",
           "von_mises"]

_EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]
_a, _b = 0.5854101966249685, 0.1381966011250105
_GP = np.array([[_b, _b, _b], [_a, _b, _b], [_b, _a, _b], [_b, _b, _a]])
_GW = np.full(4, 1.0 / 24.0)
_NODES_NAT = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
                       [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5]], float)

Select = Callable[[np.ndarray], Any]


def _dN(p: np.ndarray) -> np.ndarray:
    """Derivatives (10, 3) of the T10 shape functions at the natural point p."""
    xi, et, ze = p
    L = np.array([1 - xi - et - ze, xi, et, ze])
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    d = np.zeros((10, 3))
    for i in range(4):
        d[i] = (4 * L[i] - 1) * dL[i]
    for k, (i, j) in enumerate(_EDGES):
        d[4 + k] = 4 * (L[i] * dL[j] + L[j] * dL[i])
    return d


_DN_GP = np.array([_dN(p) for p in _GP])
_DN_NODES = np.array([_dN(p) for p in _NODES_NAT])


def d_iso(E: float, nu: float) -> np.ndarray:
    """Isotropic elasticity matrix, Voigt order [εx, εy, εz, γyz, γxz, γxy]."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = c * nu
    D[np.arange(3), np.arange(3)] = c * (1 - nu)
    D[np.arange(3, 6), np.arange(3, 6)] = c * (1 - 2 * nu) / 2
    return D


def von_mises(s: np.ndarray) -> np.ndarray:
    sx, sy, sz, tyz, txz, txy = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (tyz ** 2 + txz ** 2 + txy ** 2))


@dataclass
class TetMesh:
    X: np.ndarray            # (nn, 3)
    T: np.ndarray            # (ne, 10)
    faces: np.ndarray        # (nf, 6) boundary 6-node triangles
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ndof(self) -> int:
        return 3 * len(self.X)

    @property
    def volume(self) -> float:
        J = np.einsum("gnk,enl->egkl", _DN_GP, self.X[self.T])
        return float(np.einsum("eg,g->", np.linalg.det(J), _GW))

    @property
    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.faces)


def tet_mesh(V: np.ndarray, F: np.ndarray, size: float, angle_deg: float = 40.0, timeout: float = 600.0,
             modes: str = "reparametrize,discrete") -> TetMesh:
    """10-node tet mesh of the closed surface (V, F) with gmsh in a subprocess (target edge ``size``)."""
    from toporlogy.post.export import write_stl
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        stl, out = Path(td) / "surface.stl", Path(td) / "mesh.npz"
        write_stl(stl, V, F)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
        worker = Path(__file__).with_name("_gmsh_worker.py")
        cmd = [sys.executable, str(worker), str(stl), str(out), repr(float(size)), repr(angle_deg), modes]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)  # noqa: S603
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"gmsh worker failed ({r.returncode}): {r.stderr.strip()[-2000:]}")
        with np.load(out) as d:
            m = TetMesh(np.array(d["X"]), d["T"].astype(np.int64), d["faces"].astype(np.int64),
                        {"mode": str(d["mode"]), "gmsh_s": float(d["seconds"])})
    m.info.update({"size": float(size), "elements": int(len(m.T)), "nodes": int(len(m.X)),
                   "mesh_s": time.perf_counter() - t0})
    return m


@cite("zienkiewicz_taylor_2013", "cook_2002", "keast_1986", note="10-node tetrahedron, 4-point rule")
def stiffness_t10(m: TetMesh, E: float, nu: float, chunk: int = 20000) -> sp.csr_matrix:
    """Global stiffness (full, CSR) of the T10 mesh."""
    D = d_iso(E, nu)
    ne = len(m.T)
    edof = (3 * m.T[:, :, None] + np.arange(3)).reshape(ne, 30)
    rows, cols, vals = [], [], []
    for s in range(0, ne, chunk):
        Xe = m.X[m.T[s:s + chunk]]
        Ke = np.zeros((len(Xe), 30, 30))
        for g in range(4):
            J = np.einsum("nk,enl->ekl", _DN_GP[g], Xe)
            dNx = np.einsum("ekl,nl->enk", np.linalg.inv(J), _DN_GP[g])
            B = _bmat(dNx)
            w = _GW[g] * np.linalg.det(J)
            if (w <= 0).any():
                raise ValueError(f"{int((w <= 0).sum())} inverted/degenerate T10 Jacobians")
            Ke += w[:, None, None] * (B.transpose(0, 2, 1) @ (D @ B))      # batched BLAS, not einsum
        ed = edof[s:s + chunk]
        rows.append(np.repeat(ed, 30, axis=1).ravel())
        cols.append(np.tile(ed, (1, 30)).ravel())
        vals.append(Ke.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(m.ndof, m.ndof))


def _bmat(dNx: np.ndarray) -> np.ndarray:
    B = np.zeros((len(dNx), 6, 30))
    B[:, 0, 0::3] = dNx[:, :, 0]
    B[:, 1, 1::3] = dNx[:, :, 1]
    B[:, 2, 2::3] = dNx[:, :, 2]
    B[:, 3, 1::3] = dNx[:, :, 2]
    B[:, 3, 2::3] = dNx[:, :, 1]
    B[:, 4, 0::3] = dNx[:, :, 2]
    B[:, 4, 2::3] = dNx[:, :, 0]
    B[:, 5, 0::3] = dNx[:, :, 1]
    B[:, 5, 1::3] = dNx[:, :, 0]
    return B


def _face_weights(m: TetMesh, selected: np.ndarray | None = None) -> np.ndarray:
    """Consistent nodal areas (uniform traction on 6-node triangles: corners 0, midsides A/3).

    With ``selected`` (bool per node) only boundary faces whose six nodes are all selected contribute, so a node on
    the rim of the loaded patch does not collect area from the neighbouring faces (summing the
    areas of every face at the node makes the traction non-uniform at patch edges — 1.7 % compliance error in
    the uniaxial patch test here)."""
    fc = m.faces if selected is None else m.faces[selected[m.faces].all(axis=1)]
    P = m.X[fc]
    A = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    a = np.zeros(len(m.X))
    np.add.at(a, fc[:, 3:].ravel(), np.repeat(A / 3.0, 3))
    return a


def _supports(items: Sequence[Any]) -> list[tuple[Select, tuple[int, ...]]]:
    out = []
    for it in items:
        if callable(it):
            out.append((it, (0, 1, 2)))
        else:
            sel, dirs = it
            out.append((sel, tuple(int(d) for d in (dirs if np.ndim(dirs) else [dirs]))))
    return out


def _cases(load_cases: Any) -> dict[str, list[tuple[Select, np.ndarray]]]:
    if isinstance(load_cases, Mapping):
        return {str(k): [(s, np.asarray(f, float)) for s, f in v] for k, v in load_cases.items()}
    return {"case0": [(s, np.asarray(f, float)) for s, f in load_cases]}


def _solve(K: sp.csr_matrix, F: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, str, float]:
    from toporlogy.solvers import DirectSolver
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    Kf = K[free][:, free]
    A = sp.triu(Kf, format="csr")
    A.sort_indices()
    t0 = time.perf_counter()
    s = DirectSolver()
    s.factor(A)
    U = np.zeros_like(F)
    U[free] = s.solve(F[free]).reshape(F[free].shape)
    s.release()
    return U, s.method, time.perf_counter() - t0


def solve_t10(m: TetMesh, supports: Sequence[Any], load_cases: Any, E: float, nu: float = 0.3,
              exclusions: Sequence[Select] = ()) -> dict[str, Any]:
    """Linear static solve; per case compliance, max displacement, von Mises (max and outside the exclusions)."""
    t0 = time.perf_counter()
    bnd = m.boundary_nodes
    Xb = m.X[bnd]
    fixed = []
    n_sup = []
    for sel, dirs in _supports(supports):
        nn = bnd[np.asarray(sel(Xb), bool)]
        if not len(nn):
            raise ValueError("a support selects no boundary node")
        n_sup.append(int(len(nn)))
        fixed.append((3 * nn[:, None] + np.array(dirs)).ravel())
    fixed_d = np.unique(np.concatenate(fixed))
    cases = _cases(load_cases)
    F = np.zeros((m.ndof, len(cases)))
    n_load = []
    for c, (_, loads) in enumerate(cases.items()):
        for sel, f in loads:
            nn = bnd[np.asarray(sel(Xb), bool)]
            if not len(nn):
                raise ValueError("a load selects no boundary node")
            mask = np.zeros(len(m.X), bool)
            mask[nn] = True
            a = _face_weights(m, mask)
            w = a[nn] if a[nn].sum() > 0 else np.ones(len(nn))
            for k in range(3):
                np.add.at(F[:, c], 3 * nn + k, f[k] * w / w.sum())
            n_load.append(int(len(nn)))
    F_applied = F.copy()
    F[fixed_d] = 0.0
    K = stiffness_t10(m, E, nu)
    t1 = time.perf_counter()
    U, method, t_solve = _solve(K, F, fixed_d)
    D = d_iso(E, nu)
    nc = len(cases)
    sn = np.zeros((nc, len(m.X), 6))
    cnt = np.bincount(m.T.ravel(), minlength=len(m.X)).astype(float)
    edof = (3 * m.T[:, :, None] + np.arange(3)).reshape(len(m.T), 30)
    for s in range(0, len(m.T), 20000):
        T = m.T[s:s + 20000]
        Xe = m.X[T]
        for k in range(10):
            J = np.einsum("nk,enl->ekl", _DN_NODES[k], Xe)
            dNx = np.einsum("ekl,nl->enk", np.linalg.inv(J), _DN_NODES[k])
            for c in range(nc):
                ue = U[:, c][edof[s:s + 20000]].reshape(-1, 10, 3)
                g = np.einsum("eni,enj->eij", ue, dNx)
                eps = np.stack([g[:, 0, 0], g[:, 1, 1], g[:, 2, 2], g[:, 1, 2] + g[:, 2, 1], g[:, 0, 2] + g[:, 2, 0],
                                g[:, 0, 1] + g[:, 1, 0]], axis=1)
                np.add.at(sn[c], T[:, k], eps @ D.T)
    sn /= cnt[None, :, None]
    excl = np.zeros(len(m.X), bool)
    for sel in exclusions:
        excl |= np.asarray(sel(m.X), bool)
    out = []
    for c, name in enumerate(cases):
        vm = von_mises(sn[c])
        u = U[:, c].reshape(-1, 3)
        disp = np.linalg.norm(u, axis=1)
        vo = np.where(excl, 0.0, vm)
        R = (K @ U[:, c] - F[:, c]).reshape(-1, 3)
        out.append({"name": name, "compliance": float(F[:, c] @ U[:, c]), "max_displacement": float(disp.max()),
                    "max_displacement_at": m.X[int(np.argmax(disp))].tolist(),
                    "vm_max": float(vm.max()), "vm_max_at": m.X[int(np.argmax(vm))].tolist(),
                    "vm_max_outside_exclusions": float(vo.max()), "vm_outside_at": m.X[int(np.argmax(vo))].tolist(),
                    "applied_force": F_applied[:, c].reshape(-1, 3).sum(0).tolist(),
                    "reaction_sum": R[np.unique(fixed_d // 3)].sum(0).tolist()})
    return {"cases": out, "elements": int(len(m.T)), "nodes": int(len(m.X)), "dofs": int(m.ndof),
            "volume": m.volume, "solver": method, "solve_s": t_solve, "assemble_s": t1 - t0,
            "total_s": time.perf_counter() - t0, "support_nodes": n_sup, "load_nodes": n_load, "mesh": m.info}


def voxel_reference(domain: Any, xphys: np.ndarray, supports: Sequence[Any], load_cases: Any, E: float,
                    nu: float = 0.3, penal: float = 3.0, emin_ratio: float = 1e-9) -> dict[str, Any]:
    """Same callables on the optimisation grid (SIMP modulus E·x̄^p, H8): compliance and max displacement per case.

    Supports and loads select grid nodes of the part (all of them, not only the boundary); each total force is
    spread equally over its selected nodes."""
    from toporlogy.fem import Elasticity
    fixed = []
    for sel, dirs in _supports(supports):
        nn = domain.nodes_where(lambda X, s=sel: np.asarray(s(X), bool))
        if not len(nn):
            raise ValueError("a support selects no grid node")
        fixed.append(domain.dofs(nn, list(dirs)))
    cases = _cases(load_cases)
    Fv = domain.load_vector(len(cases))
    for c, (_, loads) in enumerate(cases.items()):
        for sel, f in loads:
            nn = domain.nodes_where(lambda X, s=sel: np.asarray(s(X), bool))
            if not len(nn):
                raise ValueError("a load selects no grid node")
            for k in range(3):
                Fv[domain.dofs(nn, k), c] += f[k] / len(nn)
    fx = np.unique(np.concatenate(fixed))
    phys = Elasticity(domain, fx, nu=nu, element="H8")
    x = np.asarray(xphys, float).reshape(-1)[phys.elements]
    fr = np.asarray(domain.frac, float).reshape(-1)[phys.elements]
    Ee = (emin_ratio + x ** penal * (1 - emin_ratio)) * E * fr
    t0 = time.perf_counter()
    U = phys.solve_numpy(Ee, Fv)
    out = []
    for c, name in enumerate(cases):
        u = U[:, c].reshape(-1, 3)
        out.append({"name": name, "compliance": float(Fv[:, c] @ U[:, c]),
                    "max_displacement": float(np.linalg.norm(u, axis=1).max())})
    return {"cases": out, "solve_s": time.perf_counter() - t0, "volume": float((x * fr).sum()) * domain.h ** 3}


def reverify(res: Any, supports: Sequence[Any], load_cases: Any, E: float, nu: float = 0.3,
             size: float | None = None, exclusions: Sequence[Select] = (), domain: Any = None,
             xphys: np.ndarray | None = None, timeout: float = 600.0, penal: float = 3.0,
             surface_edge: float | None = None) -> dict[str, Any]:
    """Mesh the finished surface, solve, and (with ``domain``/``xphys``) compare with the voxel model.

    ``surface_edge``: edge length of the surface handed to gmsh (default size/2; 0 = use the finished surface as
    it is). A :class:`~toporlogy.post.pipeline.PostResult` is remeshed from its implicit shape
    (:func:`~toporlogy.post.pipeline.remesh`); a plain (V, F) is decimated."""
    from toporlogy.post import mesh as M
    V, F = (res.V, res.F) if hasattr(res, "V") else res
    if size is None:
        size = 1.5 * (domain.h if domain is not None else float(np.ptp(V, axis=0).max()) / 30)
    # a surface much finer than the tet size makes gmsh slow (and huge in its "discrete" fallback, which keeps the
    # surface triangles): remesh the finished shape at edge ≈ size/2 (implicit re-contouring + relaxation; topology
    # checked), or decimate a plain surface
    dec: dict[str, Any] = {"applied": False}
    if surface_edge != 0:
        se = 0.5 * size if surface_edge is None else float(surface_edge)
        if hasattr(res, "implicit") and float(np.mean(M.quality(V, F)["edge_mean"])) < 0.7 * se:
            from toporlogy.post.pipeline import remesh
            V, F, dec = remesh(res, edge=se)
            dec["applied"] = "edge" in dec
        elif not hasattr(res, "implicit"):
            target = int(4 * M.area(V, F) / (np.sqrt(3) * se ** 2))
            if len(F) > 1.5 * target:
                V, F, dec = M.decimate(V, F, target)
    m = tet_mesh(V, F, size, timeout=timeout)
    m.info["surface_decimation"] = dec
    out: dict[str, Any] = {"t10": solve_t10(m, supports, load_cases, E, nu, exclusions)}
    if domain is not None and xphys is not None:
        vx = voxel_reference(domain, xphys, supports, load_cases, E, nu, penal=penal)
        out["voxel"] = vx
        out["change"] = [{"name": a["name"], "compliance_ratio": a["compliance"] / b["compliance"],
                          "max_displacement_ratio": a["max_displacement"] / b["max_displacement"]}
                         for a, b in zip(out["t10"]["cases"], vx["cases"], strict=True)]
        out["volume_ratio"] = out["t10"]["volume"] / vx["volume"]
    return out
