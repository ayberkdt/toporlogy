"""Supports resolved to DOFs, local frames and springs; the constrained physics.
"""
from __future__ import annotations

import scipy.sparse as sp
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from toporlogy.domain import Domain
from toporlogy.fem.elastic import Elasticity
from typing import Any

import numpy as np

from toporlogy.boundary.part import PartInfo
from toporlogy.boundary.supports import Support


# =============================================================================================== constraint set
def _constraint_key(cs: ConstraintSet) -> bytes:
    """Content of a resolved support set: fixed DOFs, local frames (node, constrained count, axes) and springs."""
    parts = [np.asarray(cs.fixed, np.int64).tobytes(), b"|frames"]
    for node in sorted(cs.frames):
        T, k = cs.frames[node]
        parts += [np.int64(node).tobytes(), np.int64(k).tobytes(), np.ascontiguousarray(T, np.float64).tobytes()]
    parts += [b"|springs", np.asarray(cs.spring_nodes, np.int64).tobytes(),
              np.ascontiguousarray(cs.spring_K, np.float64).tobytes()]
    return b"".join(parts)


@dataclass
class ConstraintSet:
    """Supports resolved to DOFs: global fixed DOFs, local frames for oblique constraints, nodal springs."""

    fixed: np.ndarray                                   # global DOF numbers (sorted)
    frames: dict[int, tuple[np.ndarray, int]]           # node → (T (dim×dim, columns = local axes), n constrained)
    spring_nodes: np.ndarray
    spring_K: np.ndarray                                # (k, dim, dim) global frame
    per_support: list[tuple[str, str, np.ndarray, int]]  # (name, kind, nodes, constraint rows)
    multiply_constrained: np.ndarray                     # nodes touched by more than one support
    redundant_rows: int

    @property
    def needs_extension(self) -> bool:
        return bool(self.frames) or len(self.spring_nodes) > 0

    def constrained_directions(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """(nodes, D (m, dim)) one row per constrained direction (for checks)."""
        nodes, rows = [], []
        if len(self.fixed):
            nodes.append(self.fixed // dim)
            rows.append(np.eye(dim)[self.fixed % dim])
        for n, (T, k) in self.frames.items():
            nodes.append(np.full(k, n))
            rows.append(T[:, :k].T)
        if not nodes:
            return np.zeros(0, np.int64), np.zeros((0, dim))
        return np.concatenate(nodes), np.concatenate(rows)


def _resolve(supports: Sequence[Support], info: PartInfo) -> ConstraintSet:
    dim = info.dim
    nodes_l, dirs_l, owner_l = [], [], []
    per: list[tuple[str, str, np.ndarray, int]] = []
    sn, sk = [], []
    for s_i, s in enumerate(supports):
        n, D = s.directions(info)
        n2, K = s.springs(info)
        rows = int(D.shape[0] * D.shape[1]) if D.size else 0
        per.append((s.name, s.kind, np.union1d(n, n2), rows))
        if len(n):
            m = D.shape[1]
            nodes_l.append(np.repeat(n, m))
            dirs_l.append(D.reshape(-1, dim))
            owner_l.append(np.full(len(n) * m, s_i))
        if len(n2):
            sn.append(n2)
            sk.append(K)
    touched = np.concatenate([p[2] for p in per]) if per else np.zeros(0, np.int64)
    u, c = np.unique(touched, return_counts=True)
    multi = u[c > 1]
    fixed: list[np.ndarray] = []
    frames: dict[int, tuple[np.ndarray, int]] = {}
    redundant = 0
    if nodes_l:
        N = np.concatenate(nodes_l)
        D = np.concatenate(dirs_l)
        order = np.argsort(N, kind="stable")
        N, D = N[order], D[order]
        axis_like = np.isclose(np.abs(D).max(axis=1), 1.0, rtol=0, atol=1e-12)
        starts = np.flatnonzero(np.r_[True, N[1:] != N[:-1]])
        ends = np.r_[starts[1:], len(N)]
        all_axis = np.logical_and.reduceat(axis_like, starts)
        # fast path: nodes whose constraint directions are all coordinate axes
        fa = np.flatnonzero(all_axis)
        if len(fa):
            sel = np.concatenate([np.arange(starts[k], ends[k]) for k in fa])
            ax = np.argmax(np.abs(D[sel]), axis=1)
            dof = np.unique(dim * N[sel] + ax)
            redundant += len(sel) - len(dof)
            fixed.append(dof)
        for k in np.flatnonzero(~all_axis):
            n = int(N[starts[k]])
            Dn = D[starts[k]:ends[k]]
            _, s, Vt = np.linalg.svd(Dn)
            rank = int((s > 1e-9 * s[0]).sum())
            redundant += len(Dn) - rank
            if rank == dim:
                fixed.append(dim * n + np.arange(dim))
                continue
            B = Vt[:rank]
            P = B.T @ B                                           # projector onto the constrained span
            on_axes = np.flatnonzero(np.isclose(np.diag(P), 1.0, atol=1e-10))
            if len(on_axes) == rank:
                fixed.append(dim * n + on_axes)
            else:
                frames[n] = (Vt.T.copy(), rank)                   # columns: constrained first, then free
    fixed_arr = np.unique(np.concatenate(fixed)) if fixed else np.zeros(0, np.int64)
    spring_nodes = np.concatenate(sn) if sn else np.zeros(0, np.int64)
    spring_K = np.concatenate(sk) if sk else np.zeros((0, dim, dim))
    return ConstraintSet(fixed_arr, frames, spring_nodes, spring_K, per, multi, redundant)


# =============================================================================================== physics extension
class ConstrainedElasticity(Elasticity):
    """:class:`~toporlogy.fem.Elasticity` with oblique constraints (local frames, removed after
    ``K' = Tᵀ K T``) and nodal springs to ground (added to K before the transformation).

    ``free`` keeps the parent meaning (active DOFs not fixed globally); the reduced system has the free DOFs minus
    the locally constrained ones. ``U`` is returned in the global frame, so responses, ``element_energy`` and the
    core adjoint are unchanged.
    """

    def __init__(self, domain: Domain, constraints: ConstraintSet, nu: float = 0.3, element: str | None = None,
                 solver: Any = None) -> None:
        super().__init__(domain, constraints.fixed, nu=nu, element=element, solver=solver)
        dim = domain.dim
        nf = len(self.free)
        gmap = np.full(domain.ndof, -1, np.int64)
        gmap[self.free] = np.arange(nf)
        rows, cols, vals = [np.arange(nf)], [np.arange(nf)], [np.ones(nf)]
        keep = np.ones(nf, bool)
        ident = np.ones(nf, bool)
        for n, (T, k) in constraints.frames.items():
            g = gmap[dim * n + np.arange(dim)]
            if np.any(g < 0):
                raise ValueError(f"node {n}: a locally constrained node must have all its DOFs free and active")
            ident[g] = False
            rr, cc = np.meshgrid(g, g, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(T.ravel())
            keep[g[:k]] = False
        rows[0], cols[0], vals[0] = rows[0][ident], cols[0][ident], vals[0][ident]
        self.T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, nf))
        self.keep = np.flatnonzero(keep)
        S = sp.csr_matrix((nf, nf))
        if len(constraints.spring_nodes):
            sr, sc, sv = [], [], []
            for n, K in zip(constraints.spring_nodes, constraints.spring_K, strict=True):
                g = gmap[dim * int(n) + np.arange(dim)]
                ok = g >= 0
                rr, cc = np.meshgrid(g[ok], g[ok], indexing="ij")
                sr.append(rr.ravel())
                sc.append(cc.ravel())
                sv.append(K[np.ix_(ok, ok)].ravel())
            S = sp.csr_matrix((np.concatenate(sv), (np.concatenate(sr), np.concatenate(sc))), shape=(nf, nf))
        self.S = S
        self.constraints = constraints

    def matrix(self, E_active: np.ndarray) -> sp.csr_matrix:
        A = super().matrix(E_active)
        K = (A + sp.triu(A, k=1).T + self.S).tocsr()
        Kl = (self.T.T @ K @ self.T).tocsr()[self.keep][:, self.keep]
        out = sp.triu(Kl).tocsr()
        out.sort_indices()
        return out

    def _solve_free(self, B_full: np.ndarray) -> np.ndarray:
        X = np.zeros_like(B_full)
        b = B_full[self.free]
        b2 = b[:, None] if b.ndim == 1 else b
        bl = (self.T.T @ b2)[self.keep]
        y = np.asarray(self.solver.solve(bl)).reshape(bl.shape)
        z = np.zeros_like(b2)
        z[self.keep] = y
        x = self.T @ z
        X[self.free] = x.reshape(b.shape)
        return X


class PerCaseSupports:
    """Physics for load cases with different supports: column groups solved by their own physics.

    ``groups``: list of (physics, column indices). ``solve`` returns (ndof, n_cases) in the original column order.
    """

    def __init__(self, groups: Sequence[tuple[Any, Sequence[int]]]) -> None:
        self.groups = [(p, np.asarray(c, np.int64)) for p, c in groups]
        cols = np.concatenate([c for _, c in self.groups])
        if sorted(cols.tolist()) != list(range(len(cols))):
            raise ValueError("column groups must partition 0 ... n_cases-1")
        self._inv = torch.as_tensor(np.argsort(cols), dtype=torch.long)
        first = self.groups[0][0]
        self.domain, self.elements, self.geom, self.KE = first.domain, first.elements, first.geom, first.KE
        self.n_cases = len(cols)

    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.shape[1] != self.n_cases:
            raise ValueError(f"F has {F2.shape[1]} columns, the support groups cover {self.n_cases}")
        parts = [p.solve(E_active, F2[:, c]) for p, c in self.groups]
        return torch.cat(parts, dim=1).index_select(1, self._inv.to(E_active.device))


def reaction_forces(physics: Elasticity, E_active: np.ndarray, U: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(nn, dim[, n_cases]) ``K U − F`` with the unconstrained stiffness: support reactions at constrained nodes
    (≈ 0 at free nodes; at spring nodes it is the spring force). Σ over all nodes equals −Σ F (equilibrium)."""
    d = physics.domain
    U2 = U[:, None] if U.ndim == 1 else U
    F2 = F[:, None] if F.ndim == 1 else F
    Ue = U2[physics.edof]                                           # (na, nloc, nc)
    fe = np.einsum("ab,ebc->eac", physics.KE, Ue) * (np.asarray(E_active, float) * physics.geom)[:, None, None]
    R = np.zeros_like(U2)
    np.add.at(R, physics.edof.ravel(), fe.reshape(-1, U2.shape[1]))
    R -= F2
    out = R.reshape(d.nn, d.dim, -1)
    return out[..., 0] if U.ndim == 1 else out
