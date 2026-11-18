"""Zero-energy motions of an elastic voxel part under its supports (docs/gelistirme_plani.md B-22).

With positive element moduli the reduced stiffness is singular exactly when the part can move without strain energy
while respecting its supports. A direct solver does not report that: on a floating piece it returns finite numbers
of order 1e15, which pass every ``isfinite`` test. The question is purely kinematic, so it is answered exactly here,
before the first solve, from the geometry and the supports alone.

Model. Face-connected active elements form *clusters* that can only move rigidly together (an element's stiffness
has exactly the rigid motions as null space; Q4 and H8/H8IM do). Clusters that touch only along an edge or at a node
act as hinges or ball joints. A zero-energy motion is one rigid motion θ_c per cluster (2-D: 2 translations + 1
rotation; 3-D: 3 + 3) such that

* clusters sharing a node move that node identically, and
* every support row holds at its node: ``r · u(x) = 0`` for a fixed DOF (unit row), an oblique constraint (its
  constrained directions) or a spring (its stiffness rows, since a spring stores energy unless ``K_s u = 0``).

The number of such motions is ``n_unknowns − rank``; rank from the eigenvalues of the normal matrix of that linear
system (coordinates centred and scaled by the part size, so rotations and translations are of the same order).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

__all__ = ["MechanismError", "SolveResidualError", "fixed_dof_rows", "zero_energy_modes", "check_supports"]


class MechanismError(ValueError):
    """The supports leave the part a motion without strain energy: the stiffness system is singular."""


class SolveResidualError(RuntimeError):
    """A linear solve returned x with ‖K x − b‖ / ‖b‖ above the tolerance (singular or unsolvable system)."""


def _rigid_basis(X: np.ndarray, dim: int) -> np.ndarray:
    """(n, dim, nr) rigid-motion basis at the points X (n, dim): translations, then rotations."""
    n = len(X)
    if dim == 2:
        R = np.zeros((n, 2, 3))
        R[:, 0, 0] = R[:, 1, 1] = 1.0
        R[:, 0, 2], R[:, 1, 2] = -X[:, 1], X[:, 0]
        return R
    R = np.zeros((n, 3, 6))
    R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1.0
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    R[:, 1, 3], R[:, 2, 3] = -z, y                   # rotation about x: (0, -z, y)
    R[:, 0, 4], R[:, 2, 4] = z, -x                   # about y: (z, 0, -x)
    R[:, 0, 5], R[:, 1, 5] = -y, x                   # about z: (-y, x, 0)
    return R


def fixed_dof_rows(fixed: np.ndarray, dim: int) -> dict[int, np.ndarray]:
    """Node → unit rows of its fixed components (DOF = node · dim + component)."""
    out: dict[int, list[np.ndarray]] = {}
    for dof in np.asarray(fixed, np.int64).tolist():
        out.setdefault(dof // dim, []).append(np.eye(dim)[dof % dim])
    return {n: np.array(r) for n, r in out.items()}


def check_supports(domain: Any, elements: np.ndarray, rows: Mapping[int, np.ndarray]) -> None:
    """Raise :class:`MechanismError` when the supports leave a zero-energy motion (see module notes)."""
    n, desc = zero_energy_modes(domain, elements, rows)
    if n:
        raise MechanismError(f"the supports leave {n} zero-energy motion(s), the stiffness is singular; free: "
                             + "; ".join(desc))


def zero_energy_modes(domain: Any, elements: np.ndarray, rows: Mapping[int, np.ndarray],
                      tol: float = 1e-10) -> tuple[int, list[str]]:
    """Number of independent zero-energy motions and a description of the free clusters (largest first).

    ``elements``: active element numbers of the part; ``rows``: node → (m, dim) support rows acting on that node's
    displacement (see module notes). Nodes of inactive elements and rows at nodes outside the part are ignored.
    """
    dim = domain.dim
    nr = 3 if dim == 2 else 6
    elements = np.asarray(elements, np.int64)
    mask = np.zeros(int(np.prod(domain.shape)), bool)
    mask[elements] = True
    lab, k = ndimage.label(mask.reshape(domain.shape), structure=ndimage.generate_binary_structure(dim, 1))
    cl = lab.reshape(-1)[elements] - 1                              # cluster of each element
    enodes = domain.edof[elements][:, ::dim] // dim                 # (ne, 2^dim)
    pairs = np.unique(np.column_stack([enodes.ravel(), np.repeat(cl, enodes.shape[1])]), axis=0)
    nodes, first = np.unique(pairs[:, 0], return_index=True)
    X = domain.node_coords()
    Xa = X[nodes]
    centre = Xa.mean(axis=0)
    L = float(np.ptp(Xa, axis=0).max()) or 1.0
    R = _rigid_basis((X - centre) / L, dim)                         # (nn, dim, nr)

    blocks_r, blocks_c, blocks_v = [], [], []
    nrow = 0

    def add(node_ids: np.ndarray, clusters: np.ndarray, M: np.ndarray, sign: float, row0: np.ndarray) -> None:
        """Rows row0 + i of (M_i @ R(x_i)) at the columns of cluster c_i (M: (n, m, dim))."""
        B = sign * np.einsum("nmd,ndr->nmr", M, R[node_ids])        # (n, m, nr)
        n, m, _ = B.shape
        blocks_r.append((row0[:, None, None] + np.arange(m)[None, :, None] + np.zeros((1, 1, nr), np.int64)).ravel())
        blocks_c.append((clusters[:, None, None] * nr + np.arange(nr)[None, None, :]
                         + np.zeros((1, m, 1), np.int64)).ravel())
        blocks_v.append(B.ravel())

    # shared nodes: the first cluster's motion equals every other cluster's motion there
    owner = pairs[first, 1]
    multi = pairs[:, 1] != owner[np.searchsorted(nodes, pairs[:, 0])]
    if multi.any():
        sn, sc = pairs[multi, 0], pairs[multi, 1]
        so = owner[np.searchsorted(nodes, sn)]
        eye = np.broadcast_to(np.eye(dim), (len(sn), dim, dim))
        row0 = nrow + dim * np.arange(len(sn))
        add(sn, so, eye, 1.0, row0)
        add(sn, sc, eye, -1.0, row0)
        nrow += dim * len(sn)
    # support rows
    for node, M in rows.items():
        i = np.searchsorted(nodes, node)
        if i >= len(nodes) or nodes[i] != node:
            continue
        M = np.atleast_2d(np.asarray(M, float))
        # a support row is a constraint r · u = 0, its scale is irrelevant: use an orthonormal basis of the row space
        # (a spring of stiffness 1e10 next to a unit row would otherwise make the unit row look like zero)
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)
        M = Vt[sv > 1e-12 * max(float(sv.max(initial=0.0)), 1e-300)]
        if not len(M):
            continue
        add(np.array([node]), np.array([owner[i]]), M[None], 1.0, np.array([nrow]))
        nrow += len(M)
    n_unk = nr * k
    if nrow == 0:
        return n_unk, [f"{k} cluster(s), no support acts on the part"]
    A = sp.csr_matrix((np.concatenate(blocks_v), (np.concatenate(blocks_r), np.concatenate(blocks_c))),
                      shape=(nrow, n_unk))
    N = (A.T @ A).toarray()
    w, V = np.linalg.eigh(N)
    null = w <= tol * max(float(w.max()), 1e-300)
    n_modes = int(null.sum())
    if n_modes == 0:
        return 0, []
    part = np.linalg.norm(V[:, null].reshape(k, nr, n_modes), axis=(1, 2))   # participation of each cluster
    size = np.bincount(cl, minlength=k)
    centres = np.zeros((k, dim))
    np.add.at(centres, cl, X[enodes].mean(axis=1))
    centres /= np.maximum(size, 1)[:, None]
    free = np.flatnonzero(part > 1e-6)
    free = free[np.argsort(-size[free], kind="stable")]
    desc = [f"{int(size[c])} element(s) around ({', '.join(f'{v:.4g}' for v in centres[c])})" for c in free[:5]]
    if len(free) > 5:
        desc.append(f"... {len(free) - 5} more cluster(s)")
    return n_modes, desc
