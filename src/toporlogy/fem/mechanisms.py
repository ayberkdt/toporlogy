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

The number of such motions is ``n_unknowns − rank``. Clusters couple only through shared nodes, so the system is
block-diagonal by connected component of the cluster contact graph and each component is decided on its own: rank
from the singular values of its block, coordinates centred and scaled on the component (rotations and translations of
the same order). Cost (measured 18 Nov 2026): 1000 separately supported islands 0.5 s (one dense 6000-unknown problem
before: 28.9 s, 574 MB); one component of 500 clusters (3000 unknowns, :data:`MAX_COMPONENT_UNKNOWNS`) 3.0 s. Above
that limit :class:`MechanismCheckIndeterminate` is raised; the check never passes by default.

Contract (PR-02). The check is exact under these assumptions, and they are enforced: every active element has a
positive, finite modulus (:func:`check_moduli`, called before every factorisation, also by the multigrid and the
modal paths); the support set of a physics object is frozen (``fixed`` and ``free`` are read-only), so the result
computed once per object stays valid when only the moduli change. It does not detect ill-conditioning (very small
but positive moduli, thin necks), constitutive instability or buckling; ill-conditioning is left to the per-load-case
residual check (``fem.elastic.checked_residual``). The three failures carry distinct codes: ``mechanism``
(:class:`MechanismError`), ``invalid_moduli`` (:class:`InvalidModuliError`), ``solve_residual``
(:class:`SolveResidualError`).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse import csgraph

__all__ = ["MechanismError", "SolveResidualError", "InvalidModuliError", "MechanismCheckIndeterminate",
           "MAX_COMPONENT_UNKNOWNS", "check_moduli", "fixed_dof_rows",
           "zero_energy_modes", "check_supports"]


class MechanismError(ValueError):
    """The supports leave the part a motion without strain energy: the stiffness system is singular."""

    code = "mechanism"


class SolveResidualError(RuntimeError):
    """A linear solve returned x with ‖K x − b‖ / ‖b‖ above the tolerance (singular or unsolvable system)."""

    code = "solve_residual"


class MechanismCheckIndeterminate(RuntimeError):
    """The support check could not be decided within its size limit (one contact component too large)."""

    code = "mechanism_indeterminate"


class InvalidModuliError(ValueError):
    """Element moduli (or densities) that break the model's assumptions: wrong shape, not finite, or not positive."""

    code = "invalid_moduli"


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


def check_moduli(values: Any, n: int, what: str = "modulus", allow_zero: bool = False) -> np.ndarray:
    """Element moduli (or densities) as a float array of shape (n,), or raise :class:`InvalidModuliError`.

    The support check is exact only for positive, finite moduli: then every element's stiffness has exactly the rigid
    motions as null space. A zero modulus would add motions the check does not see; a negative one makes K
    indefinite, which a direct solver still solves with a small residual (the residual check cannot catch a wrong
    material model). Very small positive values (SIMP voids, 1e-9 E0) are legal: high contrast is ill-conditioning,
    not a mechanism. A scalar is broadcast. ``allow_zero``: zero is legal (element densities: the two-range mass
    interpolation makes void elements massless on purpose; the mass matrix may be semi-definite), negative is not.
    """
    arr = np.asarray(values, float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise InvalidModuliError(f"{what}: shape {arr.shape}, expected ({n},) (one value per active element)")
    bad = ~np.isfinite(arr)
    if bad.any():
        raise InvalidModuliError(f"{what}: {int(bad.sum())} value(s) not finite (first at active element "
                                 f"{int(np.flatnonzero(bad)[0])})")
    nonpos = arr < 0 if allow_zero else arr <= 0
    if nonpos.any():
        i = int(np.flatnonzero(nonpos)[0])
        kind = "negative" if allow_zero else "not positive"
        need = "non-negative" if allow_zero else "positive"
        raise InvalidModuliError(f"{what}: {int(nonpos.sum())} value(s) {kind} (first at active element {i}: "
                                 f"{arr[i]:g}); the model assumes {need} values")
    return arr


def fixed_dof_rows(fixed: np.ndarray, dim: int) -> dict[int, np.ndarray]:
    """Node → unit rows of its fixed components (DOF = node · dim + component)."""
    out: dict[int, list[np.ndarray]] = {}
    for dof in np.asarray(fixed, np.int64).tolist():
        out.setdefault(dof // dim, []).append(np.eye(dim)[dof % dim])
    return {n: np.array(r) for n, r in out.items()}


MAX_COMPONENT_UNKNOWNS = 3000   # rigid-motion unknowns of ONE contact component decided exactly (dense SVD, ~n³)


def check_supports(domain: Any, elements: np.ndarray, rows: Mapping[int, np.ndarray],
                   max_unknowns: int = MAX_COMPONENT_UNKNOWNS) -> None:
    """Raise :class:`MechanismError` when the supports leave a zero-energy motion (see module notes), or
    :class:`MechanismCheckIndeterminate` when a contact component exceeds ``max_unknowns``."""
    n, desc = zero_energy_modes(domain, elements, rows, max_unknowns=max_unknowns)
    if n:
        raise MechanismError(f"the supports leave {n} zero-energy motion(s), the stiffness is singular; free: "
                             + "; ".join(desc))




def zero_energy_modes(domain: Any, elements: np.ndarray, rows: Mapping[int, np.ndarray], tol: float = 1e-9,
                      max_unknowns: int = MAX_COMPONENT_UNKNOWNS) -> tuple[int, list[str]]:
    """Number of independent zero-energy motions and a description of the free clusters (largest first).

    ``elements``: active element numbers of the part; ``rows``: node → (m, dim) support rows acting on that node's
    displacement (see module notes). Nodes of inactive elements and rows at nodes outside the part are ignored.

    Clusters are coupled only through shared nodes, so the linear system is block-diagonal by connected component of
    the cluster contact graph; every component is decided on its own (exact, not an approximation), centred and
    scaled on its own nodes, with the rank from the singular values of its block (no normal matrix, which would square
    the condition number): a singular value ≤ ``tol`` × the largest counts as zero. A component with more than
    ``max_unknowns`` unknowns raises :class:`MechanismCheckIndeterminate` instead of a long dense computation.
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
    owner = pairs[first, 1]                                         # first cluster of every part node
    multi = pairs[:, 1] != owner[np.searchsorted(nodes, pairs[:, 0])]
    sn, sc = pairs[multi, 0], pairs[multi, 1]                       # shared node, the other cluster at it
    so = owner[np.searchsorted(nodes, sn)]
    _, comp = csgraph.connected_components(sp.coo_matrix((np.ones(len(sn)), (so, sc)), shape=(k, k)),
                                           directed=False)
    # support rows, orthonormalised (a row is a constraint r · u = 0, its scale is irrelevant: a spring of stiffness
    # 1e10 next to a unit row would otherwise make the unit row look like zero)
    s_nodes, s_rows = [], []
    for node, M in rows.items():
        i = np.searchsorted(nodes, node)
        if i >= len(nodes) or nodes[i] != node:
            continue
        M = np.atleast_2d(np.asarray(M, float))
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)
        M = Vt[sv > 1e-12 * max(float(sv.max(initial=0.0)), 1e-300)]
        if len(M):
            s_nodes.append(int(node))
            s_rows.append(M)
    s_node_arr = np.asarray(s_nodes, np.int64)
    s_owner = owner[np.searchsorted(nodes, s_node_arr)] if len(s_nodes) else np.zeros(0, np.int64)
    s_comp = comp[s_owner]

    def by(keys: np.ndarray) -> dict[int, np.ndarray]:
        order = np.argsort(keys, kind="stable")
        bounds = np.flatnonzero(np.diff(keys[order])) + 1
        return {int(keys[g[0]]): g for g in np.split(order, bounds) if len(g)}

    g_clusters, g_shared, g_support, g_nodes = by(comp), by(comp[so]), by(s_comp), by(comp[owner])
    empty = np.zeros(0, np.int64)
    n_modes = 0
    free: list[int] = []
    for c, cls in g_clusters.items():
        kc = len(cls)
        sh, sup = g_shared.get(c, empty), g_support.get(c, empty)
        if kc == 1 and not len(sup):                                # a single body nothing touches: free
            n_modes += nr
            free.append(int(cls[0]))
            continue
        n_unk = nr * kc
        if n_unk > max_unknowns:
            raise MechanismCheckIndeterminate(
                f"support check indeterminate: one contact component has {kc} clusters ({n_unk} unknowns), above "
                f"the limit {max_unknowns}; raise max_unknowns (the cost grows with its cube) or check the supports "
                "of this geometry another way")
        lmap = np.full(k, -1, np.int64)
        lmap[cls] = np.arange(kc)
        pn = nodes[g_nodes[c]]
        centre = X[pn].mean(axis=0)
        L = float(np.ptp(X[pn], axis=0).max()) or 1.0
        m = dim * len(sh) + sum(len(s_rows[j]) for j in sup)
        A = np.zeros((m, n_unk))
        cols = np.arange(nr)
        if len(sh):
            Rs = _rigid_basis((X[sn[sh]] - centre) / L, dim)       # (ns, dim, nr): the node moves equally with both
            r = dim * np.arange(len(sh))
            for a in range(dim):
                A[(r + a)[:, None], lmap[so[sh]][:, None] * nr + cols] += Rs[:, a, :]
                A[(r + a)[:, None], lmap[sc[sh]][:, None] * nr + cols] -= Rs[:, a, :]
        row = dim * len(sh)
        if len(sup):
            Rsup = _rigid_basis((X[s_node_arr[sup]] - centre) / L, dim)          # one call per component
            ccol = lmap[s_owner[sup]][:, None] * nr + cols
            for q, j in enumerate(sup.tolist()):
                M = s_rows[j]
                A[row:row + len(M), ccol[q]] = M @ Rsup[q]
                row += len(M)
        if m:
            sv, Vt = np.linalg.svd(A, full_matrices=False)[1:]
            rank = int((sv > tol * max(float(sv.max(initial=0.0)), 1e-300)).sum())
        else:
            rank, Vt = 0, np.zeros((0, n_unk))
        if rank < n_unk:
            n_modes += n_unk - rank
            fixed_part = (Vt[:rank].reshape(rank, kc, nr) ** 2).sum(axis=(0, 2))   # ‖row-space projection‖² per block
            free.extend(int(v) for v in cls[nr - fixed_part > 1e-6])
    if n_modes == 0:
        return 0, []
    size = np.bincount(cl, minlength=k)
    centres = np.zeros((k, dim))
    np.add.at(centres, cl, X[enodes].mean(axis=1))
    centres /= np.maximum(size, 1)[:, None]
    fr = np.asarray(free, np.int64)
    fr = fr[np.argsort(-size[fr], kind="stable")]
    desc = [f"{int(size[c])} element(s) around ({', '.join(f'{v:.4g}' for v in centres[c])})" for c in fr[:5]]
    if len(fr) > 5:
        desc.append(f"... {len(fr) - 5} more cluster(s)")
    return n_modes, desc
