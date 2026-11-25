"""Fill-reducing elimination order for matrices on a regular node grid: geometric nested dissection.

The SPD systems of :mod:`toporlogy.fem` live on the node grid of a :class:`~toporlogy.Domain`. Nested dissection
(George 1973) splits the node box along its longest axis by one plane of nodes (the separator), orders both halves
recursively and the separator last; a box with at most ``leaf`` nodes (or less than three nodes along its longest
axis) keeps its C order. On a grid the separators are known exactly, so no graph partitioner is needed.

Measured on this repository's grids against SuperLU's ``MMD_AT_PLUS_A`` (the previous order;
``docs/olcumler/cozucu_siralama.md``): fill 24.3 → 18.0 and factorisation 2.1x faster at 52 920 DOF (3-D), 2.5x at
241 200 DOF (2-D); compliance unchanged to 2e-14. :class:`~toporlogy.solvers.DirectSolver` uses the order with
diagonal pivots (``SymmetricMode``, ``diag_pivot_thresh = 0``), i.e. as a symmetric elimination order, which is
stable for SPD matrices only: callers set it for stiffness and conductivity matrices, not for K − σM or tangents.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from toporlogy.references import cite

__all__ = ["LEAF_NODES", "nested_dissection_nodes", "nested_dissection_dofs"]

LEAF_NODES = 64
"""Boxes with at most this many nodes are not split further (16, 64 and 256 measured; 64 the fastest, 2-D and 3-D)."""


@cite("george_1973")
def nested_dissection_nodes(node_shape: Sequence[int], leaf: int = LEAF_NODES) -> np.ndarray:
    """Permutation of the C-order node numbers of a grid with ``node_shape`` nodes per axis (every node once)."""
    dims = tuple(int(n) for n in node_shape)
    if not dims or min(dims) < 1:
        raise ValueError(f"node_shape must be positive, got {node_shape}")
    if leaf < 1:
        raise ValueError("leaf >= 1 is required")
    strides = np.array([int(np.prod(dims[i + 1:], dtype=np.int64)) for i in range(len(dims))], dtype=np.int64)
    parts: list[np.ndarray] = []

    def box(lo: list[int], hi: list[int]) -> np.ndarray:
        grids = np.meshgrid(*[np.arange(a, b, dtype=np.int64) for a, b in zip(lo, hi, strict=True)], indexing="ij")
        return np.asarray(sum(g.ravel() * s for g, s in zip(grids, strides, strict=True)), dtype=np.int64)

    stack: list[tuple[list[int], list[int], bool]] = [(list([0] * len(dims)), list(dims), False)]
    # explicit stack (post-order: part 1, part 2, separator) so deep grids cannot hit the recursion limit
    while stack:
        lo, hi, emit = stack.pop()
        if emit:
            parts.append(box(lo, hi))
            continue
        ext = [b - a for a, b in zip(lo, hi, strict=True)]
        if min(ext) <= 0:
            continue
        a = int(np.argmax(ext))
        if int(np.prod(ext)) <= leaf or ext[a] < 3:
            parts.append(box(lo, hi))
            continue
        mid = (lo[a] + hi[a]) // 2
        hi1, lo2, los, his = list(hi), list(lo), list(lo), list(hi)
        hi1[a], lo2[a], los[a], his[a] = mid, mid + 1, mid, mid + 1
        stack.append((los, his, True))      # popped last: the separator comes after both halves
        stack.append((lo2, hi, False))
        stack.append((lo, hi1, False))
    return np.concatenate(parts)


def nested_dissection_dofs(node_shape: Sequence[int], dofs_per_node: int, free: np.ndarray,
                           leaf: int = LEAF_NODES) -> np.ndarray:
    """Order of the reduced (free-DOF) system: ``perm[k]`` = index into ``free`` of the k-th unknown eliminated.

    DOF numbering as :class:`~toporlogy.Domain`: DOF ``dofs_per_node · node + d``; the DOFs of one node stay together.
    """
    free = np.asarray(free, np.int64)
    nodes = nested_dissection_nodes(node_shape, leaf)
    dofs = (dofs_per_node * nodes[:, None] + np.arange(dofs_per_node, dtype=np.int64)[None, :]).ravel()
    n_total = int(np.prod([int(n) for n in node_shape], dtype=np.int64)) * dofs_per_node
    if free.size and (free.min() < 0 or free.max() >= n_total):
        raise ValueError("free DOF outside the node grid")
    pos = np.full(n_total, -1, np.int64)
    pos[free] = np.arange(len(free), dtype=np.int64)
    p = pos[dofs]
    return p[p >= 0]
