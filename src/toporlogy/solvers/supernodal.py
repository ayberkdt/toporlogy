"""Supernodal (multifrontal) sparse Cholesky for SPD matrices with a given elimination order and block partition.

``K = L Lᵀ`` in a given elimination order ``perm`` whose unknowns are grouped into consecutive blocks (supernodes):
each block is eliminated as one dense frontal matrix with LAPACK/BLAS (``potrf``, ``trsm``, ``gemm``), so the dense
work runs in the multithreaded BLAS of NumPy/SciPy instead of SuperLU's sequential kernels, and the Cholesky does
half the arithmetic of an LU. The grid physics pass the geometric nested dissection of
:mod:`toporlogy.solvers.ordering`, whose parts (leaf boxes and separator planes) are the blocks.

Symbolic analysis (once per sparsity pattern, reused while the pattern does not change): for block b with unknowns
``B`` (positions ``s_b … e_b − 1`` of the order), its structure is

    struct(b) = {i ≥ e_b : K[i, j] ≠ 0 structurally for some j ∈ B} ∪ ⋃_{children c} {i ∈ struct(c) : i ≥ e_b},

and the parent of b is the block of min(struct(b)). This is the block elimination structure, a superset of the
column structures of the scalar Cholesky factor (columns of one block share one row set), so the factor is exact
(rounding aside) for any consecutive partition; a poor partition only stores more zeros.

Numeric factorisation of block b (front ``[B, struct(b)]``):

    F11 = K[B, B] + Σ_c U_c[B, B],   F21 = K[S, B] + Σ_c U_c[S, B],   F22 = Σ_c U_c[S, S]      (S = struct(b))
    L11 = chol(F11),   L21 = F21 L11⁻ᵀ,   U_b = F22 − L21 L21ᵀ  (passed to the parent).

A non-positive pivot (the matrix is not SPD at these values, or singular) raises :class:`NotPositiveDefiniteError`;
there is no pivoting and no fallback.

The result differs from SuperLU's LU (same order, diagonal pivots) by rounding only; it is not bit for bit the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

__all__ = ["SupernodalCholesky", "NotPositiveDefiniteError", "AMALGAMATE"]

AMALGAMATE = 0
"""Default of ``amalgamate``: a block is merged with its parent when the parent is the next block in the order and the
merged block has at most this many unknowns (fewer, larger dense blocks: less Python per block, more stored zeros).
0 = off. Measured post hoc (docs/olcumler/hizlandirma.md §P1): 256 / 1024 make the 2-D factorisation 1.5x / 1.8x
faster at 60 600 DOF but store more than SuperLU's L + U at 120 x 40 (the pre-registered H3g fails), and do not help
in 3-D; so off by default."""


class NotPositiveDefiniteError(RuntimeError):
    """A pivot block of the supernodal Cholesky is not positive definite."""


@dataclass
class _Block:
    start: int                      # first position in the elimination order
    stop: int                       # one past the last
    struct: np.ndarray              # positions (in the order) of the later unknowns coupled to the block, ascending
    a_src: np.ndarray               # indices into A_upper.data of the entries assembled into this front
    a_dst11: np.ndarray             # their flat positions in F11 (ns × ns) or -1
    a_dst21: np.ndarray             # their flat positions in F21 (len(struct) × ns) or -1 (aligned with a_src)
    children: list[tuple[int, int, Any, Any]]   # (child, k1, pos in block, pos in struct) as torch index tensors


class SupernodalCholesky:
    """Factor and solve an SPD matrix given by its upper triangle (CSR, as :class:`~toporlogy.solvers.DirectSolver`).

    Parameters
    ----------
    order : array (n,)
        Elimination order: ``order[k]`` is the unknown eliminated k-th.
    blocks : array of int
        Sizes of the consecutive groups of ``order`` eliminated as one dense block; they sum to n.
    amalgamate : int
        Merge a block into its parent when the parent follows it and the merged size is at most this (0: never).
    """

    def __init__(self, order: np.ndarray, blocks: np.ndarray, amalgamate: int = AMALGAMATE) -> None:
        self.amalgamate = int(amalgamate)
        self.order = np.asarray(order, np.int64)
        n = len(self.order)
        sizes = np.asarray(blocks, np.int64)
        if sizes.ndim != 1 or (sizes < 0).any() or int(sizes.sum()) != n:
            raise ValueError(f"blocks: non-negative sizes summing to the number of unknowns ({n})")
        if not np.array_equal(np.sort(self.order), np.arange(n)):
            raise ValueError("order is not a permutation")
        self.n = n
        sizes = sizes[sizes > 0]
        self._given = np.concatenate([[0], np.cumsum(sizes)])     # the partition as given
        self._bounds = self._given                                # after amalgamation (set by the analysis)
        self._pattern: tuple[np.ndarray, np.ndarray] | None = None
        self._blocks: list[_Block] = []
        self._L11: list = []                                 # torch tensors (lower factors per block)
        self._L21: list = []
        self.factor_nnz = 0                                  # stored entries of L (dense blocks, both parts)

    def matches(self, order: np.ndarray, blocks: np.ndarray) -> bool:
        """True when ``order`` and ``blocks`` are those of this instance (its analysis can be reused)."""
        sizes = np.asarray(blocks, np.int64)
        sizes = sizes[sizes > 0]
        return (np.array_equal(np.asarray(order), self.order)
                and np.array_equal(np.concatenate([[0], np.cumsum(sizes)]), self._given))

    # ------------------------------------------------------------------ symbolic
    def _analyse(self, A: sp.csr_matrix) -> None:
        self._bounds = self._given
        self._structure(A)
        if self.amalgamate > 0:
            b = self._bounds
            keep = [0]
            size = 0
            for k, blk in enumerate(self._blocks[:-1]):
                size += blk.stop - blk.start
                nxt = self._blocks[k + 1]
                parent_next = len(blk.struct) > 0 and nxt.start <= blk.struct[0] < nxt.stop
                if parent_next and size + (nxt.stop - nxt.start) <= self.amalgamate:
                    continue                                       # merge k with k + 1
                keep.append(k + 1)
                size = 0
            keep.append(len(self._blocks))
            merged = b[np.asarray(keep)]
            if len(merged) < len(b):
                self._bounds = merged
                self._structure(A)

    def _structure(self, A: sp.csr_matrix) -> None:
        import torch
        n = self.n
        iperm = np.empty(n, np.int64)
        iperm[self.order] = np.arange(n)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
        pr, pc = iperm[rows], iperm[A.indices.astype(np.int64)]
        lo_row, lo_col = np.maximum(pr, pc), np.minimum(pr, pc)       # lower-triangle position in the order
        bounds = self._bounds
        nb = len(bounds) - 1
        blk_of = np.repeat(np.arange(nb), np.diff(bounds))
        col_blk = blk_of[lo_col]
        srt = np.argsort(col_blk, kind="stable")
        cuts = np.searchsorted(col_blk[srt], np.arange(nb + 1))
        pending: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(nb)]
        blocks: list[_Block] = []
        nnz = 0
        for b in range(nb):
            s0, s1 = int(bounds[b]), int(bounds[b + 1])
            ns = s1 - s0
            src = srt[cuts[b]:cuts[b + 1]]
            r, c = lo_row[src], lo_col[src] - s0
            parts = [r[r >= s1]] + [st[st >= s1] for _, st in pending[b]]
            struct = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
            in11 = r < s1
            d11 = np.where(in11, (r - s0) * ns + c, -1)
            d21 = np.where(in11, -1, np.searchsorted(struct, r) * ns + c)
            children = []
            for child, st in pending[b]:
                k1 = int(np.searchsorted(st, s1))
                children.append((child, k1, torch.from_numpy(st[:k1] - s0),
                                 torch.from_numpy(np.searchsorted(struct, st[k1:]))))
            blocks.append(_Block(s0, s1, struct, src, d11, d21, children))
            nnz += ns * (ns + 1) // 2 + ns * len(struct)
            if len(struct):
                pending[int(blk_of[struct[0]])].append((b, struct))
            pending[b] = []
        self._blocks = blocks
        self.factor_nnz = nnz
        self._pattern = (A.indptr.copy(), A.indices.copy())

    # ------------------------------------------------------------------ numeric
    def factor(self, A_upper: sp.csr_matrix) -> None:
        import torch                       # lazily: toporlogy loads PARDISO before torch (toporlogy/__init__)
        A = sp.csr_matrix(A_upper)
        if A.shape != (self.n, self.n):
            raise ValueError(f"matrix has shape {A.shape}, the order {self.n} unknowns")
        if (self._pattern is None or not np.array_equal(self._pattern[0], A.indptr)
                or not np.array_equal(self._pattern[1], A.indices)):
            self._analyse(A)
        data = np.asarray(A.data, float)
        updates: dict[int, Any] = {}
        L11s: list[torch.Tensor] = []
        L21s: list[torch.Tensor] = []
        for b, blk in enumerate(self._blocks):
            ns, nsr = blk.stop - blk.start, len(blk.struct)
            F11 = np.zeros((ns, ns))
            F21 = np.zeros((nsr, ns))
            F22 = np.zeros((nsr, nsr))
            m11 = blk.a_dst11 >= 0
            F11.ravel()[blk.a_dst11[m11]] = data[blk.a_src[m11]]
            F21.ravel()[blk.a_dst21[~m11]] = data[blk.a_src[~m11]]
            T11, T21, T22 = torch.from_numpy(F11), torch.from_numpy(F21), torch.from_numpy(F22)
            for child, k1, p1, p2 in blk.children:
                U = updates.pop(child)
                # extend-add: one addition per position (indices unique), the same as F[ix] += U
                if k1:
                    T11.index_put_((p1[:, None], p1[None, :]), U[:k1, :k1], accumulate=True)
                    if len(p2):
                        T21.index_put_((p2[:, None], p1[None, :]), U[k1:, :k1], accumulate=True)
                if len(p2):
                    T22.index_put_((p2[:, None], p2[None, :]), U[k1:, k1:], accumulate=True)
            # F11 holds K's entries in its lower triangle and the children's updates in both; Cholesky reads the
            # lower one
            L11, info = torch.linalg.cholesky_ex(T11)
            if int(info) != 0:
                raise NotPositiveDefiniteError(
                    f"supernodal Cholesky: block {b} (order positions {blk.start}..{blk.stop - 1}) is not positive "
                    f"definite (pivot {int(info)}): the matrix is singular or not SPD at these values")
            if nsr:
                L21 = torch.linalg.solve_triangular(L11.mT, T21, upper=True, left=False)
                T22.addmm_(L21, L21.mT, alpha=-1.0)              # F22 − L21 L21ᵀ, in place
                updates[b] = T22
            else:
                L21 = T21
            L11s.append(L11)
            L21s.append(L21)
        self._L11, self._L21 = L11s, L21s

    def solve(self, B: np.ndarray) -> np.ndarray:
        """Solve with the current factor; ``B`` (n,) or (n, k) in the original numbering."""
        import torch
        if not self._L11:
            raise RuntimeError("factor() first")
        B = np.asarray(B, float)
        Yn = B.reshape(self.n, -1)[self.order]
        Y = torch.from_numpy(Yn)
        for blk, L11, L21 in zip(self._blocks, self._L11, self._L21, strict=True):
            y = torch.linalg.solve_triangular(L11, Y[blk.start:blk.stop], upper=False)
            Y[blk.start:blk.stop] = y
            if len(blk.struct):
                Yn[blk.struct] -= (L21 @ y).numpy()
        for blk, L11, L21 in zip(reversed(self._blocks), reversed(self._L11), reversed(self._L21), strict=True):
            r = Y[blk.start:blk.stop]
            if len(blk.struct):
                r = r - L21.mT @ torch.from_numpy(Yn[blk.struct])
            Y[blk.start:blk.stop] = torch.linalg.solve_triangular(L11.mT, r, upper=True)
        X = np.empty_like(Yn)
        X[self.order] = Yn
        return X.reshape(B.shape)
