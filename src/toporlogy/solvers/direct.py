"""Sparse direct solver for the reduced (free-DOF) system: MKL PARDISO when loadable, else SciPy SuperLU; on request
the supernodal Cholesky of :mod:`toporlogy.solvers.supernodal`.

The factorisation of the last matrix is kept, so the adjoint solve of the same iteration is a back-substitution
only. PARDISO is located by :mod:`toporlogy.solvers.pardiso`.

Elimination order. ``factor(A, order=perm)`` (SuperLU only) eliminates the unknowns in the given order with diagonal
pivots (``SymmetricMode``, ``diag_pivot_thresh = 0``): a symmetric elimination, stable for SPD matrices. The grid
physics pass a geometric nested dissection (:mod:`toporlogy.solvers.ordering`); without ``order`` SuperLU orders
by ``MMD_AT_PLUS_A`` with its default threshold pivoting, as before. PARDISO always uses its own ordering.

``method="cholesky"`` (opt-in, never chosen by "auto"): supernodal Cholesky in the given order with the given block
partition, ``factor(A, order=perm, blocks=sizes)``; the grid physics pass the nested dissection and its parts. SPD
matrices only; the symbolic analysis is kept while the sparsity pattern and the order do not change. It agrees with
SuperLU to rounding, not bit for bit (docs/olcumler/hizlandirma.md).
"""
from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["DirectSolver", "METHODS"]

METHODS = ("auto", "pardiso", "splu", "cholesky")


class DirectSolver:
    """Factor once per matrix, solve many right-hand sides.

    Parameters
    ----------
    method : {"auto", "pardiso", "splu", "cholesky"}
        "auto" = PARDISO if it can be loaded, else SuperLU. "cholesky" = supernodal Cholesky (module notes).
    """

    name = "direct"

    def __init__(self, method: str = "auto") -> None:
        if method not in METHODS:
            raise ValueError(f"method: {' | '.join(METHODS)}, got {method!r}")
        pp = None
        if method in ("auto", "pardiso"):
            from toporlogy.solvers import pardiso
            pp = pardiso.load()
            if pp is None and method == "pardiso":
                raise RuntimeError(f"PARDISO requested but not available: {pardiso.state()['reason']}")
        self._pp: Any = pp
        self.method = method if method == "cholesky" else ("pardiso" if pp is not None else "splu")
        self.name = f"direct-{self.method}"
        self._solver: Any = None
        self._A: sp.csr_matrix | None = None
        self._order: np.ndarray | None = None
        self._perm_map: _PermutedPattern | None = None    # SuperLU input in elimination order, per pattern
        self._chol: Any = None                             # SupernodalCholesky, its analysis kept across factorisations
        self.n_factor = 0
        self.n_solve = 0

    def factor(self, A_upper: sp.csr_matrix, order: np.ndarray | None = None,
               blocks: np.ndarray | None = None) -> None:
        """Factorise the SPD matrix given by its upper triangle (CSR, free DOFs only). ``order``: elimination order
        (a permutation of the unknowns) for SuperLU, applied with diagonal pivots; ignored by PARDISO. ``blocks``:
        sizes of the consecutive groups of ``order`` eliminated as dense blocks; required with ``order`` by method
        "cholesky", ignored by the others."""
        if self.method == "cholesky":
            self._factor_cholesky(A_upper, order, blocks)
            return
        self.release()
        if self._pp is not None:
            s = self._pp.PyPardisoSolver(mtype=2)
            s.factorize(A_upper)
            self._solver, self._A = s, A_upper
        else:
            if order is None:
                full = (A_upper + sp.triu(A_upper, k=1).T).tocsr()
                self._solver = spla.splu(full.tocsc(), permc_spec="MMD_AT_PLUS_A")
            else:
                order = np.asarray(order, np.int64)
                if order.shape != (A_upper.shape[0],):
                    raise ValueError(f"order: one entry per unknown ({A_upper.shape[0]}), got shape {order.shape}")
                pm = self._perm_map
                if pm is None or not pm.matches(A_upper, order):
                    pm = self._perm_map = _PermutedPattern(A_upper, order)
                self._solver = spla.splu(pm.matrix(A_upper.data), permc_spec="NATURAL",
                                         diag_pivot_thresh=0.0, options={"SymmetricMode": True})
                self._order = order
            self._A = A_upper
        self.n_factor += 1

    def _factor_cholesky(self, A_upper: sp.csr_matrix, order: np.ndarray | None, blocks: np.ndarray | None) -> None:
        if order is None or blocks is None:
            raise ValueError("method 'cholesky' needs the elimination order and its blocks, factor(A, order=..., "
                             "blocks=...); the grid physics (Elasticity, HeatConduction) pass the nested dissection")
        from toporlogy.solvers.supernodal import SupernodalCholesky
        self._solver, self._A = None, None
        ch = self._chol
        if ch is None or not ch.matches(order, blocks):
            ch = self._chol = SupernodalCholesky(order, blocks)
        ch.factor(A_upper)
        self._solver, self._A = ch, A_upper
        self.n_factor += 1

    def solve(self, B: np.ndarray) -> np.ndarray:
        """Solve with the current factorisation; ``B`` (n,) or (n, k)."""
        if self._solver is None:
            raise RuntimeError("factor() first")
        B = np.ascontiguousarray(B, dtype=float)
        self.n_solve += 1
        if self._pp is not None:
            return np.asarray(self._solver.solve(self._A, B)).reshape(B.shape)
        if self.method == "cholesky" or self._order is None:
            return np.asarray(self._solver.solve(B)).reshape(B.shape)
        X = np.empty_like(B)
        X[self._order] = np.asarray(self._solver.solve(np.ascontiguousarray(B[self._order]))).reshape(B.shape)
        return X

    def release(self) -> None:
        if self._pp is not None and self._solver is not None:
            self._solver.free_memory(everything=True)
        self._solver, self._A, self._order = None, None, None

    def __del__(self) -> None:  # pragma: no cover - best effort
        with contextlib.suppress(Exception):
            self.release()


class _PermutedPattern:
    """Index map from the upper triangle (CSR) of a symmetric matrix to the CSC arrays of ``K[order][:, order]``.

    Built once per sparsity pattern and order; :meth:`matrix` then gathers the values each factorisation. It gives
    the same CSC arrays (values, row indices, column pointers) as the earlier construction
    ``(A + triu(A, 1).T).tocsr()[order][:, order].tocsc()``: the values are copied, never summed (an upper entry
    and its mirror are distinct positions), and entries that are exactly zero are left out, as SciPy's sparse
    addition leaves them out. Same input to SuperLU, so the same factor bit for bit (test_direct_solver).
    """

    def __init__(self, A_upper: sp.csr_matrix, order: np.ndarray) -> None:
        n = A_upper.shape[0]
        self._indptr = A_upper.indptr.copy()
        self._indices = A_upper.indices.copy()
        self._order = order.copy()
        iperm = np.empty(n, np.int64)
        iperm[order] = np.arange(n)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(A_upper.indptr))
        cols = A_upper.indices.astype(np.int64)
        strict = rows != cols
        src = np.concatenate([np.arange(len(rows)), np.flatnonzero(strict)])            # upper, then mirrors
        r = np.concatenate([iperm[rows], iperm[cols[strict]]])
        c = np.concatenate([iperm[cols], iperm[rows[strict]]])
        srt = np.lexsort((r, c))                                                          # column-major, rows up
        self._src, self._row, self._col = src[srt], r[srt].astype(np.int32), c[srt]
        self._n = n
        self._full_indptr = np.searchsorted(self._col, np.arange(n + 1)).astype(np.int32)

    def matches(self, A_upper: sp.csr_matrix, order: np.ndarray) -> bool:
        return (A_upper.shape[0] == self._n and np.array_equal(A_upper.indptr, self._indptr)
                and np.array_equal(A_upper.indices, self._indices) and np.array_equal(order, self._order))

    def matrix(self, data: np.ndarray) -> sp.csc_matrix:
        v = np.asarray(data, float)[self._src]
        nz = v != 0                                       # -0.0 too, as the sparse sum leaves it out
        if nz.all():
            return sp.csc_matrix((v, self._row, self._full_indptr), shape=(self._n, self._n))
        cnt = np.bincount(self._col[nz], minlength=self._n)
        indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
        return sp.csc_matrix((v[nz], self._row[nz], indptr), shape=(self._n, self._n))
