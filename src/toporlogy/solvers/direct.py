"""Sparse direct solver for the reduced (free-DOF) system: MKL PARDISO when loadable, else SciPy SuperLU.

The factorisation of the last matrix is kept, so the adjoint solve of the same iteration is a back-substitution
only. PARDISO is located by :mod:`toporlogy.solvers.pardiso`.

Elimination order. ``factor(A, order=perm)`` (SuperLU only) eliminates the unknowns in the given order with diagonal
pivots (``SymmetricMode``, ``diag_pivot_thresh = 0``): a symmetric elimination, stable for SPD matrices. The grid
physics pass a geometric nested dissection (:mod:`toporlogy.solvers.ordering`); without ``order`` SuperLU orders
by ``MMD_AT_PLUS_A`` with its default threshold pivoting, as before. PARDISO always uses its own ordering.
"""
from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["DirectSolver"]


class DirectSolver:
    """Factor once per matrix, solve many right-hand sides.

    Parameters
    ----------
    method : {"auto", "pardiso", "splu"}
        "auto" = PARDISO if it can be loaded, else SuperLU.
    """

    name = "direct"

    def __init__(self, method: str = "auto") -> None:
        if method not in ("auto", "pardiso", "splu"):
            raise ValueError("method: auto | pardiso | splu")
        pp = None
        if method in ("auto", "pardiso"):
            from toporlogy.solvers import pardiso
            pp = pardiso.load()
            if pp is None and method == "pardiso":
                raise RuntimeError(f"PARDISO requested but not available: {pardiso.state()['reason']}")
        self._pp: Any = pp
        self.method = "pardiso" if pp is not None else "splu"
        self.name = f"direct-{self.method}"
        self._solver: Any = None
        self._A: sp.csr_matrix | None = None
        self._order: np.ndarray | None = None
        self.n_factor = 0
        self.n_solve = 0

    def factor(self, A_upper: sp.csr_matrix, order: np.ndarray | None = None) -> None:
        """Factorise the SPD matrix given by its upper triangle (CSR, free DOFs only). ``order``: elimination order
        (a permutation of the unknowns) for SuperLU, applied with diagonal pivots; ignored by PARDISO."""
        self.release()
        if self._pp is not None:
            s = self._pp.PyPardisoSolver(mtype=2)
            s.factorize(A_upper)
            self._solver, self._A = s, A_upper
        else:
            full = (A_upper + sp.triu(A_upper, k=1).T).tocsr()
            if order is None:
                self._solver = spla.splu(full.tocsc(), permc_spec="MMD_AT_PLUS_A")
            else:
                order = np.asarray(order, np.int64)
                if order.shape != (full.shape[0],):
                    raise ValueError(f"order: one entry per unknown ({full.shape[0]}), got shape {order.shape}")
                self._solver = spla.splu(full[order][:, order].tocsc(), permc_spec="NATURAL",
                                         diag_pivot_thresh=0.0, options={"SymmetricMode": True})
                self._order = order
            self._A = A_upper
        self.n_factor += 1

    def solve(self, B: np.ndarray) -> np.ndarray:
        """Solve with the current factorisation; ``B`` (n,) or (n, k)."""
        if self._solver is None:
            raise RuntimeError("factor() first")
        B = np.ascontiguousarray(B, dtype=float)
        self.n_solve += 1
        if self._pp is not None:
            return np.asarray(self._solver.solve(self._A, B)).reshape(B.shape)
        if self._order is None:
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
