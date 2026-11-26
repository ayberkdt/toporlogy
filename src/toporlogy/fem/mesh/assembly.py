"""Sparse assembly with a reusable sparsity pattern.

The pattern of ``Σ_e P_eᵀ K_e P_e`` depends only on the element DOF lists; it is built once (:class:`Pattern`) and
every later assembly is one ``bincount`` into the stored CSR structure (the same scheme as
:class:`toporlogy.fem.elastic.Elasticity`). Rectangular blocks (the u-p coupling of a mixed formulation) use the
same class with different row and column DOF lists.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["Pattern", "assemble_vector"]


class Pattern:
    """CSR pattern of a sum of element blocks.

    Parameters
    ----------
    rows : (ne, m) int
        Global row index of each element row.
    cols : (ne, n) int, optional
        Global column index of each element column (default ``rows``: a square, symmetric pattern).
    shape : (int, int)
    """

    def __init__(self, rows: np.ndarray, shape: tuple[int, int], cols: np.ndarray | None = None) -> None:
        rows = np.asarray(rows, np.int64)
        cols = rows if cols is None else np.asarray(cols, np.int64)
        if rows.ndim != 2 or cols.ndim != 2 or len(rows) != len(cols):
            raise ValueError("rows (ne, m) and cols (ne, n) with the same ne")
        if rows.size and (rows.min() < 0 or rows.max() >= shape[0] or cols.min() < 0 or cols.max() >= shape[1]):
            raise ValueError("element DOF out of range")
        self.shape = (int(shape[0]), int(shape[1]))
        self.block = (rows.shape[1], cols.shape[1])
        self.n_elements = len(rows)
        r = np.repeat(rows, cols.shape[1], axis=1).ravel()
        c = np.tile(cols, (1, rows.shape[1])).ravel()
        key = r * self.shape[1] + c
        ukey, inv = np.unique(key, return_inverse=True)
        self._inv = inv.astype(np.int64)
        self._csr = sp.csr_matrix((np.zeros(len(ukey)), (ukey // self.shape[1], ukey % self.shape[1])),
                                  shape=self.shape)
        self._csr.sort_indices()
        self.n_assemblies = 0

    @property
    def nnz(self) -> int:
        return int(self._csr.nnz)

    def assemble(self, Ke: np.ndarray) -> sp.csr_matrix:
        """Global matrix of element blocks Ke (ne, m, n) in this pattern (a new CSR matrix, shared structure)."""
        Ke = np.asarray(Ke, float)
        if Ke.shape != (self.n_elements, *self.block):
            raise ValueError(f"element blocks must be ({self.n_elements}, {self.block[0]}, {self.block[1]}), "
                             f"got {Ke.shape}")
        A = self._csr.copy()
        A.data = np.bincount(self._inv, weights=Ke.ravel(), minlength=self._csr.nnz)
        self.n_assemblies += 1
        return A


def assemble_vector(n: int, dofs: np.ndarray, fe: np.ndarray) -> np.ndarray:
    """Global vector (n,) of element vectors fe (ne, m) at element DOFs ``dofs`` (ne, m)."""
    return np.bincount(np.asarray(dofs, np.int64).ravel(), weights=np.asarray(fe, float).ravel(), minlength=n)
