"""Linear solves of the general mesh: solver eligibility by matrix structure, factor reuse, conditioning.

* ``"spd"``: :class:`toporlogy.solvers.direct.DirectSolver` (PARDISO when loadable, else SuperLU in symmetric mode
  with diagonal pivots — valid for SPD matrices only);
* ``"symmetric_indefinite"`` / ``"general"``: SciPy SuperLU with threshold partial pivoting (saddle-point systems of
  the mixed u-p formulation); never the SPD path;
* :func:`pcg`: Jacobi-preconditioned conjugate gradients, refused for anything but ``"spd"``.

A :class:`Factorization` is bound to one matrix; solving many right-hand sides reuses it, and a different matrix
needs a new object (no silent reuse of a stale factor). Every solve is checked with
:func:`toporlogy.fem.elastic.checked_residual` (the residual / backward-error rule of the grid physics).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.fem.elastic import checked_residual, matrix_norm1
from toporlogy.fem.mechanisms import SolveResidualError
from toporlogy.solvers.direct import DirectSolver

__all__ = ["STRUCTURES", "Factorization", "pcg", "condition_1norm", "jacobi_scaled"]

STRUCTURES = ("spd", "symmetric_indefinite", "general")


class Factorization:
    """Direct factorisation of a square sparse matrix A (full storage) of a declared structure."""

    def __init__(self, A: sp.spmatrix, structure: str = "spd", solver: Any = None) -> None:
        if structure not in STRUCTURES:
            raise ValueError(f"structure: {' | '.join(STRUCTURES)}, got {structure!r}")
        A = sp.csr_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError("square matrix required")
        self.structure = structure
        self.A = A
        self.n = A.shape[0]
        self._upper = sp.triu(A, format="csr")
        if structure == "spd":
            self.solver = solver if solver is not None else DirectSolver()
            self.solver.factor(self._upper)
            self.method = getattr(self.solver, "name", "direct")
        else:
            if solver is not None:
                raise ValueError("a custom solver is accepted for the SPD path only")
            self.solver = spla.splu(A.tocsc(), permc_spec="COLAMD")
            self.method = "splu-pivoting"
        self._norm1 = matrix_norm1(self._upper) if structure != "general" else float(abs(A).sum(axis=0).max())
        self.n_solves = 0

    @property
    def fill_nnz(self) -> int | None:
        """nnz(L) + nnz(U) of a SuperLU factorisation, None for PARDISO."""
        s: Any = self.solver._solver if self.structure == "spd" else self.solver
        return int(s.L.nnz + s.U.nnz) if hasattr(s, "L") else None

    def solve(self, B: np.ndarray) -> np.ndarray:
        """X with A X = B, B (n,) or (n, k); residual-checked."""
        B = np.asarray(B, float)
        Bm = B.reshape(self.n, -1)
        X = np.asarray(self.solver.solve(np.ascontiguousarray(Bm))).reshape(Bm.shape)
        self.n_solves += 1
        if self.structure == "general":
            if not np.isfinite(X).all():
                raise SolveResidualError("the solution is not finite")
        else:
            checked_residual(self._upper, X, Bm, self._norm1)
        return X.reshape(B.shape)


def pcg(A: sp.spmatrix, b: np.ndarray, structure: str = "spd", rtol: float = 1e-10,
        maxiter: int | None = None) -> tuple[np.ndarray, int, bool]:
    """Jacobi-preconditioned CG: (x, iterations, converged). Refused unless ``structure == "spd"``."""
    if structure != "spd":
        raise ValueError(f"PCG needs a symmetric positive definite matrix, the system is {structure!r}")
    A = sp.csr_matrix(A)
    d = A.diagonal()
    if (d <= 0).any():
        raise ValueError("PCG: a non-positive diagonal entry, the matrix is not SPD")
    M = sp.diags(1.0 / d)
    it = [0]

    def cb(_: np.ndarray) -> None:
        it[0] += 1
    x, info = spla.cg(A, b, rtol=rtol, maxiter=maxiter or 20 * A.shape[0], M=M, callback=cb)
    return x, it[0], info == 0


def jacobi_scaled(A: sp.spmatrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """(S A S, s) with S = diag(s), s = 1 / sqrt(diag A) (symmetric Jacobi scaling)."""
    d = sp.csr_matrix(A).diagonal()
    if (d <= 0).any():
        raise ValueError("Jacobi scaling needs a positive diagonal")
    s = 1.0 / np.sqrt(d)
    S = sp.diags(s)
    return sp.csr_matrix(S @ A @ S), s


def condition_1norm(A: sp.spmatrix, factor: Factorization | None = None) -> float:
    """Estimate of κ₁(A) = ‖A‖₁ ‖A⁻¹‖₁ (Higham–Tisseur block estimator, :func:`scipy.sparse.linalg.onenormest`)."""
    A = sp.csr_matrix(A)
    f = factor if factor is not None else Factorization(A, "general")
    n = A.shape[0]
    slv: Any = f.solver

    def rmat(v: np.ndarray) -> np.ndarray:
        if f.structure == "general":
            return np.asarray(slv.solve(np.asarray(v, float), trans="T"))
        return np.asarray(slv.solve(np.asarray(v, float)))
    inv = spla.LinearOperator((n, n), matvec=lambda v: np.asarray(slv.solve(np.asarray(v, float))), rmatvec=rmat)
    return float(spla.onenormest(A) * spla.onenormest(inv))
