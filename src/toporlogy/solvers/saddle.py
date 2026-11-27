"""Saddle-point block operators, Schur complement and block preconditioners (WP05 L06;
``docs/foundations/SADDLE_POINT.md``, preregistration ``docs/foundations/prereg/L6_saddle.md``).

The system is ::

    K [u; p] = [f; g],      K = [[A, Bᵀ], [B, −C]],      A symmetric positive definite (n × n),
                                                          B (m × n),  C symmetric positive semi-definite (m × m)

(the sign convention of :mod:`toporlogy.fem.mesh.mixed`, where ``C = M / K_bulk`` and C = 0 is the exact constraint).
Eliminating u = A⁻¹(f − Bᵀp) gives the Schur-complement system ::

    S p = B A⁻¹ f − g,        S = C + B A⁻¹ Bᵀ        (symmetric positive semi-definite).

* :class:`SaddleOperator`: K applied block-wise, S applied **without forming it** (one solve with a factorised A per
  application), the reduced right-hand side and the recovery of u.
* :func:`block_diagonal_preconditioner` diag(Â⁻¹, Ŝ⁻¹) (SPD, for MINRES) and
  :func:`block_triangular_preconditioner` [[Â, Bᵀ], [0, −Ŝ]]⁻¹ (for GMRES), with Â a factorisation of A (exact LU or
  incomplete LU) and Ŝ the scaled pressure mass matrix.
* :func:`saddle_solve`: MINRES / GMRES through :func:`toporlogy.solvers.diagnostics.krylov_solve` (explicit residual
  recomputation and status), restarted on the explicit residual, separate u / p residuals, preconditioner solve counts,
  and a known pressure null space (K = ∞ with a fully constrained boundary) handled by compatibility check and
  projection — no diagonal shift.

This module does not import ``toporlogy.fem`` (layer rule); the blocks are passed as sparse matrices.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.solvers.diagnostics import krylov_solve

__all__ = ["factorised_solver", "SaddleOperator", "BlockPreconditioner", "block_diagonal_preconditioner",
           "block_triangular_preconditioner", "SaddleResult", "saddle_solve"]

#: relative tolerance for the null-vector and compatibility checks of a gauge-fixed pressure space
GAUGE_TOL = 1e-10


def factorised_solver(M: Any, kind: str = "lu", drop_tol: float = 1e-4, fill_factor: float = 10.0) -> Callable:
    """Solve function v ↦ M⁻¹ v of a sparse square matrix: ``kind="lu"`` an exact sparse LU (SuperLU, COLAMD),
    ``kind="ilu"`` an incomplete LU (``scipy.sparse.linalg.spilu`` with ``drop_tol``, ``fill_factor``): AMG-free, and
    not mesh-independent as a preconditioner of an elasticity block (no such claim is made)."""
    if kind not in ("lu", "ilu"):
        raise ValueError(f"kind: lu | ilu, got {kind!r}")
    Mc = sp.csc_matrix(M, dtype=float)
    if Mc.shape[0] != Mc.shape[1]:
        raise ValueError("a square matrix is required")
    if not np.all(np.isfinite(Mc.data)):
        raise ValueError("the matrix has non-finite entries")
    fac = spla.splu(Mc, permc_spec="COLAMD") if kind == "lu" else spla.spilu(Mc, drop_tol=drop_tol,
                                                                           fill_factor=fill_factor)

    def solve(v: np.ndarray) -> np.ndarray:
        return np.asarray(fac.solve(np.ascontiguousarray(v, dtype=float)))
    solve.kind = kind  # type: ignore[attr-defined]
    return solve


class SaddleOperator:
    """The block matrix K = [[A, Bᵀ], [B, −C]] (module notes) given as sparse blocks.

    Parameters
    ----------
    A : (n, n) symmetric positive definite.  B : (m, n).  C : (m, m) symmetric positive semi-definite, or None (C = 0).
    A_solve : callable v ↦ A⁻¹ v. Default: an exact sparse LU of A built here (factorised once, reused by every
        application of S).
    """

    def __init__(self, A: Any, B: Any, C: Any = None, A_solve: Callable | None = None) -> None:
        self.A = sp.csr_matrix(A, dtype=float)
        self.B = sp.csr_matrix(B, dtype=float)
        n, m = self.A.shape[0], self.B.shape[0]
        if self.A.shape != (n, n) or self.B.shape != (m, n):
            raise ValueError(f"block shapes: A {self.A.shape}, B {self.B.shape} (need A (n,n), B (m,n))")
        self.C = None if C is None else sp.csr_matrix(C, dtype=float)
        if self.C is not None and self.C.shape != (m, m):
            raise ValueError(f"C has shape {self.C.shape}, need ({m}, {m})")
        self.n, self.m = n, m
        self.A_solve = factorised_solver(self.A) if A_solve is None else A_solve
        self.n_A_solves = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n + self.m, self.n + self.m)

    def matrix(self) -> sp.csr_matrix:
        """K as one sparse matrix (an assembled sparse matrix is cheap; the Schur complement is never assembled)."""
        Cn = None if self.C is None else -self.C
        return sp.bmat([[self.A, self.B.T], [self.B, Cn]], format="csr")

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """K v for v = [u; p]."""
        v = np.asarray(v, float).reshape(-1)
        u, p = v[:self.n], v[self.n:]
        top = self.A @ u + self.B.T @ p
        bot = self.B @ u - (0.0 if self.C is None else self.C @ p)
        return np.concatenate([top, bot])

    def _A_inv(self, v: np.ndarray) -> np.ndarray:
        self.n_A_solves += 1
        return np.asarray(self.A_solve(v))

    def schur_apply(self, q: np.ndarray) -> np.ndarray:
        """S q = C q + B A⁻¹ Bᵀ q without forming S (one A-solve)."""
        q = np.asarray(q, float).reshape(-1)
        out = self.B @ self._A_inv(self.B.T @ q)
        return out if self.C is None else out + self.C @ q

    def schur_operator(self) -> spla.LinearOperator:
        """S as a symmetric :class:`scipy.sparse.linalg.LinearOperator` (matrix-free)."""
        return spla.LinearOperator((self.m, self.m), matvec=self.schur_apply, dtype=float)

    def reduced_rhs(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """B A⁻¹ f − g, the right-hand side of S p = B A⁻¹ f − g."""
        return self.B @ self._A_inv(np.asarray(f, float).reshape(-1)) - np.asarray(g, float).reshape(-1)

    def recover_u(self, f: np.ndarray, p: np.ndarray) -> np.ndarray:
        """u = A⁻¹(f − Bᵀp) from a pressure p."""
        return self._A_inv(np.asarray(f, float).reshape(-1) - self.B.T @ np.asarray(p, float).reshape(-1))


class BlockPreconditioner(spla.LinearOperator):
    """Block preconditioner as an operator (v ↦ P⁻¹ v) that counts its block solves.

    ``kind``: ``"diagonal"``: P = diag(Â, Ŝ), SPD (MINRES);  ``"triangular"``: P = [[Â, Bᵀ], [0, −Ŝ]] (GMRES).
    """

    def __init__(self, kind: str, A_hat_solve: Callable, S_hat_solve: Callable, n: int, m: int,
                 B: sp.csr_matrix | None = None) -> None:
        super().__init__(dtype=float, shape=(n + m, n + m))
        self.kind, self.n, self.m = kind, n, m
        self._a, self._s, self._B = A_hat_solve, S_hat_solve, B
        self.n_A_solves = 0
        self.n_S_solves = 0

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float).reshape(-1)
        r1, r2 = v[:self.n], v[self.n:]
        self.n_S_solves += 1
        self.n_A_solves += 1
        if self.kind == "diagonal":
            return np.concatenate([self._a(r1), self._s(r2)])
        x2 = -np.asarray(self._s(r2))                       # −Ŝ x₂ = r₂
        x1 = np.asarray(self._a(r1 - self._B.T @ x2))       # Â x₁ = r₁ − Bᵀ x₂
        return np.concatenate([x1, x2])


def _solver_of(x: Any) -> Callable:
    """A callable v ↦ x⁻¹ v from a callable or a sparse SPD matrix (factorised by exact LU)."""
    return x if callable(x) else factorised_solver(x)


def block_diagonal_preconditioner(A_hat: Any, S_hat: Any, n: int, m: int) -> BlockPreconditioner:
    """diag(Â⁻¹, Ŝ⁻¹), SPD when Â and Ŝ are. ``A_hat``, ``S_hat``: a solve callable or a sparse SPD matrix."""
    return BlockPreconditioner("diagonal", _solver_of(A_hat), _solver_of(S_hat), n, m)


def block_triangular_preconditioner(A_hat: Any, S_hat: Any, B: Any) -> BlockPreconditioner:
    """[[Â, Bᵀ], [0, −Ŝ]]⁻¹ for K = [[A, Bᵀ], [B, −C]]. With Â = A and Ŝ = S exactly, K P⁻¹ = [[I, 0], [B A⁻¹, I]]
    and (K P⁻¹ − I)² = 0: GMRES needs at most two iterations."""
    B = sp.csr_matrix(B, dtype=float)
    return BlockPreconditioner("triangular", _solver_of(A_hat), _solver_of(S_hat), B.shape[1], B.shape[0], B)


@dataclass
class SaddleResult:
    u: np.ndarray
    p: np.ndarray
    status: str                     # converged | max_iterations | breakdown | not_finite
    iterations: int                 # Krylov iterations summed over the restarts
    restarts: int                   # extra Krylov runs started from the previous iterate (explicit residual too large)
    residual: float                 # ‖b − K x‖₂ / ‖b‖₂ recomputed from (u, p) and the *original* (f, g)
    residual_u: float               # ‖f − A u − Bᵀp‖₂ / ‖b‖₂
    residual_p: float               # ‖g − B u + C p‖₂ / ‖b‖₂
    method: str
    preconditioner: str             # none | diagonal | triangular
    n_A_solves: int                 # applications of Â⁻¹ inside the preconditioner
    n_S_solves: int                 # applications of Ŝ⁻¹
    gauge: bool                     # a pressure null space was projected out
    history: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def saddle_solve(op: SaddleOperator, f: np.ndarray, g: np.ndarray, preconditioner: str = "diagonal",
                 S_hat: Any = None, A_hat: Any = None, rtol: float = 1e-8, maxiter: int | None = None,
                 max_restarts: int = 3, null_vector: np.ndarray | None = None,
                 gauge_weights: np.ndarray | None = None) -> SaddleResult:
    """Solve K [u; p] = [f; g] by preconditioned MINRES (``"diagonal"``, ``"none"``) or GMRES (``"triangular"``).

    ``S_hat``: sparse SPD matrix or solve callable for Ŝ (required unless ``preconditioner="none"``); for a stable
    mixed pair the scaled pressure mass matrix c M with c ≈ 1/G + 1/K_bulk (``docs/foundations/SADDLE_POINT.md``).
    ``A_hat``: solve callable or sparse matrix for Â (default: the operator's own A-solve, i.e. exact).
    ``rtol``: target of the explicit ‖b − K x‖₂/‖b‖₂ (converged when ≤ 10 rtol, the convention of
    :func:`toporlogy.solvers.diagnostics.krylov_solve`); a Krylov run that stops on its recursive (preconditioned)
    estimate with a larger explicit residual is restarted from its iterate (``max_restarts``, each with rtol/10).

    Pressure null space: ``null_vector`` z with Bᵀz = 0 and C z = 0 (checked; K has the null vector (0, z), e.g. z = 1 for
    C = 0 and every boundary DOF constrained). The load must be compatible, zᵀg = 0 (checked, ``ValueError`` otherwise);
    the iteration runs on g − z (zᵀg)/(zᵀz) and the returned p is projected to wᵀp = 0 with ``gauge_weights`` w
    (default z; the mixed classes use w = M 1, the discrete ∫p = 0). The reported residual is that of the original
    (f, g). No diagonal shift."""
    if preconditioner not in ("none", "diagonal", "triangular"):
        raise ValueError(f"preconditioner: none | diagonal | triangular, got {preconditioner!r}")
    n, m = op.n, op.m
    f = np.asarray(f, float).reshape(-1)
    g = np.asarray(g, float).reshape(-1)
    if f.shape != (n,) or g.shape != (m,):
        raise ValueError(f"f must have {n} and g {m} entries, got {f.shape} and {g.shape}")
    if preconditioner != "none" and S_hat is None:
        raise ValueError("S_hat (the scaled pressure mass matrix) is required for a preconditioned solve")
    z = w = None
    g_eff = g
    if null_vector is not None:
        z = np.asarray(null_vector, float).reshape(-1)
        w = z if gauge_weights is None else np.asarray(gauge_weights, float).reshape(-1)
        if z.shape != (m,) or w.shape != (m,):
            raise ValueError("null_vector and gauge_weights must have one entry per pressure DOF")
        nz = float(np.linalg.norm(z))
        scaleB = float(sp.linalg.norm(op.B)) if op.B.nnz else 1.0
        if float(np.linalg.norm(op.B.T @ z)) > GAUGE_TOL * scaleB * nz:
            raise ValueError("null_vector is not a null vector of the pressure coupling (Bᵀ z ≠ 0)")
        if op.C is not None and float(np.linalg.norm(op.C @ z)) > GAUGE_TOL * max(float(sp.linalg.norm(op.C)), 1e-300) * nz:
            raise ValueError("null_vector is not a null vector of C (C z ≠ 0)")
        if abs(float(w @ z)) <= 1e-12 * float(np.linalg.norm(w)) * nz:
            raise ValueError("the gauge functional wᵀp vanishes on the null vector (wᵀz = 0)")
        if abs(float(z @ g)) > GAUGE_TOL * nz * max(float(np.linalg.norm(g)), 1e-300):
            raise ValueError("incompatible load: zᵀg ≠ 0 (no solution exists for this pressure null space)")
        g_eff = g - z * (float(z @ g) / float(z @ z))
    K = op.matrix()
    b = np.concatenate([f, g])
    b_eff = np.concatenate([f, g_eff])
    nb = float(np.linalg.norm(b)) or 1.0
    P: BlockPreconditioner | None = None
    if preconditioner == "diagonal":
        P = block_diagonal_preconditioner(op.A_solve if A_hat is None else A_hat, S_hat, n, m)
    elif preconditioner == "triangular":
        P = block_triangular_preconditioner(op.A_solve if A_hat is None else A_hat, S_hat, op.B)
    method = "gmres" if preconditioner == "triangular" else "minres"
    x = None
    its = 0
    hist: list[float] = []
    restarts = 0
    status = "max_iterations"
    tol = rtol
    for k in range(max_restarts + 1):
        r = krylov_solve(K, b_eff, method, "symmetric_indefinite", M=P, rtol=tol, maxiter=maxiter, x0=x)
        x, its, status = r.x, its + r.iterations, r.status
        hist.extend(r.history)
        if status in ("converged", "not_finite", "breakdown"):
            break
        if k < max_restarts:
            restarts += 1
            tol = tol / 10.0
    x = np.asarray(x, float)
    u, p = x[:n].copy(), x[n:].copy()
    if z is not None and np.all(np.isfinite(p)):
        p = p - z * (float(w @ p) / float(w @ z))
    xa = np.concatenate([u, p])
    res = b - K @ xa
    if status == "converged" and float(np.linalg.norm(res)) / nb > 10.0 * rtol:
        status = "max_iterations"      # the projection / compatibility step changed the residual beyond the target
    return SaddleResult(u, p, status, its, restarts, float(np.linalg.norm(res)) / nb,
                        float(np.linalg.norm(res[:n])) / nb, float(np.linalg.norm(res[n:])) / nb, method,
                        preconditioner, 0 if P is None else P.n_A_solves, 0 if P is None else P.n_S_solves,
                        z is not None, hist)
