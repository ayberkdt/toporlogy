"""Linear-solve diagnostics (WP05; ``docs/foundations/SOLVER_CONTRACT.md``, preregistration
``docs/foundations/prereg/L_solver.md``).

The production solves (:class:`toporlogy.solvers.direct.DirectSolver`, :class:`toporlogy.fem.mesh.linear.Factorization`)
check their residual. This module reports what a residual does and does not say about a solution, and adds the solves
the general-mesh and mixed paths need:

* :func:`solve_report`: residual on the original A, normwise backward error η_N = ‖r‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞),
  componentwise (Oettli–Prager) ω = maxᵢ |rᵢ| / (|A||x| + |b|)ᵢ, a condition estimate (labelled an estimate) and
  ``forward_established`` = κ η_N ≤ :data:`FORWARD_TOL`. A small residual with κ η_N above that is reported as
  "forward accuracy not established", never as accurate.
* :func:`inertia`: (n₊, n₋, n₀) of a small dense symmetric matrix from the Bunch–Kaufman LDLᵀ (1×1 and 2×2 pivots).
* :func:`krylov_solve`: CG / MINRES / GMRES / BiCGSTAB with their structural requirements enforced (a claimed
  symmetry is checked numerically), explicit residual recomputation and a status.
* :func:`check_compatible` / :func:`solve_with_gauge`: singular systems with a known null space — compatibility of
  the load and the bordered (gauge-fixed) solve. No diagonal shift.
* :func:`refine`: iterative refinement with the residual on the original matrix, stopping at convergence,
  stagnation or an iteration cap.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["FORWARD_TOL", "SYMMETRY_TOL", "SolveReport", "solve_report", "inertia", "KrylovResult", "KRYLOV",
           "krylov_solve", "check_compatible", "solve_with_gauge", "RefinementResult", "refine"]

#: κ · η_N at or below which the forward error is reported as established (first-order bound κ η_N)
FORWARD_TOL = 1e-8
#: ‖A − Aᵀ‖_F ≤ SYMMETRY_TOL ‖A‖_F for a matrix declared symmetric
SYMMETRY_TOL = 1e-12

KRYLOV = {"cg": ("spd",), "minres": ("spd", "symmetric_indefinite"),
          "gmres": ("spd", "symmetric_indefinite", "general"), "bicgstab": ("spd", "symmetric_indefinite", "general")}


def _mat(A: Any) -> Any:
    return A if sp.issparse(A) else np.asarray(A, float)


def _norm_inf(A: Any) -> float:
    return float(abs(A).sum(axis=1).max()) if sp.issparse(A) else float(np.abs(A).sum(axis=1).max())


@dataclass
class SolveReport:
    """Residual diagnostics of one solve A x = b (module notes)."""

    residual_norm: float
    backward_normwise: float
    backward_componentwise: float
    condition: float | None
    condition_kind: str
    forward_established: bool
    forward_error: float | None = None

    @property
    def forward_bound(self) -> float | None:
        """First-order forward error bound κ η_N (None without a condition number)."""
        return None if self.condition is None else self.condition * self.backward_normwise


def solve_report(A: Any, x: np.ndarray, b: np.ndarray, condition: float | None = None,
                 reference: np.ndarray | None = None) -> SolveReport:
    """Diagnostics of x for A x = b (one right-hand side). ``condition``: a known κ∞ (kind "given"); else the 1-norm
    estimate of :func:`toporlogy.fem.mesh.linear.condition_1norm` (kind "1-norm estimate"). ``reference``: a solution
    of independent accuracy for the forward error."""
    A = _mat(A)
    x = np.asarray(x, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    r = b - A @ x
    nA = _norm_inf(A)
    den = nA * float(np.abs(x).max()) + float(np.abs(b).max())
    eta = float(np.abs(r).max()) / den if den > 0 else (0.0 if not np.any(r) else np.inf)
    cden = np.asarray(abs(A) @ np.abs(x)).reshape(-1) + np.abs(b)
    ar = np.abs(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cden > 0, ar / np.where(cden > 0, cden, 1.0), np.where(ar > 0, np.inf, 0.0))
    omega = float(ratio.max()) if len(ratio) else 0.0
    if condition is not None:
        kappa, kind = float(condition), "given"
    else:
        from toporlogy.fem.mesh.linear import condition_1norm
        kappa, kind = condition_1norm(sp.csr_matrix(A)), "1-norm estimate"
    fe = None
    if reference is not None:
        ref = np.asarray(reference, float).reshape(-1)
        fe = float(np.abs(x - ref).max() / np.abs(ref).max())
    return SolveReport(float(np.linalg.norm(r)), eta, omega, kappa, kind,
                       bool(np.isfinite(kappa) and kappa * eta <= FORWARD_TOL), fe)


def inertia(A: Any, zero_tol: float | None = None) -> tuple[int, int, int]:
    """(positive, negative, zero) eigenvalue counts of a small dense symmetric A by Sylvester's law of inertia on the
    Bunch–Kaufman factorisation A = L D Lᵀ (``scipy.linalg.ldl``; D has 1×1 and 2×2 blocks, each block's eigenvalues
    counted). ``zero_tol`` (default n · eps · ‖A‖₂ estimate) separates zero from nonzero."""
    A = np.asarray(A.toarray() if sp.issparse(A) else A, float)
    n = A.shape[0]
    if A.shape != (n, n) or np.linalg.norm(A - A.T) > SYMMETRY_TOL * max(np.linalg.norm(A), 1e-300):
        raise ValueError("inertia needs a square symmetric matrix")
    _L, D, _p = sla.ldl(A, lower=True)
    tol = n * np.finfo(float).eps * max(np.abs(A).sum(axis=1).max(), 1e-300) if zero_tol is None else float(zero_tol)
    pos = neg = zer = 0
    i = 0
    while i < n:
        if i + 1 < n and D[i + 1, i] != 0.0:
            ev = np.linalg.eigvalsh(D[i:i + 2, i:i + 2])
            i += 2
        else:
            ev = np.array([D[i, i]])
            i += 1
        pos += int((ev > tol).sum())
        neg += int((ev < -tol).sum())
        zer += int((np.abs(ev) <= tol).sum())
    return pos, neg, zer


@dataclass
class KrylovResult:
    x: np.ndarray
    status: str                     # converged | max_iterations | breakdown | not_finite
    iterations: int
    residual: float                 # ‖b − A x‖₂ / ‖b‖₂, recomputed from x
    method: str
    history: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def _check_symmetric(A: Any, structure: str) -> None:
    if structure in ("spd", "symmetric_indefinite"):
        d = A - A.T
        nd = float(sp.linalg.norm(d)) if sp.issparse(d) else float(np.linalg.norm(d))
        na = float(sp.linalg.norm(A)) if sp.issparse(A) else float(np.linalg.norm(A))
        if nd > SYMMETRY_TOL * na:
            raise ValueError(f"the matrix is declared {structure!r} but ‖A − Aᵀ‖/‖A‖ = {nd / na:.2e}: not symmetric")


def krylov_solve(A: Any, b: np.ndarray, method: str, structure: str, M: Any = None, rtol: float = 1e-10,
                 maxiter: int | None = None, x0: np.ndarray | None = None) -> KrylovResult:
    """Solve A x = b by ``method`` (cg | minres | gmres | bicgstab) for a matrix of the declared ``structure``
    (spd | symmetric_indefinite | general). CG needs spd, MINRES a symmetric matrix (and an SPD preconditioner);
    a declared symmetry is checked numerically. ``rtol``: relative residual target; the reported residual is
    recomputed from the returned x."""
    if method not in KRYLOV:
        raise ValueError(f"method: {' | '.join(KRYLOV)}, got {method!r}")
    if structure not in ("spd", "symmetric_indefinite", "general"):
        raise ValueError(f"structure: spd | symmetric_indefinite | general, got {structure!r}")
    if structure not in KRYLOV[method]:
        raise ValueError(f"{method} needs a matrix of structure {' or '.join(KRYLOV[method])}, the system is "
                         f"{structure!r}")
    A = _mat(A)
    _check_symmetric(A, structure)
    b = np.asarray(b, float).reshape(-1)
    nb = float(np.linalg.norm(b))
    hist: list[float] = []
    its = [0]

    def cb(arg: Any) -> None:
        its[0] += 1
        if np.ndim(arg) == 0:                        # gmres legacy callback gives the residual norm
            hist.append(float(arg))
        else:
            hist.append(float(np.linalg.norm(b - A @ arg)) / (nb if nb > 0 else 1.0))
    mi = maxiter if maxiter is not None else 10 * len(b)
    kw: dict[str, Any] = {"M": M, "maxiter": mi, "x0": x0, "callback": cb}
    if method == "cg":
        x, info = spla.cg(A, b, rtol=rtol, **kw)
    elif method == "minres":
        x, info = spla.minres(A, b, rtol=rtol, **kw)
    elif method == "gmres":
        x, info = spla.gmres(A, b, rtol=rtol, atol=0.0, callback_type="x", **kw)
    else:
        x, info = spla.bicgstab(A, b, rtol=rtol, atol=0.0, **kw)
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return KrylovResult(x, "not_finite", its[0], np.inf, method, hist)
    res = float(np.linalg.norm(b - A @ x)) / (nb if nb > 0 else 1.0)
    if info == 0 and res <= 10.0 * rtol:
        status = "converged"
    elif info > 0 or info == 0:
        status = "max_iterations"
    else:
        status = "breakdown"
    return KrylovResult(x, status, its[0], res, method, hist)


def _orth(N: np.ndarray) -> np.ndarray:
    N = np.asarray(N, float)
    N = N[:, None] if N.ndim == 1 else N
    q, r = np.linalg.qr(N)
    if np.abs(np.diag(r)).min() <= 1e-12 * np.abs(np.diag(r)).max():
        raise ValueError("the null-space basis is rank deficient")
    return q


def check_compatible(A: Any, b: np.ndarray, N: np.ndarray, rtol: float = 1e-12) -> bool:
    """Whether b is orthogonal to the null space spanned by the columns of N (for symmetric A: b in range(A)):
    ‖Qᵀ b‖ ≤ rtol ‖b‖ with Q an orthonormal basis of N. Also checks that N is a null space of A
    (‖A Q‖ ≤ rtol ‖A‖), raising otherwise."""
    A = _mat(A)
    Q = _orth(N)
    nA = _norm_inf(A)
    if float(np.abs(A @ Q).max()) > rtol * max(nA, 1e-300):
        raise ValueError("N is not a null space of A")
    b = np.asarray(b, float).reshape(-1)
    return bool(np.linalg.norm(Q.T @ b) <= rtol * max(np.linalg.norm(b), 1e-300))


def solve_with_gauge(A: Any, b: np.ndarray, N: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """x with A x = b and the gauge Qᵀ x = 0 (Q orthonormal basis of the null space N of a symmetric A), from the
    bordered system [[A, Q], [Qᵀ, 0]] [x; μ] = [b; 0] (sparse LU). Refused when b is incompatible."""
    if not check_compatible(A, b, N, rtol):
        raise ValueError("incompatible load: b has a component in the null space of A (no solution exists)")
    A = sp.csr_matrix(_mat(A))
    Q = _orth(N)
    k = Q.shape[1]
    Kb = sp.bmat([[A, sp.csr_matrix(Q)], [sp.csr_matrix(Q.T), None]], format="csc")
    rhs = np.concatenate([np.asarray(b, float).reshape(-1), np.zeros(k)])
    sol = spla.splu(Kb, permc_spec="COLAMD").solve(rhs)
    return np.asarray(sol[:A.shape[0]])


@dataclass
class RefinementResult:
    x: np.ndarray
    omega: list[float]              # componentwise backward error after each iterate (index 0: the start)
    stop: str                       # converged | stagnated | max_iter
    iterations: int


def refine(A: Any, b: np.ndarray, solve: Callable[[np.ndarray], np.ndarray], x0: np.ndarray,
           max_iter: int = 10) -> RefinementResult:
    """Iterative refinement x ← x + solve(b − A x) with the residual in float64 on the original A. Stops when the
    componentwise backward error ω ≤ eps (converged), when ω is not at least halved (stagnated; the better iterate is
    kept) or after ``max_iter`` corrections."""
    A = _mat(A)
    b = np.asarray(b, float).reshape(-1)
    x = np.asarray(x0, float).reshape(-1).copy()

    def om(v: np.ndarray) -> float:
        r = np.abs(b - A @ v)
        den = np.asarray(abs(A) @ np.abs(v)).reshape(-1) + np.abs(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(den > 0, r / np.where(den > 0, den, 1.0), np.where(r > 0, np.inf, 0.0))
        return float(q.max())
    hist = [om(x)]
    eps = np.finfo(float).eps
    for k in range(int(max_iter)):
        if hist[-1] <= eps:
            return RefinementResult(x, hist, "converged", k)
        xn = x + np.asarray(solve(b - A @ x), float).reshape(-1)
        w = om(xn)
        if not w <= 0.5 * hist[-1]:
            if w < hist[-1]:
                x = xn
                hist.append(w)
            return RefinementResult(x, hist, "stagnated", k + 1)
        x = xn
        hist.append(w)
    return RefinementResult(x, hist, "converged" if hist[-1] <= eps else "max_iter", int(max_iter))
