"""Path following for R(x, λ) = f_int(x) − λ q = 0: load control, displacement control and arc length (04 §1).

A problem supplies ``linearize(x, lam) -> Linearization`` (residual on the unknowns, tangent K = ∂R/∂x, ‖K‖₁), the
load vector ``load`` = q = −∂R/∂λ on the unknowns, ``solve(lin, rhs)`` (rhs (n,) or (n, k)) and optionally
``det_sign(lin)`` (sign of det K, for the stability of a point) and ``commit(x, lam)`` (called once per accepted
point: a path-dependent material commits its trial state there; nothing is committed for a rejected step).

* :func:`load_control`: λ from ``start`` to ``end`` in increments of at most ``max_step``; a failed increment is
  halved (failure below ``min_step`` ends the path with ``ok = False``), two successes in a row grow it by 1.5. It
  takes the Newton step as a callback (the physics keeps its own Newton loop and records).
* :func:`displacement_control`: the load factor is an unknown and one unknown x_c is prescribed per step
  (x_c = w_k); the corrector solves the bordered system K δx = −R + δλ q with e_cᵀ(x + δx) = w_k. The reported
  load is λ q: the work-conjugate "reaction" of the controlled DOF is not a separate unknown (this is not a
  prescribed displacement with a reaction, which would remove x_c from the unknowns).
* :func:`arc_length`: Crisfield's cylindrical arc length (ψ = 0; ψ > 0 adds ψ² Δλ² qᵀq, the spherical family).
  Constraint ΔxᵀΔx + ψ²Δλ²qᵀq = Δs². Predictor Δλ = ± Δs/√(δx_qᵀδx_q + ψ²qᵀq), δx_q = K⁻¹q, sign of
  Δx_prevᵀ δx_q + ψ²Δλ_prev qᵀq (the first step follows ``direction``). Corrector: δx = δx_R + δλ δx_q with the
  quadratic a δλ² + b δλ + c = 0; the root whose new increment makes the smallest angle with the current one is
  taken; a negative discriminant rejects the step (code ``complex_roots``); near-equal angles are logged as
  ``path ambiguity``. Rejected steps halve Δs (below ``ds_min``: :class:`ContinuationError` with code
  ``step_limit``); accepted steps scale Δs by √(``desired_iterations``/iterations), within [ds_min, ds_max].
  The method follows ONE path through limit points; at a bifurcation point of a symmetric problem it continues on
  the primary path; branch switching is a separate step (below). ``start_tangent`` = (Δx, Δλ) orients the first
  predictor as a previous increment would (used after a branch switch).

Stability, critical points and branch switching (``docs/olcumler/catallanma_onkayit.md``):

* :class:`TangentSpectrum` gives the eigenvalues of the symmetric tangent at a point. ``dense``: every eigenvalue of
  the full matrix (numpy ``eigh``); the index (number of negative eigenvalues) is Sylvester's inertia, exact.
  ``sparse``: the index from a symmetric sparse LDLᵀ (SuperLU with diagonal pivots in symmetric mode, P K Pᵀ = L U,
  U = D Lᵀ, so the index is the number of negative diagonal entries of U: Sylvester's law of inertia; refused with
  :class:`LinearSolveFailure` when SuperLU took an off-diagonal pivot), and the ``k`` eigenpairs nearest zero
  (shift-invert Lanczos, σ = 0). The ``k`` eigenvalues nearest zero are consecutive in the sorted spectrum, so their
  global position is known (``Spectrum.offset`` = index − negative ones in the window). A tangent stored
  as its upper triangle (the FE convention of :class:`IndefiniteDirectSolver`) is completed to the full matrix.
  Given to :func:`arc_length` / :func:`displacement_control`, it is evaluated at every accepted point
  (``ContinuationPoint.index``, ``eigenvalues``) and changes nothing in the path (it only observes).
* :func:`find_critical`: intervals (i, i + 1) of a path whose index differs (without a spectrum: whose ``det_sign``
  differs, which sees odd numbers of crossings only).
* :func:`locate_critical`: on the chord d = (x_B − x_A, λ_B − λ_A) of the interval, equilibria with the hyperplane
  constraint d_xᵀ(x − x_A) = t d_xᵀd_x (λ unknown; the bordered Newton of displacement control with the constraint
  vector d_x), and the Illinois (modified regula falsi) root of the crossing eigenvalue μ(t): the j-th of the sorted
  spectrum, j = min(m_A, m_B) (continuous in t, opposite signs at the ends; with ``sparse`` taken from the window
  through its offset). Stop when |μ| ≤ max(``eig_tol`` · max(|μ(0)|, |μ(1)|), 16 ε ‖K‖₁) (the second term: the
  absolute accuracy of a backward-stable eigensolver) or the bracket is shorter than ``t_tol``. Classification by
  the load-direction test: with the unit critical vector φ, |φᵀq|/‖q‖ ≤ ``bifurcation_tol`` is a bifurcation point,
  otherwise a limit point (φ has a load component: the path turns in λ). φ's sign is fixed so that its largest
  component is positive.
* :func:`switch_branch` (bifurcation points only): the equilibrium on the hyperplane φᵀ(x − x_c) = ξ (λ unknown,
  predictor (x_c + ξφ, λ_c)), then :func:`arc_length` from there with ``start_tangent`` = (x_s − x_c, λ_s − λ_c), i.e.
  away from x_c. On the primary path of a symmetric problem φᵀ(x − x_c) ≡ 0, so the hyperplane does not meet it.
  Location and switching probe points between accepted ones: they are refused for path-dependent problems (a
  ``store`` of committed material state), whose state at a probe would be the wrong one.

:class:`IndefiniteDirectSolver` factors the full symmetric matrix with SuperLU and reports the sign of the
determinant, for tangents past a limit point (indefinite) where a Cholesky / PARDISO SPD factorisation does not
apply.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from toporlogy.core.constitutive import ConstitutiveFailure
from toporlogy.solvers.newton import LinearSolveFailure, Linearization, StepRecord

__all__ = ["ContinuationError", "ContinuationProblem", "LoadControlResult", "load_control", "ContinuationPoint",
           "ContinuationPath", "ArcLengthSettings", "arc_length", "displacement_control", "IndefiniteDirectSolver",
           "full_tangent", "sparse_inertia", "Spectrum", "TangentSpectrum", "find_critical", "CriticalSettings",
           "CriticalPoint", "locate_critical", "BranchSwitch", "switch_branch"]


class ContinuationError(RuntimeError):
    """Path following stopped: ``code`` (step_limit, ...), ``cause`` (code of the last failed step), ``path``."""

    def __init__(self, message: str, code: str, cause: str, path: Any = None) -> None:
        super().__init__(message)
        self.code, self.cause, self.path = code, cause, path


class ContinuationProblem(Protocol):
    load: np.ndarray

    def linearize(self, x: np.ndarray, lam: float) -> Linearization: ...

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray: ...


# ----------------------------------------------------------------------------------------------- load control
@dataclass
class LoadControlResult:
    """``ok``: ``end`` reached; ``x`` and ``lam``: last accepted point; ``dlam``: the increment after the last
    failure; ``records``: every attempted increment; ``points``: accepted (λ, x); ``cuts``: halvings; ``last``: the
    (relative residual, backward error) of the last accepted increment."""

    ok: bool
    x: Any
    lam: float
    dlam: float
    records: list[StepRecord]
    points: list[tuple[float, Any]]
    cuts: int
    last: tuple[float, float]


def load_control(attempt: Callable[[float, Any], tuple[bool, Any, StepRecord, float, float]], x0: Any,
                 start: float = 0.0, end: float = 1.0, max_step: float = 0.1, min_step: float = 1e-4,
                 commit: Callable[[float, Any], None] | None = None, keep_points: bool = False) -> LoadControlResult:
    """Step λ from ``start`` to ``end`` (module notes). ``attempt(target, x_start) -> (ok, x, record, rel, eta)``
    runs Newton at load factor ``target`` from ``x_start``; on failure it returns the start point."""
    lam, dlam, x, streak = start, max_step, x0, 0
    sgn = 1.0 if end >= start else -1.0
    res = LoadControlResult(False, x0, start, dlam, [], [], 0, (0.0, 0.0))
    while (lam < end) if sgn > 0 else (lam > end):
        target = min(end, lam + dlam) if sgn > 0 else max(end, lam - dlam)
        if abs(end - target) < 1e-12:
            target = end
        ok, xnew, rec, rel, eta = attempt(target, x)
        res.records.append(rec)
        if ok:
            lam, x, streak = target, xnew, streak + 1
            if commit is not None:
                commit(lam, x)
            if keep_points:
                res.points.append((lam, x))
            if streak >= 2:
                dlam = min(1.5 * dlam, max_step)
            res.last = (rel, eta)
            continue
        streak = 0
        res.cuts += 1
        dlam *= 0.5
        if dlam < min_step:
            res.x, res.lam, res.dlam = x, lam, dlam
            return res
    res.ok, res.x, res.lam, res.dlam = True, x, lam, dlam
    return res


# ----------------------------------------------------------------------------------------------- path types
@dataclass
class ContinuationPoint:
    lam: float
    x: np.ndarray
    det_sign: int | None
    iterations: int
    ds: float
    residual: float
    index: int | None = None                  # negative tangent eigenvalues (with a spectrum)
    eigenvalues: np.ndarray | None = None     # tangent eigenvalues nearest zero, ascending (with a spectrum)


@dataclass
class ContinuationPath:
    points: list[ContinuationPoint] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    rejected: int = 0

    @property
    def lam(self) -> np.ndarray:
        return np.array([p.lam for p in self.points])

    def component(self, i: int) -> np.ndarray:
        return np.array([p.x[i] for p in self.points])


def _det_sign(problem: Any, lin: Linearization) -> int | None:
    f = getattr(problem, "det_sign", None)
    return None if f is None else f(lin)


def _commit(problem: Any, x: np.ndarray, lam: float) -> None:
    f = getattr(problem, "commit", None)
    if f is not None:
        f(x, lam)


def _linearize(problem: ContinuationProblem, x: np.ndarray, lam: float, rec: StepRecord) -> Linearization | None:
    try:
        return problem.linearize(x, lam)
    except ConstitutiveFailure as e:
        rec.code, rec.reason = e.code, f"inadmissible state ({e})"
        return None


def _solve2(problem: ContinuationProblem, lin: Linearization, R: np.ndarray, rec: StepRecord
            ) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        X = problem.solve(lin, np.column_stack([-R, problem.load]))
    except LinearSolveFailure as e:
        rec.code, rec.reason = "singular_tangent", f"tangent solve failed ({e})"
        return None
    X = np.asarray(X).reshape(len(R), 2)
    return X[:, 0], X[:, 1]


@dataclass(frozen=True)
class ArcLengthSettings:
    ds: float
    ds_min: float | None = None              # default ds / 1024
    ds_max: float | None = None              # default ds
    psi: float = 0.0
    max_steps: int = 500
    max_iter: int = 25
    rtol: float = 1e-10                      # ‖R‖ ≤ rtol ‖q‖
    eta_tol: float = 1e-14
    desired_iterations: int = 4
    direction: float = 1.0                   # sign of the first load increment


def _converged(R: np.ndarray, lin: Linearization, x: np.ndarray, nq: float, rtol: float, eta_tol: float
               ) -> tuple[bool, float, float]:
    nr = float(np.linalg.norm(R))
    rel = nr / nq
    eta = nr / (lin.norm1 * float(np.linalg.norm(x)) + nq)
    return (rel <= rtol or eta <= eta_tol), rel, eta


def arc_length(problem: ContinuationProblem, x0: np.ndarray, lam0: float, settings: ArcLengthSettings,
               stop: Callable[[np.ndarray, float], bool], spectrum: TangentSpectrum | None = None,
               start_tangent: tuple[np.ndarray, float] | None = None,
               stop_at_critical: bool = False) -> ContinuationPath:
    """Follow the equilibrium path from the equilibrium (``x0``, ``lam0``) until ``stop(x, lam)`` is true or
    ``max_steps`` points are accepted (module notes). ``spectrum``: evaluated at every accepted point;
    ``start_tangent``: (Δx, Δλ) orienting the first predictor; ``stop_at_critical``: return at the first point whose
    index (or, without a spectrum, det sign) differs from the previous one."""
    st = settings
    ds = st.ds
    ds_min = st.ds / 1024 if st.ds_min is None else st.ds_min
    ds_max = st.ds if st.ds_max is None else st.ds_max
    q = np.asarray(problem.load, float)
    qq = float(q @ q)
    nq = math.sqrt(qq)
    psi2 = st.psi ** 2
    path = ContinuationPath()
    x, lam = np.asarray(x0, float).copy(), float(lam0)
    lin0 = problem.linearize(x, lam)
    path.points.append(ContinuationPoint(lam, x.copy(), _det_sign(problem, lin0), 0, 0.0,
                                         float(np.linalg.norm(lin0.residual)) / nq))
    _observe(path.points[-1], spectrum, lin0)
    dx_prev: np.ndarray | None = None
    dlam_prev = 0.0
    if start_tangent is not None:
        dx_prev, dlam_prev = np.asarray(start_tangent[0], float), float(start_tangent[1])
    while len(path.points) <= st.max_steps:
        rec = StepRecord(lam, False, [])
        rec.notes.append(f"ds = {ds:.6g}")
        ok = False
        try:
            dq = np.asarray(problem.solve(lin0, q)).reshape(q.shape)
        except LinearSolveFailure as e:
            rec.code, rec.reason = "singular_tangent", f"predictor solve failed ({e})"
            dq = None
        if dq is not None:
            if dx_prev is None:
                sgn = 1.0 if st.direction >= 0 else -1.0
            else:
                sgn = 1.0 if float(dx_prev @ dq) + psi2 * dlam_prev * qq >= 0 else -1.0
            dlam = sgn * ds / math.sqrt(float(dq @ dq) + psi2 * qq)
            dx = dlam * dq
            for it in range(st.max_iter + 1):
                lin = _linearize(problem, x + dx, lam + dlam, rec)
                if lin is None:
                    break
                R = lin.residual
                conv, rel, eta = _converged(R, lin, x + dx, nq, st.rtol, st.eta_tol)
                rec.residuals.append(rel)
                rec.backward_errors.append(eta)
                if not math.isfinite(rel):
                    rec.code, rec.reason = "non_finite", "non-finite residual"
                    break
                if conv:
                    ok = True
                    break
                if it == st.max_iter:
                    rec.code, rec.reason = "max_iterations", f"no convergence in {st.max_iter} iterations"
                    break
                if len(rec.residuals) > 1 and rel > 1e6 * rec.residuals[0]:
                    rec.code, rec.reason = "divergence", "residual diverging"
                    break
                sol = _solve2(problem, lin, R, rec)
                if sol is None:
                    break
                dr, dqk = sol
                a = float(dqk @ dqk) + psi2 * qq
                v = dx + dr
                b = 2.0 * (float(dqk @ v) + psi2 * dlam * qq)
                c = float(v @ v) + psi2 * dlam * dlam * qq - ds * ds
                disc = b * b - 4.0 * a * c
                if disc < 0:
                    rec.code, rec.reason = "complex_roots", f"arc-length constraint has no real root (disc {disc:.3e})"
                    break
                sq = math.sqrt(disc)
                roots = ((-b + sq) / (2 * a), (-b - sq) / (2 * a))
                cands = [(v + r * dqk, dlam + r) for r in roots]
                nd = math.sqrt(float(dx @ dx) + psi2 * dlam * dlam * qq)
                cos = [(float(cx @ dx) + psi2 * cl * dlam * qq) / (nd * math.sqrt(float(cx @ cx) + psi2 * cl * cl * qq))
                       for cx, cl in cands]
                k = 0 if cos[0] >= cos[1] else 1
                if abs(cos[0] - cos[1]) <= 1e-12:
                    rec.notes.append(f"iteration {it}: path ambiguity (both roots at angle cosine {cos[0]:.6f})")
                dx, dlam = cands[k]
                rec.updates.append(float(np.linalg.norm(dr + roots[k] * dqk)))
        if ok and lin is not None:
            x, lam = x + dx, lam + dlam
            rec.load_factor, rec.converged, rec.code = lam, True, "converged"
            _commit(problem, x, lam)
            lin0 = lin
            path.points.append(ContinuationPoint(lam, x.copy(), _det_sign(problem, lin), len(rec.residuals) - 1, ds,
                                                 rec.residuals[-1]))
            _observe(path.points[-1], spectrum, lin)
            path.records.append(rec)
            dx_prev, dlam_prev = dx, dlam
            n_it = max(len(rec.residuals) - 1, 1)
            ds = min(max(ds * math.sqrt(st.desired_iterations / n_it), ds_min), ds_max)
            if stop(x, lam):
                return path
            if stop_at_critical and _crossed(path.points[-2], path.points[-1]):
                return path
            continue
        rec.rolled_back = True
        path.records.append(rec)
        path.rejected += 1
        ds *= 0.5
        if ds < ds_min:
            raise ContinuationError(f"arc length: step rejected at ds {2 * ds:.3g} and ds_min {ds_min:.3g} reached "
                                    f"({rec.code}: {rec.reason}); load factor {lam:.6g}", "step_limit", rec.code, path)
    return path


def displacement_control(problem: ContinuationProblem, x0: np.ndarray, lam0: float, dof: int,
                         targets: Sequence[float], max_iter: int = 25, rtol: float = 1e-10, eta_tol: float = 1e-14,
                         max_bisections: int = 10, spectrum: TangentSpectrum | None = None) -> ContinuationPath:
    """Equilibria with the unknown ``dof`` prescribed to each of ``targets`` in turn and λ unknown (module notes).
    A failed step is bisected (at most ``max_bisections`` levels). ``spectrum``: evaluated at every accepted
    point."""
    q = np.asarray(problem.load, float)
    nq = float(np.linalg.norm(q))
    path = ContinuationPath()
    x, lam = np.asarray(x0, float).copy(), float(lam0)
    lin0 = problem.linearize(x, lam)
    path.points.append(ContinuationPoint(lam, x.copy(), _det_sign(problem, lin0), 0, 0.0,
                                         float(np.linalg.norm(lin0.residual)) / nq))
    _observe(path.points[-1], spectrum, lin0)
    pending = [float(t) for t in targets][::-1]
    depth: dict[float, int] = {}
    while pending:
        w = pending[-1]
        rec = StepRecord(lam, False, [])
        rec.notes.append(f"target x[{dof}] = {w:.9g}")
        ok = False
        try:
            dq = np.asarray(problem.solve(lin0, q)).reshape(q.shape)
            dlam = (w - x[dof]) / dq[dof]
            dx = dlam * dq
        except LinearSolveFailure as e:
            rec.code, rec.reason = "singular_tangent", f"predictor solve failed ({e})"
            dq = None
        if dq is not None:
            for it in range(max_iter + 1):
                lin = _linearize(problem, x + dx, lam + dlam, rec)
                if lin is None:
                    break
                R = lin.residual
                conv, rel, eta = _converged(R, lin, x + dx, nq, rtol, eta_tol)
                rec.residuals.append(rel)
                rec.backward_errors.append(eta)
                if conv and abs(x[dof] + dx[dof] - w) <= 1e-14 * max(1.0, abs(w)):
                    ok = True
                    break
                if not math.isfinite(rel) or it == max_iter:
                    rec.code = "non_finite" if not math.isfinite(rel) else "max_iterations"
                    rec.reason = rec.code
                    break
                sol = _solve2(problem, lin, R, rec)
                if sol is None:
                    break
                dr, dqk = sol
                dl = -(x[dof] + dx[dof] + dr[dof] - w) / dqk[dof]
                dx = dx + dr + dl * dqk
                dlam += dl
        if ok and lin is not None:
            x, lam = x + dx, lam + dlam
            rec.load_factor, rec.converged, rec.code = lam, True, "converged"
            _commit(problem, x, lam)
            lin0 = lin
            path.points.append(ContinuationPoint(lam, x.copy(), _det_sign(problem, lin), len(rec.residuals) - 1, 0.0,
                                                 rec.residuals[-1]))
            _observe(path.points[-1], spectrum, lin)
            path.records.append(rec)
            pending.pop()
            continue
        rec.rolled_back = True
        path.records.append(rec)
        path.rejected += 1
        d = depth.get(w, 0)
        if d >= max_bisections:
            raise ContinuationError(f"displacement control: target {w:.6g} not reached after {d} bisections "
                                    f"({rec.code})", "step_limit", rec.code, path)
        mid = 0.5 * (w + x[dof])
        depth[mid] = d + 1
        pending.append(mid)
    return path


# ----------------------------------------------------------------------------------------------- linear solver
def _parity(p: np.ndarray) -> int:
    """+1 / −1: sign of the permutation ``p``."""
    p = np.asarray(p).copy()
    seen = np.zeros(len(p), bool)
    sign = 1
    for i in range(len(p)):
        if seen[i]:
            continue
        j, n = i, 0
        while not seen[j]:
            seen[j] = True
            j = int(p[j])
            n += 1
        if n % 2 == 0:
            sign = -sign
    return sign


class IndefiniteDirectSolver:
    """SuperLU of the full symmetric matrix given by its upper triangle (no SPD assumption); ``det_sign()`` of the
    last factorisation. Raises :class:`LinearSolveFailure` for a singular matrix or a non-finite solution."""

    name = "direct-indefinite"

    def __init__(self) -> None:
        self._lu: Any = None
        self.n_factor = 0

    def factor(self, A_upper: sp.csr_matrix) -> None:
        full = (A_upper + sp.triu(A_upper, k=1).T).tocsc()
        try:
            self._lu = spla.splu(full, permc_spec="MMD_AT_PLUS_A")
        except RuntimeError as e:                               # "Factor is exactly singular"
            raise LinearSolveFailure(f"RuntimeError: {e}") from e
        self.n_factor += 1

    def solve(self, B: np.ndarray) -> np.ndarray:
        X = self._lu.solve(np.ascontiguousarray(B, float))
        if not np.all(np.isfinite(X)):
            raise LinearSolveFailure("non-finite solution")
        return X

    def det_sign(self) -> int:
        d = self._lu.U.diagonal()
        s = int(np.prod(np.sign(d)))
        return s * _parity(self._lu.perm_r) * _parity(self._lu.perm_c)



# ----------------------------------------------------------------------------------------------- stability
def full_tangent(matrix: Any, storage: str = "auto") -> Any:
    """The full symmetric tangent of ``matrix`` (dense array or sparse). ``storage``: ``full`` (as given), ``upper``
    (upper triangle stored: A + triu(A, 1)ᵀ) or ``auto`` (upper when the strict lower triangle is empty and the
    strict upper one is not: such a matrix cannot be a full symmetric one)."""
    if storage not in ("auto", "full", "upper"):
        raise ValueError("storage: auto | full | upper")
    if sp.issparse(matrix):
        A = sp.csr_matrix(matrix)
        upper = storage == "upper" or (storage == "auto" and sp.tril(A, -1).nnz == 0 and sp.triu(A, 1).nnz > 0)
        return (A + sp.triu(A, 1).T).tocsr() if upper else A
    M = np.asarray(matrix, float)
    upper = storage == "upper" or (storage == "auto" and not np.any(np.tril(M, -1)) and bool(np.any(np.triu(M, 1))))
    return M + np.triu(M, 1).T if upper else M


@dataclass
class Spectrum:
    """Tangent eigenvalues ``values`` (ascending), unit eigenvectors ``vectors`` (n, m), ``index`` (negative
    eigenvalues of the whole tangent); ``window``: True when only the eigenvalues nearest zero were computed
    (``sparse``); ``offset``: position of ``values[0]`` in the sorted spectrum of the whole tangent."""

    values: np.ndarray
    vectors: np.ndarray
    index: int
    window: bool
    offset: int = 0

    def position(self, j: int) -> tuple[float, np.ndarray]:
        """The j-th eigenpair of the sorted spectrum (0-based); ValueError when it is outside the window."""
        r = j - self.offset
        if not 0 <= r < len(self.values):
            raise ValueError(f"eigenvalue {j} of the sorted spectrum is outside the computed window "
                             f"(positions {self.offset} .. {self.offset + len(self.values) - 1}); raise k")
        return float(self.values[r]), self.vectors[:, r]


@dataclass(frozen=True)
class TangentSpectrum:
    """Eigenvalues of the symmetric tangent (module notes). ``method``: ``dense`` (all, exact inertia; refused above
    ``dense_limit`` unknowns) or ``sparse`` (inertia from a sparse LDLᵀ, the ``k`` nearest zero); ``storage`` as
    :func:`full_tangent`; ``keep``: eigenvalues nearest zero stored per path point."""

    method: str = "dense"
    k: int = 6
    dense_limit: int = 4000
    storage: str = "auto"
    keep: int = 6

    def __post_init__(self) -> None:
        if self.method not in ("dense", "sparse"):
            raise ValueError("method: dense | sparse")
        if self.storage not in ("auto", "full", "upper"):
            raise ValueError("storage: auto | full | upper")
        if not (self.k >= 1 and self.dense_limit >= 1 and self.keep >= 1):
            raise ValueError("k >= 1, dense_limit >= 1, keep >= 1 are required")

    def __call__(self, lin: Linearization) -> Spectrum:
        A = full_tangent(lin.matrix, self.storage)
        n = A.shape[0]
        if self.method == "dense":
            if n > self.dense_limit:
                raise ValueError(f"dense spectrum of {n} unknowns refused (dense_limit {self.dense_limit}); use "
                                 f"method='sparse'")
            M = A.toarray() if sp.issparse(A) else np.asarray(A, float)
            vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
            return Spectrum(vals, vecs, int(np.count_nonzero(vals < 0)), False)
        if not self.k < n - 1:
            raise ValueError(f"sparse spectrum needs k < n - 1 = {n - 1}, got k = {self.k}")
        S = sp.csc_matrix(A) if sp.issparse(A) else sp.csc_matrix(np.asarray(A, float))
        index = sparse_inertia(S)
        v0 = np.random.default_rng(0).standard_normal(n)                     # fixed start: reproducible
        try:
            vals, vecs = spla.eigsh(S, k=self.k, sigma=0.0, which="LM", v0=v0)
        except (RuntimeError, spla.ArpackError) as e:
            raise LinearSolveFailure(f"sparse spectrum failed ({type(e).__name__}: {e})") from e
        o = np.argsort(vals)
        return Spectrum(vals[o], vecs[:, o], index, True, index - int(np.count_nonzero(vals < 0)))


def sparse_inertia(A: Any) -> int:
    """Number of negative eigenvalues of the symmetric sparse ``A`` (full storage) from P A Pᵀ = L D Lᵀ: SuperLU
    with diagonal pivots in symmetric mode (U = D Lᵀ), Sylvester's law of inertia. Raises
    :class:`LinearSolveFailure` for a singular matrix or when SuperLU took an off-diagonal pivot."""
    S = sp.csc_matrix(A)
    try:
        lu = spla.splu(S, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    except RuntimeError as e:
        raise LinearSolveFailure(f"inertia: factorisation failed ({e})") from e
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise LinearSolveFailure("inertia: SuperLU took an off-diagonal pivot (no symmetric LDL^T)")
    return int(np.count_nonzero(lu.U.diagonal() < 0))


def _observe(point: ContinuationPoint, spectrum: TangentSpectrum | None, lin: Linearization) -> None:
    if spectrum is None:
        return
    s = spectrum(lin)
    point.index = s.index
    near = np.argsort(np.abs(s.values))[:spectrum.keep]
    point.eigenvalues = np.sort(s.values[near])


def _crossed(a: ContinuationPoint, b: ContinuationPoint) -> bool:
    if a.index is not None and b.index is not None:
        return a.index != b.index
    return a.det_sign is not None and b.det_sign is not None and a.det_sign != b.det_sign


def find_critical(path: ContinuationPath) -> list[int]:
    """Indices i such that a critical point lies between ``path.points[i]`` and ``[i + 1]`` (module notes)."""
    return [i for i in range(len(path.points) - 1) if _crossed(path.points[i], path.points[i + 1])]


# ----------------------------------------------------------------------------------------------- critical points
@dataclass(frozen=True)
class CriticalSettings:
    """Controls of :func:`locate_critical` and of the hyperplane solve of :func:`switch_branch` (module notes)."""

    eig_tol: float = 1e-10
    t_tol: float = 1e-14
    max_evaluations: int = 60
    bifurcation_tol: float = 1e-6
    max_iter: int = 25
    rtol: float = 1e-10
    eta_tol: float = 1e-14


@dataclass
class CriticalPoint:
    """A located critical point: equilibrium (``x``, ``lam``), unit critical vector ``phi`` (largest component
    positive) and its ``eigenvalue``; ``kind`` ``limit`` | ``bifurcation`` from ``load_test`` = φᵀq/‖q‖;
    ``index_before`` / ``index_after`` of the bracketing points; ``lam_extremum``: λ at both bracketing points on the
    same side of λ_c; ``t`` on the chord; ``evaluations``; ``converged`` (eigenvalue tolerance met); ``records``: the
    constrained solves; ``interval``: i of the bracket (i, i + 1)."""

    lam: float
    x: np.ndarray
    phi: np.ndarray
    eigenvalue: float
    kind: str
    load_test: float
    index_before: int
    index_after: int
    lam_extremum: bool
    t: float
    evaluations: int
    converged: bool
    interval: int
    records: list[StepRecord] = field(default_factory=list)

    @property
    def multiplicity(self) -> int:
        """Number of eigenvalues that crossed zero in the bracket."""
        return abs(self.index_after - self.index_before)


def _refuse_path_dependent(problem: Any, what: str) -> None:
    if getattr(problem, "store", None) is not None:
        raise ValueError(f"{what}: path-dependent problem (committed material state); probe points between accepted "
                         f"ones would use the wrong state")


def _hyperplane_newton(problem: ContinuationProblem, x: np.ndarray, lam: float, c: np.ndarray, w: float,
                       cs: CriticalSettings, rec: StepRecord) -> tuple[bool, np.ndarray, float, Linearization | None]:
    """R(x, λ) = 0 with cᵀx = w, λ unknown (bordered Newton from the predictor (x, λ))."""
    q = np.asarray(problem.load, float)
    nq = float(np.linalg.norm(q))
    nc = float(np.linalg.norm(c))
    x = np.asarray(x, float).copy()
    for it in range(cs.max_iter + 1):
        lin = _linearize(problem, x, lam, rec)
        if lin is None:
            return False, x, lam, None
        R = lin.residual
        conv, rel, eta = _converged(R, lin, x, nq, cs.rtol, cs.eta_tol)
        rec.residuals.append(rel)
        rec.backward_errors.append(eta)
        g = float(c @ x) - w
        if not math.isfinite(rel):
            rec.code, rec.reason = "non_finite", "non-finite residual"
            return False, x, lam, None
        if conv and abs(g) <= 1e-12 * (nc * float(np.linalg.norm(x)) + abs(w)):
            return True, x, lam, lin
        if it == cs.max_iter:
            rec.code, rec.reason = "max_iterations", f"no convergence in {cs.max_iter} iterations"
            return False, x, lam, None
        if len(rec.residuals) > 1 and rel > 1e6 * rec.residuals[0]:
            rec.code, rec.reason = "divergence", "residual diverging"
            return False, x, lam, None
        sol = _solve2(problem, lin, R, rec)
        if sol is None:
            return False, x, lam, None
        dr, dqk = sol
        cq = float(c @ dqk)
        if cq == 0.0 or not math.isfinite(cq):
            rec.code, rec.reason = "singular_tangent", "bordered system singular (c'K^-1 q = 0)"
            return False, x, lam, None
        dl = (w - float(c @ (x + dr))) / cq
        step = dr + dl * dqk
        rec.updates.append(float(np.linalg.norm(step)))
        x = x + step
        lam += dl
    return False, x, lam, None                                                # pragma: no cover


def _signed(phi: np.ndarray) -> np.ndarray:
    v = np.asarray(phi, float) / float(np.linalg.norm(phi))
    return -v if v[int(np.argmax(np.abs(v)))] < 0 else v


def locate_critical(problem: ContinuationProblem, path: ContinuationPath, i: int, spectrum: TangentSpectrum,
                    settings: CriticalSettings | None = None) -> CriticalPoint:
    """Locate and classify the critical point between ``path.points[i]`` and ``[i + 1]`` (module notes). Raises
    ValueError when the spectrum index does not change over the interval, :class:`ContinuationError` (code
    ``locate``) when a constrained solve fails."""
    cs = settings if settings is not None else CriticalSettings()
    _refuse_path_dependent(problem, "locate_critical")
    if not 0 <= i < len(path.points) - 1:
        raise ValueError(f"interval {i} outside the path (0 .. {len(path.points) - 2})")
    A, B = path.points[i], path.points[i + 1]
    sA, sB = spectrum(problem.linearize(A.x, A.lam)), spectrum(problem.linearize(B.x, B.lam))
    mA, mB = sA.index, sB.index
    if mA == mB:
        raise ValueError(f"no critical point between points {i} and {i + 1}: index {mA} at both")
    j = mA if mB > mA else mA - 1                                    # position of the crossing eigenvalue
    fA, vA = sA.position(j)
    fB, vB = sB.position(j)
    if not fA * fB < 0:
        raise ValueError(f"crossing eigenvalue not bracketed ({fA:.3e} at point {i}, {fB:.3e} at {i + 1})")
    c = B.x - A.x
    cc = float(c @ c)
    w0 = float(c @ A.x)
    scale = max(abs(fA), abs(fB))
    q = np.asarray(problem.load, float)
    t0, f0, x0s, l0s = 0.0, fA, A.x, A.lam
    t1, f1, x1s, l1s = 1.0, fB, B.x, B.lam
    best = (abs(fB), 1.0, B.x, B.lam, fB, vB) if abs(fB) < abs(fA) else (abs(fA), 0.0, A.x, A.lam, fA, vA)
    records: list[StepRecord] = []
    side, ev, converged = 0, 0, False
    while ev < cs.max_evaluations:
        t = (t0 * f1 - t1 * f0) / (f1 - f0)
        if not t0 < t < t1:
            t = 0.5 * (t0 + t1)
        s_ = (t - t0) / (t1 - t0)
        xp, lp = x0s + s_ * (x1s - x0s), l0s + s_ * (l1s - l0s)
        rec = StepRecord(lp, False, [])
        rec.notes.append(f"locate t = {t:.17g}")
        ok, xt, lt, lin = _hyperplane_newton(problem, xp, lp, c, w0 + t * cc, cs, rec)
        records.append(rec)
        ev += 1
        if not ok or lin is None:
            raise ContinuationError(f"locate_critical: constrained solve failed at t = {t:.6g} ({rec.code}: "
                                    f"{rec.reason})", "locate", rec.code, path)
        rec.load_factor, rec.converged, rec.code = lt, True, "converged"
        f, v = spectrum(lin).position(j)
        if abs(f) < best[0]:
            best = (abs(f), t, xt, lt, f, v)
        if abs(f) <= max(cs.eig_tol * scale, 16 * np.finfo(float).eps * lin.norm1):
            converged = True
            break
        if f * f1 > 0:
            t1, f1, x1s, l1s = t, f, xt, lt
            if side == -1:
                f0 *= 0.5
            side = -1
        else:
            t0, f0, x0s, l0s = t, f, xt, lt
            if side == 1:
                f1 *= 0.5
            side = 1
        if t1 - t0 <= cs.t_tol:
            break
    _a, t, xc, lc, mu, phi = best
    phi = _signed(phi)
    load_test = float(phi @ q) / float(np.linalg.norm(q))
    kind = "bifurcation" if abs(load_test) <= cs.bifurcation_tol else "limit"
    return CriticalPoint(lc, np.asarray(xc, float).copy(), phi, mu, kind, load_test, mA, mB,
                         (A.lam - lc) * (B.lam - lc) > 0, t, ev, converged, i, records)


# ----------------------------------------------------------------------------------------------- branch switching
@dataclass
class BranchSwitch:
    """The first equilibrium on the secondary branch (``x``, ``lam``, its hyperplane offset ``xi`` and solve
    ``record``) and the path followed from it."""

    x: np.ndarray
    lam: float
    xi: float
    record: StepRecord
    path: ContinuationPath


def switch_branch(problem: ContinuationProblem, critical: CriticalPoint, xi: float, settings: ArcLengthSettings,
                  stop: Callable[[np.ndarray, float], bool], spectrum: TangentSpectrum | None = None,
                  critical_settings: CriticalSettings | None = None,
                  stop_at_critical: bool = False) -> BranchSwitch:
    """Leave a bifurcation point along its critical vector: the equilibrium with φᵀ(x − x_c) = ``xi`` (λ unknown),
    then :func:`arc_length` from it away from x_c (module notes). Raises ValueError for a limit point or ξ = 0,
    :class:`ContinuationError` (code ``branch_switch``) when the hyperplane solve fails."""
    if critical.kind != "bifurcation":
        raise ValueError(f"switch_branch: {critical.kind} point (load test {critical.load_test:.3e}); branch "
                         f"switching applies to bifurcation points")
    if xi == 0.0 or not math.isfinite(xi):
        raise ValueError("switch_branch: xi must be finite and non-zero")
    _refuse_path_dependent(problem, "switch_branch")
    cs = critical_settings if critical_settings is not None else CriticalSettings(
        max_iter=settings.max_iter, rtol=settings.rtol, eta_tol=settings.eta_tol)
    phi = critical.phi
    rec = StepRecord(critical.lam, False, [])
    rec.notes.append(f"branch switch xi = {xi:.6g}")
    ok, xs, ls, lin = _hyperplane_newton(problem, critical.x + xi * phi, critical.lam, phi,
                                         float(phi @ critical.x) + xi, cs, rec)
    if not ok or lin is None:
        raise ContinuationError(f"switch_branch: hyperplane solve failed ({rec.code}: {rec.reason})",
                                "branch_switch", rec.code)
    rec.load_factor, rec.converged, rec.code = ls, True, "converged"
    _commit(problem, xs, ls)
    path = arc_length(problem, xs, ls, settings, stop, spectrum=spectrum,
                      start_tangent=(xs - critical.x, ls - critical.lam), stop_at_critical=stop_at_critical)
    return BranchSwitch(xs, ls, xi, rec, path)
