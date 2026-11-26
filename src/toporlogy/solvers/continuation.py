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
  the primary path. No branch switching is attempted or guaranteed.

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
           "ContinuationPath", "ArcLengthSettings", "arc_length", "displacement_control", "IndefiniteDirectSolver"]


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
               stop: Callable[[np.ndarray, float], bool]) -> ContinuationPath:
    """Follow the equilibrium path from the equilibrium (``x0``, ``lam0``) until ``stop(x, lam)`` is true or
    ``max_steps`` points are accepted (module notes)."""
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
    dx_prev: np.ndarray | None = None
    dlam_prev = 0.0
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
            path.records.append(rec)
            dx_prev, dlam_prev = dx, dlam
            n_it = max(len(rec.residuals) - 1, 1)
            ds = min(max(ds * math.sqrt(st.desired_iterations / n_it), ds_min), ds_max)
            if stop(x, lam):
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
                         max_bisections: int = 10) -> ContinuationPath:
    """Equilibria with the unknown ``dof`` prescribed to each of ``targets`` in turn and λ unknown (module notes).
    A failed step is bisected (at most ``max_bisections`` levels)."""
    q = np.asarray(problem.load, float)
    nq = float(np.linalg.norm(q))
    path = ContinuationPath()
    x, lam = np.asarray(x0, float).copy(), float(lam0)
    lin0 = problem.linearize(x, lam)
    path.points.append(ContinuationPoint(lam, x.copy(), _det_sign(problem, lin0), 0, 0.0,
                                         float(np.linalg.norm(lin0.residual)) / nq))
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
