"""General Newton–Raphson for R(x) = 0 with separate residual, tangent and linear-solver contracts (04 §1).

The solver knows neither the material nor the discretisation. A problem supplies

* ``linearize(x) -> Linearization``: the residual R(x) on the unknowns, the tangent in whatever form its own
  ``solve`` accepts, ‖K‖₁ (for the backward error), optionally the energy Π(x) (conservative problems) and an
  opaque ``aux`` (the trial state of a path-dependent material, element data). It may raise
  :class:`~toporlogy.core.constitutive.ConstitutiveFailure` (code ``invalid_deformation`` / ``return_mapping``);
* ``solve(lin, rhs) -> dx``: solves K dx = rhs with that tangent; raises :class:`LinearSolveFailure` (code
  ``singular_tangent``) when the factorisation fails, the solution is not finite or fails its residual check.

Convergence (both reported per iteration; a small update alone is never success): relative residual
‖R‖/``scale`` ≤ ``rtol`` or backward error η = ‖R‖ / (‖K‖₁ ‖x‖ + scale) ≤ ``eta_tol`` (``scale`` = ‖λF‖ of the
load step). Failure codes (:data:`CODES`): ``non_finite`` residual, ``max_iterations``, ``divergence`` (‖R‖ above
``divergence`` × the first residual), ``stagnation`` (relative residual not halved over ``stagnation_window``
iterations), ``singular_tangent``, ``invalid_deformation``, ``return_mapping`` (from the problem), ``line_search``
(no acceptable step length). On failure the start point is returned (rollback is the caller's: nothing it
committed has changed).

Line search (``line_search="backtracking"``): after the Newton direction, trial points x − s dx; accepted when the
merit decreases by the Armijo condition with c = ``ls_c``; otherwise s is reduced by a safeguarded quadratic
interpolation (s ∈ [0.1 s, 0.5 s]), at most ``ls_max`` reductions. Merit ``residual``: ½‖R‖² (the Newton direction
is a descent direction for it whenever the solve is exact). Merit ``energy``: Π(x), used only when the problem
declares it conservative AND the direction is a descent direction (Rᵀdx > 0); otherwise the residual merit is used
and the switch is logged (an indefinite tangent past a limit point does not give a descent direction of Π). The
accepted trial linearisation is reused for the next iteration. ``line_search="none"`` is the plain full step and
performs exactly the arithmetic of the original ``fem.nonlinear`` Newton loop.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from toporlogy.core.constitutive import ConstitutiveFailure

__all__ = ["CODES", "LinearSolveFailure", "Linearization", "NewtonProblem", "NewtonSettings", "StepRecord",
           "NewtonOutcome", "newton"]

CODES = ("converged", "non_finite", "max_iterations", "divergence", "stagnation", "singular_tangent",
         "invalid_deformation", "return_mapping", "line_search", "step_limit", "complex_roots")


class LinearSolveFailure(RuntimeError):
    """The tangent could not be solved (singular / failed factorisation, non-finite solution, residual check)."""

    code = "singular_tangent"


@dataclass
class Linearization:
    """Residual R (n,), tangent ``matrix`` (form of the problem's solver), ‖K‖₁, energy Π or None, ``aux``."""

    residual: np.ndarray
    matrix: Any
    norm1: float
    energy: float | None = None
    aux: Any = None


class NewtonProblem(Protocol):
    conservative: bool

    def linearize(self, x: np.ndarray) -> Linearization: ...

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class NewtonSettings:
    """Controls of :func:`newton` (module notes)."""

    max_iter: int = 30
    rtol: float = 1e-9
    eta_tol: float = 1e-13
    divergence: float = 1e6
    stagnation_window: int = 5
    stagnation_factor: float = 0.5
    line_search: str = "none"               # none | backtracking
    merit: str = "residual"                 # residual | energy
    ls_c: float = 1e-4
    ls_max: int = 8

    def __post_init__(self) -> None:
        if self.line_search not in ("none", "backtracking"):
            raise ValueError("line_search: none | backtracking")
        if self.merit not in ("residual", "energy"):
            raise ValueError("merit: residual | energy")
        if not (self.max_iter >= 1 and self.rtol > 0 and self.eta_tol >= 0):
            raise ValueError("max_iter >= 1, rtol > 0, eta_tol >= 0 are required")


@dataclass
class StepRecord:
    """One load increment (or Newton solve): target load factor, accepted or not, failure ``code`` and ``reason``;
    per iteration the relative residual ‖R‖/‖λF‖ and backward error η (first entries before any update), the update
    norm ‖dx‖, the accepted line-search step length; after the step: min det F, energy, dissipation, notes."""

    load_factor: float
    converged: bool
    residuals: list[float]
    reason: str = ""
    backward_errors: list[float] = field(default_factory=list)
    code: str = ""
    updates: list[float] = field(default_factory=list)
    step_lengths: list[float] = field(default_factory=list)
    min_det_f: float | None = None
    energy: float | None = None
    dissipation: float | None = None
    rolled_back: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.residuals) - 1


@dataclass
class NewtonOutcome:
    converged: bool
    x: np.ndarray
    lin: Linearization | None
    record: StepRecord
    relative_residual: float
    backward_error: float


def _fail(rec: StepRecord, code: str, reason: str, x0: np.ndarray, rel: float, eta: float) -> NewtonOutcome:
    rec.code, rec.reason, rec.rolled_back = code, reason, True
    return NewtonOutcome(False, x0, None, rec, rel, eta)


def newton(problem: NewtonProblem, x0: np.ndarray, scale: float, settings: NewtonSettings | None = None,
           load_factor: float = 1.0, describe: Callable[[ConstitutiveFailure], str] | None = None) -> NewtonOutcome:
    """Solve R(x) = 0 from ``x0`` (not modified). ``scale``: the norm the relative residual divides by (‖λF‖)."""
    s = settings if settings is not None else NewtonSettings()
    x = x0.copy()
    rec = StepRecord(load_factor, False, [])
    first: float | None = None
    lin: Linearization | None = None
    for it in range(s.max_iter + 1):
        if lin is None:
            try:
                lin = problem.linearize(x)
            except ConstitutiveFailure as e:
                msg = describe(e) if describe is not None else f"inadmissible state ({e})"
                return _fail(rec, e.code, msg, x0, np.inf, np.inf)
        R = lin.residual
        nr = float(np.linalg.norm(R))
        k1 = lin.norm1
        rel = nr / scale
        eta = nr / (k1 * float(np.linalg.norm(x)) + scale)
        rec.residuals.append(rel)
        rec.backward_errors.append(eta)
        if not math.isfinite(nr):
            return _fail(rec, "non_finite", "non-finite residual", x0, rel, eta)
        if rel <= s.rtol or eta <= s.eta_tol:
            rec.converged, rec.code = True, "converged"
            return NewtonOutcome(True, x, lin, rec, rel, eta)
        first = nr if first is None else first
        if it == s.max_iter:
            return _fail(rec, "max_iterations", f"no convergence in {s.max_iter} iterations", x0, rel, eta)
        if nr > s.divergence * first:
            return _fail(rec, "divergence", "residual diverging", x0, rel, eta)
        w = s.stagnation_window
        if it >= w + 1 and rel > s.stagnation_factor * rec.residuals[-(w + 1)]:
            return _fail(rec, "stagnation", "residual stagnating (less than halved over 5 iterations)"
                         if w == 5 else f"residual stagnating (less than halved over {w} iterations)", x0, rel, eta)
        try:
            dx = problem.solve(lin, R)
        except LinearSolveFailure as e:
            return _fail(rec, "singular_tangent", f"tangent solve failed ({e})", x0, rel, eta)
        rec.updates.append(float(np.linalg.norm(dx)))
        if s.line_search == "none":
            x -= dx
            lin = None
            continue
        ok, x, lin, step, note = _line_search(problem, x, dx, lin, s)
        rec.step_lengths.append(step)
        if note:
            rec.notes.append(f"iteration {it}: {note}")
        if not ok:
            return _fail(rec, "line_search", f"line search: no acceptable step ({note})", x0, rel, eta)
    return _fail(rec, "max_iterations", "unreachable", x0, np.inf, np.inf)       # pragma: no cover


def _line_search(problem: NewtonProblem, x: np.ndarray, dx: np.ndarray, lin: Linearization,
                 s: NewtonSettings) -> tuple[bool, np.ndarray, Linearization | None, float, str]:
    R = lin.residual
    slope_r = -float(R @ R)                                   # d/ds ½‖R(x − s dx)‖² at s = 0 (exact Newton step)
    use_energy = s.merit == "energy" and getattr(problem, "conservative", False) and lin.energy is not None
    note = ""
    if s.merit == "energy" and not use_energy:
        note = "energy merit not available (non-conservative problem or no energy); residual merit used"
    if use_energy:
        slope_e = -float(R @ dx)                              # dΠ/ds = ∇Π·(−dx) = −Rᵀdx
        if not slope_e < 0:
            use_energy = False
            note = f"not a descent direction of the energy (R'dx = {-slope_e:.3e}); residual merit used"
    m0 = float(lin.energy) if (use_energy and lin.energy is not None) else 0.5 * float(R @ R)
    slope = slope_e if use_energy else slope_r
    step = 1.0
    for _ in range(s.ls_max + 1):
        xt = x - step * dx
        try:
            lt = problem.linearize(xt)
            mt = float(lt.energy) if (use_energy and lt.energy is not None) else \
                0.5 * float(lt.residual @ lt.residual)
        except ConstitutiveFailure as e:
            lt, mt = None, math.inf
            note = (note + "; " if note else "") + f"trial step {step:.3g}: {e.code}"
        if lt is not None and math.isfinite(mt) and mt <= m0 + s.ls_c * step * slope:
            return True, xt, lt, step, note
        # safeguarded quadratic model of the merit along the line: m(s) ≈ m0 + slope s + c s²
        if math.isfinite(mt):
            denom = 2.0 * (mt - m0 - slope * step)
            trial = -slope * step * step / denom if denom > 0 else 0.5 * step
        else:
            trial = 0.5 * step
        step = min(max(trial, 0.1 * step), 0.5 * step)
    return False, x, lin, step, note or "merit did not decrease"
