"""Multi-objective optimisation: dominance, weighted-sum and ε-constraint scalarisations (all objectives minimised).

* :func:`dominates` / :func:`nondominated`: a dominates b when a ≤ b in every objective and a < b in one.
* :func:`weighted_sum`: min Σ w_i (f_i − o_i) / s_i. It reaches only points on the convex hull of the front: on a
  non-convex part no weight returns an interior point, however fine the weights are.
* :func:`epsilon_constraint`: min f_k s.t. f_i ≤ ε_i. It also reaches non-convex parts; a point is reported
  feasible only when every bound holds.

Both continuous solvers here are SciPy SLSQP from several starts (local method: the result is the best local solution
found, not a proof of global optimality). For structural problems the same two scalarisations are a
:class:`~toporlogy.optimization.study.Study` with ``Compliance(weights=...)`` as objective, or with
:class:`CaseCompliance` as objective and constraint.

Verification: ``tests/toporlogy/test_pareto.py`` (records P1-P5 of ``docs/olcumler/laminat_uq_onkayit.md``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from toporlogy.optimization.responses import Compliance, EvalContext, Response

__all__ = ["dominates", "nondominated", "normalise_weights", "ParetoPoint", "weighted_sum", "epsilon_constraint",
           "CaseCompliance"]


def dominates(a: Any, b: Any, tol: float = 0.0) -> bool:
    """a ≤ b + tol everywhere and a < b − tol somewhere."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("objective vectors of different length")
    return bool(np.all(x <= y + tol) and np.any(x < y - tol))


def nondominated(F: Any, tol: float = 0.0) -> np.ndarray:
    """Mask (n,) of the rows of F (n, m) not dominated by any other row. Equal rows are all kept."""
    V = np.asarray(F, float)
    if V.ndim != 2 or not np.isfinite(V).all():
        raise ValueError("F must be a finite (n, m) array")
    return np.array([not any(dominates(V[j], V[i], tol) for j in range(len(V)) if j != i) for i in range(len(V))])


def normalise_weights(weights: Any) -> np.ndarray:
    """Non-negative weights with a positive sum, scaled to sum 1."""
    w = np.asarray(weights, float).reshape(-1)
    if not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0:
        raise ValueError(f"weights must be non-negative with a positive sum, got {w.tolist()}")
    return w / w.sum()


@dataclass(frozen=True)
class ParetoPoint:
    """One solution: its objective values, variables, the scalarisation parameter and whether its bounds hold."""

    objectives: np.ndarray
    x: np.ndarray
    parameter: np.ndarray
    feasible: bool
    success: bool
    message: str = ""


def _solve(fun: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], starts: np.ndarray,
           constraints: list[dict[str, Any]], tol: float) -> tuple[np.ndarray, float, float, bool, str]:
    """Best SLSQP solution over the starts: (x, value, violation, success, message); feasible solutions first."""
    best: tuple[tuple[bool, float], np.ndarray, float, float, bool, str] | None = None
    for x0 in starts:
        r = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                     options={"ftol": 1e-14, "maxiter": 500})
        x = np.clip(r.x, [b[0] for b in bounds], [b[1] for b in bounds])
        viol = max([0.0] + [-float(c["fun"](x)) for c in constraints])
        value = float(fun(x))
        if not math.isfinite(value):
            continue
        key = (viol > tol, value if viol <= tol else viol)
        if best is None or key < best[0]:
            best = (key, x, value, viol, bool(r.success), str(r.message))
    if best is None:
        raise RuntimeError("no start gave a finite objective value")
    return best[1:]


def _starts(bounds: Sequence[tuple[float, float]], n_starts: int, seed: int) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (lo < hi).all()):
        raise ValueError("finite bounds with low < high are required")
    if n_starts < 1:
        raise ValueError("n_starts >= 1")
    return lo + (hi - lo) * np.random.default_rng(seed).random((n_starts, lo.size))


def weighted_sum(objectives: Sequence[Callable[[np.ndarray], float]], weights: Any,
                 bounds: Sequence[tuple[float, float]], ideal: Any = None, scale: Any = None, n_starts: int = 8,
                 seed: int = 0) -> ParetoPoint:
    """min Σ w_i (f_i(x) − ideal_i) / scale_i inside the box ``bounds`` (module notes)."""
    w = normalise_weights(weights)
    m = len(objectives)
    if w.size != m:
        raise ValueError(f"{w.size} weights for {m} objectives")
    o = np.zeros(m) if ideal is None else np.asarray(ideal, float).reshape(m)
    s = np.ones(m) if scale is None else np.asarray(scale, float).reshape(m)
    if not (s > 0).all():
        raise ValueError("scales must be positive")

    def fun(x: np.ndarray) -> float:
        return float(sum(w[i] * (f(x) - o[i]) / s[i] for i, f in enumerate(objectives) if w[i] > 0))

    x, _, _, ok, msg = _solve(fun, bounds, _starts(bounds, n_starts, seed), [], 0.0)
    return ParetoPoint(np.array([f(x) for f in objectives], float), x, w, True, ok, msg)


def epsilon_constraint(objectives: Sequence[Callable[[np.ndarray], float]], primary: int,
                       epsilon: dict[int, float], bounds: Sequence[tuple[float, float]], n_starts: int = 8,
                       seed: int = 0, tol: float = 1e-8) -> ParetoPoint:
    """min f_primary(x) s.t. f_i(x) ≤ epsilon[i] inside the box. ``feasible`` False when no start satisfies every
    bound to ``tol`` (the returned point then is the least violating one and is not a Pareto point)."""
    m = len(objectives)
    if not 0 <= primary < m:
        raise ValueError("primary objective index out of range")
    if not epsilon or primary in epsilon or not all(0 <= i < m for i in epsilon):
        raise ValueError("epsilon: {objective index: bound} for objectives other than the primary one")
    cons = [{"type": "ineq", "fun": (lambda x, i=i, e=float(e): e - objectives[i](x))} for i, e in epsilon.items()]
    x, _, viol, ok, msg = _solve(objectives[primary], bounds, _starts(bounds, n_starts, seed), cons, tol)
    return ParetoPoint(np.array([f(x) for f in objectives], float), x,
                       np.array([epsilon.get(i, math.nan) for i in range(m)]), viol <= tol, ok, msg)


class CaseCompliance(Response):
    """Compliance F_cᵀ U_c of one load case (objective or ε-constraint of a structural trade-off)."""

    needs = frozenset({"u"})

    def __init__(self, case: int) -> None:
        if case < 0:
            raise ValueError("case >= 0")
        self.case = int(case)
        self.name = f"compliance_case{self.case}"

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        c = Compliance().per_case(ctx)
        if self.case >= c.numel():
            raise ValueError(f"load case {self.case} of {c.numel()}")
        return c[self.case]
