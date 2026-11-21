"""Many local constraints ``g_j(x) ≤ 0`` (one per element, per load case, per neighbourhood …) in a form an optimiser
with few constraints can use. Two standard routes, both built from the same :class:`LocalField`:

* :class:`Aggregated` — one global response ``c · A(v)`` with a smooth aggregate A of
  :mod:`toporlogy.optimization.aggregates` (P-norm, KS, induced …) and, optionally, the adaptive scale of Le et al.
  that ties it to the true maximum. One constraint for MMA / GCMMA.
* :class:`AugmentedLagrangian` — the local constraints move into the objective,

      J = f / f_ref + (1/N) Σ_j [ λ_j h_j + (μ/2) h_j² ],     h_j = max(g_j, −λ_j / μ),

  with the multiplier update ``λ_j ← λ_j + μ h_j`` and ``μ ← min(α μ, μ_max)`` done by :class:`MultiplierUpdate`
  between groups of optimiser iterations. Whatever N is, a gradient costs one adjoint solve, because the sum is one
  scalar. This is the aggregation-free strategy of Senhora et al. (2020) and Giraldo-Londoño & Paulino (2021); only
  the records and abstracts of those papers were seen in this work, the formula above is the textbook augmented
  Lagrangian for inequalities (Bertsekas) and the default parameters are NOT taken from those papers
  (``docs/literatur_2026-11-21.md`` §2.1).

A new local criterion (stress, strain energy density, local volume, temperature, fatigue damage …) is therefore one
class with ``__call__(ctx) -> tensor``; both strategies, their gradients and the YAML components come with it.
:class:`FunctionField` wraps a plain function for experiments.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from toporlogy.optimization.aggregates import AdaptiveScale, aggregate
from toporlogy.optimization.responses import EvalContext, Response
from toporlogy.optimizers.schedule import Schedule
from toporlogy.references import cite

__all__ = ["LocalField", "FunctionField", "Aggregated", "AugmentedLagrangian", "MultiplierUpdate"]


class LocalField:
    """Base class of a field of local values: ``__call__(ctx) -> tensor`` of any shape.

    ``needs`` as for :class:`~toporlogy.optimization.responses.Response` (``"u"`` = displacements). Convention for
    constraints: a value ≤ 0 is feasible. ``weights(ctx)`` may return per-entry weights (e.g. volumes) or None.
    """

    name = "field"
    needs: frozenset[str] = frozenset()

    def __call__(self, ctx: EvalContext) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def weights(self, ctx: EvalContext) -> torch.Tensor | None:
        return None


class FunctionField(LocalField):
    """A :class:`LocalField` from a function ``fn(ctx) -> tensor`` (experiments, tests)."""

    def __init__(self, fn: Callable[[EvalContext], torch.Tensor], name: str = "field",
                 needs: frozenset[str] | set[str] = frozenset()) -> None:
        self.fn, self.name, self.needs = fn, name, frozenset(needs)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        return self.fn(ctx)


class Aggregated(Response):
    """``scale · A(field)``: a smooth stand-in for ``max_j field_j`` (module notes).

    Parameters
    ----------
    field : LocalField
    kind : str
        Aggregate of :data:`toporlogy.optimization.aggregates.AGGREGATES` or ``"max"``.
    parameter : float, optional
        ρ (KS-type) or p (power-type); None = the aggregate's default.
    adaptive : float, optional
        Damping a of :class:`~toporlogy.optimization.aggregates.AdaptiveScale` (0 < a ≤ 1); None = no correction.
        The factor is updated at every gradient-enabled evaluation (one per optimiser iteration) and held constant
        in line searches and re-evaluations, so a returned design's value uses the factor of the last iteration.
    weighted : bool
        Use the field's weights.

    Attributes
    ----------
    last_max, last_aggregate : float
        True maximum and unscaled aggregate of the last evaluation (reports, tests).
    """

    def __init__(self, field: LocalField, kind: str = "pnorm", parameter: float | None = None,
                 adaptive: float | None = None, weighted: bool = False, name: str | None = None) -> None:
        self.field, self.kind, self.parameter = field, kind, parameter
        self.weighted = bool(weighted)
        self.scale = None if adaptive is None else AdaptiveScale(adaptive)
        self.needs = frozenset(field.needs)
        self.name = name or f"{field.name}_{kind}"
        self.last_max = float("nan")
        self.last_aggregate = float("nan")

    def reset_scale(self) -> None:
        """Forget the adaptive factor (after a discontinuous change of the problem)."""
        if self.scale is not None:
            self.scale.reset()

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        v = self.field(ctx)
        a = aggregate(v, self.kind, self.parameter, self.field.weights(ctx) if self.weighted else None)
        self.last_max, self.last_aggregate = float(v.detach().max()), float(a.detach())
        if self.scale is None:
            return a
        if torch.is_grad_enabled() or self.scale.value is None:
            self.scale.update(self.last_max, self.last_aggregate)
        return a * float(self.scale.value if self.scale.value is not None else 1.0)


@cite("senhora_2020", "giraldo_londono_paulino_2021", "bertsekas_1999",
      note="augmented Lagrangian with the 1/N normalisation; records seen, texts not read; parameters are this "
           "library's defaults")
class AugmentedLagrangian(Response):
    """``f / f_ref + (1/N) Σ [λ_j h_j + (μ/2) h_j²]`` with ``h_j = max(g_j, −λ_j/μ)`` (module notes).

    Parameters
    ----------
    objective : Response
        f (e.g. volume or compliance).
    field : LocalField
        g_j ≤ 0.
    mu0, alpha, mu_max : float
        Initial penalty, its growth factor per multiplier update and its cap.
    reference : float or "first"
        f_ref: a number, or the value of f at the first evaluation after :meth:`reset` ("first").

    The multipliers have the field's shape and start at zero; they are state of this object, changed only by
    :meth:`update` (called by :class:`MultiplierUpdate`). For fixed multipliers the value is a smooth function of the
    design (h is C¹ in g), so the finite-difference check applies.
    """

    def __init__(self, objective: Response, field: LocalField, mu0: float = 10.0, alpha: float = 1.1,
                 mu_max: float = 1e4, reference: float | str = "first", name: str | None = None) -> None:
        if not (mu0 > 0 and alpha >= 1 and mu_max >= mu0):
            raise ValueError("need mu0 > 0, alpha >= 1 and mu_max >= mu0")
        if not (reference == "first" or (isinstance(reference, (int, float)) and reference > 0)):
            raise ValueError("reference: a positive number or 'first'")
        self.objective, self.field = objective, field
        self.mu0, self.alpha, self.mu_max = float(mu0), float(alpha), float(mu_max)
        self.reference = reference
        self.needs = frozenset(objective.needs) | frozenset(field.needs)
        self.name = name or f"al_{objective.name}_{field.name}"
        self.reset()

    def reset(self) -> None:
        """Zero multipliers, initial penalty, forget f_ref and the stored constraint values."""
        self.mu = self.mu0
        self.lam: torch.Tensor | None = None
        self.f_ref: float | None = None if self.reference == "first" else float(self.reference)
        self.last_g: torch.Tensor | None = None
        self.last_objective = float("nan")
        self.n_updates = 0

    @property
    def max_violation(self) -> float:
        """Largest g_j of the last gradient-enabled evaluation (NaN before the first)."""
        return float("nan") if self.last_g is None else float(self.last_g.max())

    def penalty(self, g: torch.Tensor) -> torch.Tensor:
        lam = self.lam if self.lam is not None else torch.zeros_like(g.detach())
        h = torch.maximum(g, -lam / self.mu)
        return (lam * h + 0.5 * self.mu * h * h).sum() / g.numel()

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        f = self.objective(ctx)
        g = self.field(ctx)
        if self.lam is None or self.lam.shape != g.shape:
            self.lam = torch.zeros_like(g.detach())
        if self.f_ref is None:
            self.f_ref = max(abs(float(f.detach())), 1e-300)
        if torch.is_grad_enabled():
            self.last_g = g.detach().clone()
            self.last_objective = float(f.detach())
        return f / self.f_ref + self.penalty(g)

    def update(self) -> None:
        """``λ ← λ + μ h`` (so λ ≥ 0), then ``μ ← min(α μ, μ_max)``; uses the constraint values of the last
        gradient-enabled evaluation."""
        if self.last_g is None or self.lam is None:
            return
        h = torch.maximum(self.last_g, -self.lam / self.mu)
        self.lam = (self.lam + self.mu * h).clamp_min(0.0)
        self.mu = min(self.alpha * self.mu, self.mu_max)
        self.n_updates += 1


class MultiplierUpdate(Schedule):
    """Calls :meth:`AugmentedLagrangian.update` every ``every`` iterations.

    ``reset_optimizer`` (default False): report the update to the study as a change of the problem, so that it
    resets the optimizer as at a β step. Measured on the L-bracket (21 Nov 2026, ``docs/olcumler/yontemler.md``):
    with resets (MMA asymptotes re-initialised every few iterations) the stress never settled; without them the
    multipliers change the objective gradually and MMA follows.

    ``finished`` once ``min_updates`` updates were made and the largest local constraint value of the last iteration
    is ≤ ``tol``: the study may then stop on its design-change test. ``inner`` (optional) is another schedule (e.g.
    β continuation) that is advanced as well; both must be finished.
    """

    def __init__(self, al: AugmentedLagrangian, every: int = 5, tol: float = 1e-3, min_updates: int = 1,
                 inner: Schedule | None = None, reset_optimizer: bool = False) -> None:
        if every < 1:
            raise ValueError("every >= 1 is required")
        self.al, self.every, self.tol, self.min_updates = al, int(every), float(tol), int(min_updates)
        self.inner = inner
        self.reset_optimizer = bool(reset_optimizer)
        self._count = 0

    def reset(self) -> None:
        self.al.reset()
        self._count = 0
        if self.inner is not None:
            self.inner.reset()

    @property
    def finished(self) -> bool:  # type: ignore[override]
        ok = self.al.n_updates >= self.min_updates and self.al.max_violation <= self.tol
        return ok and (self.inner is None or self.inner.finished)

    def update(self, iteration: int, change: float) -> bool:
        changed = False if self.inner is None else bool(self.inner.update(iteration, change))
        self._count += 1
        if self._count >= self.every:
            self._count = 0
            self.al.update()
            changed = changed or self.reset_optimizer
        return changed

    def state(self) -> dict[str, Any]:
        s: dict[str, Any] = {} if self.inner is None else dict(self.inner.state())
        s["al_mu"] = self.al.mu                    # the largest g_j is not a state: it is unknown before an evaluation
        s["al_updates"] = float(self.al.n_updates)
        return s
