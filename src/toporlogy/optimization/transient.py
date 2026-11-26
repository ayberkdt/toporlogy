"""Responses of the transient physics: time-averaged thermal and dynamic compliance.

Physics :class:`~toporlogy.fem.transient_heat.TransientHeatPhysics` (θ-method heat conduction) and
:class:`~toporlogy.fem.dynamics_newmark.NewmarkPhysics` (Newmark average acceleration). Neither has a static state u;
the responses declare ``needs = {"history"}`` (so the study runs no static solve) and integrate the history through
``ctx.physics`` once per evaluation (cached in ``ctx.extra``). Gradients: every step is a ``GridSPD`` autograd solve,
so torch's chain rule over the steps is the discrete adjoint of the time-stepping scheme (the factorisation of the
forward run is reused backwards; the graph holds every step, no checkpointing).

Both responses are trapezoidal time averages over [0, t_N], t_N = N Δt, weights w = Δt (½, 1, …, 1, ½):

* :class:`TransientThermalCompliance`: J = (1/t_N) Σ_n w_n qᵀ T_n. With a constant heat input it tends to the static
  thermal compliance qᵀ K⁻¹ q as t_N → ∞.
* :class:`DynamicCompliance`: J = (1/t_N) Σ_n w_n f(t_n)ᵀ u_n, f(t) = s(t) F. Under an undamped step load from rest
  every mode oscillates about its static value, so J tends to the static compliance Fᵀ K⁻¹ F.

Pre-registered checks: ``docs/olcumler/gecici_study_onkayit.md`` (HS1-HS7, DS1-DS7).
"""
from __future__ import annotations

import numpy as np
import torch

from toporlogy.fem.dynamics_newmark import NewmarkPhysics
from toporlogy.fem.transient_heat import TransientHeatPhysics
from toporlogy.optimization.responses import EvalContext, Response

__all__ = ["DynamicCompliance", "TransientThermalCompliance", "time_weights"]


def time_weights(n_steps: int, dt: float) -> np.ndarray:
    """Trapezoidal weights (n_steps + 1,) of ∫ over [0, n_steps · dt] divided by its length (they sum to 1)."""
    w = np.full(int(n_steps) + 1, float(dt))
    w[0] = w[-1] = 0.5 * float(dt)
    return w / (int(n_steps) * float(dt))


def _active_moduli(ctx: EvalContext, elements: np.ndarray) -> torch.Tensor:
    return ctx.E.reshape(-1)[torch.as_tensor(elements, dtype=torch.long, device=ctx.E.device)]


class TransientThermalCompliance(Response):
    """(1/t_N) Σ_n w_n qᵀ T_n of :class:`TransientHeatPhysics` (module notes)."""

    name = "transient_thermal_compliance"
    needs = frozenset({"history"})

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        ph = ctx.physics
        if not isinstance(ph, TransientHeatPhysics):
            raise TypeError("transient_thermal_compliance needs physics transient_heat")
        T = ctx.extra.get("temperature_history")
        if T is None:
            T = ph.history(_active_moduli(ctx, ph.elements), ctx.xphys)
            ctx.extra["temperature_history"] = T
        w = torch.as_tensor(time_weights(ph.n_steps, ph.dt), dtype=T.dtype)
        q = torch.as_tensor(ph.q, dtype=T.dtype)
        return (w * (T @ q)).sum()


class DynamicCompliance(Response):
    """(1/t_N) Σ_n w_n f(t_n)ᵀ u_n of :class:`NewmarkPhysics` with the study's single load case (module notes)."""

    name = "dynamic_compliance"
    needs = frozenset({"history"})

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        ph = ctx.physics
        if not isinstance(ph, NewmarkPhysics):
            raise TypeError("dynamic_compliance needs physics dynamics_newmark")
        if ctx.F is None or np.asarray(ctx.F).reshape(ctx.domain.ndof, -1).shape[1] != 1:
            raise ValueError("dynamic_compliance needs exactly one load case")
        F = np.asarray(ctx.F, float).reshape(-1)
        res = ctx.extra.get("newmark_history")
        if res is None:
            res = ph.history(_active_moduli(ctx, ph.elements), ctx.xphys, F)
            ctx.extra["newmark_history"] = res
        t = ph.times()
        w = time_weights(ph.n_steps, ph.dt) * np.array([ph.load_factor(float(s)) for s in t])
        work = res.u @ torch.as_tensor(F, dtype=res.u.dtype)                 # Fᵀ u_n
        return (torch.as_tensor(w, dtype=work.dtype) * work).sum()
