"""Component constructors: schedules (moved from toporlogy.registry).
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.registry import BuildContext, register


def register_schedules() -> None:
    """Register the schedules components (moved from registry._register_features)."""

    # ---------------------------------------------------------------- schedule
    def robust_continuation(ctx: BuildContext, beta0: float = 1.0, beta_max: float = 32.0, every: int = 50,
                            tol: float | None = None, volume_every: int = 20, target: float | None = None,
                            update_volume: bool = True) -> Any:
        """beta doubles every `every` iterations or when change <= tol (null = run.tol); every volume_every
        iterations the dilated volume bound becomes target V_d / V_i (target null = its initial upper)."""
        from toporlogy.robust import RealisationVolume, Robust, RobustContinuation
        vol = next((c for c in ctx.constraints if isinstance(c.response, RealisationVolume)), None) \
            if update_volume else None
        rescale = [r for r in [ctx.objective, *[c.response for c in ctx.constraints]] if isinstance(r, Robust)]
        return RobustContinuation(ctx.chain, ctx.robust, beta0, beta_max, every,
                                  float(ctx.run.get("tol", 0.01)) if tol is None else tol, volume=vol,
                                  target=(None if vol is None else (vol.upper if target is None else target)),
                                  volume_every=volume_every, rescale=rescale)

    register("schedule", "robust_continuation", robust_continuation)
