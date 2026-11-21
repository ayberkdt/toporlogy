"""Component constructors: schedules.
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.registry import BuildContext, register


def register_schedules() -> None:
    """Register the schedules components."""

    # ---------------------------------------------------------------- schedule
    def robust_continuation(ctx: BuildContext, beta0: float = 1.0, beta_max: float = 32.0, every: int = 50,
                            tol: float | None = None, volume_every: int = 20, target: float | None = None,
                            update_volume: bool = True) -> Any:
        """beta doubles every `every` iterations or when change <= tol (null = run.tol); every volume_every
        iterations the dilated volume bound becomes target V_d / V_i (target null = its initial upper)."""
        from toporlogy.optimization.robust import RealisationVolume, Robust, RobustContinuation
        vol = next((c for c in ctx.constraints if isinstance(c.response, RealisationVolume)), None) \
            if update_volume else None
        rescale = [r for r in [ctx.objective, *[c.response for c in ctx.constraints]] if isinstance(r, Robust)]
        return RobustContinuation(ctx.chain, ctx.robust, beta0, beta_max, every,
                                  float(ctx.run.get("tol", 0.01)) if tol is None else tol, volume=vol,
                                  target=(None if vol is None else (vol.upper if target is None else target)),
                                  volume_every=volume_every, rescale=rescale)

    def multiplier_update(ctx: BuildContext, every: int = 5, tol: float = 1.0e-3, min_updates: int = 1,
                          beta0: float | None = None, beta_max: float = 16.0, beta_every: int = 25,
                          reset_optimizer: bool = False) -> Any:
        """Augmented-Lagrangian multiplier update every `every` iterations (objective must be an augmented
        Lagrangian, e.g. stress_augmented_lagrangian); finished when the largest local constraint value <= tol.
        beta0 not null: beta continuation of the projection runs alongside (beta0 .. beta_max, every beta_every)."""
        from toporlogy.optimization.local import AugmentedLagrangian, MultiplierUpdate
        from toporlogy.optimizers.schedule import BetaContinuation
        if not isinstance(ctx.objective, AugmentedLagrangian):
            raise ValueError("multiplier_update needs an augmented-Lagrangian objective "
                             "(e.g. {type: stress_augmented_lagrangian, ...})")
        inner = None if beta0 is None else BetaContinuation(ctx.chain, beta0, beta_max, beta_every,
                                                            float(ctx.run.get("tol", 0.01)))
        return MultiplierUpdate(ctx.objective, every, tol, min_updates, inner, reset_optimizer)

    register("schedule", "robust_continuation", robust_continuation)
    register("schedule", "multiplier_update", multiplier_update)
