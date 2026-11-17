"""Continuation schedules (β of the projection, later SIMP penalty, move limits)."""
from __future__ import annotations

from toporlogy.chain import Chain, HeavisideProjection

__all__ = ["Schedule", "BetaContinuation"]


class Schedule:
    """Called after every design update. ``update(iteration, change) -> bool``: True when the problem changed
    (the study then skips its convergence test and resets the optimizer). ``finished`` = continuation done."""

    finished = True

    def update(self, iteration: int, change: float) -> bool:
        return False

    def reset(self) -> None:
        """Back to the initial continuation state; Study.run calls it before the loop."""

    def state(self) -> dict[str, float]:
        return {}


class BetaContinuation(Schedule):
    """β ← min(2β, β_max) every ``every`` iterations or when the design change falls to ``tol`` (topo3d rule).

    The counter restarts at every β step. Without a projection link in the chain this schedule does nothing.
    """

    def __init__(self, chain: Chain, beta0: float = 1.0, beta_max: float = 16.0, every: int = 25,
                 tol: float = 0.01) -> None:
        link = chain.find(HeavisideProjection)
        self.proj: HeavisideProjection | None = link if isinstance(link, HeavisideProjection) else None
        self.beta_max, self.every, self.tol = float(beta_max), int(every), float(tol)
        self.beta0 = float(beta0)
        self.reset()

    def reset(self) -> None:
        if self.proj is not None:
            self.proj.beta = self.beta0
        self._count = 0

    @property
    def beta(self) -> float:
        return self.proj.beta if self.proj is not None else float("nan")

    @property
    def finished(self) -> bool:  # type: ignore[override]
        return self.proj is None or self.proj.beta >= self.beta_max

    def update(self, iteration: int, change: float) -> bool:
        if self.proj is None:
            return False
        self._count += 1
        if self.proj.beta < self.beta_max and (self._count >= self.every or change <= self.tol):
            self.proj.beta = min(2.0 * self.proj.beta, self.beta_max)
            self._count = 0
            return True
        return False

    def state(self) -> dict[str, float]:
        return {"beta": self.beta}
