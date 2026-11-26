"""Mass-property responses of the physical density (no state solve): centre of mass and inertia tensor components,
from :func:`toporlogy.core.mass_properties.mass_properties` with the relative mass density x̄ (linear) and the
material density ``rho0``. Differentiable in x̄ through torch. Pre-registered checks:
``docs/olcumler/kutle_ozellikleri_onkayit.md`` (MP7-MP9).
"""
from __future__ import annotations

from typing import Any

import torch

from toporlogy.core.mass_properties import mass_properties
from toporlogy.optimization.responses import EvalContext, Response

__all__ = ["CentreOfMass", "InertiaComponent"]

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis(a: Any) -> int:
    if isinstance(a, str):
        if a not in _AXES:
            raise ValueError(f"axis: x | y | z, got {a!r}")
        return _AXES[a]
    i = int(a)
    if i not in (0, 1, 2):
        raise ValueError(f"axis: 0, 1 or 2, got {a!r}")
    return i


class CentreOfMass(Response):
    """Coordinate ``axis`` of the centre of mass of x̄ (length unit of the domain)."""

    name = "centre_of_mass"

    def __init__(self, axis: Any = "x", rho0: float = 1.0) -> None:
        self.axis, self.rho0 = _axis(axis), float(rho0)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        p = mass_properties(ctx.domain, ctx.xphys, rho0=self.rho0)
        out: torch.Tensor = p.centre[self.axis]
        return out


class InertiaComponent(Response):
    """Component I_ij of the inertia tensor about the centre of mass (``about="centre"``) or a point."""

    name = "inertia"

    def __init__(self, i: Any = "z", j: Any = None, rho0: float = 1.0, about: Any = "centre") -> None:
        self.i = _axis(i)
        self.j = self.i if j is None else _axis(j)
        self.rho0, self.about = float(rho0), about

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        p = mass_properties(ctx.domain, ctx.xphys, rho0=self.rho0, about=self.about)
        out: torch.Tensor = p.inertia[self.i, self.j]
        return out
