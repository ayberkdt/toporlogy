"""Component constructors: design (moved from toporlogy.registry).
"""
from __future__ import annotations

from typing import Any

from toporlogy.configuration.helpers import _axis, _coord, _point
from toporlogy.configuration.registry import BuildContext, register


def register_design() -> None:
    """Register the design components (moved from registry._register_features)."""
    from toporlogy.design import manufacturing as MF
    length = "length"

    # ---------------------------------------------------------------- chain links (manufacturing)
    def symmetry_link(ctx: BuildContext, planes: Any) -> Any:
        """Mirror tying about planes {x: 30} or [[x, 30], [y, mid]] (coordinate: number, min, mid, max); first
        link of the chain."""
        items = list(planes.items()) if isinstance(planes, dict) else [tuple(p) for p in planes]
        d = ctx.domain
        return MF.Symmetry(d, [(_axis(a, d.dim), _coord(c, _axis(a, d.dim), d)) for a, c in items])

    def pattern(ctx: BuildContext, kind: str = "translational", offset: list | None = None, n: int | None = None,
                center: list | None = None, axis: Any = "z") -> Any:
        """Pattern repetition: translational (offset vector) or cyclic (n sectors about center[, axis])."""
        d = ctx.domain
        if kind == "translational":
            if offset is None:
                raise ValueError("translational pattern needs offset")
            return MF.Pattern.translational(d, offset)
        if kind == "cyclic":
            if n is None or center is None:
                raise ValueError("cyclic pattern needs n and center")
            return MF.Pattern.cyclic(d, n, _point(center, d, "center"), axis)
        raise ValueError(f'kind: translational | cyclic, got "{kind}"')

    def group_average(ctx: BuildContext) -> Any:
        """Re-ties the filtered field to the groups of the chain's tying link (put right after density_filter)."""
        tying = next((lk for lk in ctx.links if isinstance(lk, MF.Tying)), None)
        if tying is None:
            raise ValueError("group_average needs a tying link (symmetry, extrusion, pattern) first in the chain")
        return MF.GroupAverage(tying)

    register("link", "am_filter", MF.AMFilter, context=("domain",), types={"build": None},
             units={"base_coordinate": length})
    register("link", "casting", MF.Casting, context=("domain",), types={"direction": None},
             units={"parting": length})
    register("link", "symmetry", symmetry_link, types={"planes": None}, units={"planes": length})
    register("link", "extrusion", MF.Extrusion, context=("domain",), types={"axis": None})
    register("link", "pattern", pattern, types={"axis": None}, units={"offset": length, "center": length})
    register("link", "group_average", group_average)
