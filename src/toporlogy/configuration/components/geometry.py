"""Component constructors: geometry (moved from toporlogy.registry).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from toporlogy.configuration.helpers import _combine, _region_mask, find_bore
from toporlogy.configuration.registry import BuildContext, register


def register_geometry() -> None:
    """Register the geometry components (moved from registry._register_features)."""
    from toporlogy import geometry as G
    from toporlogy.core.domain import Domain

    length = "length"

    # ---------------------------------------------------------------- shapes (geometry CSG primitives)
    def _op(s: Any, op: str) -> Any:
        o = str(op).strip().lower()
        if o not in ("add", "cut", "intersect"):
            raise ValueError(f'op: add | cut | intersect, got "{op}"')
        s.op = o
        return s

    def box_shape(lo: list, hi: list, op: str = "add") -> Any:
        """Axis-aligned box lo <= x <= hi (2-D rectangle or 3-D box)."""
        return _op(G.Box(lo, hi), op)

    def ball(center: list, radius: float, op: str = "add") -> Any:
        """Sphere (3-D) or circle (2-D)."""
        return _op(G.Ball(center, radius), op)

    def cylinder(p0: list, p1: list, radius: float, op: str = "add") -> Any:
        """Capped circular cylinder with axis p0 -> p1 (3-D)."""
        return _op(G.Cylinder(p0, p1, radius), op)

    def capsule(p0: list, p1: list, radius: float, op: str = "add") -> Any:
        """Points within radius of the segment p0-p1 (2-D rounded slot / 3-D capsule)."""
        return _op(G.Capsule(p0, p1, radius), op)

    def halfspace(point: list, normal: list, op: str = "intersect") -> Any:
        """(x - point).normal <= 0 (material opposite to the normal); unbounded: use with op intersect or cut."""
        return _op(G.HalfSpace(point, normal), op)

    def polygon(vertices: list, op: str = "add") -> Any:
        """2-D simple polygon (vertices in order)."""
        return _op(G.Polygon(vertices), op)

    def prism(polygon: list, axis: Any = "z", range: list = (0.0, 1.0), op: str = "add") -> Any:  # noqa: A002
        """Polygon extruded along axis over range [lo, hi]; polygon (x, y) for z, (x, z) for y, (y, z) for x."""
        return _op(G.Prism(polygon, axis, (float(range[0]), float(range[1]))), op)

    def slot(p0: list, p1: list, radius: float, depth: list, op: str = "add") -> Any:
        """Rounded slot: the stadium of half-width radius about p0-p1 extruded along the vector depth (3-D)."""
        return _op(G.Slot(p0, p1, radius, depth), op)

    def bore_shape(ctx: BuildContext, index: int | None = None, near: list | None = None, extra_radius: float = 0.0,
                   extend: float = 0.0, through: bool = True, op: str = "cut") -> Any:
        """An auto-detected bore of the part as a cylinder (finalize: preserve it exactly). index 1-based (B1 ...),
        near = point (nearest axis); through: infinite in post-processing."""
        if ctx.domain is None:
            raise ValueError("a bore shape needs a built domain (use it in post steps or selectors)")
        b = find_bore(ctx, index, near)
        s = _op(b.shape(extra_radius, extend), op)
        s.through, s.bore = bool(through), b
        return s

    for name, fn in (("box", box_shape), ("ball", ball), ("sphere", ball), ("circle", ball),
                     ("cylinder", cylinder), ("capsule", capsule), ("halfspace", halfspace),
                     ("polygon", polygon), ("prism", prism), ("slot", slot)):
        register("shape", name, fn, types={"axis": None},
                 units={k: length for k in ("lo", "hi", "center", "radius", "p0", "p1", "point", "vertices",
                                            "polygon", "range", "depth")})
    register("shape", "bore", bore_shape, units={"near": length, "extra_radius": length, "extend": length})

    # ---------------------------------------------------------------- domains from geometry
    def csg_domain(ctx: BuildContext, shapes: Any, h: float, padding: int = 0, bounds: list | None = None,
                   sub: int = 4, min_frac: float = 0.05, connectivity: str | None = "largest", keep: Any = None,
                   void: Any = None, thickness: float = 1.0) -> Domain:
        """Part = CSG of shapes (applied in order by their op), voxelised with partial volumes on a grid of edge h
        (padding empty layers; bounds [[lo], [hi]] for unbounded shapes); keep / void: shapes."""
        shape = _combine(shapes)
        bb = None if bounds is None else (list(bounds[0]), list(bounds[1]))
        d = G.voxelize(shape, h, padding=padding, bounds=bb, sub=sub, min_frac=min_frac, connectivity=connectivity,
                       thickness=thickness)
        return _with_regions(d, keep, void)

    def stl_domain(ctx: BuildContext, path: str, h: float, scale: float = 1.0, padding: int = 1, sub: int = 3,
                   min_frac: float = 0.05, connectivity: str | None = "largest", bounds: list | None = None,
                   keep: Any = None, void: Any = None) -> Domain:
        """Part = closed triangle surface (binary or ASCII STL; vertices x scale, in the unit of h; a relative path
        is taken from the working directory), voxelised with partial volumes."""
        from pathlib import Path
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"STL file not found: {p.resolve()}")
        V, F = G.read_stl(p)
        bb = None if bounds is None else (list(bounds[0]), list(bounds[1]))
        d = G.voxelize_mesh(np.asarray(V, float) * float(scale), F, h, padding=padding, sub=sub, min_frac=min_frac,
                            connectivity=connectivity, bounds=bb)
        return _with_regions(d, keep, void)

    def _with_regions(d: Domain, keep: Any, void: Any) -> Domain:
        if keep is None and void is None:
            return d
        km = d.keep | (_region_mask(keep, d) if keep is not None else False)
        vm = d.void | (_region_mask(void, d) if void is not None else False)
        out = Domain(d.shape, d.h, origin=d.origin, frac=d.frac, keep=km, void=vm, thickness=d.thickness)
        out.voxelization_info = d.voxelization_info
        return out

    regions = {"keep": "shape[]", "void": "shape[]"}
    register("domain", "csg", csg_domain, nested={"shapes": "shape[]", **regions},
             units={"h": length, "bounds": length, "thickness": length})
    register("domain", "stl", stl_domain, nested=regions, units={"h": length, "bounds": length})
