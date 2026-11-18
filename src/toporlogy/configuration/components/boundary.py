"""Component constructors: boundary (moved from toporlogy.registry).
"""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np

from toporlogy.configuration.helpers import (
    _axis, _coord, _is_cylinder, _point, _vector, as_selector, checked_selector, find_bore,
)
from toporlogy.configuration.registry import BuildContext, register


def register_boundary() -> None:
    """Register the boundary components (moved from registry._register_features)."""
    from toporlogy import boundary as B

    length = "length"

    # ---------------------------------------------------------------- selectors (bc)
    def all_nodes(ctx: BuildContext) -> Any:
        """Every node of the part."""
        return B.AllNodes()

    def surface(ctx: BuildContext) -> Any:
        """Nodes on the part surface."""
        return B.SurfaceNodes()

    def in_box(ctx: BuildContext, lo: list | None = None, hi: list | None = None, surface: bool = False,
               tol: float | None = None) -> Any:
        """Nodes with lo <= x <= hi; entries numbers, "min", "mid", "max" or null (unbounded); surface: only
        surface nodes."""
        d = ctx.domain
        lo_ = [None] * d.dim if lo is None else lo
        hi_ = [None] * d.dim if hi is None else hi
        if len(lo_) != d.dim or len(hi_) != d.dim:
            raise ValueError(f"lo and hi need {d.dim} entries")
        a = [_coord(v, k, d, -np.inf) for k, v in enumerate(lo_)]
        b = [_coord(v, k, d, np.inf) for k, v in enumerate(hi_)]
        return B.InBox(a, b, surface=surface, tol=tol)

    def _plane(ctx: BuildContext, axis: Any, at: Any, point: list | None, normal: Any) -> tuple[np.ndarray, Any]:
        d = ctx.domain
        if axis is not None:
            if point is not None or normal is not None:
                raise ValueError("give axis + at, or point + normal")
            k = _axis(axis, d.dim)
            p = d.origin.astype(float).copy()
            p[k] = _coord(at if at is not None else "min", k, d)
            return p, k
        if point is None or normal is None:
            raise ValueError("give axis + at, or point + normal")
        return _point(point, d), _vector(normal, d.dim, "normal")

    def on_plane(ctx: BuildContext, axis: Any = None, at: Any = None, point: list | None = None,
                 normal: Any = None, tol: float | None = None, facing: str | None = "any",
                 max_angle: float = 60.0) -> Any:
        """Surface nodes on a plane (axis + at, or point + normal) within tol (default h/2) whose outward normal is
        (anti)parallel to it ("any"), along it ("out"), against it ("in") or not filtered (null)."""
        p, n = _plane(ctx, axis, at, point, normal)
        return B.OnPlane(p, n, tol=tol, facing=facing, max_angle=max_angle)

    def half_space(ctx: BuildContext, point: list, normal: Any, surface: bool = False,
                   tol: float | None = None) -> Any:
        """Nodes with (x - point).normal <= tol."""
        return B.HalfSpaceNodes(_point(point, ctx.domain), _vector(normal, ctx.domain.dim, "normal"), surface, tol)

    def on_cylinder(ctx: BuildContext, p0: list, radius: float, p1: list | None = None, band: Any = None,
                    facing: str | None = "inward", max_angle: float = 60.0, extend: float = 0.0) -> Any:
        """Surface nodes on a cylinder face (3-D axis p0 -> p1; 2-D circle about p0): a bore ("inward") or a boss
        ("outward"); band = radial tolerance (default 1.5 h inside / 0.75 h outside)."""
        d = ctx.domain
        q1 = None if p1 is None else _point(p1, d, "p1")
        return B.OnCylinder(_point(p0, d, "p0"), q1, radius, band=band, facing=facing, max_angle=max_angle,
                            extend=extend)

    def bore(ctx: BuildContext, index: int | None = None, near: list | None = None, band: Any = None,
             facing: str | None = "inward", max_angle: float = 60.0, extend: float = 0.0,
             min_diameter: float | None = None) -> Any:
        """Wall of an auto-detected bore (find_bores): index 1-based (B1, B2 ...), near = a point (nearest axis),
        or the only bore of the part."""
        b = find_bore(ctx, index, near, min_diameter)
        return B.OnCylinder.from_bore(b, facing=facing, band=band, max_angle=max_angle, extend=extend)

    def in_sphere(ctx: BuildContext, center: list, radius: float, surface: bool = False) -> Any:
        """Nodes within radius of center."""
        return B.InSphere(_point(center, ctx.domain, "center"), radius, surface)

    def in_shape(ctx: BuildContext, shape: Any, surface: bool = False, tol: float | None = None) -> Any:
        """Nodes inside a shape (sdf <= tol)."""
        return B.InShape(shape, surface, tol)

    def at_points(ctx: BuildContext, points: list, tol: float | None = None) -> Any:
        """Nodes at the given points (each must hit a node within tol, default 1e-6 h)."""
        return B.AtPoints([_point(p, ctx.domain) for p in points], tol)

    def nearest(ctx: BuildContext, points: list, k: int = 1, surface: bool = True) -> Any:
        """The k nearest (surface) nodes to each point."""
        return B.Nearest([_point(p, ctx.domain) for p in points], k, surface)

    def node_ids(ctx: BuildContext, ids: list) -> Any:
        """Explicit node numbers (advanced)."""
        return B.NodeIds([int(i) for i in ids])

    def _fold(of: Sequence[Any], op: str) -> Any:
        if len(of) < 2:
            raise ValueError("of needs at least two selectors")
        out = as_selector(of[0])
        for s in of[1:]:
            t = as_selector(s)
            out = out | t if op == "|" else (out & t if op == "&" else out - t)
        return out

    def union(ctx: BuildContext, of: Any) -> Any:
        """Nodes selected by any of the selectors."""
        return _fold(of, "|")

    def intersect(ctx: BuildContext, of: Any) -> Any:
        """Nodes selected by all selectors."""
        return _fold(of, "&")

    def subtract(ctx: BuildContext, of: Any) -> Any:
        """Nodes of the first selector minus those of the others."""
        return _fold(of, "-")

    pts = {"lo": length, "hi": length, "point": length, "p0": length, "p1": length, "center": length,
           "radius": length, "near": length, "points": length, "tol": length, "band": length, "at": length,
           "extend": length, "min_diameter": length, "max_angle": "deg"}
    for name, fn, nest in (("all", all_nodes, {}), ("surface", surface, {}), ("in_box", in_box, {}),
                           ("on_plane", on_plane, {}), ("half_space", half_space, {}),
                           ("on_cylinder", on_cylinder, {}), ("bore", bore, {}), ("in_sphere", in_sphere, {}),
                           ("in_shape", in_shape, {"shape": "shape"}), ("at_points", at_points, {}),
                           ("nearest", nearest, {}), ("node_ids", node_ids, {}),
                           ("union", union, {"of": "selector[]"}), ("intersect", intersect, {"of": "selector[]"}),
                           ("subtract", subtract, {"of": "selector[]"})):
        sig = inspect.signature(fn).parameters
        register("selector", name, fn, nested=nest, types={k: None for k in ("axis", "at", "normal", "band")
                                                          if k in sig},
                 units={k: u for k, u in pts.items() if k in sig})

    # ---------------------------------------------------------------- supports (bc)
    def roller(ctx: BuildContext, where: Any, normal: Any, name: str | None = None) -> Any:
        """Zero displacement along normal (axis or vector) at the selected nodes; free in the tangent plane."""
        return B.Roller(checked_selector(ctx, where), _vector(normal, ctx.domain.dim, "normal"), name)

    def symmetry(ctx: BuildContext, axis: Any = None, at: Any = None, point: list | None = None, normal: Any = None,
                 tol: float | None = None, name: str | None = None) -> Any:
        """Symmetry plane (axis + at, or point + normal): zero normal displacement of the surface nodes on it."""
        p, n = _plane(ctx, axis, at, point, normal)
        s = B.Symmetry(p, n, tol=tol, name=name)
        checked_selector(ctx, s.where, "at" if axis is not None else "point")
        return s

    def pinned_bore(ctx: BuildContext, bore: Any, axial: bool = False, tangential: bool = False,
                    name: str | None = None) -> Any:
        """Pin in a bore: radial displacement zero, rotation about the pin free (axial / tangential: also fix those)."""
        sel = _is_cylinder(checked_selector(ctx, bore, "bore"), "pinned_bore")
        return B.PinnedBore(sel, axial=axial, tangential=tangential, name=name)

    def elastic(ctx: BuildContext, where: Any, k: float = 0.0, direction: Any = "normal", total: float | None = None,
                name: str | None = None) -> Any:
        """Springs to ground: stiffness per area k x nodal area ("normal", "all" or a vector direction), or a total
        stiffness shared by area."""
        if not k > 0 and total is None:
            raise ValueError("give k > 0 (per area) or total")
        dirn = direction if isinstance(direction, str) and direction in ("normal", "all") else \
            _vector(direction, ctx.domain.dim).tolist()
        return B.ElasticSupport(checked_selector(ctx, where), k=k, direction=dirn, total=total, name=name)

    register("support", "roller", roller, nested={"where": "selector"}, types={"normal": None})
    register("support", "symmetry", symmetry, types={"axis": None, "at": None, "normal": None},
             units={"at": length, "point": length, "tol": length})
    register("support", "pinned_bore", pinned_bore, nested={"bore": "selector"})
    register("support", "elastic", elastic, nested={"where": "selector"}, types={"direction": None},
             units={"k": "force/length^3", "total": "force/length"})

    # ---------------------------------------------------------------- loads (bc)
    def point_load(ctx: BuildContext, at: list, force: list, radius: float | None = None,
                   name: str | None = None) -> Any:
        """Force at a point, spread over the surface nodes within radius (default 1.5 h) with zero moment about it."""
        return B.PointLoad(_point(at, ctx.domain, "at"), force, radius, name)

    def surface_force(ctx: BuildContext, where: Any, force: list, name: str | None = None) -> Any:
        """Total force over the selected surface, shared by geometric nodal area (uniform traction)."""
        return B.SurfaceForce(checked_selector(ctx, where), force, name)

    def traction(ctx: BuildContext, where: Any, traction: list, geometric: bool = True,
                 name: str | None = None) -> Any:
        """Traction vector (force per area) on the complete surface faces of the selection."""
        return B.Traction(checked_selector(ctx, where), traction, geometric, name)

    def pressure(ctx: BuildContext, where: Any, p: float, normals: str = "faces", name: str | None = None) -> Any:
        """Pressure p (positive pushes into the part) on the complete surface faces of the selection."""
        return B.Pressure(checked_selector(ctx, where), p, normals, name)

    def bearing(ctx: BuildContext, bore: Any, force: list, mode: str = "parallel", on: str = "surface",
                name: str | None = None) -> Any:
        """Pin load on a bore: radial part over the loaded half with weight max(0, cos theta) (projected area)."""
        sel = _is_cylinder(checked_selector(ctx, bore, "bore"), "bearing")
        return B.BearingLoad(sel, force, mode=mode, on=on, name=name)

    def remote_force(ctx: BuildContext, where: Any, force: list, point: list, moment: Any = None,
                     weights: str = "area", name: str | None = None) -> Any:
        """Force (and moment) acting at point, carried by the selected nodes (rigid-body least squares)."""
        return B.RemoteForce(checked_selector(ctx, where), force, _point(point, ctx.domain), moment, weights, name)

    def remote_moment(ctx: BuildContext, where: Any, moment: Any, point: list | None = None, weights: str = "area",
                      name: str | None = None) -> Any:
        """Pure moment about point (default: area centroid of the selection); 2-D: scalar, 3-D: (Mx, My, Mz)."""
        p = None if point is None else _point(point, ctx.domain)
        return B.RemoteMoment(checked_selector(ctx, where), moment, p, weights, name)

    def torque(ctx: BuildContext, bore: Any, torque: float, name: str | None = None) -> Any:
        """Torque about a bore axis (right-hand rule; 2-D: about the out-of-plane axis), zero net force."""
        sel = _is_cylinder(checked_selector(ctx, bore, "bore"), "torque")
        return B.Torque(sel, torque, name)

    def body_force(ctx: BuildContext, acceleration: list, density: float | None = None,
                   name: str | None = None) -> Any:
        """Design-independent body force density x acceleration on the whole part (density null = material.rho)."""
        rho = ctx.material.get("rho", 1.0) if density is None else density
        return B.BodyForce(acceleration, rho, name)

    w = {"where": "selector"}
    register("load", "point", point_load, units={"at": length, "force": "force", "radius": length})
    register("load", "surface_force", surface_force, nested=w, units={"force": "force (total)"})
    register("load", "traction", traction, nested=w, units={"traction": "force/length^2"})
    register("load", "pressure", pressure, nested=w, units={"p": "force/length^2"})
    register("load", "bearing", bearing, nested={"bore": "selector"}, units={"force": "force (total)"})
    register("load", "remote_force", remote_force, nested=w, types={"moment": None},
             units={"force": "force", "point": length, "moment": "force*length"})
    register("load", "remote_moment", remote_moment, nested=w, types={"moment": None},
             units={"moment": "force*length", "point": length})
    register("load", "torque", torque, nested={"bore": "selector"}, units={"torque": "force*length"})
    register("load", "body_force", body_force, units={"acceleration": "length/time^2", "density": "mass/length^3"})
