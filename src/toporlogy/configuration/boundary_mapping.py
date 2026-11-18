"""Supports and loads as callables for the T10 re-analysis, with their fidelity (moved from toporlogy.registry).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bc_mapping(bcs: Any) -> dict[str, Any]:
    """Supports ``[(select, dirs)]`` and load cases ``{name: [(select, force)]}`` for
    :func:`toporlogy.post.reverify` from a :class:`~toporlogy.bc.BoundaryConditions` (the bc selectors as callables
    ``select(X (n, dim)) -> bool`` in physical coordinates), and how faithful that translation is.

    Returns ``{"supports", "cases", "items", "notes", "fidelity"}``. ``items``: one ``{"name", "fidelity",
    "note"}`` per support or load that is not carried over as it is: "approximate" (carried over as a different
    condition: pinned bore as fixed, oblique as fixed in all directions, bearing / remote force as an area-spread
    resultant, first case's supports for every case) or "unsupported" (left out: springs, body forces, pure
    moments, other support types). ``fidelity``: "unsupported" if any item is, else "approximate" if any item is,
    else "equivalent" (every support and load mapped by type with no known approximation; the nodal distribution
    on the tetrahedral mesh still differs from the voxel grid's). Only an "equivalent" re-analysis is the
    optimisation's problem."""
    from toporlogy import boundary as B
    info, h = bcs.info, bcs.domain.h
    mapped: list[dict[str, str]] = []

    def note(name: str, fidelity: str, text: str) -> None:
        mapped.append({"name": str(name), "fidelity": fidelity, "note": text})
    sup_list = bcs.cases[0].supports if bcs.cases[0].supports is not None else bcs.supports
    if any(c.supports is not None for c in bcs.cases):
        note("per-case supports", "approximate", "the supports of the first case are used for every case")
    supports = []
    for s in sup_list:
        if isinstance(s, B.ElasticSupport):
            note(s.name, "unsupported", "springs are not re-verified (left free)")
            continue
        if isinstance(s, B.PinnedBore):
            supports.append((selector_callable(s.sel, info, h), (0, 1, 2)))
            note(s.name, "approximate", "pinned bore re-verified as a fixed bore")
            continue
        if isinstance(s, B.Fixed):
            rows = np.eye(info.dim) if s.dirs is None else np.array(
                [np.eye(info.dim)[int(r)] if np.isscalar(r) else np.asarray(r, float) / np.linalg.norm(r)
                 for r in s.dirs])
            ax = [int(np.argmax(np.abs(r))) for r in rows if np.isclose(np.abs(r).max(), 1.0)]
            if len(ax) != len(rows):
                note(s.name, "approximate", "oblique direction(s) re-verified as fixed in all directions")
                ax = [0, 1, 2]
            supports.append((selector_callable(s.where, info, h), tuple(sorted(set(ax)))))
            continue
        note(s.name, "unsupported", f"support type {type(s).__name__} not re-verified")
    if not supports:
        raise ValueError("reverify: no support can be re-verified")
    cases: dict[str, list[Any]] = {}
    for c in bcs.cases:
        items = []
        for ld, a in zip(c.loads, bcs.applied[c.name], strict=True):
            F = np.asarray(a.force, float)
            if isinstance(ld, B.BodyForce) or not np.linalg.norm(F) > 0:
                note(a.name, "unsupported", f"{'body force' if isinstance(ld, B.BodyForce) else 'pure moment'} not "
                                             "re-verified")
                continue
            if isinstance(ld, B.PointLoad):
                sel = B.InSphere(ld.point, 1.5 * h if ld.radius is None else float(ld.radius))
            else:
                sel = getattr(ld, "where", None) or getattr(ld, "sel", None)
            if sel is None:
                note(a.name, "unsupported", "no selection, not re-verified")
                continue
            if isinstance(ld, B.BearingLoad | B.RemoteForce):
                note(a.name, "approximate", "re-verified as the resultant spread by area (no cosine / moment)")
            items.append((selector_callable(sel, info, h), F))
        if items:
            cases[c.name] = items
    if not cases:
        raise ValueError("reverify: no load can be re-verified")
    levels = {it["fidelity"] for it in mapped}
    fidelity = "unsupported" if "unsupported" in levels else "approximate" if levels else "equivalent"
    return {"supports": supports, "cases": cases, "items": mapped,
            "notes": [f"{it['name']}: {it['note']}" for it in mapped], "fidelity": fidelity}


def bc_callables(bcs: Any) -> tuple[list[Any], dict[str, list[Any]], list[str]]:
    """``(supports, cases, notes)`` of :func:`bc_mapping` (kept for callers that need no fidelity)."""
    m = bc_mapping(bcs)
    return m["supports"], m["cases"], m["notes"]


def selector_callable(sel: Any, info: Any, h: float) -> Callable[[np.ndarray], np.ndarray]:
    """``select(X) -> bool`` in physical coordinates for a bc selector: geometric selectors keep their geometry with a
    tolerance of h/2 (the finished surface moves by up to about that), others select points within h of the nodes
    they pick on the grid."""
    from scipy.spatial import cKDTree

    from toporlogy import boundary as B
    t = 0.5 * h
    if isinstance(sel, B.InBox):
        lo, hi = sel.lo, sel.hi
        return lambda X: np.all((X - lo >= -t) & (hi - X >= -t), axis=1)        # bounds may be infinite
    if isinstance(sel, B.OnPlane):
        p, n = sel.point, sel.normal
        tol = max(t, 0.5 * h if sel.tol is None else float(sel.tol))
        return lambda X: np.abs((X - p) @ n) <= tol
    if isinstance(sel, B.HalfSpaceNodes):
        p, n = sel.point, sel.normal
        return lambda X: (X - p) @ n <= t
    if isinstance(sel, B.InSphere):
        c, r = sel.center, sel.radius
        return lambda X: np.linalg.norm(X - c, axis=1) <= r + t
    if isinstance(sel, B.InShape):
        shp = sel.shape
        return lambda X: shp.sdf(X) <= t
    if isinstance(sel, B.OnCylinder):
        bi, bo = sel.bands(h)

        def cyl(X: np.ndarray) -> np.ndarray:
            tt, r, _ = sel.coords(X)
            ok = (r >= sel.radius - bi) & (r <= sel.radius + bo)
            if sel.axis is not None:
                ok &= (tt >= -sel.extend - t) & (tt <= sel.length + sel.extend + t)
            return ok
        return cyl
    if isinstance(sel, B.AllNodes | B.SurfaceNodes):
        return lambda X: np.ones(len(X), bool)
    op, a, b = getattr(sel, "op", None), getattr(sel, "a", None), getattr(sel, "b", None)
    if op in ("|", "&", "-") and isinstance(a, B.Selector) and isinstance(b, B.Selector):
        fa, fb = selector_callable(a, info, h), selector_callable(b, info, h)
        if op == "|":
            return lambda X: fa(X) | fb(X)
        if op == "&":
            return lambda X: fa(X) & fb(X)
        return lambda X: fa(X) & ~fb(X)
    nodes = sel.select(info)
    tree = cKDTree(info.X[nodes])
    return lambda X: np.isfinite(tree.query(X, distance_upper_bound=h * (1 + 1e-9))[0])
