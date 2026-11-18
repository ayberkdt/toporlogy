"""Helpers shared by the component constructors and the post steps (moved from toporlogy.registry).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from toporlogy.boundary import Applied, Load
from toporlogy.configuration.registry import BuildContext, ConfigError, join


# ============================================================================================== shared helpers
def _axis(a: Any, dim: int) -> int:
    """Axis index from "x" / "y" / "z" (also "+x", "-z") or 0 / 1 / 2."""
    if isinstance(a, str) and a.strip().lower().lstrip("+-") in ("x", "y", "z"):
        k = "xyz".index(a.strip().lower().lstrip("+-"))
    elif isinstance(a, int | float) and not isinstance(a, bool) and float(a).is_integer():
        k = int(a)
    else:
        raise ValueError(f"axis must be x, y, z (or 0, 1, 2), got {a!r}")
    if not 0 <= k < dim:
        raise ValueError(f"axis {a!r} does not exist in {dim}-D")
    return k


def _bounds(d: Any) -> tuple[np.ndarray, np.ndarray]:
    lo = d.origin
    return lo, lo + d.h * np.asarray(d.shape, float)


def _coord(v: Any, k: int, d: Any, default: float | None = None) -> float:
    """Coordinate k from a number, "min", "max" or "mid" (of the domain box); null → ``default``."""
    lo, hi = _bounds(d)
    if v is None:
        if default is None:
            raise ValueError("a coordinate is null")
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "min":
            return float(lo[k])
        if s == "max":
            return float(hi[k])
        if s in ("mid", "centre", "center"):
            return float(0.5 * (lo[k] + hi[k]))
        try:
            return float(s)
        except ValueError:
            raise ValueError(f'coordinate must be a number, "min", "mid" or "max", got {v!r}') from None
    return float(v)


def _point(v: Any, d: Any, what: str = "point") -> np.ndarray:
    """A point of the domain's dimension; entries may be numbers, "min", "mid" or "max"."""
    if not isinstance(v, list | tuple) or len(v) != d.dim:
        raise ValueError(f"{what} needs {d.dim} coordinates, got {v!r}")
    return np.array([_coord(c, k, d) for k, c in enumerate(v)])


def _vector(v: Any, dim: int, what: str = "direction") -> np.ndarray:
    """Unit vector from an axis spec ("x", "-z", 1) or components."""
    if isinstance(v, str) or (isinstance(v, int | float) and not isinstance(v, bool)):
        sign = -1.0 if isinstance(v, str) and v.strip().startswith("-") else 1.0
        e = np.zeros(dim)
        e[_axis(v, dim)] = sign
        return e
    a = np.asarray(v, float).reshape(-1)
    if a.size != dim or not np.linalg.norm(a) > 0:
        raise ValueError(f"{what} must be an axis or a non-zero vector with {dim} components, got {v!r}")
    return a / np.linalg.norm(a)


def _nonempty(nodes: np.ndarray, what: str, d: Any) -> np.ndarray:
    if nodes.size == 0:
        lo, hi = _bounds(d)
        raise ValueError(f"{what} selects no node of the part (part box {np.round(lo, 6).tolist()} to "
                         f"{np.round(hi, 6).tolist()}, node spacing h = {d.h:g})")
    return nodes


def _directions(dofs: Any, dim: int) -> list[Any]:
    """Constrained directions: "all", "x", "xy", [x, z], [0, 2] (axes) or [[1, 1, 0]] (vectors)."""
    items: list[Any]
    if isinstance(dofs, str):
        s = dofs.strip().lower()
        if s == "all":
            return list(range(dim))
        items = list(s.replace(",", "").replace(" ", ""))
    elif isinstance(dofs, list | tuple):
        items = list(dofs)
    else:
        items = [dofs]
    axes = sorted({_axis(v, dim) for v in items if not isinstance(v, list | tuple)})
    vecs = [_vector(v, dim) for v in items if isinstance(v, list | tuple)]
    if not axes and not vecs:
        raise ValueError("dofs is empty")
    return [*axes, *[v.tolist() for v in vecs]]


def as_selector(v: Any) -> Any:
    """A built selector (bc.Selector, or node numbers from a node-array selector) as a bc.Selector."""
    from toporlogy.boundary import NodeIds, Selector
    if isinstance(v, Selector):
        return v
    a = np.asarray(v)
    if a.ndim == 1 and a.dtype.kind in "iu":
        return NodeIds(a.astype(np.int64))
    raise ValueError(f"expected a selector, got {type(v).__name__}")


def checked_selector(ctx: BuildContext, v: Any, key: str = "where") -> Any:
    """:func:`as_selector`, and the selection must not be empty (reported at ``<component path>.<key>``)."""
    from toporlogy.boundary import EmptySelectionError
    s = as_selector(v)
    try:
        s.select(ctx.domain)
    except EmptySelectionError as e:
        raise ConfigError(f"selector: {e}", join(ctx.path, key)) from None
    return s


def _case_weights(ctx: BuildContext, weights: list | None) -> list[float] | None:
    w = ctx.case_weights if weights is None else [float(v) for v in weights]
    if len(w) != ctx.n_cases:
        raise ValueError(f"{len(w)} weights for {ctx.n_cases} load cases")
    return None if all(v == 1.0 for v in w) else w


def _region_mask(shapes: Sequence[Any], d: Any) -> np.ndarray:
    """Elements of ``d`` more than half covered by the CSG of ``shapes`` (list of built shapes with ``op``)."""
    from toporlogy.geometry import coverage
    return coverage(_combine(shapes), d) > 0.5


def _combine(shapes: Sequence[Any]) -> Any:
    from toporlogy.geometry import csg
    if not shapes:
        raise ValueError("at least one shape is required")
    ops = [(getattr(s, "op", "add"), s) for s in shapes]
    if ops[0][0] != "add":
        raise ValueError(f'the first shape must have op "add", got "{ops[0][0]}"')
    return csg(ops)


class SpreadForce(Load):
    """Total force spread equally over the selected nodes (the ``force`` load; not area-weighted)."""

    kind = "spread"

    def __init__(self, where: Any, force: Sequence[float], name: str | None = None) -> None:
        self.where, self.F = where, np.asarray(force, float)
        self.name = name or f"Force {where!r}"

    def apply(self, info: Any) -> Applied:
        n = self.where.select(info)
        if self.F.size != info.dim:
            raise ValueError(f"{self.name}: force needs {info.dim} components")
        f = np.broadcast_to(self.F / len(n), (len(n), info.dim)).copy()
        return Applied(self.name, self.kind, n, f, self.F.copy(), info.X[n].mean(axis=0), None, [])


def _bores(ctx: BuildContext, min_diameter: float | None, max_roundness: float, min_length: float | None) -> list:
    from toporlogy.geometry import find_bores
    key = ("bores", id(ctx.domain), min_diameter, max_roundness, min_length)
    if key not in ctx.extra:
        ctx.extra[key] = find_bores(ctx.domain, min_diameter, max_roundness, min_length)
        ctx.extra["bores_found"] = ctx.extra[key]
    return ctx.extra[key]


def _describe_bores(bores: Sequence[Any]) -> str:
    def f(v: Any) -> str:
        return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"
    return "; ".join(f"{b.id} r {b.radius:.4g} at {f(b.center)}" + (f" axis {'xyz'[int(np.argmax(np.abs(b.axis)))]}"
                     if b.axis is not None else "") for b in bores) or "none"


def find_bore(ctx: BuildContext, index: int | None, near: list | None, min_diameter: float | None = None,
              max_roundness: float = 0.35, min_length: float | None = None) -> Any:
    """An auto-detected bore: ``index`` (1-based, = id "B<index>"), ``near`` (point: nearest axis), or the only one."""
    bores = _bores(ctx, min_diameter, max_roundness, min_length)
    if not bores:
        raise ValueError("no bore found in the part (find_bores: enclosed round holes, diameter >= 3 h)")
    if index is not None:
        if not 1 <= int(index) <= len(bores):
            raise ValueError(f"bore index {index} out of range; found: {_describe_bores(bores)}")
        return bores[int(index) - 1]
    if near is not None:
        p = _point(near, ctx.domain, "near")

        def dist(b: Any) -> float:
            q = p - b.center
            if b.axis is not None:
                q = q - (q @ b.axis) * b.axis
            return float(np.linalg.norm(q))
        return min(bores, key=dist)
    if len(bores) > 1:
        raise ValueError(f"{len(bores)} bores found, give index or near: {_describe_bores(bores)}")
    return bores[0]


def _is_cylinder(s: Any, what: str) -> Any:
    from toporlogy.boundary import OnCylinder
    if not isinstance(s, OnCylinder):
        raise ValueError(f"{what} needs a bore / cylinder selector (type bore or on_cylinder), got {s!r}")
    return s


def _robust_set(ctx: BuildContext) -> Any:
    from toporlogy.optimization.robust import RobustSet
    if ctx.robust is None:
        ctx.robust = RobustSet()
        ctx.extra.setdefault("notes", []).append("robust: no robust block; default thresholds eta_e 0.75, eta_d 0.25")
    return ctx.robust


def _field_source(ctx: BuildContext, source: str | None) -> Any:
    if source in (None, "xphys", "nominal"):
        return None
    if source not in ("eroded", "intermediate", "dilated"):
        raise ValueError(f'source: xphys | eroded | intermediate | dilated, got "{source}"')
    return _robust_set(ctx).field(source)


def _interfaces(ctx: BuildContext) -> dict[str, np.ndarray]:
    if not ctx.interfaces:
        raise ValueError("no interface regions (connectivity.interfaces, or supports and loads)")
    return ctx.interfaces


def _post_state(ctx: BuildContext) -> dict[str, Any]:
    return ctx.extra.setdefault("post_state", {})


def _design(state: dict[str, Any], res: Any) -> tuple[np.ndarray, str]:
    """x̄ for the geometry steps and its label: the repaired variant of ``res`` when a connectivity_gate repaired
    this result ("repaired"), else ``res.xphys`` ("optimized"). ``res`` itself is never modified, so its x, xphys,
    objective, constraints and U keep describing the optimised design."""
    d = state.get("design")
    if d is not None and d["source"] is res:
        return d["xphys"], d["label"]
    return res.xphys, "optimized"


def _reevaluate(study: Any, xphys: np.ndarray, run: dict[str, Any]) -> dict[str, Any]:
    """Objective and constraints of a repaired density (Study.evaluate_density), with the run_setup feasibility
    rule. Constraints that need the design chain are listed with ``value: None`` and the reason; ``feasible`` is
    False if an evaluated constraint is violated, None if none is violated but some could not be evaluated."""
    ev = study.evaluate_density(xphys)
    ftol = float(run.get("feasibility_tol", 1e-3))
    cons = []
    for c in study.constraints:
        if c.key in ev.values:
            v = ev.values[c.key]
            cons.append({"name": c.key, "value": v, "upper": c.upper,
                         "satisfied": bool(v <= c.upper + ftol * max(abs(c.upper), 1e-12))})
        else:
            cons.append({"name": c.key, "value": None, "upper": c.upper, "satisfied": None,
                         "not_evaluated": ev.skipped[c.key]})
    obj = study.objective.name
    feasible: bool | None = True
    if any(c["satisfied"] is False for c in cons):
        feasible = False
    elif any(c["satisfied"] is None for c in cons):
        feasible = None
    return {"objective": ev.values.get(obj), "objective_name": obj, "constraints": cons, "feasible": feasible,
            **({"objective_not_evaluated": ev.skipped[obj]} if obj in ev.skipped else {})}


def _rel(files: dict[str, str], folder: Any) -> dict[str, str]:
    from pathlib import Path
    out = {}
    for k, v in files.items():
        try:
            out[k] = str(Path(v).relative_to(Path(folder)))
        except ValueError:
            out[k] = str(v)
    return out
