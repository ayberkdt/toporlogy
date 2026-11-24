"""Core components: domain box, selectors plane/box/point, fixed support, force, chain links, interpolations,
solver, physics, compliance/volume, OC/MMA, beta continuation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from toporlogy.configuration.helpers import (
    SpreadForce, _axis, _case_weights, _coord, _directions, _nonempty, _region_mask, checked_selector,
)
from toporlogy.configuration.registry import BuildContext, REGISTRY, join, register


# ============================================================================================== core components
# Imported here (not at the top) so that toporlogy's import-order guard (PARDISO before torch) runs first.
def register_core() -> None:
    from toporlogy.design.chain import DensityFilter, Embed, HeavisideProjection, Passive
    from toporlogy.core.domain import Domain
    from toporlogy.design.interp import RAMP, SIMP
    from toporlogy.optimizers.oc import OC
    from toporlogy.optimizers.schedule import BetaContinuation
    from toporlogy.optimization.responses import Compliance, Volume
    from toporlogy.solvers.direct import DirectSolver

    # ---------------------------------------------------------------- domain
    def box(ctx: BuildContext, shape: list, h: float = 1.0, origin: list | None = None, thickness: float = 1.0,
            keep: Any = None, void: Any = None) -> Domain:
        """Full rectangular grid: shape (nx, ny) or (nx, ny, nz) elements of edge h; keep / void: shapes
        (elements more than half covered are frozen solid / empty)."""
        if len(shape) not in (2, 3):
            raise ValueError(f"shape needs 2 or 3 entries, got {len(shape)}")
        try:
            shp = [int(s) for s in shape]
        except (TypeError, ValueError):
            raise ValueError("shape entries must be integers") from None
        d = Domain(shp, h=float(h), origin=None if origin is None else [float(o) for o in origin],
                   thickness=float(thickness))
        if keep is None and void is None:
            return d
        km, vm = (None if s is None else _region_mask(s, d) for s in (keep, void))
        return Domain(shp, h=d.h, origin=d.origin, keep=km, void=vm, thickness=d.thickness)

    register("domain", "box", box, nested={"keep": "shape[]", "void": "shape[]"},
             units={"h": "length", "origin": "length", "thickness": "length"})

    # ---------------------------------------------------------------- selectors (node sets of the part)
    def plane(ctx: BuildContext, axis: Any, at: Any) -> np.ndarray:
        """Nodes on the plane (3-D) or line (2-D) coordinate[axis] = at; at: number, "min" or "max"."""
        d = ctx.domain
        k = _axis(axis, d.dim)
        c = _coord(at, k, d)
        tol = 1e-6 * d.h
        return _nonempty(d.nodes_where(lambda X: np.abs(X[:, k] - c) <= tol), f"plane {'xyz'[k]} = {c:g}", d)

    def box_sel(ctx: BuildContext, min: list | None = None, max: list | None = None) -> np.ndarray:  # noqa: A002
        """Nodes inside the closed box [min, max]; entries may be numbers, "min", "max" or null (unbounded)."""
        d = ctx.domain
        lo = [-np.inf] * d.dim if min is None else min
        hi = [np.inf] * d.dim if max is None else max
        if len(lo) != d.dim or len(hi) != d.dim:
            raise ValueError(f"min and max need {d.dim} entries")
        a = np.array([_coord(v, k, d, -np.inf) for k, v in enumerate(lo)])
        b = np.array([_coord(v, k, d, np.inf) for k, v in enumerate(hi)])
        tol = 1e-6 * d.h
        return _nonempty(d.nodes_where(lambda x: np.all((x >= a - tol) & (x <= b + tol), axis=1)),
                         f"box {a.tolist()} to {b.tolist()}", d)

    def point(ctx: BuildContext, at: list) -> np.ndarray:
        """The part node nearest to ``at`` (entries: numbers, "min" or "max"); must lie within one element edge."""
        d = ctx.domain
        if len(at) != d.dim:
            raise ValueError(f"at needs {d.dim} entries")
        p = np.array([_coord(v, k, d) for k, v in enumerate(at)])
        cand = d.nodes_where(lambda X: np.ones(len(X), bool))
        dist = np.linalg.norm(d.node_coords()[cand] - p, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > d.h * (1 + 1e-9):
            raise ValueError(f"no part node within h = {d.h:g} of {p.tolist()} (nearest at {dist[i]:g})")
        return cand[i:i + 1]

    register("selector", "plane", plane, units={"at": "length"})
    register("selector", "box", box_sel, units={"min": "length", "max": "length"})
    register("selector", "point", point, units={"at": "length"})

    # ---------------------------------------------------------------- supports and loads
    from toporlogy.boundary import Fixed

    def fixed(ctx: BuildContext, where: Any, dofs: Any = "all", name: str | None = None) -> Fixed:
        """Zero displacement of the selected nodes in the directions ``dofs`` ("all", "x", "xy", [x, z], or
        vectors [[1, 1, 0]])."""
        return Fixed(checked_selector(ctx, where), _directions(dofs, ctx.domain.dim), name=name)

    def force(ctx: BuildContext, where: Any, force: list, name: str | None = None) -> Any:
        """Total force vector (Fx, Fy[, Fz]) spread equally over the selected nodes."""
        if len(force) != ctx.domain.dim:
            raise ValueError(f"force needs {ctx.domain.dim} components, got {len(force)}")
        return SpreadForce(checked_selector(ctx, where), [float(f) for f in force], name)

    register("support", "fixed", fixed, nested={"where": "selector"}, types={"dofs": None})
    register("load", "force", force, nested={"where": "selector"}, units={"force": "force (total)"})

    # ---------------------------------------------------------------- chain links
    register("link", "embed", Embed, context=("domain",))
    register("link", "density_filter", DensityFilter, context=("domain",), types={"rmin": float},
             units={"rmin": "elements"})
    register("link", "heaviside_projection", HeavisideProjection)
    register("link", "passive", Passive, context=("domain",))

    # ---------------------------------------------------------------- interpolation (E0 from material)
    def simp(ctx: BuildContext, penal: float = 3.0, emin_ratio: float = 1e-6) -> SIMP:
        """Modified SIMP E = (Emin + x^p (E0 - Emin)) frac; E0 from material.E0."""
        return SIMP(ctx.domain, E0=ctx.material["E0"], penal=penal, emin_ratio=emin_ratio)

    def ramp(ctx: BuildContext, q: float = 8.0, emin_ratio: float = 1e-6) -> RAMP:
        """RAMP (experimental); E0 from material.E0."""
        return RAMP(ctx.domain, E0=ctx.material["E0"], q=q, emin_ratio=emin_ratio)

    register("interpolation", "simp", simp)
    register("interpolation", "ramp", ramp)

    # ---------------------------------------------------------------- solver and physics
    register("solver", "direct", DirectSolver)

    def elasticity(ctx: BuildContext, element: str | None = None, solver: Any = "direct") -> Any:
        """Static linear elasticity; nu from material.nu; element null = H8 (3-D) / Q4 (2-D). Oblique supports and
        springs use bc.ConstrainedElasticity, load cases with their own supports bc.PerCaseSupports."""
        spec = (ctx.config.get("physics") or {}).get("solver")
        made = [0]

        def factory() -> Any:              # first physics gets the built solver, further ones a fresh one each
            made[0] += 1
            if made[0] == 1 or not isinstance(spec, dict):
                return solver
            return REGISTRY.build("solver", spec, ctx, join(ctx.path, "solver"))

        return ctx.bc.physics(nu=ctx.material["nu"], element=element, solver_factory=factory)

    register("physics", "elasticity", elasticity, nested={"solver": "solver"})

    # ---------------------------------------------------------------- responses
    def compliance(ctx: BuildContext, weights: list | None = None) -> Compliance:
        """Weighted compliance sum_c w_c F_c.U_c; weights null = the load-case weights."""
        return Compliance(_case_weights(ctx, weights))

    register("response", "compliance", compliance)
    register("response", "volume", Volume)

    # ---------------------------------------------------------------- optimizers
    def oc(ctx: BuildContext, move: float = 0.2, l1: float = 0.0, l2: float = 1e9, bisection_tol: float = 1e-4,
           max_bisect: int = 80, dv_scale: float | None = None) -> OC:
        """Optimality criteria (one constraint); dv_scale null = number of active elements (classic bisection)."""
        return OC(move=move, l1=l1, l2=l2, bisection_tol=bisection_tol, max_bisect=max_bisect,
                  dv_scale=float(ctx.domain.n_active) if dv_scale is None else dv_scale)

    register("optimizer", "oc", oc)
    register("optimizer", "mma", "toporlogy.optimizers.mma:MMA")
    register("optimizer", "gcmma", "toporlogy.optimizers.gcmma:GCMMA")
    register("optimizer", "simpl", "toporlogy.optimizers.simpl:SiMPL")

    # ---------------------------------------------------------------- schedules
    def beta_continuation(ctx: BuildContext, beta0: float = 1.0, beta_max: float = 16.0, every: int = 25,
                          tol: float | None = None) -> BetaContinuation:
        """beta doubles every `every` iterations or when change <= tol (null = run.tol)."""
        return BetaContinuation(ctx.chain, beta0, beta_max, every,
                                float(ctx.run.get("tol", 0.01)) if tol is None else tol)

    register("schedule", "beta_continuation", beta_continuation)
