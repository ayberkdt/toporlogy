"""Post steps (registered as kind ``post``) and their lazy constructors.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from toporlogy.configuration.boundary_mapping import bc_mapping
from toporlogy.configuration.helpers import _axis, _coord, _design, _interfaces, _post_state, _reevaluate, _rel
from toporlogy.configuration.registry import BuildContext, ConfigError, REGISTRY, join, register


def _gate_dict(rep: Any) -> dict[str, Any]:
    m = rep.min_neck
    return {"n_bodies": int(rep.n_bodies), "connected": bool(rep.connected), "fragile": bool(rep.fragile),
            "reasons": list(rep.reasons), "unattached": list(rep.unattached),
            "interface_bodies": {k: list(v) for k, v in rep.interface_bodies.items()},
            "body_volume": [float(v) for v in rep.body_volume[:10]], "enclosed_voids": len(rep.enclosed_voids),
            "min_neck": None if m is None else {"between": [m.a, m.b], "cells": m.cells, "area": m.area},
            "checks": [{"label": c.label, "n_bodies": c.n_bodies, "connected": c.connected} for c in rep.checks],
            "text": rep.summary()}


# ============================================================================================== post steps (lazy)
# The post package (and its compiled kernels) is imported only when one of these steps is listed or used; their
# parameters are the fields of PostParams / CheckThresholds (signature built on first access, PEP 562).
_LAZY: dict[str, Any] = {}


def _post_shape(s: Any) -> Any:
    """geometry shape → post primitive (exact for box, ball, cylinder, halfspace; else an implicit wrapper)."""
    from toporlogy import geometry as G
    from toporlogy.post import primitives as P
    if s.dim != 3:
        raise ValueError("post-processing primitives are 3-D")
    if isinstance(s, G.Box):
        return P.BoxPrimitive(s.lo.tolist(), s.hi.tolist())
    if isinstance(s, G.Ball):
        return P.SpherePrimitive(s.center.tolist(), s.radius)
    if isinstance(s, G.Cylinder):
        c = 0.5 * (s.p0 + s.p1)
        length = None if getattr(s, "through", False) else s.length
        return P.CylinderPrimitive(c.tolist(), s.axis.tolist(), s.radius, length)
    if isinstance(s, G.HalfSpace):
        return P.PlanePrimitive(s.point.tolist(), s.normal.tolist())
    return P.ImplicitPrimitive(s.sdf, name=type(s).__name__.lower())


def _roles(ctx: BuildContext, preserve: Sequence[Any] | None) -> list[Any]:
    """preserve: [{role: keep | cut | bound, shape: <shape>, stock: 0, functional: null, name: null}]."""
    from toporlogy.post import primitives as P
    out = []
    for i, item in enumerate(preserve or []):
        path = join(join(ctx.path, "preserve"), i)
        if not isinstance(item, dict) or "role" not in item or "shape" not in item:
            raise ConfigError("expected {role: keep | cut | bound, shape: {...}, stock, functional, name}", path)
        bad = [k for k in item if k not in ("role", "shape", "stock", "functional", "name")]
        if bad:
            raise ConfigError(f"unknown key(s) {bad} (keys: role, shape, stock, functional, name)", path)
        cls = {"keep": P.Keep, "cut": P.Cut, "bound": P.Bound}.get(str(item["role"]).lower())
        if cls is None:
            raise ConfigError(f'role: keep | cut | bound, got "{item["role"]}"', join(path, "role"))
        errors: list[tuple[str, str]] = []
        spec = REGISTRY.resolve("shape", item["shape"], join(path, "shape"), errors)
        if errors:
            raise ConfigError(errors)
        assert spec is not None                         # resolve returns None only after appending to errors
        shp = _post_shape(REGISTRY.build("shape", spec, ctx, join(path, "shape")))
        kw: dict[str, Any] = {"stock": float(item.get("stock") or 0.0), "name": str(item.get("name") or
                                                                                   f"{item['role']}{i + 1}")}
        if item.get("functional") is not None:
            kw["functional"] = bool(item["functional"])
        out.append(cls(shp, **kw))
    return out


def _signature_with(fn: Callable[..., Any], dc: type, skip: Sequence[str] = ()) -> Callable[..., Any]:
    import dataclasses
    sig = inspect.signature(fn)
    base = [p for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD]
    extra = []
    for f in dataclasses.fields(dc):
        if f.name in skip:
            continue
        if f.default is not dataclasses.MISSING:
            dv = f.default
        elif f.default_factory is not dataclasses.MISSING:
            dv = f.default_factory()
        else:
            raise TypeError(f"{dc.__name__}.{f.name} has no default")
        if isinstance(dv, tuple):
            dv = list(dv)
        extra.append(inspect.Parameter(f.name, inspect.Parameter.KEYWORD_ONLY, default=dv, annotation=f.type))
    fn.__signature__ = sig.replace(parameters=[*base, *extra])  # type: ignore[attr-defined]
    return fn


def _make_finalize() -> Callable[..., Any]:
    from toporlogy.post import PostParams

    def finalize_step(ctx: BuildContext, preserve: list | None = None, name: str = "part", write: bool = True,
                      **params: Any) -> Any:
        """Optimised density -> smooth, watertight surface (toporlogy.post.finalize; parameters = PostParams);
        preserve: exact keep / cut / bound shapes with stock; writes <name>.stl and <name>_report.json (3-D)."""
        from toporlogy.post import export_result, finalize
        if ctx.domain.dim != 3:
            raise ValueError("finalize needs a 3-D domain")
        pp = PostParams(**params)
        if pp.envelope == "cad" and ctx.part is None:
            raise ValueError('envelope: cad needs the part geometry: a domain of type csg or stl')
        part = ctx.part
        roles = _roles(ctx, preserve)
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            xp, label = _design(state, res)
            pr = finalize(study.domain, xp, preserve=roles, params=pp, part=part)
            state["result"] = pr
            state["result_design"] = label
            rep = pr.report
            files = _rel(export_result(pr, folder, name, ("stl",), print_stock=False), folder) if write else {}
            topo = rep.get("topology_mesh", {})
            out = {"design": label, "triangles": int(len(pr.F)), "volume_mesh": rep.get("volume_mesh"),
                   "volume_vs_target": rep.get("volume_vs_target"), "topology": topo, "edges": rep.get("edges"),
                   "interfaces": rep.get("interfaces"), "seconds": rep.get("seconds", {}).get("total"),
                   "files": files}
            out["summary"] = (f"{out['triangles']} triangles, volume {out['volume_mesh']:.4g} "
                              f"({100 * out['volume_vs_target']:+.2f}% vs target), {topo.get('bodies', '?')} body"
                              + (f" -> {files['stl']}" if files else ""))
            return out
        return step

    return _signature_with(finalize_step, PostParams)


def _make_checks() -> Callable[..., Any]:
    from toporlogy.post import CheckThresholds

    def checks_step(ctx: BuildContext, **thresholds: Any) -> Any:
        """Manufacturing checks of the finalized part (watertight, bodies, voids, wall, overhang, fillet); gated
        values failing make the run exit 1. Needs a finalize step before it."""
        from toporlogy.post import manufacturing_checks
        th = dict(thresholds)
        th["build_direction"] = tuple(float(v) for v in th.get("build_direction", (0, 0, 1)))
        ct = CheckThresholds(**th)
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            pr = _need_result(state, "manufacturing_checks")
            ck = manufacturing_checks(pr, ct)
            state["checks"] = ck
            failed = [it["name"] for it in ck.items if it["passed"] is False]
            return {"passed": bool(ck.passed), "items": ck.items, "failed": failed,
                    "summary": ("all gated checks pass" if ck.passed else "FAILED: " + ", ".join(failed))
                    + f" ({sum(it['passed'] is not None for it in ck.items)} gated)"}
        return step

    return _signature_with(checks_step, CheckThresholds)


def _need_result(state: dict[str, Any], who: str) -> Any:
    pr = state.get("result")
    if pr is None:
        raise RuntimeError(f"{who} needs a finalize step before it")
    return pr


def export_step(ctx: BuildContext, formats: Sequence[str] = ("stl",), name: str = "part",
                print_stock: bool = True) -> Any:
    """Write the finalized surface (stl / ply / obj), the print variant (roles with stock) and <name>_report.json
    (pipeline report, checks) into the run folder. Needs a finalize step before it."""
    bad = [f for f in formats if f not in ("stl", "ply", "obj")]
    if bad:
        raise ValueError(f"formats: stl | ply | obj, got {bad}")
    state = _post_state(ctx)

    def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
        from toporlogy.post import export_result
        pr = _need_result(state, "export")
        ck = state.get("checks")
        files = _rel(export_result(pr, folder, name, tuple(formats), checks=ck, print_stock=print_stock,
                                   extra={"reverify": state.get("reverify")} if state.get("reverify") else None),
                     folder)
        return {"files": files, "summary": ", ".join(files.values())}
    return step


def reverify_step(ctx: BuildContext, E: float | None = None, nu: float | None = None, size: float | None = None,
                  surface_edge: float | None = None, timeout: float = 600.0, compare_voxel: bool = True) -> Any:
    """Re-verify the finalized part: gmsh T10 mesh + static solve with the config's supports and loads (as
    callables in physical coordinates); with compare_voxel the change against the voxel model. E / nu null =
    material. Needs a finalize step before it."""
    import importlib.util
    if importlib.util.find_spec("gmsh") is None:
        raise ValueError("reverify needs the gmsh package (not importable)")
    state = _post_state(ctx)
    E_ = float(ctx.material["E0"] if E is None else E)
    nu_ = float(ctx.material["nu"] if nu is None else nu)
    bcm = bc_mapping(ctx.bc)
    supports, cases = bcm["supports"], bcm["cases"]

    def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
        from toporlogy.post import reverify
        pr = _need_result(state, "reverify")
        xp, label = _design(state, res)                 # the voxel model of the same design that was finalized
        out = reverify(pr, supports, cases, E_, nu_, size=size, surface_edge=surface_edge,
                       domain=study.domain if compare_voxel else None, xphys=xp if compare_voxel else None,
                       timeout=timeout)
        out["design"] = state.get("result_design", label)
        # the T10 / voxel comparison is between two meshes under the SAME mapped conditions; whether those are the
        # optimisation's conditions is bc_fidelity ("equivalent" only when nothing was approximated or left out)
        out["notes"] = bcm["notes"]
        out["bc_fidelity"] = bcm["fidelity"]
        out["bc_mapping"] = bcm["items"]
        state["reverify"] = out
        ch = out.get("change") or []
        out["summary"] = (", ".join(f"{c['name']} compliance x{c['compliance_ratio']:.3f}" for c in ch)
                          or f"{len(out['t10']['cases'])} case(s) solved") + f" ({out['t10']['elements']} T10)"
        if bcm["fidelity"] != "equivalent":
            out["summary"] += f"; {bcm['fidelity']} BC mapping ({len(bcm['items'])} item(s), see bc_mapping)"
        return out
    return step


def __getattr__(name: str) -> Any:                     # PEP 562: lazy post-step constructors
    makers = {"finalize_step": _make_finalize, "checks_step": _make_checks}
    if name in makers:
        if name not in _LAZY:
            _LAZY[name] = makers[name]()
        return _LAZY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_post_steps() -> None:
    """Register the post steps components."""
    length = "length"

    # ---------------------------------------------------------------- post steps
    def connectivity_gate(ctx: BuildContext, threshold: float = 0.5, connectivity: Any = "face",
                          robust_thresholds: Sequence[float] | None = (0.3, 0.7), erosion: bool = True,
                          voids: str = "fail", repair: str = "none", section: int = 2) -> Any:
        """One-body gate (analyse_connectivity) over the interfaces; verdict PASS / FRAGILE / FAIL into
        summary.json (FAIL: exit 1). repair bridge: on FAIL join the bodies into a repaired variant
        (result_repaired.npz; later geometry steps use it, the optimisation result is not modified), gate it again
        and re-evaluate its objective and constraints ("repaired_design"); a repair that violates a constraint,
        or whose constraints cannot all be evaluated (responses that need the design chain), fails the gate."""
        from toporlogy.optimization.connectivity import analyse_connectivity, bridge
        if repair not in ("none", "bridge"):
            raise ValueError(f'repair: none | bridge, got "{repair}"')
        masks = _interfaces(ctx)
        state = _post_state(ctx)

        def gate(dom: Any, x: np.ndarray) -> Any:
            return analyse_connectivity(dom, x, masks, threshold, connectivity, robust_thresholds, erosion,
                                        voids=voids)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            x0, label = _design(state, res)
            rep = gate(study.domain, x0)
            out: dict[str, Any] = {"verdict": rep.verdict, "design": label, **_gate_dict(rep)}
            if rep.verdict == "FAIL" and repair == "bridge":        # FRAGILE is one body already: nothing to join
                br = bridge(study.domain, x0, masks, threshold=threshold, section=section)
                rep2 = gate(study.domain, br.xphys)
                np.savez_compressed(f"{folder}/result_repaired.npz", xphys=br.xphys)
                state["design"] = {"source": res, "xphys": br.xphys, "label": "repaired"}
                out = {"verdict": rep2.verdict, "verdict_before_repair": rep.verdict, "design": "repaired",
                       **_gate_dict(rep2),
                       "repair": {"bridges": br.n_bridges, "added_cells": br.added_cells,
                                  "added_volume": br.added_volume, "removed_cells": br.removed_cells,
                                  "file": "result_repaired.npz"},
                       "repaired_design": _reevaluate(study, br.xphys, ctx.run)}
                d = study.domain                            # plain volume fraction (information, not a gate)
                vf = lambda x: float(np.asarray(x)[d.active].sum() / d.n_active)   # noqa: E731
                out["repaired_design"]["volume_fraction_plain"] = {"optimized": vf(x0), "repaired": vf(br.xphys)}
            rd = out.get("repaired_design")
            # a repaired design passes only if every constraint was evaluated and holds: responses that need the
            # design chain (robust realisations, eroded-field connectivity) cannot be evaluated on it (None)
            out["passed"] = out["verdict"] != "FAIL" and (rd is None or rd["feasible"] is True)
            rp = out.get("repair")
            bad = [c["name"] for c in rd["constraints"] if c["satisfied"] is False] if rd else []
            unknown = [c["name"] for c in rd["constraints"] if c["satisfied"] is None] if rd else []
            out["summary"] = (f"{out['verdict']} ({out['n_bodies']} bod{'y' if out['n_bodies'] == 1 else 'ies'}"
                              f"{', unattached ' + ', '.join(out['unattached']) if out['unattached'] else ''})"
                              + (f" after {rp['bridges']} bridge(s), +{rp['added_cells']} cells; was "
                                 f"{out['verdict_before_repair']}" if rp else "")
                              + (f"; repaired design violates {', '.join(bad)}" if bad else "")
                              + (f"; repaired design not verified: {', '.join(unknown)} could not be evaluated"
                                 if unknown and not bad else ""))
            return out
        return step

    def lengthscale_step(ctx: BuildContext, threshold: float = 0.5, outside: Any = "void",
                         upsample: int | None = None, smooth: float = 0.0, margin: float = 0.0,
                         percentiles: Sequence[float] = (1.0, 5.0, 50.0), radii: Sequence[float] = ()) -> Any:
        """Measure member thickness and void size of the final design (lengths in the unit of h)."""
        from toporlogy.optimization.lengthscale import measure_lengthscale
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            d = study.domain
            xp, label = _design(state, res)
            rep = measure_lengthscale(xp, d.h, threshold, outside, upsample, d.active, percentiles, radii,
                                      margin, smooth)
            keys = ("solid_min", "solid_max", "void_min", "void_max", "grey_fraction", "m_nd", "solid_fraction",
                    "resolution")
            out: dict[str, Any] = {k: float(getattr(rep, k)) for k in keys}
            out["design"] = label
            out.update(solid_percentiles=rep.solid_percentiles, void_percentiles=rep.void_percentiles,
                       opening_residue=rep.opening_residue, closing_residue=rep.closing_residue)
            out["summary"] = (f"solid {out['solid_min']:.3g} .. {out['solid_max']:.3g}, void min "
                              f"{out['void_min']:.3g} (resolution {out['resolution']:.3g})")
            return out
        return step

    def discrete_check(ctx: BuildContext, volume: Any = "constraint", threshold: float = 0.5,
                       max_ratio: float | None = None) -> Any:
        """Objective and constraints of the thresholded (0/1) design, re-evaluated on the same mesh
        (optimization.benchmark). volume: number, "constraint" (bound of the volume constraint) or null (fixed
        threshold). Fails when a constraint that could be evaluated is violated, or objective_binary / objective
        exceeds max_ratio (null = no limit)."""
        from toporlogy.optimization.benchmark import discrete_report

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            out = discrete_report(study, res, volume, threshold)
            sat = [c["satisfied"] for c in out["constraints"].values()]
            ok = not any(s is False for s in sat)
            if max_ratio is not None and out["ratio"] is not None:
                ok = ok and out["ratio"] <= float(max_ratio)
            out["passed"] = bool(ok)
            ratio = "not evaluated" if out["ratio"] is None else f"{out['ratio']:.4f} x the grey value"
            out["summary"] = (f"0/1 design ({out['threshold_rule']}): objective {ratio}, M_nd {out['m_nd']:.3f}"
                              + ("" if not out["not_evaluated"] else
                                 f"; not evaluated: {', '.join(out['not_evaluated'])}"))
            return out
        return step

    def symmetry_check(ctx: BuildContext, planes: Any, threshold: float = 0.5, mesh: bool = True) -> Any:
        """Mirror-symmetry diagnosis about planes {x: 500} or [[x, 500], [y, mid]] (as the symmetry link): e_sym =
        ‖f − R(f)‖₂ / ‖f‖₂ of frac (design space), keep (frozen cells, ties included) and xphys (design used by the
        geometry steps), and, after a finalize step with mesh true, the mirror distance of the final surface
        (post.mesh.mirror_distance). Information only: no verdict, never changes the design. Mirror tying is the
        separate, opt-in symmetry link."""
        from toporlogy.optimization.diagnostics import field_symmetry
        d0 = ctx.domain
        items = list(planes.items()) if isinstance(planes, dict) else [tuple(pl) for pl in planes]
        if not items:
            raise ValueError("planes: at least one plane is required")
        pls = [(_axis(a, d0.dim), _coord(c, _axis(a, d0.dim), d0)) for a, c in items]
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            d = study.domain
            xp, label = _design(state, res)
            pr = state.get("result") if mesh else None
            fields = {"frac": d.frac, "keep": d.keep.astype(float), "xphys": xp}
            rows = []
            for k, c in pls:
                row = field_symmetry(d, fields, k, c, threshold)
                if pr is not None:
                    from toporlogy.post.mesh import mirror_distance
                    row["mesh"] = mirror_distance(pr.V, pr.F, k, c)
                rows.append(row)
            out: dict[str, Any] = {"design": label, "threshold": float(threshold),
                                   "mesh_design": state.get("result_design") if pr is not None else None,
                                   "planes": rows}
            out["summary"] = "; ".join(
                f"{r['axis']}={r['coordinate']:g}: e_sym frac {r['fields']['frac']['e_sym']:.3f}, keep "
                f"{r['fields']['keep']['e_sym']:.3f}, xphys {r['fields']['xphys']['e_sym']:.3f}"
                + (f", surface mean {r['mesh']['mean']:.3g} / max {r['mesh']['hausdorff']:.3g}" if "mesh" in r
                   else "") for r in rows)
            return out
        return step

    register("post", "connectivity_gate", connectivity_gate, types={"connectivity": None})
    register("post", "symmetry_check", symmetry_check, types={"planes": None}, units={"planes": length})
    register("post", "measure_lengthscale", lengthscale_step, types={"outside": None},
             units={"margin": length, "radii": length, "smooth": "elements"})
    register("post", "finalize", "toporlogy.configuration.post_steps:finalize_step",
             units={"sigma": length, "r_open": length, "r_close": length, "r_fillet_min": length})
    register("post", "manufacturing_checks", "toporlogy.configuration.post_steps:checks_step",
             units={"min_wall": length, "plate_level": length, "min_concave_radius": length})
    register("post", "reverify", "toporlogy.configuration.post_steps:reverify_step", units={"E": "force/length^2",
                                                                                "size": length})
    register("post", "export", "toporlogy.configuration.post_steps:export_step")
    register("post", "discrete_check", discrete_check, types={"volume": None})
