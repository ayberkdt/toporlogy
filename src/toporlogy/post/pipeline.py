"""Optimised density → manufacturing-ready, measured surface (one call: :func:`finalize`).

Stages and defaults (h = element size of the optimisation grid, h_f = h / upsample):

1. **Fine grid** h_f = h/4 (``upsample=4``), padded by the largest morphology radius + 2 h_f.
2. **Density → field**: normalised trilinear interpolation (``interp_order=1``; 3 = cubic B-spline, smoother), then
   normalised Gaussian blur(ρ·w)/blur(w) over the *care* region w = inside the Bounds and outside the Cuts
   (σ = h/2 by default: removes the C⁰ creases of the trilinear interpolant at the element faces, keeps skins at
   the envelope, which a plain Gaussian over the empty exterior would dilute). Outside the care region the field
   takes the value of the nearest care sample (EDT feature transform): members continue straight through envelope
   faces and holes are bridged, so the later morphology does not round the edges that the exact re-imposition creates.
3. **Volume-preserving threshold**: bisection on the level so that the finished part (exact interfaces included)
   has the optimised volume Σ frac·x̄·h³; sub-voxel volume from the partial-volume estimate.
4. **Signed distance**: sub-voxel redistancing (:func:`toporlogy.post.sdf.redistance`).
5. **Islands** (bodies without load path: neither largest nor touching a Keep) removed.
6. **Adaptive opening** from r = h (``r_open``) down until topology is unchanged and the loss ≤ 2 %
   (``max_open_loss``; a typical value from earlier use, not from a standard).
7. **Adaptive closing** from r = max(h, r_fillet_min) down, topology unchanged, gain ≤ 5 % (``max_close_gain``,
   typical value). A radius below ``r_fillet_min`` is reported as a failed fillet requirement.
8. **Exact re-imposition**: union with Keep, intersection with Bound (envelope, planes), Cut subtracted — functional
   surfaces have nominal size and position. Optional final level shift to restore the optimised volume
   (``volume_match="final"``; the shift acts on the free surface only).
9. **Clean-up**: islands again, enclosed voids filled (``fill_voids=True``; trapped powder) or reported.
10. **Surface**: flying edges on the field; Taubin λ|μ (20 passes, λ 0.5, μ −0.53) with functional vertices
    projected back after every pass; tangential relaxation (5 passes) with every vertex projected onto the implicit
    shape (cubic-spline φ + exact primitives); optional decimation to ``target_faces`` followed by relaxation.
11. **Report**: per-stage volumes, radii trials, grid and mesh topology, interface measurements (bore radius
    mean/min/max over the full face, face flatness), comparison with the raw iso-surface (volume change, Hausdorff
    both ways, normal deviation), triangle quality.

The envelope (outer boundary of the design domain) is, by default (``envelope="voxel"``), the 0.5 level of the
element partial volumes interpolated to the fine grid: within ~h of the CAD surface, wavy at the element scale on
curved faces (bracket, h = 7.5 mm: p95 1.29 mm off the CAD). ``envelope="cad"`` samples the signed distance to the
part geometry itself (CSG shapes or the STL triangles, :func:`toporlogy.geometry.mesh_sdf_grid`) on the fine grid.

Smoothing changes stiffness, stress and even feasibility (Swierstra, Gupta & Langelaar 2020; Subedi, Verma &
Suresh 2020 — neither read in this session), so the finished part is re-verified (:mod:`toporlogy.post.verify`).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from toporlogy.post import mesh as M
from toporlogy.post import sdf as S
from toporlogy.post import smooth as SM
from toporlogy.post.primitives import (Bound, BoxPrimitive, Cut, CylinderPrimitive, GridPrimitive, Keep, PlanePrimitive,
                                          Role, SpherePrimitive, assemble)
from toporlogy.post.regularize import adaptive_closing, adaptive_opening, fill_voids, remove_islands
from toporlogy.post.topology import grid_topology

__all__ = ["PostParams", "PostResult", "finalize", "measure_interfaces", "Implicit", "print_variant",
           "optimised_volume", "remesh"]


@dataclass
class PostParams:
    upsample: int = 4
    interp_order: int = 1
    sigma: float | None = None            # Gaussian σ (length); None → 0.5 h
    volume_match: str = "threshold"       # "threshold" | "final" | "none" (then ``level`` is used)
    level: float = 0.5
    r_open: float | None = None           # start radius (length); None → 1.0 h; 0 = no opening
    max_open_loss: float = 0.02
    r_close: float | None = None          # start radius (length); None → max(1.0 h, r_fillet_min); 0 = none
    r_fillet_min: float = 0.0             # required minimum concave fillet radius (length)
    max_close_gain: float = 0.05
    radius_factor: float = 0.75
    radius_trials: int = 6
    keep_bodies: str = "interfaces"       # "largest" | "interfaces" | "all"
    fill_voids: bool = True
    smoothing: str = "taubin"             # "taubin" | "sinc" | "none"
    smooth_iter: int = 20
    lam: float = 0.5
    mu: float = -0.53
    pass_band: float = 0.1                # windowed sinc only
    relax_iter: int = 5
    project_to_implicit: bool = True
    feature_tol: float | None = None      # vertex-to-functional-surface distance for locking; None → 0.75 h_f
    feature_angle_deg: float = 60.0
    adaptive_radius: float = 0.0          # > 0: curvature-adaptive Taubin weights 1/(1 + (κ r)²)
    target_faces: int | None = None
    envelope: str = "voxel"               # "voxel" (element partial volumes) | "cad" (signed distance to the part)


def optimised_volume(domain: Any, xphys: np.ndarray) -> float:
    """Σ frac · x̄ · h³ over the part."""
    return float((np.asarray(domain.frac) * np.asarray(xphys).reshape(domain.shape)).sum()) * domain.h ** 3


class Implicit:
    """φ(X) of the finished shape at arbitrary points: cubic-spline free field + exact primitives."""

    def __init__(self, grid: S.Grid, phi_free: np.ndarray, roles: Sequence[Role]) -> None:
        self.grid, self.roles = grid, list(roles)
        self._c = ndimage.spline_filter(np.asarray(phi_free, float), order=3)

    def free(self, X: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(self._c, self.grid.index_coords(X), order=3, mode="nearest", prefilter=False)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return assemble(self.free(X), self.roles, X)


@dataclass
class PostResult:
    V: np.ndarray
    F: np.ndarray
    V_raw: np.ndarray
    F_raw: np.ndarray
    grid: S.Grid
    phi: np.ndarray                   # final field on the grid (MC input)
    phi_free: np.ndarray              # smooth free-form part before re-imposition
    roles: list[Role]
    implicit: Implicit
    params: PostParams
    report: dict[str, Any] = field(default_factory=dict)

    def checks(self, **kw: Any) -> Any:
        from toporlogy.post.checks import manufacturing_checks
        return manufacturing_checks(self, **kw)

    def export(self, folder: Any, name: str = "part", **kw: Any) -> dict[str, str]:
        from toporlogy.post.export import export_result
        return export_result(self, folder, name, **kw)


def _envelope_role(domain: Any, grid: S.Grid, mode: str = "voxel", part: Any = None) -> Role:
    """Bound role of the design domain's outer boundary on ``grid`` (see the module notes).

    ``mode="cad"`` needs ``part``: a :class:`toporlogy.geometry.Shape` (CSG) or a closed triangle surface ``(V, F)``
    in the length unit of the domain. Its signed distance is sampled on the fine grid (exact within 3 h_f of the
    surface for a mesh; the CSG bound with exact zero level for shapes)."""
    if mode not in ("voxel", "cad"):
        raise ValueError(f'envelope: "voxel" | "cad", got "{mode}"')
    if mode == "cad":
        if part is None:
            raise ValueError('envelope="cad" needs the part geometry (a domain built from CSG shapes or an STL '
                             "surface; Python: finalize(..., part=shape or (V, F)))")
        from toporlogy import geometry as G
        if isinstance(part, G.Shape):
            phi = part.sdf(grid.points()).reshape(grid.shape)
        else:
            V, F = part
            phi = G.mesh_sdf_grid(V, F, grid.origin, grid.h, grid.shape, band=3.0 * grid.h)
        return Bound(GridPrimitive(phi, grid.origin, grid.h, name="envelope"), functional=False, name="envelope")
    if np.all(np.asarray(domain.frac) >= 1.0):
        lo = np.asarray(domain.origin, float)
        return Bound(BoxPrimitive(lo, lo + domain.h * np.asarray(domain.shape)), functional=True, name="envelope")
    return Bound(GridPrimitive(S.envelope_field(domain, grid), grid.origin, grid.h, name="envelope"), functional=False,
                 name="envelope")


def _cache(roles: Sequence[Role], X: np.ndarray, shape: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {id(r): r.shape.sdf(X).reshape(shape) for r in roles}


def finalize(domain: Any, xphys: np.ndarray, preserve: Sequence[Role] = (), params: PostParams | None = None,
             envelope: Role | None = None, target_volume: float | None = None,
             log: Callable[[str], None] | None = None, part: Any = None) -> PostResult:
    """Run the whole pipeline on the physical density ``xphys`` (``domain.shape``) of a 3-D :class:`Domain`.

    ``envelope`` (a Bound role) replaces the one built from ``params.envelope``; ``part`` is the part geometry that
    ``params.envelope="cad"`` needs (see :func:`_envelope_role`)."""
    p = params or PostParams()
    say = log or (lambda s: None)
    t_start = time.perf_counter()
    times: dict[str, float] = {}
    tic = time.perf_counter()
    h = domain.h
    r_open = h if p.r_open is None else float(p.r_open)
    r_close = max(h, p.r_fillet_min) if p.r_close is None else float(p.r_close)
    sigma = 0.5 * h if p.sigma is None else float(p.sigma)
    grid = S.grid_for_domain(domain, p.upsample, pad=max(r_open, r_close) + h)
    hf = grid.h
    X = grid.points()
    env = envelope if envelope is not None else _envelope_role(domain, grid, p.envelope, part)
    roles: list[Role] = [env] + list(preserve)
    cache = _cache(roles, X, grid.shape)

    def combine(phi: np.ndarray) -> np.ndarray:
        return assemble(phi, roles, cache=cache)

    keep_masks = [cache[id(r)] < 0 for r in roles if isinstance(r, Keep)]
    times["grid"] = time.perf_counter() - tic
    say(f"post: fine grid {grid.shape} h_f={hf:g} ({grid.size / 1e6:.2f} M samples)")
    # ---- density → field → volume-matched threshold
    tic = time.perf_counter()
    rf = S.upsample(domain, xphys, grid, order=p.interp_order)
    # "care" region = inside every Bound, outside every Cut. Outside it the density is a don't-care: it is replaced by
    # the value of the nearest care sample, so members run straight through envelope faces and across holes, and
    # opening/closing never round the edges that the exact re-imposition cuts later (bore mouths, envelope faces)
    care_phi = np.full(grid.shape, -np.inf)
    for r in roles:
        if isinstance(r, Bound):
            care_phi = np.maximum(care_phi, cache[id(r)])
        elif isinstance(r, Cut):
            care_phi = np.maximum(care_phi, -cache[id(r)])
    w = S.partial_volume(care_phi, hf)
    rf = S.normalised_blur(rf, w, sigma / hf)
    rf = S.extrapolate(rf, care_phi < 0)
    target = optimised_volume(domain, xphys) if target_volume is None else float(target_volume)
    stages: dict[str, float] = {"target": target}
    if p.volume_match in ("threshold", "final"):
        t_level, phi_t, v_t = S.volume_threshold(rf, grid, target, combine, redistanced=True)
    elif p.volume_match == "none":
        t_level = p.level
        g = np.gradient(rf, hf)
        phi_t = (t_level - rf) / np.maximum(np.sqrt(sum(x * x for x in g)), 1e-3 / hf)
        v_t = S.solid_volume(combine(phi_t), hf)
    else:
        raise ValueError("volume_match: threshold | final | none")
    stages["thresholded"] = v_t
    phi0 = phi_t if p.volume_match != "none" else S.redistance(phi_t, hf)
    times["threshold_sdf"] = time.perf_counter() - tic
    say(f"post: level {t_level:.4f} -> volume {v_t:.6g} (target {target:.6g})")
    # ---- islands, opening, closing
    tic = time.perf_counter()
    phi0, isl = remove_islands(phi0, combine, keep_masks, hf, p.keep_bodies)
    stages["islands_removed"] = S.solid_volume(combine(phi0), hf)
    if r_open > 0:
        phi_o, op = adaptive_opening(phi0, combine, hf, r_open, p.max_open_loss, p.radius_factor, p.radius_trials)
    else:
        phi_o, op = phi0, {"r_used": 0.0, "volume_change": 0.0, "trials": []}
    stages["opened"] = S.solid_volume(combine(phi_o), hf)
    say(f"post: opening r={op['r_used']:g} ({100 * op['volume_change']:+.2f} %)")
    # closing acts on the free field united with the Keep bodies (junction fillets); Bounds and Cuts are re-imposed
    # afterwards, so their sharp edges never enter the morphology (the free field runs through them)
    phi1 = S.redistance(assemble(phi_o, [r for r in roles if isinstance(r, Keep)], cache=cache), hf)
    if r_close > 0:
        phi_c, cl = adaptive_closing(phi1, combine, hf, r_close, p.max_close_gain, p.radius_factor, p.radius_trials)
    else:
        phi_c, cl = phi1, {"r_used": 0.0, "volume_change": 0.0, "trials": []}
    stages["closed"] = S.solid_volume(combine(phi_c), hf)
    say(f"post: closing r={cl['r_used']:g} ({100 * cl['volume_change']:+.2f} %)")
    times["morphology"] = time.perf_counter() - tic
    # ---- optional final volume match (free surface only), clean-up
    tic = time.perf_counter()
    delta = 0.0
    if p.volume_match == "final":
        a, b = -2 * hf, 2 * hf                        # φ + δ: δ > 0 shrinks
        for _ in range(40):
            delta = 0.5 * (a + b)
            v = S.solid_volume(combine(phi_c + delta), hf)
            if abs(v - target) <= 1e-6 * target:
                break
            if v > target:
                a = delta
            else:
                b = delta
        phi_c = phi_c + delta
    phi_c, isl2 = remove_islands(phi_c, combine, keep_masks, hf, p.keep_bodies)
    phi = combine(phi_c)
    phi_filled, voids = fill_voids(phi, hf, p.fill_voids)
    if voids["filled"]:
        cav = (phi_filled < 0) & ~(phi < 0)
        phi_c = np.where(cav, np.minimum(phi_c, -0.5 * hf), phi_c)
        phi = combine(phi_c)
    stages["final_grid"] = S.solid_volume(phi, hf)
    topo_grid = grid_topology(phi < 0)
    times["cleanup"] = time.perf_counter() - tic
    # ---- surface
    tic = time.perf_counter()
    Vr, Fr = M.extract(phi, grid)
    implicit = Implicit(grid, phi_c, roles)
    functional = [r for r in roles if r.functional]
    feats = SM.Features(functional, p.feature_tol if p.feature_tol is not None else 0.75 * hf, p.feature_angle_deg)
    feats.update(Vr, Fr)
    proj = implicit if p.project_to_implicit else None
    step = 1e-3 * hf
    V = Vr.copy()
    weights = None
    if p.adaptive_radius > 0:
        kap = _mean_curvature(phi_c, hf)
        weights = SM.curvature_weights(V, lambda P: grid.sample(kap, P), p.adaptive_radius)
    if p.smoothing == "taubin":
        V = SM.taubin(V, Fr, p.smooth_iter, p.lam, p.mu, features=feats, weights=weights)
    elif p.smoothing == "sinc":
        V = SM.windowed_sinc(V, Fr, p.smooth_iter, p.pass_band, features=feats)
    elif p.smoothing != "none":
        raise ValueError("smoothing: taubin | sinc | none")
    if p.relax_iter:
        V = SM.relax(V, Fr, p.relax_iter, features=feats, phi=proj, phi_step=step, max_move=hf)
    F = Fr
    dec = {"applied": False}
    if p.target_faces:
        V, F, dec = M.decimate(V, F, int(p.target_faces))
        if dec.get("applied"):
            feats.update(V, F)
            V = SM.relax(V, F, max(p.relax_iter, 5), features=feats, phi=proj, phi_step=step, max_move=hf)
    V, F = M.clean(V, F)
    if M.enclosed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    times["surface"] = time.perf_counter() - tic
    # ---- report
    tic = time.perf_counter()
    rep: dict[str, Any] = {
        "params": asdict(p), "grid": {"shape": list(grid.shape), "h_fine": hf, "origin": grid.origin.tolist()},
        "radii": {"r_open_start": r_open, "r_close_start": r_close, "sigma": sigma},
        "threshold_level": float(t_level), "final_level_shift": float(delta),
        "volumes": stages, "volume_mesh": M.enclosed_volume(V, F), "volume_raw_mesh": M.enclosed_volume(Vr, Fr),
        "volume_vs_target": M.enclosed_volume(V, F) / target - 1.0,
        "islands": {"after_threshold": isl, "final": isl2}, "voids": voids,
        "opening": _strip(op), "closing": _strip(cl),
        "fillet_radius_ok": bool(cl["r_used"] >= p.r_fillet_min - 1e-12),
        "topology_grid": topo_grid, "topology_mesh": _topo_summary(M.mesh_topology(V, F)),
        "edges": M.edge_report(F, len(V)), "decimation": dec, "quality": M.quality(V, F),
        "quality_raw": M.quality(Vr, Fr), "vs_raw": M.compare(V, F, Vr, Fr),
        "vs_implicit": _implicit_error(V, implicit),
        "interfaces": measure_interfaces(V, F, roles, hf),
        "roles": [r.describe() for r in roles],
        "locked_vertices": int(feats.locked.sum()) if feats.assign.size else 0,
    }
    times["report"] = time.perf_counter() - tic
    rep["seconds"] = {**times, "total": time.perf_counter() - t_start}
    say(f"post: {len(F)} triangles, volume {rep['volume_mesh']:.6g} ({100 * rep['volume_vs_target']:+.2f} % vs "
        f"target), {rep['seconds']['total']:.1f} s")
    return PostResult(V, F, Vr, Fr, grid, phi, phi_c, roles, implicit, p, rep)


def _strip(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items()}


def _topo_summary(t: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in t.items() if k != "shell_list"} | {"shells_faces": [s["faces"] for s in t["shell_list"]]}


def _implicit_error(V: np.ndarray, implicit: Implicit) -> dict[str, float]:
    d = np.abs(implicit(V))
    return {"max": float(d.max()), "mean": float(d.mean())} if len(d) else {}


def _mean_curvature(phi: np.ndarray, h: float) -> np.ndarray:
    g = np.gradient(ndimage.gaussian_filter(phi, 1.0), h)
    n = np.sqrt(sum(x * x for x in g)) + 1e-12
    div = sum(np.gradient(g[k] / n, h, axis=k) for k in range(3))
    return 0.5 * div


def remesh(res: PostResult, edge: float | None = None, target_faces: int | None = None, relax_iter: int = 8,
           max_halvings: int = 3) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Isotropic remesh of the finished shape at edge length ``edge`` (or for about ``target_faces`` triangles).

    The implicit shape (cubic-spline free field + exact primitives) is sampled on a grid of spacing ``edge``,
    contoured (flying edges), and the vertices are relaxed tangentially with projection onto the implicit shape and
    the functional primitives. If the topology (bodies, χ) differs from the finished mesh, the spacing is halved
    (members thinner than the spacing). Unlike decimation this keeps triangles near-equilateral and of one size,
    which is what a tetrahedral mesher wants.
    """
    if edge is None:
        if not target_faces:
            raise ValueError("give edge or target_faces")
        edge = float(np.sqrt(4 * M.area(res.V, res.F) / (np.sqrt(3) * target_faces))) * 0.85   # MC: ~1.4 tri/cell
    t_ref = M.mesh_topology(res.V, res.F)
    lo = res.V.min(axis=0) - 2.0 * edge
    hi = res.V.max(axis=0) + 2.0 * edge
    info: dict[str, Any] = {"tries": []}
    e = float(edge)
    for _ in range(max_halvings + 1):
        shape = tuple(int(np.ceil((hi[k] - lo[k]) / e)) + 1 for k in range(3))
        g = S.Grid(lo, e, shape)                                    # type: ignore[arg-type]
        phi = res.implicit(g.points()).reshape(shape)
        V, F = M.extract(phi, g)
        feats = SM.Features([r for r in res.roles if r.functional], 0.5 * e, res.params.feature_angle_deg)
        feats.update(V, F)
        V = SM.relax(V, F, relax_iter, features=feats, phi=res.implicit, phi_step=1e-3 * e, max_move=e)
        V, F = M.clean(V, F)
        t = M.mesh_topology(V, F)
        ok = t["bodies"] == t_ref["bodies"] and t["euler"] == t_ref["euler"] and M.is_watertight(F, len(V))
        info["tries"].append({"edge": e, "faces": int(len(F)), "bodies": t["bodies"], "euler": t["euler"], "ok": ok})
        if ok:
            info.update({"edge": e, "faces": int(len(F)), "quality": M.quality(V, F),
                         "volume": M.enclosed_volume(V, F), "volume_finished": M.enclosed_volume(res.V, res.F)})
            return V, F, info
        e *= 0.5
    info["reason"] = "topology not reproduced at any spacing; finished mesh returned"
    return res.V, res.F, info


def measure_interfaces(V: np.ndarray, F: np.ndarray, roles: Sequence[Role], h: float, band: float = 2.0,
                       angle_deg: float = 45.0) -> list[dict[str, Any]]:
    """Size / flatness of every functional surface over its full face.

    Face vertices: |g| ≤ band·h and normal within ``angle_deg`` of ∇g (locked or not). Cylinder: radius (distance to
    the axis) mean/min/max vs nominal; sphere: radius; plane: signed offset mean and flatness (max − min); other
    shapes: oriented distance statistics.
    """
    n = M.vertex_normals(V, F)
    out = []
    c = np.cos(np.radians(angle_deg))
    for r in roles:
        if not r.functional:
            continue
        g = r.oriented(V)
        on = (np.abs(g) <= band * h) & (np.einsum("ij,ij->i", n, r.oriented_grad(V)) >= c)
        row: dict[str, Any] = {"name": r.name or r.shape.kind, "role": r.role, "shape": r.shape.kind,
                               "vertices": int(on.sum())}
        if on.any():
            P = V[on]
            sh = r.shape
            if isinstance(sh, CylinderPrimitive):
                rad = sh.radial(P)
                row.update({"nominal_radius": sh.radius, "radius_mean": float(rad.mean()),
                            "radius_min": float(rad.min()), "radius_max": float(rad.max()),
                            "radius_rel_error_max": float(np.max(np.abs(rad - sh.radius)) / sh.radius)})
            elif isinstance(sh, SpherePrimitive):
                rad = np.linalg.norm(P - np.asarray(sh.center, float), axis=1)
                row.update({"nominal_radius": sh.radius, "radius_mean": float(rad.mean()),
                            "radius_min": float(rad.min()), "radius_max": float(rad.max())})
            elif isinstance(sh, PlanePrimitive):
                d = sh.sdf(P)
                row.update({"offset_mean": float(d.mean()), "flatness": float(d.max() - d.min())})
            gg = r.oriented(P)
            row.update({"distance_mean": float(gg.mean()), "distance_min": float(gg.min()),
                        "distance_max": float(gg.max())})
        out.append(row)
    return out


def print_variant(res: PostResult, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Print geometry with machining stock on every role with ``stock > 0``.

    Cut (bore) → printed smaller: shape offset by −stock. Keep → printed larger: offset by +stock. Bound plane
    (mounting face) → the face is moved out by stock and the part's footprint on the face is extruded into the new
    slab. Other Bound shapes with stock are not supported (raise). Returns ``{"V", "F", "report"}``.
    """
    grid, roles = res.grid, res.roles
    X = grid.points()
    new_roles: list[Role] = []
    slabs = []
    for r in roles:
        if r.stock <= 0:
            new_roles.append(r)
            continue
        if isinstance(r, Cut):
            new_roles.append(Cut(r.shape.offset(-r.stock), r.functional, 0.0, r.name + "_print"))
        elif isinstance(r, Keep):
            new_roles.append(Keep(r.shape.offset(r.stock), r.functional, 0.0, r.name + "_print"))
        elif isinstance(r, Bound) and isinstance(r.shape, PlanePrimitive):
            new_roles.append(Bound(r.shape.offset(r.stock), r.functional, 0.0, r.name + "_print"))
            slabs.append(r)
        else:
            raise ValueError(f"stock on {r.role} {r.shape.kind} is not supported (use a PlanePrimitive bound)")
    phi_free = res.phi_free.copy()
    phi = assemble(phi_free, new_roles, X)
    for r in slabs:
        n = r.shape._n()                                          # type: ignore[attr-defined]
        s = r.shape.sdf(X)
        Xin = X - (np.maximum(s, 0.0) + 0.5 * grid.h)[:, None] * n   # just inside the nominal face
        foot = grid.sample(res.phi, Xin).reshape(grid.shape)
        slab = np.maximum(foot, np.maximum(s.reshape(grid.shape) - r.stock, -s.reshape(grid.shape) - grid.h))
        phi = np.minimum(phi, slab)
    for r in new_roles:                                           # holes stay open through the slabs
        if isinstance(r, Cut):
            phi = np.maximum(phi, -r.shape.sdf(X).reshape(grid.shape))
    V, F = M.extract(phi, grid)
    feats = SM.Features([r for r in new_roles if r.functional], 0.75 * grid.h, res.params.feature_angle_deg)
    feats.update(V, F)
    V = SM.taubin(V, F, res.params.smooth_iter, res.params.lam, res.params.mu, features=feats)
    V, F = M.clean(V, F)
    rep = {"volume": M.enclosed_volume(V, F), "volume_nominal": M.enclosed_volume(res.V, res.F),
           "edges": M.edge_report(F, len(V)),
           "interfaces": measure_interfaces(V, F, new_roles, grid.h),
           "stock": {r.name or r.shape.kind: r.stock for r in roles if r.stock > 0}}
    if log:
        log(f"post: print variant volume {rep['volume']:.6g} (nominal {rep['volume_nominal']:.6g})")
    return {"V": V, "F": F, "report": rep}
