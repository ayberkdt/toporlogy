"""Per-stage record of the geometry transfer: density → threshold → SDF → morphology → surface → (volume mesh) →
re-analysis (06 §3; ``docs/olcumler/burkulma_geometri_onkayit.md`` GC1–GC9).

:func:`record_chain` takes the optimised density and a :class:`~toporlogy.post.pipeline.PostResult` built with
``finalize(..., keep_fields=True)`` and measures every stage separately:

========================  ======================================================================================
stage                     measured
========================  ======================================================================================
``density``               Σ frac·x̄·h³; bodies and cavities of the voxels x̄ ≥ 0.5 (26- / 6-connected)
``threshold``             fine-grid field after the volume-preserving threshold and redistancing
``islands_removed``       after removing bodies without load path
``opened``, ``closed``    after the adaptive opening / closing
``final``                 after the exact re-imposition of Keep / Bound / Cut and the clean-up
``surface``               the smoothed triangle mesh (divergence volume, shells, edges, ray thickness)
``volume_mesh``           a tetrahedral mesh if one is given, else NOT VERIFIED
``reanalysis_t10``        a :func:`~toporlogy.post.verify.reverify` result if given, else NOT VERIFIED
``reanalysis_voxel``      a :func:`~toporlogy.post.reanalysis.staircase_series` result if given (measured only)
========================  ======================================================================================

On every field stage: volume (partial-volume estimate), bodies / cavities / χ of the samples φ < 0, local thickness
(:func:`~toporlogy.post.checks.sdf_local_thickness` of the redistanced field: min, p1, median), and the CAD
tolerance = the largest signed distance to the envelope of a solid sample (> 0: material outside the envelope).
On the surface the CAD tolerance is the largest envelope distance of a vertex.

Statuses: ``PASS`` / ``FAIL`` for a check with a threshold; ``MEASURED`` without one; ``UNVERIFIED`` for a
constraint that this module cannot evaluate (self-intersection of the surface, symmetry, preservation of a hole
that is not a Cut role) — it is listed only when the caller names it in ``required``; ``NOT VERIFIED`` for an
analysis that did not run (no gmsh). The geometry verdict is PASS only if no gated check failed and nothing required
is UNVERIFIED; the structural verdict comes from the T10 re-analysis alone (a staircase re-analysis is not
body-fitted and never makes it PASS). The two verdicts are never merged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from toporlogy.post import mesh as M
from toporlogy.post import sdf as S
from toporlogy.post.checks import ray_thickness, sdf_local_thickness
from toporlogy.post.primitives import Bound
from toporlogy.post.topology import grid_topology

__all__ = ["PASS", "FAIL", "MEASURED", "UNVERIFIED", "NOT_VERIFIED", "ChainRequirements", "record_chain",
           "UNEVALUABLE"]

PASS, FAIL, MEASURED, UNVERIFIED, NOT_VERIFIED = "PASS", "FAIL", "MEASURED", "UNVERIFIED", "NOT VERIFIED"
UNEVALUABLE = ("self_intersection", "symmetry", "hole_preservation")
FIELD_STAGES = ("threshold", "islands_removed", "opened", "closed", "final")


@dataclass
class ChainRequirements:
    """Thresholds of the gated checks (None = measured, not gated) and the unevaluable constraints the part must
    meet (each becomes UNVERIFIED)."""

    max_bodies: int | None = 1
    max_voids: int | None = 0
    min_wall: float | None = None            # length; local thickness p1 on the fields, ray thickness p1 on the mesh
    cad_tol: float | None = None             # length; largest excursion outside the envelope
    volume_tol: float | None = None          # relative, final surface volume against the optimised volume
    required: tuple[str, ...] = ()           # names from UNEVALUABLE


def _check(name: str, value: Any, threshold: Any, ok: bool | None, note: str = "") -> dict[str, Any]:
    status = MEASURED if ok is None else (PASS if ok else FAIL)
    return {"name": name, "value": value, "threshold": threshold, "status": status, "note": note}


def _envelope(res: Any) -> Any:
    for r in res.roles:
        if isinstance(r, Bound) and r.name == "envelope":
            return r
    return None


def _field_stage(name: str, phi: np.ndarray, res: Any, env_phi: np.ndarray | None,
                 req: ChainRequirements) -> dict[str, Any]:
    h = res.grid.h
    solid = phi < 0
    topo = grid_topology(solid)
    lt = sdf_local_thickness(S.redistance(phi, h), h)
    lt.pop("field")
    metrics: dict[str, Any] = {"volume": S.solid_volume(phi, h), "topology": topo,
                               "thickness": {k: lt.get(k) for k in ("min", "p1", "median", "uncovered")}}
    checks = [_check("bodies", topo["bodies"], req.max_bodies,
                     None if req.max_bodies is None else topo["bodies"] <= req.max_bodies),
              _check("enclosed_voids", topo["cavities"], req.max_voids,
                     None if req.max_voids is None else topo["cavities"] <= req.max_voids)]
    p1 = lt.get("p1", float("nan"))
    checks.append(_check("min_wall_p1", p1, req.min_wall, None if req.min_wall is None else bool(p1 >= req.min_wall),
                         "inscribed-ball local thickness of the redistanced field"))
    if env_phi is not None:
        exc = float(env_phi[solid].max()) if solid.any() else float("-inf")
        metrics["cad_excursion"] = exc
        checks.append(_check("cad_tolerance", exc, req.cad_tol, None if req.cad_tol is None else exc <= req.cad_tol,
                             "largest envelope distance of a solid sample (> 0: outside)"))
    else:
        checks.append({"name": "cad_tolerance", "value": None, "threshold": req.cad_tol, "status": UNVERIFIED,
                       "note": "no envelope role"})
    return {"name": name, "kind": "field", "metrics": metrics, "checks": checks}


def record_chain(domain: Any, xphys: np.ndarray, res: Any, requirements: ChainRequirements | None = None,
                 tet_mesh: Any = None, reverify: dict[str, Any] | None = None,
                 voxel_reanalysis: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stage-by-stage record (module notes). ``res`` must carry the stage fields (``finalize(keep_fields=True)``)."""
    req = requirements or ChainRequirements()
    bad = [r for r in req.required if r not in UNEVALUABLE]
    if bad:
        raise ValueError(f"required: names from {UNEVALUABLE}, got {bad}")
    missing = [s for s in FIELD_STAGES if s not in res.fields]
    if missing:
        raise ValueError(f"the PostResult has no stage fields {missing}: run finalize(..., keep_fields=True)")
    stages: list[dict[str, Any]] = []
    # ---- density (optimisation grid)
    x = np.asarray(xphys, float).reshape(domain.shape)
    frac = np.asarray(domain.frac, float)
    target = float((frac * x).sum()) * domain.h ** domain.dim
    topo_d = grid_topology((x >= 0.5) & (frac > 0))
    stages.append({"name": "density", "kind": "voxel",
                   "metrics": {"volume": target, "volume_voxels_x_ge_half": float(((x >= 0.5) * frac).sum())
                               * domain.h ** domain.dim, "topology": topo_d, "grey_fraction":
                               float(((x > 0.05) & (x < 0.95) & (frac > 0)).sum() / max(int((frac > 0).sum()), 1))},
                   "checks": [_check("bodies", topo_d["bodies"], req.max_bodies, None if req.max_bodies is None
                                     else topo_d["bodies"] <= req.max_bodies),
                              {"name": "min_wall_p1", "value": None, "threshold": req.min_wall,
                               "status": UNVERIFIED if req.min_wall is not None else MEASURED,
                               "note": "no surface on a grey density"}]})
    # ---- fine-grid fields
    env = _envelope(res)
    env_phi = None if env is None else env.shape.sdf(res.grid.points()).reshape(res.grid.shape)
    for name in FIELD_STAGES:
        stages.append(_field_stage(name, res.fields[name], res, env_phi, req))
    # ---- surface
    V, F = res.V, res.F
    er = M.edge_report(F, len(V))
    tp = M.mesh_topology(V, F)
    vol = M.enclosed_volume(V, F)
    rt = ray_thickness(V, F, S.redistance(res.phi, res.grid.h), res.grid)
    wt = er["open"] == 0 and er["non_manifold"] == 0 and er["misoriented"] == 0 and er["non_manifold_vertices"] == 0
    sc = [_check("watertight_manifold", er, "all 0", wt),
          _check("bodies", tp["bodies"], req.max_bodies, None if req.max_bodies is None
                 else tp["bodies"] <= req.max_bodies),
          _check("enclosed_voids", tp["cavities"], req.max_voids, None if req.max_voids is None
                 else tp["cavities"] <= req.max_voids),
          _check("min_wall_p1", rt.get("p1"), req.min_wall, None if req.min_wall is None
                 else bool(rt.get("p1", float("nan")) >= req.min_wall), "ray thickness along the inner normal")]
    rel = vol / target - 1.0 if target > 0 else float("nan")
    sc.append(_check("volume_vs_optimised", rel, req.volume_tol, None if req.volume_tol is None
                     else abs(rel) <= req.volume_tol))
    surf_metrics: dict[str, Any] = {"volume": vol, "topology": {k: v for k, v in tp.items() if k != "shell_list"},
                                    "edges": er, "thickness_ray": {k: rt.get(k) for k in ("min", "p1", "median")},
                                    "faces": int(len(F)), "vertices": int(len(V))}
    if env is not None:
        exc = float(env.shape.sdf(V).max())
        surf_metrics["cad_excursion"] = exc
        sc.append(_check("cad_tolerance", exc, req.cad_tol, None if req.cad_tol is None else exc <= req.cad_tol,
                         "largest envelope distance of a vertex (> 0: outside)"))
    for name in req.required:
        sc.append({"name": name, "value": None, "threshold": "required", "status": UNVERIFIED,
                   "note": "not evaluated by toporlogy.post"})
    stages.append({"name": "surface", "kind": "surface", "metrics": surf_metrics, "checks": sc})
    # ---- volume mesh and analyses
    if tet_mesh is None:
        stages.append({"name": "volume_mesh", "kind": "volume_mesh", "status": NOT_VERIFIED,
                       "note": _gmsh_note(), "metrics": {}, "checks": []})
    else:
        from toporlogy.post.verify import tet_quality
        stages.append({"name": "volume_mesh", "kind": "volume_mesh", "status": MEASURED,
                       "metrics": {"elements": int(len(tet_mesh.T)), "volume": tet_mesh.volume,
                                   "quality": tet_quality(tet_mesh)}, "checks": []})
    if reverify is None:
        stages.append({"name": "reanalysis_t10", "kind": "analysis", "status": NOT_VERIFIED, "note": _gmsh_note(),
                       "metrics": {}, "checks": []})
    else:
        stages.append({"name": "reanalysis_t10", "kind": "analysis", "status": MEASURED, "metrics": reverify,
                       "checks": []})
    if voxel_reanalysis is not None:
        stages.append({"name": "reanalysis_voxel", "kind": "analysis", "status": MEASURED,
                       "note": "staircase voxels, not body-fitted", "metrics": voxel_reanalysis, "checks": []})
    statuses = [c["status"] for st in stages for c in st["checks"] if st["kind"] in ("field", "surface")]
    final_checks = [c["status"] for c in sc]
    if FAIL in final_checks:
        geometry = FAIL
    elif UNVERIFIED in final_checks:
        geometry = UNVERIFIED
    else:
        geometry = PASS
    structural = NOT_VERIFIED if reverify is None else MEASURED
    return {"stages": stages, "geometry_verdict": geometry, "structural_verdict": structural,
            "intermediate_failures": [(st["name"], c["name"]) for st in stages for c in st["checks"]
                                      if c["status"] == FAIL and st["name"] != "surface"],
            "requirements": asdict(req), "statuses_seen": sorted(set(statuses))}


def _gmsh_note() -> str:
    try:
        import gmsh  # noqa: F401
    except ImportError:
        return "gmsh is not installed: no tetrahedral mesh, no T10 re-analysis"
    return "not run"

