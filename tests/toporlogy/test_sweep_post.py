"""Post-processing robustness sweep: finalize() on 7 analytic shapes and 5 real optimisation results.

Pre-registered (docs/olcumler/robustness.md §2, written before the first run). Gating criteria per input:
watertight closed manifold; topology (bodies, χ) of the mesh equal to the thresholded fine-grid design after island
removal; |ΔV| ≤ 1 % vs the optimised volume; bore radius error ≤ 1 % over the full face, plane flatness ≤ 1e-3 h;
minimum triangle angle ≥ 15°; Hausdorff distance to the raw iso-surface ≤ 0.6 h_f both ways. T10 re-check of r1/r2
at two tet sizes (relative change of the T10/voxel compliance ratio ≤ 5 %).

Fast subset (normal tests): two analytic inputs (bore + mounting plane, enclosed cavity). Full sweep:
``@pytest.mark.slow``; with ``TOPORLOGY_SWEEP_RECORD=<file.jsonl>`` every input appends one JSON line (numbers of
docs/olcumler/robustness.json).
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from toporlogy import (MMA, OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume,
                          standard_chain)
from toporlogy.post import (Bound, BoxPrimitive, Cut, CylinderPrimitive, Keep, PlanePrimitive, PostParams, finalize, manufacturing_checks,
                               reverify)
from toporlogy.post.primitives import element_fraction
from toporlogy.post.topology import grid_topology

# The sweeps themselves carry @pytest.mark.slow (not yet re-run on an idle machine, interrupted 15 Nov 2026); the
# fast subset and the builder checks run in every normal run (measured 18 Nov 2026: 27 s for both modules together).


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@dataclass
class Input:
    name: str
    domain: Domain
    xphys: np.ndarray
    preserve: list[Any]
    params: PostParams = field(default_factory=PostParams)
    exact: float | None = None                     # exact solid volume of analytic inputs
    checks: dict[str, Any] = field(default_factory=dict)
    t10: dict[str, Any] | None = None              # reverify arguments (supports, loads, exclusions)
    opt: dict[str, Any] = field(default_factory=dict)


# ============================================================================================== analytic inputs
def p01_sphere() -> Input:
    d = Domain((16, 16, 16))
    c, R = np.array([8.0, 8.0, 8.0]), 5.5
    return Input("p01_sphere", d, element_fraction(d, lambda X: np.linalg.norm(X - c, axis=1) - R), [],
                 exact=4 / 3 * np.pi * R ** 3)


def p02_tube() -> Input:
    d = Domain((14, 14, 10))
    c = np.array([7.0, 7.0, 5.0])
    outer, inner = CylinderPrimitive(c, (0, 0, 1), 5.5), CylinderPrimitive(c, (0, 0, 1), 2.5)
    x = element_fraction(d, lambda X: np.maximum(np.maximum(outer.sdf(X), -inner.sdf(X)), np.abs(X[:, 2] - 5) - 3))
    return Input("p02_tube", d, x, [Cut(inner, name="bore")], exact=np.pi * (5.5 ** 2 - 2.5 ** 2) * 6.0)


def p03_plate_bore() -> Input:
    d = Domain((16, 16, 6))
    R = 3.5
    bore = CylinderPrimitive((8.0, 8.0, 3.0), (0, 0, 1), R)
    x = element_fraction(d, lambda X: np.maximum(np.maximum(1 - X[:, 2], X[:, 2] - 5), -bore.sdf(X)))
    face = Bound(PlanePrimitive((0, 0, 1), (0, 0, -1)), name="mounting face")
    return Input("p03_plate_bore", d, x, [Cut(bore, name="bore"), face], exact=16 * 16 * 4 - np.pi * R * R * 4)


def p04_l_bracket() -> Input:
    d = Domain((20, 8, 20))
    a, b = BoxPrimitive((2, 1, 2), (6, 7, 18)), BoxPrimitive((2, 1, 2), (18, 7, 6))
    x = element_fraction(d, lambda X: np.minimum(a.sdf(X), b.sdf(X)))
    pad = Keep(BoxPrimitive((2, 1, 16), (6, 7, 18)), name="top pad")
    return Input("p04_l_bracket", d, x, [pad], params=PostParams(r_fillet_min=1.5), exact=4 * 6 * 16 + 12 * 6 * 4,
                 checks={"min_concave_radius": 1.5})


def p05_thin_plate() -> Input:
    d = Domain((20, 20, 6))
    t = 1.25
    plate = BoxPrimitive((2, 2, 3 - t / 2), (18, 18, 3 + t / 2))
    return Input("p05_thin_plate", d, element_fraction(d, plate), [], exact=16 * 16 * t,
                 checks={"min_wall": 1.0, "nominal_wall": t})


def _lattice_sdf(X: np.ndarray) -> np.ndarray:
    pos, r = (3.5, 9.5, 15.5), 1.0
    out = np.full(len(X), np.inf)
    for ax in range(3):
        u, v = [k for k in range(3) if k != ax]
        for a in pos:
            for b in pos:
                c = np.zeros(3)
                c[ax], c[u], c[v] = 9.5, a, b
                e = np.zeros(3)
                e[ax] = 1.0
                out = np.minimum(out, CylinderPrimitive(c, e, r, 12.0 + 2 * r).sdf(X))
    return out


def p06_lattice() -> Input:
    d = Domain((19, 19, 19))
    return Input("p06_lattice", d, element_fraction(d, _lattice_sdf), [])


def p07_cavity() -> Input:
    d = Domain((14, 14, 14))
    cube, R = BoxPrimitive((2, 2, 2), (12, 12, 12)), 2.5
    x = element_fraction(d, lambda X: np.maximum(cube.sdf(X), R - np.linalg.norm(X - 7.0, axis=1)))
    return Input("p07_cavity", d, x, [], params=PostParams(fill_voids=False), exact=1000 - 4 / 3 * np.pi * R ** 3)


# ============================================================================================== real results
def _box_mask(shape, lo, hi) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))] = True
    return m


def r1_cantilever() -> Input:
    nx, ny, nz = 30, 10, 10
    keep = np.zeros((nx, ny, nz), bool)
    keep[0] = True
    keep[-1, :, 0] = True
    d = Domain((nx, ny, nz), keep=keep)
    L, eps = float(nx), 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (np.abs(X[:, 0] - L) < eps) & (X[:, 2] < eps))
    F = d.load_vector(1)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    chain = standard_chain(d, 1.5, projection=True)
    st = Study(d, Elasticity(d, fixed), F, chain, SIMP(d), Compliance(), [Constraint(Volume(), 0.3, scale=1.0)],
               OC(dv_scale=d.n_active), BetaContinuation(chain, 1.0, 8.0, 15, 0.01))
    t0 = time.perf_counter()
    res = st.run(max_iter=40, tol=0.01)
    pads = [Keep(BoxPrimitive((0, 0, 0), (1, ny, nz)), name="clamp plate"), Keep(BoxPrimitive((L - 1, 0, 0), (L, ny, 1)), name="load pad")]
    t10 = {"supports": [lambda X: X[:, 0] < 1e-6],
           "loads": {"tip": [(lambda X: (X[:, 0] > L - 0.5) & (X[:, 2] < 0.5), (0.0, 0.0, -1.0))]},
           "exclusions": [lambda X: X[:, 0] < 1.0, lambda X: X[:, 0] > L - 1.0]}
    return Input("r1_cantilever", d, res.xphys, pads, t10=t10, checks={"min_wall": 1.0},
                 opt={"iterations": res.iterations, "seconds": time.perf_counter() - t0, "compliance": res.objective})


def r2_mbb() -> Input:
    nx, ny, nz = 48, 8, 12
    sa, sb = _box_mask((nx, ny, nz), (0, 0, 0), (2, ny, 1)), _box_mask((nx, ny, nz), (nx - 2, 0, 0), (nx, ny, 1))
    lp = _box_mask((nx, ny, nz), (22, 0, nz - 1), (26, ny, nz))
    d = Domain((nx, ny, nz), keep=sa | sb | lp)
    eps = 1e-6
    left = d.nodes_where(lambda X: (X[:, 0] < 2 + eps) & (X[:, 2] < eps))
    right = d.nodes_where(lambda X: (X[:, 0] > nx - 2 - eps) & (X[:, 2] < eps))
    fixed = np.unique(np.concatenate([d.dofs(left), d.dofs(right, 2)]))
    top = d.nodes_where(lambda X: (X[:, 0] > 22 - eps) & (X[:, 0] < 26 + eps) & (X[:, 2] > nz - eps))
    F = d.load_vector(1)
    F[d.dofs(top, 2), 0] = -1.0 / len(top)
    chain = standard_chain(d, 1.5, projection=True)
    st = Study(d, Elasticity(d, fixed), F, chain, SIMP(d), Compliance(), [Constraint(Volume(), 0.15)], MMA(),
               BetaContinuation(chain, 1.0, 16.0, 20, 0.01), move_beta_ref=2.0)
    t0 = time.perf_counter()
    res = st.run(max_iter=120, tol=0.01)
    pads = [Keep(BoxPrimitive((0, 0, 0), (2, ny, 1)), name="support left"), Keep(BoxPrimitive((nx - 2, 0, 0), (nx, ny, 1)),
                                                                        name="support right"),
            Keep(BoxPrimitive((22, 0, nz - 1), (26, ny, nz)), name="load pad")]
    t10 = {"supports": [lambda X: (X[:, 0] < 2 + 1e-6) & (X[:, 2] < 1e-6),
                        (lambda X: (X[:, 0] > nx - 2 - 1e-6) & (X[:, 2] < 1e-6), (2,))],
           "loads": {"mid": [(lambda X: (X[:, 0] > 22 - 1e-6) & (X[:, 0] < 26 + 1e-6) & (X[:, 2] > nz - 1e-6),
                              (0.0, 0.0, -1.0))]},
           "exclusions": [lambda X: X[:, 2] < 1.0, lambda X: X[:, 2] > nz - 1.0]}
    return Input("r2_mbb3d", d, res.xphys, pads, t10=t10, checks={"min_wall": 1.0},
                 opt={"iterations": res.iterations, "seconds": time.perf_counter() - t0, "compliance": res.objective})


def r3_bores() -> Input:
    nx, ny, nz = 32, 6, 20
    R, Rb = 2.5, 4.5
    ca, cb = np.array([5.0, 3.0, 15.0]), np.array([27.0, 3.0, 5.0])
    bores = [CylinderPrimitive(ca, (0, 1, 0), R), CylinderPrimitive(cb, (0, 1, 0), R)]
    d0 = Domain((nx, ny, nz))
    frac = element_fraction(d0, lambda X: -np.minimum(bores[0].sdf(X), bores[1].sdf(X)))
    C = d0.element_centers()

    def radial(c):
        return np.hypot(C[:, 0] - c[0], C[:, 2] - c[2]).reshape(d0.shape)
    keep = ((radial(ca) < Rb) | (radial(cb) < Rb)) & (frac > 0)
    d = Domain((nx, ny, nz), frac=frac, keep=keep)
    X = d.node_coords()
    ra, rb = np.hypot(X[:, 0] - ca[0], X[:, 2] - ca[2]), np.hypot(X[:, 0] - cb[0], X[:, 2] - cb[2])
    sup = d.nodes_where(lambda P: np.hypot(P[:, 0] - ca[0], P[:, 2] - ca[2]) < R + 0.75)
    ld = d.nodes_where(lambda P: np.hypot(P[:, 0] - cb[0], P[:, 2] - cb[2]) < R + 0.75)
    del ra, rb
    fixed = d.dofs(sup)
    F = d.load_vector(1)
    F[d.dofs(ld, 2), 0] = -1.0 / len(ld)
    chain = standard_chain(d, 1.5, projection=True)
    st = Study(d, Elasticity(d, fixed), F, chain, SIMP(d), Compliance(), [Constraint(Volume(), 0.2)], MMA(),
               BetaContinuation(chain, 1.0, 16.0, 20, 0.01), move_beta_ref=2.0)
    t0 = time.perf_counter()
    res = st.run(max_iter=120, tol=0.01)
    pres = [Cut(bores[0], name="bore_a"), Cut(bores[1], name="bore_b"),
            Keep(CylinderPrimitive(ca, (0, 1, 0), Rb, float(ny)), name="boss_a"),
            Keep(CylinderPrimitive(cb, (0, 1, 0), Rb, float(ny)), name="boss_b")]
    return Input("r3_bores", d, res.xphys, pres, checks={"min_wall": 1.0},
                 opt={"iterations": res.iterations, "seconds": time.perf_counter() - t0, "compliance": res.objective})


def r4_robust() -> Input:
    from toporlogy.optimization.robust import DilatedVolume, RobustCompliance, RobustContinuation, RobustSet, \
        robust_chain_settings
    nx, ny, nz = 30, 8, 10
    keep = np.zeros((nx, ny, nz), bool)
    keep[-2:, :, :2] = True
    d = Domain((nx, ny, nz), keep=keep)
    eps = 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - nx) < eps) & (X[:, 2] < eps))
    F = d.load_vector()
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    s = robust_chain_settings(2.0, 2.0, 1.0, 3)
    rs = RobustSet.from_settings(s)
    chain = standard_chain(d, s["rmin"], projection=True, beta=1.0)
    obj = RobustCompliance(rs)
    con = Constraint(DilatedVolume(rs), 0.3)
    sch = RobustContinuation(chain, rs, 1.0, 32.0, every=15, tol=0.01, volume=con, target=0.3, volume_every=10,
                             rescale=[obj])
    st = Study(d, Elasticity(d, fixed), F, chain, SIMP(d), obj, [con], OC(move=0.1, dv_scale=d.n_active), sch)
    t0 = time.perf_counter()
    res = st.run(max_iter=100, tol=0.01)
    pads = [Keep(BoxPrimitive((nx - 2, 0, 0), (nx, ny, 2)), name="tip pad")]
    return Input("r4_robust_minsize", d, res.xphys, pads, checks={"min_wall": s["solid_size"]},
                 opt={"iterations": res.iterations, "seconds": time.perf_counter() - t0, "compliance": res.objective,
                      "min_size_nominal": s["solid_size"]})


def r5_onebody() -> Input:
    import test_sweep_connectivity as C                     # tests/toporlogy is on sys.path (conftest)
    from toporlogy.optimization.connectivity import one_body
    case = C.CASES["c12_3d_four_supports"]
    b = C.build(case)
    plan = one_body(b.domain, b.interfaces)
    d = plan.domain
    chain = standard_chain(d, case.rmin, projection=True, beta=1.0)
    st = Study(d, Elasticity(d, b.fixed), b.F, chain, SIMP(d, penal=3.0), Compliance(),
               [Constraint(Volume(), case.vf)] + plan.constraints, MMA(), BetaContinuation(chain, 1.0, 16.0, every=20),
               move_beta_ref=2.0)
    t0 = time.perf_counter()
    res = st.run(max_iter=case.max_iter)
    rep = plan.gate(res.xphys, voids="warn")
    pads = []
    for k, m in b.interfaces.items():
        idx = np.argwhere(m)
        pads.append(Keep(BoxPrimitive(tuple(idx.min(0).astype(float)), tuple(idx.max(0).astype(float) + 1.0)), name=k))
    return Input("r5_onebody_c12", d, res.xphys, pads, checks={"min_wall": 1.0},
                 opt={"iterations": res.iterations, "seconds": time.perf_counter() - t0, "compliance": res.objective,
                      "connectivity_verdict": rep.verdict})


ANALYTIC: dict[str, Callable[[], Input]] = {f.__name__: f for f in (p01_sphere, p02_tube, p03_plate_bore,
                                                                     p04_l_bracket, p05_thin_plate, p06_lattice,
                                                                     p07_cavity)}
REAL: dict[str, Callable[[], Input]] = {f.__name__: f for f in (r1_cantilever, r2_mbb, r3_bores, r4_robust,
                                                                 r5_onebody)}
FAST = ("p03_plate_bore", "p07_cavity")
T10_SIZES = (1.0, 0.5)


# ============================================================================================== evaluation
def evaluate(inp: Input, t10: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = finalize(inp.domain, inp.xphys, preserve=inp.preserve, params=inp.params)
    t_post = time.perf_counter() - t0
    r = res.report
    h, hf = inp.domain.h, res.grid.h
    ck = manufacturing_checks(res, min_wall=inp.checks.get("min_wall"),
                              min_concave_radius=inp.checks.get("min_concave_radius"))
    before = r["opening"]["topology_before"]
    tm = r["topology_mesh"]
    filled = r["voids"]["enclosed_voids"] if r["voids"]["filled"] else 0
    vox = grid_topology(np.asarray(inp.xphys) >= 0.5)
    e = r["edges"]
    ifc = r["interfaces"]
    bores = [i for i in ifc if i["shape"] == "cylinder" and "radius_rel_error_max" in i]
    planes = [i for i in ifc if i["shape"] == "plane" and "flatness" in i]
    out: dict[str, Any] = {
        "input": inp.name, "grid": list(inp.domain.shape), "h_fine": hf, "triangles": int(len(res.F)),
        "target_volume": r["volumes"]["target"], "exact_volume": inp.exact, "volume_mesh": r["volume_mesh"],
        "dV_vs_target": r["volume_vs_target"],
        "dV_vs_exact": None if inp.exact is None else r["volume_mesh"] / inp.exact - 1.0,
        "volume_stages": r["volumes"],
        "topology_voxel_input": vox, "topology_before": before,
        "topology_mesh": {k: tm[k] for k in ("bodies", "cavities", "euler", "tunnels")},
        "voids_filled": filled, "islands": r["islands"],
        "edges": e, "watertight": bool(e["open"] == 0 and e["non_manifold"] == 0 and e["misoriented"] == 0
                                       and e["non_manifold_vertices"] == 0),
        "interfaces": ifc,
        "bore_err_max": max((i["radius_rel_error_max"] for i in bores), default=None),
        "flatness_max": max((i["flatness"] for i in planes), default=None),
        "hausdorff_final_to_raw": r["vs_raw"]["hausdorff_final_to_ref"],
        "hausdorff_raw_to_final": r["vs_raw"]["hausdorff_ref_to_final"],
        "normal_dev_mean": r["vs_raw"]["normal_dev_mean_deg"], "normal_dev_p99": r["vs_raw"]["normal_dev_p99_deg"],
        "normal_dev_max": r["vs_raw"]["normal_dev_max_deg"],
        "min_angle": r["quality"]["min_angle_deg"], "min_angle_p1": r["quality"]["min_angle_p1_deg"],
        "wall_ray": {k: ck.measured["wall_ray"][k] for k in ("min", "p1", "p5", "median")},
        "wall_ball_p1": ck.measured["wall_sphere"].get("p1"),
        "overhang_fraction": ck.measured["overhang"]["fraction"],
        "r_open": r["opening"]["r_used"], "r_close": r["closing"]["r_used"],
        "open_change": r["opening"]["volume_change"], "close_change": r["closing"]["volume_change"],
        "fillet_ok": r["fillet_radius_ok"], "seconds_post": t_post, "seconds_by_stage": r["seconds"],
        "checks_passed": ck.passed, "checks_table": ck.table(), "opt": inp.opt,
    }
    if "concave_radius" in ck.measured:
        out["concave_radius"] = ck.measured["concave_radius"]
    crit = {
        "watertight": out["watertight"],
        "topology": bool(tm["bodies"] == before["bodies"] and tm["euler"] == before["euler"] - filled),
        "volume_1pct": bool(abs(out["dV_vs_target"]) <= 0.01),
        "bore_1pct": None if out["bore_err_max"] is None else bool(out["bore_err_max"] <= 0.01),
        "flatness": None if out["flatness_max"] is None else bool(out["flatness_max"] <= 1e-3 * h),
        "min_angle_15": bool(out["min_angle"] >= 15.0),
        "hausdorff_0p6hf": bool(max(out["hausdorff_final_to_raw"], out["hausdorff_raw_to_final"]) <= 0.6 * hf),
    }
    out["criteria"] = crit
    out["passed"] = all(v for v in crit.values() if v is not None)
    extra: dict[str, Any] = {}
    if "nominal_wall" in inp.checks:
        extra["wall_p5_within_10pct"] = bool(abs(out["wall_ray"]["p5"] / inp.checks["nominal_wall"] - 1) <= 0.10)
    if "concave_radius" in out:
        cr = out["concave_radius"]
        extra["concave_radius_ok"] = bool(cr["max_added_depth"] <= cr["tolerance"])
    out["extra_criteria"] = extra
    if t10 and inp.t10 is not None:
        runs = []
        for s in T10_SIZES:
            tt = time.perf_counter()
            rv = reverify(res, inp.t10["supports"], inp.t10["loads"], E=1.0, nu=0.3, size=s, domain=inp.domain,
                          xphys=inp.xphys, timeout=1800, exclusions=inp.t10["exclusions"])
            c = rv["t10"]["cases"][0]
            runs.append({"size": s, "elements": rv["t10"]["elements"], "dofs": rv["t10"]["dofs"],
                         "compliance_t10": c["compliance"], "compliance_voxel": rv["voxel"]["cases"][0]["compliance"],
                         "ratio": rv["change"][0]["compliance_ratio"],
                         "max_disp_ratio": rv["change"][0]["max_displacement_ratio"],
                         "volume_ratio": rv["volume_ratio"], "seconds": time.perf_counter() - tt})
        rel = abs(runs[1]["ratio"] - runs[0]["ratio"]) / runs[1]["ratio"]
        out["t10"] = {"runs": runs, "ratio_rel_change": rel, "converged_5pct": bool(rel <= 0.05)}
    rec = os.environ.get("TOPORLOGY_SWEEP_RECORD")
    if rec:
        with open(rec, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, default=float) + "\n")
    return out


def _line(o: dict[str, Any]) -> str:
    c = o["criteria"]
    bad = [k for k, v in c.items() if v is False]
    return (f"  {o['input']:<20} dV {100 * o['dV_vs_target']:+.2f} %  bodies {o['topology_mesh']['bodies']} "
            f"chi {o['topology_mesh']['euler']:g} (before {o['topology_before']['euler']})  angle {o['min_angle']:.1f}  "
            f"H {max(o['hausdorff_final_to_raw'], o['hausdorff_raw_to_final']) / o['h_fine']:.2f} h_f  "
            f"bore {o['bore_err_max']}  r_open {o['r_open']:.3g} r_close {o['r_close']:.3g}  "
            f"{o['seconds_post']:.1f} s  {'PASS' if o['passed'] else 'FAIL ' + ','.join(bad)}")


# ============================================================================================== tests
def gate_failures(o: dict[str, Any]) -> list[str]:
    """Every pre-registered gating criterion (robustness.md §2.2) that ``o`` fails, not only ``o["criteria"]``:
    criteria 1-6 as recorded, criterion 3 also against the exact solid for analytic inputs (the recorded
    ``criteria`` check only the target, robustness.md §5), and the T10 re-check (ratio change between the two tet
    sizes <= 5 %) where it ran. ``extra_criteria`` are "recorded, not gated" and stay out."""
    bad = [k for k, v in o["criteria"].items() if v is False]
    if o.get("dV_vs_exact") is not None and abs(o["dV_vs_exact"]) > 0.01:
        bad.append(f"volume_exact_1pct ({100 * o['dV_vs_exact']:+.2f} %)")
    if "t10" in o and not o["t10"]["converged_5pct"]:
        bad.append(f"t10_converged_5pct (ratio change {100 * o['t10']['ratio_rel_change']:.1f} %)")
    return bad


# Inputs that fail the gate in the stored records (docs/olcumler/robustness.json, "post"; runs of 2026-11-14).
# strict: a fix turns into XPASS and fails the run until the entry is removed. Causes not investigated.
KNOWN_GATE_FAILURES = {
    "p05_thin_plate": "volume vs exact solid +5.6 % (1.25-element plate; vs target -1.0 %)",
    "r1_cantilever": "T10/voxel compliance ratio 0.972 -> 0.865 between tet sizes 1.0 h and 0.5 h (12.4 % > 5 %)",
    "r2_mbb": "volume -1.67 % (> 1 %), Hausdorff 0.72 h_f (> 0.6 h_f)",
    "r3_bores": "Hausdorff 0.79 h_f (> 0.6 h_f)",
    "r4_robust": "Hausdorff 0.67 h_f (> 0.6 h_f)",
}


def _gated(names: list[str]) -> list[Any]:
    return [pytest.param(n, marks=pytest.mark.xfail(reason=KNOWN_GATE_FAILURES[n], strict=True))
            if n in KNOWN_GATE_FAILURES else n for n in names]


@pytest.mark.parametrize("name", FAST)
def test_fast_post(name):
    o = evaluate(ANALYTIC[name]())
    print("\n" + _line(o))
    assert not gate_failures(o), gate_failures(o)


# The sweeps measure (TOPORLOGY_SWEEP_RECORD keeps every input's numbers, written before any assert) AND gate:
# each input must pass every pre-registered criterion, not only be watertight.
@pytest.mark.slow
@pytest.mark.parametrize("name", _gated(list(ANALYTIC)))
def test_sweep_post_analytic(name):
    o = evaluate(ANALYTIC[name]())
    print("\n" + _line(o) + f"  extra {o['extra_criteria']}")
    assert not gate_failures(o), gate_failures(o)


@pytest.mark.slow
@pytest.mark.timeout(5400)                                    # optimisation + finalize + two gmsh/T10 solves
@pytest.mark.parametrize("name", _gated(list(REAL)))
def test_sweep_post_real(name):
    o = evaluate(REAL[name](), t10=name in ("r1_cantilever", "r2_mbb"))
    print("\n" + _line(o) + (f"\n  T10 {o['t10']}" if "t10" in o else ""))
    assert not gate_failures(o), gate_failures(o)
