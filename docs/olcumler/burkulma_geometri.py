"""Measurements of docs/olcumler/burkulma_geometri.md (pre-registration: burkulma_geometri_onkayit.md).

    python docs/olcumler/burkulma_geometri.py                # every section, from the repository root
    python docs/olcumler/burkulma_geometri.py A SP RL         # some sections (A, SP, RL, GC, CANT, VC)

Problems: ``tests/toporlogy/_buckling_cases.py`` (the tests use the same functions). Results are merged into
``burkulma_geometri.json`` section by section (a section re-run replaces only its own key). The figure:
``burkulma_geometri_figur.py``.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests" / "toporlogy")]
OUT = HERE / "burkulma_geometri.json"

import toporlogy  # noqa: E402,F401  (before torch: PARDISO, see the package)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _buckling_cases as C  # noqa: E402
from toporlogy.post.verify import CONVERGED, NOT_CONVERGED, assess_convergence  # noqa: E402


def _check(value: Any, threshold: str, ok: bool) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "pass": bool(ok)}


_D2: dict[str, Any] = {}


def _d2() -> tuple[np.ndarray, Any, Any]:
    if not _D2:
        t0 = time.perf_counter()
        x, xd, st, d = C.d2_design()
        _D2.update(x=x, xd=xd, st=st, d=d, seconds=time.perf_counter() - t0)
    return _D2["x"], _D2["st"], _D2["d"]


# ============================================================================================================ §A
def section_A() -> dict[str, Any]:
    out: dict[str, Any] = {"checks": {}}
    nbs = (2, 4, 8, 16)
    pt = C.euler_timoshenko()
    q4 = [C.euler_column(nb) for nb in nbs]
    q4im = [C.euler_column(nb, "Q4IM") for nb in nbs]
    a1 = assess_convergence([C.B_COL / nb for nb in nbs], q4)
    ext = a1.get("extrapolated", float("nan"))
    out["A1"] = {"nb": nbs, "lambda_q4": q4, "lambda_q4im": q4im, "P_T": pt, "convergence": a1}
    p = a1.get("order", float("nan"))
    out["checks"]["A1"] = _check({"verdict": a1["verdict"], "order": p, "ext_over_PT_minus_1": ext / pt - 1},
                                 "CONVERGED; 1.5 <= p <= 2.5; |ext/P_T - 1| <= 1e-3",
                                 a1["verdict"] == CONVERGED and 1.5 <= p <= 2.5 and abs(ext / pt - 1) <= 1e-3)
    out["checks"]["A2"] = _check(q4im[-1] / pt - 1, "|lambda/P_T - 1| <= 1e-3", abs(q4im[-1] / pt - 1) <= 1e-3)
    emb = [C.embedded_member(nb) for nb in nbs]
    alone = [C.embedded_member(nb, alone=True) for nb in nbs]
    ratio = [e["lambda1"] / a["lambda1"] for e, a in zip(emb, alone, strict=True)]
    a3 = assess_convergence([C.B_MEM / nb for nb in nbs], [e["lambda1"] for e in emb])
    out["A3"] = {"nb": nbs, "embedded": emb, "alone": alone, "ratio": ratio, "convergence": a3,
                 "convergence_alone": assess_convergence([C.B_MEM / nb for nb in nbs],
                                                         [a["lambda1"] for a in alone])}
    out["checks"]["A3"] = _check({"max_ratio_dev": max(abs(r - 1) for r in ratio), "verdict": a3["verdict"]},
                                 "|ratio - 1| <= 1e-3 on every mesh; CONVERGED",
                                 max(abs(r - 1) for r in ratio) <= 1e-3 and a3["verdict"] == CONVERGED)
    ns = (40, 80, 160, 320)
    wall = [C.wall(n) for n in ns]
    a4 = assess_convergence([1.0 / n for n in ns], [w[0] for w in wall])
    loc = [float(w[1]) for w in wall]
    out["A4"] = {"n": ns, "lambda1": [float(w[0]) for w in wall], "localization": loc,
                 "energy_near_clamp_end": [w[2] for w in wall], "convergence": a4}
    out["checks"]["A4"] = _check({"verdict": a4["verdict"], "localization": loc},
                                 "NOT CONVERGED; localisation strictly increasing",
                                 a4["verdict"] == NOT_CONVERGED
                                 and all(b > a for a, b in zip(loc, loc[1:], strict=False)))
    ns5 = (40, 80, 160)
    w5 = [C.wall(n, exclude_r=0.05) for n in ns5]
    a5 = assess_convergence([1.0 / n for n in ns5], [w[0] for w in w5])
    out["A5"] = {"n": ns5, "lambda1": [float(w[0]) for w in w5], "localization": [float(w[1]) for w in w5],
                 "energy_near_clamp_end": [w[2] for w in w5], "convergence": a5}
    out["checks"]["A5"] = _check(a5["verdict"], "NOT CONVERGED", a5["verdict"] == NOT_CONVERGED)
    rs = (0.025, 0.05, 0.1)
    w6 = [C.wall(80, exclude_r=r) for r in rs]
    lam6 = [float(w[0]) for w in w6]
    out["A6"] = {"r": rs, "lambda1": lam6, "localization": [float(w[1]) for w in w6],
                 "lambda1_no_exclusion": float(wall[1][0])}
    out["checks"]["A6"] = _check(lam6[2] / lam6[0] - 1, ">= 0.10", lam6[2] / lam6[0] - 1 >= 0.10)
    return out


# ============================================================================================================ §B
def section_SP() -> dict[str, Any]:
    out: dict[str, Any] = {"checks": {}}
    emins = (1e-3, 1e-6, 1e-9)
    ref = C.embedded_member(4, alone=True)["lambda1"]
    d1 = {f"{m}|{e:g}": C.embedded_member(4, e, m) for m in C.EG_METHODS for e in emins}
    out["D1"] = {"lambda_alone": ref, "cases": d1}
    dev = {k: v["lambda1"] / ref - 1 for k, v in d1.items()}
    gated = {k: v for k, v in dev.items() if float(k.split("|")[1]) <= 1e-6}
    out["checks"]["SP1"] = _check(max(abs(v) for v in gated.values()), "<= 1e-3",
                                  max(abs(v) for v in gated.values()) <= 1e-3)
    x, st, d = _d2()
    d2 = {f"{m}|{e:g}": C.evaluate_density(st, d, x, e, m) for m in C.EG_METHODS for e in emins}
    out["D2"] = {"cases": d2, "grey_fraction": float(np.mean((x > 0.05) & (x < 0.95))),
                 "volume_fraction": float(x.mean()), "optimisation_seconds": _D2["seconds"]}
    a, b = d2["pg=1|1e-06"], d2["pg=3|1e-06"]
    out["checks"]["SP2"] = _check({"low_density_share": a["low_density_share"],
                                   "lambda_ratio": a["lambda1"] / b["lambda1"]},
                                  "share >= 0.5 and ratio <= 0.5",
                                  a["low_density_share"] >= 0.5 and a["lambda1"] / b["lambda1"] <= 0.5)
    sh = {k: v["low_density_share"] for k, v in d2.items() if k.split("|")[0] in ("pg=3", "pg=4")}
    out["checks"]["SP3"] = _check(sh, "every share < 0.5", all(v < 0.5 for v in sh.values()))
    return out


# ============================================================================================================ §C
def section_RL() -> dict[str, Any]:
    from toporlogy.optimization.buckling import BucklingKS
    from toporlogy.optimization.diagnostics import study_gradcheck
    out: dict[str, Any] = {"checks": {}}
    rel = {}
    for n in (1, 2):
        st, resp, d = C.square_column_study(n)
        C.response_value(st, resp, np.ones(d.n_design))
        rel[n] = resp.last_reliability.as_dict()
    out["RL1"] = rel
    out["checks"]["RL1"] = _check({n: (r["reliable"], r["flags"]) for n, r in rel.items()},
                                  "n=1: not reliable, cluster_cut; n=2: reliable",
                                  (not rel[1]["reliable"]) and "cluster_cut" in rel[1]["flags"]
                                  and rel[2]["reliable"])
    asym: dict[int, list[float]] = {}
    for n in (1, 2):
        st, resp, d = C.square_column_study(n)
        x0 = np.ones(d.n_design)
        dirs = np.random.default_rng(0).standard_normal((4, d.n_design))
        vals = []
        for dv in dirs:
            dp, dm = C.one_sided_derivatives(st, resp, x0, dv)
            vals.append({"D+": dp, "D-": dm, "asym": abs(dp - dm) / max(abs(dp), abs(dm))})
        asym[n] = vals
    out["RL2"] = asym
    a1 = max(v["asym"] for v in asym[1])
    out["checks"]["RL2"] = _check(a1, ">= 1e-2", a1 >= 1e-2)
    st, resp, d = C.square_column_study(2)
    x0 = np.ones(d.n_design)
    r1 = study_gradcheck(st, x0, 1, n_dirs=4, h=2e-4, seed=2)
    r2 = study_gradcheck(st, x0, 1, n_dirs=4, h=4e-4, seed=2)
    err = max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
              for a, b in zip(r1["directions"], r2["directions"], strict=True))
    a2 = max(v["asym"] for v in asym[2])
    out["RL2b"] = {"asym_max": a2, "ad_vs_fd": err}
    out["checks"]["RL2b"] = _check({"asym": a2, "ad_vs_fd": err}, "asym <= 1e-5 and AD vs FD <= 1e-6",
                                   a2 <= 1e-5 and err <= 1e-6)
    _d2()
    resp = BucklingKS(n_modes=6, rho=30.0)
    st2b, _ = C.d2_study()
    for _ in range(2):
        resp(st2b.context(torch.as_tensor(_D2["xd"]), True))
    tr = resp.last_tracking
    assert tr is not None
    out["RL3"] = {"order": tr.order.tolist(), "mac": tr.mac.tolist()}
    out["checks"]["RL3"] = _check({"identity": bool((tr.order == np.arange(len(tr.order))).all()),
                                   "mac_min": float(tr.mac.min())}, "identity; MAC >= 1 - 1e-8",
                                  bool((tr.order == np.arange(len(tr.order))).all()) and tr.mac.min() >= 1 - 1e-8)
    out["RL4"] = _rl4()
    return out


class _Recording:
    """BucklingKS wrapper that logs reliability and tracking after every evaluation."""

    def __new__(cls, **kw: Any) -> Any:
        from toporlogy.optimization.buckling import BucklingKS

        class Rec(BucklingKS):
            def __init__(self, **k: Any) -> None:
                super().__init__(**k)
                self.log: list[dict[str, Any]] = []

            def __call__(self, ctx: Any) -> torch.Tensor:
                v = super().__call__(ctx)
                tr, rl = self.last_tracking, self.last_reliability
                self.log.append({"lambda": self.last_load_factors.tolist(),
                                 "reliable": rl.reliable if rl else None, "flags": list(rl.flags) if rl else [],
                                 "switch": None if tr is None else bool((tr.order != np.arange(len(tr.order))).any()),
                                 "mac_mode1": None if tr is None else float(tr.mac[0]),
                                 "order": None if tr is None else tr.order.tolist()})
                return v
        return Rec(**kw)


def _rl4() -> dict[str, Any]:
    from toporlogy.optimization.buckling import BucklingKS
    _d2()
    st0, _ = C.d2_study()
    mon = BucklingKS(n_modes=6, rho=30.0)
    mon(st0.context(torch.as_tensor(_D2["xd"]), True))
    lam0 = float(mon.last_load_factors[0])
    rec = _Recording(n_modes=6, rho=30.0)
    st, _ = C.d2_study(rec, 1.6 * lam0)
    t0 = time.perf_counter()
    res = st.run(max_iter=60, tol=1e-4)
    log = rec.log
    sw = [e["switch"] for e in log if e["switch"] is not None]
    macs = [e["mac_mode1"] for e in log if e["mac_mode1"] is not None]
    return {"lambda0": lam0, "target": 1.6 * lam0, "evaluations": len(log), "switches": int(sum(sw)),
            "mac_mode1_min": float(min(macs)) if macs else None, "unreliable": int(sum(not e["reliable"] for e in log)),
            "flags_seen": sorted({f for e in log for f in e["flags"]}), "iterations": int(res.iterations),
            "objective": float(res.objective), "seconds": time.perf_counter() - t0, "log": log}


# ============================================================================================================ §D
def _bar_sdf(X: np.ndarray) -> np.ndarray:
    from toporlogy.post.primitives import BoxPrimitive
    a = BoxPrimitive((0, 2, 2), (12, 10, 10)).sdf(X)
    b = BoxPrimitive((20, 2, 2), (32, 10, 10)).sdf(X)
    c = BoxPrimitive((12, 4, 2), (20, 8, 10)).sdf(X)
    return np.minimum(np.minimum(a, b), c)


def section_GC() -> dict[str, Any]:
    from toporlogy import Domain
    from toporlogy.post import finalize
    from toporlogy.post import sdf as S
    from toporlogy.post.chain import UNEVALUABLE, UNVERIFIED, ChainRequirements, record_chain
    from toporlogy.post.checks import sdf_local_thickness
    from toporlogy.post.primitives import element_fraction
    from toporlogy.post.reanalysis import solve_voxels, staircase_series, voxel_solid
    out: dict[str, Any] = {"checks": {}}
    d = Domain((32, 12, 12), 1.0)
    x = element_fraction(d, _bar_sdf)
    t0 = time.perf_counter()
    res = finalize(d, x, keep_fields=True)
    out["finalize_seconds"] = time.perf_counter() - t0
    rec = record_chain(d, x, res, ChainRequirements(required=UNEVALUABLE))
    out["record"] = rec
    st = {s["name"]: s for s in rec["stages"]}
    exact = 1792.0
    v = (st["density"]["metrics"]["volume"], st["threshold"]["metrics"]["volume"], st["surface"]["metrics"]["volume"])
    out["checks"]["GC1"] = _check([vv / exact - 1 for vv in v], "<= 1e-9 / 5e-3 / 3e-2",
                                  abs(v[0] / exact - 1) <= 1e-9 and abs(v[1] / exact - 1) <= 5e-3
                                  and abs(v[2] / exact - 1) <= 3e-2)
    topo = {n: (s["metrics"]["topology"]["bodies"], s["metrics"]["topology"].get("cavities"))
            for n, s in st.items() if "topology" in s["metrics"]}
    out["checks"]["GC2"] = _check(topo, "bodies 1, cavities 0 on every stage",
                                  all(b == 1 and (c in (0, None)) for b, c in topo.values())
                                  and topo["density"][1] == 0)
    er = st["surface"]["metrics"]["edges"]
    out["checks"]["GC3"] = _check(er, "open = non-manifold = misoriented = 0",
                                  er["open"] == 0 and er["non_manifold"] == 0 and er["misoriented"] == 0)
    g = res.grid
    ax = g.axes()
    ix = np.flatnonzero((ax[0] >= 14) & (ax[0] <= 18))
    iy, iz = int(np.argmin(np.abs(ax[1] - 6))), int(np.argmin(np.abs(ax[2] - 6)))
    neck = {}
    for name in ("threshold", "final"):
        lt = sdf_local_thickness(S.redistance(res.fields[name], g.h), g.h)["field"]
        vals = lt[ix, iy, iz]
        neck[name] = {"min": float(vals.min()), "max": float(vals.max()), "y": float(ax[1][iy]),
                      "z": float(ax[2][iz])}
    out["checks"]["GC4"] = _check(neck, "4 +- 0.5", all(abs(n["min"] - 4) <= 0.5 and abs(n["max"] - 4) <= 0.5
                                                        for n in neck.values()))
    exc = st["surface"]["metrics"]["cad_excursion"]
    out["checks"]["GC5"] = _check({"excursion": exc, "h_f": g.h}, "<= 0.05 h_f", exc <= 0.05 * g.h)
    unv = {c["name"]: c["status"] for c in st["surface"]["checks"] if c["name"] in UNEVALUABLE}
    out["checks"]["GC6"] = _check({"statuses": unv, "geometry_verdict": rec["geometry_verdict"]},
                                  "UNVERIFIED each; verdict not PASS",
                                  all(s == UNVERIFIED for s in unv.values()) and len(unv) == 3
                                  and rec["geometry_verdict"] != "PASS")
    sup = [lambda X: X[:, 0] < 1e-9]
    load = [(lambda X: X[:, 0] > 32 - 1e-9, (1.0, 0.0, 0.0))]
    solid, org = voxel_solid(res.implicit, (0, 0, 0), (32, 12, 12), 0.5)
    r7 = solve_voxels(solid, org, 0.5, sup, load, 1.0, 0.0, backends=("toporlogy", "skfem"))
    dif = abs(r7["toporlogy"]["compliance"] / r7["skfem"]["compliance"] - 1)
    out["GC7"] = r7
    out["checks"]["GC7"] = _check(dif, "<= 1e-8", dif <= 1e-8)
    t0 = time.perf_counter()
    r8 = staircase_series(res.implicit, (0, 0, 0), (32, 12, 12), (1 / 2, 1 / 3, 1 / 4), sup, load, 1.0, 0.0,
                          volume_reference=st["surface"]["metrics"]["volume"], solver="gmg")
    r8["seconds"] = time.perf_counter() - t0
    out["GC8"] = r8
    c_fine = r8["meshes"][-1]["toporlogy"]["compliance"]
    out["checks"]["GC8"] = _check({"C_fine": c_fine, "C_over_Ca_minus_1": c_fine / 0.625 - 1,
                                   "verdict": r8["convergence"]["verdict"]}, "|C/C_a - 1| <= 0.05",
                                  abs(c_fine / 0.625 - 1) <= 0.05)
    t10 = st["reanalysis_t10"]["status"]
    out["checks"]["GC9"] = _check({"status": t10, "note": st["reanalysis_t10"].get("note")}, "NOT VERIFIED",
                                  t10 == "NOT VERIFIED")
    return out


def section_CANT() -> dict[str, Any]:
    from toporlogy import OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume
    from toporlogy import standard_chain
    from toporlogy.post import BoxPrimitive, Keep, PostParams, finalize
    from toporlogy.post.chain import ChainRequirements, record_chain
    from toporlogy.post.reanalysis import staircase_series
    out: dict[str, Any] = {"checks": {}}
    nx, ny, nz = 30, 10, 10
    keep = np.zeros((nx, ny, nz), bool)
    keep[0] = True
    keep[-1, :, 0] = True
    d = Domain((nx, ny, nz), h=1.0, keep=keep)
    L, eps = nx * d.h, 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    tip = d.nodes_where(lambda X: (np.abs(X[:, 0] - L) < eps) & (X[:, 2] < eps))
    F = d.load_vector(1)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    chain = standard_chain(d, 1.5, projection=True)
    st = Study(d, Elasticity(d, fixed, nu=0.3), F, chain, SIMP(d), Compliance(),
               [Constraint(Volume(), 0.3, scale=1.0)], OC(dv_scale=d.n_active),
               BetaContinuation(chain, 1.0, 8.0, 15, 0.01))
    t0 = time.perf_counter()
    opt = st.run(max_iter=40, tol=0.01)
    out["optimisation"] = {"seconds": time.perf_counter() - t0, "iterations": int(opt.iterations),
                           "objective": float(opt.objective)}
    x = np.asarray(opt.xphys, float).reshape(d.shape)
    pads = [Keep(BoxPrimitive((0, 0, 0), (1, ny, nz)), name="clamp plate"),
            Keep(BoxPrimitive((L - 1, 0, 0), (L, ny, 1)), name="load pad")]
    sup = [lambda X: X[:, 0] < 1e-9]
    load = [(lambda X: (X[:, 0] > L - 1.0 - 1e-9) & (X[:, 2] < 1e-9), (0.0, 0.0, -1.0))]
    target = float(x.sum())
    for vm in ("threshold", "final"):
        res = finalize(d, x, preserve=pads, params=PostParams(volume_match=vm), keep_fields=True)
        rec = record_chain(d, x, res, ChainRequirements(min_wall=1.0))
        stg = {s["name"]: s for s in rec["stages"]}
        series = staircase_series(res.implicit, (0, 0, 0), (L, ny, nz), (1 / 2, 1 / 3, 1 / 4), sup, load, 1.0, 0.3,
                                  volume_reference=stg["surface"]["metrics"]["volume"], solver="gmg")
        out[vm] = {"record": rec, "reanalysis_voxel": series,
                   "surface_volume_over_target": stg["surface"]["metrics"]["volume"] / target - 1}
    # the grey design itself on the optimisation grid (same callables, consistent face loads, SIMP p = 3)
    out["grey_voxel_compliance_grid"] = _grey_compliance(d, x, sup, load)
    vt = {vm: out[vm]["surface_volume_over_target"] for vm in ("threshold", "final")}
    out["checks"]["VC3"] = _check(vt, "|V/V_target - 1| <= 0.01 for both", all(abs(v) <= 0.01 for v in vt.values()))
    return out


def _grey_compliance(d: Any, x: np.ndarray, sup: Any, load: Any) -> float:
    from toporlogy.fem.elastic import Elasticity
    from toporlogy.post.reanalysis import face_weights
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(np.asarray(sup[0](X), bool)))
    sel, f = load[0]
    w = face_weights(np.ones(d.shape, bool), d.h, np.asarray(sel(X), bool))
    F = np.zeros(d.ndof)
    for k in range(3):
        F[3 * np.arange(len(X)) + k] = f[k] * w / w.sum()
    m = Elasticity(d, fixed, nu=0.3, element="H8")
    xa = x.reshape(-1)[m.elements]
    U = m.solve_numpy(1e-9 + (1 - 1e-9) * xa ** 3, F)[:, 0]
    return float(F @ U)


# ============================================================================================================ §E
def section_VC() -> dict[str, Any]:
    out: dict[str, Any] = {"checks": {}}
    x, st, d = _d2()
    var = C.volume_variants(x)
    ev = {k: C.evaluate_density(st, d, v) for k, v in var.items()}
    out["variants"] = ev
    keys = list(ev)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va, vb = ev[a]["volume"], ev[b]["volume"]
            if abs(va / vb - 1) <= 5e-3:
                pairs.append({"pair": [a, b], "lambda_rel": abs(ev[a]["lambda1"] / ev[b]["lambda1"] - 1),
                              "vm_rel": abs(ev[a]["vm_max_solid"] / ev[b]["vm_max_solid"] - 1),
                              "compliance_rel": abs(ev[a]["compliance"] / ev[b]["compliance"] - 1)})
    out["pairs"] = pairs
    best = max((max(p["lambda_rel"], p["vm_rel"]) for p in pairs), default=0.0)
    out["checks"]["VC1"] = _check(best, "some volume-equal pair differs by >= 5 % in lambda1 or max vM",
                                  best >= 0.05)
    return out


def section_POST() -> dict[str, Any]:
    """Post-hoc (not pre-registered), after RL2b failed: the whole-cluster asymmetry and the AD/FD difference as a
    function of the step. A noise floor (eigen-solver tolerance 1e-10 / h) falls with a larger step, a kink does
    not; a smooth truncation error grows with the step."""
    from toporlogy.optimization.diagnostics import study_gradcheck
    out: dict[str, Any] = {"label": "post-hoc"}
    rows = []
    for h in (1e-5, 1e-4, 1e-3, 1e-2):
        st, resp, d = C.square_column_study(2)
        x0 = np.ones(d.n_design)
        asym = []
        for dv in np.random.default_rng(0).standard_normal((4, d.n_design)):
            dp, dm = C.one_sided_derivatives(st, resp, x0, dv, h=h)
            asym.append(abs(dp - dm) / max(abs(dp), abs(dm)))
        r1 = study_gradcheck(st, x0, 1, n_dirs=4, h=2 * h, seed=2)
        r2 = study_gradcheck(st, x0, 1, n_dirs=4, h=4 * h, seed=2)
        err = max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
                  for a, b in zip(r1["directions"], r2["directions"], strict=True))
        st1, resp1, _ = C.square_column_study(1)
        cut = []
        for dv in np.random.default_rng(0).standard_normal((4, d.n_design)):
            dp, dm = C.one_sided_derivatives(st1, resp1, x0, dv, h=h)
            cut.append(abs(dp - dm) / max(abs(dp), abs(dm)))
        rows.append({"h": h, "asym_whole_max": max(asym), "ad_vs_fd": err, "asym_cut_max": max(cut)})
        print(f"  post-hoc h={h:g}: whole {max(asym):.3e}, AD-FD {err:.3e}, cut {max(cut):.3e}", flush=True)
    out["steps"] = rows
    out["GC8_sections"] = _gc8_sections()
    out["A6_mode_location"] = _a6_location()
    return out


def _a6_location() -> dict[str, Any]:
    """Post-hoc for A5/A6: share of the mode-1 energy within 0.05 of the two re-entrant corners of the opening
    ((0.4, 0.6) and (0.8, 0.6)) on the 80² wall, without and with exclusion (r = 0.025, 0.05, 0.1)."""
    import toporlogy as T
    from toporlogy.fem.buckling import LinearBuckling
    out = {}
    n = 80
    for r in (None, 0.025, 0.05, 0.1):
        lam, loc, near = C.wall(n, exclude_r=r)
        out[str(r)] = {"lambda1": float(lam), "near_clamp_end": near}
    # the energy field itself: rebuild the 80² wall with the same function's pieces
    d = T.Domain((n, n), 1.0 / n)
    c = d.element_centers()
    corners = np.array([[0.4, 0.6], [0.8, 0.6]])
    dist = np.min(np.hypot(c[:, 0][:, None] - corners[None, :, 0], c[:, 1][:, None] - corners[None, :, 1]), axis=1)
    for r in (None, 0.05):
        en = C.wall_mode_energy(n, exclude_r=r)
        out[str(r)]["near_opening_corners"] = float(en[dist < 0.05].sum() / en.sum())
    _ = LinearBuckling
    return out


def _gc8_sections() -> dict[str, Any]:
    """Post-hoc for GC8: the series-bar value with the cross-section areas A(x) of the finished staircase geometry
    (h = 1/4), Σ h / A_i, against the nominal 0.625 and the 3-D staircase compliance."""
    from toporlogy import Domain
    from toporlogy.post import finalize
    from toporlogy.post.primitives import element_fraction
    from toporlogy.post.reanalysis import voxel_solid
    d = Domain((32, 12, 12), 1.0)
    res = finalize(d, element_fraction(d, C.stepped_bar_sdf))
    h = 0.25
    solid, _ = voxel_solid(res.implicit, (0, 0, 0), (32, 12, 12), h)
    A = solid.sum(axis=(1, 2)) * h * h
    xs = (np.arange(len(A)) + 0.5) * h
    return {"series_measured_areas": float(np.sum(h / A)), "nominal": 0.625,
            "area_wide_mid": float(A[np.argmin(np.abs(xs - 6))]), "area_neck_mid": float(A[np.argmin(np.abs(xs - 16))]),
            "nominal_areas": [64.0, 32.0]}


SECTIONS = {"A": section_A, "SP": section_SP, "RL": section_RL, "GC": section_GC, "CANT": section_CANT,
            "VC": section_VC, "POST": section_POST}


def _env() -> dict[str, Any]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()  # noqa: S603, S607
    import scipy
    try:
        import skfem
        skv = skfem.__version__
    except ImportError:
        skv = None
    return {"sha": sha, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "skfem": skv, "platform": platform.platform(),
            "date": time.strftime("%Y-%m-%d %H:%M %z")}


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def main(names: list[str]) -> None:
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names or list(SECTIONS):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = SECTIONS[name]()
        r["seconds_section"] = time.perf_counter() - t0
        r["env"] = _env()
        data[name] = r
        OUT.write_text(json.dumps(data, indent=1, default=_default, ensure_ascii=False), encoding="utf-8")
        for k, c in r.get("checks", {}).items():
            print(f"{k}: {'PASS' if c['pass'] else 'FAIL'}  {c['value']}  ({c['threshold']})", flush=True)
        print(f"[{name}] {r['seconds_section']:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
