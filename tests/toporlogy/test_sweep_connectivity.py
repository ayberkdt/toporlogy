"""Connectivity robustness sweep: 14 scenarios × {plain, one_body default, ties only, conduction only}.

Pre-registered (docs/olcumler/robustness.md §1, written before the first run): case list, methods and pass criteria.
Criterion per run: ``analyse_connectivity(..., voids="warn")`` verdict PASS — one body (threshold 0.5) holding every
interface, every interface still in one body at thresholds 0.3 and 0.7 and after a one-element erosion. Enclosed
voids are reported separately. Recorded per run: bodies, verdict, compliance ratio vs plain, volume feasibility,
iterations, time.

Fast subset (normal tests): two small 2-D scenarios. Full sweep: ``@pytest.mark.slow``; with the environment variable
``TOPORLOGY_SWEEP_RECORD=<file.jsonl>`` every run appends one JSON line (the numbers of docs/olcumler/robustness.json).
The scenario builders are plain functions so ``scratch/robustness`` can run single cases.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from toporlogy import (MMA, OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume,
                          standard_chain)
from toporlogy.connectivity import _nodes_of, analyse_connectivity, one_body

# The sweeps themselves carry @pytest.mark.slow (not yet re-run on an idle machine, interrupted 15 Nov 2026); the
# fast subset and the builder checks run in every normal run (measured 18 Nov 2026: 27 s for both modules together).


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ============================================================================================== scenarios
Box = tuple[tuple[int, int], ...]          # element index range [lo, hi) per axis


@dataclass(frozen=True)
class Case:
    """One scenario. Pads are element boxes; every pad is a keep region and an interface. Support pads: all DOFs of
    their nodes fixed. Load pads: per load case a force vector, total spread evenly over the pad nodes."""

    name: str
    dim: int
    shape: tuple[int, ...]
    supports: dict[str, Box]
    loads: dict[str, tuple[Box, tuple[tuple[float, ...], ...]]]   # name → (box, force per load case)
    vf: float
    rmin: float
    projection: bool
    optimizer: str                                                # "MMA" | "OC"
    outside: Box | None = None                                    # frac = 0 (L-shaped domain)
    void: tuple[Box, ...] = ()                                    # obstacle (void region, still in the part)
    max_iter: int = 150
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def _mask(shape: tuple[int, ...], box: Box) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in box)] = True
    return m


@dataclass
class Built:
    case: Case
    domain: Domain
    interfaces: dict[str, np.ndarray]
    fixed: np.ndarray
    F: np.ndarray


def build(case: Case) -> Built:
    shape = case.shape
    frac = None
    if case.outside is not None:
        frac = (~_mask(shape, case.outside)).astype(float)
    void = np.zeros(shape, bool)
    for b in case.void:
        void |= _mask(shape, b)
    ifc = {k: _mask(shape, b) for k, b in case.supports.items()}
    ifc.update({k: _mask(shape, b) for k, (b, _) in case.loads.items()})
    keep = np.logical_or.reduce(list(ifc.values()))
    if (keep & void).any():
        raise ValueError(f"{case.name}: a pad overlaps the obstacle")
    d = Domain(shape, 1.0, frac=frac, keep=keep, void=void)
    for k, m in ifc.items():
        if (m & ~d.active).any():
            raise ValueError(f"{case.name}: pad {k} leaves the part")
    fixed = d.dofs(np.unique(np.concatenate([_nodes_of(d, ifc[k]) for k in case.supports])))
    n_cases = len(next(iter(case.loads.values()))[1])
    F = d.load_vector(n_cases)
    for k, (_, forces) in case.loads.items():
        if len(forces) != n_cases:
            raise ValueError(f"{case.name}: load {k} has {len(forces)} cases, expected {n_cases}")
        nodes = np.setdiff1d(_nodes_of(d, ifc[k]), fixed // d.dim)
        for c, f in enumerate(forces):
            for a, v in enumerate(f):
                F[d.dim * nodes + a, c] += v / nodes.size
    return Built(case, d, ifc, fixed, F)


def _split2d(nx: int = 60, ny: int = 24, sw: int = 6, pad: int = 4, inset: int = 4) -> tuple[dict, dict]:
    """Two support pads (bottom, width sw) near the ends; a pad × pad load pad at mid-height above each."""
    xa, xb = inset, nx - inset - sw
    z0 = ny // 2 - pad // 2
    sup = {"support_a": ((xa, xa + sw), (0, 1)), "support_b": ((xb, xb + sw), (0, 1))}
    la, lb = xa + (sw - pad) // 2, xb + (sw - pad) // 2
    loads = {"load_a": ((la, la + pad), (z0, z0 + pad)), "load_b": ((lb, lb + pad), (z0, z0 + pad))}
    return sup, loads


def cases() -> list[Case]:
    out: list[Case] = []
    sup, lp = _split2d()
    pull = 0.02
    out.append(Case("c01_2d_split_far", 2, (60, 24), sup,
                    {"load_a": (lp["load_a"], ((-pull, -1.0),)), "load_b": (lp["load_b"], ((pull, -1.0),))},
                    0.15, 2.0, True, "MMA", note="two spans, pads far apart, weak pull apart", tags=("2d", "far")))
    out.append(Case("c02_2d_three_loads", 2, (72, 24),
                    {"support_a": ((2, 8), (0, 1)), "support_b": ((64, 70), (0, 1))},
                    {"load_a": (((14, 18), (10, 14)), ((0.0, -1.0),)),
                     "load_b": (((34, 38), (10, 14)), ((0.0, -1.0),)),
                     "load_c": (((54, 58), (10, 14)), ((0.0, -1.0),))},
                    0.12, 2.0, True, "MMA", note="3 load pads between 2 supports", tags=("2d", "3loads")))
    out.append(Case("c03_2d_four_columns", 2, (80, 20),
                    {f"support_{k}": ((x, x + 6), (0, 1)) for k, x in zip("abcd", (4, 24, 48, 68))},
                    {f"load_{k}": (((x + 1, x + 5), (10, 14)), ((0.0, -1.0),)) for k, x in zip("abcd", (4, 24, 48, 68))},
                    0.10, 1.5, True, "MMA", note="4 loads, each above its own support (4 columns)",
                    tags=("2d", "4loads", "4supports")))
    out.append(Case("c04_2d_close_OC_noproj", 2, (48, 24),
                    {"support_a": ((14, 20), (0, 1)), "support_b": ((28, 34), (0, 1))},
                    {"load_a": (((15, 19), (12, 16)), ((-pull, -1.0),)), "load_b": (((29, 33), (12, 16)), ((pull, -1.0),))},
                    0.20, 2.0, False, "OC", note="pads close (gap 8), density filter only, OC", tags=("2d", "close")))
    out.append(Case("c05_2d_low_vf_0p08", 2, (60, 24), sup,
                    {"load_a": (lp["load_a"], ((-pull, -1.0),)), "load_b": (lp["load_b"], ((pull, -1.0),))},
                    0.08, 1.5, True, "MMA", note="c01 at volume fraction 0.08, rmin 1.5", tags=("2d", "lowvf")))
    out.append(Case("c06_2d_obstacle", 2, (60, 24), sup,
                    {"load_a": (lp["load_a"], ((-pull, -1.0),)), "load_b": (lp["load_b"], ((pull, -1.0),))},
                    0.15, 2.0, True, "MMA", void=(((28, 32), (0, 18)),),
                    note="c01 with a void wall between the spans (gap only above y = 18)", tags=("2d", "obstacle")))
    out.append(Case("c07_2d_L_domain", 2, (48, 48),
                    {"support_a": ((4, 14), (47, 48)), "support_b": ((47, 48), (4, 14))},
                    {"load_a": (((7, 11), (36, 40)), ((-1.0, -0.2),)), "load_b": (((36, 40), (7, 11)), ((-0.2, -1.0),))},
                    0.12, 2.0, True, "MMA", outside=((20, 48), (20, 48)),
                    note="L-shaped part; each load pad next to its own support at the two arm ends",
                    tags=("2d", "L")))
    out.append(Case("c08_2d_asym_weak_OC", 2, (60, 24), sup,
                    {"load_a": (lp["load_a"], ((0.0, -1.0),)), "load_b": (lp["load_b"], ((0.005, -0.25),))},
                    0.15, 2.5, False, "OC", note="asymmetric loads 1 / 0.25, pull 0.005 on one side, OC, no projection",
                    tags=("2d", "asym")))
    out.append(Case("c09_2d_multiload", 2, (60, 24), sup,
                    {"load_a": (lp["load_a"], ((0.0, -1.0), (0.0, 0.0), (0.3, 0.0))),
                     "load_b": (lp["load_b"], ((0.0, 0.0), (0.0, -1.0), (-0.3, 0.0)))},
                    0.15, 2.0, True, "MMA", note="3 load cases: each pad alone, then a squeeze", tags=("2d", "multi")))
    out.append(Case("c10_2d_one_support", 2, (60, 24),
                    {"support": ((27, 33), (0, 1))},
                    {"load_a": (((2, 6), (18, 22)), ((0.0, -1.0),)), "load_b": (((54, 58), (18, 22)), ((0.0, -1.0),))},
                    0.12, 2.0, True, "MMA", note="control: one support, two loads (no reason to split)",
                    tags=("2d", "1support")))
    # ---- 3-D (z up)
    out.append(Case("c11_3d_split", 3, (30, 4, 12),
                    {"support_a": ((2, 5), (0, 4), (0, 1)), "support_b": ((25, 28), (0, 4), (0, 1))},
                    {"load_a": (((3, 5), (0, 4), (5, 7)), ((-pull, 0.0, -1.0),)),
                     "load_b": (((25, 27), (0, 4), (5, 7)), ((pull, 0.0, -1.0),))},
                    0.15, 1.5, True, "MMA", note="3-D two-span split benchmark (as split_benchmark(3, 12))",
                    tags=("3d", "far")))
    out.append(Case("c12_3d_four_supports", 3, (24, 16, 8),
                    {f"support_{k}": ((x, x + 4), (y, y + 4), (0, 1))
                     for k, (x, y) in zip("abcd", ((0, 0), (20, 0), (20, 12), (0, 12)))},
                    {"load_a": (((0, 4), (0, 4), (7, 8)), ((0.0, 0.0, -1.0),)),
                     "load_c": (((20, 24), (12, 16), (7, 8)), ((0.0, 0.0, -1.0),))},
                    0.10, 1.5, True, "MMA", max_iter=120,
                    note="4 corner supports, 2 loads above two diagonal supports; the other supports carry nothing",
                    tags=("3d", "4supports")))
    out.append(Case("c13_3d_obstacle_three_loads", 3, (32, 8, 12),
                    {"support_a": ((1, 5), (0, 8), (0, 1)), "support_b": ((27, 31), (0, 8), (0, 1))},
                    {"load_a": (((2, 4), (3, 5), (6, 8)), ((0.0, 0.0, -1.0),)),
                     "load_b": (((15, 17), (3, 5), (10, 12)), ((0.0, 0.0, -1.0),)),
                     "load_c": (((28, 30), (3, 5), (6, 8)), ((0.0, 0.0, -1.0),))},
                    0.12, 2.0, False, "MMA", void=(((10, 13), (0, 8), (0, 8)), ((19, 22), (0, 8), (0, 8))),
                    max_iter=120, note="3 loads, two void walls between them, density filter only",
                    tags=("3d", "obstacle", "3loads")))
    out.append(Case("c14_3d_close_OC", 3, (24, 6, 12),
                    {"support_a": ((6, 10), (0, 6), (0, 1)), "support_b": ((14, 18), (0, 6), (0, 1))},
                    {"load_a": (((7, 9), (2, 4), (6, 8)), ((-pull, 0.0, -1.0),)),
                     "load_b": (((15, 17), (2, 4), (6, 8)), ((pull, 0.0, -1.0),))},
                    0.20, 2.0, True, "OC", max_iter=120, note="3-D pads close (gap 4), OC with projection",
                    tags=("3d", "close")))
    return out


CASES = {c.name: c for c in cases()}
FAST = ("c01_2d_split_far", "c07_2d_L_domain")

# ============================================================================================== methods
METHODS = ("plain", "default", "ties", "conduction")


def _optimizer(name: str, d: Domain) -> Any:
    return OC(dv_scale=d.n_active) if name == "OC" else MMA()


def run(case: Case, method: str, optimizer: str | None = None, max_iter: int | None = None,
        label: str | None = None) -> dict[str, Any]:
    """Optimise one scenario with one method and gate the result. Methods with a connectivity constraint need MMA
    (OC handles one constraint): for an OC scenario they run with MMA and say so in ``optimizer``."""
    b = build(case)
    d0 = b.domain
    plan = None
    if method == "plain":
        d, extra = d0, []
    elif method == "default":
        plan = one_body(d0, b.interfaces)
    elif method == "ties":
        plan = one_body(d0, b.interfaces, ties=1, constraint=None)
    elif method == "conduction":
        plan = one_body(d0, b.interfaces, ties=None, constraint="conduction", eta=0.7)
    else:
        raise ValueError(method)
    if plan is not None:
        d, extra = plan.domain, list(plan.constraints)
    opt = optimizer or case.optimizer
    if extra and opt == "OC":
        opt = "MMA"
    chain = standard_chain(d, case.rmin, projection=case.projection, beta=1.0)
    sched = BetaContinuation(chain, 1.0, 16.0, every=20) if case.projection else None
    st = Study(d, Elasticity(d, b.fixed), b.F, chain, SIMP(d, penal=3.0), Compliance(),
               [Constraint(Volume(), case.vf)] + extra, _optimizer(opt, d), sched,
               move_beta_ref=2.0 if case.projection else None)
    t0 = time.perf_counter()
    res = st.run(max_iter=max_iter or case.max_iter)
    sec = time.perf_counter() - t0
    rep = analyse_connectivity(d, res.xphys, b.interfaces, voids="warn")
    vals = res.final.values
    out: dict[str, Any] = {
        "case": case.name, "method": label or method, "optimizer": opt, "dim": case.dim, "shape": list(case.shape),
        "n_design": int(d.n_design), "vf": case.vf, "rmin": case.rmin, "projection": case.projection,
        "iterations": res.iterations, "converged": bool(res.converged), "seconds": round(sec, 2),
        "compliance": float(res.objective), "volume": float(vals["volume"]),
        "volume_feasible": bool(vals["volume"] <= case.vf * 1.01),
        "verdict": rep.verdict, "verdict_voids_fail": "FAIL" if (rep.verdict == "PASS" and rep.voids_matter
                                                               and rep.enclosed_voids) else rep.verdict,
        "n_bodies": rep.n_bodies, "connected": bool(rep.connected), "unattached": rep.unattached,
        "floating": len(rep.floating), "enclosed_voids": len(rep.enclosed_voids),
        "checks": {c.label: {"bodies": c.n_bodies, "connected": bool(c.connected)} for c in rep.checks},
        "min_neck_cells": None if rep.min_neck is None else rep.min_neck.cells, "reasons": rep.reasons,
        "tie_cells": None if plan is None or plan.ties is None else int(plan.ties.sum()),
    }
    if "connectivity" in vals:
        bound = plan.constraints[0].upper
        out.update({"connectivity_value": float(vals["connectivity"]), "connectivity_bound": float(bound),
                    "connectivity_feasible": bool(vals["connectivity"] <= bound * 1.01)})
    fin = plan if plan is not None else one_body(d0, b.interfaces, ties=None, constraint=None)
    xr, rep2, br = fin.finish(res.xphys, voids="warn")
    out.update({"after_finish": rep2.verdict, "repaired": br is not None,
                "repair_cells": None if br is None else br.added_cells})
    if br is not None:                       # repaired design: compliance (same β) and volume, not re-optimised
        with torch.no_grad():
            E = st.interp(torch.as_tensor(xr))
            U = st.physics.solve(E.reshape(-1)[torch.as_tensor(d.active_elements)], b.F)
            out["repair_compliance"] = float((torch.as_tensor(b.F) * U).sum())
        out["repair_volume"] = float(xr[d.active].sum() / d.n_active)
    rec = os.environ.get("TOPORLOGY_SWEEP_RECORD")
    if rec:
        with open(rec, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")
    out["_xphys"], out["_domain"], out["_interfaces"] = res.xphys, d, b.interfaces     # not recorded
    out["_history"] = res.history
    return out


def _line(r: dict[str, Any]) -> str:
    ch = " ".join(f"{k}:{'ok' if v['connected'] else 'SPLIT'}" for k, v in r["checks"].items())
    return (f"  {r['case']:<28} {r['method']:<10} {r['optimizer']:<3} it={r['iterations']:<3} {r['seconds']:6.1f} s  "
            f"c={r['compliance']:.4g} V={r['volume']:.4f}  {r['verdict']:<7} bodies={r['n_bodies']} {ch}")


# ============================================================================================== tests
def test_case_builders_are_consistent():
    """Every scenario builds; pads lie in the part; forces sum to the intended resultant per case."""
    assert len(CASES) >= 12 and {c.dim for c in CASES.values()} == {2, 3}
    for c in CASES.values():
        b = build(c)
        tot = b.F.reshape(b.domain.nn, b.domain.dim, -1).sum(axis=0)          # (dim, n_cases)
        want = np.sum([np.asarray(f, float).T for _, f in c.loads.values()], axis=0)
        assert np.allclose(tot, want), c.name
        assert not np.any(b.F[b.fixed]), c.name


@pytest.mark.parametrize("name", FAST)
def test_fast_plain_splits_default_plan_passes(name):
    """Fast subset: plain compliance TO falls apart, the default one_body plan gives one robust body."""
    c = CASES[name]
    plain = run(c, "plain")
    dflt = run(c, "default")
    print("\n" + _line(plain) + "\n" + _line(dflt))
    assert plain["verdict"] != "PASS"
    assert dflt["verdict"] == "PASS", dflt["reasons"]
    assert dflt["volume_feasible"]


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_sweep_connectivity(name):
    """Full sweep (all methods). Asserts only what the library guarantees: the default plan followed by
    ``plan.finish`` (gate + bridge repair) ends PASS. Pass rates per method are reported, not asserted."""
    c = CASES[name]
    rows = [run(c, m) for m in METHODS]
    if c.optimizer == "OC":
        rows.append(run(c, "plain", optimizer="MMA", label="plain_mma"))
    print("\n" + "\n".join(_line(r) for r in rows))
    dflt = next(r for r in rows if r["method"] == "default")
    assert dflt["after_finish"] == "PASS"


# ============================================================================================== regression (finish)
def test_finish_repairs_fragile_and_split_designs_to_pass():
    """Regression (found by the sweep, 14 Nov 2026): ``OneBodyPlan.finish`` bridged only the nominal bodies with
    section 2, which the one-element erosion of its own gate removes, and added nothing to a FRAGILE design. Before
    the fix both designs below ended FRAGILE; now both end PASS."""
    shape = (24, 12)
    d = Domain(shape)
    a, b = np.zeros(shape, bool), np.zeros(shape, bool)
    a[0:2], b[22:24] = True, True
    ifc = {"a": a, "b": b}
    plan = one_body(d, ifc, ties=None, constraint=None)
    x = np.zeros(shape)
    x[0:8], x[16:24] = 1.0, 1.0
    x[8:16, 5:7] = 1.0                                       # 2-wide link: one body, splits when eroded
    assert plan.gate(x).verdict == "FRAGILE"
    xr, rep, br = plan.finish(x)
    assert rep.verdict == "PASS" and br is not None and br.added_cells > 0
    assert np.all(xr[br.added] == 1.0)
    x2 = np.zeros(shape)
    x2[0:8], x2[16:24] = 1.0, 1.0                            # two bodies
    x2[8:16, 3:9] = 0.6                                      # grey: one body at 0.5, split at 0.7
    assert plan.gate(x2).verdict == "FRAGILE"
    assert plan.finish(x2)[1].verdict == "PASS"
    x3 = np.zeros(shape)
    x3[0:8], x3[16:24] = 1.0, 1.0                            # split at the nominal threshold
    xr3, rep3, br3 = plan.finish(x3)
    assert rep3.verdict == "PASS" and br3.n_bridges >= 1
