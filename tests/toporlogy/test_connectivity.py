"""Connectivity: detection gate, min cut, ties, bridges, differentiable connectivity responses, split scenario.

Pre-registered: gradient checks ≤ 1e-6 relative (float64); detection results exact (synthetic masks with known
answers); scenario thresholds stated in each test.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
import torch

from toporlogy import (SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume,
                          standard_chain, study_gradcheck)
from toporlogy.optimization.connectivity import (ConductionConnectivity, CouplingCompliance, analyse_connectivity, bridge,
                                       elements_touching_nodes, label_bodies, min_cut_cells, split_benchmark,
                                       tie_regions, with_keep)
from toporlogy.optimization.connectivity import _nodes_of
from toporlogy.fem.heat import ConductivityInterp, HeatConduction
from toporlogy.optimizers.mma import MMA

TOL = 1e-6
# Central-difference step of the response gradient checks. The extra solves (clamped coupling cases, eroded
# realisation) are less well conditioned than the main problem: measured on CouplingCompliance 2-D, the FD error
# falls as 1/h (roundoff of the solves) — 9.0e-6, 2.0e-7, 1.9e-8 at h = 1e-6, 1e-5, 1e-4 (PARDISO; SuperLU
# 3.7e-6, 7.4e-7, 5.3e-8) — so the truncation error is negligible at h = 1e-4 and that step is used.
FD_STEP = 1e-4


def _box(shape, *slices):
    m = np.zeros(shape, bool)
    for s in slices:
        m[s] = True
    return m


# ============================================================================================== detection
def test_two_blocks_fail_and_interface_map():
    shape = (20, 10)
    d = Domain(shape)
    x = _box(shape, np.s_[1:7, 2:8], np.s_[12:19, 2:8]).astype(float)
    ifc = {"left": _box(shape, np.s_[1:3, 3:5]), "right": _box(shape, np.s_[17:19, 3:5])}
    r = analyse_connectivity(d, x, ifc)
    assert r.n_bodies == 2 and not r.connected and r.verdict == "FAIL"
    assert r.body_cells == [42, 36]                          # sorted by size: right block (7×6) is #1
    assert r.interface_bodies == {"left": [2], "right": [1]}
    s = r.summary().splitlines()
    assert s[0].startswith("Connectivity: FAIL") and s[-1].startswith("Result: FAIL")
    print("\n" + r.summary())


def test_face_versus_vertex_contact_3d():
    shape = (6, 6, 6)
    d = Domain(shape)
    x = _box(shape, np.s_[0:3, 0:3, 0:3], np.s_[3:6, 3:6, 3:6]).astype(float)     # touch at one corner
    ifc = {"a": _box(shape, np.s_[0, 0, 0]), "b": _box(shape, np.s_[5, 5, 5])}
    assert analyse_connectivity(d, x, ifc).n_bodies == 2
    assert analyse_connectivity(d, x, ifc, connectivity="vertex").n_bodies == 1
    assert analyse_connectivity(d, x, ifc, connectivity=18).n_bodies == 2         # corner ≠ edge
    x2 = _box(shape, np.s_[0:3, 0:3, :], np.s_[3:6, 3:6, :]).astype(float)        # share an edge line
    assert analyse_connectivity(d, x2, ifc, connectivity="edge").n_bodies == 1
    assert analyse_connectivity(d, x2, ifc, connectivity="face").n_bodies == 2


def test_enclosed_void_3d():
    shape = (12, 12, 12)
    d = Domain(shape)
    x = _box(shape, np.s_[1:11, 1:11, 1:11])
    x[5:7, 5:7, 5:7] = False                                 # 8-cell cavity
    x[1:3, 5, 5] = False                                     # a dead-end notch open to the outside: not enclosed
    ifc = {"bottom": _box(shape, np.s_[1:11, 1:11, 1]), "top": _box(shape, np.s_[1:11, 1:11, 10])}
    r = analyse_connectivity(d, x.astype(float), ifc)
    assert r.enclosed_voids == [8] and r.verdict == "FAIL" and r.connected
    r2 = analyse_connectivity(d, x.astype(float), ifc, voids="warn")
    assert r2.enclosed_voids == [8] and r2.verdict == "PASS"
    # 2-D closed holes are through holes: reported, not failed
    d2 = Domain((10, 10))
    x2 = _box((10, 10), np.s_[1:9, 1:9])
    x2[4:6, 4:6] = False
    r3 = analyse_connectivity(d2, x2.astype(float), {"a": _box((10, 10), np.s_[1, 1:9]),
                                                     "b": _box((10, 10), np.s_[8, 1:9])})
    assert r3.enclosed_voids == [4] and not r3.voids_matter and r3.verdict == "PASS"


@pytest.mark.parametrize("width", [1, 2, 3])
def test_neck_min_cut_2d(width):
    shape = (24, 12)
    d = Domain(shape)
    x = _box(shape, np.s_[0:8, 0:12], np.s_[16:24, 0:12], np.s_[8:16, 5:5 + width])
    ifc = {"a": _box(shape, np.s_[0:2, 0:12]), "b": _box(shape, np.s_[22:24, 0:12])}
    r = analyse_connectivity(d, x.astype(float), ifc)
    assert r.connected and r.n_bodies == 1
    neck = r.min_neck
    assert neck is not None and neck.cells == width and neck.area == pytest.approx(width * d.h * d.thickness)
    assert len(neck.where) == width
    assert x.ravel()[neck.where].all()
    # erosion by one element removes bars of width ≤ 2 → FRAGILE; width 3 survives → PASS
    assert r.verdict == ("PASS" if width >= 3 else "FRAGILE")


def test_neck_min_cut_3d_section():
    shape = (16, 8, 8)
    d = Domain(shape)
    x = _box(shape, np.s_[0:6], np.s_[10:16], np.s_[6:10, 3:5, 3:5])               # 2×2 bar
    ifc = {"a": _box(shape, np.s_[0]), "b": _box(shape, np.s_[15])}
    r = analyse_connectivity(d, x.astype(float), ifc)
    assert r.min_neck.cells == 4 and r.min_neck.area == pytest.approx(4 * d.h ** 2)


def test_min_cut_matches_brute_force():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(60):
        s = rng.random((6, 5)) < 0.75
        s[0, :] = s[-1, :] = True
        a, b = _box(s.shape, np.s_[0, :]), _box(s.shape, np.s_[-1, :])
        val, cut = min_cut_cells(s, a, b)
        cells = np.flatnonzero((s & ~a & ~b).ravel())

        def connected(sol):
            lab, _ = label_bodies(sol)
            return bool(np.intersect1d(lab[a & sol], lab[b & sol]).size)
        if not connected(s):
            assert val == 0
            continue
        best = None
        for k in range(4):
            for sub in itertools.combinations(cells, k):
                t = s.copy().ravel()
                t[list(sub)] = False
                if not connected(t.reshape(s.shape)):
                    best = k
                    break
            if best is not None:
                break
        if best is None:
            continue                                         # min cut > 3: brute force too expensive, skip
        assert val == best and len(cut) == best
        t = s.copy().ravel()
        t[cut] = False
        assert not connected(t.reshape(s.shape))
        checked += 1
    assert checked >= 20


def test_threshold_robustness_and_grey_link():
    shape = (24, 10)
    d = Domain(shape)
    x = _box(shape, np.s_[0:8], np.s_[16:24]).astype(float)
    ifc = {"a": _box(shape, np.s_[0:2]), "b": _box(shape, np.s_[22:24])}
    xg = x.copy()
    xg[8:16, 3:7] = 0.4                                      # grey film: one body only at threshold 0.3
    r = analyse_connectivity(d, xg, ifc)
    assert r.verdict == "FAIL" and r.n_bodies == 2
    assert {c.label: c.connected for c in r.checks}["t=0.3"]
    xg[8:16, 3:7] = 0.6                                      # connected at 0.5, gone at 0.7 → FRAGILE
    r = analyse_connectivity(d, xg, ifc)
    assert r.passed and r.fragile and r.verdict == "FRAGILE"
    assert not {c.label: c.connected for c in r.checks}["t=0.7"]


def test_unattached_interface_and_floating_island():
    shape = (16, 8)
    d = Domain(shape)
    x = _box(shape, np.s_[0:10, 0:8], np.s_[13:15, 2:4]).astype(float)              # main block + island
    ifc = {"a": _box(shape, np.s_[0, :]), "pad": _box(shape, np.s_[11:13, 6:8])}    # pad lies in void
    r = analyse_connectivity(d, x, ifc)
    assert r.unattached == ["pad"] and r.verdict == "FAIL" and r.floating == [2]
    assert r.interface_solid_fraction["pad"] == 0.0


def test_elements_touching_nodes():
    d = Domain((4, 3))
    m = elements_touching_nodes(d, d.nodes_where(lambda X: X[:, 1] < 1e-9))
    assert m[:, 0].all() and not m[:, 1:].any()


# ============================================================================================== ties and bridges
def test_tie_regions_connect_and_avoid_void():
    shape = (30, 16)
    void = _box(shape, np.s_[12:14, 0:13])                   # wall with a gap at the top
    keep_a, keep_b, keep_c = _box(shape, np.s_[2:4, 2:4]), _box(shape, np.s_[26:28, 2:4]), \
        _box(shape, np.s_[26:28, 12:14])
    d = Domain(shape, keep=keep_a | keep_b | keep_c, void=void)
    groups = {"a": keep_a, "b": keep_b, "c": keep_c}
    tie = tie_regions(d, groups, section=1)
    assert not (tie & d.void).any()
    d2 = with_keep(d, tie)
    r = analyse_connectivity(d2, d2.keep.astype(float), groups, erosion=False)
    assert r.connected and r.n_bodies == 1
    tie2 = tie_regions(d, groups, section=2)
    assert tie2.sum() > tie.sum()
    r2 = analyse_connectivity(d, (d.keep | tie2).astype(float), groups, erosion=False)
    assert r2.connected and r2.min_neck.cells >= 2
    print(f"\n  tie cells: section 1 -> {int(tie.sum())}, section 2 -> {int(tie2.sum())}")


def test_tie_straight_path_length():
    shape = (20, 9)
    a, b = _box(shape, np.s_[0:2, 4]), _box(shape, np.s_[18:20, 4])
    d = Domain(shape, keep=a | b)
    tie = tie_regions(d, {"a": a, "b": b}, section=1)
    assert tie.sum() == 16 and tie[2:18, 4].all()            # the straight segment between the pads
    tie3 = tie_regions(d, {"a": a, "b": b}, section=3)
    assert tie3[2:18, 3:6].all() and tie3.sum() == 56          # 3×16 plus the box overhang at the pads


def test_bridge_repairs_to_one_body():
    shape = (40, 20)
    d = Domain(shape)
    rng = np.random.default_rng(2)
    x = 0.2 * rng.random(shape)
    x[2:10, 2:18] = 1.0
    x[30:38, 2:18] = 1.0
    x[18:21, 8:11] = 1.0                                     # floating island between them (on the way)
    x[18:20, 17:19] = 0.9                                    # floating island off the way
    ifc = {"a": _box(shape, np.s_[2:4, 2:18]), "b": _box(shape, np.s_[36:38, 2:18]),
           "pad": _box(shape, np.s_[34:36, 18:20])}         # pad lies in void (touching b) → forced solid
    before = analyse_connectivity(d, x, ifc)
    assert before.verdict == "FAIL"
    res = bridge(d, x, ifc, section=3)
    after = analyse_connectivity(d, res.xphys, ifc)
    print(f"\n  bridge: {res.n_bridges} bridges, {res.added_cells} cells added, {res.removed_cells} removed, "
          f"forced {int(res.forced.sum())}; after: {after.verdict}")
    assert after.connected and after.n_bodies == 1 and after.verdict == "PASS"
    assert res.forced.sum() == 4 and res.removed_cells == 4  # the off-path island (2×2) is dropped
    assert res.added_cells < 140                             # 20-cell gap × section 3 plus the pad link
    assert (res.xphys[res.added] == 1.0).all()
    rb = bridge(d, x, ifc, section=1, binary=True)
    assert set(np.unique(rb.xphys)) <= {0.0, 1.0} and rb.added_cells < res.added_cells
    assert analyse_connectivity(d, rb.xphys, ifc, erosion=False).connected


def test_bridge_unreachable_raises():
    shape = (20, 10)
    void = _box(shape, np.s_[9:11, :])
    d = Domain(shape, void=void)
    x = _box(shape, np.s_[0:5], np.s_[15:20]).astype(float)
    with pytest.raises(RuntimeError, match="cannot be reached"):
        bridge(d, x, {"a": _box(shape, np.s_[0]), "b": _box(shape, np.s_[19])})


def test_bridge_interface_entirely_in_void_raises_clearly():
    """B-08: an interface with no cell in the design domain made the first body's label 0 (the background); here
    that gave a misleading "bodies [1] cannot be reached" (an empty-argmax error once the loop ends). It is now a
    ValueError that names the interface."""
    shape = (20, 10)
    void = _box(shape, np.s_[0:3, :])
    d = Domain(shape, void=void)
    x = _box(shape, np.s_[5:8], np.s_[15:20]).astype(float)
    with pytest.raises(ValueError, match=r"interface\(s\) \['a'\] have no cell"):
        bridge(d, x, {"a": _box(shape, np.s_[0]), "b": _box(shape, np.s_[19])})


# ============================================================================================== responses
def _small_problem(dim):
    if dim == 2:
        shape = (14, 7)
        mask = np.ones(shape, bool)
        mask[10:, 5:] = False
        d0 = Domain(shape, 1.0, frac=mask.astype(float))
    else:
        shape = (7, 4, 4)
        d0 = Domain(shape, 0.5)
    up = dim - 1
    ex = np.indices(shape)
    ifc = {"s1": (ex[0] <= 1) & (ex[up] == 0), "s2": (ex[0] >= shape[0] - 2) & (ex[up] == 0),
           "p1": (ex[0] == shape[0] // 2) & (ex[up] == shape[up] - 1)}
    keep = ifc["s1"] | ifc["s2"] | ifc["p1"]
    d = Domain(shape, d0.h, frac=d0.frac, keep=keep)
    fixed_nodes = d.nodes_where(lambda X: (X[:, up] < 1e-9) & ((X[:, 0] < 1e-9) | (X[:, 0] > shape[0] * d.h - 1e-9)))
    fixed = d.dofs(fixed_nodes)
    F = d.load_vector(1)
    top = d.nodes_where(lambda X: X[:, up] > shape[up] * d.h - 1e-9)
    F[d.dim * top + up, 0] = -1.0 / len(top)
    return d, ifc, fixed, F


def _study(d, fixed, F, resp, bound=10.0):
    chain = standard_chain(d, 1.5, beta=3.0)
    return Study(d, Elasticity(d, fixed), F, chain, SIMP(d, penal=3.0), Compliance(),
                 [Constraint(Volume(), 0.4), Constraint(resp, bound)], MMA())


@pytest.mark.parametrize("dim,eta,agg", [(2, None, "pnorm"), (2, 0.7, "ks"), (3, None, "pnorm"),
                                         (3, 0.65, "mean")])
def test_gradcheck_conduction_connectivity(dim, eta, agg):
    d, ifc, fixed, F = _small_problem(dim)
    resp = ConductionConnectivity(d, ifc, sink="s1", eta=eta, aggregate=agg, p=8.0 if agg == "pnorm" else 0.05)
    st = _study(d, fixed, F, resp)
    rng = np.random.default_rng(dim)
    x = 0.2 + 0.6 * rng.random(st.n_var)
    r = study_gradcheck(st, x, which=1, n_dirs=5, h=FD_STEP)
    print(f"\n  gradcheck ConductionConnectivity {dim}-D eta={eta} {agg}: {r['max_rel_error']:.2e} (g = {r['f']:.3g})")
    assert r["max_rel_error"] <= TOL


@pytest.mark.parametrize("dim,eta", [(2, None), (2, 0.7), (3, None)])
def test_gradcheck_coupling_compliance(dim, eta):
    d, ifc, fixed, F = _small_problem(dim)
    resp = CouplingCompliance(d, ifc, held="s1", eta=eta)
    st = _study(d, fixed, F, resp, bound=50.0)
    rng = np.random.default_rng(10 + dim)
    x = 0.2 + 0.6 * rng.random(st.n_var)
    r = study_gradcheck(st, x, which=1, n_dirs=5, h=FD_STEP)
    print(f"\n  gradcheck CouplingCompliance {dim}-D eta={eta}: {r['max_rel_error']:.2e} (g = {r['f']:.3g})")
    assert r["max_rel_error"] <= TOL


def _hand_designs():
    """Benchmark designs built by hand: two columns (split) and the same plus a bridge (one body)."""
    b = split_benchmark(2, 24)
    d = b.domain
    x = d.keep.astype(float)
    for m in (b.interfaces["load_a"], b.interfaces["load_b"]):
        cols = np.flatnonzero(m.any(axis=1))
        top = np.flatnonzero(m.any(axis=0)).max()
        x[cols.min() - 1:cols.max() + 2, :top + 1] = 1.0
    xb = bridge(d, x, b.interfaces, section=3).xphys
    return b, x, xb


def test_conduction_ratios_separate_split_from_connected():
    b, xs, xb = _hand_designs()
    assert analyse_connectivity(b.domain, xs, b.interfaces).n_bodies == 2
    resp = ConductionConnectivity(b.domain, b.interfaces, sink="support_a")
    rs, rb = resp.ratios_of(xs), resp.ratios_of(xb)
    print(f"\n  conduction ratios split {rs}\n  conduction ratios bridged {rb}")
    assert min(rs["support_b"], rs["load_b"]) > 100 * max(rb.values())
    assert max(rb.values()) < 20


def test_sinks_on_all_supports_do_not_see_the_split():
    """Why one sink: with T = 0 on both supports, the split design (each pad on its own support) looks connected."""
    b, xs, xb = _hand_designs()
    d = b.domain
    nodes = np.concatenate([_nodes_of(d, b.interfaces[k]) for k in ("support_a", "support_b")])
    heat = HeatConduction(d, nodes)
    k = ConductivityInterp(d)
    Q = np.zeros((d.nn, 2))
    for c, name in enumerate(("load_a", "load_b")):
        nn = _nodes_of(d, b.interfaces[name])
        Q[nn, c] = 1.0 / len(nn)
    R = []
    for x in (xs, xb):
        kk = k(torch.as_tensor(x)).reshape(-1)[torch.as_tensor(d.active_elements)].numpy()
        T = heat.solve_numpy(kk, Q)
        R.append((Q * T).sum(axis=0))
    print(f"\n  two-sink resistances: split {R[0]}, bridged {R[1]}")
    assert R[0].max() < 2.0 * R[1].max()                     # indistinguishable: the measure misses the split


# ============================================================================================== scenario
def _run(b, variant, iters, vf=0.15, rmin=2.0, bound=None, eta=0.7):
    d = b.domain
    if variant == "ties":
        d = with_keep(d, tie_regions(d, b.interfaces, section=1))
    chain = standard_chain(d, rmin, beta=1.0)
    cons = [Constraint(Volume(), vf)]
    if variant == "conduction":
        cons.append(Constraint(ConductionConnectivity(d, b.interfaces, sink="support_a", eta=eta), bound))
    elif variant == "coupling":
        # bound relative to the start design: ratios are 1 there, the p-norm of n cases is n^(1/8)
        cons.append(Constraint(CouplingCompliance(d, b.interfaces, held="support_a", eta=eta, reference="first"),
                               bound))
    st = Study(d, Elasticity(d, b.fixed), b.loads, chain, SIMP(d, penal=3.0), Compliance(), cons, MMA(),
               BetaContinuation(chain, 1.0, 16.0, every=25))
    t0 = time.perf_counter()
    res = st.run(max_iter=iters)
    sec = time.perf_counter() - t0
    # main-problem compliance of the final physical design (same β) — the comparable number
    with torch.no_grad():
        E = st.interp(torch.as_tensor(res.xphys))
        U = st.physics.solve(E.reshape(-1)[torch.as_tensor(d.active_elements)], b.loads)
        c = float((torch.as_tensor(b.loads) * U).sum())
    rep = analyse_connectivity(d, res.xphys, b.interfaces)
    print(f"\n  [{variant}] it={res.iterations} {sec:.1f} s  compliance={c:.4g}  vol={res.history['volume'][-1]:.4f}  "
          f"verdict={rep.verdict} bodies={rep.n_bodies} "
          + " ".join(f"{ch.label}:{ch.n_bodies}/{'ok' if ch.connected else 'SPLIT'}" for ch in rep.checks))
    return res, rep, c


def test_split_scenario_plain_ties_bridge_2d():
    """Plain compliance TO splits (≥ 2 bodies); frozen ties and bridge repair give one body with every interface."""
    b = split_benchmark(2, 16)
    res, rep, c_plain = _run(b, "plain", 80, rmin=1.5)
    assert rep.n_bodies >= 2 and not rep.connected
    rb = bridge(b.domain, res.xphys, b.interfaces, section=3)
    after = analyse_connectivity(b.domain, rb.xphys, b.interfaces)
    print(f"  [bridge] +{rb.added_cells} cells, {rb.n_bridges} bridges -> {after.verdict}, {after.n_bodies} body")
    assert after.connected and after.n_bodies == 1
    _, rep_t, c_t = _run(b, "ties", 80, rmin=1.5)
    assert rep_t.connected and rep_t.n_bodies == 1
    print(f"  compliance ratio ties/plain = {c_t / c_plain:.3f}")


@pytest.mark.slow
@pytest.mark.parametrize("dim", [2, 3])
def test_split_scenario_all_measures(dim):
    """Full comparison (plain, ties, conduction, coupling, all at the same volume) — reported numbers.

    B-20 (diagnosed 18 Nov 2026): with 100 iterations in 3-D the last beta step (to 16) lands on iteration 100; each
    beta step drives the coupling measure far above its bound (18, 122, 75 after the steps to 2, 4, 8) and at
    iteration 100 it was 11.0 against 2.5, so the returned design was infeasible and carried a 7-cell island. With
    the budget below MMA converges at iteration 219 with both constraints active and one body. The claim "keeps the
    part in one body" is only meaningful for a feasible design, so feasibility is asserted first. That assertion
    then showed the same in 2-D, where the old test passed: with 120 iterations coupling ended at 2.93 > 2.5
    (max_iter). With 300 all four 2-D variants converge feasible (plain 84, ties 114, conduction 205, coupling 288)."""
    if dim == 2:
        b, iters, rmin, bc, bk = split_benchmark(2, 24), 300, 2.0, 20.0, 2.5
    else:
        b, iters, rmin, bc, bk = split_benchmark(3, 12), 300, 1.5, 20.0, 2.5
    out = {}
    for v, bound in (("plain", None), ("ties", None), ("conduction", bc), ("coupling", bk)):
        out[v] = _run(b, v, iters, rmin=rmin, bound=bound)
    for v in ("plain", "ties", "conduction", "coupling"):
        res = out[v][0]
        assert res.feasible, f"{v}: returned design infeasible {res.constraint_values} ({res.termination})"
    assert out["plain"][1].n_bodies >= 2 and not out["plain"][1].connected
    for v in ("ties", "conduction", "coupling"):
        assert out[v][1].connected and out[v][1].n_bodies == 1, v
    for v in ("ties", "conduction", "coupling"):
        print(f"  {dim}-D compliance {v}/plain = {out[v][2] / out['plain'][2]:.3f}")
