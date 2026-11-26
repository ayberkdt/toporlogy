"""Buckling reliability and geometry transfer: pre-registered checks of docs/olcumler/burkulma_geometri_onkayit.md
(A1–A6, SP1–SP3, RL1–RL3, GC1–GC9, VC1, VC3) and unit tests of the new pieces.

The problems are in ``_buckling_cases.py`` (shared with ``docs/olcumler/burkulma_geometri.py``, which writes the
record ``burkulma_geometri.json``). Thresholds are the pre-registered ones; a failed check is a strict xfail with
the measured value in its reason.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import _buckling_cases as C
import toporlogy as T
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.optimization.buckling import BucklingKS, reliability
from toporlogy.post.verify import CONVERGED, NOT_CONVERGED, assess_convergence


# ================================================================================================ units
def test_reliability_flags_from_the_diagnostics() -> None:
    base = {"cluster_cut": False, "low_density_share": [0.1], "stress_energy_share": [0.2], "localization": [0.05],
            "mu_next": 0.1}
    r = reliability(np.array([0.5, 0.2]), base)
    assert r.reliable and r.flags == () and r.load_factor_1 == pytest.approx(2.0)
    assert not reliability(np.array([0.5]), {**base, "cluster_cut": True}).reliable
    s = reliability(np.array([0.5]), {**base, "low_density_share": [0.5]})
    assert not s.reliable and s.flags == ("spurious_low_density",)
    n = reliability(np.array([-0.1]), base)
    assert not n.reliable and "not_buckling" in n.flags and n.load_factor_1 == float("inf")
    loc = reliability(np.array([0.5]), {**base, "localization": [0.9]})
    assert loc.reliable and loc.flags == ("localized",)            # reported, does not lower ``reliable``


def test_stress_energy_share_limits() -> None:
    d = T.Domain((4, 30))
    X = d.node_coords()
    top = np.flatnonzero(np.isclose(X[:, 1], 30))
    F = d.load_vector()
    F[2 * top + 1, 0] = -1.0 / len(top)
    m = T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 1] == 0)), 0.3)
    E = np.full(d.n_active, 1000.0)
    U = m.solve_numpy(E, F)[:, 0]
    lb = LinearBuckling(m)
    _, modes = lb.modes(E, E, U, 3)
    assert np.allclose(lb.stress_energy_share(E, U, modes, np.full(d.n_active, 0.9)), 0.0)
    assert np.allclose(lb.stress_energy_share(E, U, modes, np.full(d.n_active, 0.1)), 1.0)
    x = np.where(d.element_centers()[d.active_elements, 1] > 15, 0.2, 0.9)
    s = lb.stress_energy_share(E, U, modes, x)
    assert np.all((s > 0) & (s < 1))


def test_tracking_of_a_repeated_evaluation_is_the_identity() -> None:
    st, d = C.d2_study()
    resp = BucklingKS(n_modes=4, rho=30.0)
    x = torch.as_tensor(np.full(d.n_design, 0.4))
    resp(st.context(x, True))
    assert resp.last_tracking is None and resp.last_reliability is not None
    resp(st.context(x, True))
    tr = resp.last_tracking
    assert tr is not None and (tr.order == np.arange(4)).all() and tr.mac.min() >= 1 - 1e-8
    resp.reset_tracking()
    assert resp.last_tracking is None


def test_voxel_reanalysis_prismatic_bar_is_exact() -> None:
    """ν = 0, uniform traction: the H8 solution of a prismatic bar is exact, C = L/(EA), in both backends."""
    pytest.importorskip("skfem")
    from toporlogy.post.reanalysis import face_weights, solve_voxels
    r = solve_voxels(np.ones((8, 4, 4), bool), (0, 0, 0), 0.5, [lambda X: X[:, 0] < 1e-9],
                     [(lambda X: X[:, 0] > 4 - 1e-9, (1.0, 0.0, 0.0))], 2.0, 0.0)
    assert abs(r["toporlogy"]["compliance"] - 4 / (2 * 4)) <= 1e-12
    assert abs(r["skfem"]["compliance"] - 4 / (2 * 4)) <= 1e-12
    assert np.allclose(r["skfem"]["reaction_sum"], [-1, 0, 0], atol=1e-12)
    w = face_weights(np.ones((2, 2, 2), bool), 0.5, np.ones(27, bool))
    assert abs(w.sum() - 6 * 1.0) <= 1e-12                         # surface area of the unit cube


def test_finalize_keep_fields_changes_nothing() -> None:
    from toporlogy.post import finalize
    from toporlogy.post.chain import ChainRequirements, record_chain
    from toporlogy.post.primitives import element_fraction
    d = T.Domain((10, 8, 8), 1.0)
    x = element_fraction(d, lambda X: np.max(np.abs(X - np.array([5, 4, 4])) - np.array([4, 2, 2]), axis=1))
    a = finalize(d, x)
    b = finalize(d, x, keep_fields=True)
    assert np.array_equal(a.V, b.V) and np.array_equal(a.F, b.F) and not a.fields
    assert set(b.fields) == {"threshold", "islands_removed", "opened", "closed", "final"}
    with pytest.raises(ValueError, match="keep_fields"):
        record_chain(d, x, a)
    with pytest.raises(ValueError, match="required"):
        record_chain(d, x, b, ChainRequirements(required=("watertightness_by_magic",)))
    rec = record_chain(d, x, b)
    assert rec["structural_verdict"] == "NOT VERIFIED" and rec["geometry_verdict"] == "PASS"


# ================================================================================================ §A
def test_a1_a2_euler_column_series() -> None:
    nbs = (2, 4, 8, 16)
    pt = C.euler_timoshenko()
    lam = [C.euler_column(nb) for nb in nbs]
    a1 = assess_convergence([C.B_COL / nb for nb in nbs], lam)
    q4im = C.euler_column(16, "Q4IM")
    print(f"\n  Q4 lambda/P_T - 1: {np.round(np.array(lam) / pt - 1, 6)}; order {a1.get('order')}; "
          f"extrapolated/P_T - 1 {a1.get('extrapolated', np.nan) / pt - 1:.2e}; Q4IM 16: {q4im / pt - 1:.2e}")
    assert a1["verdict"] == CONVERGED and 1.5 <= a1["order"] <= 2.5
    assert abs(a1["extrapolated"] / pt - 1) <= 1e-3
    assert abs(q4im / pt - 1) <= 1e-3


@pytest.mark.slow
def test_a3_embedded_compression_member() -> None:
    nbs = (2, 4, 8, 16)
    emb = [C.embedded_member(nb)["lambda1"] for nb in nbs]
    alone = [C.embedded_member(nb, alone=True)["lambda1"] for nb in nbs]
    ratio = np.array(emb) / np.array(alone)
    a3 = assess_convergence([C.B_MEM / nb for nb in nbs], emb)
    print(f"\n  embedded / alone - 1: {ratio - 1}; {a3['verdict']} {a3.get('order')}")
    assert np.max(np.abs(ratio - 1)) <= 1e-3 and a3["verdict"] == CONVERGED


@pytest.mark.slow
def test_a4_singular_wall_does_not_converge() -> None:
    ns = (40, 80, 160, 320)
    w = [C.wall(n) for n in ns]
    a4 = assess_convergence([1 / n for n in ns], [x[0] for x in w])
    loc = [x[1] for x in w]
    print(f"\n  wall lambda1 {[round(x[0], 4) for x in w]}, localisation {np.round(loc, 3)}, {a4['verdict']}")
    assert a4["verdict"] == NOT_CONVERGED and all(b > a for a, b in zip(loc, loc[1:], strict=False))


@pytest.mark.slow
def test_a5_exclusion_series_does_not_converge() -> None:
    ns = (40, 80, 160)
    a5 = assess_convergence([1 / n for n in ns], [C.wall(n, exclude_r=0.05)[0] for n in ns])
    print(f"\n  exclusion 0.05 series: {a5['values']} {a5['verdict']}")
    assert a5["verdict"] == NOT_CONVERGED


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="A6 pre-registered check failed (first run 2026-11-26): lambda1(r = 0.1) / "
                   "lambda1(r = 0.025) - 1 = 9.7e-5 (bound >= 0.10): once the clamp-end mode is removed the mode sits "
                   "at the opening corner, away from every zone")
def test_a6_exclusion_result_depends_on_the_zone_size() -> None:
    lam = [C.wall(80, exclude_r=r)[0] for r in (0.025, 0.05, 0.1)]
    assert lam[2] / lam[0] - 1 >= 0.10


# ================================================================================================ §B
def test_sp1_void_embedded_member_all_methods() -> None:
    ref = C.embedded_member(4, alone=True)["lambda1"]
    dev = {(m, e): C.embedded_member(4, e, m)["lambda1"] / ref - 1 for m in C.EG_METHODS for e in (1e-6, 1e-9)}
    print(f"\n  SP1 max |lambda/lambda_alone - 1| = {max(abs(v) for v in dev.values()):.2e}")
    assert max(abs(v) for v in dev.values()) <= 1e-3


@pytest.fixture(scope="module")
def d2():
    return C.d2_design()


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="SP2 pre-registered check failed (first run 2026-11-26): pg = 1 gives "
                   "lambda1 ratio 0.768 (bound <= 0.5) and a mode-1 low-density share 0.026 (bound >= 0.5): no "
                   "spurious mode on this design")
def test_sp2_low_stress_exponent_produces_a_spurious_mode(d2) -> None:
    x, _, st, d = d2
    a, b = C.evaluate_density(st, d, x, 1e-6, "pg=1"), C.evaluate_density(st, d, x, 1e-6, "pg=3")
    assert a["low_density_share"] >= 0.5 and a["lambda1"] / b["lambda1"] <= 0.5


@pytest.mark.slow
def test_sp3_stress_exponent_at_least_p_has_no_low_density_mode(d2) -> None:
    x, _, st, d = d2
    for m in ("pg=3", "pg=4"):
        for e in (1e-3, 1e-6, 1e-9):
            assert C.evaluate_density(st, d, x, e, m)["low_density_share"] < 0.5


# ================================================================================================ §C
def test_rl1_cluster_cut_is_marked_unreliable() -> None:
    out = {}
    for n in (1, 2):
        st, resp, d = C.square_column_study(n)
        C.response_value(st, resp, np.ones(d.n_design))
        out[n] = resp.last_reliability
    assert not out[1].reliable and "cluster_cut" in out[1].flags
    assert out[2].reliable


def _asymmetry(n: int) -> float:
    st, resp, d = C.square_column_study(n)
    x0 = np.ones(d.n_design)
    vals = []
    for dv in np.random.default_rng(0).standard_normal((4, d.n_design)):
        dp, dm = C.one_sided_derivatives(st, resp, x0, dv)
        vals.append(abs(dp - dm) / max(abs(dp), abs(dm)))
    return max(vals)


def test_rl2_one_sided_derivatives_of_a_cut_cluster_differ() -> None:
    """Measured 2026-11-26 (first run): 1.994."""
    a = _asymmetry(1)
    print(f"\n  |D+ - D-|/|D|, one mode of a double pair: {a:.3e}")
    assert a >= 1e-2


@pytest.mark.xfail(strict=True, reason="RL2b pre-registered check failed (first run 2026-11-26): whole-cluster "
                   "asymmetry 1.079e-5 > 1e-5 and AD vs FD 1.16e-6 > 1e-6; post-hoc in burkulma_geometri.md")
def test_rl2b_whole_cluster_aggregate_is_differentiable() -> None:
    from toporlogy.optimization.diagnostics import study_gradcheck
    a = _asymmetry(2)
    st, resp, d = C.square_column_study(2)
    x0 = np.ones(d.n_design)
    r1 = study_gradcheck(st, x0, 1, n_dirs=4, h=2e-4, seed=2)
    r2 = study_gradcheck(st, x0, 1, n_dirs=4, h=4e-4, seed=2)
    err = max(abs((4 * u["fd"] - v["fd"]) / 3 - u["ad"]) / max(abs(u["ad"]), 1e-300)
              for u, v in zip(r1["directions"], r2["directions"], strict=True))
    print(f"\n  whole cluster: |D+ - D-|/|D| {a:.3e}; AD vs FD {err:.2e}")
    assert a <= 1e-5 and err <= 1e-6


@pytest.mark.slow
def test_rl3_tracking_on_the_grey_design(d2) -> None:
    _, xd, _, _ = d2
    st, _ = C.d2_study()
    resp = BucklingKS(n_modes=6, rho=30.0)
    for _ in range(2):
        resp(st.context(torch.as_tensor(xd), True))
    tr = resp.last_tracking
    assert tr is not None and (tr.order == np.arange(6)).all() and tr.mac.min() >= 1 - 1e-8


# ================================================================================================ §D, §E
@pytest.fixture(scope="module")
def bar():
    from toporlogy.post import finalize
    from toporlogy.post.primitives import element_fraction
    d = T.Domain((32, 12, 12), 1.0)
    x = element_fraction(d, C.stepped_bar_sdf)
    return d, x, finalize(d, x, keep_fields=True)


@pytest.mark.slow
def test_gc1_to_gc3_gc5_gc6_gc9_stepped_bar_chain(bar) -> None:
    from toporlogy.post.chain import UNEVALUABLE, ChainRequirements, record_chain
    d, x, res = bar
    rec = record_chain(d, x, res, ChainRequirements(required=UNEVALUABLE))
    st = {s["name"]: s for s in rec["stages"]}
    v = [st[n]["metrics"]["volume"] / 1792 - 1 for n in ("density", "threshold", "surface")]
    assert abs(v[0]) <= 1e-9 and abs(v[1]) <= 5e-3 and abs(v[2]) <= 3e-2                         # GC1
    for s in st.values():                                                                         # GC2
        if "topology" in s["metrics"]:
            assert s["metrics"]["topology"]["bodies"] == 1 and s["metrics"]["topology"]["cavities"] == 0
    er = st["surface"]["metrics"]["edges"]
    assert er["open"] == 0 and er["non_manifold"] == 0 and er["misoriented"] == 0                 # GC3
    assert st["surface"]["metrics"]["cad_excursion"] <= 0.05 * res.grid.h                        # GC5
    unv = [c["status"] for c in st["surface"]["checks"] if c["name"] in UNEVALUABLE]
    assert unv == ["UNVERIFIED"] * 3 and rec["geometry_verdict"] != "PASS"                        # GC6
    assert st["reanalysis_t10"]["status"] == "NOT VERIFIED"                                       # GC9


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="GC4 pre-registered check failed (first run 2026-11-26): neck centre-line "
                   "local thickness 4.084-4.569 at the threshold stage, 4.084-4.343 final (bound 4 +- 0.5)")
def test_gc4_neck_thickness(bar) -> None:
    from toporlogy.post import sdf as S
    from toporlogy.post.checks import sdf_local_thickness
    _, _, res = bar
    g = res.grid
    ax = g.axes()
    ix = np.flatnonzero((ax[0] >= 14) & (ax[0] <= 18))
    iy, iz = int(np.argmin(np.abs(ax[1] - 6))), int(np.argmin(np.abs(ax[2] - 6)))
    for name in ("threshold", "final"):
        vals = sdf_local_thickness(S.redistance(res.fields[name], g.h), g.h)["field"][ix, iy, iz]
        assert np.all(np.abs(vals - 4) <= 0.5), (name, vals.min(), vals.max())


@pytest.mark.slow
def test_gc7_toporlogy_and_scikit_fem_agree_on_the_voxel_mesh(bar) -> None:
    pytest.importorskip("skfem")
    from toporlogy.post.reanalysis import solve_voxels, voxelize
    _, _, res = bar
    solid, org = voxelize(res.implicit, (0, 0, 0), (32, 12, 12), 0.5)
    r7 = solve_voxels(solid, org, 0.5, [lambda X: X[:, 0] < 1e-9], [(lambda X: X[:, 0] > 32 - 1e-9, (1.0, 0, 0))],
                      1.0, 0.0)
    assert abs(r7["toporlogy"]["compliance"] / r7["skfem"]["compliance"] - 1) <= 1e-8


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="GC8 pre-registered check failed (first run 2026-11-26): staircase h = 1/4 "
                   "compliance 0.67561, nominal series value 0.625 (+8.1 %, bound 5 %); post-hoc in "
                   "burkulma_geometri.md")
def test_gc8_staircase_compliance_against_the_nominal_series_bar(bar) -> None:
    from toporlogy.post.reanalysis import staircase_series
    _, _, res = bar
    r8 = staircase_series(res.implicit, (0, 0, 0), (32, 12, 12), (1 / 2, 1 / 3, 1 / 4), [lambda X: X[:, 0] < 1e-9],
                          [(lambda X: X[:, 0] > 32 - 1e-9, (1.0, 0.0, 0.0))], 1.0, 0.0, solver="gmg")
    c = r8["meshes"][-1]["toporlogy"]["compliance"]
    assert abs(c / 0.625 - 1) <= 0.05


@pytest.mark.slow
def test_vc1_volume_equal_variants_differ_in_buckling_or_stress(d2) -> None:
    x, _, st, d = d2
    ev = {k: C.evaluate_density(st, d, v) for k, v in C.volume_variants(x).items()}
    keys = list(ev)
    best = 0.0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if abs(ev[a]["volume"] / ev[b]["volume"] - 1) <= 5e-3:
                best = max(best, abs(ev[a]["lambda1"] / ev[b]["lambda1"] - 1),
                           abs(ev[a]["vm_max_solid"] / ev[b]["vm_max_solid"] - 1))
    assert best >= 0.05


def test_cluster_warning_still_issued() -> None:
    st, resp, d = C.square_column_study(1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st.evaluate(np.ones(d.n_design), gradients=False)
    assert any("cluster" in str(x.message) for x in w)


@pytest.mark.slow
def test_vc3_both_volume_match_options_keep_the_volume() -> None:
    """3-D cantilever of test_post_cantilever (OC, 40 iterations): finished-part volume within 1 % of the optimised
    volume for volume_match = "threshold" and "final" (measured 2026-11-26: -0.21 %, -0.48 %)."""
    from toporlogy import OC, SIMP, BetaContinuation, Compliance, Constraint, Domain, Elasticity, Study, Volume
    from toporlogy import standard_chain
    from toporlogy.post import BoxPrimitive, Keep, PostParams, finalize
    from toporlogy.post import mesh as M
    nx, ny, nz = 30, 10, 10
    keep = np.zeros((nx, ny, nz), bool)
    keep[0] = True
    keep[-1, :, 0] = True
    d = Domain((nx, ny, nz), h=1.0, keep=keep)
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-6))
    tip = d.nodes_where(lambda X: (np.abs(X[:, 0] - nx) < 1e-6) & (X[:, 2] < 1e-6))
    F = d.load_vector(1)
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    chain = standard_chain(d, 1.5, projection=True)
    st = Study(d, Elasticity(d, fixed, nu=0.3), F, chain, SIMP(d), Compliance(), [Constraint(Volume(), 0.3, scale=1.0)],
               OC(dv_scale=d.n_active), BetaContinuation(chain, 1.0, 8.0, 15, 0.01))
    x = np.asarray(st.run(max_iter=40, tol=0.01).xphys, float).reshape(d.shape)
    pads = [Keep(BoxPrimitive((0, 0, 0), (1, ny, nz)), name="clamp plate"),
            Keep(BoxPrimitive((nx - 1, 0, 0), (nx, ny, 1)), name="load pad")]
    for vm in ("threshold", "final"):
        res = finalize(d, x, preserve=pads, params=PostParams(volume_match=vm))
        assert abs(M.enclosed_volume(res.V, res.F) / x.sum() - 1) <= 0.01
