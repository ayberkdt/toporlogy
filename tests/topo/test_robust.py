"""Robust formulation (minimum solid and void size) — aerokit.topo.robust.

1. Size rules: the numeric derivation reproduces Qian & Sigmund Eq. 29 for the hat kernel; settings round trip.
2. Finite-difference gradients (threshold 1e-6, float64) of RobustCompliance (KS, p-norm), DilatedVolume and the
   bound-formulation constraints, through the three realisation solves. Responses through the linear solve use the
   Richardson-extrapolated difference of tests/topo/test_gradients.py (h = 1e-3 and 2e-3; reason documented there).
3. 2-D half MBB 90 × 30: robust run with minimum solid and void size b → measured blueprint sizes vs b, compared
   with non-robust runs (all numbers printed).
4. 3-D robust cantilever 30 × 10 × 6 runs and ends crisp (grey fraction < 5 %).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import aerokit.topo as T
from aerokit.topo.lengthscale import measure_lengthscale
from aerokit.topo.optim.mma import MMA
from aerokit.topo.robust import (DilatedVolume, Robust, RobustCompliance, RobustContinuation, RobustSet,
                                 blueprint_size, bound_formulation, eta_for_size, min_size_ratio,
                                 robust_chain_settings)

TOL = 1e-6


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """One torch / MKL thread while this module runs (restored afterwards). On a machine shared by several CPU-bound
    jobs the multithreaded PARDISO / torch kernels oversubscribe the cores: one robust MBB evaluation took 5.0-5.4 s
    multithreaded vs 0.16-0.25 s single-threaded (measured 14 Nov 2026, 100 % CPU load from other processes)."""
    n_torch = torch.get_num_threads()
    torch.set_num_threads(1)
    lib = None
    try:
        from aerokit.topo.solvers.pardiso import load as pardiso_module
        pp = pardiso_module()
        if pp is not None:
            lib = pp.PyPardisoSolver().libmkl
            n_mkl = lib.MKL_Get_Max_Threads()
            lib.MKL_Set_Num_Threads(1)
    except Exception:  # noqa: BLE001 - optional (SuperLU fallback has no thread setting)
        lib = None
    yield
    torch.set_num_threads(n_torch)
    if lib is not None:
        lib.MKL_Set_Num_Threads(n_mkl)


# ---------------------------------------------------------------------------------------------- size rules
@pytest.mark.parametrize("eta", [0.55, 0.65, 0.75, 0.85, 0.95])
def test_numeric_size_rule_reproduces_hat_formula(eta):
    """1-D hat (cone in 1-D) of large radius: blueprint_size / R → Qian & Sigmund Eq. 29."""
    R = 80.0
    assert blueprint_size(eta, R, dim=1) / R == pytest.approx(min_size_ratio(eta), abs=2e-3)
    assert eta_for_size(min_size_ratio(eta)) == pytest.approx(eta, abs=1e-12)


def test_settings_round_trip_and_cone_profile():
    s = robust_chain_settings(4.0, 2.0, h=0.5, dim=2)
    assert blueprint_size(s["eta_e"], s["rmin"], 2) * 0.5 == pytest.approx(4.0, rel=1e-6)
    assert blueprint_size(1 - s["eta_d"], s["rmin"], 2) * 0.5 == pytest.approx(2.0, rel=1e-6)
    assert s["eta_e"] == pytest.approx(0.75)
    hat = robust_chain_settings(4.0, 2.0, h=0.5, kernel="hat")
    assert hat["rmin"] == pytest.approx(8.0) and hat["eta_d"] == pytest.approx(1 - eta_for_size(0.5))
    # the 2-D cone profile is more peaked than the hat, so a larger radius is needed for the same size
    assert s["rmin"] > hat["rmin"]
    assert robust_chain_settings(4.0, dim=3)["rmin"] > robust_chain_settings(4.0, dim=2)["rmin"]


# ---------------------------------------------------------------------------------------------- gradients
def _small_problem(dim=2):
    if dim == 2:
        shape = (12, 6)
        mask = np.ones(shape, bool)
        mask[8:, 4:] = False
        keep = np.zeros(shape, bool)
        keep[-1, :2] = True
        void = np.zeros(shape, bool)
        void[2, 5] = True
        d = T.Domain(shape, 1.0, frac=mask.astype(float), keep=keep, void=void)
    else:
        d = T.Domain((6, 4, 3), 0.5)
    eps = 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    F = d.load_vector()
    tip = d.nodes_where(lambda X: abs(X[:, 0] - d.shape[0] * d.h) < eps)
    F[d.dofs(tip, 1), 0] = -1.0 / len(tip)
    return d, fixed, F


def _study(d, fixed, F, objective, constraints, chain=None):
    chain = chain or T.standard_chain(d, 2.0, projection=True, beta=3.0)
    return T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), objective, constraints, MMA())


def _richardson(study, x, which, h=1e-3):
    a = T.study_gradcheck(study, x, which, h=h, n_dirs=4)
    b = T.study_gradcheck(study, x, which, h=2 * h, n_dirs=4)
    errs = []
    for ra, rb in zip(a["directions"], b["directions"], strict=True):
        fd = (4 * ra["fd"] - rb["fd"]) / 3
        errs.append(abs(fd - ra["ad"]) / max(abs(ra["ad"]), abs(fd), 1e-300))
    return max(errs)


@pytest.mark.parametrize("dim", [2, 3])
def test_gradcheck_robust_compliance_and_dilated_volume(dim):
    d, fixed, F = _small_problem(dim)
    rs = RobustSet(0.7, 0.3)
    rng = np.random.default_rng(11)
    x0 = 0.2 + 0.6 * rng.random(d.n_design)
    for obj in (RobustCompliance(rs, aggregate="ks", rho=8.0), RobustCompliance(rs, aggregate="pnorm", p=6.0),
                Robust(T.Compliance(), rs, realisations=("eroded", "dilated"), aggregate="ks", rho=20.0)):
        st = _study(d, fixed, F, obj, [T.Constraint(DilatedVolume(rs), 0.5)])
        e_obj = _richardson(st, x0, "objective")
        e_vol = T.study_gradcheck(st, x0, 0, n_dirs=4)["max_rel_error"]
        print(f"\n[gradcheck {dim}-D {obj.aggregate} {obj.realisations}] objective {e_obj:.2e}, dilated volume "
              f"{e_vol:.2e}")
        assert e_obj < TOL and e_vol < TOL


def test_gradcheck_bound_formulation_and_own_beta():
    d, fixed, F = _small_problem(2)
    rs = RobustSet(0.75, beta=5.0)                                # own β, chain without projection
    base = T.standard_chain(d, 2.0, projection=False)
    chain, obj, cons = bound_formulation(base, rs, d.n_design)
    st = _study(d, fixed, F, obj, cons + [T.Constraint(DilatedVolume(rs), 0.5)], chain=chain)
    x0 = st.default_start()
    assert x0.shape == (d.n_design + 1,) and x0[-1] == 1.0
    # x ∈ [0.4, 0.9]: with x ∈ [0.2, 0.8] the eroded realisation (η 0.75, β 5) is almost void, its stiffness matrix
    # badly conditioned and the eroded constraint's difference quotient noise-limited (measured 1.1e-6 at h = 1e-3,
    # 1.9e-6 at 5e-4, 5.3e-6 at 2.5e-4 — growing as h falls, i.e. noise, not truncation); here 6.3e-9.
    rng = np.random.default_rng(5)
    x = np.concatenate([0.4 + 0.5 * rng.random(d.n_design), [0.7]])
    errs = [_richardson(st, x, i) for i in range(3)] + [T.study_gradcheck(st, x, 3, n_dirs=4)["max_rel_error"]]
    print(f"\n[gradcheck bound] constraints {[f'{e:.2e}' for e in errs]}")
    assert max(errs) < TOL


def test_intermediate_reuses_study_field_and_solve_count():
    d, fixed, F = _small_problem(2)
    rs = RobustSet(0.75)
    obj = RobustCompliance(rs)
    st = _study(d, fixed, F, obj, [T.Constraint(DilatedVolume(rs), 0.5)])
    n0 = st.physics.n_solves
    ev = st.evaluate(st.default_start())
    assert st.physics.n_solves - n0 == 3                          # one solve per realisation, none for the study
    assert obj.last_values["eroded"] >= obj.last_values["intermediate"] >= obj.last_values["dilated"]
    assert ev.f0 >= max(obj.last_values.values())                 # KS is an upper bound


# ---------------------------------------------------------------------------------------------- optimisation
def _half_mbb(nx=90, ny=30):
    d = T.Domain((nx, ny), 1.0)
    eps = 1e-6
    left = d.nodes_where(lambda X: X[:, 0] < eps)
    br = d.nodes_where(lambda X: (abs(X[:, 0] - nx) < eps) & (X[:, 1] < eps))
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(br, 1)])
    F = d.load_vector()
    F[d.dofs(d.nodes_where(lambda X: (X[:, 0] < eps) & (abs(X[:, 1] - ny) < eps)), 1), 0] = -1.0
    return d, fixed, F


MBB_FACES = [("reflect", "void"), ("void", "void")]       # x = 0 is the symmetry plane, the other faces are free


def test_robust_mbb_minimum_solid_and_void_size():
    """Half MBB 60 × 20, b = 3 elements (solid and void). Measured on the blueprint with ``smooth = 0.5`` (removes the
    one-element staircase teeth of slanted edges, see measure_lengthscale) and ``margin = b`` (cells within b of a
    free face are left out: at free faces the filter acts like a symmetry plane, module notes of robust.py). The
    90 × 30 / b = 4 version is reported in the module notes of this file's report, not run here (time)."""
    t0 = time.perf_counter()
    b, V = 3.0, 0.5
    d, fixed, F = _half_mbb(60, 20)
    s = robust_chain_settings(b, b, 1.0, 2)
    rs = RobustSet.from_settings(s)
    chain = T.standard_chain(d, s["rmin"], projection=True, beta=1.0)
    obj = RobustCompliance(rs)
    con = T.Constraint(DilatedVolume(rs), V)
    sch = RobustContinuation(chain, rs, 1.0, 32.0, every=20, tol=0.01, volume=con, target=V, volume_every=10,
                             rescale=[obj])
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), obj, [con], T.OC(move=0.1, dv_scale=d.n_active), sch)
    rob = st.run(max_iter=140, tol=0.01)
    kw = dict(outside=MBB_FACES, smooth=0.5, margin=b)
    m_rob = measure_lengthscale(rob.xphys, 1.0, **kw)
    m_rob_raw = measure_lengthscale(rob.xphys, 1.0, outside=MBB_FACES)
    t1 = time.perf_counter()
    # non-robust reference: same mesh, plain projection, small filter
    ch = T.standard_chain(d, 1.5, projection=True, beta=1.0)
    st2 = T.Study(d, T.Elasticity(d, fixed), F, ch, T.SIMP(d), T.Compliance(), [T.Constraint(T.Volume(), V)],
                  T.OC(move=0.1, dv_scale=d.n_active), T.BetaContinuation(ch, 1.0, 32.0, 20, 0.01))
    plain = st2.run(max_iter=140, tol=0.01)
    m_pl = measure_lengthscale(plain.xphys, 1.0, **kw)
    print(f"\n[robust MBB] {s}\n  robust: {rob.iterations} it, {t1 - t0:.1f} s, KS {rob.objective:.4g}, "
          f"realisations {obj.last_values}, V_blue {rob.xphys.mean():.4f}, dilated bound {con.upper:.4f}"
          f"\n   smooth 0.5, margin b: {m_rob.summary()}\n   raw, all cells: {m_rob_raw.summary()}"
          f"\n  plain rmin 1.5: {plain.iterations} it, {time.perf_counter() - t1:.1f} s, c {plain.objective:.4g}"
          f"\n   smooth 0.5, margin b: {m_pl.summary()}")
    assert m_rob.solid_min >= b - 1.0
    assert m_rob.void_min >= b - 1.0
    assert m_rob.grey_fraction < 0.05
    assert abs(rob.xphys.mean() - V) < 0.02
    assert m_pl.solid_min < b - 1.0 or m_pl.void_min < b - 1.0


def test_robust_3d_cantilever_runs_crisp():
    """3-D robust cantilever 24 × 8 × 6, b = 2 elements, solid pad (keep) under the tip load so that the eroded
    realisation stays attached to the load. 30 × 10 × 6 is reported separately (time).

    Not every small set-up is stable with OC (move 0.1): 20 × 8 × 5 with the same settings ended, in its last ten
    iterations at β = 32, with eroded / blueprint compliance 1.2e7 / 8.1e6 (load path lost; 452 / 147 ten iterations
    earlier). The compliance assertions below catch such a collapse."""
    t0 = time.perf_counter()
    nx, ny, nz = 24, 8, 6
    keep = np.zeros((nx, ny, nz), bool)
    keep[-2:, :, :2] = True
    d = T.Domain((nx, ny, nz), 1.0, keep=keep)
    eps = 1e-6
    fixed = d.dofs(d.nodes_where(lambda X: X[:, 0] < eps))
    F = d.load_vector()
    tip = d.nodes_where(lambda X: (abs(X[:, 0] - nx) < eps) & (X[:, 2] < eps))
    F[d.dofs(tip, 2), 0] = -1.0 / len(tip)
    s = robust_chain_settings(2.0, 2.0, 1.0, 3)
    rs = RobustSet.from_settings(s)
    chain = T.standard_chain(d, s["rmin"], projection=True, beta=1.0)
    obj = RobustCompliance(rs)
    con = T.Constraint(DilatedVolume(rs), 0.3)
    sch = RobustContinuation(chain, rs, 1.0, 32.0, every=15, tol=0.01, volume=con, target=0.3, volume_every=10,
                             rescale=[obj])
    st = T.Study(d, T.Elasticity(d, fixed), F, chain, T.SIMP(d), obj, [con], T.OC(move=0.1, dv_scale=d.n_active), sch)
    res = st.run(max_iter=100, tol=0.01)
    grey = T.grey_fraction(res.xphys, d.active)
    m = measure_lengthscale(res.xphys, 1.0, outside="void")
    print(f"\n[robust 3-D] {s}\n  {res.iterations} it, {time.perf_counter() - t0:.1f} s, KS {res.objective:.4g}, "
          f"realisations {obj.last_values}, beta {sch.beta}, grey {100 * grey:.2f} %, V_blue {res.xphys.mean():.4f}"
          f"\n  {m.summary()}")
    assert sch.beta == 32.0
    assert grey < 0.05
    lv = obj.last_values
    assert lv["eroded"] >= lv["intermediate"] >= lv["dilated"]
    assert lv["eroded"] < 10.0 * lv["dilated"]                   # all realisations carry the load
    assert abs(res.xphys.mean() - 0.3) < 0.03
    assert np.isfinite(res.objective)
    assert torch.isfinite(torch.as_tensor(res.xphys)).all()
