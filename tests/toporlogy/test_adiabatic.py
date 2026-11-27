"""Adiabatic heating of a Perzyna J2 point (``toporlogy.core.thermomechanics``), checks AD1-AD8.

Thresholds are the preregistered ones of ``docs/foundations/prereg/R5_adiabatic.md`` (committed before the first run).
All constants are synthetic, not material data. References are closed forms and a ``scipy.integrate.solve_ivp``
integration of the scalar ODE system written in this file; tests labelled "regression" compare the production code with
another construction of the same kernel (they are not independent references).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (Kinematics, MaterialRequest, StateStore, TimeStep, drive_material_point)
from toporlogy.core.properties import Constant, Polynomial
from toporlogy.core.thermomechanics import AdiabaticPerzyna
from toporlogy.core.viscoplastic import Perzyna

E0, NU = 2.0e9, 0.35                         # synthetic
K0, G0 = E0 / (3.0 * (1.0 - 2.0 * NU)), E0 / (2.0 * (1.0 + NU))
HH, TAU0, SREF, MM = 100.0e6, 1.0, 10.0e6, 2.0
RHO, TREF = 1200.0, 293.15
SY_A, SY_B = 20.0e6, -1.0e6                  # sigma_y(T) = SY_A + SY_B (T - TREF) Pa on [273.15, 310]
CP_A, CP_B = 1000.0, 2.0                     # c_p(T) = CP_A + CP_B (T - TREF) J/(kg K)
CTRL_XX = [True, False, False, False, False, False]
E_STRESS_TOL = 1e-5
VA = np.array([p[0] for p in T.VOIGT_3D])
VB = np.array([p[1] for p in T.VOIGT_3D])


def sy_fn() -> Polynomial:
    return Polynomial([SY_A, SY_B], TREF, (273.15, 310.0), "Pa", "synthetic", positive=True)


def cp_fn() -> Polynomial:
    return Polynomial([CP_A, CP_B], TREF, (200.0, 600.0), "J/(kg K)", "synthetic", positive=True)


def make(beta: float, sy=None, cls=AdiabaticPerzyna) -> AdiabaticPerzyna:
    return cls(K0, G0, sy_fn() if sy is None else sy, HH, TAU0, SREF, MM, RHO, cp_fn(), beta, TREF)


# ---- references written in the test (independent of the production property objects)
def sy_ref(Tk: float) -> float:
    return SY_A + SY_B * (Tk - TREF)


def h_ref(T0: float, T1: float) -> float:
    """integral of c_p = CP_A + CP_B (T - TREF) from T0 to T1."""
    return CP_A * (T1 - T0) + 0.5 * CP_B * ((T1 - TREF) ** 2 - (T0 - TREF) ** 2)


def sig_t(v: np.ndarray) -> np.ndarray:
    S = np.zeros((3, 3))
    S[VA, VB] = v
    S[VB, VA] = v
    return S


def run_uniaxial(model: AdiabaticPerzyna, n: int, rate: float = 1.0, eps_end: float = 0.08, store=None):
    t = np.linspace(0.0, eps_end / rate, n + 1)[1:]
    strain = np.zeros((n, 6))
    strain[:, 0] = rate * t
    return drive_material_point(model, strain, np.zeros((n, 6)), CTRL_XX, tol=E_STRESS_TOL, times=t, store=store)


@functools.cache
def cached(beta: float, n: int):
    return run_uniaxial(make(beta), n)


def temps(h) -> np.ndarray:
    return np.array([TREF] + [float(s["temperature"][0]) for s in h.states])


def alphas(h) -> np.ndarray:
    return np.array([0.0] + [float(s["alpha"][0]) for s in h.states])


def budget(h):
    """Per step: trapezoid work W, stored-energy increment, D, and the predicted mismatch 1/2 (s0 - s1):de_p."""
    S = [np.zeros((3, 3))] + [sig_t(s) for s in h.stress]
    E = [np.zeros((3, 3))] + [T.voigt_to_strain(e) for e in h.strain]
    EP = [np.zeros((3, 3))] + [s["plastic_strain"][0] for s in h.states]
    psi = np.concatenate([[0.0], h.energy])
    W, dpsi, mism = [], np.diff(psi), []
    for k in range(len(h.strain)):
        W.append(0.5 * np.sum((S[k] + S[k + 1]) * (E[k + 1] - E[k])))
        mism.append(0.5 * np.sum((S[k] - S[k + 1]) * (EP[k + 1] - EP[k])))
    return np.array(W), dpsi, np.array(h.dissipation), np.array(mism)


def ode_uniaxial(beta: float, rate: float = 1.0, eps_end: float = 0.08) -> tuple[float, float]:
    """(p_end, T_end) of the scalar ODE system of derivation 6 of the record."""
    Eu = 9.0 * K0 * G0 / (3.0 * K0 + G0)

    def rhs(t, y):
        p, Tk = y
        sig = Eu * (rate * t - p)
        f = sig - sy_ref(Tk) - HH * p
        pdot = max(f, 0.0) ** MM / (TAU0 * SREF ** MM)
        return [pdot, beta * (sig - HH * p) * pdot / (RHO * (CP_A + CP_B * (Tk - TREF)))]

    sol = solve_ivp(rhs, (0.0, eps_end / rate), [0.0, TREF], method="Radau", rtol=1e-12, atol=[1e-16, 1e-12])
    assert sol.success
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def ode_stress(p_end: float, eps_end: float = 0.08) -> float:
    return 9.0 * K0 * G0 / (3.0 * K0 + G0) * (eps_end - p_end)


# =============================================================================================================== AD1
EPS_PATH = np.array([[0.012, -0.003, 0.001, 0.002, -0.001, 0.006],
                     [0.0, 0.030, 0.0, 0.0, 0.0, 0.004],
                     [0.020, -0.010, 0.0, 0.0, 0.002, 0.010]])


def test_AD1a_beta_zero_is_bitwise_perzyna_regression() -> None:
    """beta = 0 equals Perzyna(sigma_y(T0)) bitwise on a 3-point multiaxial 3-step path (regression: same kernel)."""
    m = make(0.0)
    ref = Perzyna(K0, G0, float(sy_ref(TREF)), HH, TAU0, SREF, MM)
    cm, cr = m.initial_state(3), ref.initial_state(3)
    eps = np.array([T.voigt_to_strain(e) for e in EPS_PATH])
    for k, (dt, scale) in enumerate([(0.01, 1.0), (0.01, 1.7), (0.02, 0.9)]):
        req = MaterialRequest(energy=True, step=TimeStep(0.01 * k, dt))
        a, b = m.update(eps * scale, cm, req), ref.update(eps * scale, cr, req)
        assert np.array_equal(a.stress, b.stress)
        assert np.array_equal(a.tangent, b.tangent)
        assert np.array_equal(a.energy, b.energy)
        assert np.array_equal(a.dissipation, b.dissipation)
        assert np.array_equal(a.trial_state["plastic_strain"], b.trial_state["plastic_strain"])
        assert np.array_equal(a.trial_state["alpha"], b.trial_state["alpha"])
        assert np.all(a.trial_state["temperature"] == TREF)
        assert np.allclose(a.trial_state["cold_work"] - cm["cold_work"], a.dissipation, rtol=1e-12, atol=0.0)  # beta = 0: D all cold
        cm = {**a.trial_state}
        cr = {"plastic_strain": b.trial_state["plastic_strain"], "alpha": b.trial_state["alpha"]}
    assert np.all(cm["alpha"] > 0.0)                      # the path really was plastic


def test_AD1b_constant_sigma_y_no_feedback_but_heats() -> None:
    sy = Constant(SY_A, "Pa", "synthetic", domain=(200.0, 600.0), positive=True)
    h0, h1 = run_uniaxial(make(0.0, sy), 50), run_uniaxial(make(1.0, sy), 50)
    assert np.array_equal(h0.stress, h1.stress)
    assert np.array_equal(alphas(h0), alphas(h1))
    assert temps(h1)[-1] - TREF > 1e-3
    assert np.all(temps(h0) == TREF)


# =============================================================================================================== AD2
def test_AD2a_step_identity() -> None:
    h = cached(1.0, 200)
    W, dpsi, D, mism = budget(h)
    res = W - dpsi - D
    v = np.max(np.abs(res - mism)) / np.max(np.abs(W))
    print(f"AD2a identity residual/max W = {v:.3e}; max |res| = {np.max(np.abs(res)):.3e}, max W = {np.max(np.abs(W)):.3e}")
    assert v <= 1e-9


@pytest.mark.parametrize("beta", [1.0, 0.4])
def test_AD2b_history_closure_converges_with_dt(beta: float) -> None:
    ns = [25, 50, 100, 200]
    R, rel = [], []
    for n in ns:
        h = cached(beta, n)
        W, dpsi, D, _ = budget(h)
        R.append(abs(np.sum(W) - np.sum(dpsi) - np.sum(D)))
        rel.append(R[-1] / np.sum(W))
    print(f"AD2b beta={beta}: |R|/sumW = {rel}, R = {R}, ratios = {[a / b for a, b in zip(R[:-1], R[1:])]}")
    assert rel[-1] <= 5e-3, rel
    for a, b in zip(R[:-1], R[1:]):
        assert a / b >= 1.5, R


@pytest.mark.parametrize("beta", [1.0, 0.4])
def test_AD2c_heat_closure_cold_work_and_monotone_T(beta: float) -> None:
    h = cached(beta, 200)
    Tk, D = temps(h), np.array(h.dissipation)
    step_res = np.array([RHO * h_ref(Tk[k], Tk[k + 1]) - beta * D[k] for k in range(len(D))])
    print(f"AD2c beta={beta}: max step res/max(beta D) = {np.max(np.abs(step_res)) / np.max(beta * D):.3e}, "
          f"total {abs(RHO * h_ref(Tk[0], Tk[-1]) - beta * D.sum()) / (beta * D.sum()):.3e}, dT = {Tk[-1] - TREF:.5f} K, "
          f"cold {float(h.states[-1]['cold_work'][0]) / D.sum() / (1 - beta) if beta < 1 else float(h.states[-1]['cold_work'][0]):.9g}")
    assert np.max(np.abs(step_res)) <= 1e-8 * np.max(beta * D)
    assert abs(RHO * h_ref(Tk[0], Tk[-1]) - beta * D.sum()) <= 1e-9 * beta * D.sum()
    wc = float(h.states[-1]["cold_work"][0])
    assert abs(wc - (1.0 - beta) * D.sum()) <= 1e-9 * D.sum()
    assert np.all(np.diff(Tk) >= 0.0)
    elastic = np.array([d["plastic"] == 0.0 for d in h.diagnostics])
    assert elastic.any() and np.all(Tk[1:][elastic] == TREF)


def test_AD2d_hardening_energy_excluded_from_heat() -> None:
    h = cached(1.0, 200)
    D = np.array(h.dissipation)
    pN = alphas(h)[-1]
    assert abs(np.sum(h.plastic_work) - D.sum() - 0.5 * HH * pN ** 2) <= 1e-9 * D.sum()
    Q = RHO * h_ref(TREF, temps(h)[-1])
    # the prereg wrote 'Q_total <= sum D' without a roundoff allowance; at beta = 1 Q = sum D to roundoff (see the record)
    print(f"AD2d Q - sum D = {Q - D.sum():.3e} J/m3 (relative {(Q - D.sum()) / D.sum():.2e}); psi_h(p_N) = {0.5 * HH * pN ** 2:.6g}")
    assert D.sum() * (1.0 + 1e-12) >= Q
    assert abs(Q - D.sum()) <= 1e-9 * D.sum()             # beta = 1: all of D, none of the hardening energy


# =============================================================================================================== AD3
@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_AD3_adiabatic_rise_against_ode(beta: float) -> None:
    p_ode, T_ode = ode_uniaxial(beta)
    dT_ode, s_ode = T_ode - TREF, ode_stress(p_ode)
    ns = [25, 50, 100, 200]
    dT, sg = [], []
    for n in ns:
        h = cached(beta, n)
        dT.append(temps(h)[-1] - TREF)
        sg.append(h.stress[-1, 0])
    eT = [abs(x - dT_ode) / dT_ode for x in dT]
    eS = [abs(x - s_ode) / SY_A for x in sg]
    print(f"AD3 beta={beta}: dT_ode = {dT_ode:.6f} K, errT = {eT}, errS = {eS}, rich T = "
          f"{abs(2.0 * dT[3] - dT[2] - dT_ode) / dT_ode:.3e}, rich S = {abs(2.0 * sg[3] - sg[2] - s_ode) / SY_A:.3e}")
    for e in (eT, eS):
        for a, b in zip(e[:-1], e[1:]):
            assert a / b >= 1.5, (eT, eS)
    assert abs(2.0 * dT[3] - dT[2] - dT_ode) / dT_ode <= 3e-3
    assert abs(2.0 * sg[3] - sg[2] - s_ode) / SY_A <= 3e-3


# =============================================================================================================== AD4
def test_AD4_thermal_softening_reduces_stress() -> None:
    h0, h1 = cached(0.0, 200), cached(1.0, 200)
    s0, s1 = h0.stress[:, 0], h1.stress[:, 0]
    assert np.all(s1 <= s0)
    first = int(np.argmax([d["plastic"] > 0.0 for d in h0.diagnostics]))
    assert np.all(s1[first:] < s0[first:])
    p0, _ = ode_uniaxial(0.0)
    p1, _ = ode_uniaxial(1.0)
    red_ode = ode_stress(p0) - ode_stress(p1)
    red = s0[-1] - s1[-1]
    assert red > 0.0
    assert abs(red - red_ode) <= 0.05 * red_ode
    print(f"AD4 reduction {red:.6g} Pa, ODE {red_ode:.6g} Pa, ratio to |sy'|dT_end = "
          f"{red / (abs(SY_B) * (temps(h1)[-1] - TREF)):.4f}")


# =============================================================================================================== AD5
@pytest.mark.parametrize("beta", [-0.1, 1.0 + 1e-12, float("nan"), float("inf")])
def test_AD5_beta_outside_unit_interval_rejected(beta: float) -> None:
    with pytest.raises(ValueError, match="taylor_quinney"):
        make(beta)


def test_AD5_beta_endpoints_accepted() -> None:
    make(0.0), make(1.0)


def test_AD5_parameter_validation() -> None:
    good = dict(K=K0, G=G0, yield_stress=sy_fn(), hardening=HH, tau0=TAU0, sigma_ref=SREF, m=MM, density=RHO,
                specific_heat=cp_fn(), taylor_quinney=0.5, temperature0=TREF)
    AdiabaticPerzyna(**good)
    bad = [dict(density=0.0), dict(density=-1.0), dict(temperature0=0.0), dict(temperature0=-5.0),
           dict(yield_stress=20e6), dict(specific_heat=1000.0), dict(tau0=0.0), dict(sigma_ref=-1.0), dict(m=0.0),
           dict(hardening=-1.0), dict(K=0.0), dict(G=-1.0),
           dict(yield_stress=Polynomial([20.0, -1e-3], TREF, (273.15, 310.0), "MPa", "synthetic", positive=True)),
           dict(yield_stress=Polynomial([20e6, -1e6], TREF, (273.15, 310.0), "Pa", "synthetic", positive=False)),
           dict(specific_heat=Polynomial([1000.0], TREF, (200.0, 600.0), "J/kg", "synthetic", positive=True)),
           dict(specific_heat=Polynomial([1000.0], TREF, (200.0, 600.0), "J/(kg K)", "synthetic", positive=False)),
           dict(temperature0=500.0)]            # outside the validity interval of sigma_y
    for patch in bad:
        with pytest.raises(ValueError):
            AdiabaticPerzyna(**{**good, **patch})


def test_AD5_step_rules_and_domain_exit() -> None:
    m = make(1.0)
    eps = T.voigt_to_strain(EPS_PATH[0])[None]
    with pytest.raises(ValueError, match="physical time step"):
        m.update(eps, m.initial_state(1), MaterialRequest())
    with pytest.raises(ValueError, match="internal state"):
        m.update(eps, m.initial_state(1), MaterialRequest(step=TimeStep(0.0, 0.01, 300.0, 300.0)))
    big = T.voigt_to_strain(np.array([3.0, 0, 0, 0, 0, 0]))[None]
    with pytest.raises(ValueError, match="validity interval"):
        m.update(big, m.initial_state(1), MaterialRequest(step=TimeStep(0.0, 100.0)))


# =============================================================================================================== AD6
def test_AD6_rollback_purity_and_failed_step_leave_state_unchanged() -> None:
    m = make(1.0)
    store = StateStore(m, 1)
    run_uniaxial(m, 20, eps_end=0.04, store=store)
    before = store.checkpoint()
    assert before["alpha"][0] > 0.0 and before["temperature"][0] > TREF
    Hs = T.voigt_to_strain(np.array([0.05, -0.01, -0.01, 0.0, 0.0, 0.002]))[None]
    req = MaterialRequest(energy=True, step=TimeStep(0.04, 0.01))
    r1 = m.evaluate(Kinematics(Hs, "small"), None, store.committed, req)
    r2 = m.evaluate(Kinematics(Hs, "small"), None, store.committed, req)
    assert np.array_equal(r1.stress, r2.stress) and np.array_equal(r1.tangent, r2.tangent)
    for k in before:
        assert np.array_equal(r1.trial_state[k], r2.trial_state[k])
    assert r1.trial_state["temperature"][0] > before["temperature"][0]
    store.set_trial(r1.trial_state)
    store.rollback()
    for k, v in before.items():
        assert np.array_equal(store.committed[k], v) and not store.committed[k].flags.writeable
    big = T.voigt_to_strain(np.array([3.0, 0, 0, 0, 0, 0]))[None]
    with pytest.raises(ValueError):
        m.evaluate(Kinematics(big, "small"), None, store.committed, MaterialRequest(step=TimeStep(0.04, 100.0)))
    for k, v in before.items():
        assert np.array_equal(store.committed[k], v)


# =============================================================================================================== AD7
class _TotalWorkHeat(AdiabaticPerzyna):
    """NEGATIVE CONTROL: heat from the total plastic work (stored hardening energy included)."""

    def _heat_source(self, dissipation, diag):
        return diag["plastic_work"]


def _heat_check(h, beta: float) -> tuple[float, float, float, float]:
    """Energy check independent of the heat hook: (sum(Q - beta D), sum|Q - beta D| / (beta sum D), cold work, dT)."""
    Tk, D = temps(h), np.array(h.dissipation)
    Q = np.array([RHO * h_ref(Tk[k], Tk[k + 1]) for k in range(len(D))])
    r = Q - beta * D
    return float(r.sum()), float(np.abs(r).sum() / (beta * D.sum())), float(h.states[-1]["cold_work"][0]), \
        float(Tk[-1] - TREF)


def test_AD7_total_plastic_work_as_heat_is_detected_by_the_energy_check() -> None:
    hw = run_uniaxial(make(1.0, cls=_TotalWorkHeat), 200)
    hr = cached(1.0, 200)
    rs_w, rel_w, cold_w, dT_w = _heat_check(hw, 1.0)
    rs_r, rel_r, cold_r, dT_r = _heat_check(hr, 1.0)
    psi_h = 0.5 * HH * alphas(hw)[-1] ** 2
    print(f"AD7 wrong: sum(Q-D) = {rs_w:.6g} vs psi_h {psi_h:.6g}; rel {rel_w:.4f}; cold {cold_w:.6g}; right: rel {rel_r:.2e}, cold {cold_r:.3e}")
    assert abs(rs_w - psi_h) <= 1e-9 * psi_h                  # telescoping identity, beta = 1
    assert rel_w >= 0.02 and rel_r <= 1e-9
    assert cold_w < 0.0 and abs(cold_w + psi_h) <= 1e-6 * psi_h and cold_r >= -1e-9 * np.sum(hr.dissipation)
    rho_cp = RHO * (CP_A + CP_B * (dT_r / 2.0))
    assert abs((dT_w - dT_r) - psi_h / rho_cp) <= 0.10 * psi_h / rho_cp
    print(f"AD7 wrong-heat residual {rel_w:.4f} of sum D, dT right {dT_r:.5f} K, wrong {dT_w:.5f} K, psi_h {psi_h:.4g}")


# =============================================================================================================== AD8
def _fd_tangent(m: AdiabaticPerzyna, committed, e: np.ndarray, req: MaterialRequest, h: float = 1e-7) -> np.ndarray:
    cols = []
    for j in range(6):
        ep, em = e.copy(), e.copy()
        ep[j] += h
        em[j] -= h
        sp = m.update(T.voigt_to_strain(ep)[None], committed, req).stress[0]
        sm = m.update(T.voigt_to_strain(em)[None], committed, req).stress[0]
        cols.append(((sp - sm) / (2.0 * h))[VA, VB])
    return np.array(cols).T


def test_AD8_tangent_matches_finite_difference_and_coupling_matters() -> None:
    m = make(1.0)
    store = StateStore(m, 1)
    h = run_uniaxial(m, 20, eps_end=0.04, store=store)
    e = h.strain[-1] + np.array([0.004, -0.0015, -0.0015, 0.0008, 0.0005, 0.001])
    req = MaterialRequest(energy=False, step=TimeStep(0.04, 0.01))
    u = m.update(T.voigt_to_strain(e)[None], store.committed, req)
    assert u.diagnostics["plastic"][0] == 1.0 and u.diagnostics["delta_T"][0] > 0.0
    C = u.tangent[0]
    Cfd = _fd_tangent(m, store.committed, e, req)
    assert np.max(np.abs(C - Cfd)) / np.max(np.abs(C)) <= 1e-6
    # negative control: the isothermal Perzyna tangent (no dsigma/dT x dT/deps term) at sigma_y(T_{n+1})
    iso = m.isothermal(float(u.diagnostics["temperature"][0]))
    ci = {"plastic_strain": store.committed["plastic_strain"], "alpha": store.committed["alpha"]}
    Ciso = iso.update(T.voigt_to_strain(e)[None], ci, req).tangent[0]
    miss = np.max(np.abs(Ciso - Cfd)) / np.max(np.abs(C))
    print(f"AD8 coupling share of the tangent: {miss:.3e}")
    assert miss >= 1e-3
    # post hoc (the record predicted a non-symmetric tangent, which was wrong): both factors of the coupling are
    # proportional to the deviatoric direction, so the rank-one term n (x) n is symmetric
    assert np.allclose(C, C.T, rtol=0.0, atol=1e-9 * np.max(np.abs(C)))
    assert math.isfinite(miss)
