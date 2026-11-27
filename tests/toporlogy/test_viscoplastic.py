"""Physical J2, Perzyna, Eyring and Drucker-Prager (``toporlogy.core.viscoplastic``), checks R1-R7.

Thresholds are the preregistered ones of ``docs/foundations/prereg/R_rate.md`` (committed before the first run; two
thresholds were corrected before any run, see its "Amendments" section). All constants are synthetic, not material
data. References are closed forms written in this file, ``scipy`` ODE / root finders, mpmath, or central finite
differences of the production stress; tests labelled "cross-implementation" compare with the legacy law.
"""
from __future__ import annotations

import math
import warnings

import mpmath
import numpy as np
import pytest
import scipy.constants as sc
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (J2Plasticity, Kinematics, MaterialRequest, StateStore, TimeStep,
                                         drive_material_point)
from toporlogy.core.viscoplastic import (DruckerPrager, Perzyna, PhysicalJ2, eyring_log_rate, eyring_rate,
                                         eyring_rate_molar, eyring_stress, eyring_stress_from_log_rate,
                                         eyring_stress_molar)

E0, NU = 2.0e9, 0.35                       # synthetic
SY, HH = 20.0e6, 100.0e6                   # synthetic
VA = np.array([p[0] for p in T.VOIGT_3D])
VB = np.array([p[1] for p in T.VOIGT_3D])
CTRL_XX = [True, False, False, False, False, False]
CTRL_XY = [False, False, False, False, False, True]
E_STRESS_TOL = 1e-5                        # Pa, absolute tolerance of the mixed-control Newton (stresses ~1e7 Pa)


def KG(E: float = E0, nu: float = NU) -> tuple[float, float]:
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def eps3(v: list[float] | np.ndarray) -> np.ndarray:
    return T.voigt_to_strain(np.asarray(v, float))[None]


def uni_path(strains: np.ndarray) -> np.ndarray:
    out = np.zeros((len(strains), 6))
    out[:, 0] = strains
    return out


def drive_uni(model, strains, times=None):
    k = len(strains)
    return drive_material_point(model, uni_path(np.asarray(strains, float)), np.zeros((k, 6)), CTRL_XX,
                                tol=E_STRESS_TOL, times=times)


def plastic_trial(model, eps_voigt, committed=None, step=None):
    st = model.initial_state(1) if committed is None else committed
    return model.update(eps3(eps_voigt), st, MaterialRequest(step=step, energy=True))


def sig_voigt(u) -> np.ndarray:
    return u.stress[0][VA, VB]


# ================================================================================================ R1: physical J2
def j2_uniaxial_ref(eps: np.ndarray, E: float, sy: float = SY, H: float = HH) -> np.ndarray:
    return np.minimum(E * eps, (sy + H * eps) / (1.0 + H / E))


def test_R1a_uniaxial_closed_form():
    K, G = KG()
    eps = np.linspace(0.0, 0.03, 61)[1:]
    h = drive_uni(PhysicalJ2(K, G, SY, HH), eps)
    err = np.abs(h.stress[:, 0] - j2_uniaxial_ref(eps, E0)).max() / SY
    assert err <= 1e-9, err
    assert np.abs(h.stress[:, 1:]).max() <= 1e-4          # lateral stress free (mixed control)


def yield_stress_by_bisection(model, E: float, nu: float) -> float:
    st = model.initial_state(1)
    lo, hi = 0.0, 0.2
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        eps = np.zeros((1, 3, 3))
        eps[0, 0, 0] = mid
        eps[0, 1, 1] = eps[0, 2, 2] = -nu * mid
        if model.update(eps, st, MaterialRequest()).diagnostics["plastic"][0] > 0.5:
            hi = mid
        else:
            lo = mid
    return E * hi


def test_R1b_P2_doubling_E_keeps_yield_stress():
    ys, slopes = [], []
    for E in (E0, 2.0 * E0):
        K, G = KG(E)
        m = PhysicalJ2(K, G, SY, HH)
        ys.append(yield_stress_by_bisection(m, E, NU))
        eps = np.linspace(0.0, 0.03, 61)[1:]
        h = drive_uni(m, eps)
        assert np.abs(h.stress[:, 0] - j2_uniaxial_ref(eps, E)).max() / SY <= 1e-9
        slopes.append(h.stress[0, 0] / eps[0])
    assert all(abs(y - SY) / SY <= 1e-9 for y in ys), ys
    assert abs(slopes[1] / slopes[0] - 2.0) <= 2e-12, slopes
    # negative control: a model whose yield stress scales with E (yield defined relative to the modulus) is detected
    K, G = KG(2.0 * E0)
    scaled = yield_stress_by_bisection(PhysicalJ2(K, G, SY * 2.0, HH), 2.0 * E0, NU)
    assert abs(scaled - SY) / SY > 0.5


def test_R1c_load_unload_reverse():
    K, G = KG()
    peak, mid, rev = 0.03, 0.015, -0.02
    path = np.concatenate([np.linspace(0, peak, 61)[1:], np.linspace(peak, mid, 31)[1:], np.linspace(mid, rev, 71)[1:]])
    h = drive_uni(PhysicalJ2(K, G, SY, HH), path)
    sp = (SY + HH * peak) / (1 + HH / E0)
    p1 = peak - sp / E0
    kap1 = SY + HH * p1
    ref = np.empty_like(path)
    for i, e in enumerate(path):
        if i < 60:
            ref[i] = j2_uniaxial_ref(np.array(e), E0)
        else:
            su = sp - E0 * (peak - e)
            pr = (2 * p1 - e - SY / E0) / (1 + HH / E0)
            ref[i] = max(su, -(SY + HH * pr)) if e < peak else su
    assert np.abs(h.stress[:, 0] - ref).max() / SY <= 1e-9
    # elastic unloading leaves the plastic strain unchanged
    assert h.states[60]["alpha"][0] == pytest.approx(p1, abs=1e-12)
    assert h.states[89]["alpha"][0] == h.states[60]["alpha"][0]
    assert kap1 > SY


def test_R1d_pure_shear_hydrostatic_voce():
    K, G = KG()
    m = PhysicalJ2(K, G, SY, HH)
    gam = np.linspace(0, 0.05, 101)[1:]
    st = np.zeros((len(gam), 6))
    st[:, 5] = gam
    h = drive_material_point(m, st, np.zeros_like(st), CTRL_XY, tol=E_STRESS_TOL)
    s3 = math.sqrt(3.0)
    ref = np.where(s3 * G * gam <= SY, G * gam, (gam + s3 * SY / HH) / (1 / G + 3 / HH))
    assert np.abs(h.stress[:, 5] - ref).max() / SY <= 1e-9
    # hydrostatic strain: no yield, no plastic strain
    e = 0.5
    h2 = drive_material_point(m, np.array([[e, e, e, 0, 0, 0.0]]), np.zeros((1, 6)), [True] * 6)
    assert abs(h2.stress[0, 0] - 3 * K * e) <= 1e-9 * 3 * K * e
    assert h2.states[0]["alpha"][0] == 0.0
    # Voce uniaxial against a scalar root of kappa(p) = E (eps - p)
    s_inf, dl = 30e6, 200.0
    mv = PhysicalJ2(K, G, SY, HH, saturation_stress=s_inf, saturation_rate=dl)
    eps = np.linspace(0, 0.03, 61)[1:]
    hv = drive_uni(mv, eps)
    ref = []
    for e in eps:
        phi = lambda p: SY + HH * p + (s_inf - SY) * (1 - math.exp(-dl * p)) - E0 * (e - p)   # noqa: E731
        ref.append(E0 * e if phi(0.0) >= 0 else E0 * (e - brentq(phi, 0.0, e, xtol=1e-18, rtol=4 * np.finfo(float).eps)))
    assert np.abs(hv.stress[:, 0] - np.array(ref)).max() / SY <= 1e-9


def test_R1e_cross_implementation_legacy_j2():
    """Cross-implementation consistency (not a closed form): PhysicalJ2 = E x legacy unit-modulus J2."""
    K, G = KG()
    for kw in ({}, dict(saturation_stress=30e6, saturation_rate=200.0)):
        phys = PhysicalJ2(K, G, SY, HH, **kw)
        legacy_kw = {k: (v / E0 if k == "saturation_stress" else v) for k, v in kw.items()}
        leg = J2Plasticity(NU, SY / E0, HH / E0, **legacy_kw)
        c1, c2 = phys.initial_state(1), leg.initial_state(1)
        for v in ([0.02, -0.008, 0.004, 0.01, -0.006, 0.012], [0.01, 0.015, -0.01, -0.012, 0.02, -0.005]):
            kin = Kinematics(eps3(v), "small")
            r1 = phys.evaluate(kin, None, c1, MaterialRequest())
            r2 = leg.evaluate(kin, None, c2, MaterialRequest())
            assert r1.diagnostics["plastic"][0] > 0.5
            assert np.abs(r1.stress - E0 * r2.stress).max() / np.abs(r1.stress).max() <= 1e-10
            assert abs(r1.trial_state["alpha"][0] - r2.trial_state["alpha"][0]) <= 1e-10 * r2.trial_state["alpha"][0]
            assert np.abs(r1.tangent - E0 * r2.tangent).max() / np.abs(r1.tangent).max() <= 1e-9
            c1, c2 = r1.trial_state, r2.trial_state


# ================================================================================================ R2: Perzyna residual
TAU0, SREF = 1.0, 10.0e6
STEP1 = [0.02, -0.008, 0.004, 0.01, -0.006, 0.012]
STEP2 = [0.01, 0.015, -0.01, -0.012, 0.02, -0.005]


def ref_perzyna_step(K, G, sy, H, tau0, sref, m, ep, p, eps, dt):
    """Independent dense-numpy reference: returns (sigma 3x3, ep, p, dp)."""
    ee = eps - ep
    tr = np.trace(ee)
    s = 2 * G * (ee - tr / 3 * np.eye(3))
    q = math.sqrt(1.5 * np.sum(s * s))
    ftr = q - (sy + H * p)
    if ftr <= 0 or dt == 0:
        return s + K * tr * np.eye(3), ep, p, 0.0
    Xi = 3 * G + H
    if m == 1:
        dp = ftr / (sref * tau0 / dt + Xi)
    else:
        res = lambda d: d - dt / tau0 * max((ftr - Xi * d) / sref, 0.0) ** m     # noqa: E731
        dp = brentq(res, 0.0, ftr / Xi, xtol=1e-24, rtol=4 * np.finfo(float).eps, maxiter=500)
    n = s / np.sqrt(np.sum(s * s))
    sig = (1 - 3 * G * dp / q) * s + K * tr * np.eye(3)
    return sig, ep + math.sqrt(1.5) * dp * n, p + dp, dp


@pytest.mark.parametrize("m,tol_dp", [(1.0, 1e-11), (2.0, 1e-9), (0.5, 1e-9)])
def test_R2a_active_branch_matches_reference(m, tol_dp):
    K, G = KG()
    dt = 1e-3
    model = Perzyna(K, G, SY, HH, TAU0, SREF, m)
    committed = model.initial_state(1)
    ep, p = np.zeros((3, 3)), 0.0
    for v in (STEP1, STEP2):
        e = eps3(v)
        u = model.update(e, committed, MaterialRequest(step=TimeStep(0.0, dt)))
        sig, ep_r, p_r, dp_r = ref_perzyna_step(K, G, SY, HH, TAU0, SREF, m, ep, p, e[0], dt)
        assert dp_r > 0.0
        dp = u.diagnostics["delta_p"][0]
        assert abs(dp - dp_r) <= tol_dp * dp_r, (dp, dp_r)
        assert np.abs(u.stress[0] - sig).max() <= 1e-9 * SY
        committed = u.trial_state
        ep, p = ep_r, p_r
    # sanity: the active branch is genuinely viscous (neither elastic nor the rate-independent return)
    pj = PhysicalJ2(K, G, SY, HH).update(eps3(STEP1), PhysicalJ2(K, G, SY, HH).initial_state(1), MaterialRequest())
    assert 0.0 < dp_r < pj.diagnostics["delta_p"][0] * 5


def fd_tangent(model, committed, v, step, h=1e-6):
    C = np.zeros((6, 6))
    v = np.asarray(v, float)
    for c in range(6):
        d = np.zeros(6)
        d[c] = h
        sp = sig_voigt(model.update(eps3(v + d), committed, MaterialRequest(step=step, tangent=False)))
        sm = sig_voigt(model.update(eps3(v - d), committed, MaterialRequest(step=step, tangent=False)))
        C[:, c] = (sp - sm) / (2 * h)
    return C


@pytest.mark.parametrize("m", [1.0, 2.0, 0.5])
def test_R2b_perzyna_tangent_fd(m):
    K, G = KG()
    step = TimeStep(0.0, 1e-3)
    model = Perzyna(K, G, SY, HH, TAU0, SREF, m)
    committed = model.update(eps3(STEP1), model.initial_state(1), MaterialRequest(step=step)).trial_state
    assert committed["alpha"][0] > 0
    u = model.update(eps3(STEP2), committed, MaterialRequest(step=step))
    assert u.diagnostics["plastic"][0] > 0.5
    Cfd = fd_tangent(model, committed, STEP2, step)
    assert np.abs(Cfd - u.tangent[0]).max() / np.abs(u.tangent[0]).max() <= 1e-6
    # negative control: the elastic tangent fails the same metric
    assert np.abs(Cfd - model.Ce6).max() / np.abs(u.tangent[0]).max() >= 1e-2


def test_R2b_physical_j2_tangent_fd():
    K, G = KG()
    for kw in ({}, dict(saturation_stress=30e6, saturation_rate=200.0)):
        model = PhysicalJ2(K, G, SY, HH, **kw)
        committed = model.update(eps3(STEP1), model.initial_state(1), MaterialRequest()).trial_state
        u = model.update(eps3(STEP2), committed, MaterialRequest())
        assert u.diagnostics["plastic"][0] > 0.5
        Cfd = fd_tangent(model, committed, STEP2, None)
        assert np.abs(Cfd - u.tangent[0]).max() / np.abs(u.tangent[0]).max() <= 1e-6


def test_R2c_elastic_below_yield_dt0_and_missing_step():
    K, G = KG()
    model = Perzyna(K, G, SY, HH, TAU0, SREF, 2.0)
    v = [0.002, -0.001, 0.0005, 0.001, 0.0, -0.001]
    u = model.update(eps3(v), model.initial_state(1), MaterialRequest(step=TimeStep(0.0, 5.0)))
    assert np.abs(sig_voigt(u) - model.Ce6 @ np.asarray(v)).max() <= 1e-12 * np.abs(model.Ce6 @ np.asarray(v)).max()
    assert np.all(u.trial_state["alpha"] == 0) and np.all(u.trial_state["plastic_strain"] == 0)
    assert u.dissipation[0] == 0.0
    # dt = 0: no flow from a plastic committed state, stress = elastic trial of the reference
    committed = model.update(eps3(STEP1), model.initial_state(1), MaterialRequest(step=TimeStep(0.0, 1e-3))).trial_state
    u0 = model.update(eps3(STEP2), committed, MaterialRequest(step=TimeStep(1.0, 0.0)))
    assert np.array_equal(u0.trial_state["plastic_strain"], committed["plastic_strain"])
    assert np.array_equal(u0.trial_state["alpha"], committed["alpha"])
    sig, *_ = ref_perzyna_step(K, G, SY, HH, TAU0, SREF, 2.0, committed["plastic_strain"][0], 0.0, eps3(STEP2)[0], 0.0)
    assert np.abs(u0.stress[0] - sig).max() <= 1e-12 * np.abs(sig).max()
    with pytest.raises(ValueError, match="time step"):
        model.update(eps3(STEP1), model.initial_state(1), MaterialRequest())


@pytest.mark.parametrize("kw", [dict(K=0.0), dict(G=-1.0), dict(yield_stress=0.0), dict(hardening=-1.0),
                                dict(tau0=0.0), dict(sigma_ref=-1.0), dict(m=0.0), dict(m=float("nan"))])
def test_perzyna_refuses_invalid_parameters(kw):
    K, G = KG()
    base = dict(K=K, G=G, yield_stress=SY, hardening=HH, tau0=TAU0, sigma_ref=SREF, m=2.0)
    base.update(kw)
    with pytest.raises(ValueError):
        Perzyna(**base)


# ================================================================================================ R3: rate sweep
PZ = dict(yield_stress=SY, hardening=HH, tau0=1.0, sigma_ref=10.0e6, m=2.0)


def ode_end_stress(rate: float, eps_end: float = 0.03) -> float:
    pz = PZ

    def rhs(t, y):
        return [max((E0 * (rate * t - y[0]) - pz["yield_stress"] - pz["hardening"] * y[0]) / pz["sigma_ref"], 0.0)
                ** pz["m"] / pz["tau0"]]
    t_end = eps_end / rate
    sol = solve_ivp(rhs, (0.0, t_end), [0.0], method="Radau", rtol=1e-12, atol=1e-16)
    return E0 * (eps_end - sol.y[0, -1])


def ramp_end_stress(model, rate: float, n: int, eps_end: float = 0.03) -> float:
    strains = np.linspace(0.0, eps_end, n + 1)[1:]
    times = np.linspace(0.0, eps_end / rate, n + 1)[1:]
    return float(drive_uni(model, strains, times=times).stress[-1, 0])


@pytest.fixture(scope="module")
def ramp_table():
    K, G = KG()
    model = Perzyna(K, G, **PZ)
    out = {}
    for r in (1e-3, 0.1, 1.0):
        ode = ode_end_stress(r)
        out[r] = (ode, {n: ramp_end_stress(model, r, n) for n in (25, 50, 100, 200)})
    return out


@pytest.mark.parametrize("rate", [
    pytest.param(1e-3, marks=pytest.mark.xfail(strict=True, reason=(
        "preregistered hypothesis wrong at this rate: after the yield transient (exponential, linearised rate estimate ~13 1/s) the "
        "response sits on the steady state where backward Euler is exact; the end-stress errors are 1.9e-8, 3.7e-8, "
        "3.7e-8, 1.5e-8 Pa (1 ulp of 2.2e7 Pa) for n = 25, 50, 100, 200, i.e. the floor, not a first-order series"))),
    0.1, 1.0])
def test_R3a_time_convergence_against_ode(ramp_table, rate):
    ode, tab = ramp_table[rate]
    err = {n: abs(s - ode) for n, s in tab.items()}
    ns = (25, 50, 100, 200)
    for a, b in zip(ns[:-1], ns[1:]):
        assert err[a] >= 1.5 * err[b], (rate, err)
    assert err[200] <= err[25] / 5.0, (rate, err)
    rich = 2.0 * tab[200] - tab[100]
    assert abs(rich - ode) <= 3e-3 * ode, (rate, rich, ode)


def test_R3a_post_hoc_rate_1e3_matches_ode_to_roundoff(ramp_table):
    """Post hoc (written after the first run, not preregistered): at 1e-3 1/s all increments agree with the ODE to
    1e-12 relative (the quasi-static steady state is integrated exactly by backward Euler)."""
    ode, tab = ramp_table[1e-3]
    assert max(abs(s - ode) for s in tab.values()) <= 1e-12 * ode


def test_R3b_rate_sensitivity_and_rate_independent_control(ramp_table):
    rates = (1e-3, 0.1, 1.0)
    ode = [ramp_table[r][0] for r in rates]
    prod = [ramp_table[r][1][200] for r in rates]
    assert ode[0] < ode[1] < ode[2]
    assert prod[0] < prod[1] < prod[2]
    K, G = KG()
    j2 = [ramp_end_stress(PhysicalJ2(K, G, SY, HH), r, 50) for r in rates]
    assert max(abs(s - j2[0]) for s in j2) <= 1e-12 * j2[0]        # the metric sees no rate effect in the rate-independent law


def test_R3c_limits_tau0():
    K, G = KG()
    n, rate, e_end = 100, 1.0, 0.03
    strains = np.linspace(0.0, e_end, n + 1)[1:]
    times = np.linspace(0.0, e_end / rate, n + 1)[1:]
    hj = drive_uni(PhysicalJ2(K, G, SY, HH), strains, times=times)
    hv = drive_uni(Perzyna(K, G, **{**PZ, "tau0": 1e-12}), strains, times=times)
    assert np.abs(hv.stress - hj.stress).max() / np.abs(hj.stress).max() <= 1e-5
    he = drive_uni(Perzyna(K, G, **{**PZ, "tau0": 1e12}), strains, times=times)
    assert np.abs(he.stress[:, 0] - E0 * strains).max() / (E0 * e_end) <= 1e-6
    assert max(s["alpha"][0] for s in he.states) <= 1e-9


# ================================================================================================ R4: hold
def test_R4a_strain_hold_overstress_decays():
    K, G = KG()
    model = Perzyna(K, G, **PZ)
    ramp_t = np.linspace(0.0, 0.02, 21)[1:]
    hold_t = 0.02 + np.geomspace(1e-3, 1e4, 120)
    times = np.concatenate([ramp_t, hold_t])
    strains = np.concatenate([ramp_t, np.full(120, 0.02)])
    h = drive_uni(model, strains, times=times)
    sig = h.stress[:, 0]
    over = np.array([d["overstress"] for d in h.diagnostics])
    ov, sg = over[20:], sig[20:]
    assert over[19] > 0.0 and np.all(ov >= 0.0)
    assert np.all(np.diff(ov) <= 1e-9 * SY), np.diff(ov).max()
    assert np.all(np.diff(sg) <= 1e-9 * SY)
    assert ov[0] > 1e5 and ov[-1] < ov[0] * 1e-3
    s_eq = (SY + HH * 0.02) / (1 + HH / E0)
    assert abs(sig[-1] - s_eq) / s_eq <= 1e-5, (sig[-1], s_eq)


def test_R4b_hold_below_yield_no_flow():
    K, G = KG()
    model = Perzyna(K, G, **PZ)
    ramp_t = np.linspace(0.0, 0.005, 6)[1:]
    hold_t = 0.005 + np.geomspace(1.0, 1e6, 10)
    strains = np.concatenate([ramp_t, np.full(10, 0.005)])
    h = drive_uni(model, strains, times=np.concatenate([ramp_t, hold_t]))
    assert all(np.all(s["alpha"] == 0.0) for s in h.states)
    assert np.abs(h.stress[:, 0] - E0 * strains).max() <= 1e-12 * E0 * 0.005


# ================================================================================================ R5: Eyring
A_E, V_E, T_E = 1e-3, 1e-27, 300.0          # synthetic: 1/s, m^3, K
KT = sc.k * T_E


def mp_rate(tau: float) -> mpmath.mpf:
    mpmath.mp.dps = 50
    x = mpmath.mpf(tau) * mpmath.mpf(V_E) / (mpmath.mpf(sc.k) * mpmath.mpf(T_E))
    return mpmath.mpf(A_E) * mpmath.sinh(x)


def test_R5a_round_trip_and_value():
    assert sc.k == 1.380649e-23 and abs(sc.N_A * sc.k - 8.31446261815324) < 1e-12
    xs = [1e-8, 1e-3, 0.5, 3.0, 50.0, 300.0, 700.0]
    for x in xs + [-v for v in xs]:
        tau = x * KT / V_E
        r = eyring_rate(tau, A_E, V_E, T_E)
        ref = mp_rate(tau)
        assert abs(mpmath.mpf(float(r)) - ref) <= (1e-13 + 5e-16 * abs(x)) * abs(ref), (x, r, ref)
        back = eyring_stress(r, A_E, V_E, T_E)
        assert abs(back - tau) <= 1e-12 * abs(tau), (x, back, tau)


def test_R5b_extreme_arguments_without_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for x in (5000.0, -5000.0):
            tau = x * KT / V_E
            L = eyring_log_rate(tau, A_E, V_E, T_E)
            assert np.isfinite(L)
            assert abs(L - (math.log(A_E) + abs(x) - math.log(2.0))) <= 1e-12 * abs(x)
            back = eyring_stress_from_log_rate(L, A_E, V_E, T_E)
            assert abs(back - abs(tau)) <= 1e-12 * abs(tau)
            r = eyring_rate(tau, A_E, V_E, T_E)             # overflows only in the value: +-inf, no warning
            assert np.isinf(r) and np.sign(r) == np.sign(x)
        for rate in (1e300, -1e300, 1e-300):
            tau = eyring_stress(rate, A_E, V_E, T_E)
            ref = KT / V_E * math.copysign(1.0, rate) * (math.log(2.0) + math.log(abs(rate)) - math.log(A_E)) \
                if abs(rate) > 1.0 else None
            if ref is not None:
                assert abs(tau - ref) <= 1e-12 * abs(ref)
            else:
                assert abs(tau - KT / V_E * rate / A_E) <= 1e-12 * abs(tau)
        assert eyring_log_rate(0.0, A_E, V_E, T_E) == -np.inf
    with np.errstate(all="ignore"):
        naive = np.arcsinh(np.sinh(np.float64(800.0)))        # negative control: the naive route overflows
    assert not np.isfinite(naive)


def test_R5c_units_limits_molar_and_validation():
    rate = 3.7e-2
    t1 = eyring_stress(rate, A_E, V_E, T_E)
    assert abs(eyring_stress(rate, A_E, V_E, 2 * T_E) - 2 * t1) <= 1e-14 * t1
    x = 1e-6
    assert abs(eyring_rate(x * KT / V_E, A_E, V_E, T_E) / (A_E * x) - 1.0) <= 1e-8
    xb = 40.0
    r = eyring_rate(xb * KT / V_E, A_E, V_E, T_E)
    assert abs(eyring_stress(r, A_E, V_E, T_E) - KT / V_E * (math.log(2 * r / A_E))) <= 1e-12 * KT / V_E * xb
    tau = 2.5e6
    Vm = V_E * sc.N_A
    assert abs(eyring_rate_molar(tau, A_E, Vm, T_E) - eyring_rate(tau, A_E, Vm / sc.N_A, T_E)) <= 1e-12 * abs(
        eyring_rate(tau, A_E, V_E, T_E))
    assert abs(eyring_stress_molar(rate, A_E, Vm, T_E) - t1) <= 1e-12 * t1
    for bad in ((0.0, V_E, T_E), (A_E, -1.0, T_E), (A_E, V_E, 0.0), (A_E, V_E, float("nan"))):
        with pytest.raises(ValueError):
            eyring_rate(1e6, *bad)
        with pytest.raises(ValueError):
            eyring_stress(1e-3, *bad)
    with pytest.raises(ValueError):
        eyring_rate(float("nan"), A_E, V_E, T_E)


# ================================================================================================ R6: Drucker-Prager
A_DP, B0, H_DP = 0.6, 15.0e6, 50.0e6


def dp_models(h: float):
    K, G = KG()
    return {"assoc": DruckerPrager(K, G, A_DP, B0, h), "nonassoc": DruckerPrager(K, G, A_DP, B0, h, a_flow=0.2)}


def dp_tension_ref(eps, h, ag):
    sp = (B0 + h * eps / (1 + ag / 3)) / ((1 + A_DP / 3) + h / (E0 * (1 + ag / 3)))
    return np.minimum(E0 * eps, sp)


def dp_compression_ref(eps, h, ag):                # eps < 0, returns the (negative) stress
    c = (B0 + h * (-eps) / (1 - ag / 3)) / ((1 - A_DP / 3) + h / (E0 * (1 - ag / 3)))
    return -np.minimum(-E0 * eps, c)


@pytest.mark.parametrize("h", [0.0, H_DP])
def test_R6a_uniaxial_tension_compression_closed_form(h):
    for name, m in dp_models(h).items():
        ag = m.a_g
        et = np.linspace(0, 0.05, 101)[1:]
        ht = drive_uni(m, et)
        assert np.abs(ht.stress[:, 0] - dp_tension_ref(et, h, ag)).max() / B0 <= 1e-9, name
        hc = drive_uni(m, -et)
        assert np.abs(hc.stress[:, 0] - dp_compression_ref(-et, h, ag)).max() / B0 <= 1e-9, name
        if h == 0.0:
            assert abs(ht.stress[-1, 0] - B0 / (1 + A_DP / 3)) <= 1e-9 * B0
            assert abs(hc.stress[-1, 0] + B0 / (1 - A_DP / 3)) <= 1e-9 * B0


def test_R6b_pressure_sign_negative_control():
    m = dp_models(0.0)["assoc"]
    et = np.linspace(0, 0.05, 101)[1:]
    t_plateau = drive_uni(m, et).stress[-1, 0]
    c_plateau = -drive_uni(m, -et).stress[-1, 0]
    assert t_plateau < c_plateau                                  # pressure strengthening: compression is stronger
    flipped_tension = B0 / (1 - A_DP / 3)                         # what a wrong sign of the pressure term would give
    assert (flipped_tension - t_plateau) / t_plateau >= 0.05
    # the metric of R6a applied to the flipped closed form fails by far more than its threshold
    assert abs(t_plateau - flipped_tension) / B0 > 1e-9


@pytest.mark.parametrize("h", [0.0, H_DP])
def test_R6c_hydrostatic_and_apex(h):
    K, G = KG()
    for name, m in dp_models(h).items():
        ag = m.a_g
        # compression never yields
        for e in (-0.01, -0.05):
            u = plastic_trial(m, [e, e, e, 0, 0, 0])
            assert u.diagnostics["plastic"][0] == 0.0
            assert abs(u.stress[0, 0, 0] - 3 * K * e) <= 1e-12 * abs(3 * K * e)
        e_ap = B0 / (3 * A_DP * K)
        assert plastic_trial(m, [0.99 * e_ap] * 3 + [0, 0, 0]).diagnostics["plastic"][0] == 0.0
        assert plastic_trial(m, [1.01 * e_ap] * 3 + [0, 0, 0]).diagnostics["apex"][0] == 1.0
        for e in (0.01, 0.02):
            u = plastic_trial(m, [e, e, e, 0, 0, 0])
            assert u.diagnostics["apex"][0] == 1.0
            dl = (3 * A_DP * K * e - B0) / (A_DP * K * ag + h)
            sig_iso = (B0 + h * dl) / A_DP
            S = u.stress[0]
            assert abs(S[0, 0] - sig_iso) <= 1e-12 * sig_iso and abs(S[1, 1] - sig_iso) <= 1e-12 * sig_iso
            assert np.abs(S - sig_iso * np.eye(3)).max() <= 1e-9 * sig_iso
            assert abs(np.trace(u.trial_state["plastic_strain"][0]) - ag * dl) <= 1e-12 * ag * dl
            assert abs(u.trial_state["kappa"][0] - dl) <= 1e-12 * dl


def dp_cone_state(m):
    return m.update(eps3(STEP1), m.initial_state(1), MaterialRequest()).trial_state


@pytest.mark.parametrize("h", [0.0, H_DP])
def test_R6d_tangent_fd_cone_apex_symmetry(h):
    K, G = KG()
    for name, m in dp_models(h).items():
        committed = dp_cone_state(m)
        assert committed["kappa"][0] > 0
        u = m.update(eps3(STEP2), committed, MaterialRequest())
        assert u.diagnostics["plastic"][0] > 0.5 and u.diagnostics["apex"][0] == 0.0, name
        C = u.tangent[0]
        Cfd = fd_tangent(m, committed, STEP2, None)
        scale = max(np.abs(C).max(), 1e-3 * K)
        assert np.abs(Cfd - C).max() / scale <= 1e-6, (name, h)
        assert np.abs(Cfd - m.Ce6).max() / scale >= 1e-2             # negative control: elastic tangent fails the metric
        asym = np.linalg.norm(C - C.T) / np.linalg.norm(C)
        if name == "assoc":
            assert asym <= 1e-10, asym
        else:
            assert asym >= 1e-3, asym
        # apex branch: hydrostatic tension plus a small deviatoric perturbation
        va = [0.02 + 1e-4, 0.02 - 1e-4, 0.02, 3e-4, -2e-4, 1e-4]
        ua = m.update(eps3(va), m.initial_state(1), MaterialRequest())
        assert ua.diagnostics["apex"][0] == 1.0
        Ca = ua.tangent[0]
        Cfa = fd_tangent(m, m.initial_state(1), va, None)
        assert np.abs(Cfa - Ca).max() / max(np.abs(Ca).max(), 1e-3 * K) <= 1e-6


def test_R6e_refusals():
    K, G = KG()
    ok = dict(K=K, G=G, a=0.6, b0=B0, hardening=H_DP)
    DruckerPrager(**ok)
    DruckerPrager(**{**ok, "a_flow": 0.6})
    for bad in (dict(a=0.0), dict(a=-0.1), dict(a=3.0), dict(a=3.5), dict(a_flow=0.0), dict(a_flow=-0.2),
                dict(a_flow=0.7), dict(b0=0.0), dict(hardening=-1.0), dict(K=0.0), dict(G=-1.0),
                dict(a=float("nan")), dict(b0=float("inf"))):
        with pytest.raises(ValueError):
            DruckerPrager(**{**ok, **bad})


def cyclic_strains(n: int = 80, amp: float = 0.02) -> np.ndarray:
    t = np.linspace(0, 1, n + 1)[1:, None]
    ph = np.array([0.0, 1.1, 2.3, 0.7, 1.9, 2.9])[None]
    fr = np.array([2.0, 3.0, 2.0, 1.0, 2.0, 3.0])[None]
    return amp * np.sin(2 * np.pi * fr * t + ph)


def test_R6f_dissipation_nonnegative_and_plastic_volume():
    path = cyclic_strains()
    for h in (0.0, H_DP):
        for name, m in dp_models(h).items():
            if name == "nonassoc" and h > 0:
                continue                              # not claimed admissible (R6g)
            hist = drive_material_point(m, path, np.zeros_like(path), [True] * 6)
            assert hist.dissipation.min() >= -1e-6, (name, h, hist.dissipation.min())
            assert hist.dissipation.sum() > 0
    m = dp_models(0.0)["nonassoc"]
    u = m.update(eps3([0.02, -0.012, -0.004, 0.01, -0.006, 0.012]), m.initial_state(1), MaterialRequest())
    assert u.diagnostics["apex"][0] == 0.0 and u.diagnostics["plastic"][0] == 1.0
    assert abs(u.diagnostics["plastic_volume_strain"][0] - m.a_g * u.diagnostics["delta_lambda"][0]) <= 1e-12 * \
        u.diagnostics["plastic_volume_strain"][0]


def test_R6g_documented_inadmissible_region():
    K, G = KG()
    a, ag, h = 0.6, 0.06, 50e6
    m = DruckerPrager(K, G, a, B0, h, a_flow=ag)
    e = 0.1
    u = plastic_trial(m, [e, e, e, 0, 0, 0])
    dl = (3 * a * K * e - B0) / (a * K * ag + h)
    ref = dl * (ag * B0 / a + h * dl * (ag / a - 0.5))
    assert ref < 0 and u.diagnostics["apex"][0] == 1.0
    assert abs(u.dissipation[0] - ref) <= 1e-12 * abs(ref)


# ================================================================================================ R7: energy, rollback
MODEL_FACTORIES = {
    "j2": lambda: PhysicalJ2(*KG(), SY, HH),
    "j2_voce": lambda: PhysicalJ2(*KG(), SY, HH, saturation_stress=30e6, saturation_rate=200.0),
    "perzyna": lambda: Perzyna(*KG(), **PZ),
    "dp_assoc": lambda: DruckerPrager(*KG(), A_DP, B0, H_DP),
}


def test_R7a_dissipation_nonnegative_cyclic_multiaxial():
    path = cyclic_strains(120, 0.02)
    for name in ("j2", "j2_voce", "perzyna"):
        m = MODEL_FACTORIES[name]()
        times = None
        if name == "perzyna":
            times = np.cumsum(np.where(np.arange(120) % 7 == 3, 0.0, 1e-3))      # includes dt = 0 steps
        hist = drive_material_point(m, path, np.zeros_like(path), [True] * 6, times=times)
        assert hist.dissipation.min() >= -1e-6, (name, hist.dissipation.min())
        assert hist.dissipation.sum() > 0, name


def test_R7b_rollback_and_restart_identical():
    path = cyclic_strains(10, 0.02)
    times = np.cumsum(np.full(10, 2e-3))
    for name, make in MODEL_FACTORIES.items():
        m = make()
        timed = times if name == "perzyna" else None
        # repeated evaluation from one committed state: bitwise identical, committed arrays untouched
        store = StateStore(m, 1)
        req = MaterialRequest(energy=True, step=TimeStep(0.0, 1e-3) if timed is not None else None)
        kin = Kinematics(eps3(path[0] * 1.5), "small")
        r1 = m.evaluate(kin, None, store.committed, req)
        store.set_trial(r1.trial_state)
        store.rollback()                                         # rejected step
        assert all(np.all(v == 0) for v in store.committed.values())
        r2 = m.evaluate(kin, None, store.committed, req)
        assert np.array_equal(r1.stress, r2.stress) and np.array_equal(r1.tangent, r2.tangent)
        assert all(np.array_equal(r1.trial_state[k], r2.trial_state[k]) for k in r1.trial_state)
        # restart: 10 steps at once = 5 steps + restart from the checkpoint
        full = drive_material_point(m, path, np.zeros_like(path), [True] * 6, times=timed)
        a = drive_material_point(m, path[:5], np.zeros((5, 6)), [True] * 6, times=None if timed is None else timed[:5])
        st = StateStore(m, 1)
        st.restore(a.states[-1])
        b = drive_material_point(m, path[5:], np.zeros((5, 6)), [True] * 6, store=st, strain0=a.strain[-1],
                                 times=None if timed is None else timed[5:], t0=0.0 if timed is None else timed[4])
        assert np.array_equal(np.concatenate([a.stress, b.stress]), full.stress), name
        assert np.array_equal(np.concatenate([a.strain, b.strain]), full.strain), name
        for k in full.states[-1]:
            assert np.array_equal(b.states[-1][k], full.states[-1][k]), (name, k)


def test_R7c_energy_balance_converges_under_refinement():
    K, G = KG()
    errs = []
    for n in (100, 200, 400):
        path = np.concatenate([np.linspace(0, 0.03, n // 2 + 1)[1:], np.linspace(0.03, 0.0, n // 2 + 1)[1:]])
        h = drive_uni(PhysicalJ2(K, G, SY, HH), path)
        sig = np.concatenate([[0.0], h.stress[:, 0]])
        eps = np.concatenate([[0.0], path])
        W = float(np.sum(0.5 * (sig[1:] + sig[:-1]) * np.diff(eps)))
        # total energy after the cycle: stored energy of the last state (E at the end) + accumulated dissipation
        errs.append(abs(W - h.energy[-1] - h.dissipation.sum()) / abs(W))
    assert errs[2] <= 1e-3, errs
    assert errs[0] >= 1.5 * errs[1] and errs[1] >= 1.5 * errs[2], errs


def test_R7d_evaluate_packing_plane_strain_and_physical_flag():
    K, G = KG()
    for name, make in MODEL_FACTORIES.items():
        m = make()
        assert m.physical is True and m.kinematics == "small" and m.conservative is False
        step = TimeStep(0.0, 1e-3) if name == "perzyna" else None
        req = MaterialRequest(step=step)
        v = np.array(STEP1)
        r = m.evaluate(Kinematics(eps3(v), "small"), None, m.initial_state(1), req)
        u = m.update(eps3(v), m.initial_state(1), req)
        assert np.array_equal(r.stress[0], sig_voigt(u))
        v2 = np.array([0.02, -0.008, 0.0, 0.0, 0.0, 0.012])
        H2 = T.voigt_to_strain(np.array([v2[0], v2[1], v2[5]]))[None]
        r2 = m.evaluate(Kinematics(H2, "small", plane="strain"), None, m.initial_state(1), req)
        u2 = m.update(eps3(v2), m.initial_state(1), req)
        assert np.array_equal(r2.stress[0], sig_voigt(u2)[[0, 1, 5]])
        with pytest.raises(ValueError):
            m.evaluate(Kinematics(H2, "small", plane="stress"), None, m.initial_state(1), req)
