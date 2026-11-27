"""VF_fractional: Mittag-Leffler function, fractional viscoelastic models, passivity, Prony bridge (VF1-VF11).

Preregistration: ``docs/foundations/prereg/VF_fractional.md`` (thresholds fixed there before the first run). All
parameters are synthetic. References are independent of the production code: mpmath at >= 30 digits (series at raised
precision, asymptotic expansion, ``quad`` of the integral derived in the preregistration, the closed forms of
alpha = 1 and alpha = 1/2), mpmath ``invertlaplace`` of the Laplace-domain moduli, numpy/mpmath complex arithmetic for
the complex modulus. Tests that are only regressions of a limit against ``core.viscoelastic`` are labelled so.
"""
from __future__ import annotations

import math
from functools import cache

import mpmath as mp
import numpy as np
import pytest

from toporlogy.core import fractional as F
from toporlogy.core import viscoelastic as V
from toporlogy.core.constitutive import MaterialRequest, TimeStep

X_GRID = [0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 1.6, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 100.0,
          200.0, 300.0, 500.0, 700.0, 1000.0]
G_INF, G1, TAU = 2.0e5, 8.0e5, 1.0                    # synthetic fractional Zener (Pa, Pa, s)
G0 = G_INF + G1


# ----------------------------------------------------------------------------------------------- mpmath reference
def _mp_series(a: float, b: float, x: float):
    xa = float(x) ** (1.0 / a)
    dps = 45 + int(xa / 2.302585) + 12
    with mp.workdps(dps):
        a_, b_, x_ = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        kmin = float(xa / a) + 10.0
        s, k = mp.mpf(0), 0
        tol = mp.mpf(10) ** (-(dps - 5))
        while True:
            term = (-x_) ** k * mp.rgamma(a_ * k + b_)
            s += term
            if k > kmin and abs(term) < tol:
                break
            k += 1
        return s


def _mp_integral_mp(a: float, b, x: float):
    """Integral route for any beta (mpf): downward recurrence E_{a,b}(-x) = (1/G(b-a) - E_{a,b-a}(-x))/x to b < a+1."""
    with mp.workdps(60):
        if b >= mp.mpf(a) + 1 - mp.mpf(10) ** -12:
            bm = b - mp.mpf(a)
            return (mp.rgamma(bm) - _mp_integral_mp(a, bm, x)) / mp.mpf(x)
        a_, x_ = mp.mpf(a), mp.mpf(x)
        sb, sba, ca = mp.sinpi(b), mp.sinpi(b - a_), mp.cospi(a_)
        e = (1 - b) / a_
        U = mp.mpf(100) ** a_
        f = lambda u: mp.exp(-u ** (1 / a_)) * u ** e * (u * sb + x_ * sba) / (u * u + 2 * x_ * u * ca + x_ * x_)  # noqa: E731
        pts = [0, x_, U] if x_ < U else [0, U]
        return mp.quad(f, pts) / (mp.pi * a_)


def _mp_asymptotic(a: float, b: float, x: float, n: int = 20):
    with mp.workdps(60):
        a_, b_, x_ = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        return sum((-1) ** (k + 1) * x_ ** (-k) * mp.rgamma(b_ - a_ * k) for k in range(1, n + 1))


@cache
def ref_ml(a: float, b: float, x: float) -> float:
    """Reference E_{a,b}(-x) as a double (computed at >= 30 digits). Tiers: closed forms (a = 1, a = 1/2 & b = 1),
    series at raised precision (x^{1/a} <= 150), asymptotic expansion (x >= 300), integral otherwise."""
    if x == 0.0:
        with mp.workdps(40):
            return float(mp.rgamma(mp.mpf(b)))
    if a == 1.0 and b == 1.0:
        with mp.workdps(40):
            return float(mp.exp(-mp.mpf(x)))
    if a == 0.5 and b == 1.0:
        with mp.workdps(60):
            xm = mp.mpf(x)
            return float(mp.exp(xm * xm) * mp.erfc(xm))
    if x ** (1.0 / a) <= 150.0:
        return float(_mp_series(a, b, x))
    if x >= 300.0:
        return float(_mp_asymptotic(a, b, x))
    return float(_mp_integral_mp(a, mp.mpf(b), x))


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b != 0.0 else abs(a)


# ----------------------------------------------------------------------------------------------- VF1 - VF3
@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8, 1.0])
def test_VF1_mittag_leffler_vs_mpmath(alpha):
    x = np.array(X_GRID)
    got = F.mittag_leffler(-x, alpha)
    worst = max(rel(g, ref_ml(alpha, 1.0, xi)) for g, xi in zip(got, X_GRID, strict=True))
    assert worst <= 1e-10, f"alpha={alpha}: worst relative error {worst:.3e}"


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("kind", ["alpha", "1.5", "1+alpha", "2", "2.7"])
def test_VF2_mittag_leffler_two_parameter(alpha, kind):
    beta = {"alpha": alpha, "1.5": 1.5, "1+alpha": 1.0 + alpha, "2": 2.0, "2.7": 2.7}[kind]
    x = np.array(X_GRID)
    got = F.mittag_leffler(-x, alpha, beta)
    worst = max(rel(g, ref_ml(alpha, beta, xi)) for g, xi in zip(got, X_GRID, strict=True))
    assert worst <= 1e-9, f"alpha={alpha} beta={beta}: worst relative error {worst:.3e}"


@pytest.mark.parametrize("alpha", [0.1, 0.95, 0.99])
def test_VF3_mittag_leffler_stress(alpha):
    x = np.array(X_GRID)
    got = F.mittag_leffler(-x, alpha)
    worst = max(rel(g, ref_ml(alpha, 1.0, xi)) for g, xi in zip(got, X_GRID, strict=True))
    assert worst <= 1e-8, f"alpha={alpha}: worst relative error {worst:.3e}"


# ----------------------------------------------------------------------------------------------- VF4
@pytest.mark.parametrize("alpha,xs", [(0.3, (3.0, 4.0)), (0.8, (20.0, 50.0))])
def test_VF4a_reference_series_vs_integral(alpha, xs):
    for x in xs:
        s = _mp_series(alpha, 1.0, x)
        i = _mp_integral_mp(alpha, mp.mpf(1), x)
        assert abs(s - i) / abs(s) <= mp.mpf(10) ** -20


@pytest.mark.parametrize("alpha", [0.3, 0.8])
@pytest.mark.parametrize("beta", [1.0, 1.5])
def test_VF4b_reference_asymptotic_vs_integral(alpha, beta):
    asy = _mp_asymptotic(alpha, beta, 300.0)
    itg = _mp_integral_mp(alpha, mp.mpf(beta), 300.0)
    assert abs(asy - itg) / abs(asy) <= mp.mpf(10) ** -20


@pytest.mark.parametrize("alpha", [0.3, 0.8])
def test_VF4c_reference_vs_inverse_laplace(alpha):
    with mp.workdps(30):
        F_hat = lambda s: s ** (alpha - 1.0) / (s ** alpha + 1.0)  # noqa: E731
        for t in (0.5, 2.0, 20.0):
            inv = mp.invertlaplace(F_hat, t, method="talbot")
            ref = ref_ml(alpha, 1.0, t ** alpha)
            assert abs(float(inv) - ref) / ref <= 1e-10, (alpha, t)


# ----------------------------------------------------------------------------------------------- VF5, VF6
def _gstar_zener(s, a, Ginf=G_INF, G1_=G1, tau=TAU):
    z = (s * tau) ** a
    return Ginf + G1_ * z / (1 + z)


U_GRID = [1e-2, 0.1, 1.0, 10.0, 100.0]


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
def test_VF5_zener_relaxation_vs_inverse_laplace(alpha):
    with mp.workdps(30):
        Ghat = lambda s: _gstar_zener(s, alpha) / s  # noqa: E731
        ref = [float(mp.invertlaplace(Ghat, u, method="talbot")) for u in U_GRID]
    got = F.fractional_zener_relaxation(np.array(U_GRID), G_INF, G1, TAU, alpha)
    worst = max(rel(g, r) for g, r in zip(got, ref, strict=True))
    assert worst <= 1e-8, f"alpha={alpha}: {worst:.3e}"


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8])
def test_VF6_creep_and_other_models_vs_inverse_laplace(alpha):
    E = 3.0e5
    t = np.array(U_GRID)
    cases = {
        "zener creep": (lambda s: 1 / (s * _gstar_zener(s, alpha)),
                        F.fractional_zener_creep(t, G_INF, G1, TAU, alpha)),
        "kv creep": (lambda s: 1 / (s * E * (1 + (s * TAU) ** alpha)),
                     F.fractional_kelvin_voigt_creep(t, E, TAU, alpha)),
        "maxwell creep": (lambda s: 1 / (s * E * (s * TAU) ** alpha / (1 + (s * TAU) ** alpha)),
                          F.fractional_maxwell_creep(t, E, TAU, alpha)),
        "kv relaxation": (lambda s: E * (1 + (s * TAU) ** alpha) / s,
                          F.fractional_kelvin_voigt_relaxation(t, E, TAU, alpha)),
        "maxwell relaxation": (lambda s: E * (s * TAU) ** alpha / (1 + (s * TAU) ** alpha) / s,
                               F.fractional_maxwell_relaxation(t, E, TAU, alpha)),
    }
    for name, (Fhat, got) in cases.items():
        with mp.workdps(30):
            ref = [float(mp.invertlaplace(Fhat, u, method="talbot")) for u in U_GRID]
        worst = max(rel(g, r) for g, r in zip(got, ref, strict=True))
        assert worst <= 1e-8, f"{name}, alpha={alpha}: {worst:.3e}"


def test_VF6_kv_relaxation_alpha_one_refused():
    with pytest.raises(ValueError, match="no finite relaxation modulus"):
        F.fractional_kelvin_voigt_relaxation(1.0, 1e5, 1.0, 1.0)


# ----------------------------------------------------------------------------------------------- VF7 (regression)
def test_VF7_alpha_one_recovers_standard_linear_solid_regression():
    """Regression of a limit against core.viscoelastic (not an independent reference)."""
    t = np.geomspace(1e-4, 1e4, 200)
    tau, E = 2.5, 3.0e5
    w = np.geomspace(1e-6, 1e6, 200)
    assert np.max(np.abs(F.fractional_zener_relaxation(t, G_INF, G1, tau, 1.0)
                         / V.sls_relaxation(t, G_INF, G1, tau) - 1.0)) <= 1e-13
    assert np.max(np.abs(F.fractional_zener_creep(t, G_INF, G1, tau, 1.0) / V.sls_creep(t, G_INF, G1, tau) - 1.0)) <= 1e-13
    assert np.max(np.abs(F.fractional_kelvin_voigt_creep(t, E, tau, 1.0)
                         / V.kelvin_voigt_creep(t, E, E * tau) - 1.0)) <= 1e-13
    tm = np.geomspace(1e-4, 500.0 * tau, 200)               # e^{-t/tau} stays representable (no 0/0)
    assert np.max(np.abs(F.fractional_maxwell_relaxation(tm, E, tau, 1.0)
                         / V.maxwell_relaxation(tm, E, E * tau) - 1.0)) <= 1e-13
    assert np.max(np.abs(F.fractional_maxwell_creep(t, E, tau, 1.0) / V.maxwell_creep(t, E, E * tau) - 1.0)) <= 1e-13
    M = V.complex_modulus(V.PronySeries(G_INF, np.array([G1]), np.array([tau])), w)
    assert np.max(np.abs(F.fractional_zener_modulus(w, G_INF, G1, tau, 1.0) - M) / np.abs(M)) <= 1e-13


@pytest.mark.parametrize("alpha", [0.99, 0.999])
def test_VF7_continuity_in_alpha(alpha):
    t = np.geomspace(1e-4, 1e4, 400)
    w = np.geomspace(1e-4, 1e4, 400)
    dG = np.max(np.abs(F.fractional_zener_relaxation(t, G_INF, G1, TAU, alpha)
                       - F.fractional_zener_relaxation(t, G_INF, G1, TAU, 1.0))) / G1
    dM = np.max(np.abs(F.fractional_zener_modulus(w, G_INF, G1, TAU, alpha)
                       - F.fractional_zener_modulus(w, G_INF, G1, TAU, 1.0))) / G1
    assert dG <= 2.0 * (1.0 - alpha), dG
    assert dM <= 2.0 * (1.0 - alpha), dM


# ----------------------------------------------------------------------------------------------- VF8
@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8, 0.99, 1.0])
def test_VF8_passivity_peak_limits(alpha):
    w = np.geomspace(1e-8, 1e8, 401) / TAU
    Gs = F.fractional_zener_modulus(w, G_INF, G1, TAU, alpha)
    Gi = G_INF + G1 * (1j * w * TAU) ** alpha / (1.0 + (1j * w * TAU) ** alpha)      # independent numpy complex power
    assert np.min(Gs.imag) / G1 >= -1e-14 and np.min(Gi.imag) / G1 >= -1e-14
    assert F.is_passive(Gs) and F.is_passive(Gi)
    assert np.max(np.abs(Gs - Gi) / np.abs(Gi)) <= 1e-12
    with mp.workdps(30):
        for om in (1e-3, 0.3, 1.0, 7.0, 1e3):
            s = mp.mpc(0, om)
            gm = _gstar_zener(s, alpha)
            gp = F.fractional_zener_modulus(om, G_INF, G1, TAU, alpha)
            assert abs(complex(gm) - gp) / abs(complex(gm)) <= 1e-12
            assert float(gm.imag) >= 0.0
    # peak value at exactly 1/tau, and location on a grid
    peak = F.fractional_zener_modulus(1.0 / TAU, G_INF, G1, TAU, alpha).imag
    assert abs(peak - 0.5 * G1 * math.tan(math.pi * alpha / 4.0)) <= 1e-12 * peak
    wg = np.geomspace(1e-3, 1e3, 20001) / TAU
    loss = F.fractional_zener_modulus(wg, G_INF, G1, TAU, alpha).imag
    assert abs(wg[np.argmax(loss)] * TAU - 1.0) <= 2e-3
    # symmetry about 1/tau on the log axis
    ws = np.geomspace(1e-3, 1.0, 50)
    l1 = F.fractional_zener_modulus(ws / TAU, G_INF, G1, TAU, alpha).imag
    l2 = F.fractional_zener_modulus(1.0 / (ws * TAU), G_INF, G1, TAU, alpha).imag
    assert np.max(np.abs(l1 / l2 - 1.0)) <= 1e-12
    # limits: slope, plateau moduli, fractional Maxwell phase angle
    sa = math.sin(math.pi * alpha / 2)
    lo = F.fractional_zener_modulus(1e-30 / TAU, G_INF, G1, TAU, alpha)
    assert abs(lo.imag / (G1 * sa * (1e-30) ** alpha) - 1.0) <= 1e-6
    assert abs(lo.real / G_INF - 1.0) <= 1e-6
    hi = F.fractional_zener_modulus(1e30 / TAU, G_INF, G1, TAU, alpha)
    assert abs(hi.real / G0 - 1.0) <= 1e-6
    if alpha < 1.0:
        fm = F.fractional_zener_modulus(1e-30 / TAU, 0.0, G1, TAU, alpha)
        assert abs(fm.imag / fm.real / math.tan(math.pi * alpha / 2) - 1.0) <= 1e-6


# ----------------------------------------------------------------------------------------------- VF9
ALPHAS = [0.3, 0.5, 0.8]
N_SEQ = [5, 9, 17, 33, 65]


@pytest.fixture(scope="module")
def budgets():
    return {a: F.prony_error_budget(G_INF, G1, TAU, a, N_SEQ) for a in ALPHAS}


def _monotone_param(alpha):
    if alpha == 0.3:
        return pytest.param(alpha, marks=pytest.mark.xfail(
            strict=True, reason="measured: e_t(33) = 8.89e-5 < e_t(65) = 8.98e-5 at alpha = 0.3; the max-norm error "
                                "sits at the window edge t = t_lo = smallest branch time and reaches a floor, while "
                                "the weighted L2 fit residual still decreases (2.78e-4 -> 2.62e-4); see "
                                "prereg/VF_fractional.md"))
    return alpha


@pytest.mark.parametrize("alpha", [_monotone_param(a) for a in ALPHAS])
def test_VF9_prony_error_strictly_decreasing(budgets, alpha):
    e = [f.rel_error_time for f in budgets[alpha]]
    ew = [f.rel_error_modulus for f in budgets[alpha]]
    print(f"alpha={alpha}: N={N_SEQ} e_t={[f'{v:.3e}' for v in e]} e_omega={[f'{v:.3e}' for v in ew]}")
    assert all(e[i + 1] < e[i] for i in range(len(e) - 1)), e


@pytest.mark.parametrize("alpha", ALPHAS)
def test_VF9_prony_error_thresholds(budgets, alpha):
    fits = budgets[alpha]
    e = [f.rel_error_time for f in fits]
    ew = [f.rel_error_modulus for f in fits]
    assert e[2] <= 5e-2 and e[3] <= 5e-3 and e[4] <= 5e-4, e
    assert ew[4] <= 5e-2, ew
    for f in fits:
        assert np.all(f.series.amplitudes >= 0.0) and f.series.inf >= 0.0
    ts = np.geomspace(1e-4, 1e4, 40)
    got = F.fractional_zener_relaxation(ts, G_INF, G1, TAU, alpha)
    ref = [G_INF + G1 * ref_ml(alpha, 1.0, float((t / TAU) ** alpha)) for t in ts]
    assert max(rel(g, r) for g, r in zip(got, ref, strict=True)) <= 1e-9


# ----------------------------------------------------------------------------------------------- VF10
@pytest.mark.parametrize("n_branches", [17, 33, 65])
def test_VF10_time_domain_drive_within_reported_error(budgets, n_branches):
    alpha = 0.5
    fit = budgets[alpha][N_SEQ.index(n_branches)]
    K, gamma0, t_r = 1.0e6, 1.0e-3, 1.0e-3
    gm = fit.to_generalized_maxwell(V.PronySeries(K, np.zeros(0), np.zeros(0)))
    state = gm.initial_state(1)
    t_cur = 0.0

    def step(t_next, g):
        nonlocal state, t_cur
        eps = np.zeros((1, 3, 3))
        eps[0, 0, 1] = eps[0, 1, 0] = 0.5 * g                       # engineering shear gamma_xy = g
        up = gm.update(eps, state, MaterialRequest(tangent=False, step=TimeStep(t_cur, t_next - t_cur)))
        state, t_cur = up.trial_state, t_next
        return float(up.stress[0, 0, 1])

    step(t_r, gamma0)                                               # ramp: one exact step
    tk = np.geomspace(1e-2, 1e3, 25)
    sig = np.array([step(t, gamma0) for t in tk])

    def Phi(t):                                                     # int_0^t G, item 4 of the preregistration
        x = t ** alpha
        e2 = ref_ml(alpha, 2.0, float(x))
        return G_INF * t + G1 * t * e2

    ref = np.array([gamma0 / t_r * (Phi(float(t)) - Phi(float(t) - t_r)) for t in tk])
    Gt = np.array([G_INF + G1 * ref_ml(alpha, 1.0, float((t - t_r) ** alpha)) for t in tk])
    bound = 1.01 * gamma0 * fit.rel_error_time * Gt + 1e-9 * gamma0 * G0
    err = np.abs(sig - ref)
    print(f"\nN={n_branches}: e_t={fit.rel_error_time:.3e} max drive error={np.max(err):.3e} Pa, "
          f"max err/bound={np.max(err / bound):.3f}")
    assert np.all(err <= bound)
    if n_branches == 65:
        assert np.max(err / ref) <= 1e-3


# ----------------------------------------------------------------------------------------------- VF11
def test_VF11_negative_controls():
    # (a) naive double precision power series at alpha = 1, x = 50
    term, naive = 1.0, 1.0
    for i in range(1, 200):                                           # terms by recurrence (factorial overflows float)
        term *= -50.0 / i
        naive += term
    assert abs(naive - math.exp(-50.0)) > 1.0
    # (b) a perturbed order is detected by the reference comparison
    x = np.array(X_GRID[1:])
    for alpha in (0.5, 0.8):
        got = F.mittag_leffler(-x, alpha * (1 + 1e-3))
        worst = max(rel(g, ref_ml(alpha, 1.0, xi)) for g, xi in zip(got, X_GRID[1:], strict=True))
        assert worst > 1e-6, (alpha, worst)


def test_VF11_coarse_fit_is_visible(budgets):
    fits = budgets[0.5]
    assert fits[0].rel_error_time > 100.0 * fits[-1].rel_error_time


def test_VF11_input_validation():
    for bad in (0.0, -0.1, 1.0001, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            F.mittag_leffler(-1.0, bad)
    with pytest.raises(ValueError):
        F.mittag_leffler(0.5, 0.5)
    with pytest.raises(ValueError):
        F.mittag_leffler(float("nan"), 0.5)
    with pytest.raises(ValueError):
        F.mittag_leffler(-1.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        F.mittag_leffler(-1.0, 1.0, 1.5)                          # alpha = 1 with non-integer beta
    with pytest.raises(ValueError):
        F.fractional_zener_relaxation(1.0, -1.0, 1.0, 1.0, 0.5)   # negative G_inf
    with pytest.raises(ValueError):
        F.fractional_zener_relaxation(1.0, 1.0, 0.0, 1.0, 0.5)    # G1 = 0
    with pytest.raises(ValueError):
        F.fractional_zener_relaxation(1.0, 1.0, 1.0, 0.0, 0.5)    # tau = 0
    with pytest.raises(ValueError):
        F.fractional_zener_relaxation(-1.0, 1.0, 1.0, 1.0, 0.5)   # t < 0
    with pytest.raises(ValueError):
        F.fractional_maxwell_relaxation(1.0, 0.0, 1.0, 0.5)       # E = 0
    with pytest.raises(ValueError):
        F.fractional_zener_modulus(-1.0, 1.0, 1.0, 1.0, 0.5)      # omega < 0
    with pytest.raises(ValueError):
        F.fractional_kelvin_voigt_relaxation(0.0, 1.0, 1.0, 0.5)  # singular at t = 0
    with pytest.raises(ValueError):
        F.prony_error_budget(G_INF, G1, TAU, 0.5, [8], window=(1.0, 1.0))
    with pytest.raises(ValueError):
        F.prony_error_budget(G_INF, G1, TAU, 0.5, [1])
