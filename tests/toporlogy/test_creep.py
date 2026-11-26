"""Irreversible creep (``toporlogy.core.creep``): Norton, Norton-Bailey, Arrhenius temperature, adaptive long holds.

Thresholds and references are the preregistered ones of ``docs/foundations/prereg/C_creep.md`` (committed before the
first run). References are closed forms written out in this file, ``scipy.integrate.quad`` and ``solve_ivp``, never
a call of the production kernel, except where a test says "regression". All constants are synthetic (not material
data).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from toporlogy.core import creep as CR
from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (MaterialRequest, StateStore, TimeStep, drive_material_point)
from toporlogy.core.creep import (ArrheniusFactor, LoadSegment, NortonBaileyCreep, NortonCreep, integrate_creep)

K0, G0 = 1.7e10, 8e9                      # synthetic isotropic moduli [Pa]
A0, SREF, N0 = 1e-6, 1e7, 3.0             # synthetic Norton constants: [1/s], 10 MPa, exponent
R = 8.314462618                           # J/(mol K), written out here, not imported
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
ALL_STRESS = [False] * 6
ALL_STRAIN = [True] * 6


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def vm(sig6: np.ndarray) -> float:
    """von Mises stress of a Voigt stress (physical shear)."""
    s = np.asarray(sig6, float)
    return math.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
                     + 3.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))


def norton(n: float = N0, **kw) -> NortonCreep:
    return NortonCreep(K0, G0, kw.pop("A", A0), SREF, n, **kw)


def stress_path(model, sigma6, times, temperatures=None, temperature0=None):
    """All six components stress controlled (lateral stresses free = 0), constant target ``sigma6``."""
    k = len(times)
    return drive_material_point(model, np.zeros((k, 6)), np.tile(np.asarray(sigma6, float), (k, 1)), ALL_STRESS,
                                tol=1e-6, times=times, temperatures=temperatures, temperature0=temperature0)


def relaxation(model, q0: float, times, dt_first_zero: bool = True):
    """Held deviatoric strain e = diag(e0, -e0/2, -e0/2), q0 = 3 G e0, loaded at t = 0 (a dt = 0 step first)."""
    e0 = q0 / (3.0 * G0)
    eps = np.array([e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0])
    tt = np.concatenate([[0.0], np.asarray(times, float)]) if dt_first_zero else np.asarray(times, float)
    return drive_material_point(model, np.tile(eps, (len(tt), 1)), np.zeros((len(tt), 6)), ALL_STRAIN, tol=1e-6,
                                times=tt, t0=0.0)


def q_closed(q0: float, t: float, n: float, A: float = A0) -> float:
    """q(t) of qdot = -3G A (q/sigma_ref)^n (WP13)."""
    k = 3.0 * G0 * A / SREF ** n
    if n == 1.0:
        return q0 * math.exp(-k * t)
    return (q0 ** (1.0 - n) + (n - 1.0) * k * t) ** (1.0 / (1.0 - n))


# ================================================================================================ CR1 Arrhenius
def test_cr1_arrhenius_factor_derivative_overflow_and_validation() -> None:
    Q, Tr = 5e4, 300.0
    f = ArrheniusFactor(Q, Tr)
    assert float(f.factor(Tr)) == 1.0
    for Tt in (250.0, 300.0, 360.0, 500.0):
        ref = math.exp(-Q / R * (1.0 / Tt - 1.0 / Tr))
        assert rel(float(f.factor(Tt)), ref) <= 1e-13
        h = 1e-3
        fd = (math.exp(-Q / R * (1.0 / (Tt + h) - 1.0 / Tr)) - math.exp(-Q / R * (1.0 / (Tt - h) - 1.0 / Tr))) / (2 * h)
        assert rel(float(f.dfactor(Tt)), fd) <= 1e-8
    big = ArrheniusFactor(1e7, Tr)
    assert math.isfinite(float(big.log_factor(1000.0)))
    with pytest.raises(ValueError):
        big.factor(1000.0)
    for bad in ((-1.0, 300.0), (1e4, 0.0), (1e4, -3.0), (float("nan"), 300.0), (1e4, float("inf"))):
        with pytest.raises(ValueError):
            ArrheniusFactor(*bad)
    for Tbad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError):
            f.factor(Tbad)


# ================================================================================================ C2 Norton
def _creep_state(h):
    return h.states[-1]["creep_strain"][0], float(h.states[-1]["p_cr"][0])


def test_c2a_uniaxial_tension_rate_and_strain() -> None:
    h = stress_path(norton(), [2e7, 0, 0, 0, 0, 0], [100.0])
    ecr, p = _creep_state(h)
    rate = A0 * (2e7 / SREF) ** N0
    assert rel(rate, 8e-6) <= 1e-12                                   # the acceptance number itself
    assert rel(ecr[0, 0], 8e-4) <= 1e-9
    assert rel(ecr[1, 1], -4e-4) <= 1e-9 and rel(ecr[2, 2], -4e-4) <= 1e-9
    assert rel(p, 8e-4) <= 1e-9
    assert np.abs(ecr - np.diag(np.diag(ecr))).max() <= 1e-15
    assert rel(float(h.dissipation[-1]), 2e7 * 8e-4) <= 1e-9
    assert np.abs(h.stress[-1] - [2e7, 0, 0, 0, 0, 0]).max() <= 1e-5


def test_c2b_compression_sign_and_positive_accumulated() -> None:
    h = stress_path(norton(), [-2e7, 0, 0, 0, 0, 0], [100.0])
    ecr, p = _creep_state(h)
    assert rel(ecr[0, 0], -8e-4) <= 1e-9
    assert rel(ecr[1, 1], 4e-4) <= 1e-9 and rel(ecr[2, 2], 4e-4) <= 1e-9
    assert p > 0.0 and rel(p, 8e-4) <= 1e-9
    assert float(h.dissipation[-1]) >= 0.0


def test_c2c_hydrostatic_stress_gives_no_creep() -> None:
    h = stress_path(norton(), [1e8, 1e8, 1e8, 0, 0, 0], [100.0])
    ecr, p = _creep_state(h)
    assert np.abs(ecr).max() <= 1e-30 and p <= 1e-30


def test_c2d_pure_shear_stress() -> None:
    tau = 1e7
    h = stress_path(norton(), [0, 0, 0, 0, 0, tau], [100.0])
    ecr, p = _creep_state(h)
    q = math.sqrt(3.0) * tau
    p_ref = A0 * (q / SREF) ** N0 * 100.0
    assert rel(p, p_ref) <= 1e-9
    assert rel(ecr[0, 1], p_ref * math.sqrt(3.0) / 2.0) <= 1e-9          # N_xy = 3 tau / (2 q)
    assert np.abs(np.diag(ecr)).max() <= 1e-15


def test_c2e_arrhenius_isothermal_at_tref_and_above() -> None:
    Q, Tr = 5e4, 300.0
    m = norton(arrhenius=ArrheniusFactor(Q, Tr))
    for Tt in (300.0, 360.0):
        h = stress_path(m, [2e7, 0, 0, 0, 0, 0], [100.0], temperatures=[Tt], temperature0=Tt)
        _, p = _creep_state(h)
        ref = A0 * math.exp(-Q / R * (1.0 / Tt - 1.0 / Tr)) * 8.0 * 100.0
        assert rel(p, ref) <= 1e-9


def test_c2f_no_evolution_for_dt_zero_and_zero_deviatoric_stress() -> None:
    m = norton()
    ecr = np.array([[1e-4, 3e-5, 1e-5], [3e-5, -4e-5, 2e-5], [1e-5, 2e-5, -6e-5]])[None]
    committed = {"creep_strain": ecr, "p_cr": np.array([3e-4])}
    eps = np.array([[[3e-3, 1e-4, 0.0], [1e-4, -1e-3, 2e-4], [0.0, 2e-4, 5e-4]]])
    u = m.update(eps, committed, MaterialRequest(step=TimeStep(0.0, 0.0)))
    assert np.array_equal(u.trial_state["creep_strain"], ecr) and np.array_equal(u.trial_state["p_cr"], [3e-4])
    dev = eps - (np.trace(eps[0]) / 3.0) * np.eye(3)
    sig_el = K0 * np.trace(eps[0]) * np.eye(3) + 2.0 * G0 * (dev[0] - ecr[0])
    assert np.abs(u.stress[0] - sig_el).max() <= 1e-13 * np.abs(sig_el).max()
    assert float(u.dissipation[0]) == 0.0
    hyd = np.full((1, 3, 3), 0.0)
    hyd[0] = 0.25 * np.eye(3)                                         # dyadic: the deviator is exactly zero
    z = {"creep_strain": np.zeros((1, 3, 3)), "p_cr": np.zeros(1)}
    u = m.update(hyd, z, MaterialRequest(step=TimeStep(0.0, 1e6)))
    assert np.array_equal(u.trial_state["creep_strain"], z["creep_strain"]) and u.trial_state["p_cr"][0] == 0.0
    assert np.array_equal(u.tangent[0], m.Ce6)


def test_c2g_invalid_parameters_and_missing_step_or_temperature() -> None:
    good = dict(K=K0, G=G0, A=A0, sigma_ref=SREF, n=3.0)
    for key, bad in (("K", 0.0), ("G", -1.0), ("A", -1e-6), ("n", 0.0), ("sigma_ref", 0.0), ("n", float("nan")),
                     ("G", float("inf"))):
        with pytest.raises(ValueError):
            NortonCreep(**{**good, key: bad})
    with pytest.raises(ValueError):
        NortonCreep(**good, arrhenius="Arrhenius")                           # type: ignore[arg-type]
    nb = dict(K=K0, G=G0, B=1e-4, sigma_ref=SREF, n=3.0, m=0.5, t_ref=10.0)
    for key, bad in (("B", -1.0), ("m", 0.0), ("t_ref", 0.0), ("n", -1.0), ("sigma_ref", -1.0)):
        with pytest.raises(ValueError):
            NortonBaileyCreep(**{**nb, key: bad})
    with pytest.raises(ValueError):
        NortonBaileyCreep(**nb, hardening="kinematic")
    eps = np.zeros((1, 3, 3))
    st = norton().initial_state(1)
    with pytest.raises(ValueError):
        norton().update(eps, st, MaterialRequest())                                   # no step
    with pytest.raises(ValueError):
        norton(arrhenius=ArrheniusFactor(5e4, 300.0)).update(eps, st, MaterialRequest(step=TimeStep(0.0, 1.0)))
    with pytest.raises(ValueError):
        TimeStep(0.0, -1.0)


# ================================================================================================ C3 relaxation
Q0 = 2e7


def test_c3a_relaxation_n1_exponential() -> None:
    h = relaxation(norton(n=1.0), Q0, 0.025 * np.arange(1, 1601))
    assert rel(vm(h.stress[-1]), q_closed(Q0, 40.0, 1.0)) <= 2e-3


def test_c3b_relaxation_n3_closed_form() -> None:
    h = relaxation(norton(n=3.0), Q0, 0.025 * np.arange(1, 1601))
    q_ref = q_closed(Q0, 40.0, 3.0)
    assert rel(vm(h.stress[-1]), q_ref) <= 2e-3
    assert q_ref < 0.95 * Q0                                                      # the relaxation is not trivial


@pytest.mark.parametrize("n", [1.0, 3.0])
def test_c3c_backward_euler_first_order(n: float) -> None:
    errs = []
    dts = [4.0, 2.0, 1.0, 0.5, 0.25]
    for dt in dts:
        h = relaxation(norton(n=n), Q0, dt * np.arange(1, int(round(40.0 / dt)) + 1))
        errs.append(rel(vm(h.stress[-1]), q_closed(Q0, 40.0, n)))
    assert all(a > b for a, b in zip(errs, errs[1:]))
    orders = [math.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(0.9 <= o <= 1.1 for o in orders[-3:]), (errs, orders)


def _tangent_case(model, committed, eps6, step):
    req = MaterialRequest(step=step)
    ev = T.voigt_to_strain(np.asarray(eps6, float))[None]
    u = model.update(ev, committed, req)
    return u, req


def _fd_tangent(model, committed, eps6, step, h):
    C = np.zeros((6, 6))
    req = MaterialRequest(step=step, tangent=False)
    for j in range(6):
        out = []
        for sgn in (+1.0, -1.0):
            e = np.array(eps6, float)
            e[j] += sgn * h
            out.append(model.update(T.voigt_to_strain(e)[None], committed, req).stress[0][_VA, _VB])
        C[:, j] = (out[0] - out[1]) / (2.0 * h)
    return C


ECR0 = 1e-4 * np.array([[1.0, 0.3, 0.1], [0.3, -0.4, 0.2], [0.1, 0.2, -0.6]])
EPS6 = np.array([2.6e-3, -1.0e-3, 5e-4, 4e-4, -3e-4, 6e-4])


def _norton_state():
    return {"creep_strain": ECR0[None].copy(), "p_cr": np.array([2e-4])}


def _fd_error(model, committed, step) -> tuple[float, np.ndarray]:
    u, _ = _tangent_case(model, committed, EPS6, step)
    C = u.tangent[0]
    errs = []
    for h in np.array([1e-4, 1e-5, 1e-6, 1e-7]) * 1e-3:
        Cfd = _fd_tangent(model, committed, EPS6, step, h)
        errs.append(np.abs(Cfd - C).max() / np.abs(C).max())
    return min(errs), C


def test_c3d_tangent_against_central_fd_and_c3e_frozen_direction_is_detected() -> None:
    m = norton(n=3.0)
    step = TimeStep(0.0, 30.0)
    committed = _norton_state()
    u, _ = _tangent_case(m, committed, EPS6, step)
    theta = float(u.diagnostics["q"][0]) / float(vm_tensor_trial(m, committed, EPS6))
    assert 0.3 <= theta <= 0.9, theta                           # set-up: active branch with a changing direction
    err, C = _fd_error(m, committed, step)
    assert err <= 1e-6, err
    # C3e: the tangent with the direction of N frozen, C_e - 4 G^2 dp' N x N, written here from h' of the law
    eps_t = T.voigt_to_strain(EPS6)
    dev = eps_t - np.trace(eps_t) / 3.0 * np.eye(3)
    s_tr = 2.0 * G0 * (dev - ECR0)
    q_tr = math.sqrt(1.5 * np.sum(s_tr * s_tr))
    N = 1.5 * s_tr / q_tr
    nv = N[_VA, _VB]
    q = float(u.diagnostics["q"][0])
    dp = float(u.diagnostics["delta_p"][0])
    hp = step.dt * A0 * N0 * (q / SREF) ** (N0 - 1.0) / SREF
    dpp = hp / (1.0 + 3.0 * G0 * hp)
    one6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    idev6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(one6, one6) / 3.0
    Ce = K0 * np.outer(one6, one6) + 2.0 * G0 * idev6
    C_frozen = Ce - 4.0 * G0 ** 2 * dpp * np.outer(nv, nv)
    C_derived = (K0 * np.outer(one6, one6) + 2.0 * G0 * (q / q_tr) * idev6
                 - 4.0 * G0 ** 2 * (dpp - dp / q_tr) * np.outer(nv, nv))
    assert np.abs(C_derived - C).max() / np.abs(C).max() <= 1e-10         # the doc formula, re-derived in the test
    best_fro = min(np.abs(_fd_tangent(m, committed, EPS6, step, h) - C_frozen).max() / np.abs(C_frozen).max()
                   for h in np.array([1e-4, 1e-5, 1e-6, 1e-7]) * 1e-3)
    assert best_fro >= 1e-3, best_fro                                      # the FD check discriminates


def vm_tensor_trial(model, committed, eps6) -> float:
    eps_t = T.voigt_to_strain(np.asarray(eps6, float))
    dev = eps_t - np.trace(eps_t) / 3.0 * np.eye(3)
    s_tr = 2.0 * model.G * (dev - committed["creep_strain"][0])
    return math.sqrt(1.5 * np.sum(s_tr * s_tr))


@pytest.mark.parametrize("n", [0.7, 1.0, 3.0, 8.0])
def test_c3f_local_root_residual_independent_recomputation(n: float) -> None:
    m = norton(n=n)
    rng = np.random.default_rng(12345)
    for _ in range(30):
        a = 2e-3 * rng.standard_normal((3, 3))
        eps = 0.5 * (a + a.T)
        b = 1e-4 * rng.standard_normal((3, 3))
        ecr = 0.5 * (b + b.T)
        ecr -= np.trace(ecr) / 3.0 * np.eye(3)
        p0 = float(np.linalg.norm(ecr))
        dt = float(10.0 ** rng.uniform(-1.0, 2.0))
        u = m.update(eps[None], {"creep_strain": ecr[None], "p_cr": np.array([p0])},
                     MaterialRequest(step=TimeStep(0.0, dt)))
        sig = u.stress[0]
        s = sig - np.trace(sig) / 3.0 * np.eye(3)
        q_new = math.sqrt(1.5 * np.sum(s * s))
        dp = float(u.trial_state["p_cr"][0]) - p0
        dev = eps - np.trace(eps) / 3.0 * np.eye(3)
        s_tr = 2.0 * G0 * (dev - ecr)
        q_tr = math.sqrt(1.5 * np.sum(s_tr * s_tr))
        assert 0.0 <= dp <= q_tr / (3.0 * G0) * (1.0 + 1e-12)
        r = dp - dt * A0 * (q_new / SREF) ** n
        assert abs(r) <= 1e-10 * dp, (n, dt, r, dp)


def _c3g_case(qtr: float):
    m = norton(n=20.0)
    e0 = qtr / (3.0 * G0)
    eps = np.diag([e0, -0.5 * e0, -0.5 * e0])[None]
    u = m.update(eps, m.initial_state(1), MaterialRequest(step=TimeStep(0.0, 1.0)))
    return u, float(u.trial_state["p_cr"][0]), float(u.diagnostics["q"][0])


def test_c3g_high_exponent_and_near_zero_stress_are_safe() -> None:
    cases = {}
    for name, qtr in (("low", 1e-3 * SREF), ("mid", 1.5 * SREF), ("high", 10.0 * SREF)):
        u, dp, q_new = _c3g_case(qtr)
        assert np.all(np.isfinite(u.stress)) and np.all(np.isfinite(u.tangent))
        assert dp >= 0.0 and q_new >= 0.0
        if dp > 0.0:
            assert abs(dp - A0 * (q_new / SREF) ** 20.0) <= 1e-10 * dp
        cases[name] = (q_new, qtr)
    assert cases["low"][0] <= cases["low"][1]


@pytest.mark.xfail(strict=True, reason="preregistered claim wrong: for n = 20, q_tr = 10 sigma_ref, dt = 1 s the "
                   "update does not relax below 1e-3 q_tr; measured q_new = 1.505e7 Pa = 0.15 q_tr (the stress at "
                   "which the creep rate balances the elastic unloading, see the post hoc check)")
def test_c3g_prereg_full_relaxation_claim() -> None:
    _, _, q_new = _c3g_case(10.0 * SREF)
    assert q_new < 1e-3 * 10.0 * SREF


def test_c3g_post_hoc_high_stress_root_against_brentq() -> None:
    """Post hoc (after the claim above failed): q_new of the n = 20, q_tr = 10 sigma_ref update is the root of the
    balance dt A (q/sigma_ref)^n = (q_tr - q)/(3G), found independently by scipy brentq."""
    from scipy.optimize import brentq
    qtr = 10.0 * SREF
    _, _, q_new = _c3g_case(qtr)
    root = brentq(lambda q: 1.0 * A0 * (q / SREF) ** 20.0 - (qtr - q) / (3.0 * G0), 0.0, qtr, xtol=1e-9, rtol=1e-14)
    assert rel(q_new, root) <= 1e-10


def test_c3h_dissipation_nonnegative_and_equals_stress_times_creep_increment() -> None:
    m = norton()
    committed = _norton_state()
    ev = T.voigt_to_strain(EPS6)[None]
    u = m.update(ev, committed, MaterialRequest(step=TimeStep(0.0, 30.0)))
    dec = u.trial_state["creep_strain"][0] - committed["creep_strain"][0]
    work = float(np.sum(u.stress[0] * dec))
    assert float(u.dissipation[0]) >= 0.0 and rel(float(u.dissipation[0]), work) <= 1e-12


# ================================================================================================ C4 temperature
def _f_ref(Tt: float, Q: float, Tr: float = 300.0) -> float:
    return math.exp(-Q / R * (1.0 / Tt - 1.0 / Tr))


@pytest.mark.parametrize("Q", [6e4, 2e5])
@pytest.mark.parametrize("T0,T1", [(300.0, 400.0), (300.0, 600.0), (600.0, 300.0), (350.0, 350.0)])
def test_c4a_step_average_against_quad(Q: float, T0: float, T1: float) -> None:
    f = ArrheniusFactor(Q, 300.0)
    ref, _ = quad(lambda s: _f_ref(T0 + (T1 - T0) * s, Q), 0.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=500)
    assert rel(float(f.step_average(T0, T1)), ref) <= 1e-10


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_c4b_variable_temperature_ramp_against_quad(k: int) -> None:
    Q = 6e4
    m = norton(arrhenius=ArrheniusFactor(Q, 300.0))
    times = np.linspace(0.0, 10.0, k + 1)[1:]
    h = stress_path(m, [2e7, 0, 0, 0, 0, 0], times, temperatures=300.0 + 10.0 * times, temperature0=300.0)
    _, p = _creep_state(h)
    integral, _ = quad(lambda t: _f_ref(300.0 + 10.0 * t, Q), 0.0, 10.0, epsabs=0.0, epsrel=1e-13, limit=500)
    assert rel(p, A0 * 8.0 * integral) <= 1e-9


def test_c4c_endpoint_rule_is_detected_as_wrong() -> None:
    Q = 6e4
    integral, _ = quad(lambda t: _f_ref(300.0 + 10.0 * t, Q), 0.0, 10.0, epsabs=0.0, epsrel=1e-13, limit=500)
    endpoint = 10.0 * _f_ref(400.0, Q)
    assert rel(endpoint, integral) >= 1e-2


def test_c4d_temperature_input_validation() -> None:
    m = norton(arrhenius=ArrheniusFactor(5e4, 300.0))
    with pytest.raises(ValueError):
        m.update(np.zeros((1, 3, 3)), m.initial_state(1), MaterialRequest(step=TimeStep(0.0, 1.0)))
    with pytest.raises(ValueError):
        TimeStep(0.0, 1.0, temperature=-1.0)


# ================================================================================================ C5 Norton-Bailey
BN = dict(K=K0, G=G0, B=1e-4, sigma_ref=SREF, n=3.0, m=0.5, t_ref=10.0)


def test_c5a_primary_creep_exact_increment_from_zero() -> None:
    m = NortonBaileyCreep(**BN)
    c = 1e-4 * 8.0
    grid = [1e-6, 0.37, 5.0, 100.0]
    h = stress_path(m, [2e7, 0, 0, 0, 0, 0], grid)
    for k, t in enumerate(grid):
        assert rel(float(h.states[k]["p_cr"][0]), c * math.sqrt(t / 10.0)) <= 1e-9
        assert rel(float(h.states[k]["creep_strain"][0][0, 0]), c * math.sqrt(t / 10.0)) <= 1e-9
    h1 = stress_path(m, [2e7, 0, 0, 0, 0, 0], [1e-6])
    assert rel(float(h1.states[-1]["p_cr"][0]), c * math.sqrt(1e-6 / 10.0)) <= 1e-9


def _stress_step_run(hardening: str) -> float:
    m = NortonBaileyCreep(**BN, hardening=hardening)
    times = [20.0, 40.0, 100.0, 150.0, 250.0, 400.0]
    sig = np.array([[s, 0, 0, 0, 0, 0] for s in (1e7, 1e7, 1e7, 2e7, 2e7, 2e7)], float)
    h = drive_material_point(m, np.zeros((6, 6)), sig, ALL_STRESS, tol=1e-6, times=times)
    return float(h.states[-1]["p_cr"][0])


def test_c5b_time_versus_strain_hardening_under_a_stress_step() -> None:
    B, tref, m_, t1, t = 1e-4, 10.0, 0.5, 100.0, 400.0
    c1, c2 = B * 1.0, B * 8.0
    p_time = c1 * math.sqrt(t1 / tref) + c2 * (math.sqrt(t / tref) - math.sqrt(t1 / tref))
    p_strain = (c1 ** (1.0 / m_) * t1 / tref + c2 ** (1.0 / m_) * (t - t1) / tref) ** m_
    assert rel(_stress_step_run("time"), p_time) <= 1e-9
    assert rel(_stress_step_run("strain"), p_strain) <= 1e-9
    p1 = c1 * math.sqrt(t1 / tref)
    s_time = solve_ivp(lambda tt, y: [c2 * m_ * tt ** (m_ - 1.0) / tref ** m_], (t1, t), [p1], method="DOP853",
                       rtol=1e-12, atol=1e-18)
    s_strain = solve_ivp(lambda tt, y: [(m_ / tref) * c2 ** (1.0 / m_) * y[0] ** (1.0 - 1.0 / m_)], (t1, t), [p1],
                         method="DOP853", rtol=1e-12, atol=1e-18)
    assert rel(float(s_time.y[0, -1]), p_time) <= 1e-8
    assert rel(float(s_strain.y[0, -1]), p_strain) <= 1e-8
    assert abs(p_time - p_strain) / p_time >= 0.2


def test_c5c_time_hardening_relaxation_first_order_in_phi() -> None:
    B, phi_end = 1e-6, 40.0
    errs = []
    for N in (4, 8, 16, 32):
        m = NortonBaileyCreep(K0, G0, B, SREF, 3.0, 0.5, 1.0)
        times = (np.arange(1, N + 1) * phi_end / N) ** 2
        h = relaxation(m, Q0, times)
        k = 3.0 * G0 * B / SREF ** 3
        q_ref = (Q0 ** -2 + 2.0 * k * phi_end) ** -0.5
        errs.append(rel(vm(h.stress[-1]), q_ref))
    orders = [math.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(0.9 <= o <= 1.1 for o in orders[-2:]), (errs, orders)


@pytest.mark.parametrize("hardening", ["time", "strain"])
def test_c5d_norton_bailey_tangent_against_fd(hardening: str) -> None:
    m = NortonBaileyCreep(**BN, hardening=hardening)
    committed = {"creep_strain": ECR0[None].copy(), "p_cr": np.array([2e-4]), "t_load": np.array([100.0])}
    step = TimeStep(100.0, 50.0)
    eps6 = np.array([1.0e-3, -5e-4, 0.0, 2.4e-4, 1e-4, -1e-4])
    u = m.update(T.voigt_to_strain(eps6)[None], committed, MaterialRequest(step=step))
    assert float(u.trial_state["p_cr"][0]) > 2e-4
    C = u.tangent[0]
    errs = [np.abs(_fd_tangent(m, committed, eps6, step, h) - C).max() / np.abs(C).max()
            for h in np.array([1e-4, 1e-5, 1e-6, 1e-7]) * 1e-3]
    assert min(errs) <= 1e-6, errs


def test_c5e_m1_equals_norton_regression_and_clock_semantics() -> None:
    """Regression (two production paths, not independent): time hardening with m = 1 is Norton with A = B/t_ref."""
    tref, B = 10.0, 1e-5
    times = np.arange(1, 21) * 0.5
    h1 = relaxation(NortonBaileyCreep(K0, G0, B, SREF, 3.0, 1.0, tref), Q0, times)
    h2 = relaxation(NortonCreep(K0, G0, B / tref, SREF, 3.0), Q0, times)
    assert np.abs(h1.stress - h2.stress).max() <= 1e-12 * np.abs(h2.stress).max()
    m = NortonBaileyCreep(**BN)
    st = {"creep_strain": np.zeros((3, 3, 3)), "p_cr": np.zeros(3), "t_load": np.array([5.0, 7.0, 9.0])}
    r = m.reset_load_time(st, 0.0, mask=np.array([True, False, True]))
    assert np.array_equal(r["t_load"], [0.0, 7.0, 0.0]) and np.array_equal(st["t_load"], [5.0, 7.0, 9.0])
    u = m.update(np.zeros((3, 3, 3)), r, MaterialRequest(step=TimeStep(123.0, 3.0)))
    assert np.array_equal(u.trial_state["t_load"], [3.0, 10.0, 3.0])


# ================================================================================================ C6 long holds
def _c6a(rtol: float):
    outs = np.logspace(-6.0, 8.0, 29)
    e0 = Q0 / (3.0 * G0)
    eps = (e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0)
    run = integrate_creep(norton(), ALL_STRAIN, [LoadSegment(float(outs[-1]), strain=eps)], outs, rtol=rtol,
                          atol=1e-12, max_steps=5000)
    ep, eq = [], []
    for k, t in zip(run.output_index, outs):
        q_ref = q_closed(Q0, float(t), 3.0)
        ep.append(rel(float(run.p_cr[k]), (Q0 - q_ref) / (3.0 * G0)))
        eq.append(abs(vm(run.stress[k]) - q_ref) / Q0)
    return run, outs, max(ep), max(eq)


def test_c6a_steps_bounded_and_outputs_exactly_on_the_grid() -> None:
    run, outs, _, _ = _c6a(1e-4)
    assert run.n_accepted <= 2000, run.n_accepted
    assert np.array_equal(run.times[run.output_index], outs)


@pytest.mark.xfail(strict=True, reason="preregistered global accuracy not met: with rtol = 1e-4 the measured maximum "
                   "relative error of eps_cr over the outputs is 2.10e-3 (at t = 100 s) and max |q - q_ref|/q0 = "
                   "1.16e-3, against the preregistered 1e-3 each; step doubling controls the local error only")
def test_c6a_prereg_global_accuracy_rtol_1e4() -> None:
    _, _, ep, eq = _c6a(1e-4)
    assert ep <= 1e-3 and eq <= 1e-3, (ep, eq)


def test_c6a_post_hoc_global_error_decreases_with_rtol() -> None:
    """Post hoc characterisation (not preregistered): the global error falls with rtol (measured at preregistration
    follow-up: 2.1e-3 / 6.7e-4 / 2.1e-4 for rtol 1e-4 / 1e-5 / 1e-6, i.e. ~ 0.2 sqrt(rtol))."""
    _, _, e4, q4 = _c6a(1e-4)
    run5, _, e5, q5 = _c6a(1e-5)
    assert e5 <= 0.5 * e4 and q5 <= 0.5 * q4, (e4, e5, q4, q5)
    assert run5.n_accepted <= 2000


def test_c6b_long_hold_and_step_release_with_exact_event() -> None:
    m = NortonBaileyCreep(**BN)
    t_rel = 1e4
    outs = np.unique(np.concatenate([np.logspace(-6.0, 8.0, 29), [t_rel]]))
    segs = [LoadSegment(t_rel, stress=(2e7, 0, 0, 0, 0, 0)), LoadSegment(float(outs[-1]), stress=(0,) * 6)]
    run = integrate_creep(m, ALL_STRESS, segs, outs, rtol=1e-6, atol=1e-12, max_steps=2000)
    assert run.n_accepted <= 300, run.n_accepted
    assert t_rel in run.times
    c = 1e-4 * 8.0
    for k, t in zip(run.output_index, outs):
        assert rel(float(run.p_cr[k]), c * math.sqrt(min(float(t), t_rel) / 10.0)) <= 1e-8
        if t > t_rel:
            assert np.abs(run.stress[k]).max() <= 1e-2
    assert float(run.p_cr[-1]) > 0.0 and run.creep_strain[-1][0, 0] > 0.0          # irreversible strain stays


# ================================================================================================ C7 rollback/restart
def test_c7a_rollback_restores_the_committed_state_bitwise() -> None:
    m = norton()
    store = StateStore(m, 1)
    e0 = Q0 / (3.0 * G0)
    eps = np.array([[e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0]])
    kw = dict(strain=eps, stress=np.zeros((1, 6)), control=ALL_STRAIN, tol=1e-6)
    h0 = drive_material_point(m, **kw, times=[5.0], store=store)
    cp = store.checkpoint()
    h1 = drive_material_point(m, **kw, times=[20.0], t0=5.0, store=store, strain0=h0.strain[-1])
    assert not np.array_equal(store.committed["creep_strain"], cp["creep_strain"])
    store.restore(cp)
    store.rollback()
    for k in cp:
        assert np.array_equal(store.committed[k], cp[k])
    h2 = drive_material_point(m, **kw, times=[20.0], t0=5.0, store=store, strain0=h0.strain[-1])
    assert np.array_equal(h1.stress, h2.stress)
    assert np.array_equal(h1.states[-1]["creep_strain"], h2.states[-1]["creep_strain"])
    assert np.array_equal(h1.states[-1]["p_cr"], h2.states[-1]["p_cr"])


def _replay(model, run, control, stress_tol, temperature0=None):
    return drive_material_point(model, run.strain_target, run.stress_target, control, tol=stress_tol, times=run.times,
                                temperatures=run.temperature, t0=0.0, temperature0=temperature0)


def test_c7b_rejected_steps_leave_no_trace() -> None:
    m = norton()
    e0 = Q0 / (3.0 * G0)
    eps = (e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0)
    run = integrate_creep(m, ALL_STRAIN, [LoadSegment(1000.0, strain=eps)], [1.0, 10.0, 1000.0], rtol=1e-4,
                          atol=1e-12, dt0=500.0, stress_tol=1e-6)
    assert run.n_rejected >= 1
    rp = _replay(m, run, ALL_STRAIN, 1e-6)
    assert np.array_equal(rp.stress[-1], run.stress[-1])
    assert np.array_equal(rp.states[-1]["creep_strain"][0], run.creep_strain[-1])
    assert np.array_equal(rp.states[-1]["p_cr"][0], run.p_cr[-1])
    # mixed control with a temperature ramp: epsilon_xx held, lateral stresses free
    mt = norton(arrhenius=ArrheniusFactor(5e4, 300.0))
    ctrl = [True, False, False, False, False, False]
    segs = [LoadSegment(100.0, strain=(3e-3, 0, 0, 0, 0, 0), temperature=330.0),
            LoadSegment(400.0, strain=(3e-3, 0, 0, 0, 0, 0))]
    run2 = integrate_creep(mt, ctrl, segs, [50.0, 100.0, 400.0], temperature0=300.0, rtol=1e-4, atol=1e-12,
                           dt0=60.0, stress_tol=1e-3)
    assert run2.n_rejected >= 1 and 100.0 in run2.times
    rp2 = _replay(mt, run2, ctrl, 1e-3, temperature0=300.0)
    assert np.array_equal(rp2.stress[-1], run2.stress[-1])
    assert np.array_equal(rp2.states[-1]["creep_strain"][0], run2.creep_strain[-1])
    assert np.array_equal(rp2.states[-1]["p_cr"][0], run2.p_cr[-1])


def _restart_pair(model, control, segs1, segs2, outs1, outs2, **kw):
    full = integrate_creep(model, control, segs1 + segs2, list(outs1) + list(outs2), **kw)
    first = integrate_creep(model, control, segs1, outs1, **kw)
    st = StateStore(model, 1)
    st.restore(first.store.checkpoint())
    second = integrate_creep(model, control, segs2, outs2, t0=first.t_final, store=st, strain0=first.strain_final,
                             dt_next=first.dt_next, **{k: v for k, v in kw.items() if k != "dt0"})
    return full, first, second


def test_c7c_restart_equals_uninterrupted_run() -> None:
    e0 = Q0 / (3.0 * G0)
    eps = (e0, -0.5 * e0, -0.5 * e0, 0.0, 0.0, 0.0)
    full, first, second = _restart_pair(norton(), ALL_STRAIN, [LoadSegment(10.0, strain=eps)],
                                        [LoadSegment(1000.0, strain=eps)], [1.0, 5.0, 10.0], [100.0, 1000.0],
                                        rtol=1e-5, atol=1e-12, dt0=1e-3, stress_tol=1e-6)
    for name in ("times", "stress", "strain", "creep_strain", "p_cr"):
        assert np.array_equal(getattr(full, name), np.concatenate([getattr(first, name), getattr(second, name)]))
    assert full.n_accepted == first.n_accepted + second.n_accepted
    # stress-controlled Norton-Bailey with a stress step at the restart time (the clock t_load is in the state)
    nb = NortonBaileyCreep(**BN)
    s1, s2 = (1e7, 0, 0, 0, 0, 0), (2e7, 0, 0, 0, 0, 0)
    full, first, second = _restart_pair(nb, ALL_STRESS, [LoadSegment(100.0, stress=s1)],
                                        [LoadSegment(1000.0, stress=s2)], [1.0, 100.0], [300.0, 1000.0],
                                        rtol=1e-6, atol=1e-12, dt0=1e-3, stress_tol=1e-6)
    for name in ("times", "stress", "strain", "creep_strain", "p_cr"):
        assert np.array_equal(getattr(full, name), np.concatenate([getattr(first, name), getattr(second, name)]))
    assert np.array_equal(full.extra["t_load"], second.extra["t_load"])


# ================================================================================================ C8
def test_c8_coupling_tertiary_and_rupture_are_not_implemented() -> None:
    text = " ".join(CR.NOT_IMPLEMENTED).lower()
    for word in ("creep-plasticity", "tertiary", "rupture"):
        assert word in text
    public = [n for n in dir(CR) if not n.startswith("_")]
    assert not any(("plastic" in n.lower() or "damage" in n.lower() or "rupture" in n.lower()) for n in public
                   if n != "NOT_IMPLEMENTED")
