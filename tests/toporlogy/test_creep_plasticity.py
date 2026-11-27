"""Coupled J2 creep + rate-independent plasticity (``toporlogy.core.creep_plasticity.CreepPlasticJ2``).

Thresholds and references are the preregistered ones of ``docs/foundations/prereg/C8_creep_plasticity.md`` (committed
before the first run). References are closed forms derived in that file and written out here, ``solve_ivp``, ``quad``,
``brentq`` and dense numpy; a production kernel is called as a reference only in the limit checks C8a/C8b and the
restart check C8k, which are labelled "regression". All constants are synthetic (not material data).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import MaterialRequest, StateStore, TimeStep, drive_material_point
from toporlogy.core.creep import ArrheniusFactor, NortonCreep
from toporlogy.core.creep_plasticity import CreepPlasticJ2
from toporlogy.core.viscoplastic import PhysicalJ2

K0, G0 = 1.7e10, 8e9                      # synthetic isotropic moduli [Pa]
A0, SREF, N0 = 1e-6, 1e7, 3.0             # synthetic Norton constants
SY, H0 = 1.5e7, 2e9                       # synthetic yield stress and hardening modulus [Pa]
R = 8.314462618                           # J/(mol K), written out here, not imported
E0 = 9.0 * K0 * G0 / (3.0 * K0 + G0)
NU0 = (3.0 * K0 - 2.0 * G0) / (2.0 * (3.0 * K0 + G0))
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
ALL_STRESS = [False] * 6
ALL_STRAIN = [True] * 6
EYE = np.eye(3)
ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
IDEV6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(ONE6, ONE6) / 3.0
CE6 = K0 * np.outer(ONE6, ONE6) + 2.0 * G0 * IDEV6
NAX = np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0])           # Voigt strain of N = diag(1, -1/2, -1/2)


def mk(**kw) -> CreepPlasticJ2:
    return CreepPlasticJ2(K0, G0, kw.get("A", A0), SREF, kw.get("n", N0), kw.get("sy", SY), kw.get("H", H0),
                          arrhenius=kw.get("arr"))


def req(dt: float, T1=None, T0=None, tangent: bool = True, energy: bool = True) -> MaterialRequest:
    return MaterialRequest(energy=energy, tangent=tangent, step=TimeStep(0.0, dt, T1, T0))


def relmax(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.abs(a - b).max()) if a.size else 0.0
    s = float(np.abs(b).max()) if b.size else 0.0
    return d / s if s > 0.0 else d


def vm(sig6) -> float:
    s = np.asarray(sig6, float)
    return math.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
                     + 3.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))


def dev(a: np.ndarray) -> np.ndarray:
    return a - np.trace(a, axis1=-2, axis2=-1)[..., None, None] / 3.0 * EYE


def rand_dev(rng, scale: float) -> np.ndarray:
    a = rng.standard_normal((3, 3))
    d = dev(0.5 * (a + a.T))
    return scale * d / np.linalg.norm(d)


def batch(rng, npts: int, factors, H: float = H0, alpha_max: float = 2e-3, sy: float = SY):
    """Random committed states (ε_cr, ε_p, p_cr, α) and strains whose trial stress is q_tr = factor κ_n at each point."""
    ecr = np.stack([rand_dev(rng, 1e-4) for _ in range(npts)])
    ep = np.stack([rand_dev(rng, 1e-3) for _ in range(npts)])
    al = alpha_max * rng.random(npts)
    pc = np.linalg.norm(ecr, axis=(1, 2)) * math.sqrt(2.0 / 3.0)
    kn = sy + H * al
    f = np.array([factors[i % len(factors)] for i in range(npts)])
    ee = np.stack([rand_dev(rng, 1.0) * (f[i] * kn[i] / (2.0 * G0 * math.sqrt(1.5))) for i in range(npts)])
    vol = 1e-3 * rng.standard_normal(npts)
    eps = ee + ecr + ep + (vol / 3.0)[:, None, None] * EYE
    return eps, {"creep_strain": ecr, "p_cr": pc, "plastic_strain": ep, "alpha": al}


def trial_of(eps: np.ndarray, c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent trial quantities s_tr, q_tr, N = 3 s_tr/(2 q_tr) of every point (dense numpy, written here)."""
    s_tr = 2.0 * G0 * dev(eps - c["creep_strain"] - c["plastic_strain"])
    q_tr = np.sqrt(1.5 * np.einsum("nij,nij->n", s_tr, s_tr))
    return s_tr, q_tr, 1.5 * s_tr / q_tr[:, None, None]


def qof(sig: np.ndarray) -> np.ndarray:
    s = dev(sig)
    return np.sqrt(1.5 * np.einsum("nij,nij->n", s, s))


def stress_path(model, sigma6, times, temperatures=None, temperature0=None):
    k = len(times)
    return drive_material_point(model, np.zeros((k, 6)), np.tile(np.asarray(sigma6, float), (k, 1)), ALL_STRESS,
                                tol=1e-6, times=times, temperatures=temperatures, temperature0=temperature0)


def strain_path(model, eps6, times, t0=0.0, **kw):
    return drive_material_point(model, eps6, np.zeros((len(eps6), 6)), ALL_STRAIN, tol=1e-6, times=times, t0=t0, **kw)


# ======================================================================================================= C8a limit
@pytest.mark.parametrize("flag", [None, math.inf])
@pytest.mark.parametrize("with_T", [False, True])
def test_c8a_infinite_yield_is_norton_creep_bitwise_regression(flag, with_T: bool) -> None:
    arr = ArrheniusFactor(5e4, 300.0) if with_T else None
    m = CreepPlasticJ2(K0, G0, A0, SREF, N0, flag, H0, arrhenius=arr)
    ref = NortonCreep(K0, G0, A0, SREF, N0, arrhenius=arr)
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        eps, c = batch(rng, 40, [0.3, 1.0, 2.0, 3.0])
        eps[0] = 0.25 * EYE                                                    # q_tr = 0 point (hydrostatic)
        c["creep_strain"][0] = 0.0
        c["plastic_strain"][:] = 0.0
        c["alpha"][:] = 0.0
        cref = {"creep_strain": c["creep_strain"], "p_cr": c["p_cr"]}
        for dt in (0.0, 30.0):
            st = TimeStep(0.0, dt, 340.0, 300.0) if with_T else TimeStep(0.0, dt)
            rq = MaterialRequest(energy=True, step=st)
            a, b = m.update(eps, c, rq), ref.update(eps, cref, rq)
            assert np.array_equal(a.stress, b.stress) and np.array_equal(a.tangent, b.tangent)
            assert np.array_equal(a.energy, b.energy) and np.array_equal(a.dissipation, b.dissipation)
            assert np.array_equal(a.trial_state["creep_strain"], b.trial_state["creep_strain"])
            assert np.array_equal(a.trial_state["p_cr"], b.trial_state["p_cr"])
            assert not a.trial_state["plastic_strain"].any() and not a.trial_state["alpha"].any()
    k = 40
    eps6 = np.array([[2e-3 * math.sin(0.3 * j), -1e-3 * math.sin(0.3 * j + 0.4), 3e-4, 4e-4, 0.0, 6e-4 * math.sin(j)]
                     for j in range(1, k + 1)])
    tt = 2.0 * np.arange(1, k + 1)
    kw = dict(temperatures=300.0 + np.arange(1, k + 1), temperature0=300.0) if with_T else {}
    ha = strain_path(m, eps6, tt, **kw)
    hb = strain_path(ref, eps6, tt, **kw)
    assert np.array_equal(ha.stress, hb.stress) and np.array_equal(ha.dissipation, hb.dissipation)
    assert np.array_equal(ha.energy, hb.energy, equal_nan=True)
    assert np.array_equal(ha.states[-1]["creep_strain"], hb.states[-1]["creep_strain"])
    assert float(ha.states[-1]["p_cr"][0]) > 0.0                                  # the path did creep


# ======================================================================================================= C8b limit
def _cyclic_path(k: int = 36):
    j = np.arange(1, k + 1)
    ph = 2.0 * math.pi * 3.0 * j / k
    return np.stack([3e-3 * np.sin(ph), -1e-3 * np.sin(ph + 0.5), 5e-4 * np.sin(2.0 * ph), 8e-4 * np.sin(ph),
                     np.zeros(k), 6e-4 * np.sin(ph + 1.0)], axis=1)


@pytest.mark.parametrize("H", [H0, 0.0])
def test_c8b_zero_creep_is_physical_j2(H: float) -> None:
    m, ref = mk(A=0.0, H=H), PhysicalJ2(K0, G0, SY, H)
    rng = np.random.default_rng(7)
    eps, c = batch(rng, 50, [0.5, 1.2, 2.0, 3.0], H=H)
    c["creep_strain"][:] = 0.0
    c["p_cr"][:] = 0.0
    cref = {"plastic_strain": c["plastic_strain"], "alpha": c["alpha"]}
    a = m.update(eps, c, req(5.0))
    b = ref.update(eps, cref, MaterialRequest(energy=True))
    assert relmax(a.stress, b.stress) <= 1e-13 and relmax(a.tangent, b.tangent) <= 1e-13
    assert relmax(a.energy, b.energy) <= 1e-13 and relmax(a.dissipation, b.dissipation) <= 1e-13
    assert relmax(a.trial_state["plastic_strain"], b.trial_state["plastic_strain"]) <= 1e-13
    assert relmax(a.trial_state["alpha"], b.trial_state["alpha"]) <= 1e-13
    assert not a.trial_state["creep_strain"].any() and not a.trial_state["p_cr"].any()
    assert (a.diagnostics["plastic"] > 0).sum() >= 20                                # plastic points occur
    eps6 = _cyclic_path()
    tt = np.arange(1.0, 37.0)
    ha = strain_path(m, eps6, tt)
    hb = drive_material_point(ref, eps6, np.zeros((36, 6)), ALL_STRAIN, tol=1e-6)
    assert relmax(ha.stress, hb.stress) <= 1e-13 and relmax(ha.dissipation, hb.dissipation) <= 1e-13
    assert relmax(ha.energy, hb.energy) <= 1e-13
    for sa, sb in zip(ha.states, hb.states):
        assert relmax(sa["plastic_strain"], sb["plastic_strain"]) <= 1e-13
        assert relmax(sa["alpha"], sb["alpha"]) <= 1e-13
        assert not sa["creep_strain"].any() and not sa["p_cr"].any()
    assert float(hb.states[-1]["alpha"][0]) > 1e-3                                    # the path yielded repeatedly


# ======================================================================================== C8c constant-stress creep
def _hold_reference(sig0: float, T_iso: float | None):
    f = 1.0 if T_iso is None else math.exp(-5e4 / R * (1.0 / T_iso - 1.0 / 300.0))
    return A0 * f * (abs(sig0) / SREF) ** N0


@pytest.mark.parametrize("sig0", [2e7, -2e7])
@pytest.mark.parametrize("T_iso", [None, 360.0])
def test_c8c_constant_stress_above_yield_closed_form(sig0: float, T_iso) -> None:
    sg = math.copysign(1.0, sig0)
    arr = None if T_iso is None else ArrheniusFactor(5e4, 300.0)
    m = mk(arr=arr)
    times = [0.0, 5.0, 25.0, 100.0]
    kw = {} if T_iso is None else dict(temperatures=[T_iso] * 4, temperature0=T_iso)
    h = stress_path(m, [sig0, 0, 0, 0, 0, 0], times, **kw)
    c = _hold_reference(sig0, T_iso)
    a0 = (abs(sig0) - SY) / H0                                                       # q = σ0 = κ(α0) at the load step
    t = np.array(times)
    ax = np.array([e[0] for e in h.strain])
    lat = np.array([e[1] for e in h.strain])
    assert relmax(ax, sig0 / E0 + sg * a0 + sg * c * t) <= 1e-9
    assert relmax(lat, -NU0 * sig0 / E0 - sg * a0 / 2.0 - sg * c * t / 2.0) <= 1e-9
    ecr = np.array([s["creep_strain"][0] for s in h.states])
    ep = np.array([s["plastic_strain"][0] for s in h.states])
    assert relmax(ecr[:, 0, 0], sg * c * t) <= 1e-9 and relmax(ecr[:, 1, 1], -sg * c * t / 2.0) <= 1e-9
    assert relmax(ep[:, 0, 0], np.full(4, sg * a0)) <= 1e-9 and relmax(ep[:, 1, 1], np.full(4, -sg * a0 / 2.0)) <= 1e-9
    assert relmax([s["p_cr"][0] for s in h.states], c * t) <= 1e-9 and (np.array([s["p_cr"][0] for s in h.states]) >= 0).all()
    assert relmax([s["alpha"][0] for s in h.states], np.full(4, a0)) <= 1e-9
    d_ref = np.array([SY * a0 + 0.5 * H0 * a0 ** 2] + [abs(sig0) * c * dt for dt in np.diff(times)])
    assert relmax(h.dissipation, d_ref) <= 1e-9


@pytest.mark.parametrize("k", [1, 4])
def test_c8c_variable_temperature_ramp_after_load(k: int) -> None:
    Q, Tr, sig0 = 5e4, 300.0, 2e7
    m = mk(arr=ArrheniusFactor(Q, Tr))
    times = [0.0] + [10.0 * j / k for j in range(1, k + 1)]
    temps = [300.0] + [300.0 + 100.0 * j / k for j in range(1, k + 1)]
    h = stress_path(m, [sig0, 0, 0, 0, 0, 0], times, temperatures=temps, temperature0=300.0)
    integral, _ = quad(lambda t: math.exp(-Q / R * (1.0 / (300.0 + 10.0 * t) - 1.0 / Tr)), 0.0, 10.0, epsabs=1e-13,
                       epsrel=1e-13)
    p_ref = A0 * (sig0 / SREF) ** N0 * integral
    assert abs(float(h.states[-1]["p_cr"][0]) - p_ref) / p_ref <= 1e-9
    assert abs(float(h.states[-1]["alpha"][0]) - (sig0 - SY) / H0) <= 1e-9 * (sig0 - SY) / H0


# ========================================================================================== reference rate system
def ode_reference(phases, e0=0.0, z0=(0.0, 0.0), A=A0, n=N0, sy=SY, H=H0):
    """Independent reference (prereg section 4): z = (p_cr, alpha) of the rate system with active-set logic.
    ``phases`` = [(t_a, t_b, e_dot)]; returns the states at the phase ends and the time of the first yield."""
    g3 = 3.0 * G0
    r = lambda q: A * (max(q, 0.0) / SREF) ** n                               # noqa: E731
    z = np.array(z0, float)
    t_y = None
    e_a = e0
    out = []
    for ta, tb, edot in phases:
        e_ph = e_a
        q = g3 * (e_ph - z[0] - z[1])
        plastic = (q - (sy + H * z[1]) > -1e-6) and (g3 * (edot - r(q)) / (g3 + H) > 0.0)
        t = ta
        guard = 0
        while t < tb:
            guard += 1
            assert guard < 10
            if plastic:
                def rhs(_t, zz):
                    rq = r(sy + H * zz[1])
                    return [rq, g3 * (edot - rq) / (g3 + H)]

                def ev(_t, zz):
                    return g3 * (edot - r(sy + H * zz[1])) / (g3 + H)
                ev.direction = -1.0
            else:
                def rhs(_t, zz, e_ph=e_ph, ta=ta):
                    return [r(g3 * (e_ph + edot * (_t - ta) - zz[0] - zz[1])), 0.0]

                def ev(_t, zz, e_ph=e_ph, ta=ta):
                    return g3 * (e_ph + edot * (_t - ta) - zz[0] - zz[1]) - (sy + H * zz[1])
                ev.direction = 1.0
            ev.terminal = True
            sol = solve_ivp(rhs, (t, tb), z, method="DOP853", rtol=1e-12, atol=1e-18, events=ev)
            z = sol.y[:, -1].copy()
            if sol.status == 1:
                t = float(sol.t_events[0][0])
                if not plastic and t_y is None:
                    t_y = t
                plastic = not plastic
            else:
                t = tb
        out.append(z.copy())
        e_a = e_ph + edot * (tb - ta)
    return out, t_y


def q_closed(q0: float, t: float, n: float = N0) -> float:
    k = 3.0 * G0 * A0 / SREF ** n
    return (q0 ** (1.0 - n) + (n - 1.0) * k * t) ** (1.0 / (1.0 - n))


# ================================================================================================ C8d relaxation
E1 = 3e-3


def _relax(m, dt: float, t_end: float = 60.0):
    n = int(round(t_end / dt))
    tt = np.concatenate([[0.0], dt * np.arange(1, n + 1)])
    return strain_path(m, np.tile(E1 * NAX, (len(tt), 1)), tt)


def test_c8d_held_strain_relaxation_from_plastic_state() -> None:
    a0 = (3.0 * G0 * E1 - SY) / (3.0 * G0 + H0)
    q0 = SY + H0 * a0
    h = _relax(mk(), 0.025)
    q_ref = q_closed(q0, 60.0)
    p_ref = (q0 - q_ref) / (3.0 * G0)
    assert abs(vm(h.stress[0]) - q0) <= 1e-9 * q0 and abs(float(h.states[0]["alpha"][0]) - a0) <= 1e-9 * a0
    assert abs(vm(h.stress[-1]) - q_ref) / q_ref <= 2e-3
    assert abs(float(h.states[-1]["p_cr"][0]) - p_ref) / p_ref <= 2e-3
    zs, _ = ode_reference([(0.0, 60.0, 0.0)], e0=E1, z0=(0.0, a0))
    assert abs(zs[0][0] - p_ref) / p_ref <= 1e-9                              # two independent references agree
    assert abs(zs[0][1] - a0) <= 1e-15
    for k in range(1, len(h.states)):
        assert np.array_equal(h.states[k]["alpha"], h.states[0]["alpha"])
        assert np.array_equal(h.states[k]["plastic_strain"], h.states[0]["plastic_strain"])
        assert h.diagnostics[k]["plastic"] == 0.0
    errs = [abs(vm(_relax(mk(), dt).stress[-1]) - q_ref) / q_ref for dt in (0.2, 0.1, 0.05)]
    orders = [math.log2(a / b) for a, b in zip(errs, errs[1:])]
    assert all(0.9 <= o <= 1.1 for o in orders), (errs, orders)


# ================================================================================================ C8e ramp + hold
def _ramp_hold(m, N: int):
    tk = np.concatenate([30.0 * np.arange(1, N + 1) / N, 30.0 + 30.0 * np.arange(1, N + 1) / N])
    e = np.where(tk <= 30.0, 1e-4 * tk, 3e-3)
    return strain_path(m, e[:, None] * NAX[None], tk)


def test_c8e_ramp_and_hold_against_ivp_with_events() -> None:
    zs, t_y = ode_reference([(0.0, 30.0, 1e-4), (30.0, 60.0, 0.0)])
    (p30, a30), (p60, a60) = zs
    assert t_y is not None and 0.0 < t_y < 30.0 and a30 > 0.0 and p30 > 0.0 and a60 == a30
    q30, q60 = 3.0 * G0 * (3e-3 - p30 - a30), 3.0 * G0 * (3e-3 - p60 - a60)
    errq = []
    for N in (250, 500, 1000):
        h = _ramp_hold(mk(), N)
        got = {30: (vm(h.stress[N - 1]), float(h.states[N - 1]["p_cr"][0]), float(h.states[N - 1]["alpha"][0])),
               60: (vm(h.stress[2 * N - 1]), float(h.states[2 * N - 1]["p_cr"][0]),
                    float(h.states[2 * N - 1]["alpha"][0]))}
        errq.append(abs(got[60][0] - q60) / q60)
        if N == 1000:
            for tt_, (qq, pp, aa), (qr, pr, ar) in ((30, got[30], (q30, p30, a30)), (60, got[60], (q60, p60, a60))):
                assert abs(qq - qr) / qr <= 2e-3 and abs(pp - pr) / pr <= 2e-3 and abs(aa - ar) / ar <= 2e-3, tt_
            assert np.array_equal(h.states[2 * N - 1]["alpha"], h.states[N - 1]["alpha"])
    orders = [math.log2(a / b) for a, b in zip(errq, errq[1:])]
    assert all(0.8 <= o <= 1.2 for o in orders), (errq, orders)


# ===================================================================================================== C8f tangent
def _fd_tangent(model, committed, eps6, step_req, h):
    C = np.zeros((6, 6))
    rq = MaterialRequest(step=step_req.step, tangent=False)
    for j in range(6):
        out = []
        for sgn in (1.0, -1.0):
            e = np.array(eps6, float)
            e[j] += sgn * h
            out.append(model.update(T.voigt_to_strain(e)[None], committed, rq).stress[0][_VA, _VB])
        C[:, j] = (out[0] - out[1]) / (2.0 * h)
    return C


HS = np.array([1e-4, 1e-5, 1e-6, 1e-7]) * 1e-3


def _best_fd_error(model, committed, eps6, rq, C) -> float:
    return min(relmax(_fd_tangent(model, committed, eps6, rq, h), C) for h in HS)


def _single(c, i):
    return {k: v[i:i + 1] for k, v in c.items()}


def _voigt(eps_t):
    return np.array([eps_t[a, b] * (1.0 if a == b else 2.0) for a, b in T.VOIGT_3D])


@pytest.mark.parametrize("n,H", [(3.0, H0), (1.0, H0), (3.0, 0.0)])
def test_c8f_tangent_plastic_and_creeping(n: float, H: float) -> None:
    m = mk(n=n, H=H)
    eps, c = batch(np.random.default_rng(11), 12, [1.6], H=H)
    worst = 0.0
    for i in range(12):
        ci = _single(c, i)
        rq = req(30.0)
        u = m.update(eps[i:i + 1], ci, rq)
        d = u.diagnostics
        assert d["plastic"][0] == 1.0 and d["delta_p_cr"][0] > 0.0 and d["delta_alpha"][0] > 0.0
        assert d["delta_p_cr"][0] / d["delta_alpha"][0] >= 0.05
        worst = max(worst, _best_fd_error(m, ci, _voigt(eps[i]), rq, u.tangent[0]))
    assert worst <= 1e-6, worst


def test_c8f_tangent_creep_only_and_continuum_refused() -> None:
    m = mk()
    eps, c = batch(np.random.default_rng(12), 12, [0.8])
    worst = 0.0
    for i in range(12):
        ci = _single(c, i)
        rq = req(30.0)
        u = m.update(eps[i:i + 1], ci, rq)
        _, q_tr, _ = trial_of(eps[i:i + 1], ci)
        assert u.diagnostics["plastic"][0] == 0.0 and u.diagnostics["q"][0] / q_tr[0] <= 0.97
        worst = max(worst, _best_fd_error(m, ci, _voigt(eps[i]), rq, u.tangent[0]))
    assert worst <= 1e-6, worst
    with pytest.raises(ValueError):
        m.update(eps[:1], _single(c, 0), MaterialRequest(step=TimeStep(0.0, 1.0), tangent_kind="continuum"))


def _derived_tangent(model, committed, eps, dt, frozen=False):
    """The tangent of the preregistration written out in the test (plastic branch), from the returned step."""
    u = model.update(eps, committed, req(dt))
    d = u.diagnostics
    s_tr, q_tr, N = trial_of(eps, committed)
    G, H = G0, model.H
    q, x, y = float(d["q"][0]), float(d["delta_p_cr"][0]), float(d["delta_alpha"][0])
    hp = dt * A0 * model.n * (q / SREF) ** (model.n - 1.0) / SREF
    w = H / (3.0 * G + H)
    xd = w * hp / (1.0 + 3.0 * G * w * hp)
    lam_p = xd + (1.0 - 3.0 * G * xd) / (3.0 * G + H)
    nv = N[0][_VA, _VB]
    if frozen:
        return u, CE6 - 4.0 * G ** 2 * lam_p * np.outer(nv, nv)
    th = q / q_tr[0]
    return u, (K0 * np.outer(ONE6, ONE6) + 2.0 * G * th * IDEV6
               - 4.0 * G ** 2 * (lam_p - (x + y) / q_tr[0]) * np.outer(nv, nv))


def test_c8g_wrong_tangents_are_detected() -> None:
    m = mk()
    eps, c = batch(np.random.default_rng(13), 8, [1.6])
    for i in range(8):
        ci, rq = _single(c, i), req(30.0)
        u, C_der = _derived_tangent(m, ci, eps[i:i + 1], 30.0)
        assert relmax(u.tangent[0], C_der) <= 1e-10                                # the formula of the doc
        _, C_fro = _derived_tangent(m, ci, eps[i:i + 1], 30.0, frozen=True)
        assert min(relmax(_fd_tangent(m, ci, _voigt(eps[i]), rq, h), C_fro) for h in HS) >= 1e-3
    m2 = mk(H=1e10)
    eps2, c2 = batch(np.random.default_rng(14), 8, [1.6], H=1e10, alpha_max=1e-4)
    for i in range(8):
        ci, rq = _single(c2, i), req(50.0)
        u, C_der = _derived_tangent(m2, ci, eps2[i:i + 1], 50.0)
        assert u.diagnostics["plastic"][0] == 1.0 and relmax(u.tangent[0], C_der) <= 1e-10
        # creep ignored in the plastic part: the right Lambda' differs by w x' (about 3e-12 1/Pa)
        s_tr, q_tr, N = trial_of(eps2[i:i + 1], ci)
        nv = N[0][_VA, _VB]
        d = u.diagnostics
        th = float(d["q"][0]) / q_tr[0]
        lam = float(d["delta_p_cr"][0] + d["delta_alpha"][0])
        C_wrong = (K0 * np.outer(ONE6, ONE6) + 2.0 * G0 * th * IDEV6
                   - 4.0 * G0 ** 2 * (1.0 / (3.0 * G0 + 1e10) - lam / q_tr[0]) * np.outer(nv, nv))
        assert min(relmax(_fd_tangent(m2, ci, _voigt(eps2[i]), rq, h), C_wrong) for h in HS) >= 1e-3
        assert _best_fd_error(m2, ci, _voigt(eps2[i]), rq, u.tangent[0]) <= 1e-6


# ================================================================ C8h separation, C8i local system, C8j dissipation
PARAMS = [(3.0, H0), (1.0, 0.0), (0.7, H0), (8.0, H0)]
FACTORS = [0.5, 0.95, 1.3, 1.8, 2.5]


def _states_for(n: float, H: float, dt: float, seed: int, npts: int = 20):
    eps, c = batch(np.random.default_rng(seed), npts, FACTORS, H=H)
    return mk(n=n, H=H), eps, c


def test_c8h_separation_and_no_double_counting() -> None:
    n_pl = n_el = 0
    for (n, H) in PARAMS[:2]:
        for dt in (0.0, 0.5, 30.0):
            m, eps, c = _states_for(n, H, dt, 100 + int(dt))
            u = m.update(eps, c, req(dt))
            ts = u.trial_state
            s_tr, q_tr, N = trial_of(eps, c)
            x = ts["p_cr"] - c["p_cr"]
            y = ts["alpha"] - c["alpha"]
            assert (x >= 0).all() and (y >= 0).all()
            dcr, dep = ts["creep_strain"] - c["creep_strain"], ts["plastic_strain"] - c["plastic_strain"]
            xN, yN = x[:, None, None] * N, y[:, None, None] * N
            assert np.abs(dcr - xN).max() <= 1e-12 * max(np.abs(xN).max(), 1e-30)
            assert np.abs(dep - yN).max() <= 1e-12 * max(np.abs(yN).max(), 1e-30)
            assert np.abs(dcr + dep - (x + y)[:, None, None] * N).max() <= 1e-12 * max(np.abs(xN + yN).max(), 1e-30)
            for dd, inc in ((dcr, x), (dep, y)):
                eq = math.sqrt(2.0 / 3.0) * np.linalg.norm(dd, axis=(1, 2))
                assert np.abs(eq - inc).max() <= 1e-12 * max(inc.max(), 1e-30)
            ee = eps - ts["creep_strain"] - ts["plastic_strain"]
            sig = K0 * np.trace(ee, axis1=1, axis2=2)[:, None, None] * EYE + 2.0 * G0 * dev(ee)
            assert relmax(u.stress, sig) <= 1e-12
            q_new = qof(u.stress)
            assert np.abs(q_new - (q_tr - 3.0 * G0 * (x + y))).max() <= 1e-12 * q_tr.max()
            n_pl += int((u.diagnostics["plastic"] > 0).sum())
            n_el += int(((u.diagnostics["plastic"] == 0) & (q_tr > 0)).sum())
    assert n_pl >= 10 and n_el >= 10


def _independent_solution(n, H, dt, eps, c, i):
    """Both branches by brentq (prereg C8i): returns (plastic?, x, y) of point i."""
    s_tr, q_tr, _ = trial_of(eps[i:i + 1], _single(c, i))
    qt = float(q_tr[0])
    kn = SY + H * float(c["alpha"][i])
    hf = lambda q: dt * A0 * (max(q, 0.0) / SREF) ** n                          # noqa: E731
    cap = qt / (3.0 * G0)
    xc = 0.0 if dt == 0.0 else brentq(lambda x: x - hf(qt - 3.0 * G0 * x), 0.0, cap, xtol=1e-22, rtol=1e-15, maxiter=500)
    qc = qt - 3.0 * G0 * xc
    if not qc > kn:
        return False, xc, 0.0
    if H == 0.0:
        xp = hf(kn)
    else:
        w = H / (3.0 * G0 + H)
        xmax = (qt - kn) / (3.0 * G0)
        xp = 0.0 if dt == 0.0 else brentq(lambda x: x - hf(kn + w * (qt - kn - 3.0 * G0 * x)), 0.0, xmax, xtol=1e-22,
                                          rtol=1e-15, maxiter=500)
    return True, xp, (qt - kn - 3.0 * G0 * xp) / (3.0 * G0 + H)


@pytest.mark.parametrize("n,H", PARAMS)
def test_c8i_local_system_kkt_and_independent_active_set(n: float, H: float) -> None:
    for dt in (0.0, 0.5, 30.0):
        m, eps, c = _states_for(n, H, dt, 200 + int(dt))
        u = m.update(eps, c, req(dt))
        ts = u.trial_state
        s_tr, q_tr, _ = trial_of(eps, c)
        x, y = ts["p_cr"] - c["p_cr"], ts["alpha"] - c["alpha"]
        q_new = qof(u.stress)
        kn = SY + H * c["alpha"]
        cap = q_tr / (3.0 * G0)
        r1 = x - dt * A0 * (q_new / SREF) ** n
        assert (np.abs(r1) <= 1e-10 * np.maximum(x, 1e-3 * cap)).all(), np.abs(r1).max()
        pl = y > 0.0
        assert (np.abs(q_new[pl] - (SY + H * ts["alpha"][pl])) <= 1e-12 * SY).all()
        assert (q_new[~pl] <= kn[~pl] * (1.0 + 1e-12)).all()
        fy = q_new - (SY + H * ts["alpha"])
        assert (fy <= 1e-12 * SY).all() and (y * np.maximum(fy, 0.0) <= 1e-12 * SY * np.maximum(y, 1e-30)).all()
        for i in range(len(eps)):
            ip, xr, yr = _independent_solution(n, H, dt, eps, c, i)
            assert ip == bool(pl[i]), (i, dt)
            assert abs(x[i] - xr) <= 1e-9 * (xr + yr) + 1e-30 and abs(y[i] - yr) <= 1e-9 * (xr + yr) + 1e-30


@pytest.mark.parametrize("n,H", PARAMS)
def test_c8j_dissipation_split_and_nonnegative(n: float, H: float) -> None:
    for dt in (0.0, 0.5, 30.0):
        m, eps, c = _states_for(n, H, dt, 300 + int(dt))
        u = m.update(eps, c, req(dt))
        ts = u.trial_state
        x, y = ts["p_cr"] - c["p_cr"], ts["alpha"] - c["alpha"]
        q_new = qof(u.stress)
        dpsi = 0.5 * H * (ts["alpha"] ** 2 - c["alpha"] ** 2)
        d_cr, d_p = q_new * x, q_new * y - dpsi
        assert (d_cr >= 0).all() and (d_p >= SY * y * (1.0 - 1e-9)).all()
        assert relmax(u.dissipation, d_cr + d_p) <= 1e-12
        dein = (ts["creep_strain"] - c["creep_strain"]) + (ts["plastic_strain"] - c["plastic_strain"])
        assert relmax(np.einsum("nij,nij->n", u.stress, dein), q_new * (x + y)) <= 1e-12
        assert relmax(u.diagnostics["dissipation_creep"], d_cr) <= 1e-12 and relmax(
            u.diagnostics["dissipation_plastic"], d_p) <= 1e-12
    m = mk()
    h = strain_path(m, _cyclic_path(), 2.0 * np.arange(1, 37))
    assert sum(d["plastic"] for d in h.diagnostics) >= 6 and sum(d["delta_p_cr"] for d in h.diagnostics) > 0.0
    for dis, d in zip(h.dissipation, h.diagnostics):
        assert dis >= -1e-9 * d["inelastic_work"]


# ===================================================================================================== C8k restart
def _mixed_targets(k: int):
    j = np.arange(1, k + 1)
    ph = 2.0 * math.pi * j / 8.0
    eps6 = np.stack([2.5e-3 * np.sin(ph), -8e-4 * np.sin(ph + 0.4), 4e-4 * np.sin(2 * ph), 5e-4 * np.sin(ph),
                     np.zeros(k), np.zeros(k)], axis=1)
    sig6 = np.zeros((k, 6))
    sig6[:, 5] = 5e6
    return eps6, sig6


def test_c8k_rollback_and_restart_bitwise_regression() -> None:
    m = mk(arr=ArrheniusFactor(5e4, 300.0))
    k = 12
    eps6, sig6 = _mixed_targets(k)
    ctrl = [True, True, True, True, True, False]
    tt = 3.0 * np.arange(1, k + 1)
    TT = 300.0 + 4.0 * np.arange(1, k + 1)

    def run(lo, hi, store=None, strain0=None):
        t0 = 0.0 if lo == 0 else float(tt[lo - 1])
        T0 = 300.0 if lo == 0 else float(TT[lo - 1])
        return drive_material_point(m, eps6[lo:hi], sig6[lo:hi], ctrl, tol=1e-6, times=tt[lo:hi],
                                    temperatures=TT[lo:hi], t0=t0, temperature0=T0, store=store, strain0=strain0)

    full = run(0, k)
    assert sum(d["plastic"] for d in full.diagnostics) >= 2 and float(full.states[-1]["p_cr"][0]) > 0.0
    store = StateStore(m, 1)
    h1 = run(0, 3, store)
    cp = store.checkpoint()
    first = run(3, 4, store, h1.strain[-1])
    store.restore(cp)
    assert all(np.array_equal(store.committed[f], cp[f]) for f in m.state_fields)
    again = run(3, 4, store, h1.strain[-1])
    assert np.array_equal(first.stress, again.stress) and np.array_equal(first.dissipation, again.dissipation)
    assert all(np.array_equal(first.states[-1][f], again.states[-1][f]) for f in m.state_fields)
    store2 = StateStore(m, 1)
    ha = run(0, 7, store2)
    hb = run(7, k, store2, ha.strain[-1])
    assert np.array_equal(np.concatenate([ha.stress, hb.stress]), full.stress)
    assert np.array_equal(np.concatenate([ha.dissipation, hb.dissipation]), full.dissipation)
    assert all(np.array_equal(hb.states[-1][f], full.states[-1][f]) for f in m.state_fields)


# ============================================================================================ C8l sequential split
def _seq(dt, qt, kn, H, n=N0):
    hf = lambda q: dt * A0 * (max(q, 0.0) / SREF) ** n                          # noqa: E731
    xs = brentq(lambda x: x - hf(qt - 3.0 * G0 * x), 0.0, qt / (3.0 * G0), xtol=1e-22, rtol=1e-15, maxiter=500)
    qc = qt - 3.0 * G0 * xs
    ys = (qc - kn) / (3.0 * G0 + H) if qc > kn else 0.0
    return xs, ys, hf


def test_c8l_sequential_split_differs_from_coupled() -> None:
    m = mk()
    dt, qt = 20.0, 3e7
    e = qt / (3.0 * G0)
    eps = (e * np.diag([1.0, -0.5, -0.5]))[None]
    u = m.update(eps, m.initial_state(1), req(dt))
    xc, yc = float(u.trial_state["p_cr"][0]), float(u.trial_state["alpha"][0])
    xs, ys, hf = _seq(dt, qt, SY, H0)
    assert abs(xs - xc) / xc >= 0.5, (xs, xc)
    q_seq = qt - 3.0 * G0 * (xs + ys)
    assert abs(xs - hf(q_seq)) / xs >= 0.5                                      # the sequential end state breaks r1
    q_cpl = float(qof(u.stress)[0])
    assert abs(xc - hf(q_cpl)) <= 1e-10 * xc                                    # the coupled update satisfies it
    assert yc > 0.0 and ys > 0.0
    # (iii) from the yield surface with a fixed strain rate: the difference falls with the step
    a0, edot = 1e-3, 1e-4
    kn = SY + H0 * a0
    n0 = np.diag([1.0, -0.5, -0.5])
    committed = {"creep_strain": np.zeros((1, 3, 3)), "p_cr": np.zeros(1), "plastic_strain": (a0 * n0)[None],
                 "alpha": np.array([a0])}
    diffs = []
    for dt in (8.0, 2.0, 0.5, 0.125):
        eps = ((a0 + kn / (3.0 * G0) + edot * dt) * n0)[None]
        u = m.update(eps, committed, req(dt))
        xc = float(u.trial_state["p_cr"][0])
        xs, ys, _ = _seq(dt, kn + 3.0 * G0 * edot * dt, kn, H0)
        assert ys > 0.0 and u.diagnostics["plastic"][0] == 1.0
        diffs.append(abs(xs - xc) / xc)
    assert all(a > b for a, b in zip(diffs, diffs[1:])) and diffs[-2] / diffs[-1] >= 2.0, diffs


# ===================================================================================================== C8m validation
def test_c8m_validation_and_degenerate_input() -> None:
    good = dict(K=K0, G=G0, A=A0, sigma_ref=SREF, n=N0, yield_stress=SY, hardening=H0)
    for key, bad in (("K", 0.0), ("G", -1.0), ("A", -1e-6), ("n", 0.0), ("sigma_ref", 0.0), ("yield_stress", 0.0),
                     ("yield_stress", -1e6), ("yield_stress", float("nan")), ("hardening", -1.0),
                     ("hardening", float("inf")), ("hardening", float("nan")), ("n", float("nan"))):
        with pytest.raises(ValueError):
            CreepPlasticJ2(**{**good, key: bad})
    with pytest.raises(ValueError):
        CreepPlasticJ2(**good, arrhenius="Arrhenius")                                       # type: ignore[arg-type]
    eps = np.zeros((1, 3, 3))
    st = mk().initial_state(1)
    with pytest.raises(ValueError):
        mk().update(eps, st, MaterialRequest())                                             # no physical step
    with pytest.raises(ValueError):
        mk(arr=ArrheniusFactor(5e4, 300.0)).update(eps, st, MaterialRequest(step=TimeStep(0.0, 1.0)))
    a = CreepPlasticJ2(**{**good, "yield_stress": None}).update(np.full((1, 3, 3), 0.0) + 1e-3 * np.diag([1.0, -1.0, 0.2]),
                                                                  st, req(10.0))
    b = CreepPlasticJ2(**{**good, "yield_stress": math.inf}).update(
        np.full((1, 3, 3), 0.0) + 1e-3 * np.diag([1.0, -1.0, 0.2]), st, req(10.0))
    assert np.array_equal(a.stress, b.stress) and np.array_equal(a.tangent, b.tangent)
    m = mk()
    e = (3e-3 * np.diag([1.0, -0.5, -0.5]))[None]
    u = m.update(e, m.initial_state(1), req(0.0))                                           # dt = 0: plastic, no creep
    assert u.trial_state["p_cr"][0] == 0.0 and not u.trial_state["creep_strain"].any()
    assert u.diagnostics["plastic"][0] == 1.0 and u.trial_state["alpha"][0] > 0.0
    hyd = np.full((1, 3, 3), 0.0) + 0.25 * np.eye(3)                                        # exactly zero deviator
    u = m.update(hyd, st, req(1e6))
    assert all(np.array_equal(u.trial_state[f], st[f]) for f in m.state_fields)
    assert np.array_equal(u.tangent[0], m.Ce6) and u.diagnostics["plastic"][0] == 0.0
