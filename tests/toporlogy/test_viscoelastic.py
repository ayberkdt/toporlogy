"""V_viscoelastic: linear viscoelasticity, generalized Maxwell, DMA, time–temperature shift (V1–V10, C1).

Preregistration: ``docs/foundations/prereg/V_viscoelastic.md`` (thresholds fixed there before the first run). All
parameters are synthetic. References are independent of the production recursion: element ODEs (``solve_ivp``),
Laplace transforms by quadrature, mpmath, closed forms derived in the preregistration, a viscous-strain ODE oracle, a
Van Loan matrix-exponential oracle, a DAE oracle for the uniaxial-stress constraint and a residue oracle. Tests that are
only regressions of the contract are labelled "regression".
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

from toporlogy.core import tensor as T
from toporlogy.core import viscoelastic as V
from toporlogy.core.constitutive import Kinematics, MaterialRequest, StateStore, TimeStep, drive_material_point

R = 8.314462618
VA = [p[0] for p in T.VOIGT_3D]
VB = [p[1] for p in T.VOIGT_3D]


def PS(inf, A, tau):
    return V.PronySeries(inf, np.array(A, float), np.array(tau, float))


SETS = {
    "A": (PS(0.5e6, [1.0e6, 0.5e6], [2, 30]), PS(2.0e6, [1.5e6, 0.5e6, 1.0e6], [0.5, 8, 200])),
    "W": (PS(0.3e6, [0.4e6, 0.6e6, 0.5e6], [1e-4, 1.0, 1e4]), PS(1.0e6, [1.0e6, 2.0e6], [1e-3, 50.0])),
    "AN": (PS(1.0e6, [], []), PS(1.0e6, [3.0e6], [10.0])),
    "NP": (PS(0.5e6, [0.3e6, 0.2e6], [4, 60]), PS(1.0e6, [2.0e6, 1.0e6], [10, 150])),
    "D": (PS(0.6e6, [0.8e6], [5.0]), PS(1.5e6, [2.0e6], [2.0])),
    "Z": (PS(0.5e6, [1.0e6, 0.0], [2, 7]), PS(2.0e6, [0.0, 1.0e6], [1, 9])),
}
_nu = 0.3
_G = lambda E: E / (2 * (1 + _nu))                    # noqa: E731
_K = lambda E: E / (3 * (1 - 2 * _nu))                # noqa: E731
SETS["P"] = (PS(_G(1e6), [_G(2e6)], [10.0]), PS(_K(1e6), [_K(2e6)], [10.0]))


def model(name, shift=None):
    sh, bk = SETS[name]
    return V.GeneralizedMaxwell(sh, bk, shift)


def vstress(sig):
    """(k, 3, 3) → (k, 6) Voigt, physical shear."""
    return sig[:, VA, VB]


def ten(ev):
    """(k, 6) engineering-shear Voigt strain → (k, 3, 3)."""
    return np.array([T.voigt_to_strain(e) for e in ev])


def run(m, eps_nodes, times, temps=None, T0=None, control=None, stress=None, tol=1e-9):
    k = len(times)
    ctrl = [True] * 6 if control is None else control
    st = np.zeros((k, 6)) if stress is None else stress
    return drive_material_point(m, eps_nodes, st, ctrl, tol=tol, times=times, temperatures=temps, temperature0=T0)


# ================================================================================================= oracles
def _devtr(e):
    tr = np.trace(e, axis1=-2, axis2=-1)
    return e - (tr / 3.0)[..., None, None] * np.eye(3), tr


def oracle_stress(shear, bulk, times, eps, form="strain", rtol=1e-12, aT=None):
    """Stress (k, 3, 3) at the nodes of a strain path that is piecewise linear in t, from rest at t = 0 with zero strain.

    form "strain": viscous-strain ODE  e_v' = (e − e_v)/(τ a_T(t)),  σ = 2G∞ e + Σ 2G_i (e − e_v,i) + ...
    form "stress": branch-stress ODE   s' = −s/(τ a_T) + 2G_i e',  (jumps: s += 2G_i Δe).
    ``aT`` (optional) = (t_a, t_b, T_a, T_b, fn) data: fn(T) → a_T, T linear per segment (else a_T = 1)."""
    Gi, ts, Kj, tk = shear.amplitudes, shear.taus, bulk.amplitudes, bulk.taus
    ms, mk = len(Gi), len(Kj)
    k = len(times)
    out = np.zeros((k, 3, 3))
    ts_all = np.concatenate([[0.0], times])
    e_nodes = np.concatenate([np.zeros((1, 3, 3)), eps])
    y = np.zeros(ms * 9 + mk)
    scale = max(float(np.abs(eps).max()), 1e-30)
    for n in range(k):
        t_a, t_b = ts_all[n], ts_all[n + 1]
        e_a, th_a = _devtr(e_nodes[n])
        e_b, th_b = _devtr(e_nodes[n + 1])
        dt = t_b - t_a
        if dt == 0.0:
            if form == "stress":
                y[:ms * 9] += (2.0 * Gi[:, None, None] * (e_b - e_a)[None]).reshape(-1)
                y[ms * 9:] += Kj * (th_b - th_a)
        else:
            de, dth = (e_b - e_a) / dt, (th_b - th_a) / dt
            if aT is None:
                a_of = lambda u: 1.0                                  # noqa: E731
            else:
                Ta, Tb, fn = aT[0][n], aT[1][n], aT[2]
                a_of = lambda u, Ta=Ta, Tb=Tb, fn=fn: fn(Ta + (Tb - Ta) * u / dt)   # noqa: E731
            tau_vec = np.concatenate([np.repeat(ts, 9), tk])

            def rhs(u, yy):
                a = a_of(u)
                if form == "strain":
                    e = e_a + de * u
                    th = th_a + dth * u
                    tgt = np.concatenate([np.tile(e.reshape(-1), ms), np.full(mk, th)])
                    return (tgt - yy) / (tau_vec * a)
                src = np.concatenate([np.repeat(2.0 * Gi, 9) * np.tile(de.reshape(-1), ms), Kj * dth])
                return -yy / (tau_vec * a) + src

            def jac(u, yy):
                return np.diag(-1.0 / (tau_vec * a_of(u)))

            sol = solve_ivp(rhs, (0.0, dt), y, method="Radau", jac=jac, rtol=rtol, atol=1e-18 * scale * 1e6 * 1e-6,
                            first_step=min(dt, float(tau_vec.min())) * 1e-3)
            assert sol.success
            y = sol.y[:, -1]
        e, th = e_b, th_b
        v = y[:ms * 9].reshape(ms, 3, 3)
        w = y[ms * 9:]
        if form == "strain":
            sig = 2.0 * shear.inf * e + sum((2.0 * Gi[i] * (e - v[i]) for i in range(ms)), np.zeros((3, 3)))
            pr = bulk.inf * th + float(np.sum(Kj * (th - w)))
        else:
            sig = 2.0 * shear.inf * e + v.sum(0)
            pr = bulk.inf * th + float(w.sum())
        out[n] = sig + pr * np.eye(3)
    return out


def oracle_expm(shear, bulk, times, eps):
    """Van Loan matrix-exponential oracle (stress form, constant τ), nodes (k, 3, 3)."""
    Gi, ts, Kj, tk = shear.amplitudes, shear.taus, bulk.amplitudes, bulk.taus
    k = len(times)
    out = np.zeros((k, 3, 3))
    ts_all = np.concatenate([[0.0], times])
    e_nodes = np.concatenate([np.zeros((1, 3, 3)), eps])
    s = np.zeros((len(Gi), 3, 3))
    p = np.zeros(len(Kj))

    def adv(x0, tau, c, dt):                                # x' = −x/τ + c over dt, via expm of the augmented matrix
        E = expm(np.array([[-dt / tau, c * dt], [0.0, 0.0]]))
        return E[0, 0] * x0 + E[0, 1]

    for n in range(k):
        dt = ts_all[n + 1] - ts_all[n]
        ea, tha = _devtr(e_nodes[n])
        eb, thb = _devtr(e_nodes[n + 1])
        if dt == 0.0:
            s = s + 2.0 * Gi[:, None, None] * (eb - ea)[None]
            p = p + Kj * (thb - tha)
        else:
            for i in range(len(Gi)):
                for a in range(3):
                    for b in range(3):
                        s[i, a, b] = adv(s[i, a, b], ts[i], 2.0 * Gi[i] * (eb - ea)[a, b] / dt, dt)
            for j in range(len(Kj)):
                p[j] = adv(p[j], tk[j], Kj[j] * (thb - tha) / dt, dt)
        out[n] = 2.0 * shear.inf * eb + s.sum(0) + (bulk.inf * thb + p.sum()) * np.eye(3)
    return out


def oracle_energy(shear, bulk, times, eps, rtol=1e-12):
    """Stress-form ODE with accumulators: returns (ψ_end, ∫D dt, ∫σ:ε̇ dt) for a continuous piecewise-linear path from rest
    (no instantaneous jumps)."""
    Gi, ts, Kj, tk = shear.amplitudes, shear.taus, bulk.amplitudes, bulk.taus
    ms, mk = len(Gi), len(Kj)
    ts_all = np.concatenate([[0.0], times])
    e_nodes = np.concatenate([np.zeros((1, 3, 3)), eps])
    y = np.zeros(ms * 9 + mk + 2)
    scale = float(np.abs(eps).max()) * float(shear.instantaneous)
    tau_vec = np.concatenate([np.repeat(ts, 9), tk])
    G9 = np.repeat(Gi, 9)
    for n in range(len(times)):
        dt = ts_all[n + 1] - ts_all[n]
        assert dt > 0.0
        ea, tha = _devtr(e_nodes[n])
        eb, thb = _devtr(e_nodes[n + 1])
        de, dth = (eb - ea) / dt, (thb - tha) / dt

        def rhs(u, yy):
            s = yy[:ms * 9].reshape(ms, 3, 3)
            p = yy[ms * 9:ms * 9 + mk]
            e = ea + de * u
            th = tha + dth * u
            src = np.concatenate([G9 * 2.0 * np.tile(de.reshape(-1), ms), Kj * dth])
            d = -yy[:-2] / tau_vec + src
            sig_dev_work = 2.0 * shear.inf * np.sum(e * de) + np.sum(s * de[None])
            work = sig_dev_work + (bulk.inf * th + p.sum()) * dth
            with np.errstate(divide="ignore", invalid="ignore"):
                dis_s = np.sum(np.where(G9 > 0, (yy[:ms * 9] ** 2) / np.where(G9 > 0, 2.0 * G9 * np.repeat(ts, 9), 1.0), 0.0))
                dis_k = np.sum(np.where(Kj > 0, p ** 2 / np.where(Kj > 0, Kj * tk, 1.0), 0.0))
            return np.concatenate([d, [dis_s + dis_k, work]])

        jd = np.concatenate([-1.0 / tau_vec, [0.0, 0.0]])
        sol = solve_ivp(rhs, (0.0, dt), y, method="Radau", jac=lambda u, yy: np.diag(jd), rtol=rtol,
                        atol=1e-14 * max(scale, 1e-30) * 1e-3)
        assert sol.success
        y = sol.y[:, -1]
    e, th = _devtr(e_nodes[-1])
    s = y[:ms * 9].reshape(ms, 3, 3)
    p = y[ms * 9:ms * 9 + mk]
    psi = shear.inf * np.sum(e * e) + 0.5 * bulk.inf * th ** 2
    psi += sum((np.sum(s[i] * s[i]) / (4 * Gi[i]) for i in range(ms) if Gi[i] > 0), 0.0)
    psi += sum((p[j] ** 2 / (2 * Kj[j]) for j in range(mk) if Kj[j] > 0), 0.0)
    return float(psi), float(y[-2]), float(y[-1])


def uniax_oracle(shear, bulk, mode, amp, t_eval, rtol=1e-12):
    """DAE oracle of uniaxial stress (σ_yy = σ_zz = 0): returns σ_xx(t) (mode "relax", ε_xx = amp for t > 0) or ε_xx(t)
    (mode "creep", σ_xx = amp). States (e_xx, θ, s_i, p_j); constraint σ̇_yy = 0 differentiated (and σ̇_xx = 0 for creep);
    initial values from the instantaneous response."""
    Gi, ts, Kj, tk = shear.amplitudes, shear.taus, bulk.amplitudes, bulk.taus
    G0, K0 = shear.instantaneous, bulk.instantaneous
    ms, mk = len(Gi), len(Kj)
    if mode == "relax":
        th0 = 3 * G0 * amp / (G0 + 3 * K0)
        e0 = 3 * K0 * amp / (G0 + 3 * K0)
    else:
        e0, th0 = amp / (3 * G0), amp / (3 * K0)
    y0 = np.concatenate([[e0, th0], 2 * Gi * e0, Kj * th0])

    def rhs(t, y):
        s, p = y[2:2 + ms], y[2 + ms:]
        S1 = np.sum(s / ts) if ms else 0.0
        P1 = np.sum(p / tk) if mk else 0.0
        R_ = P1 - 0.5 * S1
        if mode == "relax":
            de = -R_ / (G0 + 3 * K0)
            dth = -3 * de
        else:
            de = S1 / (2 * G0)
            dth = P1 / K0
        return np.concatenate([[de, dth], -s / ts + 2 * Gi * de, -p / tk + Kj * dth])

    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), y0, method="Radau", t_eval=t_eval, rtol=rtol,
                    atol=1e-15 * abs(amp) * 1e-3 * max(G0, K0) if mode == "relax" else 1e-18)
    assert sol.success
    e, th, s, p = sol.y[0], sol.y[1], sol.y[2:2 + ms], sol.y[2 + ms:]
    if mode == "relax":
        return 2 * shear.inf * e + s.sum(0) + bulk.inf * th + p.sum(0)
    return e + th / 3.0


def creep_compliance_residues(series, t):
    """Shear (or bulk) creep compliance of a Prony series with M∞ > 0 from the residues of Q(s)/(s N(s)), Q = Π(1 + sτ_i),
    N = M∞ Q + Σ M_i τ_i s Π_{j≠i}(1 + sτ_j)  (Laplace domain; numpy.roots)."""
    taus, A = series.taus, series.amplitudes
    Q = np.poly1d([1.0])
    for tau in taus:
        Q = Q * np.poly1d([tau, 1.0])
    N = series.inf * Q
    for i, tau in enumerate(taus):
        term = np.poly1d([A[i] * tau, 0.0])
        for j, tj in enumerate(taus):
            if j != i:
                term = term * np.poly1d([tj, 1.0])
        N = N + term
    roots = np.roots(N.coeffs)
    assert np.all(np.abs(roots.imag) <= 1e-9 * np.abs(roots.real))
    roots = roots.real
    dN = N.deriv()
    J = np.full_like(t, Q(0.0) / N(0.0), dtype=float)
    for sk in roots:
        J = J + np.exp(sk * t) * Q(sk) / (sk * dN(sk))
    return J


def log_times(m, lo=1e-3, hi=1e3):
    n = int(round(m * math.log10(hi / lo))) + 1
    return np.concatenate([[0.0], np.geomspace(lo, hi, n)])


def rec(label, value):
    """Print a measured value (shown with ``pytest -rP``) for the first-result record."""
    print(f"MEAS {label} = {value!r}")


def order(e_coarse, e_fine):
    return math.log2(e_coarse / e_fine)


# ================================================================================================= V1 scalar
SCAL = dict(maxwell=(1e6, 1e7), kv=(1e6, 1e7), sls=(1e6, 2e6, 10.0), burgers=(2e6, 4e7, 1e6, 5e6))


_V1A = []


def _chk(a, ref):
    e = float(np.max(np.abs(a - ref)) / np.max(ref))
    _V1A.append(e)
    assert e <= 1e-9, e


def _ode(f, y0, t_eval):
    sol = solve_ivp(f, (0.0, float(t_eval[-1])), y0, method="DOP853", t_eval=t_eval, rtol=1e-13, atol=1e-30)
    assert sol.success
    return sol.y


def test_V1a_scalar_curves_against_element_odes():
    s0 = 1.0e3
    e0 = 1.0e-3
    E, eta = SCAL["maxwell"]
    t = np.linspace(0.0, 50 * eta / E, 201)
    # Maxwell: relaxation σ' = −(E/η)σ, σ(0+) = E ε0 ; creep ε' = σ0/η, ε(0+) = σ0/E
    ref = _ode(lambda t_, y: [-(E / eta) * y[0]], [E * e0], t)[0]
    _chk(V.maxwell_relaxation(t, E, eta) * e0, ref)
    ref = _ode(lambda t_, y: [s0 / eta], [s0 / E], t)[0]
    _chk(V.maxwell_creep(t, E, eta) * s0, ref)
    # Kelvin–Voigt creep: η ε' + E ε = σ0
    E, eta = SCAL["kv"]
    ref = _ode(lambda t_, y: [(s0 - E * y[0]) / eta], [0.0], t)[0]
    _chk(V.kelvin_voigt_creep(t, E, eta) * s0, ref)
    # SLS: σ = E∞ ε + σ1, σ1' = E1 ε' − σ1/τ.  Relaxation: ε' = 0, σ1(0+) = E1 ε0. Creep: ε' = (σ0 − E∞ ε)/(τ E0)
    Ei, E1, tau = SCAL["sls"]
    t = np.linspace(0.0, 10 * tau * (Ei + E1) / Ei, 201)
    ref = _ode(lambda t_, y: [-y[0] / tau], [E1 * e0], t)[0] + Ei * e0
    _chk(V.sls_relaxation(t, Ei, E1, tau) * e0, ref)
    ref = _ode(lambda t_, y: [(s0 - Ei * y[0]) / (tau * (Ei + E1))], [s0 / (Ei + E1)], t)[0]
    _chk(V.sls_creep(t, Ei, E1, tau) * s0, ref)
    # Burgers: ε = ε_M + ε_K. Creep: ε_M' = σ0/η_M, ε_K' = (σ0 − E_K ε_K)/η_K. Relaxation: states (σ, ε_K), ε' = 0
    EM, etaM, EK, etaK = SCAL["burgers"]
    t = np.linspace(0.0, 30 * etaM / EM, 301)
    y = _ode(lambda t_, y: [s0 / etaM, (s0 - EK * y[1]) / etaK], [s0 / EM, 0.0], t)
    ref = y[0] + y[1]
    _chk(V.burgers_creep(t, EM, etaM, EK, etaK) * s0, ref)

    def rel(t_, y):
        flow = (y[0] - EK * y[1]) / etaK
        return [-EM * (y[0] / etaM + flow), flow]

    ref = _ode(rel, [EM * e0, 0.0], t)[0]
    _chk(V.burgers_relaxation(t, EM, etaM, EK, etaK) * e0, ref)
    rec("V1a max rel err over 8 curves", max(_V1A))


def test_V1b_special_values_and_refusals():
    E, eta = SCAL["maxwell"]
    assert V.maxwell_relaxation(eta / E, E, eta) == pytest.approx(E / math.e, rel=1e-12)
    assert V.maxwell_creep(0.0, E, eta) == pytest.approx(1 / E, rel=1e-12)
    assert V.maxwell_creep(eta / E, E, eta) == pytest.approx(1 / E + 1 / E, rel=1e-12)   # t = η/E: t/η = 1/E
    Ei, E1, tau = SCAL["sls"]
    E0 = Ei + E1
    assert V.sls_relaxation(0.0, Ei, E1, tau) == pytest.approx(E0, rel=1e-12)
    assert V.sls_relaxation(1e4, Ei, E1, tau) == pytest.approx(Ei, rel=1e-12)
    assert V.sls_creep(0.0, Ei, E1, tau) == pytest.approx(1 / E0, rel=1e-12)
    assert V.sls_creep(1e5, Ei, E1, tau) == pytest.approx(1 / Ei, rel=1e-12)
    assert V.sls_creep_time(Ei, E1, tau) == pytest.approx(tau * E0 / Ei, rel=1e-14)
    EM, etaM, EK, etaK = SCAL["burgers"]
    assert V.burgers_creep(0.0, EM, etaM, EK, etaK) == pytest.approx(1 / EM, rel=1e-12)
    t1, t2 = 1e6, 2e6
    slope = (V.burgers_creep(t2, EM, etaM, EK, etaK) - V.burgers_creep(t1, EM, etaM, EK, etaK)) / (t2 - t1)
    assert slope == pytest.approx(1 / etaM, rel=1e-12)
    assert V.burgers_relaxation(0.0, EM, etaM, EK, etaK) == pytest.approx(EM, rel=1e-12)
    assert V.burgers_relaxation(1e5, EM, etaM, EK, etaK) < 1e-12 * EM
    EKv, etaKv = SCAL["kv"]
    assert V.kelvin_voigt_creep(0.0, EKv, etaKv) == 0.0
    assert V.kelvin_voigt_creep(1e3 * etaKv / EKv, EKv, etaKv) == pytest.approx(1 / EKv, rel=1e-12)
    with pytest.raises(ValueError, match="impuls"):
        V.kelvin_voigt_relaxation(1.0, EKv, etaKv)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            V.maxwell_relaxation(1.0, bad, 1.0)
    with pytest.raises(ValueError):
        V.sls_creep(-1.0, 1.0, 1.0, 1.0)
    # dimension check: J [1/Pa] and E_R [Pa]: scaling the moduli by c scales E_R by c and J by 1/c
    c = 7.0
    assert V.sls_relaxation(3.0, c * Ei, c * E1, tau) == pytest.approx(c * V.sls_relaxation(3.0, Ei, E1, tau), rel=1e-14)
    assert V.sls_creep(3.0, c * Ei, c * E1, tau) == pytest.approx(V.sls_creep(3.0, Ei, E1, tau) / c, rel=1e-14)


def test_V1c_laplace_identity():
    models = {
        "maxwell": (lambda t: V.maxwell_relaxation(t, *SCAL["maxwell"]), lambda t: V.maxwell_creep(t, *SCAL["maxwell"])),
        "sls": (lambda t: V.sls_relaxation(t, *SCAL["sls"]), lambda t: V.sls_creep(t, *SCAL["sls"])),
        "burgers": (lambda t: V.burgers_relaxation(t, *SCAL["burgers"]), lambda t: V.burgers_creep(t, *SCAL["burgers"])),
    }
    for name, (ER, J) in models.items():
        for s in (0.05, 0.3, 2.0):
            Eh = quad(lambda t: math.exp(-s * t) * float(ER(t)), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
            Jh = quad(lambda t: math.exp(-s * t) * float(J(t)), 0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
            assert s * s * Eh * Jh == pytest.approx(1.0, rel=1e-7), (name, s)
    # J is not 1/E_R
    t = 10.0
    assert abs(V.sls_creep(t, *SCAL["sls"]) * V.sls_relaxation(t, *SCAL["sls"]) - 1.0) > 0.05


def test_V1d_unloading_and_recovery():
    s0, tu = 1.0e3, 40.0
    Ei, E1, tau = SCAL["sls"]
    J = lambda t: V.sls_creep(t, Ei, E1, tau)                    # noqa: E731
    eps_u = V.recovery_strain(J, np.array([tu, tu + 3000.0]), s0, tu)
    assert abs(eps_u[1]) <= 1e-12 * s0 / Ei
    E0 = Ei + E1
    tc = tau * E0 / Ei
    peak = V.recovery_strain(J, tu - 1e-12, s0, tu)
    assert float(peak) == pytest.approx(s0 * (1 / Ei - (1 / Ei - 1 / E0) * math.exp(-tu / tc)), rel=1e-9)
    E, eta = 1e6, 1e7
    Jm = lambda t: V.maxwell_creep(t, E, eta)                    # noqa: E731
    r = V.recovery_strain(Jm, np.array([tu + 1.0, tu + 1e3, tu + 1e6]), s0, tu)
    assert np.allclose(r, s0 * tu / eta, rtol=1e-12)
    EM, etaM, EK, etaK = SCAL["burgers"]
    Jb = lambda t: V.burgers_creep(t, EM, etaM, EK, etaK)        # noqa: E731
    rb = V.recovery_strain(Jb, np.array([tu + 1e5]), s0, tu)
    assert rb[0] == pytest.approx(s0 * tu / etaM, rel=1e-12)
    # the recoverable (SLS) and the permanent (Maxwell) parts are different things
    assert abs(r[0]) > 1e-3 * s0 / E and abs(eps_u[1]) < 1e-12 * s0 / Ei


# ================================================================================================= V2
def _exact_relax(series, t):
    return series.inf + sum(a * math.exp(-t / ta) for a, ta in zip(series.amplitudes, series.taus))


def test_V2_pure_shear_and_volume_relaxation():
    sh, bk = SETS["A"]
    m = model("A")
    times = np.array([0.0, 0.05, 0.4, 1.0, 1.7, 5.0, 12.0, 33.0, 90.0, 300.0, 1000.0, 2000.0])
    g0, th0 = 2.0e-3, 1.5e-3
    eps = np.zeros((len(times), 6))
    eps[:, 5] = g0                                           # engineering γ_xy
    h = run(m, eps, times)
    rec("V2 shear max rel err", max(abs(h.stress[k, 5] / (_exact_relax(sh, t) * g0) - 1) for k, t in enumerate(times)))
    for k, t in enumerate(times):
        assert h.stress[k, 5] == pytest.approx(_exact_relax(sh, t) * g0, rel=1e-10)
        assert np.max(np.abs(h.stress[k, :5])) <= 1e-10 * _exact_relax(sh, 0.0) * g0
    eps = np.zeros((len(times), 6))
    eps[:, :3] = th0 / 3.0
    h = run(m, eps, times)
    rec("V2 volume max rel err", max(abs(h.stress[k, c] / (_exact_relax(bk, t) * th0) - 1)
                                     for k, t in enumerate(times) for c in range(3)))
    for k, t in enumerate(times):
        for c in range(3):
            assert h.stress[k, c] == pytest.approx(_exact_relax(bk, t) * th0, rel=1e-10)
        assert np.max(np.abs(h.stress[k, 3:])) <= 1e-10 * _exact_relax(bk, 0.0) * th0


# ================================================================================================= V3
V3_DT = [1e-14, 3e-13, 1e-9, 2.5e-6, 1e-4, 7.3e-3, 0.4, 1.0, 12.5, 100.0, 100.0, 100.0, 3.3, 0.1, 1e-3, 40.0, 1e-9,
         5.0, 100.0, 2.0]
V3_LAM = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 0.2, 0.0, -0.2, -0.5, -0.5, -0.5, 0.5, 0.5, 0.0]
V3_E0 = np.array([1.0e-3, -0.4e-3, 0.7e-3, 0.3e-3, -0.5e-3, 0.8e-3])


def _v3_path():
    times = np.cumsum(V3_DT)
    return times, np.array(V3_LAM)[:, None] * V3_E0[None]


def test_V3_irregular_ramp_hold_unload_against_hereditary_oracle():
    sh, bk = SETS["W"]
    times, ev = _v3_path()
    eps = ten(ev)
    # oracle control: three independent evaluations of the same hereditary response
    ref = oracle_stress(sh, bk, times, eps, "strain")
    ref_b = oracle_stress(sh, bk, times, eps, "stress")
    ref_c = oracle_expm(sh, bk, times, eps)
    scale = float(np.max(np.abs(ref)))
    assert np.max(np.abs(ref - ref_b)) <= 1e-9 * scale
    assert np.max(np.abs(ref - ref_c)) <= 1e-9 * scale
    h = run(model("W"), ev, times)
    err = np.max(np.abs(vstress(ref) - h.stress))
    rec("V3 prod-vs-oracle err/scale", float(err / scale))
    rec("V3 oracle control strain-vs-stress, strain-vs-expm (rel)", (float(np.max(np.abs(ref - ref_b)) / scale),
                                                                      float(np.max(np.abs(ref - ref_c)) / scale)))
    assert err <= 1e-8 * scale, (err / scale)
    assert np.all(np.isfinite(h.stress)) and np.all(np.isfinite(h.energy)) and np.all(h.dissipation >= -1e-12 * h.energy.max())


# ================================================================================================= V4
def _prod_uniax(m, mode, times, amp, unload_at=None):
    k = len(times)
    strain = np.zeros((k, 6))
    stress = np.zeros((k, 6))
    if mode == "relax":
        ctrl = [True, False, False, True, True, True]
        strain[:, 0] = amp
    else:
        ctrl = [False, False, False, True, True, True]
        stress[:, 0] = amp
        if unload_at is not None:
            stress[np.arange(k) > unload_at, 0] = 0.0
    h = run(m, strain, times, control=ctrl, stress=stress, tol=1e-9)
    return h


def _closed_AN():
    sh, bk = SETS["AN"]
    G, K1, Kinf, tau = sh.inf, bk.amplitudes[0], bk.inf, bk.taus[0]
    K0 = Kinf + K1
    Einf = 9 * Kinf * G / (3 * Kinf + G)
    E0 = 9 * K0 * G / (3 * K0 + G)
    taup = tau * (3 * K0 + G) / (3 * Kinf + G)
    return Einf, E0, taup


def test_V4a_analytic_uniaxial_relaxation_and_creep_convergence():
    Einf, E0, taup = _closed_AN()
    tc = taup * E0 / Einf
    eps0, sig0 = 1e-3, 1e3
    errs_r, errs_c = {}, {}
    for mm in (25, 50, 100):
        t = log_times(mm)
        h = _prod_uniax(model("AN"), "relax", t, eps0)
        ref = (Einf + (E0 - Einf) * np.exp(-t / taup)) * eps0
        errs_r[mm] = np.max(np.abs(h.stress[:, 0] - ref)) / np.max(ref)
        assert np.max(np.abs(h.stress[:, 1:3])) <= 1e-8 * np.max(ref)
        h = _prod_uniax(model("AN"), "creep", t, sig0)
        refc = sig0 * (1 / Einf - (1 / Einf - 1 / E0) * np.exp(-t / tc))
        errs_c[mm] = np.max(np.abs(h.strain[:, 0] - refc)) / np.max(refc)
        assert np.max(np.abs(h.stress[:, 1:3])) <= 1e-8 * sig0
    rec("V4a relax errs", errs_r)
    rec("V4a creep errs", errs_c)
    for errs in (errs_r, errs_c):
        assert errs[100] <= 5e-4, errs
        assert order(errs[50], errs[100]) >= 1.6, errs


def test_V4b_nonproportional_against_dae_and_residue_oracles():
    sh, bk = SETS["NP"]
    eps0, sig0 = 1e-3, 1e3
    # oracle control: creep by the DAE oracle against the residue (Laplace) oracle ε = σ0 (J_G/3 + J_K/9)
    t = log_times(25)[1:]
    cr = uniax_oracle(sh, bk, "creep", sig0, t)
    cres = sig0 * (creep_compliance_residues(sh, t) / 3.0 + creep_compliance_residues(bk, t) / 9.0)
    rec("V4b oracle control DAE-vs-residue rel", float(np.max(np.abs(cr - cres)) / np.max(cres)))
    assert np.max(np.abs(cr - cres)) <= 1e-8 * np.max(cres)
    errs_r, errs_c = {}, {}
    for mm in (25, 50, 100):
        t = log_times(mm)
        h = _prod_uniax(model("NP"), "relax", t, eps0)
        ref = uniax_oracle(sh, bk, "relax", eps0, t)
        errs_r[mm] = np.max(np.abs(h.stress[:, 0] - ref)) / np.max(ref)
        h = _prod_uniax(model("NP"), "creep", t, sig0)
        ref = uniax_oracle(sh, bk, "creep", sig0, t)
        errs_c[mm] = np.max(np.abs(h.strain[:, 0] - ref)) / np.max(ref)
    rec("V4b relax errs", errs_r)
    rec("V4b creep errs", errs_c)
    for errs in (errs_r, errs_c):
        assert errs[100] <= 5e-4, errs
        assert order(errs[50], errs[100]) >= 1.6, errs


def test_V4c_lateral_stress_and_independent_reevaluation():
    sh, bk = SETS["NP"]
    t = log_times(50)
    h = _prod_uniax(model("NP"), "creep", t, 1e3)
    assert np.max(np.abs(h.stress[:, 1:3])) <= 1e-8 * 1e3
    assert np.max(np.abs(h.stress[:, 0] - 1e3)) <= 1e-8 * 1e3
    ref = oracle_stress(sh, bk, t, ten(h.strain), "strain")
    rec("V4c creep: max |sigma_trans| prod, oracle re-eval err, oracle sigma_trans (Pa)",
        (float(np.max(np.abs(h.stress[:, 1:3]))), float(np.max(np.abs(vstress(ref) - h.stress))),
         float(np.max(np.abs(vstress(ref)[:, 1:3])))))
    assert np.max(np.abs(vstress(ref) - h.stress)) <= 1e-8 * 1e3
    assert np.max(np.abs(vstress(ref)[:, 1:3])) <= 1e-8 * 1e3                 # the oracle also finds σ_trans ≈ 0
    t = log_times(50)
    h = _prod_uniax(model("NP"), "relax", t, 1e-3)
    ref = oracle_stress(sh, bk, t, ten(h.strain), "strain")
    scale = np.max(np.abs(h.stress))
    rec("V4c relax re-eval err/scale", float(np.max(np.abs(vstress(ref) - h.stress)) / scale))
    assert np.max(np.abs(vstress(ref) - h.stress)) <= 1e-8 * scale


def test_V4d_negative_control_pointwise_formula_is_not_the_uniaxial_modulus():
    Einf, E0, taup = _closed_AN()
    sh, bk = SETS["AN"]
    t = np.geomspace(1.0, 100.0, 400)
    Er = Einf + (E0 - Einf) * np.exp(-t / taup)
    G = sh.relaxation(t)
    K = bk.relaxation(t)
    Eptw = 9 * K * G / (3 * K + G)
    rec("V4d AN pointwise-vs-true max rel diff", float(np.max(np.abs(Eptw - Er) / Er)))
    assert np.max(np.abs(Eptw - Er) / Er) >= 3e-2                              # not the uniaxial modulus
    # proportional spectra: constant ν, the pointwise formula coincides
    shp, bkp = SETS["P"]
    tt = np.geomspace(1e-2, 1e3, 300)
    Gp, Kp = shp.relaxation(tt), bkp.relaxation(tt)
    Ept = 9 * Kp * Gp / (3 * Kp + Gp)
    Erp = 1e6 + 2e6 * np.exp(-tt / 10.0)
    rec("V4d P pointwise-vs-true max rel diff", float(np.max(np.abs(Ept - Erp) / Erp)))
    assert np.max(np.abs(Ept - Erp) / Erp) <= 1e-12
    # lateral strain ratio of the production relaxation run
    t = log_times(25)
    h = _prod_uniax(model("AN"), "relax", t, 1e-3)
    ratio = -h.strain[:, 1] / h.strain[:, 0]
    rec("V4d AN lateral ratio spread", float((ratio.max() - ratio.min()) / ratio.max()))
    assert (ratio.max() - ratio.min()) / ratio.max() >= 5e-2
    t = log_times(25)
    h = _prod_uniax(model("P"), "relax", t, 1e-3)
    ratio = -h.strain[:, 1] / h.strain[:, 0]
    rec("V4d P lateral ratio dev from 0.3", float(np.max(np.abs(ratio - 0.3))))
    assert np.max(np.abs(ratio - 0.3)) <= 1e-9


# ================================================================================================= V5
def _state_after(m, n_steps=6, shift=False):
    times = np.array([0.5, 1.5, 4.0, 9.0, 20.0, 45.0, 90.0])[:n_steps]
    ev = np.array([0.2, 0.5, 1.0, 0.7, -0.2, 0.4, 0.4])[:n_steps, None] * np.array([1e-3, -4e-4, 7e-4, 3e-4, -5e-4, 8e-4])
    if shift:
        temps = np.linspace(310.0, 340.0, n_steps)
        return run(m, ev, times, temps=temps, T0=300.0), times, ev, temps
    return run(m, ev, times), times, ev, None


@pytest.mark.parametrize("with_shift", [False, True])
def test_V5_tangent_against_central_differences(with_shift):
    shift = V.ArrheniusShift(320.0, 6e4) if with_shift else None
    m = model("A", shift)
    h, times, ev, temps = _state_after(m, 6, with_shift)
    committed = h.states[-1]
    e0 = h.strain[-1].copy()
    dt = 7.0
    tstep = TimeStep(times[-1], dt, 345.0 if with_shift else None, temps[-1] if with_shift else None)
    req = MaterialRequest(step=tstep, energy=True)

    def stress(e):
        return m.evaluate(Kinematics(T.voigt_to_strain(e)[None], "small"), None, committed, req).stress[0]

    r0 = m.evaluate(Kinematics(T.voigt_to_strain(e0)[None], "small"), None, committed, req)
    C = r0.tangent[0]
    d = np.array([3e-4, -1e-4, 2e-4, 4e-4, -2e-4, 1e-4])
    for hstep in (1e-3, 1e-4, 1e-5, 1e-6):
        Cfd = np.zeros((6, 6))
        for j in range(6):
            ej = np.zeros(6)
            ej[j] = hstep
            Cfd[:, j] = (stress(e0 + ej) - stress(e0 - ej)) / (2 * hstep)
        rec(f"V5 shift={with_shift} h={hstep} FD err/|C|", float(np.max(np.abs(Cfd - C)) / np.max(np.abs(C))))
        assert np.max(np.abs(Cfd - C)) <= 1e-6 * np.max(np.abs(C)), hstep
        rem = stress(e0 + hstep * d / 1e-4) - stress(e0) - C @ (hstep * d / 1e-4)
        assert np.max(np.abs(rem)) <= 1e-12 * np.max(np.abs(stress(e0)))
    # symmetry of the algorithmic tangent
    assert np.max(np.abs(C - C.T)) <= 1e-12 * np.max(np.abs(C))


def test_V5_instantaneous_tangent_and_newton_convergence():
    m = model("A")
    sh, bk = SETS["A"]
    G0, K0 = sh.instantaneous, bk.instantaneous
    Ci = np.zeros((6, 6))
    Ci[:3, :3] = K0 - 2 * G0 / 3
    Ci[0, 0] = Ci[1, 1] = Ci[2, 2] = K0 + 4 * G0 / 3
    Ci[3, 3] = Ci[4, 4] = Ci[5, 5] = G0
    req = MaterialRequest(step=TimeStep(0.0, 0.0))
    r = m.evaluate(Kinematics(np.zeros((1, 3, 3)), "small"), None, StateStore(m, 1).committed, req)
    assert np.max(np.abs(r.tangent[0] - Ci)) <= 1e-12 * G0
    assert np.max(np.abs(m.instantaneous_tangent() - Ci)) <= 1e-12 * G0
    req = MaterialRequest(step=TimeStep(0.0, 1e-12), tangent_kind="algorithmic")
    r = m.evaluate(Kinematics(np.zeros((1, 3, 3)), "small"), None, StateStore(m, 1).committed, req)
    assert np.max(np.abs(r.tangent[0] - Ci)) <= 1e-9 * G0
    # mixed control path: one Newton update per step with the algorithmic tangent
    t = log_times(10, 1e-2, 1e2)
    h = _prod_uniax(model("NP"), "creep", t, 1e3)
    rec("V5 max Newton iterations per step", max(h.iterations))
    assert max(h.iterations) <= 1


# ================================================================================================= V6
def _v6_path(seed=20261127, k=24):
    rng = np.random.default_rng(seed)
    dts = rng.uniform(0.5, 6.0, k)
    times = np.cumsum(dts)
    ev = rng.uniform(-1e-3, 1e-3, (k, 6))
    return times, ev


def _refine(times, ev, m):
    t0 = np.concatenate([[0.0], times])
    e0 = np.concatenate([np.zeros((1, 6)), ev])
    tt, ee = [], []
    for i in range(len(times)):
        for j in range(1, m + 1):
            f = j / m
            tt.append(t0[i] + f * (t0[i + 1] - t0[i]))
            ee.append(e0[i] + f * (e0[i + 1] - e0[i]))
    tt[m - 1::m] = list(times)
    ee = np.array(ee)
    ee[m - 1::m] = ev
    return np.array(tt), ee


@pytest.mark.parametrize("name", ["A", "Z"])
def test_V6_energy_and_dissipation_against_augmented_oracle_and_closure(name):
    sh, bk = SETS[name]
    times, ev = _v6_path()
    psi_o, D_o, W_o = oracle_energy(sh, bk, times, ten(ev))
    assert W_o == pytest.approx(psi_o + D_o, rel=1e-8)                       # the oracle closes (formulas of ψ and D)
    assert D_o > 0
    resid = {}
    for mm in (4, 8, 16, 32, 64):
        tt, ee = _refine(times, ev, mm)
        h = run(model(name), ee, tt)
        psi, D = h.energy[-1], h.dissipation.sum()
        assert psi == pytest.approx(psi_o, rel=1e-8) and pytest.approx(D_o, rel=1e-8) == D, mm
        assert np.all(np.isfinite(h.energy)) and np.all(np.isfinite(h.dissipation))
        assert np.all(h.energy >= 0.0) and np.all(h.dissipation >= -1e-12 * h.energy.max())
        sig = np.vstack([np.zeros((1, 6)), h.stress])
        eps = np.vstack([np.zeros((1, 6)), ee])
        Wt = float(np.sum(0.5 * (sig[1:] + sig[:-1]) * np.diff(eps, axis=0)))
        resid[mm] = abs(Wt - (psi + D)) / (psi + D)
        rec(f"V6 {name} m={mm} psi rel, D rel", (float(psi / psi_o - 1), float(D / D_o - 1)))
    rec(f"V6 {name} trapezoid residuals", resid)
    rec(f"V6 {name} oracle closure W/(psi+D)-1", float(W_o / (psi_o + D_o) - 1))
    assert resid[64] <= 5e-3, resid
    assert resid[16] / resid[32] >= 3.0 and resid[32] / resid[64] >= 3.0, resid


# ================================================================================================= V7
def test_V7a_shift_values_and_reference_point():
    wlf = V.WLFShift(300.0, 17.44, 51.6)
    arr = V.ArrheniusShift(300.0, 8e4)
    assert float(wlf.ln_aT(300.0)) == 0.0 and float(arr.ln_aT(300.0)) == 0.0
    assert float(wlf.aT(300.0)) == 1.0 and float(arr.aT(300.0)) == 1.0
    mpmath.mp.dps = 50
    for Tk in (250.0, 290.0, 310.0, 345.5, 400.0):
        ref_w = -mpmath.mpf("17.44") * (mpmath.mpf(Tk) - 300) / (mpmath.mpf("51.6") + mpmath.mpf(Tk) - 300) * mpmath.log(10)
        assert float(wlf.ln_aT(Tk)) == pytest.approx(float(ref_w), rel=1e-13)
        ref_a = (mpmath.mpf(8e4) / mpmath.mpf("8.314462618")) * (1 / mpmath.mpf(Tk) - 1 / mpmath.mpf(300))
        assert float(arr.ln_aT(Tk)) == pytest.approx(float(ref_a), rel=1e-13)
        assert float(arr.aT(Tk)) == pytest.approx(float(mpmath.exp(ref_a)), rel=1e-13)
    assert V.R_GAS == 8.314462618
    assert float(wlf.ln_aT(np.array([310.0, 320.0])[0])) < 0.0                 # hotter: shorter times


def test_V7b_refusals():
    wlf = V.WLFShift(300.0, 17.44, 50.0)
    for Tk in (250.0, 240.0, 100.0):                            # C2 + T − T_ref = 0 and below
        with pytest.raises(ValueError, match="C2"):
            wlf.ln_aT(Tk)
    for Tk in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError):
            wlf.ln_aT(Tk)
        with pytest.raises(ValueError):
            V.ArrheniusShift(300.0, 8e4).ln_aT(Tk)
    rng = V.WLFShift(300.0, 17.44, 51.6, T_min=280.0, T_max=340.0)
    rng.ln_aT(300.0)
    for Tk in (279.0, 341.0):
        with pytest.raises(ValueError, match="calibrated"):
            rng.ln_aT(Tk)
    for bad in (dict(T_ref=300.0, C1=0.0, C2=51.6), dict(T_ref=300.0, C1=17.4, C2=-1.0), dict(T_ref=-1.0, C1=1.0, C2=1.0)):
        with pytest.raises(ValueError):
            V.WLFShift(**bad)
    for bad in (dict(T_ref=300.0, Ea=0.0), dict(T_ref=300.0, Ea=-1.0), dict(T_ref=0.0, Ea=1.0)):
        with pytest.raises(ValueError):
            V.ArrheniusShift(**bad)
    with pytest.raises(ValueError):
        V.WLFShift(300.0, 17.44, 51.6, T_min=350.0, T_max=340.0)


def test_V7c_reduced_time_increment():
    wlf = V.WLFShift(300.0, 17.44, 51.6)
    arr = V.ArrheniusShift(300.0, 8e4)
    cases = [(wlf, 290.0, 350.0, 100.0), (arr, 250.0, 400.0, 1.0), (arr, 300.0, 330.0, 50.0), (wlf, 350.0, 290.0, 7.0)]
    for sh, Ta, Tb, dt in cases:
        got = float(sh.reduced_time_increment(Ta, Tb, dt))
        ref = quad(lambda t: math.exp(-float(sh.ln_aT(Ta + (Tb - Ta) * t / dt))), 0.0, dt, epsabs=0, epsrel=1e-13,
                   limit=500)[0]
        rec(f"V7c reduced time rel err {type(sh).__name__} {Ta}->{Tb}", abs(got / ref - 1))
        assert got == pytest.approx(ref, rel=1e-11)
    mpmath.mp.dps = 50
    for sh, Tk in ((wlf, 320.0), (arr, 270.0)):
        ref = 13.0 * mpmath.exp(-mpmath.mpf(float(sh.ln_aT(Tk))))
        assert float(sh.reduced_time_increment(Tk, Tk, 13.0)) == pytest.approx(float(ref), rel=1e-14)
    assert float(arr.reduced_time_increment(300.0, 350.0, 0.0)) == 0.0
    with pytest.raises(ValueError, match="overflow"):                        # a_T ~ e^{-9622}: ξ overflows
        V.ArrheniusShift(1.0, 8e4).reduced_time_increment(1e5, 1e5, 1.0)
    cold = float(arr.reduced_time_increment(5.0, 5.0, 1.0))                  # a_T ~ e^{1892}: ξ underflows to the limit 0
    assert cold == 0.0
    arrv = arr.reduced_time_increment(np.array([300.0, 310.0]), np.array([300.0, 330.0]), 10.0)
    assert arrv.shape == (2,) and arrv[0] == 10.0


def _aT_hand(Ea, Tref):
    return lambda T: math.exp((Ea / 8.314462618) * (1.0 / T - 1.0 / Tref))


def test_V7d_shifted_model_against_oracles():
    arr = V.ArrheniusShift(320.0, 6e4)
    sh, bk = SETS["A"]
    # (i) regression: T = T_ref throughout is bitwise the unshifted model
    times = np.array([0.5, 1.5, 4.0, 9.0, 20.0, 45.0])
    ev = np.array([0.2, 0.5, 1.0, 0.7, -0.2, 0.4])[:, None] * V3_E0[None]
    h0 = run(model("A"), ev, times)
    h1 = run(model("A", arr), ev, times, temps=np.full(6, 320.0), T0=320.0)
    assert np.array_equal(h0.stress, h1.stress) and np.array_equal(h0.strain, h1.strain)
    # (ii) isothermal T ≠ T_ref: oracle with τ ← a_T τ
    fn = _aT_hand(6e4, 320.0)
    Tiso = 305.0
    h2 = run(model("A", arr), ev, times, temps=np.full(6, Tiso), T0=Tiso)
    ref = oracle_stress(sh, bk, times, ten(ev), "strain", aT=(np.full(6, Tiso), np.full(6, Tiso), fn))
    scale = float(np.max(np.abs(ref)))
    rec("V7d(ii) isothermal shifted err/scale", float(np.max(np.abs(vstress(ref) - h2.stress)) / scale))
    assert np.max(np.abs(vstress(ref) - h2.stress)) <= 1e-8 * scale
    # (iii) thermal cycle with ramp, hold, jump (dt = 0) and strain ramp; refinement
    nodes_t = np.array([100.0, 200.0, 300.0, 300.0, 400.0, 500.0, 600.0])
    nodes_T = np.array([320.0, 340.0, 340.0, 310.0, 310.0, 300.0, 320.0])
    nodes_l = np.array([0.5, 1.0, 1.0, 1.0, 0.2, 0.2, 0.0])
    T_start, l_start = 300.0, 0.0
    errs = {}
    for mm in (4, 8, 16, 32, 64):
        tt, TT, ll = [], [], []
        pt, pT, pl = 0.0, T_start, l_start
        for t1, T1, l1 in zip(nodes_t, nodes_T, nodes_l):
            if t1 == pt:                                      # the jump: one dt = 0 step to the new temperature
                tt.append(t1)
                TT.append(T1)
                ll.append(l1)
            else:
                for j in range(1, mm + 1):
                    f = j / mm
                    last = j == mm
                    tt.append(t1 if last else pt + f * (t1 - pt))
                    TT.append(T1 if last else pT + f * (T1 - pT))
                    ll.append(l1 if last else pl + f * (l1 - pl))
            pt, pT, pl = t1, T1, l1
        tt, TT, ll = np.array(tt), np.array(TT), np.array(ll)
        evc = ll[:, None] * V3_E0[None]
        h = run(model("A", arr), evc, tt, temps=TT, T0=T_start)
        Tprev = np.concatenate([[T_start], TT[:-1]])
        ref = oracle_stress(sh, bk, tt, ten(evc), "strain", rtol=1e-12, aT=(Tprev, TT, fn))
        scale = float(np.max(np.abs(ref)))
        errs[mm] = float(np.max(np.abs(vstress(ref) - h.stress))) / scale
    mms = sorted(errs)
    for a, b in zip(mms[:-1], mms[1:]):
        assert errs[b] < errs[a], errs
    rec("V7d thermal cycle errors", errs)
    assert errs[64] <= 5e-3, errs
    assert errs[32] / errs[64] >= 1.8, errs


# ================================================================================================= V8
def test_V8a_single_branch_dma_and_limits():
    Gi, G1, tau = 0.5e6, 1.0e6, 2.0
    s = PS(Gi, [G1], [tau])
    w = 1.0 / tau
    Gc = s.complex_modulus(w)
    assert Gc.real == pytest.approx(Gi + G1 / 2, rel=1e-14) and Gc.imag == pytest.approx(G1 / 2, rel=1e-14)
    st, lo, td = V.storage_loss_tan(Gc)
    assert td == pytest.approx((G1 / 2) / (Gi + G1 / 2), rel=1e-14)
    assert V.complex_modulus(s, 1e-9 / tau) == pytest.approx(Gi, rel=1e-8)
    assert V.complex_modulus(s, 1e9 / tau).real == pytest.approx(Gi + G1, rel=1e-8)
    assert V.complex_modulus(s, 0.0) == Gi
    # loss peak at ωτ = 1 for a single branch
    x = np.geomspace(1e-3, 1e3, 600001)
    loss = V.complex_modulus(PS(0.0, [1.0], [1.0]), x).imag
    assert abs(math.log(x[np.argmax(loss)])) <= 1e-3
    for name in ("A", "W", "NP", "D"):
        for series in SETS[name]:
            ww = np.geomspace(1e-8, 1e8, 200)
            Mc = series.complex_modulus(ww)
            assert np.all(Mc.imag >= 0.0) and np.all(Mc.real >= series.inf * (1 - 1e-14))
    assert V.frequency_hz_to_rad(1.0) == pytest.approx(2 * math.pi, rel=1e-15)
    with pytest.raises(ValueError):
        V.complex_modulus(s, -1.0)
    big = V.complex_modulus(s, 1e300)                                       # no overflow for huge ωτ
    assert np.isfinite(big) and big.real == pytest.approx(Gi + G1, rel=1e-12)


def _fit(t, y, w, amp):
    A = np.column_stack([np.sin(w * t), np.cos(w * t)]) * amp
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _sine_run(setname, uniax):
    w = 0.6
    per = 2 * math.pi / w
    n = 200
    cycles = 30
    dt = per / n
    t = dt * np.arange(1, cycles * n + 1)
    amp = 1e-3
    ev = np.zeros((len(t), 6))
    if uniax:
        ev[:, 0] = amp * np.sin(w * t)
        h = run(model(setname), ev, t, control=[True, False, False, True, True, True], tol=1e-9)
    else:
        ev[:, 5] = amp * np.sin(w * t)
        h = run(model(setname), ev, t)
    return h, t, w, n, amp


def test_V8b_time_domain_shear_cycle_energy():
    sh, bk = SETS["D"]
    h, t, w, n, amp = _sine_run("D", False)
    Gc = complex(sh.complex_modulus(w))
    a, b = _fit(t[-n:], h.stress[-n:, 5], w, amp)
    Wc = h.dissipation[-n:].sum()
    rec("V8b G* fit rel err, cycle energy rel err, psi periodicity",
        (float(abs((a + 1j * b) - Gc) / abs(Gc)), float(Wc / (math.pi * Gc.imag * amp ** 2) - 1),
         float(h.energy[-1] / h.energy[-1 - n] - 1)))
    assert abs((a + 1j * b) - Gc) <= 1e-3 * abs(Gc)
    assert Wc == pytest.approx(math.pi * Gc.imag * amp ** 2, rel=1e-3)
    assert h.energy[-1] == pytest.approx(h.energy[-1 - n], rel=1e-9)


def test_V8c_time_domain_uniaxial_E_and_nu():
    sh, bk = SETS["D"]
    h, t, w, n, amp = _sine_run("D", True)
    res = V.dma_moduli(sh, bk, w)
    a, b = _fit(t[-n:], h.stress[-n:, 0], w, amp)
    c, d = _fit(t[-n:], h.strain[-n:, 1], w, amp)
    rec("V8c E* rel err, nu* rel err", (float(abs((a + 1j * b) - complex(res.E)) / abs(res.E)),
                                         float(abs(-(c + 1j * d) - complex(res.nu)) / abs(res.nu))))
    assert abs((a + 1j * b) - complex(res.E)) <= 2e-3 * abs(res.E)
    assert abs(-(c + 1j * d) - complex(res.nu)) <= 2e-3 * abs(res.nu)
    Gc, Kc = complex(res.G), complex(res.K)
    nu = complex(res.nu)
    E = complex(res.E)
    assert abs(E - 2 * Gc * (1 + nu)) <= 1e-14 * abs(E) and abs(E - 3 * Kc * (1 - 2 * nu)) <= 1e-14 * abs(E)
    assert res.E.imag > 0.0


# ================================================================================================= V9
def test_V9a_restart_during_dwell_regression():
    m = model("A")
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 35.0, 65.0, 95.0, 125.0, 155.0])
    ev = np.array([1, 2, 3, 3, 3, 3, 3, 3, 3, 3.0])[:, None] * V3_E0[None] * 0.1
    full = run(m, ev, times)
    first = run(m, ev[:5], times[:5])
    store = StateStore(m, 1)
    store.restore(first.states[-1])
    rest = drive_material_point(m, ev[5:], np.zeros((5, 6)), [True] * 6, times=times[5:], t0=times[4], store=store,
                                strain0=first.strain[-1])
    assert np.array_equal(np.vstack([first.stress, rest.stress]), full.stress)
    assert np.array_equal(np.vstack([first.strain, rest.strain]), full.strain)
    for key in m.state_fields:
        assert np.array_equal(rest.states[-1][key], full.states[-1][key])


def test_V9b_hold_subdivision_semigroup():
    m = model("A")
    e1 = V3_E0 * 0.5
    base_t = [1e-3]
    base_e = [e1]
    one = run(m, np.array(base_e + [e1]), np.array(base_t + [1e-3 + 10.0]))
    ten_ = run(m, np.array(base_e + [e1] * 10), np.array(base_t + [1e-3 + k for k in range(1, 11)]))
    rng = np.random.default_rng(3)
    cuts = np.sort(rng.uniform(0.0, 10.0, 7))
    tt = np.concatenate([[1e-3], 1e-3 + cuts, [1e-3 + 10.0]])
    rnd = run(m, np.array([e1] * len(tt)), tt)
    scale = np.max(np.abs(one.stress[-1]))
    rec("V9b subdivision stress rel diff (10 steps, random split)",
        (float(np.max(np.abs(ten_.stress[-1] - one.stress[-1])) / scale),
         float(np.max(np.abs(rnd.stress[-1] - one.stress[-1])) / scale)))
    for other in (ten_, rnd):
        assert np.max(np.abs(other.stress[-1] - one.stress[-1])) <= 1e-12 * scale
        for key in ("s", "p"):
            a, b = one.states[-1][key], other.states[-1][key]
            assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))
    # the exact exponential semigroup is not trivial: the subdivided hold really relaxed
    assert np.max(np.abs(one.stress[-1] - one.stress[0])) > 1e-3 * scale


def test_V9c_rejection_does_not_mutate_regression():
    m = model("A")
    h = run(m, V3_E0[None] * 0.3, np.array([1.0]))
    store = StateStore(m, 1)
    store.restore(h.states[-1])
    snap = {k: v.copy() for k, v in store.committed.items()}
    for v in store.committed.values():
        assert not v.flags.writeable
    kin = Kinematics(T.voigt_to_strain(V3_E0 * 0.7)[None], "small")
    req = MaterialRequest(step=TimeStep(1.0, 2.0), energy=True)
    r1 = m.evaluate(kin, None, store.committed, req)
    r2 = m.evaluate(kin, None, store.committed, req)
    assert np.array_equal(r1.stress, r2.stress) and np.array_equal(r1.tangent, r2.tangent)
    assert np.array_equal(r1.energy, r2.energy) and np.array_equal(r1.dissipation, r2.dissipation)
    store.set_trial(r1.trial_state)
    store.rollback()
    for k in snap:
        assert np.array_equal(store.committed[k], snap[k])
    with pytest.raises(ValueError, match="time step"):
        m.evaluate(kin, None, store.committed, MaterialRequest())
    with pytest.raises(ValueError):
        TimeStep(0.0, -1.0)
    ms = model("A", V.ArrheniusShift(320.0, 6e4))
    with pytest.raises(ValueError, match="temperature"):
        ms.evaluate(kin, None, store.committed, req)
    for k in snap:
        assert np.array_equal(store.committed[k], snap[k])


# ================================================================================================= V10
def test_V10_invalid_input_and_fluid_limit():
    bad = [
        (0.5, [-1.0], [1.0]), (0.5, [1.0], [0.0]), (0.5, [1.0], [-2.0]), (0.5, [1.0], [float("nan")]),
        (0.5, [float("inf")], [1.0]), (-1.0, [1.0], [1.0]), (float("nan"), [1.0], [1.0]), (0.0, [], []),
        (0.0, [0.0], [1.0]), (0.5, [1.0, 2.0], [1.0]),
    ]
    for inf, A, tau in bad:
        with pytest.raises(ValueError):
            PS(inf, A, tau)
    with pytest.raises(ValueError):
        V.PronySeries(1.0, np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        V.PronySeries.from_weights(1e6, [0.7, 0.5], [1.0, 2.0])
    ok = V.PronySeries.from_weights(1e6, [0.25, 0.25], [1.0, 2.0])
    assert ok.inf == pytest.approx(0.5e6, rel=1e-15) and ok.instantaneous == pytest.approx(1e6, rel=1e-15)
    assert V.PronySeries.from_weights(1e6, [0.5, 0.5], [1.0, 2.0]).is_fluid
    with pytest.raises(ValueError):
        V.GeneralizedMaxwell(SETS["A"][0], SETS["A"][1], shift="WLF")
    with pytest.raises(ValueError):
        V.GeneralizedMaxwell(SETS["A"][0], 3.0)
    with pytest.raises(ValueError):
        V.sls_relaxation(1.0, 0.0, 1.0, 1.0)
    # zero equilibrium shear: accepted, flagged, long-time stiffness singular and not hidden
    fl_shear = PS(0.0, [0.8e6, 0.4e6], [1.0, 5.0])
    fl = V.GeneralizedMaxwell(fl_shear, PS(1e6, [1e6], [2.0]))
    assert fl_shear.is_fluid and fl.is_fluid and fl.equilibrium_moduli() == (1e6, 0.0)
    assert not model("A").is_fluid
    t = np.array([0.0, 1.0, 3.0, 200.0, 400.0])
    ev = np.zeros((5, 6))
    ev[:, 5] = 1e-3
    h = run(fl, ev, t)
    rec("V10 fluid held-strain stress ratio at 400 s", float(abs(h.stress[-1, 5]) / h.stress[0, 5]))
    assert abs(h.stress[-1, 5]) <= 1e-12 * h.stress[0, 5]
    # steady shear flow: γ̇ = τ_xy/η, η = Σ G_i τ_i
    t = np.array([0.0, 1e3, 2e3, 3e3])
    stress = np.zeros((4, 6))
    stress[:, 5] = 1e3
    h = run(fl, np.zeros((4, 6)), t, control=[True, True, True, True, True, False], stress=stress, tol=1e-9)
    rate = (h.strain[-1, 5] - h.strain[-2, 5]) / 1e3
    rec("V10 flow rate rel err", float(rate * fl_shear.viscosity / 1e3 - 1))
    assert rate == pytest.approx(1e3 / fl_shear.viscosity, rel=1e-6)
    assert fl_shear.viscosity == pytest.approx(0.8e6 * 1.0 + 0.4e6 * 5.0, rel=1e-15)
    # zero-amplitude branches: finite energy and dissipation
    h = run(model("Z"), np.array([[1, 2, 3, 4, 5, 6.0]]) * 1e-4, np.array([1.0]))
    assert np.all(np.isfinite(h.energy)) and np.all(np.isfinite(h.dissipation)) and h.dissipation[0] >= 0.0


# ================================================================================================= C1
def _J_sls(t, Ei=1e6, E1=2e6, tau=10.0):
    E0 = Ei + E1
    return 1.0 / Ei - (1.0 / Ei - 1.0 / E0) * np.exp(-t / (tau * E0 / Ei))


def test_C1a_sls_recovery_and_maxwell_residual():
    s0, tu = 1e3, 40.0
    Ei, E1, tau = 1e6, 2e6, 10.0
    assert V.sls_creep_time(Ei, E1, tau) == pytest.approx(30.0, rel=1e-14)
    t = np.array([0.0, 10.0, 40.0, 40.0 + 30.0, 40.0 + 3000.0])
    eps = V.recovery_strain(lambda x: V.sls_creep(x, Ei, E1, tau), t, s0, tu)
    hand = np.array([s0 * (1 / 3e6),
                     s0 * (1e-6 - (1e-6 - 1 / 3e6) * math.exp(-10 / 30)),
                     s0 * (1e-6 - (1e-6 - 1 / 3e6) * math.exp(-40 / 30)) - s0 / 3e6,   # t = t_u is after the unloading
                     s0 * (_J_sls(70.0) - _J_sls(30.0)), 0.0])
    rec("C1a max rel err", float(np.max(np.abs(eps[:4] / hand[:4] - 1.0))))
    assert np.allclose(eps[:4], hand[:4], rtol=1e-12, atol=0.0)
    assert abs(eps[4]) <= 1e-12 * s0 / Ei
    eta, E = 1e7, 1e6
    r = V.recovery_strain(lambda x: V.maxwell_creep(x, E, eta), np.array([41.0, 500.0, 1e5]), s0, tu)
    assert np.allclose(r, s0 * tu / eta, rtol=1e-12) and r[0] == pytest.approx(4e-3, rel=1e-12)


def _c1b_run(mm):
    n1 = int(round(mm * math.log10(40.0 / 1e-3))) + 1
    n2 = int(round(mm * math.log10(3000.0 / 1e-3))) + 1
    t = np.concatenate([[0.0], np.geomspace(1e-3, 40.0, n1), [40.0], 40.0 + np.geomspace(1e-3, 3000.0, n2)])
    k_unload = n1 + 1                                          # first index with σ = 0 (the duplicate node at t = 40)
    stress = np.zeros((len(t), 6))
    stress[:k_unload, 0] = 1e3
    h = run(model("P"), np.zeros((len(t), 6)), t, control=[False, False, False, True, True, True], stress=stress)
    after = np.arange(len(t)) >= k_unload
    ref = 1e3 * (_J_sls(t) - np.where(after, _J_sls(np.maximum(t - 40.0, 0.0)), 0.0))
    return t, h, ref, k_unload


def test_C1b_production_creep_and_recovery_proportional_spectra():
    errs = {}
    for mm in (25, 50, 100):
        t, h, ref, ku = _c1b_run(mm)
        errs[mm] = float(np.max(np.abs(h.strain[:, 0] - ref))) / (1e3 / 1e6)
        if mm == 100:
            peak = h.strain[:, 0].max()
            sel = np.abs(h.strain[:, 0]) >= 1e-3 * peak
            rec("C1b final strain / peak, lateral ratio dev", (float(abs(h.strain[-1, 0]) / peak),
                float(np.max(np.abs(-h.strain[sel, 1] / h.strain[sel, 0] - 0.3)))))
            assert abs(h.strain[-1, 0]) <= 1e-6 * peak
            assert np.max(np.abs(-h.strain[sel, 1] / h.strain[sel, 0] - 0.3)) <= 1e-9
            assert np.max(np.abs(h.stress[:, 1:3])) <= 1e-8 * 1e3
    rec("C1b errs", errs)
    assert errs[100] <= 5e-4, errs
    assert order(errs[50], errs[100]) >= 1.6, errs
