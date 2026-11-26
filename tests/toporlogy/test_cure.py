"""Cure kinetics, glass transition, reaction heat, chemical shrinkage, Avrami / Nakamura and the curing material point
(``toporlogy.core.cure``). Checks TS1-TS6, TP3, TP4 of docs/foundations/prereg/TS_cure.md (thresholds pre-registered
2026-11-27 01:54 +0300, committed in 5003f24 before this file was run). All numbers are synthetic, not material data.

References are independent of the production code: closed forms, ``scipy.integrate.solve_ivp`` (DOP853, rtol 1e-12) with
the rate law and the Arrhenius formula written in this file, ``scipy.integrate.quad``, central finite differences, a
dense numpy stiffness. A comparison with the hand-derived recurrence of backward Euler is labelled "derivation"."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from toporlogy.core import cure as C
from toporlogy.core import properties as P
from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (Kinematics, MaterialRequest, StateStore, TimeStep, drive_material_point)

R = 8.314462618


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def k_arr(T_: Any, k_ref: float, T_ref: float, Q: float) -> Any:
    """The test's own Arrhenius law with k(T_ref) = k_ref."""
    return k_ref * np.exp(-Q / R * (1.0 / np.asarray(T_, float) - 1.0 / T_ref))


def first_order() -> C.NthOrder:
    return C.NthOrder(C.Arrhenius(0.1, 0.0), 1.0)                       # k = 0.1 1/s at every T


# ============================================================================================== TS1 first order
def test_ts1a_closed_form() -> None:
    exact = 1.0 - 0.8 * math.exp(-1.0)
    assert rel(C.isothermal_first_order(0.2, 0.1, 10.0), exact) <= 1e-14
    assert rel(C.isothermal_nth_order(0.2, 0.1, 1.0, 10.0), exact) <= 1e-14
    assert C.isothermal_first_order(0.2, 0.1, 0.0) == 0.2


def recurrence_n1(alpha0: float, k_of_step: Any, dt: float, n_steps: int) -> np.ndarray:
    """Derivation: backward Euler for n = 1, 1 - a_{j+1} = (1 - a_j)/(1 + k dt), k at the END temperature."""
    a = [alpha0]
    for j in range(n_steps):
        kd = k_of_step(j) * dt
        a.append((a[-1] + kd) / (1.0 + kd))
    return np.array(a)


def test_ts1b_matches_hand_recurrence() -> None:
    law = first_order()
    a = 0.2
    got = [a]
    for _ in range(10):
        a = law.advance(a, 300.0, 300.0, 1.0)
        got.append(a)
    assert np.max(np.abs(np.array(got) - recurrence_n1(0.2, lambda j: 0.1, 1.0, 10))) <= 1e-12
    # temperature dependent k: Arrhenius with Q != 0, ramp 380 -> 420 K over 20 s
    law = C.NthOrder(C.Arrhenius.from_reference(0.05, 400.0, 70e3), 1.0)
    Tg = 380.0 + 2.0 * np.arange(21)
    a, got = 0.0, [0.0]
    for j in range(20):
        a = law.advance(a, Tg[j], Tg[j + 1], 1.0)
        got.append(a)
    ref = recurrence_n1(0.0, lambda j: k_arr(Tg[j + 1], 0.05, 400.0, 70e3), 1.0, 20)
    assert np.max(np.abs(np.array(got) - ref)) <= 1e-12
    # regression (same kernel): the batched call equals the scalar loop
    vec = law.advance(np.array([0.0, 0.3, 0.9]), np.array([380.0, 390.0, 400.0]), np.array([385.0, 395.0, 410.0]), 2.0)
    one = [law.advance(a0, t0, t1, 2.0) for a0, t0, t1 in ((0.0, 380.0, 385.0), (0.3, 390.0, 395.0), (0.9, 400.0, 410.0))]
    assert float(np.max(np.abs(vec - np.array(one)))) <= 1e-14


@pytest.mark.parametrize("dt,steps", [(0.01, 1000), (1.0, 10), (100.0, 1)])
def test_ts1c_monotone_bounded(dt: float, steps: int) -> None:
    law = first_order()
    a, seq = 0.2, [0.2]
    for _ in range(steps):
        a = law.advance(a, 320.0, 320.0, dt)
        seq.append(a)
    s = np.array(seq)
    assert np.all(s >= 0.2) and np.all(s <= 1.0) and np.all(np.diff(s) >= 0.0)


def test_ts1d_first_order_convergence() -> None:
    law = first_order()
    exact = 1.0 - 0.8 * math.exp(-1.0)
    dts = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    errs = []
    for dt in dts:
        a = 0.2
        for _ in range(int(round(10.0 / dt))):
            a = law.advance(a, 300.0, 300.0, dt)
        errs.append(abs(a - exact))
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(dts) - 1)]
    for dt, p in zip(dts[1:-1], orders[1:]):                              # pairs with dt <= 1
        assert 0.9 <= p <= 1.1, (dt, p)
    assert errs[-1] <= 1.2e-3
    assert abs(errs[-1] / (0.8 * math.exp(-1.0) * 0.05 * 0.0625) - 1.0) <= 0.05


def test_ts1e_negative_control_explicit_euler_overshoots() -> None:
    dt, k = 100.0, 0.1
    a_exp = 0.2 + dt * k * (1.0 - 0.2)                                   # explicit Euler, one step
    assert rel(a_exp, 8.2) <= 1e-14 and a_exp > 1.0
    a_imp = first_order().advance(0.2, 300.0, 300.0, dt)
    assert 0.2 <= a_imp <= 1.0


def test_ts1f_nth_order_closed_forms() -> None:
    k = 0.05
    for n, t_end in ((2.0, 40.0), (0.5, 30.0)):
        sol = solve_ivp(lambda t, y, n=n: [k * (1.0 - y[0]) ** n], (0.0, t_end), [0.2], method="DOP853", rtol=1e-12,
                        atol=1e-14, t_eval=[t_end])
        assert rel(C.isothermal_nth_order(0.2, k, n, t_end), sol.y[0, -1]) <= 1e-9
    t_star = 0.8 ** 0.5 / (0.5 * k)
    assert rel(C.finite_completion_time(0.2, k, 0.5), t_star) <= 1e-14
    assert C.finite_completion_time(0.2, k, 2.0) == math.inf and C.finite_completion_time(0.2, k, 1.0) == math.inf
    assert C.isothermal_nth_order(0.2, k, 0.5, t_star * 1.0001) == 1.0
    assert C.isothermal_nth_order(0.2, k, 0.5, 2 * t_star) == 1.0
    # the implicit step with n < 1 reaches 1 and stays at 1 without leaving [0, 1]
    law = C.NthOrder(C.Arrhenius(k, 0.0), 0.5)
    a = 0.2
    for _ in range(10):
        b = law.advance(a, 300.0, 300.0, 20.0)
        assert a <= b <= 1.0
        a = b


# ============================================================================================== TS2 Kamal-Sourour
Q1, Q2, TREF = 60e3, 50e3, 400.0


def kamal_law() -> C.KamalSourour:
    return C.KamalSourour(C.Arrhenius.from_reference(0.01, TREF, Q1), C.Arrhenius.from_reference(0.05, TREF, Q2),
                          1.0, 2.0)


def ramp_T(t: Any) -> Any:
    return 380.0 + 0.2 * np.asarray(t, float)


def kamal_rhs(t: float, y: Any) -> list[float]:
    a, Tt = y[0], 380.0 + 0.2 * t
    return [(k_arr(Tt, 0.01, TREF, Q1) + k_arr(Tt, 0.05, TREF, Q2) * a) * (1.0 - a) ** 2]


def run_ramp(law: C.CureKinetics, alpha0: float, n_steps: int) -> np.ndarray:
    t = np.linspace(0.0, 300.0, n_steps + 1)
    Tt = ramp_T(t)
    a, out = alpha0, [alpha0]
    for j in range(n_steps):
        a = law.advance(a, Tt[j], Tt[j + 1], t[j + 1] - t[j])
        out.append(a)
    return np.array(out)


@pytest.mark.parametrize("alpha0", [0.0, 0.1])
def test_ts2a_kamal_ramp_against_ode_and_order(alpha0: float) -> None:
    law = kamal_law()
    t_fine = np.linspace(0.0, 300.0, 961)
    sol = solve_ivp(kamal_rhs, (0.0, 300.0), [alpha0], method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t_fine)
    ref = sol.y[0]
    errs = []
    for n_steps in (240, 480, 960):
        a = run_ramp(law, alpha0, n_steps)
        stride = 960 // n_steps
        errs.append(float(np.max(np.abs(a - ref[::stride]))))
    p1, p2 = math.log2(errs[0] / errs[1]), math.log2(errs[1] / errs[2])
    assert 0.8 <= p1 <= 1.2 and 0.8 <= p2 <= 1.2, (errs, p1, p2)
    assert errs[2] <= 2e-2


@pytest.mark.parametrize("n_steps", [3, 1])
def test_ts2b_bounded_at_coarse_steps(n_steps: int) -> None:
    a = run_ramp(kamal_law(), 0.0, n_steps)
    assert np.all(a >= 0.0) and np.all(a <= 1.0) and np.all(np.diff(a) >= 0.0)
    assert a[-1] > 0.0


def test_ts2c_autocatalytic_start() -> None:
    law = C.KamalSourour(C.Arrhenius(0.0, 0.0), C.Arrhenius(0.05, 0.0), 1.0, 2.0)
    assert law.advance(0.0, 400.0, 400.0, 10.0) == 0.0                    # trivial equilibrium, documented
    a = law.advance(1e-3, 400.0, 400.0, 10.0)
    assert 1e-3 < a <= 1.0
    assert law.advance(1.0, 400.0, 400.0, 10.0) == 1.0
    assert kamal_law().advance(0.0, 400.0, 400.0, 10.0) > 0.0              # k1 > 0 initiates


def test_ts2d_rate_derivatives_against_fd() -> None:
    laws = [C.NthOrder(C.Arrhenius.from_reference(0.02, 400.0, 60e3), 1.5),
            C.KamalSourour(C.Arrhenius.from_reference(0.01, 400.0, 60e3), C.Arrhenius.from_reference(0.05, 400.0, 50e3),
                           1.3, 2.2)]
    for law in laws:
        for a in (0.3, 0.7):
            h = 1e-6
            fd_a = (law.rate(a + h, 400.0) - law.rate(a - h, 400.0)) / (2 * h)
            assert abs(law.drate_dalpha(a, 400.0) - fd_a) <= 1e-6 * abs(fd_a)
            hT = 1e-3
            fd_T = (law.rate(a, 400.0 + hT) - law.rate(a, 400.0 - hT)) / (2 * hT)
            assert abs(law.drate_dT(a, 400.0) - fd_T) <= 1e-6 * abs(fd_T)


# ============================================================================================== TS3 DiBenedetto
def test_ts3_dibenedetto() -> None:
    g = C.DiBenedetto(250.0, 420.0, 0.5)
    a = np.array([0.0, 0.5, 1.0])
    assert rel(g.Tg(a), [250.0, 250.0 + 170.0 / 3.0, 420.0]) <= 1e-14
    assert rel(g.dTg_dalpha(a), [85.0, 170.0 * 0.5 / 0.5625, 340.0]) <= 1e-14

    def f(x: float) -> float:
        return float(g.Tg(x))

    for x in (0.25, 0.5, 0.75):
        h = 1e-5
        assert abs(g.dTg_dalpha(x) - (f(x + h) - f(x - h)) / (2 * h)) <= 1e-6 * g.dTg_dalpha(x)
    h = 1e-4
    d0 = (-3 * f(0.0) + 4 * f(h) - f(2 * h)) / (2 * h)
    d1 = (3 * f(1.0) - 4 * f(1.0 - h) + f(1.0 - 2 * h)) / (2 * h)
    assert abs(g.dTg_dalpha(0.0) - d0) <= 1e-6 * d0 and abs(g.dTg_dalpha(1.0) - d1) <= 1e-6 * d1
    lin = C.DiBenedetto(250.0, 420.0, 1.0)
    assert rel(lin.Tg(a), 250.0 + 170.0 * a) <= 1e-14 and rel(lin.dTg_dalpha(a), np.full(3, 170.0)) <= 1e-14
    al = np.linspace(0.0, 1.0, 11)
    assert float(np.max(np.abs(g.alpha_at_Tg(g.Tg(al)) - al))) <= 1e-13        # round trip (alpha in [0, 1])
    assert np.all(np.diff(g.Tg(al)) > 0.0)


# ============================================================================================== TS4 heat
def cp_const() -> P.Constant:
    return P.Constant(2000.0, "J/(kg K)", "synthetic", positive=True)


def test_ts4a_adiabatic_rise_constant_cp() -> None:
    h = C.ReactionHeat(2e5)
    assert rel(h.adiabatic_temperature(300.0, 0.0, 0.5, cp_const()), 350.0) <= 1e-14
    assert rel(h.adiabatic_temperature(300.0, 0.2, 0.7, cp_const()), 350.0) <= 1e-14
    assert rel(h.heat(0.2, 0.7), 1e5) <= 1e-14
    with pytest.raises(ValueError):
        h.heat(0.7, 0.2)


def test_ts4b_variable_cp_against_ode() -> None:
    h = C.ReactionHeat(2e5)
    cp = P.Polynomial([1800.0, 4.0], 300.0, (300.0, 500.0), "J/(kg K)", "synthetic", positive=True)
    dT = (-1800.0 + math.sqrt(1800.0 ** 2 + 8e5)) / 4.0
    assert rel(h.adiabatic_temperature(300.0, 0.0, 0.5, cp) - 300.0, dT) <= 1e-12

    def rhs(t: float, y: Any) -> list[float]:
        a, Tt = y
        da = (k_arr(Tt, 0.01, TREF, Q1) + k_arr(Tt, 0.05, TREF, Q2) * a) * (1.0 - a) ** 2
        return [da, 2e5 * da / (1800.0 + 4.0 * (Tt - 300.0))]

    te = np.linspace(0.0, 1500.0, 26)
    sol = solve_ivp(rhs, (0.0, 1500.0), [0.0, 350.0], method="DOP853", rtol=1e-12, atol=1e-12, t_eval=te)
    assert sol.y[0, -1] > 0.5                                             # the run actually cures
    worst = 0.0
    for al, Tt in zip(sol.y[0], sol.y[1]):
        al = min(max(float(al), 0.0), 1.0)                                # reference post-processing only
        T_prod = h.adiabatic_temperature(350.0, 0.0, al, cp)
        worst = max(worst, abs(T_prod - Tt) / max(Tt - 350.0, 1.0))
    assert worst <= 1e-9
    with pytest.raises(ValueError):                                       # leaves the c_p validity interval
        C.ReactionHeat(2e6).adiabatic_temperature(350.0, 0.0, 1.0, cp)


def test_ts4c_heat_budget() -> None:
    h = C.ReactionHeat(2e5)
    law = first_order()
    a, seq = 0.2, [0.2]
    for _ in range(20):
        a = law.advance(a, 300.0, 300.0, 0.5)
        seq.append(a)
    steps = sum(h.heat(seq[j], seq[j + 1]) for j in range(20))
    assert rel(steps, 2e5 * (seq[-1] - 0.2)) <= 1e-12
    rho, k, t_end = 1200.0, 0.1, 10.0
    val, _ = quad(lambda t: float(h.power_density(rho, k * 0.8 * math.exp(-k * t))), 0.0, t_end, epsabs=0, epsrel=1e-13)
    exact = rho * 2e5 * (C.isothermal_first_order(0.2, k, t_end) - 0.2)
    assert rel(val, exact) <= 1e-10
    with pytest.raises(ValueError):
        h.power_density(rho, -1.0)


# ============================================================================================== TS5 shrinkage
def test_ts5a_shrinkage_function() -> None:
    s = C.ChemicalShrinkage(-0.03)
    assert rel(s.volumetric(1.0), -0.03) <= 1e-14 and rel(s.linear(1.0), -0.01) <= 1e-14
    assert s.linear(1.0) < 0.0
    path = np.array([0.0, 0.1, 0.35, 0.4, 0.8, 1.0])
    tot = sum(s.linear_increment(path[j], path[j + 1]) for j in range(5))
    assert rel(tot, -0.03 * 1.0 / 3.0) <= 1e-13
    g = C.ChemicalShrinkage(-0.03, alpha_gel=0.3)
    assert g.volumetric(0.3) == 0.0 and g.volumetric(0.1) == 0.0
    assert rel(g.volumetric(0.8), -0.03 * 0.5) <= 1e-14


def elastic_synth() -> tuple[P.IsotropicModuliOfT, P.IsotropicModuliOfT]:
    mu = P.IsotropicModuliOfT(E=P.Constant(1e5, "Pa", "synthetic"), nu=P.Constant(0.45, "1", "synthetic"))
    mc = P.IsotropicModuliOfT(E=P.Constant(3e9, "Pa", "synthetic"), nu=P.Constant(0.35, "1", "synthetic"))
    return mu, mc


def make_point(alpha_gel: float = 0.0, kinetics: C.CureKinetics | None = None, beta: float = -0.03
               ) -> C.CuringThermoElasticPoint:
    mu, mc = elastic_synth()
    exp = P.ThermalExpansion(P.Constant(5e-5, "1/K", "synthetic"), 300.0)
    return C.CuringThermoElasticPoint(kinetics or first_order(), mu, mc, exp, C.ChemicalShrinkage(beta, alpha_gel))


K_U, K_C = 1e5 / (3 * (1 - 0.9)), 3e9 / (3 * (1 - 0.7))


def test_ts5b_free_shrinkage_and_thermal() -> None:
    model = make_point()
    n = 20
    times = np.arange(1.0, n + 1.0)
    temps = 300.0 + 2.0 * times
    z = np.zeros((n, 6))
    h = drive_material_point(model, z, z, np.zeros(6, bool), tol=1e-6, times=times, temperatures=temps, t0=0.0,
                             temperature0=300.0)
    alpha = recurrence_n1(0.0, lambda j: 0.1, 1.0, n)[1:]
    e = 5e-5 * (temps - 300.0) + (-0.03 / 3.0) * alpha
    expected = np.zeros((n, 6))
    expected[:, :3] = e[:, None]
    assert np.max(np.abs(h.strain - expected)) <= 1e-12
    assert np.max(np.abs(h.stress)) <= 1e-5
    assert np.all(e[:5] != 0.0)
    assert rel([s["alpha"][0] for s in h.states], alpha) <= 1e-12


@pytest.mark.parametrize("alpha_gel", [0.0, 0.3])
def test_ts5c_fully_constrained(alpha_gel: float) -> None:
    model = make_point(alpha_gel)
    n = 20
    times = np.arange(1.0, n + 1.0)
    z = np.zeros((n, 6))
    h = drive_material_point(model, z, z, np.ones(6, bool), tol=1e-6, times=times, temperatures=np.full(n, 300.0),
                             t0=0.0, temperature0=300.0)
    alpha = recurrence_n1(0.0, lambda j: 0.1, 1.0, n)[1:]
    eps_lin = (-0.03 / 3.0) * np.maximum(alpha - alpha_gel, 0.0)
    Kmix = (1 - alpha) * K_U + alpha * K_C
    sigma = -3.0 * Kmix * eps_lin
    for c in range(3):
        assert np.max(np.abs(h.stress[:, c] - sigma)) <= 1e-12 * np.max(np.abs(sigma))
    assert np.max(np.abs(h.stress[:, 3:])) <= 1e-9
    assert np.all(sigma >= 0.0) and sigma[-1] > 0.0                       # shrinkage held back = tension
    if alpha_gel > 0.0:
        pre = alpha <= alpha_gel
        assert pre.any() and np.max(np.abs(h.stress[pre])) <= 1e-9


def test_ts5d_contract_and_tangent() -> None:
    model = make_point()
    assert model.physical is True and model.kinematics == "small" and model.state_fields == ("alpha",)
    assert model.conservative is False
    H0 = np.zeros((1, 3, 3))
    store = StateStore(model, 1)
    with pytest.raises(ValueError):
        model.evaluate(Kinematics(H0, "small"), None, store.committed, MaterialRequest())
    with pytest.raises(ValueError):
        model.evaluate(Kinematics(H0, "small"), None, store.committed, MaterialRequest(step=TimeStep(0.0, 1.0)))
    req = MaterialRequest(step=TimeStep(0.0, 1.0, 320.0, 310.0), energy=True)
    e6 = np.array([1e-3, -2e-4, 3e-4, 4e-4, -5e-4, 6e-4])
    r = model.evaluate(Kinematics(T.voigt_to_strain(e6)[None], "small"), None, store.committed, req)
    assert r.energy is None and r.dissipation is None
    a1 = 0.1 / 1.1                                                        # derivation: one backward Euler step
    K, G = (1 - a1) * K_U + a1 * K_C, (1 - a1) * (1e5 / 2.9) + a1 * (3e9 / 2.7)
    one = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    Id = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(one, one) / 3.0
    dense = K * np.outer(one, one) + 2.0 * G * Id
    assert rel(r.tangent[0], dense) <= 1e-12

    def sig(e: np.ndarray) -> np.ndarray:
        return model.evaluate(Kinematics(T.voigt_to_strain(e)[None], "small"), None, store.committed, req).stress[0]

    h = 1e-7
    fd = np.stack([(sig(e6 + h * np.eye(6)[i]) - sig(e6 - h * np.eye(6)[i])) / (2 * h) for i in range(6)], axis=1)
    assert rel(r.tangent[0], fd) <= 1e-6


# ============================================================================================== TS6 history
def test_ts6a_trial_does_not_change_committed() -> None:
    model = make_point()
    store = StateStore(model, 1)
    before = store.committed["alpha"].copy()
    assert not store.committed["alpha"].flags.writeable
    req = MaterialRequest(step=TimeStep(0.0, 1.0, 300.0, 300.0))
    kin = Kinematics(np.zeros((1, 3, 3)), "small")
    r1 = model.evaluate(kin, None, store.committed, req)
    store.set_trial(r1.trial_state)
    assert r1.trial_state["alpha"][0] > 0.0
    assert np.array_equal(store.committed["alpha"], before)
    store.rollback()
    assert np.array_equal(store.committed["alpha"], before)
    r2 = model.evaluate(kin, None, store.committed, req)
    assert np.array_equal(r1.trial_state["alpha"], r2.trial_state["alpha"])
    bad = {"alpha": np.array([1.5])}
    with pytest.raises(ValueError):
        model.evaluate(kin, None, bad, req)
    assert np.array_equal(store.committed["alpha"], before)


class Probe:
    """Records the committed alpha at every evaluate (CT5 style)."""

    def __init__(self, m: Any) -> None:
        self.m = m
        self.seen: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.m, name)

    def evaluate(self, kin: Any, params: Any, committed: Any, request: Any) -> Any:
        self.seen.append(float(committed["alpha"][0]))
        return self.m.evaluate(kin, params, committed, request)


def test_ts6b_committed_constant_inside_newton_loop() -> None:
    probe = Probe(make_point())
    store = StateStore(probe, 1)
    n = 4
    times = np.arange(1.0, n + 1.0)
    strain = np.zeros((n, 6))
    stress = np.zeros((n, 6))
    ctrl = np.array([True, False, False, False, False, False])
    drive_material_point(probe, strain, stress, ctrl, tol=1e-6, times=times, temperatures=np.full(n, 300.0), t0=0.0,
                         temperature0=300.0, store=store)
    assert store.n_commits == n
    seen = np.array(probe.seen)
    assert len(seen) >= 2 * n and len(np.unique(seen)) == n               # >= 2 evaluations per step, same committed
    assert np.all(np.diff(seen) >= 0.0)


def schedule(ramp_first: bool) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(1.0, 31.0)
    if ramp_first:
        temps = np.where(times <= 20.0, 380.0 + 2.0 * times, 420.0)
    else:
        temps = np.where(times <= 10.0, 380.0, 380.0 + 2.0 * (times - 10.0))
    return times, temps


def test_ts6c_history_restart_and_order_matters() -> None:
    kin = C.NthOrder(C.Arrhenius.from_reference(0.05, 400.0, 70e3), 1.0)
    model = make_point(kinetics=kin)
    times, temps = schedule(True)
    z = np.zeros((30, 6))
    h = drive_material_point(model, z, z, np.ones(6, bool), tol=1e-6, times=times, temperatures=temps, t0=0.0,
                             temperature0=380.0)
    alpha = np.array([s["alpha"][0] for s in h.states])
    ref = recurrence_n1(0.0, lambda j: k_arr(temps[j], 0.05, 400.0, 70e3), 1.0, 30)[1:]
    assert np.max(np.abs(alpha - ref)) <= 1e-12
    # restart after 15 steps from a checkpoint reproduces the end state bitwise
    store1 = StateStore(model, 1)
    h1 = drive_material_point(model, z[:15], z[:15], np.ones(6, bool), tol=1e-6, times=times[:15], temperatures=temps[:15],
                              t0=0.0, temperature0=380.0, store=store1)
    store2 = StateStore(model, 1)
    store2.restore(store1.checkpoint())
    h2 = drive_material_point(model, z[15:], z[15:], np.ones(6, bool), tol=1e-6, times=times[15:],
                              temperatures=temps[15:], t0=float(times[14]), temperature0=float(temps[14]),
                              store=store2, strain0=h1.strain[-1])
    assert np.array_equal(h2.states[-1]["alpha"], h.states[-1]["alpha"])
    # negative control: the same end temperature reached hold-then-ramp gives a different degree of cure
    t2, T2 = schedule(False)
    h3 = drive_material_point(model, z, z, np.ones(6, bool), tol=1e-6, times=t2, temperatures=T2, t0=0.0,
                              temperature0=380.0)
    assert abs(h3.states[-1]["alpha"][0] - h.states[-1]["alpha"][0]) > 1e-3


def test_ts6d_invalid_input_raises() -> None:
    law = first_order()
    for args in ((1.1, 300.0, 300.0, 1.0), (-0.1, 300.0, 300.0, 1.0), (float("nan"), 300.0, 300.0, 1.0),
                 (0.5, 300.0, 300.0, -1.0), (0.5, 0.0, 300.0, 1.0), (0.5, 300.0, -3.0, 1.0),
                 (0.5, float("nan"), 300.0, 1.0), (0.5, 300.0, 300.0, float("inf"))):
        with pytest.raises(ValueError):
            law.advance(*args)
    with pytest.raises(ValueError):
        law.advance(0.5, 300.0, 300.0, 1.0, substeps=0)
    with pytest.raises(ValueError):
        law.rate(1.5, 300.0)
    for ctor in (lambda: C.Arrhenius(-1.0, 0.0), lambda: C.Arrhenius(1.0, -1.0), lambda: C.Arrhenius(float("nan"), 1.0),
                 lambda: C.Arrhenius.from_reference(0.0, 400.0, 1e3), lambda: C.Arrhenius.from_reference(1.0, 0.0, 1e3),
                 lambda: C.NthOrder(C.Arrhenius(1.0, 0.0), 0.0), lambda: C.NthOrder(C.Arrhenius(1.0, 0.0), -1.0),
                 lambda: C.NthOrder(1.0, 1.0),
                 lambda: C.KamalSourour(C.Arrhenius(1.0, 0.0), C.Arrhenius(1.0, 0.0), -0.1, 1.0),
                 lambda: C.KamalSourour(C.Arrhenius(1.0, 0.0), C.Arrhenius(1.0, 0.0), 1.0, 0.0),
                 lambda: C.DiBenedetto(250.0, 250.0, 0.5), lambda: C.DiBenedetto(0.0, 250.0, 0.5),
                 lambda: C.DiBenedetto(250.0, 420.0, 0.0), lambda: C.ReactionHeat(0.0),
                 lambda: C.ReactionHeat(float("inf")), lambda: C.ChemicalShrinkage(float("nan")),
                 lambda: C.ChemicalShrinkage(-0.03, alpha_gel=1.0), lambda: C.ChemicalShrinkage(-0.03, alpha_gel=-0.1),
                 lambda: C.Nakamura(0.0, P.Constant(0.1, "1/s", "synthetic")),
                 lambda: C.Nakamura(2.0, P.Constant(0.1, "1", "synthetic"))):
        with pytest.raises(ValueError):
            ctor()
    g = C.DiBenedetto(250.0, 420.0, 0.5)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            g.Tg(bad)
    with pytest.raises(ValueError):
        g.alpha_at_Tg(200.0)
    h = C.ReactionHeat(2e5)
    with pytest.raises(ValueError):                                       # c_p not built with positive=True
        h.adiabatic_temperature(300.0, 0.0, 0.5, P.Constant(2000.0, "J/(kg K)", "synthetic"))
    with pytest.raises(ValueError):                                       # wrong units
        h.adiabatic_temperature(300.0, 0.0, 0.5, P.Constant(2000.0, "Pa", "synthetic", positive=True))
    for call in (lambda: C.avrami(1.0, 0.0, 2.0), lambda: C.avrami(1.0, 0.01, 0.0), lambda: C.avrami(-1.0, 0.01, 2.0),
                 lambda: C.avrami_time(1.0, 0.01, 2.0), lambda: C.avrami_time(-0.1, 0.01, 2.0),
                 lambda: C.nakamura_K_from_avrami(-1.0, 2.0)):
        with pytest.raises(ValueError):
            call()
    nk = C.Nakamura(2.0, P.Constant(0.1, "1/s", "synthetic"))
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            nk.advance(bad, 300.0, 300.0, 1.0)
    with pytest.raises(ValueError):
        nk.advance(0.0, 300.0, 300.0, -1.0)
    neg = C.Nakamura(2.0, P.PiecewiseLinear([350.0, 450.0], [-0.1, -0.1], "1/s", "synthetic"))
    with pytest.raises(ValueError):
        neg.advance_theta(0.0, 400.0, 410.0, 1.0)
    win = C.Nakamura(2.0, P.PiecewiseLinear([350.0, 450.0], [0.01, 0.1], "1/s", "synthetic"))
    with pytest.raises(ValueError):                                       # outside the validity interval of K
        win.advance_theta(0.0, 440.0, 455.0, 1.0)


# ============================================================================================== TP3 Avrami
def test_tp3a_avrami_closed_form() -> None:
    assert rel(C.avrami(10.0, 0.01, 2.0), 1.0 - math.exp(-1.0)) <= 1e-13
    assert C.avrami(0.0, 0.01, 2.0) == 0.0
    t = np.linspace(0.0, 30.0, 61)
    X = C.avrami(t, 0.01, 2.0)
    assert np.all(np.diff(X) > 0.0) and np.all(X < 1.0)
    assert rel(C.avrami_time(C.avrami(10.0, 0.01, 2.0), 0.01, 2.0), 10.0) <= 1e-13
    for n in (1.0, 2.0, 2.5, 3.0):
        K = 0.1
        k = K ** n
        ref = 1.0 - math.exp(-k * 5.0 ** n)
        assert rel(C.avrami(5.0, k, n), ref) <= 1e-13
        assert rel(C.nakamura_K_from_avrami(k, n), K) <= 1e-13
        nk = C.Nakamura(n, P.Constant(K, "1/s", "synthetic"))
        th = 0.0
        for _ in range(5):
            th = nk.advance_theta(th, 300.0, 300.0, 1.0)
        assert rel(nk.X_of_theta(th), ref) <= 1e-13


def test_tp3b_nakamura_rate_at_constant_T() -> None:
    nk = C.Nakamura(2.0, P.Constant(0.1, "1/s", "synthetic"))
    X10 = 1.0 - math.exp(-1.0)
    assert rel(nk.rate(X10, 300.0), 0.2 * math.exp(-1.0)) <= 1e-13
    assert nk.rate(1.0, 300.0) == 0.0


# ============================================================================================== TP4 Nakamura
def K_table() -> P.PiecewiseLinear:
    return P.PiecewiseLinear([350.0, 400.0, 450.0], [0.01, 0.1, 0.01], "1/s", "synthetic")


def K_ref(T_: Any) -> Any:
    return np.interp(T_, [350.0, 400.0, 450.0], [0.01, 0.1, 0.01])


def test_tp4a_constant_T_reduces_to_avrami() -> None:
    nk = C.Nakamura(2.0, P.Constant(0.1, "1/s", "synthetic"))
    exact = 1.0 - math.exp(-1.0)
    assert abs(nk.advance(0.0, 400.0, 400.0, 10.0) - exact) <= 1e-13
    th = 0.0
    for _ in range(100):
        th = nk.advance_theta(th, 400.0, 400.0, 0.1)
    assert abs(nk.X_of_theta(th) - exact) <= 1e-13
    X = 0.0
    for _ in range(100):
        X = nk.advance(X, 400.0, 400.0, 0.1)
    assert abs(X - exact) <= 1e-12                                         # via the X <-> theta round trip
    assert nk.advance(0.3, 400.0, 400.0, 0.0) == 0.3 and nk.advance(1.0, 400.0, 400.0, 5.0) == 1.0


def varying_T_reference(X0: float) -> tuple[np.ndarray, np.ndarray]:
    te = np.arange(0.0, 20.5, 0.5)
    th0 = (-math.log1p(-X0)) ** 0.5
    s1 = solve_ivp(lambda t, y: [float(K_ref(440.0 - 4.0 * t))], (0.0, 10.0), [th0], method="DOP853", rtol=1e-12,
                   atol=1e-14, t_eval=te[te <= 10.0])
    s2 = solve_ivp(lambda t, y: [float(K_ref(440.0 - 4.0 * t))], (10.0, 20.0), [s1.y[0, -1]], method="DOP853",
                   rtol=1e-12, atol=1e-14, t_eval=te[te > 10.0])
    return te, np.concatenate([s1.y[0], s2.y[0]])


def test_tp4b_varying_T_against_theta_ode() -> None:
    nk = C.Nakamura(2.0, K_table())
    te, th_ref = varying_T_reference(0.0)
    Tg = 440.0 - 4.0 * te
    th, X = 0.0, [0.0]
    for j in range(40):
        th = nk.advance_theta(th, Tg[j], Tg[j + 1], 0.5)
        X.append(float(nk.X_of_theta(th)))
    assert float(np.max(np.abs(np.array(X) - (1.0 - np.exp(-th_ref ** 2))))) <= 1e-10
    assert X[-1] > 0.5 and np.all(np.diff(X) >= 0.0)


def test_tp4c_varying_T_against_X_ode() -> None:
    nk = C.Nakamura(2.0, K_table())
    X0 = 0.01
    te = np.arange(0.0, 20.5, 0.5)

    def rhs(t: float, y: Any) -> list[float]:
        return [2.0 * float(K_ref(440.0 - 4.0 * t)) * (1.0 - y[0]) * math.sqrt(-math.log(1.0 - y[0]))]

    s1 = solve_ivp(rhs, (0.0, 10.0), [X0], method="DOP853", rtol=1e-12, atol=1e-14, t_eval=te[te <= 10.0])
    s2 = solve_ivp(rhs, (10.0, 20.0), [s1.y[0, -1]], method="DOP853", rtol=1e-12, atol=1e-14, t_eval=te[te > 10.0])
    ref = np.concatenate([s1.y[0], s2.y[0]])
    Tg = 440.0 - 4.0 * te
    X, got = X0, [X0]
    for j in range(40):
        X = nk.advance(X, Tg[j], Tg[j + 1], 0.5)
        got.append(X)
    assert float(np.max(np.abs(np.array(got) - ref))) <= 1e-8


def test_tp4d_negative_control_X_form_never_starts() -> None:
    nk = C.Nakamura(2.0, P.Constant(0.1, "1/s", "synthetic"))
    X, dt = 0.0, 0.1
    for _ in range(100):                                                   # explicit Euler on the X-form, from X = 0
        X = X + dt * 2.0 * 0.1 * (1.0 - X) * math.sqrt(-math.log1p(-X)) if X > 0.0 else X
    assert X == 0.0 and abs((1.0 - math.exp(-1.0)) - X) > 0.5
    assert nk.rate(0.0, 400.0) == 0.0                                      # the singular start-up of the X form
    assert abs(nk.advance(0.0, 400.0, 400.0, 10.0) - (1.0 - math.exp(-1.0))) <= 1e-13
