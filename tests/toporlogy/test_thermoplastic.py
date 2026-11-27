"""Glass-transition property model, crystalline mixing, crystallising thermoelastic point, latent heat and shrinkage
(``toporlogy.core.thermoplastic``). Checks TP1, TP2, TP5, MX1-MX4 of docs/foundations/prereg/TP_thermoplastic.md
(thresholds pre-registered and committed in b7fade3 before this file was run). All numbers are synthetic, not material
data.

References are independent of the production code: closed forms written in this file (own logistic, own softplus
integral, own Hashin-Shtrikman formulas, own Avrami), ``scipy.integrate.quad``, central finite differences, a dense numpy
stiffness. A comparison with a scalar loop of the same kernel is labelled "regression"."""
from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad

from toporlogy.core import micromechanics as mm
from toporlogy.core import properties as P
from toporlogy.core import tensor as T
from toporlogy.core import thermoplastic as TPM
from toporlogy.core.constitutive import (Kinematics, MaterialRequest, StateStore, TimeStep, drive_material_point)
from toporlogy.core.cure import Nakamura

TM, W, LO, HI = 380.0, 6.0, 300.0, 480.0
EG, ER, NU_A = 2.0e9, 2.0e7, 0.38
AG, AR = 6e-5, 2e-4
EC, NU_C = 5e9, 0.30
XMAX, BETA, LAT, RHO, CP = 0.4, -0.06, 1.2e8, 1300.0, 1600.0


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------- the test's own closed forms
def s_of(Tk: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-(np.asarray(Tk, float) - TM) / W))


def E_closed(Tk: Any) -> Any:
    return EG * (ER / EG) ** s_of(Tk)


def alpha_closed(Tk: Any) -> Any:
    return AG + (AR - AG) * s_of(Tk)


def eth_closed(Tk: Any, T0: float) -> Any:
    sp = lambda x: np.log1p(np.exp(x))                                                      # noqa: E731
    return AG * (Tk - T0) + (AR - AG) * W * (sp((np.asarray(Tk, float) - TM) / W) - sp((T0 - TM) / W))


def KG_from_E(E: Any, nu: float) -> tuple[Any, Any]:
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def hs_lo(phi: Any, Ka: float, Ga: float, Kc: float, Gc: float) -> tuple[Any, Any]:
    """HS lower bound, soft phase 1 = amorphous (matrix), hard phase 2 = crystal, f2 = phi."""
    f1 = 1.0 - np.asarray(phi, float)
    K = Ka + phi / (1.0 / (Kc - Ka) + 3.0 * f1 / (3.0 * Ka + 4.0 * Ga))
    G = Ga + phi / (1.0 / (Gc - Ga) + 6.0 * f1 * (Ka + 2.0 * Ga) / (5.0 * Ga * (3.0 * Ka + 4.0 * Ga)))
    return K, G


def hs_hi(phi: Any, Ka: float, Ga: float, Kc: float, Gc: float) -> tuple[Any, Any]:
    """HS upper bound: matrix = the hard phase (crystal), inclusion = the soft phase with f = 1 - phi."""
    f1 = np.asarray(phi, float)
    K = Kc + (1.0 - f1) / (1.0 / (Ka - Kc) + 3.0 * f1 / (3.0 * Kc + 4.0 * Gc))
    G = Gc + (1.0 - f1) / (1.0 / (Ga - Gc) + 6.0 * f1 * (Kc + 2.0 * Gc) / (5.0 * Gc * (3.0 * Kc + 4.0 * Gc)))
    return K, G


def E_of_KG(K: Any, G: Any) -> Any:
    return 9.0 * K * G / (3.0 * K + G)


# ---------------------------------------------------------------- fixtures
def E_fn() -> TPM.LogisticTransition:
    return TPM.LogisticTransition(EG, ER, TM, W, (LO, HI), "Pa", "synthetic", form="log")


def a_fn() -> TPM.LogisticTransition:
    return TPM.LogisticTransition(AG, AR, TM, W, (LO, HI), "1/K", "synthetic")


def moduli_amorphous() -> P.IsotropicModuliOfT:
    return P.IsotropicModuliOfT(E=E_fn(), nu=P.Constant(NU_A, "1", "synthetic"))


def moduli_crystal() -> P.IsotropicModuliOfT:
    return P.IsotropicModuliOfT(E=P.Constant(EC, "Pa", "synthetic"), nu=P.Constant(NU_C, "1", "synthetic"))


def const_nakamura(K0: float = 0.05, n: float = 2.0) -> Nakamura:
    return Nakamura(n, P.Constant(K0, "1/s", "synthetic"))


def make_point(nak: Nakamura | None = None, T_ref: float = 400.0, beta: float = BETA, X0: float = 0.0,
               rule: str = "hs_lower") -> TPM.CrystallisingThermoElasticPoint:
    return TPM.CrystallisingThermoElasticPoint(
        nak or const_nakamura(), TPM.CrystallineMixture(moduli_amorphous(), moduli_crystal(), rule),
        P.ThermalExpansion(a_fn(), T_ref), beta, LAT, XMAX, X0)


Z6 = np.zeros(6)


def drive(model: Any, ctrl: Any, times: np.ndarray, temps: np.ndarray, T0: float, strain: np.ndarray | None = None):
    n = len(times)
    st = np.zeros((n, 6)) if strain is None else strain
    return drive_material_point(model, st, np.zeros((n, 6)), np.asarray(ctrl, bool), tol=1e-6, times=times,
                                temperatures=temps, t0=0.0, temperature0=T0)


# ============================================================================================== TP2 transition
def test_tp2a_exact_points() -> None:
    E, a = E_fn(), a_fn()
    L3 = W * math.log(3.0)
    assert rel(E(TM), math.sqrt(EG * ER)) <= 1e-14 and rel(E(TM), 2e8) <= 1e-14
    assert rel(E(TM + L3), EG * (ER / EG) ** 0.75) <= 1e-14
    assert rel(E(TM - L3), EG * (ER / EG) ** 0.25) <= 1e-14
    assert rel(a(TM), 1.3e-4) <= 1e-14
    assert rel(a(TM + L3), AG + 0.75 * (AR - AG)) <= 1e-14
    assert rel(a(TM - L3), AG + 0.25 * (AR - AG)) <= 1e-14
    assert rel(a.derivative(TM), (AR - AG) / (4.0 * W)) <= 1e-13
    assert rel(E.derivative(TM), 2e8 * math.log(ER / EG) / (4.0 * W)) <= 1e-13


def test_tp2b_monotone_and_peak_slope() -> None:
    E, a = E_fn(), a_fn()
    g = np.linspace(LO, HI, 2001)
    assert np.all(np.diff(E(g)) < 0.0) and np.all(np.diff(a(g)) > 0.0)
    fine = np.linspace(LO, HI, 24001)
    assert abs(float(fine[np.argmax(np.abs(a.derivative(fine)))]) - TM) <= 0.01


def test_tp2c_derivative_vs_fd() -> None:
    h = 1e-3
    for f in (E_fn(), a_fn()):
        Tk = np.linspace(304.0, 476.0, 9)
        fd = (f(Tk + h) - f(Tk - h)) / (2 * h)
        assert rel(f.derivative(Tk), fd) <= 1e-6


def test_tp2d_integral_and_thermal_strain() -> None:
    a = a_fn()
    for T0, T1 in ((300.0, 420.0), (340.0, 400.0), (420.0, 330.0)):
        ref, _ = quad(alpha_closed, T0, T1, epsabs=0, epsrel=1e-13, limit=200)
        assert rel(a.integral(T0, T1), ref) <= 1e-12
        assert rel(a.integral(T0, T1), eth_closed(T1, T0)) <= 1e-12
    te = P.ThermalExpansion(a, 300.0)
    for Tk in (320.0, 380.0, 440.0):
        ref, _ = quad(alpha_closed, 300.0, Tk, epsabs=0, epsrel=1e-13, limit=200)
        assert rel(te.strain(Tk), ref) <= 1e-12
    with pytest.raises(ValueError):
        E_fn().integral(300.0, 400.0)


def test_tp2e_refusal_outside_range() -> None:
    for f in (E_fn(), a_fn()):
        for bad in (299.9, 480.1, float("nan"), 0.0, -1.0):
            with pytest.raises(ValueError):
                f.value(bad)
            with pytest.raises(ValueError):
                f.derivative(bad)
        with pytest.raises(ValueError):
            f.integral(299.9, 400.0)
        with pytest.raises(ValueError):
            f.integral(300.0, 480.1)
        assert math.isfinite(float(f.value(300.0))) and math.isfinite(float(f.value(480.0)))
    with pytest.raises(ValueError):
        TPM.LogisticTransition(AG, AR, TM, W, (LO, HI), "1/K", "synthetic", extrapolation="clamp")


def test_tp2f_no_universal_tg() -> None:
    params = inspect.signature(TPM.LogisticTransition).parameters
    for name in ("p_glassy", "p_rubbery", "T_mid", "width", "domain", "units", "source"):
        assert params[name].default is inspect.Parameter.empty, name
    with pytest.raises(TypeError):
        TPM.LogisticTransition(AG, AR, width=W, domain=(LO, HI), units="1/K", source="synthetic")  # type: ignore[call-arg]
    ok = dict(p_glassy=AG, p_rubbery=AR, T_mid=TM, width=W, domain=(LO, HI), units="1/K", source="synthetic")
    bad = [dict(width=0.0), dict(width=-1.0), dict(p_rubbery=AG), dict(T_mid=250.0), dict(T_mid=500.0),
           dict(domain=(LO, math.inf)), dict(form="log", p_glassy=-1.0), dict(form="log", p_rubbery=0.0),
           dict(form="cubic"), dict(temperature_units="degC"), dict(source=""), dict(width=float("nan")),
           dict(positive=True, p_glassy=-1e-5)]
    for b in bad:
        with pytest.raises(ValueError):
            TPM.LogisticTransition(**{**ok, **b})
    other = TPM.LogisticTransition(**{**ok, "T_mid": 360.0})
    assert abs(float(other(TM)) - float(a_fn()(TM))) > 1e-5               # the midpoint is a parameter, not a constant


# ============================================================================================== TP1 history
def tp1_path() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(1.0, 186.0)
    temps = np.where(t <= 120.0, 300.0 + t, np.where(t <= 140.0, 420.0, 420.0 - 2.0 * (t - 140.0)))
    return t, temps, 1e-3 * t / 185.0


def tp1_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, temps, exx = tp1_path()
    eth = np.array([quad(alpha_closed, 300.0, Tk, epsabs=0, epsrel=1e-13, limit=200)[0] for Tk in temps])
    E = E_closed(temps)
    return E * (exx - eth), eth, E


def tp1_strain(exx: np.ndarray) -> np.ndarray:
    st = np.zeros((len(exx), 6))
    st[:, 0] = exx
    return st


def test_tp1a_ramp_hold_cool_thermoelastic() -> None:
    t, temps, exx = tp1_path()
    sig_ref, eth, E = tp1_reference()
    model = P.ThermoElasticPoint(moduli_amorphous(), P.ThermalExpansion(a_fn(), 300.0))
    h = drive(model, [True, False, False, False, False, False], t, temps, 300.0, tp1_strain(exx))
    assert rel(h.stress[:, 0], sig_ref) <= 1e-9
    assert np.max(np.abs(h.stress[:, 1:])) <= 1e-6
    eyy = eth - NU_A * (exx - eth)
    assert np.max(np.abs(h.strain[:, 1] - eyy)) <= 1e-12 and np.max(np.abs(h.strain[:, 2] - eyy)) <= 1e-12
    assert E.max() / E.min() > 50.0                                       # the path really goes through the transition


def test_tp1b_same_history_without_crystallisation() -> None:
    t, temps, exx = tp1_path()
    sig_ref, _eth, _E = tp1_reference()
    nak = Nakamura(2.0, P.Constant(0.0, "1/s", "synthetic"))
    model = make_point(nak, T_ref=300.0)
    h = drive(model, [True, False, False, False, False, False], t, temps, 300.0, tp1_strain(exx))
    assert rel(h.stress[:, 0], sig_ref) <= 1e-9
    assert all(float(s["X"][0]) == 0.0 for s in h.states)


def test_tp1c_negative_control_secant_for_instantaneous() -> None:
    t, temps, exx = tp1_path()
    sig_ref, _eth, E = tp1_reference()
    wrong = E * (exx - alpha_closed(temps) * (temps - 300.0))             # instantaneous CTE used as a secant CTE
    assert float(np.max(np.abs(wrong - sig_ref)) / np.max(np.abs(sig_ref))) > 1e-2


# ============================================================================================== mixing
A_PH, C_PH = mm.IsoPhase.from_E_nu(3e7, 0.38), mm.IsoPhase.from_E_nu(5e9, 0.30)


def closed_rule(rule: str, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Ka, Ga, Kc, Gc = A_PH.K, A_PH.G, C_PH.K, C_PH.G
    Kv, Gv = (1 - phi) * Ka + phi * Kc, (1 - phi) * Ga + phi * Gc
    Kr, Gr = 1.0 / ((1 - phi) / Ka + phi / Kc), 1.0 / ((1 - phi) / Ga + phi / Gc)
    if rule == "voigt":
        return Kv, Gv
    if rule == "reuss":
        return Kr, Gr
    if rule == "hill":
        return 0.5 * (Kv + Kr), 0.5 * (Gv + Gr)
    return (hs_lo if rule == "hs_lower" else hs_hi)(phi, Ka, Ga, Kc, Gc)


def inside_voigt_reuss(K: float, G: float, phi: float) -> bool:
    Kv, Gv = closed_rule("voigt", np.array(phi))
    Kr, Gr = closed_rule("reuss", np.array(phi))
    return bool(Kr * (1 - 1e-12) <= K <= Kv * (1 + 1e-12) and Gr * (1 - 1e-12) <= G <= Gv * (1 + 1e-12))


PHIS = np.linspace(0.0, 1.0, 21)


@pytest.mark.parametrize("rule", TPM.MIXING_RULES)
def test_mx1_rules_inside_bounds_and_monotone(rule: str) -> None:
    got = np.array([TPM.mix_isotropic(p, A_PH, C_PH, rule) for p in PHIS])
    for (K, G), p in zip(got, PHIS):
        assert inside_voigt_reuss(K, G, p), (rule, p)
    # production bounds as a second check (not the reference of the rule itself)
    for p, (K, G) in zip(PHIS, got):
        f = np.array([1.0 - p, p])
        Kv, Gv = mm.voigt(f, [A_PH, C_PH])
        Kr, Gr = mm.reuss(f, [A_PH, C_PH])
        assert Kr * (1 - 1e-12) <= K <= Kv * (1 + 1e-12) and Gr * (1 - 1e-12) <= G <= Gv * (1 + 1e-12)
        if rule.startswith("hs"):
            b = mm.hs_bounds(f, [A_PH, C_PH])
            assert b.K_lower * (1 - 1e-12) <= K <= b.K_upper * (1 + 1e-12)
            assert b.G_lower * (1 - 1e-12) <= G <= b.G_upper * (1 + 1e-12)
            assert Kr * (1 - 1e-12) <= b.K_lower <= b.K_upper <= Kv * (1 + 1e-12)
    assert np.all(np.diff(got[:, 0]) > 0.0) and np.all(np.diff(got[:, 1]) > 0.0)


@pytest.mark.parametrize("rule", TPM.MIXING_RULES)
def test_mx2_closed_forms_and_endpoints(rule: str) -> None:
    got = np.array([TPM.mix_isotropic(p, A_PH, C_PH, rule) for p in PHIS])
    Kc_, Gc_ = closed_rule(rule, PHIS)
    assert rel(got[:, 0], Kc_) <= 1e-12 and rel(got[:, 1], Gc_) <= 1e-12
    assert rel(got[0], [A_PH.K, A_PH.G]) <= 1e-13
    assert rel(got[-1], [C_PH.K, C_PH.G]) <= 1e-13


def test_mx3_refusal_and_batching() -> None:
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            TPM.mix_isotropic(bad, A_PH, C_PH)
    with pytest.raises(ValueError):
        TPM.mix_isotropic(0.5, A_PH, C_PH, "linear_in_E")
    with pytest.raises(ValueError):
        TPM.mix_isotropic(0.5, mm.VOID, C_PH)
    with pytest.raises(ValueError):
        TPM.mix_isotropic(0.5, A_PH, 3.0)                                                # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TPM.CrystallineMixture(moduli_amorphous(), P.IsotropicModuliOfT(E=P.Constant(5e3, "MPa", "synthetic"),
                                                                        nu=P.Constant(0.3, "1", "synthetic")))
    with pytest.raises(ValueError):
        TPM.CrystallineMixture(moduli_amorphous(), moduli_crystal(), "nope")
    mix = TPM.CrystallineMixture(moduli_amorphous(), moduli_crystal())
    Tk, ph = np.array([310.0, 380.0, 450.0]), np.array([0.0, 0.2, 0.4])
    K, G = mix.moduli(Tk, ph)
    for i in range(3):                                                                    # regression (same kernel)
        Ka, Ga = KG_from_E(float(E_closed(Tk[i])), NU_A)
        ref = TPM.mix_isotropic(float(ph[i]), mm.IsoPhase(Ka, Ga), mm.IsoPhase.from_E_nu(EC, NU_C), "hs_lower")
        assert abs(K[i] - ref[0]) <= 1e-14 * ref[0] and abs(G[i] - ref[1]) <= 1e-14 * ref[1]
    with pytest.raises(ValueError):
        mix.moduli(500.0, 0.1)                                                            # outside the validity range


def test_mx4_negative_control_bound_check_detects_violation() -> None:
    Kv, Gv = closed_rule("voigt", np.array(0.5))
    assert inside_voigt_reuss(float(Kv), float(Gv), 0.5)
    assert not inside_voigt_reuss(float(Kv) * 1.01, float(Gv), 0.5)


# ============================================================================================== TP5 latent heat
def avrami_X(t: Any) -> Any:
    return -np.expm1(-0.0025 * np.asarray(t, float) ** 2)                                 # K = 0.05, n = 2: k = 0.0025


def test_tp5a_latent_heat_balance() -> None:
    n = 30
    times = 2.0 * np.arange(1.0, n + 1.0)
    h = drive(make_point(), np.ones(6, bool), times, np.full(n, 400.0), 400.0)
    heat = np.array([d["latent_heat"] for d in h.diagnostics])
    t_prev = np.concatenate([[0.0], times[:-1]])
    assert np.max(np.abs(heat - LAT * (avrami_X(times) - avrami_X(t_prev)))) <= 1e-10 * LAT
    X_end = 1.0 - math.exp(-9.0)
    assert abs(avrami_X(60.0) - X_end) <= 1e-15
    assert abs(heat.sum() - LAT * X_end) <= 1e-12 * LAT
    dT = sum(TPM.adiabatic_temperature_rise(LAT, hj / LAT, RHO, CP) for hj in heat)
    assert abs(dT - LAT * X_end / (RHO * CP)) <= 1e-12 * dT
    val, _ = quad(lambda t: LAT * 2.0 * 0.0025 * t * math.exp(-0.0025 * t * t), 0.0, 60.0, epsabs=0, epsrel=1e-13)
    assert abs(val - LAT * X_end) <= 1e-10 * LAT
    assert abs(TPM.adiabatic_temperature_rise(LAT, 0.5, RHO, CP) - 28.846153846153847) <= 1e-14 * 28.85
    assert np.all(heat > 0.0)
    for args in ((LAT, -0.1, RHO, CP), (0.0, 0.1, RHO, CP), (LAT, 0.1, 0.0, CP), (LAT, 0.1, RHO, -1.0),
                 (LAT, float("nan"), RHO, CP)):
        with pytest.raises(ValueError):
            TPM.adiabatic_temperature_rise(*args)


def test_tp5b_free_shrinkage_with_cooling() -> None:
    K_tab = P.PiecewiseLinear([300.0, 400.0, 480.0], [0.0, 0.03, 0.0], "1/s", "synthetic")
    model = make_point(Nakamura(2.0, K_tab), T_ref=400.0)
    n = 40
    t = np.arange(1.0, n + 1.0)
    temps = 440.0 - 2.0 * t
    h = drive(model, np.zeros(6, bool), t, temps, 440.0)
    theta = np.array([quad(lambda s: float(np.interp(440.0 - 2.0 * s, [300.0, 400.0, 480.0], [0.0, 0.03, 0.0])),
                           0.0, tk, points=[20.0] if tk > 20.0 else None, epsabs=0, epsrel=1e-13, limit=200)[0]
                      for tk in t])
    X = -np.expm1(-theta ** 2)
    eth = np.array([float(eth_closed(Tk, 400.0)) for Tk in temps])
    e = eth + BETA * X / 3.0
    expected = np.zeros((n, 6))
    expected[:, :3] = e[:, None]
    assert np.max(np.abs(h.strain - expected)) <= 1e-11
    assert np.max(np.abs(h.stress)) <= 1e-5
    assert 0.3 < X[-1] < 0.9
    assert np.all((h.strain[:, 0] - eth) < 0.0)                           # crystallisation shrinks
    assert np.all(np.diff([float(s["X"][0]) for s in h.states]) >= 0.0)
    assert np.max(np.abs(np.array([float(s["X"][0]) for s in h.states]) - X)) <= 1e-12


def amorphous_KG_400() -> tuple[float, float]:
    return KG_from_E(float(E_closed(400.0)), NU_A)


def tp5_isothermal(ctrl: Any, **kw: Any):
    n = 30
    times = 2.0 * np.arange(1.0, n + 1.0)
    return times, drive(make_point(**kw), ctrl, times, np.full(n, 400.0), 400.0)


def mixture_400(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Ka, Ga = amorphous_KG_400()
    Kc, Gc = KG_from_E(EC, NU_C)
    return hs_lo(XMAX * X, Ka, Ga, Kc, Gc)


def test_tp5c_constrained_uniaxial_bar_tension() -> None:
    times, h = tp5_isothermal([True, False, False, False, False, False])
    X = avrami_X(times)
    K, G = mixture_400(X)
    sig = -E_of_KG(K, G) * BETA * X / 3.0
    assert rel(h.stress[:, 0], sig) <= 1e-9
    assert np.all(h.stress[:, 0] > 0.0)                                   # shrinkage held back = tension
    assert np.max(np.abs(h.stress[:, 1:])) <= 1e-6


def test_tp5d_fully_constrained() -> None:
    times, h = tp5_isothermal(np.ones(6, bool))
    X = avrami_X(times)
    K, _G = mixture_400(X)
    sig = -K * BETA * X
    for c in range(3):
        assert rel(h.stress[:, c], sig) <= 1e-9
    assert np.max(np.abs(h.stress[:, 3:])) <= 1e-9
    assert np.all(sig > 0.0)


def test_tp5e_negative_control_wrong_sign_detected() -> None:
    times, h = tp5_isothermal([True, False, False, False, False, False])
    X = avrami_X(times)
    K, G = mixture_400(X)
    flipped = -E_of_KG(K, G) * (-BETA) * X / 3.0                          # expansion instead of shrinkage
    assert np.all(flipped < 0.0)
    assert float(np.max(np.abs(h.stress[:, 0] - flipped)) / np.max(np.abs(h.stress[:, 0]))) >= 1.9
    for b in (0.06, 0.0):
        with pytest.raises(ValueError):
            make_point(beta=b)


def test_tp5f_reference_state_and_boundedness() -> None:
    n = 10
    times = 2.0 * np.arange(1.0, n + 1.0)
    h = drive(make_point(X0=0.1), np.zeros(6, bool), times, np.full(n, 400.0), 400.0)
    theta0 = math.sqrt(-math.log(0.9))
    X = -np.expm1(-(theta0 + 0.05 * times) ** 2)
    assert np.max(np.abs(h.strain[:, :3] - (BETA * (X - 0.1) / 3.0)[:, None])) <= 1e-11
    assert np.max(np.abs(h.stress)) <= 1e-5
    model = make_point(X0=0.1)
    kin = Kinematics(np.zeros((1, 3, 3)), "small")
    x = np.array([0.1])
    for _ in range(3):                                                    # coarse steps drive X to 1 without clipping
        r = model.evaluate(kin, None, {"X": x}, MaterialRequest(step=TimeStep(0.0, 1000.0, 400.0, 400.0)))
        x1 = r.trial_state["X"]
        assert 0.1 <= x[0] <= x1[0] <= 1.0 and np.all(np.isfinite(r.stress))
        x = x1
    assert x[0] == 1.0


def test_tp5g_contract_and_tangent() -> None:
    model = make_point()
    assert model.physical is True and model.kinematics == "small" and model.state_fields == ("X",)
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
    X1 = -math.expm1(-0.0025)                                             # K = 0.05 1/s, n = 2, dt = 1 s, X0 = 0
    Ka, Ga = KG_from_E(float(E_closed(320.0)), NU_A)
    Kc, Gc = KG_from_E(EC, NU_C)
    K, G = hs_lo(XMAX * X1, Ka, Ga, Kc, Gc)
    one = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    Id = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(one, one) / 3.0
    dense = K * np.outer(one, one) + 2.0 * G * Id
    assert rel(r.tangent[0], dense) <= 1e-12

    def sig(e: np.ndarray) -> np.ndarray:
        return model.evaluate(Kinematics(T.voigt_to_strain(e)[None], "small"), None, store.committed, req).stress[0]

    h = 1e-7
    fd = np.stack([(sig(e6 + h * np.eye(6)[i]) - sig(e6 - h * np.eye(6)[i])) / (2 * h) for i in range(6)], axis=1)
    assert rel(r.tangent[0], fd) <= 1e-6
    nak, mix, exp = const_nakamura(), TPM.CrystallineMixture(moduli_amorphous(), moduli_crystal()), \
        P.ThermalExpansion(a_fn(), 400.0)
    for kw in (dict(beta_c=0.0), dict(beta_c=0.1), dict(beta_c=float("nan")), dict(latent_heat=0.0),
               dict(latent_heat=-1.0), dict(X_max=0.0), dict(X_max=1.1), dict(X0=-0.1), dict(X0=1.1),
               dict(beta_c=True)):
        args = {**dict(beta_c=BETA, latent_heat=LAT, X_max=XMAX, X0=0.0), **kw}
        with pytest.raises(ValueError):
            TPM.CrystallisingThermoElasticPoint(nak, mix, exp, **args)
    for ctor in (lambda: TPM.CrystallisingThermoElasticPoint(1.0, mix, exp, BETA, LAT),
                 lambda: TPM.CrystallisingThermoElasticPoint(nak, moduli_amorphous(), exp, BETA, LAT),
                 lambda: TPM.CrystallisingThermoElasticPoint(nak, mix, a_fn(), BETA, LAT)):
        with pytest.raises(ValueError):
            ctor()


def test_tp5h_trial_does_not_change_committed() -> None:
    model = make_point(X0=0.1)
    store = StateStore(model, 1)
    before = store.committed["X"].copy()
    assert not store.committed["X"].flags.writeable
    req = MaterialRequest(step=TimeStep(0.0, 5.0, 400.0, 400.0))
    kin = Kinematics(np.zeros((1, 3, 3)), "small")
    r1 = model.evaluate(kin, None, store.committed, req)
    store.set_trial(r1.trial_state)
    assert r1.trial_state["X"][0] > 0.1
    assert np.array_equal(store.committed["X"], before)
    store.rollback()
    assert np.array_equal(store.committed["X"], before)
    r2 = model.evaluate(kin, None, store.committed, req)
    assert np.array_equal(r1.trial_state["X"], r2.trial_state["X"])
    store.set_trial(r2.trial_state)
    store.commit()
    assert np.array_equal(store.committed["X"], r1.trial_state["X"])
    assert not store.committed["X"].flags.writeable
    with pytest.raises(ValueError):
        model.evaluate(kin, None, {"X": np.array([1.5])}, req)
