"""Temperature-dependent properties, thermal strain and the thermoelastic material point
(``toporlogy.core.properties``). Checks PP1-PP17 of docs/foundations/prereg/P_properties.md (thresholds pre-registered
2026-11-27 01:27 +0300, committed before this file was run). All numbers are synthetic, not material data."""
from __future__ import annotations


import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad

from toporlogy.core import properties as P
from toporlogy.core.constitutive import (Kinematics, MaterialRequest, StateStore, TimeStep, drive_material_point)

TREF = 300.0
DOM = (280.0, 350.0)


def rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def E_syn(units: str = "Pa") -> P.Polynomial:
    """Synthetic E(T) = 3e9 - 1e7 (T - 300) Pa on 280-350 K."""
    return P.Polynomial([3e9, -1e7], TREF, DOM, units, "synthetic", positive=True)


def nu_syn(v: float = 0.35) -> P.Constant:
    return P.Constant(v, "1", "synthetic", domain=DOM, positive=True)


def moduli_syn() -> P.IsotropicModuliOfT:
    return P.IsotropicModuliOfT(E=E_syn(), nu=nu_syn())


# ============================================================================================== PP1 / PP2 moduli
def test_pp1_synthetic_moduli_closed_form_and_fd() -> None:
    m = moduli_syn()
    T = np.linspace(285.0, 345.0, 13)
    E = 3e9 - 1e7 * (T - 300.0)
    K, G = E / 0.9, E / 2.7
    mod, dmod = m.moduli(T), m.dmoduli(T)
    assert rel(mod.E, E) <= 1e-13 and rel(mod.K, K) <= 1e-13 and rel(mod.G, G) <= 1e-13
    assert rel(mod.nu, np.full_like(T, 0.35)) <= 1e-13
    assert rel(mod.lam, K - 2 * G / 3) <= 1e-13
    assert rel(dmod.E, np.full_like(T, -1e7)) <= 1e-12
    assert rel(dmod.K, np.full_like(T, -1e7 / 0.9)) <= 1e-12 and rel(dmod.G, np.full_like(T, -1e7 / 2.7)) <= 1e-12
    assert float(np.max(np.abs(dmod.nu))) <= 1e-12 * 0.35
    assert rel(dmod.lam, np.full_like(T, -1e7 / 0.9 + 2e7 / 2.7 / 3)) <= 1e-12

    def fK(t):
        return (3e9 - 1e7 * (t - 300.0)) / (3 * (1 - 2 * 0.35))

    def fG(t):
        return (3e9 - 1e7 * (t - 300.0)) / (2 * (1 + 0.35))

    h = 1e-2
    assert rel(dmod.K, (fK(T + h) - fK(T - h)) / (2 * h)) <= 1e-7
    assert rel(dmod.G, (fG(T + h) - fG(T - h)) / (2 * h)) <= 1e-7


def test_pp2_endpoints_out_of_range_batch() -> None:
    m = moduli_syn()
    assert rel(m.E(280.0), 3.2e9) <= 1e-13 and rel(m.E(350.0), 2.5e9) <= 1e-13
    assert rel(m.K(280.0), 3.2e9 / 0.9) <= 1e-13 and rel(m.G(350.0), 2.5e9 / 2.7) <= 1e-13
    for bad in (279.999, 350.001, float("nan"), float("inf"), -float("inf"), 0.0, -5.0):
        with pytest.raises(ValueError):
            m.moduli(bad)
        with pytest.raises(ValueError):
            m.dmoduli(bad)
    with pytest.raises(ValueError):
        m.moduli(np.array([300.0, 351.0]))
    out = m.E(np.array([280.0, 300.0, 350.0]))
    assert out.shape == (3,)
    assert m.stiffness(np.array([280.0, 300.0])).shape == (2, 6, 6)
    assert m.stiffness(300.0).shape == (6, 6)


# ============================================================================================== PP3 / PP4 pairs
def _pair_functions() -> dict[str, P.Polynomial]:
    mk = lambda c, u: P.Polynomial(c, TREF, DOM, u, "synthetic")                      # noqa: E731
    return {"E": mk([3e9, -1e7, 2e4], "Pa"), "K": mk([2.5e9, -8e6, 1e4], "Pa"), "G": mk([1.4e9, -4e6, 5e3], "Pa"),
            "nu": mk([0.30, 1e-4, 1e-6], "1")}


def _mp_moduli(pair: tuple[str, str], coeffs: dict[str, list[float]], T):
    """E, nu, K, G, lam in mpmath from the pair, with the E01 conversions written out here."""
    v = {k: sum(mp.mpf(c) * (T - TREF) ** i for i, c in enumerate(coeffs[k])) for k in pair}
    s = set(pair)
    if s == {"E", "nu"}:
        K, G = v["E"] / (3 * (1 - 2 * v["nu"])), v["E"] / (2 * (1 + v["nu"]))
    elif s == {"E", "K"}:
        K, G = v["K"], 3 * v["K"] * v["E"] / (9 * v["K"] - v["E"])
    elif s == {"E", "G"}:
        K, G = v["E"] * v["G"] / (3 * (3 * v["G"] - v["E"])), v["G"]
    elif s == {"K", "G"}:
        K, G = v["K"], v["G"]
    elif s == {"K", "nu"}:
        K, G = v["K"], 3 * v["K"] * (1 - 2 * v["nu"]) / (2 * (1 + v["nu"]))
    else:
        K, G = 2 * v["G"] * (1 + v["nu"]) / (3 * (1 - 2 * v["nu"])), v["G"]
    return {"E": 9 * K * G / (3 * K + G), "nu": (3 * K - 2 * G) / (2 * (3 * K + G)), "K": K, "G": G,
            "lam": K - 2 * G / 3}


PAIRS = [("E", "nu"), ("E", "K"), ("E", "G"), ("K", "G"), ("K", "nu"), ("G", "nu")]


@pytest.mark.parametrize("pair", PAIRS)
def test_pp3_pp4_all_pairs_chain_rule_vs_mpmath(pair: tuple[str, str]) -> None:
    fns = _pair_functions()
    coeffs = {k: list(f.coefficients) for k, f in fns.items()}
    m = P.IsotropicModuliOfT(**{k: fns[k] for k in pair})
    mp.mp.dps = 50
    for T in (285.0, 300.0, 333.3):
        mod, dmod = m.moduli(T), m.dmoduli(T)
        for name in ("E", "nu", "K", "G", "lam"):
            f = lambda t, name=name: _mp_moduli(pair, coeffs, t)[name]                 # noqa: E731
            ref = f(mp.mpf(T))
            dref = mp.diff(f, mp.mpf(T))
            assert abs(float(getattr(mod, name)) - float(ref)) <= 1e-10 * abs(float(ref)), (pair, name, T)
            assert abs(float(getattr(dmod, name)) - float(dref)) <= 1e-10 * abs(float(dref)), (pair, name, T)
    # PP4: the given pair is reproduced
    Ts = np.linspace(282.0, 348.0, 17)
    for k in pair:
        assert rel(getattr(m.moduli(Ts), k), fns[k].value(Ts)) <= 1e-13, (pair, k)


# ============================================================================================== PP5 piecewise linear
def test_pp5_piecewise_linear() -> None:
    x = np.array([280.0, 300.0, 330.0, 350.0])
    y = np.array([5.0, 3.0, 4.5, 4.0])
    pl = P.PiecewiseLinear(x, y, "GPa", "synthetic")
    assert np.array_equal(pl.value(x), y)
    assert pl.value(290.0) == pytest.approx(4.0, rel=1e-14) and pl.value(340.0) == pytest.approx(4.25, rel=1e-14)
    assert pl.integral(280.0, 350.0) == pytest.approx(277.5, rel=1e-13)
    assert pl.integral(290.0, 340.0) == pytest.approx(191.25, rel=1e-13)
    assert pl.integral(340.0, 290.0) == pytest.approx(-191.25, rel=1e-13)
    assert pl.integral(310.0, 310.0) == 0.0
    assert pl.derivative(290.0) == pytest.approx(-0.1, rel=1e-13)
    assert pl.derivative(315.0) == pytest.approx(0.05, rel=1e-13)
    assert pl.derivative(300.0) == pytest.approx(0.05, rel=1e-13)                   # right-hand at an interior knot
    assert P.PiecewiseLinear(x, y, "GPa", "synthetic", side="left").derivative(300.0) == pytest.approx(-0.1, rel=1e-13)
    assert pl.derivative(280.0) == pytest.approx(-0.1, rel=1e-13)
    assert pl.derivative(350.0) == pytest.approx(-0.025, rel=1e-13)                 # last knot: the adjacent slope
    assert [bool(pl.nonsmooth(t)) for t in (280.0, 290.0, 300.0, 330.0, 350.0)] == [False, False, True, True, False]
    assert pl.smoothness.startswith("C0")
    assert pl.value(np.array([[285.0, 295.0], [305.0, 345.0]])).shape == (2, 2)


# ============================================================================================== PP6 PCHIP
def test_pp6_pchip_monotone_integral_derivative() -> None:
    x = np.arange(280.0, 351.0, 10.0)
    y = np.array([3.2, 3.1, 2.95, 2.7, 2.65, 2.3, 2.2, 2.1])
    pc = P.Pchip(x, y, "GPa", "synthetic", positive=True)
    assert rel(pc.value(x), y) <= 1e-14
    g = np.linspace(280.0, 350.0, 2001)
    v = pc.value(g)
    assert float(np.max(np.diff(v))) <= 1e-12 * (y.max() - y.min())               # non-increasing
    assert v.min() >= y.min() - 1e-12 and v.max() <= y.max() + 1e-12
    for a, b in ((280.0, 350.0), (293.0, 337.5), (350.0, 300.0)):
        ref = quad(lambda t: float(pc.value(t)), a, b, epsabs=0.0, epsrel=1e-13, limit=200)[0]
        assert abs(pc.integral(a, b) - ref) <= 1e-10 * abs(ref)
    pts = np.array([283.3, 296.1, 311.7, 327.9, 344.4])
    h = 1e-6
    fd = (pc.value(pts + h) - pc.value(pts - h)) / (2 * h)
    assert rel(pc.derivative(pts), fd) <= 1e-6
    assert pc.smoothness.startswith("C1")


# ============================================================================================== PP7 polynomial
def test_pp7_polynomial() -> None:
    c = [2.0, -0.03, 4e-4, -1e-6]
    pn = P.Polynomial(c, 310.0, (290.0, 340.0), "GPa", "synthetic", positive=True)
    T = np.linspace(290.0, 340.0, 11)
    x = T - 310.0
    ref = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
    assert rel(pn.value(T), ref) <= 1e-13
    dref = c[1] + x * (2 * c[2] + x * 3 * c[3])
    assert rel(pn.derivative(T), dref) <= 1e-13
    F = lambda xx: c[0] * xx + c[1] * xx ** 2 / 2 + c[2] * xx ** 3 / 3 + c[3] * xx ** 4 / 4                # noqa: E731
    assert rel(pn.integral(290.0, 340.0), F(30.0) - F(-20.0)) <= 1e-13
    q = quad(lambda t: float(pn.value(t)), 290.0, 340.0, epsabs=0.0, epsrel=1e-13)[0]
    assert rel(pn.integral(290.0, 340.0), q) <= 1e-12
    cst = P.Constant(7.0, "GPa", "synthetic")
    assert cst.integral(300.0, 320.0) == pytest.approx(140.0, rel=1e-14) and cst.derivative(310.0) == 0.0


# ============================================================================================== PP8 / PP9 thermal strain
A, B = 1.2e-5, 2e-8


def _alpha(domain=(240.0, 400.0)) -> P.Polynomial:
    return P.Polynomial([A, B], TREF, domain, "1/K", "synthetic", positive=True)


def test_pp8_thermal_strain_secant_and_roundtrip() -> None:
    te = P.ThermalExpansion(_alpha(), TREF)
    T = np.linspace(240.0, 380.0, 57)
    dT = T - TREF
    ref = A * dT + 0.5 * B * dT ** 2
    assert float(np.max(np.abs(te.strain(T) - ref))) <= 1e-13 * float(np.max(np.abs(ref)))
    assert te.strain(TREF) == 0.0
    sec = te.secant(T)
    ref_sec = A + 0.5 * B * dT
    assert rel(sec, ref_sec) <= 1e-13
    assert te.secant(TREF) == pytest.approx(A, rel=1e-14)                              # limit at T = T_ref
    assert te.secant(TREF + 1e-12) == pytest.approx(A, rel=1e-14)
    ts = P.ThermalExpansion.from_secant(P.Polynomial([A, 0.5 * B], TREF, (240.0, 400.0), "1/K", "synthetic"), TREF)
    assert float(np.max(np.abs(ts.strain(T) - ref))) <= 1e-13 * float(np.max(np.abs(ref)))
    assert rel(ts.alpha(T), A + B * dT) <= 1e-13
    assert rel(te.alpha(T), A + B * dT) <= 1e-13
    assert rel(ts.secant(T), ref_sec) <= 1e-13


def test_pp9_negative_control_secant_used_as_instantaneous() -> None:
    dT = 80.0
    right = A * dT + 0.5 * B * dT ** 2
    sec_poly = P.Polynomial([A, 0.5 * B], TREF, (240.0, 400.0), "1/K", "synthetic")
    wrong = sec_poly.integral(TREF, TREF + dT)                                           # secant CTE integrated as if instantaneous
    assert wrong == pytest.approx(A * dT + 0.25 * B * dT ** 2, rel=1e-13)
    assert abs(wrong / right - 1.0) > 1e-3                                              # the confusion is detectable
    prod = P.ThermalExpansion(_alpha(), TREF).strain(TREF + dT)
    assert abs(prod - right) < abs(prod - wrong)
    assert abs(prod / right - 1.0) <= 1e-13
    # b = 0: secant and instantaneous coincide, the control then sees no difference
    c0 = P.Polynomial([A, 0.0], TREF, (240.0, 400.0), "1/K", "synthetic")
    assert abs(c0.integral(TREF, TREF + dT) / (A * dT) - 1.0) <= 1e-14


# ============================================================================================== PP10 enthalpy
def test_pp10_enthalpy() -> None:
    cp = P.Polynomial([900.0, 0.5], TREF, (250.0, 400.0), "J/(kg K)", "synthetic", positive=True)
    H = P.Enthalpy(cp, TREF)
    T = np.array([250.0, 280.0, 300.0, 333.0, 400.0])
    d = T - TREF
    ref = 900.0 * d + 0.25 * d ** 2
    assert float(np.max(np.abs(H.H(T) - ref))) <= 1e-10 * float(np.max(np.abs(ref)))
    x = np.array([250.0, 300.0, 350.0, 400.0])
    for f in (P.PiecewiseLinear(x, [900.0, 950.0, 1000.0, 1100.0], "J/(kg K)", "synthetic", positive=True),
              P.Pchip(x, [900.0, 950.0, 1000.0, 1100.0], "J/(kg K)", "synthetic", positive=True)):
        Hh = P.Enthalpy(f, TREF)
        for t in (260.0, 300.0, 377.0, 400.0):
            q = quad(lambda s: float(f.value(s)), TREF, t, epsabs=0.0, epsrel=1e-13)[0]
            assert abs(Hh.H(t) - q) <= 1e-10 * max(abs(q), 1.0)
    with pytest.raises(ValueError):
        P.Constant(-1.0, "J/(kg K)", "synthetic", positive=True)
    with pytest.raises(ValueError):
        P.Enthalpy(P.Constant(900.0, "J/(kg K)", "synthetic"), TREF)                    # c_p not declared positive
    with pytest.raises(ValueError):
        P.Enthalpy(P.Constant(900.0, "J", "synthetic", positive=True), TREF)           # wrong units


# ============================================================================================== PP11 tensor CTE
def test_pp11_tensor_cte_and_rotation() -> None:
    alpha = {"xx": P.Polynomial([1e-5, 1e-8], TREF, (240.0, 400.0), "1/K", "synthetic"),
             "yy": P.Constant(2e-5, "1/K", "synthetic"), "zz": P.Constant(3e-5, "1/K", "synthetic"),
             "xy": P.Constant(1e-6, "1/K", "synthetic")}
    te = P.ThermalExpansionTensor(alpha, TREF)
    T = np.array([250.0, 300.0, 340.0, 390.0])
    d = T - TREF
    ref = np.zeros((4, 3, 3))
    ref[:, 0, 0] = 1e-5 * d + 0.5e-8 * d ** 2
    ref[:, 1, 1] = 2e-5 * d
    ref[:, 2, 2] = 3e-5 * d
    ref[:, 0, 1] = ref[:, 1, 0] = 1e-6 * d
    e = te.strain(T)
    assert e.shape == (4, 3, 3) and np.array_equal(e, np.swapaxes(e, 1, 2))
    assert float(np.max(np.abs(e - ref))) <= 1e-13 * float(np.max(np.abs(ref)))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    er = P.ThermalExpansionTensor(alpha, TREF, rotation=R).strain(T)
    want = np.einsum("ij,njk,lk->nil", R, ref, R)
    assert float(np.max(np.abs(er - want))) <= 1e-13 * float(np.max(np.abs(ref)))
    assert np.allclose(er[:, 0, 0], ref[:, 1, 1], rtol=1e-13, atol=0) and np.allclose(er[:, 1, 1], ref[:, 0, 0],
                                                                                  rtol=1e-13, atol=0)
    assert np.array_equal(er, np.swapaxes(er, 1, 2))
    assert te.strain(TREF).shape == (3, 3) and not np.any(te.strain(TREF))


# ============================================================================================== PP12-PP16 material point
def _model() -> P.ThermoElasticPoint:
    alpha = P.Polynomial([A, B], TREF, DOM, "1/K", "synthetic", positive=True)
    return P.ThermoElasticPoint(moduli_syn(), alpha, T_ref=TREF)


def _ramp(k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return 10.0 * np.arange(1, k + 1), TREF + 5.0 * np.arange(1, k + 1)


def _eth(T: np.ndarray) -> np.ndarray:
    d = T - TREF
    return A * d + 0.5 * B * d ** 2


def _E(T: np.ndarray) -> np.ndarray:
    return 3e9 - 1e7 * (T - TREF)


def test_pp12_free_expansion_zero_stress_even_with_varying_E() -> None:
    t, T = _ramp()
    z = np.zeros((len(t), 6))
    h = drive_material_point(_model(), z, z, [False] * 6, tol=1e-6, times=t, temperatures=T, t0=0.0, temperature0=TREF)
    assert float(np.max(np.abs(h.stress))) <= 1e-5
    want = np.zeros((len(t), 6))
    want[:, :3] = _eth(T)[:, None]
    assert float(np.max(np.abs(h.strain - want))) <= 1e-12 * float(np.max(want))
    assert _E(T[-1]) < _E(T[0])                                                         # E really changed along the path


def test_pp13_uniaxial_bar_axially_constrained() -> None:
    t, T = _ramp()
    z = np.zeros((len(t), 6))
    h = drive_material_point(_model(), z, z, [True, False, False, False, False, False], tol=1e-6, times=t,
                             temperatures=T, t0=0.0, temperature0=TREF)
    eth = _eth(T)
    assert rel(h.stress[:, 0], -_E(T) * eth) <= 1e-9
    assert float(np.max(np.abs(h.stress[:, 1:]))) <= 1e-5
    assert rel(h.strain[:, 1], 1.35 * eth) <= 1e-9 and rel(h.strain[:, 2], 1.35 * eth) <= 1e-9
    assert float(np.max(np.abs(h.strain[:, 0]))) == 0.0


def test_pp14_three_d_fully_constrained() -> None:
    t, T = _ramp()
    z = np.zeros((len(t), 6))
    h = drive_material_point(_model(), z, z, [True] * 6, tol=1e-6, times=t, temperatures=T, t0=0.0, temperature0=TREF)
    K = _E(T) / 0.9
    ref = -3.0 * K * _eth(T)
    for i in range(3):
        assert rel(h.stress[:, i], ref) <= 1e-12
    assert float(np.max(np.abs(h.stress[:, 3:]))) <= 1e-12 * float(np.max(np.abs(ref)))


def _dense_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.full((6, 6), 0.0)
    C[:3, :3] = lam
    C[:3, :3] += 2 * mu * np.eye(3)
    C[3:, 3:] = mu * np.eye(3)
    return C


def test_pp15_tangent_dense_and_fd() -> None:
    model = _model()
    rng = np.random.default_rng(0)
    H = rng.normal(size=(3, 3, 3)) * 1e-3
    eps = 0.5 * (H + np.swapaxes(H, 1, 2))
    req = MaterialRequest(step=TimeStep(0.0, 1.0, 320.0, 310.0))
    resp = model.evaluate(Kinematics(eps, "small"), None, {}, req)
    Cd = _dense_C(3e9 - 1e7 * 20.0, 0.35)
    for i in range(3):
        assert rel(resp.tangent[i], Cd) <= 1e-12
    # finite differences of the stress at fixed T
    h = 1e-7
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for j, (a, b) in enumerate(idx):
        d = np.zeros((3, 3))
        d[a, b] = d[b, a] = (h if a == b else 0.5 * h)                               # engineering shear step h
        sp = model.evaluate(Kinematics(eps + d, "small"), None, {}, req).stress
        sm = model.evaluate(Kinematics(eps - d, "small"), None, {}, req).stress
        col = (sp - sm) / (2 * h)
        assert rel(col, np.broadcast_to(Cd[:, j], col.shape)) <= 1e-6, j


def test_pp16_contract() -> None:
    m = _model()
    assert m.physical is True and m.kinematics == "small" and m.state_fields == () and m.initial_state(4) == {}
    eps = np.zeros((2, 3, 3))
    kin = Kinematics(eps, "small")
    with pytest.raises(ValueError):
        m.evaluate(kin, None, {}, MaterialRequest())                                     # no step
    with pytest.raises(ValueError):
        m.evaluate(kin, None, {}, MaterialRequest(step=TimeStep(0.0, 1.0)))              # step without temperature
    with pytest.raises(ValueError):
        m.evaluate(kin, None, {}, MaterialRequest(step=TimeStep(0.0, 1.0, 400.0)))       # outside the property domain
    with pytest.raises(ValueError):
        TimeStep(0.0, 1.0, -5.0)
    r = m.evaluate(kin, None, {}, MaterialRequest(step=TimeStep(0.0, 1.0, 320.0), tangent=False))
    assert r.energy is None and r.dissipation is None and r.tangent is None and r.trial_state == {}
    assert "elastic_strain_energy" in r.diagnostics
    store = StateStore(m, 2)
    assert store.checkpoint() == {}
    # per-point temperatures (n,)
    r2 = m.evaluate(kin, None, {}, MaterialRequest(step=TimeStep(0.0, 1.0, np.array([300.0, 340.0]))))
    assert r2.stress.shape == (2, 6)
    assert abs(r2.stress[0, 0]) <= 1e-9 and r2.stress[1, 0] < 0.0                    # held at zero strain: compressive
    with pytest.raises(ValueError):
        P.ThermoElasticPoint(moduli_syn(), P.ThermalExpansion(_alpha((280.0, 350.0)), TREF), T_ref=310.0)
    with pytest.raises(ValueError):
        P.ThermoElasticPoint(moduli_syn(), _alpha((280.0, 350.0)))                       # bare function needs T_ref


# ============================================================================================== PP17 explicit failures
def test_pp17_property_construction_errors() -> None:
    good_x, good_y = [280.0, 300.0, 350.0], [1.0, 2.0, 3.0]
    for bad_x in ([280.0, 280.0, 350.0], [350.0, 300.0, 280.0], [280.0, float("nan"), 350.0], [0.0, 300.0, 350.0],
                  [-10.0, 300.0, 350.0]):
        with pytest.raises(ValueError):
            P.PiecewiseLinear(bad_x, good_y, "GPa", "synthetic")
        with pytest.raises(ValueError):
            P.Pchip(bad_x, good_y, "GPa", "synthetic")
    for cls in (P.PiecewiseLinear, P.Pchip):
        with pytest.raises(ValueError):
            cls(good_x, [1.0, 2.0], "GPa", "synthetic")                                  # length mismatch
        with pytest.raises(ValueError):
            cls([300.0], [1.0], "GPa", "synthetic")                                      # < 2 knots
        with pytest.raises(ValueError):
            cls(good_x, [1.0, float("inf"), 3.0], "GPa", "synthetic")
        with pytest.raises(ValueError):
            cls(good_x, [1.0, 0.0, 3.0], "GPa", "synthetic", positive=True)
        with pytest.raises(ValueError):
            cls(good_x, [1.0, -2.0, 3.0], "GPa", "synthetic", positive=True)
        for pol in ("clamp", "linear", "constant", ""):
            with pytest.raises(ValueError):
                cls(good_x, good_y, "GPa", "synthetic", extrapolation=pol)
        for u in ("", "   ", None, 5):
            with pytest.raises(ValueError):
                cls(good_x, good_y, u, "synthetic")
        for tu in ("degC", "C", "°C", "Celsius", "kelvin", ""):
            with pytest.raises(ValueError):
                cls(good_x, good_y, "GPa", "synthetic", temperature_units=tu)
        with pytest.raises(ValueError):
            cls(good_x, good_y, "GPa", "")                                               # no source
    with pytest.raises(ValueError):
        P.Constant(-1.0, "Pa", "synthetic", positive=True)
    with pytest.raises(ValueError):
        P.Constant(float("nan"), "Pa", "synthetic")
    with pytest.raises(ValueError):
        P.Constant(1.0, "Pa", "synthetic", extrapolation="clamp")
    for dom in ((350.0, 280.0), (-1.0, 300.0), (float("nan"), 300.0), (300.0, 300.0)):
        with pytest.raises(ValueError):
            P.Constant(1.0, "Pa", "synthetic", domain=dom)
    with pytest.raises(ValueError):
        P.Polynomial([1.0], 300.0, (280.0, float("inf")), "Pa", "synthetic")
    with pytest.raises(ValueError):
        P.Polynomial([1.0, float("nan")], 300.0, DOM, "Pa", "synthetic")
    with pytest.raises(ValueError):
        P.Polynomial([1.0], -300.0, DOM, "Pa", "synthetic")
    with pytest.raises(ValueError):
        P.Polynomial([1.0, -0.1], 300.0, (280.0, 330.0), "Pa", "synthetic", positive=True)       # negative at the end
    # interior dip below zero while both end points are positive
    with pytest.raises(ValueError):
        P.Polynomial([0.05, -0.02, 1e-3], 300.0, (280.0, 330.0), "Pa", "synthetic", positive=True)
    P.Polynomial([0.05, -0.02, 1e-3], 300.0, (280.0, 330.0), "Pa", "synthetic", positive=False)   # allowed without it


def test_pp17_evaluation_errors() -> None:
    pl = P.PiecewiseLinear([280.0, 300.0, 350.0], [1.0, 2.0, 3.0], "GPa", "synthetic")
    pc = P.Pchip([280.0, 300.0, 350.0], [1.0, 2.0, 3.0], "GPa", "synthetic")
    pn = P.Polynomial([1.0, 0.1], 300.0, DOM, "GPa", "synthetic")
    for f in (pl, pc, pn):
        for bad in (float("nan"), float("inf"), 0.0, -1.0, 279.9, 350.1):
            with pytest.raises(ValueError):
                f.value(bad)
            with pytest.raises(ValueError):
                f.derivative(bad)
            with pytest.raises(ValueError):
                f.integral(300.0, bad)
            with pytest.raises(ValueError):
                f.integral(bad, 300.0)
        with pytest.raises(ValueError):
            f.value(np.array([300.0, 400.0]))
    with pytest.raises(ValueError):
        P.Constant(1.0, "Pa", "synthetic", domain=DOM).value(360.0)


def test_pp17_moduli_errors() -> None:
    E, nu = E_syn(), nu_syn()
    K = P.Polynomial([2.5e9, -8e6], TREF, DOM, "Pa", "synthetic")
    G = P.Polynomial([1.1e9, -4e6], TREF, DOM, "Pa", "synthetic")
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E, nu=nu, K=K)                                            # redundant (three)
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E, nu=nu, K=K, G=G)
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E)
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT()
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E, lam=K)                                                 # unsupported modulus
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E, mu=G)
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E, nu=0.35)                                               # a number, not a function
    # inadmissible values
    cst = lambda v, u="Pa": P.Constant(v, u, "synthetic", domain=DOM)                      # noqa: E731
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(3e9), G=cst(0.9e9))                                     # E >= 3G: K <= 0
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(3e9), nu=cst(0.5, "1"))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(3e9), nu=cst(-1.0, "1"))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(3e9), nu=cst(0.6, "1"))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(-3e9), nu=cst(0.3, "1"))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(K=cst(-1e9), G=cst(1e9))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(K=cst(1e9), G=cst(0.0))
    with pytest.raises(ValueError):                                                      # E crosses zero inside the domain
        P.IsotropicModuliOfT(E=P.Polynomial([1e9, -5e7], TREF, DOM, "Pa", "synthetic"), nu=nu)
    # a narrow inadmissible dip between the constructor's sample points is caught at evaluation
    dip = P.PiecewiseLinear([280.0, 300.03, 300.04, 300.05, 320.0], [3e9, 3e9, -1e9, 3e9, 3e9], "Pa", "synthetic")
    m = P.IsotropicModuliOfT(E=dip, nu=P.Constant(0.3, "1", "synthetic"))
    with pytest.raises(ValueError):
        m.moduli(300.04)
    assert m.E(310.0) == pytest.approx(3e9, rel=1e-13)
    # domains and units
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=P.Constant(3e9, "Pa", "synthetic", domain=(280.0, 300.0)),
                             nu=P.Constant(0.3, "1", "synthetic", domain=(310.0, 350.0)))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(K=cst(1e9, "Pa"), G=cst(1e6, "MPa"))                          # mixed units: convert first
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(3e9, "psi"), nu=nu)
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=E, nu=cst(0.3, "Pa"))
    with pytest.raises(ValueError):
        P.IsotropicModuliOfT(E=cst(3e9, "1/K"), nu=nu)


def test_pp17_thermal_errors() -> None:
    a = P.Polynomial([A, B], TREF, (240.0, 400.0), "1/K", "synthetic")
    with pytest.raises(ValueError):
        P.ThermalExpansion(P.Constant(1e-5, "1/degC", "synthetic"), TREF)                  # unit not accepted
    with pytest.raises(ValueError):
        P.ThermalExpansion(P.Constant(1e-5, "ppm/K", "synthetic"), TREF)
    for bad in (0.0, -300.0, float("nan"), 500.0, 100.0):
        with pytest.raises(ValueError):
            P.ThermalExpansion(a, bad)
    with pytest.raises(ValueError):
        P.ThermalExpansion(a, TREF).strain(500.0)
    with pytest.raises(ValueError):
        P.ThermalExpansion(a, TREF).secant(float("nan"))
    with pytest.raises(ValueError):
        P.ThermalExpansionTensor({}, TREF)
    with pytest.raises(ValueError):
        P.ThermalExpansionTensor({"xx": a, "ab": a}, TREF)
    with pytest.raises(ValueError):
        P.ThermalExpansionTensor({"xx": a}, TREF, rotation=np.diag([1.0, 1.0, -1.0]))     # improper rotation
    with pytest.raises(ValueError):
        P.ThermalExpansionTensor({"xx": a}, TREF, rotation=2.0 * np.eye(3))
    with pytest.raises(ValueError):
        P.ThermalExpansionTensor({"xx": P.Constant(1e-5, "1/degC", "synthetic")}, TREF)
    with pytest.raises(ValueError):
        P.ThermalExpansion(a, TREF).strain(np.array([300.0, 450.0]))
