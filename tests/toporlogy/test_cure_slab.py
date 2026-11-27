"""Coupled thermo-chemical cure slab and its one-way residual stress (``toporlogy.fem.mesh.cure_slab``). Checks TS7a-TS7g
of docs/foundations/prereg/TS7_cure_slab.md (thresholds pre-registered 2026-11-27 04:45 +0300, committed in 7b3b925
before this file was run). All numbers are synthetic, not material data.

References are independent of the production code: the closed recurrence of backward Euler for n = 1, a scalar coupled
solve by ``scipy.optimize.brentq`` on closed forms, and a method of lines written here (three-point finite differences on
a fine grid, ``scipy.integrate.solve_ivp`` BDF with tight tolerances, the rate law and Arrhenius formula coded in this
file). Comparisons labelled "regression" call the same kernel as the production code."""
from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from toporlogy.core import cure as C
from toporlogy.core import properties as P
from toporlogy.fem.mesh.cure_slab import CureSlab, residual_stress
from toporlogy.fem.mesh.history import StepRejected
from toporlogy.fem.mesh.mesh import rectangle

R_GAS = 8.314462618
# ---- synthetic problem P (prereg)
L, LX = 0.02, 0.004
RHO, CP, KC, DH = 1200.0, 1500.0, 0.2, 7.5e4
T0, HC = 350.0, 40.0
K_REF, Q_ACT, N_ORD = 1e-3, 4e4, 1.5
T_OUT = np.array([600.0, 1200.0, 1800.0, 2400.0, 3000.0])


def kin_p() -> C.NthOrder:
    return C.NthOrder(C.Arrhenius.from_reference(K_REF, T0, Q_ACT), N_ORD)


def make_slab(n: int, bc: str = "robin", k: float = KC, dh: float = DH, kin: C.CureKinetics | None = None,
              alpha0: float = 0.0) -> CureSlab:
    mesh = rectangle(1, n, LX, L)
    slab = CureSlab(mesh, kin or kin_p(), C.ReactionHeat(dh), RHO, CP, k, T0, alpha0)
    if bc == "robin":
        slab.add_robin(mesh.facet_sets["top"], HC, T0)
    elif bc == "dirichlet":
        slab.add_dirichlet(mesh.node_sets["top"], T0)
    elif bc != "none":
        raise ValueError(bc)
    return slab


def abar(slab: CureSlab) -> float:
    w = slab.weights
    return float(np.sum(w * slab.alpha) / np.sum(w))


def run_prod(n: int, dt: float, bc: str = "robin", dh: float = DH) -> dict[str, np.ndarray]:
    slab = make_slab(n, bc, dh=dh)
    tc, ts, ab = [], [], []
    for t in T_OUT:
        slab.advance_to(float(t), dt)
        tc.append(slab.T[0])
        ts.append(slab.T[n])
        ab.append(abar(slab))
    return {"Tc": np.array(tc), "Ts": np.array(ts), "abar": np.array(ab)}


# ====================================================================================== independent MOL reference
@cache
def mol(N: int, bc: str, rtol: float) -> dict[str, np.ndarray]:
    """Method of lines for the half slab [0, L]: symmetric at y = 0, Robin or Dirichlet at y = L. Three-point central
    differences (ghost nodes), unknowns (T_i, alpha_i), BDF with tight tolerances; rate law and Arrhenius formula coded
    here (n = 1.5)."""
    h = L / N
    kap = KC / (RHO * CP)

    def rate(T: np.ndarray, a: np.ndarray) -> np.ndarray:
        return K_REF * np.exp(-Q_ACT / R_GAS * (1.0 / T - 1.0 / T0)) * np.maximum(1.0 - a, 0.0) ** N_ORD

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        T, a = y[:N + 1], y[N + 1:]
        f = rate(T, a)
        d2 = np.empty(N + 1)
        d2[1:-1] = (T[2:] - 2.0 * T[1:-1] + T[:-2]) / h ** 2
        d2[0] = 2.0 * (T[1] - T[0]) / h ** 2
        if bc == "robin":
            d2[-1] = 2.0 * (T[-2] - T[-1]) / h ** 2 - 2.0 * HC / (KC * h) * (T[-1] - T0)
        dT = kap * d2 + DH / CP * f
        if bc == "dirichlet":
            dT[-1] = 0.0
        return np.concatenate([dT, f])

    S = sp.lil_matrix((2 * (N + 1), 2 * (N + 1)))
    for i in range(N + 1):
        for j in (i - 1, i, i + 1):
            if 0 <= j <= N:
                S[i, j] = 1
        S[i, N + 1 + i] = 1
        S[N + 1 + i, i] = 1
        S[N + 1 + i, N + 1 + i] = 1
    y0 = np.concatenate([np.full(N + 1, T0), np.zeros(N + 1)])
    sol = solve_ivp(rhs, (0.0, float(T_OUT[-1])), y0, method="BDF", t_eval=T_OUT, rtol=rtol,
                    atol=np.concatenate([np.full(N + 1, 1e-9), np.full(N + 1, 1e-12)]), jac_sparsity=S.tocsr())
    assert sol.success and sol.y.shape[1] == len(T_OUT)
    T, a = sol.y[:N + 1], sol.y[N + 1:]
    return {"Tc": T[0].copy(), "Ts": T[-1].copy(), "abar": np.trapezoid(a, dx=h, axis=0) / L}


def dT_ref(ref: dict[str, np.ndarray]) -> float:
    d = float(ref["Tc"].max() - T0)
    assert d >= 5.0, f"set-up invalid: exotherm of the reference is only {d} K"
    return d


def errors(prod: dict[str, np.ndarray], ref: dict[str, np.ndarray], scale: float) -> tuple[float, float]:
    e_t = max(np.max(np.abs(prod["Tc"] - ref["Tc"])), np.max(np.abs(prod["Ts"] - ref["Ts"]))) / scale
    e_a = float(np.max(np.abs(prod["abar"] - ref["abar"])))
    return float(e_t), e_a


def order(e_coarse: float, e_fine: float) -> float:
    return math.log2(e_coarse / e_fine)


# ============================================================================================ TS7b0 the reference
def test_ts7b0_reference_is_accurate() -> None:
    r400, r800, r400t = mol(400, "robin", 1e-9), mol(800, "robin", 1e-9), mol(400, "robin", 1e-11)
    scale = dT_ref(r800)
    assert errors(r400, r800, scale)[0] <= 2e-5
    assert errors(r400, r800, scale)[1] <= 2e-5
    assert errors(r400, r400t, scale)[0] <= 1e-6
    assert errors(r400, r400t, scale)[1] <= 1e-6
    d = dT_ref(mol(800, "dirichlet", 1e-9))
    assert d >= 5.0


# ============================================================================================ TS7a adiabatic limit
def recurrence(a0: float, k: float, dt: float, n: int) -> np.ndarray:
    """Derivation: n = 1 backward Euler with a constant rate, a_{j+1} = (a_j + k dt)/(1 + k dt)."""
    a = [a0]
    for _ in range(n):
        a.append((a[-1] + k * dt) / (1.0 + k * dt))
    return np.array(a)


@pytest.mark.parametrize("cond", [0.2, 0.0])
def test_ts7a_adiabatic_closed_form_constant_rate(cond: float) -> None:
    n_steps, dt, a0 = 50, 60.0, 0.1
    kin = C.NthOrder(C.Arrhenius(1e-3, 0.0), 1.0)
    slab = make_slab(4, "none", k=cond, kin=kin, alpha0=a0)
    aj = recurrence(a0, 1e-3, dt, n_steps)
    for j in range(1, n_steps + 1):
        slab.step(dt)
        expect = T0 + DH / CP * (aj[j] - a0)
        rise = DH / CP * (aj[j] - a0)
        assert np.max(np.abs(slab.T - expect)) <= 1e-10 * rise
        assert np.ptp(slab.T) <= 1e-9
        assert np.max(np.abs(slab.alpha - aj[j])) <= 1e-12
    assert slab.T[0] - T0 == pytest.approx(DH / CP * (aj[-1] - a0), rel=1e-10)


@pytest.mark.parametrize("cond", [0.2, 0.0])
def test_ts7a_adiabatic_coupled_arrhenius_vs_scalar_brentq(cond: float) -> None:
    n_steps, dt, a0 = 50, 60.0, 0.1
    kin = C.NthOrder(C.Arrhenius.from_reference(K_REF, T0, Q_ACT), 1.0)
    slab = make_slab(4, "none", k=cond, kin=kin, alpha0=a0)
    Tj, aj = T0, a0
    for _ in range(n_steps):
        def a_new(T: float, aj: float = aj) -> float:
            kd = K_REF * math.exp(-Q_ACT / R_GAS * (1.0 / T - 1.0 / T0)) * dt
            return (aj + kd) / (1.0 + kd)

        Tj = brentq(lambda T, Tj=Tj, aj=aj: CP * (T - Tj) - DH * (a_new(T) - aj), Tj, Tj + 60.0, xtol=1e-14,
                    rtol=4 * np.finfo(float).eps)
        aj = a_new(Tj)
        slab.step(dt)
    assert Tj - T0 > 10.0                                                     # the exotherm is not negligible
    assert abs(slab.T[0] - Tj) <= 1e-10 * (Tj - T0)
    assert np.ptp(slab.T) <= 1e-9
    # (iii) regression: the stored alpha is a replay of core.cure.advance on the saved temperature history
    alpha = np.full(slab.shape_q, a0)
    for h in slab.history:
        alpha = np.asarray(kin.advance(alpha, slab.quadrature_temperature(h.T_prev),
                                       slab.quadrature_temperature(h.T_next), h.dt))
    assert np.max(np.abs(slab.alpha - alpha)) <= 1e-13 * np.max(alpha)


# ================================================================================================ TS7b convergence
def test_ts7b1_time_convergence_first_order() -> None:
    ref = mol(800, "robin", 1e-9)
    scale = dT_ref(ref)
    dts = [100.0, 50.0, 25.0, 12.5, 6.25]
    e = [errors(run_prod(32, dt), ref, scale)[0] for dt in dts]
    assert all(e[i + 1] < e[i] for i in range(len(e) - 1)), e
    p1, p2 = order(e[2], e[3]), order(e[3], e[4])
    assert 0.85 <= p1 <= 1.15, (p1, e)
    assert 0.85 <= p2 <= 1.15, (p2, e)


def test_ts7b2_space_convergence_second_order() -> None:
    ref = mol(800, "robin", 1e-9)
    scale = dT_ref(ref)
    et, ea = [], []
    for n in (2, 4, 8, 16):
        a, b = run_prod(n, 3.0), run_prod(n, 1.5)
        star = {k: 2.0 * b[k] - a[k] for k in a}                            # first-order time error removed
        t_, a_ = errors(star, ref, scale)
        et.append(t_)
        ea.append(a_)
    assert all(et[i + 1] < et[i] for i in range(3)), et
    p48, p816 = order(et[1], et[2]), order(et[2], et[3])
    assert 1.7 <= p48 <= 2.3, (p48, et)
    assert 1.5 <= p816 <= 2.5, (p816, et)
    assert et[3] <= 3e-3 and ea[3] <= 3e-3, (et, ea)


def test_ts7b3_dirichlet_surface() -> None:
    ref = mol(800, "dirichlet", 1e-9)
    scale = dT_ref(ref)
    e_t, e_a = errors(run_prod(32, 1.5, "dirichlet"), ref, scale)
    assert e_t <= 1e-2 and e_a <= 1e-2, (e_t, e_a)


def test_ts7b_negative_control_without_exotherm() -> None:
    ref = mol(800, "robin", 1e-9)
    scale = dT_ref(ref)
    e_t, e_a = errors(run_prod(32, 6.25, dh=1e-9), ref, scale)
    assert e_t >= 0.9 and e_a >= 0.1, (e_t, e_a)


# =============================================================================================== TS7c energy balance
def test_ts7c_energy_balance_and_controls() -> None:
    n, dt, nsteps = 16, 10.0, 60
    slab = make_slab(n)
    y = np.linspace(0.0, L, n + 1)
    assert abs(slab.weights.sum() - LX * L) <= 1e-13 * LX * L                    # the weights integrate the area
    mism, mism_a, mism_b = [], [], []
    tot = {"stored": 0.0, "reaction": 0.0, "inflow": 0.0}
    for _ in range(nsteps):
        T0_, a0_ = slab.T.copy(), slab.alpha.copy()
        r = slab.step(dt)
        T1 = slab.T
        stored = RHO * CP * LX * np.trapezoid(T1[:n + 1] - T0_[:n + 1], y)       # column x = 0 (x-uniform field)
        reaction = RHO * DH * float(np.sum(slab.weights * (slab.alpha - a0_)))
        inflow = dt * HC * LX * (T0 - T1[n])                                    # backward Euler: flux at t_{n+1}
        inflow_old = dt * HC * LX * (T0 - T0_[n])                               # start-of-step flux (explicit)
        sc = max(abs(stored), abs(reaction), abs(inflow))
        mism.append(abs(stored - reaction - inflow) / sc)
        mism_a.append(abs(stored - inflow) / sc)
        mism_b.append(abs(stored - reaction - inflow_old) / sc)
        for got, want in ((r.stored, stored), (r.reaction, reaction), (r.robin_heat, inflow)):
            assert abs(got - want) <= 1e-9 * sc
        assert r.dirichlet_heat == 0.0
        tot["stored"] += stored
        tot["reaction"] += reaction
        tot["inflow"] += inflow
    assert max(mism) <= 1e-9, max(mism)
    assert abs(tot["stored"] - tot["reaction"] - tot["inflow"]) <= 1e-9 * max(abs(v) for v in tot.values())
    assert tot["reaction"] > 0.0 and tot["stored"] > 0.0                         # exothermic, heating: signs
    assert max(mism_a) >= 1e-2, max(mism_a)                                     # (a) without the reaction term
    assert max(mism_b) >= 1e-4, max(mism_b)                                     # (b) explicit-in-time flux


# =================================================================================================== TS7d rollback
def test_ts7d_rollback_and_validation() -> None:
    def advanced() -> CureSlab:
        s = make_slab(8)
        s.advance_to(1000.0, 25.0)
        return s

    slab = advanced()
    T_b, a_b, t_b, nh = slab.T.copy(), slab.alpha.copy(), slab.t, len(slab.history)
    with pytest.raises(StepRejected):
        slab.step(25.0, max_iter=1)
    assert np.array_equal(slab.T, T_b) and np.array_equal(slab.alpha, a_b)
    assert slab.t == t_b and len(slab.history) == nh
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            slab.step(bad)
    with pytest.raises(ValueError):
        slab.step(25.0, max_iter=0)
    assert np.array_equal(slab.T, T_b) and np.array_equal(slab.alpha, a_b) and len(slab.history) == nh
    assert not slab.alpha.flags.writeable and not slab.T.flags.writeable
    slab.step(25.0)                                                             # the retry with the default limit
    fresh = advanced()
    fresh.step(25.0)
    assert np.array_equal(slab.T, fresh.T) and np.array_equal(slab.alpha, fresh.alpha)
    assert slab.t == fresh.t == 1025.0 and len(slab.history) == nh + 1


def test_construction_validation() -> None:
    mesh = rectangle(1, 2, LX, L)
    args: dict[str, Any] = dict(rho=RHO, cp=CP, conductivity=KC, T0=T0)
    ok = CureSlab(mesh, kin_p(), C.ReactionHeat(DH), **args)
    assert ok.t == 0.0
    for bad in ({"rho": 0.0}, {"cp": -1.0}, {"conductivity": -0.1}, {"T0": 0.0}, {"alpha0": 1.5}, {"thickness": 0.0}):
        with pytest.raises(ValueError):
            CureSlab(mesh, kin_p(), C.ReactionHeat(DH), **{**args, **bad})
    with pytest.raises(ValueError):
        CureSlab(mesh, "not kinetics", C.ReactionHeat(DH), **args)       # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ok.add_dirichlet(np.array([0]), 350.0)
        ok.add_dirichlet(np.array([0]), 350.0)


# ======================================================================================== TS7e mechanics, uniform
def synth_moduli() -> tuple[P.IsotropicModuliOfT, P.IsotropicModuliOfT]:
    mu = P.IsotropicModuliOfT(E=P.Constant(1e5, "Pa", "synthetic"), nu=P.Constant(0.45, "1", "synthetic"))
    mc = P.IsotropicModuliOfT(E=P.Constant(3e9, "Pa", "synthetic"), nu=P.Constant(0.35, "1", "synthetic"))
    return mu, mc


def synth_expansion() -> P.ThermalExpansion:
    return P.ThermalExpansion(P.Constant(5e-5, "1/K", "synthetic"), 350.0)


def closed_form_sigma(alpha: float, T: float, gel: float) -> tuple[float, float, float]:
    """sigma_xx = -E e/(1 - nu) at a uniform state (derivation in the prereg), from the test's own K, G, E, nu."""
    Ku, Gu = 1e5 / (3 * (1 - 0.9)), 1e5 / (2 * 1.45)
    Kc, Gc = 3e9 / (3 * (1 - 0.7)), 3e9 / (2 * 1.35)
    K, G = (1 - alpha) * Ku + alpha * Kc, (1 - alpha) * Gu + alpha * Gc
    E, nu = 9 * K * G / (3 * K + G), (3 * K - 2 * G) / (2 * (3 * K + G))
    e = 5e-5 * (T - 350.0) + (-0.03 / 3.0) * max(alpha - gel, 0.0)
    return -E * e / (1.0 - nu), E, e


def test_ts7e_constrained_and_free_uniform_slab() -> None:
    n_steps, dt, a0 = 50, 60.0, 0.1
    kin = C.NthOrder(C.Arrhenius(1e-3, 0.0), 1.0)
    slab = make_slab(4, "none", kin=kin, alpha0=a0)
    slab.advance_to(n_steps * dt, dt)
    mu, mc = synth_moduli()
    shr = C.ChemicalShrinkage(-0.03, 0.1)
    con = residual_stress(slab, mu, mc, synth_expansion(), shr, "in_plane", tol=1e-12)
    # set-up: reference force 1e3 N on a supported DOF (the stress-free state has rounding-level internal forces)
    free = residual_stress(slab, mu, mc, synth_expansion(), shr, "free", tol=1e-12, force_scale=1e3)
    aj = recurrence(a0, 1e-3, dt, n_steps)
    exp_sig = np.array([closed_form_sigma(aj[j], T0 + DH / CP * (aj[j] - a0), 0.1)[0] for j in range(1, n_steps + 1)])
    scale = np.max(np.abs(exp_sig))
    assert exp_sig[-1] > 0.0 and scale > 1e6                                    # shrinkage + constraint: tension
    for j in range(n_steps):
        s = con.stress_history[j]
        assert np.max(np.abs(s[..., 0] - exp_sig[j])) <= 1e-9 * scale
        assert np.max(np.abs(s[..., 1])) <= 1e-9 * scale and np.max(np.abs(s[..., 2])) <= 1e-9 * scale
        assert np.max(np.abs(free.stress_history[j])) <= 1e-9 * scale
    # negative control: E instead of E/(1 - nu) is wrong by the factor nu >= 0.35
    wrong = np.array([closed_form_sigma(aj[j], T0 + DH / CP * (aj[j] - a0), 0.1)[1:] for j in range(1, n_steps + 1)])
    sig_wrong = -wrong[:, 0] * wrong[:, 1]
    assert np.max(np.abs(sig_wrong - exp_sig) / np.abs(exp_sig)) >= 0.2
    with pytest.raises(ValueError):
        residual_stress(slab, mu, mc, synth_expansion(), shr, "clamped")


# ================================================================================ TS7f one-way chain, TS7g Newton
@cache
def coupled_gradient_run() -> tuple[CureSlab, list[Any]]:
    slab = make_slab(16)
    return slab, slab.advance_to(3000.0, 25.0)


def test_ts7f_one_way_consistency_and_sigma_yy_mean() -> None:
    slab, _ = coupled_gradient_run()
    mu, mc = synth_moduli()
    mech = residual_stress(slab, mu, mc, synth_expansion(), C.ChemicalShrinkage(-0.03, 0.0), "in_plane", tol=1e-12)
    assert np.max(np.abs(mech.alpha - slab.alpha)) <= 1e-13 * np.max(slab.alpha)
    sxx_max = float(np.max(np.abs(mech.stress[..., 0])))
    assert sxx_max >= 1e5
    mean_yy = np.sum(slab.weights * mech.stress[..., 1], axis=1) / np.sum(slab.weights, axis=1)
    assert np.max(np.abs(mean_yy)) <= 1e-8 * sxx_max
    assert np.ptp(slab.T) > 0.0 and np.ptp(mech.stress[..., 0]) > 0.0           # a real through-thickness gradient


def test_ts7g_newton_iterations_and_jacobian() -> None:
    slab, results = coupled_gradient_run()
    assert max(r.iterations for r in results) <= 8, max(r.iterations for r in results)
    assert all(r.residual >= 0.0 for r in results)
    # Jacobian against central differences of the production residual (derivative regression), mid-run state
    mid = make_slab(16)
    mid.advance_to(1500.0, 25.0)
    rng = np.random.default_rng(7)
    Tt = mid.T + 0.5 * rng.random(mid.T.shape)
    Rt, J = mid.residual_jacobian(Tt, 25.0, mid.t + 25.0)
    Jd = J.toarray()
    h = 1e-3
    Jfd = np.zeros_like(Jd)
    for j in range(len(Tt)):
        e = np.zeros_like(Tt)
        e[j] = h
        Jfd[:, j] = (mid.residual_jacobian(Tt + e, 25.0, mid.t + 25.0)[0]
                     - mid.residual_jacobian(Tt - e, 25.0, mid.t + 25.0)[0]) / (2.0 * h)
    assert np.max(np.abs(Jd - Jfd)) <= 1e-6 * np.max(np.abs(Jd))
    assert len(Rt) == len(Tt)
