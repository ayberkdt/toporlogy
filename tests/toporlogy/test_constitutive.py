"""Constitutive protocol and material-point checks (``toporlogy.core.constitutive``).

Tolerances are the pre-registered ones of ``docs/olcumler/nonlineer_cozucu_onkayit.md`` §2 (committed 26 Nov 2026
02:51 +03:00, before the first run); the P- and J-numbers refer to that table. Checks added after the first run are
marked "post hoc".
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (J2Plasticity, Kinematics, LinearElastic, NeoHooke, MaterialRequest, ReturnMappingFailure,
                                         SaintVenantKirchhoff, StateStore, drive_material_point, energy_oracle,
                                         neo_hooke_energy_torch, svk_energy_torch)
from toporlogy.fem.elements import D_iso, D_plane_strain

NUS = (-0.3, 0.0, 0.3, 0.45)
SY, HARD, NU = 1e-3, 0.05, 0.3
EY = SY                                   # ε_y = σ_y / E, E = 1
VOCE = dict(saturation_stress=1.5e-3, saturation_rate=500.0)


def _random_H(rng: np.random.Generator, n: int, dim: int, amp: float = 0.3) -> np.ndarray:
    out = []
    while len(out) < n:
        H = amp * (2 * rng.random((dim, dim)) - 1)
        if np.linalg.norm(H, 2) <= 0.3 and np.linalg.det(np.eye(dim) + H) >= 0.5:
            out.append(H)
    return np.array(out)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


# ============================================================================================== P1: AD oracle
# SVK in plane stress is the 2-D law with plane-stress constants, not a condensed 3-D energy: not part of P1
P1_CASES = [("svk", "3d"), ("svk", "strain"), ("neo_hooke", "3d"), ("neo_hooke", "strain"), ("neo_hooke", "stress")]


@pytest.mark.parametrize("nu", NUS)
@pytest.mark.parametrize("material,plane", P1_CASES)
def test_p1_energy_stress_tangent_match_the_ad_oracle(material, plane, nu):
    """P1: ψ, S ≤ 1e-11 and ℂ ≤ 1e-10 (max|Δ|/max|ref|) against torch autograd of ψ(E) (20 random states)."""
    rng = np.random.default_rng(100 + int(10 * nu) + (0 if plane == "3d" else 7))
    dim = 3 if plane == "3d" else 2
    H = _random_H(rng, 20, dim)
    if material == "svk":
        model = SaintVenantKirchhoff(D_iso(nu) if dim == 3 else D_plane_strain(nu))
        psi = svk_energy_torch(D_iso(nu))
    else:
        model = NeoHooke(nu)
        psi = neo_hooke_energy_torch(model.kappa, model.mu)
    r = model.evaluate(Kinematics(H, "finite", plane), None, {}, MaterialRequest(energy=True))
    errs = [0.0, 0.0, 0.0]
    for k in range(len(H)):
        E3 = np.zeros((3, 3))
        E3[:dim, :dim] = T.green_lagrange(H[k:k + 1])[0]
        if plane == "stress":
            E3[2, 2] = r.diagnostics["E33"][k]
        p, S6, C6 = energy_oracle(psi, T.strain_to_voigt(E3, rtol=None))
        if dim == 3:
            S, C = S6, C6
        else:
            ip, op = [0, 1, 5], [2, 3, 4]
            S = S6[ip]
            C = C6[np.ix_(ip, ip)]
            if plane == "stress":
                C = C - C6[np.ix_(ip, op)] @ np.linalg.solve(C6[np.ix_(op, op)], C6[np.ix_(op, ip)])
                assert abs(S6[2]) <= 1e-12 * np.abs(S6).max()
        assert r.energy is not None and r.tangent is not None
        errs[0] = max(errs[0], abs(r.energy[k] - p) / abs(p))
        errs[1] = max(errs[1], _rel(r.stress[k], S))
        errs[2] = max(errs[2], _rel(r.tangent[k], C))
    print(f"P1 {material} {plane} nu={nu}: psi {errs[0]:.2e}, S {errs[1]:.2e}, C {errs[2]:.2e}")
    assert errs[0] <= 1e-11 and errs[1] <= 1e-11 and errs[2] <= 1e-10


# ============================================================================================== P2: objectivity
@pytest.mark.parametrize("material", ["svk", "neo_hooke"])
def test_p2_rigid_rotation_objectivity(material):
    """P2: ψ(QF) = ψ(F) and S(QF) = S(F), 3-D, 20 random proper rotations: relative ≤ 1e-12."""
    rng = np.random.default_rng(21)
    H = _random_H(rng, 20, 3)
    model = SaintVenantKirchhoff(D_iso(0.3)) if material == "svk" else NeoHooke(0.3)
    Q = np.array([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(20)])
    Q[np.linalg.det(Q) < 0] *= -1
    HQ = Q @ (np.eye(3) + H) - np.eye(3)
    r0 = model.evaluate(Kinematics(H, "finite"), None, {}, MaterialRequest(energy=True))
    r1 = model.evaluate(Kinematics(HQ, "finite"), None, {}, MaterialRequest(energy=True))
    assert r0.energy is not None and r1.energy is not None
    e_s, e_p = _rel(r1.stress, r0.stress), _rel(r1.energy, r0.energy)
    print(f"P2 {material}: S {e_s:.2e}, psi {e_p:.2e}")
    assert e_s <= 1e-12 and e_p <= 1e-12


# ============================================================================================== P3: trial vs committed
def _sha(state: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(np.ascontiguousarray(state[k]).tobytes())
    return h.hexdigest()


def test_p3_trial_evaluation_does_not_change_the_committed_state():
    """P3: J2 evaluate with the read-only committed state of a StateStore raises nothing and leaves its bytes."""
    m = J2Plasticity(NU, SY, HARD)
    store = StateStore(m, 5)
    rng = np.random.default_rng(3)
    eps = 5 * EY * (2 * rng.random((5, 3, 3)) - 1)
    eps = 0.5 * (eps + np.swapaxes(eps, 1, 2))
    r = m.evaluate(Kinematics(eps, "small"), None, store.committed, MaterialRequest())
    store.set_trial(r.trial_state)
    store.commit()                                   # a committed plastic state
    before = _sha(store.committed)
    assert not store.committed["alpha"].flags.writeable
    r2 = m.evaluate(Kinematics(2 * eps, "small"), None, store.committed, MaterialRequest())
    assert _sha(store.committed) == before
    assert r2.trial_state is not None and not np.shares_memory(r2.trial_state["alpha"], store.committed["alpha"])
    store.set_trial(r2.trial_state)
    store.rollback()
    assert _sha(store.committed) == before and store.trial is None


def test_protocol_refuses_wrong_kinematics_and_plane_stress_j2():
    """Post hoc (contract): a small-strain model refuses finite kinematics and vice versa; J2 refuses plane stress."""
    H = np.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="small-strain model"):
        LinearElastic(np.eye(3)).evaluate(Kinematics(H, "finite", "stress"))
    with pytest.raises(ValueError, match="finite-strain model"):
        NeoHooke(0.3).evaluate(Kinematics(H, "small", "strain"))
    with pytest.raises(ValueError, match="plane strain only"):
        J2Plasticity(NU, SY).evaluate(Kinematics(H, "small", "stress"), None, J2Plasticity(NU, SY).initial_state(1))
    with pytest.raises(ValueError, match="does not fit"):
        Kinematics(H, "small", "3d")


# ============================================================================================== J1, J2: uniaxial
def _uniaxial_reference(eps: np.ndarray, H: float) -> tuple[np.ndarray, np.ndarray]:
    """1-D isotropic hardening (E = 1), leg by leg in closed form: elastic until |σ| reaches σ_y + Hα in the loading
    direction, then slope E_t = H/(1 + H); α grows by |Δε_plastic part|/(1 + H). Independent of radial return."""
    Et = H / (1.0 + H)
    sig, al = 0.0, 0.0
    out_s, out_a = [], []
    prev = 0.0
    legs: list[tuple[float, float, float]] = []   # (start strain, start stress, start alpha) per leg
    sgn = 0.0
    for e in eps:
        d = e - prev
        s_new = np.sign(d)
        if s_new != 0 and s_new != sgn:            # a new monotone leg starts at the previous point
            legs.append((prev, sig, al))
            sgn = s_new
        ea, sa, aa = legs[-1]
        k = SY + H * aa
        e_switch = ea + (sgn * k - sa)             # E = 1
        if sgn * (e - e_switch) <= 0:
            sig, al = sa + (e - ea), aa
        else:
            sig = sgn * k + Et * (e - e_switch)
            al = aa + abs(e - e_switch) / (1.0 + H)
        out_s.append(sig)
        out_a.append(al)
        prev = e
    return np.array(out_s), np.array(out_a)


def _drive_uniaxial(model: J2Plasticity, eps11: np.ndarray):
    k = len(eps11)
    strain = np.zeros((k, 6))
    strain[:, 0] = eps11
    ctrl = [True, False, False, False, False, False]
    return drive_material_point(model, strain, np.zeros((k, 6)), ctrl, tol=1e-14 * 1.0)


def _yield_f(model: J2Plasticity, stress6: np.ndarray, alpha: float) -> float:
    s = T.voigt_to_stress(stress6)
    dev = s - np.trace(s) / 3 * np.eye(3)
    return math.sqrt(1.5) * float(np.linalg.norm(dev)) - float(model.kappa(np.array(alpha)))


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["tension", "compression"])
def test_j1_uniaxial_stress_monotonic_closed_form(sign):
    """J1: uniaxial stress to ±10 ε_y in 50 steps: max|σ₁₁ − σ_ref| ≤ 1e-10 σ_y, max|α − α_ref| ≤ 1e-10 ε_y."""
    m = J2Plasticity(NU, SY, HARD)
    eps = sign * np.linspace(0, 10 * EY, 51)[1:]
    h = _drive_uniaxial(m, eps)
    s_ref, a_ref = _uniaxial_reference(eps, HARD)
    a = np.array([st["alpha"][0] for st in h.states])
    es, ea = np.abs(h.stress[:, 0] - s_ref).max(), np.abs(a - a_ref).max()
    lateral = np.abs(h.stress[:, 1:]).max()
    print(f"J1 {sign:+.0f}: stress {es / SY:.2e} sy, alpha {ea / EY:.2e} ey, lateral stress {lateral:.1e}")
    assert es <= 1e-10 * SY and ea <= 1e-10 * EY


def test_j2_cyclic_uniaxial_closed_form_and_elastic_unloading():
    """J2: 0 → +5ε_y → −5ε_y → +8ε_y, 40 steps per leg; J1 thresholds; on elastic unloading Δσ = E Δε and α stays
    bit-for-bit."""
    m = J2Plasticity(NU, SY, HARD)
    legs = [np.linspace(0, 5 * EY, 41)[1:], np.linspace(5 * EY, -5 * EY, 41)[1:], np.linspace(-5 * EY, 8 * EY, 41)[1:]]
    eps = np.concatenate(legs)
    h = _drive_uniaxial(m, eps)
    s_ref, a_ref = _uniaxial_reference(eps, HARD)
    a = np.array([st["alpha"][0] for st in h.states])
    es, ea = np.abs(h.stress[:, 0] - s_ref).max(), np.abs(a - a_ref).max()
    plastic = np.diff(np.concatenate([[0.0], a])) > 0
    elastic_after_plastic = [k for k in range(1, len(eps)) if not plastic[k] and a[k - 1] > 0]
    assert elastic_after_plastic, "no elastic unloading step"
    for k in elastic_after_plastic:
        assert a[k] == a[k - 1]
        assert abs((h.stress[k, 0] - h.stress[k - 1, 0]) - (eps[k] - eps[k - 1])) <= 1e-12 * SY
    print(f"J2 cyclic: stress {es / SY:.2e} sy, alpha {ea / EY:.2e} ey, {len(elastic_after_plastic)} elastic "
          f"unloading steps, final alpha {a[-1] / EY:.3f} ey")
    assert es <= 1e-10 * SY and ea <= 1e-10 * EY


def test_j3_proportional_mixed_path_closed_form():
    """J3: ε = t ε̂ (deviatoric + volumetric), ē(1) = 10 ε_y equivalent: q = 3μ(σ_y + Hē)/(3μ + H) once 3μē > σ_y,
    p = K tr ε; max|Δq|, |Δp| ≤ 1e-11 σ_y."""
    m = J2Plasticity(NU, SY, HARD)
    D = np.array([[0.3, 0.8, -0.2], [0.8, -0.5, 0.4], [-0.2, 0.4, 0.2]])
    D = D - np.trace(D) / 3 * np.eye(3)
    D *= 10 * EY / (math.sqrt(2.0 / 3.0) * np.linalg.norm(D))     # ē = √(2/3)‖dev ε‖ = 10 ε_y at t = 1
    epsh = D + 0.002 * np.eye(3)
    ts = np.linspace(0, 1, 31)[1:]
    h = drive_material_point(m, [t * T.strain_to_voigt(epsh, rtol=None) for t in ts], np.zeros((30, 6)), [True] * 6)
    eq, ep = 0.0, 0.0
    for t, s6 in zip(ts, h.stress, strict=True):
        ebar = t * math.sqrt(2.0 / 3.0) * np.linalg.norm(D)
        q_ref = 3 * m.mu * ebar if 3 * m.mu * ebar <= SY else 3 * m.mu * (SY + HARD * ebar) / (3 * m.mu + HARD)
        s = T.voigt_to_stress(s6)
        q = math.sqrt(1.5) * np.linalg.norm(s - np.trace(s) / 3 * np.eye(3))
        eq = max(eq, abs(q - q_ref))
        ep = max(ep, abs(np.trace(s) / 3 - m.K * t * np.trace(epsh)))
    print(f"J3: q {eq / SY:.2e} sy, p {ep / SY:.2e} sy")
    assert eq <= 1e-11 * SY and ep <= 1e-11 * SY


# ============================================================================================== J4: tangent
def _random_states(rng: np.random.Generator, n: int):
    ep = rng.standard_normal((n, 3, 3))
    ep = 0.5 * (ep + np.swapaxes(ep, 1, 2))
    ep -= np.trace(ep, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    ep *= (3 * EY * rng.random(n) / np.linalg.norm(ep, axis=(1, 2)))[:, None, None]
    al = 5 * EY * rng.random(n)
    e = rng.standard_normal((n, 3, 3))
    e = 0.5 * (e + np.swapaxes(e, 1, 2))
    e = ep + (4 * EY * rng.random(n) / np.linalg.norm(e, axis=(1, 2)))[:, None, None] * e
    return {"plastic_strain": ep, "alpha": al}, e


def _fd_tangent(m: J2Plasticity, e: np.ndarray, st: dict[str, np.ndarray], h: float) -> np.ndarray:
    C = np.zeros((6, 6))
    ev = T.strain_to_voigt(e, rtol=None)
    for j in range(6):
        out = []
        for s in (1, -1):
            v = ev.copy()
            v[j] += s * h
            out.append(m.evaluate(Kinematics(T.voigt_to_strain(v)[None], "small"), None, st, MaterialRequest()).stress[0])
        C[:, j] = (out[0] - out[1]) / (2 * h)
    return C


@pytest.mark.parametrize("hardening", ["linear", "voce"])
def test_j4_consistent_tangent_matches_finite_difference(hardening):
    """J4: algorithmic tangent vs central FD of the return map (step 1e-6 max|ε|), 20 random states: max|C − C_FD| ≤
    1e-6 max|C|; points within 1e-6 σ_y of the yield surface are excluded."""
    m = J2Plasticity(NU, SY, HARD, **(VOCE if hardening == "voce" else {}))
    rng = np.random.default_rng(44)
    states, eps = _random_states(rng, 20)
    worst, n_pl, used = 0.0, 0, 0
    for k in range(20):
        st = {key: v[k:k + 1] for key, v in states.items()}
        r = m.evaluate(Kinematics(eps[k:k + 1], "small"), None, st, MaterialRequest())
        # distance of the trial point from the yield surface
        s_tr = 2 * m.mu * (eps[k] - np.trace(eps[k]) / 3 * np.eye(3) - st["plastic_strain"][0])
        f_tr = math.sqrt(1.5) * np.linalg.norm(s_tr) - float(m.kappa(st["alpha"]))
        if abs(f_tr) <= 1e-6 * SY:
            continue
        used += 1
        n_pl += int(r.diagnostics["plastic"][0])
        assert r.tangent is not None
        Cfd = _fd_tangent(m, eps[k], st, 1e-6 * np.abs(eps[k]).max())
        worst = max(worst, _rel(Cfd, r.tangent[0]))
    print(f"J4 {hardening}: {used} states ({n_pl} plastic), max rel {worst:.2e}")
    assert n_pl >= 5 and used - n_pl >= 3
    assert worst <= 1e-6


def test_j4b_continuum_tangent_is_not_the_consistent_one():
    """J4b (negative control): the continuum tangent differs from FD by ≥ 1e-3 max|C| at a plastic point with
    Δα ≥ ε_y."""
    m = J2Plasticity(NU, SY, HARD)
    st = m.initial_state(1)
    D = np.diag([1.0, -0.5, -0.5])
    e = (8 * EY / (math.sqrt(2.0 / 3.0) * np.linalg.norm(D))) * D
    r = m.evaluate(Kinematics(e[None], "small"), None, st, MaterialRequest(tangent_kind="continuum"))
    assert r.trial_state is not None and r.trial_state["alpha"][0] >= EY and r.tangent is not None
    Cfd = _fd_tangent(m, e, st, 1e-6 * np.abs(e).max())
    d = np.abs(r.tangent[0] - Cfd).max() / np.abs(Cfd).max()
    print(f"J4b: continuum vs FD {d:.2e}, delta alpha {r.trial_state['alpha'][0] / EY:.2f} ey")
    assert d >= 1e-3


# ============================================================================================== J5, J6: yield, energy
@pytest.mark.parametrize("hardening", ["linear", "voce"])
def test_j5_j6_yield_surface_dissipation_and_energy_identity(hardening):
    """J5: |f| ≤ 1e-12 σ_y on plastic steps, f ≤ 0 on elastic steps; J6: D = σ_y Δα ≥ 0 and σ_{n+1}:Δε_p ≥ 0
    (≥ −1e-14 σ_y ε_y), and Σ σ_{n+1}:Δε = ψ_e(end) + Σ ½Δε_e:C:Δε_e + Σ σ_{n+1}:Δε_p (relative ≤ 1e-12)."""
    m = J2Plasticity(NU, SY, HARD, **(VOCE if hardening == "voce" else {}))
    legs = [np.linspace(0, 5 * EY, 41)[1:], np.linspace(5 * EY, -5 * EY, 41)[1:], np.linspace(-5 * EY, 8 * EY, 41)[1:]]
    h = _drive_uniaxial(m, np.concatenate(legs))
    al = np.array([s["alpha"][0] for s in h.states])
    dal = np.diff(np.concatenate([[0.0], al]))
    f = np.array([_yield_f(m, h.stress[k], al[k]) for k in range(len(al))])
    pl = dal > 0
    fmax_pl, fmax_el = np.abs(f[pl]).max(), f[~pl].max()
    work = np.zeros(1)
    lhs, el, alg, wp = 0.0, 0.0, 0.0, 0.0
    e_prev, ep_prev = np.zeros(6), np.zeros((3, 3))
    for k in range(len(al)):
        de = h.strain[k] - e_prev
        sig = T.voigt_to_stress(h.stress[k])
        ep = h.states[k]["plastic_strain"][0]
        dep = ep - ep_prev
        lhs += float(h.stress[k] @ de)
        dee = T.voigt_to_strain(de) - dep
        alg += 0.5 * float(np.tensordot(dee, 2 * m.mu * (dee - np.trace(dee) / 3 * np.eye(3))
                                        + m.K * np.trace(dee) * np.eye(3)))
        wp += float(np.tensordot(sig, dep))
        e_prev, ep_prev = h.strain[k], ep
    ee = T.voigt_to_strain(h.strain[-1]) - ep_prev
    el = 0.5 * float(np.tensordot(T.voigt_to_stress(h.stress[-1]), ee))
    scale = sum(abs(float(h.stress[k] @ (h.strain[k] - (h.strain[k - 1] if k else 0)))) for k in range(len(al)))
    ident = abs(lhs - (el + alg + wp)) / scale
    del work
    print(f"J5/J6 {hardening}: |f| plastic {fmax_pl / SY:.1e} sy, max f elastic {fmax_el / SY:.1e} sy, "
          f"min D {h.dissipation.min():.1e}, min W_p {h.plastic_work.min():.1e}, identity {ident:.1e}, "
          f"total dissipation {h.dissipation.sum():.3e}")
    assert fmax_pl <= 1e-12 * SY and fmax_el <= 0.0
    assert h.dissipation.min() >= -1e-14 * SY * EY and h.plastic_work.min() >= -1e-14 * SY * EY
    assert ident <= 1e-12


def test_return_mapping_failure_is_reported():
    """Post hoc (N1 at the material point): Voce with one local iteration raises ReturnMappingFailure."""
    m = J2Plasticity(NU, SY, HARD, local_max_iter=1, **VOCE)
    e = np.diag([1.0, -0.5, -0.5]) * 5 * EY
    with pytest.raises(ReturnMappingFailure) as ei:
        m.evaluate(Kinematics(e[None], "small"), None, m.initial_state(1), MaterialRequest())
    assert ei.value.code == "return_mapping"
