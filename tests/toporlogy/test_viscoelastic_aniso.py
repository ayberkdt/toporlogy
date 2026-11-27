"""VA_anisotropic_ve: anisotropic generalized Maxwell with PSD branch tensors in Mandel form (VA1-VA12, NC1).

Preregistration: ``docs/foundations/prereg/VA_anisotropic_ve.md`` (thresholds fixed there before the first run). All
parameters are synthetic. The references are independent of the production recursion: closed forms derived in the
preregistration (written in Voigt, without the Mandel map), ``solve_ivp`` of the branch ODEs, Gauss-Legendre
quadrature of the exact branch stress with the test's own ``numpy.linalg.pinv``, mpmath, and a rotation written with
``einsum`` in the test. Tests that are only regressions are labelled "regression".
"""
from __future__ import annotations

import json
import math
import os

import mpmath
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from toporlogy.core import tensor as T
from toporlogy.core import viscoelastic as V
from toporlogy.core import viscoelastic_aniso as VA
from toporlogy.core.constitutive import Kinematics, MaterialRequest, StateStore, TimeStep, drive_material_point
from toporlogy.core.material import Orthotropic

PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
MV = np.array([1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)])
MM = np.outer(MV, MV)
_REC: dict[str, object] = {}


def rec(name: str, value: object) -> None:
    """Keep the measured value (written to the file named by $VA_RECORD when set)."""
    _REC[name] = value
    path = os.environ.get("VA_RECORD")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_REC, fh, indent=1, default=float)


# ------------------------------------------------------------------------------------------------ test-side maps
def eps_v(e: np.ndarray) -> np.ndarray:
    return np.array([e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])


def v_eps(v: np.ndarray) -> np.ndarray:
    e = np.zeros((3, 3))
    for (i, j), x, w in zip(PAIRS, v, [1, 1, 1, 0.5, 0.5, 0.5]):
        e[i, j] = e[j, i] = w * x
    return e


def sig_v(s: np.ndarray) -> np.ndarray:
    return np.array([s[i, j] for i, j in PAIRS])


def c4(C6: np.ndarray) -> np.ndarray:
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(PAIRS):
        for J, (k, l) in enumerate(PAIRS):
            C[i, j, k, l] = C[j, i, k, l] = C[i, j, l, k] = C[j, i, l, k] = C6[I, J]
    return C


def c6(C: np.ndarray) -> np.ndarray:
    return np.array([[C[i, j, k, l] for (k, l) in PAIRS] for (i, j) in PAIRS])


def rot_c6(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    return c6(np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, c4(C6)))


def psd_voigt(rng: np.random.Generator, rank: int, scale: float) -> np.ndarray:
    """Random Voigt stiffness whose MANDEL matrix is positive semi-definite of the given rank."""
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    w = np.zeros(6)
    w[:rank] = scale * rng.uniform(0.2, 1.0, rank)
    return ((Q * w) @ Q.T) / MM


def rand_sym(rng: np.random.Generator, scale: float) -> np.ndarray:
    A = rng.normal(size=(3, 3))
    return scale * 0.5 * (A + A.T)


def run(model, eps: list[np.ndarray], dts: np.ndarray, state=None, t0: float = 0.0):
    st = model.initial_state(1) if state is None else state
    t = t0
    out: dict[str, list] = {"sig": [], "C": [], "psi": [], "D": []}
    for e, dt in zip(eps, dts):
        u = model.update(e[None], st, MaterialRequest(energy=True, step=TimeStep(t, float(dt))))
        out["sig"].append(u.stress[0])
        out["C"].append(u.tangent[0])
        out["psi"].append(u.energy[0])
        out["D"].append(u.dissipation[0])
        st = u.trial_state
        t += float(dt)
    return {k: np.array(v) for k, v in out.items()}, st


def rand_history(rng: np.random.Generator, n: int, scale: float = 1e-3) -> list[np.ndarray]:
    e = np.zeros((3, 3))
    out = []
    for _ in range(n):
        e = e + rand_sym(rng, scale)
        out.append(e.copy())
    return out


ORTH_A = Orthotropic(40e9, 10e9, 8e9, 4e9, 3.5e9, 2.5e9, 0.30, 0.28, 0.40)
ORTH_B = Orthotropic(80e9, 6e9, 5e9, 2e9, 2e9, 1.5e9, 0.25, 0.25, 0.45)


def orth_model(rotation=None, taus=(0.7, 12.0, 300.0)):
    rng = np.random.default_rng(11)
    C_inf = ORTH_A.stiffness()
    branches = [(0.5 * ORTH_A.stiffness(), taus[0]), (0.3 * ORTH_B.stiffness(), taus[1]),
                (psd_voigt(rng, 3, 5e9), taus[2])]
    return VA.AnisotropicMaxwell(C_inf, branches, rotation=rotation), C_inf, branches


# ================================================================================================= VA1
def _iso_models(kind: str):
    if kind == "separate":
        Ginf, Kinf = 0.4e9, 1.5e9
        G = [(0.6e9, 0.3), (0.5e9, 4.0), (0.2e9, 60.0)]
        K = [(1.0e9, 0.8), (0.7e9, 25.0)]
        gm = V.GeneralizedMaxwell(V.PronySeries(Ginf, np.array([g for g, _ in G]), np.array([t for _, t in G])),
                                  V.PronySeries(Kinf, np.array([k for k, _ in K]), np.array([t for _, t in K])))
        br = [(VA.isotropic_branch_tensor(0.0, g), t) for g, t in G] + [(VA.isotropic_branch_tensor(k, 0.0), t) for k, t in K]
        return gm, VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(Kinf, Ginf), br)
    Ginf, Kinf = 0.4e9, 1.5e9
    B = [(1.0e9, 0.6e9, 0.5), (0.7e9, 0.5e9, 7.0), (0.3e9, 0.2e9, 90.0)]
    gm = V.GeneralizedMaxwell(V.PronySeries(Ginf, np.array([b[1] for b in B]), np.array([b[2] for b in B])),
                              V.PronySeries(Kinf, np.array([b[0] for b in B]), np.array([b[2] for b in B])))
    br = [(VA.isotropic_branch_tensor(k, g), t) for k, g, t in B]
    return gm, VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(Kinf, Ginf), br)


@pytest.mark.parametrize("kind", ["separate", "combined"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_VA1_isotropic_reduction_vs_generalized_maxwell(kind, seed):
    """Cross-implementation check against GeneralizedMaxwell (not an independent oracle)."""
    gm, am = _iso_models(kind)
    rng = np.random.default_rng(seed)
    dts = 10.0 ** rng.uniform(-3, 2, 40)
    dts[7] = 0.0
    eps = rand_history(rng, 40)
    a, _ = run(gm, eps, dts)
    b, _ = run(am, eps, dts)
    e_sig = np.abs(a["sig"] - b["sig"]).max() / np.abs(a["sig"]).max()
    e_C = np.abs(a["C"] - b["C"]).max() / np.abs(a["C"]).max()
    e_psi = np.abs(a["psi"] - b["psi"]).max() / a["psi"].max()
    e_D = np.abs(a["D"] - b["D"]).max() / a["D"].max()
    rec(f"VA1 {kind} seed{seed} sig, C, psi, D", (e_sig, e_C, e_psi, e_D))
    assert e_sig <= 1e-13 and e_C <= 1e-13 and e_psi <= 1e-12 and e_D <= 1e-11


# ================================================================================================= VA2
def test_VA2_orthotropic_step_relaxation_closed_form():
    model, C_inf, branches = orth_model()
    times = np.logspace(-3, 4, 12)
    rng = np.random.default_rng(2)
    modes = []
    for k in range(6):
        e0 = np.zeros(6)
        e0[k] = 1e-3
        modes.append(v_eps(e0 * np.array([1, 1, 1, 1, 1, 1])))      # engineering component set to 1e-3 -> tensor
    modes.append(rand_sym(rng, 1e-3))
    worst = 0.0
    for e0 in modes:
        dts = np.concatenate([[0.0], times[:1], np.diff(times)])
        eps = [e0] * len(dts)
        h, _ = run(model, eps, dts)
        tt = np.cumsum(dts)
        ev = eps_v(e0)
        for k, t in enumerate(tt):
            ref = (C_inf + sum(C * math.exp(-t / tau) for C, tau in branches)) @ ev
            worst = max(worst, np.abs(sig_v(h["sig"][k]) - ref).max())
        s0 = np.abs((C_inf + sum(C for C, _ in branches)) @ ev).max()
        worst_rel = worst / s0
        assert worst_rel <= 1e-12, (worst_rel,)
    rec("VA2 max abs err / sigma(0+)", float(worst_rel))


# ================================================================================================= VA3
def test_VA3_objectivity_under_rotation():
    rng = np.random.default_rng(3)
    R = Rotation.random(random_state=7).as_matrix()
    mat, _, _ = orth_model(None)
    rot, _, _ = orth_model(R)
    dts = 10.0 ** rng.uniform(-2, 2, 30)
    eps = rand_history(rng, 30)
    eps_mat = [R.T @ e @ R for e in eps]
    A, _ = run(rot, eps, dts)
    B, _ = run(mat, eps_mat, dts)
    sig_B = np.array([R @ s @ R.T for s in B["sig"]])
    CB = np.array([rot_c6(C, R) for C in B["C"]])
    e_s = np.abs(A["sig"] - sig_B).max() / np.abs(sig_B).max()
    e_C = np.abs(A["C"] - CB).max() / np.abs(CB).max()
    e_p = np.abs(A["psi"] - B["psi"]).max() / B["psi"].max()
    e_D = np.abs(A["D"] - B["D"]).max() / B["D"].max()
    rec("VA3 orthotropic sig, C, psi, D", (e_s, e_C, e_p, e_D))
    assert max(e_s, e_C, e_p, e_D) <= 1e-12
    # the rotation is not a no-op for this material
    assert np.abs(A["sig"] - B["sig"]).max() > 1e-3 * np.abs(B["sig"]).max()
    # isotropic material: the rotation does nothing
    _, iso0 = _iso_models("combined")
    iso1 = VA.AnisotropicMaxwell(iso0.C_inf, list(zip(iso0.branch_tensors, iso0.taus)), rotation=R)
    a, _ = run(iso0, eps, dts)
    b, _ = run(iso1, eps, dts)
    e_iso = np.abs(a["sig"] - b["sig"]).max() / np.abs(a["sig"]).max()
    rec("VA3 isotropic rotation invariance", e_iso)
    assert e_iso <= 1e-12


# ================================================================================================= VA4
def test_VA4_hereditary_reference_solve_ivp():
    rng = np.random.default_rng(4)
    R = Rotation.random(random_state=5).as_matrix()
    C_inf = ORTH_A.stiffness()
    br = [(0.5 * ORTH_A.stiffness(), 0.3), (0.3 * ORTH_B.stiffness(), 5.0), (psd_voigt(rng, 3, 5e9), 40.0)]
    model = VA.AnisotropicMaxwell(C_inf, br, rotation=R)
    times = np.concatenate([[0.0], np.sort(rng.uniform(0.5, 50.0, 11))])
    Ev = np.vstack([np.zeros(6), 2e-3 * rng.normal(size=(11, 6))])
    # reference in Voigt: s' = -s/tau + C eps'
    Cg = [rot_c6(C, R) for C in [C_inf] + [b for b, _ in br]]
    taus = np.array([t for _, t in br])
    s = np.zeros((3, 6))
    ref = []
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        edot = (Ev[k] - Ev[k - 1]) / dt

        def rhs(_t, y, edot=edot):
            S = y.reshape(3, 6)
            return (-S / taus[:, None] + np.array([Cg[i + 1] @ edot for i in range(3)])).ravel()

        sol = solve_ivp(rhs, (0.0, dt), s.ravel(), method="DOP853", rtol=1e-12, atol=1e-3)
        s = sol.y[:, -1].reshape(3, 6)
        ref.append(Cg[0] @ Ev[k] + s.sum(0))
    ref = np.array(ref)
    h, _ = run(model, [v_eps(e) for e in Ev[1:]], np.diff(times), state=None)
    # the model starts from zero strain; the first step goes from eps = 0 (Ev[0]) to Ev[1]
    got = np.array([sig_v(x) for x in h["sig"]])
    err = np.abs(got - ref).max() / np.abs(ref).max()
    rec("VA4 max rel err at knots", float(err))
    assert err <= 1e-9


# ================================================================================================= VA5
def test_VA5_tangent_vs_central_fd():
    rng = np.random.default_rng(5)
    model, _, _ = orth_model(Rotation.random(random_state=9).as_matrix())
    eps = rand_history(rng, 7)
    dts = 10.0 ** rng.uniform(-1, 1.5, 7)
    _, st = run(model, eps[:-1], dts[:-1])
    t = float(dts[:-1].sum())
    req = MaterialRequest(step=TimeStep(t, float(dts[-1])))
    E = eps_v(eps[-1])
    u = model.update(eps[-1][None], st, req)
    h = 1e-6
    J = np.zeros((6, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        sp = sig_v(model.update(v_eps(E + d)[None], st, req).stress[0])
        sm = sig_v(model.update(v_eps(E - d)[None], st, req).stress[0])
        J[:, k] = (sp - sm) / (2 * h)
    err = np.abs(J - u.tangent[0]).max() / np.abs(u.tangent[0]).max()
    rec("VA5 FD tangent rel err", float(err))
    assert err <= 1e-8
    uc = model.update(eps[-1][None], st, MaterialRequest(step=TimeStep(t, float(dts[-1])), tangent_kind="continuum"))
    C0 = model.C_inf + model.branch_tensors.sum(0)
    ec = np.abs(uc.tangent[0] - C0).max() / np.abs(C0).max()
    rec("VA5 continuum tangent rel err", float(ec))
    assert ec <= 1e-14
    # the algorithmic tangent differs from the continuum one for a finite step and tends to it for dt -> 0
    assert np.abs(u.tangent[0] - C0).max() > 1e-3 * np.abs(C0).max()


# ================================================================================================= VA6
def rand_model(rng: np.random.Generator, taus=(0.5, 5.0, 80.0), ranks=(6, 4, 2)) -> VA.AnisotropicMaxwell:
    C_inf = psd_voigt(rng, 6, 3e9)
    # SPD C_inf: eigenvalues >= 0.2 * scale > 0 from psd_voigt with rank 6
    return VA.AnisotropicMaxwell(C_inf, [(psd_voigt(rng, r, 2e9), t) for r, t in zip(ranks, taus)])


def _gl():
    x, w = np.polynomial.legendre.leggauss(10)
    return 0.5 * (x + 1.0), 0.5 * w


def _branch_path(model, s0, de_M, dt, u):
    """Exact branch stresses (nb, 6) in Mandel components at fraction u of the step (test-side formula)."""
    a = dt / model.taus
    Mb = model.branch_tensors * MM[None]
    h = (1.0 - np.exp(-a * u)) / a
    return np.exp(-a * u)[:, None] * s0 + h[:, None] * np.einsum("bij,j->bi", Mb, de_M)


def test_VA6_energy_dissipation_nonnegative_and_quadrature():
    worst_psi, worst_D = np.inf, np.inf
    for seed in range(5):
        rng = np.random.default_rng(60 + seed)
        model = rand_model(rng)
        dts = 10.0 ** rng.uniform(-4, 3, 60)
        h, _ = run(model, rand_history(rng, 60), dts)
        worst_psi = min(worst_psi, h["psi"].min() / h["psi"].max())
        worst_D = min(worst_D, h["D"].min() / h["D"].max())
        assert h["psi"].min() >= -1e-12 * h["psi"].max()
        assert h["D"].min() >= -1e-11 * h["D"].max()
    rec("VA6 min psi/max psi, min D/max D", (worst_psi, worst_D))
    # D against Gauss-Legendre of the rate sum s^T pinv(M) s / tau along the exact branch stress (dt <= 4 tau_min)
    gu, gw = _gl()
    worst = 0.0
    for seed in range(3):
        rng = np.random.default_rng(70 + seed)
        model = rand_model(rng)
        pinv = [np.linalg.pinv(C * MM, rcond=1e-10, hermitian=True) for C in model.branch_tensors]
        dts = rng.uniform(0.01, 2.0, 40)
        eps = rand_history(rng, 40)
        st = model.initial_state(1)
        t, e_prev, Dmax, errs = 0.0, np.zeros(6), 0.0, []
        for e, dt in zip(eps, dts):
            u = model.update(e[None], st, MaterialRequest(energy=True, step=TimeStep(t, float(dt))))
            e_M = eps_v(e) / MV
            de_M = e_M - e_prev
            D = 0.0
            for ui, wi in zip(gu, gw):
                s = _branch_path(model, st["s"][0], de_M, dt, ui)
                D += wi * dt * sum(s[b] @ pinv[b] @ s[b] / model.taus[b] for b in range(3))
            errs.append(abs(D - u.dissipation[0]))
            Dmax = max(Dmax, D)
            st, e_prev, t = u.trial_state, e_M, t + float(dt)
        worst = max(worst, max(errs) / Dmax)
    rec("VA6 D vs GL max abs diff / max D", float(worst))
    assert worst <= 1e-10


# ================================================================================================= VA7
def test_VA7_energy_closure_exact_work():
    gu, gw = _gl()
    rng = np.random.default_rng(77)
    model = rand_model(rng, taus=(0.5, 2.0, 6.0))
    dts = rng.uniform(0.01, 2.0, 40)
    eps = rand_history(rng, 40)
    st = model.initial_state(1)
    t, e_prev, W, D_sum = 0.0, np.zeros(6), 0.0, 0.0
    psi = 0.0
    M_inf = model.C_inf * MM
    for e, dt in zip(eps, dts):
        u = model.update(e[None], st, MaterialRequest(energy=True, step=TimeStep(t, float(dt))))
        e_M = eps_v(e) / MV
        de_M = e_M - e_prev
        for ui, wi in zip(gu, gw):
            s = _branch_path(model, st["s"][0], de_M, dt, ui)
            sig_M = M_inf @ (e_prev + ui * de_M) + s.sum(0)
            W += wi * (sig_M @ de_M)
        D_sum += u.dissipation[0]
        psi = u.energy[0]
        st, e_prev, t = u.trial_state, e_M, t + float(dt)
    resid = abs(W - psi - D_sum) / max(abs(W), D_sum, psi)
    rec("VA7(i) exact-work closure rel residual", float(resid))
    assert resid <= 1e-10


def test_VA7_trapezoid_work_refinement():
    rng = np.random.default_rng(78)
    model = rand_model(rng, taus=(0.3, 1.5, 5.0))
    Ea, Eb = rand_sym(rng, 1e-3), rand_sym(rng, 1e-3)
    Tend = 2.0
    res, Ws = {}, {}
    for N in (40, 80, 160):
        t = np.linspace(0.0, Tend, N + 1)
        eps = [Ea * math.sin(3.0 * math.pi * x / Tend) + Eb * (1.0 - math.cos(2.0 * math.pi * x / Tend)) for x in t]
        h, _ = run(model, eps[1:], np.diff(t))
        sig = np.concatenate([np.zeros((1, 3, 3)), h["sig"]])
        de = np.array(eps[1:]) - np.array(eps[:-1])
        W = float(np.sum(0.5 * (sig[:-1] + sig[1:]) * de))
        res[N] = W - h["psi"][-1] - h["D"].sum()
        Ws[N] = W
    r1, r2 = abs(res[40] / res[80]), abs(res[80] / res[160])
    rec("VA7(ii) residuals N=40,80,160, ratios, |R160|/W", (res[40], res[80], res[160], r1, r2,
                                                           abs(res[160]) / Ws[160]))
    assert r1 >= 3.2 and r2 >= 3.2
    assert abs(res[160]) <= 1e-3 * Ws[160]


# ================================================================================================= VA8
def _good():
    return VA.isotropic_branch_tensor(2e9, 1e9), [(VA.isotropic_branch_tensor(1e9, 5e8), 3.0)]


def _neg_branch(frac=-1e-3):
    rng = np.random.default_rng(8)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    w = np.array([frac, 0.3, 0.5, 0.7, 0.9, 1.0]) * 1e9
    return ((Q * w) @ Q.T) / MM


def test_VA8_invalid_input_refused_valid_accepted():
    Ci, br = _good()
    with pytest.raises(ValueError, match="not positive semi-definite"):
        VA.AnisotropicMaxwell(Ci, [(_neg_branch(), 1.0)])
    asym = br[0][0].copy()
    asym[0, 1] *= 1.01
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(Ci, [(asym, 1.0)])
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(Ci, [(np.eye(5), 1.0)])
    nan = br[0][0].copy()
    nan[2, 2] = np.nan
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(Ci, [(nan, 1.0)])
    for tau in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            VA.AnisotropicMaxwell(Ci, [(br[0][0], tau)])
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(_neg_branch(-0.2), br)                           # C_inf indefinite
    with pytest.raises(ValueError, match="allow_fluid"):
        VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(0.0, 1e9), br)             # singular C_inf
    fl = VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(0.0, 1e9), br, allow_fluid=True)
    assert fl.is_fluid and not VA.AnisotropicMaxwell(Ci, br).is_fluid
    with pytest.raises(ValueError):                                             # fluid, C0 still singular
        VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(0.0, 1e9), [(VA.isotropic_branch_tensor(0.0, 5e8), 1.0)],
                              allow_fluid=True)
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(Ci, br, rotation=1.1 * np.eye(3))
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(Ci, br, rotation=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="time step"):
        VA.AnisotropicMaxwell(Ci, br).update(np.zeros((1, 3, 3)), VA.AnisotropicMaxwell(Ci, br).initial_state(1))
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(Ci, br).update(np.zeros((1, 3, 3)), VA.AnisotropicMaxwell(Ci, br).initial_state(1),
                                             MaterialRequest(step=TimeStep(0.0, 1.0), tangent_kind="nonsense"))
    # accepted: zero eigenvalue, zero tensor, no branches
    rng = np.random.default_rng(81)
    VA.AnisotropicMaxwell(Ci, [(psd_voigt(rng, 4, 1e9), 1.0), (np.zeros((6, 6)), 2.0)])
    el = VA.AnisotropicMaxwell(Ci, [])
    u = el.update(np.eye(3)[None] * 1e-3, el.initial_state(1), MaterialRequest(step=TimeStep(0.0, 1.0)))
    assert np.allclose(u.tangent[0], Ci, rtol=0, atol=0)
    # the classification is by the Mandel spectrum: the negative branch has a Voigt-matrix and a Mandel-matrix
    # spectrum that both go negative here, the refusal message reports the Mandel one
    wm = np.linalg.eigvalsh(_neg_branch() * MM)
    rec("VA8 negative branch Mandel lambda_min/lambda_max", float(wm[0] / wm[-1]))
    assert wm[0] / wm[-1] < -1e-12


# ================================================================================================= VA9
def _drive_inputs(rng, k):
    return np.cumsum(1e-3 * rng.normal(size=(k, 6)), axis=0), np.cumsum(rng.uniform(0.05, 5.0, k))


def _manual(model, Ev, times, state=None, t0=0.0):
    st = model.initial_state(1) if state is None else state
    out, t = [], t0
    for e, tn in zip(Ev, times):
        H = T.voigt_to_strain(e)[None]
        eps = 0.5 * (H + np.swapaxes(H, 1, 2))
        u = model.update(eps, st, MaterialRequest(energy=True, step=TimeStep(t, float(tn - t))))
        out.append(u.stress[0].copy())
        st = {k: v.copy() for k, v in u.trial_state.items()}
        t = float(tn)
    return np.array(out), st


def test_VA9_restart_rollback_bitwise():
    rng = np.random.default_rng(9)
    model, _, br = orth_model()
    Ev, times = _drive_inputs(rng, 12)
    full, st_full = _manual(model, Ev, times)
    first, st6 = _manual(model, Ev[:6], times[:6])
    model2, _, _ = orth_model()                                             # a new instance
    rest, st_end = _manual(model2, Ev[6:], times[6:], state=st6, t0=float(times[5]))
    assert np.array_equal(np.concatenate([first, rest]), full)
    for k in model.state_fields:
        assert np.array_equal(st_end[k], st_full[k])
    # rollback: a discarded wild trial changes neither the committed arrays nor the next result
    ro = {k: v.copy() for k, v in st6.items()}
    for v in ro.values():
        v.setflags(write=False)
    snap = {k: v.tobytes() for k, v in ro.items()}
    model.update(rand_sym(rng, 5.0)[None], ro, MaterialRequest(energy=True, step=TimeStep(float(times[5]), 100.0)))
    assert {k: v.tobytes() for k, v in ro.items()} == snap
    again, _ = _manual(model, Ev[6:], times[6:], state=ro, t0=float(times[5]))
    assert np.array_equal(again, rest)
    # drive_material_point (regression of the contract path): same stresses, and a restart through StateStore
    ctrl = [True] * 6
    hist = drive_material_point(model, Ev, np.zeros((12, 6)), ctrl, times=times)
    assert np.array_equal(hist.stress, np.array([sig_v(s) for s in full]))
    h1 = drive_material_point(model, Ev[:6], np.zeros((6, 6)), ctrl, times=times[:6])
    store = StateStore(model, 1)
    store.restore(h1.states[-1])
    h2 = drive_material_point(model, Ev[6:], np.zeros((6, 6)), ctrl, times=times[6:], t0=float(times[5]),
                              store=store, strain0=h1.strain[-1])
    assert np.array_equal(np.vstack([h1.stress, h2.stress]), hist.stress)
    rec("VA9", "bitwise equal: restart, rollback, drive_material_point restart")


# ================================================================================================= VA10
def _models_dma():
    rng = np.random.default_rng(10)
    m1, _, _ = orth_model(Rotation.random(random_state=3).as_matrix(), taus=(0.01, 3.0, 400.0))
    m2 = rand_model(rng, taus=(0.02, 1.0, 50.0), ranks=(3, 1, 5))
    m3 = VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(2e9, 1e9), [(psd_voigt(rng, 6, 1e9), 2.0)])
    return [m1, m2, m3]


def test_VA10_dma_passivity_and_limits():
    om = np.logspace(-6, 6, 400)
    worst = [np.inf, np.inf]
    for m in _models_dma():
        Ms = m.complex_modulus(om, "mandel")
        Mi = m.C_inf * MM
        for k in range(len(om)):
            lo = np.linalg.eigvalsh(0.5 * (Ms[k].imag + Ms[k].imag.T))
            worst[0] = min(worst[0], lo[0] / lo[-1])
            hi = np.linalg.eigvalsh(0.5 * ((Ms[k].real - Mi) + (Ms[k].real - Mi).T))
            worst[1] = min(worst[1], hi[0] / np.linalg.eigvalsh(Ms[k].real)[-1])
        assert worst[0] >= -1e-12 and worst[1] >= -1e-12
        # closed form of the complex modulus in Voigt (test side)
        for w in (0.37, 12.0):
            x = w * m.taus
            ref = m.C_inf + sum(C * ((xi * xi + 1j * xi) / (1 + xi * xi)) for C, xi in zip(m.branch_tensors, x))
            assert np.abs(m.complex_modulus(w) - ref).max() <= 1e-12 * np.abs(ref).max()
        C0 = m.C_inf + m.branch_tensors.sum(0)
        e0 = np.abs(m.complex_modulus(0.0) - m.C_inf).max() / np.abs(m.C_inf).max()
        einf = np.abs(m.complex_modulus(1e12 / m.taus.min()) - C0).max() / np.abs(C0).max()
        assert e0 <= 1e-14 and einf <= 1e-10, (e0, einf)
    rec("VA10 min eig(M'')/max, min eig(M'-Minf)/max", tuple(worst))
    # single branch: loss peak at omega tau = 1 with value M_i / 2
    Cb = psd_voigt(np.random.default_rng(101), 4, 1e9)
    m = VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(2e9, 1e9), [(Cb, 2.0)])
    w = np.logspace(-3, 3, 20001) / 2.0
    loss = m.complex_modulus(w).imag
    kmax = np.argmax(np.einsum("kij,ij->k", loss, Cb))
    assert abs(w[kmax] * 2.0 - 1.0) <= 1e-3
    assert np.abs(m.complex_modulus(0.5).imag - 0.5 * Cb).max() <= 1e-12 * np.abs(Cb).max()


def test_VA10_steady_cycle_work_equals_pi_e0_Mloss_e0():
    rng = np.random.default_rng(102)
    model = rand_model(rng, taus=(0.3, 1.0, 2.0))
    om = 2.0
    e0 = 1e-3 * rng.normal(size=6)                        # Mandel amplitude
    Mb = [C * MM for C in model.branch_tensors]
    x = om * model.taus
    s_init = np.array([Mb[i] @ e0 * x[i] / (1 + x[i] ** 2) for i in range(3)])
    Mloss = sum(Mb[i] * x[i] / (1 + x[i] ** 2) for i in range(3))
    W_ref = math.pi * e0 @ Mloss @ e0
    N = 2000
    T0 = 2 * math.pi / om
    t = np.linspace(0.0, T0, N + 1)
    eps = [v_eps(MV * e0 * math.sin(om * ti)) for ti in t[1:]]
    st = {"strain_prev": np.zeros((1, 3, 3)), "s": s_init[None]}
    h, st_end = run(model, eps, np.diff(t), state=st)
    D = h["D"].sum()
    err = abs(D - W_ref) / W_ref
    rec("VA10 cycle dissipation rel err", float(err))
    assert err <= 1e-4
    # periodicity of the analytic start state (the state returns to itself)
    assert np.abs(st_end["s"][0] - s_init).max() <= 1e-3 * np.abs(s_init).max()


# ================================================================================================= VA11
def test_VA11_history_contract():
    model, _, _ = orth_model()
    assert model.physical is True and model.kinematics == "small" and model.conservative is False
    assert model.state_fields == ("strain_prev", "s")
    st = model.initial_state(4)
    assert st["strain_prev"].shape == (4, 3, 3) and st["s"].shape == (4, 3, 6)
    rng = np.random.default_rng(11)
    H = np.array([rand_sym(rng, 1e-3) for _ in range(4)])
    ro = {k: v.copy() for k, v in st.items()}
    for v in ro.values():
        v.setflags(write=False)
    req = MaterialRequest(energy=True, step=TimeStep(0.0, 2.0))
    u = model.update(H, ro, req)
    r = model.evaluate(Kinematics(H, "small"), None, ro, req)
    assert np.array_equal(r.stress, np.array([sig_v(s) for s in u.stress]))
    assert np.array_equal(r.tangent, u.tangent) and np.array_equal(r.energy, u.energy)
    assert all(not np.shares_memory(u.trial_state[k], ro[k]) for k in ro)
    r2 = model.evaluate(Kinematics(H[:, :2, :2], "small", plane="strain"), None, ro, req)
    assert r2.stress.shape == (4, 3)
    with pytest.raises(ValueError):
        model.evaluate(Kinematics(H[:, :2, :2], "small", plane="stress"), None, ro, req)
    with pytest.raises(ValueError):
        model.update(H + np.swapaxes(np.array([np.triu(np.ones((3, 3)), 1)] * 4), 1, 2), ro, req)  # asymmetric strain
    rec("VA11", "contract holds")


# ================================================================================================= VA12
def test_VA12_stability_phi_vs_mpmath():
    mpmath.mp.dps = 50
    Cinf = VA.isotropic_branch_tensor(2e9, 1e9)
    C1 = VA.isotropic_branch_tensor(1e9, 5e8)
    model = VA.AnisotropicMaxwell(Cinf, [(C1, 1.0)])
    e1 = rand_sym(np.random.default_rng(12), 1e-3)
    worst = 0.0
    for a in (1e-10, 1e-3, 1.0, 50.0, 1e6):
        u = model.update(e1[None], model.initial_state(1), MaterialRequest(energy=True, step=TimeStep(0.0, a)))
        phi = float((1 - mpmath.exp(-mpmath.mpf(a))) / mpmath.mpf(a))
        ref = Cinf @ eps_v(e1) + phi * (C1 @ eps_v(e1))
        got = sig_v(u.stress[0])
        worst = max(worst, np.abs(got - ref).max() / np.abs(ref).max())
        assert np.isfinite(u.energy[0]) and np.isfinite(u.dissipation[0]) and u.dissipation[0] >= 0.0
        hold = model.update(e1[None], u.trial_state, MaterialRequest(energy=True, step=TimeStep(a, 1e6)))
        assert np.isfinite(hold.stress).all() and hold.dissipation[0] >= -1e-12 * max(u.dissipation[0], 1e-300)
        if a == 1e6:
            assert np.all(hold.trial_state["s"] == 0.0)
    rec("VA12 max rel err of stress vs mpmath phi", float(worst))
    assert worst <= 1e-13


# ================================================================================================= NC1
def _cycle_work(Cv: np.ndarray, e0_M: np.ndarray, tau: float, om: float, ncyc: int = 30) -> float:
    """Work s . d(eps) over the last cycle of the branch ODE s' = -s/tau + C eps' (Voigt, solve_ivp)."""
    ev = MV * e0_M
    Tp = 2 * math.pi / om

    def rhs(t, y):
        edot = om * math.cos(om * t) * ev
        s = y[:6]
        return np.concatenate([-s / tau + Cv @ edot, [s @ edot]])

    y = np.zeros(7)
    for c in range(ncyc):
        y[6] = 0.0
        sol = solve_ivp(rhs, (c * Tp, (c + 1) * Tp), y, method="DOP853", rtol=1e-12, atol=1e-6)
        y = sol.y[:, -1]
    return float(y[6])


def test_NC1_nonpsd_branch_refused_and_dissipates_negatively():
    rng = np.random.default_rng(13)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    lam = np.array([-0.2, 0.3, 0.5, 0.7, 0.9, 1.0]) * 1e9
    C_bad = ((Q * lam) @ Q.T) / MM
    lam_ok = np.abs(lam)
    C_ok = ((Q * lam_ok) @ Q.T) / MM
    with pytest.raises(ValueError):
        VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(2e9, 1e9), [(C_bad, 0.5)])
    VA.AnisotropicMaxwell(VA.isotropic_branch_tensor(2e9, 1e9), [(C_ok, 0.5)])         # the PSD control is accepted
    om, tau = 2.0, 0.5
    x = om * tau
    e0 = 1e-3 * Q[:, 0]                                  # along the negative eigenvector (Mandel)
    W_bad = _cycle_work(C_bad, e0, tau, om)
    W_ok = _cycle_work(C_ok, e0, tau, om)
    ref_bad = math.pi * (x / (1 + x * x)) * lam[0] * (e0 @ e0)
    ref_ok = math.pi * (x / (1 + x * x)) * abs(lam[0]) * (e0 @ e0)
    rec("NC1 W_bad, ref_bad, W_ok, ref_ok", (W_bad, ref_bad, W_ok, ref_ok))
    assert W_bad < 0.0 < W_ok
    assert abs(W_bad - ref_bad) <= 1e-6 * abs(ref_bad) and abs(W_ok - ref_ok) <= 1e-6 * abs(ref_ok)
