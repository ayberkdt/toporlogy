"""General Newton, path following, state handling and small-strain plasticity in the FE physics
(``solvers.newton``, ``solvers.continuation``, ``fem.nonlinear``), energy-interpolation gradients and the new
``EndStrainEnergy`` response.

Tolerances are the pre-registered ones of ``docs/olcumler/nonlineer_cozucu_onkayit.md`` §2 (committed 26 Nov 2026
02:51 +03:00, before the first run); the letters and numbers refer to that table. Checks added after the first run
are marked "post hoc".
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from toporlogy.core.constitutive import J2Plasticity, Kinematics, LinearElastic, MaterialRequest, StateStore, \
    drive_material_point
from toporlogy.core.domain import Domain
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import D_plane_strain, D_plane_stress, D_iso
from toporlogy.fem.nonlinear import EnergyInterpolation, NonlinearConvergenceError, NonlinearElasticity
from toporlogy.optimization.nonlinear import EndCompliance, EndStrainEnergy
from toporlogy.optimization.study import Study
from toporlogy.optimizers.mma import MMA
from toporlogy.solvers.continuation import ArcLengthSettings, ContinuationError, arc_length, \
    displacement_control, load_control
from toporlogy.solvers.newton import LinearSolveFailure, Linearization, NewtonSettings, newton

sys.path.insert(0, str(Path(__file__).parent))
from _truss import TrussAtLoad, TwoBarTruss  # noqa: E402

SY, HARD, NU = 1e-3, 0.05, 0.3


def _clamped(shape, h=1.0, load=(0.0, -1.0), where="tip"):
    d = Domain(shape, h=h)
    X = d.node_coords()
    dim = d.dim
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - X[:, 0].max()) < 1e-9)
    F = np.zeros(d.ndof)
    for i, v in enumerate(load):
        F[dim * tip + i] = v / len(tip)
    return d, fixed, F


# ============================================================================================== P4, P5: small strain
@pytest.mark.parametrize("dim", [2, 3])
def test_p4_small_strain_linear_model_equals_elasticity(dim):
    """P4: LinearElastic (small-strain kinematics) in NonlinearElasticity gives Elasticity's U: relative ≤ 1e-12."""
    d, fixed, F = _clamped((6, 3) if dim == 2 else (4, 2, 2))
    E = 0.3 + 0.7 * np.random.default_rng(1).random(d.n_active)
    D = D_plane_stress(0.3) if dim == 2 else D_iso(0.3)
    nl = NonlinearElasticity(d, fixed, material=LinearElastic(D), rtol=1e-13)
    lin = Elasticity(d, fixed, nu=0.3)
    U1, U0 = nl.solve_numpy(E, F), lin.solve_numpy(E, F)
    rel = np.abs(U1 - U0).max() / np.abs(U0).max()
    print(f"P4 {dim}D: rel {rel:.2e}")
    assert rel <= 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_p5_j2_elastic_limit_equals_linear(dim):
    """P5: J2 with σ_y = 1e30 equals the small-strain linear solution (plane strain in 2-D): relative ≤ 1e-12."""
    d, fixed, F = _clamped((6, 3) if dim == 2 else (4, 2, 2))
    E = 0.3 + 0.7 * np.random.default_rng(1).random(d.n_active)
    D = D_plane_strain(NU) if dim == 2 else D_iso(NU)
    kw = dict(plane="strain") if dim == 2 else {}
    U0 = NonlinearElasticity(d, fixed, material=LinearElastic(D), rtol=1e-13, **kw).solve_numpy(E, F)
    U1 = NonlinearElasticity(d, fixed, material=J2Plasticity(NU, 1e30, HARD), rtol=1e-13, **kw).solve_numpy(E, F)
    rel = np.abs(U1 - U0).max() / np.abs(U0).max()
    print(f"P5 {dim}D: rel {rel:.2e}")
    assert rel <= 1e-12


def test_stateful_model_refuses_autograd_interpolation_and_warm_start():
    """Post hoc (fail-closed contract): no energy interpolation or warm start for J2. The design-sensitivity refusal
    of this test was replaced by the path adjoint (``plastik_duyarlilik_onkayit.md``, planned change): a
    path-dependent model without ``evaluate_vjp`` is still refused (test_plastic_sensitivity.py PS9)."""
    d, fixed, F = _clamped((4, 2))
    j2 = J2Plasticity(NU, SY, HARD)
    with pytest.raises(ValueError, match="finite-strain"):
        NonlinearElasticity(d, fixed, material=j2, plane="strain", energy_interpolation=EnergyInterpolation())
    with pytest.raises(ValueError, match="warm_start"):
        NonlinearElasticity(d, fixed, material=j2, plane="strain", warm_start=True)


# ============================================================================================== J7: homogeneous bar
def _bar(dim):
    d = Domain((4, 1) if dim == 2 else (4, 1, 1), h=1.0)
    X = d.node_coords()
    fixed = [d.dofs(np.flatnonzero(X[:, 0] < 1e-9))[0::dim]]           # u_x = 0 at x = 0
    for a in range(1, dim):
        fixed.append(dim * np.flatnonzero(X[:, a] < 1e-9) + a)           # rollers on the y (z) = 0 faces
    tip = np.flatnonzero(np.abs(X[:, 0] - 4.0) < 1e-9)
    F = np.zeros(d.ndof)
    F[dim * tip] = 1.0 / len(tip)                                        # unit total force on a unit face
    return d, np.concatenate(fixed), F, tip


@pytest.mark.parametrize("dim", [2, 3])
def test_j7_homogeneous_bar_matches_the_material_point(dim):
    """J7: uniform traction 0 → 2σ_y → 0 (10 + 10 steps): tip ε₁₁ and the permanent strain match the material-point
    driver with the same control (relative ≤ 1e-9). The α clause: the two tests after this one."""
    d, fixed, F, tip = _bar(dim)
    j2 = J2Plasticity(NU, SY, HARD)
    nl = NonlinearElasticity(d, fixed, material=j2, plane="strain" if dim == 2 else "stress", rtol=1e-13,
                             load_steps=10)
    E = np.ones(nl.n_elements)
    lam_path, eps_fe = [], []
    cp = None
    U, rep, cp = nl.solve_history(E, 2 * SY * F, [1.0, 0.0])
    lam_path = [s.load_factor for s in rep.steps if s.converged]
    assert rep.cuts == 0 and len(lam_path) == 20
    # replay the same load factors one at a time to get the strain after every increment
    eps_fe, cp = [], None
    alphas = []
    for lam in lam_path:
        Ui, _r, cp = nl.solve_history(E, 2 * SY * F, [lam], checkpoint=cp)
        eps_fe.append(float(np.mean(Ui[dim * tip, 0])) / 4.0)
        alphas.append(cp["state"]["alpha"])
    k = len(lam_path)
    strain, stress = np.zeros((k, 6)), np.zeros((k, 6))
    stress[:, 0] = 2 * SY * np.array(lam_path)
    ctrl = [False] * 6
    if dim == 2:
        ctrl[2] = True                                                   # ε_zz = 0 (plane strain)
    mp = drive_material_point(j2, strain, stress, ctrl)
    e_ref = mp.strain[:, 0]
    rel = np.abs(np.array(eps_fe) - e_ref).max() / np.abs(e_ref).max()
    perm = abs(eps_fe[-1] - e_ref[-1]) / abs(e_ref[-1])
    spread = max(float(np.ptp(a) / np.abs(a).max()) if np.abs(a).max() > 0 else 0.0 for a in alphas)
    spread_path = max(float(np.ptp(a)) for a in alphas) / max(float(np.abs(a).max()) for a in alphas)
    print(f"J7 {dim}D: eps rel {rel:.2e}, permanent eps {eps_fe[-1]:.4e} (ref {e_ref[-1]:.4e}, rel {perm:.1e}), "
          f"alpha spread per step {spread:.1e}, relative to the path maximum {spread_path:.1e}")
    assert e_ref[-1] > 0.0
    assert rel <= 1e-9 and perm <= 1e-9


def _j7_spreads(dim):
    d, fixed, F, tip = _bar(dim)
    nl = NonlinearElasticity(d, fixed, material=J2Plasticity(NU, SY, HARD), plane="strain" if dim == 2 else "stress",
                             rtol=1e-13, load_steps=10)
    E = np.ones(nl.n_elements)
    cp, alphas = None, []
    for lam in [0.1 * k for k in range(1, 11)] + [1.0 - 0.1 * k for k in range(1, 11)]:
        _U, _r, cp = nl.solve_history(E, 2 * SY * F, [lam], checkpoint=cp)
        alphas.append(cp["state"]["alpha"])
    per_step = max(float(np.ptp(a) / np.abs(a).max()) if np.abs(a).max() > 0 else 0.0 for a in alphas)
    path = max(float(np.ptp(a)) for a in alphas) / max(float(np.abs(a).max()) for a in alphas)
    print(f"J7 alpha {dim}D: per step {per_step:.1e}, relative to the path maximum {path:.1e}")
    return per_step, path


J7_ALPHA_REASON = ("pre-registered J7 alpha clause FAILED as written (26 Nov 2026): per-step relative spread of "
                   "alpha over the Gauss points H8 1.0 at the step that lands exactly on the yield stress "
                   "(lambda = 0.5, sigma = sigma_y: f_trial = +-1e-19, some points return with alpha = 3.6e-19, others "
                   "stay elastic); Q4 plane strain 1.1e-12 at the first plastic step (rounding of a small alpha). "
                   "Relative to the path maximum of alpha: 1.5e-14 (H8), 3.2e-13 (Q4) (post hoc "
                   "test_j7_alpha_uniform_relative_to_the_path_maximum).")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.xfail(strict=True, reason=J7_ALPHA_REASON)
def test_j7_alpha_uniform_over_gauss_points_as_registered(dim):
    """J7 (alpha clause as pre-registered): per-step relative spread of α over the Gauss points ≤ 1e-12."""
    assert _j7_spreads(dim)[0] <= 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_j7_alpha_uniform_relative_to_the_path_maximum(dim):
    """Post hoc (J7 alpha clause, same threshold, a scale that exists at first yield): max over the steps of the
    α spread over the Gauss points ≤ 1e-12 × the largest α of the path."""
    assert _j7_spreads(dim)[1] <= 1e-12


# ============================================================================================== J8: plastic cantilever
def _plastic_cantilever(tangent_kind="algorithmic", load_steps=10):
    d, fixed, F = _clamped((20, 5))
    P = 1.6 * SY * 25 / (6 * 20)                                          # 1.6 × first yield at the root
    nl = NonlinearElasticity(d, fixed, material=J2Plasticity(NU, SY, HARD), plane="strain", rtol=1e-12,
                             load_steps=load_steps, tangent_kind=tangent_kind, max_iter=30)
    return d, nl, P * F


def _gauss_strains(nl, u):
    Ue = u[nl.edof]
    n, nl_ = Ue.shape
    Ut = np.swapaxes(Ue.reshape(n, nl_ // nl.dim, nl.dim), 1, 2)
    return [Ut @ dN.T for dN in nl._dN]


def test_j8_plastic_cantilever_newton_unloading_dissipation_energy():
    """J8 (clause b: test_j8b_*): (a) loading 10 + unloading 10 converges; (c) permanent max|u_res| ≥ 1e-3 max|u|; (d) Σ w σ_y Δα ≥ 0
    every step; (e) Σ λ_{n+1} Fᵀ Δu_n = Σ_gp w E [ψ_e(end) + Σ ½Δε_e:C:Δε_e + Σ σ:Δε_p] (relative ≤ 1e-8)."""
    d, nl, F = _plastic_cantilever()
    E = np.ones(nl.n_elements)
    U, rep, cp = nl.solve_history(E, F, [1.0, 0.0])
    it_alg = rep.iterations
    diss = [s.dissipation for s in rep.steps if s.converged]
    try:
        _d, nlc, _F = _plastic_cantilever("continuum")
        _U, repc, _cp = nlc.solve_history(E, F, [1.0, 0.0])
        it_cont: int | None = repc.iterations
    except NonlinearConvergenceError:
        it_cont = None
    umax = np.abs(U[:, 0]).max()
    ures = np.abs(U[:, 1]).max()
    # (e): replay the converged load factors increment by increment from checkpoints
    lams = [s.load_factor for s in rep.steps if s.converged]
    j2 = nl.model
    cpk = None
    u_prev = np.zeros(d.ndof)
    st_prev = StateStore(j2, nl.n_elements * nl.ngp).committed
    ext, alg, wp = 0.0, 0.0, 0.0
    w = nl._w
    for lam in lams:
        Ui, _r, cpk = nl.solve_history(E, F, [lam], checkpoint=cpk)
        u = Ui[:, 0]
        ext += lam * float(F @ (u - u_prev))
        Hn, Hp = _gauss_strains(nl, u), _gauss_strains(nl, u_prev)
        n = nl.n_elements
        for g in range(nl.ngp):
            sl = {k: v[g * n:(g + 1) * n] for k, v in st_prev.items()}
            r = j2.evaluate(Kinematics(Hn[g], "small", "strain"), None, sl, MaterialRequest(tangent=False))
            de = 0.5 * (Hn[g] - Hp[g] + np.swapaxes(Hn[g] - Hp[g], 1, 2))
            de3 = np.zeros((n, 3, 3))
            de3[:, :2, :2] = de
            assert r.trial_state is not None
            dep = r.trial_state["plastic_strain"] - sl["plastic_strain"]
            dee = de3 - dep
            tr = np.trace(dee, axis1=1, axis2=2)
            Cdee = 2 * j2.mu * (dee - tr[:, None, None] / 3 * np.eye(3)) + j2.K * tr[:, None, None] * np.eye(3)
            sig = np.zeros((n, 3, 3))
            sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1] = r.stress[:, 0], r.stress[:, 1], r.stress[:, 2]
            sig[:, 1, 0] = sig[:, 0, 1]
            sig[:, 2, 2] = r.diagnostics["sigma_zz"]
            alg += w * float(np.sum(E * 0.5 * np.einsum("nij,nij->n", dee, Cdee)))
            wp += w * float(np.sum(E * np.einsum("nij,nij->n", sig, dep)))
        st_prev = {k: np.asarray(v) for k, v in cpk["state"].items()}
        u_prev = u
    # stored elastic energy at the end
    el = 0.0
    Hn = _gauss_strains(nl, u_prev)
    n = nl.n_elements
    for g in range(nl.ngp):
        sl = {k: v[g * n:(g + 1) * n] for k, v in st_prev.items()}
        r = j2.evaluate(Kinematics(Hn[g], "small", "strain"), None, sl, MaterialRequest(tangent=False, energy=True))
        assert r.energy is not None
        el += w * float(np.sum(E * (r.energy - j2.hardening_energy(sl["alpha"]))))
    ident = abs(ext - (el + alg + wp)) / abs(sum(abs(x) for x in (el, alg, wp)))
    print(f"J8: iterations algorithmic {it_alg}, continuum {it_cont}; max|u| {umax:.3e}, residual {ures:.3e}; "
          f"min step dissipation {min(diss):.2e}; total {sum(diss):.3e}; energy identity {ident:.1e} "
          f"(external {ext:.4e}, elastic {el:.3e}, algorithmic {alg:.3e}, plastic work {wp:.3e})")
    assert all(s.converged for s in rep.steps)                                    # (a) (no cut at all)
    assert ures >= 1e-3 * umax                                                    # (c)
    assert min(diss) >= 0.0                                                       # (d)
    assert ident <= 1e-8                                                          # (e)


J8B_REASON = ("pre-registered J8(b) FAILED (26 Nov 2026): total Newton iterations 27 (algorithmic) vs 34 "
              "(continuum), ratio 1.26 < 1.5. The total includes the 10 elastic unloading steps and the elastic "
              "loading steps (one or two iterations with either tangent); per-step histories: "
              "test_j8b_posthoc_per_step_iterations.")


@pytest.mark.xfail(strict=True, reason=J8B_REASON)
def test_j8b_iterations_algorithmic_vs_continuum():
    """J8(b) as pre-registered: total iterations algorithmic < continuum / 1.5 (or the continuum tangent fails)."""
    _d, nl, F = _plastic_cantilever()
    E = np.ones(nl.n_elements)
    _U, rep, _cp = nl.solve_history(E, F, [1.0, 0.0])
    try:
        _d, nlc, _F = _plastic_cantilever("continuum")
        _U, repc, _cp = nlc.solve_history(E, F, [1.0, 0.0])
        it_cont: int | None = repc.iterations
    except NonlinearConvergenceError:
        it_cont = None
    print(f"J8b: algorithmic {rep.iterations}, continuum {it_cont}")
    assert it_cont is None or rep.iterations < it_cont / 1.5


def test_j8b_posthoc_per_step_iterations():
    """Post hoc: Newton iterations and the last relative residuals per loading step; with the continuum tangent
    each loading step needs at least as many iterations as with the algorithmic one."""
    out = {}
    for kind in ("algorithmic", "continuum"):
        _d, nl, F = _plastic_cantilever(kind)
        _U, rep, _cp = nl.solve_history(np.ones(nl.n_elements), F, [1.0])
        out[kind] = [(round(s.load_factor, 2), s.iterations, [f"{r:.0e}" for r in s.residuals[-3:]])
                     for s in rep.steps if s.converged]
    for kind, rows in out.items():
        print(f"J8b {kind}: " + "; ".join(f"{lam}: {n} it {r}" for lam, n, r in rows))
    assert all(c[1] >= a[1] for a, c in zip(out["algorithmic"], out["continuum"], strict=True))


# ============================================================================================== J9: rollback, restart
def _sha(state) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(np.ascontiguousarray(state[k]).tobytes())
    return h.hexdigest()


def test_j9_rollback_checkpoint_restart_and_case_isolation():
    """J9: (a) a failed increment leaves the committed bytes; (b) checkpoint + restart reproduces the uninterrupted
    path bit for bit; (c) the second of two load cases equals its solve alone bit for bit (U and state)."""
    d, nl, F = _plastic_cantilever()
    E = np.ones(nl.n_elements)
    # (a)
    store = StateStore(nl.model, nl.n_elements * nl.ngp)
    _U, _r, cp = nl.solve_history(E, F, [0.8])
    store.restore(cp["state"])
    before = _sha(store.committed)
    nl1 = NonlinearElasticity(d, nl.fixed, material=nl.model, plane="strain", max_iter=1, rtol=1e-12)
    ok, u, rec, _rel, _eta = nl1._newton(E, 3.0 * F, cp["u"], None, 3.0, store)
    assert not ok and rec.rolled_back and store.trial is None and _sha(store.committed) == before
    # (b)
    Ua, _ra, cpa = nl.solve_history(E, F, [0.5, 1.0, 0.0])
    _U1, _r1, cp1 = nl.solve_history(E, F, [0.5])
    Ub, _rb, cpb = nl.solve_history(E, F, [1.0, 0.0], checkpoint=cp1)
    assert np.array_equal(Ua[:, 2], Ub[:, 1]) and _sha(cpa["state"]) == _sha(cpb["state"])
    # (c)
    F2 = np.column_stack([0.5 * F, F])
    U2 = nl.solve_numpy(E, F2)
    s2 = nl.last_states[1]
    U1 = nl.solve_numpy(E, F)
    assert np.array_equal(U2[:, 1], U1[:, 0]) and _sha(s2) == _sha(nl.last_states[0])
    print(f"J9: committed {before[:12]} unchanged; restart and isolation bit-identical")


# ============================================================================================== N: Newton
class _Scalar:
    """R(x) = r(x) with a user tangent (deliberately wrong when asked), unit scale."""

    conservative = False

    def __init__(self, r, k, energy=None):
        self.r, self.k, self.energy = r, k, energy
        if energy is not None:
            self.conservative = True

    def linearize(self, x):
        R = np.atleast_1d(self.r(x))
        K = np.atleast_2d(self.k(x))
        en = None if self.energy is None else float(self.energy(x))
        return Linearization(R, K, float(np.abs(K).sum()), en)

    def solve(self, lin, rhs):
        try:
            out = np.linalg.solve(lin.matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise LinearSolveFailure(f"LinAlgError: {e}") from e
        return out


def test_n1_failure_codes():
    """N1: every failure code is produced by a constructed problem and appears in the step record."""
    S = NewtonSettings
    cases = {
        "singular_tangent": (_Scalar(lambda x: x - 2.0, lambda x: np.zeros((1, 1))), S()),
        "divergence": (_Scalar(lambda x: x, lambda x: -0.01 * np.ones((1, 1))), S()),
        "stagnation": (_Scalar(lambda x: x, lambda x: 10.0 * np.ones((1, 1))), S()),
        "max_iterations": (_Scalar(lambda x: x, lambda x: 2.0 * np.ones((1, 1))), S(max_iter=5, rtol=1e-12)),
        "non_finite": (_Scalar(lambda x: np.nan * x, lambda x: np.ones((1, 1))), S()),
    }
    got = {}
    for code, (prob, st) in cases.items():
        out = newton(prob, np.array([1.0]), 1.0, st)
        got[code] = out.record.code
        assert not out.converged and out.record.rolled_back
    # invalid deformation: neo-Hooke bar under a huge compression in one increment
    d, fixed, F = _clamped((4, 2), load=(-5.0, 0.0))
    nl = NonlinearElasticity(d, fixed, material="neo_hooke", load_steps=1)
    nl.check_kinematics()
    ok, _u, rec, *_ = nl._newton(np.ones(nl.n_elements), F, np.zeros(d.ndof), None, 1.0)
    got["invalid_deformation"] = rec.code
    # return mapping: Voce with one local iteration in the FE bar
    d2, fixed2, F2, _tip = _bar(2)
    j2 = J2Plasticity(NU, SY, HARD, saturation_stress=1.5e-3, saturation_rate=500.0, local_max_iter=1)
    nl2 = NonlinearElasticity(d2, fixed2, material=j2, plane="strain", load_steps=1)
    store = StateStore(j2, nl2.n_elements * nl2.ngp)
    ok2, _u2, rec2, *_ = nl2._newton(np.ones(nl2.n_elements), 3 * SY * F2, np.zeros(d2.ndof), None, 1.0, store)
    got["return_mapping"] = rec2.code
    # step limit: one iteration allowed, no halving room
    d3, fixed3, F3 = _clamped((6, 3), load=(0.0, -0.05))
    nl3 = NonlinearElasticity(d3, fixed3, material="svk", max_iter=1, load_steps=1, min_step=0.5)
    with pytest.raises(NonlinearConvergenceError) as ei:
        nl3.solve_numpy(np.ones(nl3.n_elements), F3)
    got["step_limit"] = ei.value.code
    assert ei.value.cause == "max_iterations"
    # complex roots: arc length with an inexact tangent (orthogonal corrector larger than the arc)
    got["complex_roots"] = _complex_roots_code()
    print("N1:", got)
    assert all(k == v for k, v in got.items()), got


class _BadTangentPath:
    """x ∈ R², R = (x0 − λ, x1 − x0²), solved with the wrong tangent diag(1, 0.01)."""

    load = np.array([1.0, 0.0])

    def linearize(self, x, lam):
        R = np.array([x[0] - lam, x[1] - x[0] ** 2])
        return Linearization(R, np.diag([1.0, 0.01]), 1.0)

    def solve(self, lin, rhs):
        return np.linalg.solve(lin.matrix, rhs)


def _complex_roots_code() -> str:
    with pytest.raises(ContinuationError) as ei:
        arc_length(_BadTangentPath(), np.zeros(2), 0.0, ArcLengthSettings(ds=0.1, ds_min=0.1), lambda x, lam: False)
    assert ei.value.code == "step_limit"
    return ei.value.cause


def test_n2_line_search_rescues_newton_on_atan():
    """N2: R = atan(x), x₀ = 2: full Newton fails (max_iterations | stagnation | divergence); with backtracking
    |x| ≤ 1e-14 within 15 iterations."""
    prob = _Scalar(np.arctan, lambda x: np.atleast_2d(1.0 / (1.0 + x * x)))
    full = newton(prob, np.array([2.0]), 1.0, NewtonSettings(rtol=1e-14))
    ls = newton(prob, np.array([2.0]), 1.0, NewtonSettings(rtol=1e-14, line_search="backtracking", max_iter=15))
    print(f"N2: full Newton {full.record.code} after {full.record.iterations}; line search {ls.record.code} in "
          f"{ls.record.iterations}, x = {ls.x[0]:.1e}, steps {[round(s, 3) for s in ls.record.step_lengths]}")
    assert not full.converged and full.record.code in ("max_iterations", "stagnation", "divergence")
    assert ls.converged and abs(ls.x[0]) <= 1e-14 and ls.record.iterations <= 15


def test_n2b_energy_merit_only_for_a_descent_direction():
    """N2b: Π = −x²/2 + x⁴/4 at x = 0.5 (indefinite tangent): the energy merit is refused and the switch logged."""
    prob = _Scalar(lambda x: -x + x ** 3, lambda x: np.atleast_2d(-1.0 + 3 * x * x),
                   energy=lambda x: float(-0.5 * x[0] ** 2 + 0.25 * x[0] ** 4))
    out = newton(prob, np.array([0.5]), 1.0, NewtonSettings(line_search="backtracking", merit="energy", rtol=1e-12))
    print("N2b:", out.record.notes[:2], out.record.code)
    assert any("not a descent direction" in n for n in out.record.notes)
    nc = newton(_Scalar(lambda x: x, lambda x: np.ones((1, 1))), np.array([1.0]), 1.0,
                NewtonSettings(line_search="backtracking", merit="energy"))
    assert any("energy merit not available" in n for n in nc.record.notes)


def test_n3_step_log_fields():
    """N3: every converged step records residuals, update norms, code, min det F (finite), energy (log_energy) and,
    for J2, the dissipation; line-search step lengths when enabled. First run: with the residual merit the
    backtracking stalls on this cantilever (step ≈ 0.1 every iteration, stagnation at λ = 0.1; the full Newton step
    raises ‖R‖ in bending, see test_residual_merit_stalls_where_full_newton_converges); the energy merit is used."""
    d, fixed, F = _clamped((8, 3), load=(0.0, -0.05))
    nl = NonlinearElasticity(d, fixed, material="neo_hooke", log_energy=True, line_search="backtracking",
                             merit="energy")
    nl.solve_numpy(np.ones(nl.n_elements), F)
    for s in nl.last_reports[0].steps:
        assert s.converged and s.code == "converged" and len(s.updates) == s.iterations
        assert len(s.step_lengths) == s.iterations
        assert s.min_det_f is not None and s.min_det_f > 0 and s.energy is not None and math.isfinite(s.energy)
    _d, nlp, Fp = _plastic_cantilever()
    _U, rep, _cp = nlp.solve_history(np.ones(nlp.n_elements), Fp, [1.0])
    assert all(s.dissipation is not None and s.dissipation >= 0 for s in rep.steps if s.converged)
    print(f"N3: {len(nl.last_reports[0].steps)} steps, min det F {min(s.min_det_f for s in nl.last_reports[0].steps):.3f}")


def test_residual_merit_stalls_where_full_newton_converges():
    """Post hoc (finding of the first N3 run): on the 8 × 3 neo-Hooke cantilever the full Newton step converges,
    the residual-merit backtracking fails at the first increment, the energy merit converges."""
    d, fixed, F = _clamped((8, 3), load=(0.0, -0.05))
    res = {}
    for ls, merit in (("none", "residual"), ("backtracking", "residual"), ("backtracking", "energy")):
        nl = NonlinearElasticity(d, fixed, material="neo_hooke", line_search=ls, merit=merit, load_steps=10,
                                 min_step=0.1)
        nl.check_kinematics()
        ok, _u, rec, *_ = nl._newton(np.ones(nl.n_elements), 0.1 * F, np.zeros(d.ndof), None, 0.1)
        res[(ls, merit)] = (ok, rec.code, rec.iterations, [round(x, 3) for x in rec.step_lengths[:4]])
    print("merit:", res)
    assert res[("none", "residual")][0] and not res[("backtracking", "residual")][0]         and res[("backtracking", "energy")][0]


# ============================================================================================== A: arc length, truss
def _truss_path():
    tr = TwoBarTruss()
    st = ArcLengthSettings(ds=0.005, rtol=1e-12, eta_tol=0.0, max_steps=2000)
    path = arc_length(tr, np.zeros(2), 0.0, st, lambda x, lam: -x[1] >= 2.5 * tr.h)
    return tr, path


def test_a1_a3_arc_length_traces_the_truss_snap_through():
    """A1: every point on the closed-form curve (|λ − P(w)/P̂| ≤ 1e-9, |u| ≤ 1e-12), w reaches 2.5h; A3: det K
    sign equals the sign of dP/dw at every point farther than one arc length from a limit point."""
    tr, path = _truss_path()
    lam = path.lam
    u = path.component(0)
    w = -path.component(1)
    e1 = np.abs(lam - tr.P(w) / tr.P_ref).max()
    e_u = np.abs(u).max()
    ds = 0.005
    away = (np.abs(w - tr.w1) > ds) & (np.abs(w - tr.w2) > ds)
    sg = np.array([p.det_sign for p in path.points])
    agree = float(np.mean(sg[away] == np.sign(tr.dPdw(w[away]))))
    print(f"A1/A3: {len(lam)} points, {path.rejected} rejected; A1 {e1:.1e}, |u| {e_u:.1e}; w end {w[-1] / tr.h:.3f} h;"
          f" det sign agreement {agree:.3f} over {int(away.sum())} points")
    assert e1 <= 1e-9 and e_u <= 1e-12 and w[-1] >= 2.5 * tr.h
    assert agree == 1.0


A2_REASON = ("pre-registered A2a/A2b FAILED as written (26 Nov 2026): 'max λ over the path ≤ 1 + 1e-9' and the "
             "refinement around the path maximum assumed the first limit point is the maximum, but the path to "
             "w = 2.5h passes w = 2h (P = 0) into the tension branch, where λ = P(2.5h)/P_max = 4.87 at the end point "
             "(the closed form itself, A1 passes). A specification error of the pre-registration, not of the solver; "
             "the post hoc test_a2_posthoc_local_limit_points checks the two local extrema on w ∈ [0, 2h].")


@pytest.mark.xfail(strict=True, reason=A2_REASON)
def test_a2_pre_registered_global_extrema():
    """A2a (as pre-registered): max λ ∈ [1 − 5e-3, 1 + 1e-9], min λ ∈ [−1 − 1e-9, −1 + 5e-3]; A2b: parabola through
    the three points around the maximum, |λ_peak − 1| ≤ 1e-3, |w_peak − w₁| ≤ 0.05h."""
    tr, path = _truss_path()
    lam, w = path.lam, -path.component(1)
    print(f"A2 as registered: max lam {lam.max():.4f} at w {w[int(np.argmax(lam))] / tr.h:.3f} h, min lam "
          f"{lam.min():.4f}")
    assert 1 - 5e-3 <= lam.max() <= 1 + 1e-9 and -1 - 1e-9 <= lam.min() <= -1 + 5e-3
    k = int(np.argmax(lam))
    c2, c1, c0 = np.linalg.solve(np.vander(w[k - 1:k + 2], 3), lam[k - 1:k + 2])
    assert abs(c0 - c1 * c1 / (4 * c2) - 1) <= 1e-3 and abs(-c1 / (2 * c2) - tr.w1) <= 0.05 * tr.h


def test_a2_posthoc_local_limit_points():
    """Post hoc (replaces A2 as written, same thresholds): on w ∈ [0, 2h] the discrete max / min λ lie in
    [1 − 5e-3, 1 + 1e-9] / [−1 − 1e-9, −1 + 5e-3]; parabolic refinement at both: |λ_peak ∓ 1| ≤ 1e-3,
    |w_peak − w₁,₂| ≤ 0.05h."""
    tr, path = _truss_path()
    lam, w = path.lam, -path.component(1)
    inside = w <= 2 * tr.h
    li, wi = lam[inside], w[inside]
    out = []
    for k, target, wt in ((int(np.argmax(li)), 1.0, tr.w1), (int(np.argmin(li)), -1.0, tr.w2)):
        c2, c1, c0 = np.linalg.solve(np.vander(wi[k - 1:k + 2], 3), li[k - 1:k + 2])
        out.append((li[k], c0 - c1 * c1 / (4 * c2), -c1 / (2 * c2), target, wt))
    print("A2 post hoc: " + "; ".join(f"discrete {a:.6f}, refined {b:.7f} at w {c / tr.h:.5f} h (exact {e / tr.h:.5f} h)"
                                      for a, b, c, _t, e in out))
    assert 1 - 5e-3 <= out[0][0] <= 1 + 1e-9 and -1 - 1e-9 <= out[1][0] <= -1 + 5e-3
    for _a, b, c, t, e in out:
        assert abs(b - t) <= 1e-3 and abs(c - e) <= 0.05 * tr.h


def _truss_load_control(tr, end=1.2):
    pts = [(0.0, np.zeros(2))]

    def attempt(lam, x):
        out = newton(TrussAtLoad(tr, lam), x, float(np.linalg.norm(lam * tr.load)), NewtonSettings(rtol=1e-12))
        return out.converged, (out.x if out.converged else x), out.record, out.relative_residual, out.backward_error

    res = load_control(attempt, np.zeros(2), 0.0, end, 1.0 / 20, 1e-4, keep_points=True)
    return res, pts + res.points


def test_a4_load_control_fails_at_the_limit_point():
    """A4 (negative control): load control to λ = 1.2 either ends with step_limit or jumps by ≥ w₂ − w₁ between two
    accepted points; no accepted point is on the unstable branch (det K < 0)."""
    tr = TwoBarTruss()
    res, pts = _truss_load_control(tr)
    w = np.array([-x[1] for _l, x in pts])
    jumps = np.abs(np.diff(w))
    unstable = [lam for lam, x in pts if np.linalg.det(tr.internal(x)[1]) < 0]
    print(f"A4: ok {res.ok}, reached {res.lam:.5f}, cuts {res.cuts}, last code "
          f"{res.records[-1].code}, max jump {jumps.max() / tr.h:.3f} h (w2 - w1 = {(tr.w2 - tr.w1) / tr.h:.3f} h), "
          f"unstable points {len(unstable)}")
    assert (not res.ok) or jumps.max() >= tr.w2 - tr.w1
    assert not unstable


# ============================================================================================== A5: FE arc length
def test_a5_fe_arc_length_matches_load_control():
    """A5: stocky cantilever 16 × 6, SVK plane stress, tip load 0.06: at every arc-length point the load-control
    solution at that λ agrees, ‖u_arc − u_load‖∞/‖u_load‖∞ ≤ 1e-7."""
    d, fixed, F = _clamped((16, 6), load=(0.0, -0.06))
    nl = NonlinearElasticity(d, fixed, nu=0.3, material="svk", rtol=1e-12)
    E = np.ones(nl.n_elements)
    prob = nl.continuation_problem(E, F)
    u_end = nl.solve_numpy(E, F)[:, 0]
    ds = 0.08 * float(np.linalg.norm(u_end[nl.free]))
    path = arc_length(prob, np.zeros(len(nl.free)), 0.0, ArcLengthSettings(ds=ds, rtol=1e-12),
                      lambda x, lam: lam >= 1.0)
    worst = 0.0
    for p in path.points[1:]:
        ul = nl.solve_numpy(E, p.lam * F)[:, 0][nl.free]
        worst = max(worst, float(np.abs(p.x - ul).max() / np.abs(ul).max()))
    print(f"A5: {len(path.points)} points, last lam {path.points[-1].lam:.4f}, max|u|/L "
          f"{np.abs(u_end).max() / 16:.3f}, worst rel {worst:.2e}")
    assert worst <= 1e-7


# ============================================================================================== A6: FE shallow arch
def _arch():
    nx, ny = 40, 8
    frac = np.zeros((nx, ny))
    for i in range(nx):
        yc = int(round(3 * (1 - ((i + 0.5 - 20) / 20) ** 2)))
        frac[i, yc:yc + 2] = 1.0
    d = Domain((nx, ny), h=1.0, frac=frac)
    X = d.node_coords()
    pins = [int(np.flatnonzero((np.abs(X[:, 0] - x0) < 1e-9) & (np.abs(X[:, 1]) < 1e-9))[0]) for x0 in (0.0, 40.0)]
    fixed = np.concatenate([d.dofs(np.array([p])) for p in pins])
    crown = int(np.flatnonzero((np.abs(X[:, 0] - 20) < 1e-9) & (np.abs(X[:, 1] - 5) < 1e-9))[0])
    F = np.zeros(d.ndof)
    F[2 * crown + 1] = -1.0
    return d, fixed, F, crown


def _arch_run():
    d, fixed, F, crown = _arch()
    nl = NonlinearElasticity(d, fixed, nu=0.3, material="svk", rtol=1e-12)
    E = np.ones(nl.n_elements)
    prob = nl.continuation_problem(E, F)
    c = int(np.flatnonzero(nl.free == 2 * crown + 1)[0])
    path = arc_length(prob, np.zeros(len(nl.free)), 0.0, ArcLengthSettings(ds=0.25, rtol=1e-12, max_steps=800),
                      lambda x, lam: -x[c] >= 6.0)
    return nl, E, F, c, path


@pytest.mark.slow
def test_a6_fe_shallow_arch_arc_length_vs_displacement_control():
    """A6: staircase shallow arch: arc length to crown deflection 6 vs independent displacement control at the same
    crown deflections: |λ_dc − λ_arc| ≤ 1e-7 max|λ|; no λ maximum before w = 6 → 'NO LIMIT POINT' (fails)."""
    nl, E, F, c, path = _arch_run()
    lam, w = path.lam, -path.component(c)
    kmax = int(np.argmax(lam))
    has_limit = 0 < kmax < len(lam) - 1 and lam[kmax + 1] < lam[kmax]
    dc = displacement_control(nl.continuation_problem(E, F), np.zeros(len(nl.free)), 0.0, c, list(-w[1:]))
    err = float(np.abs(dc.lam[1:] - lam[1:]).max() / np.abs(lam).max())
    sg = [p.det_sign for p in path.points]
    first_neg = next((i for i, v in enumerate(sg) if v is not None and v < 0), None)
    print(f"A6: {len(lam)} arc points ({path.rejected} rejected), max lam {lam.max():.5e} at w {w[kmax]:.3f}, "
          f"limit {'yes' if has_limit else 'NO LIMIT POINT'}, min lam after {lam[kmax:].min():.4e}; dc vs arc "
          f"{err:.2e}; first det K < 0 at w {w[first_neg] if first_neg is not None else None}, lam/max "
          f"{lam[first_neg] / lam.max() if first_neg is not None else None}")
    assert has_limit
    assert err <= 1e-7


A6B_REASON = ("pre-registered A6b FAILED (26 Nov 2026): load control to 1.2 x the arc-length maximum ends with "
              "step_limit at 0.9998 of the maximum (the jump/fail clause holds), but 3 accepted points have "
              "det K < 0 (lambda/max 0.9975-0.9998). The symmetric arch has an ANTISYMMETRIC bifurcation at "
              "lambda = 0.9972 lambda_max (w = 2.30) before the symmetric limit point (w = 2.47): the smallest "
              "eigenvector there is 100 % antisymmetric. Load control (and arc length) stay on the symmetric primary "
              "path through the bifurcation; the points beyond it are unstable equilibria. No branch switching.")


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=A6B_REASON)
def test_a6b_arch_load_control_negative_control():
    """A6b: load control to 1.2 × max λ fails or jumps by ≥ w(min λ) − w(max λ) and never accepts det K < 0."""
    nl, E, F, c, path = _arch_run()
    lam, w = path.lam, -path.component(c)
    kmax = int(np.argmax(lam))
    lc = nl.continuation_problem(E, F)
    pts = [(0.0, np.zeros(len(nl.free)))]
    target = 1.2 * lam.max()

    class _At:
        conservative = True

        def __init__(self, lmb):
            self.lmb = lmb

        def linearize(self, xx):
            return lc.linearize(xx, self.lmb)

        def solve(self, lin, rhs):
            return lc.solve(lin, rhs)

    def attempt(lmb, x):
        out = newton(_At(lmb), x, float(np.linalg.norm(lmb * lc.load)), NewtonSettings(rtol=1e-12))
        return out.converged, (out.x if out.converged else x), out.record, out.relative_residual, out.backward_error

    res = load_control(attempt, np.zeros(len(nl.free)), 0.0, target, target / 20, 1e-4 * target, keep_points=True)
    pts += res.points
    wl = np.array([-x[c] for _l, x in pts])
    jumps = np.abs(np.diff(wl))
    unstable = [lm / lam.max() for lm, x in pts if lc.det_sign(lc.linearize(x, lm)) < 0]
    kmin = kmax + int(np.argmin(lam[kmax:]))
    print(f"A6b: ok {res.ok}, reached {res.lam / lam.max():.4f} of max, last code {res.records[-1].code}, max jump "
          f"{jumps.max():.3f} (w(min) - w(max) = {w[kmin] - w[kmax]:.3f}), unstable {[round(u, 4) for u in unstable]}")
    assert (not res.ok) or jumps.max() >= w[kmin] - w[kmax]
    assert not unstable


# ============================================================================================== E1, T: gradients
def _study(domain, physics, F, objective, rmin=1.5):
    chain = Chain([Embed(domain), DensityFilter(domain, rmin), Passive(domain)])
    return Study(domain, physics, F, chain, SIMP(domain, E0=1.0, penal=3.0, emin_ratio=1e-6), objective, [], MMA())


def _gradient_problem(material="neo_hooke", cutoff=0.1):
    d, fixed, F = _clamped((12, 4), load=(0.0, -0.01))
    nl = NonlinearElasticity(d, fixed, nu=0.3, material=material, rtol=1e-13,
                             energy_interpolation=EnergyInterpolation(cutoff=cutoff, cutoff_step=0.0))
    return d, nl, F


@pytest.mark.parametrize("cutoff", [0.1, 0.3])
@pytest.mark.parametrize("response", ["end_compliance", "end_strain_energy"])
def test_e1_t3_fixed_cutoff_gradient_finite_difference(response, cutoff):
    """E1 / T3: fixed cutoff (cutoff_step = 0), random grey design, 8 components, central FD step 1e-6:
    max|g − g_FD|/max|g_FD| ≤ 1e-6; max|u| ≥ 5 % of the span."""
    d, nl, F = _gradient_problem(cutoff=cutoff)
    st = _study(d, nl, F, EndCompliance() if response == "end_compliance" else EndStrainEnergy())
    x = 0.05 + 0.95 * np.random.default_rng(8).random(d.n_design)
    ev = st.evaluate(x)
    idx = np.random.default_rng(9).choice(d.n_design, 8, replace=False)
    fd = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        fd.append((st.evaluate(xp, gradients=False).f0 - st.evaluate(xm, gradients=False).f0) / 2e-6)
    fd = np.array(fd)
    rel = float(np.abs(ev.df0[idx] - fd).max() / np.abs(fd).max())
    umax = np.abs(ev.U).max()
    print(f"E1 {response} c={cutoff}: rel {rel:.2e}, max|u|/span {umax / 12:.3f}, cutoffs {nl.cutoffs()}")
    assert umax >= 0.05 * 12
    assert rel <= 1e-6


def _taylor(response_cls, material, scale=1.0):
    d, nl, F = _gradient_problem(material)
    st = _study(d, nl, F, response_cls())
    rng = np.random.default_rng(12)
    x = 0.3 + 0.4 * rng.random(d.n_design)
    dvec = 2 * rng.random(d.n_design) - 1
    dvec /= np.abs(dvec).max()
    ev = st.evaluate(x)
    f0, g = ev.f0, float(ev.df0 @ dvec) * scale
    hs = 0.05 * 2.0 ** -np.arange(7)
    r = np.array([abs(st.evaluate(x + h * dvec, gradients=False).f0 - f0 - h * g) for h in hs])
    p = np.log2(r[:-1] / r[1:])
    return p, r


def _consecutive(p, lo=1.8, hi=2.2) -> bool:
    return any(lo <= p[k] <= hi and lo <= p[k + 1] <= hi for k in range(len(p) - 1))


@pytest.mark.parametrize("material", ["svk", "neo_hooke"])
def test_t1_t2_end_strain_energy_taylor_remainder(material):
    """T1: two consecutive Taylor orders of EndStrainEnergy in [1.8, 2.2]; T2: with the directional derivative
    × 1.01 the last two orders ≤ 1.3."""
    p, r = _taylor(EndStrainEnergy, material)
    p2, _r2 = _taylor(EndStrainEnergy, material, 1.01)
    print(f"T {material}: p = {np.round(p, 3).tolist()}, r = {[f'{v:.1e}' for v in r]}; "
          f"T2 p = {np.round(p2, 3).tolist()}")
    assert _consecutive(p)
    assert p2[-1] <= 1.3 and p2[-2] <= 1.3


# ============================================================================================== D: V4 cause
D1_REASON = ("pre-registered D1 FAILED in its second clause (26 Nov 2026): the last relative residual of every "
             "step is at the rounding floor (<= 3.1e2 phi in all 20 steps; the first clause holds), but the residual "
             "before it is below 1e4 phi at lambda = 0.6, 0.7 (SVK: 1.4e3, 6.9e2 phi) and 0.9, 1.0 (neo-Hooke: "
             "4.6e3, 1.7e3 phi). Newton reaches the floor in two quadratic steps from ~1e-1, so no step has three "
             "residuals in the asymptotic range above the floor: the three-point order of V4 always contains a "
             "floor-limited value (record D2).")


@pytest.mark.xfail(strict=True, reason=D1_REASON)
def test_d1_v4_last_residual_is_at_the_rounding_floor():
    """D1 (pre-registered hypothesis): in the V4 beam (4 layers, L/H = 50, α = 2, 10 steps), at every step with
    λ ≥ 0.6, the last relative residual is ≤ 1e3 φ and the one before ≥ 1e4 φ, φ = ε (‖K_T‖₁‖u‖ + ‖λF‖)/‖λF‖ at the
    converged state; SVK and neo-Hooke."""
    from toporlogy.fem.elastic import matrix_norm1
    out = {}
    for material in ("svk", "neo_hooke"):
        H, n, slender = 1.0, 4, 50
        d = Domain((slender * n, n), h=H / n)
        X = d.node_coords()
        L = slender * H
        fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
        tip = np.flatnonzero(np.abs(X[:, 0] - L) < 1e-9)
        wts = np.ones(len(tip))
        wts[[0, -1]] = 0.5
        wts /= wts.sum()
        F = np.zeros(d.ndof)
        F[2 * tip + 1] = -2.0 * (H ** 3 / 12) / L ** 2 * wts
        nl = NonlinearElasticity(d, fixed, nu=0.0, material=material, rtol=1e-10, load_steps=10)
        E = np.ones(nl.n_elements)
        _U, rep, snaps = nl._solve_case(E, F, None, factors=[k / 10 for k in range(1, 11)])
        rows = []
        for s, lam, u in zip([s for s in rep.steps if s.converged], [k / 10 for k in range(1, 11)], snaps,
                             strict=True):
            A = nl._assemble_mat(nl.element_state(u, E)["K"])
            fl = lam * F[nl.free]
            phi = np.finfo(float).eps * (matrix_norm1(A) * np.linalg.norm(u[nl.free]) + np.linalg.norm(fl)) \
                / np.linalg.norm(fl)
            rows.append((lam, s.residuals[-1] / phi, s.residuals[-2] / phi))
        out[material] = rows
    for m, rows in out.items():
        print(f"D1 {m}: " + "; ".join(f"{lam:.1f}: last {a:.1e} phi, previous {b:.1e} phi" for lam, a, b in rows))
    ok = all(a <= 1e3 and b >= 1e4 for rows in out.values() for lam, a, b in rows if lam >= 0.6 - 1e-12)
    assert ok
