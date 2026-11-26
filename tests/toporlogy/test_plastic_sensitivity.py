"""Design sensitivities of a path-dependent material (J2 plasticity) by the path adjoint of
:class:`toporlogy.fem.nonlinear.NonlinearElasticity` and the local vector-Jacobian product
:meth:`toporlogy.core.constitutive.J2Plasticity.evaluate_vjp`.

Checks and thresholds are the pre-registered ones of ``docs/olcumler/plastik_duyarlilik_onkayit.md`` (committed
26 Nov 2026 12:36 +03:00, commit 2f0352e, before the first run); the labels PS0-PS9 refer to that table. Checks
added after the first run are marked "post hoc".
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from toporlogy.core.constitutive import J2Plasticity, Kinematics, LinearElastic, MaterialRequest
from toporlogy.core.domain import Domain
from toporlogy.design.chain import Chain, DensityFilter, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.elements import D_iso, D_plane_strain
from toporlogy.fem.nonlinear import NonlinearElasticity
from toporlogy.optimization.nonlinear import EndCompliance, EndStrainEnergy, PlasticDissipation
from toporlogy.optimization.study import Study
from toporlogy.optimizers.mma import MMA

SY, HARD, NU = 1e-3, 0.05, 0.3
VOCE = dict(saturation_stress=1.5e-3, saturation_rate=500.0)
H_FD = 1e-5


# ---------------------------------------------------------------------------------------------- set-up (pre-registered)
def _cantilever(dim=2, sy=SY, voce=False, cases=1, tangent_kind="algorithmic", load_steps=10,
                load_scale=1.6):
    """Q4 plane strain 12 × 4 or H8 6 × 2 × 2 cantilever, clamped at x = 0; tip load 1.6 × the beam's first yield;
    E = 0.5 + 0.5 U(0, 1) (seed 3). ``cases`` 2: second case tip down + ½ axial (2-D)."""
    shape = (12, 4) if dim == 2 else (6, 2, 2)
    d = Domain(shape, h=1.0)
    X = d.node_coords()
    L = float(shape[0])
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(np.abs(X[:, 0] - L) < 1e-9)
    P = load_scale * SY * (4.0 ** 2 / (6 * 12) if dim == 2 else 2.0 ** 3 / (6 * 6))
    F = np.zeros((d.ndof, cases))
    F[dim * tip + 1, 0] = -P / len(tip)
    if cases == 2:
        F[dim * tip + 1, 1] = -P / len(tip)
        F[dim * tip, 1] = 0.5 * P / len(tip)
    kw = dict(plane="strain") if dim == 2 else {}
    mat = J2Plasticity(NU, sy, HARD, **(VOCE if voce else {}))
    nl = NonlinearElasticity(d, fixed, material=mat, rtol=1e-12, load_steps=load_steps, tangent_kind=tangent_kind,
                             **kw)
    E = 0.5 + 0.5 * np.random.default_rng(3).random(nl.n_elements)
    return d, nl, F, E


def _plastic_sets(nl):
    """Per case and increment: Gauss points with Δα > 0 (bool arrays, rows g·n + e)."""
    out = []
    for c, path in enumerate(nl.last_paths):
        alphas = [s["alpha"] for s in path.state_before] + [nl.last_states[c]["alpha"]]
        out.append([alphas[k + 1] - alphas[k] > 0 for k in range(len(path.u))])
    return out


def _same_sets(a, b):
    return len(a) == len(b) and all(len(x) == len(y) and all(np.array_equal(p, q) for p, q in zip(x, y, strict=True))
                                    for x, y in zip(a, b, strict=True))


def _plastic_elements(nl, sets):
    n = nl.n_elements
    hit = np.zeros(n, bool)
    for case in sets:
        for s in case:
            hit |= s.reshape(nl.ngp, n).any(axis=0)
    return hit


def _components(nl, sets):
    """8 components (seed 4): 4 elements with plastic points, 4 without (pre-registration: ≥ 4 plastic)."""
    hit = _plastic_elements(nl, sets)
    rng = np.random.default_rng(4)
    pl, el = np.flatnonzero(hit), np.flatnonzero(~hit)
    assert len(pl) >= 4, "PS0: fewer than 4 elements with plastic points"
    return np.concatenate([rng.choice(pl, 4, replace=False), rng.choice(el, min(4, len(el)), replace=False)])


def _responses(nl, E, F, weights):
    U = nl.solve_numpy(E, F)
    J = float(sum(w * F[:, c] @ U[:, c] for c, w in enumerate(weights)))
    return J, nl.path_dissipation(E, weights=weights)


def _fd(nl, E, F, idx, weights):
    """Central differences of (end compliance, dissipation) and the PS0 preconditions."""
    _responses(nl, E, F, weights)
    sets0 = _plastic_sets(nl)
    cuts = [sum(r.cuts for r in nl.last_reports)]
    same = True
    gJ, gD = [], []
    for i in idx:
        v = []
        for s in (1.0, -1.0):
            Ep = E.copy()
            Ep[i] += s * H_FD
            v.append(_responses(nl, Ep, F, weights))
            cuts.append(sum(r.cuts for r in nl.last_reports))
            same &= _same_sets(sets0, _plastic_sets(nl))
        gJ.append((v[0][0] - v[1][0]) / (2 * H_FD))
        gD.append((v[0][1] - v[1][1]) / (2 * H_FD))
    n_plastic_inc = max(sum(int(s.any()) for s in case) for case in sets0)
    return np.array(gJ), np.array(gD), {"cuts": int(sum(cuts)), "same_sets": bool(same),
                                        "plastic_increments": n_plastic_inc}


def _adjoint(nl, E, F, weights):
    """End compliance gradient through the autograd solve and the dissipation gradient."""
    Et = torch.tensor(E, dtype=torch.float64, requires_grad=True)
    U = nl.solve(Et, F)
    w = torch.tensor(weights, dtype=torch.float64)
    J = (w * (torch.as_tensor(F) * U).sum(dim=0)).sum()
    J.backward()
    assert Et.grad is not None
    gD = nl.path_adjoint_gradient(E, nl.last_paths, dissipation_weights=weights)
    return Et.grad.numpy().copy(), gD


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def _check_ps0(info, tag):
    print(f"PS0 {tag}: {info}")
    assert info["cuts"] == 0 and info["same_sets"] and info["plastic_increments"] >= 3


# ---------------------------------------------------------------------------------------------- PS1
@pytest.mark.parametrize("dim", [2, 3])
def test_ps1_elastic_limit_equals_the_small_strain_sensitivity(dim):
    """PS1: σ_y = 1e30: the path-adjoint end-compliance gradient equals (a) the single adjoint of LinearElastic and
    (b) the linear Elasticity compliance gradient: ‖Δg‖∞/‖g‖∞ ≤ 1e-10."""
    d, nl, F, E = _cantilever(dim, sy=1e30)
    g_path, _ = _adjoint(nl, E, F, [1.0])
    D = D_plane_strain(NU) if dim == 2 else D_iso(NU)
    kw = dict(plane="strain") if dim == 2 else {}
    lin = NonlinearElasticity(d, nl.fixed, material=LinearElastic(D), rtol=1e-12, **kw)
    Et = torch.tensor(E, dtype=torch.float64, requires_grad=True)
    (torch.as_tensor(F) * lin.solve(Et, F)).sum().backward()
    assert Et.grad is not None
    g_lin = Et.grad.numpy().copy()
    el = Elasticity(d, nl.fixed, elasticity=D)
    Et2 = torch.tensor(E, dtype=torch.float64, requires_grad=True)
    (torch.as_tensor(F) * el.solve(Et2, F)).sum().backward()
    assert Et2.grad is not None
    g_el = Et2.grad.numpy().copy()
    ra, rb = _rel(g_path, g_lin), _rel(g_path, g_el)
    print(f"PS1 {dim}D: vs LinearElastic {ra:.2e}, vs Elasticity {rb:.2e}; adjoint solves {nl.n_adjoint_solves}")
    assert ra <= 1e-10 and rb <= 1e-10


# ---------------------------------------------------------------------------------------------- PS2, PS4, PS6
def _last_tangent_gradient(nl, E, F, w):
    """PS6: one adjoint with the last tangent per case, history ignored: −λ_Nᵀ f_unit(u_N, s_{N−1})."""
    g_last = np.zeros(nl.n_elements)
    for c, path in enumerate(nl.last_paths):
        u, s = path.u[-1], path.state_before[-1]
        st = nl.element_state(u, E, need_K=True, need_dE=True, state=s)
        A = nl._assemble_mat(st["K"])
        nl.solver.factor(A)
        lam = np.zeros(nl.domain.ndof)
        lam[nl.free] = np.asarray(nl.solver.solve(w[c] * F[nl.free, c])).reshape(-1)
        g_last -= np.einsum("ep,ep->e", lam[nl.edof], st["dfdE"])
    return g_last


@pytest.fixture(scope="module")
def q4_two_cases():
    """The PS2a / PS4 / PS6 problem: Q4, linear hardening, two cases (weights 1, 0.5), FD of 8 components."""
    _d, nl, F, E = _cantilever(2, cases=2)
    w = [1.0, 0.5]
    gJ, gD = _adjoint(nl, E, F, w)
    D0 = nl.path_dissipation(E, weights=w)
    idx = _components(nl, _plastic_sets(nl))
    g_last = _last_tangent_gradient(nl, E, F, w)
    fdJ, fdD, info = _fd(nl, E, F, idx, w)
    return {"gJ": gJ[idx], "gD": gD[idx], "D": D0, "last": g_last[idx], "fdJ": fdJ, "fdD": fdD, "info": info,
            "idx": idx}


def test_ps2a_ps4_q4_linear_hardening_two_cases(q4_two_cases):
    """PS2a end compliance (two cases, weights 1 and 0.5) and PS4 plastic dissipation against central differences
    (h = 1e-5) ≤ 1e-5, with the PS0 preconditions."""
    r = q4_two_cases
    _check_ps0(r["info"], "PS2a/PS4")
    rJ, rD = _rel(r["gJ"], r["fdJ"]), _rel(r["gD"], r["fdD"])
    print(f"PS2a: rel {rJ:.2e}; PS4: D = {r['D']:.4e}, rel {rD:.2e}; components {r['idx'].tolist()}")
    assert rJ <= 1e-5
    assert r["D"] > 0 and rD <= 1e-5


@pytest.mark.xfail(strict=True, reason="PS6 pre-registered FAIL (first run 26 Nov 2026): the last-tangent gradient "
                                       "is off by 3.15e-3 relative, below the required 1e-2 (the path adjoint: "
                                       "1.9e-8); docs/olcumler/plastik_duyarlilik_onkayit.md")
def test_ps6_last_tangent_gradient_differs_from_finite_differences(q4_two_cases):
    """PS6 negative control: the gradient that ignores the history differs from FD by ≥ 1e-2 relative."""
    r = q4_two_cases
    rL = _rel(r["last"], r["fdJ"])
    print(f"PS6 last tangent: rel {rL:.2e}")
    assert rL >= 1e-2


@pytest.mark.parametrize("scale", [1.6, 2.5, 3.5])
def test_ps6_posthoc_last_tangent_error_versus_load(scale):
    """Post hoc measurement (no threshold; PS6 FAIL follow-up): ‖g_last − g_path‖∞/‖g_path‖∞ for tip loads 1.6,
    2.5, 3.5 × first yield (the path adjoint is the reference, verified by PS2a at 1.6)."""
    _d, nl, F, E = _cantilever(2, load_scale=scale)
    g_path, _gD = _adjoint(nl, E, F, [1.0])
    g_last = _last_tangent_gradient(nl, E, F, [1.0])
    frac = float(np.mean(_plastic_elements(nl, _plastic_sets(nl))))
    print(f"PS6 post hoc x{scale}: last tangent rel {_rel(g_last, g_path):.2e}; elements with plastic points "
          f"{frac:.2f}")
    assert np.all(np.isfinite(g_last))


@pytest.mark.parametrize("variant", ["voce", "h8"])
def test_ps2b_ps2c_voce_and_h8(variant):
    """PS2b (Voce hardening, Q4) and PS2c (H8 3-D): end compliance against central differences ≤ 1e-5 (PS0)."""
    _d, nl, F, E = _cantilever(3 if variant == "h8" else 2, voce=variant == "voce")
    gJ, gD = _adjoint(nl, E, F, [1.0])
    idx = _components(nl, _plastic_sets(nl))
    fdJ, fdD, info = _fd(nl, E, F, idx, [1.0])
    _check_ps0(info, variant)
    rJ, rD = _rel(gJ[idx], fdJ), _rel(gD[idx], fdD)
    print(f"PS2 {variant}: rel {rJ:.2e} (dissipation, not pre-registered here: {rD:.2e})")
    assert rJ <= 1e-5


# ---------------------------------------------------------------------------------------------- PS3
def _points(model, n=20, seed=5):
    rng = np.random.default_rng(seed)
    ey = SY
    eps = rng.normal(size=(n, 3, 3)) * ey * rng.uniform(0.2, 3.0, size=(n, 1, 1))
    eps = 0.5 * (eps + np.swapaxes(eps, 1, 2))
    ep = rng.normal(size=(n, 3, 3)) * 0.5 * ey
    ep = 0.5 * (ep + np.swapaxes(ep, 1, 2))
    ep -= (np.trace(ep, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
    committed = {"plastic_strain": ep, "alpha": rng.uniform(0, 2 * ey, size=n)}
    return eps, committed


def _sym(rng, n, scale=1.0):
    a = rng.normal(size=(n, 3, 3)) * scale
    return 0.5 * (a + np.swapaxes(a, 1, 2))


@pytest.mark.parametrize("voce", [False, True])
def test_ps3_local_vjp_against_finite_differences_and_the_tangent(voce):
    """PS3a: ⟨VJP(ē), δ⟩ = d/dt ⟨ē, update(x + tδ)⟩ (central FD) per point, relative ≤ 1e-6; ≥ 10 of 20 plastic.
    PS3b: the ε̄ of a σ̄-only cotangent equals C_algᵀ σ̄ (Voigt) per point, relative ≤ 1e-12."""
    model = J2Plasticity(NU, SY, HARD, **(VOCE if voce else {}))
    eps, com = _points(model)
    n = len(eps)
    _s, C6, _t, diag, _p = model.update(eps, com)
    plastic = diag["plastic"] > 0.5
    rng = np.random.default_rng(6)
    sb, eb = _sym(rng, n), _sym(rng, n)
    ab, db = rng.normal(size=n), rng.normal(size=n)
    de, dep = _sym(rng, n, SY), _sym(rng, n, SY)
    dep -= (np.trace(dep, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
    da = rng.uniform(0.0, SY, size=n)

    def phi(t):
        c = {"plastic_strain": com["plastic_strain"] + t * dep, "alpha": com["alpha"] + t * da}
        s, _C, tr, dg, _p = model.update(eps + t * de, c, MaterialRequest(tangent=False))
        val = (np.einsum("nij,nij->n", sb, s) + np.einsum("nij,nij->n", eb, tr["plastic_strain"])
               + ab * tr["alpha"] + db * dg["delta_alpha"])
        return val, dg["plastic"]

    t = 1e-6
    (vp, pp), (vm, pm) = phi(t), phi(-t)
    assert np.array_equal(pp, pm) and np.array_equal(pp, diag["plastic"]), "plastic set changed within ±t"
    fd = (vp - vm) / (2 * t)
    e_bar, ep_bar, a_bar = model.update_vjp(eps, com, sb, eb, ab, db)
    vjp = (np.einsum("nij,nij->n", e_bar, de) + np.einsum("nij,nij->n", ep_bar, dep) + a_bar * da)
    rel_a = float(np.max(np.abs(vjp - fd) / np.abs(fd)))
    # PS3b
    e_only, _x, _y = model.update_vjp(eps, com, sb)
    VA = np.array([0, 1, 2, 1, 0, 0])
    VB = np.array([0, 1, 2, 2, 2, 1])
    m = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    ref = np.einsum("nvw,nv->nw", C6, m * sb[:, VA, VB])
    got = e_only[:, VA, VB]
    rel_b = float(np.max(np.abs(got - ref).max(axis=1) / np.abs(ref).max(axis=1)))
    print(f"PS3 voce={voce}: {int(plastic.sum())}/{n} plastic; PS3a rel {rel_a:.2e}; PS3b rel {rel_b:.2e}")
    assert plastic.sum() >= 10 and (~plastic).sum() >= 1
    assert rel_a <= 1e-6
    assert rel_b <= 1e-12


def test_ps3_evaluate_vjp_conventions_match_evaluate():
    """Post hoc (convention check of the protocol wrapper): evaluate_vjp in Voigt / grad form against central
    differences of evaluate, Q4 plane strain and 3-D points, relative ≤ 1e-6."""
    model = J2Plasticity(NU, SY, HARD)
    rng = np.random.default_rng(7)
    for dim, plane in ((2, "strain"), (3, "3d")):
        n = 12
        H = rng.normal(size=(n, dim, dim)) * 2 * SY
        com = model.initial_state(n)
        com["alpha"] = rng.uniform(0, SY, n)
        nv = 3 if dim == 2 else 6
        Sb, dB = rng.normal(size=(n, nv)), rng.normal(size=n)
        dH = rng.normal(size=(n, dim, dim)) * SY

        def phi(t, H=H, dH=dH, com=com, Sb=Sb, dB=dB, plane=plane):
            r = model.evaluate(Kinematics(H + t * dH, "small", plane), None, com, MaterialRequest(tangent=False))
            return np.einsum("nv,nv->n", Sb, r.stress) + dB * r.dissipation
        t = 1e-6
        fd = (phi(t) - phi(-t)) / (2 * t)
        Hb, _cb = model.evaluate_vjp(Kinematics(H, "small", plane), None, com, Sb, None, dB)
        vjp = np.einsum("nij,nij->n", Hb, dH)
        rel = float(np.max(np.abs(vjp - fd) / np.abs(fd)))
        print(f"evaluate_vjp {dim}D: rel {rel:.2e}")
        assert rel <= 1e-6


# ---------------------------------------------------------------------------------------------- PS5
def test_ps5_gradient_independent_of_newton_iterations():
    """PS5: forward Newton with the continuum tangent (more iterations) and with the algorithmic one give the same
    end-compliance and dissipation gradients: ≤ 1e-8 relative."""
    _d, nl, F, E = _cantilever(2)
    gJ, gD = _adjoint(nl, E, F, [1.0])
    it_a = sum(r.iterations for r in nl.last_reports)
    _d, nlc, _F, _E = _cantilever(2, tangent_kind="continuum")
    gJc, gDc = _adjoint(nlc, E, F, [1.0])
    it_c = sum(r.iterations for r in nlc.last_reports)
    rJ, rD = _rel(gJc, gJ), _rel(gDc, gD)
    print(f"PS5: iterations algorithmic {it_a}, continuum {it_c}; rel compliance {rJ:.2e}, dissipation {rD:.2e}")
    assert it_c > it_a
    assert rJ <= 1e-8 and rD <= 1e-8


# ---------------------------------------------------------------------------------------------- PS8
def test_ps8_study_end_compliance_gradient():
    """PS8: Study (density filter + SIMP p = 3, x ∈ [0.4, 1]) with the J2 physics and EndCompliance: gradient in x
    against central differences (h = 1e-5, 6 components) ≤ 1e-5."""
    d, nl, F, _E = _cantilever(2)
    chain = Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)])
    st = Study(d, nl, F[:, 0], chain, SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6), EndCompliance(), [], MMA())
    x = 0.4 + 0.6 * np.random.default_rng(10).random(d.n_design)
    ev = st.evaluate(x)
    plastic = int(sum(s.sum() for case in _plastic_sets(nl) for s in case))
    idx = np.random.default_rng(11).choice(d.n_design, 6, replace=False)
    fd = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += H_FD
        xm[i] -= H_FD
        fd.append((st.evaluate(xp, gradients=False).f0 - st.evaluate(xm, gradients=False).f0) / (2 * H_FD))
    rel = _rel(ev.df0[idx], np.array(fd))
    print(f"PS8: rel {rel:.2e}; plastic point-increments {plastic}")
    assert plastic > 0
    assert rel <= 1e-5


# ---------------------------------------------------------------------------------------------- PS9
class _NoVJP:
    """A path-dependent model without evaluate_vjp (J2 delegated)."""

    name, kinematics, conservative = "no_vjp", "small", False

    def __init__(self) -> None:
        self._m = J2Plasticity(NU, SY, HARD)
        self.state_fields = self._m.state_fields

    def initial_state(self, n):
        return self._m.initial_state(n)

    def evaluate(self, kin, params, committed, request):
        return self._m.evaluate(kin, params, committed, request)


def test_ps9_fail_closed_contracts():
    """PS9: a path-dependent model without evaluate_vjp is refused (NotImplementedError); a path recorded at other
    moduli is refused (ValueError); PlasticDissipation needs a path-dependent physics; EndStrainEnergy refuses J2."""
    d, nl, F, E = _cantilever(2)
    bad = NonlinearElasticity(d, nl.fixed, material=_NoVJP(), plane="strain")
    with pytest.raises(NotImplementedError, match="evaluate_vjp"):
        bad.solve(torch.tensor(E), F)
    nl.solve_numpy(E, F)
    with pytest.raises(ValueError, match="other moduli"):
        nl.path_adjoint_gradient(E * 1.01, nl.last_paths, G=F)
    with pytest.raises(ValueError, match="other moduli"):
        nl.path_dissipation(E * 1.01)
    chain = Chain([Embed(d), Passive(d)])
    simp = SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6)
    hyper = NonlinearElasticity(d, nl.fixed, material="svk", plane="strain")
    st = Study(d, hyper, F[:, 0], chain, simp, PlasticDissipation(), [], MMA())
    with pytest.raises(ValueError, match="path-dependent"):
        st.evaluate(np.full(d.n_design, 0.8))
    st2 = Study(d, nl, F[:, 0], chain, simp, EndStrainEnergy(), [], MMA())
    with pytest.raises(ValueError, match="hyperelastic"):
        st2.evaluate(np.full(d.n_design, 0.8))


def test_ps4_study_plastic_dissipation_gradient():
    """Post hoc: PlasticDissipation inside Study (density filter + SIMP) against central differences in x
    (h = 1e-5, 6 components) ≤ 1e-5 (the PS4 threshold; the pre-registered PS4 is the physics-level check)."""
    d, nl, F, _E = _cantilever(2)
    chain = Chain([Embed(d), DensityFilter(d, 1.5), Passive(d)])
    st = Study(d, nl, F[:, 0], chain, SIMP(d, E0=1.0, penal=3.0, emin_ratio=1e-6), PlasticDissipation(), [], MMA())
    x = 0.4 + 0.6 * np.random.default_rng(10).random(d.n_design)
    ev = st.evaluate(x)
    idx = np.random.default_rng(11).choice(d.n_design, 6, replace=False)
    fd = []
    for i in idx:
        xp, xm = x.copy(), x.copy()
        xp[i] += H_FD
        xm[i] -= H_FD
        fd.append((st.evaluate(xp, gradients=False).f0 - st.evaluate(xm, gradients=False).f0) / (2 * H_FD))
    rel = _rel(ev.df0[idx], np.array(fd))
    print(f"PS4 study: D = {ev.f0:.4e}, rel {rel:.2e}")
    assert ev.f0 > 0
    assert rel <= 1e-5


# ---------------------------------------------------------------------------------------------- measurement
def test_ps_measurement_cost_and_load_step_dependence():
    """Measurement (no threshold): adjoint solves and backward / forward time; J and gradient change 10 → 20 steps."""
    _d, nl, F, E = _cantilever(2)
    t0 = time.perf_counter()
    U = nl.solve_numpy(E, F)
    t1 = time.perf_counter()
    n0 = nl.n_adjoint_solves
    g10 = nl.path_adjoint_gradient(E, nl.last_paths, G=F)
    t2 = time.perf_counter()
    J10 = float(F[:, 0] @ U[:, 0])
    _d, nl20, _F, _E = _cantilever(2, load_steps=20)
    U20 = nl20.solve_numpy(E, F)
    g20 = nl20.path_adjoint_gradient(E, nl20.last_paths, G=F)
    J20 = float(F[:, 0] @ U20[:, 0])
    print(f"PS measure: increments {len(nl.last_paths[0].u)}, adjoint solves {nl.n_adjoint_solves - n0}, "
          f"forward {t1 - t0:.3f} s, backward {t2 - t1:.3f} s; 10 -> 20 steps: J rel {abs(J20 - J10) / J10:.2e}, "
          f"gradient rel {_rel(g20, g10):.2e}")
    assert nl.n_adjoint_solves - n0 >= 1
