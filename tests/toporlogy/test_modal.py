"""Eigenfrequency layer: consistent mass, eigen solve, eigenvalue sensitivities, dynamic interpolations, responses.

Pre-registered thresholds: gradients ≤ 1e-6 relative error (:func:`toporlogy.checks.gradcheck`, float64,
modes well separated); mass matrices equal to the repository's element library (aerokit.fem.elements) to 1e-12;
eigenpairs equal to a dense ``scipy.linalg.eigh`` to 1e-9 relative. Euler–Bernoulli ratios are reported and
bounded loosely (the FE model includes shear and rotary inertia and a fully clamped root, EB does not).

FD step for eigenvalue checks: h = 1e-4 (central differences) instead of the default 1e-6. Measured on the 2-D test
problem (13 Nov 2026): along a direction with a small directional derivative (3e-5 against λ up to 0.33) the
h = 1e-6 difference quotient is round-off limited (1.6e-5 relative with the sparse solver, 4.7e-5 with a dense
``scipy.linalg.eigh``), while h = 1e-4 / 1e-5 agree with the analytic value to 9e-8 / 2e-8. The analytic
gradient is the same in both cases; only the reference is noisy at h = 1e-6.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from toporlogy import SIMP, Constraint, Domain, Study, Volume, gradcheck, standard_chain, study_gradcheck
from toporlogy.dynamics import (EigenvalueGap, Eigenvalue, Frequency, LinearMass, MinEigenvalueKS, Negated,
                                   TwoRangeMass, TwoRangeSIMP, min_frequency_constraint)
from toporlogy.fem.modal import Modal, RepeatedEigenvalueWarning, element_mass, find_clusters
from toporlogy.optim import Optimizer

TOL = 1e-6
H_FD = 1e-4        # FD step for eigenvalue checks (docstring)
BETA1 = 1.8751040687119611          # first root of cos βL cosh βL = −1 (clamped-free beam)


def _clamped_x0(d: Domain) -> np.ndarray:
    return d.dofs(d.nodes_where(lambda X: X[:, 0] < 1e-9))


def _report(name: str, r: dict) -> float:
    print(f"\n  gradcheck {name}: max_rel_error = {r['max_rel_error']:.2e}")
    return r["max_rel_error"]


def _domain_2d() -> Domain:
    shape = (12, 5)
    mask = np.ones(shape, bool)
    mask[8:, 3:] = False                          # L shape
    mask[4:6, 1:3] = False                        # hole
    keep = np.zeros(shape, bool)
    keep[-1, :2] = True
    void = np.zeros(shape, bool)
    void[2, 4] = True
    frac = np.where(mask, 1.0, 0.0)
    frac[0, 0] = 0.7
    return Domain(shape, 0.5, frac=frac, keep=keep, void=void, thickness=0.4)


# ============================================================================================ mass matrices
def test_element_mass_matches_element_library():
    pytest.importorskip("aerokit.fem.elements", reason="optional cross-check against the general FE element library")
    from aerokit.fem.elements.plane import Q4, D_isotropic
    from aerokit.fem.elements.solid import H8, D_isotropic3d
    X2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    m2 = Q4([0, 1, 2, 3], D_isotropic(1.0, 0.3, "stress"), t=1.0, rho=1.0).m(X2)
    assert np.allclose(element_mass(2), m2, atol=1e-12)
    X3 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    m3 = H8(list(range(8)), D_isotropic3d(1.0, 0.3), rho=1.0).m(X3)
    assert np.allclose(element_mass(3), m3, atol=1e-12)
    # closed form: ∫ N_a N_b = Π_axes (1/3 same, 1/6 different) per direction
    assert np.isclose(element_mass(3)[0, 0], 1 / 27) and np.isclose(element_mass(2)[0, 2], 1 / 18)


@pytest.mark.parametrize("dim", [2, 3])
def test_assembled_mass_total_and_point_mass(dim):
    d = _domain_2d() if dim == 2 else Domain((4, 3, 2), 0.5)
    m = Modal(d, fixed=[], n_modes=2, sigma=-1.0)
    rho = np.random.default_rng(3).uniform(0.5, 2.0, m.n_elements)
    M = m.mass_matrix(rho)
    vol = d.h ** 3 if dim == 3 else d.thickness * d.h ** 2
    for k in range(dim):                                     # rigid translation t: tᵀ M t = total mass
        t = np.zeros(d.ndof)
        t[k::dim] = 1.0
        tf = t[m.free]
        assert np.isclose(tf @ (M @ tf), (rho * vol).sum(), rtol=1e-12)
    node = d.nodes_where(lambda X: np.all(np.abs(X - X.max(axis=0)) < 1e-9, axis=1) if dim == 3
                         else (X[:, 0] < 1e-9) & (X[:, 1] < 1e-9))
    m.add_point_mass(node, 0.37)
    M2 = m.mass_matrix(rho)
    t = np.zeros(d.ndof)
    t[0::dim] = 1.0
    assert np.isclose(t[m.free] @ (M2 @ t[m.free]), (rho * vol).sum() + 0.37 * len(node), rtol=1e-12)
    K = m.stiffness(rho)
    assert abs(K - K.T).max() < 1e-12 and abs(M - M.T).max() < 1e-14


# ============================================================================================ eigen solve
@pytest.mark.parametrize("dim", [2, 3])
def test_eigenpairs_match_dense(dim):
    d = _domain_2d() if dim == 2 else Domain((6, 3, 2), 0.5)
    m = Modal(d, _clamped_x0(d), n_modes=5)
    rng = np.random.default_rng(4)
    E, rho = rng.uniform(0.3, 1.0, m.n_elements), rng.uniform(0.3, 1.0, m.n_elements)
    m.add_point_mass([d.nn - 1], 0.05)
    lam, Phi = m.solve_numpy(E, rho, 5)
    K, M = m.stiffness(E).toarray(), m.mass_matrix(rho).toarray()
    ref = sla.eigh(K, M, eigvals_only=True, subset_by_index=[0, 4])
    assert np.allclose(lam, ref, rtol=1e-9)
    assert np.allclose(Phi.T @ M @ Phi, np.eye(5), atol=1e-10)
    assert np.allclose(Phi.T @ K @ Phi, np.diag(lam), atol=1e-9 * lam.max())
    res = np.linalg.norm(K @ Phi - (M @ Phi) * lam, axis=0) / np.linalg.norm(K @ Phi, axis=0)
    assert res.max() < 1e-9


def _cantilever_ratio(n: int, element: str, dim: int) -> tuple[float, int]:
    """ω1 / ω_EB for a 10:1 cantilever with n elements through the depth (depth 1, length 10)."""
    h = 1.0 / n
    if dim == 3:
        d = Domain((10 * n, n, n), h)
        I, A = 1.0 / 12, 1.0
    else:
        d = Domain((10 * n, n), h, thickness=0.3)
        I, A = 0.3 / 12, 0.3
    m = Modal(d, _clamped_x0(d), element=element, n_modes=2)
    E0, rho0 = 70e9, 2700.0                                   # SI: λ in rad²/s² directly
    lam, _ = m.modes_numpy(np.full(m.n_elements, E0), np.full(m.n_elements, rho0), 2)
    w_eb = BETA1 ** 2 * np.sqrt(E0 * I / (rho0 * A * 10.0 ** 4))
    if dim == 3:                                              # square section: two equal bending modes
        assert abs(lam[1] - lam[0]) <= 1e-8 * lam[0]
        assert len(find_clusters(lam, 1e-6)) == 1
    return float(np.sqrt(lam[0]) / w_eb), m.n_free


def test_cantilever_vs_euler_bernoulli_3d():
    out = {}
    t0 = time.perf_counter()
    for el in ("H8", "H8IM"):
        out[el] = [_cantilever_ratio(n, el, 3)[0] for n in (2, 4)]
        print(f"\n  3-D cantilever 10:1, {el}: w1/w_EB at 20x2x2, 40x4x4 = {np.round(out[el], 4)}")
    print(f"  ({time.perf_counter() - t0:.1f} s)")
    # H8 (full integration) is stiffer and converges from above; H8IM is near-converged already
    assert out["H8"][0] > out["H8"][1] > 0.99 and out["H8"][1] < 1.03
    assert 0.98 < out["H8IM"][1] < 1.01 and abs(out["H8IM"][1] - out["H8IM"][0]) < 0.01


def test_cantilever_vs_euler_bernoulli_2d():
    r = [_cantilever_ratio(n, "Q4", 2)[0] for n in (2, 4, 8, 16)]
    print(f"\n  2-D Q4 cantilever 10:1: w1/w_EB at depth 2, 4, 8, 16 = {np.round(r, 4)}")
    assert all(a > b for a, b in zip(r, r[1:], strict=False))        # monotone from above (shear-locking Q4)
    assert 0.98 < r[-1] < 1.0                                        # shear + rotary inertia lower EB slightly


def test_frequency_units_hz():
    d = Domain((40, 4), 0.025, thickness=0.01)                      # 1 m × 0.1 m aluminium cantilever
    m = Modal(d, _clamped_x0(d), n_modes=2)
    E = SIMP(d, E0=70e9, penal=3.0)
    m.mass = LinearMass(d, rho0=2700.0)
    from toporlogy.responses import EvalContext
    x = torch.ones(d.shape, dtype=torch.float64)
    ctx = EvalContext(d, x, E(x), physics=m)
    f = float(Frequency(1)(ctx))
    assert np.isclose(f, np.sqrt(float(Eigenvalue(1)(ctx))) / (2 * np.pi), rtol=1e-14)
    I, A = 0.01 * 0.1 ** 3 / 12, 0.01 * 0.1
    f_eb = BETA1 ** 2 * np.sqrt(70e9 * I / (2700.0 * A)) / (2 * np.pi)
    print(f"\n  f1 = {f:.2f} Hz, Euler-Bernoulli {f_eb:.2f} Hz, ratio {f / f_eb:.4f}")
    assert 0.98 < f / f_eb < 1.03
    # same model in mm / MPa / (t/mm³) units → identical Hz with the unit factors
    d2 = Domain((40, 4), 25.0, thickness=10.0)
    m2 = Modal(d2, _clamped_x0(d2), n_modes=2, mass=LinearMass(d2, rho0=2.7e-9))
    ctx2 = EvalContext(d2, x, SIMP(d2, E0=70e3)(x), physics=m2)
    f2 = float(Frequency(1, E_unit=1e6, rho_unit=1e12, length_unit=1e-3)(ctx2))
    assert np.isclose(f2, f, rtol=1e-8)


# ============================================================================================ gradients
def test_eigenvalue_gradient_direct_2d():
    d = _domain_2d()
    m = Modal(d, _clamped_x0(d), n_modes=4)
    m.add_point_mass([d.nn // 2], 0.02)
    na = m.n_elements
    rng = np.random.default_rng(5)
    lam0, _ = m.solve_numpy(np.ones(na), np.ones(na), 5)
    gaps = np.diff(lam0) / lam0[1:]
    print(f"\n  relative gaps of modes 1-5: {np.round(gaps, 3)}")
    w = torch.as_tensor(rng.standard_normal(4), dtype=torch.float64)

    def fn(v):
        return (w * m.eigenvalues(v[:na], v[na:], 4)).sum()
    r = gradcheck(fn, rng.uniform(0.4, 1.0, 2 * na), n_dirs=5, h=H_FD)
    assert _report("lam_1..4(E, rho) 2-D", r) <= TOL


@pytest.mark.parametrize("element", ["H8", "H8IM"])
def test_eigenvalue_gradient_through_chain_3d(element):
    shape = (6, 3, 2)
    keep = np.zeros(shape, bool)
    keep[-1, 0, :] = True
    d = Domain(shape, 0.5, keep=keep)
    m = Modal(d, _clamped_x0(d), element=element, n_modes=3, mass=TwoRangeMass(d, rho0=1.0))
    chain = standard_chain(d, rmin=1.5, projection=True, beta=2.0)
    interp = TwoRangeSIMP(d, penal=3.0)
    act = torch.as_tensor(m.elements)
    rng = np.random.default_rng(6)

    def fn(x):
        xp = chain(x)
        E, rho = interp(xp).reshape(-1)[act], m.density(xp).reshape(-1)[act]
        lam = m.eigenvalues(E, rho, 3)
        return lam[0] + 0.3 * lam[2]
    x0 = rng.uniform(0.2, 0.8, d.n_design)
    lam = m.solve_numpy(*[np.asarray(t) for t in _ER(chain, interp, m, x0)], 4)[0]
    print(f"\n  {element}: lam1..4 = {lam}")
    r = gradcheck(fn, x0, n_dirs=4, h=H_FD)
    assert _report(f"lam through chain 3-D {element}", r) <= TOL


def _ER(chain, interp, m, x0):
    with torch.no_grad():
        xp = chain(torch.as_tensor(x0))
        act = torch.as_tensor(m.elements)
        return interp(xp).reshape(-1)[act].numpy(), m.density(xp).reshape(-1)[act].numpy()


@pytest.mark.parametrize("mod", ["two_range_simp", "two_range_mass", "linear_mass"])
def test_interpolations_gradient(mod):
    d = _domain_2d()
    rng = np.random.default_rng(8)
    link = {"two_range_simp": TwoRangeSIMP(d, E0=2.0, q=1.0), "two_range_mass": TwoRangeMass(d, rho0=3.0),
            "linear_mass": LinearMass(d, rho0=3.0, rhomin_ratio=1e-3)}[mod]
    W = torch.as_tensor(rng.standard_normal(d.shape))
    x = rng.uniform(0.01, 0.6, d.shape)
    x[np.abs(x - 0.1) < 0.01] = 0.05                               # stay away from the kink at x_t = 0.1
    r = gradcheck(lambda v: (W * link(v)).sum(), x, n_dirs=5)
    assert _report(mod, r) <= TOL
    if mod != "linear_mass":                                        # continuity at the threshold
        dd = Domain((1, 2))
        lk = TwoRangeSIMP(dd) if mod == "two_range_simp" else TwoRangeMass(dd)
        v = lk(torch.tensor([[0.1 - 1e-12, 0.1 + 1e-12]], dtype=torch.float64))
        assert abs(float(v[0, 0] - v[0, 1])) < 1e-9


def _modal_study(objective, constraints, d=None, mass=None, point=0.0, n_modes=4):
    d = d or _domain_2d()
    m = Modal(d, _clamped_x0(d), n_modes=n_modes, mass=mass)
    if point:
        m.add_point_mass(d.nodes_where(lambda X: X[:, 0] > X[:, 0].max() - 1e-9)[:1], point)
    chain = standard_chain(d, rmin=1.5, projection=False)

    class _Null(Optimizer):
        pass
    return Study(d, m, d.load_vector(), chain, TwoRangeSIMP(d), objective, constraints, _Null())


@pytest.mark.parametrize("which", ["frequency", "ks", "pnorm", "margin", "gap"])
def test_response_gradients(which):
    resp = {"frequency": Frequency(2, E_unit=1e3, length_unit=0.1), "ks": Negated(MinEigenvalueKS(3, rho=20.0)),
            "pnorm": MinEigenvalueKS(3, rho=8.0, kind="pnorm"), "gap": EigenvalueGap(1)}
    if which == "margin":
        st = _modal_study(Volume(), [min_frequency_constraint(0.05, n_modes=3, rho=30.0)], point=0.01)
        target = 0
    else:
        st = _modal_study(resp[which], [Constraint(Volume(), 0.5)], mass=TwoRangeMass(_domain_2d()), point=0.01)
        target = "objective"
    x0 = np.random.default_rng(9).uniform(0.2, 0.8, st.domain.n_design)
    ev = st.evaluate(x0)
    print(f"\n  {which}: values {ev.values}")
    r = study_gradcheck(st, x0, which=target, n_dirs=4, h=H_FD)
    assert _report(which, r) <= TOL


# ============================================================================================ repeated eigenvalues
def test_repeated_eigenvalue_detection_and_ks():
    d = Domain((8, 2, 2), 0.5)                                      # square section: λ1 = λ2 by symmetry
    m = Modal(d, _clamped_x0(d), n_modes=3)
    chain = standard_chain(d, rmin=1.2, projection=False)
    interp = SIMP(d, penal=3.0)
    act = torch.as_tensor(m.elements)
    x0 = np.full(d.n_design, 0.6)                                   # symmetric design

    def lam_of(x):
        xp = chain(x)
        return m.eigenvalues(interp(xp).reshape(-1)[act], m.density(xp).reshape(-1)[act], 3)
    with torch.no_grad():
        lam = lam_of(torch.as_tensor(x0)).numpy()
    assert len(m.last_clusters) == 1 and list(m.last_clusters[0]) == [0, 1]
    xt = torch.as_tensor(x0).clone().requires_grad_(True)
    with pytest.warns(RepeatedEigenvalueWarning):
        lam_of(xt)[0].backward()
    ks = MinEigenvalueKS(2, rho=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RepeatedEigenvalueWarning)
        r = gradcheck(lambda x: ks.aggregate(lam_of(x)[:2]), x0, n_dirs=4, h=H_FD)
    print(f"\n  square cantilever lam = {lam}")
    assert _report("KS over a double eigenvalue (symmetric design)", r) <= TOL
    # cutting the cluster (KS over mode 1 only, = λ1) must warn
    ks1 = MinEigenvalueKS(1, rho=20.0)
    xt = torch.as_tensor(x0).clone().requires_grad_(True)
    with pytest.warns(RepeatedEigenvalueWarning):
        ks1.aggregate(lam_of(xt)[:1]).backward()
    m.repeated = "raise"
    xt = torch.as_tensor(x0).clone().requires_grad_(True)
    with pytest.raises(FloatingPointError):
        lam_of(xt)[1].backward()


def test_ks_bounds():
    lam = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)
    for kind in ("ks", "pnorm"):
        a = MinEigenvalueKS(3, rho=40.0, scale=2.0, kind=kind).aggregate(lam)
        assert a <= 2.0 + 1e-12 and a >= 2.0 - 2.0 * np.log(3) / 40.0 - 1e-12


# ============================================================================================ spurious modes + optimisation
def _cc_beam(nx=56, ny=8):
    d = Domain((nx, ny), 1.0)
    fixed = d.dofs(d.nodes_where(lambda X: (X[:, 0] < 1e-9) | (X[:, 0] > nx - 1e-9)))
    centre = d.nodes_where(lambda X: (np.abs(X[:, 0] - nx / 2) < 1e-9) & (np.abs(X[:, 1] - ny / 2) < 1e-9))
    return d, fixed, centre


def test_spurious_modes_plain_simp_vs_two_range():
    d, fixed, _ = _cc_beam(32, 8)
    x = np.full(d.shape, 1e-3)
    x[:, :2] = 1.0                                                   # solid strip at the bottom, void above
    xt = torch.as_tensor(x)
    out = {}
    for name, interp in (("SIMP p=3", SIMP(d, penal=3.0, emin_ratio=1e-9)), ("TwoRangeSIMP", TwoRangeSIMP(d))):
        m = Modal(d, fixed, n_modes=3)
        act = m.elements
        E = interp(xt).reshape(-1)[act].numpy()
        rho = m.density(xt).reshape(-1)[act].numpy()
        lam, modes = m.modes_numpy(E, rho, 3)
        share = m.low_density_share(x, rho, modes)
        out[name] = (lam[0], share[0])
        print(f"\n  {name}: lam1 = {lam[0]:.3e}, low-density kinetic-energy share of mode 1 = {share[0]:.3f}")
    assert out["SIMP p=3"][1] > 0.9                                   # localised spurious mode in the void
    assert out["TwoRangeSIMP"][1] < 0.1 and out["TwoRangeSIMP"][0] > 100 * out["SIMP p=3"][0]


def test_maximise_fundamental_eigenvalue_clamped_beam():
    """Du & Olhoff-style: clamped–clamped beam with a non-structural mass at the centre, maximise λ1 (KS)."""
    from toporlogy.optim.mma import MMA
    d, fixed, centre = _cc_beam(56, 8)
    vf = 0.5
    m = Modal(d, fixed, n_modes=4)
    m_ns = 0.2 * vf * d.ne * d.h ** 2 * d.thickness                 # 20 % of the admissible structural mass
    m.add_point_mass(centre, m_ns)
    chain = standard_chain(d, rmin=2.0, projection=False)
    ks = MinEigenvalueKS(3, rho=40.0)
    st = Study(d, m, d.load_vector(), chain, TwoRangeSIMP(d, penal=3.0), Negated(ks),
               [Constraint(Volume(), vf)], MMA(move=0.1))
    t0 = time.perf_counter()
    lam_hist = []
    res = st.run(max_iter=60, tol=1e-3, callback=lambda it, ev, ch: lam_hist.append(m.last_lambda[:3].copy()))
    dt = time.perf_counter() - t0
    lam_hist = np.array(lam_hist)
    print(f"\n  clamped beam 56x8, m_ns = {m_ns:g}: {res.iterations} it, {dt:.1f} s")
    for i in list(range(0, len(lam_hist), 10)) + [len(lam_hist) - 1]:
        print(f"    it {i + 1:3d}: lam1..3 = {np.array2string(lam_hist[i], formatter={'float_kind': '{:.4e}'.format})}  vol = "
              f"{res.history['volume'][i]:.4f}")
    xp = res.xphys
    act = m.elements
    E = st.interp(torch.as_tensor(xp)).reshape(-1)[act].numpy()
    rho = m.density(torch.as_tensor(xp)).reshape(-1)[act].numpy()
    lam, modes = m.modes_numpy(E, rho, 3)
    share = m.low_density_share(xp, rho, modes)
    print(f"    final lam1..3 = {lam}, low-density (xbar<0.1) kinetic share = {np.round(share, 4)}, "
          f"xbar<0.1 in {np.mean(xp < 0.1):.2f} of the domain")
    E_plain = SIMP(d, penal=3.0, emin_ratio=1e-9)(torch.as_tensor(xp)).reshape(-1)[act].numpy()
    lam_p, modes_p = m.modes_numpy(E_plain, rho, 3)
    print(f"    same design with plain SIMP p=3: lam1 = {lam_p[0]:.4e}, low-density share = "
          f"{m.low_density_share(xp, rho, modes_p)[0]:.3f}")
    assert lam_hist[-1, 0] > 1.3 * lam_hist[0, 0]
    assert res.constraint_values["volume"] <= vf * 1.01           # of the returned design (res.xphys above)
    assert share.max() < 0.05
