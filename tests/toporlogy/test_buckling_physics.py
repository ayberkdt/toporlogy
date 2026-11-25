"""Buckling physics added to fem/buckling.py and optimization/buckling.py: preload, cluster cut warning, low-density
share, first-failure load factor, non-isotropic elasticity.

References: the linearity of K_G in the load (a preload c·P on the same pattern lowers λ by exactly c); the
definition of the low-density share; uniaxial tension (stress = traction, so first failure at allowable/traction,
with preload (allowable − σ0)/σ1); the isotropic material passed as a general D must give the legacy λ; a slender
orthotropic column buckles with its axial modulus (Euler); finite differences for the preload gradient (1e-6, as
test_buckling.py). Tolerances: 1e-8 relative where only the eigen-solver tolerance (1e-10) separates the sides.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.core.material import Orthotropic, Strength, unit_elasticity
from toporlogy.fem.buckling import BucklingClusterWarning, LinearBuckling
from toporlogy.optimization.buckling import BucklingKS, first_failure_load_factor
from toporlogy.optimization.diagnostics import study_gradcheck
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.strength import make_criterion
from toporlogy.optimization.stress import FailureIndex


def _column(shape=(4, 60), h=1.0, D=None, side=0.0):
    d = T.Domain(shape, h)
    X = d.node_coords()
    ax = d.dim - 1
    fixed = d.dofs(np.flatnonzero(X[:, ax] == 0))
    top = np.flatnonzero(np.isclose(X[:, ax], X[:, ax].max()))
    F = d.load_vector(2)
    F[d.dim * top + ax, 0] = -1.0 / len(top)                 # case 0: unit compression (scaled)
    F[d.dim * top[:1], 0] += side
    F[:, 1] = F[:, 0]                                        # case 1: same pattern (preload, scaled below)
    m = T.Elasticity(d, fixed, 0.3, elasticity=D)
    return d, m, F


def test_preload_on_the_same_pattern_shifts_lambda_by_its_factor():
    d, m, F = _column()
    E = np.full(d.n_active, 1000.0)
    U = m.solve_numpy(E, F)
    lb = LinearBuckling(m)
    mu0, _ = lb.modes(E, E, U[:, 0], 3)
    lam0 = 1 / mu0
    for c in (0.3 * lam0[0], 0.8 * lam0[0]):
        mu, _ = lb.modes(E, E, U[:, 0], 3, KG0=lb.geometric_matrix(E, c * U[:, 1]))
        print(f"\n  preload c = {c:.4g}: lambda {1 / mu[0]:.8g}, expected {lam0[0] - c:.8g}")
        assert np.max(np.abs((1 / mu) - (lam0 - c)) / (lam0 - c)) <= 1e-8
    with pytest.raises(ValueError, match="preload alone buckles"):
        lb.modes(E, E, U[:, 0], 3, KG0=lb.geometric_matrix(E, 1.05 * lam0[0] * U[:, 1]))


def _study(response, preload_scale=0.0):
    d = T.Domain((6, 18), 1.0, thickness=0.8)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] == 0))
    top = np.flatnonzero(np.isclose(X[:, 1], 18))
    F = d.load_vector(2)
    F[2 * top + 1, 0] = -1.0 / len(top)
    F[2 * top[:1], 0] = 0.02
    F[2 * top + 1, 1] = -preload_scale / len(top)
    F[2 * top[-1:], 1] = -0.01
    st = T.Study(d, T.Elasticity(d, fixed, 0.3), F, T.standard_chain(d, 1.5, beta=2.0), T.SIMP(d, E0=50.0),
                 T.Compliance(), [T.Constraint(T.Volume(), 0.6), T.Constraint(response, 1.0, scale=1.0)], T.MMA())
    return st, d


def test_gradient_with_preload():
    resp = BucklingKS(n_modes=4, rho=20.0, scale=1.0, preload_case=1)
    st, d = _study(resp, preload_scale=0.5)
    x0 = 0.45 + 0.4 * np.random.default_rng(11).random(d.n_design)
    r1 = study_gradcheck(st, x0, 1, n_dirs=4, h=2e-4, seed=2)
    r2 = study_gradcheck(st, x0, 1, n_dirs=4, h=4e-4, seed=2)
    err = max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
              for a, b in zip(r1["directions"], r2["directions"], strict=True))
    print(f"\n  buckling KS with preload, gradient: {err:.2e}; load factors {np.round(resp.last_load_factors, 3)}")
    assert err <= 1e-6


def test_cluster_cut_warning_on_a_square_column():
    """A square 3-D column has double eigenvalues: aggregating 1 mode cuts the first pair (warning), 2 do not."""
    d = T.Domain((3, 3, 24), 0.5)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 2] == 0))
    top = np.flatnonzero(np.isclose(X[:, 2], X[:, 2].max()))
    F = d.load_vector()
    F[3 * top + 2, 0] = -1.0 / len(top)
    for n, expect in ((1, True), (2, False)):
        resp = BucklingKS(n_modes=n, rho=20.0, scale=1.0)
        st = T.Study(d, T.Elasticity(d, fixed, 0.3), F, T.standard_chain(d, 1.2), T.SIMP(d, E0=50.0),
                     T.Compliance(), [T.Constraint(T.Volume(), 0.6), T.Constraint(resp, 1.0, scale=1.0)], T.MMA())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            st.evaluate(np.full(d.n_design, 1.0), gradients=False)
        hit = any(issubclass(x.category, BucklingClusterWarning) for x in w)
        assert hit is expect and resp.last_cluster_cut is expect


def test_low_density_share():
    d, m, F = _column((4, 30))
    E = np.full(d.n_active, 1000.0)
    U = m.solve_numpy(E, F)
    lb = LinearBuckling(m)
    _, modes = lb.modes(E, E, U[:, 0], 3)
    assert np.allclose(lb.low_density_share(E, modes, np.full(d.n_active, 0.9)), 0.0)
    assert np.allclose(lb.low_density_share(E, modes, np.full(d.n_active, 0.1)), 1.0)
    x = np.full(d.n_active, 0.9)
    upper = d.element_centers()[d.active_elements, 1] > 15
    x[upper] = 0.2
    share = lb.low_density_share(E, modes, x)
    assert np.all((share > 0) & (share < 1))


def test_first_failure_load_factor_uniaxial_tension():
    nx, ny = 8, 2
    d = T.Domain((nx, ny))
    X = d.node_coords()
    left = np.flatnonzero(X[:, 0] == 0)
    right = np.flatnonzero(np.isclose(X[:, 0], nx))
    fixed = np.concatenate([d.dofs(left, 0), d.dofs(left[:1], 1)])
    F = d.load_vector(2)
    w = np.ones(len(right))
    w[[0, -1]] = 0.5
    F[2 * right, 0] = 2.0 * w                                   # σ1 = 2 (variable)
    F[2 * right, 1] = 5.0 * w                                   # σ0 = 5 (preload)
    m = T.Elasticity(d, fixed, 0.3)
    U = torch.as_tensor(m.solve_numpy(np.full(d.n_active, 7.0), F))
    ctx = EvalContext(domain=d, xphys=torch.ones(d.shape, dtype=torch.float64),
                      E=torch.full(d.shape, 7.0, dtype=torch.float64), U=U)
    field = FailureIndex(d, make_criterion("von_mises", Strength(30.0, safety_factor=1.5), eps=0.0), E0=7.0)
    lam = first_failure_load_factor(ctx, field, case=0)
    lam_p = first_failure_load_factor(ctx, field, case=0, preload_case=1)
    print(f"\n  first failure: {lam:.12g} (expected 10), with preload {lam_p:.12g} (expected 7.5)")
    assert abs(lam - 10.0) <= 1e-9 * 10 and abs(lam_p - 7.5) <= 1e-9 * 7.5


def test_general_D_buckling():
    d, m_iso, F = _column((4, 60))
    E = np.full(d.n_active, 1000.0)
    U = m_iso.solve_numpy(E, F)
    lam_iso = 1 / LinearBuckling(m_iso).modes(E, E, U[:, 0], 2)[0]
    D = unit_elasticity(Orthotropic.isotropic(1.0, 0.3), 2, "stress")
    _, m_d, _ = _column((4, 60), D=D)
    lam_d = 1 / LinearBuckling(m_d).modes(E, E, m_d.solve_numpy(E, F)[:, 0], 2)[0]
    assert np.max(np.abs(lam_d / lam_iso - 1)) <= 1e-8


def test_orthotropic_column_buckles_with_its_axial_modulus():
    """A slender column (4 × 60, clamped-free) whose material is E/4 along the axis and stiffer across must buckle
    like the isotropic E/4 column (Euler: only the axial modulus enters; shear differs by about 0.15 % here).

    Record: the first version compared one mesh (4 elements across) with a 1 % bound and FAILED at 2.1 %. Cause
    measured: Q4 bending error (parasitic shear grows with G/E_axial, 2× larger for this material). The ratio is
    1.0210, 1.0059, 1.0021 at 4, 8, 16 elements across (second order). The test now asserts that convergence."""
    ortho = Orthotropic(E1=1.0, E2=0.25, E3=0.25, G12=0.2, G13=0.2, G23=0.1, nu12=0.3, nu13=0.3, nu23=0.3)
    err = []
    for nb in (4, 8, 16):
        lam = []
        for D in (unit_elasticity(Orthotropic.isotropic(0.25, 0.3), 2, "stress") * 0.25,
                  unit_elasticity(ortho, 2, "stress")):
            d = T.Domain((nb, 15 * nb), 4.0 / nb)
            X = d.node_coords()
            top = np.flatnonzero(np.isclose(X[:, 1], X[:, 1].max()))
            F = d.load_vector()
            F[2 * top + 1, 0] = -1.0 / len(top)
            m = T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 1] == 0)), elasticity=D)
            E = np.full(d.n_active, 1000.0)
            lam.append(1 / LinearBuckling(m).modes(E, E, m.solve_numpy(E, F)[:, 0], 1)[0][0])
        err.append(abs(lam[1] / lam[0] - 1))
    print(f"\n  orthotropic / isotropic E/4 column, 4 / 8 / 16 across: {np.round(err, 5)}")
    assert err[2] <= 5e-3 and err[0] / err[1] >= 2.5 and err[1] / err[2] >= 2.5


# ================================================================================ singular stresses
def _wall(n: int, exclude_r: float | None = None):
    """The solid wall of Ferrari, Sigmund & Guest (2021) §5.2 on an n × n mesh (n = 40 k; supported node columns of
    the 320 mesh in units of n/40), opening void, x-load 1e-2 on the left edge. Returns λ1, its localisation and
    the energy share of mode 1 within 0.02 of a clamped-patch end."""
    h = 1.0 / n
    d = T.Domain((n, n), h)
    t = n // 40
    ix, iy = np.indices((n, n))
    row, col = n - 1 - iy, ix
    void = (row >= 2 * n // 5 - 1) & (col >= 2 * n // 5 - 1) & (col <= n - n // 5 - 1)
    x = np.where(void, 0.0, 1.0).ravel()
    X = d.node_coords()
    xi = np.rint(X[:, 0] / h).astype(int)
    cols = np.r_[0:t + 1, 2 * n // 5 - t - 1:2 * n // 5, 4 * n // 5:4 * n // 5 + t + 1, n - t:n + 1]
    fixed = d.dofs(np.flatnonzero(np.isclose(X[:, 1], 0) & np.isin(xi, cols)))
    left = np.flatnonzero(np.isclose(X[:, 0], 0))
    F = d.load_vector()
    w = np.ones(len(left))
    w[[0, -1]] = 0.5
    F[2 * left, 0] = 1e-2 * w / w.sum()
    m = T.Elasticity(d, fixed, 0.3)
    E = 1e-6 + x ** 3 * (1 - 1e-6)
    U = m.solve_numpy(E, F)[:, 0]
    c = d.element_centers()
    ends = np.array([t, 2 * n // 5 - t - 1, 2 * n // 5 - 1, 4 * n // 5, 4 * n // 5 + t, n - t]) * h
    dist = np.min(np.hypot(c[:, 0][:, None] - ends[None, :], c[:, 1][:, None]), axis=1)
    EG = x ** 3 if exclude_r is None else x ** 3 * (dist >= exclude_r)
    lb = LinearBuckling(m)
    mu, modes = lb.modes(E, EG, U, 2)
    _, kE = lb.mode_forms(modes)
    en = E * kE[:, 0]
    return 1 / mu[0], lb.localization(E, modes)[0], float(en[dist < 0.02].sum() / en.sum())


def test_singular_stress_mode_does_not_converge_and_is_flagged():
    """The solid wall's first mode sits at the end of a clamped patch (singular stress): under refinement λ1 keeps
    falling (no mesh limit) and the mode's energy concentrates (localisation rising). A global mode (Euler column)
    keeps its energy spread. Measured 2026-11-25: λ1 2.852, 2.106, 1.602; localisation 0.881, 0.921, 0.954."""
    lam, loc, near = zip(*(_wall(n) for n in (40, 80, 160)))
    print(f"\n  wall lambda1 {np.round(lam, 4)}, localisation {np.round(loc, 3)}, energy near a clamp end "
          f"{np.round(near, 3)}")
    assert lam[0] / lam[1] > 1.2 and lam[1] / lam[2] > 1.2
    assert loc[0] < loc[1] < loc[2] and loc[2] > 0.9 and near[2] > 0.7
    d, m, F = _column((4, 60))
    E = np.full(d.n_active, 1000.0)
    lb = LinearBuckling(m)
    _, modes = lb.modes(E, E, m.solve_numpy(E, F)[:, 0], 1)
    print(f"  Euler column localisation {lb.localization(E, modes)[0]:.3f}")
    assert lb.localization(E, modes)[0] < 0.1


def test_excluding_stress_stiffness_moves_the_singular_mode():
    """With no stress stiffness within 0.05 of the clamped-patch ends the clamp-end mode disappears (its energy
    there drops) and λ1 rises; the mode then localises at the opening corner (moved, not removed: module notes)."""
    lam0, _, near0 = _wall(80)
    lam1, loc1, near1 = _wall(80, exclude_r=0.05)
    print(f"\n  wall 80²: lambda1 {lam0:.4f} -> {lam1:.4f} with exclusion; energy near clamp ends {near0:.3f} -> "
          f"{near1:.3f}; localisation {loc1:.3f}")
    assert lam1 > 1.5 * lam0 and near1 < 0.01 and near0 > 0.5


# ================================================================================ incompatible modes in K_G
def _fs_column_error(nx: int, element: str, modes_in_kg: bool = True) -> float:
    """Ferrari & Sigmund (2019) column (Lx 10, Ly 1, t 0.005, E 2e11, ν 0, f0 2.5e5): (λ1 f0 − Pc) / Pc."""
    ny = nx // 10
    d = T.Domain((nx, ny), 10.0 / nx, thickness=0.005)
    X = d.node_coords()
    tip = np.flatnonzero(np.isclose(X[:, 0], 10.0))
    w = np.ones(len(tip))
    w[np.isclose(X[tip, 1], 0) | np.isclose(X[tip, 1], 1)] = 0.5
    F = d.load_vector()
    F[2 * tip, 0] = -2.5e5 * w / w.sum()
    m = T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 0] == 0)), 0.0, element=element)
    E = np.full(d.n_active, 2e11)
    mu, _ = LinearBuckling(m, modes_in_kg=modes_in_kg).modes(E, E, m.solve_numpy(E, F)[:, 0], 2)
    return (2.5e5 / mu[0] - 2.05617e6) / 2.05617e6


def test_modes_in_kg_improve_coarse_meshes():
    """Q4IM with the condensed modes in K_G is closer to the converged 2-D value (Richardson from 80×8 and 160×16)
    than with the compatible field only, on 20×2 and 40×4. Measured 2026-11-25: −0.00561 / −0.00511 (20×2); the
    scratch code with rectangular elements gave −0.005614 for the same formulation."""
    e80, e160 = _fs_column_error(80, "Q4IM"), _fs_column_error(160, "Q4IM")
    limit = e160 + (e160 - e80) / 3
    for nx in (20, 40):
        new, old = _fs_column_error(nx, "Q4IM"), _fs_column_error(nx, "Q4IM", modes_in_kg=False)
        print(f"\n  column {nx}x{nx // 10}: modes in K_G {new:.5f}, compatible only {old:.5f}, limit {limit:.5f}")
        assert abs(new - limit) < abs(old - limit)
    assert abs(_fs_column_error(20, "Q4IM") - (-0.005614)) <= 1e-5


def test_gradient_q4im_with_modes_in_kg():
    d = T.Domain((6, 18), 1.0, thickness=0.8)
    X = d.node_coords()
    top = np.flatnonzero(np.isclose(X[:, 1], 18))
    F = d.load_vector()
    F[2 * top + 1, 0] = -1.0 / len(top)
    F[2 * top[:1], 0] = 0.02
    resp = BucklingKS(n_modes=4, rho=20.0, scale=1.0)
    st = T.Study(d, T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 1] == 0)), 0.3, element="Q4IM"), F,
                 T.standard_chain(d, 1.5, beta=2.0), T.SIMP(d, E0=50.0), T.Compliance(),
                 [T.Constraint(T.Volume(), 0.6), T.Constraint(resp, 1.0, scale=1.0)], T.MMA())
    x0 = 0.45 + 0.4 * np.random.default_rng(12).random(d.n_design)
    r1 = study_gradcheck(st, x0, 1, n_dirs=4, h=2e-4, seed=2)
    r2 = study_gradcheck(st, x0, 1, n_dirs=4, h=4e-4, seed=2)
    err = max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
              for a, b in zip(r1["directions"], r2["directions"], strict=True))
    print(f"\n  Q4IM buckling KS gradient: {err:.2e}")
    assert err <= 1e-6
