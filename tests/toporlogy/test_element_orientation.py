"""Per-element material orientation (fibre angle field) and orthotropic T10 re-analysis.

Pre-registered in ``docs/olcumler/eleman_yonu.md`` §0 (2026-11-25 17:54 +0300, before the first run); the test names
carry the item numbers P1 … P7. Tests added after the pre-registration are marked "post hoc" in their docstring.

Independent references: the single-D path of the library (P1, P6); closed forms of a two-orientation strip, parallel
(Voigt, same strain) and series (Reuss, same stress) layers, chosen so that the exact solution is piecewise linear and
therefore reproduced by Q4 / H8 / T10 to rounding (P2); the strength ratio of Tsai-Wu under uniaxial stress along a
material axis, σ/Xt and σ/Yt (P2c); finite differences with Richardson extrapolation (P3); the off-axis modulus
1/E(θ) = c⁴/E1 + (1/G12 − 2ν12/E1)c²s² + s⁴/E2 (Jones 1999, standard transformation; written from memory and
independent of the library's tensor rotation) (P4); results of the isotropic T10 path recorded at d566290 (P5).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.core.material import (Orthotropic, OrthotropicStrength, rotation_about_z, rotation_from_axes,
                                     rotations_about_z, rotations_from_axes, unit_elasticity, unit_elasticity_field)
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.fem.modal import Modal
from toporlogy.optimization.buckling import BucklingKS
from toporlogy.optimization.diagnostics import study_gradcheck
from toporlogy.optimization.local import Aggregated
from toporlogy.optimization.responses import EvalContext
from toporlogy.optimization.strength import make_criterion
from toporlogy.optimization.stress import FailureIndex
from toporlogy.post.verify import solve_t10, stiffness_t10, tet_voxel_rows, voxel_reference

sys.path.insert(0, str(Path(__file__).parent))
from test_post_verify import _box_t10  # noqa: E402

GE = Orthotropic(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)
VOIGT3 = Orthotropic(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.28)
REUSS3 = Orthotropic(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.28 * 9 / 40)
STRENGTH = OrthotropicStrength(Xt=15.0, Xc=12.0, Yt=0.5, Yc=2.0, S12=0.7)


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _R(dim: int) -> np.ndarray:
    return rotation_about_z(25.0) if dim == 2 else rotation_from_axes([1.0, 2.0, 0.5], [0.0, 0.3, 1.0])


def _clamped(shape, F_dir=-1.0):
    d = T.Domain(shape)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    F = d.load_vector()
    tip = np.flatnonzero(np.isclose(X[:, 0], X[:, 0].max()))
    F[d.dim * tip + 1, 0] = F_dir / len(tip)
    F[d.dim * tip[:1], 0] += 0.3
    return d, fixed, F


# ===================================================================================================== P1
@pytest.mark.parametrize("dim,element", [(2, "Q4"), (2, "Q4IM"), (3, "H8"), (3, "H8IM")])
def test_p1_uniform_field_equals_single_D(dim, element):
    """P1: the same R in every element gives D, KE, K, U, compliance and the Tsai-Wu field of the single-D path
    (≤ 1e-14; by construction expected to be identical)."""
    d, fixed, F = _clamped((8, 4) if dim == 2 else (5, 3, 2))
    R = _R(dim)
    D1 = unit_elasticity(GE, dim, "stress", R)
    Dn = unit_elasticity_field(GE, dim, "stress", np.broadcast_to(R, (d.n_active, 3, 3)))
    assert Dn.shape == (d.n_active,) + D1.shape and _rel(Dn, np.broadcast_to(D1, Dn.shape)) <= 1e-14
    m1 = T.Elasticity(d, fixed, element=element, elasticity=D1)
    mn = T.Elasticity(d, fixed, element=element, elasticity=Dn)
    assert mn.per_element and not m1.per_element and len(mn.KE) == 1
    assert _rel(mn.element_stiffness(), np.broadcast_to(m1.KE, mn.element_stiffness().shape)) <= 1e-14
    E = 40.0 * (0.3 + 0.7 * np.random.default_rng(3).random(d.n_active))
    errs = {"K": _rel(mn.matrix(E).toarray(), m1.matrix(E).toarray())}
    U1, Un = m1.solve_numpy(E, F), mn.solve_numpy(E, F)
    errs["U"] = _rel(Un, U1)
    errs["c"] = _rel(float(F[:, 0] @ Un[:, 0]), float(F[:, 0] @ U1[:, 0]))
    errs["energy"] = _rel(mn.element_energy(Un), m1.element_energy(U1))
    ctx = EvalContext(domain=d, xphys=torch.full(d.shape, 0.8, dtype=torch.float64),
                      E=torch.full(d.shape, 40.0, dtype=torch.float64), U=torch.as_tensor(U1))
    f1 = FailureIndex(d, make_criterion("tsai_wu", STRENGTH, R), D=D1, E0=40.0)(ctx)
    fn = FailureIndex(d, make_criterion("tsai_wu", STRENGTH, np.broadcast_to(R, (d.n_active, 3, 3)).copy()),
                      D=Dn, E0=40.0)(ctx)
    errs["failure_index"] = _rel(fn.numpy(), f1.numpy())
    print(f"\n  P1 {dim}-D {element}: " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    assert max(errs.values()) <= 1e-14


@pytest.mark.parametrize("dim,element", [(2, "Q4"), (2, "Q4IM"), (3, "H8"), (3, "H8IM")])
def test_p1_uniform_field_equals_single_D_buckling_and_modal(dim, element):
    """P1: buckling λ (LinearBuckling, modes_in_kg for the IM elements) and modal λ of the uniform field equal the
    single-D values (≤ 1e-14)."""
    d = T.Domain((4, 14) if dim == 2 else (2, 2, 8))
    X = d.node_coords()
    ax = dim - 1
    fixed = d.dofs(np.flatnonzero(X[:, ax] == 0))
    top = np.flatnonzero(np.isclose(X[:, ax], X[:, ax].max()))
    F = d.load_vector()
    F[dim * top + ax, 0] = -1.0 / len(top)
    R = _R(dim)
    D1 = unit_elasticity(GE, dim, "stress", R)
    Dn = unit_elasticity_field(GE, dim, "stress", np.broadcast_to(R, (d.n_active, 3, 3)))
    E = np.full(d.n_active, 40.0)
    lam = []
    for D in (D1, Dn):
        m = T.Elasticity(d, fixed, element=element, elasticity=D)
        u = m.solve_numpy(E, F)[:, 0]
        mu, _ = LinearBuckling(m).modes(E, E, u, 3)
        mo = Modal(d, fixed, element=element, elasticity=D, n_modes=3)
        lm, _ = mo.modes_numpy(E, np.ones(d.n_active), 3)
        lam.append((1 / mu, lm))
    eb, em = _rel(lam[1][0], lam[0][0]), _rel(lam[1][1], lam[0][1])
    print(f"\n  P1 {dim}-D {element}: buckling {eb:.1e} (lambda1 {lam[0][0][0]:.6g}), modal {em:.1e}")
    assert eb <= 1e-14 and em <= 1e-14


# ===================================================================================================== P2
def _right_edge_loads_2d(d, sigma_of_element):
    """Consistent nodal forces of a traction σ_e on the right edge (x = L) of a 2-D grid: σ h t / 2 per node of
    every edge segment, σ taken from the element the segment belongs to."""
    F = d.load_vector()
    nx, ny = d.shape
    for j in range(ny):
        e = (nx - 1) * ny + j
        s = sigma_of_element[e] * d.h * d.thickness / 2
        for jj in (j, j + 1):
            F[2 * d.node_number(np.array([nx, jj])), 0] += s
    return F


def _right_face_loads_3d(d, sigma_of_element):
    F = d.load_vector()
    nx, ny, nz = d.shape
    for j in range(ny):
        for k in range(nz):
            e = ((nx - 1) * ny + j) * nz + k
            s = sigma_of_element[e] * d.h ** 2 / 4
            for jj in (j, j + 1):
                for kk in (k, k + 1):
                    F[3 * d.node_number(np.array([nx, jj, kk])), 0] += s
    return F


def _strip_supports(d):
    X = d.node_coords()
    fx = [d.dofs(np.flatnonzero(X[:, 0] == 0), 0)]
    o = np.flatnonzero(np.all(X == 0, axis=1))
    fx.append(d.dofs(o, list(range(1, d.dim))))
    if d.dim == 3:
        top = np.flatnonzero((X[:, 0] == 0) & (X[:, 1] == 0) & np.isclose(X[:, 2], X[:, 2].max()))
        fx.append(d.dofs(top, 1))
    return np.unique(np.concatenate(fx))


@pytest.mark.parametrize("dim", [2, 3])
def test_p2a_parallel_layers_voigt(dim):
    """P2a: layers stacked across the load (0° for y < H/2, 90° above) share the axial strain: E_eff = Σ E_i H_i / H
    and u_x = ε L on the whole loaded end (≤ 1e-10). 3-D uses ν13 = ν23 so that the layers contract alike in z."""
    mat = GE if dim == 2 else VOIGT3
    d = T.Domain((8, 4) if dim == 2 else (6, 4, 2))
    iy = d.element_ijk()[:, 1]
    theta = np.where(iy < d.shape[1] // 2, 0.0, 90.0)
    D = unit_elasticity_field(mat, dim, "stress", rotations_about_z(theta))
    m = T.Elasticity(d, _strip_supports(d), element="Q4" if dim == 2 else "H8", elasticity=D)
    eps = 1e-3
    Ei = np.where(theta == 0.0, mat.E1, mat.E2)
    sig = Ei * eps
    F = _right_edge_loads_2d(d, sig) if dim == 2 else _right_face_loads_3d(d, sig)
    U = m.solve_numpy(np.full(d.n_active, mat.E1), F)[:, 0]
    X = d.node_coords()
    L = X[:, 0].max()
    end = np.flatnonzero(np.isclose(X[:, 0], L))
    ux = U[dim * end]
    H = X[:, 1].max()
    A = H * (d.thickness if dim == 2 else X[:, 2].max())
    E_eff = (F[:, 0].sum() / A) / (ux.mean() / L)
    E_voigt = 0.5 * (mat.E1 + mat.E2)
    print(f"\n  P2a {dim}-D: E_eff / Voigt - 1 = {E_eff / E_voigt - 1:+.1e}, u_x spread {np.ptp(ux) / ux.mean():.1e}")
    assert abs(E_eff / E_voigt - 1) <= 1e-10
    assert _rel(ux, np.full_like(ux, eps * L)) <= 1e-10


@pytest.mark.parametrize("dim", [2, 3])
def test_p2b_series_layers_reuss(dim):
    """P2b: layers in series along the load (0° for x < L/2, 90° beyond) carry the same stress:
    1/E_eff = (1/E1 + 1/E2)/2 (≤ 1e-10). 3-D uses ν13/E1 = ν23/E2 so that the lateral strains match at the
    interface."""
    mat = GE if dim == 2 else REUSS3
    d = T.Domain((8, 3) if dim == 2 else (6, 2, 2))
    ix = d.element_ijk()[:, 0]
    theta = np.where(ix < d.shape[0] // 2, 0.0, 90.0)
    m = T.Elasticity(d, _strip_supports(d), element="Q4IM" if dim == 2 else "H8IM",
                     elasticity=unit_elasticity_field(mat, dim, "stress", rotations_about_z(theta)))
    sigma = 0.2
    F = (_right_edge_loads_2d(d, np.full(d.ne, sigma)) if dim == 2
         else _right_face_loads_3d(d, np.full(d.ne, sigma)))
    U = m.solve_numpy(np.full(d.n_active, mat.E1), F)[:, 0]
    X = d.node_coords()
    L = X[:, 0].max()
    ux = U[dim * np.flatnonzero(np.isclose(X[:, 0], L))]
    E_eff = sigma / (ux.mean() / L)
    E_reuss = 1.0 / (0.5 / mat.E1 + 0.5 / mat.E2)
    print(f"\n  P2b {dim}-D: E_eff / Reuss - 1 = {E_eff / E_reuss - 1:+.1e}, Reuss {E_reuss:.12g}")
    assert abs(E_eff / E_reuss - 1) <= 1e-10
    assert np.ptp(ux) <= 1e-10 * abs(ux.mean())


def test_p2c_tsai_wu_index_in_series_layers():
    """P2c: in the series strip under σ the Tsai-Wu index is σ/Xt in the 0° layer and σ/Yt in the 90° layer
    (per-element D and R; eps = 0; ≤ 1e-10)."""
    d = T.Domain((8, 3))
    ix = d.element_ijk()[:, 0]
    theta = np.where(ix < 4, 0.0, 90.0)
    Rf = rotations_about_z(theta)
    D = unit_elasticity_field(GE, 2, "stress", Rf)
    m = T.Elasticity(d, _strip_supports(d), elasticity=D)
    sigma = 0.2
    U = m.solve_numpy(np.full(d.n_active, GE.E1), _right_edge_loads_2d(d, np.full(d.ne, sigma)))
    ctx = EvalContext(domain=d, xphys=torch.ones(d.shape, dtype=torch.float64),
                      E=torch.full(d.shape, GE.E1, dtype=torch.float64), U=torch.as_tensor(U))
    fi = FailureIndex(d, make_criterion("tsai_wu", STRENGTH, Rf, eps=0.0), D=D, E0=GE.E1, q=0.0)
    got = fi.solid_ratio(ctx)[:, 0].numpy()
    want = np.where(theta == 0.0, sigma / STRENGTH.Xt, sigma / STRENGTH.Yt)
    print(f"\n  P2c: index error {_rel(got, want):.1e} (0 deg: {got[0]:.6g}, 90 deg: {got[-1]:.6g})")
    assert _rel(got, want) <= 1e-10


# ===================================================================================================== P3
def _fd_error(st, x0, which, h=2e-4):
    r1 = study_gradcheck(st, x0, which, n_dirs=4, h=h, seed=2)
    r2 = study_gradcheck(st, x0, which, n_dirs=4, h=2 * h, seed=2)
    return max(abs((4 * a["fd"] - b["fd"]) / 3 - a["ad"]) / max(abs(a["ad"]), 1e-300)
               for a, b in zip(r1["directions"], r2["directions"], strict=True))


def _angle_field(d) -> np.ndarray:
    c = d.element_centers()[d.active_elements]
    return 30.0 + 40.0 * np.sin(0.7 * c[:, 0]) * np.cos(0.9 * c[:, 1])


@pytest.mark.parametrize("dim,element", [(2, "Q4"), (3, "H8IM")])
def test_p3a_compliance_gradient(dim, element):
    """P3a: compliance gradient with a continuous orientation field, central differences (Richardson) ≤ 1e-6."""
    d, fixed, F = _clamped((10, 4) if dim == 2 else (6, 3, 2))
    th = np.deg2rad(_angle_field(d))
    if dim == 2:
        R = rotations_about_z(np.rad2deg(th))
    else:
        R = rotations_from_axes(np.column_stack([np.cos(th), np.sin(th), 0.3 * np.ones_like(th)]), [0.0, 0.0, 1.0])
    D = unit_elasticity_field(GE, dim, "stress", R)
    st = T.Study(d, T.Elasticity(d, fixed, element=element, elasticity=D), F, T.standard_chain(d, 1.5, beta=2.0),
                 T.SIMP(d, E0=40.0), T.Compliance(), [T.Constraint(T.Volume(), 0.6)], T.MMA())
    x0 = 0.4 + 0.5 * np.random.default_rng(8).random(d.n_design)
    err = _fd_error(st, x0, "objective")
    print(f"\n  P3a {dim}-D {element}: {len(st.physics.KE)} distinct element matrices, gradient {err:.2e}")
    assert err <= 1e-6


def test_p3b_failure_index_gradient():
    """P3b: p-norm of the relaxed Tsai-Wu index, per-element D and R, ≤ 1e-6."""
    d = T.Domain((10, 4))
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    F = d.load_vector()
    F[2 * np.flatnonzero(np.isclose(X[:, 0], 10) & (X[:, 1] == 0)) + 1, 0] = -1.0
    F[2 * np.flatnonzero(np.isclose(X[:, 0], 10) & (X[:, 1] == 4)), 0] = 0.3
    R = rotations_about_z(_angle_field(d))
    D = unit_elasticity_field(GE, 2, "stress", R)
    field = FailureIndex(d, make_criterion("tsai_wu", STRENGTH, R), D=D, E0=40.0, q=0.5)
    st = T.Study(d, T.Elasticity(d, fixed, elasticity=D), F, T.standard_chain(d, 1.5, beta=2.0),
                 T.SIMP(d, E0=40.0), Aggregated(field, "pnorm", 8.0), [T.Constraint(T.Volume(), 0.6)], T.MMA())
    x0 = 0.4 + 0.5 * np.random.default_rng(8).random(d.n_design)
    err = _fd_error(st, x0, "objective", h=1e-4)
    print(f"\n  P3b: Tsai-Wu p-norm gradient {err:.2e}")
    assert err <= 1e-6


def test_p3c_buckling_gradient_q4im():
    """P3c: buckling KS gradient with per-element D, Q4IM with the modes in K_G, ≤ 1e-6."""
    d = T.Domain((6, 18), 1.0, thickness=0.8)
    X = d.node_coords()
    top = np.flatnonzero(np.isclose(X[:, 1], 18))
    F = d.load_vector()
    F[2 * top + 1, 0] = -1.0 / len(top)
    F[2 * top[:1], 0] = 0.02
    D = unit_elasticity_field(GE, 2, "stress", rotations_about_z(60.0 + 0.5 * _angle_field(d)))
    resp = BucklingKS(n_modes=4, rho=20.0, scale=1.0)
    st = T.Study(d, T.Elasticity(d, d.dofs(np.flatnonzero(X[:, 1] == 0)), element="Q4IM", elasticity=D), F,
                 T.standard_chain(d, 1.5, beta=2.0), T.SIMP(d, E0=40.0), T.Compliance(),
                 [T.Constraint(T.Volume(), 0.6), T.Constraint(resp, 1.0, scale=1.0)], T.MMA())
    x0 = 0.45 + 0.4 * np.random.default_rng(12).random(d.n_design)
    err = _fd_error(st, x0, 1)
    print(f"\n  P3c: buckling KS gradient {err:.2e}; load factors {np.round(resp.last_load_factors, 4)}")
    assert err <= 1e-6


# ===================================================================================================== P4
def _E_theta(mat: Orthotropic, theta_deg: float) -> float:
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    return 1.0 / (c ** 4 / mat.E1 + (1 / mat.G12 - 2 * mat.nu12 / mat.E1) * c * c * s * s + s ** 4 / mat.E2)


def _bar_bc(L: float):
    tol = 1e-9
    sup = [(lambda X: X[:, 0] < tol, [0]),
           (lambda X: np.linalg.norm(X, axis=1) < tol, [1, 2]),
           (lambda X: np.linalg.norm(X - [0.0, 0.0, 1.0], axis=1) < tol, [1])]
    return sup, (lambda X: X[:, 0] > L - tol)


@pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0])
def test_p4a_p4b_t10_and_h8_off_axis_bar(theta):
    """P4a/P4b: uniform orthotropic bar (4 × 1 × 1) in tension along x, material axis 1 at θ about z, supports
    that only remove rigid motions: compliance P² L / (E(θ) A) with T10 and with H8 (``voxel_reference``) ≤ 1e-9."""
    L, P = 4.0, 3.0
    sup, end = _bar_bc(L)
    Du = unit_elasticity(GE, 3, "stress", rotation_about_z(theta))
    want = P * P * L / _E_theta(GE, theta)
    r = solve_t10(_box_t10((4, 1, 1), (L, 1.0, 1.0)), sup, [(end, (P, 0.0, 0.0))], GE.E1, elasticity=Du)
    d = T.Domain((4, 1, 1), h=1.0)
    vx = voxel_reference(d, np.ones(d.shape), sup, [(end, (P, 0.0, 0.0))], GE.E1, elasticity=Du)
    ct, cv = r["cases"][0]["compliance"], vx["cases"][0]["compliance"]
    print(f"\n  P4 theta = {theta:g} deg: E(theta) {_E_theta(GE, theta):.10g}; T10 {ct / want - 1:+.1e}, H8 {cv / want - 1:+.1e}")
    assert abs(ct / want - 1) <= 1e-9 and abs(cv / want - 1) <= 1e-9
    assert r["material"] == "general"


def test_p4c_t10_series_bar_orientation_from_voxels():
    """P4c: bar of two orientations in series (0° for x < 2, 90° beyond; ν13/E1 = ν23/E2), the tets take the D of
    the voxel holding their centroid: T10 and H8 compliance P² (L_A/E1 + L_B/E2) / A ≤ 1e-9."""
    L, P = 4.0, 3.0
    sup, end = _bar_bc(L)
    d = T.Domain((4, 1, 1), h=1.0)
    theta = np.where(d.element_ijk()[:, 0] < 2, 0.0, 90.0)
    Dv = unit_elasticity_field(REUSS3, 3, "stress", rotations_about_z(theta))
    m = _box_t10((8, 2, 2), (L, 1.0, 1.0))
    rows, n_out = tet_voxel_rows(m, d)
    assert n_out == 0
    r = solve_t10(m, sup, [(end, (P, 0.0, 0.0))], REUSS3.E1, elasticity=Dv[rows])
    vx = voxel_reference(d, np.ones(d.shape), sup, [(end, (P, 0.0, 0.0))], REUSS3.E1, elasticity=Dv)
    want = P * P * (2.0 / REUSS3.E1 + 2.0 / REUSS3.E2)
    ct, cv = r["cases"][0]["compliance"], vx["cases"][0]["compliance"]
    print(f"\n  P4c: T10 {ct / want - 1:+.1e}, H8 {cv / want - 1:+.1e}")
    assert abs(ct / want - 1) <= 1e-9 and abs(cv / want - 1) <= 1e-9
    assert r["material"] == "per-element"


def test_tet_voxel_rows_outside_centroid_takes_nearest_voxel():
    """Post hoc (not in the pre-registration): a tet whose centroid lies outside the active voxels takes the
    nearest active voxel, and is counted."""
    d = T.Domain((2, 1, 1), h=1.0)
    m = _box_t10((2, 1, 1), (2.0, 1.0, 1.0))
    rows, n_out = tet_voxel_rows(m, d)
    c = m.X[m.T[:, :4]].mean(axis=1)
    assert n_out == 0 and np.array_equal(rows, (c[:, 0] >= 1.0).astype(np.int64))
    m.X = m.X + np.array([2.5, 0.0, 0.0])                  # every centroid beyond x = 2: nearest is voxel 1
    rows, n_out = tet_voxel_rows(m, d)
    assert n_out == len(m.T) and np.all(rows == 1)


# ===================================================================================================== P5
def test_p5_isotropic_t10_unchanged():
    """P5: the isotropic T10 / voxel path reproduces the values recorded at d566290 on this machine (exact).

    Changed at the merge (2026-11-25, coordinator): the voxel reference solves with ``Elasticity``, whose SuperLU
    elimination order became nested dissection (``solvers.ordering``, docs/olcumler/cozucu_siralama.md). Its two
    values now differ from the d566290 record in the last digits (compliance 57.11866917633062 against
    57.1186691763344, 6.6e-14 relative), so that line compares to 1e-12; the T10 part is unchanged and still exact."""
    m = _box_t10((3, 2, 2), (1.5, 1.0, 0.8))
    m.X = m.X + 0.02 * np.random.default_rng(7).standard_normal(m.X.shape)
    K = stiffness_t10(m, 7.0, 0.27).tocsr()
    K.sort_indices()
    h = hashlib.sha256(K.data.tobytes() + K.indices.tobytes() + K.indptr.tobytes()).hexdigest()
    r = solve_t10(m, [(lambda X: X[:, 0] < 0.05, [0, 1, 2])], [(lambda X: X[:, 0] > 1.45, (0.0, -1.0, 0.3))],
                  7.0, 0.27)
    c = r["cases"][0]
    d = T.Domain((4, 2, 2), h=0.5)
    vx = voxel_reference(d, np.linspace(0.3, 1.0, d.ne), [(lambda X: X[:, 0] < 1e-9, [0, 1, 2])],
                         [(lambda X: X[:, 0] > 2 - 1e-9, (0.0, -1.0, 0.2))], 7.0, 0.27)
    assert h == "c37c6bb4a85341e2bb619a86df7cd6bf8ddf5071c2829a42b485e52ac6a7df6f"
    assert (c["compliance"], c["max_displacement"], c["vm_max"]) == (3.482339726026348, 3.70297627743529,
                                                                       13.115959479529819)
    for got, rec in zip((vx["cases"][0]["compliance"], vx["cases"][0]["max_displacement"]),
                        (57.1186691763344, 59.143123678603374), strict=True):
        assert abs(got - rec) <= 1e-12 * rec
    assert "material" not in r


# ===================================================================================================== P6
ORTHO = {"E1": 40.0, "E2": 9.0, "E3": 9.0, "G12": 4.0, "G13": 4.0, "G23": 3.2, "nu12": 0.28, "nu13": 0.28,
         "nu23": 0.4}


def _cfg(orient: dict, shape=(12, 4)) -> dict:
    extra = [0] if len(shape) == 3 else []
    at = ["max", "min"] + (["min"] if len(shape) == 3 else [])
    return {
        "domain": {"type": "box", "shape": list(shape), "h": 1.0},
        "material": {"E0": 40.0, "nu": 0.3, "orthotropic": {**ORTHO, **orient},
                     "strength": {"Xt": 15.0, "Xc": 12.0, "Yt": 0.5, "Yc": 2.0, "S12": 0.7}},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point", "at": at},
                                                  "force": [0, -1] + extra}]}],
        "objective": "compliance",
        "constraints": [{"response": "volume", "upper": 0.5}, {"response": {"type": "stress"}}],
        "optimizer": {"type": "mma", "move": 0.1},
        "run": {"max_iter": 2},
    }


def test_p6_yaml_uniform_angle_field_equals_angle(tmp_path):
    """P6: ``angle_field`` with 25 degrees everywhere (as a .npy file) builds the same study as ``angle: 25``: same
    compliance (pre-registered, ≤ 1e-14). Post hoc (added after the first run, which compared values and gradients
    together and measured 1.3e-14 on the stress constraint's gradient only): compliance gradient and stress
    constraint value ≤ 1e-14 (measured 0), stress constraint gradient ≤ 1e-13 (measured 1.3e-14: the per-element
    stress is a batched einsum, the single-D one a matrix product, so the backward pass rounds differently)."""
    from toporlogy.configuration.build import study_from_dict
    f = tmp_path / "angle.npy"
    np.save(f, np.full((12, 4), 25.0))
    a = study_from_dict(_cfg({"angle": 25.0}))
    b = study_from_dict(_cfg({"angle_field": str(f)}))
    assert b.physics.per_element and b.physics.D.shape == (b.domain.n_active, 3, 3)
    assert b.constraints[1].response.field.criterion.R.shape == (b.domain.n_active, 3, 3)
    x = a.default_start()
    ea, eb = a.evaluate(x), b.evaluate(x)
    errs = {k: _rel(getattr(eb, k), getattr(ea, k)) for k in ("f0", "df0", "g", "dg")}
    print(f"\n  P6: compliance {ea.f0:.12g} / {eb.f0:.12g}; " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    assert errs["f0"] <= 1e-14                                   # pre-registered
    assert errs["df0"] <= 1e-14 and errs["g"] <= 1e-14           # post hoc
    assert errs["dg"] <= 1e-13                                   # post hoc


def test_yaml_axis_field_and_errors(tmp_path):
    """Post hoc: ``axis_field`` (3-D) builds and evaluates; (*grid, 3) needs axis2; wrong shapes, 2-D/3-D mix-ups
    and two orientations are refused with the key in the message."""
    from toporlogy.configuration.build import study_from_dict
    from toporlogy.configuration.registry import ConfigError
    shape = (6, 3, 2)
    ax = np.zeros(shape + (3,))
    ax[..., 0], ax[..., 1] = 1.0, np.linspace(0, 1, 6)[:, None, None]
    f = tmp_path / "axes.npy"
    np.save(f, ax)
    st = study_from_dict(_cfg({"axis_field": str(f), "axis2": [0, 0, 1]}, shape))
    assert st.physics.D.shape == (st.domain.n_active, 6, 6)
    assert np.all(np.isfinite(st.evaluate(st.default_start()).dg))
    both = np.stack([ax, np.broadcast_to([0.0, 0.0, 1.0], ax.shape)], axis=-2)
    np.save(tmp_path / "both.npy", both)
    st2 = study_from_dict(_cfg({"axis_field": str(tmp_path / "both.npy")}, shape))
    assert _rel(st2.physics.D, st.physics.D) <= 1e-14
    for orient, sh, msg in [({"axis_field": str(f)}, shape, "axis2"),
                            ({"angle_field": str(tmp_path / "nope.npy")}, (12, 4), "not found"),
                            ({"angle_field": str(f)}, (12, 4), "element grid"),
                            ({"angle": 3.0, "angle_field": str(f)}, (12, 4), "one of"),
                            ({"axis_field": str(f), "axis2": [0, 0, 1]}, (12, 4), "3-D")]:
        with pytest.raises(ConfigError, match=msg):
            study_from_dict(_cfg(orient, sh))


# ===================================================================================================== P7
def test_p7_gmg_refuses_per_element_D():
    """P7: ElasticityGMG refuses a per-element elasticity with a clear error."""
    from toporlogy.fem.elastic_gmg import ElasticityGMG
    d, fixed, _ = _clamped((4, 2, 2))
    D = unit_elasticity_field(GE, 3, "stress", np.broadcast_to(np.eye(3), (d.n_active, 3, 3)))
    with pytest.raises(ValueError, match="per-element elasticity"):
        ElasticityGMG(d, fixed, elasticity=D)


def test_shape_checks():
    """Post hoc: a per-element D of the wrong length or size is refused."""
    d, fixed, _ = _clamped((4, 2))
    with pytest.raises(ValueError, match="active elements"):
        T.Elasticity(d, fixed, elasticity=np.broadcast_to(np.eye(3), (3, 3, 3)).copy())
    with pytest.raises(ValueError, match="per-element D"):
        T.Elasticity(d, fixed, elasticity=np.broadcast_to(np.eye(6), (d.n_active, 6, 6)).copy())
    D = unit_elasticity_field(GE, 2, "stress", rotations_about_z(np.zeros(d.n_active)))
    with pytest.raises(ValueError, match="material frames"):
        FailureIndex(d, make_criterion("tsai_wu", STRENGTH, rotations_about_z([0.0, 1.0])), D=D)


@pytest.mark.slow
def test_cost_of_a_continuous_field():
    """Post hoc, a measurement (no tolerance): set-up time of Elasticity for a continuous field (every element its own
    D) against one D, 3-D H8, 24 × 12 × 12."""
    d, fixed, _ = _clamped((24, 12, 12))
    R = rotations_from_axes(np.column_stack([np.ones(d.n_active), np.linspace(0, 1, d.n_active),
                                             np.zeros(d.n_active)]), [0.0, 0.0, 1.0])
    t0 = time.perf_counter()
    D = unit_elasticity_field(GE, 3, "stress", R)
    t1 = time.perf_counter()
    m = T.Elasticity(d, fixed, elasticity=D)
    t2 = time.perf_counter()
    T.Elasticity(d, fixed, elasticity=D[0])
    t3 = time.perf_counter()
    m.matrix(np.ones(d.n_active))
    t4 = time.perf_counter()
    print(f"\n  cost, {d.n_active} elements, {len(m.KE)} distinct: D field {t1 - t0:.2f} s, Elasticity "
          f"{t2 - t1:.2f} s (one D: {t3 - t2:.2f} s), assembly {t4 - t3:.2f} s")

