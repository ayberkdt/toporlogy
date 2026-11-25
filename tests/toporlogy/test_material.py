"""Material models: elasticity matrices (core/material.py), element matrices for a general D (fem/elements.py),
plane strain, and the Hashin-Shtrikman bounds of a porous solid (design/interp.py).

Independent references used here: the closed forms D_iso / D_plane_stress already in the library (isotropic case);
the directional compliance 1/E(n) = n_i n_j n_k n_l S_ijkl (rotation); the element patch identities (rigid motions
have no energy, a uniform strain ε has energy εᵀDε · volume); a 3-D model with every z displacement fixed (exactly
plane strain) against the 2-D plane-strain model; the Lamé solution of a hollow sphere / hollow disk under external
pressure (the exact bulk modulus of the coated-sphere / coated-cylinder assemblage, which IS the HS bound).
Tolerances: rounding (1e-12 relative) wherever the two sides are the same algebra; stated otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import toporlogy as T
from toporlogy.core.material import (Orthotropic, Strength, OrthotropicStrength, out_of_plane_row, plane_strain,
                                     plane_stress, rotate_stiffness, rotation_about_z, rotation_from_axes,
                                     unit_elasticity)
from toporlogy.design.interp import HashinShtrikman, hashin_shtrikman_moduli, simp_admissible_penal
from toporlogy.fem.elements import (D_iso, D_plane_strain, D_plane_stress, element_matrix, ke_h8_from_D,
                                    ke_q4_from_D)

GLASS_EPOXY = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


def test_isotropic_special_case_equals_the_closed_forms():
    m = Orthotropic.isotropic(7.0, 0.3)
    C = m.stiffness() / 7.0
    assert _rel(C, D_iso(0.3)) <= 1e-14
    assert _rel(plane_stress(C), D_plane_stress(0.3)) <= 1e-14
    assert _rel(plane_strain(C), D_plane_strain(0.3)) <= 1e-14
    nu = 0.3                                            # plane strain: σzz = ν(σxx + σyy) = λ(εxx + εyy)
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    assert _rel(out_of_plane_row(C), [lam, lam, 0.0]) <= 1e-14


def test_rotations():
    rng = np.random.default_rng(1)
    iso = Orthotropic.isotropic(3.0, 0.25).stiffness()
    for _ in range(5):
        R = rotation_from_axes(rng.standard_normal(3), rng.standard_normal(3))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-14) and np.linalg.det(R) > 0
        assert _rel(rotate_stiffness(iso, R), iso) <= 1e-13              # isotropic: invariant
    m = Orthotropic(**GLASS_EPOXY)
    C90 = m.stiffness(rotation_about_z(90.0))
    assert abs(C90[0, 0] - m.stiffness()[1, 1]) <= 1e-12 * C90[0, 0]      # 90° about z swaps 1 and 2
    assert abs(C90[1, 1] - m.stiffness()[0, 0]) <= 1e-12 * C90[1, 1]
    # directional modulus along global x for material axis 1 at θ: 1/E_x = n·n·n·n·S (tensor contraction)
    S = m.compliance()
    for th in (0.0, 17.0, 45.0, 73.0):
        Sx = np.linalg.inv(m.stiffness(rotation_about_z(th)))[0, 0]
        c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        n = np.array([c, -s, 0.0])                                      # global x in material axes
        St = np.zeros((3, 3, 3, 3))
        pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for a, (i, j) in enumerate(pairs):
            for b, (k, l) in enumerate(pairs):
                f = (0.5 if i != j else 1.0) * (0.5 if k != l else 1.0)  # engineering → tensor shear
                for p, q in {(i, j), (j, i)}:
                    for r, t in {(k, l), (l, k)}:
                        St[p, q, r, t] = S[a, b] * f
        assert abs(Sx - np.einsum("i,j,k,l,ijkl->", n, n, n, n, St)) <= 1e-13 * Sx


def test_bad_constants_are_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        Orthotropic(E1=1, E2=1, E3=1, G12=1, G13=1, G23=1, nu12=0.9, nu13=0.9, nu23=0.9)
    with pytest.raises(ValueError):
        rotation_from_axes([1, 0, 0], [2, 0, 0])
    with pytest.raises(ValueError):
        Strength(tension=1.0, safety_factor=0.5)
    with pytest.raises(ValueError):
        OrthotropicStrength(Xt=1, Xc=1, Yt=1, Yc=1, S12=1, f12=1.5)


@pytest.mark.parametrize("dim,element", [(2, "Q4"), (3, "H8"), (3, "H8IM")])
def test_element_patch_identities_for_a_general_D(dim, element):
    """Rigid motions carry no energy; the linear field u = ε·x carries εᵀ D ε · volume (unit element), for an
    orthotropic, rotated D. Rounding only."""
    m = Orthotropic(**GLASS_EPOXY)
    R = rotation_from_axes([1, 2, 0.5], [0, 1, 3]) if dim == 3 else rotation_about_z(31.0)
    D = unit_elasticity(m, dim, "strain" if dim == 2 else "stress", R)
    K = ke_q4_from_D(D) if dim == 2 else ke_h8_from_D(D, element)[0]
    corners = (T.Domain((1, 1)) if dim == 2 else T.Domain((1, 1, 1))).corners.astype(float)
    rng = np.random.default_rng(3)
    for _ in range(3):
        if dim == 2:
            ex, ey, gxy, w = rng.standard_normal(4)
            eps = np.array([ex, ey, gxy])
            G = np.array([[ex, gxy / 2], [gxy / 2, ey]])
            W = np.array([[0, -w], [w, 0]])
        else:
            g = rng.standard_normal(6)
            eps = g
            G = np.array([[g[0], g[5] / 2, g[4] / 2], [g[5] / 2, g[1], g[3] / 2], [g[4] / 2, g[3] / 2, g[2]]])
            a = rng.standard_normal(3)
            W = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        u_rigid = (corners @ W.T + rng.standard_normal(dim)).ravel()
        u_lin = (corners @ G.T).ravel()
        assert abs(u_rigid @ K @ u_rigid) <= 1e-12 * np.abs(K).max()
        assert abs(u_lin @ K @ u_lin - eps @ D @ eps) <= 1e-12 * abs(eps @ D @ eps)


def test_general_D_reproduces_the_isotropic_elements():
    assert _rel(ke_q4_from_D(D_plane_stress(0.3)), element_matrix(2, "Q4", 0.3)) <= 1e-14
    assert _rel(ke_h8_from_D(D_iso(0.3), "H8IM")[0], element_matrix(3, "H8IM", 0.3)) <= 1e-14


@pytest.mark.parametrize("orthotropic", [False, True])
def test_plane_strain_2d_equals_3d_slab_with_z_fixed(orthotropic):
    """A one-element-thick 3-D slab with every w = 0 is exactly plane strain (u, v independent of z): its
    displacements must equal the 2-D plane-strain model's (thickness h) to rounding; checked for an isotropic and
    an orthotropic material rotated about z."""
    nx, ny, h = 8, 3, 0.5
    mat = Orthotropic(**GLASS_EPOXY) if orthotropic else Orthotropic.isotropic(1.0, 0.3)
    R = rotation_about_z(25.0) if orthotropic else None
    d2 = T.Domain((nx, ny), h, thickness=h)
    X2 = d2.node_coords()
    fx2 = d2.dofs(np.flatnonzero(X2[:, 0] == 0))
    F2 = d2.load_vector()
    tip2 = np.flatnonzero(np.isclose(X2[:, 0], nx * h))
    F2[2 * tip2 + 1, 0] = -1.0 / len(tip2)
    m2 = T.Elasticity(d2, fx2, elasticity=unit_elasticity(mat, 2, "strain", R))
    U2 = m2.solve_numpy(np.ones(d2.n_active), F2)[:, 0]
    d3 = T.Domain((nx, ny, 1), h)
    X3 = d3.node_coords()
    fx3 = np.union1d(d3.dofs(np.flatnonzero(X3[:, 0] == 0)), d3.dofs(np.arange(d3.nn), 2))
    F3 = d3.load_vector()
    tip3 = np.flatnonzero(np.isclose(X3[:, 0], nx * h))
    F3[3 * tip3 + 1, 0] = -1.0 / len(tip3)                 # same total load over twice as many nodes
    m3 = T.Elasticity(d3, fx3, elasticity=unit_elasticity(mat, 3, "stress", R), element="H8")
    U3 = m3.solve_numpy(np.ones(d3.n_active), F3)[:, 0]
    ij = np.indices((nx + 1, ny + 1)).reshape(2, -1).T
    n3 = d3.node_number(np.column_stack([ij, np.zeros(len(ij), int)]))
    n2 = d2.node_number(ij)
    for c in (0, 1):
        assert np.max(np.abs(U3[3 * n3 + c] - U2[2 * n2 + c])) <= 1e-10 * np.abs(U2).max()


def test_orthotropic_bar_modulus_along_the_axes():
    """Uniform tension of a 2-D plane-stress strip, x roller at x = 0 (one y pin), with material axis 1 along x
    (θ = 0) or along y (θ = 90): the elongation gives E1 and E2 exactly (no shear coupling at these angles)."""
    nx, ny, h = 10, 2, 1.0
    for th, E_expect in ((0.0, 40.0), (90.0, 9.0)):
        mat = Orthotropic(**GLASS_EPOXY)
        d = T.Domain((nx, ny), h)
        X = d.node_coords()
        left = np.flatnonzero(X[:, 0] == 0)
        right = np.flatnonzero(np.isclose(X[:, 0], nx * h))
        fixed = np.concatenate([d.dofs(left, 0), d.dofs(left[:1], 1)])
        F = d.load_vector()
        w = np.full(len(right), h)
        w[[0, -1]] = h / 2
        F[2 * right, 0] = w                                    # unit traction, total ny·h
        D = unit_elasticity(mat, 2, "stress", rotation_about_z(th))
        m = T.Elasticity(d, fixed, elasticity=D)
        U = m.solve_numpy(np.full(d.n_active, mat.reference_modulus), F)[:, 0]
        strain = U[2 * right].mean() / (nx * h)
        assert abs(1.0 / strain - E_expect) <= 1e-10 * E_expect


# ================================================================================ Hashin-Shtrikman
def _lame_hollow_sphere_bulk(E: float, nu: float, porosity: float) -> float:
    """Effective bulk modulus of a hollow sphere (inner radius a, outer b, (a/b)³ = porosity) under external
    pressure, from the Lamé field u = A r + B / r² solved numerically for its two boundary conditions; the field's
    equilibrium is checked by finite differences."""
    K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
    a, b, p = porosity ** (1 / 3), 1.0, 1.0
    lam = K - 2 * G / 3

    def stresses(A, B, r):
        ur, dur = A * r + B / r ** 2, A - 2 * B / r ** 3
        tr = dur + 2 * ur / r
        return lam * tr + 2 * G * dur, lam * tr + 2 * G * ur / r
    # σrr(a) = 0, σrr(b) = −p : linear in (A, B)
    M = np.array([[stresses(1, 0, a)[0], stresses(0, 1, a)[0]], [stresses(1, 0, b)[0], stresses(0, 1, b)[0]]])
    A, B = np.linalg.solve(M, [0.0, -p])
    r, dr = 0.5 * (a + b), 1e-4
    srr = lambda rr: stresses(A, B, rr)[0]                                       # noqa: E731
    eq = (srr(r + dr) - srr(r - dr)) / (2 * dr) + 2 * (srr(r) - stresses(A, B, r)[1]) / r
    assert abs(eq) <= 1e-6 * p
    return -p / (3 * (A * b + B / b ** 2) / b)                                    # p = −K_eff · ΔV/V


def _lame_hollow_disk_bulk(E: float, nu: float, porosity: float) -> float:
    """Plane-stress hollow disk ((a/b)² = porosity), u = A r + B / r, external pressure: effective plane bulk
    modulus p = −κ_eff · 2 u(b)/b."""
    a, b, p = porosity ** 0.5, 1.0, 1.0
    c = E / (1 - nu ** 2)

    def srr(A, B, r):
        return c * ((A - B / r ** 2) + nu * (A + B / r ** 2))
    M = np.array([[srr(1, 0, a), srr(0, 1, a)], [srr(1, 0, b), srr(0, 1, b)]])
    A, B = np.linalg.solve(M, [0.0, -p])
    return -p / (2 * (A * b + B / b) / b)


@pytest.mark.parametrize("nu", [0.0, 0.25, 0.3, 0.45])
def test_hs_bulk_bound_is_the_coated_sphere_and_cylinder(nu):
    for x in (0.1, 0.4, 0.75, 0.95):
        K3 = float(hashin_shtrikman_moduli(torch.tensor(x, dtype=torch.float64), 1.0, nu, 3)[0])
        K2 = float(hashin_shtrikman_moduli(torch.tensor(x, dtype=torch.float64), 1.0, nu, 2)[0])
        assert abs(K3 / _lame_hollow_sphere_bulk(1.0, nu, 1 - x) - 1) <= 1e-12
        assert abs(K2 / _lame_hollow_disk_bulk(1.0, nu, 1 - x) - 1) <= 1e-12


def test_hs_shear_bound_limits_and_simp_admissibility():
    """Shear bounds: 0 at x = 0, the solid's G at x = 1, monotone (the formula itself is not independently checked).
    SIMP x^p with p = simp_admissible_penal lies below both HS bounds on (0, 1]; 5 % less violates it near x = 1."""
    x = torch.linspace(1e-4, 1.0, 2001, dtype=torch.float64)
    for dim in (2, 3):
        for nu in (0.0, 0.2, 0.3, 0.4):
            E = 1.0
            K, G = hashin_shtrikman_moduli(x, E, nu, dim)
            K1 = E / (3 * (1 - 2 * nu)) if dim == 3 else E / (2 * (1 - nu))
            G1 = E / (2 * (1 + nu))
            assert abs(float(G[-1]) - G1) <= 1e-14 and abs(float(K[-1]) - K1) <= 1e-12 * K1
            assert bool((torch.diff(G) > 0).all()) and bool((torch.diff(K) > 0).all())
            p = simp_admissible_penal(nu, dim)
            assert bool((x ** p * K1 <= K * (1 + 1e-12)).all()) and bool((x ** p * G1 <= G * (1 + 1e-12)).all())
            q = 0.95 * p
            assert bool((x ** q * K1 > K).any()) or bool((x ** q * G1 > G).any())
    assert abs(simp_admissible_penal(0.3, 2) - 40 / 13) <= 1e-14         # max(2/0.7, 4/1.3) = 4/1.3
    assert abs(simp_admissible_penal(0.3, 3) - 2.625) <= 1e-14           # 3·0.7 / (2·0.4)


def test_hashin_shtrikman_interpolation():
    d = T.Domain((4, 3))
    hs = HashinShtrikman(d, E0=5.0, nu=0.3, emin_ratio=1e-9)
    x = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)
    y = hs.young(x)
    assert abs(float(y[0])) <= 1e-15 and abs(float(y[-1]) - 1.0) <= 1e-14 and bool((torch.diff(y) > 0).all())
    nu_hs = hs.poisson(x[1:])
    print(f"\n  HS (2-D, nu = 0.3): E/E0 at x = 0.25, 0.5: {y[1]:.4f}, {y[2]:.4f}; nu_HS at 0.25, 0.5, 1: "
          f"{np.round(nu_hs.numpy(), 4)} (the element keeps 0.3)")
    assert abs(float(nu_hs[-1]) - 0.3) <= 1e-12
    E = hs(torch.full(d.shape, 0.5, dtype=torch.float64))
    assert abs(float(E.mean()) - 5.0 * (1e-9 + float(y[2]) * (1 - 1e-9))) <= 1e-12
