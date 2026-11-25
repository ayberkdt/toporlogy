"""Tensor core (``toporlogy.core.tensor``): records T1-T9, K1-K2, S1 of docs/olcumler/tensor_elastisite.md.

Thresholds are those pre-registered there (2026-11-26 00:29 +0300) before the module existed. Random inputs are
seeded; random stiffnesses are built in Mandel form from a random orthogonal basis and eigenvalues in [1, 1e4]
(condition number ≤ 1e4 by construction), so the κ-scaled tolerances are known in advance.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from toporlogy.core import tensor as T

RNG = np.random.default_rng(20261126)


def random_sym(n: int, d: int = 3, rng: np.random.Generator = RNG) -> np.ndarray:
    A = rng.standard_normal((n, d, d))
    return 0.5 * (A + np.swapaxes(A, 1, 2))


def random_spd_voigt(n: int, rng: np.random.Generator = RNG, cond: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
    """(C_voigt (n, 6, 6), κ (n,)): Mandel M = Q diag(λ) Qᵀ, λ log-uniform in [1, cond], Q random orthogonal."""
    out, ks = [], []
    for _ in range(n):
        Q, _r = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.exp(rng.uniform(0.0, math.log(cond), 6))
        lam[0], lam[1] = 1.0, cond ** rng.uniform(0.2, 1.0)
        M = (Q * lam) @ Q.T
        out.append(T.mandel_to_stiffness(0.5 * (M + M.T)))
        ks.append(lam.max() / lam.min())
    return np.array(out), np.array(ks)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(b))), 1e-300))


# ============================================================================================== T1 Voigt / Mandel
@pytest.mark.parametrize("d", [2, 3])
def test_t1_voigt_roundtrip_exact(d: int) -> None:
    A = random_sym(200, d)
    assert np.array_equal(T.voigt_to_stress(T.stress_to_voigt(A)), A)
    assert np.array_equal(T.voigt_to_strain(T.strain_to_voigt(A)), A)
    v = T.strain_to_voigt(A)
    pairs = T.voigt_pairs(d)
    for a, (i, j) in enumerate(pairs):                      # engineering shear γ = 2 ε, physical σ_ij
        assert np.array_equal(v[:, a], A[:, i, j] * (1.0 if i == j else 2.0))
        assert np.array_equal(T.stress_to_voigt(A)[:, a], A[:, i, j])
    At = torch.as_tensor(A)
    assert torch.equal(T.voigt_to_stress(T.stress_to_voigt(At)), At)
    assert torch.equal(T.voigt_to_strain(T.strain_to_voigt(At)), At)


@pytest.mark.parametrize("d", [2, 3])
def test_t1b_mandel_roundtrip(d: int) -> None:
    A = random_sym(200, d)
    B = T.mandel_to_tensor(T.tensor_to_mandel(A))
    assert np.abs(B - A).max() <= 1e-15 * np.abs(A).max()


def test_t1c_nonsymmetric_tensor_refused_and_rounding_accepted() -> None:
    F = np.eye(3) + 0.3 * RNG.standard_normal((3, 3))
    S = random_sym(1)[0]
    P = T.pk1_from_pk2(F, S)
    with pytest.raises(ValueError, match="not symmetric"):
        T.stress_to_voigt(P)
    with pytest.raises(ValueError, match="not symmetric"):
        T.strain_to_voigt(P)
    A = random_sym(1)[0]
    A2 = A.copy()
    A2[0, 1] += 1e-14 * np.abs(A).max()
    T.stress_to_voigt(A2)                                   # rounding-level asymmetry: accepted


# ============================================================================================== T2 energy
def test_t2_energy_equivalence() -> None:
    sig, eps = random_sym(200), random_sym(200)
    ref = T.ddot(sig, eps)
    v = np.einsum("na,na->n", T.stress_to_voigt(sig), T.strain_to_voigt(eps))
    m = np.einsum("na,na->n", T.tensor_to_mandel(sig), T.tensor_to_mandel(eps))
    scale = T.tensor_norm(sig) * T.tensor_norm(eps)
    assert np.max(np.abs(v - ref) / scale) <= 1e-13
    assert np.max(np.abs(m - ref) / scale) <= 1e-13


# ============================================================================================== T3, T4 conversions
def _deterministic_C() -> list[np.ndarray]:
    from toporlogy.core import linear_elastic as L
    mono = L.orthotropic_compliance(3.0, 2.0, 1.0, 0.7, 0.5, 0.4, 0.2, 0.25, 0.3)
    mono[0, 5] = mono[5, 0] = 0.05
    mono[1, 5] = mono[5, 1] = -0.04
    tri = np.linalg.inv(random_spd_voigt(1, np.random.default_rng(3))[0][0])
    return [L.isotropic_stiffness(1.0, 0.3), L.cubic_stiffness(1.7, 1.2, 0.75),
            np.linalg.inv(L.transversely_isotropic_compliance(1.0, 5.0, 0.3, 0.2, 0.4)),
            np.linalg.inv(L.orthotropic_compliance(40.0, 9.0, 9.0, 4.0, 4.0, 3.2, 0.28, 0.28, 0.4)),
            np.linalg.inv(mono), np.linalg.inv(tri)]


def _samples() -> tuple[np.ndarray, np.ndarray]:
    C, k = random_spd_voigt(120, np.random.default_rng(11))
    det = [0.5 * (c + c.T) for c in _deterministic_C()]
    kd = [float(np.linalg.cond(T.stiffness_to_mandel(c))) for c in det]
    return np.concatenate([C, np.array(det)]), np.concatenate([k, kd])


def test_t3_voigt_tensor_roundtrip_exact() -> None:
    C, _ = _samples()
    assert len(C) == 126
    assert np.array_equal(T.tensor_to_stiffness(T.stiffness_to_tensor(C)), C)
    assert np.array_equal(T.tensor_to_compliance(T.compliance_to_tensor(C)), C)


def test_t3b_inverse_in_tensor_and_mandel_form() -> None:
    C, kappa = _samples()
    Is = T.identity4_sym(3)
    worst = []
    for c, k in zip(C, kappa, strict=True):
        S = np.linalg.inv(c)
        prod = T.ddot44(T.compliance_to_tensor(S), T.stiffness_to_tensor(c))
        e1 = np.abs(prod - Is).max()
        e2 = np.abs(T.compliance_to_mandel(S) - np.linalg.inv(T.stiffness_to_mandel(c))).max() \
            / np.abs(T.compliance_to_mandel(S)).max()
        tol = max(1e-12, 1e-12 * k)
        worst.append(max(e1, e2) / tol)
        assert e1 <= tol and e2 <= tol, (e1, e2, k)


def test_t4_minor_major_symmetry_exact() -> None:
    C, _ = _samples()
    for c in C[:30]:
        assert T.symmetry_errors4(T.stiffness_to_tensor(c)) == (0.0, 0.0)
        assert T.symmetry_errors4(T.compliance_to_tensor(c)) == (0.0, 0.0)


def test_t4b_asymmetric_stiffness_refused_not_symmetrised() -> None:
    from toporlogy.core import linear_elastic as L
    from toporlogy.core.material import rotate_stiffness
    C = L.isotropic_stiffness(1.0, 0.3)
    bad = C.copy()
    bad[0, 1] += 1e-6 * np.abs(C).max()
    with pytest.raises(ValueError, match="not symmetric"):
        L.check_positive_definite(bad)
    with pytest.raises(ValueError, match="not symmetric"):
        rotate_stiffness(bad, np.eye(3))
    with pytest.raises(ValueError, match="not symmetric"):
        L.ElasticTensor(bad)
    ok = C.copy()
    ok[0, 1] += 1e-14 * np.abs(C).max()
    L.check_positive_definite(ok)


# ============================================================================================== T5 rotations
def test_t5_random_rotations_and_negative_controls() -> None:
    R = T.random_rotations(100, np.random.default_rng(5))
    assert np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max() <= 1e-14
    assert np.abs(np.linalg.det(R) - 1.0).max() <= 1e-14
    T.check_rotation(R)
    refl = R[0] @ np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="proper"):
        T.check_rotation(refl)
    T.check_rotation(refl, proper=False)
    bent = R[1] + 1e-8 * np.random.default_rng(6).standard_normal((3, 3))
    with pytest.raises(ValueError, match="orthonormal"):
        T.check_rotation(bent)


def test_t5b_rotation_roundtrip_energy_and_spectrum() -> None:
    C, _ = random_spd_voigt(100, np.random.default_rng(12))
    R = T.random_rotations(100, np.random.default_rng(13))
    eps = random_sym(100, 3, np.random.default_rng(14))
    for c, r, e in zip(C, R, eps, strict=True):
        C4 = T.stiffness_to_tensor(c)
        Cr = T.rotate4(C4, r)
        back = T.rotate4(Cr, r, inverse=True)
        assert rel(back, C4) <= 1e-12
        er = T.rotate2(e, r)
        w0 = T.ddot(e, T.ddot42(C4, e))
        w1 = T.ddot(er, T.ddot42(Cr, er))
        assert abs(w1 - w0) <= 1e-12 * abs(w0)
        l0 = np.linalg.eigvalsh(T.stiffness_to_mandel(c))
        l1 = np.linalg.eigvalsh(T.stiffness_to_mandel(T.tensor_to_stiffness(Cr)))
        assert rel(l1, l0) <= 1e-12


def test_t5c_isotropic_invariant() -> None:
    from toporlogy.core import linear_elastic as L
    C = L.isotropic_stiffness(3.0, 0.27)
    C4 = T.stiffness_to_tensor(C)
    for r in T.random_rotations(50, np.random.default_rng(15)):
        assert np.abs(T.rotate4(C4, r) - C4).max() <= 1e-13 * np.abs(C4).max()


def test_t5d_rotvec_roundtrip() -> None:
    rng = np.random.default_rng(16)
    ax = rng.standard_normal((200, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    ang = rng.uniform(1e-6, math.pi - 1e-3, 200)
    w = ax * ang[:, None]
    R = T.rotation_from_rotvec(w)
    T.check_rotation(R)
    assert np.abs(T.rotvec_from_rotation(R) - w).max() <= 1e-12
    assert np.abs(T.rotation_from_rotvec(T.rotvec_from_rotation(R)) - R).max() <= 1e-12


def test_t5e_plane_angle_periodicity() -> None:
    from toporlogy.core.material import Orthotropic, rotate_stiffness, rotation_about_z
    C = Orthotropic(40.0, 9.0, 9.0, 4.0, 4.0, 3.2, 0.28, 0.28, 0.4).stiffness()
    for th in (0.0, 17.0, 45.0, 101.0):
        a = rotate_stiffness(C, rotation_about_z(th))
        b = rotate_stiffness(C, rotation_about_z(th + 180.0))
        c = rotate_stiffness(C, rotation_about_z(th + 90.0))
        assert rel(b, a) <= 1e-13
        assert rel(c, a) > 1e-3


# ============================================================================================== T6 projectors
def test_t6_identity_projectors_and_isotropic_split() -> None:
    from toporlogy.core import linear_elastic as L
    A = RNG.standard_normal((50, 3, 3))
    Is, Pv, Pd = T.identity4_sym(), T.volumetric_projector(), T.deviatoric_projector()
    assert rel(T.ddot42(Is, A), T.sym(A)) <= 1e-13
    for P in (Pv, Pd):
        assert np.abs(T.ddot44(P, P) - P).max() <= 1e-13
    assert np.abs(T.ddot44(Pv, Pd)).max() <= 1e-13
    assert np.abs(T.trace(T.deviator(A))).max() <= 1e-13 * np.abs(A).max()
    m = L.isotropic_moduli(E=2.5, nu=-0.3)
    split = 3 * m.K * Pv + 2 * m.G * Pd
    assert rel(T.stiffness_to_tensor(L.isotropic_stiffness(2.5, -0.3)), split) <= 1e-13


# ============================================================================================== T7 invariants
def test_t7_invariants_against_eigenvalues() -> None:
    A = random_sym(200)
    w = np.linalg.eigvalsh(A)
    I1, I2, I3 = T.principal_invariants(A)
    s = np.linalg.norm(A, axis=(1, 2))
    assert np.max(np.abs(I1 - w.sum(1)) / s) <= 1e-12
    assert np.max(np.abs(I2 - (w[:, 0] * w[:, 1] + w[:, 1] * w[:, 2] + w[:, 0] * w[:, 2])) / s ** 2) <= 1e-12
    assert np.max(np.abs(I3 - w.prod(1)) / s ** 3) <= 1e-12
    dev = w - w.mean(1, keepdims=True)
    assert np.max(np.abs(T.j2(A) - 0.5 * (dev ** 2).sum(1)) / s ** 2) <= 1e-12
    assert np.max(np.abs(T.j3(A) - dev.prod(1)) / s ** 3) <= 1e-12
    assert np.max(np.abs(T.von_mises_squared(T.stress_to_voigt(A)) - 3 * T.j2(A)) / s ** 2) <= 1e-13
    plane = A.copy()
    plane[:, 2, :] = plane[:, :, 2] = 0.0
    v3 = T.stress_to_voigt(plane[:, :2, :2])
    assert np.max(np.abs(T.von_mises_squared(v3) - 3 * T.j2(plane)) / s ** 2) <= 1e-13


def test_t7b_lode_angle() -> None:
    e = np.diag([1.0, 0.0, 0.0])
    assert abs(T.lode_angle(2.5 * e) - 0.0) <= 1e-6
    assert abs(T.lode_angle(-2.5 * e) - math.pi / 3) <= 1e-6
    shear = np.zeros((3, 3))
    shear[0, 1] = shear[1, 0] = 1.7
    assert abs(T.lode_angle(shear) - math.pi / 6) <= 1e-12
    assert np.isnan(T.lode_angle(3.0 * np.eye(3)))


# ============================================================================================== T8 spectral derivatives
def test_t8_principal_value_gradients_fd() -> None:
    rng = np.random.default_rng(17)
    n = 0
    while n < 50:
        A = random_sym(1, 3, rng)[0]
        w = np.linalg.eigvalsh(A)
        if np.min(np.diff(w)) < 0.1 * np.linalg.norm(A):
            continue
        n += 1
        _, G, rep = T.principal_value_gradients(A)
        assert not rep.any()
        D = random_sym(1, 3, rng)[0]
        h = 1e-6 * np.linalg.norm(A)
        fd = (np.linalg.eigvalsh(A + h * D) - np.linalg.eigvalsh(A - h * D)) / (2 * h)
        an = np.einsum("iab,ab->i", G, D)
        assert np.max(np.abs(fd - an)) <= 1e-6 * np.max(np.abs(an)), (fd, an)


def test_t8b_repeated_eigenvalue_flagged() -> None:
    A = np.diag([2.0, 2.0, -1.0])
    w, G, rep = T.principal_value_gradients(A)
    assert rep.tolist() == [False, True, True] or rep.tolist() == [True, True, False]
    assert np.isnan(G[rep]).all() and np.isfinite(G[~rep]).all()


# ============================================================================================== T9 smoothing
def test_t9_smoothed_von_mises_error_bound() -> None:
    from toporlogy.optimization.strength import von_mises_equivalent
    s = torch.as_tensor(RNG.standard_normal((1000, 6)) * np.exp(RNG.uniform(-12, 2, (1000, 1))))
    e = 1e-6
    err = (von_mises_equivalent(s, e) - von_mises_equivalent(s, 0.0)).numpy()
    assert err.min() >= 0.0 and err.max() <= e * (1 + 1e-12)


# ============================================================================================== K1 kinematics
def _random_F(n: int, rng: np.random.Generator, scale: float = 0.3) -> np.ndarray:
    F = []
    while len(F) < n:
        f = np.eye(3) + scale * rng.standard_normal((3, 3))
        if np.linalg.det(f) > 0.05:
            F.append(f)
    return np.array(F)


def test_k1_polar_and_cauchy_green() -> None:
    F = _random_F(100, np.random.default_rng(21))
    C, B = T.right_cauchy_green(F), T.left_cauchy_green(F)
    assert rel(C, np.einsum("nki,nkj->nij", F, F)) <= 1e-15
    assert rel(B, np.einsum("nik,njk->nij", F, F)) <= 1e-15
    R, U, V = T.polar_decomposition(F)
    nf = np.linalg.norm(F, axis=(1, 2))[:, None, None]
    assert np.max(np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3))) <= 1e-13
    assert np.max(np.abs(np.linalg.det(R) - 1)) <= 1e-13
    assert np.max(np.abs(R @ U - F) / nf) <= 1e-13 and np.max(np.abs(V @ R - F) / nf) <= 1e-13
    assert np.linalg.eigvalsh(U).min() > 0 and np.linalg.eigvalsh(V).min() > 0


def test_k1b_strain_measures() -> None:
    F = _random_F(100, np.random.default_rng(22))
    H = F - np.eye(3)
    E = T.green_lagrange(H)
    assert rel(E, 0.5 * (T.right_cauchy_green(F) - np.eye(3))) <= 1e-12
    Fi = np.linalg.inv(F)
    assert rel(T.euler_almansi(F), np.swapaxes(Fi, 1, 2) @ E @ Fi) <= 1e-12
    lnU = T.hencky_strain(F)
    w, Q = np.linalg.eigh(2 * lnU)
    expC = np.einsum("nik,nk,njk->nij", Q, np.exp(w), Q)
    assert rel(expC, T.right_cauchy_green(F)) <= 1e-12
    R, _, _ = T.polar_decomposition(F)
    assert rel(T.hencky_strain(F, "spatial"), R @ lnU @ np.swapaxes(R, 1, 2)) <= 1e-12


def test_k1c_infinitesimal_limit_bound() -> None:
    rng = np.random.default_rng(23)
    H = rng.standard_normal((100, 3, 3))
    H *= 1e-6 / np.linalg.norm(H, axis=(1, 2))[:, None, None]
    d = np.linalg.norm(T.green_lagrange(H) - T.infinitesimal_strain(H), axis=(1, 2))
    assert np.all(d <= 0.5 * np.linalg.norm(H, axis=(1, 2)) ** 2 * (1 + 1e-6))


def _exact_green_and_jm1(H: np.ndarray) -> tuple[np.ndarray, float]:
    Hf = [[Fraction(float(x)) for x in row] for row in H]
    E = [[(Hf[i][j] + Hf[j][i] + sum(Hf[k][i] * Hf[k][j] for k in range(3))) / 2 for j in range(3)]
         for i in range(3)]
    F = [[Hf[i][j] + (1 if i == j else 0) for j in range(3)] for i in range(3)]
    det = (F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
           + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]))
    return np.array([[float(x) for x in r] for r in E]), float(det - 1)


def test_k1d_small_displacement_gradient_precision() -> None:
    rng = np.random.default_rng(24)
    errF = []
    for _ in range(10):
        H = 1e-10 * rng.standard_normal((3, 3))
        E_ex, jm1_ex = _exact_green_and_jm1(H)
        E = T.green_lagrange(H[None])[0]
        assert rel(E, E_ex) <= 1e-14
        assert abs(T.jacobian_minus_one(H[None])[0] - jm1_ex) <= 1e-14 * abs(jm1_ex)
        F = np.eye(3) + H
        errF.append(rel(0.5 * (F.T @ F - np.eye(3)), E_ex))
    print(f"K1d (report only): F-form Green-Lagrange relative error at |H| = 1e-10: max {max(errF):.3e}")


# ============================================================================================== K2 admissibility
def test_k2_inverted_singular_near_singular() -> None:
    F = np.diag([1.0, 1.0, -0.5])
    with pytest.raises(T.InvertedDeformationError):
        T.jacobian(F)
    with pytest.raises(T.InvertedDeformationError):
        T.polar_decomposition(F)
    with pytest.raises(T.InvertedDeformationError):
        T.cauchy_from_pk2(F, np.eye(3))
    Z = np.diag([1.0, 1.0, 0.0])
    with pytest.raises(T.SingularDeformationError):
        T.jacobian(Z)
    with pytest.raises(T.SingularDeformationError):
        T.hencky_strain(Z)
    near = np.diag([1.0, 1.0, 1e-9])
    st = T.deformation_state(np.array([np.eye(3), F, Z, near]))
    assert st.tolist() == ["regular", "inverted", "singular", "near_singular"]
    assert T.jacobian(near) > 0                              # admissible, flagged only


# ============================================================================================== S1 stress measures
def _svk(C4: np.ndarray, F: np.ndarray) -> tuple[float, np.ndarray]:
    E = T.green_lagrange(F - np.eye(3))
    S = T.ddot42(C4, E)
    return 0.5 * float(T.ddot(E, S).sum()), S


def test_s1_pk1_is_energy_derivative() -> None:
    from toporlogy.core import linear_elastic as L
    C4 = T.stiffness_to_tensor(L.isotropic_stiffness(1.0, 0.3))
    rng = np.random.default_rng(31)
    for F in _random_F(20, rng):
        _, S = _svk(C4, F[None])
        P = T.pk1_from_pk2(F, S[0])
        fd = np.zeros((3, 3))
        h = 1e-6
        for i in range(3):
            for j in range(3):
                dF = np.zeros((3, 3))
                dF[i, j] = h
                fd[i, j] = (_svk(C4, (F + dF)[None])[0] - _svk(C4, (F - dF)[None])[0]) / (2 * h)
        assert rel(fd, P) <= 1e-6


def test_s1b_work_conjugacy() -> None:
    rng = np.random.default_rng(32)
    F = _random_F(100, rng)
    S = random_sym(100, 3, rng)
    Fdot = rng.standard_normal((100, 3, 3))
    Edot = T.sym(np.swapaxes(F, 1, 2) @ Fdot)                # Ė = sym(Fᵀ Ḟ)
    d = T.rate_of_deformation(F, Fdot)
    P, tau, sig = T.pk1_from_pk2(F, S), T.kirchhoff_from_pk2(F, S), T.cauchy_from_pk2(F, S)
    J = T.jacobian(F)
    w = T.ddot(S, Edot)
    scale = np.linalg.norm(P, axis=(1, 2)) * np.linalg.norm(Fdot, axis=(1, 2))
    for other in (T.ddot(P, Fdot), T.ddot(tau, d), J * T.ddot(sig, d)):
        assert np.max(np.abs(other - w) / scale) <= 1e-12


def test_s1c_stress_roundtrips() -> None:
    rng = np.random.default_rng(33)
    F = _random_F(100, rng)
    S = random_sym(100, 3, rng)
    sig = T.cauchy_from_pk2(F, S)
    assert np.max(np.abs(sig - np.swapaxes(sig, 1, 2))) <= 1e-13 * np.abs(sig).max()
    assert rel(T.pk2_from_cauchy(F, sig), S) <= 1e-12
    assert rel(T.pk2_from_pk1(F, T.pk1_from_pk2(F, S)), S) <= 1e-12
    T.stress_to_voigt(sig, rtol=1e-12)                       # Cauchy has a Voigt form


def test_backend_torch_matches_numpy_and_is_differentiable() -> None:
    """Post hoc (not in the pre-registered list): one implementation serves both backends."""
    A = random_sym(10)
    R = T.random_rotations(10, np.random.default_rng(40))
    At = torch.as_tensor(A, dtype=torch.float64).requires_grad_(True)
    Rt = torch.as_tensor(R)
    out = T.von_mises_squared(T.stress_to_voigt(T.rotate2(At, Rt, inverse=True), rtol=None))
    assert np.allclose(out.detach().numpy(), T.von_mises_squared(T.stress_to_voigt(T.rotate2(A, R, inverse=True))),
                       rtol=1e-13, atol=0)
    out.sum().backward()
    assert At.grad is not None and torch.isfinite(At.grad).all()
