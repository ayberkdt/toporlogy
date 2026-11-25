"""Linear elastic family (``toporlogy.core.linear_elastic``, plane reductions of ``toporlogy.core.material``):
records L1-L7 of docs/olcumler/tensor_elastisite.md (thresholds pre-registered 2026-11-26 00:29 +0300)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from toporlogy.core import linear_elastic as L
from toporlogy.core import material as M
from toporlogy.core import tensor as T
from toporlogy.fem.elements import D_plane_stress

from .test_tensor import random_spd_voigt, rel

NUS = [-0.99, -0.5, -0.3, 0.0, 0.3, 0.49]


# ============================================================================================== L1 isotropic
@pytest.mark.parametrize("nu", NUS)
def test_l1_isotropic_conversions_roundtrip(nu: float) -> None:
    ref = L.isotropic_moduli(E=2.0, nu=nu)
    pairs = [dict(E=ref.E, nu=ref.nu), dict(E=ref.E, G=ref.G), dict(G=ref.G, K=ref.K), dict(lam=ref.lam, G=ref.G),
             dict(E=ref.E, K=ref.K), dict(M=ref.M, G=ref.G)]
    for kw in pairs:
        m = L.isotropic_moduli(**kw)
        for k in ("E", "nu", "G", "K", "lam", "M"):
            a, b = getattr(m, k), getattr(ref, k)
            scale = max(abs(b), ref.G)                       # ν and λ can be 0: compare on the modulus scale
            if k == "nu":
                scale = max(abs(b), 1.0)
            assert abs(a - b) <= 1e-13 * scale, (kw, k, a, b)


def test_l1b_isotropic_admissibility() -> None:
    for nu in (0.5, -1.0, 0.6, -1.2):
        with pytest.raises(ValueError):
            L.isotropic_moduli(E=1.0, nu=nu)
        with pytest.raises(ValueError):
            L.isotropic_stiffness(1.0, nu)
    for kw in (dict(E=0.0, nu=0.3), dict(E=-1.0, nu=0.3), dict(G=0.0, K=1.0), dict(G=-1.0, K=1.0),
               dict(G=1.0, K=-0.1)):
        with pytest.raises(ValueError):
            L.isotropic_moduli(**kw)
    m = L.isotropic_moduli(E=1.0, nu=-0.3)
    assert m.lam < 0                                         # λ < 0 alone is admissible
    L.ElasticTensor.isotropic(1.0, -0.3)


@pytest.mark.parametrize("nu", NUS)
def test_l1c_isotropic_stiffness_lame_form(nu: float) -> None:
    m = L.isotropic_moduli(E=3.0, nu=nu)
    lame = 2 * m.mu * T.identity4_sym() + m.lam * T.outer(np.eye(3), np.eye(3))
    assert rel(T.stiffness_to_tensor(L.isotropic_stiffness(3.0, nu)), lame) <= 1e-14
    assert rel(np.linalg.inv(L.isotropic_stiffness(3.0, nu)), L.isotropic_compliance(3.0, nu)) <= 1e-13 * max(
        1.0, float(np.linalg.cond(T.stiffness_to_mandel(L.isotropic_stiffness(3.0, nu)))))


# ============================================================================================== L2 orthotropic
ORTHO = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)


def test_l2_orthotropic_constants_roundtrip_and_reciprocity() -> None:
    S = L.orthotropic_compliance(**ORTHO)
    ec = L.engineering_constants(S)
    for k, v in ORTHO.items():
        assert abs(ec[k] - v) <= 1e-13 * abs(v), k
    assert abs(ec["nu21"] - ORTHO["nu12"] * ORTHO["E2"] / ORTHO["E1"]) <= 1e-13 * ec["nu21"]
    assert abs(ec["nu31"] - ORTHO["nu13"] * ORTHO["E3"] / ORTHO["E1"]) <= 1e-13 * ec["nu31"]
    assert abs(ec["nu32"] - ORTHO["nu23"] * ORTHO["E3"] / ORTHO["E2"]) <= 1e-13 * ec["nu32"]
    assert ec["coupling"] == 0.0
    et = L.ElasticTensor.orthotropic(**ORTHO)
    ec2 = et.engineering_constants()
    for k, v in ORTHO.items():
        assert abs(ec2[k] - v) <= 1e-12 * abs(v), k


def test_l2b_anisotropic_poisson_outside_isotropic_bounds() -> None:
    """ν12 = 2 with E1/E2 = 10 is admissible; ν12² > E1/E2 is not; an SPD example with ν_ij < −1 exists. Forcing every
    ν_ij into the isotropic (−1, 1/2) would reject the first and the last."""
    good = dict(E1=10.0, E2=1.0, E3=1.0, G12=0.5, G13=0.5, G23=0.4, nu12=2.0, nu13=0.1, nu23=0.1)
    M.Orthotropic(**good)
    L.ElasticTensor.orthotropic(**good)
    bad = dict(good, nu12=math.sqrt(10.0) * 1.001)
    with pytest.raises(ValueError, match="positive definite"):
        M.Orthotropic(**bad)
    with pytest.raises(L.NotPositiveDefiniteError):
        L.check_positive_definite(L.orthotropic_compliance(**bad), "compliance")
    neg = dict(E1=10.0, E2=1.0, E3=1.0, G12=0.5, G13=0.5, G23=0.4, nu12=-2.0, nu13=0.1, nu23=0.1)
    diag = L.check_positive_definite(L.orthotropic_compliance(**neg), "compliance")
    assert diag["min_eigenvalue"] > 0 and neg["nu12"] < -1


# ============================================================================================== L3 transversely isotropic
def test_l3_transversely_isotropic_invariance() -> None:
    rng = np.random.default_rng(41)
    for axis in (0, 1, 2):
        S = L.transversely_isotropic_compliance(2.0, 7.0, 0.25, 0.15, 0.9, axis=axis)
        C = np.linalg.inv(S)
        C = 0.5 * (C + C.T)
        assert L.symmetry_class(C) == f"transversely_isotropic_{'xyz'[axis]}"
        e = np.eye(3)[axis]
        for th in rng.uniform(0, 2 * math.pi, 10):
            R = T.rotation_from_rotvec(th * e)
            assert np.abs(M.rotate_stiffness(C, R) - C).max() <= 1e-13 * np.abs(C).max()
        p, q = [k for k in range(3) if k != axis]
        assert abs(C[3 + axis, 3 + axis] - 0.5 * (C[p, p] - C[p, q])) <= 1e-13 * np.abs(C).max()
        perp = T.rotation_from_rotvec(0.6 * np.eye(3)[(axis + 1) % 3])
        assert rel(M.rotate_stiffness(C, perp), C) > 1e-3


# ============================================================================================== L4 general
def test_l4_general_from_C_or_S_and_refusals() -> None:
    C, kappa = random_spd_voigt(30, np.random.default_rng(42))
    for c, k in zip(C, kappa, strict=True):
        a = L.ElasticTensor.from_stiffness(c)
        b = L.ElasticTensor.from_compliance(np.linalg.inv(c))
        assert rel(b.C_material, a.C_material) <= max(1e-12, 1e-12 * k)
        assert abs(a.diagnostics["condition_number"] - k) <= 1e-8 * k
    c = C[0]
    w, Q = np.linalg.eigh(T.stiffness_to_mandel(c))
    w[0] = -0.1 * w[0]
    indef = T.mandel_to_stiffness(0.5 * ((Q * w) @ Q.T + ((Q * w) @ Q.T).T))
    with pytest.raises(L.NotPositiveDefiniteError):
        L.ElasticTensor(indef)
    asym = c.copy()
    asym[0, 5] *= 1 + 1e-6
    with pytest.raises(ValueError, match="not symmetric"):
        L.ElasticTensor(asym)
    with pytest.raises(ValueError, match="not symmetric"):
        L.ElasticTensor.from_compliance(asym)


# ============================================================================================== L5 symmetry class
def test_l5_symmetry_classes() -> None:
    iso = L.isotropic_stiffness(1.0, 0.3)
    assert L.symmetry_class(iso) == "isotropic"
    assert L.symmetry_class(M.rotate_stiffness(iso, T.random_rotations(1, np.random.default_rng(1))[0])) == "isotropic"
    assert L.symmetry_class(L.cubic_stiffness(1.7, 1.2, 0.75)) == "cubic"
    assert L.symmetry_class(np.linalg.inv(L.transversely_isotropic_compliance(1.0, 3.0, 0.3, 0.2, 0.4))) \
        == "transversely_isotropic_z"
    assert L.symmetry_class(np.linalg.inv(L.orthotropic_compliance(**ORTHO))) == "orthotropic"
    assert L.symmetry_class(random_spd_voigt(1, np.random.default_rng(2))[0][0]) == "anisotropic"


# ============================================================================================== L6 strong ellipticity
def test_l6_isotropic_acoustic_tensor() -> None:
    m = L.isotropic_moduli(E=1.0, nu=0.3)
    n = L.fibonacci_sphere(500)
    w = np.linalg.eigvalsh(L.acoustic_tensor(L.isotropic_stiffness(1.0, 0.3), n))
    ref = np.array(sorted([m.mu, m.mu, m.lam + 2 * m.mu]))
    assert np.abs(w - ref).max() <= 1e-13 * ref.max()


def test_l6b_positive_definite_implies_sampled_ellipticity() -> None:
    C, _ = random_spd_voigt(120, np.random.default_rng(43))
    for c in C:
        r = L.strong_ellipticity(c, 400, refine=False)
        assert r["min_eigenvalue"] > 0 and r["label"] == "sampled"


def test_l6c_ellipticity_is_not_positive_definiteness() -> None:
    lam, mu = -1.5, 1.0
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[3:, 3:] = np.eye(3) * mu
    with pytest.raises(L.NotPositiveDefiniteError):
        L.check_positive_definite(C)
    r = L.strong_ellipticity(C, 2000)
    assert abs(r["min_eigenvalue"] - min(mu, lam + 2 * mu)) <= 1e-12 and r["label"] == "refined"
    C2 = C.copy()
    C2[:3, :3] = -2.5
    C2[np.arange(3), np.arange(3)] = -2.5 + 2 * mu               # λ + 2μ = −0.5 < 0
    assert L.strong_ellipticity(C2, 500)["min_eigenvalue"] < 0


# ============================================================================================== L7 plane reductions
_P, _O = [0, 1, 5], [2, 3, 4]


def _schur(C: np.ndarray, out: list[int]) -> np.ndarray:
    return C[np.ix_(_P, _P)] - C[np.ix_(_P, out)] @ np.linalg.solve(C[np.ix_(out, out)], C[np.ix_(out, _P)])


def test_l7_plane_stress_compliance_form_is_schur_complement() -> None:
    C, kappa = random_spd_voigt(120, np.random.default_rng(44))
    for c, k in zip(C, kappa, strict=True):
        assert rel(M.plane_stress(c), _schur(c, _O)) <= max(1e-12, 1e-12 * k)


def test_l7b_isotropic_and_3d_reference() -> None:
    for nu in (-0.5, 0.0, 0.3, 0.45):
        assert rel(M.plane_stress(L.isotropic_stiffness(1.0, nu)), D_plane_stress(nu)) <= 1e-12
    rng = np.random.default_rng(45)
    C, _ = random_spd_voigt(30, rng)
    for c in C:
        ep = rng.standard_normal(3)
        # 3-D: unknown out-of-plane strains with σ_o = 0
        eo = np.linalg.solve(c[np.ix_(_O, _O)], -c[np.ix_(_O, _P)] @ ep)
        e6 = np.zeros(6)
        e6[_P], e6[_O] = ep, eo
        s6 = c @ e6
        assert rel(M.plane_stress(c) @ ep, s6[_P]) <= 1e-12
        assert rel(M.plane_stress_recovery(c) @ ep, eo) <= 1e-12


def test_l7c_szz_only_condensation_versus_full() -> None:
    C = M.Orthotropic(**ORTHO).stiffness()
    Cz = M.rotate_stiffness(C, M.rotation_about_z(35.0))
    assert rel(_schur(Cz, [2]), M.plane_stress(Cz)) <= 1e-13           # out-of-plane shear decoupled
    Cg = M.rotate_stiffness(C, M.rotation_from_axes([1.0, 0.4, 0.7], [0.0, 1.0, -0.3]))
    assert rel(_schur(Cg, [2]), M.plane_stress(Cg)) > 1e-3               # σzz alone is not enough


def test_l7d_plane_strain_recovery() -> None:
    rng = np.random.default_rng(46)
    C, _ = random_spd_voigt(30, rng)
    for c in C:
        ep = rng.standard_normal(3)
        e6 = np.zeros(6)
        e6[_P] = ep
        s6 = c @ e6
        assert rel(M.plane_strain(c) @ ep, s6[_P]) <= 1e-13
        assert rel(M.plane_strain_recovery(c) @ ep, s6[_O]) <= 1e-13
        assert np.array_equal(M.out_of_plane_row(c), M.plane_strain_recovery(c)[0])


def test_l7e_plane_stress_derivative_fd() -> None:
    rng = np.random.default_rng(47)
    C, _ = random_spd_voigt(20, rng, cond=1e2)
    for c in C:
        d = rng.standard_normal((6, 6))
        d = 0.5 * (d + d.T)
        h = 1e-6 * np.abs(c).max() / np.abs(d).max()
        fd = (M.plane_stress(c + h * d) - M.plane_stress(c - h * d)) / (2 * h)
        an = M.plane_stress_derivative(c, d)
        assert rel(fd, an) <= 1e-6


def test_post_verify_d_iso_agrees_with_the_core() -> None:
    """Post hoc: post.verify.d_iso stays as written (bit-exact T10 record P5) and equals the core to rounding."""
    from toporlogy.post.verify import d_iso
    for E, nu in ((7.0, 0.27), (1.0, -0.5), (3.0, 0.45)):
        assert rel(d_iso(E, nu), L.isotropic_stiffness(E, nu)) <= 1e-15
