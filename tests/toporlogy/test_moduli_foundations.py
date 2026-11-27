"""Aggregate moduli, anisotropy indices and the plane-stress reduction of a rotated 3-D tensor: records MO1-MO4 and
E5 of docs/foundations/prereg/T_spectral.md (thresholds preregistered 2026-11-27 01:29 +0300, commit 828d9be, before
this file was run). All stiffnesses are synthetic (not material data). E1 (isotropic pairs and the ν sweep) is
already covered by test_linear_elastic.py::test_l1_* and test_anisotropy.py::test_p6_*.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core import anisotropy as A
from toporlogy.core import linear_elastic as L
from toporlogy.core import material as M

from .test_tensor import random_spd_voigt, rel

_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
ORTHO = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)


def to4(C6: np.ndarray) -> np.ndarray:
    """C_ijkl from a Voigt stiffness, written out here (shear entries are C_yzyz etc.)."""
    C4 = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    C4[p, q, r, s] = C6[a, b]
    return C4


def to6(C4: np.ndarray) -> np.ndarray:
    return np.array([[C4[i, j, k, l] for (k, l) in _PAIRS] for (i, j) in _PAIRS])


def rotate(C6: np.ndarray, R: np.ndarray) -> np.ndarray:
    C6r = to6(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, to4(C6)))
    return 0.5 * (C6r + C6r.T)


def rot(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# ================================================================================================ MO1
@pytest.mark.parametrize("E,nu", [(7.0, 0.3), (1.0, -0.5), (3.0, 0.45), (2.0, 0.0)])
def test_mo1_isotropic_rotated(E: float, nu: float) -> None:
    K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
    rng = np.random.default_rng(101)
    for _ in range(10):
        C = rotate(L.isotropic_stiffness(E, nu), rot(rng))
        m = A.aggregate_moduli(C)
        for k, ref in (("K_V", K), ("K_R", K), ("K_H", K), ("G_V", G), ("G_R", G), ("G_H", G), ("E_H", E),
                       ("nu_H", nu)):
            assert abs(m[k] - ref) <= 1e-12 * max(abs(ref), G), (k, m[k], ref)
        assert abs(A.universal_anisotropy_index(C)) <= 1e-12
    C0 = L.isotropic_stiffness(E, nu)
    assert abs(A.zener_ratio(C0) - 1.0) <= 1e-12


# ================================================================================================ MO2
C11, C12, C44 = 2.0, 1.0, 1.5                                # synthetic cubic, C11 − C12 = 1 ≠ 2 C44 = 3


def _cubic_closed_forms() -> dict[str, float]:
    K = (C11 + 2 * C12) / 3
    Gv = (C11 - C12 + 3 * C44) / 5
    Gr = 5 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    return {"K": K, "Gv": Gv, "Gr": Gr, "AU": 5 * Gv / Gr + K / K - 6}


def test_mo2_cubic_zener_and_anisotropy_index() -> None:
    C = L.cubic_stiffness(C11, C12, C44)
    ref = _cubic_closed_forms()
    z = A.zener_ratio(C)
    assert abs(z - 2 * C44 / (C11 - C12)) <= 1e-14 and abs(z - 3.0) <= 1e-14 and abs(z - 1.0) > 1.0
    m = A.aggregate_moduli(C)
    assert abs(m["K_V"] - ref["K"]) <= 1e-12 * ref["K"] and abs(m["K_R"] - ref["K"]) <= 1e-12 * ref["K"]
    assert abs(m["G_V"] - ref["Gv"]) <= 1e-12 * ref["Gv"] and abs(m["G_R"] - ref["Gr"]) <= 1e-12 * ref["Gr"]
    au = A.universal_anisotropy_index(C)
    assert au > 0.1 and abs(au - ref["AU"]) <= 1e-12 * ref["AU"]


def test_mo2_rotated_cubic_same_aggregate_but_zener_refused() -> None:
    C = L.cubic_stiffness(C11, C12, C44)
    rng = np.random.default_rng(102)
    ref = A.aggregate_moduli(C)
    for _ in range(5):
        Cr = rotate(C, rot(rng))
        m = A.aggregate_moduli(Cr)
        assert all(abs(m[k] - ref[k]) <= 1e-12 * abs(ref[k]) for k in ref)
        with pytest.raises(ValueError, match="cubic axes|cubic symmetry"):
            A.zener_ratio(Cr)


def test_mo2_zener_refuses_non_cubic() -> None:
    ortho = np.linalg.inv(L.orthotropic_compliance(**ORTHO))
    ti = np.linalg.inv(L.transversely_isotropic_compliance(E_p=10.0, E_t=40.0, nu_p=0.3, nu_pt=0.25, G_t=4.0))
    for C in (0.5 * (ortho + ortho.T), 0.5 * (ti + ti.T)):
        with pytest.raises(ValueError, match="cubic"):
            A.zener_ratio(C)
    with pytest.raises(ValueError):
        A.zener_ratio(np.eye(3))


# ================================================================================================ MO3
def test_mo3_voigt_reuss_vs_expanded_formulas_random_spd() -> None:
    Cs, _ = random_spd_voigt(30, np.random.default_rng(103), cond=1e4)
    for C in Cs:
        S = np.linalg.inv(C)
        c, s = C, 0.5 * (S + S.T)
        Kv = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9
        Gv = (c[0, 0] + c[1, 1] + c[2, 2] - (c[0, 1] + c[0, 2] + c[1, 2]) + 3 * (c[3, 3] + c[4, 4] + c[5, 5])) / 15
        Kr = 1 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[0, 2] + s[1, 2]))
        Gr = 15 / (4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
                   + 3 * (s[3, 3] + s[4, 4] + s[5, 5]))
        m = A.aggregate_moduli(C)
        for k, ref in (("K_V", Kv), ("G_V", Gv), ("K_R", Kr), ("G_R", Gr)):
            assert abs(m[k] - ref) <= 1e-10 * abs(ref), (k, m[k], ref)
        assert m["K_V"] >= m["K_R"] * (1 - 1e-12) and m["G_V"] >= m["G_R"] * (1 - 1e-12)
        assert A.universal_anisotropy_index(C) >= -1e-12


# ================================================================================================ MO4
def test_mo4_hill_and_rotation_invariance() -> None:
    rng = np.random.default_rng(104)
    Cs, _ = random_spd_voigt(8, rng, cond=1e3)
    for C in Cs:
        m = A.aggregate_moduli(C)
        assert abs(m["K_H"] - 0.5 * (m["K_V"] + m["K_R"])) <= 1e-14 * m["K_H"]
        assert abs(m["G_H"] - 0.5 * (m["G_V"] + m["G_R"])) <= 1e-14 * m["G_H"]
        K, G = m["K_H"], m["G_H"]
        assert abs(m["E_H"] - 9 * K * G / (3 * K + G)) <= 1e-13 * m["E_H"]
        assert abs(m["nu_H"] - (3 * K - 2 * G) / (2 * (3 * K + G))) <= 1e-13
        mr = A.aggregate_moduli(rotate(C, rot(rng)))
        assert all(abs(mr[k] - m[k]) <= 1e-12 * max(abs(m[k]), 1e-300) for k in m)


def test_mo4_invalid_inputs() -> None:
    C = L.isotropic_stiffness(1.0, 0.3)
    bad = C.copy()
    bad[0, 1] += 0.1
    for x in (bad, -C, np.eye(3), np.ones(6)):
        with pytest.raises(ValueError):
            A.aggregate_moduli(x)
    with pytest.raises(L.NotPositiveDefiniteError):
        A.universal_anisotropy_index(-C)


# ================================================================================================ E5
_P, _O = [0, 1, 5], [2, 3, 4]


def _rotated_ortho(rng: np.random.Generator, about_z: bool = False) -> np.ndarray:
    S = L.orthotropic_compliance(**ORTHO)
    C = np.linalg.inv(0.5 * (S + S.T))
    if about_z:
        t = 0.61
        R = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    else:
        R = rot(rng)
    return rotate(0.5 * (C + C.T), R)


def test_e5_plane_stress_reproduces_3d_solve_and_zero_out_of_plane_stress() -> None:
    rng = np.random.default_rng(105)
    for _ in range(10):
        C = _rotated_ortho(rng)
        ep = rng.standard_normal(3)
        eo = np.linalg.solve(C[np.ix_(_O, _O)], -C[np.ix_(_O, _P)] @ ep)      # dense 3-D solve with σ_o = 0
        e6 = np.zeros(6)
        e6[_P], e6[_O] = ep, eo
        s6 = C @ e6
        assert np.abs(s6[_O]).max() <= 1e-12 * np.abs(s6[_P]).max()
        assert rel(M.plane_stress(C) @ ep, s6[_P]) <= 1e-12
        e6b = np.zeros(6)
        e6b[_P], e6b[_O] = ep, M.plane_stress_recovery(C) @ ep
        s6b = C @ e6b
        assert np.abs(s6b[_O]).max() <= 1e-12 * np.abs(s6b[_P]).max()


def test_e5_plane_strain_matches_zz_zero_full_strain() -> None:
    rng = np.random.default_rng(106)
    for _ in range(10):
        C = _rotated_ortho(rng)
        ep = rng.standard_normal(3)
        e6 = np.zeros(6)
        e6[_P] = ep
        s6 = C @ e6
        assert rel(M.plane_strain(C) @ ep, s6[_P]) <= 1e-12
        assert rel(M.plane_strain_recovery(C) @ ep, s6[_O]) <= 1e-12


def _schur_zz_only(C: np.ndarray) -> np.ndarray:
    return C[np.ix_(_P, _P)] - np.outer(C[_P, 2], C[2, _P]) / C[2, 2]


def test_e5_negative_control_szz_only_elimination() -> None:
    rng = np.random.default_rng(107)
    coupled = [_rotated_ortho(rng) for _ in range(5)]
    for C in coupled:
        assert rel(_schur_zz_only(C), M.plane_stress(C)) > 1e-3        # coupled rotated tensor: σzz alone is not enough
    Cz = _rotated_ortho(rng, about_z=True)
    assert rel(_schur_zz_only(Cz), M.plane_stress(Cz)) <= 1e-13        # rotation about z decouples the out-of-plane shears
