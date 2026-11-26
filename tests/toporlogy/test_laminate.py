"""Classical lamination theory and first-ply failure: records L1-L11 of ``docs/olcumler/laminat_uq_onkayit.md``.

Every material and strength number here is SYNTHETIC (chosen for a well-conditioned test, not a measured material).
The independent sides are written in this file: the classical trigonometric Q̄ formulas, the 3-D rotation + plane
stress route through ``core.material`` and a Gauss quadrature through the thickness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from toporlogy.core import laminate as L
from toporlogy.core.linear_elastic import ElasticTensor
from toporlogy.core.material import OrthotropicStrength, plane_stress, rotate_stiffness, rotation_about_z
from toporlogy.optimization.laminate_failure import (failure_modes, first_ply_failure, load_factor_to_failure,
                                                     ply_failure_index)

E1, E2, G12, NU12 = 140e3, 10e3, 5e3, 0.3
MAT = ElasticTensor.orthotropic(E1, E2, E2, G12, G12, 3.5e3, NU12, NU12, 0.4)
ALPHA = (-0.5e-6, 30e-6)
STRENGTH = OrthotropicStrength(Xt=1500.0, Xc=1200.0, Yt=50.0, Yc=200.0, S12=70.0, S23=50.0)
T_PLY = 0.125


def classical_qbar(theta_deg: float) -> np.ndarray:
    """Tsai-Pagano trigonometric transformation (written here, not in the library)."""
    nu21 = NU12 * E2 / E1
    d = 1.0 - NU12 * nu21
    q11, q22, q12, q66 = E1 / d, E2 / d, NU12 * E2 / d, G12
    m, n = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    b11 = q11 * m**4 + 2 * (q12 + 2 * q66) * m**2 * n**2 + q22 * n**4
    b22 = q11 * n**4 + 2 * (q12 + 2 * q66) * m**2 * n**2 + q22 * m**4
    b12 = (q11 + q22 - 4 * q66) * m**2 * n**2 + q12 * (m**4 + n**4)
    b16 = (q11 - q12 - 2 * q66) * m**3 * n - (q22 - q12 - 2 * q66) * m * n**3
    b26 = (q11 - q12 - 2 * q66) * m * n**3 - (q22 - q12 - 2 * q66) * m**3 * n
    b66 = (q11 + q22 - 2 * q12 - 2 * q66) * m**2 * n**2 + q66 * (m**4 + n**4)
    return np.array([[b11, b12, b16], [b12, b22, b26], [b16, b26, b66]])


def route_3d(theta_deg: float) -> np.ndarray:
    """Rotate the 3-D stiffness, then reduce to plane stress (independent of the 2-D rotation of the library)."""
    return plane_stress(rotate_stiffness(MAT.stiffness(), rotation_about_z(theta_deg)))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def random_stacks(n: int = 20, seed: int = 11) -> list[L.Laminate]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(1, 13))
        out.append(L.Laminate.from_angles(MAT, rng.uniform(-90, 90, k), rng.uniform(0.05, 0.3, k), ALPHA))
    return out


def quad(lam: L.Laminate, points: int = 3):
    """Gauss points (ply, z, weight) through the thickness."""
    x, w = np.polynomial.legendre.leggauss(points)
    z = lam.z
    for k in range(lam.n_plies):
        a, b = z[k], z[k + 1]
        for xi, wi in zip(x, w, strict=True):
            yield k, 0.5 * (a + b) + 0.5 * (b - a) * xi, 0.5 * (b - a) * wi


# --------------------------------------------------------------------------------------------------- L1-L3
def test_l1_reduced_stiffness_matches_the_analytic_form() -> None:
    Q = L.ply_reduced_stiffness(MAT)
    nu21 = NU12 * E2 / E1
    d = 1.0 - NU12 * nu21
    want = np.array([[E1 / d, NU12 * E2 / d, 0], [NU12 * E2 / d, E2 / d, 0], [0, 0, G12]])
    assert rel(Q, want) <= 1e-12


def test_l2_qbar_matches_classical_formulas_and_the_3d_route() -> None:
    angles = np.linspace(-90, 90, 25)
    Qb = L.transformed_reduced_stiffness(L.ply_reduced_stiffness(MAT), angles)
    for a, q in zip(angles, Qb, strict=True):
        assert rel(q, classical_qbar(a)) <= 1e-12
        assert rel(q, route_3d(a)) <= 1e-12


def test_l3_single_ply() -> None:
    h = 0.4
    lam = L.Laminate.from_angles(MAT, [30.0], h)
    Qb = classical_qbar(30.0)
    assert rel(lam.A, Qb * h) <= 1e-12
    assert rel(lam.D, Qb * h**3 / 12) <= 1e-12
    assert np.abs(lam.B).max() <= 1e-12 * np.abs(lam.A).max() * h


def test_l4_abd_matches_through_thickness_quadrature() -> None:
    for lam in random_stacks():
        K = np.zeros((6, 6))
        for k, z, w in quad(lam):
            q = route_3d(lam.angles[k])
            K[:3, :3] += w * q
            K[:3, 3:] += w * z * q
            K[3:, :3] += w * z * q
            K[3:, 3:] += w * z * z * q
        ref = lam.abd()
        for sl in ((slice(0, 3), slice(0, 3)), (slice(0, 3), slice(3, 6)), (slice(3, 6), slice(3, 6))):
            scale = np.abs(ref[:3, :3]).max() * lam.thickness ** (sl[0].start // 3 + sl[1].start // 3)
            assert np.abs(ref[sl] - K[sl]).max() <= 1e-12 * scale


# --------------------------------------------------------------------------------------------------- L5-L6
def test_l5a_symmetric_stacks_have_no_coupling() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        k = int(rng.integers(1, 7))
        lam = L.Laminate.from_angles(MAT, rng.uniform(-90, 90, k), rng.uniform(0.05, 0.3, k), symmetric=True)
        c = lam.classification()
        assert c["symmetric_stack"] and c["B_zero"]
        assert np.abs(lam.B).max() <= 1e-12 * np.abs(lam.A).max() * lam.thickness


def test_l5b_balanced_stacks_have_no_shear_extension_coupling() -> None:
    for angles in ([30, -30], [45, -45, 0, 90], [15, 60, -60, -15], [20, -20, 20, -20]):
        lam = L.Laminate.from_angles(MAT, angles, T_PLY)
        c = lam.classification()
        assert c["balanced_stack"] and c["A16_A26_zero"]
        assert max(abs(lam.A[0, 2]), abs(lam.A[1, 2])) <= 1e-12 * np.abs(lam.A).max()
    unbalanced = L.Laminate.from_angles(MAT, [30, 30], T_PLY).classification()
    assert not unbalanced["balanced_stack"] and not unbalanced["A16_A26_zero"]


def test_l5c_cross_ply_coupling_is_analytic() -> None:
    t = 0.2
    lam = L.Laminate.from_angles(MAT, [0.0, 90.0], t)
    Q = L.ply_reduced_stiffness(MAT)
    b = (Q[1, 1] - Q[0, 0]) * t**2 / 2
    want = np.array([[b, 0, 0], [0, -b, 0], [0, 0, 0]])
    assert np.abs(lam.B - want).max() <= 1e-12 * abs(b)
    c = lam.classification()
    assert not c["symmetric_stack"] and not c["B_zero"] and c["A16_A26_zero"]


def test_l6_quasi_isotropic() -> None:
    lam = L.Laminate.from_angles(MAT, [0, 45, -45, 90], T_PLY, symmetric=True)
    A = lam.A
    s = np.abs(A).max()
    assert abs(A[0, 0] - A[1, 1]) <= 1e-12 * s
    assert abs(A[2, 2] - 0.5 * (A[0, 0] - A[0, 1])) <= 1e-12 * s
    assert max(abs(A[0, 2]), abs(A[1, 2])) <= 1e-12 * s


# --------------------------------------------------------------------------------------------------- L7-L8
def test_l7a_isotropic_ply_expands_freely() -> None:
    E, nu, a, dT = 70e3, 0.33, 23e-6, 80.0
    lam = L.Laminate.from_angles(ElasticTensor.isotropic(E, nu), [17.0], 1.3, (a, a))
    st = lam.response(dT0=dT)
    assert np.abs(st.eps0 - np.array([a * dT, a * dT, 0])).max() <= 1e-12 * a * dT
    assert np.abs(st.kappa).max() <= 1e-12 * a * dT
    assert np.abs(lam.ply_stresses(st).stress).max() <= 1e-12 * E * a * dT


def integrate_stress(lam: L.Laminate, st: L.LaminateState) -> tuple[np.ndarray, np.ndarray, float]:
    N, M, smax = np.zeros(3), np.zeros(3), 0.0
    for k, z, w in quad(lam):
        s = lam.stresses_at(st, k, z)[0]
        N += w * s
        M += w * z * s
        smax = max(smax, float(np.abs(s).max()))
    return N, M, smax


def test_l7b_cross_ply_residual_stress_is_self_equilibrated() -> None:
    lam = L.Laminate.from_angles(MAT, [0, 90], T_PLY, ALPHA, symmetric=True)
    N, M, smax = integrate_stress(lam, lam.response(dT0=-120.0))
    assert smax > 1.0                                                     # the residual stress is not trivially zero
    assert np.abs(N).max() <= 1e-12 * smax * lam.thickness
    assert np.abs(M).max() <= 1e-12 * smax * lam.thickness**2


def test_l8_recovered_stresses_give_back_the_applied_resultants() -> None:
    rng = np.random.default_rng(8)
    for lam in random_stacks(20, seed=12):
        N = rng.uniform(-100, 100, 3)
        M = rng.uniform(-20, 20, 3)
        dT0, dT1 = rng.uniform(-100, 100), rng.uniform(-200, 200)
        got_N, got_M, smax = integrate_stress(lam, lam.response(N, M, dT0, dT1))
        h = lam.thickness
        assert np.abs(got_N - N).max() <= 1e-10 * max(np.abs(N).max(), smax * h)
        assert np.abs(got_M - M).max() <= 1e-10 * max(np.abs(M).max(), smax * h**2)


# --------------------------------------------------------------------------------------------------- L9
LAYUP = np.array([12.0, -37.0, 71.0, 5.0])
THICK = np.array([0.11, 0.2, 0.15, 0.09])
LOAD = np.array([120.0, -35.0, 40.0, 3.0, -1.5, 2.0])


def test_l9a_abd_derivatives_match_central_differences() -> None:
    lam = L.Laminate.from_angles(MAT, LAYUP, THICK)
    dA, dT = L.abd_design_derivatives(lam)
    Q = lam.Q_plies

    def K(a: np.ndarray, t: np.ndarray) -> np.ndarray:
        return L.abd_matrix(L.transformed_reduced_stiffness(Q, a), t)

    for k in range(4):
        e = np.zeros(4)
        e[k] = 1.0
        for jac, ha, ht, h in ((dA, 1e-4, 0.0, 1e-4), (dT, 0.0, 1e-7, 1e-7)):
            fd = (K(LAYUP + ha * e, THICK + ht * e) - K(LAYUP - ha * e, THICK - ht * e)) / (2 * h)
            assert rel(jac[k], fd) <= 1e-6


def test_l9b_energy_taylor_remainder_is_second_order() -> None:
    Q = L.Laminate.from_angles(MAT, LAYUP, THICK).Q_plies
    a = torch.tensor(LAYUP, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(THICK, dtype=torch.float64, requires_grad=True)
    W = L.complementary_energy(Q, a, t, LOAD)
    ga, gt = torch.autograd.grad(W, (a, t))
    rng = np.random.default_rng(9)
    da, dt = rng.uniform(-1, 1, 4) * 5.0, rng.uniform(-1, 1, 4) * 0.01
    slope = float(ga @ torch.as_tensor(da) + gt @ torch.as_tensor(dt))
    hs = np.array([2.0**-i for i in range(2, 8)])
    res = [abs(L.complementary_energy(Q, LAYUP + h * da, THICK + h * dt, LOAD) - float(W) - h * slope) for h in hs]
    order = np.polyfit(np.log(hs), np.log(res), 1)[0]
    assert order >= 1.9
    # NumPy and Torch paths are the same code
    assert abs(float(W) - L.complementary_energy(Q, LAYUP, THICK, LOAD)) <= 1e-12 * abs(float(W))


# --------------------------------------------------------------------------------------------------- L10
def test_l10a_tsai_wu_unit_index_at_the_fibre_strength() -> None:
    h = T_PLY
    lam = L.Laminate.from_angles(MAT, [0.0], h)
    r = first_ply_failure(lam, STRENGTH, "tsai_wu", N=[STRENGTH.Xt * h, 0, 0])
    assert abs(r.index.max() - 1.0) <= 1e-9
    assert abs(r.load_factor - 1.0) <= 1e-9


def test_l10b_load_factor_with_residual_stress_lands_on_the_surface() -> None:
    lam = L.Laminate.from_angles(MAT, [0, 90, 45], T_PLY, ALPHA, symmetric=True)
    N = np.array([40.0, 5.0, 10.0])
    for criterion in ("tsai_wu", "hashin"):
        r = first_ply_failure(lam, STRENGTH, criterion, N=N, dT0=-100.0)
        assert 0 < r.load_factor < math.inf
        s = lam.ply_stresses(lam.response(N * r.load_factor, dT0=-100.0)).stress_material
        point = s[r.critical_ply, r.critical_surface]
        mode = [m for m in failure_modes(criterion, STRENGTH) if m.name == r.critical_mode][0]
        assert abs(float(mode.value(point)) - 1.0) <= 1e-9
        every = [float(m.value(p)) for m in failure_modes(criterion, STRENGTH)
                 for p in s.reshape(-1, 3) if bool(m.active(p))]
        assert max(every) <= 1.0 + 1e-9                                   # nothing failed before λ*


def test_l10c_hashin_modes_have_unit_index_on_their_surfaces() -> None:
    a = STRENGTH
    uniaxial = {"fibre_tension": [a.Xt, 0, 0], "fibre_compression": [-a.Xc, 0, 0],
                "matrix_tension": [0, a.Yt, 0], "matrix_compression": [0, -a.Yc, 0]}
    for name, s in uniaxial.items():
        index, mode = ply_failure_index(np.array(s, float), "hashin", a)
        assert abs(float(index) - 1.0) <= 1e-12 and str(mode) == name
    modes = {m.name: m for m in failure_modes("hashin", a)}
    for s in ([300.0, -20.0, 30.0], [-100.0, -80.0, 45.0], [50.0, 30.0, -20.0]):
        index, mode = ply_failure_index(np.array(s), "hashin", a)
        on = np.array(s) / float(index)
        assert abs(float(modes[str(mode)].value(on)) - 1.0) <= 1e-12


def test_l10d_off_axis_tsai_wu_strength() -> None:
    for theta in (0.0, 10.0, 30.0, 45.0, 75.0, 90.0):
        lam = L.Laminate.from_angles(MAT, [theta], 1.0)
        r = first_ply_failure(lam, STRENGTH, "tsai_wu", N=[1.0, 0, 0])
        m, n = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        s1, s2, s12 = m * m, n * n, -m * n                                 # ply stresses of unit σ_x
        X, Xc, Y, Yc, S = 1500.0, 1200.0, 50.0, 200.0, 70.0
        f1, f2 = 1 / X - 1 / Xc, 1 / Y - 1 / Yc
        f11, f22, f66 = 1 / (X * Xc), 1 / (Y * Yc), 1 / S**2
        f12 = -0.5 * math.sqrt(f11 * f22)
        A = f11 * s1**2 + f22 * s2**2 + f66 * s12**2 + 2 * f12 * s1 * s2
        B = f1 * s1 + f2 * s2
        want = (-B + math.sqrt(B * B + 4 * A)) / (2 * A)
        assert abs(r.load_factor - want) <= 1e-10 * want
        assert abs(float(r.index.max()) - 1.0 / want) <= 1e-10 / want


def test_load_factor_edge_cases() -> None:
    modes = failure_modes("tsai_wu", STRENGTH)
    assert load_factor_to_failure([0, 0, 0], [0, 0, 0], modes)[0] == math.inf
    assert load_factor_to_failure([1.0, 0, 0], [0, 80.0, 0], modes)[0] == 0.0   # residual alone already failed


# --------------------------------------------------------------------------------------------------- L11
def test_l11_bad_input_is_refused() -> None:
    with pytest.raises(ValueError):
        L.Ply(MAT, 0.0, 0.0)
    with pytest.raises(ValueError):
        L.Ply(MAT, 0.0, -0.1)
    with pytest.raises(ValueError):
        L.Laminate(())
    with pytest.raises(TypeError):
        L.Ply(np.eye(6), 0.0, 0.1)
    with pytest.raises(ValueError):
        L.Laminate.from_angles(MAT, [0, 90], [0.1])
    with pytest.raises(ValueError):
        L.transformed_reduced_stiffness(np.stack([np.eye(3)] * 2), [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):                                       # thermal load without expansion data
        L.Laminate.from_angles(MAT, [0], 0.1).response(dT0=10.0)
    lam = L.Laminate.from_angles(MAT, [0], 0.1)
    no_s23 = OrthotropicStrength(Xt=1500.0, Xc=1200.0, Yt=50.0, Yc=200.0, S12=70.0)
    with pytest.raises(ValueError):
        first_ply_failure(lam, no_s23, "hashin", N=[1, 0, 0])
    with pytest.raises(ValueError):
        first_ply_failure(lam, STRENGTH, "max_stress", N=[1, 0, 0])
    with pytest.raises(ValueError):
        first_ply_failure(lam, [STRENGTH, STRENGTH], "tsai_wu", N=[1, 0, 0])
    with pytest.raises(TypeError):
        failure_modes("hashin", None)
