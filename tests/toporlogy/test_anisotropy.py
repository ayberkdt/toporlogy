"""Directional moduli, Poisson extrema, isotropic sweep, sensitivities and SIMP/HS admissibility
(``toporlogy.core.anisotropy``, ``toporlogy.design.interp``): records P1-P11 of docs/olcumler/tensor_elastisite.md
(thresholds pre-registered 2026-11-26 00:29 +0300). P6b, P8 and P10b are measurements without a threshold; they are
produced by ``docs/olcumler/tensor_elastisite.py``."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from toporlogy.core import anisotropy as A
from toporlogy.core import linear_elastic as L
from toporlogy.core import material as M
from toporlogy.core import tensor as T
from toporlogy.design.interp import (hashin_shtrikman_bounds, hs_admissible_exponents, simp_admissible_penal)

from .test_tensor import random_spd_voigt, rel

SWEEP = [-0.99, -0.9, -0.5, 0.0, 0.2, 0.3, 0.45, 0.49, 0.499, 0.4999]
ORTHO = dict(E1=40.0, E2=9.0, E3=9.0, G12=4.0, G13=4.0, G23=3.2, nu12=0.28, nu13=0.28, nu23=0.4)


def _frame(n: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random m ⊥ n for every n."""
    P = A.plane_basis(n)
    a = rng.uniform(0, 2 * math.pi, len(n))
    return n, P[..., 0] * np.cos(a)[:, None] + P[..., 1] * np.sin(a)[:, None]


# ============================================================================================== P1 isotropic limit
def test_p1_isotropic_directional_constants() -> None:
    E, nu = 2.0, 0.27
    S = L.isotropic_compliance(E, nu)
    m = L.isotropic_moduli(E=E, nu=nu)
    n, mm = _frame(L.fibonacci_sphere(2000), np.random.default_rng(51))
    assert np.abs(A.young_modulus(S, n) / E - 1).max() <= 1e-12
    assert np.abs(A.poisson_ratio(S, n, mm) / nu - 1).max() <= 1e-12
    assert np.abs(A.shear_modulus(S, n, mm) / m.G - 1).max() <= 1e-12
    assert np.abs(A.linear_compressibility(S, n) * 3 * m.K - 1).max() <= 1e-12
    assert abs(A.volumetric_compressibility(S) * m.K - 1) <= 1e-12


# ============================================================================================== P2 independent route
def test_p2_directional_formulas_against_rotated_voigt() -> None:
    rng = np.random.default_rng(52)
    C, _ = random_spd_voigt(50, rng)
    for c in C:
        S = np.linalg.inv(c)
        S = 0.5 * (S + S.T)
        for _ in range(20):
            n = rng.standard_normal(3)
            n /= np.linalg.norm(n)
            _, m = _frame(n[None], rng)
            m = m[0]
            R = np.column_stack([n, m, np.cross(n, m)])         # local axis 1 = n, 2 = m
            Sp = np.linalg.inv(M.rotate_stiffness(c, R.T))      # compliance in the (n, m, n × m) frame
            assert abs(A.young_modulus(S, n) * Sp[0, 0] - 1) <= 1e-11
            assert abs(A.poisson_ratio(S, n, m) - (-Sp[1, 0] / Sp[0, 0])) <= 1e-11 * max(1.0, abs(Sp[1, 0] / Sp[0, 0]))
            assert abs(A.shear_modulus(S, n, m) * Sp[5, 5] - 1) <= 1e-11
            beta = Sp[0, 0] + Sp[0, 1] + Sp[0, 2]
            assert abs(A.linear_compressibility(S, n) - beta) <= 1e-11 * np.abs(Sp[0, :3]).sum()


# ============================================================================================== P3 frame invariance
@pytest.mark.xfail(strict=True, reason="pre-registered P3 (1e-12 relative) FAILED on the first run: 1.21e-12 for the "
                   "linear compressibility. The check rotates C and inverts it (S_q = inv(rotate(C))), which scales "
                   "rounding by the Mandel condition number (up to 1e4 here); the tolerance did not provide for that. "
                   "Post hoc: test_p3_rotation_invariance_rotated_compliance rotates S directly")
def test_p3_rotation_invariance() -> None:
    rng = np.random.default_rng(53)
    C, _ = random_spd_voigt(20, rng)
    n, m = _frame(L.fibonacci_sphere(100), rng)
    for c, Q in zip(C, T.random_rotations(20, rng), strict=True):
        S = np.linalg.inv(c)
        Sq = np.linalg.inv(M.rotate_stiffness(c, Q))
        nq, mq = n @ Q.T, m @ Q.T
        for f in (A.young_modulus, A.linear_compressibility):
            assert rel(f(Sq, nq), f(S, n)) <= 1e-12
        for g in (A.poisson_ratio, A.shear_modulus):
            assert rel(g(Sq, nq, mq), g(S, n, m)) <= 1e-12


def test_p3_rotation_invariance_rotated_compliance() -> None:
    """Post hoc (after the P3 failure, same 1e-12): the compliance is rotated as a tensor, no inversion."""
    rng = np.random.default_rng(53)
    C, _ = random_spd_voigt(20, rng)
    n, m = _frame(L.fibonacci_sphere(100), rng)
    for c, Q in zip(C, T.random_rotations(20, rng), strict=True):
        S = np.linalg.inv(c)
        S = 0.5 * (S + S.T)
        Sq = T.tensor_to_compliance(T.rotate4(T.compliance_to_tensor(S), Q))
        Sq = 0.5 * (Sq + Sq.T)
        nq, mq = n @ Q.T, m @ Q.T
        for f in (A.young_modulus, A.linear_compressibility):
            assert rel(f(Sq, nq), f(S, n)) <= 1e-12
        for g in (A.poisson_ratio, A.shear_modulus):
            assert rel(g(Sq, nq, mq), g(S, n, m)) <= 1e-12


# ============================================================================================== P4 extremum over m
def test_p4_poisson_extrema_over_m() -> None:
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(54)
    mats = [L.orthotropic_compliance(**ORTHO), np.linalg.inv(L.cubic_stiffness(1.7, 1.2, 0.75))]
    mats += [np.linalg.inv(c) for c in random_spd_voigt(8, rng)[0]]
    phi = np.linspace(0, 2 * math.pi, 3600, endpoint=False)
    for S in mats:
        S = 0.5 * (S + S.T)
        for n in L.fibonacci_sphere(7):
            ex = A.poisson_extrema(S, n)
            P = A.plane_basis(n)
            ms = P[:, 0][None, :] * np.cos(phi)[:, None] + P[:, 1][None, :] * np.sin(phi)[:, None]
            nu = A.poisson_ratio(S, np.broadcast_to(n, ms.shape), ms)
            amp = max(abs(ex["nu_max"]), abs(ex["nu_min"]), 1e-300)
            assert nu.max() <= ex["nu_max"] + 1e-12 * amp and nu.min() >= ex["nu_min"] - 1e-12 * amp
            assert abs(nu.max() - ex["nu_max"]) <= 1e-5 * amp and abs(nu.min() - ex["nu_min"]) <= 1e-5 * amp
            for sign, key in ((-1.0, "nu_max"), (1.0, "nu_min")):
                k = int(np.argmax(-sign * nu))
                f = lambda t: sign * float(A.poisson_ratio(S, n, P[:, 0] * math.cos(t) + P[:, 1] * math.sin(t)))
                r = minimize_scalar(f, bounds=(phi[k] - 0.01, phi[k] + 0.01), method="bounded",
                                    options={"xatol": 1e-12})
                assert abs(sign * r.fun - ex[key]) <= 1e-10 * amp


# ============================================================================================== P5 extremum over n
@pytest.mark.parametrize("which", ["ortho", "cubic", "random0", "random1"])
def test_p5_directional_extrema_sampling_convergence(which: str) -> None:
    rng = np.random.default_rng(55)
    rnd = random_spd_voigt(2, rng)[0]
    S = {"ortho": L.orthotropic_compliance(**ORTHO), "cubic": np.linalg.inv(L.cubic_stiffness(1.7, 1.2, 0.75)),
         "random0": np.linalg.inv(rnd[0]), "random1": np.linalg.inv(rnd[1])}[which]
    S = 0.5 * (S + S.T)
    for q in A.QUANTITIES:
        a = A.directional_extrema(S, q, 2000)
        b = A.directional_extrema(S, q, 8000)
        assert a["label"] == b["label"] == "refined"
        assert a["max"] >= a["sampled_max"] and a["min"] <= a["sampled_min"]
        for k in ("min", "max"):
            assert abs(a[k] - b[k]) <= 1e-8 * max(abs(a["min"]), abs(a["max"])), (q, k, a[k], b[k])


# ============================================================================================== P6 isotropic sweep
@pytest.mark.parametrize("fixed", ["E", "G"])
def test_p6_isotropic_sweep_condition_number(fixed: str) -> None:
    rows = A.isotropic_sweep(SWEEP, fixed)
    assert [r["nu"] for r in rows] == SWEEP
    for r in rows:
        assert abs(r["cond_C"] / r["cond_closed_form"] - 1) <= 1e-10
        assert abs(r["cond_S"] / r["cond_closed_form"] - 1) <= 1e-10
        assert (r["E"] if fixed == "E" else r["G"]) == 1.0


# ============================================================================================== P7 unbounded ν
@pytest.mark.parametrize("k", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_p7_unbounded_poisson_spd_family(k: float, sign: float) -> None:
    a = 0.5 * sign
    S = L.orthotropic_compliance(E1=k * k, E2=1.0, E3=1.0, G12=1.0, G13=1.0, G23=1.0, nu12=a * k, nu13=a * k,
                                 nu23=0.0)
    assert (a * k) ** 2 < k * k / 2                          # analytic PD condition ν12² < E1/2 (E2 = E3 = 1)
    L.check_positive_definite(S, "compliance", "S")
    ec = L.engineering_constants(S)
    assert abs(ec["nu12"] - a * k) <= 1e-13 * abs(a * k)
    ex = A.poisson_extrema(S, np.array([1.0, 0.0, 0.0]))
    assert ex["nu_max" if sign > 0 else "nu_min"] == pytest.approx(a * k, rel=1e-12)
    bad = L.orthotropic_compliance(E1=k * k, E2=1.0, E3=1.0, G12=1.0, G13=1.0, G23=1.0, nu12=0.75 * k,
                                   nu13=0.75 * k, nu23=0.0)       # a² = 0.5625 > 1/2: negative control
    with pytest.raises(L.NotPositiveDefiniteError):
        L.check_positive_definite(bad, "compliance", "S")


# ============================================================================================== P9 sensitivities
def test_p9_isotropic_and_compliance_derivatives_fd() -> None:
    for E, nu in ((1.0, 0.3), (2.5, -0.4), (0.7, 0.1)):
        dE, dn = A.isotropic_stiffness_derivatives(E, nu)
        hE, hn = 1e-6 * E, 1e-6
        fdE = (L.isotropic_stiffness(E + hE, nu) - L.isotropic_stiffness(E - hE, nu)) / (2 * hE)
        fdn = (L.isotropic_stiffness(E, nu + hn) - L.isotropic_stiffness(E, nu - hn)) / (2 * hn)
        assert rel(fdE, dE) <= 1e-7 and rel(fdn, dn) <= 1e-7
    rng = np.random.default_rng(56)
    for c in random_spd_voigt(20, rng, cond=1e2)[0]:
        d = rng.standard_normal((6, 6))
        d = 0.5 * (d + d.T)
        h = 1e-6 * np.abs(c).max() / np.abs(d).max()
        S = np.linalg.inv(c)
        fd = (np.linalg.inv(c + h * d) - np.linalg.inv(c - h * d)) / (2 * h)
        assert rel(fd, A.compliance_derivative(S, d)) <= 1e-7


def _rz(t: float) -> np.ndarray:
    return M.rotation_about_z(math.degrees(t))


def test_p9_young_angle_derivative_fd() -> None:
    C = M.Orthotropic(**ORTHO).stiffness()
    W = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 0]])
    for th in (0.1, 0.7, 1.3):
        for n in (np.array([1.0, 0, 0]), np.array([0.6, 0.8, 0.0]), np.array([0.0, 0.6, 0.8])):
            R = _rz(th)
            Ct = M.rotate_stiffness(C, R)
            dC = A.rotated_stiffness_derivative(C, R, W @ R)
            St = np.linalg.inv(Ct)
            an = A.young_modulus_derivative(St, n, A.compliance_derivative(St, dC))
            h = 1e-6

            def E_at(t: float) -> float:
                return float(A.young_modulus(np.linalg.inv(M.rotate_stiffness(C, _rz(t))), n))
            fd = (E_at(th + h) - E_at(th - h)) / (2 * h)
            assert abs(fd - an) <= 1e-6 * max(abs(an), 1e-3 * E_at(th)), (th, n, fd, an)


def _taylor_slopes(f, x0: float, d: float, df: np.ndarray) -> list[float]:
    h0 = 1e-2
    r = []
    for k in range(7):
        h = h0 / 2 ** k
        r.append(float(np.abs(f(x0 + h * d) - f(x0) - h * d * df).max()))
    return [math.log2(r[k] / r[k + 1]) for k in range(6)]


def test_p9b_taylor_remainder_second_order() -> None:
    _, dn = A.isotropic_stiffness_derivatives(1.0, 0.3)
    s = _taylor_slopes(lambda v: L.isotropic_stiffness(1.0, v), 0.3, 1.0, dn)
    assert min(s) >= 1.9, s
    C = M.Orthotropic(**ORTHO).stiffness()
    W = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 0]])
    dC = A.rotated_stiffness_derivative(C, _rz(0.4), W @ _rz(0.4))
    s2 = _taylor_slopes(lambda t: M.rotate_stiffness(C, _rz(t)), 0.4, 1.0, dC)
    assert min(s2) >= 1.9, s2


@pytest.mark.parametrize("nu", [0.4999, -0.999])
def test_p9c_derivatives_near_the_admissible_boundary(nu: float) -> None:
    _, dn = A.isotropic_stiffness_derivatives(1.0, nu)
    h = 1e-6 * min(1.0, 0.5 - nu, nu + 1.0) * 10            # stays inside (-1, 1/2)
    fd = (L.isotropic_stiffness(1.0, nu + h) - L.isotropic_stiffness(1.0, nu - h)) / (2 * h)
    kappa = float(np.linalg.cond(T.stiffness_to_mandel(L.isotropic_stiffness(1.0, nu))))
    assert rel(fd, dn) <= 1e-6 * kappa, (rel(fd, dn), kappa)


# ============================================================================================== P10 Cholesky chart
@pytest.mark.xfail(strict=True, reason="pre-registered P10 FAILED on the first run. (a) 'Cholesky passes': 37 of the "
                   "1000 theta fail np.linalg.cholesky of the floating-point Mandel matrix L L^T (all 37 from the "
                   "N(0, 10) group, condition numbers 7.6e16 .. 1e35): SPD in exact arithmetic is not SPD in float64 "
                   "once cond > 1/eps. (b) roundtrip <= 1e-12: worst 3.1e-8 (N(0, 2) group, cond up to 2.4e11), "
                   "98 skipped as numerically singular; the tolerance did not scale with cond. Post hoc: "
                   "test_p10_cholesky_parametrisation_well_conditioned")
def test_p10_cholesky_parametrisation() -> None:
    rng = np.random.default_rng(57)
    th = np.concatenate([np.column_stack([rng.uniform(-2, 2, (900, 6)), rng.normal(0, 2, (900, 15))]),
                         rng.normal(0, 10, (100, 21))])
    worst = 0.0
    for t in th:
        C = A.stiffness_from_cholesky(t)
        np.linalg.cholesky(T.stiffness_to_mandel(C))
        try:
            back = A.cholesky_parameters(C)
        except L.NotPositiveDefiniteError:                   # numerically singular by the 64 eps rule
            continue
        worst = max(worst, float(np.abs(back - t).max()))
    assert worst <= 1e-12, worst


def test_p10_cholesky_parametrisation_well_conditioned() -> None:
    """Post hoc (after the P10 failure): moderate theta (diag U(-1, 1), off-diagonal N(0, 0.5)); SPD and roundtrip
    within 1e-12 * cond(M_C)."""
    rng = np.random.default_rng(58)
    th = np.column_stack([rng.uniform(-1, 1, (500, 6)), rng.normal(0, 0.5, (500, 15))])
    for t in th:
        C = A.stiffness_from_cholesky(t)
        k = L.check_positive_definite(C)["condition_number"]
        assert np.abs(A.cholesky_parameters(C) - t).max() <= 1e-12 * k


# ============================================================================================== P11 SIMP / HS
@pytest.mark.parametrize("nu", [-0.9, -0.5, 0.0, 0.2, 0.3, 0.45, 0.49])
def test_p11_general_exponents_equal_closed_forms(nu: float) -> None:
    pk, pg = hs_admissible_exponents(nu, 2, "stress")
    assert abs(pk / (2 / (1 - nu)) - 1) <= 1e-14 and abs(pg / (4 / (1 + nu)) - 1) <= 1e-14
    pk3, pg3 = hs_admissible_exponents(nu, 3)
    assert abs(pk3 / (3 * (1 - nu) / (2 * (1 - 2 * nu))) - 1) <= 1e-14
    assert abs(pg3 / (15 * (1 - nu) / (7 - 5 * nu)) - 1) <= 1e-14
    assert simp_admissible_penal(nu, 2) == max(pk, pg) and simp_admissible_penal(nu, 3) == max(pk3, pg3)


@pytest.mark.parametrize("case", [(2, "stress"), (2, "strain"), (3, "stress")])
@pytest.mark.parametrize("nu", [-0.5, 0.0, 0.3, 0.45])
def test_p11b_simp_below_hs_bounds(case: tuple[int, str], nu: float) -> None:
    dim, plane = case
    m = L.isotropic_moduli(E=1.0, nu=nu)
    K = m.K if dim == 3 else L.plane_bulk_modulus(1.0, nu, plane)
    x = torch.linspace(1e-4, 1.0, 10_000, dtype=torch.float64)
    Kb, Gb = hashin_shtrikman_bounds(x, K, m.G, dim)
    p = max(hs_admissible_exponents(nu, dim, plane))
    assert bool((x ** p * K <= Kb * (1 + 1e-12)).all()) and bool((x ** p * m.G <= Gb * (1 + 1e-12)).all())
    q = 0.95 * p
    assert bool((x ** q * K > Kb * (1 + 1e-12)).any()) or bool((x ** q * m.G > Gb * (1 + 1e-12)).any())


def test_p11c_plane_strain_exponent_differs() -> None:
    ps = max(hs_admissible_exponents(0.3, 2, "strain"))
    pst = max(hs_admissible_exponents(0.3, 2, "stress"))
    assert abs(ps - 3.5) <= 1e-14 * 3.5 and abs(pst - 40 / 13) <= 1e-14 * 40 / 13
    nu = 0.3
    nup = nu / (1 - nu)                                      # plane strain = plane-stress formula at ν/(1 − ν)
    assert abs(ps - max(2 / (1 - nup), 4 / (1 + nup))) <= 1e-14 * ps
