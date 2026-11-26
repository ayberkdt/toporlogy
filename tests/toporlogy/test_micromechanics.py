"""Micromechanics of isotropic phases (docs/foundations/prereg/MM_micromechanics.md; checks MM1-MM4, E7, MMI1-MMI2).

Independent references written in this file: the textbook two-phase HS form, Hashin's coated-sphere assemblage solved
from the Lamé solution (3 linear equations), Eshelby's sphere components (remembered, text not read), the scalar closed
forms of the dilute and Mori-Tanaka schemes, the exact layered (Backus) average, Voigt/Reuss of the phase matrices.
Checks labelled "cross-check" or "regression" call existing repository code and are not independent.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy.core import micromechanics as mm
from toporlogy.core.domain import Domain
from toporlogy.core.linear_elastic import symmetry_class
from toporlogy.design import interp
from toporlogy.fem.elements import D_plane_stress
from toporlogy.fem.homogenization import homogenize_elasticity

P = mm.IsoPhase


def rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def textbook(ref: P, oth: P, f_oth: float) -> tuple[float, float]:
    """HS form with ``ref`` as comparison medium (3-D): Hashin-Shtrikman 1963 two-phase result."""
    f1 = 1.0 - f_oth
    K = ref.K + f_oth / (1.0 / (oth.K - ref.K) + 3.0 * f1 / (3.0 * ref.K + 4.0 * ref.G))
    G = ref.G + f_oth / (1.0 / (oth.G - ref.G) + 6.0 * f1 * (ref.K + 2.0 * ref.G) / (5.0 * ref.G * (3.0 * ref.K + 4.0 * ref.G)))
    return K, G


def coated_sphere_K(core: P, coat: P, f_core: float) -> float:
    """Exact bulk modulus of Hashin's coated-sphere assemblage: hydrostatic displacement u = ε b on the outer radius
    b = 1, core u = A_c r, coating u = A_s r + B_s / r²; continuity of u and σ_rr (σ_rr = 3K A − 4 G B / r³) at r = a."""
    a = f_core ** (1.0 / 3.0)
    M = np.array([[a, -a, -1.0 / a ** 2],
                  [3.0 * core.K, -3.0 * coat.K, 4.0 * coat.G / a ** 3],
                  [0.0, 1.0, 1.0]])
    Ac, As, Bs = np.linalg.solve(M, np.array([0.0, 0.0, 1.0]))
    return (3.0 * coat.K * As - 4.0 * coat.G * Bs) / 3.0


STIFF, SOFT = P(10.0, 6.0), P(2.0, 1.0)            # ordered pair (both K and G)
X1, X2 = P(10.0, 1.0), P(1.0, 10.0)                # crossed pair: K1 > K2 but G1 < G2


# ---------------------------------------------------------------------------------------------------------- MM1
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("f", [0.0, 0.37, 1.0])
def test_mm1a_identical_phases_and_endpoints(dim: int, f: float) -> None:
    ph = P(7.0, 3.0)
    fr = [1.0 - f, f]
    for K, G in (mm.voigt(fr, [ph, ph]), mm.reuss(fr, [ph, ph])):
        assert abs(K - 7.0) <= 1e-13 * 7 and abs(G - 3.0) <= 1e-13 * 3
    b = mm.hs_bounds(fr, [ph, ph], dim)
    for v, ref in ((b.K_lower, 7.0), (b.K_upper, 7.0), (b.G_lower, 3.0), (b.G_upper, 3.0)):
        assert abs(v - ref) <= 1e-13 * ref
    for scheme in (mm.mori_tanaka_sphere, mm.dilute_sphere):
        r = scheme(f, ph, ph)
        assert abs(r.K - 7.0) <= 1e-13 * 7 and abs(r.G - 3.0) <= 1e-13 * 3


@pytest.mark.parametrize("dim", [2, 3])
def test_mm1a_pure_phase_endpoints(dim: int) -> None:
    for pair in ((STIFF, SOFT), (X1, X2)):
        a, b = pair
        for f, ph in ((0.0, a), (1.0, b)):
            fr = [1.0 - f, f]
            res = [mm.voigt(fr, pair), mm.reuss(fr, pair)]
            hs = mm.hs_bounds(fr, pair, dim)
            res += [(hs.K_lower, hs.G_lower), (hs.K_upper, hs.G_upper)]
            for K, G in res:
                assert abs(K - ph.K) <= 1e-13 * ph.K and abs(G - ph.G) <= 1e-13 * ph.G
    # mean-field schemes: f = 0 is the matrix; Mori-Tanaka at f = 1 is the inclusion
    r0 = mm.mori_tanaka_sphere(0.0, STIFF, SOFT)
    r1 = mm.mori_tanaka_sphere(1.0, STIFF, SOFT)
    assert abs(r0.K - STIFF.K) <= 1e-13 * STIFF.K and abs(r0.G - STIFF.G) <= 1e-13 * STIFF.G
    assert abs(r1.K - SOFT.K) <= 1e-12 * SOFT.K and abs(r1.G - SOFT.G) <= 1e-12 * SOFT.G
    d0 = mm.dilute_sphere(0.0, STIFF, SOFT)
    assert abs(d0.K - STIFF.K) <= 1e-13 * STIFF.K
    # wording correction (recorded in the prereg): the dilute scheme is NOT exact at f = 1 for distinct phases
    d1 = mm.dilute_sphere(1.0, STIFF, SOFT)
    assert abs(d1.K - SOFT.K) > 1e-3 * SOFT.K


@pytest.mark.parametrize("f", [0.1, 0.5, 0.9])
def test_mm1b_ordered_pair_equals_textbook_form(f: float) -> None:
    b = mm.hs_bounds([1.0 - f, f], [STIFF, SOFT], 3)
    Ku, Gu = textbook(STIFF, SOFT, f)                      # stiff phase as comparison medium: upper
    Kl, Gl = textbook(SOFT, STIFF, 1.0 - f)                # soft phase as comparison medium: lower (stiff fraction 1 - f)
    assert rel([b.K_upper, b.G_upper, b.K_lower, b.G_lower], [Ku, Gu, Kl, Gl]) <= 1e-12


@pytest.mark.parametrize("x", [0.05, 0.4, 0.8, 1.0])
def test_mm1c_void_limits(x: float) -> None:
    K, G = 5.0, 2.0
    solid = P(K, G)
    fr = [1.0 - x, x]
    Kv, Gv = mm.voigt(fr, [mm.VOID, solid])
    assert abs(Kv - x * K) <= 1e-14 * K and abs(Gv - x * G) <= 1e-14 * G
    Kr, Gr = mm.reuss(fr, [mm.VOID, solid])
    hs = mm.hs_bounds(fr, [mm.VOID, solid], 3)
    if x < 1.0:
        assert Kr == 0.0 and Gr == 0.0 and hs.K_lower == 0.0 and hs.G_lower == 0.0
    # cross-check with the existing void-limit closed forms of design/interp.py (3-D and 2-D)
    for dim in (3, 2):
        h = mm.hs_bounds(fr, [mm.VOID, solid], dim)
        Ke, Ge = interp.hashin_shtrikman_bounds(x, K, G, dim)
        assert rel([h.K_upper, h.G_upper], [Ke, Ge]) <= 1e-12
    if x < 1.0:
        # independent: hollow sphere (Lamé), void core with volume fraction 1 - x
        assert abs(coated_sphere_K(mm.VOID, solid, 1.0 - x) - hs.K_upper) <= 1e-10 * K


def test_mm1d_bulk_bound_against_coated_spheres() -> None:
    for f_core in (0.15, 0.35, 0.7):
        # ordered pair: stiff coating attains the upper bound, soft coating the lower bound (core = other phase)
        up = mm.hs_bounds([f_core, 1.0 - f_core], [SOFT, STIFF], 3)
        lo = mm.hs_bounds([1.0 - f_core, f_core], [SOFT, STIFF], 3)
        assert abs(coated_sphere_K(SOFT, STIFF, f_core) - up.K_upper) <= 1e-10 * STIFF.K
        assert abs(coated_sphere_K(STIFF, SOFT, f_core) - lo.K_lower) <= 1e-10 * STIFF.K
    # crossed pair: both realisable coating orders lie inside the bounds
    for f1 in (0.2, 0.5, 0.8):
        b = mm.hs_bounds([f1, 1.0 - f1], [X1, X2], 3)
        for k in (coated_sphere_K(X2, X1, 1.0 - f1), coated_sphere_K(X1, X2, f1)):
            assert b.K_lower - 1e-12 * X1.K <= k <= b.K_upper + 1e-12 * X1.K
        assert b.K_lower <= b.K_upper


# ---------------------------------------------------------------------------------------------------------- MM2
def _eshelby_components(nu: float) -> np.ndarray:
    d = np.eye(3)
    S4 = ((5 * nu - 1) * np.einsum("ij,kl->ijkl", d, d)
          + (4 - 5 * nu) * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))) / (15 * (1 - nu))
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    basis = []
    for i, j in pairs:
        E = np.zeros((3, 3))
        E[i, j] += 1.0
        E[j, i] += 1.0
        basis.append(E / np.linalg.norm(E))
    return np.array([[np.einsum("ij,ijkl,kl->", Ea, S4, Eb) for Eb in basis] for Ea in basis])


@pytest.mark.parametrize("nu", [-0.4, 0.0, 0.2, 0.3, 0.45])
def test_mm2a_eshelby_sphere(nu: float) -> None:
    S = mm.eshelby_sphere(nu)
    assert np.abs(S - _eshelby_components(nu)).max() <= 1e-14
    G = 1.0
    K = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))
    e = np.array([1, 1, 1, 0, 0, 0]) / np.sqrt(3.0)
    assert abs(e @ S @ e - 3 * K / (3 * K + 4 * G)) <= 1e-14


def _closed(f: float, m: P, i: P, scheme: str) -> tuple[float, float]:
    Ks = 4.0 * m.G / 3.0
    zeta = m.G * (9 * m.K + 8 * m.G) / (6 * (m.K + 2 * m.G))
    aK, aG = (i.K - m.K) / (m.K + Ks), (i.G - m.G) / (m.G + zeta)
    c = 1.0 if scheme == "dilute" else 1.0 - f
    return m.K + f * (i.K - m.K) / (1 + c * aK), m.G + f * (i.G - m.G) / (1 + c * aG)


def _from_E_nu(E: float, nu: float) -> P:
    return P(E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu)))


PAIRS = [(_from_E_nu(70.0, 0.3), _from_E_nu(3.0, 0.4)), (_from_E_nu(70.0, 0.3), mm.VOID),
         (_from_E_nu(1.0, 0.2), _from_E_nu(1e3, 0.35)), (_from_E_nu(1.0, 0.3), _from_E_nu(1e-3, 0.3)),
         (_from_E_nu(5.0, 0.1), _from_E_nu(5.5, 0.1))]


@pytest.mark.parametrize("pair", range(len(PAIRS)))
@pytest.mark.parametrize("scheme", ["dilute", "mori_tanaka"])
def test_mm2b_tensor_schemes_equal_closed_forms(pair: int, scheme: str) -> None:
    m, i = PAIRS[pair]
    f = 0.15
    r = (mm.dilute_sphere if scheme == "dilute" else mm.mori_tanaka_sphere)(f, m, i)
    K, G = _closed(f, m, i, scheme)
    assert rel([r.K, r.G], [K, G]) <= 1e-12
    assert r.isotropy_error <= 1e-13
    assert symmetry_class(r.C) == "isotropic"
    assert np.abs(r.C - r.C.T).max() <= 1e-12 * np.abs(r.C).max()
    assert np.linalg.eigvalsh(r.C).min() > 0


@pytest.mark.parametrize("pair", [0, 1, 3])
def test_mm2c_mori_tanaka_minus_dilute(pair: int) -> None:
    m, i = PAIRS[pair]
    ratios = {}
    for f in (1e-3, 5e-4):
        d, t = mm.dilute_sphere(f, m, i), mm.mori_tanaka_sphere(f, m, i)
        Ks, zeta = 4 * m.G / 3, m.G * (9 * m.K + 8 * m.G) / (6 * (m.K + 2 * m.G))
        for got, a in (((t.K - m.K) / (d.K - m.K) - 1, (i.K - m.K) / (m.K + Ks)),
                       ((t.G - m.G) / (d.G - m.G) - 1, (i.G - m.G) / (m.G + zeta))):
            pred = f * a / (1 + (1 - f) * a)
            assert abs(got - pred) <= 1e-9 * abs(pred) + 1e-13
        ratios[f] = (t.K - m.K) / (d.K - m.K) - 1
    assert abs(ratios[1e-3] / ratios[5e-4] - 2.0) <= 0.01                   # O(f)


@pytest.mark.parametrize("f", [0.1, 0.3, 0.6])
def test_mm2d_mori_tanaka_is_the_hs_form_with_the_matrix_as_comparison(f: float) -> None:
    stiff, soft = _from_E_nu(70.0, 0.3), _from_E_nu(3.0, 0.3)
    # stiff matrix, soft spheres: the HS UPPER bound (the brief's wording said lower)
    t = mm.mori_tanaka_sphere(f, stiff, soft)
    b = mm.hs_bounds([1.0 - f, f], [stiff, soft], 3)
    Kt, Gt = textbook(stiff, soft, f)
    assert rel([t.K, t.G], [b.K_upper, b.G_upper]) <= 1e-12 and rel([t.K, t.G], [Kt, Gt]) <= 1e-12
    # soft matrix, stiff spheres: the HS lower bound
    t2 = mm.mori_tanaka_sphere(f, soft, stiff)
    b2 = mm.hs_bounds([1.0 - f, f], [soft, stiff], 3)
    assert rel([t2.K, t2.G], [b2.K_lower, b2.G_lower]) <= 1e-12
    # the wording "equals the lower bound" is false for the stiff-matrix pair
    if f == 0.3:
        assert abs(t.K - b.K_lower) > 1e-2 * b.K_lower and abs(t.G - b.G_lower) > 1e-2 * b.G_lower


@pytest.mark.parametrize("ratio", [1e-9, 1e-3, 1e3])
def test_mm2e_mori_tanaka_inside_hs_for_contrast(ratio: float) -> None:
    m, i = _from_E_nu(1.0, 0.3), _from_E_nu(ratio, 0.3)
    f = 0.3
    t = mm.mori_tanaka_sphere(f, m, i)
    b = mm.hs_bounds([1.0 - f, f], [m, i], 3)
    for v, lo, up in ((t.K, b.K_lower, b.K_upper), (t.G, b.G_lower, b.G_upper)):
        assert lo - 1e-12 * up <= v <= up + 1e-12 * up


# ---------------------------------------------------------------------------------------------------------- MM3
def test_mm3_fraction_conversion() -> None:
    v, rho = np.array([0.6, 0.3, 0.1]), np.array([1800.0, 1200.0, 0.0])
    assert abs(mm.mixture_density(v, rho) - 1440.0) <= 1e-11
    w = mm.volume_to_mass_fractions(v, rho)
    assert np.abs(w - [0.75, 0.25, 0.0]).max() <= 1e-14
    v2 = mm.mass_to_volume_fractions([0.75, 0.25], [1800.0, 1200.0])
    assert np.abs(v2 - [2 / 3, 1 / 3]).max() <= 1e-14
    rng = np.random.default_rng(3)
    for _ in range(20):
        vv, rr = rng.dirichlet(np.ones(4)), rng.uniform(100, 8000, 4)
        assert np.abs(mm.mass_to_volume_fractions(mm.volume_to_mass_fractions(vv, rr), rr) - vv).max() <= 1e-14
    for bad in (lambda: mm.mixture_density([0.5, 0.6], [1.0, 2.0]), lambda: mm.mixture_density([-0.1, 1.1], [1.0, 2.0]),
                lambda: mm.mixture_density([np.nan, 1.0], [1.0, 2.0]), lambda: mm.mixture_density([0.5, 0.5], [1.0]),
                lambda: mm.mixture_density([0.5, 0.5], [1.0, -2.0]),
                lambda: mm.mass_to_volume_fractions([0.5, 0.5], [1.0, 0.0]),
                lambda: mm.volume_to_mass_fractions([0.0, 1.0], [1.0, 0.0])):
        with pytest.raises(ValueError):
            bad()


# ---------------------------------------------------------------------------------------------------------- MM4
NU = 0.3


def _backus_y(Cs: list[np.ndarray], fr: list[float]) -> np.ndarray:
    """Exact average of layers normal to y (2-D): ε_xx, σ_yy, σ_xy continuous."""
    Tn, N = [0], [1, 2]
    inv = np.linalg.inv
    avg = lambda g: sum(w * g(C) for C, w in zip(Cs, fr, strict=True))              # noqa: E731
    CNN = inv(avg(lambda C: inv(C[np.ix_(N, N)])))
    A = avg(lambda C: C[np.ix_(Tn, N)] @ inv(C[np.ix_(N, N)]))
    CTN = A @ CNN
    CTT = avg(lambda C: C[np.ix_(Tn, Tn)] - C[np.ix_(Tn, N)] @ inv(C[np.ix_(N, N)]) @ C[np.ix_(N, Tn)]) + A @ CNN @ A.T
    out = np.zeros((3, 3))
    out[np.ix_(N, N)], out[np.ix_(Tn, N)], out[np.ix_(N, Tn)], out[np.ix_(Tn, Tn)] = CNN, CTN, CTN.T, CTT
    return out


def _loewner_bounds(C: np.ndarray, E: np.ndarray) -> tuple[float, float]:
    """(min eig of C_V − C, min eig of C − C_R) for a cell of plane-stress phases with moduli E (D at E = 1)."""
    D = D_plane_stress(NU)
    CV, CR = E.mean() * D, D / np.mean(1.0 / E)
    s = np.abs(C).max()
    return np.linalg.eigvalsh(CV - C).min() / s, np.linalg.eigvalsh(C - CR).min() / s


def test_mm4a_mm4b_layered_cell() -> None:
    d = Domain((4, 12))
    idx = np.indices(d.shape)[1]
    vals = np.where(idx < 3, 1.0, np.where(idx < 8, 7.0, 3.0))
    E = vals.ravel()[d.active_elements]
    D = D_plane_stress(NU)
    ref = _backus_y([1.0 * D, 7.0 * D, 3.0 * D], [3 / 12, 5 / 12, 4 / 12])
    C = homogenize_elasticity(d, E, NU).C
    assert rel(C, ref) <= 1e-9
    lo, hi = _loewner_bounds(C, E)
    assert lo >= -1e-10 and hi >= -1e-10


def _random_cell(seed: int) -> tuple[Domain, np.ndarray]:
    d = Domain((24, 24))
    E = np.full(d.n_active, 1.0)
    E[np.random.default_rng(seed).permutation(d.n_active)[: d.n_active // 2]] = 10.0
    return d, E


@pytest.mark.parametrize("seed", range(5))
def test_mm4b_mm4c_random_two_phase_cell(seed: int) -> None:
    d, E = _random_cell(seed)
    assert E.mean() == 5.5
    C = homogenize_elasticity(d, E, NU).C
    lo, hi = _loewner_bounds(C, E)
    assert lo >= -1e-10 and hi >= -1e-10                                  # rigorous: Voigt / Reuss
    # measured, not a theorem (finite sample): isotropic parts against the 2-D HS bounds of the phases
    kappa_c = (C[0, 0] + C[1, 1] + 2 * C[0, 1]) / 4
    mu_c = (C[0, 0] + C[1, 1] - 2 * C[0, 1] + 4 * C[2, 2]) / 8
    ph = [P(e / (2 * (1 - NU)), e / (2 * (1 + NU))) for e in (1.0, 10.0)]
    b = mm.hs_bounds([0.5, 0.5], ph, 2)
    assert b.K_lower <= kappa_c <= b.K_upper
    assert b.G_lower <= mu_c <= b.G_upper


# ---------------------------------------------------------------------------------------------------------- E7
def test_e7_bound_ordering_and_rejection() -> None:
    rng = np.random.default_rng(7)
    crossed = 0
    for _ in range(200):
        n, dim = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        K, G = 10 ** rng.uniform(-3, 3, n), 10 ** rng.uniform(-3, 3, n)
        f = rng.dirichlet(np.ones(n))
        phases = [P(k, g) for k, g in zip(K, G, strict=True)]
        crossed += int(np.argmax(K) != np.argmax(G))
        Kv, Gv = mm.voigt(f, phases)
        Kr, Gr = mm.reuss(f, phases)
        b = mm.hs_bounds(f, phases, dim)
        for lo, a, bb, hi in ((Kr, b.K_lower, b.K_upper, Kv), (Gr, b.G_lower, b.G_upper, Gv)):
            tol = 1e-12 * hi
            assert lo - tol <= a <= bb + tol and bb <= hi + tol
    assert crossed >= 20
    for bad in ([0.5, 0.6], [-0.1, 1.1], [1.2, -0.2], [np.nan, 1.0], [1.0], [0.3, 0.3, 0.4]):
        with pytest.raises(ValueError):
            mm.hs_bounds(bad, [STIFF, SOFT])
    for args in ((-1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (np.nan, 1.0), (np.inf, 1.0)):
        with pytest.raises(ValueError):
            P(*args)
    with pytest.raises(ValueError):
        mm.mori_tanaka_sphere(1.2, STIFF, SOFT)
    with pytest.raises(ValueError):
        mm.mori_tanaka_sphere(0.2, mm.VOID, SOFT)


# ---------------------------------------------------------------------------------------------------------- MMI1
NU2 = [0.0, 0.1, 0.3, 1 / 3, 0.45]
NU3 = [0.0, 0.2, 0.3, 0.45]


def _closed_p(nu: float, dim: int) -> float:
    if dim == 2:
        return max(2 / (1 - nu), 4 / (1 + nu))
    return max(3 * (1 - nu) / (2 * (1 - 2 * nu)), 15 * (1 - nu) / (7 - 5 * nu))


@pytest.mark.parametrize(("dim", "nu"), [(2, v) for v in NU2] + [(3, v) for v in NU3])
def test_mmi1_simp_and_ramp_admissibility(dim: int, nu: float) -> None:
    p, q = mm.required_interpolation_parameters(nu, dim)
    assert abs(p - _closed_p(nu, dim)) <= 1e-12 * p and abs(q - (p - 1)) <= 1e-12
    assert abs(p - interp.simp_admissible_penal(nu, dim)) <= 1e-12 * p        # cross-check with design/interp.py
    assert mm.interpolation_admissibility("simp", p + 0.02, nu, dim).admissible
    assert mm.interpolation_admissibility("ramp", q + 0.02, nu, dim).admissible
    assert mm.interpolation_admissibility("simp", p, nu, dim).margin_K >= -1e-12
    low = mm.interpolation_admissibility("simp", p - 0.05, nu, dim)
    assert not low.admissible and low.x_worst > 0.9
    assert not mm.interpolation_admissibility("ramp", q - 0.05, nu, dim).admissible
    if dim == 2:
        assert p >= 3.0 - 1e-12 and (abs(p - 3.0) <= 1e-12) == (abs(nu - 1 / 3) < 1e-12)


def test_mmi1_regression_against_the_repository_laws() -> None:
    d = Domain((3, 3))
    x = torch.tensor(np.linspace(0.05, 1.0, d.n_active), dtype=torch.float64).reshape(d.shape)
    assert torch.allclose(interp.SIMP(d, E0=1.0, penal=3.2, emin_ratio=0.0)(x), x ** 3.2, rtol=1e-14, atol=0)
    assert torch.allclose(interp.RAMP(d, E0=1.0, q=5.0, emin_ratio=0.0)(x), x / (1 + 5.0 * (1 - x)), rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------------------------------------- MMI2
@pytest.mark.parametrize("pair", [(P(5.0, 3.0), P(40.0, 9.0)), (X1, X2)])
@pytest.mark.parametrize("scheme", mm.SCHEMES)
def test_mmi2_interpolation_endpoints_positivity_derivatives(pair: tuple[P, P], scheme: str) -> None:
    a, b = pair
    x = np.linspace(0.0, 1.0, 101)
    r = mm.interpolate_moduli(x, a, b, scheme)
    assert abs(r.K[0] - a.K) <= 1e-14 * a.K and abs(r.G[0] - a.G) <= 1e-14 * a.G
    assert abs(r.K[-1] - b.K) <= 1e-14 * b.K and abs(r.G[-1] - b.G) <= 1e-14 * b.G
    assert np.all(r.K > 0) and np.all(r.G > 0) and np.all(r.E > 0) and np.all(r.nu > -1) and np.all(r.nu < 0.5)
    h = 1e-6
    xi = np.linspace(0.05, 0.95, 19)
    rp, rm = mm.interpolate_moduli(xi + h, a, b, scheme), mm.interpolate_moduli(xi - h, a, b, scheme)
    ri = mm.interpolate_moduli(xi, a, b, scheme)
    for got, num, scale in ((ri.dK, (rp.K - rm.K) / (2 * h), max(a.K, b.K)), (ri.dG, (rp.G - rm.G) / (2 * h), max(a.G, b.G)),
                            (ri.dE, (rp.E - rm.E) / (2 * h), np.abs(ri.E).max())):
        assert np.abs(got - num).max() <= 1e-6 * scale
    # closed derivatives of Voigt and Reuss written here
    if scheme == "voigt":
        assert np.abs(ri.dK - (b.K - a.K)).max() <= 1e-13 * b.K
    if scheme == "reuss":
        assert np.abs(ri.dK - ri.K ** 2 * (1 / a.K - 1 / b.K)).max() <= 1e-12 * max(a.K, b.K)
    # ordering Reuss <= HS- <= HS+ <= Voigt
    s = {k: mm.interpolate_moduli(x, a, b, k) for k in mm.SCHEMES}
    for q in ("K", "G"):
        v = {k: getattr(s[k], q) for k in mm.SCHEMES}
        tol = 1e-12 * v["voigt"].max()
        assert np.all(v["reuss"] <= v["hs_lower"] + tol) and np.all(v["hs_lower"] <= v["hs_upper"] + tol)
        assert np.all(v["hs_upper"] <= v["voigt"] + tol)


def test_mmi2_ordered_pair_against_textbook_and_legacy_simp() -> None:
    a, b = P(5.0, 3.0), P(40.0, 9.0)
    x = np.linspace(0.0, 1.0, 21)
    up, lo = mm.interpolate_moduli(x, a, b, "hs_upper"), mm.interpolate_moduli(x, a, b, "hs_lower")
    for k, xx in enumerate(x[1:-1], start=1):
        Ku, Gu = textbook(b, a, 1.0 - xx)                  # stiff phase b as reference, fraction of a = 1 - x
        Kl, Gl = textbook(a, b, xx)                        # soft phase a as reference
        assert rel([up.K[k], up.G[k], lo.K[k], lo.G[k]], [Ku, Gu, Kl, Gl]) <= 1e-12
    # legacy scalar path: equal ν, phase a = r × b: Voigt E(x) is SIMP with p = 1 and emin_ratio r
    r = 0.01
    solid = _from_E_nu(2.0, 0.3)
    soft = P(r * solid.K, r * solid.G)
    v = mm.interpolate_moduli(x, soft, solid, "voigt")
    d = Domain((3, 3))
    xs = torch.tensor(np.linspace(0.0, 1.0, d.n_active), dtype=torch.float64).reshape(d.shape)
    E_simp = interp.SIMP(d, E0=2.0, penal=1.0, emin_ratio=r)(xs).numpy()
    E_mm = mm.interpolate_moduli(xs.numpy(), soft, solid, "voigt").E
    assert np.abs(E_simp - E_mm).max() <= 1e-13 * 2.0
    assert np.abs(v.nu - 0.3).max() <= 1e-13
    for bad in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            mm.interpolate_moduli(bad, a, b)
    with pytest.raises(ValueError):
        mm.interpolate_moduli(0.5, mm.VOID, b)
