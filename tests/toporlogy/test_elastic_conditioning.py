"""Precision of anisotropic elasticity at extreme conditioning, acoustic tensor and wave speeds, Cholesky
parameterisation (risk R02): records EC1-EC8, AC1-AC5 and R02a-R02b of docs/foundations/prereg/E3_conditioning.md
(thresholds preregistered before this file was run). Every tensor is synthetic (seeded random orthogonal Q), not
material data. The reference is mpmath with 50 digits (100 for the R02 cases) on the **float64 inputs exactly**, written
in this file; no toporlogy function is called by a reference. The energy row's float side is the production Mandel
conversion plus a numpy product (there is no production energy function): a regression of that path, not an
independent check of it. ``measure_levels`` / ``r02_family`` also feed docs/foundations/scripts/e3_conditioning.py.
"""
from __future__ import annotations

import functools
import math
from typing import Any

import mpmath as mpm
import numpy as np
import pytest

from toporlogy.core import anisotropy as A
from toporlogy.core import linear_elastic as L
from toporlogy.core import tensor as T

U = float(np.finfo(float).eps) / 2                    # unit roundoff 2^-53
C_FACTOR = 100.0                                      # the preregistered c in "error <= c kappa u"
LEVELS = (1.0, 1e2, 1e6, 1e10, 1e12, 1e14)            # designed kappa
ASSERTED = (1.0, 1e2, 1e6, 1e10)
SEED = 20261127
_MV = np.array([1.0, 1.0, 1.0, math.sqrt(2), math.sqrt(2), math.sqrt(2)])      # Mandel factors (float)


# ============================================================================================== mp reference
def _mpf(x: float) -> Any:
    return mpm.mpf(float(x))


def _mp_mandel(C6: np.ndarray) -> Any:
    """M = diag(m) C diag(m) with m = (1, 1, 1, sqrt2, sqrt2, sqrt2) exactly, of the float entries of C6."""
    r2 = mpm.sqrt(2)
    m = [1, 1, 1, r2, r2, r2]
    return mpm.matrix([[m[i] * m[j] * _mpf(C6[i, j]) for j in range(6)] for i in range(6)])


def _mp_voigt(C6: np.ndarray) -> Any:
    return mpm.matrix([[_mpf(C6[i, j]) for j in range(6)] for i in range(6)])


def _mp_eigs(M: Any) -> tuple[Any, Any]:
    w = mpm.eigsy(M, eigvals_only=True)
    ws = [w[i] for i in range(w.rows)]
    return min(ws), max(ws)


def _mp_spd(M: Any) -> bool:
    try:
        mpm.cholesky(M)
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _avec(n: np.ndarray) -> Any:
    """a(n) = (n1^2, n2^2, n3^2, n2 n3, n1 n3, n1 n2) in mp from the float n (derivation in the prereg)."""
    x = [_mpf(v) for v in n]
    return mpm.matrix([x[0] * x[0], x[1] * x[1], x[2] * x[2], x[1] * x[2], x[0] * x[2], x[0] * x[1]])


def _quad(a: Any, S: Any, b: Any) -> Any:
    return (a.T * S * b)[0]


def _rel(x: float, ref: Any) -> float:
    return float(abs(_mpf(x) - ref) / abs(ref))


def _rel_nu(x: float, ref: Any) -> float:
    """Poisson-ratio error |x - ref| / max(|ref|, 1). Set-up correction after the first run (see the prereg record):
    at kappa = 1 (M = I, isotropic nu = 0 for every direction pair) the reference is 0 and a relative error is
    undefined; identical to the relative error wherever |ref| >= 1."""
    return float(abs(_mpf(x) - ref) / max(abs(ref), 1))


# ============================================================================================== synthetic data
def _unit_rows(rng: np.random.Generator, k: int) -> np.ndarray:
    v = rng.standard_normal((k, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@functools.cache
def level_data() -> dict[float, dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    out: dict[float, dict[str, np.ndarray]] = {}
    for kappa in LEVELS:
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        lam = np.geomspace(1.0, kappa, 6)
        M = (Q * lam) @ Q.T
        M = 0.5 * (M + M.T)
        n = _unit_rows(rng, 20)
        mm = rng.standard_normal((20, 3))
        mm = mm - np.sum(mm * n, axis=1, keepdims=True) * n
        mm = mm / np.linalg.norm(mm, axis=1, keepdims=True)
        out[kappa] = {"C6": T.mandel_to_stiffness(M), "eps": rng.standard_normal((20, 6)), "n": n, "m": mm}
    return out


def _float_compliance(C6: np.ndarray) -> np.ndarray:
    """Mirrors ElasticTensor.compliance_material (inverse, symmetrised) without the SPD gate that refuses 1e14."""
    S = np.linalg.inv(C6)
    return 0.5 * (S + S.T)


def _accepts(C6: np.ndarray) -> bool:
    try:
        L.check_positive_definite(C6)
    except L.NotPositiveDefiniteError:
        return False
    return True


# ============================================================================================== measurements
def measure_level(kappa_design: float) -> dict[str, Any]:
    d = level_data()[kappa_design]
    C6, eps, n, m = d["C6"], d["eps"], d["n"], d["m"]
    S_f = _float_compliance(C6)
    Mf = T.stiffness_to_mandel(C6)
    res: dict[str, Any] = {"kappa_design": kappa_design}
    with mpm.workdps(50):
        Cm = _mp_voigt(C6)
        Mm = _mp_mandel(C6)
        wmin, wmax = _mp_eigs(Mm)
        kappa = float(wmax / wmin)
        res.update(kappa=kappa, ku=kappa * U, bound=C_FACTOR * kappa * U, mp_spd=bool(wmin > 0))
        # soft strain: eigenvector of the smallest eigenvalue, converted to the Voigt engineering strain m * e
        E_, Q_ = mpm.eigsy(Mm)
        k0 = min(range(6), key=lambda i: E_[i])
        r2 = mpm.sqrt(2)
        mmp = [1, 1, 1, r2, r2, r2]
        eps_soft = np.array([float(mmp[i] * Q_[i, k0]) for i in range(6)])
        errs_rand, errs_soft = [], []
        for e_v in list(eps) + [eps_soft]:
            ev = mpm.matrix([_mpf(x) for x in e_v])
            W_ref = _quad(ev, Cm, ev) / 2
            e_m = e_v / _MV
            W_f = 0.5 * float(e_m @ Mf @ e_m)
            (errs_soft if e_v is eps_soft else errs_rand).append(_rel(W_f, W_ref))
        res["energy_random"], res["energy_soft"] = max(errs_rand), errs_soft[0]
        # compliance
        S_ref = Cm ** -1
        Mi = np.array([[1 / (_MV[i] * _MV[j]) for j in range(6)] for i in range(6)])
        dnorm = mpm.sqrt(sum((_mpf(S_f[i, j]) - S_ref[i, j]) ** 2 * _mpf(Mi[i, j]) ** 2
                             for i in range(6) for j in range(6)))
        snorm = mpm.sqrt(sum(S_ref[i, j] ** 2 * _mpf(Mi[i, j]) ** 2 for i in range(6) for j in range(6)))
        res["S_normwise"] = float(dnorm / snorm)
        res["S_componentwise_max"] = float(max(abs(_mpf(S_f[i, j]) - S_ref[i, j]) / abs(S_ref[i, j])
                                               for i in range(6) for j in range(6)))
        # directional Young and nu(n, m)
        E_f = A.young_modulus(S_f, n)
        nu_f = A.poisson_ratio(S_f, n, m)
        ey, en = [], []
        for k in range(len(n)):
            a, b = _avec(n[k]), _avec(m[k])
            ey.append(_rel(float(E_f[k]), 1 / _quad(a, S_ref, a)))
            en.append(_rel_nu(float(nu_f[k]), -_quad(b, S_ref, a) / _quad(a, S_ref, a)))
        res["young"], res["nu_directional"] = max(ey), max(en)
        # engineering constants and poisson_uniaxial
        eng = L.engineering_constants(S_f)
        e_eng, e_uni = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    ref = -S_ref[j, i] / S_ref[i, i]
                    e_eng.append(_rel_nu(float(eng[f"nu{i + 1}{j + 1}"]), ref))
                    e_uni.append(_rel_nu(float(L.poisson_uniaxial(C6, i, j)), ref))
        res["nu_engineering"], res["nu_uniaxial"] = max(e_eng), max(e_uni)
        # reciprocity
        smax = float(np.abs(S_f).max())
        rf = max(abs(eng[f"nu{i + 1}{j + 1}"] / eng[f"E{i + 1}"] - eng[f"nu{j + 1}{i + 1}"] / eng[f"E{j + 1}"])
                 for i in range(3) for j in range(i + 1, 3)) / smax
        smax_mp = max(abs(S_ref[i, j]) for i in range(6) for j in range(6))
        rm = max(abs(S_ref[j, i] - S_ref[i, j]) for i in range(3) for j in range(i + 1, 3)) / smax_mp
        res["reciprocity_float"], res["reciprocity_mp"] = float(rf), float(rm)
    res["accepted"] = _accepts(C6)
    res["verdict"] = L.forward_accuracy_established(C6)["verdict"]
    return res


@functools.cache
def measure_levels() -> dict[float, dict[str, Any]]:
    return {k: measure_level(k) for k in LEVELS}


_TRIL = np.tril_indices(6, -1)


def _chol_theta(r: float, rng: np.random.Generator) -> np.ndarray:
    diag = rng.permutation(r * np.linspace(-1.0, 1.0, 6))
    off = rng.standard_normal(15) * np.exp(diag[_TRIL[0]])
    return np.concatenate([diag, off])


def _mp_LLt(theta: np.ndarray) -> Any:
    Lm = mpm.zeros(6, 6)
    for i in range(6):
        Lm[i, i] = mpm.exp(_mpf(theta[i]))
    for k, (i, j) in enumerate(zip(*_TRIL)):
        Lm[i, j] = _mpf(theta[6 + k])
    return Lm * Lm.T


def measure_theta(theta: np.ndarray) -> dict[str, Any]:
    C = A.stiffness_from_cholesky(theta)
    Mf = T.stiffness_to_mandel(C)
    Mfs = 0.5 * (Mf + Mf.T)
    try:
        np.linalg.cholesky(Mfs)
        float_chol = True
    except np.linalg.LinAlgError:
        float_chol = False
    out: dict[str, Any] = {"float_cholesky_ok": float_chol, "accepted": _accepts(C),
                           "verdict": L.forward_accuracy_established(C)["verdict"]}
    with mpm.workdps(100):
        Mref = _mp_LLt(theta)
        wmin, wmax = _mp_eigs(Mref)
        out["kappa_exact"] = float(wmax / wmin)
        out["exact_spd"] = _mp_spd(Mref)
        Mfm = mpm.matrix([[_mpf(Mfs[i, j]) for j in range(6)] for i in range(6)])
        out["float_matrix_spd"] = _mp_spd(Mfm)
        out["map_error"] = float(mpm.sqrt(sum((_mpf(Mf[i, j]) - Mref[i, j]) ** 2 for i in range(6) for j in range(6)))
                                 / mpm.sqrt(sum(Mref[i, j] ** 2 for i in range(6) for j in range(6))))
        wf = np.linalg.eigvalsh(Mfs)[0]
        out["lambda_min_rel_error"] = float(abs(_mpf(wf) - wmin) / wmin)
    return out


@functools.cache
def r02_family() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    rows = []
    for r in (8.0, 12.0, 16.0, 20.0):
        for _ in range(10):
            row = measure_theta(_chol_theta(r, rng))
            row["range"] = r
            rows.append(row)
    return rows


# ============================================================================================== EC1-EC5
@pytest.mark.parametrize("kappa", ASSERTED)
def test_ec_forward_error_within_c_kappa_u(kappa: float) -> None:
    """EC1-EC5: every measured error <= 100 kappa u for kappa <= 1e10 (sampled directions, random + soft strain)."""
    r = measure_levels()[kappa]
    assert r["mp_spd"]
    assert abs(r["kappa"] / kappa - 1) < 1e-3                              # the designed level is the realised one
    b = r["bound"]
    for key in ("energy_random", "energy_soft", "S_normwise", "young", "nu_directional", "nu_engineering",
                "nu_uniaxial"):
        assert r[key] <= b, f"{key}: {r[key]:.3e} > {b:.3e} (kappa {r['kappa']:.3e}, kappa u {r['ku']:.3e})"
    assert r["reciprocity_float"] <= b
    assert r["reciprocity_mp"] <= 1e-30


@pytest.mark.parametrize("kappa", (1e12, 1e14))
def test_ec_reported_levels_not_asserted(kappa: float) -> None:
    """kappa >= 1e12: "forward accuracy not established" (reported, not asserted): the reference is SPD, the float
    results are finite, the helper says so; the measured errors are recorded by docs/foundations/scripts/."""
    r = measure_levels()[kappa]
    assert r["mp_spd"] and r["ku"] > 1e-4
    assert all(np.isfinite(r[k]) for k in ("energy_random", "energy_soft", "S_normwise", "young", "nu_directional",
                                           "nu_engineering", "nu_uniaxial", "reciprocity_float"))
    assert r["verdict"] in ("not_established", "undecidable")


# ============================================================================================== EC6, EC7, EC8
@pytest.mark.parametrize("kappa", (1.0, 1e2, 1e6, 1e10, 1e12))
def test_ec6a_acceptance_up_to_1e12(kappa: float) -> None:
    assert measure_levels()[kappa]["accepted"]
    L.ElasticTensor.from_stiffness(level_data()[kappa]["C6"])


def test_ec6a_kappa_1e14_is_spd_but_refused_as_numerically_singular() -> None:
    C6 = level_data()[1e14]["C6"]
    r = measure_levels()[1e14]
    assert r["mp_spd"]                                                      # exact arithmetic: SPD
    with pytest.raises(L.NotPositiveDefiniteError, match="numerically singular"):
        L.check_positive_definite(C6)
    assert not r["accepted"] and r["verdict"] == "undecidable"


def _spectrum_tensor(lam: np.ndarray, seed: int = 7) -> np.ndarray:
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((6, 6)))
    M = (Q * lam) @ Q.T
    return T.mandel_to_stiffness(0.5 * (M + M.T))


@pytest.mark.parametrize("t", (1e-1, 1e-6, 1e-12))
def test_ec6b_non_spd_rejected(t: float) -> None:
    C6 = _spectrum_tensor(np.array([-10.0 * t, 1.0, 2.0, 3.0, 4.0, 10.0]))
    with pytest.raises(L.NotPositiveDefiniteError):
        L.check_positive_definite(C6)
    with pytest.raises(L.NotPositiveDefiniteError):
        L.ElasticTensor.from_stiffness(C6)
    assert L.forward_accuracy_established(C6)["verdict"] == "not_positive_definite"


def test_ec6b_singular_and_indefinite_compliance_rejected() -> None:
    C6 = _spectrum_tensor(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0]))
    with pytest.raises(L.NotPositiveDefiniteError):
        L.check_positive_definite(C6)
    assert L.forward_accuracy_established(C6)["verdict"] in ("not_positive_definite", "undecidable")
    S_bad = L.orthotropic_compliance(20.0, 1.0, 1.0, 0.4, 0.4, 0.4, 5.0, 0.0, 0.0)       # det [[.05,-.25],[-.25,1]] < 0
    with pytest.raises(L.NotPositiveDefiniteError):
        L.ElasticTensor.from_compliance(S_bad)
    assert L.forward_accuracy_established(S_bad, "compliance")["verdict"] == "not_positive_definite"


def test_ec6c_valid_anisotropic_poisson_outside_isotropic_range_is_accepted() -> None:
    S = L.orthotropic_compliance(100.0, 1.0, 1.0, 0.4, 0.4, 0.4, 5.0, -3.0, 0.2)
    et = L.ElasticTensor.from_compliance(S)                                 # full SPD gate
    eng = et.engineering_constants()
    assert abs(eng["nu12"] - 5.0) <= 1e-12 * 5.0 and abs(eng["nu13"] + 3.0) <= 1e-12 * 3.0
    assert eng["nu12"] > 0.5 and eng["nu13"] < -1.0                         # outside [-1, 0.5]
    with mpm.workdps(50):
        Sm = mpm.matrix([[_mpf(S[i, j]) for j in range(6)] for i in range(6)])
        assert _mp_eigs(Sm)[0] > 0
    assert L.forward_accuracy_established(et.stiffness())["verdict"] == "established"
    # reciprocity of this tensor
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a = eng[f"nu{i + 1}{j + 1}"] / eng[f"E{i + 1}"]
        b = eng[f"nu{j + 1}{i + 1}"] / eng[f"E{j + 1}"]
        assert abs(a - b) <= 1e-12 * abs(a)


@functools.cache
def ec7_counts() -> dict[str, int]:
    rng = np.random.default_rng(SEED + 1)
    c = {"accepted_spd": 0, "false_accept": 0, "rejected_spd": 0, "rejected_indefinite": 0}
    for _ in range(300):
        s = float(rng.choice([-1.0, 1.0]))
        lam = np.concatenate([[s * 10.0 * 10.0 ** rng.uniform(-18, -10)], np.sort(rng.uniform(1, 10, 4)), [10.0]])
        C6 = _spectrum_tensor(lam, int(rng.integers(1, 10 ** 6)))
        M = T.stiffness_to_mandel(C6)
        M = 0.5 * (M + M.T)
        with mpm.workdps(50):
            spd = _mp_spd(mpm.matrix([[_mpf(M[i, j]) for j in range(6)] for i in range(6)]))
        acc = _accepts(C6)
        key = ("accepted_spd" if spd else "false_accept") if acc else ("rejected_spd" if spd else "rejected_indefinite")
        c[key] += 1
    return c


def test_ec7_no_false_acceptance() -> None:
    c = ec7_counts()
    assert sum(c.values()) == 300
    assert c["false_accept"] == 0, c


def test_ec8_verdict_helper() -> None:
    lv = measure_levels()
    for k in (1.0, 1e2, 1e6, 1e10):
        assert lv[k]["verdict"] == "established"
    assert lv[1e12]["verdict"] == "not_established" and lv[1e14]["verdict"] == "undecidable"
    r = L.forward_accuracy_established(L.isotropic_stiffness(1.0, 0.3))
    # kappa = 3K / 2G = (1/(1-2nu)) / (1/(1+nu)) = (1+nu)/(1-2nu) = 1.3/0.4 = 3.25 (the prereg text says 2.3333: slip)
    assert abs(r["condition_number"] - 3.25) <= 1e-12 * 3.25 and r["verdict"] == "established"
    assert abs(r["kappa_u"] - 3.25 * U) <= 1e-12 * 3.25 * U
    with pytest.raises(ValueError):
        L.forward_accuracy_established(np.eye(4))


# ============================================================================================== acoustic tensor
def _iso_C6(K: float, G: float) -> np.ndarray:
    lam = K - 2.0 * G / 3.0
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2.0 * G
    C[3:, 3:] = G * np.eye(3)
    return C


@functools.cache
def directions() -> np.ndarray:
    return np.vstack([L.fibonacci_sphere(500), _unit_rows(np.random.default_rng(SEED + 2), 200)])


def _bounds_data(kappa: float) -> tuple[np.ndarray, float, float]:
    C6 = level_data()[kappa]["C6"]
    Q = L.acoustic_tensor(C6, directions())
    lam = np.linalg.eigvalsh(0.5 * (Q + np.swapaxes(Q, -1, -2)))
    with mpm.workdps(50):
        wmin, wmax = (float(x) for x in _mp_eigs(_mp_mandel(C6)))
    return lam, wmin, wmax


@pytest.mark.parametrize("kappa", ASSERTED)
def test_ac1_acoustic_tensor_symmetric(kappa: float) -> None:
    Q = L.acoustic_tensor(level_data()[kappa]["C6"], directions())
    assert np.abs(Q - np.swapaxes(Q, -1, -2)).max() / np.abs(Q).max() <= 1e-14


@pytest.mark.parametrize("kappa", [pytest.param(1.0, marks=pytest.mark.xfail(
    strict=True, reason="E3 AC2 kappa=1: the proved bound w_min/2 is attained (Q = I/2) and float eigenvalues fall "
    "7e-16 below it, more than the preregistered 4 u w_max = 4.4e-16 slack; see test_ac2_amended")), 1e2, 1e6, 1e10])
def test_ac2_acoustic_positive_definite_within_proved_bounds(kappa: float) -> None:
    lam, wmin, wmax = _bounds_data(kappa)
    slack = 4 * U * wmax
    assert (lam[:, 0] >= wmin / 2 - slack).all(), (lam[:, 0].min(), wmin / 2)                   # proved lower bound
    assert (lam[:, -1] <= wmax + slack).all()
    assert (lam[:, 0] > 0).all()


def test_ac2_amended_slack_16u_at_kappa_1() -> None:
    """Post-hoc (after the AC2 kappa = 1 failure above): same bounds with a slack of 16 u w_max, labelled amended."""
    lam, wmin, wmax = _bounds_data(1.0)
    assert (lam[:, 0] >= wmin / 2 - 16 * U * wmax).all() and (lam[:, -1] <= wmax + 16 * U * wmax).all()


@pytest.mark.parametrize("K,G", [(1.7e11, 8.0e10), (3.0, 1.0)])
def test_ac3_isotropic_acoustic_eigenvalues(K: float, G: float) -> None:
    N = _unit_rows(np.random.default_rng(SEED + 3), 50)
    lam = np.linalg.eigvalsh(L.acoustic_tensor(_iso_C6(K, G), N))
    ref = np.array([G, G, K + 4.0 * G / 3.0])
    assert np.abs(lam / ref - 1).max() <= 1e-14


def test_ac4_wave_speeds_units_and_scaling() -> None:
    K, G, rho = 1.7e11, 8.0e10, 7800.0                      # synthetic Pa, Pa, kg/m^3 -> speeds in m/s
    C6 = _iso_C6(K, G)
    N = _unit_rows(np.random.default_rng(SEED + 4), 50)
    c = L.wave_speeds(C6, N, rho)
    ref = np.array([math.sqrt(G / rho), math.sqrt(G / rho), math.sqrt((K + 4 * G / 3) / rho)])
    assert np.abs(c / ref - 1).max() <= 1e-14
    assert 3000.0 < ref[0] < 3500.0                          # sqrt(8e10 / 7800) = 3202.6 m/s: a speed in m/s, not m^2/s^2
    assert np.abs(L.wave_speeds(C6, N, 4 * rho) / (c / 2) - 1).max() <= 1e-15
    assert np.abs(L.wave_speeds(9.0 * C6, N, rho) / (3.0 * c) - 1).max() <= 1e-15
    assert np.abs(L.wave_speeds((C6 / 1e9) * 1e9, N, rho) / c - 1).max() <= 1e-15
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            L.wave_speeds(C6, N, bad)
    cn = L.wave_speeds(_iso_C6(3.0, -1.0), N[:5], 1.0)       # G < 0: two negative acoustic eigenvalues -> NaN
    assert np.isnan(cn[:, :2]).all() and np.isfinite(cn[:, 2]).all()


def test_ac5_sampled_ellipticity_is_labelled_sampled_and_respects_the_proved_bound() -> None:
    C6 = level_data()[1e6]["C6"]
    res = L.strong_ellipticity(C6, n_directions=500)
    assert res["label"] in ("sampled", "refined") and "certif" not in res["label"]
    with mpm.workdps(50):
        wmin, wmax = (float(x) for x in _mp_eigs(_mp_mandel(C6)))
    assert wmin / 2 - 4 * U * wmax <= res["min_eigenvalue"] <= wmax
    res0 = L.strong_ellipticity(C6, n_directions=500, refine=False)
    assert res0["label"] == "sampled"


# ============================================================================================== R02
def test_r02a_historical_extreme_case_reproduced() -> None:
    theta = np.concatenate([np.linspace(-20.0, 20.0, 6), np.zeros(15)])
    C = A.stiffness_from_cholesky(theta)
    assert np.isfinite(C).all()
    with mpm.workdps(100):
        M = _mp_mandel(C)
        wmin, wmax = _mp_eigs(M)
        assert 1e34 <= float(wmax / wmin) <= 1e36                            # analytic exp(80) = 5.5e34
        assert _mp_spd(M)                                                    # SPD as stored and by construction
    with pytest.raises(L.NotPositiveDefiniteError):
        L.check_positive_definite(C)
    with pytest.raises(L.NotPositiveDefiniteError):
        A.cholesky_parameters(C)
    assert L.forward_accuracy_established(C)["verdict"] == "undecidable"


def test_r02b_dense_family() -> None:
    rows = r02_family()
    assert len(rows) == 40
    assert all(r["exact_spd"] for r in rows)                                 # (i) SPD by construction
    big = [r for r in rows if r["kappa_exact"] >= 1e15]
    assert big and all(not r["accepted"] for r in big)                       # (ii) refused, never accepted
    assert not any(r["accepted"] and not r["float_matrix_spd"] for r in rows)  # (iii) no false acceptance
    assert max(r["map_error"] for r in rows) <= C_FACTOR * U                 # (iv) theta -> C accurate normwise
