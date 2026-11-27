"""Second derivative and Hessian-vector product of spectral functions (``toporlogy.core.spectral``): records SD1-SD10 of
docs/foundations/prereg/T2_second_derivative.md (thresholds preregistered 2026-11-27 06:51 +0300, commit fe92d84,
before this file was run).

References independent of the production formula: block-matrix functions (``scipy.linalg.expm/sqrtm/logm`` and mpmath
at 40 digits on [[A,H,0],[0,A,K],[0,0,A]]), mpmath divided differences from the definition, a word series for exp,
closed forms, and central differences of the (separately verified) first derivative. Tests that call the production
kernel twice are labelled (consistency).
"""
from __future__ import annotations

import math
import zlib

import mpmath as mp
import numpy as np
import pytest
import scipy.linalg as sla

from toporlogy.core import spectral as SP

RNG = np.random.default_rng(20261127 + 6)

CASES = [("exp", None), ("log", None), ("sqrt", None), ("inv_sqrt", None), ("power", 3.0), ("power", -1.5),
         ("power", 0.25)]
IDS = [f"{f}" if p is None else f"{f}{p:g}" for f, p in CASES]


def tol(kappa: float) -> float:
    return 1e-10 if kappa <= 1e2 else 1e-12 * kappa


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def rand_sym(n: int = 3, rng: np.random.Generator = RNG) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return 0.5 * (a + a.T)


def unit_sym(n: int = 3, rng: np.random.Generator = RNG) -> np.ndarray:
    a = rand_sym(n, rng)
    return a / np.linalg.norm(a, 2)


def rand_rot(n: int = 3, rng: np.random.Generator = RNG) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def spd(spectrum, rng: np.random.Generator = RNG) -> np.ndarray:
    Q = rand_rot(len(spectrum), rng)
    return (Q * np.asarray(spectrum, float)) @ Q.T


def spd_kappa(n: int, kappa: float, rng: np.random.Generator = RNG) -> np.ndarray:
    spec = [1.0, kappa] if n == 2 else [1.0, kappa ** rng.uniform(0.2, 0.8), kappa]
    return spd(spec, rng)


# ---------------------------------------------------------------------------------------------- block oracles
def block(A, H, K) -> np.ndarray:
    n = A.shape[0]
    M = np.zeros((3 * n, 3 * n))
    for i in range(3):
        M[i * n:(i + 1) * n, i * n:(i + 1) * n] = A
    M[:n, n:2 * n] = H
    M[n:2 * n, 2 * n:] = K
    return M


def _real(X: np.ndarray) -> np.ndarray:
    assert np.max(np.abs(np.imag(X))) <= 1e-9 * max(1.0, np.max(np.abs(X)))
    return np.real(X)


def scipy_second(A, H, K, f: str) -> np.ndarray:
    fn = {"exp": sla.expm, "sqrt": sla.sqrtm, "log": sla.logm}[f]
    n = A.shape[0]
    return _real(fn(block(A, H, K))[:n, 2 * n:] + fn(block(A, K, H))[:n, 2 * n:])


def _mpm(M: np.ndarray):
    return mp.matrix(np.asarray(M, float).tolist())


def _mp_block(M: np.ndarray, f: str, p: float | None):
    m = _mpm(M)
    if f == "exp":
        return mp.expm(m)
    if f == "log":
        return mp.logm(m)
    if f == "sqrt":
        return mp.sqrtm(m)
    if f == "inv_sqrt":
        return mp.sqrtm(m) ** -1
    return mp.expm(mp.mpf(p) * mp.logm(m))


def mp_second(A, H, K, f: str, p: float | None = None) -> np.ndarray:
    n = A.shape[0]
    with mp.workdps(40):
        X1 = _mp_block(block(A, H, K), f, p)
        X2 = _mp_block(block(A, K, H), f, p)
        X = [[X1[i, 2 * n + j] + X2[i, 2 * n + j] for j in range(n)] for i in range(n)]
        # mpmath matrix functions can return a complex type with a ~1e-40 imaginary part: check, then drop it
        assert max(abs(mp.im(v)) for row in X for v in row) <= 1e-25 * max(abs(v) for row in X for v in row)
        return np.array([[float(mp.re(v)) for v in row] for row in X])


# ---------------------------------------------------------------------------------------------------- SD1
@pytest.mark.parametrize("n", [3, 2])
@pytest.mark.parametrize("kappa", [1e1, 1e2, 1e3, 1e4])
@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd1_block_matrix_oracle_random_spd(n, kappa, f, p):
    A, H, K = spd_kappa(n, kappa), rand_sym(n), rand_sym(n)
    if f == "exp":                                   # set-up fix: e^1000 overflows; keep cond κ but λmax = 5
        A = A * (5.0 / kappa)
    got = SP.spectral_second(A, H, K, f, p)
    if f in ("exp", "log", "sqrt"):
        ref = scipy_second(A, H, K, f)
        t = tol(kappa) if f == "exp" else max(tol(kappa), 1e-8)
        assert rel(got, ref) <= t
    else:
        assert rel(got, mp_second(A, H, K, f, p)) <= tol(kappa)
    if f in ("log", "sqrt") and kappa <= 1e2:        # the scipy reference is also checked against mpmath where cheap
        assert rel(got, mp_second(A, H, K, f)) <= tol(kappa)


# ---------------------------------------------------------------------------------------------------- SD2
@pytest.mark.parametrize("spec", [[2.0, 2.0, 5.0], [3.0, 3.0, 3.0], [2.0, 2.0 + 1e-10, 5.0]],
                         ids=["225", "333", "near_repeat"])
@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd2_repeated_and_near_repeated_spectra_vs_mpmath_block(spec, f, p):
    rng = np.random.default_rng(zlib.crc32(repr((tuple(spec), f, p)).encode()))
    worst = 0.0
    for _ in range(5):                                # 5 random rotations per case (mpmath at 40 digits is slow)
        A = spd(spec, rng)
        H, K = rand_sym(3, rng), rand_sym(3, rng)
        worst = max(worst, rel(SP.spectral_second(A, H, K, f, p), mp_second(A, H, K, f, p)))
    assert worst <= 1e-10


# ---------------------------------------------------------------------------------------------------- SD3
def _mp_f(x, f: str, p: float | None):
    if f == "exp":
        return mp.exp(x)
    if f == "log":
        return mp.log(x)
    if f == "sqrt":
        return mp.sqrt(x)
    if f == "inv_sqrt":
        return 1 / mp.sqrt(x)
    return x ** mp.mpf(p)


def mp_dd2(x: float, y: float, z: float, f: str, p: float | None):
    """f[x, y, z] at 80 digits from the definition; coincident float nodes by the limit formulas."""
    g = lambda t: _mp_f(t, f, p)
    x, y, z = mp.mpf(x), mp.mpf(y), mp.mpf(z)
    nodes = sorted([x, y, z])
    if nodes[0] == nodes[1] == nodes[2]:
        return mp.diff(g, nodes[0], 2) / 2
    if nodes[0] == nodes[1] or nodes[1] == nodes[2]:
        a, c = (nodes[0], nodes[2]) if nodes[0] == nodes[1] else (nodes[2], nodes[0])
        return ((g(c) - g(a)) / (c - a) - mp.diff(g, a)) / (c - a)
    return sum(g(u) / ((u - v) * (u - w)) for u, v, w in ((x, y, z), (y, x, z), (z, x, y)))


DELTAS = [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]


@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd3_second_divided_differences_vs_mpmath_gap_sweep(f, p):
    worst, where = 0.0, None
    with mp.workdps(80):
        for a in (0.5, 2.0, 40.0):
            for c in (a + 3.0, 2.5 * a):
                for d in DELTAS:
                    w = np.array([a, a + d, c])
                    F = SP._second_divided_differences(w, f, p)
                    ref = np.array([[[float(mp_dd2(w[i], w[j], w[k], f, p)) for k in range(3)] for j in range(3)]
                                    for i in range(3)])
                    e = rel(F, ref)
                    if e > worst:
                        worst, where = e, (a, c, d)
    assert worst <= 1e-11, where


# ---------------------------------------------------------------------------------------------------- SD4
@pytest.mark.parametrize("spec", [None, [2.0, 2.0, 5.0], [3.0, 3.0, 3.0]])
@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd4_symmetry_in_h_and_k_consistency(spec, f, p):
    A = spd_kappa(3, 1e3) if spec is None else spd(spec)
    if spec is None and f == "exp":                  # set-up fix: e^1000 overflows; keep cond 1e3 with λmax = 5
        A = A * 5e-3
    H, K = rand_sym(), rand_sym()
    X, Y = SP.spectral_second(A, H, K, f, p), SP.spectral_second(A, K, H, f, p)
    assert rel(X, Y) <= 1e-13
    assert rel(X, X.T) <= 1e-13


# ---------------------------------------------------------------------------------------------------- SD5
def _d2f_scalar(f: str, p: float | None, x: float) -> float:
    return {"exp": math.exp(x), "log": -1.0 / x ** 2, "sqrt": -0.25 * x ** -1.5, "inv_sqrt": 0.75 * x ** -2.5,
            "power": (p or 0.0) * ((p or 0.0) - 1.0) * x ** ((p or 0.0) - 2.0)}[f]


@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd5_triple_repeat_is_half_f_second_times_anticommutator(f, p):
    A = spd([3.0, 3.0, 3.0])
    H, K = rand_sym(), rand_sym()
    ref = 0.5 * _d2f_scalar(f, p, 3.0) * (H @ K + K @ H)
    assert rel(SP.spectral_second(A, H, K, f, p), ref) <= 1e-12


@pytest.mark.parametrize("spec", [[0.7, 2.0, 5.0], [2.0, 2.0, 5.0], [3.0, 3.0, 3.0], [1.0, 1.5, 4.0]])
def test_sd5_polynomial_and_inverse_closed_forms(spec):
    A = spd(spec)
    H, K = rand_sym(), rand_sym()
    Ai = np.linalg.inv(A)
    assert rel(SP.spectral_second(A, H, K, "power", 2.0), H @ K + K @ H) <= 1e-12
    words3 = H @ K @ A + H @ A @ K + A @ H @ K + K @ H @ A + K @ A @ H + A @ K @ H
    assert rel(SP.spectral_second(A, H, K, "power", 3.0), words3) <= 1e-12
    inv = Ai @ H @ Ai @ K @ Ai + Ai @ K @ Ai @ H @ Ai
    assert rel(SP.spectral_second(A, H, K, "power", -1.0), inv) <= 1e-12


@pytest.mark.parametrize("spec", [[0.7, 2.0, 5.0], [2.0, 2.0, 5.0], [3.0, 3.0, 3.0], [1.0, 1.0 + 1e-10, 4.0]])
def test_sd5_exp_word_series(spec):
    A = spd(spec)
    H, K = rand_sym(), rand_sym()
    N = 45
    P = [np.eye(3)]
    for _ in range(N):
        P.append(P[-1] @ A)
    tot = np.zeros((3, 3))
    for pp in range(N - 1):
        for q in range(N - 1 - pp):
            HK = P[pp] @ H @ P[q]
            KH = P[pp] @ K @ P[q]
            for r in range(N - 1 - pp - q):
                c = 1.0 / math.factorial(pp + q + r + 2)
                tot += c * (HK @ K @ P[r] + KH @ H @ P[r])
    assert rel(SP.spectral_second(A, H, K, "exp"), tot) <= 1e-11


# ---------------------------------------------------------------------------------------------------- SD6
def _central(fn, h):
    return (fn(h) - fn(-h)) / (2.0 * h)


HS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


@pytest.mark.parametrize("spec", [[1.0, 2.5, 5.0], [2.0, 2.0, 5.0]])
@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd6_central_fd_of_first_derivative_step_sweep(spec, f, p):
    A = spd(spec)
    H, K = unit_sym(), unit_sym()
    ref = SP.spectral_second(A, H, K, f, p)
    errs = [rel(ref, _central(lambda t: SP.spectral_frechet(A + t * K, H, f, p), h)) for h in HS]
    assert min(errs) <= 1e-7, errs
    if (f, p) == ("power", 3.0):
        # A ↦ L_A[H] is quadratic for the cubic, so the central difference has NO truncation error: the
        # preregistered O(h²) slope check does not apply (failed prediction of its wording, see the record)
        assert max(errs[:3]) <= 1e-9, errs
    else:
        assert errs[1] / max(errs[2], 1e-300) >= 30.0, errs  # O(h^2) truncation between h = 1e-2 and 1e-3


# ---------------------------------------------------------------------------------------------------- SD7
@pytest.mark.parametrize("spec", [[1.0, 2.5, 5.0], [2.0, 2.0, 5.0]])
@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd7_hvp_vs_fd_of_vjp(spec, f, p):
    A = spd(spec)
    G, H, K = rand_sym(), unit_sym(), unit_sym()
    hv = SP.spectral_hvp(A, G, H, f, p)
    errs = [rel(hv, _central(lambda t: SP.spectral_vjp(A + t * H, G, f, p), h)) for h in HS]
    assert min(errs) <= 1e-7, errs
    # consistency identities (same kernel): <K, hvp(G,H)> = <G, d2f[H,K]> and symmetry of the Hessian
    lhs = float(np.sum(K * hv))
    assert abs(lhs - float(np.sum(G * SP.spectral_second(A, H, K, f, p)))) <= 1e-12 * (1.0 + abs(lhs))
    assert abs(lhs - float(np.sum(H * SP.spectral_hvp(A, G, K, f, p)))) <= 1e-12 * (1.0 + abs(lhs))


# ---------------------------------------------------------------------------------------------------- SD8
@pytest.mark.parametrize("f,p", CASES, ids=IDS)
def test_sd8_batch_broadcast_and_rotation_covariance(f, p):
    A = np.stack([spd_kappa(3, 1e2) for _ in range(5)])
    H, K = rand_sym(), np.stack([rand_sym() for _ in range(5)])
    got = SP.spectral_second(A, H, K, f, p)
    assert got.shape == (5, 3, 3)
    loop = np.stack([SP.spectral_second(A[i], H, K[i], f, p) for i in range(5)])
    assert rel(got, loop) <= 1e-13
    R = rand_rot()
    rot = SP.spectral_second(R @ A[0] @ R.T, R @ H @ R.T, R @ K[0] @ R.T, f, p)
    assert rel(rot, R @ got[0] @ R.T) <= 1e-13


# ---------------------------------------------------------------------------------------------------- SD9
def test_sd9_invalid_inputs():
    A, H, K = spd([1.0, 2.0, 3.0]), rand_sym(), rand_sym()
    bad_nonsym = H + np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]])
    nan = H.copy()
    nan[0, 0] = np.nan
    for call in (lambda: SP.spectral_second(A, bad_nonsym, K, "log"),
                 lambda: SP.spectral_second(A, H, bad_nonsym, "log"),
                 lambda: SP.spectral_second(A + bad_nonsym * 0.1, H, K, "log"),
                 lambda: SP.spectral_second(A, nan, K, "log"),
                 lambda: SP.spectral_second(np.eye(4), np.eye(4), np.eye(4), "log"),
                 lambda: SP.spectral_second(np.ones(3), H, K, "log"),
                 lambda: SP.spectral_second(np.float64(1.0), H, K, "log"),
                 lambda: SP.spectral_second(A, np.eye(2), K, "log"),
                 lambda: SP.spectral_second(A, H, np.eye(2), "log"),
                 lambda: SP.spectral_second(A, H, K, "sin"),
                 lambda: SP.spectral_second(A, H, K, "power"),
                 lambda: SP.spectral_second(A, H, K, "log", 2.0),
                 lambda: SP.spectral_second(A, H, K, "power", float("nan")),
                 lambda: SP.spectral_hvp(A, bad_nonsym, H, "log"),
                 lambda: SP.spectral_hvp(A, H, bad_nonsym, "log"),
                 lambda: SP.spectral_hvp(A, H, K, "power")):
        with pytest.raises(ValueError):
            call()
    indef = spd([-1.0, 2.0, 3.0])
    for f, p in (("log", None), ("sqrt", None), ("inv_sqrt", None), ("power", 3.0)):
        with pytest.raises(ValueError):
            SP.spectral_second(indef, H, K, f, p)
        with pytest.raises(ValueError):
            SP.spectral_hvp(indef, H, K, f, p)
        with pytest.raises(ValueError):
            SP.spectral_second(np.diag([0.0, 2.0, 3.0]), H, K, f, p)
    assert rel(SP.spectral_second(indef, H, K, "exp"), mp_second(indef, H, K, "exp")) <= 1e-10


# --------------------------------------------------------------------------------------------------- SD10
def test_sd10_negative_control_naive_recurrence_fails_at_near_repeat():
    a = 2.0
    b, c = a + 1e-10, a + 2e-10

    def D(x, y):                                      # stable first divided difference of log, written here
        lo, hi = min(x, y), max(x, y)
        return math.log1p((hi - lo) / lo) / (hi - lo)

    naive = (D(a, b) - D(b, c)) / (a - c)
    with mp.workdps(80):
        ref = float(mp_dd2(a, b, c, "log", None))
    assert abs(naive - ref) / abs(ref) > 1e-8
    # the production value at the same nodes is accurate
    F = SP._second_divided_differences(np.array([a, b, c]), "log", None)
    assert abs(F[0, 1, 2] - ref) / abs(ref) <= 1e-11
