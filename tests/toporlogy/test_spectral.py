"""Spectral functions of symmetric tensors (``toporlogy.core.spectral``): records TS1-TS9 of
docs/foundations/prereg/T_spectral.md (thresholds preregistered 2026-11-27 01:29 +0300, commit 828d9be, before this
file was run).

References are independent of the production kernel: closed forms, mpmath at 60 digits (central differences of
``mp.sqrtm/logm/expm``), ``scipy.linalg`` (``expm_frechet``, ``solve_sylvester``, ``sqrtm``, ``logm``) and a
projector formula written here. Tests that call the production kernel twice are labelled (consistency / regression).
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
import scipy.linalg as sla

from toporlogy.core import spectral as SP

RNG = np.random.default_rng(20261127)

# scalar functions and derivatives written out here (no import from the production module)
_F = {
    "sqrt": (np.sqrt, lambda x: 0.5 / np.sqrt(x), None),
    "log": (np.log, lambda x: 1.0 / x, None),
    "exp": (np.exp, np.exp, None),
    "inv_sqrt": (lambda x: x ** -0.5, lambda x: -0.5 * x ** -1.5, None),
}
CASES = [("sqrt", None), ("log", None), ("exp", None), ("inv_sqrt", None), ("power", 3.0), ("power", -1.5),
         ("power", 0.25)]


def _fd(f: str, p: float | None):
    if f == "power":
        return (lambda x: x ** p), (lambda x: p * x ** (p - 1.0))
    return _F[f][0], _F[f][1]


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def rand_sym(n: int = 3, rng: np.random.Generator = RNG) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return 0.5 * (a + a.T)


def rand_rot(n: int = 3, rng: np.random.Generator = RNG) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def spd(spectrum, rng: np.random.Generator = RNG) -> tuple[np.ndarray, np.ndarray]:
    Q = rand_rot(len(spectrum), rng)
    return (Q * np.asarray(spectrum, float)) @ Q.T, Q


# ---------------------------------------------------------------------------------------- mpmath references
def _mpf(A: np.ndarray):
    return mp.matrix(np.asarray(A, float).tolist())


def _mp_f(M, f: str, p: float | None):
    if f == "sqrt":
        return mp.sqrtm(M)
    if f == "log":
        return mp.logm(M)
    if f == "exp":
        return mp.expm(M)
    if f == "inv_sqrt":
        return mp.sqrtm(M) ** -1
    return mp.expm(mp.mpf(p) * mp.logm(M))


def mp_value(A, f: str, p: float | None = None) -> np.ndarray:
    with mp.workdps(60):
        R = _mp_f(_mpf(A), f, p)
        return np.array([[float(mp.re(R[i, j])) for j in range(R.cols)] for i in range(R.rows)])


def mp_dir(A, H, f: str, p: float | None = None, dps: int = 60) -> np.ndarray:
    """Central difference at ``dps`` digits, h = 10^(-dps/3) (truncation and rounding ~ 10^(-2 dps/3))."""
    with mp.workdps(dps):
        h = mp.mpf(10) ** (-(dps // 3))
        MA, MH = _mpf(A), _mpf(H)
        R = (_mp_f(MA + h * MH, f, p) - _mp_f(MA - h * MH, f, p)) / (2 * h)
        return np.array([[float(mp.re(R[i, j])) for j in range(R.cols)] for i in range(R.rows)])


def mp_log_dir_eigsy(A, H, dps: int = 100) -> np.ndarray:
    """Log derivative at ``dps`` digits via mpmath's symmetric eigensolver (``mp.logm`` does not converge at κ = 1e12):
    central difference of Q log(Λ) Qᵀ, h = 1e-30."""
    with mp.workdps(dps):
        def logsym(M):
            w, Q = mp.eigsy(M)
            return Q * mp.diag([mp.log(x) for x in w]) * Q.T
        h = mp.mpf("1e-30")
        MA, MH = _mpf(A), _mpf(H)
        R = (logsym(MA + h * MH) - logsym(MA - h * MH)) / (2 * h)
        return np.array([[float(R[i, j]) for j in range(3)] for i in range(3)])


def projector_oracle(Q: np.ndarray, groups: list[list[int]], lam: list[float], H, f: str, p: float | None):
    """L[H] = sum_kl dd(λk, λl) P_k H P_l with P_k the spectral projectors of the exactly known A = Q diag Qᵀ
    (distinct eigenvalue groups only; no eigenvector of a degenerate subspace is used)."""
    fv, fp = _fd(f, p)
    P = [Q[:, g] @ Q[:, g].T for g in groups]
    out = np.zeros_like(H)
    for k in range(len(groups)):
        for l in range(len(groups)):
            dd = fp(lam[k]) if k == l else (fv(lam[k]) - fv(lam[l])) / (lam[k] - lam[l])
            out += dd * P[k] @ H @ P[l]
    return out


def scipy_value(A, f, p):
    if f == "sqrt":
        return sla.sqrtm(A).real
    if f == "log":
        return sla.logm(A).real
    if f == "exp":
        return sla.expm(A)
    if f == "inv_sqrt":
        return np.linalg.inv(sla.sqrtm(A).real)
    return sla.expm(p * sla.logm(A).real)


# ================================================================================================ TS1 values
def test_ts1_values_diag_1_4_9() -> None:
    A = np.diag([1.0, 4.0, 9.0])
    d = np.array([1.0, 4.0, 9.0])
    assert rel(SP.spectral_function(A, "sqrt"), np.diag([1.0, 2.0, 3.0])) <= 1e-12
    assert rel(SP.spectral_function(A, "inv_sqrt"), np.diag([1.0, 0.5, 1 / 3])) <= 1e-12
    assert rel(SP.spectral_function(A, "log"), np.diag(np.log(d))) <= 1e-12
    assert rel(SP.spectral_function(A, "exp"), np.diag(np.exp(d))) <= 1e-12
    assert rel(SP.spectral_function(A, "power", 3.0), np.diag(d ** 3)) <= 1e-12
    assert rel(SP.spectral_function(A, "power", -1.0), np.diag(1 / d)) <= 1e-12


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("f,p", CASES)
def test_ts1_values_rotated_vs_mpmath(n: int, f: str, p: float | None) -> None:
    rng = np.random.default_rng(11 + n)
    for _ in range(4):
        lam = np.exp(rng.uniform(0.0, math.log(1e4), n)) if f != "exp" else rng.uniform(-3.0, 3.0, n)
        A, _Q = spd(lam, rng)
        assert rel(SP.spectral_function(A, f, p), mp_value(A, f, p)) <= 1e-12


def test_ts1_exp_indefinite_ok() -> None:
    A, _ = spd([-2.0, 0.5, 3.0], np.random.default_rng(3))
    assert rel(SP.spectral_function(A, "exp"), mp_value(A, "exp")) <= 1e-12


# ================================================================================================ TS2 (T3)
H_OFF = np.array([[0.0, 1.0, -2.0], [1.0, 0.0, 0.5], [-2.0, 0.5, 0.0]])
A149 = np.diag([1.0, 4.0, 9.0])


def _dd_diag(f: str, p: float | None, lam):
    fv, fp = _fd(f, p)
    n = len(lam)
    return np.array([[fp(lam[i]) if i == j else (fv(lam[i]) - fv(lam[j])) / (lam[i] - lam[j]) for j in range(n)]
                     for i in range(n)])


@pytest.mark.parametrize("f,p", CASES)
def test_ts2_derivative_diag_analytic_and_mpmath(f: str, p: float | None) -> None:
    L = SP.spectral_frechet(A149, H_OFF, f, p)
    ana = _dd_diag(f, p, [1.0, 4.0, 9.0]) * H_OFF
    assert rel(L, ana) <= 1e-12
    assert rel(L, mp_dir(A149, H_OFF, f, p)) <= 1e-12


def test_ts2_sqrt_closed_values() -> None:
    L = SP.spectral_frechet(A149, H_OFF, "sqrt")
    # 1/(√a+√b): (1,2) 1/3, (1,3) 1/4, (2,3) 1/5
    assert abs(L[0, 1] - 1.0 / 3.0) <= 1e-14 and abs(L[0, 2] + 2.0 / 4.0) <= 1e-14 and abs(L[1, 2] - 0.5 / 5.0) <= 1e-14
    assert np.abs(np.diag(L)).max() == 0.0


def test_ts2_sqrt_sylvester() -> None:
    S = np.diag([1.0, 2.0, 3.0])
    X = sla.solve_sylvester(S, S, H_OFF)               # X S + S X = H
    assert rel(SP.spectral_frechet(A149, H_OFF, "sqrt"), X) <= 1e-12
    rng = np.random.default_rng(21)
    for _ in range(10):                                # rotated, general A with scipy sqrtm
        A, _Q = spd(np.exp(rng.uniform(0, math.log(1e3), 3)), rng)
        H = rand_sym(3, rng)
        S = sla.sqrtm(A).real
        assert rel(SP.spectral_frechet(A, H, "sqrt"), sla.solve_sylvester(S, S, H)) <= 1e-11


@pytest.mark.parametrize("f", ["sqrt", "log", "exp"])
def test_ts2_double_precision_fd_step_sweep(f: str) -> None:
    H = H_OFF / np.abs(H_OFF).max()
    L = SP.spectral_frechet(A149, H, f)
    errs = []
    for h in 10.0 ** -np.arange(1, 9):
        fd = (scipy_value(A149 + h * H, f, None) - scipy_value(A149 - h * H, f, None)) / (2 * h)
        errs.append(rel(L, fd))
    assert min(errs) <= 1e-6
    assert errs[0] > errs[2]                           # the sweep has the expected truncation-error shape


# ================================================================================================ TS3 (T4)
def test_ts3_at_two_identity_exact_limits() -> None:
    A = 2.0 * np.eye(3)
    for _ in range(5):
        H = rand_sym()
        assert rel(SP.spectral_frechet(A, H, "log"), H / 2.0) <= 1e-12
        assert rel(SP.spectral_frechet(A, H, "sqrt"), H / (2.0 * math.sqrt(2.0))) <= 1e-12
        assert rel(SP.spectral_frechet(A, H, "exp"), math.exp(2.0) * H) <= 1e-12
        assert rel(SP.spectral_frechet(A, H, "inv_sqrt"), -0.5 * 2.0 ** -1.5 * H) <= 1e-12
        assert rel(SP.spectral_frechet(A, H, "power", 3.0), 3 * 4.0 * H) <= 1e-12


@pytest.mark.parametrize("f,p", CASES)
def test_ts3_repeated_rotated_projector_oracle(f: str, p: float | None) -> None:
    rng = np.random.default_rng(31)
    for _ in range(8):
        A, Q = spd([2.0, 2.0, 5.0], rng)
        H = rand_sym(3, rng)
        ref = projector_oracle(Q, [[0, 1], [2]], [2.0, 5.0], H, f, p)
        assert rel(SP.spectral_frechet(A, H, f, p), ref) <= 1e-12


def test_ts3_exp_vs_expm_frechet() -> None:
    rng = np.random.default_rng(32)
    for spec in ([2.0, 2.0, 5.0], [1.0, 1.0, 1.0], [-1.0, 0.3, 2.5]):
        A, _ = spd(spec, rng)
        H = rand_sym(3, rng)
        ref = sla.expm_frechet(A, H, compute_expm=False)
        assert rel(SP.spectral_frechet(A, H, "exp"), ref) <= 1e-12


def test_ts3_log_vs_inverse_of_expm_frechet() -> None:
    """L_log[H] solves expm_frechet(logm(A), X) = H (chain rule on exp(log A) = A)."""
    rng = np.random.default_rng(33)
    for spec in ([2.0, 2.0, 5.0], [1.0, 4.0, 9.0], [0.5, 0.5, 0.5]):
        A, _ = spd(spec, rng)
        H = rand_sym(3, rng)
        X = sla.logm(A).real
        M9 = np.column_stack([sla.expm_frechet(X, E.reshape(3, 3), compute_expm=False).ravel() for E in np.eye(9)])
        ref = np.linalg.solve(M9, H.ravel()).reshape(3, 3)
        assert rel(SP.spectral_frechet(A, H, "log"), ref) <= 1e-11


def test_ts3_inv_sqrt_and_power_closed_forms() -> None:
    rng = np.random.default_rng(34)
    A, _ = spd([2.0, 2.0, 5.0], rng)
    H = rand_sym(3, rng)
    S = sla.sqrtm(A).real
    X = sla.solve_sylvester(S, S, H)
    Si = np.linalg.inv(S)
    assert rel(SP.spectral_frechet(A, H, "inv_sqrt"), -Si @ X @ Si) <= 1e-11
    Ai = np.linalg.inv(A)
    assert rel(SP.spectral_frechet(A, H, "power", 2.0), H @ A + A @ H) <= 1e-12
    assert rel(SP.spectral_frechet(A, H, "power", -1.0), -Ai @ H @ Ai) <= 1e-12
    assert rel(SP.spectral_frechet(A, H, "power", 1.0), H) <= 1e-12
    assert rel(SP.spectral_frechet(A, H, "power", 0.0), np.zeros((3, 3)) + 1e-300) <= 1e-12 or \
        np.abs(SP.spectral_frechet(A, H, "power", 0.0)).max() == 0.0


# ================================================================================================ TS4 near repeat
@pytest.mark.parametrize("f,p", [("sqrt", None), ("log", None), ("exp", None), ("inv_sqrt", None), ("power", 3.0)])
def test_ts4_near_repeat_matches_mpmath_and_continuity(f: str, p: float | None) -> None:
    rng = np.random.default_rng(41)
    Q = rand_rot(3, rng)
    A = (Q * np.array([2.0, 2.0 + 1e-10, 5.0])) @ Q.T
    A0 = (Q * np.array([2.0, 2.0, 5.0])) @ Q.T
    for _ in range(4):
        H = rand_sym(3, rng)
        L = SP.spectral_frechet(A, H, f, p)
        assert np.isfinite(L).all()
        assert rel(L, mp_dir(A, H, f, p)) <= 1e-10
        L0 = SP.spectral_frechet(A0, H, f, p)
        assert rel(L, L0) <= 1e-9


# ================================================================================================ TS5 rotated spaces
SPECTRA = [([1.0, 1.0, 1.0], [[0, 1, 2]]), ([1.0, 1.0, 4.0], [[0, 1], [2]]), ([1.0, 4.0, 4.0], [[0], [1, 2]])]


@pytest.mark.parametrize("f,p", CASES)
@pytest.mark.parametrize("spec,groups", SPECTRA)
def test_ts5_random_rotations_projector_and_covariance(spec, groups, f: str, p: float | None) -> None:
    rng = np.random.default_rng(51)
    lam = [spec[g[0]] for g in groups]
    Ad = np.diag(spec)
    for _ in range(40):
        Q = rand_rot(3, rng)
        A = Q @ Ad @ Q.T
        H = rand_sym(3, rng)
        L = SP.spectral_frechet(A, H, f, p)
        assert rel(L, projector_oracle(Q, groups, lam, H, f, p)) <= 1e-12
        # covariance (symmetry property of the kernel): L_{QAQᵀ}[QHQᵀ] = Q L_A[H] Qᵀ
        assert rel(SP.spectral_frechet(A, Q @ H @ Q.T, f, p), Q @ SP.spectral_frechet(Ad, H, f, p) @ Q.T) <= 1e-12


def test_ts5_near_repeat_spectrum_rotated() -> None:
    rng = np.random.default_rng(52)
    for _ in range(20):
        Q = rand_rot(3, rng)
        A = (Q * np.array([1.0, 1.0 + 1e-10, 4.0])) @ Q.T
        H = rand_sym(3, rng)
        ref = projector_oracle(Q, [[0, 1], [2]], [1.0, 4.0], H, "log", None)
        assert rel(SP.spectral_frechet(A, H, "log"), ref) <= 1e-9
        assert rel(SP.spectral_frechet(A, H, "sqrt"),
                   projector_oracle(Q, [[0, 1], [2]], [1.0, 4.0], H, "sqrt", None)) <= 1e-9


def test_ts5_negative_control_nearly_repeated_is_not_repeated() -> None:
    """Treating the pair (2, 2.001) as exactly repeated is detectably wrong: the test can fail."""
    rng = np.random.default_rng(53)
    Q = rand_rot(3, rng)
    A = (Q * np.array([2.0, 2.001, 5.0])) @ Q.T
    H = rand_sym(3, rng)
    wrong = projector_oracle(Q, [[0, 1], [2]], [2.0, 5.0], H, "sqrt", None)
    assert rel(SP.spectral_frechet(A, H, "sqrt"), wrong) > 1e-6


# ================================================================================================ TS6 duality
@pytest.mark.parametrize("f,p", CASES)
def test_ts6_vjp_jvp_duality_consistency(f: str, p: float | None) -> None:
    """Consistency (same kernel twice): ⟨L[H],G⟩ = ⟨H,L[G]⟩ for symmetric H, G."""
    rng = np.random.default_rng(61)
    for spec in ([1.0, 4.0, 9.0], [2.0, 2.0, 5.0], [1.0, 1.0 + 1e-10, 4.0]):
        A, _ = spd(spec, rng)
        H, G = rand_sym(3, rng), rand_sym(3, rng)
        a = float(np.sum(SP.spectral_frechet(A, H, f, p) * G))
        b = float(np.sum(H * SP.spectral_vjp(A, G, f, p)))
        assert abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)


@pytest.mark.parametrize("f", ["sqrt", "log", "exp", "inv_sqrt"])
def test_ts6_vjp_vs_fd_gradient_independent(f: str) -> None:
    rng = np.random.default_rng(62)
    A, _ = spd([1.5, 2.0, 6.0], rng)
    G = rand_sym(3, rng)
    g = SP.spectral_vjp(A, G, f)
    h = 1e-6
    basis = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0 if i == j else 1.0 / math.sqrt(2.0)
            basis.append(E)
    for E in basis:
        fd = (np.sum(G * scipy_value(A + h * E, f, None)) - np.sum(G * scipy_value(A - h * E, f, None))) / (2 * h)
        assert abs(np.sum(g * E) - fd) <= 1e-6 * max(abs(fd), np.abs(g).max())


# ================================================================================================ TS7 scales / cond
@pytest.mark.parametrize("s", [1e-12, 1e-6, 1.0, 1e6, 1e12])
def test_ts7_scale_laws(s: float) -> None:
    rng = np.random.default_rng(71)
    B, _ = spd([1.0, 10.0, 100.0], rng)
    H = rand_sym(3, rng)
    # scaling laws of the kernel (consistency, not independent): sqrt(sB) = √s sqrt(B), log(sB) = log s I + log B
    assert rel(SP.spectral_function(s * B, "sqrt"), math.sqrt(s) * SP.spectral_function(B, "sqrt")) <= 1e-10
    lg = SP.spectral_function(s * B, "log") - math.log(s) * np.eye(3)
    assert np.abs(lg - SP.spectral_function(B, "log")).max() <= 1e-10 * np.abs(SP.spectral_function(B, "log")).max()
    assert rel(SP.spectral_frechet(s * B, H, "sqrt"), SP.spectral_frechet(B, H, "sqrt") / math.sqrt(s)) <= 1e-10
    assert rel(SP.spectral_frechet(s * B, H, "log"), SP.spectral_frechet(B, H, "log") / s) <= 1e-10


@pytest.mark.parametrize("kappa", [1.0, 1e2, 1e4, 1e8])
@pytest.mark.parametrize("f", ["sqrt", "log", "inv_sqrt"])
def test_ts7_condition_sweep_vs_mpmath(kappa: float, f: str) -> None:
    rng = np.random.default_rng(72)
    A, _ = spd([1.0, math.sqrt(kappa), kappa], rng)
    H = rand_sym(3, rng)
    assert rel(SP.spectral_frechet(A, H, f), mp_dir(A, H, f)) <= 1e-12 + 1e-14 * kappa


def test_ts7_condition_1e12_reported_not_asserted(record_property) -> None:
    rng = np.random.default_rng(73)
    A, _ = spd([1.0, 1e6, 1e12], rng)
    H = rand_sym(3, rng)
    try:
        L = SP.spectral_frechet(A, H, "log")
        record_property("kappa_1e12_log_rel_err", rel(L, mp_log_dir_eigsy(A, H)))
        assert np.isfinite(L).all()
    except ValueError:
        record_property("kappa_1e12_log_rel_err", "refused (eigenvalue <= 0)")


# ================================================================================================ TS8 invalid input
@pytest.mark.parametrize("f,p", [("log", None), ("sqrt", None), ("inv_sqrt", None), ("power", 0.5)])
@pytest.mark.parametrize("spec", [[1.0, 2.0, 0.0], [1.0, 2.0, -1e-30], [-1.0, 2.0, 3.0]])
def test_ts8_not_spd_refused(f: str, p: float | None, spec) -> None:
    A = np.diag(spec)
    with pytest.raises(ValueError, match="positive definite"):
        SP.spectral_function(A, f, p)
    with pytest.raises(ValueError, match="positive definite"):
        SP.spectral_frechet(A, np.eye(3), f, p)
    with pytest.raises(ValueError, match="positive definite"):
        SP.spectral_vjp(A, np.eye(3), f, p)


def test_ts8_bad_arguments() -> None:
    A = np.diag([1.0, 2.0, 3.0])
    I3 = np.eye(3)
    bad_A = [np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.full((3, 3), np.nan), np.eye(4),
             np.ones(3), np.float64(2.0), np.ones((2, 3)), np.zeros((0, 3, 3))]
    for a in bad_A:
        with pytest.raises(ValueError):
            SP.spectral_function(a, "exp")
    with pytest.raises(ValueError, match="symmetric"):
        SP.spectral_frechet(A, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "sqrt")
    with pytest.raises(ValueError):
        SP.spectral_frechet(A, np.eye(2), "sqrt")
    with pytest.raises(ValueError, match="f must be"):
        SP.spectral_function(A, "sin")
    with pytest.raises(ValueError, match="exponent"):
        SP.spectral_function(A, "power")
    with pytest.raises(ValueError, match="only used"):
        SP.spectral_function(A, "log", 2.0)
    with pytest.raises(ValueError):
        SP.spectral_function(A, "power", float("nan"))
    assert SP.spectral_function(I3, "exp").shape == (3, 3)


def test_ts8_batch_broadcast_equals_loop() -> None:
    rng = np.random.default_rng(81)
    A = np.array([spd(np.exp(rng.uniform(0, 3, 3)), rng)[0] for _ in range(5)])
    H1 = rand_sym(3, rng)
    H5 = np.array([rand_sym(3, rng) for _ in range(5)])
    H25 = np.array([[rand_sym(3, rng) for _ in range(5)] for _ in range(2)])
    H21 = np.array([[rand_sym(3, rng)] for _ in range(2)])
    for f, p in CASES[:4]:
        assert SP.spectral_function(A, f, p).shape == (5, 3, 3)
        assert np.allclose(SP.spectral_function(A, f, p), np.array([SP.spectral_function(a, f, p) for a in A]),
                           rtol=1e-14, atol=0)
        assert np.allclose(SP.spectral_frechet(A, H1, f, p), np.array([SP.spectral_frechet(a, H1, f, p) for a in A]),
                           rtol=1e-13, atol=0)
        assert np.allclose(SP.spectral_frechet(A, H5, f, p),
                           np.array([SP.spectral_frechet(a, h, f, p) for a, h in zip(A, H5)]), rtol=1e-13, atol=0)
        out = SP.spectral_frechet(A, H21, f, p)
        assert out.shape == (2, 5, 3, 3)
        assert np.allclose(out[1, 3], SP.spectral_frechet(A[3], H21[1, 0], f, p), rtol=1e-13, atol=0)
        assert SP.spectral_frechet(A, H25, f, p).shape == (2, 5, 3, 3)


# ================================================================================================ TS9 2 x 2
@pytest.mark.parametrize("f,p", CASES)
def test_ts9_two_by_two_vs_mpmath(f: str, p: float | None) -> None:
    rng = np.random.default_rng(91)
    for spec in ([1.0, 4.0], [2.0, 2.0], [2.0, 2.0 + 1e-10], [0.3, 40.0]):
        A, _ = spd(spec, rng)
        H = rand_sym(2, rng)
        assert rel(SP.spectral_function(A, f, p), mp_value(A, f, p)) <= 1e-12
        tol = 1e-10 if spec[1] < 2.0 + 1e-9 and spec[1] > 2.0 else 1e-12
        assert rel(SP.spectral_frechet(A, H, f, p), mp_dir(A, H, f, p)) <= tol
