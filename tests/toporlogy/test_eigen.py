"""WP05 L06 (eigen part): reliability checks for generalized symmetric eigenproblems and the buckling pencil
(``docs/foundations/prereg/L6e_eigen.md``, ``docs/foundations/EIGEN_RELIABILITY.md``).

References are written here: dense ``scipy.linalg.eigh``, Kronecker structure of the membrane, closed forms of the
beam (roots by ``brentq``), a 2-DOF buckling determinant, fixed-eigenvector families for the crossings. The only
second call to a repository kernel is E5f, labelled as a regression. All data synthetic (model units).
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.sparse as sp

import toporlogy as T
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.solvers.eigen import (ShiftError, buckling_load_factors, eigen_report, mac, shift_invert_modes,
                                     subspace_angles, track_modes)

SRC = Path(__file__).resolve().parents[2] / "src" / "toporlogy" / "solvers" / "eigen.py"


# ------------------------------------------------------------------------------------------------ E0
def test_e0_layer_rule_no_fem_or_torch_import() -> None:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    bad = [m for m in mods if m.startswith("toporlogy.fem") or m.split(".")[0] == "torch"]
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ E1
def _dense_pair(n: int, seed: int):
    rng = np.random.default_rng(seed)
    A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n), B @ B.T + n * np.eye(n)


def test_e1a_report_on_dense_eigh() -> None:
    K, M = _dense_pair(30, 1)
    lam, Phi = sla.eigh(K, M)
    rep = eigen_report(K, M, lam, Phi)
    # independent evaluation of the definitions
    nK, nM = np.linalg.norm(K, 2), np.linalg.norm(M, 2)
    eta = np.array([np.linalg.norm(K @ Phi[:, i] - lam[i] * M @ Phi[:, i]) /
                    ((nK + abs(lam[i]) * nM) * np.linalg.norm(Phi[:, i])) for i in range(30)])
    print(f"\n  E1a: max eta {rep.max_residual:.2e}, orth {rep.orth_error:.2e}, max r {rep.max_rayleigh:.2e}")
    assert rep.max_residual <= 1e-13 and rep.orth_error <= 1e-12 and rep.max_rayleigh <= 1e-13
    assert np.abs(rep.residual - eta).max() <= 1e-15
    assert rep.passed()


def _chain(n: int, seed: int):
    rng = np.random.default_rng(seed)
    d = 2.0 + rng.random(n)
    K = sp.diags([-np.ones(n - 1), d, -np.ones(n - 1)], [-1, 0, 1]).tocsr()
    M = sp.diags([0.1 * np.ones(n - 1), 1.0 + rng.random(n), 0.1 * np.ones(n - 1)], [-1, 0, 1]).tocsr()
    return K, M


def test_e1b_shift_invert_vs_dense() -> None:
    K, M = _chain(80, 2)
    res = shift_invert_modes(K, M, 6, sigma=-0.5)
    lam_d, Phi_d = sla.eigh(K.toarray(), M.toarray())
    rel = np.abs(res.lam - lam_d[:6]) / lam_d[:6]
    macd = np.diag(mac(res.Phi, Phi_d[:, :6], M))
    print(f"\n  E1b: lam rel {rel.max():.2e}, min MAC {macd.min():.15f}, max eta {res.report.max_residual:.2e}")
    assert rel.max() <= 1e-9 and macd.min() >= 1 - 1e-9 and res.report.max_residual <= 1e-10
    assert not res.cluster_cut and res.n_zero == 0


def test_e1c_negative_controls_of_the_report() -> None:
    K, M = _dense_pair(30, 1)
    lam, Phi = sla.eigh(K, M)
    P1 = Phi.copy()
    rng = np.random.default_rng(5)
    d = rng.standard_normal(30)
    P1[:, 0] += 1e-6 * np.linalg.norm(P1[:, 0]) * d / np.linalg.norm(d)
    r1 = eigen_report(K, M, lam, P1)
    lam2 = lam.copy()
    lam2[10] *= 1 + 1e-6
    r2 = eigen_report(K, M, lam2, Phi)
    r3 = eigen_report(K, M, lam, 2.0 * Phi)
    print(f"\n  E1c: eta_0 {r1.residual[0]:.2e}, r_10 {r2.rayleigh[10]:.2e}, orth(2 Phi) {r3.orth_error:.2f}")
    assert r1.residual[0] > 1e-8 and r2.rayleigh[10] > 1e-8 and r3.orth_error > 1.0
    assert not r1.passed() and not r2.passed() and not r3.passed()


# ------------------------------------------------------------------------------------------------ E2
N1 = 12


def _membrane():
    h = 1.0 / (N1 + 1)
    e = np.ones(N1 - 1)
    K1 = (np.diag(2.0 * np.ones(N1)) - np.diag(e, 1) - np.diag(e, -1)) / h
    M1 = h * (np.diag(4.0 * np.ones(N1)) + np.diag(e, 1) + np.diag(e, -1)) / 6.0
    K = sp.csr_matrix(np.kron(K1, M1) + np.kron(M1, K1))
    M = sp.csr_matrix(np.kron(M1, M1))
    mu, psi = sla.eigh(K1, M1)
    return K, M, mu, psi


def _kron_reference(mu, psi, count: int):
    pairs = sorted(((mu[i] + mu[j], i, j) for i in range(N1) for j in range(N1)), key=lambda t: (t[0], t[1], t[2]))[:count]
    lam = np.array([p[0] for p in pairs])
    Phi = np.column_stack([np.kron(psi[:, i], psi[:, j]) for _, i, j in pairs])
    return lam, Phi


def test_e2_repeated_eigenvalues_membrane() -> None:
    K, M, mu, psi = _membrane()
    lam_ref, Phi_ref = _kron_reference(mu, psi, 6)
    res = shift_invert_modes(K, M, 6, sigma=-1.0)
    rel = np.abs(res.lam - lam_ref) / lam_ref
    assert rel.max() <= 1e-10                                                       # E2a
    assert res.report.clusters == [(0,), (1, 2), (3,), (4, 5)]
    assert not res.cluster_cut
    rep = eigen_report(K, M, res.lam, res.Phi, reference=(lam_ref, Phi_ref))        # E2b
    a1 = subspace_angles(res.Phi[:, 1:3], Phi_ref[:, 1:3], M).max()
    a2 = subspace_angles(res.Phi[:, 4:6], Phi_ref[:, 4:6], M).max()
    print(f"\n  E2: lam rel {rel.max():.2e}, angles {a1:.2e} {a2:.2e}, cluster residuals {rep.cluster_residual.max():.2e}")
    assert a1 <= 1e-8 and a2 <= 1e-8 and rep.max_cluster_angle <= 1e-8 and rep.cluster_residual.max() <= 1e-10
    # E2c negative control: a rotation inside the cluster changes the vectors, not the subspace
    c, s = math.cos(math.radians(37)), math.sin(math.radians(37))
    P = res.Phi.copy()
    P[:, 1:3] = res.Phi[:, 1:3] @ np.array([[c, -s], [s, c]])
    m_ind = mac(res.Phi[:, 1], P[:, 1], M)[0, 0]
    ang = subspace_angles(P[:, 1:3], res.Phi[:, 1:3], M).max()
    rr = eigen_report(K, M, res.lam, P)
    print(f"  E2c: individual MAC {m_ind:.3f}, subspace angle {ang:.2e}, residual {rr.max_residual:.2e}")
    assert m_ind <= 0.7 and ang <= 1e-8 and rr.max_residual <= 1e-10 and rr.cluster_residual.max() <= 1e-10


def test_e2d_cluster_cut_flag() -> None:
    K, M, _, _ = _membrane()
    cuts = {k: shift_invert_modes(K, M, k, sigma=-1.0).cluster_cut for k in (4, 5, 6)}
    print(f"\n  E2d: cluster_cut by k: {cuts}")
    assert cuts == {4: False, 5: True, 6: False}


# ------------------------------------------------------------------------------------------------ E3
EI, EA_, RHO = 2e-4, 1.0, 1.0


def _beam(nel: int = 20, clamped: bool = False):
    L = 1.0
    h = L / nel
    n = 3 * (nel + 1)
    K, M = np.zeros((n, n)), np.zeros((n, n))
    ka = EA_ / h * np.array([[1, -1], [-1, 1]])
    ma = RHO * h / 6 * np.array([[2, 1], [1, 2]])
    kb = EI / h ** 3 * np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                 [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
    mb = RHO * h / 420 * np.array([[156, 22 * h, 54, -13 * h], [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                                   [54, 13 * h, 156, -22 * h], [-13 * h, -3 * h * h, -22 * h, 4 * h * h]])
    for e in range(nel):
        a, b = 3 * e, 3 * (e + 1)
        ia, ib = [a, b], [a + 1, a + 2, b + 1, b + 2]
        K[np.ix_(ia, ia)] += ka
        M[np.ix_(ia, ia)] += ma
        K[np.ix_(ib, ib)] += kb
        M[np.ix_(ib, ib)] += mb
    x = np.linspace(0.0, L, nel + 1)
    rigid = np.zeros((n, 3))
    rigid[0::3, 0] = 1.0
    rigid[1::3, 1] = 1.0
    rigid[1::3, 2] = x
    rigid[2::3, 2] = 1.0
    if clamped:
        K, M = K[3:, 3:], M[3:, 3:]
    return sp.csr_matrix(K), sp.csr_matrix(M), rigid, K.shape[0]


def _roots(sign: float, brackets):
    return np.array([sopt.brentq(lambda x: math.cos(x) * math.cosh(x) - sign, a, b, xtol=1e-14) for a, b in brackets])


def test_e3_free_free_beam_zero_modes() -> None:
    K, M, rigid, n = _beam(20)
    w = np.linalg.eigvalsh(K.toarray())
    n_ref = int((np.abs(w) <= 1e-9 * np.linalg.norm(K.toarray(), 2)).sum())
    assert n_ref == 3                                                               # independent count
    res = shift_invert_modes(K, M, 6, assume_psd=True, expected_zero=3)
    print(f"\n  E3a: sigma {res.sigma:.3e}, n_zero {res.n_zero}, zero residual {res.zero_residual:.2e}, "
          f"max|lam_0| {np.abs(res.lam[:3]).max():.2e}, s {res.scale:.3e}")
    assert res.n_zero == 3 and res.zero_count_ok is True and res.zero_residual <= 1e-10
    assert np.abs(res.lam[res.zero_idx]).max() <= 1e-9 * res.scale
    ang = subspace_angles(res.Phi[:, res.zero_idx], rigid, M).max()                 # E3b
    print(f"  E3b: angle to the rigid space {ang:.2e}")
    assert ang <= 1e-8
    x = _roots(1.0, [(4, 5), (7, 8), (10, 11)])                                     # E3c
    ref = x ** 4 * EI
    rel = np.abs(res.lam[3:6] - ref) / ref
    print(f"  E3c: bending rel errors {rel}")
    assert rel.max() <= 1e-3
    # E3d negative controls
    with pytest.raises(ShiftError):
        shift_invert_modes(K, M, 6, sigma=0.0)
    assert shift_invert_modes(K, M, 6, assume_psd=True, expected_zero=2).zero_count_ok is False
    with pytest.raises(ValueError):
        shift_invert_modes(K, M, 6)


def test_e3e_clamped_beam_zero_shift_allowed() -> None:
    K, M, _, _ = _beam(20, clamped=True)
    res = shift_invert_modes(K, M, 3, sigma=0.0)
    x = _roots(-1.0, [(1.5, 2.5), (4, 5), (7, 8)])
    ref = x ** 4 * EI
    rel = np.abs(res.lam - ref) / ref
    print(f"\n  E3e: clamped bending rel errors {rel}, n_zero {res.n_zero}")
    assert res.n_zero == 0 and rel.max() <= 1e-3


def test_e3f_shift_refusal_on_known_spectrum() -> None:
    K, M = sp.diags(np.arange(1.0, 61.0)).tocsr(), sp.identity(60, format="csr")
    with pytest.raises(ShiftError):
        shift_invert_modes(K, M, 5, sigma=3.0)
    with pytest.raises(ShiftError):
        shift_invert_modes(K, M, 5, sigma=3.0 * (1 + 1e-12))
    res = shift_invert_modes(K, M, 5, sigma=3.0 * (1 + 1e-3))
    print(f"\n  E3f: accepted at 3.003, lam {res.lam}, distance bound {res.shift_distance:.3e}")
    assert np.abs(res.lam - np.arange(1.0, 6.0)).max() <= 1e-9


# ------------------------------------------------------------------------------------------------ E4
def _family(lam_curves, ts, M, seed):
    """K(t) = M Psi diag(lam(t)) Psi^T M with fixed M-orthonormal Psi: exact eigenvalues, eigenvectors Psi columns."""
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    w, U = np.linalg.eigh(M)
    Psi = U @ np.diag(w ** -0.5) @ U.T @ Q
    assert np.abs(Psi.T @ M @ Psi - np.eye(n)).max() < 1e-13
    lams, Phis = [], []
    for t in ts:
        Kt = M @ Psi @ np.diag([c(t) for c in lam_curves]) @ Psi.T @ M
        l, P = sla.eigh(0.5 * (Kt + Kt.T), M)
        lams.append(l)
        Phis.append(P)
    return Psi, lams, Phis


def test_e4a_e4b_two_dof_crossing() -> None:
    M = np.array([[2.0, 0.5], [0.5, 1.0]])
    ts = np.linspace(0.0, 2.0, 8)
    curves = [lambda t: 1 + t, lambda t: 3 - t]
    Psi, lams, Phis = _family(curves, ts, M, 7)
    tr = track_modes(lams, Phis, M)
    worst = min(mac(Phis[s][:, tr.order[s][t]], Psi[:, t], M)[0, 0] for s in range(8) for t in range(2))
    print(f"\n  E4a: worst MAC to Psi {worst:.15f}, min consecutive MAC {tr.mac_to_previous.min():.15f}, "
          f"crossings {[(c.step, c.track_a, c.track_b) for c in tr.crossings]}")
    assert worst >= 1 - 1e-10 and tr.mac_to_previous.min() >= 1 - 1e-10
    assert [(c.step, c.track_a, c.track_b) for c in tr.crossings] == [(4, 0, 1)] and not tr.ambiguous
    assert np.abs(tr.lam[:, 0] - (1 + ts)).max() <= 1e-12 and np.abs(tr.lam[:, 1] - (3 - ts)).max() <= 1e-12
    # E4b negative control: matching by sorted index
    m_idx = mac(Phis[3][:, 0], Phis[4][:, 0], M)[0, 0]
    dev = max(abs(lams[s][0] - (1 + ts[s])) for s in range(8))
    print(f"  E4b: index-matched MAC across the crossing {m_idx:.2e}, lam_0(index) deviates from 1 + t by {dev:.3f}")
    assert m_idx <= 1e-12 and dev > 0.5


def test_e4c_three_dof_three_crossings() -> None:
    rng = np.random.default_rng(3)
    B = rng.standard_normal((3, 3))
    M = B @ B.T + 3 * np.eye(3)
    ts = np.linspace(0.0, 5.0, 12)
    curves = [lambda t: 1 + t, lambda t: 2.5 + 0 * t, lambda t: 4.5 - 0.5 * t]
    Psi, lams, Phis = _family(curves, ts, M, 11)
    tr = track_modes(lams, Phis, M)
    got = [(c.step, c.track_a, c.track_b) for c in tr.crossings]
    print(f"\n  E4c: crossings {got}, ambiguous {tr.ambiguous}")
    assert got == [(4, 0, 1), (6, 0, 2), (9, 1, 2)] and not tr.ambiguous
    worst = min(mac(Phis[s][:, tr.order[s][t]], Psi[:, t], M)[0, 0] for s in range(12) for t in range(3))
    assert worst >= 1 - 1e-10


def test_e4d_no_crossing_family() -> None:
    M = np.array([[2.0, 0.5], [0.5, 1.0]])
    ts = np.linspace(0.0, 2.0, 6)
    _, lams, Phis = _family([lambda t: 1 + t, lambda t: 3 + t], ts, M, 7)
    tr = track_modes(lams, Phis, M)
    assert tr.crossings == [] and all(np.array_equal(o, [0, 1]) for o in tr.order)


# ------------------------------------------------------------------------------------------------ E5
K2 = np.array([[4.0, 1.0], [1.0, 3.0]])
KG2 = np.array([[-2.0, 1.0], [1.0, 1.0]])
LAM_P, LAM_N = (-4 + math.sqrt(148)) / 6, (-4 - math.sqrt(148)) / 6


def test_e5a_two_dof_closed_form() -> None:
    res = buckling_load_factors(K2, KG2, k=1)
    print(f"\n  E5a: load factor {res.load_factors[0]:.12f} (closed form {LAM_P:.12f}), reverse {res.reverse_load_factors}, "
          f"inertia {res.kg_inertia}")
    assert abs(res.load_factors[0] - LAM_P) <= 1e-12 * LAM_P
    assert res.kg_inertia == (1, 1, 0) and res.spd_evidence == "cholesky" and res.path == "dense"
    assert abs(res.reverse_load_factors[0] - abs(LAM_N)) <= 1e-12 * abs(LAM_N)
    assert abs(res.Phi[:, 0] @ K2 @ res.Phi[:, 0] - 1.0) <= 1e-12 and res.report.passed(1e-12, 1e-12, 1e-12)


def test_e5b_assumptions_enforced() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        buckling_load_factors(np.array([[1.0, 2.0], [2.0, 1.0]]), KG2)
    with pytest.raises(ValueError, match="symmetric"):
        buckling_load_factors(np.array([[4.0, 1.0], [0.0, 3.0]]), KG2)
    res = buckling_load_factors(K2, np.zeros((2, 2)))
    assert res.load_factors.size == 0 and res.notes


def test_e5c_wrong_sign_convention_is_detected() -> None:
    res = buckling_load_factors(K2, -KG2, k=1)
    print(f"\n  E5c: opposite convention gives {res.load_factors[0]:.6f} (correct {LAM_P:.6f})")
    assert abs(res.load_factors[0] - abs(LAM_N)) <= 1e-12 and abs(res.load_factors[0] - LAM_P) > 1.0


def _column(nb: int):
    shape, h = (nb, 40 * nb), 4.0 / nb
    d = T.Domain(shape, h)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] == 0))
    top = np.flatnonzero(np.isclose(X[:, 1], shape[1] * h))
    F = d.load_vector()
    w = np.where(np.isclose(X[top, 0], 0) | np.isclose(X[top, 0], shape[0] * h), 0.5, 1.0)
    F[d.dim * top + 1, 0] = -w / w.sum()                    # unit compressive resultant
    m = T.Elasticity(d, fixed, 0.0, None)
    E = np.full(d.n_active, 1000.0)
    U = m.solve_numpy(E, F)
    A = m.matrix(E)
    K = (A + A.T - sp.diags(A.diagonal())).tocsr()
    lb = LinearBuckling(m)
    KG = lb.geometric_matrix(E, U[:, 0])
    return m, lb, E, U, K, KG


@pytest.fixture(scope="module")
def columns():
    return {nb: _column(nb) for nb in (2, 4, 8)}


def test_e5d_euler_column(columns) -> None:
    pe = math.pi ** 2 * 1000.0 * 4.0 ** 3 / 12 / (4 * 160.0 ** 2)
    err = {}
    for nb, (_, _, _, _, K, KG) in columns.items():
        res = buckling_load_factors(K, KG, k=1)
        err[nb] = abs(res.load_factors[0] / pe - 1.0)
        assert res.path == "lanczos" and res.report.passed(1e-8, 1e-8, 1e-8)
    print(f"\n  E5d: |P_cr/P_Euler - 1| = {err}; spd evidence at nb=8: "
          f"{buckling_load_factors(columns[8][4], columns[8][5], 1).spd_evidence}")
    assert err[8] < 0.01 and err[8] < err[4] < err[2]


def test_e5e_dense_vs_lanczos_column(columns) -> None:
    _, _, _, _, K, KG = columns[2]
    res = buckling_load_factors(K, KG, k=3)
    mu_d = np.sort(sla.eigh(-KG.toarray(), K.toarray(), eigvals_only=True))[::-1][:3]
    rel = np.abs(res.mu - mu_d) / mu_d
    print(f"\n  E5e: Lanczos vs dense mu rel {rel}")
    assert res.path == "lanczos" and rel.max() <= 1e-8


def test_e5f_regression_against_linear_buckling_modes(columns) -> None:
    """Regression (second call to the repository kernel, same K and K_G), not an independent check."""
    m, lb, E, U, K, KG = columns[4]
    mu_repo, _ = lb.modes(E, E, U[:, 0], 1)
    res = buckling_load_factors(K, KG, k=1)
    assert abs(res.mu[0] - mu_repo[0]) <= 1e-8 * mu_repo[0]


# ------------------------------------------------------------------------------------------------ E6
def test_e6_invalid_input_rejected() -> None:
    K, M = _dense_pair(10, 0)
    lam, Phi = sla.eigh(K, M)
    with pytest.raises(ValueError):
        eigen_report(K[:, :5], M, lam, Phi)
    with pytest.raises(ValueError):
        eigen_report(K, M[:5, :5], lam, Phi)
    Kn = K.copy()
    Kn[0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        eigen_report(Kn, M, lam, Phi)
    Kx = K.copy()
    Kx[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        eigen_report(Kx, M, lam, Phi)
    with pytest.raises(ValueError):
        eigen_report(K, M, lam[:3], Phi)
    Ks, Ms = _chain(30, 1)
    with pytest.raises(ValueError, match="not symmetric positive definite"):
        shift_invert_modes(Ks, -Ms, 3, sigma=-1.0)
    with pytest.raises(ValueError, match="k must lie"):
        shift_invert_modes(Ks, Ms, 29, sigma=-1.0)
    with pytest.raises(ValueError, match="k must lie"):
        shift_invert_modes(Ks, Ms, 0, sigma=-1.0)
    with pytest.raises(ShiftError):
        shift_invert_modes(Ks, Ms, 3, sigma=float("nan"))
    with pytest.raises(ValueError):
        mac(Phi, Phi[:5], M)
    with pytest.raises(ValueError):
        subspace_angles(Phi[:, :2], Phi[:, :3], M)
    with pytest.raises(ValueError):
        track_modes([lam], [Phi], M)
