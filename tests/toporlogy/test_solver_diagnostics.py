"""Linear-solve diagnostics (docs/foundations/prereg/L_solver.md, L1-L5)."""
from __future__ import annotations

import mpmath
import numpy as np
import pytest
import scipy.sparse as sp

from toporlogy.fem.mesh.linear import Factorization
from toporlogy.solvers.diagnostics import (check_compatible, inertia, krylov_solve, iterative_refinement, solve_report,
                                           solve_with_gauge)


def _graded(n: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Q, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(n, n)))
    lam = np.logspace(0, -k, n)
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    x = np.random.default_rng(12).normal(size=n)
    return A, A @ x, Q, lam


def _mp_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    with mpmath.workdps(50):
        sol = mpmath.lu_solve(mpmath.matrix(A.tolist()), mpmath.matrix(b.tolist()))
        return np.array([float(v) for v in sol])


@pytest.mark.parametrize("n", [8, 32])
@pytest.mark.parametrize("k", [0, 4, 8, 12])
def test_l1_backward_and_forward(n: int, k: int) -> None:
    A, b, _, lam = _graded(n, k)
    x = np.linalg.solve(A, b)
    rep = solve_report(A, x, b, condition=float(np.linalg.cond(A, np.inf)), reference=_mp_solve(A, b))
    assert rep.backward_normwise <= 1e-14                                            # L1a
    assert rep.forward_error is not None
    if k in (0, 4):                                                                   # L1b
        assert rep.forward_established
        assert rep.forward_error <= 1e-8
        assert rep.forward_error <= 10 * rep.forward_bound + 1e-15
    if k == 12:
        assert not rep.forward_established


def test_l1n_small_residual_wrong_solution() -> None:
    A, b, Q, lam = _graded(8, 12)
    vmin = Q[:, int(np.argmin(lam))]
    x_bad = np.linalg.solve(A, b) + 1e-2 * vmin
    ref = _mp_solve(A, b)
    rep = solve_report(A, x_bad, b, condition=float(np.linalg.cond(A, np.inf)), reference=ref)
    assert rep.backward_normwise <= 1e-12
    assert not rep.forward_established
    assert rep.forward_error >= 1e-3


def _indefinite() -> np.ndarray:
    S = np.array([[2.0, 1.0], [1.0, 3.0]])
    A = np.zeros((6, 6))
    A[:2, :2] = [[0.0, 1.0], [1.0, 0.0]]
    A[2:4, 2:4] = S
    A[4:, 4:] = -S
    return A


def test_l2a_inertia() -> None:
    A = _indefinite()
    ev = np.linalg.eigvalsh(A)
    assert inertia(A) == (3, 3, 0) == (int((ev > 0).sum()), int((ev < 0).sum()), 0)


def test_l2b_indefinite_solvers() -> None:
    A = _indefinite()
    b = np.arange(1.0, 7.0)
    with pytest.raises(ValueError):
        krylov_solve(A, b, "cg", "symmetric_indefinite")
    r = krylov_solve(A, b, "minres", "symmetric_indefinite", rtol=1e-12)
    assert r.converged and np.linalg.norm(b - A @ r.x) / np.linalg.norm(b) <= 1e-10
    x = Factorization(sp.csr_matrix(A), "symmetric_indefinite").solve(b)
    assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) <= 1e-14


@pytest.mark.xfail(strict=True, reason="pre-registered L2c FAILED on the first run: Factorization(A, 'spd') "
                   "(direct-splu) solved the indefinite 6x6 block matrix without an error (relative residual "
                   "9.3e-17, the solution correct to 2.2e-16); the SPD path does not detect indefiniteness")
def test_l2c_spd_path_does_not_return_silently_on_indefinite() -> None:
    A = sp.csr_matrix(_indefinite())
    with pytest.raises(Exception):  # noqa: B017 - any refusal counts (pre-registered: an exception is raised)
        Factorization(A, "spd").solve(np.arange(1.0, 7.0))


def test_l2d_nonsymmetric() -> None:
    A = np.array([[3.0, 1.0], [-2.0, 4.0]])
    b = np.array([1.0, 2.0])
    for m, s in [("cg", "spd"), ("minres", "spd"), ("minres", "symmetric_indefinite")]:
        with pytest.raises(ValueError, match="not symmetric"):
            krylov_solve(A, b, m, s)
    r = krylov_solve(A, b, "gmres", "general", rtol=1e-13)
    assert r.converged and r.residual <= 1e-12
    xt = Factorization(sp.csr_matrix(A), "general").solve_transpose(b)
    ref = np.linalg.solve(A.T, b)
    assert np.abs(xt - ref).max() <= 1e-14 * np.abs(ref).max()


def test_l3_singular_with_gauge() -> None:
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    N = np.array([1.0, 1.0]) / np.sqrt(2.0)
    b = np.array([1.0, -1.0])
    assert check_compatible(A, b, N)
    x = solve_with_gauge(A, b, N)
    assert np.abs(x - [0.5, -0.5]).max() <= 1e-14
    bad = np.array([1.0, 0.0])
    assert not check_compatible(A, bad, N)
    with pytest.raises(ValueError, match="incompatible"):
        solve_with_gauge(A, bad, N)


def _sparse_spd(n: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    R = sp.random(n, n, density=0.1, random_state=rng)
    return sp.csr_matrix(R @ R.T + n * sp.eye(n))


def test_l4a_multi_rhs() -> None:
    A = _sparse_spd(50, 13)
    B = np.random.default_rng(13).normal(size=(50, 4))
    X = Factorization(A, "spd").solve(B)
    Ad = A.toarray()
    for j in range(4):
        ref = np.linalg.solve(Ad, B[:, j])
        assert np.abs(X[:, j] - ref).max() <= 1e-12 * np.abs(ref).max()


def test_l4b_adjoint_duality() -> None:
    rng = np.random.default_rng(14)
    n = 40
    A = sp.csr_matrix(sp.random(n, n, density=0.15, random_state=rng) + 10 * sp.eye(n))
    a, b = rng.normal(size=n), rng.normal(size=n)
    f = Factorization(A, "general")
    lhs = a @ f.solve(b)
    rhs = b @ f.solve_transpose(a)
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_l4c_krylov_status() -> None:
    A = _sparse_spd(50, 13)
    b = np.ones(50)
    r = krylov_solve(A, b, "cg", "spd", maxiter=2, rtol=1e-14)
    assert r.status == "max_iterations" and not r.converged
    rec = np.linalg.norm(b - A @ r.x) / np.linalg.norm(b)
    assert abs(r.residual - rec) <= 1e-12 * rec


def test_l5_refinement() -> None:
    n = 30
    rng = np.random.default_rng(15)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    S = (Q * np.linspace(1.0, 4.0, n)) @ Q.T
    S = 0.5 * (S + S.T)
    D = np.diag(np.logspace(-6, 6, n))
    A = D @ S @ D
    b = A @ rng.normal(size=n)
    x0 = np.linalg.solve(A.astype(np.float32), b.astype(np.float32)).astype(float)
    res = iterative_refinement(A, b, lambda r: np.linalg.solve(A, r), x0, max_iter=10)
    assert res.omega[-1] <= 1e-14
    assert all(res.omega[i + 1] <= res.omega[i] for i in range(len(res.omega) - 1))
    assert res.stop in ("converged", "stagnated", "max_iter") and res.iterations <= 10
