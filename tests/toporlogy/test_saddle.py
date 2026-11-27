"""WP05 L06 — saddle-point block operators, Schur complement, block preconditioning
(``docs/foundations/prereg/L6_saddle.md``, ``docs/foundations/SADDLE_POINT.md``).

Dense references are written here with numpy (``inv``, ``solve``, ``eigh``); the FE sweep compares the preconditioned
Krylov solution with ``Factorization(K, "symmetric_indefinite")``. All materials synthetic (E = 1, unit square).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from toporlogy.fem.mesh import Dirichlet, rectangle
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.fem.mesh.manufactured import Manufactured
from toporlogy.fem.mesh.mesh import perturb
from toporlogy.fem.mesh.mixed import MixedKG
from toporlogy.solvers.diagnostics import condition_1norm_estimate, inertia
from toporlogy.solvers.saddle import (SaddleOperator, block_diagonal_preconditioner, block_triangular_preconditioner,
                                      factorised_solver, saddle_solve)

RTOL = 1e-8
RES_TOL = 10 * RTOL


# ------------------------------------------------------------------------------------------------ dense systems
def _dense(c_kind: str, n: int = 14, m: int = 5, gauge: bool = False, seed: int = 3):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.linspace(1.0, 10.0, n)) @ Q.T
    A = 0.5 * (A + A.T)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, m)))
    B = U @ np.diag(np.linspace(1.0, 3.0, m)) @ V.T
    if gauge:
        B = (np.eye(m) - np.ones((m, m)) / m) @ B
    R = rng.standard_normal((m, m))
    C = None if c_kind == "zero" else 0.3 * (R @ R.T / m + np.eye(m))
    return A, B, C, rng


def _K(A, B, C):
    m = B.shape[0]
    return np.block([[A, B.T], [B, -(np.zeros((m, m)) if C is None else C)]])


@pytest.mark.parametrize("c_kind", ["zero", "spd"])
def test_l6a1_to_a4_block_operator_schur_and_inertia(c_kind):
    A, B, C, rng = _dense(c_kind)
    n, m = A.shape[0], B.shape[0]
    K = _K(A, B, C)
    op = SaddleOperator(sp.csr_matrix(A), sp.csr_matrix(B), None if C is None else sp.csr_matrix(C))
    v = rng.standard_normal(n + m)
    e1 = np.abs(op.matvec(v) - K @ v).max() / np.abs(K @ v).max()
    Ai = np.linalg.inv(A)
    S = (0 if C is None else C) + B @ Ai @ B.T
    e2 = 0.0
    for _ in range(3):
        q = rng.standard_normal(m)
        e2 = max(e2, np.abs(op.schur_apply(q) - S @ q).max() / np.abs(S @ q).max())
    xs = rng.standard_normal(n + m)
    b = K @ xs
    f, g = b[:n], b[n:]
    red = B @ Ai @ f - g
    e3a = np.abs(op.reduced_rhs(f, g) - red).max() / np.abs(red).max()
    p = np.linalg.solve(S, red)                              # dense Schur solve
    u = op.recover_u(f, p)
    e3b = max(np.abs(u - xs[:n]).max() / np.abs(xs[:n]).max(), np.abs(p - xs[n:]).max() / np.abs(xs[n:]).max())
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    inr = inertia(K)
    print(f"\n  L6a C={c_kind}: matvec {e1:.1e}, schur_apply {e2:.1e}, reduced rhs {e3a:.1e}, (u,p) {e3b:.1e}, "
          f"min eig S {ev[0]:.3f}, inertia {inr}")
    assert e1 <= 1e-14
    assert e2 <= 1e-12
    assert e3a <= 1e-12 and e3b <= 1e-10
    assert ev[0] > 0
    assert inr == (n, m, 0)


def test_l6a5_exact_blocks_terminate_in_a_handful_of_iterations():
    out = {}
    for c_kind in ("zero", "spd"):
        A, B, C, rng = _dense(c_kind)
        n, m = A.shape[0], B.shape[0]
        K = _K(A, B, C)
        xs = rng.standard_normal(n + m)
        b = K @ xs
        S = (0 if C is None else C) + B @ np.linalg.inv(A) @ B.T
        op = SaddleOperator(sp.csr_matrix(A), sp.csr_matrix(B), None if C is None else sp.csr_matrix(C))
        S_solve = lambda v, S=S: np.linalg.solve(S, v)       # noqa: E731
        rd = saddle_solve(op, b[:n], b[n:], "diagonal", S_hat=S_solve, rtol=1e-10)
        rt = saddle_solve(op, b[:n], b[n:], "triangular", S_hat=S_solve, rtol=1e-10)
        out[c_kind] = (rd.iterations, rt.iterations)
        print(f"\n  L6a5 C={c_kind}: MINRES(diag) {rd.iterations} its res {rd.residual:.1e} err "
              f"{np.abs(np.r_[rd.u, rd.p] - xs).max():.1e}; GMRES(tri) {rt.iterations} its res {rt.residual:.1e} err "
              f"{np.abs(np.r_[rt.u, rt.p] - xs).max():.1e}")
        for r in (rt, rd) if c_kind == "zero" else (rt,):
            assert r.converged and r.residual <= 1e-9
            assert np.abs(np.r_[r.u, r.p] - xs).max() <= 1e-8 * np.abs(xs).max()
        assert rt.iterations <= 3
    assert out["zero"][0] <= 4


def test_l6a6_invalid_input_is_rejected():
    A, B, C, rng = _dense("zero")
    n, m = A.shape[0], B.shape[0]
    with pytest.raises(ValueError, match="block shapes"):
        SaddleOperator(sp.csr_matrix(A), sp.csr_matrix(B[:, :-1]))
    with pytest.raises(ValueError, match="C has shape"):
        SaddleOperator(sp.csr_matrix(A), sp.csr_matrix(B), sp.eye(m + 1))
    op = SaddleOperator(sp.csr_matrix(A), sp.csr_matrix(B))
    with pytest.raises(ValueError, match="f must have"):
        saddle_solve(op, np.zeros(n + 1), np.zeros(m), "none")
    with pytest.raises(ValueError, match="preconditioner"):
        saddle_solve(op, np.zeros(n), np.zeros(m), "jacobi")
    with pytest.raises(ValueError, match="S_hat"):
        saddle_solve(op, np.zeros(n), np.zeros(m), "diagonal")
    with pytest.raises(ValueError, match="kind"):
        factorised_solver(sp.eye(3), "amg")


def test_l6a7_gauge_dense_projection_matches_the_bordered_system():
    A, B, _, rng = _dense("zero", gauge=True)
    n, m = A.shape[0], B.shape[0]
    z = np.ones(m)
    w0 = rng.uniform(0.5, 2.0, m)
    f = rng.standard_normal(n)
    g = B @ rng.standard_normal(n)
    K = _K(A, B, None)
    e = np.concatenate([np.zeros(n), w0])
    Kb = np.block([[K, e[:, None]], [e[None, :], np.zeros((1, 1))]])
    xb = np.linalg.solve(Kb, np.concatenate([f, g, [0.0]]))
    op = SaddleOperator(sp.csr_matrix(A), sp.csr_matrix(B))
    r = saddle_solve(op, f, g, "diagonal", S_hat=sp.eye(m), rtol=1e-12, null_vector=z, gauge_weights=w0)
    err = np.abs(np.r_[r.u, r.p] - xb[:n + m]).max() / np.abs(xb[:n + m]).max()
    print(f"\n  L6a7: {r.status}, {r.iterations} its, residual {r.residual:.1e}, vs bordered {err:.1e}, w0.p {w0 @ r.p:.1e}")
    assert r.converged and r.gauge
    assert err <= 1e-9
    assert abs(w0 @ r.p) <= 1e-12 * np.linalg.norm(w0) * np.linalg.norm(r.p)
    with pytest.raises(ValueError, match="incompatible"):
        saddle_solve(op, f, g + 1.0, "diagonal", S_hat=sp.eye(m), null_vector=z, gauge_weights=w0)
    with pytest.raises(ValueError, match="null vector of the pressure coupling"):
        saddle_solve(op, f, g, "diagonal", S_hat=sp.eye(m), null_vector=rng.standard_normal(m))
    with pytest.raises(ValueError, match="gauge functional"):
        saddle_solve(op, f, g, "diagonal", S_hat=sp.eye(m), null_vector=z,
                     gauge_weights=np.array([1.0, -1.0, 0.0, 0.0, 0.0]))


# ------------------------------------------------------------------------------------------------ FE helpers
def kg(nu: float, E: float = 1.0) -> tuple[float, float, float]:
    G = E / (2 * (1 + nu))
    if nu == 0.5:
        return G, math.inf, math.inf
    K = E / (3 * (1 - 2 * nu))
    return G, K, K - 2 * G / 3


def clamp(m):
    return Dirichlet.fixed(m.dofs(m.boundary_nodes()))


@functools.cache
def _case(element: str, nu: float, n: int):
    """Free-DOF blocks of a clamped unit-square MixedKG, a manufactured load f and a compatible g = B w."""
    G, K, lam = kg(nu)
    m = rectangle(n, n, element=element)
    m = perturb(m, 0.2 / n, seed=200 + n)
    ms = Manufactured.incompressible_limit(2, G, lam)
    mx = MixedKG(m, G, K)
    f_full = mx.disp.body_force(lambda X: ms.body_force(G, X), degree=2 * m.ref.degree + 4)
    bc = clamp(m)
    free = np.setdiff1d(mx.disp.active, bc.dofs)
    A = sp.csr_matrix(mx.A[free][:, free])
    B = sp.csr_matrix(mx.B[:, free])
    M = sp.csr_matrix(mx.M)
    f = np.asarray(f_full).reshape(len(f_full), -1)[free, 0]
    g = B @ np.random.default_rng(7).standard_normal(len(free))
    g *= np.linalg.norm(f) / np.linalg.norm(g)
    C = None if math.isinf(K) else sp.csr_matrix(M / K)
    c_hat = 1.0 / G + (0.0 if math.isinf(K) else 1.0 / K)
    return {"A": A, "B": B, "M": M, "C": C, "f": f, "g": g, "G": G, "K": K, "c": c_hat, "gauge": math.isinf(K)}


def _test_K(cs):
    """The assembled saddle matrix built in the test (not by the module)."""
    Cn = None if cs["C"] is None else -cs["C"]
    return sp.bmat([[cs["A"], cs["B"].T], [cs["B"], Cn]], format="csr")


def _solve(cs, preconditioner="diagonal", **kw):
    op = SaddleOperator(cs["A"], cs["B"], cs["C"])
    S_hat = cs["c"] * cs["M"]
    extra = {}
    if cs["gauge"]:
        extra = {"null_vector": np.ones(cs["M"].shape[0]), "gauge_weights": cs["M"] @ np.ones(cs["M"].shape[0])}
    return saddle_solve(op, cs["f"], cs["g"], preconditioner, S_hat=S_hat, rtol=RTOL, **extra, **kw)


NUS = (0.3, 0.49, 0.4999, 0.5)
NS = (8, 16, 32)


@functools.cache
def _run(element: str, nu: float, n: int):
    cs = _case(element, nu, n)
    r = _solve(cs)
    Kt = _test_K(cs)
    b = np.concatenate([cs["f"], cs["g"]])
    x = np.concatenate([r.u, r.p])
    rr = b - Kt @ x
    nn = cs["A"].shape[0]
    out = {"res": float(np.linalg.norm(rr) / np.linalg.norm(b)),
           "res_u": float(np.linalg.norm(rr[:nn]) / np.linalg.norm(b)),
           "res_p": float(np.linalg.norm(rr[nn:]) / np.linalg.norm(b)), "its": r.iterations, "status": r.status,
           "restarts": r.restarts, "gauge": r.gauge, "nA": r.n_A_solves, "nS": r.n_S_solves}
    m = cs["M"].shape[0]
    one = np.ones(m)
    w = cs["M"] @ one
    out["wp"] = float(abs(w @ r.p) / (np.linalg.norm(w) * np.linalg.norm(r.p)))
    if cs["gauge"]:
        gv = sp.csr_matrix(w.reshape(-1, 1))
        Kd = sp.bmat([[cs["A"], cs["B"].T, None], [cs["B"], None, gv], [None, gv.T, None]], format="csr")
        bd = np.concatenate([b, [0.0]])
    else:
        Kd, bd = Kt, b
    xd = Factorization(Kd, "symmetric_indefinite").solve(bd).reshape(-1)
    xb = np.concatenate([x, [xd[-1]]]) if cs["gauge"] else x           # the multiplier is taken from the direct solve
    rb = bd - Kd @ xb
    kappa = condition_1norm_estimate(Kd)
    out["err1"] = float(np.abs(xb - xd).sum() / np.abs(xd).sum())
    out["bound1"] = float(10 * kappa * np.abs(rb).sum() / np.abs(bd).sum())
    out["kappa"] = float(kappa)
    return out


def _table(element):
    rows = []
    for nu in NUS:
        rows.append(f"nu={nu}: " + ", ".join(f"n={n}:{_run(element, nu, n)['its']}" for n in NS))
    return "\n    ".join(rows)


# ------------------------------------------------------------------------------------------------ L6b1
@functools.cache
def _spectrum(nu: float, n: int):
    """Eigenvalues of Ŝ⁻¹S (dense, written here); the K = ∞ constant (zero) mode is removed."""
    G, K, _ = kg(nu)
    m = perturb(rectangle(n, n, element="T6"), 0.2 / n, seed=200 + n)
    mx = MixedKG(m, G, K)
    free = np.setdiff1d(mx.disp.active, clamp(m).dofs)
    A = mx.A[free][:, free].toarray()
    B = mx.B[:, free].toarray()
    M = mx.M.toarray()
    S = B @ np.linalg.solve(A, B.T) + (0.0 if math.isinf(K) else M / K)
    c = 1.0 / G + (0.0 if math.isinf(K) else 1.0 / K)
    lam = sla.eigh(0.5 * (S + S.T), c * M, eigvals_only=True)
    if math.isinf(K):
        assert abs(lam[0]) <= 1e-9, lam[0]
        lam = lam[1:]
    return lam, G, K


@pytest.mark.parametrize("nu", [0.3, 0.4999, 0.5])
def test_l6b1a_schur_complement_spectrum_is_below_three(nu):
    top = [_spectrum(nu, n)[0][-1] for n in (4, 8, 16)]
    print(f"\n  L6b1a nu={nu}: lambda_max {top}")
    assert max(top) <= 3.0 + 1e-10


@pytest.mark.parametrize("nu", [0.3, pytest.param(0.4999, marks=pytest.mark.xfail(
    strict=True, reason="pre-registered L6b1 kappa(S_hat^-1 S) <= 200 at nu = 0.4999; measured 3.72e3 / 3.75e3 / 3.75e3 for "
                        "n = 4 / 8 / 16: the constant pressure is an exact eigenvector with eigenvalue G/(G+K) = 2.0e-4, "
                        "the equivalence constant degrades with K/G on that one mode")), 0.5])
def test_l6b1b_condition_number_of_the_preconditioned_schur_complement(nu):
    kap = {n: _spectrum(nu, n)[0][-1] / _spectrum(nu, n)[0][0] for n in (4, 8, 16)}
    print(f"\n  L6b1b nu={nu}: kappa {kap}")
    assert max(kap.values()) <= 200


@pytest.mark.parametrize("nu", [0.3, 0.4999, 0.5])
def test_l6b1c_condition_number_does_not_grow_with_refinement(nu):
    k = {n: _spectrum(nu, n)[0][-1] / _spectrum(nu, n)[0][0] for n in (4, 16)}
    print(f"\n  L6b1c nu={nu}: kappa(16)/kappa(4) = {k[16] / k[4]:.4f}")
    assert k[16] / k[4] <= 1.25


@pytest.mark.parametrize("nu", [0.3, 0.49, 0.4999])
def test_l6b1_post_hoc_constant_mode_explains_the_small_eigenvalue(nu):
    """POST HOC (after the L6b1b failure at nu = 0.4999): with a clamped boundary B^T 1 = 0, so S 1 = M 1/K and the
    constant is an exact eigenvector of S_hat^-1 S with eigenvalue (1/K)/(1/G + 1/K) = G/(G+K); without it the spectrum
    is well conditioned (kappa' <= 200, threshold chosen before this test was first run)."""
    for n in (4, 8, 16):
        lam, G, K = _spectrum(nu, n)
        c = G / (G + K)
        j = int(np.argmin(np.abs(lam - c)))
        rest = np.delete(lam, j)
        print(f"\n  L6b1 post hoc nu={nu} n={n}: constant-mode eigenvalue {lam[j]:.6f} vs G/(G+K) {c:.6f}; "
              f"remaining [{rest[0]:.4f}, {rest[-1]:.4f}] kappa' {rest[-1] / rest[0]:.2f}")
        assert abs(lam[j] - c) <= 1e-8
        assert rest[-1] / rest[0] <= 200


# ------------------------------------------------------------------------------------------------ L6c, L6f
@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_l6c1_block_diagonal_minres_converges_on_every_case(element):
    print(f"\n  L6c iterations {element}:\n    {_table(element)}")
    for nu in NUS:
        for n in NS:
            r = _run(element, nu, n)
            print(f"  L6c1 {element} nu={nu} n={n}: {r['status']} its {r['its']} (restarts {r['restarts']}) res "
                  f"{r['res']:.1e} (u {r['res_u']:.1e}, p {r['res_p']:.1e}) A-solves {r['nA']} S-solves {r['nS']}")
            assert r["status"] == "converged"
            assert r["res"] <= RES_TOL and r["res_u"] <= RES_TOL and r["res_p"] <= RES_TOL
            assert r["nA"] == r["nS"] >= r["its"]                  # L6f


@pytest.mark.parametrize("element", [pytest.param("T6", marks=pytest.mark.xfail(
    strict=True, reason="pre-registered L6c2a max/min iterations <= 2 over the 12 (nu, n) cases: T6 measured 47/21 = 2.24 "
                        "(nu = 0.3 needs 21-23 iterations, nu >= 0.49 needs 38-47: the constant-mode eigenvalue "
                        "G/(G+K) of the preconditioned Schur complement)")), "Q9"])
def test_l6c2a_iteration_ratio_over_the_sweep_is_at_most_two(element):
    its = [_run(element, nu, n)["its"] for nu in NUS for n in NS]
    print(f"\n  L6c2a {element}: iterations {its}, max/min = {max(its) / min(its):.2f}")
    assert max(its) / min(its) <= 2.0


@pytest.mark.parametrize("element", [pytest.param("T6", marks=pytest.mark.xfail(
    strict=True, reason="pre-registered L6c2b max iterations <= 40: T6 measured 47 (nu = 0.4999 and 0.5, n = 32)")), "Q9"])
def test_l6c2b_iteration_count_is_at_most_forty(element):
    its = [_run(element, nu, n)["its"] for nu in NUS for n in NS]
    print(f"\n  L6c2b {element}: max iterations {max(its)}")
    assert max(its) <= 40


@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_l6c2_post_hoc_iteration_count_is_flat_in_h_at_fixed_nu(element):
    """POST HOC, descriptive (the threshold 1.3 was set after seeing the counts, largest observed 1.24): at fixed nu the
    count over n = 8, 16, 32 varies by at most 1.3; counts at nu = 0.4999 and 0.5 agree within 1."""
    for nu in NUS:
        its = [_run(element, nu, n)["its"] for n in NS]
        print(f"\n  L6c2 post hoc {element} nu={nu}: {its}, max/min {max(its) / min(its):.2f}")
        assert max(its) / min(its) <= 1.3
    for n in NS:
        assert abs(_run(element, 0.4999, n)["its"] - _run(element, 0.5, n)["its"]) <= 1


def test_l6c3_negative_control_plain_minres_does_not_converge_in_500_iterations():
    cs = _case("T6", 0.4999, 32)
    r = _solve(cs, "none", maxiter=500, max_restarts=0)
    b = np.concatenate([cs["f"], cs["g"]])
    res = np.linalg.norm(b - _test_K(cs) @ np.concatenate([r.u, r.p])) / np.linalg.norm(b)
    print(f"\n  L6c3 plain MINRES n=32 nu=0.4999: {r.status}, {r.iterations} its, explicit residual {res:.2e}")
    assert not res <= RES_TOL
    assert r.status != "converged"


@pytest.mark.xfail(strict=True, reason="pre-registered L6c4: plain MINRES iterations (cap 500, nu = 0.3) at n = 32 / n = 8 "
                                       ">= 2; measured 500 / 308 = 1.62 (the cap bounds the ratio; n = 32 hit the cap)")
def test_l6c4_negative_control_plain_minres_iterations_grow_with_refinement():
    its = {n: _solve(_case("T6", 0.3, n), "none", maxiter=500, max_restarts=0).iterations for n in (8, 32)}
    print(f"\n  L6c4 plain MINRES nu=0.3: {its}, ratio {its[32] / its[8]:.2f}")
    assert its[32] / its[8] >= 2.0


def test_l6c4_post_hoc_plain_minres_explicit_residual_after_a_fixed_budget_grows_with_refinement():
    """POST HOC (after the L6c4 failure): explicit residual after exactly 300 plain MINRES iterations, nu = 0.3."""
    r = {n: _solve(_case("T6", 0.3, n), "none", maxiter=300, max_restarts=0).residual for n in (8, 32)}
    print(f"\n  L6c4 post hoc: residual after 300 its {r}, ratio {r[32] / r[8]:.1f}")
    assert r[32] / r[8] >= 2.0


@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_l6c5_and_c6_solution_equals_the_direct_solve_and_the_gauge_holds(element):
    for nu in NUS:
        for n in NS:
            r = _run(element, nu, n)
            print(f"  L6c5 {element} nu={nu} n={n}: |x - x_direct|_1 = {r['err1']:.2e} <= bound {r['bound1']:.2e} "
                  f"(kappa_1 est. {r['kappa']:.2e}); gauge {r['gauge']}, |w.p| rel {r['wp']:.1e}")
            assert r["err1"] <= r["bound1"]
            assert r["gauge"] == (nu == 0.5)
            if nu == 0.5:
                assert r["wp"] <= 1e-10


# ------------------------------------------------------------------------------------------------ L6d, L6e
def test_l6d_block_triangular_gmres_is_bounded_independently_of_h_and_nu():
    its = []
    for nu in (0.3, 0.4999, 0.5):
        for n in NS:
            cs = _case("T6", nu, n)
            r = _solve(cs, "triangular")
            b = np.concatenate([cs["f"], cs["g"]])
            res = np.linalg.norm(b - _test_K(cs) @ np.concatenate([r.u, r.p])) / np.linalg.norm(b)
            print(f"  L6d nu={nu} n={n}: {r.status}, {r.iterations} its (restarts {r.restarts}), residual {res:.1e}, "
                  f"A-solves {r.n_A_solves}")
            assert r.status == "converged" and res <= RES_TOL
            assert r.n_A_solves == r.n_S_solves >= r.iterations
            its.append(r.iterations)
    print(f"\n  L6d iterations {its}, max/min = {max(its) / min(its):.2f}")
    assert max(its) / min(its) <= 2.0
    assert max(its) <= 40


@pytest.mark.parametrize("n", [pytest.param(8, marks=pytest.mark.xfail(
    strict=True, reason="pre-registered L6e: ILU block + MINRES converges to explicit residual <= 1e-7 in <= 200 iterations; "
                        "measured n = 8: status max_iterations after 18 iterations, residual 1.7e-6 (an ILU factor of an SPD "
                        "matrix is not symmetric, so it is not a valid MINRES preconditioner)")),
                               pytest.param(16, marks=pytest.mark.xfail(
    strict=True, reason="pre-registered L6e as above; measured n = 16: max_iterations after 17 iterations, residual 1.3e-6"))])
def test_l6e_incomplete_lu_block_converges_without_a_mesh_independence_claim(n):
    cs = _case("T6", 0.3, n)
    ilu = factorised_solver(cs["A"], "ilu", drop_tol=1e-4, fill_factor=10.0)
    r = _solve(cs, "diagonal", A_hat=ilu, maxiter=200, max_restarts=0)
    b = np.concatenate([cs["f"], cs["g"]])
    res = np.linalg.norm(b - _test_K(cs) @ np.concatenate([r.u, r.p])) / np.linalg.norm(b)
    exact = _run("T6", 0.3, n)["its"]
    print(f"\n  L6e ILU n={n}: {r.status}, {r.iterations} its (exact-LU block: {exact}), residual {res:.1e}")
    assert r.status == "converged" and res <= RES_TOL
    assert r.iterations <= 200


@pytest.mark.parametrize("n", [8, 16])
def test_l6e_post_hoc_incomplete_lu_needs_the_nonsymmetric_krylov_method(n):
    """POST HOC (after L6e): an ILU factorisation of an SPD matrix is not symmetric, so it is not a valid MINRES
    preconditioner; with the block-triangular preconditioner and GMRES (no symmetry needed) it converges."""
    cs = _case("T6", 0.3, n)
    ilu = factorised_solver(cs["A"], "ilu", drop_tol=1e-4, fill_factor=10.0)
    r = _solve(cs, "triangular", A_hat=ilu, maxiter=200, max_restarts=0)
    b = np.concatenate([cs["f"], cs["g"]])
    res = np.linalg.norm(b - _test_K(cs) @ np.concatenate([r.u, r.p])) / np.linalg.norm(b)
    print(f"\n  L6e post hoc ILU + GMRES n={n}: {r.status}, {r.iterations} its, residual {res:.1e}")
    assert r.status == "converged" and res <= RES_TOL


def test_block_preconditioner_constructors_apply_the_documented_blocks():
    A, B, _, rng = _dense("zero")
    n, m = A.shape[0], B.shape[0]
    S = B @ np.linalg.inv(A) @ B.T
    Pd = block_diagonal_preconditioner(sp.csr_matrix(A), sp.csr_matrix(S), n, m)
    Pt = block_triangular_preconditioner(sp.csr_matrix(A), sp.csr_matrix(S), sp.csr_matrix(B))
    v = rng.standard_normal(n + m)
    Dd = sla.block_diag(np.linalg.inv(A), np.linalg.inv(S))
    Dt = np.linalg.inv(np.block([[A, B.T], [np.zeros((m, n)), -S]]))
    assert np.abs(Pd @ v - Dd @ v).max() <= 1e-12 * np.abs(Dd @ v).max()
    assert np.abs(Pt @ v - Dt @ v).max() <= 1e-12 * np.abs(Dt @ v).max()
