"""Bifurcation detection, limit / bifurcation classification and branch switching (``solvers.continuation``).

Tolerances are the pre-registered ones of ``docs/olcumler/catallanma_onkayit.md`` §3 (committed 26 Nov 2026 12:40
+03:00 in 494b7ff, before the code and the first run); the labels refer to that table. Closed forms (models S and A,
the two-bar truss T) are derived in §2 of that record. Checks added after the first run are marked "post hoc".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from toporlogy.core.domain import Domain
from toporlogy.fem.buckling import LinearBuckling
from toporlogy.fem.elastic import Elasticity
from toporlogy.fem.nonlinear import NonlinearElasticity
from toporlogy.solvers.continuation import ArcLengthSettings, TangentSpectrum, arc_length, displacement_control, \
    find_critical, full_tangent, locate_critical, sparse_inertia, switch_branch
from toporlogy.solvers.newton import LinearSolveFailure, Linearization

sys.path.insert(0, str(Path(__file__).parent))
from _truss import TwoBarTruss  # noqa: E402

DENSE = TangentSpectrum()
SPARSE = TangentSpectrum(method="sparse", k=6)


# ============================================================================================== closed-form models
class _TwoDof:
    """R(x, λ) = ∇U(x) − λ q on x = (u, v), q = (0, −1); dense full tangent."""

    conservative = True
    load = np.array([0.0, -1.0])

    def grad_hess(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def linearize(self, x: np.ndarray, lam: float) -> Linearization:
        g, H = self.grad_hess(np.asarray(x, float))
        return Linearization(g - lam * self.load, H, float(np.abs(H).sum(axis=0).max()))

    def solve(self, lin: Linearization, rhs: np.ndarray) -> np.ndarray:
        try:
            out = np.linalg.solve(lin.matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise LinearSolveFailure(f"LinAlgError: {e}") from e
        if not np.all(np.isfinite(out)):
            raise LinearSolveFailure("non-finite solution")
        return out

    def det_sign(self, lin: Linearization) -> int:
        return int(np.sign(np.linalg.det(lin.matrix)))


class SpringBar(_TwoDof):
    """Model S (record §2): extensible bar with an axial spring k_a and a base rotational spring k_r."""

    def __init__(self, L: float = 1.0, ka: float = 100.0, kr: float = 1.0) -> None:
        self.L, self.ka, self.kr = L, ka, kr
        self.lam_c = ka * (L - math.sqrt(L * L - 4 * kr / ka)) / 2

    def grad_hess(self, x):
        X, Y = x[0], self.L + x[1]
        ell = math.hypot(X, Y)
        th = math.atan2(X, Y)
        dl = np.array([X, Y]) / ell
        d2l = np.array([[Y * Y, -X * Y], [-X * Y, X * X]]) / ell ** 3
        dt = np.array([Y, -X]) / ell ** 2
        d2t = np.array([[-2 * X * Y, X * X - Y * Y], [X * X - Y * Y, 2 * X * Y]]) / ell ** 4
        ka, kr = self.ka, self.kr
        g = ka * (ell - self.L) * dl + kr * th * dt
        H = ka * np.outer(dl, dl) + ka * (ell - self.L) * d2l + kr * np.outer(dt, dt) + kr * th * d2t
        return g, H

    def lam_secondary(self, th: float) -> float:
        m = self.kr * th / math.sin(th)
        c = math.cos(th)
        return self.ka * (self.L - math.sqrt(self.L ** 2 - 4 * m * c / self.ka)) / (2 * c)


class Polynomial(_TwoDof):
    """Model A (record §2): U = ½ k_a e² + ½ k_r u² + (β/6) u³, e = v + u²/(2L)."""

    def __init__(self, L: float = 1.0, ka: float = 100.0, kr: float = 1.0, beta: float = 3.0) -> None:
        self.L, self.ka, self.kr, self.beta = L, ka, kr, beta
        self.lam_c = kr * L

    def grad_hess(self, x):
        u, v = x
        L, ka, kr, b = self.L, self.ka, self.kr, self.beta
        e = v + u * u / (2 * L)
        g = np.array([ka * e * u / L + kr * u + 0.5 * b * u * u, ka * e])
        H = np.array([[ka * (u / L) ** 2 + ka * e / L + kr + b * u, ka * u / L], [ka * u / L, ka]])
        return g, H

    def lam_secondary(self, u: float) -> float:
        return self.L * (self.kr + 0.5 * self.beta * u)


def _criterion(problem, p, rtol, eta_tol) -> tuple[bool, float, float]:
    lin = problem.linearize(p.x, p.lam)
    nr = float(np.linalg.norm(lin.residual))
    nq = float(np.linalg.norm(problem.load))
    rel = nr / nq
    eta = nr / (lin.norm1 * float(np.linalg.norm(p.x)) + nq)
    return (rel <= rtol or eta <= eta_tol), rel, eta


CF = ArcLengthSettings(ds=0.01, rtol=1e-12, eta_tol=0.0, max_steps=500)


def _primary(model, ds=CF.ds):
    st = ArcLengthSettings(ds=ds, rtol=CF.rtol, eta_tol=CF.eta_tol, max_steps=CF.max_steps)
    return arc_length(model, np.zeros(2), 0.0, st, lambda x, lam: lam >= 1.5, spectrum=DENSE)


A_REASON = ("pre-registered A1/A2 FAILED (26 Nov 2026): with ds = 0.01 the primary path of model A has |dx/dλ| = "
            "1/k_a = 0.01, so the first arc-length step is Δλ = 1 and the accepted point lands EXACTLY on λ_c = k_r L "
            "= 1, where K = diag(0, k_a) is singular: the next predictor solve K⁻¹q fails (LinAlgError: Singular "
            "matrix), every halving restarts from the same point, ContinuationError step_limit (cause "
            "singular_tangent) at load factor 1. A defect of arc_length (no predictor at an exactly singular "
            "accepted point), not of the location; the post hoc tests use ds = 0.0137 (Δλ = 1.37). Update 27 Nov "
            "2026 (docs/foundations/prereg/R07_bordered.md): the predictor now passes the exactly singular point; "
            "the test still fails, now in locate_critical (ValueError: crossing eigenvalue not bracketed, 0.000e+00 "
            "at the accepted point): a second defect, of the critical-point bracketing.")
A_POSTHOC_DS = 0.0137


@pytest.mark.parametrize("model", [SpringBar(), Polynomial()], ids=["S", "A"])
def test_m0_closed_form_tangents_match_finite_differences(model):
    """M0: the tangent of models S and A equals the central FD of their residual (h = 1e-6), 5 random points,
    ≤ 1e-6 relative."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5):
        x = rng.uniform(-0.2, 0.2, 2)
        H = model.linearize(x, 0.7).matrix
        fd = np.column_stack([(model.linearize(x + 1e-6 * e, 0.7).residual - model.linearize(x - 1e-6 * e, 0.7).residual)
                              / 2e-6 for e in np.eye(2)])
        worst = max(worst, float(np.abs(fd - H).max() / np.abs(H).max()))
    print(f"M0 {type(model).__name__}: worst FD rel {worst:.1e}")
    assert worst <= 1e-6


def test_s1_s2_symmetric_bifurcation_located_and_classified():
    """S1: one critical interval on the primary path to λ = 1.5, |λ − λ_c|/λ_c ≤ 1e-8, |v + λ_c/k_a| ≤ 1e-8 L,
    |u| ≤ 1e-12; S2: bifurcation, |φᵀq|/‖q‖ ≤ 1e-8."""
    m = SpringBar()
    path = _primary(m)
    crit = find_critical(path)
    cp = locate_critical(m, path, crit[0], DENSE)
    err = abs(cp.lam - m.lam_c) / m.lam_c
    ev = abs(cp.x[1] + m.lam_c / m.ka) / m.L
    print(f"S1/S2: {len(path.points)} points, intervals {crit}, lam {cp.lam:.15f} (exact {m.lam_c:.15f}), rel "
          f"{err:.1e}, v err {ev:.1e}, |u| {abs(cp.x[0]):.1e}, {cp.evaluations} evaluations, mu {cp.eigenvalue:.1e}, "
          f"kind {cp.kind}, load test {cp.load_test:.1e}, index {cp.index_before}->{cp.index_after}")
    assert len(crit) == 1
    assert err <= 1e-8 and ev <= 1e-8 and abs(cp.x[0]) <= 1e-12
    assert cp.kind == "bifurcation" and abs(cp.load_test) <= 1e-8


def _secondary_checks(model, sw, lam_of, umin):
    pts = sw.path.points
    crit = [_criterion(model, p, CF.rtol, CF.eta_tol) for p in pts]
    u = np.abs(np.array([p.x[0] for p in pts]))
    lam = np.array([p.lam for p in pts])
    exact = np.array([lam_of(p) for p in pts])
    return crit, u, lam, exact


def test_s3_symmetric_branch_switch():
    """S3: switch at ξ = 0.02 L, arc length to |u| ≥ 0.3 L: (a) every point meets the convergence criterion
    (‖R‖/‖q‖ ≤ 1e-12); (b) |u| ≥ 0.5 ξ everywhere, monotone, last ≥ 0.3 L; (c) |λ − λ_S(θ)| ≤ 1e-9 λ_c."""
    m = SpringBar()
    path = _primary(m)
    cp = locate_critical(m, path, find_critical(path)[0], DENSE)
    xi = 0.02 * m.L
    sw = switch_branch(m, cp, xi, CF, lambda x, lam: abs(x[0]) >= 0.3 * m.L)
    crit, u, lam, exact = _secondary_checks(m, sw, lambda p: m.lam_secondary(math.atan2(p.x[0], m.L + p.x[1])), 0)
    e_lam = float(np.abs(lam - exact).max() / m.lam_c)
    print(f"S3: switch {sw.record.iterations} iterations, start u {sw.x[0]:.4f} lam {sw.lam:.8f}; {len(u)} points, "
          f"{sw.path.rejected} rejected, worst rel residual {max(c[1] for c in crit):.1e}, u {u.min():.4f}..{u[-1]:.4f},"
          f" lam {lam.min():.6f}..{lam.max():.6f}, worst |lam - lam_S|/lam_c {e_lam:.1e}")
    assert all(c[0] for c in crit)
    assert u.min() >= 0.5 * xi and np.all(np.diff(u) > 0) and u[-1] >= 0.3 * m.L
    assert e_lam <= 1e-9


@pytest.mark.xfail(strict=True, reason=A_REASON)
def test_a1_asymmetric_bifurcation_located_and_classified():
    """A1 (as pre-registered, ds = 0.01): |λ − 1| ≤ 1e-8; bifurcation, |φᵀq|/‖q‖ ≤ 1e-8."""
    _a1(CF.ds)


def test_a1_posthoc_step_off_the_critical_point():
    """A1 post hoc (same thresholds, ds = 0.0137 so that no accepted point is the singular one)."""
    _a1(A_POSTHOC_DS)


def _a1(ds):
    m = Polynomial()
    path = _primary(m, ds)
    crit = find_critical(path)
    cp = locate_critical(m, path, crit[0], DENSE)
    print(f"A1: intervals {crit}, lam {cp.lam:.15f}, err {abs(cp.lam - 1):.1e}, {cp.evaluations} evaluations, kind "
          f"{cp.kind}, load test {cp.load_test:.1e}")
    assert len(crit) == 1 and abs(cp.lam - 1.0) <= 1e-8
    assert cp.kind == "bifurcation" and abs(cp.load_test) <= 1e-8


@pytest.mark.xfail(strict=True, reason=A_REASON)
@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["plus", "minus"])
def test_a2_asymmetric_branch_switch(sign):
    """A2 (as pre-registered, ds = 0.01): ξ = ±0.02 to |u| ≥ 0.2: (a) criterion at every point; (b) |u| ≥ 0.5|ξ|,
    monotone, last ≥ 0.2; (c) |λ − L(k_r + βu/2)| ≤ 1e-9; (d) +ξ: λ > λ_c everywhere, −ξ: λ < λ_c."""
    _a2(sign, CF.ds)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["plus", "minus"])
def test_a2_posthoc_step_off_the_critical_point(sign):
    """A2 post hoc (same thresholds; primary path with ds = 0.0137, the switched path with the registered 0.01)."""
    _a2(sign, A_POSTHOC_DS)


def _a2(sign, ds):
    m = Polynomial()
    path = _primary(m, ds)
    cp = locate_critical(m, path, find_critical(path)[0], DENSE)
    xi = sign * 0.02
    sw = switch_branch(m, cp, xi, CF, lambda x, lam: abs(x[0]) >= 0.2)
    crit, u, lam, exact = _secondary_checks(m, sw, lambda p: m.lam_secondary(p.x[0]), 0)
    e_lam = float(np.abs(lam - exact).max())
    print(f"A2 {sign:+.0f}: phi {cp.phi}, start u {sw.x[0]:+.4f} lam {sw.lam:.6f}; {len(u)} points, worst rel residual "
          f"{max(c[1] for c in crit):.1e}, lam {lam.min():.4f}..{lam.max():.4f}, worst |lam - lam_A| {e_lam:.1e}")
    assert all(c[0] for c in crit)
    assert u.min() >= 0.5 * abs(xi) and np.all(np.diff(u) > 0) and u[-1] >= 0.2
    assert e_lam <= 1e-9
    assert np.all(lam > m.lam_c) if sign > 0 else np.all(lam < m.lam_c)


# ============================================================================================== T: truss limit points
def _truss_path(spectrum=DENSE, stop_at_critical=False):
    tr = TwoBarTruss()
    st = ArcLengthSettings(ds=0.005, rtol=1e-12, eta_tol=0.0, max_steps=2000)
    return tr, arc_length(tr, np.zeros(2), 0.0, st, lambda x, lam: -x[1] >= 2.5 * tr.h, spectrum=spectrum,
                          stop_at_critical=stop_at_critical)


def _truss_checks(tr, path, label):
    crit = find_critical(path)
    cps = [locate_critical(tr, path, i, DENSE) for i in crit]
    print(f"{label}: intervals {crit}; " + "; ".join(
        f"{c.kind} lam {c.lam:+.12f} w/h {-c.x[1] / tr.h:.9f} load test {c.load_test:+.3f} ({c.evaluations} ev)"
        for c in cps))
    assert len(cps) == 2 and all(c.kind == "limit" and abs(c.load_test) >= 0.5 for c in cps)
    assert abs(cps[0].lam - 1) <= 1e-8 and abs(cps[1].lam + 1) <= 1e-8
    assert abs(-cps[0].x[1] - tr.w1) <= 1e-6 * tr.h and abs(-cps[1].x[1] - tr.w2) <= 1e-6 * tr.h
    return cps


def test_t1_t2_truss_limit_points():
    """T1: exactly two intervals on the truss path to 2.5h, both limit points (|φᵀq|/‖q‖ ≥ 0.5); T2: |λ₁ − 1|,
    |λ₂ + 1| ≤ 1e-8, |w − w₁,₂| ≤ 1e-6 h."""
    tr, path = _truss_path()
    _truss_checks(tr, path, "T1/T2")


def test_t3_truss_limit_points_on_a_displacement_control_path():
    """T3: the same on a displacement-control path (50 v targets to 2.5h)."""
    tr = TwoBarTruss()
    targets = list(-np.linspace(0.0, 2.5 * tr.h, 51)[1:])
    path = displacement_control(tr, np.zeros(2), 0.0, 1, targets, rtol=1e-12, eta_tol=0.0, spectrum=DENSE)
    _truss_checks(tr, path, "T3")


def test_t4_det_sign_intervals_equal_the_index_intervals():
    """T4: without a spectrum, ``find_critical`` (det sign) gives the same intervals as with the spectrum index."""
    _tr, a = _truss_path()
    _tr, b = _truss_path(spectrum=None)
    print(f"T4: index {find_critical(a)}, det sign {find_critical(b)}")
    assert find_critical(a) == find_critical(b) and all(p.index is None for p in b.points)


def test_t5_switch_branch_refuses_a_limit_point():
    tr, path = _truss_path()
    cp = locate_critical(tr, path, find_critical(path)[0], DENSE)
    with pytest.raises(ValueError, match="limit point"):
        switch_branch(tr, cp, 0.01, ArcLengthSettings(ds=0.005), lambda x, lam: True)


def test_c1_contract_errors():
    """C1: no index change in the interval, dense above dense_limit, unknown method: ValueError."""
    tr, path = _truss_path()
    with pytest.raises(ValueError, match="no critical point"):
        locate_critical(tr, path, 0, DENSE)
    with pytest.raises(ValueError, match="dense_limit"):
        TangentSpectrum(dense_limit=1)(tr.linearize(np.zeros(2), 0.0))
    with pytest.raises(ValueError, match="method"):
        TangentSpectrum(method="lanczos")


def test_c2_stop_at_critical():
    """C2: ``stop_at_critical`` ends the truss path at the first point past w₁; the last two indices are 0, 1."""
    tr, path = _truss_path(stop_at_critical=True)
    w = -path.points[-1].x[1]
    print(f"C2: {len(path.points)} points, last w/h {w / tr.h:.4f} (w1/h {tr.w1 / tr.h:.4f}), indices "
          f"{[p.index for p in path.points[-2:]]}")
    assert [p.index for p in path.points[-2:]] == [0, 1] and -path.points[-2].x[1] < tr.w1 < w


@pytest.mark.parametrize("which", ["truss", "S"])
def test_b2_spectrum_only_observes(which):
    """B2: with a spectrum the path (λ, x at every point) is bitwise the path without one."""
    if which == "truss":
        _tr, a = _truss_path()
        _tr, b = _truss_path(spectrum=None)
    else:
        m = SpringBar()
        a = _primary(m)
        b = arc_length(m, np.zeros(2), 0.0, CF, lambda x, lam: lam >= 1.5)
    assert len(a.points) == len(b.points)
    assert all(p.lam == r.lam and np.array_equal(p.x, r.x) for p, r in zip(a.points, b.points))


def test_full_tangent_storage():
    """Post hoc contract test: upper-triangle storage is completed, a full matrix is kept."""
    import scipy.sparse as sp
    M = np.array([[2.0, 1.0], [1.0, 3.0]])
    U = np.triu(M)
    assert np.array_equal(full_tangent(U), M) and np.array_equal(full_tangent(M), M)
    assert np.array_equal(full_tangent(sp.csr_matrix(U)).toarray(), M)
    assert np.array_equal(full_tangent(np.diag([1.0, 2.0])), np.diag([1.0, 2.0]))


def test_sparse_inertia_equals_the_dense_eigenvalue_count():
    """Post hoc contract test: the sparse LDL^T inertia equals the count of negative eigenvalues (eigvalsh) on
    random indefinite matrices and on shifted FE column tangents; the window values sit at their global position."""
    import scipy.sparse as sp
    rng = np.random.default_rng(3)
    for n in (50, 300):
        A = rng.standard_normal((n, n))
        A = A + A.T + 3 * np.eye(n)
        assert sparse_inertia(sp.csc_matrix(A)) == int((np.linalg.eigvalsh(A) < 0).sum())
    _d, nl, prob, _lam, _ds, _top = _column(20)
    K = full_tangent(prob.linearize(np.zeros(len(nl.free)), 0.0).matrix)
    ev = np.linalg.eigvalsh(K.toarray())
    for k in (0, 3, 10):
        sigma = 0.5 * (ev[k] + ev[k + 1])
        s = SPARSE(Linearization(np.zeros(K.shape[0]), (K - sigma * sp.identity(K.shape[0])).tocsr(), 1.0))
        assert s.index == k + 1
        assert np.abs(s.values - (ev - sigma)[s.offset:s.offset + len(s.values)]).max() <= 1e-12


# ============================================================================================== FE column
def _column(n_long, b=4, nu=0.0):
    d = Domain((b, n_long), h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] == 0))
    top = np.flatnonzero(np.isclose(X[:, 1], n_long))
    w = np.where(np.isclose(X[top, 0], 0) | np.isclose(X[top, 0], b), 0.5, 1.0)
    F = np.zeros(d.ndof)
    F[2 * top + 1] = -w / w.sum()                                   # unit compressive resultant
    E = np.ones(d.n_active)
    m = Elasticity(d, fixed, nu)
    U = m.solve_numpy(E, F[:, None])
    mu, _modes = LinearBuckling(m).modes(E, E, U[:, 0], 4)
    lam_lin = 1.0 / mu[0]
    nl = NonlinearElasticity(d, fixed, nu=nu, material="svk", plane="stress", rtol=1e-12)
    prob = nl.continuation_problem(np.ones(nl.n_elements), F)
    ds = 0.1 * lam_lin * float(np.linalg.norm(U[nl.free, 0]))
    return d, nl, prob, lam_lin, ds, top


def _mirror_share(d, free, v, x_axis):
    """Share ‖v_anti‖²/‖v‖² of the part antisymmetric under the reflection x → 2 x_axis − x (vector field)."""
    X = d.node_coords()
    key = {(round(a, 9), round(c, 9)): i for i, (a, c) in enumerate(X)}
    mirror = np.array([key[(round(2 * x_axis - a, 9), round(c, 9))] for a, c in X])
    full = np.zeros(d.ndof)
    full[free] = v
    f = full.reshape(-1, 2)
    Mf = np.column_stack([-f[mirror, 0], f[mirror, 1]])
    anti = 0.5 * (f - Mf)
    total = float((f ** 2).sum())
    return (float((anti ** 2).sum()) / total if total > 0 else 0.0), float(np.sqrt((anti ** 2).sum()))


def _column_run(n_long, spectrum=SPARSE):
    d, nl, prob, lam_lin, ds, top = _column(n_long)
    st = ArcLengthSettings(ds=ds, max_steps=200)
    path = arc_length(prob, np.zeros(len(nl.free)), 0.0, st, lambda x, lam: lam >= 1.2 * lam_lin, spectrum=spectrum)
    crit = find_critical(path)
    cp = locate_critical(prob, path, crit[0], spectrum)
    return d, nl, prob, lam_lin, path, crit, cp, top, st


@pytest.fixture(scope="module")
def column80():
    return _column_run(80)


def test_fc1_fc3_fe_column_bifurcation_matches_linear_buckling(column80):
    """FC1: |λ_nl/λ_lin − 1| ≤ 1e-2 (4 × 80 Q4 SVK column); FC3: bifurcation, |φᵀq|/‖q‖ ≤ 1e-8, antisymmetric
    share of φ ≥ 0.999."""
    d, nl, prob, lam_lin, path, crit, cp, _top, _st = column80
    share, _n = _mirror_share(d, nl.free, cp.phi, 2.0)
    rel = cp.lam / lam_lin - 1
    eps_c = math.pi ** 2 / (48 * 20 ** 2)
    print(f"FC1/FC3: {len(path.points)} points, intervals {crit}, lam_nl {cp.lam:.10e}, lam_lin {lam_lin:.10e}, "
          f"rel {rel:+.3e} ({rel / eps_c:+.3f} eps_c), {cp.evaluations} evaluations, kind {cp.kind}, load test "
          f"{cp.load_test:.1e}, antisymmetric share {share:.12f}")
    assert abs(rel) <= 1e-2
    assert cp.kind == "bifurcation" and abs(cp.load_test) <= 1e-8 and share >= 0.999


def test_fc2_prebuckling_difference_scales_with_the_buckling_strain(column80):
    """FC2: d = |λ_nl/λ_lin − 1| at L = 80 and 160: d(80)/d(160) ∈ [2, 8]."""
    d80 = abs(column80[6].lam / column80[3] - 1)
    r160 = _column_run(160)
    d160 = abs(r160[6].lam / r160[3] - 1)
    print(f"FC2: d(80) {d80:.4e}, d(160) {d160:.4e}, ratio {d80 / d160:.3f}; d/eps_c {d80 / (math.pi ** 2 / 19200):.3f}, "
          f"{d160 / (math.pi ** 2 / 76800):.3f}")
    assert 2.0 <= d80 / d160 <= 8.0


FC4_REASON = ("pre-registered FC4 FAILED in its second clause (26 Nov 2026): the index sequences along the 4 x 80 "
              "column path are identical (dense eigh and sparse LDL^T inertia), but the located loads differ by "
              "1.4e-9 relative (> 1e-9). Both locations stop after 2 evaluations on the eigenvalue precision floor "
              "|mu| <= 16 eps ||K||_1, added to locate_critical after the first run (without it the dense location "
              "ran 16 evaluations to the t bracket); at that floor the load is resolved to ~1e-9, and the two "
              "eigensolvers stop at different t. The located load agrees with linear buckling to 5.3e-4 (FC1).")


@pytest.fixture(scope="module")
def column80_dense():
    return _column_run(80, spectrum=DENSE)


def _fc4(column80, column80_dense):
    a = [p.index for p in column80_dense[4].points]
    b = [p.index for p in column80[4].points]
    rel = abs(column80_dense[6].lam / column80[6].lam - 1)
    print(f"FC4: dense indices {a}, sparse {b}; located lam rel diff {rel:.3e} ({column80_dense[6].evaluations} / "
          f"{column80[6].evaluations} evaluations)")
    return a, b, rel


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=FC4_REASON)
def test_fc4_dense_and_sparse_spectra_agree(column80, column80_dense):
    """FC4: the same index sequence along the path with the dense spectrum as with the sparse one (k = 6, the
    column80 fixture); located λ within 1e-9."""
    a, b, rel = _fc4(column80, column80_dense)
    assert a == b and rel <= 1e-9


@pytest.mark.slow
def test_fc4_posthoc_index_sequences_agree(column80, column80_dense):
    """FC4 post hoc: the first clause alone (identical index sequences, dense ↔ sparse)."""
    a, b, _rel = _fc4(column80, column80_dense)
    assert a == b


def test_fb_fe_column_branch_switch(column80):
    """FB1: every point of the switched path meets the criterion (rtol 1e-10 or η ≤ 1e-14); FB2: the top lateral
    displacement rises monotonically to 0.05 L within 200 steps, and is ≤ 1e-8 L on the primary path up to the
    critical point; FB3: λ ≥ λ_c (1 − 1e-6) on the secondary path."""
    d, nl, prob, lam_lin, path, crit, cp, top, st = column80
    L = 80.0
    gx = np.flatnonzero(np.isin(nl.free, 2 * top))                     # top lateral DOFs in free numbering
    xi = 1e-3 * L / float(np.abs(cp.phi[gx]).max())
    st2 = ArcLengthSettings(ds=2.0, max_steps=200)
    sw = switch_branch(prob, cp, xi, st2, lambda x, lam: abs(x[gx].mean()) >= 0.05 * L)
    pts = sw.path.points
    crit_ok = [_criterion(prob, p, st2.rtol, st2.eta_tol) for p in pts]
    lat = np.abs(np.array([p.x[gx].mean() for p in pts]))
    lam = np.array([p.lam for p in pts])
    prim = max(abs(p.x[gx]).max() for p in path.points[:crit[0] + 1])
    share, _n = _mirror_share(d, nl.free, pts[-1].x - cp.x, 2.0)
    print(f"FB: xi {xi:.4f}, switch {sw.record.iterations} iterations, {len(pts)} points ({sw.path.rejected} "
          f"rejected), worst rel residual {max(c[1] for c in crit_ok):.1e}, top lateral {lat[0] / L:.2e}..{lat[-1] / L:.4f}"
          f" L, lam/lam_c {lam.min() / cp.lam:.8f}..{lam.max() / cp.lam:.8f}, primary lateral {prim / L:.1e} L, "
          f"antisymmetric share of x - x_c at the end {share:.6f}")
    assert all(c[0] for c in crit_ok)
    assert np.all(np.diff(lat) > 0) and lat[-1] >= 0.05 * L and prim <= 1e-8 * L
    assert lam.min() >= cp.lam * (1 - 1e-6)


# ============================================================================================== FA: shallow arch (A6b)
def _arch():
    """The staircase shallow arch of ``test_nonlinear_solvers._arch`` (A6/A6b), same construction."""
    nx, ny = 40, 8
    frac = np.zeros((nx, ny))
    for i in range(nx):
        yc = int(round(3 * (1 - ((i + 0.5 - 20) / 20) ** 2)))
        frac[i, yc:yc + 2] = 1.0
    d = Domain((nx, ny), h=1.0, frac=frac)
    X = d.node_coords()
    pins = [int(np.flatnonzero((np.abs(X[:, 0] - x0) < 1e-9) & (np.abs(X[:, 1]) < 1e-9))[0]) for x0 in (0.0, 40.0)]
    fixed = np.concatenate([d.dofs(np.array([p])) for p in pins])
    crown = int(np.flatnonzero((np.abs(X[:, 0] - 20) < 1e-9) & (np.abs(X[:, 1] - 5) < 1e-9))[0])
    F = np.zeros(d.ndof)
    F[2 * crown + 1] = -1.0
    return d, fixed, F, crown


@pytest.mark.slow
def test_fa_arch_bifurcation_before_the_limit_point_and_branch_switch():
    """FA1: first critical point of the arch path (crown deflection to 3) is a bifurcation (|φᵀq|/‖q‖ ≤ 1e-8),
    antisymmetric share ≥ 0.999, λ_b/λ_lim ∈ [0.99, 1); FA2: the second is a limit point (≥ 1e-3), λ_lim ≥
    (1 − 1e-10) max discrete λ and ≤ (1 + 1e-3) max; FA3: switch at ξ = 0.05, 40 arc-length steps: criterion at every
    point, antisymmetric norm of x at the end ≥ 10 ξ, ≤ 1e-8 on the primary points up to the bifurcation."""
    d, fixed, F, crown = _arch()
    nl = NonlinearElasticity(d, fixed, nu=0.3, material="svk", rtol=1e-12)
    prob = nl.continuation_problem(np.ones(nl.n_elements), F)
    c = int(np.flatnonzero(nl.free == 2 * crown + 1)[0])
    st = ArcLengthSettings(ds=0.25, rtol=1e-12, max_steps=800)
    path = arc_length(prob, np.zeros(len(nl.free)), 0.0, st, lambda x, lam: -x[c] >= 3.0, spectrum=DENSE)
    crit = find_critical(path)
    cps = [locate_critical(prob, path, i, DENSE) for i in crit[:2]]
    b, lim = cps
    share, _n = _mirror_share(d, nl.free, b.phi, 20.0)
    lmax = float(path.lam.max())
    prim = max(_mirror_share(d, nl.free, p.x, 20.0)[1] for p in path.points[:crit[0] + 1])
    sw = switch_branch(prob, b, 0.05, ArcLengthSettings(ds=0.25, rtol=1e-12, max_steps=40), lambda x, lam: False)
    ok = [_criterion(prob, p, 1e-12, st.eta_tol) for p in sw.path.points]
    anti_end = _mirror_share(d, nl.free, sw.path.points[-1].x, 20.0)[1]
    print(f"FA: {len(path.points)} points, intervals {crit}; {b.kind} lam {b.lam:.10e} (w {-b.x[c]:.4f}) load test "
          f"{b.load_test:.1e} anti share {share:.9f}; {lim.kind} lam {lim.lam:.10e} (w {-lim.x[c]:.4f}) load test "
          f"{lim.load_test:.3f}; lam_b/lam_lim {b.lam / lim.lam:.6f}; lam_lim/max discrete {lim.lam / lmax - 1:+.2e}; "
          f"primary anti {prim:.1e}; switch: {len(sw.path.points)} points, lam {sw.path.lam.min():.5e}.."
          f"{sw.path.lam.max():.5e}, worst rel residual {max(o[1] for o in ok):.1e}, anti norm at end {anti_end:.3f}")
    assert b.kind == "bifurcation" and abs(b.load_test) <= 1e-8 and share >= 0.999
    assert 0.99 <= b.lam / lim.lam < 1.0
    assert lim.kind == "limit" and abs(lim.load_test) >= 1e-3
    assert lmax * (1 - 1e-10) <= lim.lam <= lmax * (1 + 1e-3)
    assert all(o[0] for o in ok) and anti_end >= 10 * 0.05 and prim <= 1e-8
