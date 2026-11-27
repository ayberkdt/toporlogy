"""A posteriori estimators and newest-vertex bisection on T3 (WP09; prereg ``docs/foundations/prereg/A_adaptivity.md``,
ID ``A_adaptivity``, preregistered 2026-11-27 03:05 +0300 before the first run).

References are written here and do not call the production indicators: collapsed Gauss rules, a Duffy map with
``s = σ³`` at the L-shape corner, the exact gradient ``∇u`` of ``u = r^{2/3} sin(2θ/3)``, a closed-form local model
(``_model``) and brute-force geometry checks. ``HeatProblem`` solves the discrete problems (already verified
elsewhere: ``test_mesh_verification.py``)."""
from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.spatial import cKDTree

from toporlogy.fem.mesh import Dirichlet, HeatProblem, Mesh, perturb, rectangle
from toporlogy.fem.mesh.adaptivity import (cell_areas, dorfler_mark, free_dofs, mesh_from_state, mesh_state,
                                           newest_vertex_labelling, refine, refine_uniform, residual_indicators,
                                           solve_adaptive_poisson, zz_indicators)

PI = math.pi


# ---------------------------------------------------------------------------------------------- reference tools
def _tri_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed Gauss rule on the reference triangle (0,0),(1,0),(0,1); weights sum to 1/2."""
    x, w = leggauss(n)
    x, w = 0.5 * (x + 1), 0.5 * w
    P, W = [], []
    for xi, wi in zip(x, w, strict=True):
        for eta, we in zip(x, w, strict=True):
            P.append((xi, eta * (1 - xi)))
            W.append(wi * we * (1 - xi))
    return np.array(P), np.array(W)


def _subdivide(V: np.ndarray) -> np.ndarray:
    """(m, 3, 2) triangles -> (4m, 3, 2) by midpoints."""
    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
    return np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])


def _cell_integral(mesh: Mesh, f, singular: bool, chunk: int = 3000) -> np.ndarray:
    """Per-cell ∫ f(x, cell) dA of a T3 mesh. ``singular``: the origin is a point where f may blow up like r^{-2/3}:
    cells touching it use a Duffy map with s = σ³ (14 × 14 Gauss), cells within 3 diameters are split in 16
    sub-triangles (12 × 12 each), all others use 6 × 6."""
    V = mesh.points[mesh.cells]
    ne = len(V)
    diam = np.max([np.linalg.norm(V[:, i] - V[:, (i + 1) % 3], axis=1) for i in range(3)], axis=0)
    dist = np.linalg.norm(V, axis=2).min(axis=1)
    touch = singular & (dist < 1e-14)
    near = singular & ~touch & (dist < 3 * diam)
    reg = ~touch & ~near
    out = np.zeros(ne)

    def run(ids: np.ndarray, rule_n: int, split: int) -> None:
        P, W = _tri_rule(rule_n)
        for s0 in range(0, len(ids), chunk):
            c = ids[s0:s0 + chunk]
            T = V[c]
            owner = c
            for _ in range(split):
                T = _subdivide(T)
                owner = np.tile(owner, 4)
            e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
            det = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
            X = T[:, None, 0, :] + P[None, :, 0:1] * e1[:, None, :] + P[None, :, 1:2] * e2[:, None, :]
            wt = det[:, None] * W[None, :]
            own = np.broadcast_to(owner[:, None], wt.shape)
            val = f(X.reshape(-1, 2), own.reshape(-1))
            out[:] += np.bincount(own.reshape(-1), weights=(val * wt.reshape(-1)), minlength=ne)

    run(np.flatnonzero(reg), 6, 0)
    run(np.flatnonzero(near), 12, 2)
    ids = np.flatnonzero(touch)
    if len(ids):
        k = np.argmin(np.linalg.norm(V[ids], axis=2), axis=1)
        rot = (k[:, None] + np.arange(3)) % 3
        T = np.take_along_axis(V[ids], rot[:, :, None], axis=1)
        A, B = T[:, 1], T[:, 2]
        cross = np.abs(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])
        x, w = leggauss(14)
        x, w = 0.5 * (x + 1), 0.5 * w
        sg, tt = np.meshgrid(x, x, indexing="ij")
        wg = np.outer(w, w).ravel()
        sg, tt = sg.ravel(), tt.ravel()
        s = sg ** 3
        jac = s * 3 * sg ** 2                                   # |A×B| · s ds, s = σ³
        for s0 in range(0, len(ids), chunk):
            sl = slice(s0, s0 + chunk)
            a, b, cr = A[sl], B[sl], cross[sl]
            X = s[None, :, None] * (a[:, None, :] + tt[None, :, None] * (b - a)[:, None, :])
            wt = cr[:, None] * wg[None, :] * jac[None, :]
            own = np.broadcast_to(ids[sl][:, None], wt.shape)
            val = f(X.reshape(-1, 2), own.reshape(-1))
            out += np.bincount(own.reshape(-1), weights=(val * wt.reshape(-1)), minlength=ne)
    return out


def _grad_discrete(mesh: Mesh, u: np.ndarray) -> np.ndarray:
    """(ne, 2) gradient of the P1 field, from the 2x2 system (p1-p0)·g = u1-u0, (p2-p0)·g = u2-u0."""
    X = mesh.points[mesh.cells]
    M = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], axis=1)
    r = np.stack([u[mesh.cells[:, 1]] - u[mesh.cells[:, 0]], u[mesh.cells[:, 2]] - u[mesh.cells[:, 0]]], axis=1)
    return np.linalg.solve(M, r[:, :, None])[:, :, 0]


def _local_errors(mesh: Mesh, u: np.ndarray, grad_exact, singular: bool, kappa: float = 1.0) -> np.ndarray:
    """(ne,) squared energy errors ∫_T κ |∇u − ∇u_h|²."""
    gh = _grad_discrete(mesh, u)

    def f(X, cell):
        d = grad_exact(X) - gh[cell]
        return kappa * np.einsum("ij,ij->i", d, d)
    return _cell_integral(mesh, f, singular)


# ---------------------------------------------------------------------------------------------- A1 problem
def _u_sq(X):
    return np.sin(PI * X[:, 0]) * np.sin(PI * X[:, 1])


def _g_sq(X):
    return PI * np.stack([np.cos(PI * X[:, 0]) * np.sin(PI * X[:, 1]), np.sin(PI * X[:, 0]) * np.cos(PI * X[:, 1])],
                         axis=1)


def _solve_square(n: int, kappa: float = 1.0):
    m = newest_vertex_labelling(rectangle(n, n, element="T3"))
    P = HeatProblem(m, kappa)
    f = P.body_force(lambda X: kappa * 2 * PI ** 2 * _u_sq(X), degree=8)
    bn = m.boundary_nodes()
    u = P.solve(f, Dirichlet.fixed(bn)).u[:, 0]
    return m, u


@cache
def _a1(n: int, kappa: float = 1.0) -> dict:
    m, u = _solve_square(n, kappa)
    src = (lambda X: kappa * 2 * PI ** 2 * _u_sq(X))
    err = math.sqrt(_local_errors(m, u, _g_sq, False, kappa).sum())
    eta = math.sqrt(residual_indicators(m, u, src, kappa, degree=8).sum())
    zz = math.sqrt(zz_indicators(m, u, kappa).sum())
    return {"n": n, "err": err, "eta": eta, "zz": zz, "th": eta / err, "th_zz": zz / err}


def _model() -> dict:
    """Closed-form local model of the prereg: u ≈ ½ xᵀHx on squares split along (0,0)-(1,1)."""
    P, W = _tri_rule(6)

    def e2(a, b, c):
        H = np.array([[a, b], [b, c]])
        tot = 0.0
        for V, g in (([(0, 0), (1, 0), (1, 1)], np.array([a / 2, b + c / 2])),
                     ([(0, 0), (1, 1), (0, 1)], np.array([a / 2 + b, c / 2]))):
            V = np.array(V, float)
            J = np.column_stack([V[1] - V[0], V[2] - V[0]])
            r = (V[0] + P @ J.T) @ H.T - g
            tot += np.sum(W * abs(np.linalg.det(J)) * np.sum(r * r, axis=1))
        return tot

    x, w = leggauss(40)
    x, w = 0.5 * (x + 1), 0.5 * w
    X, Y = np.meshgrid(x, x, indexing="ij")
    Wt = np.outer(w, w)
    a = -PI ** 2 * np.sin(PI * X) * np.sin(PI * Y)
    c, b = a, PI ** 2 * np.cos(PI * X) * np.cos(PI * Y)
    E = np.sum(Wt * np.vectorize(e2)(a, b, c))
    N = np.sum(Wt * ((a + b) ** 2 + (b + c) ** 2 + 4 * b ** 2 + 2 * (a + c) ** 2))
    return {"e2": E, "eta2": N, "theta": math.sqrt(N / E), "e_const": math.sqrt(E)}


def test_a1_model_constants_match_prereg():
    mo = _model()
    assert mo["theta"] == pytest.approx(4 * math.sqrt(2), rel=1e-6)
    assert mo["e_const"] == pytest.approx(3.489, abs=1e-3)


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
def test_a1_report(n):
    r = _a1(n)
    print(f"\n  A1 n={n}: |e|={r['err']:.5e} eta={r['eta']:.5e} theta_res={r['th']:.4f} theta_zz={r['th_zz']:.4f}")
    assert r["err"] > 0


def test_a1a_residual_effectivity_matches_model():
    th = 4 * math.sqrt(2)
    assert abs(_a1(64)["th"] / th - 1) <= 0.15
    assert abs(_a1(32)["th"] / th - 1) <= 0.25


def test_a1b_zz_effectivity_is_one():
    assert 0.85 <= _a1(64)["th_zz"] <= 1.15
    assert 0.80 <= _a1(32)["th_zz"] <= 1.20


def test_a1c_energy_error_constant_and_slope():
    mo_const = 3.489
    assert 0.90 * mo_const <= _a1(64)["err"] * 64 <= 1.10 * mo_const
    slope = math.log(_a1(32)["err"] / _a1(64)["err"]) / math.log(2)
    assert 0.95 <= slope <= 1.05


def test_a1d_kappa_scaling_regression():
    """Regression (same kernel): κ = 4 with source 4 s leaves θ unchanged."""
    a, b = _a1(16, 1.0), _a1(16, 4.0)
    assert abs(a["th"] - b["th"]) <= 1e-9 * a["th"]
    assert abs(a["th_zz"] - b["th_zz"]) <= 1e-9 * a["th_zz"]


def test_a1e_patch_test_linear_solution_has_zero_indicators():
    m = perturb(rectangle(6, 6, element="T3"), 0.02, seed=3)

    def g(X):
        return 1 + 2 * X[:, 0] - 3 * X[:, 1]

    def dg(X):
        return np.tile([2.0, -3.0], (len(X), 1))
    P = HeatProblem(m)
    bn = m.boundary_nodes()
    u = P.solve(P.body_force(0.0), Dirichlet([(bn, g(m.points[bn]))])).u[:, 0]
    assert np.abs(u - g(m.points)).max() <= 1e-12
    assert math.sqrt(residual_indicators(m, u, 0.0, 1.0, (g, dg)).sum()) <= 1e-12
    assert math.sqrt(zz_indicators(m, u).sum()) <= 1e-12
    assert _local_errors(m, u, dg, False).sum() <= 1e-24


# ---------------------------------------------------------------------------------------------- A2 problem
def _uL(X):
    r = np.hypot(X[:, 0], X[:, 1])
    th = np.arctan2(X[:, 1], X[:, 0])
    th = np.where(th < 0, th + 2 * PI, th)
    return r ** (2 / 3) * np.sin(2 * th / 3)


def _gL(X):
    r = np.hypot(X[:, 0], X[:, 1])
    th = np.arctan2(X[:, 1], X[:, 0])
    th = np.where(th < 0, th + 2 * PI, th)
    k = (2 / 3) * r ** (-1 / 3)
    return np.stack([-k * np.sin(th / 3), k * np.cos(th / 3)], axis=1)


def _L0() -> Mesh:
    base = rectangle(2, 2, 2.0, 2.0, "T3", origin=(-1.0, -1.0))
    c = base.cell_centroids()
    keep = ~((c[:, 0] > 0) & (c[:, 1] < 0))
    used, inv = np.unique(base.cells[keep], return_inverse=True)
    m = Mesh(base.points[used], inv.reshape(-1, 3), "T3")
    return newest_vertex_labelling(m)


def _solveL(m: Mesh) -> np.ndarray:
    P = HeatProblem(m)
    bn = m.boundary_nodes()
    return P.solve(P.body_force(0.0), Dirichlet([(bn, _uL(m.points[bn]))])).u[:, 0]


def _errL(m: Mesh, u: np.ndarray) -> float:
    return math.sqrt(_local_errors(m, u, _gL, True).sum())


def test_a2f_test_quadrature_and_exact_gradient():
    m = refine_uniform(_L0(), 6).mesh
    E2 = _cell_integral(m, lambda X, c: np.einsum("ij,ij->i", _gL(X), _gL(X)), True).sum()
    ref = 2 * quad(lambda t: math.cos(t) ** (-4 / 3), 0, PI / 4, epsabs=1e-14, epsrel=1e-14)[0]
    print(f"\n  A2f: sum = {E2:.12f}, analytic = {ref:.12f}, rel = {abs(E2 / ref - 1):.2e}")
    assert abs(E2 / ref - 1) <= 1e-8
    X = np.array([[0.3, 0.4], [-0.5, 0.2], [-0.3, -0.7], [0.6, 0.1], [0.0, -0.4]])
    h = 1e-6
    fd = np.stack([(_uL(X + [h, 0]) - _uL(X - [h, 0])) / (2 * h), (_uL(X + [0, h]) - _uL(X - [0, h])) / (2 * h)], 1)
    assert np.abs(fd - _gL(X)).max() <= 1e-7


@cache
def _uniform_L() -> list[dict]:
    out = []
    cur = _L0()
    for r in range(1, 15):
        cur = refine(cur, np.arange(cur.n_cells)).mesh
        if r < 2:
            continue
        u = _solveL(cur)
        out.append({"round": r, "N": free_dofs(cur), "err": _errL(cur, u)})
    return out


def _slope(N, e) -> float:
    return float(np.polyfit(np.log(N), np.log(e), 1)[0])


@cache
def _adaptive_L(estimator: str) -> dict:
    start = refine_uniform(_L0(), 2).mesh
    res = solve_adaptive_poisson(start, 0.0, 1.0, (_uL, _gL), 0.5, estimator, max_dofs=40000, max_iterations=60)
    errs = [_errL(lv.mesh, lv.u) for lv in res.levels]
    return {"res": res, "N": np.array([lv.n_dofs for lv in res.levels]), "err": np.array(errs),
            "eta": np.array([lv.eta for lv in res.levels])}


def test_a2a_uniform_rate():
    U = _uniform_L()
    N = np.array([u["N"] for u in U])
    e = np.array([u["err"] for u in U])
    sel = (N >= 1000) & (N <= 60000)
    s = _slope(N[sel], e[sel])
    print("\n  A2a uniform: " + ", ".join(f"r{u['round']}: N={u['N']} e={u['err']:.4e}" for u in U) + f"; slope {s:.4f}")
    assert -0.38 <= s <= -0.29


def test_a2b_adaptive_rate():
    A = _adaptive_L("residual")
    sel = A["N"] >= 1000
    s = _slope(A["N"][sel], A["err"][sel])
    print("\n  A2b adaptive: " + ", ".join(f"N={n} e={e:.4e} th={h / e:.2f}" for n, e, h in
                                         zip(A["N"], A["err"], A["eta"], strict=True))
          + f"; slope {s:.4f}; stop = {A['res'].stop_reason}")
    assert -0.58 <= s <= -0.42


def _factor(A: dict) -> tuple[float, int, float]:
    U = _uniform_L()
    u12 = next(u for u in U if u["round"] == 12)
    ok = np.flatnonzero(A["N"] <= u12["N"])
    k = ok[-1]
    return u12["err"] / A["err"][k], int(A["N"][k]), float(A["err"][k])


def test_a2c_adaptive_beats_uniform_at_comparable_dofs():
    fac, Na, ea = _factor(_adaptive_L("residual"))
    u12 = next(u for u in _uniform_L() if u["round"] == 12)
    print(f"\n  A2c: uniform N={u12['N']} e={u12['err']:.4e}; adaptive N={Na} e={ea:.4e}; factor {fac:.2f}")
    assert fac >= 2.5


@pytest.mark.xfail(strict=True, reason="pre-registered A2d failed on the first run: 4 of the 6 cells touching the "
                   "origin are in the Dörfler set (theta = 0.5 marks only 6 of 1536 cells) and the true error share "
                   "of M is 0.479 < 0.5 (docs/foundations/prereg/A_adaptivity.md, first result)")
def test_a2d_estimator_detects_the_corner():
    m = refine_uniform(_L0(), 8).mesh                    # h = 1/16
    u = _solveL(m)
    eta2 = residual_indicators(m, u, 0.0, 1.0, (_uL, _gL))
    M = dorfler_mark(eta2, 0.5)
    touch = np.flatnonzero((np.linalg.norm(m.points[m.cells], axis=2) < 1e-14).any(axis=1))
    eT2 = _local_errors(m, u, _gL, True)
    share = eT2[M].sum() / eT2.sum()
    print(f"\n  A2d: |M|={len(M)} of {m.n_cells}; cells at origin {len(touch)} all marked: {np.isin(touch, M).all()}; "
          f"true error share in M = {share:.3f}")
    assert np.isin(touch, M).all()
    assert share >= 0.5


def _origin_diam(m: Mesh) -> float:
    V = m.points[m.cells]
    t = (np.linalg.norm(V, axis=2) < 1e-14).any(axis=1)
    return float(np.max([np.linalg.norm(V[t][:, i] - V[t][:, (i + 1) % 3], axis=1) for i in range(3)]))


def test_a2e_negative_control_without_jump_term_loses_the_corner():
    A, C = _adaptive_L("residual"), _adaptive_L("residual_no_jump")
    fac, Na, ea = _factor(A)
    ok = np.flatnonzero(C["N"] <= next(u for u in _uniform_L() if u["round"] == 12)["N"])
    ec = C["err"][ok[-1]]
    sel = C["N"] >= 1000
    s = _slope(C["N"][sel], C["err"][sel]) if sel.sum() >= 3 else 0.0
    dA, dC = _origin_diam(A["res"].mesh), _origin_diam(C["res"].mesh)
    print(f"\n  A2e: control e={ec:.4e} (N={C['N'][ok[-1]]}) vs adaptive e={ea:.4e}; control slope {s:.3f} "
          f"({int(sel.sum())} levels with N>=1000); origin diameter control {dC:.3e} vs adaptive {dA:.3e}; "
          f"stop = {C['res'].stop_reason}")
    assert ec >= 2 * ea
    assert s >= -0.40
    assert dC >= 8 * dA


def test_a2g_reported_zz_adaptive_and_effectivity():
    Z, A = _adaptive_L("zz"), _adaptive_L("residual")
    sel = Z["N"] >= 1000
    s = _slope(Z["N"][sel], Z["err"][sel]) if sel.sum() >= 3 else float("nan")
    print(f"\n  A2g: ZZ-driven adaptive slope {s:.3f} over {int(sel.sum())} levels; final e={Z['err'][-1]:.4e} "
          f"N={Z['N'][-1]}; residual theta along adaptive: {np.round(A['eta'] / A['err'], 2).tolist()}; "
          f"ZZ theta: {np.round(Z['eta'] / Z['err'], 2).tolist()}")
    assert np.isfinite(Z["err"]).all()


# ---------------------------------------------------------------------------------------------- A3
def _unit_square() -> Mesh:
    m = newest_vertex_labelling(rectangle(4, 4, element="T3"))
    c = m.cell_centroids()
    m.cell_tags = np.where(c[:, 0] < 0.5, 1, 2).astype(np.int64)
    return m


def _angles_min_deg(m: Mesh) -> float:
    V = m.points[m.cells]
    best = 180.0
    for i in range(3):
        a, b, c = V[:, i], V[:, (i + 1) % 3], V[:, (i + 2) % 3]
        u, v = b - a, c - a
        cosang = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        best = min(best, float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).min()))
    return best


def _check_conforming(m: Mesh, area0: float, perimeter0: float) -> None:
    c = m.cells
    V = m.points[c]
    area = 0.5 * ((V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1])
                  - (V[:, 1, 1] - V[:, 0, 1]) * (V[:, 2, 0] - V[:, 0, 0]))
    assert (area > 0).all()
    assert abs(area.sum() / area0 - 1) <= 1e-14
    assert _angles_min_deg(m) >= 45.0 - 1e-9
    ed = np.sort(np.concatenate([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]]), axis=1)
    uq, cnt = np.unique(ed, axis=0, return_counts=True)
    assert set(np.unique(cnt)) <= {1, 2}
    L = np.linalg.norm(m.points[uq[:, 0]] - m.points[uq[:, 1]], axis=1)
    assert abs(L[cnt == 1].sum() / perimeter0 - 1) <= 1e-14
    tree = cKDTree(m.points)
    mid = 0.5 * (m.points[uq[:, 0]] + m.points[uq[:, 1]])
    for k, hits in enumerate(tree.query_ball_point(mid, L * 0.5 * (1 - 1e-6))):
        assert set(hits) <= set(uq[k].tolist()), f"hanging node on edge {uq[k]}"


@pytest.mark.parametrize("name", ["square", "L"])
def test_a3a_conformity_under_random_marking(name):
    m = _unit_square() if name == "square" else _L0()
    area0, per0 = (1.0, 4.0) if name == "square" else (3.0, 8.0)
    _check_conforming(m, area0, per0)
    rng = np.random.default_rng(2026)
    for _ in range(6):
        marked = np.flatnonzero(rng.random(m.n_cells) < 0.3)
        m = refine(m, marked).mesh
        _check_conforming(m, area0, per0)
    assert m.n_cells > 6 * (4 if name == "L" else 8)


def test_a3b_tags_and_sets_are_geometric_after_refinement():
    m0 = _unit_square()
    rng = np.random.default_rng(7)
    m = m0
    for r in range(5):
        for stage in (m,):
            c = stage.cell_centroids()
            assert ((stage.cell_tags == 1) == (c[:, 0] < 0.5)).all()
            for name, val, dim in (("left", 0.0, 0), ("right", 1.0, 0), ("bottom", 0.0, 1), ("top", 1.0, 1)):
                got = {tuple(f) for f in stage.facet_sets[name].tolist()}
                want = set()
                for j, cell in enumerate(stage.cells.tolist()):
                    for f in range(3):
                        a, b = cell[f], cell[(f + 1) % 3]
                        if abs(stage.points[a, dim] - val) < 1e-14 and abs(stage.points[b, dim] - val) < 1e-14:
                            want.add((j, f))
                assert got == want, (r, name)
                assert set(stage.node_sets[name].tolist()) == set(np.flatnonzero(
                    np.abs(stage.points[:, dim] - val) < 1e-14).tolist()), (r, name)
        m = refine(m, np.flatnonzero(rng.random(m.n_cells) < 0.35)).mesh


def test_a3c_transfer_affine_field_and_cell_inheritance():
    m0 = _L0()
    m1 = refine_uniform(m0, 2).mesh
    rng = np.random.default_rng(5)
    ref = refine(m1, np.flatnonzero(rng.random(m1.n_cells) < 0.3))
    ref2 = refine(ref.mesh, np.flatnonzero(rng.random(ref.mesh.n_cells) < 0.3))
    aff = lambda X: 3.0 + 2.0 * X[:, 0] - 5.0 * X[:, 1]                  # noqa: E731
    u = aff(m1.points)
    u2 = ref2.transfer_nodal(ref.transfer_nodal(u))
    assert np.abs(u2 - aff(ref2.mesh.points)).max() <= 1e-14 * 10
    vec = np.stack([aff(m1.points), 1.0 - m1.points[:, 0]], axis=1)
    out = ref.transfer_nodal(vec)
    assert np.abs(out[:, 0] - aff(ref.mesh.points)).max() <= 1e-14 * 10
    assert np.abs(out[:, 1] - (1.0 - ref.mesh.points[:, 0])).max() <= 1e-14 * 10
    # cell field: children take the parent value and lie inside the parent triangle
    field = np.arange(m1.n_cells)
    child = ref.inherit_cell(field)
    assert (child == ref.parent_cell).all()
    cc = ref.mesh.cell_centroids()
    V = m1.points[m1.cells[ref.parent_cell]]
    M = np.stack([V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]], axis=2)
    lam = np.linalg.solve(M, (cc - V[:, 0])[:, :, None])[:, :, 0]
    assert (lam >= -1e-12).all() and (lam.sum(axis=1) <= 1 + 1e-12).all()
    # a quadratic is interpolated, not reproduced (documents the contract)
    q = m1.points[:, 0] ** 2
    assert np.abs(ref.transfer_nodal(q) - ref.mesh.points[:, 0] ** 2).max() > 1e-4


def test_a3d_determinism_and_restart():
    m = _unit_square()
    marked = np.array([0, 3, 5, 9, 11])
    a, b = refine(m, marked), refine(m, marked)
    sa, sb = mesh_state(a.mesh), mesh_state(b.mesh)
    assert sa.keys() == sb.keys() and all(np.array_equal(sa[k], sb[k]) for k in sa)
    assert np.array_equal(a.parent_cell, b.parent_cell) and np.array_equal(a.node_parents, b.node_parents)

    start = refine_uniform(_L0(), 2).mesh
    states = []
    full = solve_adaptive_poisson(start, 0.0, 1.0, (_uL, _gL), 0.5, "residual", max_dofs=3000, max_iterations=6,
                                  on_level=lambda lv: states.append(mesh_state(lv.mesh)))
    assert len(full.levels) == 7
    rest = solve_adaptive_poisson(mesh_from_state(states[3]), 0.0, 1.0, (_uL, _gL), 0.5, "residual", max_dofs=3000,
                                  max_iterations=3)
    assert len(rest.levels) == 4
    for k in range(4):
        la, lb = full.levels[3 + k], rest.levels[k]
        assert np.array_equal(la.mesh.points, lb.mesh.points) and np.array_equal(la.mesh.cells, lb.mesh.cells)
        assert np.array_equal(la.u, lb.u) and np.array_equal(la.eta2_cells, lb.eta2_cells)


def test_a3e_caps_and_stop_reasons():
    start = refine_uniform(_L0(), 2).mesh
    args = (0.0, 1.0, (_uL, _gL), 0.5, "residual")
    r = solve_adaptive_poisson(start, *args, max_dofs=60, max_iterations=30)
    assert r.stop_reason == "max_dofs" and r.levels[-1].n_dofs <= 60 and len(r.levels) >= 2
    r = solve_adaptive_poisson(start, *args, max_dofs=10 ** 6, max_iterations=2)
    assert r.stop_reason == "max_iterations" and len(r.levels) == 3
    r = solve_adaptive_poisson(start, *args, max_dofs=10 ** 6, max_iterations=5, target=1e9)
    assert r.stop_reason == "target" and len(r.levels) == 1
    r = solve_adaptive_poisson(start, 0.0, 1.0, None, 0.5, "residual", max_dofs=10 ** 6, max_iterations=5)
    assert r.stop_reason == "zero_estimator" and len(r.levels) == 1          # u = 0, f = 0: exactly zero
    with pytest.raises(ValueError):
        solve_adaptive_poisson(refine_uniform(start, 4).mesh, *args, max_dofs=10)
    with pytest.raises(ValueError):
        solve_adaptive_poisson(start, 0.0, 1.0, None, 0.5, "nope")


def test_a3f_dorfler_brute_force():
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        e = rng.integers(0, 6, size=n).astype(float)
        if e.sum() == 0:
            assert len(dorfler_mark(e, 0.5)) == 0
            continue
        th = float(rng.choice([0.1, 0.3, 0.5, 0.7, 1.0]))
        M = dorfler_mark(e, th)
        assert (np.diff(M) > 0).all()
        tot = e.sum()
        assert e[M].sum() >= th * tot
        assert e[M].sum() - e[M].min() < th * tot
        rest = np.setdiff1d(np.arange(n), M)
        if len(rest):
            lo = e[M].min()
            assert e[rest].max() <= lo
            tie = rest[e[rest] == lo]
            if len(tie):
                assert tie.min() > M[e[M] == lo].max()                   # stable: lower index first
    with pytest.raises(ValueError):
        dorfler_mark(np.ones(3), 0.0)


def test_a3_refine_input_checks():
    m = rectangle(2, 2, element="T3")
    with pytest.raises(ValueError):
        refine(m, np.array([100]))
    with pytest.raises(ValueError):
        refine(rectangle(2, 2, element="Q4"), np.array([0]))
    with pytest.raises(ValueError):
        residual_indicators(m, np.zeros(m.n_nodes), kappa=-1.0)
    lab = newest_vertex_labelling(m)
    assert (cell_areas(lab) > 0).all() and np.allclose(cell_areas(lab).sum(), 1.0)


def test_a2d_posthoc_marking_measurements():
    """Post hoc (not pre-registered, no threshold): where the Dörfler set lies on the uniform h = 1/16 L-mesh and what
    share of the true squared error it holds for theta = 0.5 and 0.8."""
    m = refine_uniform(_L0(), 8).mesh
    u = _solveL(m)
    eta2 = residual_indicators(m, u, 0.0, 1.0, (_uL, _gL))
    eT2 = _local_errors(m, u, _gL, True)
    cen = np.linalg.norm(m.cell_centroids(), axis=1)
    touch = np.flatnonzero((np.linalg.norm(m.points[m.cells], axis=2) < 1e-14).any(axis=1))
    for th in (0.5, 0.8):
        M = dorfler_mark(eta2, th)
        print(f"\n  A2d post hoc theta={th}: |M|={len(M)}, origin cells marked {int(np.isin(touch, M).sum())}/"
              f"{len(touch)}, max centroid distance {cen[M].max():.3f} (h=0.0625), true error share "
              f"{eT2[M].sum() / eT2.sum():.3f}; eta2 of origin cells {np.round(eta2[touch] / eta2.sum(), 3).tolist()}; "
              f"true e2 share of origin cells {np.round(eT2[touch] / eT2.sum(), 3).tolist()}")
