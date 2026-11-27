"""F2 / F4 investigation of two historical failed predictions (risks R04, R05); preregistration
``docs/foundations/prereg/F4_rates.md`` (committed before the first run). Record: ``docs/foundations/F4_RATES.md``.

R05 (F2-1..F2-4): the curved T10 patch test against the quadrature degree. R04 (F4-0..F4-5): the T4 L2 and T10
energy rates of the V3 manufactured-solution test, separated into reference error, mesh draw, quadrature and
pre-asymptotic behaviour. The historical xfails in ``test_mesh_verification.py`` are not touched."""
from __future__ import annotations

import math
import time
from math import factorial
from types import SimpleNamespace

import numpy as np
import pytest

from toporlogy.core.linear_elastic import isotropic_moduli, isotropic_stiffness
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, InvertedElementError, box, get, perturb, rectangle
from toporlogy.fem.mesh.manufactured import Manufactured
from toporlogy.fem.mesh.quadrature import rule

NU = 0.3
slow = pytest.mark.slow


def _lame():
    m = isotropic_moduli(E=1.0, nu=NU)
    return m.G, m.lam


def _D3():
    return isotropic_stiffness(1.0, NU)


# ====================================================================================================== F2 (R05)
def _mesh(name, n=(4, 3, 2), amp=0.2, seed=3, curved=0.0):
    """Same construction as ``_mesh`` of test_mesh_assembly.py."""
    if get(name).dim == 2:
        m = rectangle(n[0], n[1], lx=1.2, ly=0.9, element=name)
        h = min(1.2 / n[0], 0.9 / n[1])
    else:
        m = box(n[0], n[1], n[2], lx=1.0, ly=0.8, lz=0.7, element=name)
        h = min(1.0 / n[0], 0.8 / n[1], 0.7 / n[2])
    return perturb(m, amp * h, seed=seed, curved=curved * h) if amp or curved else m


def _patch(m, D, degree=None):
    """Same construction as ``_patch`` of test_mesh_assembly.py: relative max u error and stress error."""
    rng = np.random.default_rng(0)
    dim = m.dim
    A = rng.normal(size=(dim, dim)) * 1e-2
    a = rng.normal(size=dim) * 1e-2
    P = ElasticProblem(m, D, degree=degree)
    bn = m.boundary_nodes()
    val = m.points[bn] @ A.T + a
    bc = Dirichlet([(m.dofs(bn, [c]), val[:, c]) for c in range(dim)])
    s = P.solve(np.zeros(P.ndof), bc)
    used = m.used_nodes()
    ue = m.points[used] @ A.T + a
    uh = s.u[:, 0].reshape(-1, dim)[used]
    err_u = np.abs(uh - ue).max() / np.abs(ue).max()
    eps = np.array([A[0, 0], A[1, 1], A[2, 2], A[1, 2] + A[2, 1], A[0, 2] + A[2, 0], A[0, 1] + A[1, 0]])
    sx = D @ eps
    err_s = np.abs(P.stress(s.u[:, 0]) - sx).max() / np.abs(sx).max()
    return float(err_u), float(err_s)


_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_f2_1_curved_t10_cell_integrand_degree_and_quadrature_error():
    """F2-1. Exact (sympy, rational arithmetic) cell integral of cof(J) grad_xi N_i on a curved P2 tetrahedron."""
    sp = pytest.importorskip("sympy")
    xi = sp.symbols("xi eta zeta")
    cc = sp.Symbol("c")
    lam = [1 - sum(xi), *xi]
    N = [lam[i] * (2 * lam[i] - 1) for i in range(4)] + [4 * lam[i] * lam[j] for i, j in _EDGES]
    rng = np.random.default_rng(5)
    h = 0.25
    V = rng.uniform(-0.2 * h, 0.2 * h, (4, 3)) + h * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    u = rng.uniform(-1.0, 1.0, (6, 3))
    R = sp.Rational
    nodes = [[R(float(v)) for v in V[i]] for i in range(4)]
    for k, (i, j) in enumerate(_EDGES):
        nodes.append([(nodes[i][a] + nodes[j][a]) / 2 + cc * R(float(h * u[k, a])) for a in range(3)])
    x = sp.Matrix([sum(N[i] * nodes[i][a] for i in range(10)) for a in range(3)])
    J = x.jacobian(xi)
    cof = J.cofactor_matrix()
    gN = [sp.Matrix([sp.diff(N[i], s) for s in xi]) for i in range(10)]
    entries = [sp.expand((cof * gN[i])[a]) for i in range(10) for a in range(3)]
    max_deg, bad = 0, 0
    for e in entries:
        pe = sp.Poly(e, *xi, cc)
        for mono, _ in pe.terms():
            d = sum(mono[:3])
            max_deg = max(max_deg, d)
            if d == 3 and mono[3] != 2:
                bad += 1
    assert max_deg == 3
    assert bad == 0                                           # every degree-3 term carries exactly c^2
    exact = []
    for e in entries:
        pe = sp.Poly(e, *xi)
        exact.append(sum(coef * R(factorial(m[0]) * factorial(m[1]) * factorial(m[2]),
                                  factorial(m[0] + m[1] + m[2] + 3)) for m, coef in pe.terms()))
    a, b = 0.5854101966249685, 0.1381966011250105             # Keast 4-point rule typed here (degree 2)
    keast = (np.array([[b, b, b], [a, b, b], [b, a, b], [b, b, a]]), np.full(4, 1 / 24))
    r3 = rule("tet", 3)
    out = {}
    for cval in (0.0, 0.05, 0.1):
        # numeric entries at this c (substitute before lambdify of the c-dependent polynomial)
        fc = sp.lambdify(xi, [e.subs(cc, R(cval).limit_denominator(10 ** 6)) for e in entries], "numpy")
        ex = np.array([float(e.subs(cc, R(cval).limit_denominator(10 ** 6))) for e in exact])
        for tag, (P, W) in (("deg2", keast), ("deg3", (r3.points, r3.weights))):
            vals = np.array([np.broadcast_to(np.asarray(v, float), len(W)) for v in fc(*P.T)])
            q = vals @ W
            out[(cval, tag)] = float(np.abs(q - ex).max())
        out[(cval, "scale")] = float(np.abs(ex).max())
    print(f"\n  F2-1 max degree {max_deg}; quadrature error (abs, scale {out[(0.1, 'scale')]:.3e}): "
          f"c=0 deg2 {out[(0.0, 'deg2')]:.2e} deg3 {out[(0.0, 'deg3')]:.2e}; "
          f"c=.05 deg2 {out[(0.05, 'deg2')]:.2e} deg3 {out[(0.05, 'deg3')]:.2e}; "
          f"c=.1 deg2 {out[(0.1, 'deg2')]:.2e} deg3 {out[(0.1, 'deg3')]:.2e}; "
          f"ratio {out[(0.1, 'deg2')] / out[(0.05, 'deg2')]:.4f}")
    assert out[(0.0, "deg2")] <= 1e-12 * out[(0.1, "scale")]
    assert out[(0.1, "deg2")] > 1e-6 * out[(0.1, "scale")]
    assert out[(0.1, "deg3")] <= 1e-12 * out[(0.1, "scale")]
    assert 3.99 <= out[(0.1, "deg2")] / out[(0.05, "deg2")] <= 4.01


def test_f2_2_historical_patch_numbers_are_reproduced():
    m = _mesh("T10", curved=0.1)
    eu2, es2 = _patch(m, _D3())
    eu3, es3 = _patch(m, _D3(), degree=3)
    print(f"\n  F2-2 T10 curved 0.1 seed 3: degree 2 u {eu2:.3e} stress {es2:.3e}; degree 3 u {eu3:.2e} stress {es3:.2e}")
    assert 1.65e-4 <= eu2 < 1.75e-4
    assert 2.55e-3 <= es2 < 2.65e-3
    assert eu3 <= 1e-12


def test_f2_3_patch_error_against_quadrature_degree():
    """A seed whose mesh has an inverted cell at any of the tested rule degrees (detJ is checked at the points of each
    rule) is refused for all degrees and replaced by the next seed (test set-up event, listed in the output)."""
    seeds, refused, rows = [], [], {}
    s = 3
    while len(seeds) < 5:
        try:
            m = _mesh("T10", seed=s, curved=0.1)
            trial = {(s, deg): _patch(m, _D3(), degree=deg) for deg in (2, 3, 4, 6, 8)}
        except InvertedElementError:
            refused.append(s)
        else:
            rows.update(trial)
            seeds.append(s)
        s += 1
    print(f"\n  F2-3 seeds {seeds} (refused {refused})")
    for deg in (2, 3, 4, 6, 8):
        print("    degree", deg, "u", [f"{rows[(s, deg)][0]:.1e}" for s in seeds], "stress",
              [f"{rows[(s, deg)][1]:.1e}" for s in seeds])
    for s in seeds:
        assert 1e-6 <= rows[(s, 2)][0] <= 1e-2
        for deg in (3, 4, 6, 8):
            assert rows[(s, deg)][0] <= 1e-10
            assert rows[(s, deg)][1] <= 1e-9


def test_f2_4_degree_two_patch_error_scales_as_curvature_squared():
    e = {c: _patch(_mesh("T10", seed=3, curved=c), _D3())[0] for c in (0.0, 0.025, 0.05, 0.1)}
    r1, r2 = e[0.05] / e[0.025], e[0.1] / e[0.05]
    print(f"\n  F2-4 u error at degree 2: {[f'{v:.2e}' for v in e.values()]}; ratios {r1:.3f}, {r2:.3f}")
    assert e[0.0] <= 1e-10
    assert 3.0 <= r1 <= 5.0
    assert 3.0 <= r2 <= 5.0


# ====================================================================================================== F4-5 (a)
def test_f4_5_manufactured_solution_equals_sympy():
    sp = pytest.importorskip("sympy")
    X = sp.symbols("x y z")
    g = lambda t: t ** 2 * (1 - t) ** 2  # noqa: E731
    psi = g(X[0]) * g(X[1]) * g(X[2])
    c = 1.0
    mu, lam = _lame()
    d = lambda f, i: sp.diff(f, X[i])  # noqa: E731
    uvec = [d(psi, 1) - d(psi, 2) + c * d(psi, 0), d(psi, 2) - d(psi, 0) + c * d(psi, 1),
            d(psi, 0) - d(psi, 1) + c * d(psi, 2)]
    div = sum(d(uvec[i], i) for i in range(3))
    sig = [[mu * (d(uvec[i], j) + d(uvec[j], i)) + (lam * div if i == j else 0) for j in range(3)] for i in range(3)]
    f = [-sum(d(sig[i][j], j) for j in range(3)) for i in range(3)]
    gr = [[d(uvec[i], j) for j in range(3)] for i in range(3)]
    rng = np.random.default_rng(21)
    pts = rng.uniform(0, 1, (20, 3))
    fu = sp.lambdify(X, uvec, "numpy")
    fg = sp.lambdify(X, gr, "numpy")
    ff = sp.lambdify(X, f, "numpy")
    ms = Manufactured(3, c=c, lam_c=lam * c)
    ue = np.array([np.broadcast_to(np.asarray(v, float), 20) for v in fu(*pts.T)]).T
    ge = np.array([[np.broadcast_to(np.asarray(v, float), 20) for v in row] for row in fg(*pts.T)]).transpose(2, 0, 1)
    fe = np.array([np.broadcast_to(np.asarray(v, float), 20) for v in ff(*pts.T)]).T
    du = np.abs(ms.u(pts) - ue).max() / np.abs(ue).max()
    dg = np.abs(ms.grad(pts) - ge).max() / np.abs(ge).max()
    df = np.abs(ms.body_force(mu, pts) - fe).max() / np.abs(fe).max()
    print(f"\n  F4-5(i) Manufactured vs sympy: u {du:.1e}, grad {dg:.1e}, f {df:.1e}")
    assert max(du, dg, df) <= 1e-11


def _own_errors(m, uh_nodes, ms, mu, lam, npts=14):
    """Independent dense error integral on a straight-sided tetrahedral mesh (T4 or T10): monomial fit of the nodal
    field on each cell, collapsed Gauss-Legendre rule written here, own Lame energy density."""
    deg = 1 if m.cells.shape[1] == 4 else 2
    mons = [(0, 0, 0)] + [tuple(int(i == j) for j in range(3)) for i in range(3)]
    if deg == 2:
        mons += [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    def basis(Xp):
        return np.stack([Xp[:, 0] ** a * Xp[:, 1] ** b * Xp[:, 2] ** c for a, b, c in mons], axis=1)

    def dbasis(Xp):
        out = np.zeros((len(Xp), len(mons), 3))
        for k, (a, b, c) in enumerate(mons):
            e = (a, b, c)
            for j in range(3):
                if e[j] == 0:
                    continue
                ee = list(e)
                ee[j] -= 1
                out[:, k, j] = e[j] * Xp[:, 0] ** ee[0] * Xp[:, 1] ** ee[1] * Xp[:, 2] ** ee[2]
        return out

    x, w = np.polynomial.legendre.leggauss(npts)
    r01, w01 = 0.5 * (x + 1), 0.5 * w
    R, S, T = np.meshgrid(r01, r01, r01, indexing="ij")
    WR, WS, WT = np.meshgrid(w01, w01, w01, indexing="ij")
    ref = np.stack([(R * (1 - S) * (1 - T)).ravel(), (S * (1 - T)).ravel(), T.ravel()], axis=1)
    wref = (WR * WS * WT * (1 - S) * (1 - T) ** 2).ravel()
    acc = np.zeros(4)
    for cell in m.cells:
        Xn = m.points[cell]
        coef = np.linalg.solve(basis(Xn), uh_nodes.reshape(-1, 3)[cell])           # (nb, 3)
        Vv = Xn[:4]
        Jm = np.stack([Vv[1] - Vv[0], Vv[2] - Vv[0], Vv[3] - Vv[0]], axis=1)
        Xp = Vv[0] + ref @ Jm.T
        wp = wref * abs(np.linalg.det(Jm))
        uh = basis(Xp) @ coef
        Gh = np.einsum("qkj,ki->qij", dbasis(Xp), coef)
        ue, Ge = ms.u(Xp), ms.grad(Xp)
        dG = Gh - Ge
        eps_d = 0.5 * (dG + np.swapaxes(dG, 1, 2))
        eps_e = 0.5 * (Ge + np.swapaxes(Ge, 1, 2))
        dens = lambda e: 2 * mu * np.einsum("qij,qij->q", e, e) + lam * np.trace(e, axis1=1, axis2=2) ** 2  # noqa: E731
        acc += [((uh - ue) ** 2).sum(1) @ wp, (ue ** 2).sum(1) @ wp, dens(eps_d) @ wp, dens(eps_e) @ wp]
    return {"l2": math.sqrt(acc[0] / acc[1]), "energy": math.sqrt(acc[2] / acc[3])}


@pytest.mark.parametrize("name", ["T4", "T10"])
def test_f4_5_error_integral_equals_independent_dense_integral(name):
    n = 2
    m = perturb(box(n, n, n, element=name), 0.2 / n, seed=102, curved=0.0)
    mu, lam = _lame()
    ms = Manufactured(3, c=1.0, lam_c=lam)
    rng = np.random.default_rng(8)
    uh = ms.u(m.points) + 0.01 * rng.normal(size=(m.n_nodes, 3))
    own = _own_errors(m, uh.ravel(), ms, mu, lam)
    P = ElasticProblem(m, _D3())
    prod = P.errors(uh.ravel(), ms.u, ms.grad, degree=24)
    dl = abs(prod["l2"] - own["l2"]) / own["l2"]
    de = abs(prod["energy"] - own["energy"]) / own["energy"]
    print(f"\n  F4-5(ii) {name}: l2 own {own['l2']:.10e} prod {prod['l2']:.10e} ({dl:.1e}); "
          f"energy own {own['energy']:.10e} prod {prod['energy']:.10e} ({de:.1e})")
    assert dl <= 1e-9 and de <= 1e-9


# ====================================================================================================== F4 solves
_CACHE: dict = {}
_STATS: dict = {}
_DEG = {"T4": 6, "T10": 8}                                    # 2k + 4
_KEY_RATE = {"T4": "l2", "T10": "energy"}


def _peak_delta(before):
    psutil = pytest.importorskip("psutil")
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) - before


def _level(name, n, seed, offsets=(None,), load_offset=0, undistorted=False, cache=True):
    """Solve the V3 problem exactly as ``_mms_errors`` of test_mesh_verification.py (new mesh draw ``seed``) and return
    the errors at error-norm degree 2k + 4 (key None) or 2k + 4 + offset, for load degree 2k + 4 + load_offset."""
    key = (name, n, seed, load_offset, undistorted)
    got = _CACHE.get(key, {}) if cache else {}
    todo = [o for o in offsets if o not in got]
    if todo:
        psutil = pytest.importorskip("psutil")
        before = psutil.Process().memory_info().rss
        t0 = time.time()
        base = box(n, n, n, element=name)
        m = base if undistorted else perturb(base, 0.2 / n, seed=seed, curved=0.0 / n)
        mu, lam = _lame()
        ms = Manufactured(3, c=1.0, lam_c=lam)
        P = ElasticProblem(m, _D3())
        k = get(name).degree
        f = P.body_force(lambda X: ms.body_force(mu, X), degree=2 * k + 4 + load_offset)
        s = P.solve(f, Dirichlet.fixed(m.dofs(m.boundary_nodes())))
        for o in todo:
            got[o] = P.errors(s.u[:, 0], ms.u, ms.grad, degree=None if o is None else 2 * k + 4 + o)
        _STATS[(name, n)] = (time.time() - t0, _peak_delta(before), P.ndof)
        print(f"\n    [level {name} n={n} seed={seed} ndof={P.ndof} {time.time() - t0:.0f}s "
              f"peak +{_peak_delta(before) / 1e6:.0f} MB]")
        if cache:
            _CACHE[key] = got
    return got


def _pair(e1, e2, n1, n2, key):
    return math.log(e1[key] / e2[key]) / math.log(n2 / n1)


def _may_run(name, n_prev):
    """Memory rule of the preregistration: skip when the previous level took > 600 s or 2.5 x its peak does not fit."""
    psutil = pytest.importorskip("psutil")
    st = _STATS.get((name, n_prev))
    if st is None:
        return True, ""
    sec, peak, ndof = st
    avail = psutil.virtual_memory().available
    if sec > 600:
        return False, f"previous level n={n_prev} took {sec:.0f} s > 600 s"
    if 2.5 * peak > avail:
        return False, f"2.5 x previous peak {peak / 1e6:.0f} MB > available {avail / 1e6:.0f} MB"
    return True, ""


@slow
@pytest.mark.parametrize("name,ns", [("T4", (4, 8, 16)), ("T10", (2, 4, 8))])
def test_f4_0_historical_rates_reproduce_and_runs_are_bitwise_identical(name, ns):
    first = [_level(name, n, 100 + n)[None] for n in ns]
    second = [_level(name, n, 100 + n, cache=False)[None] for n in ns]
    assert first == second                                    # dict of floats: exact equality
    p_l2 = _pair(first[-2], first[-1], ns[-2], ns[-1], "l2")
    p_en = _pair(first[-2], first[-1], ns[-2], ns[-1], "energy")
    print(f"\n  F4-0 {name} {ns}: L2 p = {p_l2:.6f}, energy p = {p_en:.6f}; errors "
          f"{[(format(e['l2'], '.4e'), format(e['energy'], '.4e')) for e in first]}")
    if name == "T4":
        assert abs(p_l2 - 1.796) <= 0.0006 and abs(p_en - 0.92) <= 0.006
    else:
        assert abs(p_l2 - 2.92) <= 0.006 and abs(p_en - 1.788) <= 0.0006


_INTERP = {"T4": (8, 16, 24, 32), "T10": (4, 8, 12, 16, 24)}


def _interp_errors(name, n, seed, ms):
    m = perturb(box(n, n, n, element=name), 0.2 / n, seed=seed, curved=0.0 / n)
    stub = SimpleNamespace(mesh=m, ref=m.ref, D=_D3(), k=6, thickness=1.0,
                           element_values=lambda u: np.asarray(u, float).reshape(-1, 3)[m.cells])
    return ElasticProblem.errors(stub, ms.u(m.points).ravel(), ms.u, ms.grad)       # type: ignore[arg-type]


_INTERP_CACHE: dict = {}


def _interp_pairs(name):
    if name in _INTERP_CACHE:
        return _INTERP_CACHE[name]
    psutil = pytest.importorskip("psutil")
    mu, lam = _lame()
    ms = Manufactured(3, c=1.0, lam_c=lam)
    errs, reached = {}, []
    for n in _INTERP[name]:
        if reached and n > _INTERP[name][-2] and psutil.virtual_memory().available < 2.0e9:
            print(f"\n  F4-1 {name}: n={n} skipped (available memory)")
            break
        t0 = time.time()
        b = psutil.Process().memory_info().rss
        errs[n] = _interp_errors(name, n, 100 + n, ms)
        reached.append(n)
        print(f"\n    [interp {name} n={n} {time.time() - t0:.0f}s peak +{_peak_delta(b) / 1e6:.0f} MB] "
              f"l2 {errs[n]['l2']:.4e} energy {errs[n]['energy']:.4e}")
    pairs = {(a, b): (_pair(errs[a], errs[b], a, b, "l2"), _pair(errs[a], errs[b], a, b, "energy"))
             for a, b in zip(reached[:-1], reached[1:], strict=True)}
    print(f"\n  F4-1 {name} interpolation pair rates (L2, energy): "
          f"{ {k: (round(v[0], 3), round(v[1], 3)) for k, v in pairs.items()} }")
    _INTERP_CACHE[name] = pairs
    return pairs


_F41_I_XFAIL = {"T4": "pre-registered F4-1 (i) failed: T4 interpolation L2 p(8, 16) = 1.918 differs from the Galerkin "
                      "1.796 by 0.122 > 0.10 (the solved L2 error is more pre-asymptotic than the interpolation error); "
                      "T10 energy 1.814 vs 1.788 passes (F4_RATES.md)"}


@slow
@pytest.mark.parametrize("name", [pytest.param("T4", marks=pytest.mark.xfail(strict=True, reason=_F41_I_XFAIL["T4"])),
                                  "T10"])
def test_f4_1_i_interpolation_rate_matches_galerkin_rate_at_the_historical_pair(name):
    pairs = _interp_pairs(name)
    idx = 0 if _KEY_RATE[name] == "l2" else 1
    first = pairs[(_INTERP[name][0], _INTERP[name][1])][idx]
    galerkin = 1.796 if name == "T4" else 1.788
    assert abs(first - galerkin) <= 0.10


@slow
@pytest.mark.parametrize("name", ["T4", "T10"])
def test_f4_1_ii_iii_interpolation_rate_reaches_the_asymptotic_window(name):
    pairs = _interp_pairs(name)
    idx = 0 if _KEY_RATE[name] == "l2" else 1
    first = pairs[(_INTERP[name][0], _INTERP[name][1])][idx]
    last_pair = list(pairs)[-1]
    want = (24, 32) if name == "T4" else (16, 24)
    assert pairs[last_pair][idx] > first + 0.05                                       # (iii)
    if last_pair != want:
        pytest.skip(f"{name}: pair {want} not reached under the memory rule (last {last_pair})")
    assert pairs[want][idx] >= 1.9                                                    # (ii)


@slow
@pytest.mark.parametrize("name,ns", [("T4", (8, 16)), ("T10", (4, 8))])
def test_f4_2_error_norm_and_load_quadrature_do_not_change_the_rate(name, ns):
    key = _KEY_RATE[name]
    tol = 0.002 if name == "T4" else 0.006
    offs = (None, 6, 12)
    e = [_level(name, n, 100 + n, offsets=offs) for n in ns]
    base = _pair(e[0][None], e[1][None], ns[0], ns[1], key)
    res = {"error degree +6": _pair(e[0][6], e[1][6], ns[0], ns[1], key),
           "error degree +12": _pair(e[0][12], e[1][12], ns[0], ns[1], key)}
    el = [_level(name, n, 100 + n, offsets=(None,), load_offset=6) for n in ns]
    res["load degree +6"] = _pair(el[0][None], el[1][None], ns[0], ns[1], key)
    rel = {n: max(abs(e[i][12][k] / e[i][None][k] - 1) for k in ("l2", "energy")) for i, n in enumerate(ns)}
    print(f"\n  F4-2 {name} {ns} ({key}): historical p = {base:.5f}; "
          + "; ".join(f"{k} p = {v:.5f} (dp {v - base:+.1e})" for k, v in res.items())
          + f"; max relative change of an error norm (+12): {rel}")
    for k, v in res.items():
        assert abs(v - base) <= tol, k
        assert v < 1.8, k


_SEED_BASES = (200, 300, 400, 500)


def _scatter(name, ns):
    key = _KEY_RATE[name]
    out = {}
    for b in (100, *_SEED_BASES):
        e = [_level(name, n, b + n)[None] for n in ns]
        out[b] = _pair(e[0], e[1], ns[0], ns[1], key)
    return out


_F43_XFAIL = ("pre-registered F4-3 failed for T10: 1 of 4 other mesh draws reach 1.8 (pair rates 1.7997, 1.7855, 1.8140, "
              "1.7818; historical draw 1.7877; undistorted 1.825): the T10 miss is not a mesh-draw accident "
              "(F4_RATES.md)")


@slow
@pytest.mark.parametrize("name,ns", [("T4", (8, 16)),
                                     pytest.param("T10", (4, 8), marks=pytest.mark.xfail(strict=True, reason=_F43_XFAIL))])
def test_f4_3_rate_scatter_over_mesh_draws_and_undistorted_mesh(name, ns):
    out = _scatter(name, ns)
    und = [_level(name, n, 0, undistorted=True)[None] for n in ns]
    pu = _pair(und[0], und[1], ns[0], ns[1], _KEY_RATE[name])
    print(f"\n  F4-3 {name} {ns} ({_KEY_RATE[name]}): pair rate per base seed "
          f"{ {b: round(v, 4) for b, v in out.items()} }; undistorted {pu:.4f}; "
          f"undistorted errors {[ (round(x['l2'], 6), round(x['energy'], 6)) for x in und]}")
    ok = sum(1 for b in _SEED_BASES if out[b] >= 1.8)
    assert ok >= 2, f"{ok} of 4 other mesh draws reach 1.8"


@slow
@pytest.mark.parametrize("name,ns", [("T4", (8, 16)), ("T10", (4, 8))])
def test_f4_3b_undistorted_mesh_rate(name, ns):
    und = [_level(name, n, 0, undistorted=True)[None] for n in ns]
    pu = _pair(und[0], und[1], ns[0], ns[1], _KEY_RATE[name])
    print(f"\n  F4-3b {name} {ns}: undistorted {_KEY_RATE[name]} pair rate {pu:.4f}")
    assert pu >= 1.8


_FINER = {"T4": (16, 24, 32), "T10": (8, 12, 16)}


@slow
@pytest.mark.parametrize("name", ["T4", "T10"])
def test_f4_4_further_solved_levels(name):
    ns = _FINER[name]
    key = _KEY_RATE[name]
    e = {ns[0]: _level(name, ns[0], 100 + ns[0])[None]}
    e[ns[1]] = _level(name, ns[1], 100 + ns[1])[None]
    p1 = _pair(e[ns[0]], e[ns[1]], ns[0], ns[1], key)
    print(f"\n  F4-4 {name} p({ns[0]},{ns[1]}) {key} = {p1:.4f} (L2 {_pair(e[ns[0]], e[ns[1]], ns[0], ns[1], 'l2'):.4f}, "
          f"energy {_pair(e[ns[0]], e[ns[1]], ns[0], ns[1], 'energy'):.4f})")
    if name == "T4":
        assert abs(p1 - 1.92) <= 0.01 and p1 >= 1.9
    else:
        assert abs(p1 - 1.89) <= 0.01


@slow
@pytest.mark.parametrize("name", ["T4", "T10"])
def test_f4_4b_third_solved_level_when_memory_allows(name):
    """Third level (T4 n = 32, T10 n = 16) under the memory rule of the preregistration; skipped, not guessed, otherwise."""
    ns = _FINER[name]
    key = _KEY_RATE[name]
    e = {n: _level(name, n, 100 + n)[None] for n in ns[:2]}
    ok, why = _may_run(name, ns[1])
    if not ok:
        pytest.skip(f"{name} n={ns[2]} not run: {why}")
    e[ns[2]] = _level(name, ns[2], 100 + ns[2])[None]
    p2 = _pair(e[ns[1]], e[ns[2]], ns[1], ns[2], key)
    print(f"\n  F4-4 {name} p({ns[1]},{ns[2]}) {key} = {p2:.4f} (L2 {_pair(e[ns[1]], e[ns[2]], ns[1], ns[2], 'l2'):.4f}, "
          f"energy {_pair(e[ns[1]], e[ns[2]], ns[1], ns[2], 'energy'):.4f})")
    assert p2 >= 1.8

