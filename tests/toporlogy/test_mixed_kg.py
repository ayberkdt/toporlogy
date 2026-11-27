"""K/G mixed elasticity, incompressibility, slender-beam locking and reduced-integration modes (X1-X4).

Pre-registered in ``docs/foundations/prereg/X_mixed.md`` (2026-11-27 02:54 +0300) before the first run; derivations
and thresholds are there. Synthetic materials only (E = 1, unit square). References: the manufactured solution of
:mod:`toporlogy.fem.mesh.manufactured`, the Timoshenko–Goodier cantilever energy by ``scipy.integrate.dblquad``,
dense numpy element matrices written in this file, ``solvers.diagnostics`` (inertia, gauge solve, MINRES). Where a
check is a regression against the code itself it says so.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import dblquad

from toporlogy.fem.mesh import Dirichlet, ElasticProblem, MixedElasticity, rectangle
from toporlogy.fem.mesh.benchmarks import cantilever_case
from toporlogy.fem.mesh.linear import Factorization, condition_1norm, jacobi_scaled
from toporlogy.fem.mesh.manufactured import Manufactured
from toporlogy.fem.mesh.mesh import perturb
from toporlogy.fem.mesh.mixed import MixedKG
from toporlogy.fem.mesh.problem import stiffness_nullity
from toporlogy.solvers.diagnostics import check_compatible, inertia, krylov_solve, solve_with_gauge

BASELINE = Path(__file__).parent / "baseline" / "mixed_lambda_fingerprint.json"


# ------------------------------------------------------------------------------------------------ helpers
def kg(nu: float, E: float = 1.0) -> tuple[float, float, float]:
    """(G, K, λ) from E and ν (ν = 0.5: K = λ = ∞)."""
    G = E / (2 * (1 + nu))
    if nu == 0.5:
        return G, math.inf, math.inf
    K = E / (3 * (1 - 2 * nu))
    return G, K, K - 2 * G / 3


def plane_strain_D(E: float, nu: float) -> np.ndarray:
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, (1 - 2 * nu) / 2]])


def clamp(m):
    return Dirichlet.fixed(m.dofs(m.boundary_nodes()))


def mms_load(prob: ElasticProblem, ms: Manufactured, mu: float, k: int) -> np.ndarray:
    return prob.body_force(lambda X: ms.body_force(mu, X), degree=2 * k + 4)


def tri_div(m, U: np.ndarray) -> np.ndarray:
    """div u_h per T3 cell by the closed gradient formula (independent of the kernels)."""
    X = m.points[m.cells]                                    # (ne, 3, 2)
    Uc = U.reshape(-1, 2)[m.cells]
    x, y = X[..., 0], X[..., 1]
    a2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    dNdx = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1) / a2[:, None]
    dNdy = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1) / a2[:, None]
    return (dNdx * Uc[..., 0]).sum(1) + (dNdy * Uc[..., 1]).sum(1)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


# ------------------------------------------------------------------------------------------------ X1
@pytest.mark.parametrize("nu", [-0.5, 0.0, 0.3])
def test_x1a_t3_p0_k_based_equals_the_displacement_element(nu):
    G, K, lam = kg(nu)
    m = rectangle(8, 8, element="T3")
    ms = Manufactured(2, 1.0, lam * 1.0)
    mx = MixedKG(m, G, K, pressure="P0")
    ref = ElasticProblem(m, plane_strain_D(1.0, nu))
    f = mms_load(ref, ms, G, 1)
    s = mx.solve(f, clamp(m))
    u_ref = ref.solve(f, clamp(m)).u[:, 0]
    p_ref = -K * tri_div(m, u_ref)
    eu, ep = rel(s.u[:, 0], u_ref), rel(s.p[:, 0], p_ref)
    print(f"\n  X1a nu={nu}: u {eu:.2e}, p {ep:.2e}")
    assert eu <= 1e-9
    assert ep <= 1e-8


@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_x1a_negative_taylor_hood_differs_from_the_displacement_element(element):
    G, K, lam = kg(0.3)
    m = rectangle(4, 4, element=element)
    ms = Manufactured(2, 1.0, lam)
    mx = MixedKG(m, G, K)
    ref = ElasticProblem(m, plane_strain_D(1.0, 0.3))
    f = mms_load(ref, ms, G, m.ref.degree)
    d = rel(mx.solve(f, clamp(m)).u[:, 0], ref.solve(f, clamp(m)).u[:, 0])
    print(f"\n  X1a-neg {element}: relative difference {d:.2e}")
    assert d >= 1e-8


def _x1b_case(element, nu, n):
    G, K, lam = kg(nu)
    m = rectangle(n, n, element=element)
    ms = Manufactured(2, 1.0, lam)
    mx = MixedKG(m, G, K)
    ref = ElasticProblem(m, plane_strain_D(1.0, nu))
    f = mms_load(ref, ms, G, m.ref.degree)
    um = mx.solve(f, clamp(m))
    ud = ref.solve(f, clamp(m)).u[:, 0]
    Mm = ref.mass()
    d = um.u[:, 0] - ud
    diff = math.sqrt(d @ (Mm @ d) / (ud @ (Mm @ ud)))
    eu = mx.disp.errors(um.u[:, 0], ms.u, ms.grad)["l2"]
    ep = mx.pressure_error(um.p[:, 0], lambda X: -K * ms.divergence(X))
    return eu, diff, ep


@pytest.mark.parametrize("element", ["T6", "Q9"])
@pytest.mark.parametrize("nu", [0.0, 0.3])
def test_x1b_taylor_hood_k_based_converges_to_the_compressible_solution(element, nu):
    r = [_x1b_case(element, nu, n) for n in (4, 8, 16)]
    rate = [[math.log(r[i][k] / r[i + 1][k]) / math.log(2) for k in range(3)] for i in range(2)]
    print(f"\n  X1b {element} nu={nu}: (u err, diff, p err) {[tuple(f'{v:.2e}' for v in x) for x in r]}; rates {rate}")
    assert rate[1][0] >= 2.5 and rate[1][1] >= 2.5
    assert rate[1][2] >= 1.7


def test_x1c_t3_p0_lambda_and_k_pressures_and_the_mean_stress():
    nu = 0.3
    G, K, lam = kg(nu)
    m = rectangle(8, 8, element="T3")
    ms = Manufactured(2, 1.0, lam)
    ref = ElasticProblem(m, plane_strain_D(1.0, nu))
    f = mms_load(ref, ms, G, 1)
    sk = MixedKG(m, G, K, pressure="P0").solve(f, clamp(m))
    sl = MixedElasticity(m, G, lam, pressure="P0").solve(f, clamp(m))
    div = tri_div(m, sk.u[:, 0])
    e_u = rel(sk.u[:, 0], sl.u[:, 0])
    e_p = rel(sk.p[:, 0], sl.p[:, 0] - 2 * G / 3 * div)
    sig = ref.stress(sk.u[:, 0])[:, 0, :]
    p_mean = -(1 + nu) * (sig[:, 0] + sig[:, 1]) / 3
    e_m = rel(sk.p[:, 0], p_mean)
    print(f"\n  X1c: u_K vs u_lambda {e_u:.2e}; p_K vs p_lambda - 2G/3 div {e_p:.2e}; p_K vs mean stress {e_m:.2e}")
    assert e_u <= 1e-9 and e_p <= 1e-8 and e_m <= 1e-8
    # λ-based refuses ν = 0, K-based accepts it
    with pytest.raises(ValueError, match="λ = 0"):
        MixedElasticity(m, 1.0, 0.0)
    G0, K0, _ = kg(0.0)
    assert MixedKG(m, G0, K0, pressure="P0").lam == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        MixedKG(m, 1.0, 0.0)
    with pytest.raises(ValueError):
        MixedKG(m, 0.0, 1.0)


def test_x1d_lambda_based_class_matches_the_pre_change_fingerprint():
    """Regression (not independent): outputs of the unmodified class recorded before MixedKG was added."""
    fp = json.loads(BASELINE.read_text())
    for key, lam in (("lam7", 7.0), ("laminf", math.inf)):
        m = rectangle(3, 3, element="T6")
        mx = MixedElasticity(m, 1.0, lam)
        f = np.zeros(mx.ndof_u)
        f[1::2] = np.sin(np.arange(len(f[1::2])))
        s = mx.solve(f, clamp(m))
        u0 = np.array([float.fromhex(h) for h in fp[key]["u"]])
        p0 = np.array([float.fromhex(h) for h in fp[key]["p"]])
        bit = np.array_equal(s.u[:, 0], u0) and np.array_equal(s.p[:, 0], p0)
        print(f"\n  X1d {key}: bitwise equal = {bit}")
        np.testing.assert_allclose(s.u[:, 0], u0, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(s.p[:, 0], p0, rtol=1e-13, atol=1e-15)
        assert s.gauge == fp[key]["gauge"]


# ------------------------------------------------------------------------------------------------ X2
def _x2_case(element, nu, n, distort=0.2):
    G, K, lam = kg(nu)
    m = rectangle(n, n, element=element)
    m = perturb(m, distort / n, seed=200 + n) if distort else m
    ms = Manufactured.incompressible_limit(2, G, lam)
    mx = MixedKG(m, G, K)
    f = mms_load(mx.disp, ms, G, m.ref.degree)
    s = mx.solve(f, clamp(m))
    eu = mx.disp.errors(s.u[:, 0], ms.u, ms.grad)["l2"]
    if math.isinf(K):
        ep = mx.pressure_error(s.p[:, 0], ms.pressure, zero_mean=True)
    else:
        ep = mx.pressure_error(s.p[:, 0], lambda X: -K * ms.divergence(X))
    eps, g = mx.disp.strain(s.u[:, 0], degree=2 * m.ref.degree + 2)
    dv = eps[..., 0] + eps[..., 1]
    return {"eu": eu, "ep": ep, "div": float(math.sqrt((dv ** 2 * g.dV).sum())), "mx": mx, "s": s, "gauge": s.gauge}


@pytest.mark.parametrize("element", ["T6", "Q9"])
@pytest.mark.parametrize("nu", [0.4999, 0.5])
def test_x2a_taylor_hood_k_based_converges_near_and_at_incompressibility(element, nu):
    r = [_x2_case(element, nu, n) for n in (8, 16, 32)]
    rt = lambda key: math.log(r[1][key] / r[2][key]) / math.log(2)  # noqa: E731
    col = lambda key: [format(x[key], ".2e") for x in r]  # noqa: E731
    print(f"\n  X2a {element} nu={nu}: u {col('eu')} rate {rt('eu'):.2f}; p {col('ep')} rate {rt('ep'):.2f}; "
          f"div {col('div')} rate {rt('div'):.2f}; gauge {r[-1]['gauge']}")
    assert rt("eu") >= 2.5
    assert rt("ep") >= 1.7
    assert r[-1]["gauge"] == (nu == 0.5)
    if nu == 0.5:
        assert rt("div") >= 1.7
        mx, s = r[-1]["mx"], r[-1]["s"]
        Bu = mx.B @ s.u[:, 0]
        scale = abs(mx.B).sum(axis=1).max() * np.abs(s.u).max()
        print(f"  weak constraint: max|B u| = {np.abs(Bu).max():.2e} (scale {scale:.2e})")
        assert np.abs(Bu).max() <= 1e-10 * scale


@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_x2a_error_is_uniform_in_the_bulk_to_shear_ratio(element):
    e3 = _x2_case(element, 0.3, 16)["eu"]
    e4 = _x2_case(element, 0.4999, 16)["eu"]
    print(f"\n  X2a uniformity {element}: e(0.3) {e3:.3e}, e(0.4999) {e4:.3e}")
    assert e4 <= 2 * e3


def _saddle(mx, bc, gauge_row: bool):
    c = bc.dofs
    free = np.setdiff1d(mx.disp.active, c)
    A, B = mx.A[free][:, free], mx.B[:, free]
    C = sp.csr_matrix((mx.n_p, mx.n_p)) if math.isinf(mx.K) else -mx.M / mx.K
    S = sp.bmat([[A, B.T], [B, C]], format="csr")
    return S, len(free)


def test_x2b_inertia_of_the_saddle_matrix():
    m = rectangle(4, 4, element="T6")
    bc = clamp(m)
    for nu in (0.3, 0.4999):
        G, K, _ = kg(nu)
        mx = MixedKG(m, G, K)
        S, nu_free = _saddle(mx, bc, False)
        i = inertia(S)
        print(f"\n  X2b nu={nu}: inertia {i}, (n_u, n_p) = ({nu_free}, {mx.n_p})")
        assert i == (nu_free, mx.n_p, 0)
    G, K, _ = kg(0.5)
    mx = MixedKG(m, G, K)
    S0, nf = _saddle(mx, bc, False)
    i0 = inertia(S0)
    Sg, _, gauge = mx.system(bc)
    ig = inertia(Sg)
    print(f"\n  X2b K=inf clamped: no gauge {i0}, with gauge row {ig}, (n_u, n_p) = ({nf}, {mx.n_p}), gauge {gauge}")
    assert gauge
    assert i0 == (nf, mx.n_p - 1, 1)
    assert ig == (nf + 1, mx.n_p, 0)
    mc = rectangle(4, 2, lx=2.0, ly=1.0, element="T6")
    left = np.flatnonzero(np.abs(mc.points[:, 0]) < 1e-12)
    bcc = Dirichlet.fixed(mc.dofs(left))
    mxc = MixedKG(mc, G, K)
    Sc, nfc = _saddle(mxc, bcc, False)
    ic = inertia(Sc)
    print(f"\n  X2b cantilever K=inf: inertia {ic}, (n_u, n_p) = ({nfc}, {mxc.n_p}), gauge {mxc.system(bcc)[2]}")
    assert not mxc.system(bcc)[2]
    assert ic == (nfc, mxc.n_p, 0)


def _inf_sup(cls, element, pressure, n):
    m = rectangle(n, n, element=element)
    mx = cls(m, 1.0, math.inf, pressure=pressure)
    w = mx.inf_sup(clamp(m))
    zero = int((w <= 1e-10 * w.max()).sum())
    return zero, (math.sqrt(w[zero]) if zero < len(w) else 0.0)


@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_x2c_k_based_taylor_hood_inf_sup_is_bounded(element):
    out = [_inf_sup(MixedKG, element, "vertex", n) for n in (4, 8, 16)]
    print(f"\n  X2c KG TH {element}: (zero modes, beta) = {out}")
    assert all(z == 1 for z, _ in out)
    assert out[-1][1] >= 0.5 * out[0][1]


@pytest.mark.parametrize("pressure", ["vertex", "P0"])
def test_x2c_negative_control_equal_order_and_p0_have_spurious_modes(pressure):
    out = [_inf_sup(MixedKG, "Q4", pressure, n) for n in (4, 8, 16)]
    print(f"\n  X2c KG Q4-{pressure}: (zero modes, beta) = {out}")
    assert all(z >= 2 for z, _ in out)


def test_x2d_gauge_compatibility_and_the_old_class():
    G, K, lam = kg(0.5)
    m = rectangle(8, 8, element="T6")
    ms = Manufactured.incompressible_limit(2, G, lam)
    bc = clamp(m)
    mx = MixedKG(m, G, K)
    f = mms_load(mx.disp, ms, G, 2)
    s = mx.solve(f, bc)
    assert s.gauge
    mean = float(np.ones(mx.n_p) @ (mx.M @ s.p[:, 0]))
    S0, nf = _saddle(mx, bc, False)
    free = np.setdiff1d(mx.disp.active, bc.dofs)
    rhs = np.concatenate([f[free, 0] if f.ndim == 2 else f[free], np.zeros(mx.n_p)])
    N = np.concatenate([np.zeros(nf), np.ones(mx.n_p)])
    assert check_compatible(S0, rhs, N)
    x = solve_with_gauge(S0, rhs, N)
    eu = np.abs(s.u[free, 0] - x[:nf]).max() / np.abs(x[:nf]).max()
    d = s.p[:, 0] - x[nf:]
    ep = np.abs(d - d.mean()).max() / np.abs(s.p).max()
    print(f"\n  X2d: mean {mean:.2e}, u vs solve_with_gauge {eu:.2e}, p up to a constant {ep:.2e}")
    assert abs(mean) <= 1e-12 * np.abs(s.p).max()
    assert eu <= 1e-8 and ep <= 1e-8
    # a boundary displacement with net inflow (u = x - 1/2 on the whole boundary): no incompressible solution
    bn = m.boundary_nodes()
    X = m.points[bn]
    bad = Dirichlet([(2 * bn, X[:, 0] - 0.5), (2 * bn + 1, X[:, 1] - 0.5)])
    Uc = np.broadcast_to(bad.values, (len(bad.dofs), 1))
    free2 = np.setdiff1d(mx.disp.active, bad.dofs)
    rhs2 = np.concatenate([f[free2] - mx.A[free2][:, bad.dofs] @ Uc[:, 0], -(mx.B[:, bad.dofs] @ Uc[:, 0])])
    N2 = np.concatenate([np.zeros(len(free2)), np.ones(mx.n_p)])
    S2, _ = _saddle(mx, bad, False)
    assert not check_compatible(S2, rhs2, N2)
    with pytest.raises(ValueError, match="net volume"):
        mx.solve(f, bad)
    old = MixedElasticity(m, G, math.inf)
    so = old.solve(f, bad)                       # characterisation: the λ-based class does not check
    U = so.u[:, 0]
    viol = float(np.abs(old.B @ U).max())
    print(f"  old class on net-inflow data: no error, max|B u| = {viol:.2e}")
    assert viol >= 1e-3


def _minres_its(n: int, precond: bool) -> int:
    G, K, lam = kg(0.4999)
    m = rectangle(n, n, element="T6")
    ms = Manufactured.incompressible_limit(2, G, lam)
    mx = MixedKG(m, G, K)
    f = mms_load(mx.disp, ms, G, 2)
    bc = clamp(m)
    S, _, _ = mx.system(bc)
    free = np.setdiff1d(mx.disp.active, bc.dofs)
    nf = len(free)
    rhs = np.concatenate([f[free], np.zeros(mx.n_p)]) if f.ndim == 1 else np.concatenate([f[free, 0], np.zeros(mx.n_p)])
    Mop = None
    if precond:
        fa = spla.splu(sp.csc_matrix(mx.A[free][:, free]))
        fm = spla.splu(sp.csc_matrix((1 / G + 1 / K) * mx.M))
        Mop = spla.LinearOperator(S.shape, matvec=lambda v: np.concatenate([fa.solve(v[:nf]), fm.solve(v[nf:])]))
    r = krylov_solve(S, rhs, "minres", "symmetric_indefinite", M=Mop, rtol=1e-6, maxiter=40 * S.shape[0])
    print(f"\n  X2e n={n} precond={precond}: {r.status}, {r.iterations} iterations, residual {r.residual:.1e}")
    if precond:     # (set-up correction after the first run: the plain control is only counted, see the prereg record)
        assert r.converged
        x = Factorization(S, "symmetric_indefinite").solve(rhs).reshape(-1)
        assert np.abs(r.x[:nf] - x[:nf]).max() / np.abs(x[:nf]).max() <= 1e-3
    return r.iterations


def test_x2e_block_preconditioned_minres_is_mesh_independent():
    p8, p16 = _minres_its(8, True), _minres_its(16, True)
    print(f"\n  X2e: preconditioned {p8} -> {p16}")
    assert p16 <= 1.5 * p8


@pytest.mark.xfail(strict=True, reason="pre-registered X2e negative control: plain MINRES iterations(16)/iterations(8) "
                                       ">= 1.5; measured 148 / 120 = 1.23, and neither run reached rtol 1e-6 "
                                       "(true residuals 3.9e-4 and 1.8e-3: MINRES stops on its recursive estimate)")
def test_x2e_plain_minres_iterations_grow_with_refinement():
    u8, u16 = _minres_its(8, False), _minres_its(16, False)
    print(f"\n  X2e plain: {u8} -> {u16}")
    assert u16 / u8 >= 1.5


# ------------------------------------------------------------------------------------------------ X3
NU_B, LBEAM = 0.3, 32.0


def _u_exact_beam() -> float:
    """Strain energy of the Timoshenko–Goodier cantilever (P = D = E = 1, plane stress, ν = 0.3, L = 32) by quadrature:
    ∫∫ σx²/2E + τ²/2G with σx = −P (L−x) y / I, τ = P (D²/4 − y²) / (2I), I = D³/12."""
    I_ = 1.0 / 12
    G = 1.0 / (2 * (1 + NU_B))

    def dens(y, x):
        sx = -(LBEAM - x) * y / I_
        t = (0.25 - y * y) / (2 * I_)
        return sx * sx / 2 + t * t / (2 * G)
    val, _ = dblquad(dens, 0.0, LBEAM, -0.5, 0.5, epsabs=1e-10, epsrel=1e-12)
    return val


U_EX = _u_exact_beam()


def _beam(element: str, ny: int, nx: int):
    m = rectangle(nx, ny, lx=LBEAM, ly=1.0, element=element, origin=(0.0, -0.5))
    E, nu = 1.0, NU_B
    D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    P = ElasticProblem(m, D)
    I_ = 1.0 / 12

    def tr(sign):
        return lambda X: sign * np.column_stack([-(LBEAM - X[:, 0]) * X[:, 1] / I_, (0.25 - X[:, 1] ** 2) / (2 * I_)])
    f = P.traction(m.facets_where(lambda X: np.abs(X[:, 0]) < 1e-9), tr(-1.0)) + \
        P.traction(m.facets_where(lambda X: np.abs(X[:, 0] - LBEAM) < 1e-9), tr(1.0))
    used = m.used_nodes()

    def node(x, y):
        return int(used[np.argmin(np.hypot(m.points[used, 0] - x, m.points[used, 1] - y))])
    a, b = node(0.0, 0.0), node(0.0, 0.5)
    fixed = np.array([2 * a, 2 * a + 1, 2 * b])
    s = P.solve(f, Dirichlet.fixed(fixed))
    return P, f, fixed, s


def _R(element, ny, nx):
    P, f, fixed, s = _beam(element, ny, nx)
    return 0.5 * float(f @ s.u[:, 0]) / U_EX


def _d(R: float) -> float:
    return 1.0 / R - 1.0


def _d_pred(hx_over_D: float) -> float:
    return (1 - NU_B) / 2 * hx_over_D ** 2


def test_x3_own_cantilever_reproduces_the_benchmark_module():
    mine = _R("Q4", 2, 16)
    prod = cantilever_case("Q4", LBEAM, 2, nx=16)["R"]
    print(f"\n  X3 set-up check: own R = {mine:.10f}, cantilever_case R = {prod:.10f}")
    assert abs(mine - prod) <= 1e-10


def test_x3a_q4_shear_locking_follows_the_parasitic_shear_law_and_q9_does_not_lock():
    hx = {32: 1.0, 16: 2.0, 8: 4.0}
    d = {nx: _d(_R("Q4", 2, nx)) for nx in hx}
    q9 = {nx: _R("Q9", 2, nx) for nx in hx}
    ratio = {nx: d[nx] / _d_pred(hx[nx]) for nx in hx}
    print(f"\n  X3a Q4 d = {d}; d/d_pred = {ratio}; d(2)/d(1) = {d[16] / d[32]:.2f}, d(4)/d(2) = {d[8] / d[16]:.2f}; "
          f"Q9 R = {q9}")
    assert all(0.3 <= r <= 1.5 for r in ratio.values())
    assert 2.5 <= d[16] / d[32] <= 6 and 2.5 <= d[8] / d[16] <= 6
    assert all(r >= 0.95 for r in q9.values())


def test_x3b_incompatible_modes_remove_the_deficit_on_square_elements():
    out = {}
    for ny in (1, 2, 4):
        q4 = cantilever_case("Q4", LBEAM, ny)["R"]
        q4im = cantilever_case("Q4IM", LBEAM, ny)["R"]
        out[ny] = (q4, q4im, _d(q4) / _d_pred(1.0 / ny))
    print(f"\n  X3b (R_Q4, R_Q4IM, d/d_pred): {out}")
    assert all(v[1] >= 0.95 for v in out.values())
    assert all(0.3 <= v[2] <= 1.5 for v in out.values())


def test_x3c_locking_is_not_conditioning():
    rows = []
    for ny in (2, 4, 8):
        P, f, fixed, s = _beam("Q4", ny, 8 * ny)
        free = np.setdiff1d(P.active, fixed)
        K = P.K[free][:, free]
        kap = condition_1norm(K, P.factorization())
        R = 0.5 * float(f @ s.u[:, 0]) / U_EX
        rows.append((ny, kap, _d(R), P, f, free, s))
    print("\n  X3c (ny, kappa1, d): " + str([(r[0], f"{r[1]:.2e}", f"{r[2]:.3e}") for r in rows]))
    for lo, hi in zip(rows[:-1], rows[1:], strict=True):
        assert hi[1] / lo[1] >= 2.5            # conditioning gets worse with refinement ...
        assert lo[2] / hi[2] >= 2.5            # ... while the locking deficit gets better
    ny, kap, d, P, f, free, s = rows[0]
    K = P.K[free][:, free]
    u = s.u[free, 0]
    Ks, sc = jacobi_scaled(K)
    us = sc * Factorization(Ks, "spd").solve(sc * f[free, 0] if f.ndim == 2 else sc * f[free])
    fb = f[free, 0] if f.ndim == 2 else f[free]
    corr = P.factorization().solve(fb - K @ u)
    c1 = float(np.abs(us - u).max() / np.abs(u).max())
    c2 = float(np.abs(corr).max() / np.abs(u).max())
    print(f"  worst mesh: scaled-solve change {c1:.1e}, refinement change {c2:.1e}")
    assert c1 <= 1e-8 and c2 <= 1e-8


# ------------------------------------------------------------------------------------------------ X4
E_, NU_ = 1.0, 0.3
G_, K_ = E_ / (2 * (1 + NU_)), E_ / (3 * (1 - 2 * NU_))
D_PE = plane_strain_D(E_, NU_)
M3 = np.array([1.0, 1.0, 0.0])
D_VOL = K_ * np.outer(M3, M3)
D_DEV = np.array([[4 * G_ / 3, -2 * G_ / 3, 0], [-2 * G_ / 3, 4 * G_ / 3, 0], [0, 0, G_]])


def _bmat(xi: float, eta: float) -> tuple[np.ndarray, float]:
    """B (3×8) and detJ of the unit-square Q4 (h = 1) at reference point (xi, eta); dofs (ux, uy) per node,
    nodes (0,0), (1,0), (1,1), (0,1)."""
    sx, sy = np.array([-1, 1, 1, -1.0]), np.array([-1, -1, 1, 1.0])
    dNdxi = 0.25 * sx * (1 + sy * eta)
    dNdeta = 0.25 * sy * (1 + sx * xi)
    dNdx, dNdy = 2 * dNdxi, 2 * dNdeta                       # x = (xi+1)/2: dxi/dx = 2
    B = np.zeros((3, 8))
    B[0, 0::2] = dNdx
    B[1, 1::2] = dNdy
    B[2, 0::2] = dNdy
    B[2, 1::2] = dNdx
    return B, 0.25


def _ke(mode: str) -> np.ndarray:
    g = 1 / math.sqrt(3)
    gp = [(a, b) for a in (-g, g) for b in (-g, g)]
    if mode == "full":
        return sum(w * 1.0 * (lambda B, J: B.T @ D_PE @ B * J)(*_bmat(*p)) for p in gp for w in [1.0])
    if mode == "reduced":
        B, J = _bmat(0.0, 0.0)
        return 4 * J * B.T @ D_PE @ B
    if mode == "sri":
        Kd = sum((lambda B, J: B.T @ D_DEV @ B * J)(*_bmat(*p)) for p in gp)
        B, J = _bmat(0.0, 0.0)
        return Kd + 4 * J * B.T @ D_VOL @ B
    raise ValueError(mode)


def _assemble(n: int, mode: str) -> np.ndarray:
    ke = _ke(mode)
    nn = (n + 1) ** 2
    idx = lambda i, j: i * (n + 1) + j  # noqa: E731
    K = np.zeros((2 * nn, 2 * nn))
    for i in range(n):
        for j in range(n):
            nodes = [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            dofs = np.array([[2 * a, 2 * a + 1] for a in nodes]).ravel()
            K[np.ix_(dofs, dofs)] += ke
    return K


def _nullity(K: np.ndarray, rtol: float = 1e-10) -> int:
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    return int((np.abs(w) <= rtol * np.abs(w).max()).sum())


def test_x4_set_up_the_dense_split_reproduces_the_plane_strain_matrix():
    np.testing.assert_allclose(D_DEV + D_VOL, D_PE, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_x4a_zero_energy_mode_counts(n):
    full, red, sri = (_nullity(_assemble(n, m)) for m in ("full", "reduced", "sri"))
    lower = max(5, 2 * (n + 1) ** 2 - 3 * n * n)
    m = rectangle(n, n, element="Q4")
    prod_full = stiffness_nullity(ElasticProblem(m, D_PE).K.toarray())
    prod_red = stiffness_nullity(ElasticProblem(m, D_PE, degree=1).K.toarray())
    print(f"\n  X4a n={n}: full {full}, reduced {red} (lower bound {lower}), SRI {sri}; production full "
          f"{prod_full}, one-point {prod_red}")
    assert full == 3 and sri == 3
    assert red >= lower
    if n == 1:
        assert red == 5
    assert prod_full == full and prod_red == red


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_x4b_checkerboard_displacements_are_zero_energy_under_one_point_integration(n):
    K = _assemble(n, "reduced")
    nn = (n + 1) ** 2
    chk = np.array([(-1.0) ** (i + j) for i in range(n + 1) for j in range(n + 1)])
    zx, zy = np.zeros(2 * nn), np.zeros(2 * nn)
    zx[0::2], zy[1::2] = chk, chk
    nK = np.linalg.norm(K, 2)
    r = [float(np.linalg.norm(K @ z) / nK) for z in (zx, zy)]
    Kf = _assemble(n, "full")
    rf = float(np.linalg.norm(Kf @ zx) / np.linalg.norm(Kf, 2))
    print(f"\n  X4b n={n}: |K z|/|K| one-point {r}, full integration {rf:.2e}")
    assert max(r) <= 1e-12
