"""Poisson ratio, locking and conditioning claims (docs/olcumler/fem_genel_onkayit.md §4, L1-L4, pre-registered
2026-11-26 02:41 +0300 before the first run). The full tables are produced by docs/olcumler/fem_genel.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from toporlogy.core.linear_elastic import isotropic_plane_strain
from toporlogy.fem.mesh import Dirichlet, ElasticProblem, MixedElasticity, rectangle
from toporlogy.fem.mesh.benchmarks import cantilever_case, cantilever_energy, mms_case, moduli
from toporlogy.fem.mesh.linear import Factorization, condition_1norm, jacobi_scaled, jacobi_pcg
from toporlogy.fem.mesh.manufactured import Manufactured

N2, N3 = 16, 6


def _fmt(rows, key):
    return [format(x[key], '.2e') for x in rows]


# ---------------------------------------------------------------------------------------------- L1
@pytest.mark.parametrize("method", ["Q4", "T6", "TH:T6-T3", "Q4IM"])
@pytest.mark.parametrize("nu", [-0.5, 0.3, 0.499])
def test_l1a_e_fixed_and_g_fixed_give_the_same_relative_errors(method, nu):
    a = mms_case(method, 8, nu, fixed="E")
    b = mms_case(method, 8, nu, fixed="G")
    assert abs(a["l2"] - b["l2"]) <= 1e-10
    if not math.isnan(a["energy"]):
        assert abs(a["energy"] - b["energy"]) <= 1e-10


_T4_XFAIL = pytest.mark.xfail(strict=True, reason="pre-registered L1b T4 (n = 6): measured ratio 3.7 < 5; "
                              "e(0.4999) = 0.996 (u_h ≈ 0, complete locking) but e(0.3) = 0.27 on this coarse mesh "
                              "caps the ratio at 1/0.27 = 3.8 (fem_genel.md §4)")


@pytest.mark.parametrize("method,n", [("Q4", N2), ("T3", N2), ("H8", N3), pytest.param("T4", N3, marks=_T4_XFAIL)])
def test_l1b_displacement_elements_lock(method, n):
    lo, hi = mms_case(method, n, 0.3), mms_case(method, n, 0.4999)
    ratio = hi["l2"] / lo["l2"]
    print(f"\n  L1b {method}: e(0.3) = {lo['l2']:.3e}, e(0.4999) = {hi['l2']:.3e}, ratio {ratio:.1f}")
    assert ratio >= 5


def test_l1b_t4_posthoc_finer_mesh():
    """Post hoc (not pre-registered): T4 at n = 12, where e(0.3) is smaller, has the ratio ≥ 5."""
    lo, hi = mms_case("T4", 12, 0.3), mms_case("T4", 12, 0.4999)
    print(f"\n  L1b T4 n=12 (post hoc): {lo['l2']:.3e} -> {hi['l2']:.3e}, ratio {hi['l2'] / lo['l2']:.1f}")
    assert hi["l2"] / lo["l2"] >= 5


@pytest.mark.parametrize("method", ["TH:T6-T3", "TH:Q9-Q4"])
def test_l1c_taylor_hood_does_not_lock(method):
    e3, e4, e5 = (mms_case(method, N2, nu)["l2"] for nu in (0.3, 0.4999, 0.5))
    print(f"\n  L1c {method}: {e3:.4e} {e4:.4e} {e5:.4e}")
    assert e4 <= 2 * e3
    assert 0.9 <= e5 / e4 <= 1.1


def test_l1d_same_displacement_space_mixed_beats_displacement_only():
    th = mms_case("TH:T6-T3", N2, 0.4999)["l2"]
    t6 = mms_case("T6", N2, 0.4999)["l2"]
    print(f"\n  L1d: TH {th:.3e} < T6 {t6:.3e}")
    assert th < t6


@pytest.mark.parametrize("method", ["TH:T6-T3", "TH:Q9-Q4"])
@pytest.mark.parametrize("nu", [0.4999, 0.5])
def test_l1e_taylor_hood_converges_near_and_at_incompressibility(method, nu):
    ns = (8, 16, 32)
    r = [mms_case(method, n, nu, variant="B", distort=0.2, seed=200 + n) for n in ns]
    pu = math.log(r[-2]["l2"] / r[-1]["l2"]) / math.log(2)
    pp = math.log(r[-2]["p_error"] / r[-1]["p_error"]) / math.log(2)
    print(f"\n  L1e {method} nu={nu}: u {_fmt(r, 'l2')} p = {pu:.2f}; pressure "
          f"{_fmt(r, 'p_error')} p = {pp:.2f}; gauge {r[-1]['gauge']}")
    assert pu >= 2.8
    assert pp >= 1.8
    assert r[-1]["gauge"] == (nu == 0.5)


# ---------------------------------------------------------------------------------------------- L2
def _inf_sup(element, pressure, n):
    m = rectangle(n, n, element=element)
    mx = MixedElasticity(m, 1.0, math.inf, pressure=pressure)
    w = mx.inf_sup(Dirichlet.fixed(m.dofs(m.boundary_nodes())))
    zero = int((w <= 1e-10 * w.max()).sum())
    beta = math.sqrt(w[zero]) if zero < len(w) else 0.0
    return zero, beta


@pytest.mark.parametrize("element", ["T6", "Q9"])
def test_l2_taylor_hood_inf_sup_is_bounded(element):
    out = [_inf_sup(element, "vertex", n) for n in (4, 8, 16)]
    print(f"\n  L2 TH {element}: (zero modes, beta) = {out}")
    assert all(z == 1 for z, _ in out)
    assert out[-1][1] >= 0.5 * out[0][1]


def test_l2_q4_p0_has_a_spurious_pressure_mode():
    out = [_inf_sup("Q4", "P0", n) for n in (4, 8, 16)]
    print(f"\n  L2 Q4-P0: (zero modes, beta) = {out}")
    assert all(z >= 2 for z, _ in out)


# ---------------------------------------------------------------------------------------------- L3
def test_l3a_bilinear_elements_shear_lock_on_elongated_elements():
    q4 = cantilever_case("Q4", 100, 2, nx=40)["R"]
    h8 = cantilever_case("H8", 100, 2, nx=40, nu=0.0, dim=3)["R"]
    print(f"\n  L3a: Q4 R = {q4:.4f}, H8 R = {h8:.4f}")
    assert q4 <= 0.5 and h8 <= 0.5


@pytest.mark.parametrize("slenderness", [5, 10, 50, 100])
def test_l3b_q9_does_not_shear_lock(slenderness):
    R = cantilever_case("Q9", slenderness, 2, nx=40)["R"]
    print(f"\n  L3b Q9 L/h = {slenderness}: R = {R:.5f}")
    assert R >= 0.95


@pytest.mark.parametrize("slenderness", [10, 50, 100])
def test_l3c_incompatible_modes_on_square_elements(slenderness):
    q = cantilever_case("Q4IM", slenderness, 2)["R"]
    h = cantilever_case("H8IM", slenderness, 2, nu=0.0, dim=3)["R"]
    print(f"\n  L3c L/h = {slenderness}: Q4IM R = {q:.4f}, H8IM R = {h:.4f}")
    assert q >= 0.95 and h >= 0.95


def test_l3d_the_deficit_is_locking_not_shear_deformation():
    U, Us = cantilever_energy(1.0, 1.0, 100.0, 1.0, 0.3)
    frac = Us / U
    q4 = cantilever_case("Q4", 100, 2, nx=40)
    assert abs(q4["shear_fraction"] - frac) <= 1e-15
    print(f"\n  L3d: shear fraction {frac:.2e}, Q4 deficit {1 - q4['R']:.3f}")
    assert frac < 1e-3
    assert 1 - q4["R"] > 100 * frac


# ---------------------------------------------------------------------------------------------- L4
def _q4_system(nu, n=N2):
    E, mu, lam = moduli(nu)
    m = rectangle(n, n, element="Q4")
    P = ElasticProblem(m, isotropic_plane_strain(E, nu))
    ms = Manufactured(2)
    f = P.body_force(lambda X: ms.body_force(mu, X), degree=6)
    c = m.dofs(m.boundary_nodes())
    free = np.setdiff1d(P.active, c)
    s = P.solve(f, Dirichlet.fixed(c))
    return P, f, free, s, ms


_L4: dict[float, tuple] = {}


def _l4(nu):
    if nu not in _L4:
        P, f, free, s, ms = _q4_system(nu)
        K = P.K[free][:, free]
        fac = P.factorization()
        kap = condition_1norm(K, fac)
        Ks, sc = jacobi_scaled(K)
        us = sc * Factorization(Ks, "spd").solve(sc * f[free])
        d_scaled = np.abs(us - s.u[free, 0]).max() / np.abs(s.u[free, 0]).max()
        r = f[free] - K @ s.u[free, 0]
        d_ref = np.abs(fac.solve(r)).max() / np.abs(s.u[free, 0]).max()
        e = P.errors(s.u[:, 0], ms.u, ms.grad)["l2"]
        _, it, conv = jacobi_pcg(K, f[free])
        _L4[nu] = (kap, d_scaled, d_ref, e, it, conv)
        print(f"\n  L4 Q4 nu={nu}: kappa {kap:.2e}, scaled-solve change {d_scaled:.1e}, refinement {d_ref:.1e}, "
              f"e_L2 {e:.3e}, PCG {it} it ({conv})")
    return _L4[nu]


@pytest.mark.xfail(strict=True, reason="pre-registered L4: kappa(0.4999)/kappa(0.3) >= 100; measured 1.27e4 / 1.86e2 "
                                       "= 68 (n = 16: the h^-2 part of kappa dominates at nu = 0.3) (fem_genel.md §4)")
def test_l4_condition_number_grows_100_fold():
    assert _l4(0.4999)[0] / _l4(0.3)[0] >= 100


def test_l4_scaling_and_refinement_do_not_remove_locking():
    """The discrete solution is solved accurately (Jacobi scaling changes nothing, one refinement step changes
    nothing), yet the error ratio stays: the bias is discretisation locking, not conditioning."""
    res = {nu: _l4(nu) for nu in (0.3, 0.4999)}
    assert all(v[1] <= 1e-8 and v[2] <= 1e-8 for v in res.values())
    assert res[0.4999][3] / res[0.3][3] >= 5


def test_l4_pcg_refused_for_the_indefinite_mixed_system():
    m = rectangle(4, 4, element="T6")
    mx = MixedElasticity(m, 1.0, 100.0)
    import scipy.sparse as sp
    K = sp.bmat([[mx.A, mx.B.T], [mx.B, -mx.M / 100.0]], format="csr")
    with pytest.raises(ValueError, match="symmetric positive definite"):
        jacobi_pcg(K, np.ones(K.shape[0]), structure="symmetric_indefinite")
