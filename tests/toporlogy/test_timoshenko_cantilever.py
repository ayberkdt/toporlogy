"""Cantilever with a parabolic end shear against the closed-form plane-stress solution of Timoshenko & Goodier
(docs/olcumler/dogrulama_turu.md §7, H1-H4, pre-registered 2026-11-25 23:05 +0300 before the first run).

The formula is transcribed from memory, so H1 first checks it: the stress derived from u equals the stress formula,
the stress is in equilibrium, the top and bottom faces are traction free and the end shear sums to P. The FE problem
carries the exact tractions on both ends, three zero DOFs hold the rigid motion, and the least-squares rigid motion
of FE − exact is removed before the error is taken.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity

E, NU, P, D = 1.0, 0.3, 1.0, 2.0
L = 4 * D
I_ = D ** 3 / 12
G3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5, 8, 5]) / 9.0)


def u_exact(x, y):
    ux = -P * y / (6 * E * I_) * ((6 * L - 3 * x) * x + (2 + NU) * (y ** 2 - D ** 2 / 4))
    uy = P / (6 * E * I_) * (3 * NU * y ** 2 * (L - x) + (4 + 5 * NU) * D ** 2 * x / 4 + (3 * L - x) * x ** 2)
    return ux, uy


def sigma_exact(x, y):
    return -P * (L - x) * y / I_, np.zeros_like(np.asarray(x, float) * y), P * (D ** 2 / 4 - y ** 2) / (2 * I_)


def test_h1_the_transcribed_solution_is_consistent():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, L, 50), rng.uniform(-D / 2, D / 2, 50)
    dl = 1e-5

    def grad(f, x, y):
        return ((f(x + dl, y) - f(x - dl, y)) / (2 * dl), (f(x, y + dl) - f(x, y - dl)) / (2 * dl))
    uxx, uxy = grad(lambda a, b: u_exact(a, b)[0], x, y)
    uyx, uyy = grad(lambda a, b: u_exact(a, b)[1], x, y)
    c = E / (1 - NU ** 2)
    s_from_u = np.array([c * (uxx + NU * uyy), c * (uyy + NU * uxx), E / (2 * (1 + NU)) * (uxy + uyx)])
    s = np.array(sigma_exact(x, y))
    scale = np.abs(s).max()
    e1 = np.abs(s_from_u - s).max() / scale
    sxx_x = grad(lambda a, b: sigma_exact(a, b)[0], x, y)[0]
    txy_x, txy_y = grad(lambda a, b: sigma_exact(a, b)[2], x, y)
    syy_y = grad(lambda a, b: sigma_exact(a, b)[1], x, y)[1]
    e2 = max(np.abs(sxx_x + txy_y).max(), np.abs(txy_x + syy_y).max()) / (scale / D)
    edge = np.linspace(0, L, 7)
    e3 = max(np.abs(np.array(sigma_exact(edge, np.full(7, s_ * D / 2)))[1:]).max() for s_ in (-1, 1)) / scale
    g, w = G3
    e4 = abs(np.sum(w * sigma_exact(L, g * D / 2)[2]) * D / 2 - P) / P
    print(f"\n  H1: stress from u {e1:.1e}, equilibrium {e2:.1e}, free faces {e3:.1e}, end resultant {e4:.1e}")
    assert max(e1, e2, e3, e4) <= 1e-7


def _edge_loads(d, x_face, normal_sign):
    """Consistent nodal loads of the exact traction on the face x = x_face (outward normal ±x), linear hats."""
    X = d.node_coords()
    nodes = np.flatnonzero(np.abs(X[:, 0] - x_face) < 1e-9)
    nodes = nodes[np.argsort(X[nodes, 1])]
    F = np.zeros(d.ndof)
    g, w = G3
    for a, b in zip(nodes[:-1], nodes[1:], strict=True):
        y0, y1 = X[a, 1], X[b, 1]
        yq = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * g
        sxx, _, txy = sigma_exact(np.full(3, x_face), yq - D / 2)
        t = normal_sign * np.array([sxx, txy])                # traction = σ · n, n = (±1, 0)
        Na, Nb = (y1 - yq) / (y1 - y0), (yq - y0) / (y1 - y0)
        for c in range(2):
            F[2 * a + c] += np.sum(w * t[c] * Na) * (y1 - y0) / 2
            F[2 * b + c] += np.sum(w * t[c] * Nb) * (y1 - y0) / 2
    return F


def solve(n, element):
    d = Domain((4 * n, n), h=D / n)
    X = d.node_coords()
    mid0 = int(np.flatnonzero((X[:, 0] < 1e-9) & (np.abs(X[:, 1] - D / 2) < 1e-9))[0])
    mid1 = int(np.flatnonzero((X[:, 0] > L - 1e-9) & (np.abs(X[:, 1] - D / 2) < 1e-9))[0])
    fixed = np.array([2 * mid0, 2 * mid0 + 1, 2 * mid1 + 1])
    F = _edge_loads(d, 0.0, -1.0) + _edge_loads(d, L, 1.0)
    U = Elasticity(d, fixed, nu=NU, element=element).solve_numpy(np.full(d.n_active, E), F[:, None])[:, 0]
    U = U.reshape(-1, 2)
    x, y = X[:, 0], X[:, 1] - D / 2
    ue = np.column_stack(u_exact(x, y))
    R = np.zeros((2 * len(x), 3))                               # rigid modes (1, 0), (0, 1), (−y, x)
    R[0::2, 0], R[1::2, 1], R[0::2, 2], R[1::2, 2] = 1.0, 1.0, -y, x
    a, *_ = np.linalg.lstsq(R, (U - ue).ravel(), rcond=None)
    Uc = (U.ravel() - R @ a).reshape(-1, 2)
    En = np.abs(Uc - ue).max(1) / np.abs(ue).max()
    err = float(En.max())
    tip = mid1
    tip_err = float(abs(Uc[tip, 1] - ue[tip, 1]) / abs(ue[tip, 1]))
    middle = (x > L / 4) & (x < 3 * L / 4)
    return err, tip_err, float(np.sqrt(np.mean(En ** 2))), float(En[middle].max()), (float(x[En.argmax()]),
                                                                                   float(y[En.argmax()]))


NS = (4, 8, 16)


@pytest.fixture(scope="module")
def results():
    return {el: [solve(n, el) for n in NS] for el in ("Q4", "Q4IM")}


def _rates(e):
    return [float(np.log2(e[i] / e[i + 1])) for i in range(len(e) - 1)]


@pytest.mark.xfail(strict=True, reason=(
    "pre-registered H2 FAILED on the first run: Q4 rates 1.78, 1.82 (n = 4, 8, 16). Diagnostic run (not a criterion): "
    "n = 32, 64 give 1.85, 1.86, approaching 2 slowly from below; a max-norm h^2 |ln h| behaviour would do this "
    "(not verified)"))
def test_h2_q4_converges_at_second_order(results):
    e = [r[0] for r in results["Q4"]]
    print(f"\n  H2 Q4: errors {e}, rates {_rates(e)}")
    assert all(1.8 <= v <= 2.25 for v in _rates(e))


@pytest.mark.xfail(strict=True, reason=(
    "pre-registered H3 FAILED on the first run: Q4IM error 13x below Q4 (2.5e-5 vs 3.4e-4) but rates 1.34, 1.52; "
    "diagnostic n = 32, 64: 1.63, 1.70, still rising; cause not established"))
def test_h3_q4im_is_more_accurate(results):
    e, q = [r[0] for r in results["Q4IM"]], [r[0] for r in results["Q4"]]
    print(f"\n  H3 Q4IM: errors {e}, rates {_rates(e)}")
    assert e[-1] <= 0.5 * q[-1]
    assert e[-1] < 1e-10 or all(v >= 1.8 for v in _rates(e))


def test_h4_tip_deflection(results):
    q4, im = results["Q4"][-1][1], results["Q4IM"][-1][1]
    print(f"\n  H4 tip deflection error: Q4 {q4:.2e}, Q4IM {im:.2e}")
    assert q4 <= 2e-2 and im <= 2e-3


@pytest.mark.parametrize("element", ["Q4", "Q4IM"])
def test_h5_second_order_away_from_the_loaded_corners_post_hoc(results, element):
    """Post hoc (written after H2 and H3 failed; not pre-registered): the largest nodal error sits at a corner of a
    loaded end face, and away from it the rate is second order: RMS over all nodes and max over the middle half
    of the span, both in H2's band [1.8, 2.25]. Measured on the diagnosis run: RMS 2.03-2.14, middle 2.00-2.06."""
    r = results[element]
    corner = [(round(c[0], 9) in (0.0, L)) and (round(abs(c[1]), 9) == D / 2) for c in (v[4] for v in r)]
    rms, mid = _rates([v[2] for v in r]), _rates([v[3] for v in r])
    print(f"\n  H5 {element}: max at {[v[4] for v in r]}, RMS rates {rms}, middle-half rates {mid}")
    assert all(corner)
    assert all(1.8 <= v <= 2.25 for v in rms + mid)
