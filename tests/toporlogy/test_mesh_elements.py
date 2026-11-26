"""Element contract of the general mesh (docs/olcumler/fem_genel_onkayit.md §1, E1-E9, pre-registered
2026-11-26 02:41 +0300 before the first run): shape functions, quadrature exactness, signed detJ check, element
stiffness rank, agreement with the grid element matrices."""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from toporlogy.core.linear_elastic import isotropic_plane_strain, isotropic_plane_stress, isotropic_stiffness
from toporlogy.fem.elements import ke_h8_from_D, ke_q4_from_D
from toporlogy.fem.mesh import (DegenerateElementError, ElasticProblem, InvertedElementError, Mesh, box, get,
                                rectangle, straighten)
from toporlogy.fem.mesh.problem import zero_energy_modes
from toporlogy.fem.mesh.quadrature import MEASURE, rule
from toporlogy.fem.mesh.reference import monomial_exponents

ALL = ("L2", "L3", "Q4", "Q9", "H8", "T3", "T6", "T4", "T10")
SOLID = ("Q4", "Q9", "T3", "T6", "H8", "T4", "T10")


def _random_points(ref, n, seed=0):
    rng = np.random.default_rng(seed)
    if ref.cell in ("line", "quad", "hex"):
        return rng.uniform(-1, 1, size=(n, ref.dim))
    P = rng.dirichlet(np.ones(ref.dim + 1), size=n)[:, 1:]
    return P


@pytest.mark.parametrize("name", ALL)
def test_e1_kronecker(name):
    ref = get(name)
    err = np.abs(ref.shape(ref.nodes) - np.eye(ref.n_nodes)).max()
    assert err <= 1e-14, err


@pytest.mark.parametrize("name", ALL)
def test_e2_partition_of_unity_and_derivative_sum(name):
    ref = get(name)
    X = _random_points(ref, 50)
    assert np.abs(ref.shape(X).sum(axis=1) - 1).max() <= 1e-14
    assert np.abs(ref.grad(X).sum(axis=1)).max() <= 1e-13


@pytest.mark.parametrize("name", ALL)
def test_e3_derivatives_against_central_differences(name):
    ref = get(name)
    X = _random_points(ref, 20, seed=1) * 0.9
    h = 1e-6
    G = ref.grad(X)
    err = 0.0
    for d in range(ref.dim):
        e = np.zeros(ref.dim)
        e[d] = h
        fd = (ref.shape(X + e) - ref.shape(X - e)) / (2 * h)
        err = max(err, float(np.abs(fd - G[:, :, d]).max()))
    assert err <= 1e-8, err


@pytest.mark.parametrize("name", ALL)
def test_e4_completeness(name):
    ref = get(name)
    X = _random_points(ref, 30, seed=2)
    N = ref.shape(X)
    for e in monomial_exponents(ref):
        f = lambda P, e=e: np.prod(P ** np.array(e), axis=1)
        err = np.abs(N @ f(ref.nodes) - f(X)).max()
        assert err <= 1e-13, (e, err)


def _exact_monomial(cell, e):
    if cell in ("line", "quad", "hex"):
        return float(np.prod([2.0 / (a + 1) if a % 2 == 0 else 0.0 for a in e]))
    return float(np.prod([math.factorial(a) for a in e]) / math.factorial(sum(e) + len(e)))


@pytest.mark.parametrize("cell", ["line", "quad", "hex", "tri", "tet"])
@pytest.mark.parametrize("degree", range(9))
def test_e5_quadrature_exactness(cell, degree):
    q = rule(cell, degree)
    assert q.degree >= degree
    assert abs(q.weights.sum() - MEASURE[cell]) <= 1e-14 * MEASURE[cell]
    dim = q.points.shape[1]
    simplex = cell in ("tri", "tet")
    worst = 0.0
    for e in product(range(q.degree + 1), repeat=dim):
        if simplex and sum(e) > q.degree:
            continue
        got = float(q.weights @ np.prod(q.points ** np.array(e), axis=1))
        ex = _exact_monomial(cell, e)
        worst = max(worst, abs(got - ex) / max(abs(ex), _exact_monomial(cell, tuple(2 * (a // 2) for a in e))))
    assert worst <= 1e-14, worst


@pytest.mark.parametrize("cell", ["line", "quad", "hex"])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_e5b_gauss_rule_is_tight(cell, n):
    q = rule(cell, 2 * n - 1)
    assert len(q.weights) == n ** q.points.shape[1]
    e = (2 * n,) + (0,) * (q.points.shape[1] - 1)
    got = float(q.weights @ np.prod(q.points ** np.array(e), axis=1))
    assert abs(got - _exact_monomial(cell, e)) > 1e-10


def _single(name, distort=None):
    ref = get(name)
    A = np.array([[1.0, 0.3, 0.1], [0.2, 0.9, -0.15], [0.05, 0.1, 1.1]])[:ref.dim, :ref.dim]
    X = ref.nodes @ A.T + 0.5
    m = straighten(Mesh(X, np.arange(ref.n_nodes)[None, :], name))
    if distort is not None:
        m.points[distort[0]] += distort[1]
    return m


def _D(dim, nu=0.3):
    return isotropic_plane_strain(1.0, nu) if dim == 2 else isotropic_stiffness(1.0, nu)


def _check_rank(K, dim):
    nrbm = 3 if dim == 2 else 6
    w = np.sort(np.abs(np.linalg.eigvalsh(K)))
    assert np.abs(K - K.T).max() <= 1e-14 * np.abs(K).max()
    assert (w <= 1e-10 * w[-1]).sum() == nrbm, w[:nrbm + 2] / w[-1]
    assert w[nrbm] > 1e-8 * w[-1], w[nrbm] / w[-1]


@pytest.mark.parametrize("name", SOLID)
def test_e7_element_stiffness_rank_affine(name):
    m = _single(name)
    K = ElasticProblem(m, _D(m.dim)).element_stiffness()[0]
    _check_rank(K, m.dim)


NONAFFINE = {"Q4": (2, [0.15, 0.1]), "Q9": (2, [0.15, 0.1]), "T6": (3, [0.06, 0.05]),
             "H8": (6, [0.15, 0.1, -0.1]), "T10": (5, [0.04, 0.05, -0.03])}


@pytest.mark.parametrize("name", list(NONAFFINE))
def test_e7_element_stiffness_rank_nonaffine(name):
    node, dx = NONAFFINE[name]
    m = _single(name, (node, np.array(dx)))
    if name == "Q9":
        m = straighten(m)                     # bilinear map: corner moved, mid nodes follow
    K = ElasticProblem(m, _D(m.dim)).element_stiffness()[0]
    _check_rank(K, m.dim)


def test_e8_general_q4_h8_equal_grid_element_matrices():
    nu = 0.3
    D2 = isotropic_plane_stress(1.0, nu)
    K = ElasticProblem(rectangle(1, 1, element="Q4"), D2).element_stiffness()[0]
    ref = ke_q4_from_D(D2)
    assert np.abs(K - ref).max() <= 1e-14 * np.abs(ref).max()
    D3 = isotropic_stiffness(1.0, nu)
    K3 = ElasticProblem(box(1, 1, 1, element="H8"), D3).element_stiffness()[0]
    ref3 = ke_h8_from_D(D3, "H8")[0]
    assert np.abs(K3 - ref3).max() <= 1e-14 * np.abs(ref3).max()


def test_e9_reduced_integration_q4_shows_two_hourglass_modes():
    K = ElasticProblem(rectangle(1, 1, element="Q4"), _D(2), degree=1).element_stiffness()[0]
    assert len(rule("quad", 1).weights) == 1
    assert zero_energy_modes(K) == 3 + 2


@pytest.mark.parametrize("name", SOLID)
def test_e6_inverted_cell_is_refused_and_named(name):
    m = rectangle(3, 2, element=name) if get(name).dim == 2 else box(2, 2, 1, element=name)
    k = 3
    P = m.points.copy()
    P[:, 0] *= -1.0                             # mirror the whole mesh: every cell inverted
    with pytest.raises(InvertedElementError) as ei:
        ElasticProblem(Mesh(P, m.cells, name), _D(m.dim))
    assert len(ei.value.elements) == m.n_cells
    # one inverted cell among good ones: swap two vertices (and their edge nodes stay: still inverted)
    C = m.cells.copy()
    C[k] = C[k][np.array(_mirror_perm(get(name)))]
    with pytest.raises(InvertedElementError) as ei:
        ElasticProblem(Mesh(m.points, C, name), _D(m.dim))
    assert ei.value.elements.tolist() == [k]
    assert str(k) in str(ei.value)


def _mirror_perm(ref):
    """Local node permutation that mirrors the reference cell (reverses orientation)."""
    X = ref.nodes.copy()
    if ref.cell in ("quad", "hex"):
        Y = X.copy()
        Y[:, 0] *= -1
    else:
        Y = X[:, [1, 0] + list(range(2, ref.dim))]
    return [int(np.flatnonzero(np.abs(X - y).sum(axis=1) < 1e-12)[0]) for y in Y]


@pytest.mark.parametrize("name", SOLID)
def test_e6_degenerate_cell_is_refused(name):
    m = _single(name)
    ref = get(name)
    m.points[1] = m.points[0]                    # two vertices on top of each other
    m = straighten(m)
    with pytest.raises(DegenerateElementError) as ei:
        ElasticProblem(m, _D(m.dim))
    assert ei.value.elements.tolist() == [0]
    assert ref.n_vertices >= 2
