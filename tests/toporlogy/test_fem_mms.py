"""Independent verification of the FEM by the method of manufactured solutions (docs/gelistirme_plani.md V1).

A smooth displacement field that vanishes on the boundary is chosen; the body force it needs,
f = -[(lam + mu) grad(div u) + mu lap u] (plane stress in 2-D: lam* = E nu / (1 - nu^2)), is derived by hand
here, and the consistent nodal loads are integrated here with the multilinear hat of each node and 3-point Gauss
per axis. Nothing of the library is used except ``Domain`` (grid and node coordinates) and ``Elasticity`` (the
element matrices, assembly, constraints and solver under test). The finite-element nodal displacements must
converge to the exact field at the rate of the element, O(h^2) on a regular grid.

Measured (18 Nov 2026), max nodal error relative to max |u|:
Q4   h = 1/8, 1/16, 1/32: 7.50e-3, 1.87e-3, 4.66e-4 (rates 2.01, 2.00)
H8   h = 1/4, 1/8, 1/16:  1.01e-1, 2.56e-2, 6.29e-3 (rates 1.99, 2.02)
H8IM h = 1/4, 1/8, 1/16:  1.37e-1, 3.28e-2, 8.19e-3 (rates 2.07, 2.00)
Solving with nu = 0.25 for loads made with nu = 0.3, the error stalls: Q4 1.60e-2 -> 1.68e-2 (rate -0.07), H8 rate
0.38. These tests therefore fail on a wrong element matrix, a wrong material constant, a wrong DOF numbering or a
wrong assembly, which a symmetry or patch test can miss.
"""
from __future__ import annotations

import numpy as np
import pytest

import toporlogy as T

G3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5, 8, 5]) / 9.0)   # Gauss-Legendre on [-1, 1]

def field(dim, L):
    k = [np.pi / l for l in L]
    if dim == 2:
        A = [(1.0, (1, 1)), (0.5, (2, 1))]                 # u_c = a_c * prod sin(m_d k_d x_d)
    else:
        A = [(1.0, (1, 1, 1)), (0.5, (2, 1, 1)), (0.3, (1, 2, 1))]
    return k, A

def u_exact(X, dim, L):
    k, A = field(dim, L)
    return np.stack([a * np.prod([np.sin(m * kk * X[:, i]) for i, (m, kk) in enumerate(zip(mm, k))], axis=0) for a, mm in A], 1)

def body_force(X, dim, L, E, nu):
    """f = -[(lam+mu) grad(div u) + mu lap u]; plane stress in 2-D (lam* = E nu / (1 - nu^2))."""
    k, A = field(dim, L)
    mu = E / (2 * (1 + nu))
    lam = E * nu / (1 - nu ** 2) if dim == 2 else E * nu / ((1 + nu) * (1 - 2 * nu))
    def d(c, i, j=None):                                    # derivative of component c along i (and j)
        a, mm = A[c]
        out = np.full(len(X), a)
        for q in range(dim):
            w = mm[q] * k[q]
            s, co = np.sin(w * X[:, q]), np.cos(w * X[:, q])
            n = (q == i) + (q == j if j is not None else 0)
            out = out * (s if n == 0 else w * co if n == 1 else -w * w * s)
        return out
    f = np.zeros((len(X), dim))
    for c in range(dim):
        graddiv = sum(d(i, c, i) for i in range(dim))
        lap = sum(d(c, i, i) for i in range(dim))
        f[:, c] = -((lam + mu) * graddiv + mu * lap)
    return f

def consistent_loads(d, dim, L, E, nu):
    """sum over cells of int f N_node dV with the multilinear hat of each corner node, 3-point Gauss per axis;
    independent of the library's element routines."""
    h, shape = d.h, d.shape
    g, w = G3
    pts = np.array(np.meshgrid(*[g] * dim, indexing="ij")).reshape(dim, -1).T          # (ng, dim) in [-1,1]
    wts = np.prod(np.array(np.meshgrid(*[w] * dim, indexing="ij")).reshape(dim, -1), 0)
    corners = np.array(np.meshgrid(*[[0, 1]] * dim, indexing="ij")).reshape(dim, -1).T  # (2^dim, dim)
    cells = np.indices(shape).reshape(dim, -1).T                                         # (ne, dim)
    Xg = (cells[:, None, :] + (pts[None] + 1) / 2) * h                                    # (ne, ng, dim)
    fg = body_force(Xg.reshape(-1, dim), dim, L, E, nu).reshape(len(cells), len(pts), dim)
    jac = (h / 2) ** dim * (1.0 if dim == 3 else d.thickness)
    F = d.load_vector()[:, 0].copy()
    node_shape = np.array(shape) + 1
    for cn in corners:
        N = np.prod(np.where(cn[None, :] == 1, (1 + pts) / 2, (1 - pts) / 2), 1)          # (ng,)
        nodes = np.ravel_multi_index((cells + cn).T, node_shape)
        load = np.einsum("g,egc,g->ec", N, fg, wts) * jac
        for c in range(dim):
            np.add.at(F, nodes * dim + c, load[:, c])        # numbering checked in a test below
    return F

def solve(n, dim, element, nu_load=0.3, nu_fe=0.3):
    L = (2.0, 1.0) if dim == 2 else (1.0, 1.0, 1.0)
    h = 1.0 / n
    shape = tuple(int(round(l / h)) for l in L)
    d = T.Domain(shape, h)
    X = d.node_coords()
    bnd = np.flatnonzero(np.any((X < 1e-9) | (np.array(L) - 1e-9 < X), axis=1))
    fixed = d.dofs(bnd)
    F = consistent_loads(d, dim, L, 1.0, nu_load)
    phys = T.Elasticity(d, fixed, nu=nu_fe, element=element)
    U = phys.solve_numpy(np.ones(d.n_active), F[:, None])[:, 0].reshape(-1, dim)
    ue = u_exact(X, dim, L)
    return float(np.max(np.abs(U - ue)) / np.max(np.abs(ue)))



def _rates(errors):
    return [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]


def test_dof_numbering_assumed_by_the_loads():
    """The loads above use DOF = node * dim + component; the library's numbering must be that."""
    for shape in ((3, 2), (2, 3, 2)):
        d = T.Domain(shape)
        nodes = np.arange(d.ndof // d.dim)
        for c in range(d.dim):
            assert np.array_equal(d.dofs(nodes, c), nodes * d.dim + c)


@pytest.mark.parametrize("dim,element,ns,finest_max", [
    (2, "Q4", (8, 16, 32), 1e-3),
    (3, "H8", (4, 8, 16), 1e-2),
    (3, "H8IM", (4, 8, 16), 1.2e-2),
])
def test_manufactured_solution_converges_at_second_order(dim, element, ns, finest_max):
    e = [solve(n, dim, element) for n in ns]
    rates = _rates(e)
    assert all(1.8 <= r <= 2.25 for r in rates), (element, e, rates)
    assert e[-1] < finest_max, (element, e)


@pytest.mark.parametrize("dim,element,ns", [(2, "Q4", (8, 16, 32)), (3, "H8", (4, 8, 16))])
def test_manufactured_solution_detects_a_wrong_material_constant(dim, element, ns):
    """The same check against a solver with nu = 0.25 (loads made with 0.3) must fail: error stalls."""
    good = solve(ns[-1], dim, element)
    bad = [solve(n, dim, element, nu_fe=0.25) for n in ns]
    assert _rates(bad)[-1] < 1.0 and bad[-1] > 5 * good, (element, bad, good)
