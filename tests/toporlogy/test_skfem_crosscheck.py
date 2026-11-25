"""Cross-check against another team's code: scikit-fem (Gustafsson & McBain), docs/olcumler/dogrulama_turu.md §8,
X1-X4, pre-registered 2026-11-25 23:50 +0300 before the first run.

The same grid is built with scikit-fem's own mesh, elements, forms and quadrature; per-element moduli are a P0
field matched by element centres; nodes and DOFs are matched by coordinates. Skipped (NOT VERIFIED) without skfem.
"""
from __future__ import annotations

import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
from skfem import (Basis, BilinearForm, ElementHex0, ElementHex1, ElementQuad0, ElementQuad1,  # noqa: E402
                   ElementVector, MeshHex, MeshQuad, asm, condense, solve)
from skfem.helpers import ddot, dot, grad, sym_grad, trace  # noqa: E402

from toporlogy.core.domain import Domain  # noqa: E402
from toporlogy.fem.elastic import Elasticity  # noqa: E402
from toporlogy.fem.heat import assemble_conduction  # noqa: E402
from toporlogy.fem.modal import Modal  # noqa: E402

NU = 0.3
TOL = 1e-12            # pre-registered X1-X3
TOL_U = 1e-10          # pre-registered X4


@BilinearForm
def _elasticity(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return w["E"] * w["t"] * (2.0 * w["mu"] * ddot(eu, ev) + w["lam"] * trace(eu) * trace(ev))


@BilinearForm
def _mass(u, v, w):
    return w["rho"] * w["t"] * dot(u, v)


@BilinearForm
def _laplace(u, v, w):
    return w["k"] * w["t"] * dot(grad(u), grad(v))


def _setup(dim):
    d = Domain((5, 3), h=0.7, thickness=0.6) if dim == 2 else Domain((4, 3, 2), h=0.7)
    axes = [np.linspace(0.0, n * d.h, n + 1) for n in d.shape]
    mesh = MeshQuad.init_tensor(*axes) if dim == 2 else MeshHex.init_tensor(*axes)
    e1, e0 = (ElementQuad1(), ElementQuad0()) if dim == 2 else (ElementHex1(), ElementHex0())
    # library node of every skfem vertex, library element of every skfem element (by coordinates)
    node = np.ravel_multi_index(tuple(np.rint(mesh.p / d.h).astype(int)), np.array(d.shape) + 1)
    cent = mesh.p[:, mesh.t].mean(axis=1)
    elem = np.ravel_multi_index(tuple(np.floor(cent / d.h).astype(int)), d.shape)
    X = d.node_coords()
    assert np.allclose(X[node], mesh.p.T)
    assert np.array_equal(np.sort(elem), np.arange(d.n_active))
    return d, mesh, e1, e0, node, elem


def _p0(mesh, e0, values_by_library_element, elem):
    b0 = Basis(mesh, e0, intorder=2)
    return b0.interpolate(np.asarray(values_by_library_element)[elem])


def _vector_perm(basis, node, dim):
    """skfem DOF -> library DOF (node · dim + component)."""
    perm = np.empty(basis.N, np.int64)
    for c in range(dim):
        perm[basis.nodal_dofs[c]] = node * dim + c
    return perm


def _lame(dim):
    mu = 1.0 / (2 * (1 + NU))
    lam = NU / (1 - NU ** 2) if dim == 2 else NU / ((1 + NU) * (1 - 2 * NU))   # plane stress: lam* = E nu/(1-nu^2)
    return lam, mu


def _to_library(A, perm, n):
    A = A.tocoo()
    out = np.zeros((n, n))
    np.add.at(out, (perm[A.row], perm[A.col]), A.data)
    return out


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize("dim", [2, 3])
def test_x1_stiffness(dim):
    d, mesh, e1, e0, node, elem = _setup(dim)
    E = 0.2 + 0.8 * np.random.default_rng(0).random(d.n_active)
    b = Basis(mesh, ElementVector(e1), intorder=2)
    lam, mu = _lame(dim)
    t = d.thickness if dim == 2 else 1.0
    Ks = _to_library(asm(_elasticity, b, E=_p0(mesh, e0, E, elem), lam=lam, mu=mu, t=t),
                     _vector_perm(b, node, dim), d.ndof)
    A = Elasticity(d, np.array([], np.int64), nu=NU).matrix(E)
    Kl = (A + A.T).toarray() - np.diag(A.diagonal())
    print(f"\n  X1 {dim}-D: {_rel(Kl, Ks):.2e}")
    assert _rel(Kl, Ks) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_x2_mass(dim):
    d, mesh, e1, e0, node, elem = _setup(dim)
    rho = 0.1 + 0.9 * np.random.default_rng(1).random(d.n_active)
    b = Basis(mesh, ElementVector(e1), intorder=2)
    t = d.thickness if dim == 2 else 1.0
    Ms = _to_library(asm(_mass, b, rho=_p0(mesh, e0, rho, elem), t=t), _vector_perm(b, node, dim), d.ndof)
    Ml = Modal(d, np.array([], np.int64), nu=NU).mass_matrix(rho).toarray()
    print(f"\n  X2 {dim}-D: {_rel(Ml, Ms):.2e}")
    assert _rel(Ml, Ms) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_x3_conduction(dim):
    d, mesh, e1, e0, node, elem = _setup(dim)
    k = 0.1 + 0.9 * np.random.default_rng(2).random(d.n_active)
    b = Basis(mesh, e1, intorder=2)
    t = d.thickness if dim == 2 else 1.0
    perm = np.empty(b.N, np.int64)
    perm[b.nodal_dofs[0]] = node
    Cs = _to_library(asm(_laplace, b, k=_p0(mesh, e0, k, elem), t=t), perm, d.nn)
    Cl = assemble_conduction(d, k).toarray()
    print(f"\n  X3 {dim}-D: {_rel(Cl, Cs):.2e}")
    assert _rel(Cl, Cs) <= TOL


@pytest.mark.parametrize("dim", [2, 3])
def test_x4_cantilever_solution(dim):
    d, mesh, e1, e0, node, elem = _setup(dim)
    E = 0.2 + 0.8 * np.random.default_rng(3).random(d.n_active)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    tip = np.flatnonzero(X[:, 0] > X[:, 0].max() - 1e-9)
    F = np.zeros(d.ndof)
    F[dim * tip + 1] = -1.0 / len(tip)
    Ul = Elasticity(d, fixed, nu=NU).solve_numpy(E, F[:, None])[:, 0]
    b = Basis(mesh, ElementVector(e1), intorder=2)
    lam, mu = _lame(dim)
    t = d.thickness if dim == 2 else 1.0
    K = asm(_elasticity, b, E=_p0(mesh, e0, E, elem), lam=lam, mu=mu, t=t)
    perm = _vector_perm(b, node, dim)                         # skfem dof s <-> library dof perm[s]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    Us = solve(*condense(K, F[perm], D=inv[fixed]))
    rel = float(np.abs(Ul[perm] - Us).max() / np.abs(Ul).max())
    print(f"\n  X4 {dim}-D: {rel:.2e}")
    assert rel <= TOL_U
