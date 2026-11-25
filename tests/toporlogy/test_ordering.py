"""Geometric nested dissection for the direct solver (toporlogy.solvers.ordering): the checks pre-registered in
docs/olcumler/cozucu_siralama.md §1 (2026-11-25 19:43 +0300, before the first run).

O1 permutation; O2 separator structure; O3 less fill than MMD on grids not used for the choice; O4 same solution as
the solver's own order; O5 the same with partial volumes, keep/void and an oblique support (ConstrainedElasticity);
O6 same compliance gradient. A failing pre-registered check stays as a strict xfail with the record here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import toporlogy as T
from toporlogy import boundary
from toporlogy.solvers.ordering import nested_dissection_dofs, nested_dissection_nodes


@pytest.mark.parametrize("shape", [(1,), (5,), (2, 3), (13, 7), (1, 9, 4), (9, 5, 6), (25, 13, 13)])
@pytest.mark.parametrize("leaf", [1, 64, 10 ** 6])
def test_o1_every_node_once(shape, leaf):
    p = nested_dissection_nodes(shape, leaf)
    assert np.array_equal(np.sort(p), np.arange(int(np.prod(shape))))


@pytest.mark.parametrize("shape", [(21, 11), (17, 9, 9)])
def test_o2_top_level_separator(shape):
    """Both halves first, the separator plane last; no element touches both halves (nodes of one element differ by at
    most one index per axis), so the factor has no entry between them: checked on the scalar grid Laplacian."""
    p = nested_dissection_nodes(shape)
    coords = np.stack(np.unravel_index(p, shape), axis=1)
    a = int(np.argmax(shape))
    mid = shape[a] // 2
    n1, n2 = int((coords[:, a] < mid).sum()), int((coords[:, a] > mid).sum())
    assert (coords[:n1, a] < mid).all() and (coords[n1:n1 + n2, a] > mid).all() and (coords[n1 + n2:, a] == mid).all()
    lap = None
    for ax, n in enumerate(shape):
        D = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
        I_ = [sp.identity(m) for m in shape]
        I_[ax] = D
        term = I_[0]
        for M in I_[1:]:
            term = sp.kron(term, M)
        lap = term if lap is None else lap + term
    assert lap is not None
    Lp = (lap + 0.01 * sp.identity(lap.shape[0])).tocsr()[p][:, p].tocsc()
    lu = spla.splu(Lp, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    L = lu.L.tocsr()
    assert abs(L[n1:n1 + n2, :n1]).sum() == 0.0


def _cantilever(shape, frac=None, keep=None, void=None):
    d = T.Domain(shape, frac=frac, keep=keep, void=void)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    F = d.load_vector()
    tip = np.flatnonzero(np.isclose(X[:, 0], shape[0]) & np.isin(np.arange(d.nn), d.active_dofs // d.dim))
    F[d.dim * tip + d.dim - 1, 0] = -1.0 / len(tip)
    return d, fixed, F


def _design(d, seed):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    f = ndimage.gaussian_filter(rng.standard_normal(d.shape), 1.5).ravel()[d.active_elements]
    x = 1 / (1 + np.exp(-4 * (f - f.mean()) / f.std()))
    return 1e-6 + x ** 3 * (1 - 1e-6)


def _fill_nd_and_mmd(shape):
    d, fixed, _ = _cantilever(shape)
    m = T.Elasticity(d, fixed, 0.3)
    A = m.matrix(_design(d, 4))
    K = (A + sp.triu(A, k=1).T).tocsc()
    lu = spla.splu(K, permc_spec="MMD_AT_PLUS_A")
    p = nested_dissection_dofs(tuple(s + 1 for s in shape), d.dim, m.free)
    lu2 = spla.splu(K[p][:, p].tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    return (lu.L.nnz + lu.U.nnz) / K.nnz, (lu2.L.nnz + lu2.U.nnz) / K.nnz


def test_o3_less_fill_than_mmd_2d():
    mmd, nd = _fill_nd_and_mmd((400, 120))
    print(f"\n  O3 2-D 400x120: fill MMD {mmd:.2f}, nested dissection {nd:.2f}")
    assert nd < mmd


@pytest.mark.slow
def test_o3_less_fill_than_mmd_3d():
    mmd, nd = _fill_nd_and_mmd((32, 16, 16))
    print(f"\n  O3 3-D 32x16x16: fill MMD {mmd:.2f}, nested dissection {nd:.2f}")
    assert nd < mmd


def _compare(make, E, F):
    a, b = make(), make()
    b.ordering = "solver"
    Ua, Ub = a.solve_numpy(E, F), b.solve_numpy(E, F)
    ca, cb = float(np.sum(F * Ua)), float(np.sum(F * Ub))
    return abs(ca - cb) / abs(cb), np.linalg.norm(Ua - Ub) / np.linalg.norm(Ub), a, b


@pytest.mark.parametrize("shape", [(16, 8, 8), (120, 40)])
def test_o4_same_solution_as_the_solver_order(shape):
    d, fixed, F = _cantilever(shape)
    E = _design(d, 5)
    dc, du, a, b = _compare(lambda: T.Elasticity(d, fixed, 0.3), E, F)
    print(f"\n  O4 {shape}: compliance {dc:.1e}, U {du:.1e}; residuals {a.last_residual:.1e} / {b.last_residual:.1e}")
    assert dc <= 1e-10 and du <= 1e-7


def test_o5_partial_volumes_keep_void_and_oblique_support():
    shape = (8, 5, 5)
    rng = np.random.default_rng(6)
    frac = np.clip(rng.uniform(0.2, 1.6, shape), 0, 1)
    frac[6:, :2, :] = 0.0                                   # a notch: elements outside the part
    keep = np.zeros(shape, bool)
    keep[:, 4, :] = True
    void = np.zeros(shape, bool)
    void[3, 2, 2] = True
    d = T.Domain(shape, frac=frac, keep=keep, void=void)
    d1 = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    bottom = boundary.OnPlane([0, 0, 0], 2)
    side = boundary.Roller(boundary.OnPlane([0, 0, 0], 0), 0)
    load = [boundary.LoadCase("t", [boundary.SurfaceForce(boundary.OnPlane([0, 0, 5], 2), [1.0, 0.5, -1.0])])]
    B = boundary.BoundaryConditions(d, [boundary.Fixed(bottom, [d1, [0, 0, 1]]), side], load)
    assert B.check().ok and len(B.constraints("t").frames) > 0
    E = _design(d, 7)
    dc, du, a, _ = _compare(B.physics, E, B.F())
    print(f"\n  O5: compliance {dc:.1e}, U {du:.1e}")
    assert type(a).__name__ == "ConstrainedElasticity"
    assert dc <= 1e-10 and du <= 1e-7


def test_o6_same_compliance_gradient():
    d, fixed, F = _cantilever((14, 6, 6))
    E0 = torch.tensor(_design(d, 8), dtype=torch.float64)
    grads = []
    for ordering in ("nested_dissection", "solver"):
        m = T.Elasticity(d, fixed, 0.3)
        m.ordering = ordering
        E = E0.clone().requires_grad_(True)
        U = m.solve(E, F)
        c = (torch.as_tensor(F) * U).sum()
        c.backward()
        assert E.grad is not None
        grads.append(E.grad.numpy().copy())
    rel = np.max(np.abs(grads[0] - grads[1])) / np.max(np.abs(grads[1]))
    print(f"\n  O6: gradient {rel:.1e}")
    assert rel <= 1e-9


def test_default_is_nested_dissection_and_bad_values_are_refused():
    d, fixed, _ = _cantilever((4, 2, 2))
    m = T.Elasticity(d, fixed, 0.3)
    assert m.ordering == "nested_dissection"
    with pytest.raises(ValueError, match="ordering"):
        T.Elasticity(d, fixed, 0.3, ordering="metis")


# ------------------------------------------------------- O8, O9 (docs/olcumler/dogrulama_turu.md, pre-registered)
def _buckling_with_preload(shape, ordering):
    from toporlogy.fem.buckling import LinearBuckling
    d = T.Domain(shape, h=1.0)
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 1] < 1e-9))
    top = np.flatnonzero(X[:, 1] > X[:, 1].max() - 1e-9)
    F = np.zeros(d.ndof)
    F[d.dim * top + 1] = -1.0 / len(top)
    F0 = np.zeros(d.ndof)
    F0[d.dim * top] = 0.1 / len(top)
    E = 0.1 + 0.9 * np.random.default_rng(0).random(d.n_active)
    m = T.Elasticity(d, fixed, nu=0.3, ordering=ordering)
    U = m.solve_numpy(E, np.column_stack([F, F0]))
    lb = LinearBuckling(m)
    mu, _ = lb.modes(E, E, U[:, 0], k=6, KG0=lb.geometric_matrix(E, U[:, 1]))
    return mu


@pytest.mark.parametrize("shape", [(40, 12), (8, 4, 4)])
def test_o8_buckling_preload_same_load_factors(shape):
    a, b = _buckling_with_preload(shape, "nested_dissection"), _buckling_with_preload(shape, "solver")
    err = float(np.abs(a - b).max() / np.abs(b).max())
    print(f"\n  O8 {shape}: {err:.2e}")
    assert err <= 1e-9


def _modal(shape, sigma, ordering):
    from toporlogy.fem.modal import Modal
    from toporlogy.solvers.direct import DirectSolver
    d = T.Domain(shape, h=1.0)
    X = d.node_coords()
    mo = Modal(d, d.dofs(np.flatnonzero(X[:, 0] < 1e-9)), nu=0.3, n_modes=6, sigma=sigma, solver=DirectSolver("splu"))
    mo.elastic.ordering = ordering
    E = 0.1 + 0.9 * np.random.default_rng(1).random(d.n_active)
    lam, _ = mo.solve_numpy(E, np.ones(d.n_active), 6)
    return np.asarray(lam), mo.solver


@pytest.mark.parametrize("sigma", [0.0, -1e-3, 1e-4])
@pytest.mark.parametrize("shape", [(40, 12), (8, 4, 4)])
def test_o9_modal_same_eigenvalues(shape, sigma):
    (a, sa), (b, _) = _modal(shape, sigma, "nested_dissection"), _modal(shape, sigma, "solver")
    err = float(np.abs(a - b).max() / np.abs(b).max())
    print(f"\n  O9 {shape} sigma={sigma}: {err:.2e}")
    assert err <= 1e-9
    # σ > 0 (indefinite K − σM) keeps the pivoting order: bit for bit the old path
    assert (sa._order is None) == (sigma > 0)
    if sigma > 0:
        assert np.array_equal(a, b)


def test_o12_gmg_direct_coarse_solve_same_result(monkeypatch):
    """O12 (docs/olcumler/dogrulama_turu.md §5.2): the "direct" coarse solve of the multigrid factorises with the
    nested dissection order; the solution equals the one with the previous (MMD) coarse factorisation to 1e-12."""
    from toporlogy.fem.elastic_gmg import ElasticityGMG
    from toporlogy.solvers.direct import DirectSolver
    d = T.Domain((16, 8, 8))
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] < 1e-9))
    F = np.zeros((d.ndof, 1))
    F[d.dim * np.flatnonzero(X[:, 0] > X[:, 0].max() - 1e-9) + 2, 0] = -1.0
    E = 1e-3 + np.random.default_rng(4).random(d.n_active) ** 3

    def run():
        g = ElasticityGMG(d, fixed, tol=1e-10, direct_below=0, coarse="direct", n_geometric=2)
        return g.solve_numpy(E, F), g
    U_nd, g = run()
    assert g.n_fallbacks == 0 and g.solver.coarse._ds._order is not None
    factor = DirectSolver.factor
    monkeypatch.setattr(DirectSolver, "factor", lambda self, A, order=None: factor(self, A))
    U_old, g2 = run()
    assert g2.solver.coarse._ds._order is None
    rel = float(np.abs(U_nd - U_old).max() / np.abs(U_old).max())
    print(f"\n  O12: {rel:.2e}")
    assert rel <= 1e-12
