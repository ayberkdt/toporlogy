"""Scalar conduction on the grid (toporlogy.fem.heat): element, exact 1-D solutions, patch test, gradients.

Pre-registered thresholds: exact solutions and patch test to 1e-10 relative; gradient checks ≤ 1e-6 relative
(:func:`toporlogy.checks.gradcheck`, float64).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from toporlogy import Domain, gradcheck
from toporlogy.fem.heat import (ConductivityInterp, HeatConduction, assemble_conduction, conduction_matrix,
                                   element_nodes)

TOL = 1e-6


def test_element_matrices():
    K2 = conduction_matrix(2)
    ref = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6.0
    assert np.abs(K2 - ref).max() < 1e-14
    for dim in (2, 3):
        K = conduction_matrix(dim)
        ev = np.linalg.eigvalsh(K)
        assert np.abs(K - K.T).max() < 1e-15
        assert np.abs(K.sum(axis=1)).max() < 1e-14                  # constants are in the null space
        assert abs(ev[0]) < 1e-13 and ev[1] > 1e-3                  # exactly one zero eigenvalue
    # 3-D: diagonal of the unit-cube trilinear element is 1/3 (closed form 4/12)
    assert np.allclose(np.diag(conduction_matrix(3)), 1.0 / 3.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_bar_series_resistance_exact(dim):
    """Bar of n elements, T = 0 at x = 0, heat q into the far end: T_end = q Σ h / (k_e A) exactly."""
    n, h, t = 7, 0.4, 0.3
    shape = (n, 1) if dim == 2 else (n, 1, 1)
    d = Domain(shape, h, thickness=t)
    rng = np.random.default_rng(3)
    k = 0.5 + rng.random(n)
    X = d.node_coords()
    fixed = np.flatnonzero(X[:, 0] < 1e-12)
    end = np.flatnonzero(X[:, 0] > n * h - 1e-12)
    q = 2.5
    Q = np.zeros(d.nn)
    Q[end] = q / len(end)
    model = HeatConduction(d, fixed)
    T = model.solve_numpy(k, Q)[:, 0]
    A = h * t if dim == 2 else h * h
    exact = q * np.sum(h / (k * A))
    assert np.abs(T[end] - exact).max() <= 1e-10 * exact
    # interior nodes: linear in each element (series)
    xs = np.cumsum(np.concatenate([[0.0], q * h / (k * A)]))
    ix = np.rint(X[:, 0] / h).astype(int)
    assert np.abs(T - xs[ix]).max() <= 1e-10 * exact


def test_patch_linear_field():
    """A linear temperature field is in equilibrium at every interior node of an irregular 3-D part."""
    shape = (6, 5, 4)
    mask = np.ones(shape, bool)
    mask[4:, 3:, :] = False
    mask[1:3, 1:3, 1:3] = False
    d = Domain(shape, 0.7, frac=mask.astype(float))
    K = assemble_conduction(d, np.full(d.n_active, 1.3))
    X = d.node_coords()
    T = 0.4 + X @ np.array([1.0, -2.0, 0.5])
    r = K @ T
    # interior node: all 8 surrounding box elements active
    act = np.pad(d.active, 1, constant_values=False)
    ns = d.node_shape
    idx = np.indices(ns).reshape(3, -1).T
    interior = np.ones(len(idx), bool)
    for c in np.indices((2, 2, 2)).reshape(3, -1).T:
        e = idx + c                                    # element (i - 1 + c) in padded coordinates
        interior &= act[e[:, 0], e[:, 1], e[:, 2]]
    assert interior.sum() > 5
    assert np.abs(r[interior]).max() <= 1e-10 * np.abs(K).max() * np.abs(T).max()


def test_element_nodes_match_domain_corners():
    d = Domain((3, 2, 2), 1.0)
    en = element_nodes(d)
    ijk = d.element_ijk()
    for c_i, c in enumerate(d.corners):
        assert np.array_equal(en[:, c_i], d.node_number(ijk + c))


def test_singular_parts_rejected():
    mask = np.zeros((6, 3), bool)
    mask[:2] = True
    mask[4:] = True                                    # two separate parts
    d = Domain((6, 3), 1.0, frac=mask.astype(float))
    with pytest.raises(ValueError, match="no fixed node"):
        HeatConduction(d, [0])
    X = d.node_coords()
    HeatConduction(d, np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > 6 - 1e-9)))     # both parts sunk: fine


def _model_2d():
    shape = (8, 5)
    mask = np.ones(shape, bool)
    mask[5:, 3:] = False
    d = Domain(shape, 1.0, frac=np.where(mask, 0.8, 0.0), thickness=0.5)
    X = d.node_coords()
    m = HeatConduction(d, d.nodes_where(lambda X: X[:, 0] < 1e-9))
    rng = np.random.default_rng(5)
    Q = np.zeros((d.nn, 2))
    hot = d.nodes_where(lambda X: X[:, 0] > 4.5)
    Q[hot, 0] = 1.0 / len(hot)
    Q[:, 1] = rng.random(d.nn) * (np.isin(np.arange(d.nn), np.unique(element_nodes(d, d.active_elements))))
    del X
    return d, m, Q


def test_gradient_self_adjoint_and_general():
    d, m, Q = _model_2d()
    rng = np.random.default_rng(11)
    w = torch.as_tensor(rng.standard_normal(d.nn))
    interp = ConductivityInterp(d, penal=3.0, kmin_ratio=1e-3)
    act = torch.as_tensor(d.active_elements)
    Qt = torch.as_tensor(Q)

    def resistance(x):                                   # self-adjoint per case
        T = m.solve(interp(x).reshape(-1)[act], Q)
        return (Qt * T).sum()

    def general(x):                                      # needs an adjoint solve
        T = m.solve(interp(x).reshape(-1)[act], Q)
        return (w * T[:, 0] ** 2).sum() + (w * T[:, 1]).sum()

    x0 = 0.2 + 0.6 * rng.random(d.shape)
    for fn, name in ((resistance, "resistance"), (general, "general")):
        r = gradcheck(fn, x0, n_dirs=5)
        print(f"\n  heat gradcheck {name}: {r['max_rel_error']:.2e}")
        assert r["max_rel_error"] <= TOL


def test_factorisation_reuse_counts():
    d, m, Q = _model_2d()
    k = torch.full((d.n_active,), 0.7, dtype=torch.float64, requires_grad=True)
    f0 = m.solver.n_factor
    T = m.solve(k, Q)
    (torch.as_tensor(Q) * T).sum().backward()
    assert m.solver.n_factor - f0 == 1 and m.n_adjoint_solves == 0      # self-adjoint: no back-substitution
    k2 = k.detach().clone().requires_grad_(True)
    T = m.solve(k2, Q)
    (T[:, 0] ** 2).sum().backward()
    assert m.solver.n_factor - f0 == 2 and m.n_adjoint_solves == 1      # one adjoint solve, no refactorisation
