"""Mode tracking (docs/olcumler/mod_takibi_onkayit.md, MT1-MT6, pre-registered before the code)."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from toporlogy.core.domain import Domain
from toporlogy.fem.modal import Modal
from toporlogy.optimization.mode_tracking import clusters, subspace_angle, track_modes
from toporlogy.solvers.eigen import mac

RNG = np.random.default_rng(0)


def _spd(n: int) -> np.ndarray:
    A = RNG.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def _m_orthonormal(n: int, k: int, M: np.ndarray) -> np.ndarray:
    V = RNG.normal(size=(n, k))
    L = np.linalg.cholesky(V.T @ M @ V)
    return V @ np.linalg.inv(L).T


def test_mt1_shuffled_and_flipped_modes_are_recovered() -> None:
    M = _spd(20)
    Phi = _m_orthonormal(20, 6, M)
    perm = RNG.permutation(6)
    signs = RNG.choice([-1.0, 1.0], 6)
    new = Phi[:, perm] * signs
    t = track_modes(Phi, new, M)
    assert np.array_equal(perm[t.order], np.arange(6))
    assert np.array_equal(t.sign, signs[t.order].astype(int))
    assert (t.mac >= 1 - 1e-12).all()


def test_mt2_scale_and_sign_invariance() -> None:
    M = _spd(12)
    Phi = _m_orthonormal(12, 4, M)
    Psi = RNG.normal(size=(12, 4))
    assert np.abs(mac(Phi, -3.0 * Psi, M) - mac(Phi, Psi, M)).max() <= 1e-14


def test_mt3_repeated_eigenvalue_is_a_subspace() -> None:
    M = _spd(15)
    Phi = _m_orthonormal(15, 4, M)
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.eye(4)
    R[1:3, 1:3] = [[c, -s], [s, c]]                     # rotate inside the cluster {1, 2}
    new = Phi @ R
    lam = np.array([1.0, 2.0, 2.0, 3.0])
    t = track_modes(Phi, new, M, lam_old=lam)
    assert [1, 2] in t.clusters
    assert t.mac[1:3].min() < 0.99                      # individually ambiguous ...
    assert t.cluster_angles[t.clusters.index([1, 2])] <= 1e-7    # ... as a subspace identical


def test_mt4_square_cantilever_modes_are_tracked() -> None:
    d = Domain((10, 2, 2))
    X = d.node_coords()
    fixed = d.dofs(np.flatnonzero(X[:, 0] == 0))
    m = Modal(d, fixed, nu=0.3)
    E = np.ones(d.n_active)
    lam0, P0 = m.solve_numpy(E, np.ones(d.n_active), 6)
    E2 = E * (1 + 0.02 * np.random.default_rng(1).random(d.n_active))
    lam1, P1 = m.solve_numpy(E2, np.ones(d.n_active), 6)
    M = m.mass_matrix(np.ones(d.n_active))
    t = track_modes(P0, P1, M, lam_old=lam0, cluster_tol=1e-6)
    print(f"\n  MT4 eigenvalues {lam0}, clusters {t.clusters}, angles {t.cluster_angles}")
    assert [0, 1] in t.clusters
    assert max(t.cluster_angles) <= 0.05


def test_mt5_unrelated_modes_have_small_mac() -> None:
    M = _spd(200)
    a, b = _m_orthonormal(200, 5, M), _m_orthonormal(200, 5, M)
    assert mac(a, b, M).max() < 0.5


def test_mt6_refusals() -> None:
    with pytest.raises(ValueError, match="same n"):
        mac(np.ones((5, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="positive definite"):
        mac(np.ones((3, 1)), np.ones((3, 1)), sp.csr_matrix(-np.eye(3)))
    assert clusters(np.array([1.0, 1.0 + 1e-9, 2.0])) == [[0, 1], [2]]
    assert subspace_angle(np.eye(3)[:, :1], np.eye(3)[:, :1]) == 0.0
