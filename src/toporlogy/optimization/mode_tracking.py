"""Mode tracking between two eigen-solutions: M-weighted MAC, assignment, sign alignment and repeated eigenvalues
as subspaces (05 §3).

``MAC_ij = (φ_iᵀ M ψ_j)² / ((φ_iᵀ M φ_i)(ψ_jᵀ M ψ_j))`` is invariant to the scale and sign of either vector. The
pairing maximises the total MAC (Hungarian method). Eigenvalues within ``cluster_tol`` (relative) form a cluster:
inside a cluster the individual vectors are arbitrary (any rotation of a basis is a basis), so a cluster is compared
as a subspace by the largest principal angle between M-orthonormal bases. Pre-registered checks:
``docs/olcumler/mod_takibi_onkayit.md`` (MT1-MT6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

__all__ = ["mac", "Tracking", "track_modes", "clusters", "subspace_angle"]


def _M(M: Any, n: int) -> Any:
    if M is None:
        return sp.identity(n, format="csr")
    if M.shape != (n, n):
        raise ValueError(f"M must be ({n}, {n}), got {M.shape}")
    return M


def _check_pd(M: Any, Phi: np.ndarray) -> None:
    d = np.einsum("ik,ik->k", Phi, np.asarray(M @ Phi))
    if not (d > 0).all():
        raise ValueError("M is not positive definite on the given modes (phi' M phi <= 0)")


def mac(Phi: np.ndarray, Psi: np.ndarray, M: Any = None) -> np.ndarray:
    """(k1, k2) M-weighted modal assurance criterion of the columns of ``Phi`` and ``Psi``."""
    Phi, Psi = np.asarray(Phi, float), np.asarray(Psi, float)
    if Phi.ndim != 2 or Psi.ndim != 2 or Phi.shape[0] != Psi.shape[0]:
        raise ValueError(f"Phi and Psi must be (n, k) with the same n, got {Phi.shape} and {Psi.shape}")
    M = _M(M, Phi.shape[0])
    _check_pd(M, Phi)
    _check_pd(M, Psi)
    MPsi = np.asarray(M @ Psi)
    cross = Phi.T @ MPsi
    a = np.einsum("ik,ik->k", Phi, np.asarray(M @ Phi))
    b = np.einsum("ik,ik->k", Psi, MPsi)
    return np.asarray(cross ** 2 / np.outer(a, b))


def clusters(lam: np.ndarray, cluster_tol: float = 1e-6) -> list[list[int]]:
    """Every mode in a group: the clusters of :func:`toporlogy.fem.modal.find_clusters` (sorted eigenvalues within
    ``cluster_tol`` relative of their neighbour) and each remaining mode on its own, in index order."""
    from toporlogy.fem.modal import find_clusters
    lam = np.asarray(lam, float)
    groups = {int(c[0]): [int(i) for i in c] for c in find_clusters(lam, cluster_tol)}
    out: list[list[int]] = []
    i = 0
    while i < len(lam):
        g = groups.get(i, [i])
        out.append(g)
        i = g[-1] + 1
    return out


def _orthonormal(V: np.ndarray, M: Any) -> np.ndarray:
    G = V.T @ np.asarray(M @ V)
    L = np.linalg.cholesky(0.5 * (G + G.T))
    return np.asarray(sla.solve_triangular(L, V.T, lower=True).T)


def subspace_angle(A: np.ndarray, B: np.ndarray, M: Any = None) -> float:
    """Largest principal angle (rad) between span(A) and span(B) in the M inner product."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    M = _M(M, A.shape[0])
    Qa, Qb = _orthonormal(A, M), _orthonormal(B, M)
    s = np.linalg.svd(Qa.T @ np.asarray(M @ Qb), compute_uv=False)
    if A.shape[1] != B.shape[1]:
        return float(np.pi / 2)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


@dataclass
class Tracking:
    """``order[j]`` = column of the new modes that continues old mode j; ``sign[j]`` = ±1 to align it;
    ``mac`` = MAC of each pair; ``clusters`` of the old eigenvalues and the subspace angle of each."""

    order: np.ndarray
    sign: np.ndarray
    mac: np.ndarray
    clusters: list[list[int]]
    cluster_angles: list[float]


def track_modes(Phi_old: np.ndarray, Phi_new: np.ndarray, M: Any = None, lam_old: np.ndarray | None = None,
                cluster_tol: float = 1e-6) -> Tracking:
    """Pair the new modes with the old ones by maximal total MAC; clusters (from ``lam_old``) are compared as
    subspaces against the new columns they were paired with."""
    A = mac(Phi_old, Phi_new, M)
    rows, cols = linear_sum_assignment(-A)
    order = np.empty(A.shape[0], np.int64)
    order[rows] = cols
    Mm = _M(M, Phi_old.shape[0])
    dots = np.einsum("ik,ik->k", Phi_old, np.asarray(Mm @ Phi_new[:, order]))
    sign = np.where(dots < 0, -1, 1)
    cl = clusters(lam_old, cluster_tol) if lam_old is not None else [[j] for j in range(A.shape[0])]
    angles = [subspace_angle(Phi_old[:, c], Phi_new[:, order[c]], Mm) for c in cl]
    return Tracking(order, sign, A[np.arange(A.shape[0]), order], cl, angles)
