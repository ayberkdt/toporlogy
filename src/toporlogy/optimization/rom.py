"""POD-Galerkin reduced-order model of a parametric linear elastic problem, with an a posteriori error bound.

The reduced solution is an **approximation**, never the finite element solution: every result carries
``approximate = True``, an error bound and an extrapolation flag.

Problem: K(θ) u = f with an affine, positive decomposition K(θ) = Σ_q θ_q K_q, θ_q > 0, every K_q symmetric
positive semidefinite (e.g. one modulus per region of a fixed mesh; :meth:`AffineSystem.from_elasticity`). A
geometry, support or topology change is not of this form.

* Basis: left singular vectors of the training snapshots (Euclidean inner product);
  Σ_s ‖s − V Vᵀ s‖² = Σ_{i>r} σ_i² over the snapshots (the training projection error, not the error elsewhere).
* Online: Galerkin, (Σ θ_q Vᵀ K_q V) a = Vᵀ f (r × r).
* Error bound in the energy norm e = ‖u − V a‖_K: with the residual ρ = f − K(θ) V a, a reference θ̄ and
  α = min_q θ_q / θ̄_q (so K(θ) ≥ α K(θ̄)),  e ≤ Δ = √(ρᵀ K(θ̄)⁻¹ ρ / α). Rigorous up to rounding; it overestimates
  by at most √(max_q(θ_q/θ̄_q) / α). Evaluating it costs a full-order residual and a back substitution with the
  factorised K(θ̄): cheaper than a new factorisation, not independent of the mesh size.
* Compliance fᵀ u for the same load: fᵀ V a ≤ fᵀ u and fᵀ u − fᵀ V a = e².
* Relative bound: e / ‖u‖_K ≤ Δ / ‖V a‖_K (Galerkin orthogonality).

Not covered: nonlinear history, hyper-reduction, parameter-dependent loads or mesh.

Verification: ``tests/toporlogy/test_rom.py`` (records R1-R7 of ``docs/olcumler/laminat_uq_onkayit.md``).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = ["AffineSystem", "pod_basis", "projection_error", "PODModel", "ReducedSolution"]


def _positive_theta(theta: Any, q: int) -> np.ndarray:
    t = np.asarray(theta, float).reshape(-1)
    if t.size != q:
        raise ValueError(f"{t.size} parameters for {q} affine terms")
    if not (np.isfinite(t).all() and (t > 0).all()):
        raise ValueError("every parameter must be positive and finite (K must stay positive definite)")
    return t


@dataclass(frozen=True)
class AffineSystem:
    """K(θ) = Σ_q θ_q ``matrices[q]`` (symmetric, full storage) and load ``load`` (n,)."""

    matrices: tuple[sp.csr_matrix, ...]
    load: np.ndarray

    def __post_init__(self) -> None:
        if not self.matrices:
            raise ValueError("at least one affine term")
        n = self.matrices[0].shape[0]
        if any(m.shape != (n, n) for m in self.matrices):
            raise ValueError("the affine terms must have the same square shape")
        f = np.asarray(self.load, float).reshape(-1)
        if f.size != n:
            raise ValueError(f"the load has {f.size} entries, the matrices {n} rows")
        object.__setattr__(self, "load", f)

    @classmethod
    def from_elasticity(cls, physics: Any, regions: Any, load: Any) -> AffineSystem:
        """One affine term per region of an :class:`toporlogy.fem.Elasticity`: ``regions`` (n_active,) integer
        labels 0 … q−1 in the order of ``physics.elements``, θ_q the modulus of region q; ``load`` (ndof,)."""
        lab = np.asarray(regions).reshape(-1)
        if lab.size != physics.n_elements or not np.issubdtype(lab.dtype, np.integer) or lab.min() < 0:
            raise ValueError("regions: one non-negative integer label per active element")
        q = int(lab.max()) + 1
        if len(np.unique(lab)) != q:
            raise ValueError("region labels must be 0 … q−1 without gaps")
        physics.check_kinematics()
        mats = []
        for k in range(q):
            A = physics.matrix((lab == k).astype(float))
            mats.append((A + A.T - sp.diags(A.diagonal())).tocsr())
        F = np.asarray(load, float).reshape(-1)
        return cls(tuple(mats), F[physics.free])

    @property
    def n(self) -> int:
        return self.load.size

    @property
    def q(self) -> int:
        return len(self.matrices)

    def stiffness(self, theta: Any) -> sp.csc_matrix:
        t = _positive_theta(theta, self.q)
        K = self.matrices[0] * t[0]
        for tq, Kq in zip(t[1:], self.matrices[1:], strict=True):
            K = K + Kq * tq
        return K.tocsc()

    def solve(self, theta: Any, load: Any = None) -> np.ndarray:
        """Full-order solution (sparse direct)."""
        f = self.load if load is None else np.asarray(load, float).reshape(self.n)
        return np.asarray(splu(self.stiffness(theta)).solve(f))

    def snapshots(self, thetas: Any, load: Any = None) -> np.ndarray:
        """(n, m) full-order solutions for parameter rows ``thetas`` (m, q)."""
        return np.column_stack([self.solve(t, load) for t in np.atleast_2d(np.asarray(thetas, float))])


def pod_basis(snapshots: Any, rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(V (n, r) orthonormal, all singular values) of the snapshot matrix (n, m). ``rank`` None: the numerical rank.
    A rank above the numerical rank is refused (the extra vectors would be rounding noise)."""
    S = np.asarray(snapshots, float)
    if S.ndim != 2 or not np.isfinite(S).all():
        raise ValueError("snapshots must be a finite (n, m) array")
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    numerical = int((sv > max(S.shape) * np.finfo(float).eps * sv[0]).sum()) if sv.size and sv[0] > 0 else 0
    r = numerical if rank is None else int(rank)
    if not 1 <= r <= numerical:
        raise ValueError(f"rank must be between 1 and the numerical rank {numerical} of the snapshots, got {r}")
    return U[:, :r], sv


def projection_error(snapshots: Any, V: Any) -> float:
    """Σ over snapshots of ‖s − V Vᵀ s‖²."""
    S, Vm = np.asarray(snapshots, float), np.asarray(V, float)
    return float(((S - Vm @ (Vm.T @ S)) ** 2).sum())


@dataclass(frozen=True)
class ReducedSolution:
    """An approximation: ``u`` = V a, ``error_bound`` Δ ≥ ‖u_h − u‖_K, ``relative_error_bound`` Δ / ‖u‖_K,
    ``compliance`` fᵀ u (a lower bound of the full-order one), ``extrapolation`` True outside the training box."""

    u: np.ndarray
    coefficients: np.ndarray
    error_bound: float
    relative_error_bound: float
    compliance: float
    extrapolation: bool
    approximate: bool = True


class PODModel:
    """POD-Galerkin model of an :class:`AffineSystem` trained on parameter rows ``train`` (m, q).

    ``rank``: basis size (None: numerical rank of the snapshots). ``reference``: θ̄ of the error bound (default the
    geometric mean of the training box)."""

    def __init__(self, system: AffineSystem, train: Any, rank: int | None = None,
                 reference: Sequence[float] | None = None) -> None:
        T = np.atleast_2d(np.asarray(train, float))
        for t in T:
            _positive_theta(t, system.q)
        self.system = system
        self.train_low, self.train_high = T.min(axis=0), T.max(axis=0)
        self.snapshot_matrix = system.snapshots(T)
        self.V, self.singular_values = pod_basis(self.snapshot_matrix, rank)
        self.reference = np.sqrt(self.train_low * self.train_high) if reference is None \
            else _positive_theta(reference, system.q)
        self._ref_factor = splu(system.stiffness(self.reference))
        self._Kr = [self.V.T @ (K @ self.V) for K in system.matrices]

    @property
    def rank(self) -> int:
        return self.V.shape[1]

    def reduced_coefficients(self, theta: Any, load: Any = None) -> np.ndarray:
        """a of the Galerkin system (the online r × r solve only)."""
        t = _positive_theta(theta, self.system.q)
        f = self.system.load if load is None else np.asarray(load, float).reshape(self.system.n)
        Kr = sum(tq * K for tq, K in zip(t, self._Kr, strict=True))
        return np.linalg.solve(Kr, self.V.T @ f)

    def solve(self, theta: Any, load: Any = None) -> ReducedSolution:
        """Reduced solution with its error bound (module notes). ``load`` (n,) replaces the training load; the basis
        was not built for it, so look at the bound."""
        t = _positive_theta(theta, self.system.q)
        f = self.system.load if load is None else np.asarray(load, float).reshape(self.system.n)
        a = self.reduced_coefficients(t, f)
        u = self.V @ a
        Ku = sum(tq * (K @ u) for tq, K in zip(t, self.system.matrices, strict=True))
        res = f - Ku
        alpha = float(np.min(t / self.reference))
        delta = math.sqrt(max(float(res @ self._ref_factor.solve(res)), 0.0) / alpha)
        energy = float(u @ Ku)
        rel = delta / math.sqrt(energy) if energy > 0 else math.inf
        outside = bool(np.any(t < self.train_low * (1 - 1e-12)) or np.any(t > self.train_high * (1 + 1e-12)))
        return ReducedSolution(u, a, delta, rel, float(f @ u), outside)
