"""Non-shrinking, feature-preserving smoothing of a triangle surface.

* :func:`taubin` — λ|μ smoothing (Taubin 1995): a positive Laplacian step λ followed by a negative step μ
  (|μ| > λ) per pass, a low-pass filter with pass-band k_pb = 1/λ + 1/μ that removes the iso-surface facet noise
  without the shrinkage of plain Laplacian smoothing. Umbrella (uniform) weights.
* :func:`windowed_sinc` — VTK ``vtkWindowedSincPolyDataFilter`` (pyvista ``smooth_taubin``), the same low-pass idea
  with a windowed-sinc polynomial (VTK documentation; not the Taubin paper's λ|μ form).
* :func:`relax` — tangential relaxation: each vertex moves to its neighbour average *within its tangent plane*, then
  back onto the exact surface. Improves triangle shape without moving the surface (the tangential smoothing step of
  isotropic remeshing; Botsch et al. 2010, Polygon Mesh Processing, ch. 6 — book not re-read in this session).

Feature preservation (:class:`Features`): vertices that lie on a *functional* exact surface (bore, mating face; see
:mod:`toporlogy.post.primitives`) are projected back onto that surface after every pass — onto two surfaces
alternately where they meet (edge of a bore mouth). Optionally all other vertices are projected onto the implicit
shape φ = 0 (Newton steps on a callable φ), so the surface never drifts from the regularised shape.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from toporlogy.references import cite
from toporlogy.post.mesh import vertex_normals

__all__ = ["Features", "umbrella", "taubin", "windowed_sinc", "relax", "project_implicit"]


def umbrella(F: np.ndarray, n: int) -> sp.csr_matrix:
    """Row-normalised vertex adjacency W (W·V = neighbour average)."""
    F = np.asarray(F, np.int64)
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                                           # an edge counts once
    d = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(d, 1.0)) @ A


@dataclass
class Features:
    """Functional surfaces and the vertices assigned to them.

    ``roles``: functional :class:`~toporlogy.post.primitives.Role` objects. A vertex belongs to role i when
    ``|g_i(v)| ≤ tol`` and its normal is within ``angle_deg`` of the surface normal ∇g_i (g = oriented distance).
    """

    roles: Sequence[Any]
    tol: float
    angle_deg: float = 60.0
    assign: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))

    def update(self, V: np.ndarray, F: np.ndarray) -> np.ndarray:
        n = vertex_normals(V, F)
        c = np.cos(np.radians(self.angle_deg))
        cols = []
        for r in self.roles:
            g = r.oriented(V)
            gr = r.oriented_grad(V)
            cols.append((np.abs(g) <= self.tol) & (np.einsum("ij,ij->i", n, gr) >= c))
        self.assign = np.stack(cols, axis=1) if cols else np.zeros((len(V), 0), bool)
        return self.assign

    @property
    def locked(self) -> np.ndarray:
        return self.assign.any(axis=1) if self.assign.size else np.zeros(len(self.assign), bool)

    def project(self, V: np.ndarray, rounds: int = 6) -> np.ndarray:
        """Assigned vertices onto their surface(s) (alternating projections where several meet)."""
        if not self.assign.size or not self.assign.any():
            return V
        V = V.copy()
        multi = self.assign.sum(axis=1) > 1
        for _ in range(rounds if multi.any() else 1):
            for i, r in enumerate(self.roles):
                m = self.assign[:, i]
                if not m.any():
                    continue
                P = V[m]
                for _k in range(2):
                    P = P - r.oriented(P)[:, None] * r.oriented_grad(P)
                V[m] = P
        return V


def project_implicit(V: np.ndarray, phi: Callable[[np.ndarray], np.ndarray], step: float, iterations: int = 3,
                     max_move: float | None = None) -> np.ndarray:
    """Newton projection of points onto φ = 0 (x ← x − φ∇φ/|∇φ|², ∇φ by central differences of ``step``)."""
    P = np.asarray(V, float).copy()
    start = P.copy()
    E = np.eye(3) * step
    for _ in range(iterations):
        f = phi(P)
        g = np.stack([(phi(P + E[k]) - phi(P - E[k])) / (2 * step) for k in range(3)], axis=1)
        gg = np.maximum((g * g).sum(axis=1), 1e-12)
        P = P - (f / gg)[:, None] * g
    if max_move is not None:
        d = P - start
        n = np.linalg.norm(d, axis=1)
        far = n > max_move
        P[far] = start[far] + d[far] * (max_move / n[far])[:, None]
    return P


def _post(V: np.ndarray, features: Features | None, phi: Callable | None, phi_step: float,
          max_move: float | None) -> np.ndarray:
    if phi is not None:
        free = ~features.locked if features is not None and features.assign.size else np.ones(len(V), bool)
        if free.any():
            V = V.copy()
            V[free] = project_implicit(V[free], phi, phi_step, max_move=max_move)
    if features is not None:
        V = features.project(V)
    return V


@cite("taubin_1995", note="lambda|mu two-step smoothing; lambda 0.5, mu -0.53 (k_pb ~ 0.11) is this module's default")
def taubin(V: np.ndarray, F: np.ndarray, iterations: int = 20, lam: float = 0.5, mu: float = -0.53,
           features: Features | None = None, weights: np.ndarray | None = None,
           phi: Callable[[np.ndarray], np.ndarray] | None = None, phi_step: float = 1e-3,
           max_move: float | None = None) -> np.ndarray:
    """λ|μ smoothing; ``weights`` (n,) in [0, 1] scales the steps per vertex (curvature-adaptive smoothing);
    feature vertices are re-projected after every pass, all others onto ``phi`` if given."""
    W = umbrella(F, len(V))
    P = np.asarray(V, float).copy()
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    for _ in range(iterations):
        for s in (lam, mu):
            P = P + (s * w)[:, None] * (W @ P - P)
        P = _post(P, features, phi, phi_step, max_move) if (features is not None or phi is not None) else P
    return P


def windowed_sinc(V: np.ndarray, F: np.ndarray, iterations: int = 20, pass_band: float = 0.1,
                  features: Features | None = None, phi: Callable | None = None, phi_step: float = 1e-3,
                  max_move: float | None = None) -> np.ndarray:
    """VTK windowed-sinc smoothing; features (and φ) are re-imposed once after the VTK filter (VTK offers no
    per-pass callback)."""
    from toporlogy.post.mesh import to_polydata
    pd = to_polydata(V, F).smooth_taubin(n_iter=iterations, pass_band=pass_band, boundary_smoothing=False,
                                         feature_smoothing=False, non_manifold_smoothing=False,
                                         normalize_coordinates=True)
    P = np.asarray(pd.points, float).copy()
    if features is not None or phi is not None:
        P = _post(P, features, phi, phi_step, max_move)
    return P


@cite("botsch_kobbelt_2010", note="tangential relaxation step of isotropic remeshing")
def relax(V: np.ndarray, F: np.ndarray, iterations: int = 5, alpha: float = 0.5, features: Features | None = None,
          phi: Callable[[np.ndarray], np.ndarray] | None = None, phi_step: float = 1e-3,
          max_move: float | None = None) -> np.ndarray:
    """Tangential relaxation + projection (surface unchanged, triangles more regular)."""
    W = umbrella(F, len(V))
    P = np.asarray(V, float).copy()
    for _ in range(iterations):
        n = vertex_normals(P, F)
        U = W @ P - P
        U -= np.einsum("ij,ij->i", U, n)[:, None] * n
        P = P + alpha * U
        P = _post(P, features, phi, phi_step, max_move)
    return P


def curvature_weights(V: np.ndarray, kappa: Callable[[np.ndarray], np.ndarray], radius: float) -> np.ndarray:
    """Per-vertex smoothing weight 1/(1 + (|κ|·radius)²): full smoothing on flat parts, less on features with a
    curvature radius below ``radius`` (κ = mean curvature of the implicit shape, sampled at the vertices)."""
    k = np.abs(np.asarray(kappa(V), float))
    return 1.0 / (1.0 + (k * radius) ** 2)


__all__ += ["curvature_weights"]
