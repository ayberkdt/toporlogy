"""Cell quality (scale- and rotation-invariant) and a Jacobian positivity certificate.

Quality is *sampled*: at the reference nodes and a quadrature rule of degree 2k (k = element degree). With
``J = ∂x/∂ξ`` and ``A`` the Jacobian of the map from the reference cell to the ideal regular cell (identity for Q4, Q9,
H8; equilateral triangle / regular tetrahedron of unit edges for T3, T6 / T4, T10), ``S = J A⁻¹`` and d the dimension:

* mean ratio ``q = d · sgn(det S) |det S|^{2/d} / ‖S‖_F²``: 1 for the ideal cell, → 0 for a sliver, negative if the
  cell is inverted at that sample;
* scaled Jacobian ``SJ = det S / Π_j ‖S e_j‖`` (columns): 1 for the ideal cell; it does NOT see a sliver tetrahedron
  whose columns stay orthogonal (use ``q`` or ``κ``);
* condition ``κ = σmax(S) / σmin(S)`` (inf if singular);
* ``edge_ratio`` longest / shortest straight vertex-to-vertex edge;
* ``detj_min_rel`` sampled min of detJ / s_e (s_e = (longest edge)^d, as in :mod:`~toporlogy.fem.mesh.kernels`).

q, SJ and κ are invariant under ``x → c R x + t``. No universal acceptance threshold exists: :func:`flag_cells` takes
the threshold from the caller.

The *certificate* is separate. detJ is a polynomial of degree m = d(k − 1) in the barycentric coordinates
(simplices) or n per variable (Q4: 1, Q9: 3, H8: 2); its Bernstein coefficients follow exactly from its values on the
degree-m lattice and their minimum is a lower bound of detJ over the whole cell. ``certified`` means that bound is
positive (relative to s_e, above ``DETJ_TOL``): then detJ > 0 everywhere in the cell, not just at the samples. A cell
whose samples are positive but whose bound is not is *uncertain* (an inversion between sample points is possible).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, factorial

import numpy as np

from toporlogy.fem.mesh.kernels import _CELL_EDGES, DETJ_TOL, element_scale
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.quadrature import rule
from toporlogy.fem.mesh.reference import ReferenceElement, get

__all__ = ["CellQuality", "cell_quality", "quality_from_nodes", "sampled_detj", "flag_cells", "ideal_jacobian"]

_S3 = np.sqrt(3.0)
_IDEAL = {
    "quad": np.eye(2), "hex": np.eye(3),
    "tri": np.array([[1.0, 0.5], [0.0, _S3 / 2]]),
    "tet": np.array([[1.0, 0.5, 0.5], [0.0, _S3 / 2, _S3 / 6], [0.0, 0.0, np.sqrt(2.0 / 3.0)]]),
}


def ideal_jacobian(ref: ReferenceElement) -> np.ndarray:
    """A: Jacobian of the map from the reference cell to the ideal (regular, unit-edge for simplices) cell."""
    if ref.cell not in _IDEAL:
        raise ValueError(f"no shape quality for {ref.name} (a {ref.cell}); needs a 2-D or 3-D cell")
    return _IDEAL[ref.cell]


def _samples(ref: ReferenceElement) -> np.ndarray:
    return np.vstack([ref.nodes, rule(ref.cell, 2 * ref.degree).points])


def _jacobians(ref: ReferenceElement, X: np.ndarray, xi: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 3 or X.shape[1:] != (ref.n_nodes, ref.dim):
        raise ValueError(f"{ref.name}: node coordinates must be (ne, {ref.n_nodes}, {ref.dim}), got {X.shape}")
    return np.einsum("eni,qnj->eqij", X, ref.grad(xi))


def sampled_detj(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(detJ (ne, nq), s_e (ne,)) at the reference nodes and the degree-2k rule; signed, nothing is raised."""
    ref = mesh.ref
    X = mesh.cell_points()
    return np.linalg.det(_jacobians(ref, X, _samples(ref))), element_scale(ref, X)


@dataclass(frozen=True)
class CellQuality:
    """Per-cell quality (arrays of length n_cells); see the module docstring for the definitions."""

    element: str
    mean_ratio_min: np.ndarray
    scaled_jacobian_min: np.ndarray
    condition_max: np.ndarray
    edge_ratio: np.ndarray
    detj_min_rel: np.ndarray
    bound_rel: np.ndarray          # min Bernstein coefficient of detJ / s_e (lower bound over the whole cell)
    certified: np.ndarray          # bound_rel > DETJ_TOL

    @property
    def inverted(self) -> np.ndarray:
        """Cell ids with a negative sampled detJ (beyond ``DETJ_TOL``)."""
        return np.flatnonzero(self.detj_min_rel < -DETJ_TOL)

    @property
    def uncertain(self) -> np.ndarray:
        """Cell ids whose samples are positive but whose positivity is not certified."""
        return np.flatnonzero((self.detj_min_rel >= -DETJ_TOL) & ~self.certified)

    def histogram(self, field: str = "mean_ratio_min", bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """(counts, edges) of a per-cell field; the worst id is ``worst(field)``."""
        return np.histogram(getattr(self, field), bins=bins)

    def worst(self, field: str = "mean_ratio_min") -> int:
        """Id of the worst cell (smallest value; for ``condition_max`` / ``edge_ratio`` the largest)."""
        v = getattr(self, field)
        return int(np.argmax(v) if field in ("condition_max", "edge_ratio") else np.argmin(v))


def _edge_ratio(ref: ReferenceElement, X: np.ndarray) -> np.ndarray:
    e = np.array(_CELL_EDGES[ref.cell])
    L = np.linalg.norm(X[:, e[:, 0]] - X[:, e[:, 1]], axis=-1)
    lo = L.min(axis=1)
    return np.where(lo > 0, L.max(axis=1) / np.where(lo > 0, lo, 1.0), np.inf)


# ---------------------------------------------------------------------------------------- Bernstein lower bound
def _bernstein_lattice(ref: ReferenceElement) -> tuple[np.ndarray, np.ndarray]:
    """(reference points of the lattice (nl, dim), Bernstein basis matrix (nl, nl)) for the polynomial degree of
    detJ on this element."""
    d = ref.dim
    if ref.kind == "P":
        m = d * (ref.degree - 1)
        if m == 0:
            return np.full((1, d), 1.0 / (d + 1)), np.ones((1, 1))
        alphas = [a for a in product(range(m + 1), repeat=d + 1) if sum(a) == m]
        lam = np.array(alphas, float) / m
        B = np.empty((len(alphas), len(alphas)))
        for j, a in enumerate(alphas):
            coef = factorial(m) / np.prod([factorial(i) for i in a])
            B[:, j] = coef * np.prod(lam ** np.array(a), axis=1)
        return lam[:, 1:], B
    n = {1: 1, 2: 3}[ref.degree] if d == 2 else 2
    t = np.linspace(0.0, 1.0, n + 1)
    idx = list(product(range(n + 1), repeat=d))
    pts = np.array([[t[i] for i in ix] for ix in idx])
    B = np.empty((len(idx), len(idx)))
    for j, ix in enumerate(idx):
        B[:, j] = np.prod([comb(n, i) * pts[:, k] ** i * (1 - pts[:, k]) ** (n - i) for k, i in enumerate(ix)], axis=0)
    return 2.0 * pts - 1.0, B


def _bernstein_bound(ref: ReferenceElement, X: np.ndarray) -> np.ndarray:
    """(ne,) min Bernstein coefficient of detJ (a lower bound of detJ over the cell)."""
    xi, B = _bernstein_lattice(ref)
    det = np.linalg.det(_jacobians(ref, X, xi))                       # (ne, nl)
    coef = np.linalg.solve(B[None], det[:, :, None])[..., 0]
    return coef.min(axis=1)


def quality_from_nodes(ref: ReferenceElement, X: np.ndarray) -> CellQuality:
    """:class:`CellQuality` of cells with node coordinates X (ne, n_nodes, dim) of element ``ref``."""
    A = ideal_jacobian(ref)
    d = ref.dim
    X = np.asarray(X, float)
    J = _jacobians(ref, X, _samples(ref))
    S = J @ np.linalg.inv(A)
    det = np.linalg.det(S)
    fro2 = (S ** 2).sum(axis=(-1, -2))
    q = d * np.sign(det) * np.abs(det) ** (2.0 / d) / np.where(fro2 > 0, fro2, 1.0)
    cn = np.linalg.norm(S, axis=-2).prod(axis=-1)
    sj = np.where(cn > 0, det / np.where(cn > 0, cn, 1.0), 0.0)
    sv = np.linalg.svd(S, compute_uv=False)
    with np.errstate(divide="ignore"):
        kappa = np.where(sv[..., -1] > 0, sv[..., 0] / np.where(sv[..., -1] > 0, sv[..., -1], 1.0), np.inf)
    scale = element_scale(ref, X)
    dj = np.linalg.det(J)
    bound = _bernstein_bound(ref, X) / scale
    return CellQuality(ref.name, q.min(axis=1), sj.min(axis=1), kappa.max(axis=1), _edge_ratio(ref, X),
                       dj.min(axis=1) / scale, bound, bound > DETJ_TOL)


def cell_quality(mesh: Mesh) -> CellQuality:
    """Quality and Jacobian certificate of every cell of ``mesh`` (T3, T6, Q4, Q9, T4, T10, H8)."""
    return quality_from_nodes(get(mesh.element), mesh.cell_points())


def flag_cells(quality: CellQuality, mean_ratio: float) -> np.ndarray:
    """Cell ids with mean ratio below the caller's threshold, any non-finite value, or an inverted sample."""
    bad = (quality.mean_ratio_min < mean_ratio) | ~np.isfinite(quality.mean_ratio_min) | \
        (quality.detj_min_rel < -DETJ_TOL)
    return np.flatnonzero(bad)
