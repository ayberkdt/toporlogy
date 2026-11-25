"""Density → implicit shape on a fine grid: upsampling, normalised smoothing, volume-preserving threshold, sub-voxel
signed distance, morphological offsets.

Sign convention everywhere: φ < 0 inside the material. A :class:`Grid` holds sample points ``origin + h·(i, j, k)``
(C order). A sample "owns" the cube of edge ``h`` around it, so a field's volume is ``Σ pv(φ)·h³`` with the
partial-volume estimate ``pv(φ) = clip(1/2 − φ/h, 0, 1)`` (exact for a plane parallel to a grid face, second order
for curved surfaces).

Signed distance (:func:`redistance`). Binary distance transforms put the surface half-way between samples and give a
staircase whose iso-surface is visibly blocky (a Gaussian on the SDF is the usual work-around and rounds edges).
Here every sample next to a sign change gets a *foot point* on the zero level of the trilinear interpolant of the
input field (Newton steps x ← x − f ∇f/|∇f|²); the exact Euclidean distance transform of the interface samples
(scipy ``distance_transform_edt`` with the feature transform) gives each sample its nearest interface sample, and
the distance is taken to the tangent disks at the foot points of that sample and its 26 neighbours (disk radius
reduced at sharp edges). The result is a distance to the smooth surface, accurate to a small fraction of h near the
surface (measured in ``tests/toporlogy/test_post_sdf.py``: sphere R = 10 h, max error 0.03 h within 3 h of the surface;
next to sharp convex edges up to ~0.7 h).

Morphology on distance fields: erosion by r is the zero level of φ + r, dilation of φ − r; each is redistanced, so
opening = dilate(erode(φ)) and closing = erode(dilate(φ)) are the exact rolling-ball operations of the sampled
shape (offset rounding; Rossignac & Requicha 1986 — source not read in this session).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import math

import numba
import numpy as np
from scipy import ndimage

__all__ = ["Grid", "grid_for_domain", "upsample", "normalised_blur", "partial_volume", "solid_volume", "redistance",
           "erode", "dilate", "extrapolate", "opening", "closing", "mask_shape", "volume_threshold", "envelope_field"]


@dataclass
class Grid:
    origin: np.ndarray          # coordinates of sample (0, 0, 0)
    h: float
    shape: tuple[int, int, int]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def axes(self) -> list[np.ndarray]:
        return [self.origin[k] + self.h * np.arange(self.shape[k]) for k in range(3)]

    def points(self) -> np.ndarray:
        """(N, 3) sample coordinates in C order."""
        ax = self.axes()
        P = np.meshgrid(*ax, indexing="ij")
        return np.stack([p.ravel() for p in P], axis=1)

    def index_coords(self, X: np.ndarray) -> np.ndarray:
        """(3, n) fractional sample indices of the points X (n, 3) (for ``ndimage.map_coordinates``)."""
        return ((np.atleast_2d(X) - self.origin) / self.h).T

    def sample(self, field: np.ndarray, X: np.ndarray, order: int = 1) -> np.ndarray:
        return ndimage.map_coordinates(np.asarray(field, float), self.index_coords(X), order=order, mode="nearest")

    def to_image(self, field: np.ndarray, name: str = "phi") -> Any:
        import pyvista as pv
        im = pv.ImageData(dimensions=self.shape, spacing=(self.h,) * 3, origin=tuple(self.origin))
        im.point_data[name] = np.asarray(field, np.float64).ravel(order="F")
        return im


def grid_for_domain(domain: Any, upsample: int = 4, pad: float = 0.0) -> Grid:
    """Fine grid over the domain's box, ``h_f = domain.h / upsample``, extended by ``pad`` (length) + 2 h_f.

    Samples sit at ``box_min − pad' + h_f·(i + 1/2)`` so that the box faces lie half-way between samples.
    """
    if domain.dim != 3:
        raise ValueError("post-processing is 3-D")
    hf = domain.h / int(upsample)
    npad = int(np.ceil(pad / hf)) + 2
    lo = np.asarray(domain.origin, float) - npad * hf
    shape = tuple(int(s * upsample + 2 * npad) for s in domain.shape)
    return Grid(lo + 0.5 * hf, hf, shape)          # type: ignore[arg-type]


def upsample(domain: Any, rho: np.ndarray, grid: Grid, order: int = 1) -> np.ndarray:
    """Element densities (``domain.shape``) → fine-grid values by *normalised* interpolation.

    ``order`` 1 = trilinear, 3 = cubic B-spline (smoother; overshoot clipped to [0, 1]). Both the masked density
    ρ·m and the part mask m (m = 1 for elements with frac > 0) are interpolated and divided, so the empty exterior
    does not pull the density down at the envelope; the field is extrapolated (nearest) beyond the part and the exact
    envelope cuts it later.
    """
    m = domain.active.astype(float)
    r = np.asarray(rho, float).reshape(domain.shape) * m
    X = grid.points()
    c = ((X - np.asarray(domain.origin, float)) / domain.h - 0.5).T
    a = ndimage.map_coordinates(r, c, order=order, mode="nearest").reshape(grid.shape)
    b = ndimage.map_coordinates(m, c, order=order, mode="nearest").reshape(grid.shape)
    out = np.where(b > 1e-3, a / np.maximum(b, 1e-3), 0.0)
    return np.clip(out, 0.0, 1.0)


def normalised_blur(f: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """blur(f·w)/blur(w) with a Gaussian of ``sigma`` samples (normalised convolution: skins are not diluted)."""
    if sigma <= 0:
        return np.asarray(f, float)
    w = np.asarray(weight, float)
    a = ndimage.gaussian_filter(np.asarray(f, float) * w, sigma, mode="nearest")
    b = ndimage.gaussian_filter(w, sigma, mode="nearest")
    return np.where(b > 1e-3, a / np.maximum(b, 1e-3), 0.0)


def extrapolate(f: np.ndarray, care: np.ndarray) -> np.ndarray:
    """Values outside ``care`` replaced by the value of the nearest care sample (nearest-value extrapolation)."""
    care = np.asarray(care, bool)
    if care.all() or not care.any():
        return np.asarray(f, float)
    ind = ndimage.distance_transform_edt(~care, return_distances=False, return_indices=True)
    return np.asarray(f, float)[tuple(ind)]


def partial_volume(phi: np.ndarray, h: float) -> np.ndarray:
    """Solid fraction of each sample cube for a distance-like φ: clip(1/2 − φ/h, 0, 1)."""
    return np.clip(0.5 - np.asarray(phi, float) / h, 0.0, 1.0)


def solid_volume(phi: np.ndarray, h: float) -> float:
    return float(partial_volume(phi, h).sum()) * h ** 3


def _foot_points(f: np.ndarray, h: float, idx: np.ndarray, iterations: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Foot points (n, 3, index units) on the zero level of the trilinear interpolant, from samples ``idx`` (n, 3),
    and the unit normals there."""
    g = np.gradient(np.asarray(f, float))                       # per index unit
    P = idx.astype(float)
    start = P.copy()
    G = np.zeros_like(P)
    for _ in range(iterations + 1):
        c = P.T
        G = np.stack([ndimage.map_coordinates(gk, c, order=1, mode="nearest") for gk in g], axis=1)
        if _ == iterations:
            break
        v = ndimage.map_coordinates(f, c, order=1, mode="nearest")
        gg = np.maximum((G * G).sum(axis=1), 1e-12)
        P = P - (v / gg)[:, None] * G
        d = P - start                                           # stay within 1.5 samples of the start
        n = np.linalg.norm(d, axis=1)
        far = n > 1.5
        if far.any():
            P[far] = start[far] + d[far] * (1.5 / n[far])[:, None]
    N = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    return P, N


_DISK = 0.75          # radius (samples) of the tangent disk each foot point stands for
_EDGE = 0.65          # share of that radius kept at sharp edges (calibrated: opened cylinder and box)


def redistance(f: np.ndarray, h: float, band: float | None = None) -> np.ndarray:
    """Signed distance (length units) to the zero level of the sampled field ``f`` (φ < 0 where f < 0).

    ``band`` (length): samples farther than this from the surface keep the plain EDT value (to the interface
    sample's foot point without the 26-neighbour refinement) — saves time on big grids. Default: no limit.
    """
    f = np.asarray(f, float)
    inside = f < 0
    if inside.all() or not inside.any():
        return np.full(f.shape, -1e9 if inside.all() else 1e9)
    # interface samples: a 6-neighbour has the opposite sign
    st = ndimage.generate_binary_structure(3, 1)
    iface = (inside & ~ndimage.binary_erosion(inside, st, border_value=1)) | \
            (~inside & ndimage.binary_dilation(inside, st))
    ii = np.argwhere(iface)
    Q = np.full(f.shape + (3,), np.nan)
    Nq = np.zeros(f.shape + (3,))
    fp, fn = _foot_points(f, h, ii)
    Q[tuple(ii.T)] = fp
    Nq[tuple(ii.T)] = fn
    _, ind = ndimage.distance_transform_edt(~iface, return_distances=True, return_indices=True)
    has = np.isfinite(Q[..., 0])
    disk = _disk_radii(Nq, has, _DISK, _EDGE)
    d = _disk_distance(Q, Nq, ind.astype(np.int64), has, disk,
                       -1.0 if band is None else band / h)
    return np.where(inside, -d, d) * h


@numba.njit(parallel=True, cache=True)
def _disk_radii(Nq: np.ndarray, has: np.ndarray, disk: float, edge: float) -> np.ndarray:  # pragma: no cover - compiled
    """Disk radius per foot point: full on smooth parts, zero where the normals of neighbouring foot points differ by
    more than 45° (sharp edges; a tilted disk there would under-estimate the distance outside a convex edge and make
    a dilation grow — measured +1.1 % volume on an opened finite cylinder before this rule)."""
    nx, ny, nz = has.shape
    out = np.zeros((nx, ny, nz))
    c20, c45 = math.cos(math.radians(20.0)), math.cos(math.radians(45.0))
    for i in numba.prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not has[i, j, k]:
                    continue
                m = 1.0
                for a in range(max(i - 1, 0), min(i + 2, nx)):
                    for b in range(max(j - 1, 0), min(j + 2, ny)):
                        for c in range(max(k - 1, 0), min(k + 2, nz)):
                            if has[a, b, c]:
                                dd = (Nq[i, j, k, 0] * Nq[a, b, c, 0] + Nq[i, j, k, 1] * Nq[a, b, c, 1]
                                      + Nq[i, j, k, 2] * Nq[a, b, c, 2])
                                if dd < m:
                                    m = dd
                w = (m - c45) / (c20 - c45)
                w = min(max(w, 0.0), 1.0)
                out[i, j, k] = disk * (edge + (1.0 - edge) * w)
    return out


@numba.njit(parallel=True, cache=True)
def _disk_distance(Q: np.ndarray, Nq: np.ndarray, ind: np.ndarray, has: np.ndarray, disk: np.ndarray,
                   band: float) -> np.ndarray:  # pragma: no cover - compiled
    nx, ny, nz = has.shape
    out = np.empty((nx, ny, nz))
    for i in numba.prange(nx):
        for j in range(ny):
            for k in range(nz):
                bi, bj, bk = ind[0, i, j, k], ind[1, i, j, k], ind[2, i, j, k]
                dx, dy, dz = i - Q[bi, bj, bk, 0], j - Q[bi, bj, bk, 1], k - Q[bi, bj, bk, 2]
                d0 = np.sqrt(dx * dx + dy * dy + dz * dz)
                if band > 0 and d0 > band:
                    out[i, j, k] = d0
                    continue
                best = np.inf
                for a in range(max(bi - 1, 0), min(bi + 2, nx)):
                    for b in range(max(bj - 1, 0), min(bj + 2, ny)):
                        for c in range(max(bk - 1, 0), min(bk + 2, nz)):
                            if not has[a, b, c]:
                                continue
                            vx, vy, vz = i - Q[a, b, c, 0], j - Q[a, b, c, 1], k - Q[a, b, c, 2]
                            dn = vx * Nq[a, b, c, 0] + vy * Nq[a, b, c, 1] + vz * Nq[a, b, c, 2]
                            t2 = vx * vx + vy * vy + vz * vz - dn * dn
                            dt = np.sqrt(t2) if t2 > 0 else 0.0
                            e = dt - disk[a, b, c]
                            dd = dn * dn + (e * e if e > 0 else 0.0)
                            if dd < best:
                                best = dd
                out[i, j, k] = np.sqrt(best)
    return out


def erode(phi: np.ndarray, r: float, h: float) -> np.ndarray:
    return redistance(np.asarray(phi) + r, h)


def dilate(phi: np.ndarray, r: float, h: float) -> np.ndarray:
    return redistance(np.asarray(phi) - r, h)


def opening(phi: np.ndarray, r: float, h: float) -> np.ndarray:
    """Remove everything a ball of radius r cannot reach from inside (rounds convex edges to ≥ r)."""
    return dilate(erode(phi, r, h), r, h) if r > 0 else np.asarray(phi, float)


def closing(phi: np.ndarray, r: float, h: float) -> np.ndarray:
    """Fill everything a ball of radius r cannot reach from outside (fillets concave corners with ≥ r)."""
    return erode(dilate(phi, r, h), r, h) if r > 0 else np.asarray(phi, float)


def mask_shape(mask: np.ndarray, grid: Grid, smooth: float = 0.0, name: str = "mask") -> Any:
    """:class:`~toporlogy.post.primitives.GridPrimitive` of a boolean mask on ``grid`` (surface half-way between
    samples). ``smooth`` = Gaussian σ in samples applied to the indicator first (removes the staircase)."""
    from toporlogy.post.primitives import GridPrimitive
    ind = np.asarray(mask, float)
    if smooth > 0:
        ind = ndimage.gaussian_filter(ind, smooth)
    return GridPrimitive(redistance(0.5 - ind, grid.h), grid.origin, grid.h, name=name)


def envelope_field(domain: Any, grid: Grid) -> np.ndarray:
    """Distance-like field of the domain's part (frac > 0) on the fine grid, sub-voxel from the partial volumes.

    The element ``frac`` is interpolated (trilinear) to the fine samples and the 0.5 level redistanced: planar,
    sloped and curved envelope faces described by partial volumes come out without the element staircase.
    """
    fr = np.asarray(domain.frac, float)
    fr = np.pad(fr, 1)
    X = grid.points()
    c = ((X - np.asarray(domain.origin, float)) / domain.h + 0.5).T
    v = ndimage.map_coordinates(fr, c, order=1, mode="constant", cval=0.0).reshape(grid.shape)
    return redistance(0.5 - v, grid.h)


def volume_threshold(rf: np.ndarray, grid: Grid, target: float, combine: Any = None, lo: float = 0.0,
                     hi: float = 1.0, iterations: int = 40, redistanced: bool = False
                     ) -> tuple[float, np.ndarray, float]:
    """Level t such that the volume of {rf > t} (after ``combine``) equals ``target`` (Brent root search).

    The field (t − rf)/|∇rf| is distance-like near its zero level, so the partial-volume estimate is sub-voxel.
    ``combine(phi) -> phi`` re-imposes the exact geometry (keep, envelope, holes) inside the loop so that the volume
    counted is the volume of the finished part. With ``redistanced`` the root-search result is refined by two secant
    steps on the volume of the *redistanced* field (the first-order distance (t − rf)/|∇rf| uses the gradient at the
    sample, not at the surface, and biases the volume by ~0.2 % on a blurred step; measured on a plate).
    Returns (t, φ_t, volume); φ_t is the redistanced field when ``redistanced`` else the first-order field.
    """
    g = np.gradient(np.asarray(rf, float), grid.h)
    gm = np.maximum(np.sqrt(sum(x * x for x in g)), 1e-3 / grid.h)
    comb = combine if combine is not None else (lambda p: p)

    def field(t: float) -> np.ndarray:
        return (t - rf) / gm

    def vol(t: float) -> float:
        return solid_volume(comb(field(t)), grid.h)

    a, b = lo, hi
    va, vb = vol(a), vol(b)
    if not vb <= target <= va:
        raise ValueError(f"target volume {target:g} outside [{vb:g}, {va:g}] reachable by thresholding")
    from scipy.optimize import brentq
    t = float(brentq(lambda tt: vol(tt) - target, a, b, xtol=1e-10, rtol=1e-12, maxiter=iterations))
    v = vol(t)
    if not redistanced:
        return t, field(t), v
    dt = 1e-3
    slope = (vol(t + dt) - vol(t - dt)) / (2 * dt)              # dV/dt (< 0) of the cheap estimate
    phi = redistance(field(t), grid.h)
    v = solid_volume(comb(phi), grid.h)
    for _ in range(2):
        if abs(v - target) <= 1e-5 * target or slope >= 0:
            break
        t = float(np.clip(t - (v - target) / slope, lo, hi))
        phi = redistance(field(t), grid.h)
        v = solid_volume(comb(phi), grid.h)
    return t, phi, v


def sample_points_mask(grid: Grid, X: np.ndarray) -> np.ndarray:
    """Nearest sample index (n, 3) of points X (clipped into the grid)."""
    k = np.rint(grid.index_coords(X).T).astype(np.int64)
    return np.clip(k, 0, np.array(grid.shape) - 1)


def as_grid_points(grid: Grid) -> np.ndarray:
    return grid.points()


def axes_meshgrid(grid: Grid) -> Sequence[np.ndarray]:
    return np.meshgrid(*grid.axes(), indexing="ij", sparse=True)
