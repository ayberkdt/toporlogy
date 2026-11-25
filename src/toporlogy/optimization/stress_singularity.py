"""Does the peak stress of a mesh-refinement sequence converge?

At a point load, a sharp re-entrant corner or the end of a clamped edge the exact elastic stress is unbounded, so the
largest computed value is a property of the mesh, not of the structure: it grows with every refinement and must not
be compared with an allowable. :func:`assess_peak` reads that from a sequence of meshes and returns a status;
:func:`region_stress` gives the quantities of one mesh over a fixed physical box (peak, where it is, volume average).

The rule (pre-registered in ``docs/olcumler/gerilme_onkayit.md`` §4), from the last three meshes with increments
Δ₁ = p_{n−1} − p_{n−2} and Δ₂ = p_n − p_{n−1}: fewer than three meshes → ``NOT VERIFIED``; both increments positive
and Δ₂ ≥ ``ratio_min`` Δ₁ (the growth does not slow down) → :data:`SINGULAR`; |Δ₂| ≤ ``rel_tol`` |p_n| and shrinking
→ :data:`CONVERGED`; anything else → :data:`NOT_CONVERGED`.

It is a diagnosis from finitely many meshes, not a proof: a quantity converging as h^q with q below about 0.32 would be
called singular, and a singularity that has not yet outgrown the regular part of the stress on the meshes given would
not be seen. ``converged`` is a statement about this one scalar on these meshes only.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["SINGULAR", "CONVERGED", "NOT_CONVERGED", "NOT_VERIFIED", "PeakAssessment", "assess_peak", "RegionStress",
           "region_stress"]

SINGULAR = "singular — max stress not converged"
CONVERGED = "converged"
NOT_CONVERGED = "not converged"
NOT_VERIFIED = "NOT VERIFIED"


@dataclass(frozen=True)
class PeakAssessment:
    """Verdict of :func:`assess_peak`.

    ``slopes``: local exponents −Δ ln p / Δ ln h of consecutive mesh pairs (about λ for p ~ h^−λ, about 0 when p has
    settled). ``increment_ratio``: Δ₂ / Δ₁ (None when Δ₁ is below the noise floor or there are too few meshes).
    ``rel_change``: |Δ₂| / |p_n|.
    """

    status: str
    h: tuple[float, ...]
    peak: tuple[float, ...]
    slopes: tuple[float, ...]
    increment_ratio: float | None
    rel_change: float | None
    reason: str


def assess_peak(h: Sequence[float], peak: Sequence[float], rel_tol: float = 0.02, ratio_min: float = 0.8,
                noise: float = 1e-8) -> PeakAssessment:
    """Status of a peak (or any scalar) over meshes of strictly decreasing size ``h`` (module notes).

    Parameters
    ----------
    h : sequence of float
        Element sizes, strictly decreasing.
    peak : sequence of float
        The value on each mesh. A non-finite value gives ``NOT VERIFIED``.
    rel_tol : float
        Largest relative change of the last pair for ``converged``.
    ratio_min : float
        Δ₂ / Δ₁ from which the growth counts as not slowing down.
    noise : float
        Increments below ``noise`` |p_n| are rounding.
    """
    hs, ps = np.asarray(h, float), np.asarray(peak, float)
    if hs.ndim != 1 or ps.shape != hs.shape or hs.size == 0:
        raise ValueError("h and peak must be sequences of the same non-zero length")
    if not (np.isfinite(hs).all() and (hs > 0).all() and (np.diff(hs) < 0).all()):
        raise ValueError("h must be positive and strictly decreasing")
    slopes = tuple(float(-np.log(ps[i + 1] / ps[i]) / np.log(hs[i + 1] / hs[i]))
                   if ps[i] > 0 and ps[i + 1] > 0 else float("nan") for i in range(hs.size - 1))

    def out(status: str, reason: str, ratio: float | None = None, rel: float | None = None) -> PeakAssessment:
        return PeakAssessment(status, tuple(map(float, hs)), tuple(map(float, ps)), slopes, ratio, rel, reason)

    if not np.isfinite(ps).all():
        return out(NOT_VERIFIED, "a value is not finite")
    if hs.size < 3:
        return out(NOT_VERIFIED, "fewer than three meshes")
    d1, d2 = float(ps[-2] - ps[-3]), float(ps[-1] - ps[-2])
    floor = noise * abs(float(ps[-1]))
    ratio = d2 / d1 if abs(d1) > floor else None
    rel = abs(d2) / abs(float(ps[-1])) if ps[-1] != 0 else (0.0 if d2 == 0 else float("inf"))
    if d1 > floor and d2 > floor and d2 >= ratio_min * d1:
        return out(SINGULAR, "the peak grows with refinement and the growth does not slow down", ratio, rel)
    if rel <= rel_tol and (abs(d2) < ratio_min * abs(d1) or abs(d2) <= floor):
        return out(CONVERGED, "the last change is within the tolerance and shrinking", ratio, rel)
    return out(NOT_CONVERGED, "still changing by more than the tolerance, or not monotonically settling", ratio, rel)


@dataclass(frozen=True)
class RegionStress:
    """Scalar stress measure over the elements whose centres lie in a box."""

    peak: float
    location: tuple[float, ...]
    mean: float                      # volume-weighted
    n_elements: int


def region_stress(values: np.ndarray, points: np.ndarray, lo: Sequence[float] | None = None,
                  hi: Sequence[float] | None = None, weights: np.ndarray | None = None) -> RegionStress:
    """Peak, its location and the weighted mean of ``values`` (n,) at ``points`` (n, dim) inside the box [lo, hi]
    (default: everywhere). ``weights``: element volumes or solid fractions (default equal)."""
    v, X = np.asarray(values, float), np.asarray(points, float)
    if v.ndim != 1 or X.shape[:1] != v.shape:
        raise ValueError("values must be (n,) and points (n, dim)")
    inside = np.ones(v.size, bool)
    if lo is not None:
        inside &= (np.asarray(lo, float) <= X).all(1)
    if hi is not None:
        inside &= (np.asarray(hi, float) >= X).all(1)
    if not inside.any():
        raise ValueError("no element centre lies in the region")
    w = np.ones(v.size) if weights is None else np.asarray(weights, float)
    i = np.flatnonzero(inside)
    top = i[int(np.argmax(v[i]))]
    return RegionStress(float(v[top]), tuple(map(float, X[top])), float((v[i] * w[i]).sum() / w[i].sum()), int(i.size))
