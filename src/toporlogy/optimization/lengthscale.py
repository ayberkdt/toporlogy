"""Length-scale tools: geometry measurement of a design, grey level, and the local-volume (maximum size) response.

Measurement (:func:`measure_lengthscale`)
-----------------------------------------
Definition used everywhere in this module — the *local thickness* of a phase P (solid or void) at a point p is the
diameter of the largest ball that lies inside P and contains p::

    τ(p) = 2 · max{ r : ∃ c with p ∈ B(c, r) ⊂ P }.

This is the morphological-opening definition: τ(p) ≥ 2r exactly when p survives the opening of P by a ball of
radius r. Consequences, all intended:

* a plate / bar of thickness t has τ = t everywhere; the largest τ of a phase is twice its largest inscribed radius
  (``solid_max`` = maximum member thickness, ``void_max`` = largest hole);
* a convex corner of the solid with radius ρ gives τ ≈ 2ρ near the corner, so ``solid_min`` is
  min(thinnest member, 2 × sharpest convex corner radius); the same on the void phase makes ``void_min`` =
  min(smallest hole / gap, 2 × sharpest re-entrant corner radius of the solid). A knife edge or a sharp re-entrant
  corner therefore shows up as a small ``solid_min`` / ``void_min``. A design has minimum solid size b in this sense
  iff opening by a ball of diameter b leaves it unchanged — the property the robust formulation aims at (convex
  corners rounded with radius ≥ b/2).

Discretisation. The element field (one value per element centre) is up-sampled by multilinear interpolation onto a
grid ``upsample`` times finer and thresholded there (the threshold contour of the interpolated field); with crisp
0/1 elements the contour lies exactly on the element faces. Slanted edges of crisp designs remain staircases with
one-element teeth, which are geometrically convex corners and pull ``solid_min`` / ``void_min`` down to ≈ h; the
optional ``smooth`` (Gaussian, σ in elements) removes them, and the percentiles are insensitive to them.
On the fine grid D(c) = Euclidean distance (``scipy.ndimage.distance_transform_edt``) from fine cell c to the nearest
fine cell of the other phase (cell centre to cell centre), and τ(p) = 2·max{r·h_f : |p − c| < r ≤ D(c)} — the
largest discrete ball that contains no cell centre of the other phase — evaluated on levels of D. A member of n fine
cells across (n even) measures exactly n·h_f (odd n: (n + 1)·h_f); in general the error is about one fine cell
h_f = h/upsample. Opening by a ball of radius ρ uses the same convention (centres with D ≥ ρ/h_f, coverage
|p − c| < ρ/h_f).

Outside the grid. Every face of the box has a rule: ``"void"`` (the part ends there — true free edge), ``"solid"``,
or ``"reflect"`` (symmetry plane: the design continues as its mirror image). Elements with ``frac = 0`` count as void.
Note: the density filter (:class:`~toporlogy.chain.DensityFilter`) normalises by the in-domain weight, which
behaves like ``"reflect"`` at every face, so length-scale guarantees of filter-based methods hold in that sense at
free faces; measure with ``"void"`` to see the true free-edge geometry.

Maximum size (:class:`LocalVolume`)
-----------------------------------
Local volume constraint of Wu, Aage, Westermann & Sigmund (2018, IEEE TVCG 24(2):1127-1140; arXiv:1608.04366v2,
**read in this session**, §3.1-3.2): ρ̄_e = mean of the projected density over the elements whose centroid lies within
R_e of the centroid of e (Eq. 1-2, uniform weights), ρ̄_e ≤ α (Eq. 6), aggregated by
``(1/n Σ_e ρ̄_e^p)^(1/p) ≤ α`` with p = 16 (Eq. 9). The paper notes that this p-mean under-estimates max ρ̄ and that
the limit is exceeded at a few places; :class:`LocalVolume` therefore offers an adaptive correction factor
c = max ρ̄ / p-mean from the previous evaluation (the normalised-aggregation idea of Le et al. 2010, ``le_norato_2010``;
that paper was not read in this session). The relation α, R → maximum member thickness is purely geometric
(:func:`max_thickness_settings`).

Grey level: M_nd = mean of 4 x̄(1 − x̄) (Sigmund 2007; also Lazarov, Wang & Sigmund 2016 Eq. 30 — formula as recorded in
the registry notes and in :func:`aerokit.optimization.topology.gray_measure`; the papers were not re-read in this
session).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as tF
from scipy import ndimage

from toporlogy.references import cite
from toporlogy.optimization.responses import EvalContext, Response

__all__ = ["LengthScaleReport", "measure_lengthscale", "local_thickness", "opening_residue", "closing_residue",
           "grey_measure", "GreyLevel", "LocalVolume", "local_volume_field", "ball_kernel",
           "max_thickness_settings", "slab_fraction"]

_MODES = ("void", "solid", "reflect")


# ============================================================================================ face rules and padding
def _faces(outside: Any, dim: int, allowed: Sequence[str] = _MODES) -> list[tuple[str, str]]:
    """Normalise ``outside`` to one (low, high) rule pair per axis."""
    if isinstance(outside, str):
        out = [(outside, outside)] * dim
    else:
        seq = list(outside)
        if len(seq) != dim:
            raise ValueError(f"outside: one entry per axis ({dim}) or a single string")
        out = [(s, s) if isinstance(s, str) else (str(s[0]), str(s[1])) for s in seq]
    for lo, hi in out:
        if lo not in allowed or hi not in allowed:
            raise ValueError(f"outside rules must be in {tuple(allowed)}")
    return out


def _pad_np(a: np.ndarray, width: int, faces: list[tuple[str, str]]) -> np.ndarray:
    """Pad every face by ``width`` cells with its rule ("void" → 0/False, "solid" → 1/True, "reflect" → mirror)."""
    if width <= 0:
        return a
    for ax, rules in enumerate(faces):
        for side, rule in enumerate(rules):
            pw = [(0, 0)] * a.ndim
            pw[ax] = (width, 0) if side == 0 else (0, width)
            if rule == "reflect":
                a = np.pad(a, pw, mode="symmetric")
            else:
                a = np.pad(a, pw, mode="constant", constant_values=(rule == "solid"))
    return a


def _pad_torch(a: torch.Tensor, width: int, faces: list[tuple[str, str]], values: dict[str, float]) -> torch.Tensor:
    """Torch version of :func:`_pad_np`; ``values`` gives the constant of each non-reflect rule."""
    if width <= 0:
        return a
    for ax, rules in enumerate(faces):
        for side, rule in enumerate(rules):
            n = a.shape[ax]
            if rule == "reflect":
                reps = []
                remaining = width
                flip = torch.flip(a, dims=(ax,))
                pieces = [flip, a]                       # mirror, original, mirror, ... (symmetric padding)
                k = 0
                while remaining > 0:
                    take = min(remaining, n)
                    src = pieces[k % 2]
                    reps.append(src.narrow(ax, n - take, take) if side == 0 else src.narrow(ax, 0, take))
                    remaining -= take
                    k += 1
                if side == 0:
                    a = torch.cat(list(reversed(reps)) + [a], dim=ax)
                else:
                    a = torch.cat([a] + reps, dim=ax)
            else:
                shp = list(a.shape)
                shp[ax] = width
                c = torch.full(shp, float(values[rule]), dtype=a.dtype, device=a.device)
                a = torch.cat([c, a] if side == 0 else [a, c], dim=ax)
    return a


# ============================================================================================ measurement
def _upsample(x: np.ndarray, s: int, faces: list[tuple[str, str]]) -> np.ndarray:
    """Multilinear interpolation of element-centre values onto centres of an s-times finer grid (separable)."""
    a = _pad_np(np.asarray(x, float), 1, faces)
    for ax in range(a.ndim):
        n = a.shape[ax] - 2
        u = (np.arange(n * s) + 0.5) / s - 0.5 + 1.0      # fine centre in padded element-index coordinates
        i0 = np.floor(u).astype(np.int64)
        w = u - i0
        shp = [1] * a.ndim
        shp[ax] = -1
        a = np.take(a, i0, axis=ax) * (1.0 - w).reshape(shp) + np.take(a, i0 + 1, axis=ax) * w.reshape(shp)
    return a


def _edt_padded(ph: np.ndarray, fc: list[tuple[str, str]], pad: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """EDT of the phase on an array padded by the face rules; the padding grows until it no longer limits D."""
    cap = max(ph.shape)
    if pad is not None:
        p = int(pad)
    elif all(rule == "void" for f in fc for rule in f):
        p = 1                                               # the padding is the other phase: D is exact already
    else:
        p = min(8, cap)
    while True:
        big = _pad_np(ph, p, fc)
        if not (~big).any():
            return big, np.full(big.shape, np.inf), p
        D = ndimage.distance_transform_edt(big)
        core = tuple(slice(p, p + n) for n in ph.shape)
        if pad is not None or p >= cap or D[core].max() < p - 1:
            return big, D, p
        p = min(2 * p, cap)


def local_thickness(phase: np.ndarray, faces: list[tuple[str, str]] | str = "void", pad: int | None = None,
                    level_step: float = 0.5, max_levels: int = 48, exact: float = 6.0) -> np.ndarray:
    """Local thickness τ (in cells) of the True phase of a binary array; 0 outside the phase.

    ``faces`` rules refer to this phase ("void" = outside is NOT this phase). ``pad`` (cells) default: grown
    automatically until it does not limit the inscribed balls. Levels of D are exact up to ``exact`` cells and above
    that every step = max(``level_step``, max D / ``max_levels``) cells (D rounded down to that grid, so τ is
    under-estimated by < 2·step cells there: absolute below 48·level_step, else relative ≤ 2/max_levels ≈ 4 %).
    """
    ph = np.asarray(phase, bool)
    fc = _faces(faces, ph.ndim) if isinstance(faces, str) else faces
    if not ph.any():
        return np.zeros(ph.shape)
    big, dist, p = _edt_padded(ph, fc, pad)
    core = tuple(slice(p, p + n) for n in ph.shape)
    if np.isinf(dist).any():                                # phase fills everything: thickness unbounded
        return np.where(ph, np.inf, 0.0)
    step = max(float(level_step), float(dist[big].max()) / max_levels)
    # exact levels up to ``exact`` cells (few distinct values √(i² + j² [+ k²])), quantised above: rounding D down at
    # small radii would leave boundary cells uncovered (D = √2 floored to 1 no longer covers its neighbour)
    Dq = np.where(dist <= exact, dist, np.floor(dist / step) * step)
    levels = np.unique(Dq[big])
    tau = np.zeros(big.shape)
    tau[big] = 2.0                                          # every cell of the phase: D ≥ 1
    for r in levels[levels > 1.0]:
        # only centres within r of the core matter: crop to the core grown by ceil(r)
        g = int(math.ceil(r))
        win = tuple(slice(max(p - g, 0), min(p + n + g, N)) for n, N in zip(ph.shape, big.shape, strict=True))
        centres = Dq[win] >= r
        # ball coverage |q − c| < r ≤ D(c): the ball contains no centre of the other phase
        dcov = ndimage.distance_transform_edt(~centres)
        cov = (dcov < r) & big[win]
        tw = tau[win]
        tw[cov] = np.maximum(tw[cov], 2.0 * r)
    return tau[core]


@dataclass
class LengthScaleReport:
    """Result of :func:`measure_lengthscale`; lengths in the unit of ``h``."""

    solid_min: float
    solid_percentiles: dict[float, float]
    solid_max: float
    void_min: float
    void_percentiles: dict[float, float]
    void_max: float
    grey_fraction: float
    m_nd: float
    solid_fraction: float
    opening_residue: dict[float, float] = field(default_factory=dict)
    closing_residue: dict[float, float] = field(default_factory=dict)
    upsample: int = 1
    resolution: float = 0.0                                 # fine cell size h/upsample (≈ measurement error)
    solid_map: np.ndarray | None = None                    # τ on the fine grid (solid), 0 elsewhere
    void_map: np.ndarray | None = None

    def summary(self) -> str:
        ps = ", ".join(f"p{k:g} {v:.3g}" for k, v in self.solid_percentiles.items())
        pv = ", ".join(f"p{k:g} {v:.3g}" for k, v in self.void_percentiles.items())
        return (f"solid thickness min {self.solid_min:.3g} ({ps}) max {self.solid_max:.3g}; void size min "
                f"{self.void_min:.3g} ({pv}) max {self.void_max:.3g}; grey {100 * self.grey_fraction:.2f} %, "
                f"M_nd {100 * self.m_nd:.2f} %; resolution {self.resolution:.3g}")


def _phase_stats(tau: np.ndarray, sel: np.ndarray, h_f: float, pct: Sequence[float]) -> tuple[float, dict, float]:
    v = tau[sel]
    if v.size == 0:
        return math.nan, {float(q): math.nan for q in pct}, math.nan
    return float(v.min() * h_f), {float(q): float(np.percentile(v, q) * h_f) for q in pct}, float(v.max() * h_f)


def _opening_fine(ph: np.ndarray, r_cells: float, faces: list[tuple[str, str]]) -> np.ndarray:
    """Opening of the True phase by a Euclidean ball of radius ``r_cells`` (fine cells), exact on the grid."""
    if not ph.any():
        return ph.copy()
    pad = min(max(ph.shape), int(math.ceil(2 * r_cells)) + 2)
    big, dist_in, _ = _edt_padded(ph, faces, pad)
    if np.isinf(dist_in).any():
        return ph.copy()
    centres = dist_in >= r_cells
    dist = ndimage.distance_transform_edt(~centres) if centres.any() else np.full(big.shape, np.inf)
    opened = (dist < r_cells) & big
    core = tuple(slice(pad, pad + n) for n in ph.shape)
    return opened[core]


def _invert(faces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    sw = {"void": "solid", "solid": "void", "reflect": "reflect"}
    return [(sw[a], sw[b]) for a, b in faces]


def _prepare(xphys: np.ndarray, h: float, threshold: float, outside: Any, upsample: int | None,
             mask: np.ndarray | None, smooth: float = 0.0) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]],
                                                                     int, float]:
    x = np.asarray(xphys, float)
    if x.ndim not in (2, 3):
        raise ValueError("xphys must be a 2-D or 3-D element array")
    s = int(upsample if upsample is not None else (4 if x.ndim == 2 else 2))
    if s < 1:
        raise ValueError("upsample ≥ 1")
    m = np.ones(x.shape, bool) if mask is None else np.asarray(mask, bool).reshape(x.shape)
    x = np.where(m, x, 0.0)                                 # outside the part = void
    faces = _faces(outside, x.ndim)
    fine = _upsample(x, s, faces) if s > 1 else x
    if smooth > 0:
        sig = float(smooth) * s
        w = int(math.ceil(4 * sig))
        big = ndimage.gaussian_filter(_pad_np(fine, w, faces), sig, mode="nearest")
        fine = big[tuple(slice(w, w + n) for n in fine.shape)]
    mf = m
    for ax in range(x.ndim):
        mf = np.repeat(mf, s, axis=ax)
    solid = (fine >= threshold) & mf
    return solid, mf, faces, s, h / s


@cite("sigmund_2007", note="M_nd = Σ 4x(1 − x)/n reported alongside (registry note; not re-read in this session)")
def measure_lengthscale(xphys: np.ndarray, h: float = 1.0, threshold: float = 0.5, outside: Any = "void",
                        upsample: int | None = None, mask: np.ndarray | None = None,
                        percentiles: Sequence[float] = (1.0, 5.0, 50.0), radii: Sequence[float] = (),
                        margin: float = 0.0, smooth: float = 0.0, keep_maps: bool = False) -> LengthScaleReport:
    """Measure member thickness, void size, grey level and opening/closing residues of an element density field.

    Parameters
    ----------
    xphys : array (nx, ny[, nz])
        Physical density per element.
    h : float
        Element size; every length returned is in this unit.
    threshold : float
        Iso-value of the solid (on the multilinearly interpolated field).
    outside : str or per-axis sequence of str / (low, high)
        Rule for each box face ("void", "solid", "reflect"), see the module notes.
    upsample : int
        Fine cells per element and axis (default 4 in 2-D, 2 in 3-D). Resolution of the result ≈ h / upsample.
    mask : bool array, optional
        Part mask (e.g. ``domain.active``); elements outside count as void and are excluded from void statistics.
    percentiles : sequence of float
        Percentiles (by area/volume of the phase) of the local thickness reported for each phase.
    radii : sequence of float
        Radii (in the unit of h) for the opening / closing residues: ``opening_residue[r]`` = share of the solid removed
        by opening with a ball of radius r (thin members, knife edges, sharp convex corners);
        ``closing_residue[r]`` = solid added by closing, as a share of the solid (thin slits, small holes, sharp
        re-entrant corners).
    margin : float
        Cells closer than ``margin`` (unit of h) to a face whose rule is not "reflect" are left out of the min /
        percentile statistics (they are still part of the geometry).
    smooth : float
        Standard deviation (in elements) of a Gaussian applied to the up-sampled field before thresholding
        (default 0 = none). On a voxel grid a slanted edge is a staircase whose one-element teeth are, geometrically,
        convex corners of radius ≈ h/2, so ``solid_min`` / ``void_min`` of an unsmoothed slanted design are ≈ h no
        matter how thick its members are (the percentiles are not affected). ``smooth = 0.5``-1 removes the teeth:
        it leaves the position of a straight face unchanged but rounds corners with radius ~σ and lets members
        thinner than ~2σ shrink (conservative for minimum-size checks, so use it there, not for maximum size).
    keep_maps : bool
        Return the fine τ maps.
    """
    solid, mf, faces, s, h_f = _prepare(xphys, h, threshold, outside, upsample, mask, smooth)
    x = np.asarray(xphys, float)
    m = np.ones(x.shape, bool) if mask is None else np.asarray(mask, bool).reshape(x.shape)
    xm = x[m]
    grey = float(np.mean((xm > 0.1) & (xm < 0.9))) if xm.size else 0.0
    mnd = float(np.mean(4.0 * xm * (1.0 - xm))) if xm.size else 0.0
    void = mf & ~solid
    # void phase: cells outside the part (mask False) are void too
    void_all = ~solid
    tau_s = local_thickness(solid, faces)
    tau_v = local_thickness(void_all, _invert(faces))
    sel = np.ones(solid.shape, bool)
    if margin > 0:
        mc = int(round(margin / h_f))
        for ax, (lo, hi) in enumerate(faces):
            idx = [slice(None)] * solid.ndim
            if lo != "reflect":
                idx[ax] = slice(0, mc)
                sel[tuple(idx)] = False
            if hi != "reflect":
                idx[ax] = slice(solid.shape[ax] - mc, None)
                sel[tuple(idx)] = False
    smin, sp, smax = _phase_stats(tau_s, solid & sel, h_f, percentiles)
    vmin, vp, vmax = _phase_stats(tau_v, void & sel, h_f, percentiles)
    if solid.any():
        smax = float(tau_s[solid].max() * h_f)
    if void.any():
        vmax = float(tau_v[void].max() * h_f)
    rep = LengthScaleReport(smin, sp, smax, vmin, vp, vmax, grey, mnd, float(solid.sum() / max(mf.sum(), 1)),
                            upsample=s, resolution=h_f)
    ns = max(int(solid.sum()), 1)
    for r in radii:
        rc = float(r) / h_f
        opened = _opening_fine(solid, rc, faces)
        rep.opening_residue[float(r)] = float((solid & ~opened).sum() / ns)
        closed = ~_opening_fine(void_all, rc, _invert(faces))
        rep.closing_residue[float(r)] = float((closed & ~solid & mf).sum() / ns)
    if keep_maps:
        rep.solid_map, rep.void_map = tau_s * h_f, np.where(void, tau_v * h_f, 0.0)
    return rep


def opening_residue(xphys: np.ndarray, h: float, radius: float, **kw: Any) -> float:
    """Share of the solid removed by opening with a ball of ``radius`` (see :func:`measure_lengthscale`)."""
    return measure_lengthscale(xphys, h, radii=(radius,), **kw).opening_residue[float(radius)]


def closing_residue(xphys: np.ndarray, h: float, radius: float, **kw: Any) -> float:
    """Solid added by closing with a ball of ``radius``, as a share of the solid."""
    return measure_lengthscale(xphys, h, radii=(radius,), **kw).closing_residue[float(radius)]


# ============================================================================================ grey level
def grey_measure(xphys: np.ndarray, mask: np.ndarray | None = None) -> float:
    """M_nd = mean over the (masked) elements of 4 x̄ (1 − x̄), as a fraction (× 100 for %)."""
    v = np.asarray(xphys, float)
    v = v[np.asarray(mask, bool)] if mask is not None else v.ravel()
    return float(np.mean(4.0 * v * (1.0 - v))) if v.size else 0.0


@cite("sigmund_2007", "lazarov_wang_sigmund_2016", note="measure of non-discreteness M_nd (fraction, not %)")
class GreyLevel(Response):
    """Differentiable M_nd = mean of 4 x̄(1 − x̄) over the design elements (keep / void / outside left out).

    Use as a monitor or as a constraint ``GreyLevel() ≤ m`` (e.g. 0.05). ``source`` = optional callable
    ``ctx -> field`` (default ``ctx.xphys``).
    """

    name = "grey_level"

    def __init__(self, source: Any = None, name: str | None = None) -> None:
        self.source = source
        if name:
            self.name = name

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        x = ctx.xphys if self.source is None else self.source(ctx)
        sel = torch.as_tensor(ctx.domain.design, device=x.device)
        v = x[sel]
        return (4.0 * v * (1.0 - v)).mean()


# ============================================================================================ local volume
def ball_kernel(radius: float, dim: int) -> np.ndarray:
    """Uniform kernel over element offsets with centroid distance ≤ ``radius`` (elements), Wu et al. Eq. (2)."""
    if not radius >= 1.0:
        raise ValueError("radius ≥ 1 element is required")
    r = int(math.floor(radius))
    ax = np.arange(-r, r + 1)
    grids = np.meshgrid(*([ax] * dim), indexing="ij")
    return (sum(g.astype(float) ** 2 for g in grids) <= radius ** 2 + 1e-12).astype(float)


def local_volume_field(x: torch.Tensor | np.ndarray, radius: float, outside: Any = "void",
                       mask: np.ndarray | None = None) -> torch.Tensor:
    """ρ̄_e: mean of ``x`` over the ball of ``radius`` elements around every element (differentiable in torch).

    ``outside`` rules per face: "void" (outside counts as empty), "solid", "reflect" (symmetry plane) or "ignore"
    (outside not counted — the in-domain mean). Elements with ``mask`` False count as empty.
    """
    xt = torch.as_tensor(x, dtype=torch.float64) if not isinstance(x, torch.Tensor) else x
    dim = xt.dim()
    faces = _faces(outside, dim, _MODES + ("ignore",))
    k = torch.as_tensor(ball_kernel(radius, dim), dtype=xt.dtype, device=xt.device)
    w = k.shape[0] // 2
    if mask is not None:
        xt = xt * torch.as_tensor(np.asarray(mask, bool), device=xt.device).to(xt.dtype)
    xp = _pad_torch(xt, w, faces, {"void": 0.0, "solid": 1.0, "ignore": 0.0})
    conv = tF.conv3d if dim == 3 else tF.conv2d
    num = conv(xp[None, None], k[None, None])[0, 0]
    if any("ignore" in f for f in faces):
        ones = torch.ones_like(xt)
        mp = _pad_torch(ones, w, faces, {"void": 1.0, "solid": 1.0, "ignore": 0.0})
        den = conv(mp[None, None], k[None, None])[0, 0]
        return num / den
    return num / k.sum()


class LocalVolume(Response):
    """Aggregated local volume fraction max_e ρ̄_e (≈), for the constraint ``LocalVolume(R) ≤ α`` (maximum size).

    Wu, Aage, Westermann & Sigmund (2018), Eq. (1), (2), (9) — read in this session (arXiv:1608.04366v2; not in the
    reference registry, see module notes).

    Parameters
    ----------
    radius : float
        Ball radius R in elements (centroid distance ≤ R).
    p : float
        Exponent of the aggregation (paper: 16).
    aggregate : {"pmean", "max"}
        "pmean" = (1/n Σ ρ̄_e^p)^(1/p) over the active elements (Wu et al. Eq. 9); "max" = exact maximum (gradient
        of the arg-max element only).
    adaptive : float in [0, 1]
        0 = off. Otherwise the p-mean is multiplied by a factor c updated after every gradient-enabled evaluation as
        c ← a·(max ρ̄ / p-mean) + (1 − a)·c, so the constrained value tracks the true maximum. The factor is held
        constant inside one evaluation (its derivative is not included — standard for this correction); switch it
        off for gradient checks.
    outside : face rules, see :func:`local_volume_field`. Default "void" (the part ends at the box faces).
    source : callable ``ctx -> field``, optional
        Field to bound; default ``ctx.xphys``. With the robust formulation pass the dilated realisation, e.g.
        ``source=robust.field("dilated")``, so the thickest realisation is bounded.
    """

    name = "local_volume"

    def __init__(self, radius: float, p: float = 16.0, aggregate: str = "pmean", adaptive: float = 0.0,
                 outside: Any = "void", source: Any = None, name: str | None = None) -> None:
        if aggregate not in ("pmean", "max"):
            raise ValueError("aggregate: pmean | max")
        if not 0.0 <= adaptive <= 1.0:
            raise ValueError("adaptive must lie in [0, 1]")
        self.radius, self.p, self.aggregate = float(radius), float(p), aggregate
        self.adaptive, self.outside, self.source = float(adaptive), outside, source
        self.correction = 1.0
        self.last_max = math.nan
        if name:
            self.name = name

    def field(self, ctx: EvalContext) -> torch.Tensor:
        x = ctx.xphys if self.source is None else self.source(ctx)
        return local_volume_field(x, self.radius, self.outside, ctx.domain.active)

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        rb = self.field(ctx)
        v = rb[torch.as_tensor(ctx.domain.active, device=rb.device)]
        vmax = v.max()
        if self.aggregate == "max":
            return vmax
        pm = (v.clamp_min(0.0) ** self.p).mean() ** (1.0 / self.p)
        if torch.is_grad_enabled():
            self.last_max = float(vmax)
        out = self.correction * pm
        if self.adaptive > 0 and torch.is_grad_enabled():
            ratio = float(vmax) / max(float(pm), 1e-300)
            self.correction = self.adaptive * ratio + (1.0 - self.adaptive) * self.correction
        return out


def slab_fraction(u: float, dim: int, member: str = "plate") -> float:
    """Continuum share of a ball of radius R filled by a centred member of half-thickness u·R.

    2-D bar (strip in a disc): (2/π)(u√(1−u²) + asin u). 3-D plate (slab in a ball): (3u − u³)/2. 3-D rod (cylinder
    of radius uR through the centre): 1.5 (u²s + 2/3 − s + s³/3), s = √(1 − u²). Derived here (elementary geometry).
    """
    u = min(max(float(u), 0.0), 1.0)
    if dim == 2:
        return (2.0 / math.pi) * (u * math.sqrt(1 - u * u) + math.asin(u))
    if member == "plate":
        return 0.5 * (3 * u - u ** 3)
    if member == "rod":
        s = math.sqrt(1 - u * u)
        return 1.5 * (u * u * s + 2.0 / 3.0 - s + s ** 3 / 3.0)
    raise ValueError("member: plate | rod")


def _discrete_member_fraction(t: float, radius: float, dim: int, member: str, sub: int = 16) -> float:
    """Local volume at the centre of a straight member of thickness ``t`` elements (exact partial coverage of the
    elements of the ball kernel; plate/bar centred on an element; rod: square-in-disc area by sub-sampling)."""
    k = ball_kernel(radius, dim)
    r = k.shape[0] // 2
    off = np.arange(-r, r + 1, dtype=float)
    lo, hi = off - 0.5, off + 0.5
    cov1 = np.clip(np.minimum(hi, t / 2) - np.maximum(lo, -t / 2), 0.0, 1.0)     # slab across axis 0
    if dim == 2 or member == "plate":
        shp = [1] * dim
        shp[0] = -1
        cov = np.broadcast_to(cov1.reshape(shp), k.shape)
    else:
        q = (np.arange(sub) + 0.5) / sub - 0.5
        A = np.zeros((2 * r + 1, 2 * r + 1))
        for i, a in enumerate(off):
            for j, b in enumerate(off):
                yy, zz = np.meshgrid(a + q, b + q, indexing="ij")
                A[i, j] = np.mean(yy ** 2 + zz ** 2 <= (t / 2) ** 2)
        cov = np.broadcast_to(A[None], k.shape)                  # rod along axis 0
    return float((k * cov).sum() / k.sum())


def max_thickness_settings(t_max: float, h: float = 1.0, dim: int = 2, radius: float | None = None,
                           member: str | None = None) -> dict[str, float]:
    """Ball radius R (elements) and bound α of :class:`LocalVolume` for a maximum member thickness ``t_max``.

    α is the local volume at the centre of a straight member of thickness t_max computed on the actual discrete
    ball kernel (``discrete``); the continuum value of :func:`slab_fraction` is returned as ``continuum``. A member
    thicker than t_max has a larger local volume at its centre and violates ρ̄ ≤ α; one thinner satisfies it — as
    long as no other solid lies inside the ball. Junctions, parallel members closer than ~2R and members of another
    shape see more solid, so the constraint is conservative there.

    ``member``: 2-D "bar"; 3-D "rod" (default — calibrated on a round rod, which then also bounds plates, more
    strictly: a plate of the same thickness fills more of the ball) or "plate" (bounds plates only; rods may grow
    thicker). Default radius R = max(t_max/h, 1.5): larger R lets the constraint also space members apart (and lowers
    α); R < t_max/2 makes the bound meaningless (α → 1).
    """
    t = float(t_max) / h
    if not t > 0:
        raise ValueError("t_max > 0 is required")
    member = member or ("bar" if dim == 2 else "rod")
    if dim == 2 and member != "bar":
        raise ValueError("2-D member: bar")
    if dim == 3 and member not in ("rod", "plate"):
        raise ValueError("3-D member: rod | plate")
    rad = float(radius) if radius is not None else max(t, 1.5)
    if rad < t / 2:
        raise ValueError("radius must be at least t_max/2 (in elements)")
    alpha = _discrete_member_fraction(t, rad, dim, member)
    cont = slab_fraction(t / (2 * rad), dim, "plate" if member in ("bar", "plate") else "rod")
    return {"radius": rad, "alpha": alpha, "continuum": cont, "t_max_elements": t, "member": member}  # type: ignore[dict-item]
