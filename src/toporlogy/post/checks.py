"""Manufacturing gate for a finished surface: one result object, pass/fail per check, user thresholds.

Checks (``None`` threshold = measured and reported, not gated):

========================  ===========================================================  =========================
check                     measured                                                     default threshold
========================  ===========================================================  =========================
watertight                open, non-manifold, misoriented edges, non-manifold vertices  all 0 (always gated)
bodies                    outer shells of the mesh                                     ``max_bodies`` = 1
enclosed_voids            inner shells (cavities) of the mesh                          ``max_voids`` = 0
min_wall                  ray thickness (default) or inscribed-ball thickness, p1      ``min_wall`` None
overhang                  down-facing area steeper than ``overhang_deg`` above plate   ``max_overhang_fraction``
concave_radius            closing test at ``min_concave_radius`` (added depth)         ``min_concave_radius`` None
volume, surface_area      divergence theorem / triangle areas                          reported
========================  ===========================================================  =========================

Typical laser powder-bed fusion values, as quoted in service-bureau design guides (e.g. the Xometry metal
3-D printing design guide) and ISO/ASTM 52911-1 (design for PBF-LB/M) — neither read in this session, so they are
NOT defaults: walls ≥ 0.8–1.2 mm, self-supporting down-skin up to 45° from the build plate normal.

Local thickness (Hildebrand & Rüegsegger 1997 definition — source not read in this session): the thickness at a
point is the diameter of the largest inscribed ball that contains it. Ball centres are the ridge samples of the
inside distance (local maxima along some grid direction), the ridge radius is corrected to sub-sample accuracy
from the distance profile across the ridge (|∇d| = 1), and every solid sample gets the largest covering diameter.
Unlike 2·EDT between sample centres (which overstates walls by one sample) and, unlike a 3-D local-maximum test,
also measures necks (saddles of the distance).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numba
import numpy as np

from toporlogy.post import mesh as M
from toporlogy.post import sdf as S

__all__ = ["CheckThresholds", "CheckResult", "manufacturing_checks", "sdf_local_thickness", "ray_thickness", "overhang",
           "concave_radius_test"]


@dataclass
class CheckThresholds:
    max_bodies: int | None = 1
    max_voids: int | None = 0
    min_wall: float | None = None            # length; gated on the p1 value (``wall_statistic``)
    wall_statistic: str = "p1"               # "min" | "p1" | "p5"
    wall_method: str = "ray"                 # "ray" (mesh normals) | "sphere" (local thickness on the grid)
    build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    overhang_deg: float = 45.0               # down-skin steeper than this from the build direction is overhang
    plate_level: float | None = None         # build-plate height along the build direction; None = part minimum
    max_overhang_fraction: float | None = None
    min_concave_radius: float | None = None  # closing test radius (length)


@dataclass
class CheckResult:
    passed: bool
    items: list[dict[str, Any]]
    measured: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)

    def table(self) -> str:
        """Markdown table of the checks."""
        rows = ["| check | value | threshold | pass |", "|---|---|---|---|"]
        for it in self.items:
            rows.append(f"| {it['name']} | {_fmt(it['value'])} | {_fmt(it['threshold'])} | "
                        f"{'yes' if it['passed'] else ('n/a' if it['passed'] is None else 'NO')} |")
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "items": self.items, "measured": self.measured, "thresholds": self.thresholds}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# ------------------------------------------------------------------------------------------------ local thickness
@numba.njit(cache=True)
def _ridge_radius(D: np.ndarray, solid: np.ndarray) -> np.ndarray:  # pragma: no cover - compiled
    nx, ny, nz = D.shape
    R = np.zeros_like(D)
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1],
                     [0, 1, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if not solid[i, j, k]:
                    continue
                d0 = D[i, j, k]
                if d0 <= 0:
                    continue
                best_drop = -1.0
                rr = 0.0
                ridge = False
                for q in range(13):
                    a, b, c = dirs[q, 0], dirs[q, 1], dirs[q, 2]
                    dp = D[i + a, j + b, k + c]
                    dm = D[i - a, j - b, k - c]
                    if d0 >= dp and d0 >= dm:
                        ridge = True
                        if d0 > dp and d0 > dm:
                            L = math.sqrt(a * a + b * b + c * c)
                            hi = max(dp, dm)
                            lo = min(dp, dm)
                            slope = (d0 - lo) / L
                            drop = slope
                            if drop > best_drop:
                                best_drop = drop
                                peak = 0.5 * (d0 + hi + slope * L)
                                rr = min(max(peak, d0), d0 + 0.5 * L)
                if ridge:
                    R[i, j, k] = rr if best_drop > 0 else d0
    return R


@numba.njit(cache=True)
def _paint(R: np.ndarray, solid: np.ndarray) -> np.ndarray:  # pragma: no cover - compiled
    nx, ny, nz = R.shape
    LT = np.zeros_like(R)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = R[i, j, k]
                if r <= 0:
                    continue
                m = int(math.ceil(r))
                r2 = r * r + 1e-9
                for a in range(max(i - m, 0), min(i + m + 1, nx)):
                    da = (a - i) * (a - i)
                    for b in range(max(j - m, 0), min(j + m + 1, ny)):
                        db = da + (b - j) * (b - j)
                        if db > r2:
                            continue
                        for c in range(max(k - m, 0), min(k + m + 1, nz)):
                            if db + (c - k) * (c - k) <= r2 and solid[a, b, c] and LT[a, b, c] < 2 * r:
                                LT[a, b, c] = 2 * r
    return LT


def sdf_local_thickness(phi: np.ndarray, h: float) -> dict[str, Any]:
    """Local thickness (length) on the grid of a signed distance φ (φ < 0 inside): field + statistics."""
    solid = phi < 0
    D = np.where(solid, -phi / h, 0.0)
    R = _ridge_radius(D, solid)
    LT = _paint(R, solid) * h
    cov = LT[solid]
    covered = cov > 0
    v = cov[covered]
    ridge = 2 * R[R > 0] * h
    out: dict[str, Any] = {"field": LT, "samples": int(solid.sum()), "uncovered": int((~covered).sum())}
    if v.size:
        out.update({"min": float(v.min()), "p1": float(np.percentile(v, 1)), "p5": float(np.percentile(v, 5)),
                    "median": float(np.median(v)), "max": float(v.max()),
                    "ridge_min": float(ridge.min()), "ridge_p1": float(np.percentile(ridge, 1))})
    return out


def ray_thickness(V: np.ndarray, F: np.ndarray, phi: np.ndarray, grid: Any, max_length: float | None = None,
                  chunk: int = 20000) -> dict[str, Any]:
    """Wall thickness by rays: from every face centre along the inward normal to the first exit (φ > 0, trilinear
    φ on the grid, crossing interpolated linearly). Area-weighted statistics; rays longer than ``max_length``
    (default 2.2·max inside distance) count as ``max_length``. Unlike the inscribed-ball thickness this is not
    small at sharp convex edges; rays from end faces of long members can be long (that only overstates)."""
    h = grid.h
    L = float(max_length) if max_length is not None else 2.2 * float(max(-phi.min(), h)) + 2 * h
    n = M.face_normals(V, F)
    a = M.area(V, F, per_face=True)
    C = V[F].mean(axis=1)
    step = 0.25 * h
    ts = np.arange(1, int(np.ceil(L / step)) + 1) * step
    out = np.full(len(F), L)
    for s in range(0, len(F), chunk):
        c, d = C[s:s + chunk], -n[s:s + chunk]
        P = c[:, None, :] + ts[None, :, None] * d[:, None, :]
        f = grid.sample(phi, P.reshape(-1, 3)).reshape(len(c), len(ts))
        f0 = np.full((len(c), 1), -1e-12)
        ff = np.hstack([f0, f])
        tt = np.concatenate([[0.0], ts])
        ex = (ff[:, :-1] <= 0) & (ff[:, 1:] > 0)
        # ignore an exit in the first step if the ray started slightly outside (centre on the surface)
        has = ex.any(axis=1)
        k = np.argmax(ex, axis=1)
        r = np.arange(len(c))
        fa, fb = ff[r, k], ff[r, k + 1]
        frac = np.where(fb != fa, -fa / np.where(fb != fa, fb - fa, 1.0), 0.5)
        th = tt[k] + frac * step
        out[s:s + chunk] = np.where(has, th, L)
    order = np.argsort(out)
    cw = np.cumsum(a[order]) / a.sum()

    def pct(q: float) -> float:
        return float(out[order][min(np.searchsorted(cw, q / 100.0), len(out) - 1)])

    return {"min": float(out.min()), "p1": pct(1), "p5": pct(5), "median": pct(50), "max_length": L,
            "faces": int(len(F))}


# ------------------------------------------------------------------------------------------------ overhang etc.
def overhang(V: np.ndarray, F: np.ndarray, build_direction: Any = (0, 0, 1), angle_deg: float = 45.0,
             plate_level: float | None = None, plate_tol: float = 0.0) -> dict[str, float]:
    """Down-facing area whose normal is within (90° − angle) of −b (steeper than ``angle_deg`` from b), above the
    plate (face centre higher than plate_level + plate_tol)."""
    b = np.asarray(build_direction, float)
    b = b / np.linalg.norm(b)
    n = M.face_normals(V, F)
    a = M.area(V, F, per_face=True)
    hgt = V[F].mean(axis=1) @ b
    lvl = float((V @ b).min()) if plate_level is None else float(plate_level)
    down = n @ b < -math.cos(math.radians(angle_deg))
    above = hgt > lvl + plate_tol
    ov = down & above
    return {"area": float(a[ov].sum()), "fraction": float(a[ov].sum() / a.sum()), "plate_level": lvl,
            "plate_contact_area": float(a[down & ~above].sum()), "angle_deg": angle_deg}


def concave_radius_test(res: Any, r: float) -> dict[str, float]:
    """Closing of the finished shape by r: max depth of added material (length) and added volume fraction.

    A shape whose concave edges all have radius ≥ r is unchanged by the closing (depth ≈ 0 up to the grid error);
    a sharp 90° corner gains about 0.41 r. Material added outside Bound shapes or inside Cut shapes is ignored.
    """
    from toporlogy.post.primitives import Bound, Cut
    h = res.grid.h
    phi = S.redistance(res.phi, h)
    cl = S.closing(phi, r, h)
    X = res.grid.points()
    allowed = np.ones(res.grid.shape, bool)
    for ro in res.roles:
        s = ro.shape.sdf(X).reshape(res.grid.shape)
        if isinstance(ro, Bound):
            allowed &= s < -h
        elif isinstance(ro, Cut):
            allowed &= s > h
    added = np.where(allowed & (cl < 0), phi - cl, 0.0)
    depth = float(np.clip(added, 0, None).max())
    v0 = S.volume(phi, h)
    gain = float(S.partial_volume(cl, h)[allowed].sum() - S.partial_volume(phi, h)[allowed].sum()) * h ** 3
    return {"radius": r, "max_added_depth": depth, "added_volume_fraction": gain / v0, "tolerance": 0.5 * h}


def manufacturing_checks(res: Any, thresholds: CheckThresholds | None = None, **kw: Any) -> CheckResult:
    """All checks on a :class:`~toporlogy.post.pipeline.PostResult` (or anything with V, F, phi, grid, roles)."""
    th = thresholds or CheckThresholds(**kw)
    V, F = res.V, res.F
    items: list[dict[str, Any]] = []
    meas: dict[str, Any] = {}

    def add(name: str, value: Any, threshold: Any, passed: bool | None, note: str = "") -> None:
        items.append({"name": name, "value": value, "threshold": threshold, "passed": passed, "note": note})

    er = M.edge_report(F, len(V))
    meas["edges"] = er
    wt = er["open"] == 0 and er["non_manifold"] == 0 and er["misoriented"] == 0 and er["non_manifold_vertices"] == 0
    add("watertight_manifold", f"open {er['open']}, non-manifold {er['non_manifold']}, misoriented "
        f"{er['misoriented']}, nm-vertices {er['non_manifold_vertices']}", "all 0", wt)
    tp = M.mesh_topology(V, F)
    meas["topology"] = {k: v for k, v in tp.items() if k != "shell_list"}
    add("bodies", tp["bodies"], th.max_bodies, None if th.max_bodies is None else tp["bodies"] <= th.max_bodies)
    add("enclosed_voids", tp["cavities"], th.max_voids,
        None if th.max_voids is None else tp["cavities"] <= th.max_voids)
    h = res.grid.h
    phi_sd = S.redistance(res.phi, h)
    lt = sdf_local_thickness(phi_sd, h)
    lt.pop("field")
    rt = ray_thickness(V, F, phi_sd, res.grid)
    meas["wall_sphere"] = lt
    meas["wall_ray"] = rt
    if th.wall_method not in ("ray", "sphere"):
        raise ValueError("wall_method: ray | sphere")
    w = rt if th.wall_method == "ray" else lt
    wv = w.get(th.wall_statistic, float("nan"))
    add(f"min_wall_{th.wall_statistic}", wv, th.min_wall, None if th.min_wall is None else bool(wv >= th.min_wall),
        f"{th.wall_method} thickness; min {w.get('min', float('nan')):.4g}, p5 {w.get('p5', float('nan')):.4g}; "
        f"inscribed-ball p1 {lt.get('p1', float('nan')):.4g} (h_f = {h:g})")
    ov = overhang(V, F, th.build_direction, th.overhang_deg, th.plate_level, plate_tol=1.5 * h)
    meas["overhang"] = ov
    add("overhang_fraction", ov["fraction"], th.max_overhang_fraction,
        None if th.max_overhang_fraction is None else bool(ov["fraction"] <= th.max_overhang_fraction),
        f"area {ov['area']:.4g} steeper than {th.overhang_deg:g} deg above the plate")
    if th.min_concave_radius:
        cr = concave_radius_test(res, th.min_concave_radius)
        meas["concave_radius"] = cr
        add("concave_radius", cr["max_added_depth"], f"depth <= {cr['tolerance']:.3g} at r = {cr['radius']:g}",
            bool(cr["max_added_depth"] <= cr["tolerance"]), "closing test")
    meas["volume"] = M.volume(V, F)
    meas["surface_area"] = M.area(V, F)
    add("volume", meas["volume"], None, None)
    add("surface_area", meas["surface_area"], None, None)
    gated = [it["passed"] for it in items if it["passed"] is not None]
    return CheckResult(all(gated), items, meas, asdict(th))
