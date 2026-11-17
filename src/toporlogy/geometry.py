"""Domain construction from geometry: CSG of primitives, closed triangle meshes (STL), bore detection.

Shapes
------
Every shape is a *signed-distance-like* function ``sdf(P)``: negative inside, positive outside, zero on the surface
and 1-Lipschitz (|sdf(a) − sdf(b)| ≤ |a − b|). The primitives return the exact signed distance; the Booleans
(``a | b`` union = min, ``a & b`` intersection = max, ``a - b`` difference = max(a, −b)) return a bound with the exact
sign. 1-Lipschitz plus zero on the surface gives ``|sdf(x)| ≤ dist(x, surface)``, so a voxel whose centre has
``|sdf| > h·√dim / 2`` cannot be cut by the surface: :func:`coverage` sub-samples only the voxels near the surface.

* any dimension: :class:`Box`, :class:`Ball` (``Sphere`` / ``Circle``), :class:`Capsule` (2-D: rounded slot),
  :class:`HalfSpace`;
* 2-D: :class:`Polygon`;
* 3-D: :class:`Cylinder` (capped, arbitrary axis), :class:`Prism` (polygon extruded along x, y or z),
  :class:`Slot` (rounded slot extruded along an arbitrary vector).

``csg([("add", s1), ("cut", s2), ("intersect", s3), ...])`` applies operations in order.

Voxelisation
------------
:func:`voxelize` (shape) and :func:`voxelize_mesh` (closed triangle surface) return a :class:`~toporlogy.Domain`
whose ``frac`` is the solid volume fraction of every element (partial volume). Elements below ``min_frac`` are
dropped and, by default, only the largest *face-connected* component is kept: an element that touches the rest only
along an edge or at a corner forms a hinge (a mechanism, singular stiffness) — a frequent artefact of partial
volumes on curved surfaces.

Mesh coverage is exact along each ray: crossings of ``sub × sub`` rays per voxel column (along x, y and z) with the
triangles give the inside intervals; the covered length per voxel is integrated exactly, rays with an odd number of
crossings (open surface, ray through an edge) are dropped, and the three directions are pooled. The approach (ray
parity) follows the repository's earlier voxeliser, extended to exact interval integration.

Bores
-----
:func:`find_bores` finds axis-aligned round holes in a voxel domain (slice by slice, as the
repository's earlier hole detector) and estimates the radius and centre from the partial volumes
(sub-voxel), returning :class:`Bore` objects that the selectors and loads of :mod:`toporlogy.bc` accept.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from toporlogy.references import cite
from toporlogy.domain import Domain

__all__ = ["Shape", "Box", "Ball", "Sphere", "Circle", "Capsule", "HalfSpace", "Polygon", "Cylinder", "Prism",
           "Slot", "Union", "Intersection", "Difference", "csg", "grid_for", "coverage", "voxelize",
           "read_stl", "write_stl", "mesh_volume", "winding_number", "coverage_mesh", "voxelize_mesh",
           "Bore", "find_bores", "largest_component", "shape_from_function"]

_AX = {"x": 0, "y": 1, "z": 2}
_PLANE = {2: (0, 1), 1: (0, 2), 0: (1, 2)}         # in-plane coordinates of a prism along axis k


def _pts(P: Any, dim: int) -> np.ndarray:
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != dim:
        raise ValueError(f"points must have shape (n, {dim}), got {P.shape}")
    return P


def _extrude(d2: np.ndarray, t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Exact signed distance of an extrusion: in-plane distance ``d2``, axial coordinate ``t`` in [lo, hi]."""
    c, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    w1, w2 = d2, np.abs(t - c) - half
    return np.minimum(np.maximum(w1, w2), 0.0) + np.hypot(np.maximum(w1, 0.0), np.maximum(w2, 0.0))


def _seg_dist(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(P)) if L2 == 0 else np.clip((P - a) @ ab / L2, 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


class Shape:
    """Base class of signed-distance-like solids (see module notes)."""

    dim: int = 3

    def sdf(self, P: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def inside(self, P: Any) -> np.ndarray:
        """Membership of points ``P`` (n, dim) (surface counts as inside)."""
        return self.sdf(_pts(P, self.dim)) <= 0.0

    def __or__(self, other: Shape) -> Shape:
        return Union(self, other)

    def __and__(self, other: Shape) -> Shape:
        return Intersection(self, other)

    def __sub__(self, other: Shape) -> Shape:
        return Difference(self, other)


class Box(Shape):
    """Axis-aligned box ``lo ≤ x ≤ hi`` (2-D rectangle or 3-D box)."""

    def __init__(self, lo: Sequence[float], hi: Sequence[float]) -> None:
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        if self.lo.shape != self.hi.shape or self.lo.size not in (2, 3) or np.any(self.hi <= self.lo):
            raise ValueError("Box needs lo < hi with 2 or 3 coordinates")
        self.dim = self.lo.size

    def sdf(self, P: np.ndarray) -> np.ndarray:
        c, half = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        q = np.abs(P - c) - half
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lo.copy(), self.hi.copy()

    def __repr__(self) -> str:
        return f"Box({self.lo.tolist()}, {self.hi.tolist()})"


class Ball(Shape):
    """Sphere (3-D) or circle (2-D) of ``radius`` about ``center``."""

    def __init__(self, center: Sequence[float], radius: float) -> None:
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        if self.center.size not in (2, 3) or not self.radius > 0:
            raise ValueError("Ball needs a 2-D or 3-D centre and radius > 0")
        self.dim = self.center.size

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return np.linalg.norm(P - self.center, axis=1) - self.radius

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.center - self.radius, self.center + self.radius

    def __repr__(self) -> str:
        return f"Ball({self.center.tolist()}, r={self.radius:g})"


Sphere = Ball
Circle = Ball


class Capsule(Shape):
    """Points within ``radius`` of the segment p0–p1: a 2-D rounded slot (stadium) or a 3-D capsule."""

    def __init__(self, p0: Sequence[float], p1: Sequence[float], radius: float) -> None:
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.radius = float(radius)
        if self.p0.shape != self.p1.shape or self.p0.size not in (2, 3) or not self.radius > 0:
            raise ValueError("Capsule needs two 2-D or 3-D points and radius > 0")
        self.dim = self.p0.size

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return _seg_dist(P, self.p0, self.p1) - self.radius

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return np.minimum(self.p0, self.p1) - self.radius, np.maximum(self.p0, self.p1) + self.radius


class HalfSpace(Shape):
    """``(x − point)·n ≤ 0`` (inside is opposite to the normal). Unbounded: use it in an intersection or a cut."""

    def __init__(self, point: Sequence[float], normal: Sequence[float]) -> None:
        self.point = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.normal = n / np.linalg.norm(n)
        self.dim = self.point.size

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return (P - self.point) @ self.normal

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(self.dim, -np.inf), np.full(self.dim, np.inf)


def _polygon_sdf(Q: np.ndarray, V: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Exact signed distance of points Q (n, 2) to the closed polygon V (m, 2); sign by even–odd crossing."""
    A, B = V, np.roll(V, -1, axis=0)
    E = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", E, E), 1e-300)
    out = np.empty(len(Q))
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk, None, :]                                   # (c, 1, 2)
        w = q - A[None]                                               # (c, m, 2)
        t = np.clip(np.einsum("cmk,mk->cm", w, E) / L2, 0.0, 1.0)
        d = np.min(np.linalg.norm(w - t[..., None] * E[None], axis=2), axis=1)
        y = q[..., 1]
        cond = (A[None, :, 1] > y) != (B[None, :, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = A[None, :, 0] + (y - A[None, :, 1]) * E[None, :, 0] / E[None, :, 1]
        cross = cond & (q[..., 0] < xint)
        inside = (cross.sum(axis=1) % 2) == 1
        out[s:s + chunk] = np.where(inside, -d, d)
    return out


class Polygon(Shape):
    """2-D simple polygon (vertices in order, either orientation; not closed explicitly)."""

    dim = 2

    def __init__(self, vertices: Sequence[Sequence[float]]) -> None:
        self.V = np.asarray(vertices, float)
        if self.V.ndim != 2 or self.V.shape[1] != 2 or len(self.V) < 3:
            raise ValueError("Polygon needs at least 3 (x, y) vertices")
        if np.allclose(self.V[0], self.V[-1]):
            self.V = self.V[:-1]

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return _polygon_sdf(P, self.V)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)


class Cylinder(Shape):
    """Capped circular cylinder with axis p0 → p1 (any direction) and ``radius``."""

    dim = 3

    def __init__(self, p0: Sequence[float], p1: Sequence[float], radius: float) -> None:
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.radius = float(radius)
        d = self.p1 - self.p0
        self.length = float(np.linalg.norm(d))
        if self.p0.size != 3 or self.length == 0 or not self.radius > 0:
            raise ValueError("Cylinder needs two distinct 3-D points and radius > 0")
        self.axis = d / self.length

    def sdf(self, P: np.ndarray) -> np.ndarray:
        q = P - self.p0
        t = q @ self.axis
        rad = np.linalg.norm(q - t[:, None] * self.axis, axis=1)
        return _extrude(rad - self.radius, t, 0.0, self.length)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        ext = self.radius * np.sqrt(np.clip(1.0 - self.axis ** 2, 0.0, 1.0))   # extent of the end discs
        return np.minimum(self.p0, self.p1) - ext, np.maximum(self.p0, self.p1) + ext

    def __repr__(self) -> str:
        return f"Cylinder({self.p0.tolist()} -> {self.p1.tolist()}, r={self.radius:g})"


class Prism(Shape):
    """Polygon extruded along a coordinate axis over ``range`` = (lo, hi).

    The polygon coordinates are (x, y) for axis "z", (x, z) for axis "y" and (y, z) for axis "x".
    """

    dim = 3

    def __init__(self, polygon: Sequence[Sequence[float]], axis: str | int = "z",
                 range: tuple[float, float] = (0.0, 1.0)) -> None:  # noqa: A002 - geometric range
        self.ax = _AX[axis] if isinstance(axis, str) else int(axis)
        self.poly = Polygon(polygon)
        self.lo, self.hi = float(range[0]), float(range[1])
        if not self.hi > self.lo:
            raise ValueError("Prism range must have hi > lo")

    def sdf(self, P: np.ndarray) -> np.ndarray:
        i, j = _PLANE[self.ax]
        return _extrude(self.poly.sdf(P[:, [i, j]]), P[:, self.ax], self.lo, self.hi)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        i, j = _PLANE[self.ax]
        a, b = self.poly.bbox()
        lo, hi = np.zeros(3), np.zeros(3)
        lo[[i, j]], hi[[i, j]] = a, b
        lo[self.ax], hi[self.ax] = self.lo, self.hi
        return lo, hi


class Slot(Shape):
    """Rounded slot: the 2-D stadium of half-width ``radius`` about p0–p1, extruded from there along ``depth``.

    ``depth`` is a vector (direction and length of the extrusion); p1 − p0 should be normal to it (its component
    along ``depth`` is ignored).
    """

    dim = 3

    def __init__(self, p0: Sequence[float], p1: Sequence[float], radius: float, depth: Sequence[float]) -> None:
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        dv = np.asarray(depth, float)
        self.length = float(np.linalg.norm(dv))
        if self.length == 0 or not radius > 0:
            raise ValueError("Slot needs a non-zero depth vector and radius > 0")
        self.n = dv / self.length
        self.radius = float(radius)
        e = self.p1 - self.p0
        self.e = e - (e @ self.n) * self.n

    def sdf(self, P: np.ndarray) -> np.ndarray:
        q = P - self.p0
        t = q @ self.n
        qp = q - t[:, None] * self.n
        d2 = _seg_dist(qp, np.zeros(3), self.e) - self.radius
        return _extrude(d2, t, 0.0, self.length)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        ends = np.array([self.p0, self.p0 + self.e, self.p0 + self.length * self.n,
                         self.p0 + self.e + self.length * self.n])
        return ends.min(axis=0) - self.radius, ends.max(axis=0) + self.radius


def _common_dim(shapes: Sequence[Shape], what: str) -> int:
    dims = {s.dim for s in shapes}
    if len(dims) != 1:
        raise ValueError(f"{what} of shapes of different dimension: {sorted(dims)}")
    return dims.pop()


class Union(Shape):
    def __init__(self, *shapes: Shape) -> None:
        self.shapes = list(shapes)
        self.dim = _common_dim(self.shapes, "Union")

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return np.min([s.sdf(P) for s in self.shapes], axis=0)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        bb = [s.bbox() for s in self.shapes]
        return np.min([b[0] for b in bb], axis=0), np.max([b[1] for b in bb], axis=0)


class Intersection(Shape):
    def __init__(self, *shapes: Shape) -> None:
        self.shapes = list(shapes)
        self.dim = _common_dim(self.shapes, "Intersection")

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return np.max([s.sdf(P) for s in self.shapes], axis=0)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        bb = [s.bbox() for s in self.shapes]
        return np.max([b[0] for b in bb], axis=0), np.min([b[1] for b in bb], axis=0)


class Difference(Shape):
    def __init__(self, a: Shape, b: Shape) -> None:
        if a.dim != b.dim:
            raise ValueError("Difference of shapes of different dimension")
        self.a, self.b = a, b
        self.dim = a.dim

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return np.maximum(self.a.sdf(P), -self.b.sdf(P))

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.a.bbox()


def csg(operations: Sequence[tuple[str, Shape]]) -> Shape:
    """Apply ("add" | "cut" | "intersect", shape) in order; the first operation must be "add"."""
    if not operations or operations[0][0] != "add":
        raise ValueError("csg: the first operation must be ('add', shape)")
    out = operations[0][1]
    for op, s in operations[1:]:
        if s.dim != out.dim:
            raise ValueError("csg: all shapes must have the same dimension")
        if op == "add":
            out = Union(out, s)
        elif op == "cut":
            out = Difference(out, s)
        elif op == "intersect":
            out = Intersection(out, s)
        else:
            raise ValueError(f"csg: unknown operation {op!r} (add, cut, intersect)")
    return out


# --------------------------------------------------------------------------------------- voxelisation
def grid_for(lo: Sequence[float], hi: Sequence[float], h: float,
             padding: int = 0) -> tuple[tuple[int, ...], np.ndarray]:
    """(shape, origin) of the regular grid of edge ``h`` that covers [lo, hi] plus ``padding`` empty layers."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("unbounded geometry: pass bounds=(lo, hi)")
    n = np.maximum(np.ceil((hi - lo) / h - 1e-9).astype(int), 1) + 2 * int(padding)
    return tuple(int(v) for v in n), lo - padding * h


def _element_centres(shape: tuple[int, ...], h: float, origin: np.ndarray) -> np.ndarray:
    idx = np.indices(shape).reshape(len(shape), -1).T
    return origin + h * (idx + 0.5)


def coverage(shape: Shape, grid: Domain | tuple[tuple[int, ...], float, Sequence[float]], sub: int = 4,
             chunk: int = 20_000) -> np.ndarray:
    """Solid fraction of every element of ``grid`` (a Domain or (shape, h, origin)) inside ``shape``.

    Voxels whose centre is farther than half a diagonal from the surface (by the sdf bound) are 0 or 1 exactly;
    the others are sampled with ``sub``^dim points (error ≲ 1/sub per cut voxel, much less on average).
    """
    if isinstance(grid, Domain):
        gshape, h, origin = grid.shape, grid.h, grid.origin
    else:
        gshape, h, origin = tuple(grid[0]), float(grid[1]), np.asarray(grid[2], float)
    dim = len(gshape)
    if dim != shape.dim:
        raise ValueError(f"shape is {shape.dim}-D, grid {dim}-D")
    C = _element_centres(gshape, h, np.asarray(origin, float))
    d = shape.sdf(C)
    r = 0.5 * h * math.sqrt(dim)
    frac = (d <= -r).astype(float)
    cut = np.flatnonzero(np.abs(d) < r)
    off = (np.arange(sub) + 0.5) / sub - 0.5
    O = np.stack(np.meshgrid(*([off] * dim), indexing="ij"), axis=-1).reshape(-1, dim) * h   # (sub^dim, dim)
    per = max(1, chunk // len(O))
    for s in range(0, len(cut), per):
        idx = cut[s:s + per]
        P = (C[idx, None, :] + O[None]).reshape(-1, dim)
        frac[idx] = (shape.sdf(P) <= 0.0).reshape(len(idx), -1).mean(axis=1)
    return frac.reshape(gshape)


def largest_component(active: np.ndarray) -> np.ndarray:
    """Mask of the largest face-connected component of ``active`` (2-D edges / 3-D faces shared)."""
    lab, n = ndimage.label(active)
    if n <= 1:
        return active.copy()
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _finish(frac: np.ndarray, h: float, origin: np.ndarray, min_frac: float, connectivity: str | None,
            keep: Any, void: Any, thickness: float) -> Domain:
    frac = np.where(frac >= min_frac, frac, 0.0)
    if connectivity == "largest":
        frac = np.where(largest_component(frac > 0), frac, 0.0)
    elif connectivity not in (None, "all"):
        raise ValueError("connectivity: 'largest' | 'all' | None")
    g = (frac.shape, h, origin)
    masks = []
    for m in (keep, void):
        if isinstance(m, Shape):
            masks.append(coverage(m, g) > 0.5)
        else:
            masks.append(m)
    return Domain(frac.shape, h, origin, frac=frac, keep=masks[0], void=masks[1], thickness=thickness)


def voxelize(shape: Shape, h: float, padding: int = 0, bounds: tuple[Sequence[float], Sequence[float]] | None = None,
             sub: int = 4, min_frac: float = 0.05, connectivity: str | None = "largest", keep: Any = None,
             void: Any = None, thickness: float = 1.0) -> Domain:
    """Domain on the grid of edge ``h`` covering ``shape`` (or ``bounds``) with partial-volume ``frac``.

    ``keep`` / ``void``: bool masks or shapes (elements more than half covered). ``min_frac``: elements with a
    smaller solid fraction are removed (near-empty slivers only add ill-conditioning). ``connectivity="largest"``
    keeps the largest face-connected component (see module notes).
    """
    lo, hi = bounds if bounds is not None else shape.bbox()
    gshape, origin = grid_for(lo, hi, h, padding)
    frac = coverage(shape, (gshape, h, origin), sub=sub)
    return _finish(frac, h, origin, min_frac, connectivity, keep, void, thickness)


# --------------------------------------------------------------------------------------- triangle meshes
@cite("3dsystems_1989", note="binary STL: 80-byte header, uint32 count, 50 bytes per facet; ASCII 'facet/vertex' "
                             "(layout from memory, specification not read in this session)")
def read_stl(path: str | Path, merge_tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (nv, 3) and triangles (nf, 3) of a binary or ASCII STL file; coincident vertices merged.

    ``merge_tol`` is relative to the bounding-box diagonal.
    """
    raw = Path(path).read_bytes()
    tri: np.ndarray | None = None
    # binary when the records fit in the file and it does not read as ASCII ("solid" + "facet"); bytes after the
    # last record (padding some exporters add) are ignored. A binary header may itself start with "solid".
    looks_ascii = raw[:512].lstrip().lower().startswith(b"solid") and b"facet" in raw[:1024].lower()
    if len(raw) >= 84 and not looks_ascii:
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50 * n <= len(raw):
            rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tri = np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"].astype(float)
    if tri is None:
        txt = raw.decode("ascii", errors="replace")
        vals = [line.split()[1:4] for line in txt.splitlines() if line.strip().startswith("vertex")]
        if not vals or len(vals) % 3:
            raise ValueError(f"{path}: neither a binary STL nor an ASCII STL with complete facets")
        tri = np.asarray(vals, float).reshape(-1, 3, 3)
    P = tri.reshape(-1, 3)
    scale = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
    key = np.round(P / (merge_tol * scale)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = P[first]
    F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]   # drop degenerate facets
    return V, F


def write_stl(path: str | Path, V: np.ndarray, F: np.ndarray) -> Path:
    """Write a binary STL (normals from the vertex order)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    T = V[F]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, T
    p = Path(path)
    with p.open("wb") as fh:
        fh.write(b"binary STL written by toporlogy.geometry".ljust(80, b" "))
        fh.write(struct.pack("<I", len(F)))
        fh.write(rec.tobytes())
    return p


def mesh_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Enclosed volume of a closed, consistently oriented triangle surface (|Σ signed tetrahedra|)."""
    T = np.asarray(V, float)[np.asarray(F, np.int64)]
    return abs(float(np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum()) / 6.0)


@cite("vanoosterom_strackee_1983", note="solid angle of a triangle, tan(Ω/2) = a·(b×c) / (abc + (a·b)c + (a·c)b "
                                        "+ (b·c)a) (formula from memory, paper not read in this session)")
def winding_number(points: np.ndarray, V: np.ndarray, F: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Generalised winding number of ``points`` w.r.t. the triangle surface (≈1 inside, ≈0 outside).

    Robust against small holes and inconsistent orientation is NOT given (orientation must be consistent);
    cost O(points × triangles), use it for modest point sets (selectors, checks), not for voxelisation.
    """
    P = np.asarray(points, float)
    T = np.asarray(V, float)[np.asarray(F, np.int64)]
    out = np.zeros(len(P))
    per = max(1, chunk // max(len(T), 1))
    for s in range(0, len(P), per):
        q = P[s:s + per, None, None, :]
        R = T[None] - q                                       # (c, m, 3, 3)
        a, b, c = R[:, :, 0], R[:, :, 1], R[:, :, 2]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = np.einsum("cmk,cmk->cm", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("cmk,cmk->cm", a, b) * lc + np.einsum("cmk,cmk->cm", a, c) * lb
               + np.einsum("cmk,cmk->cm", b, c) * la)
        out[s:s + per] = 2.0 * np.arctan2(num, den).sum(axis=1) / (4.0 * np.pi)
    return out


def _ray_hits(T: np.ndarray, ax: int, a0: float, b0: float, hf: float, na: int, nb: int,
              chunk: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    """(ray id, axial coordinate) of every crossing of the ray grid along ``ax`` with the triangles T (m, 3, 3).

    Rays sit at (a0 + (A + ½)·hf, b0 + (B + ½)·hf) in the plane normal to ``ax``; ray id = A·nb + B.
    """
    i, j = _PLANE[ax]
    ta, tb, tc = T[:, :, i], T[:, :, j], T[:, :, ax]
    # a tiny irrational shift keeps rays off vertices and edges of grid-aligned meshes
    sa, sb = a0 + (0.5 + 3.71390676e-7) * hf, b0 + (0.5 + 6.18033989e-7) * hf
    ia0 = np.clip(np.ceil((ta.min(1) - sa) / hf), 0, na).astype(np.int64)
    ia1 = np.clip(np.floor((ta.max(1) - sa) / hf), -1, na - 1).astype(np.int64)
    ib0 = np.clip(np.ceil((tb.min(1) - sb) / hf), 0, nb).astype(np.int64)
    ib1 = np.clip(np.floor((tb.max(1) - sb) / hf), -1, nb - 1).astype(np.int64)
    ca, cb = np.maximum(ia1 - ia0 + 1, 0), np.maximum(ib1 - ib0 + 1, 0)
    cnt = ca * cb
    det = (tb[:, 1] - tb[:, 2]) * (ta[:, 0] - ta[:, 2]) + (ta[:, 2] - ta[:, 1]) * (tb[:, 0] - tb[:, 2])
    good = np.flatnonzero((cnt > 0) & (np.abs(det) > 1e-300))
    rays, zs = [], []
    start = 0
    while start < len(good):                                   # chunks of triangles with bounded pair count
        csum = np.cumsum(cnt[good[start:]])
        stop = start + max(1, int(np.searchsorted(csum, chunk)))
        t = good[start:stop]
        start = stop
        k = cnt[t]
        tt = np.repeat(t, k)
        loc = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
        A = ia0[tt] + loc // cb[tt]
        B = ib0[tt] + loc % cb[tt]
        pa, pb = sa + A * hf, sb + B * hf
        x1, x2, x3 = ta[tt, 0], ta[tt, 1], ta[tt, 2]
        y1, y2, y3 = tb[tt, 0], tb[tt, 1], tb[tt, 2]
        dd = det[tt]
        l1 = ((y2 - y3) * (pa - x3) + (x3 - x2) * (pb - y3)) / dd
        l2 = ((y3 - y1) * (pa - x3) + (x1 - x3) * (pb - y3)) / dd
        l3 = 1.0 - l1 - l2
        ok = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        z = l1 * tc[tt, 0] + l2 * tc[tt, 1] + l3 * tc[tt, 2]
        rays.append((A * nb + B)[ok])
        zs.append(z[ok])
    if not rays:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(rays), np.concatenate(zs)


def coverage_mesh(V: np.ndarray, F: np.ndarray, gshape: Sequence[int], h: float, origin: Sequence[float],
                  sub: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    """Solid fraction of every voxel inside a closed triangle surface (exact along rays, see module notes).

    Returns (frac, info) with info = rays used / dropped per axis.
    """
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    T = V[F]
    n = np.asarray(gshape, int)
    origin = np.asarray(origin, float)
    hf = h / sub
    covered = np.zeros(tuple(n))
    weight = np.zeros(tuple(n))
    info: dict[str, Any] = {}
    for ax in range(3):
        i, j = _PLANE[ax]
        na, nb = sub * n[i], sub * n[j]
        ray, z = _ray_hits(T, ax, origin[i], origin[j], hf, na, nb)
        order = np.lexsort((z, ray))
        ray, z = ray[order], z[order]
        # coincident crossings on one ray (ray through a shared edge) are merged when the count is odd
        starts = np.flatnonzero(np.r_[True, ray[1:] != ray[:-1]])
        cnts = np.diff(np.r_[starts, len(ray)])
        odd = np.flatnonzero(cnts % 2 == 1)
        drop = np.zeros(len(ray), bool)
        bad_rays: list[int] = []
        for k in odd:
            s, e = starts[k], starts[k] + cnts[k]
            dup = np.r_[False, np.diff(z[s:e]) <= 1e-9 * h]
            if (cnts[k] - dup.sum()) % 2 == 0:
                drop[s:e] |= dup
            else:
                bad_rays.append(int(ray[s]))
        ray, z = ray[~drop], z[~drop]
        bad = np.zeros(na * nb, bool)
        bad[bad_rays] = True
        keep = ~bad[ray]
        ray, z = ray[keep], z[keep]
        u = (z - origin[ax]) / h                                   # axial coordinate in voxel units
        enter, leave, rid = u[0::2], u[1::2], ray[0::2]
        if len(ray) % 2:
            raise RuntimeError("internal: odd number of crossings after filtering")
        lo = np.clip(enter, 0.0, n[ax])
        hi = np.clip(leave, 0.0, n[ax])
        m = hi > lo
        lo, hi, rid = lo[m], hi[m], rid[m]
        A, B = rid // nb, rid % nb
        col = (A // sub) * n[j] + (B // sub)                      # voxel column (i, j)
        cov = np.zeros((n[i] * n[j], n[ax] + 1))
        ka = np.minimum(np.floor(lo).astype(np.int64), n[ax] - 1)
        kb = np.minimum(np.floor(hi).astype(np.int64), n[ax] - 1)
        same = ka == kb
        np.add.at(cov, (col[same], ka[same]), hi[same] - lo[same])
        d = ~same
        np.add.at(cov, (col[d], ka[d]), ka[d] + 1.0 - lo[d])
        np.add.at(cov, (col[d], kb[d]), hi[d] - kb[d])
        full = np.zeros_like(cov)                                  # whole voxels strictly between ka and kb
        np.add.at(full, (col[d], ka[d] + 1), 1.0)
        np.add.at(full, (col[d], kb[d]), -1.0)
        cov += np.cumsum(full, axis=1)
        cov = cov[:, :n[ax]].reshape(n[i], n[j], n[ax])
        good_rays = (~bad).reshape(n[i], sub, n[j], sub).sum(axis=(1, 3)).astype(float)   # per voxel column
        covered += np.moveaxis(cov, [0, 1, 2], [i, j, ax])
        weight += np.moveaxis(np.broadcast_to(good_rays[:, :, None], cov.shape), [0, 1, 2], [i, j, ax])
        info["xyz"[ax]] = {"rays": int(na * nb), "dropped": int(bad.sum())}
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(weight > 0, covered / weight, 0.0)
    return np.clip(frac, 0.0, 1.0), info


def voxelize_mesh(V: np.ndarray, F: np.ndarray, h: float, padding: int = 1, sub: int = 3, min_frac: float = 0.05,
                  connectivity: str | None = "largest", keep: Any = None, void: Any = None,
                  bounds: tuple[Sequence[float], Sequence[float]] | None = None) -> Domain:
    """Domain with partial volumes of a closed triangle surface (vertices in the length unit of ``h``)."""
    V = np.asarray(V, float)
    lo, hi = bounds if bounds is not None else (V.min(axis=0), V.max(axis=0))
    gshape, origin = grid_for(lo, hi, h, padding)
    frac, info = coverage_mesh(V, F, gshape, h, origin, sub=sub)
    dom = _finish(frac, h, origin, min_frac, connectivity, keep, void, 1.0)
    dom.voxelization_info = info          # rays used / dropped per axis (diagnostic)
    return dom


# --------------------------------------------------------------------------------------- bores
@dataclass
class Bore:
    """A round hole.

    3-D: ``axis`` is a unit coordinate vector (detection is axis-aligned), ``center`` the midpoint of the hole's
    axis, ``segments`` the intervals of the axis coordinate where material surrounds the hole (several for a clevis).
    2-D: ``axis`` None, ``segments`` empty (a circular hole in the plane)."""

    center: np.ndarray
    radius: float
    axis: np.ndarray | None = None
    segments: list[tuple[float, float]] = field(default_factory=list)
    roundness: float = 0.0
    id: str = ""

    @property
    def dim(self) -> int:
        return len(self.center)

    @property
    def span(self) -> tuple[float, float]:
        """(lo, hi) along the axis (3-D), over all segments."""
        return min(a for a, _ in self.segments), max(b for _, b in self.segments)

    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis end points p0, p1 (3-D)."""
        if self.axis is None:
            raise ValueError("2-D bore has no axis end points")
        k = int(np.argmax(np.abs(self.axis)))
        lo, hi = self.span
        p0, p1 = self.center.copy(), self.center.copy()
        p0[k], p1[k] = lo, hi
        return p0, p1

    def shape(self, extra_radius: float = 0.0, extend: float = 0.0) -> Shape:
        """The hole (optionally enlarged) as a Cylinder / Circle shape."""
        if self.axis is None:
            return Ball(self.center, self.radius + extra_radius)
        p0, p1 = self.endpoints()
        return Cylinder(p0 - extend * self.axis, p1 + extend * self.axis, self.radius + extra_radius)


def _sections(sl: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Enclosed void regions of a 2-D slice (masks)."""
    lab, n = ndimage.label(~sl)
    if n == 0:
        return []
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    out = []
    for k in range(1, n + 1):
        if k not in border:
            m = lab == k
            if m.sum() >= min_area:
                out.append(m)
    return out


def _section_props(mask: np.ndarray, frac2: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(centre in pixel units, radius in pixels, roundness) of a void section using partial volumes."""
    ring = ndimage.binary_dilation(mask) & ~mask
    w = np.where(mask | ring, 1.0 - frac2, 0.0)                     # void fraction of the section and its wall
    pix = np.argwhere(w > 0)
    ww = w[w > 0]
    c = (pix * ww[:, None]).sum(axis=0) / ww.sum()
    r = math.sqrt(ww.sum() / math.pi)
    edge = mask & ~ndimage.binary_erosion(mask)
    bp = np.argwhere(edge)
    if len(bp) > 3:
        rad = np.linalg.norm(bp - c, axis=1) + 0.5
        spread = float(np.std(rad) / max(np.mean(rad), 1e-9))
    else:
        spread = 1.0
    pm = np.argwhere(mask)
    bb = pm.max(0) - pm.min(0) + 1
    fill = len(pm) / (math.pi / 4 * bb[0] * bb[1])
    return c, r, spread if fill >= 0.6 else 1.0


def find_bores(domain: Domain, min_diameter: float | None = None, max_roundness: float = 0.35,
               min_length: float | None = None) -> list[Bore]:
    """Axis-aligned round holes of the part (enclosed void regions in grid slices).

    Lengths in the domain's unit; defaults: ``min_diameter`` = 3 h, ``min_length`` = 2 h. Radius and centre come
    from the partial volumes of the wall voxels (sub-voxel estimate; ≈ 0.1 h at a few voxels per radius).
    """
    h = domain.h
    dmin = 3.0 * h if min_diameter is None else float(min_diameter)
    lmin = 2 if min_length is None else max(1, int(round(min_length / h)))
    min_area = max(1, int(math.floor(math.pi * (0.5 * dmin / h) ** 2)))
    frac = domain.frac
    solid = frac > 0.5
    if domain.dim == 2:
        out = []
        for m in _sections(solid, min_area):
            c, r, spread = _section_props(m, frac)
            if spread <= max_roundness:
                out.append(Bore(domain.origin + h * (c + 0.5), r * h, None, [], spread))
        for k, b in enumerate(out, 1):
            b.id = f"B{k}"
        return out
    tubes: list[dict[str, Any]] = []
    for ax in range(3):
        sm, fm = np.moveaxis(solid, ax, 0), np.moveaxis(frac, ax, 0)
        prev: list[dict[str, Any]] = []
        for k in range(sm.shape[0]):
            cur = []
            for mask in _sections(sm[k], min_area):
                hit = None
                for t in prev:
                    if (t["mask"] & mask).sum() > 0.5 * min(mask.sum(), t["mask"].sum()):
                        hit = t
                        break
                if hit is None:
                    hit = {"ax": ax, "k0": k, "k1": k, "r": [], "c": [], "spread": []}
                    tubes.append(hit)
                c, r, spread = _section_props(mask, fm[k])
                hit.update(k1=k, mask=mask)
                hit["r"].append(r)
                hit["c"].append(c)
                hit["spread"].append(spread)
                cur.append(hit)
            prev = cur
    bores: list[Bore] = []
    for t in tubes:
        if len(t["r"]) < lmin or float(np.median(t["spread"])) > max_roundness:
            continue
        ax = t["ax"]
        i, j = _PLANE[ax]
        c2 = np.mean(t["c"], axis=0)
        center = np.zeros(3)
        center[i] = domain.origin[i] + (c2[0] + 0.5) * h
        center[j] = domain.origin[j] + (c2[1] + 0.5) * h
        lo, hi = domain.origin[ax] + t["k0"] * h, domain.origin[ax] + (t["k1"] + 1) * h
        center[ax] = 0.5 * (lo + hi)
        bores.append(Bore(center, float(np.median(t["r"])) * h, np.eye(3)[ax], [(lo, hi)],
                          float(np.median(t["spread"]))))
    merged: list[Bore] = []                    # collinear tubes (clevis arms) → one bore with several segments
    for b in sorted(bores, key=lambda q: (int(np.argmax(q.axis)), q.segments[0][0])):
        ax = int(np.argmax(b.axis))
        for m in merged:
            if int(np.argmax(m.axis)) != ax:
                continue
            off = np.delete(m.center - b.center, ax)
            if np.linalg.norm(off) < 0.5 * max(m.radius, b.radius) and abs(m.radius - b.radius) < 0.25 * m.radius:
                m.segments.append(b.segments[0])
                lo, hi = m.span
                m.center[ax] = 0.5 * (lo + hi)
                break
        else:
            merged.append(b)
    merged.sort(key=lambda q: (int(np.argmax(q.axis)), *np.round(q.center, 9)))
    for k, b in enumerate(merged, 1):
        b.id = f"B{k}"
    return merged


def shape_from_function(fn: Callable[[np.ndarray], np.ndarray], lo: Sequence[float], hi: Sequence[float]) -> Shape:
    """Wrap a user signed-distance-like function (must be 1-Lipschitz for :func:`coverage` to be exact)."""
    class _Fn(Shape):
        def __init__(self) -> None:
            self.dim = len(lo)

        def sdf(self, P: np.ndarray) -> np.ndarray:
            return np.asarray(fn(P), float)

        def bbox(self) -> tuple[np.ndarray, np.ndarray]:
            return np.asarray(lo, float), np.asarray(hi, float)
    return _Fn()
