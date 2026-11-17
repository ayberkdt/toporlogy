"""Node selectors: planes, boxes, cylinders and bores, spheres, shapes, points, node ids.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from scipy.spatial import cKDTree
from toporlogy.domain import Domain
from toporlogy.geometry import Bore, Shape
from typing import Any

import numpy as np

from toporlogy.boundary.part import EmptySelectionError, PartInfo, part_info


# =============================================================================================== selectors
class Selector:
    """Picks nodes of the part. Combine with ``|`` (union), ``&`` (intersection), ``-`` (difference)."""

    surface: bool = False

    def mask(self, info: PartInfo) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def select(self, domain: Domain | PartInfo, allow_empty: bool = False) -> np.ndarray:
        """Sorted node numbers; raises :class:`EmptySelectionError` when nothing is selected."""
        info = domain if isinstance(domain, PartInfo) else part_info(domain)
        m = self.mask(info) & info.node_active
        out = np.flatnonzero(m)
        if not len(out) and not allow_empty:
            raise EmptySelectionError(f"{self!r} selected no node of the part (check coordinates, tolerances and "
                                      f"that the region touches the active elements; h = {info.h:g})")
        return out

    def _filter(self, info: PartInfo, m: np.ndarray) -> np.ndarray:
        return m & info.surface if self.surface else m

    def __or__(self, other: Selector) -> Selector:
        return _SetOp("|", self, other)

    def __and__(self, other: Selector) -> Selector:
        return _SetOp("&", self, other)

    def __sub__(self, other: Selector) -> Selector:
        return _SetOp("-", self, other)


class _SetOp(Selector):
    def __init__(self, op: str, a: Selector, b: Selector) -> None:
        self.op, self.a, self.b = op, a, b

    def mask(self, info: PartInfo) -> np.ndarray:
        ma, mb = self.a.mask(info), self.b.mask(info)
        return ma | mb if self.op == "|" else (ma & mb if self.op == "&" else ma & ~mb)

    def __repr__(self) -> str:
        return f"({self.a!r} {self.op} {self.b!r})"


class AllNodes(Selector):
    """Every node of the part."""

    def mask(self, info: PartInfo) -> np.ndarray:
        return info.node_active.copy()

    def __repr__(self) -> str:
        return "AllNodes()"


class SurfaceNodes(Selector):
    """Nodes on the boundary of the active elements."""

    surface = True

    def mask(self, info: PartInfo) -> np.ndarray:
        return info.surface.copy()

    def __repr__(self) -> str:
        return "SurfaceNodes()"


class InBox(Selector):
    """Nodes with lo ≤ x ≤ hi (bounds widened by ``tol``, default 1e-9·h)."""

    def __init__(self, lo: Sequence[float], hi: Sequence[float], surface: bool = False,
                 tol: float | None = None) -> None:
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.surface, self.tol = surface, tol

    def mask(self, info: PartInfo) -> np.ndarray:
        t = 1e-9 * info.h if self.tol is None else self.tol
        X = info.X
        return self._filter(info, np.all((self.lo - t <= X) & (self.hi + t >= X), axis=1))

    def __repr__(self) -> str:
        return f"InBox({self.lo.tolist()}, {self.hi.tolist()}, surface={self.surface})"


class HalfSpaceNodes(Selector):
    """Nodes with (x − point)·n ≤ tol."""

    def __init__(self, point: Sequence[float], normal: Sequence[float], surface: bool = False,
                 tol: float | None = None) -> None:
        self.point = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.normal = n / np.linalg.norm(n)
        self.surface, self.tol = surface, tol

    def mask(self, info: PartInfo) -> np.ndarray:
        t = 1e-9 * info.h if self.tol is None else self.tol
        return self._filter(info, (info.X - self.point) @ self.normal <= t)

    def __repr__(self) -> str:
        return f"HalfSpaceNodes({self.point.tolist()}, n={self.normal.round(4).tolist()})"


def _facing_ok(n: np.ndarray, ref: np.ndarray, facing: str | None, max_angle: float) -> np.ndarray:
    """Normal filter: 'out' n·ref ≥ cos, 'in' n·ref ≤ −cos, 'any' |n·ref| ≥ cos, None everything."""
    if facing is None:
        return np.ones(len(n), bool)
    c = np.einsum("ij,ij->i", n, ref) if ref.ndim == 2 else n @ ref
    cs = math.cos(math.radians(max_angle))
    if facing == "out":
        return c >= cs
    if facing == "in":
        return c <= -cs
    if facing == "any":
        return np.abs(c) >= cs
    raise ValueError("facing: 'out' | 'in' | 'any' | None")


class OnPlane(Selector):
    """Surface nodes within ``tol`` (default ½ h) of the plane through ``point`` with ``normal``.

    ``facing`` filters by the outward surface normal: "any" (default, normal (anti)parallel to the plane normal
    within ``max_angle``), "out" (part surface whose outward normal is +normal), "in", or None. For a plane that
    does not coincide with a node layer (partial volumes) pass ``tol=h``.
    """

    def __init__(self, point: Sequence[float], normal: Sequence[float] | int, tol: float | None = None,
                 facing: str | None = "any", max_angle: float = 60.0, surface: bool = True) -> None:
        self.point = np.asarray(point, float)
        if np.isscalar(normal):
            n = np.zeros(self.point.size)
            n[int(normal)] = 1.0
        else:
            n = np.asarray(normal, float)
        self.normal = n / np.linalg.norm(n)
        self.tol, self.facing, self.max_angle, self.surface = tol, facing, max_angle, surface

    def mask(self, info: PartInfo) -> np.ndarray:
        t = 0.5 * info.h if self.tol is None else self.tol
        m = np.abs((info.X - self.point) @ self.normal) <= t + 1e-9 * info.h
        m = self._filter(info, m)
        if self.surface and self.facing is not None:
            idx = np.flatnonzero(m)
            m[idx] = _facing_ok(info.normals[idx], self.normal, self.facing, self.max_angle)
        return m

    def __repr__(self) -> str:
        return f"OnPlane({self.point.tolist()}, n={self.normal.round(4).tolist()})"


def _cyl_coords(X: np.ndarray, p0: np.ndarray, axis: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(axial coordinate t, radius r, radial unit vector e_r) of points X about the axis (2-D: about a point)."""
    q = X - p0
    if axis is None:
        t = np.zeros(len(X))
        rv = q
    else:
        t = q @ axis
        rv = q - t[:, None] * axis
    r = np.linalg.norm(rv, axis=1)
    er = np.zeros_like(rv)
    ok = r > 0
    er[ok] = rv[ok] / r[ok, None]
    return t, r, er


class OnCylinder(Selector):
    """Surface nodes on a cylindrical face (3-D: axis p0 → p1; 2-D: circle about ``p0``, ``p1`` None).

    A node is taken when it is on the part surface, its distance from the axis lies in
    [radius − band_in, radius + band_out], its axial coordinate lies within the segment (± ``extend``), and its
    outward normal points toward the axis (``facing="inward"``: a bore, material outside) or away from it
    ("outward": a shaft or boss) within ``max_angle``. Defaults: bore band (1.5 h inside, 0.75 h outside) — wall
    nodes of partial-volume elements reach ≈ √2 h into the hole —, mirrored for "outward".
    The normal filter is what keeps the plane faces at the bore mouths out of the selection.
    """

    surface = True

    def __init__(self, p0: Sequence[float], p1: Sequence[float] | None, radius: float,
                 band: float | tuple[float, float] | None = None, facing: str | None = "inward",
                 max_angle: float = 60.0, extend: float = 0.0) -> None:
        self.p0 = np.asarray(p0, float)
        self.dim = self.p0.size
        if self.dim == 3:
            if p1 is None:
                raise ValueError("OnCylinder in 3-D needs both axis points p0, p1")
            d = np.asarray(p1, float) - self.p0
            self.length = float(np.linalg.norm(d))
            self.axis: np.ndarray | None = d / self.length
        else:
            self.length, self.axis = 0.0, None
        self.radius = float(radius)
        self.band, self.facing, self.max_angle, self.extend = band, facing, max_angle, float(extend)
        if facing not in ("inward", "outward", None):
            raise ValueError("facing: 'inward' | 'outward' | None")

    @classmethod
    def from_bore(cls, bore: Bore, **kw: Any) -> OnCylinder:
        if bore.axis is None:
            return cls(bore.center, None, bore.radius, **kw)
        p0, p1 = bore.endpoints()
        return cls(p0, p1, bore.radius, **kw)

    def bands(self, h: float) -> tuple[float, float]:
        if self.band is None:
            return (1.5 * h, 0.75 * h) if self.facing != "outward" else (0.75 * h, 1.5 * h)
        if np.isscalar(self.band):
            return float(self.band), float(self.band)
        return float(self.band[0]), float(self.band[1])

    def coords(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _cyl_coords(X, self.p0, self.axis)

    def mask(self, info: PartInfo) -> np.ndarray:
        bi, bo = self.bands(info.h)
        idx = np.flatnonzero(info.surface)
        t, r, er = self.coords(info.X[idx])
        ok = (r >= self.radius - bi) & (r <= self.radius + bo)
        if self.axis is not None:
            tol = self.extend + 1e-9 * info.h
            ok &= (t >= -tol) & (t <= self.length + tol)
        if self.facing is not None:
            n = info.normals[idx]
            ref = -er if self.facing == "inward" else er
            ok &= _facing_ok(n, ref, "out", self.max_angle)
        m = np.zeros(info.domain.nn, bool)
        m[idx[ok]] = True
        return m

    def __repr__(self) -> str:
        if self.axis is None:
            return f"OnCylinder(center={self.p0.tolist()}, r={self.radius:g}, {self.facing})"
        return f"OnCylinder({self.p0.tolist()} -> {(self.p0 + self.length * self.axis).tolist()}, " \
               f"r={self.radius:g}, {self.facing})"


def BoreFace(bore: Bore | Sequence[float], p1: Sequence[float] | None = None, radius: float | None = None,
             **kw: Any) -> OnCylinder:
    """Bore wall selector: ``BoreFace(bore)`` for a detected :class:`~toporlogy.geometry.Bore`, or
    ``BoreFace(p0, p1, radius)`` (2-D: ``BoreFace(center, None, radius)``)."""
    if isinstance(bore, Bore):
        return OnCylinder.from_bore(bore, facing="inward", **kw)
    if radius is None:
        raise ValueError("BoreFace(p0, p1, radius): radius missing")
    return OnCylinder(bore, p1, radius, facing="inward", **kw)


class InSphere(Selector):
    """Nodes within ``radius`` of ``center``."""

    def __init__(self, center: Sequence[float], radius: float, surface: bool = False) -> None:
        self.center, self.radius, self.surface = np.asarray(center, float), float(radius), surface

    def mask(self, info: PartInfo) -> np.ndarray:
        return self._filter(info, np.linalg.norm(info.X - self.center, axis=1) <= self.radius + 1e-9 * info.h)

    def __repr__(self) -> str:
        return f"InSphere({self.center.tolist()}, r={self.radius:g}, surface={self.surface})"


class InShape(Selector):
    """Nodes inside a :class:`~toporlogy.geometry.Shape` (sdf ≤ tol, default 1e-9·h)."""

    def __init__(self, shape: Shape, surface: bool = False, tol: float | None = None) -> None:
        self.shape, self.surface, self.tol = shape, surface, tol

    def mask(self, info: PartInfo) -> np.ndarray:
        t = 1e-9 * info.h if self.tol is None else self.tol
        return self._filter(info, self.shape.sdf(info.X) <= t)

    def __repr__(self) -> str:
        return f"InShape({self.shape!r})"


class AtPoints(Selector):
    """Nodes at the given coordinates (within ``tol``, default 1e-6·h); every point must hit a node of the part."""

    def __init__(self, points: Sequence[Sequence[float]], tol: float | None = None) -> None:
        self.points, self.tol = np.atleast_2d(np.asarray(points, float)), tol

    def mask(self, info: PartInfo) -> np.ndarray:
        t = 1e-6 * info.h if self.tol is None else self.tol
        idx = np.flatnonzero(info.node_active)
        tree = cKDTree(info.X[idx])
        d, k = tree.query(self.points)
        miss = np.flatnonzero(d > t)
        if len(miss):
            raise EmptySelectionError(f"AtPoints: no node of the part within {t:g} of {self.points[miss].tolist()} "
                                      f"(nearest distance {d[miss].round(6).tolist()}); use Nearest() instead")
        m = np.zeros(info.domain.nn, bool)
        m[idx[k]] = True
        return m

    def __repr__(self) -> str:
        return f"AtPoints({len(self.points)} point(s))"


class Nearest(Selector):
    """The ``k`` nearest nodes (surface nodes by default) to each point."""

    def __init__(self, points: Sequence[Sequence[float]], k: int = 1, surface: bool = True) -> None:
        self.points, self.k, self.surface = np.atleast_2d(np.asarray(points, float)), int(k), surface

    def mask(self, info: PartInfo) -> np.ndarray:
        pool = info.surface if self.surface else info.node_active
        idx = np.flatnonzero(pool)
        _, k = cKDTree(info.X[idx]).query(self.points, k=self.k)
        m = np.zeros(info.domain.nn, bool)
        m[idx[np.asarray(k).ravel()]] = True
        return m

    def __repr__(self) -> str:
        return f"Nearest({len(self.points)} point(s), k={self.k})"


class NodeIds(Selector):
    """Explicit node numbers (advanced; inactive nodes are reported by the checks)."""

    def __init__(self, ids: Sequence[int]) -> None:
        self.ids = np.unique(np.asarray(ids, np.int64))

    def mask(self, info: PartInfo) -> np.ndarray:
        m = np.zeros(info.domain.nn, bool)
        m[self.ids] = True
        return m

    def __repr__(self) -> str:
        return f"NodeIds({len(self.ids)})"


class Where(Selector):
    """Nodes where ``predicate(X)`` (X: (nn, dim) coordinates) is true."""

    def __init__(self, predicate: Callable[[np.ndarray], Any], surface: bool = False, label: str = "") -> None:
        self.predicate, self.surface, self.label = predicate, surface, label

    def mask(self, info: PartInfo) -> np.ndarray:
        return self._filter(info, np.asarray(self.predicate(info.X), bool))

    def __repr__(self) -> str:
        return f"Where({self.label or 'predicate'})"
