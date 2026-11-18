"""Exact implicit primitives and their roles in the finished part ("preserve" geometry).

Every primitive is a :class:`toporlogy.geometry.Shape` (the one base class of signed-distance solids in the library)
in 3-D: a signed distance ``sdf(X)`` (negative inside, ``X`` (n, 3) physical coordinates) that is exact (Euclidean) at
least near its surface. On top of a CSG shape it gives a unit gradient ``grad(X)``, the closest surface point
``project(X)`` and an exact ``offset(d)``, which the re-imposition and the vertex projection need.

A *role* says how the shape enters the finished part after smoothing (CSG on distance fields, φ = min for union, max
for intersection):

* :class:`Keep`  — material that must exist (boss, flange, pad): ``φ ← min(φ, s)``;
* :class:`Cut`   — material that must not exist (bore, clearance): ``φ ← max(φ, −s)``;
* :class:`Bound` — material only inside (design envelope, mounting plane): ``φ ← max(φ, s)``.

Order of re-imposition (:func:`assemble`): union with every Keep, then intersection with every Bound, then every Cut
subtracted. So holes are always open at their nominal size and the part never leaves its envelope, whatever the
smoothing did before. A role marked ``functional`` is a surface with a function (bore, mating face): its mesh vertices
are projected back onto the exact shape after every smoothing pass and its size/flatness is measured.

``stock`` (length) is the machining allowance of the *print* variant: a functional Cut is printed smaller (``s`` grown
by −stock, e.g. bore radius R − stock), a Keep and a planar Bound are printed larger (see
:func:`toporlogy.post.pipeline.print_variant`).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from toporlogy.geometry import Shape

__all__ = ["Primitive", "SpherePrimitive", "CylinderPrimitive", "PlanePrimitive", "BoxPrimitive", "ImplicitPrimitive",
           "GridPrimitive", "OffsetPrimitive", "Role", "Keep", "Cut", "Bound", "assemble", "oriented",
           "element_fraction"]


def _x(X: Any) -> np.ndarray:
    return np.atleast_2d(np.asarray(X, float))


class Primitive(Shape):
    """A 3-D :class:`~toporlogy.geometry.Shape` with an exact surface: gradient, closest point and offset."""

    dim = 3
    kind = "shape"

    def sdf(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def grad(self, X: np.ndarray) -> np.ndarray:
        """Unit outward gradient (central differences by default; exact in the subclasses)."""
        X = _x(X)
        e = 1e-6 * max(1.0, float(np.abs(X).max(initial=1.0)))
        g = np.stack([(self.sdf(X + e * np.eye(3)[k]) - self.sdf(X - e * np.eye(3)[k])) / (2 * e) for k in range(3)],
                     axis=1)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.where(n > 0, n, 1.0)

    def project(self, X: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Closest point on the surface (Newton steps x ← x − s ∇s; one step is exact for planes and spheres)."""
        P = _x(X).copy()
        for _ in range(iterations):
            P -= self.sdf(P)[:, None] * self.grad(P)
        return P

    def offset(self, d: float) -> Primitive:
        """The shape grown outward by ``d`` (shrunk for d < 0)."""
        return OffsetPrimitive(self, d)

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError(f"{type(self).__name__} has no bounding box (unbounded or not known)")


@dataclass
class OffsetPrimitive(Primitive):
    base: Primitive
    d: float
    kind = "offset"

    def sdf(self, X: np.ndarray) -> np.ndarray:
        return self.base.sdf(X) - self.d

    def grad(self, X: np.ndarray) -> np.ndarray:
        return self.base.grad(X)

    def describe(self) -> dict[str, Any]:
        return {"kind": "offset", "d": self.d, "base": self.base.describe()}

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.base.bbox()                       # grown by d > 0; for d < 0 the base box still bounds it
        g = max(float(self.d), 0.0)
        return lo - g, hi + g


@dataclass
class SpherePrimitive(Primitive):
    center: Sequence[float]
    radius: float
    kind = "sphere"

    def sdf(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(_x(X) - np.asarray(self.center, float), axis=1) - self.radius

    def grad(self, X: np.ndarray) -> np.ndarray:
        d = _x(X) - np.asarray(self.center, float)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.where(n > 0, n, 1.0)

    def offset(self, d: float) -> Primitive:
        return SpherePrimitive(self.center, self.radius + d)

    def describe(self) -> dict[str, Any]:
        return {"kind": "sphere", "center": list(map(float, self.center)), "radius": float(self.radius)}

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center, float)
        return c - self.radius, c + self.radius


@dataclass
class CylinderPrimitive(Primitive):
    """Circular cylinder of ``radius`` around the axis through ``center`` with direction ``axis``.

    ``length`` None = infinite (a through-bore); else capped, from center − length/2 to center + length/2 along the
    axis (exact SDF of a capped cylinder).
    """

    center: Sequence[float]
    axis: Sequence[float]
    radius: float
    length: float | None = None
    kind = "cylinder"

    def _frame(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(self.axis, float)
        a = a / np.linalg.norm(a)
        d = _x(X) - np.asarray(self.center, float)
        t = d @ a
        rv = d - t[:, None] * a
        return a, t, rv

    def radial(self, X: np.ndarray) -> np.ndarray:
        """Distance of the points from the axis."""
        return np.linalg.norm(self._frame(X)[2], axis=1)

    def axial(self, X: np.ndarray) -> np.ndarray:
        return self._frame(X)[1]

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact box of the capped cylinder: the axis segment, grown by radius · sqrt(1 − a_k²) per coordinate."""
        if self.length is None:
            raise ValueError("an infinite CylinderPrimitive (length None) has no bounding box")
        a = np.asarray(self.axis, float)
        a = a / np.linalg.norm(a)
        c = np.asarray(self.center, float)
        half = 0.5 * float(self.length) * np.abs(a) + float(self.radius) * np.sqrt(np.clip(1.0 - a * a, 0.0, None))
        return c - half, c + half

    def sdf(self, X: np.ndarray) -> np.ndarray:
        _, t, rv = self._frame(X)
        dr = np.linalg.norm(rv, axis=1) - self.radius
        if self.length is None:
            return dr
        da = np.abs(t) - 0.5 * self.length
        out = np.hypot(np.maximum(dr, 0.0), np.maximum(da, 0.0))
        return out + np.minimum(np.maximum(dr, da), 0.0)

    def grad(self, X: np.ndarray) -> np.ndarray:
        a, t, rv = self._frame(X)
        r = np.linalg.norm(rv, axis=1)
        er = rv / np.where(r > 0, r, 1.0)[:, None]
        if self.length is None:
            return er
        dr = r - self.radius
        da = np.abs(t) - 0.5 * self.length
        ea = np.sign(t)[:, None] * a
        g = np.where((dr >= da)[:, None], er, ea)
        both = (dr > 0) & (da > 0)
        if both.any():
            v = dr[both, None] * er[both] + da[both, None] * ea[both]
            g[both] = v / np.linalg.norm(v, axis=1, keepdims=True)
        return g

    def offset(self, d: float) -> Primitive:
        length = None if self.length is None else self.length + 2 * d
        return CylinderPrimitive(self.center, self.axis, self.radius + d, length)

    def describe(self) -> dict[str, Any]:
        return {"kind": "cylinder", "center": list(map(float, self.center)), "axis": list(map(float, self.axis)),
                "radius": float(self.radius), "length": None if self.length is None else float(self.length)}


@dataclass
class PlanePrimitive(Primitive):
    """Half-space ``n·(x − point) ≤ 0`` (material on the side opposite to the unit outward normal ``n``)."""

    point: Sequence[float] | np.ndarray
    normal: Sequence[float]
    kind = "plane"

    def _n(self) -> np.ndarray:
        n = np.asarray(self.normal, float)
        return n / np.linalg.norm(n)

    def sdf(self, X: np.ndarray) -> np.ndarray:
        return (_x(X) - np.asarray(self.point, float)) @ self._n()

    def grad(self, X: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._n(), _x(X).shape).copy()

    def offset(self, d: float) -> Primitive:
        return PlanePrimitive(np.asarray(self.point, float) + d * self._n(), self.normal)

    def describe(self) -> dict[str, Any]:
        return {"kind": "plane", "point": list(map(float, self.point)), "normal": list(map(float, self._n()))}


@dataclass
class BoxPrimitive(Primitive):
    """Axis-aligned box [lo, hi] (exact SDF)."""

    lo: Sequence[float] | np.ndarray
    hi: Sequence[float] | np.ndarray
    kind = "box"

    def sdf(self, X: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        c, e = 0.5 * (lo + hi), 0.5 * (hi - lo)
        q = np.abs(_x(X) - c) - e
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)

    def grad(self, X: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        c, e = 0.5 * (lo + hi), 0.5 * (hi - lo)
        d = _x(X) - c
        q = np.abs(d) - e
        s = np.sign(d)
        s[s == 0] = 1.0
        out = np.maximum(q, 0.0) * s
        n = np.linalg.norm(out, axis=1)
        inside = n == 0
        if inside.any():
            k = np.argmax(q[inside], axis=1)
            g = np.zeros((int(inside.sum()), 3))
            g[np.arange(len(k)), k] = s[inside][np.arange(len(k)), k]
            out[inside] = g
            n[inside] = 1.0
        return out / n[:, None]

    def offset(self, d: float) -> Primitive:
        if d <= 0:
            return BoxPrimitive(np.asarray(self.lo, float) - d, np.asarray(self.hi, float) + d)
        return OffsetPrimitive(self, d)                                   # rounded box: exact as an offset field

    def describe(self) -> dict[str, Any]:
        return {"kind": "box", "lo": list(map(float, self.lo)), "hi": list(map(float, self.hi))}

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.lo, float).copy(), np.asarray(self.hi, float).copy()


@dataclass
class ImplicitPrimitive(Primitive):
    """User signed distance ``fn(X) -> (n,)`` (should be a distance near its zero level)."""

    fn: Callable[[np.ndarray], np.ndarray]
    name: str = "implicit"
    kind = "implicit"

    def sdf(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(_x(X)), float).reshape(-1)

    def describe(self) -> dict[str, Any]:
        return {"kind": "implicit", "name": self.name}


class GridPrimitive(Primitive):
    """Primitive given by a sampled signed distance on a regular grid (from a keep/void mask or an envelope).

    ``phi`` holds values at the sample points ``origin + h·(i, j, k)``; evaluation is trilinear (outside the grid the
    nearest sample is used). Build it from a boolean mask with :func:`toporlogy.post.sdf.mask_shape`.
    """

    kind = "grid"

    def __init__(self, phi: np.ndarray, origin: Sequence[float] | np.ndarray, h: float, name: str = "grid") -> None:
        self.phi = np.asarray(phi, float)
        self.origin = np.asarray(origin, float)
        self.h = float(h)
        self.name = name

    def sdf(self, X: np.ndarray) -> np.ndarray:
        from scipy import ndimage
        c = ((_x(X) - self.origin) / self.h).T
        return ndimage.map_coordinates(self.phi, c, order=1, mode="nearest")

    def describe(self) -> dict[str, Any]:
        return {"kind": "grid", "name": self.name, "shape": list(self.phi.shape), "h": self.h}

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Samples with φ ≤ 0, grown by one spacing (the trilinear zero level lies within one cell of them). Outside
        the grid the nearest sample is used, so a solid sample on the grid border makes the solid unbounded."""
        inside = self.phi <= 0.0
        if any(np.take(inside, [0, -1], axis=ax).any() for ax in range(inside.ndim)):
            raise ValueError("GridPrimitive: solid samples on the grid border, the solid extends beyond the grid")
        if not inside.any():
            raise ValueError("GridPrimitive: no solid sample")
        ijk = np.argwhere(inside)
        return self.origin + self.h * (ijk.min(axis=0) - 1), self.origin + self.h * (ijk.max(axis=0) + 1)


# --------------------------------------------------------------------------------------------- roles
@dataclass
class Role:
    shape: Primitive
    functional: bool = False
    stock: float = 0.0
    name: str = ""
    role = "role"
    sign = 1.0                       # oriented field g = sign · s has the material inside (g < 0)

    def oriented(self, X: np.ndarray) -> np.ndarray:
        """g with g < 0 in the material side of this surface and ∇g = outward normal of the part on it."""
        return self.sign * self.shape.sdf(X)

    def oriented_grad(self, X: np.ndarray) -> np.ndarray:
        return self.sign * self.shape.grad(X)

    def describe(self) -> dict[str, Any]:
        return {"role": self.role, "name": self.name, "functional": self.functional, "stock": self.stock,
                "shape": self.shape.describe()}


@dataclass
class Keep(Role):
    """Material that must exist (union)."""

    role = "keep"
    sign = 1.0


@dataclass
class Cut(Role):
    """Material that must not exist (subtraction); functional by default (bores)."""

    functional: bool = True
    role = "cut"
    sign = -1.0


@dataclass
class Bound(Role):
    """Material only inside the shape (intersection); envelopes and mounting planes."""

    functional: bool = True
    role = "bound"
    sign = 1.0


def element_fraction(domain: Any, shape: Primitive | Callable[[np.ndarray], np.ndarray],
                     samples: int = 6) -> np.ndarray:
    """Solid fraction of every element of a regular-grid domain for the solid ``sdf < 0`` (``samples``³ points per
    element). Useful to build ``Domain.frac`` from an envelope primitive or a density of an analytic solid."""
    f = shape.sdf if isinstance(shape, Primitive) else shape
    n, h = tuple(domain.shape), float(domain.h)
    off = (np.arange(samples) + 0.5) / samples
    idx = np.indices(n).reshape(len(n), -1).T.astype(float)
    acc = np.zeros(len(idx))
    for a in off:
        for b in off:
            for c in off:
                acc += np.asarray(f(np.asarray(domain.origin, float) + h * (idx + [a, b, c]))) < 0
    return (acc / samples ** 3).reshape(n)


def oriented(role: Role, X: np.ndarray) -> np.ndarray:
    return role.oriented(X)


def assemble(phi_free: np.ndarray, roles: Sequence[Role], X: np.ndarray | None = None,
             cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Re-impose the exact geometry: ``max(max(min(φ, keeps), bounds), −cuts)`` (values at the points ``X``).

    ``cache`` maps ``id(role)`` to precomputed ``role.shape.sdf(X)`` arrays of ``phi_free``'s shape.
    """
    shp = np.shape(phi_free)

    def s(r: Role) -> np.ndarray:
        if cache is not None and id(r) in cache:
            return cache[id(r)]
        if X is None:
            raise ValueError("assemble needs X or a cache")
        return r.shape.sdf(X).reshape(shp)

    phi = np.asarray(phi_free, float)
    for r in roles:
        if isinstance(r, Keep):
            phi = np.minimum(phi, s(r))
    for r in roles:
        if isinstance(r, Bound):
            phi = np.maximum(phi, s(r))
    for r in roles:
        if isinstance(r, Cut):
            phi = np.maximum(phi, -s(r))
    return phi
