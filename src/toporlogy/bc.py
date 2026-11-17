"""Boundary conditions by geometry: node selectors, supports, loads, load cases, non-design pads and checks.

Users never handle DOF numbers. A *selector* picks nodes of the part by geometry; a *support* constrains selected
nodes; a *load* turns an intended resultant into consistent nodal forces; :class:`BoundaryConditions` collects
named load cases (each may have its own supports), builds ``F`` (ndof, n_cases) and the physics, and checks
everything (:meth:`BoundaryConditions.check`, :meth:`BoundaryConditions.summary`).

Part surface and normals
------------------------
A node is on the surface when it touches at least one but fewer than 2^dim active elements (boundary of the active
elements; the box boundary counts). Exposed element faces (active element, inactive neighbour) carry area
``h²`` (3-D) or ``h·thickness`` (2-D) and an axis normal — the *staircase* surface. Each node also gets a smooth
outward normal ``n̂ = −m/|m|`` from the solid-fraction moment ``m = Σ_e frac_e (c_e − x) w(|c_e − x|)`` over the
4^dim surrounding elements (cone weight ``w = max(0, 2.5h − d)``); with partial volumes it follows curved surfaces
closely. The *geometric* area of a staircase face is ``A_f |n_f · n̂|`` (Σ_f n_f,k² over the staircase faces under
a smooth surface patch equals its area), used for area-weighted distributions and springs.

Selectors exclude what they should: :class:`OnCylinder` (bore face) keeps only surface nodes whose normal points
toward the axis within ``max_angle``, so the end faces at the bore mouths are not selected; :class:`OnPlane` keeps
only surface nodes whose normal is (anti)parallel to the plane normal.

Supports and how they enter the solver
--------------------------------------
Every support yields, per node, the *directions* in which the displacement is zero. Per node the directions of all
supports are combined (SVD, rank k): k = dim → all DOFs fixed; k axis-aligned directions → those DOFs fixed (plain
:class:`~toporlogy.fem.Elasticity`); otherwise the node gets a local orthonormal frame T (constrained directions
first) and the constrained local DOFs are removed after the transformation ``K' = Tᵀ K T`` (exact multi-point
constraint by transformation). Elastic supports add nodal spring blocks ``k A_i (n nᵀ | I | d dᵀ)`` to K. Both
need :class:`ConstrainedElasticity`, a subclass of ``Elasticity`` that overrides ``matrix`` and ``_solve_free``;
the autograd adjoint of the core works unchanged (springs do not depend on the design; ``U = T S⁻¹ Tᵀ F`` is
self-adjoint for compliance).

Loads (never a single node by default)
--------------------------------------
* :class:`PointLoad` — the force spread over the surface nodes within ``radius`` (default 1.5 h) with the
  rigid-body distribution below, so it is statically equivalent to the force at the point (zero moment about it);
* :class:`RemoteForce`, :class:`RemoteMoment`, :class:`Torque` — weighted least-squares distribution
  ``f_i = w_i (λ + μ × r_i)`` (minimises Σ|f_i|²/w_i under the resultant force and moment, a rigid-body field;
  the "distributed / deformable" remote load of commercial codes, source not read in this session);
* :class:`SurfaceForce` (total force ∝ geometric area), :class:`Traction` (force per area),
  :class:`Pressure` (−p n on staircase faces: a closed pressurised cavity gives exactly zero net force) — consistent
  nodal loads of a uniform traction on bilinear faces (A/4 per corner in 3-D, A/2 per end in 2-D);
* :class:`BearingLoad` — the radial part of the force over the loaded half of a bore with weight
  ``A_i max(0, cos θ_i)`` (projected-area cosine distribution, ported and generalised from
  the repository's earlier voxel tools);
  ``mode="parallel"`` (nodal forces parallel to the load) or ``"radial"`` (normal contact pressure ∝ cos θ). The
  documentation of the commercial "Bearing Load" was not read in this session;
* body forces: :func:`body_force_matrix` (design-independent or as a linear map of x̄) and the design-dependent
  response :class:`BodyForceCompliance` with :func:`design_load_solve` (adjoint through the load).

Every load records its intended force (and moment about a reference point when it prescribes one);
:meth:`BoundaryConditions.check` compares them with the assembled nodal forces to 1e-9 relative.

Per-load-case supports
----------------------
The core ``Elasticity`` has one set of fixed DOFs and ``Study`` takes one physics. :meth:`BoundaryConditions.physics`
returns a single physics when all cases share their supports, else a :class:`PerCaseSupports` wrapper that solves
each group of columns with its own physics and concatenates (autograd flows through each solve). Responses that run
extra solves through ``ctx.physics`` (adjoint-based stress etc.) must use the group physics.
"""
from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from toporlogy.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.geometry import Bore, Shape
from toporlogy.responses import EvalContext, Response

__all__ = [
    # selectors
    "Selector", "AllNodes", "SurfaceNodes", "InBox", "OnPlane", "HalfSpaceNodes", "OnCylinder", "BoreFace",
    "InSphere", "InShape", "AtPoints", "Nearest", "NodeIds", "Where", "EmptySelectionError",
    # supports
    "Support", "Fixed", "Roller", "Symmetry", "PinnedBore", "ElasticSupport",
    # loads
    "Load", "Applied", "NodalForces", "PointLoad", "SurfaceForce", "Traction", "Pressure", "BearingLoad",
    "RemoteForce", "RemoteMoment", "Torque", "BodyForce",
    # cases, physics, checks
    "LoadCase", "Combination", "BoundaryConditions", "ConstraintSet", "ConstrainedElasticity", "PerCaseSupports",
    "Issue", "CheckReport", "part_info", "PartInfo", "rigid_distribution", "reaction_forces",
    "body_force_matrix", "design_load_solve", "BodyForceCompliance", "clearance_mask",
]


class EmptySelectionError(ValueError):
    """A selector picked no node of the part."""


# =============================================================================================== part topology
class PartInfo:
    """Surface, faces, normals and areas of a domain's part (computed once per domain, see :func:`part_info`)."""

    def __init__(self, domain: Domain) -> None:
        d = domain
        self.domain, self.dim, self.h = d, d.dim, d.h
        self.nper = 2 ** d.dim
        act = d.active_elements
        self.elem_nodes = d.edof[act][:, ::d.dim] // d.dim                 # (na, 2^dim)
        cnt = np.bincount(self.elem_nodes.ravel(), minlength=d.nn)
        self.node_count = cnt
        self.node_active = cnt > 0
        self.surface = (cnt > 0) & (cnt < self.nper)
        self.X = d.node_coords()
        xa = self.X[self.node_active]
        self.L = float(np.linalg.norm(xa.max(axis=0) - xa.min(axis=0))) or d.h
        self.face_area = d.h * d.h if d.dim == 3 else d.h * d.thickness
        self.elem_volume = d.h ** 3 if d.dim == 3 else d.h * d.h * d.thickness
        self._normals: np.ndarray | None = None
        self._faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ smooth outward normals
    @property
    def normals(self) -> np.ndarray:
        """(nn, dim) unit outward normal per node (zero where undefined, e.g. interior nodes)."""
        if self._normals is None:
            d = self.domain
            frac = np.where(d.active, d.frac, 0.0)
            P = np.pad(frac, 2)
            W = np.lib.stride_tricks.sliding_window_view(P, (4,) * self.dim)   # node i ↔ elements i−2 … i+1
            off = np.array([-1.5, -0.5, 0.5, 1.5])
            G = np.stack(np.meshgrid(*([off] * self.dim), indexing="ij"), axis=0)   # (dim, 4, …)
            w = np.maximum(0.0, 2.5 - np.sqrt((G ** 2).sum(axis=0)))
            letters = "abc"[: self.dim]
            m = np.stack([np.einsum(f"...{letters},{letters}->...", W, G[k] * w) for k in range(self.dim)], axis=-1)
            m = m.reshape(-1, self.dim)
            nrm = np.linalg.norm(m, axis=1)
            n = np.zeros_like(m)
            ok = nrm > 1e-9
            n[ok] = -m[ok] / nrm[ok, None]
            self._normals = n
        return self._normals

    # ------------------------------------------------------------------ exposed faces (staircase surface)
    @property
    def faces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(face nodes (nf, 2^(dim−1)), staircase normal n_f (nf, dim), geometric area factor (nf,), smooth face
        normal (nf, dim)).

        The smooth normal of a face is the mean of its corner normals when the element that owns it is partial
        (frac < 1: the true surface cuts it, the face is a staircase step) and n_f otherwise (a full element with an
        empty neighbour: the face lies on the surface). The area factor is |n_f · smooth normal| — exactly 1 on
        grid-aligned faces, also next to edges where the corner normals tilt."""
        if self._faces is None:
            d = self.domain
            act = d.active
            corners = np.indices((2,) * self.dim).reshape(self.dim, -1).T
            fn, fnorm, own = [], [], []
            for k in range(self.dim):
                for s in (-1, 1):
                    nb = np.zeros_like(act)
                    src = [slice(None)] * self.dim
                    dst = [slice(None)] * self.dim
                    if s > 0:
                        src[k], dst[k] = slice(1, None), slice(None, -1)
                    else:
                        src[k], dst[k] = slice(None, -1), slice(1, None)
                    nb[tuple(dst)] = act[tuple(src)]
                    idx = np.argwhere(act & ~nb)
                    if not len(idx):
                        continue
                    cs = corners[corners[:, k] == (1 if s > 0 else 0)]
                    fn.append(np.stack([d.node_number(idx + c) for c in cs], axis=1))
                    e = np.zeros((len(idx), self.dim))
                    e[:, k] = s
                    fnorm.append(e)
                    own.append(d.frac[tuple(idx.T)])
            F = np.concatenate(fn)
            N = np.concatenate(fnorm)
            partial = np.concatenate(own) < 1.0 - 1e-9
            nh = self.normals[F].mean(axis=1)
            ln = np.linalg.norm(nh, axis=1)
            use = partial & (ln > 1e-9)
            SN = N.copy()
            SN[use] = nh[use] / ln[use, None]
            fac = np.abs(np.einsum("ij,ij->i", N, SN))
            self._faces = (F, N, fac, SN)
        return self._faces

    def patch(self, nodes: np.ndarray, geometric: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Faces whose corners all lie in ``nodes`` and the consistent nodal area of ``nodes`` on them.

        Returns (face indices, nodal area aligned with ``nodes``). ``geometric`` applies the |n_f·n̂| factor.
        """
        F, _, fac, _ = self.faces
        inset = np.zeros(self.domain.nn, bool)
        inset[nodes] = True
        sel = np.flatnonzero(inset[F].all(axis=1))
        a = self.face_area * (fac[sel] if geometric else np.ones(len(sel)))
        area = np.zeros(self.domain.nn)
        np.add.at(area, F[sel].ravel(), np.repeat(a / F.shape[1], F.shape[1]))
        return sel, area[nodes]

    def patch_normals(self, nodes: np.ndarray) -> np.ndarray:
        """(k, dim) outward normal of ``nodes`` from the smooth normals of the complete faces of the selection
        (area-weighted); the node normal where a node is on no such face. Flat patches give the exact normal up
        to their edges."""
        F, _, fac, SN = self.faces
        sel, _ = self.patch(nodes)
        acc = np.zeros((self.domain.nn, self.dim))
        np.add.at(acc, F[sel].ravel(), np.repeat(fac[sel, None] * SN[sel], F.shape[1], axis=0))
        out = acc[nodes]
        ln = np.linalg.norm(out, axis=1)
        out = np.where(ln[:, None] > 1e-12, out / np.maximum(ln, 1e-300)[:, None], self.normals[nodes])
        return out


_INFO: weakref.WeakKeyDictionary[Domain, PartInfo] = weakref.WeakKeyDictionary()


def part_info(domain: Domain) -> PartInfo:
    """Cached :class:`PartInfo` of ``domain``."""
    info = _INFO.get(domain)
    if info is None:
        info = PartInfo(domain)
        _INFO[domain] = info
    return info


def _cross(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """r × f row-wise; scalar (n,) in 2-D."""
    if r.shape[-1] == 2:
        return r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]
    return np.cross(r, f)


def _vec(v: Any, dim: int, what: str) -> np.ndarray:
    a = np.asarray(v, float).reshape(-1)
    if a.size != dim:
        raise ValueError(f"{what} must have {dim} components, got {a.size}")
    return a


def _mom(v: Any, dim: int) -> np.ndarray:
    """Moment as an array: (3,) in 3-D, (1,) in 2-D (about the out-of-plane axis)."""
    a = np.asarray(v, float).reshape(-1)
    want = 3 if dim == 3 else 1
    if a.size != want:
        raise ValueError(f"moment must have {want} component(s) in {dim}-D, got {a.size}")
    return a


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


# =============================================================================================== supports
class Support:
    """Base class: ``directions(info) -> (nodes (k,), D (k, m, dim))`` zero-displacement directions per node and
    ``springs(info) -> (nodes, K (k, dim, dim))``. ``name`` labels it in reports."""

    name = "support"
    kind = "support"

    def directions(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(0, np.int64), np.zeros((0, 0, info.dim))

    def springs(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(0, np.int64), np.zeros((0, info.dim, info.dim))

    def nodes(self, info: PartInfo) -> np.ndarray:
        a, _ = self.directions(info)
        b, _ = self.springs(info)
        return np.union1d(a, b)


class Fixed(Support):
    """Zero displacement of the selected nodes in ``directions`` (default all; ints = global axes, or vectors)."""

    kind = "fixed"

    def __init__(self, where: Selector, directions: Sequence[int] | Sequence[Sequence[float]] | None = None,
                 name: str | None = None) -> None:
        self.where, self.dirs = where, directions
        self.name = name or f"Fixed {where!r}"

    def _D(self, dim: int) -> np.ndarray:
        if self.dirs is None:
            return np.eye(dim)
        rows = []
        for d in self.dirs:
            if np.isscalar(d):
                rows.append(np.eye(dim)[int(d)])
            else:
                v = _vec(d, dim, "direction")
                rows.append(v / np.linalg.norm(v))
        return np.array(rows)

    def directions(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        n = self.where.select(info)
        D = self._D(info.dim)
        return n, np.broadcast_to(D, (len(n),) + D.shape).copy()


class Roller(Fixed):
    """Zero displacement along ``normal`` (any direction) at the selected nodes; free in the tangent plane."""

    kind = "roller"

    def __init__(self, where: Selector, normal: Sequence[float] | int, name: str | None = None) -> None:
        super().__init__(where, [normal], name or f"Roller {where!r}")


def Symmetry(point: Sequence[float], normal: Sequence[float] | int, tol: float | None = None,
             name: str | None = None) -> Roller:
    """Symmetry plane: zero normal displacement of the part's surface nodes on the plane."""
    sel = OnPlane(point, normal, tol=tol, facing=None)
    return Roller(sel, sel.normal, name or f"Symmetry {sel!r}")


class PinnedBore(Support):
    """Pin in a bore: radial displacement zero (local cylindrical frame), rotation about the pin free.

    ``axial=True`` also fixes the axial displacement, ``tangential=True`` the tangential one (= fixed bore).
    ``bore``: a :class:`~toporlogy.geometry.Bore` or an :class:`OnCylinder` selector.
    """

    kind = "pinned"

    def __init__(self, bore: Bore | OnCylinder, axial: bool = False, tangential: bool = False,
                 name: str | None = None, **select_kw: Any) -> None:
        self.sel = bore if isinstance(bore, OnCylinder) else OnCylinder.from_bore(bore, facing="inward", **select_kw)
        self.axial, self.tangential = axial, tangential
        self.name = name or f"Pinned {self.sel!r}"

    def directions(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        n = self.sel.select(info)
        _, _, er = self.sel.coords(info.X[n])
        rows = [er]
        if info.dim == 3:
            a = np.broadcast_to(self.sel.axis, er.shape)
            if self.axial:
                rows.append(a.copy())
            if self.tangential:
                rows.append(np.cross(a, er))
        elif self.tangential:
            rows.append(np.column_stack([-er[:, 1], er[:, 0]]))
        bad = np.linalg.norm(er, axis=1) == 0
        if bad.any():
            raise ValueError(f"{self.name}: {int(bad.sum())} node(s) on the pin axis (radial direction undefined)")
        return n, np.stack(rows, axis=1)


class ElasticSupport(Support):
    """Springs to ground on surface nodes: stiffness per unit area ``k`` × the node's geometric area.

    ``direction``: "normal" (k n̂ n̂ᵀ with the outward normal), "all" (k I) or a vector d (k d̂ d̂ᵀ). Alternatively
    ``total`` = total stiffness, shared in proportion to area (then ``k`` is ignored).
    """

    kind = "spring"

    def __init__(self, where: Selector, k: float = 0.0, direction: str | Sequence[float] = "normal",
                 total: float | None = None, name: str | None = None) -> None:
        self.where, self.k, self.direction, self.total = where, float(k), direction, total
        self.name = name or f"Springs {where!r}"

    def springs(self, info: PartInfo) -> tuple[np.ndarray, np.ndarray]:
        n = self.where.select(info)
        _, area = info.patch(n)
        if not area.sum() > 0:
            raise ValueError(f"{self.name}: the selection contains no complete surface face (area 0)")
        kn = (self.total * area / area.sum()) if self.total is not None else self.k * area
        dim = info.dim
        if isinstance(self.direction, str) and self.direction == "normal":
            nn = info.patch_normals(n)
            B = nn[:, :, None] * nn[:, None, :]
        elif isinstance(self.direction, str) and self.direction == "all":
            B = np.broadcast_to(np.eye(dim), (len(n), dim, dim))
        elif isinstance(self.direction, str):
            raise ValueError("direction: 'normal' | 'all' | vector")
        else:
            d = _vec(self.direction, dim, "spring direction")
            d = d / np.linalg.norm(d)
            B = np.broadcast_to(np.outer(d, d), (len(n), dim, dim))
        keep = kn > 0
        return n[keep], (kn[:, None, None] * B)[keep]


# =============================================================================================== constraint set
def _constraint_key(cs: ConstraintSet) -> bytes:
    """Content of a resolved support set: fixed DOFs, local frames (node, constrained count, axes) and springs."""
    parts = [np.asarray(cs.fixed, np.int64).tobytes(), b"|frames"]
    for node in sorted(cs.frames):
        T, k = cs.frames[node]
        parts += [np.int64(node).tobytes(), np.int64(k).tobytes(), np.ascontiguousarray(T, np.float64).tobytes()]
    parts += [b"|springs", np.asarray(cs.spring_nodes, np.int64).tobytes(),
              np.ascontiguousarray(cs.spring_K, np.float64).tobytes()]
    return b"".join(parts)


@dataclass
class ConstraintSet:
    """Supports resolved to DOFs: global fixed DOFs, local frames for oblique constraints, nodal springs."""

    fixed: np.ndarray                                   # global DOF numbers (sorted)
    frames: dict[int, tuple[np.ndarray, int]]           # node → (T (dim×dim, columns = local axes), n constrained)
    spring_nodes: np.ndarray
    spring_K: np.ndarray                                # (k, dim, dim) global frame
    per_support: list[tuple[str, str, np.ndarray, int]]  # (name, kind, nodes, constraint rows)
    multiply_constrained: np.ndarray                     # nodes touched by more than one support
    redundant_rows: int

    @property
    def needs_extension(self) -> bool:
        return bool(self.frames) or len(self.spring_nodes) > 0

    def constrained_directions(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """(nodes, D (m, dim)) one row per constrained direction (for checks)."""
        nodes, rows = [], []
        if len(self.fixed):
            nodes.append(self.fixed // dim)
            rows.append(np.eye(dim)[self.fixed % dim])
        for n, (T, k) in self.frames.items():
            nodes.append(np.full(k, n))
            rows.append(T[:, :k].T)
        if not nodes:
            return np.zeros(0, np.int64), np.zeros((0, dim))
        return np.concatenate(nodes), np.concatenate(rows)


def _resolve(supports: Sequence[Support], info: PartInfo) -> ConstraintSet:
    dim = info.dim
    nodes_l, dirs_l, owner_l = [], [], []
    per: list[tuple[str, str, np.ndarray, int]] = []
    sn, sk = [], []
    for s_i, s in enumerate(supports):
        n, D = s.directions(info)
        n2, K = s.springs(info)
        rows = int(D.shape[0] * D.shape[1]) if D.size else 0
        per.append((s.name, s.kind, np.union1d(n, n2), rows))
        if len(n):
            m = D.shape[1]
            nodes_l.append(np.repeat(n, m))
            dirs_l.append(D.reshape(-1, dim))
            owner_l.append(np.full(len(n) * m, s_i))
        if len(n2):
            sn.append(n2)
            sk.append(K)
    touched = np.concatenate([p[2] for p in per]) if per else np.zeros(0, np.int64)
    u, c = np.unique(touched, return_counts=True)
    multi = u[c > 1]
    fixed: list[np.ndarray] = []
    frames: dict[int, tuple[np.ndarray, int]] = {}
    redundant = 0
    if nodes_l:
        N = np.concatenate(nodes_l)
        D = np.concatenate(dirs_l)
        order = np.argsort(N, kind="stable")
        N, D = N[order], D[order]
        axis_like = np.isclose(np.abs(D).max(axis=1), 1.0, rtol=0, atol=1e-12)
        starts = np.flatnonzero(np.r_[True, N[1:] != N[:-1]])
        ends = np.r_[starts[1:], len(N)]
        all_axis = np.logical_and.reduceat(axis_like, starts)
        # fast path: nodes whose constraint directions are all coordinate axes
        fa = np.flatnonzero(all_axis)
        if len(fa):
            sel = np.concatenate([np.arange(starts[k], ends[k]) for k in fa])
            ax = np.argmax(np.abs(D[sel]), axis=1)
            dof = np.unique(dim * N[sel] + ax)
            redundant += len(sel) - len(dof)
            fixed.append(dof)
        for k in np.flatnonzero(~all_axis):
            n = int(N[starts[k]])
            Dn = D[starts[k]:ends[k]]
            _, s, Vt = np.linalg.svd(Dn)
            rank = int((s > 1e-9 * s[0]).sum())
            redundant += len(Dn) - rank
            if rank == dim:
                fixed.append(dim * n + np.arange(dim))
                continue
            B = Vt[:rank]
            P = B.T @ B                                           # projector onto the constrained span
            on_axes = np.flatnonzero(np.isclose(np.diag(P), 1.0, atol=1e-10))
            if len(on_axes) == rank:
                fixed.append(dim * n + on_axes)
            else:
                frames[n] = (Vt.T.copy(), rank)                   # columns: constrained first, then free
    fixed_arr = np.unique(np.concatenate(fixed)) if fixed else np.zeros(0, np.int64)
    spring_nodes = np.concatenate(sn) if sn else np.zeros(0, np.int64)
    spring_K = np.concatenate(sk) if sk else np.zeros((0, dim, dim))
    return ConstraintSet(fixed_arr, frames, spring_nodes, spring_K, per, multi, redundant)


# =============================================================================================== physics extension
class ConstrainedElasticity(Elasticity):
    """:class:`~toporlogy.fem.Elasticity` with oblique constraints (local frames, removed after
    ``K' = Tᵀ K T``) and nodal springs to ground (added to K before the transformation).

    ``free`` keeps the parent meaning (active DOFs not fixed globally); the reduced system has the free DOFs minus
    the locally constrained ones. ``U`` is returned in the global frame, so responses, ``element_energy`` and the
    core adjoint are unchanged.
    """

    def __init__(self, domain: Domain, constraints: ConstraintSet, nu: float = 0.3, element: str | None = None,
                 solver: Any = None) -> None:
        super().__init__(domain, constraints.fixed, nu=nu, element=element, solver=solver)
        dim = domain.dim
        nf = len(self.free)
        gmap = np.full(domain.ndof, -1, np.int64)
        gmap[self.free] = np.arange(nf)
        rows, cols, vals = [np.arange(nf)], [np.arange(nf)], [np.ones(nf)]
        keep = np.ones(nf, bool)
        ident = np.ones(nf, bool)
        for n, (T, k) in constraints.frames.items():
            g = gmap[dim * n + np.arange(dim)]
            if np.any(g < 0):
                raise ValueError(f"node {n}: a locally constrained node must have all its DOFs free and active")
            ident[g] = False
            rr, cc = np.meshgrid(g, g, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(T.ravel())
            keep[g[:k]] = False
        rows[0], cols[0], vals[0] = rows[0][ident], cols[0][ident], vals[0][ident]
        self.T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, nf))
        self.keep = np.flatnonzero(keep)
        S = sp.csr_matrix((nf, nf))
        if len(constraints.spring_nodes):
            sr, sc, sv = [], [], []
            for n, K in zip(constraints.spring_nodes, constraints.spring_K, strict=True):
                g = gmap[dim * int(n) + np.arange(dim)]
                ok = g >= 0
                rr, cc = np.meshgrid(g[ok], g[ok], indexing="ij")
                sr.append(rr.ravel())
                sc.append(cc.ravel())
                sv.append(K[np.ix_(ok, ok)].ravel())
            S = sp.csr_matrix((np.concatenate(sv), (np.concatenate(sr), np.concatenate(sc))), shape=(nf, nf))
        self.S = S
        self.constraints = constraints

    def matrix(self, E_active: np.ndarray) -> sp.csr_matrix:
        A = super().matrix(E_active)
        K = (A + sp.triu(A, k=1).T + self.S).tocsr()
        Kl = (self.T.T @ K @ self.T).tocsr()[self.keep][:, self.keep]
        out = sp.triu(Kl).tocsr()
        out.sort_indices()
        return out

    def _solve_free(self, B_full: np.ndarray) -> np.ndarray:
        X = np.zeros_like(B_full)
        b = B_full[self.free]
        b2 = b[:, None] if b.ndim == 1 else b
        bl = (self.T.T @ b2)[self.keep]
        y = np.asarray(self.solver.solve(bl)).reshape(bl.shape)
        z = np.zeros_like(b2)
        z[self.keep] = y
        x = self.T @ z
        X[self.free] = x.reshape(b.shape)
        return X


class PerCaseSupports:
    """Physics for load cases with different supports: column groups solved by their own physics.

    ``groups``: list of (physics, column indices). ``solve`` returns (ndof, n_cases) in the original column order.
    """

    def __init__(self, groups: Sequence[tuple[Any, Sequence[int]]]) -> None:
        self.groups = [(p, np.asarray(c, np.int64)) for p, c in groups]
        cols = np.concatenate([c for _, c in self.groups])
        if sorted(cols.tolist()) != list(range(len(cols))):
            raise ValueError("column groups must partition 0 ... n_cases-1")
        self._inv = torch.as_tensor(np.argsort(cols), dtype=torch.long)
        first = self.groups[0][0]
        self.domain, self.elements, self.geom, self.KE = first.domain, first.elements, first.geom, first.KE
        self.n_cases = len(cols)

    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        if F2.shape[1] != self.n_cases:
            raise ValueError(f"F has {F2.shape[1]} columns, the support groups cover {self.n_cases}")
        parts = [p.solve(E_active, F2[:, c]) for p, c in self.groups]
        return torch.cat(parts, dim=1).index_select(1, self._inv.to(E_active.device))


def reaction_forces(physics: Elasticity, E_active: np.ndarray, U: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(nn, dim[, n_cases]) ``K U − F`` with the unconstrained stiffness: support reactions at constrained nodes
    (≈ 0 at free nodes; at spring nodes it is the spring force). Σ over all nodes equals −Σ F (equilibrium)."""
    d = physics.domain
    U2 = U[:, None] if U.ndim == 1 else U
    F2 = F[:, None] if F.ndim == 1 else F
    Ue = U2[physics.edof]                                           # (na, nloc, nc)
    fe = np.einsum("ab,ebc->eac", physics.KE, Ue) * (np.asarray(E_active, float) * physics.geom)[:, None, None]
    R = np.zeros_like(U2)
    np.add.at(R, physics.edof.ravel(), fe.reshape(-1, U2.shape[1]))
    R -= F2
    out = R.reshape(d.nn, d.dim, -1)
    return out[..., 0] if U.ndim == 1 else out


# =============================================================================================== loads
@dataclass
class Applied:
    """Nodal forces of one load and what was intended."""

    name: str
    kind: str
    nodes: np.ndarray
    forces: np.ndarray                     # (k, dim)
    intended_force: np.ndarray
    ref_point: np.ndarray
    intended_moment: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def force(self) -> np.ndarray:
        return self.forces.sum(axis=0)

    def moment(self, point: np.ndarray | None = None, X: np.ndarray | None = None) -> np.ndarray:
        """Resultant moment about ``point`` (default ``ref_point``); X: node coordinates (nn, dim)."""
        if X is None:
            raise ValueError("node coordinates needed")
        p = self.ref_point if point is None else np.asarray(point, float)
        m = _cross(X[self.nodes] - p, self.forces).sum(axis=0)
        return np.atleast_1d(m)


class Load:
    """Base class: ``apply(info) -> Applied``."""

    name = "load"

    def apply(self, info: PartInfo) -> Applied:  # pragma: no cover - interface
        raise NotImplementedError


def rigid_distribution(X: np.ndarray, w: np.ndarray, point: np.ndarray, force: np.ndarray,
                       moment: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Nodal forces f_i = w_i (λ + μ × r_i), r_i = X_i − point, with Σ f_i = force and Σ r_i × f_i = moment.

    Minimises Σ |f_i|² / w_i (weighted least squares, a rigid-body field). Raises ValueError when the nodes cannot
    carry the requested resultant (e.g. collinear nodes and a moment about their line).
    """
    dim = X.shape[1]
    r = X - point
    L = max(float(np.sqrt((w * (r ** 2).sum(axis=1)).sum() / w.sum())), 1e-300)
    rs = r / L                                                      # scaled for conditioning
    n = len(X)
    if dim == 3:
        Phi = np.zeros((n, 3, 6))
        Phi[:, :, :3] = np.eye(3)
        x, y, z = rs[:, 0], rs[:, 1], rs[:, 2]
        # μ × r = −[r]× μ
        Phi[:, 0, 4], Phi[:, 0, 5] = z, -y
        Phi[:, 1, 3], Phi[:, 1, 5] = -z, x
        Phi[:, 2, 3], Phi[:, 2, 4] = y, -x
        target = np.r_[force, np.asarray(moment, float).reshape(3) / L]
    else:
        Phi = np.zeros((n, 2, 3))
        Phi[:, :, :2] = np.eye(2)
        Phi[:, 0, 2], Phi[:, 1, 2] = -rs[:, 1], rs[:, 0]
        target = np.r_[force, np.asarray(moment, float).reshape(1) / L]
    M = np.einsum("i,iak,ial->kl", w, Phi, Phi)
    theta = np.linalg.lstsq(M, target, rcond=1e-12)[0]
    f = w[:, None] * np.einsum("iak,k->ia", Phi, theta)
    got = np.einsum("iak,ia->k", Phi, f)
    scale = max(np.linalg.norm(target), 1e-300)
    if np.linalg.norm(got - target) > max(rtol, 1e-9) * scale:
        raise ValueError(f"the {n} node(s) cannot carry the requested resultant (residual "
                         f"{np.linalg.norm(got - target) / scale:.2e}); collinear or too few nodes - enlarge the "
                         "selection or the patch radius")
    return f


def _weights(info: PartInfo, nodes: np.ndarray, mode: str, notes: list[str]) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(nodes))
    if mode != "area":
        raise ValueError("weights: 'area' | 'uniform'")
    _, a = info.patch(nodes)
    if not a.sum() > 0:
        notes.append("no complete surface face in the selection: uniform nodal weights used")
        return np.ones(len(nodes))
    if (a == 0).any():
        notes.append(f"{int((a == 0).sum())} selected node(s) outside every complete face carry no load")
    return a


class NodalForces(Load):
    """Explicit forces on nodes (advanced): ``forces`` (k, dim) or one vector for all."""

    def __init__(self, where: Selector, forces: Any, name: str | None = None) -> None:
        self.where, self.f = where, forces
        self.name = name or f"NodalForces {where!r}"

    def apply(self, info: PartInfo) -> Applied:
        n = self.where.select(info)
        f = np.broadcast_to(np.asarray(self.f, float), (len(n), info.dim)).copy()
        notes = ["single-node load: singular stresses and a grey 'load blob' are likely"] if len(n) == 1 else []
        return Applied(self.name, "nodal", n, f, f.sum(axis=0), info.X[n].mean(axis=0), None, notes)


class RemoteForce(Load):
    """Force (and optional moment) acting at ``point``, carried by the selected nodes as a rigid-body
    least-squares distribution (statically exact)."""

    kind = "remote"

    def __init__(self, where: Selector, force: Sequence[float], point: Sequence[float],
                 moment: Any = None, weights: str = "area", name: str | None = None) -> None:
        self.where, self.F, self.point, self.M, self.wmode = where, force, point, moment, weights
        self.name = name or f"RemoteForce {where!r}"

    def apply(self, info: PartInfo) -> Applied:
        return self._apply(info, _vec(self.F, info.dim, "force"), _vec(self.point, info.dim, "point"))

    def _apply(self, info: PartInfo, F: np.ndarray, p: np.ndarray) -> Applied:
        n = self.where.select(info)
        M = _mom(np.zeros(3 if info.dim == 3 else 1) if self.M is None else self.M, info.dim)
        notes: list[str] = []
        w = _weights(info, n, self.wmode, notes)
        live = w > 0
        f = np.zeros((len(n), info.dim))
        try:
            f[live] = rigid_distribution(info.X[n[live]], w[live], p, F, M)
        except ValueError as exc:
            raise ValueError(f"{self.name}: {exc}") from None
        return Applied(self.name, self.kind, n, f, F, p, M, notes)


class RemoteMoment(RemoteForce):
    """Pure moment about ``point`` (default: the area-weighted centroid of the nodes); zero net force."""

    kind = "moment"

    def __init__(self, where: Selector, moment: Any, point: Sequence[float] | None = None, weights: str = "area",
                 name: str | None = None) -> None:
        super().__init__(where, [0.0], point if point is not None else [0.0], moment, weights,
                         name or f"RemoteMoment {where!r}")
        self._auto = point is None

    def apply(self, info: PartInfo) -> Applied:
        # zero force and the reference point are local to this part: the load object is not modified, so it can be
        # applied to several parts (cases, sweeps) and read afterwards
        if self._auto:
            n = self.where.select(info)
            w = _weights(info, n, self.wmode, [])
            p = (w[:, None] * info.X[n]).sum(axis=0) / w.sum()
        else:
            p = _vec(self.point, info.dim, "point")
        return self._apply(info, np.zeros(info.dim), p)


class PointLoad(Load):
    """A force at ``point`` spread over the surface nodes within ``radius`` (default 1.5 h), statically
    equivalent to the point force (rigid-body distribution, zero moment about the point)."""

    kind = "point"

    def __init__(self, point: Sequence[float], force: Sequence[float], radius: float | None = None,
                 name: str | None = None) -> None:
        self.point, self.F, self.radius = np.asarray(point, float), force, radius
        self.name = name or f"PointLoad at {self.point.tolist()}"

    def apply(self, info: PartInfo) -> Applied:
        r = 1.5 * info.h if self.radius is None else float(self.radius)
        sel = InSphere(self.point, r, surface=True)
        n = sel.select(info, allow_empty=True)
        if not len(n):
            idx = np.flatnonzero(info.surface)
            dmin = float(np.min(np.linalg.norm(info.X[idx] - self.point, axis=1)))
            raise EmptySelectionError(f"{self.name}: no surface node within radius {r:g} (nearest surface node at "
                                      f"{dmin:g}); move the point onto the part or enlarge the radius")
        rf = RemoteForce(NodeIds(n), self.F, self.point, None, "area", self.name)
        a = rf.apply(info)
        a.kind = self.kind
        if len(n) < 3:
            a.notes.append(f"patch of {len(n)} node(s): enlarge the radius (>= 1.5 h) to avoid a load singularity")
        return a


class SurfaceForce(Load):
    """Total force over the selected surface, shared in proportion to the geometric nodal area (uniform
    traction of the given resultant)."""

    kind = "surface force"

    def __init__(self, where: Selector, force: Sequence[float], name: str | None = None) -> None:
        self.where, self.F = where, force
        self.name = name or f"SurfaceForce {where!r}"

    def apply(self, info: PartInfo) -> Applied:
        n = self.where.select(info)
        F = _vec(self.F, info.dim, "force")
        notes: list[str] = []
        w = _weights(info, n, "area", notes)
        f = np.outer(w / w.sum(), F)
        return Applied(self.name, self.kind, n, f, F, info.X[n].mean(axis=0), None, notes)


class Traction(Load):
    """Traction vector (force per area) on the complete surface faces of the selection; area = geometric
    (default) or staircase face area (``geometric=False``)."""

    kind = "traction"

    def __init__(self, where: Selector, traction: Sequence[float], geometric: bool = True,
                 name: str | None = None) -> None:
        self.where, self.t, self.geometric = where, traction, geometric
        self.name = name or f"Traction {where!r}"

    def apply(self, info: PartInfo) -> Applied:
        n = self.where.select(info)
        t = _vec(self.t, info.dim, "traction")
        faces, a = info.patch(n, self.geometric)
        if not len(faces):
            raise EmptySelectionError(f"{self.name}: the selection contains no complete surface face")
        f = np.outer(a, t)
        return Applied(self.name, self.kind, n, f, t * a.sum(), info.X[n].mean(axis=0), None, [])


class Pressure(Load):
    """Pressure ``p`` (positive = pushing into the part) on the complete surface faces of the selection.

    ``normals="faces"`` (default): staircase face normals and areas, −p Σ n_f A_f — the discrete divergence theorem
    holds exactly (a closed cavity has zero net force). ``"smooth"``: smooth normal n̂ and geometric area.
    """

    kind = "pressure"

    def __init__(self, where: Selector, p: float, normals: str = "faces", name: str | None = None) -> None:
        self.where, self.p, self.normals = where, float(p), normals
        self.name = name or f"Pressure {where!r}"

    def apply(self, info: PartInfo) -> Applied:
        n = self.where.select(info)
        sel, _ = info.patch(n, False)
        if not len(sel):
            raise EmptySelectionError(f"{self.name}: the selection contains no complete surface face")
        F, N, fac, SN = info.faces
        ncorner = F.shape[1]
        if self.normals == "faces":
            ff = -self.p * info.face_area * N[sel]
        elif self.normals == "smooth":
            ff = -self.p * info.face_area * fac[sel, None] * SN[sel]
        else:
            raise ValueError("normals: 'faces' | 'smooth'")
        acc = np.zeros((info.domain.nn, info.dim))
        np.add.at(acc, F[sel].ravel(), np.repeat(ff / ncorner, ncorner, axis=0))
        f = acc[n]
        return Applied(self.name, self.kind, n, f, ff.sum(axis=0), info.X[n].mean(axis=0), None, [])


def _cylinder_samples(info: PartInfo, sel: OnCylinder, spacing: float | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint samples of the exact cylinder (3-D) / circle (2-D) of ``sel``: points, e_r, area weights."""
    h = info.h
    ds = h / 3.0 if spacing is None else float(spacing)
    R = sel.radius
    nt = max(16, int(math.ceil(2 * math.pi * R / ds)))
    th = (np.arange(nt) + 0.5) * 2 * math.pi / nt
    if sel.axis is None:
        er = np.column_stack([np.cos(th), np.sin(th)])
        P = sel.p0 + R * er
        dA = np.full(nt, 2 * math.pi * R / nt * info.domain.thickness)
        return P, er, dA
    a = sel.axis
    u = np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    nz = max(1, int(math.ceil(sel.length / ds)))
    z = (np.arange(nz) + 0.5) * sel.length / nz
    T, Z = np.meshgrid(th, z, indexing="ij")
    er = np.cos(T).ravel()[:, None] * u + np.sin(T).ravel()[:, None] * v
    P = sel.p0 + Z.ravel()[:, None] * a + R * er
    dA = np.full(len(P), 2 * math.pi * R / nt * sel.length / nz)
    return P, er, dA


def _scatter(info: PartInfo, P: np.ndarray, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consistent nodal forces of point forces ``vec`` at points ``P`` inside active elements (bi/trilinear shape
    functions of the containing element). Returns (nodes, forces, mask of the points that were used)."""
    d = info.domain
    s = (P - d.origin) / d.h
    idx = np.floor(s).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.asarray(d.shape)), axis=1)
    use = np.zeros(len(P), bool)
    use[inside] = d.active[tuple(idx[inside].T)]
    xi = s[use] - idx[use]
    corners = np.indices((2,) * d.dim).reshape(d.dim, -1).T
    acc = np.zeros((d.nn, d.dim))
    for c in corners:
        N = np.prod(np.where(c[None, :] == 1, xi, 1.0 - xi), axis=1)
        np.add.at(acc, d.node_number(idx[use] + c), N[:, None] * vec[use])
    nodes = np.flatnonzero(np.abs(acc).sum(axis=1) > 0)
    return nodes, acc[nodes], use


def _bearing_vectors(er: np.ndarray, A: np.ndarray, F: np.ndarray, axis: np.ndarray | None, mode: str,
                     name: str) -> np.ndarray:
    """Force vectors at the bore points with radial directions ``er`` and areas ``A`` (sum exactly F)."""
    Fa = (F @ axis) * axis if axis is not None else np.zeros_like(F)
    Fr = F - Fa
    f = np.outer(A / A.sum(), Fa)
    fr = float(np.linalg.norm(Fr))
    if fr > 0:
        dhat = Fr / fr
        w = A * np.maximum(0.0, er @ dhat)
        if not w.sum() > 0:
            raise ValueError(f"{name}: no bore point on the loaded half")
        if mode == "parallel":
            f += np.outer(w / w.sum(), Fr)
        else:
            g = w[:, None] * er
            g *= fr / float(g.sum(axis=0) @ dhat)
            g -= np.outer(w / w.sum(), g.sum(axis=0) - Fr)          # exact resultant (tiny correction)
            f += g
    return f


class BearingLoad(Load):
    """Pin (bearing) load on a bore: the radial part of ``force`` over the loaded half with weight
    dA max(0, cos θ), θ = angle between the radial direction and the radial force direction; an axial part
    (if any) uniform by area. ``mode``: "parallel" (forces parallel to the radial force) or "radial" (along e_r,
    ∝ cos θ, corrected to the exact resultant). ``bore``: Bore or OnCylinder.

    ``on="surface"`` (default): the distribution is evaluated on the exact cylinder (samples every h/3) and turned
    into consistent nodal forces with the shape functions of the element containing each sample (embedded-surface
    load; samples in inactive elements are dropped and the rest renormalised). With partial volumes this does not
    load the corner nodes of near-empty wall elements that reach into the hole, which made the compliance jump
    with the grid position by several per cent (measured, see tests). ``on="nodes"``: the selected bore-wall nodes
    with their geometric nodal areas (the older rule).
    """

    kind = "bearing"

    def __init__(self, bore: Bore | OnCylinder, force: Sequence[float], mode: str = "parallel", on: str = "surface",
                 name: str | None = None, **select_kw: Any) -> None:
        self.sel = bore if isinstance(bore, OnCylinder) else OnCylinder.from_bore(bore, facing="inward", **select_kw)
        self.F, self.mode, self.on = force, mode, on
        if mode not in ("parallel", "radial"):
            raise ValueError("mode: 'parallel' | 'radial'")
        if on not in ("surface", "nodes"):
            raise ValueError("on: 'surface' | 'nodes'")
        self.name = name or f"BearingLoad {self.sel!r}"

    def apply(self, info: PartInfo) -> Applied:
        F = _vec(self.F, info.dim, "force")
        notes: list[str] = []
        a = self.sel.axis
        center = self.sel.p0 + (0.5 * self.sel.length * a if a is not None else 0.0)
        if self.on == "nodes":
            n = self.sel.select(info)
            A = _weights(info, n, "area", notes)
            _, _, er = self.sel.coords(info.X[n])
            f = _bearing_vectors(er, A, F, a, self.mode, self.name)
            return Applied(self.name, self.kind, n, f, F, center, None, notes)
        P, er, dA = _cylinder_samples(info, self.sel)
        if a is not None:
            # each sample is split into two halves just before / after it along the axis, so a sample on an end
            # face of the part counts half (no bias from the half-open element search)
            eps = 1e-7 * info.h * a
            P, er, dA = np.r_[P - eps, P + eps], np.r_[er, er], np.r_[0.5 * dA, 0.5 * dA]
        d = info.domain
        s = np.floor((P - d.origin) / d.h).astype(np.int64)
        ok = np.all((s >= 0) & (s < np.asarray(d.shape)), axis=1)
        ok[ok] = d.active[tuple(s[ok].T)]
        if not ok.any():
            raise EmptySelectionError(f"{self.name}: no point of the bore surface lies in an active element")
        vec = np.zeros((len(P), info.dim))
        vec[ok] = _bearing_vectors(er[ok], dA[ok], F, a, self.mode, self.name)
        n, f, _ = _scatter(info, P, vec)
        lost = 1.0 - dA[ok].sum() / dA.sum()
        if lost > 0.02:
            notes.append(f"info: {lost:.0%} of the bore surface lies outside the part (dropped, load "
                         "renormalised)")
        return Applied(self.name, self.kind, n, f, F, center, None, notes)


class Torque(RemoteMoment):
    """Torque about a bore axis (tangential traction, rigid-body distribution about the axis, zero net force).
    3-D: ``torque`` scalar about the bore axis (right-hand rule); 2-D: scalar about the out-of-plane axis."""

    kind = "torque"

    def __init__(self, bore: Bore | OnCylinder, torque: float, name: str | None = None, **select_kw: Any) -> None:
        sel = bore if isinstance(bore, OnCylinder) else OnCylinder.from_bore(bore, facing="inward", **select_kw)
        center = sel.p0 + (0.5 * sel.length * sel.axis if sel.axis is not None else 0.0)
        M = float(torque) * sel.axis if sel.axis is not None else [float(torque)]
        super().__init__(sel, M, center, "area", name or f"Torque {sel!r}")


def body_force_matrix(domain: Domain, acceleration: Sequence[float], density: float = 1.0) -> sp.csr_matrix:
    """(ndof, n_active) map from active-element density x̄ to nodal body forces: column e holds
    ρ frac_e V_e a / 2^dim on each corner (consistent for a constant body force on Q4 / H8)."""
    info = part_info(domain)
    a = _vec(acceleration, domain.dim, "acceleration")
    act = domain.active_elements
    m = density * domain.frac.ravel()[act] * info.elem_volume / info.nper
    edof = domain.edof[act]                                          # (na, nloc), node-major
    rows = edof.ravel()
    cols = np.repeat(np.arange(len(act)), edof.shape[1])
    vals = (m[:, None] * np.tile(a, info.nper)[None, :]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(domain.ndof, len(act)))


class BodyForce(Load):
    """Design-independent body force ρ a on every active element with full density (e.g. a non-design mass).
    For self-weight that follows the design use :class:`BodyForceCompliance`."""

    kind = "body force"

    def __init__(self, acceleration: Sequence[float], density: float, name: str | None = None) -> None:
        self.a, self.rho = acceleration, float(density)
        self.name = name or "BodyForce"

    def apply(self, info: PartInfo) -> Applied:
        d = info.domain
        G = body_force_matrix(d, self.a, self.rho)
        Fv = np.asarray(G.sum(axis=1)).ravel().reshape(d.nn, d.dim)
        n = np.flatnonzero(info.node_active)
        mass = self.rho * float(d.frac[d.active].sum()) * info.elem_volume
        return Applied(self.name, self.kind, n, Fv[n], mass * _vec(self.a, d.dim, "acceleration"),
                       info.X[n].mean(axis=0), None, [])


# =============================================================================================== design-dependent load
class _LoadSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E_active: torch.Tensor, F: torch.Tensor, model: Elasticity) -> torch.Tensor:  # type: ignore[override]
        E_np = E_active.detach().cpu().double().numpy()
        F_np = F.detach().cpu().double().numpy()
        U = model.solve_numpy(E_np, F_np)
        ctx.model, ctx.U, ctx.E, ctx.token = model, U, E_np, model._token
        return torch.from_numpy(U).to(dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, Any, None]:  # type: ignore[override]
        model = ctx.model
        if model._token != ctx.token:
            model.factor(ctx.E)
        G = gU.detach().cpu().double().numpy()
        lam = model._solve_free(G)
        model.n_adjoint_solves += 1
        dE = -model.geom * model.element_energy(ctx.U, lam).sum(axis=1)
        return torch.from_numpy(dE), torch.from_numpy(lam), None


def design_load_solve(physics: Elasticity, E_active: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Differentiable ``U = K(E)⁻¹ F`` where the load ``F`` (ndof, n_cases) is a tensor too (∂/∂F = λ, the adjoint)."""
    F2 = F[:, None] if F.ndim == 1 else F
    return _LoadSolve.apply(E_active, F2, physics)


class BodyForceCompliance(Response):
    """Compliance with a design-dependent body force: ``Σ_c w_c F_cᵀ U_c``, ``F_c = F0_c + s_c G x̄``, ``U_c = K⁻¹ F_c``.

    ``G`` from :func:`body_force_matrix` (mass ∝ x̄, linear); ``factors`` s_c per case (default 1). The response
    solves itself (``needs`` is empty, so the study skips its own solve). With SIMP the stiffness-to-mass ratio
    vanishes at low density (x̄^(p−1)); a modified interpolation at low density is usually needed (Bruyneel &
    Duysinx 2005 — source not read in this session, not in the registry).
    """

    name = "compliance_bodyforce"
    needs: frozenset[str] = frozenset()

    def __init__(self, F0: np.ndarray, G: sp.spmatrix, factors: Sequence[float] | None = None,
                 weights: Sequence[float] | None = None) -> None:
        F0 = np.asarray(F0, float)
        self.F0 = F0[:, None] if F0.ndim == 1 else F0
        nc = self.F0.shape[1]
        self.factors = np.ones(nc) if factors is None else np.asarray(factors, float).reshape(nc)
        self.weights = np.ones(nc) if weights is None else np.asarray(weights, float).reshape(nc)
        Gc = sp.coo_matrix(G)
        self._G = torch.sparse_coo_tensor(np.vstack([Gc.row, Gc.col]), Gc.data, Gc.shape, dtype=torch.float64)

    def loads(self, ctx: EvalContext) -> torch.Tensor:
        act = torch.as_tensor(ctx.domain.active_elements, dtype=torch.long)
        xa = ctx.xphys.reshape(-1)[act]
        g = torch.sparse.mm(self._G, xa[:, None])                   # (ndof, 1)
        return torch.as_tensor(self.F0) + g * torch.as_tensor(self.factors)[None, :]

    def __call__(self, ctx: EvalContext) -> torch.Tensor:
        if ctx.physics is None:
            raise ValueError("BodyForceCompliance needs ctx.physics")
        act = torch.as_tensor(ctx.domain.active_elements, dtype=torch.long)
        F = self.loads(ctx)
        U = design_load_solve(ctx.physics, ctx.E.reshape(-1)[act], F)
        return ((F * U).sum(dim=0) * torch.as_tensor(self.weights)).sum()


# =============================================================================================== cases and container
@dataclass
class LoadCase:
    """Named load case; ``supports`` None = the container's default supports."""

    name: str
    loads: Sequence[Load]
    supports: Sequence[Support] | None = None


@dataclass
class Combination:
    """Linear combination of load cases: column = Σ factor × case column (cases must share their supports)."""

    name: str
    factors: dict[str, float]


@dataclass
class Issue:
    level: str                  # "error" | "warning" | "info"
    code: str
    message: str
    case: str | None = None

    def __str__(self) -> str:
        where = f" [{self.case}]" if self.case else ""
        return f"{self.level.upper():7s} {self.code}{where}: {self.message}"


@dataclass
class CheckReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, level: str | None = None) -> list[str]:
        return [i.code for i in self.issues if level is None or i.level == level]

    def raise_on_error(self) -> None:
        if self.errors:
            raise ValueError("boundary-condition check failed:\n" + "\n".join(str(i) for i in self.errors))

    def __str__(self) -> str:
        head = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([head] + [str(i) for i in self.issues])


def _rb_basis(X: np.ndarray, c: np.ndarray, L: float) -> np.ndarray:
    """(n, dim, n_rb) rigid-body displacement of nodes X for θ = (translation, rotation·L) about c."""
    r = (X - c) / L
    n, dim = X.shape
    if dim == 3:
        Phi = np.zeros((n, 3, 6))
        Phi[:, :, :3] = np.eye(3)
        Phi[:, 0, 4], Phi[:, 0, 5] = r[:, 2], -r[:, 1]
        Phi[:, 1, 3], Phi[:, 1, 5] = -r[:, 2], r[:, 0]
        Phi[:, 2, 3], Phi[:, 2, 4] = r[:, 1], -r[:, 0]
    else:
        Phi = np.zeros((n, 2, 3))
        Phi[:, :, :2] = np.eye(2)
        Phi[:, 0, 2], Phi[:, 1, 2] = -r[:, 1], r[:, 0]
    return Phi


def _describe_mode(theta: np.ndarray, c: np.ndarray, L: float, dim: int) -> str:
    lam = theta[:dim]
    mu = theta[dim:] / L
    fmt = lambda v: "(" + ", ".join(f"{x:.3g}" for x in np.where(np.abs(v) < 1e-9, 0.0, v)) + ")"  # noqa: E731

    def canon(v: np.ndarray) -> np.ndarray:                  # a mode has no sign: largest component positive
        v = v / np.linalg.norm(v)
        return -v if v[int(np.argmax(np.abs(v)))] < 0 else v
    if np.linalg.norm(mu) * L < 1e-6 * np.linalg.norm(theta):
        return f"translation along {fmt(canon(lam))}"
    if dim == 2:
        p = c + np.array([-lam[1], lam[0]]) / mu[0]
        return f"rotation about the point {fmt(p)}"
    m2 = float(mu @ mu)
    p = c + np.cross(mu, lam) / m2
    pitch = float(lam @ mu) / m2
    s = f"rotation about the axis {fmt(canon(mu))} through {fmt(p)}"
    return s + (f" with axial translation (pitch {pitch:.3g})" if abs(pitch) > 1e-6 * L else "")


class BoundaryConditions:
    """Supports, named load cases and combinations on a domain; builds ``F``, the physics and the checks.

    Parameters
    ----------
    domain : Domain
    supports : default supports of every case without its own
    cases : load cases (names unique)
    combinations : optional linear combinations; :meth:`F` then returns cases followed by combinations
    """

    def __init__(self, domain: Domain, supports: Sequence[Support], cases: Sequence[LoadCase],
                 combinations: Sequence[Combination] = ()) -> None:
        self.domain = domain
        self.info = part_info(domain)
        self.supports = list(supports)
        self.cases = list(cases)
        self.combinations = list(combinations)
        names = [c.name for c in self.cases] + [c.name for c in self.combinations]
        if len(set(names)) != len(names):
            raise ValueError(f"load case / combination names must be unique: {names}")
        if not self.cases:
            raise ValueError("at least one load case is required")
        self.applied: dict[str, list[Applied]] = {}
        for c in self.cases:
            out = []
            for ld in c.loads:
                try:
                    out.append(ld.apply(self.info))
                except EmptySelectionError as exc:
                    raise EmptySelectionError(f"load case {c.name!r}, load {ld.name!r}: {exc}") from None
            self.applied[c.name] = out
        # support sets, identified by what they constrain on this part (fixed DOFs, local frames, springs), not by
        # the identity of the support objects: equal supports built separately are one set (one factorisation, and
        # combinations of their cases are valid)
        self._sets: list[tuple[bytes, list[Support]]] = []
        self._case_set: list[int] = []
        self._cs: dict[int, ConstraintSet] = {}
        resolved: dict[tuple[int, ...], ConstraintSet] = {}
        for c in self.cases:
            sup = list(c.supports) if c.supports is not None else self.supports
            if not sup:
                raise ValueError(f"load case {c.name!r} has no supports")
            ids = tuple(id(s) for s in sup)
            if ids not in resolved:
                try:
                    resolved[ids] = _resolve(sup, self.info)
                except EmptySelectionError as exc:
                    raise EmptySelectionError(f"supports of case {c.name!r}: {exc}") from None
            cs = resolved[ids]
            key = _constraint_key(cs)
            for k, (kk, _) in enumerate(self._sets):
                if kk == key:
                    self._case_set.append(k)
                    break
            else:
                self._sets.append((key, sup))
                self._cs[len(self._sets) - 1] = cs
                self._case_set.append(len(self._sets) - 1)
        idx = {c.name: i for i, c in enumerate(self.cases)}
        for cb in self.combinations:
            miss = [n for n in cb.factors if n not in idx]
            if miss:
                raise ValueError(f"combination {cb.name!r}: unknown case(s) {miss}")
            sets = {self._case_set[idx[n]] for n in cb.factors}
            if len(sets) > 1:
                raise ValueError(f"combination {cb.name!r} mixes cases with different supports - superposition "
                                 "is only valid for one support set")

    # ------------------------------------------------------------------ assembly
    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cases] + [c.name for c in self.combinations]

    def case_vector(self, name: str) -> np.ndarray:
        """(ndof,) load vector of one case."""
        d = self.domain
        f = np.zeros(d.ndof)
        for a in self.applied[name]:
            np.add.at(f, (d.dim * a.nodes[:, None] + np.arange(d.dim)).ravel(), a.forces.ravel())
        return f

    def F(self, which: str = "all") -> np.ndarray:
        """(ndof, n) load array: "cases", "combinations" or "all" (cases then combinations)."""
        cols = [self.case_vector(c.name) for c in self.cases]
        base = np.column_stack(cols)
        idx = {c.name: i for i, c in enumerate(self.cases)}
        comb = [sum(f * base[:, idx[n]] for n, f in cb.factors.items()) for cb in self.combinations]
        if which == "cases":
            return base
        if which == "combinations":
            if not comb:
                raise ValueError("no combinations defined")
            return np.column_stack(comb)
        if which != "all":
            raise ValueError("which: 'all' | 'cases' | 'combinations'")
        return np.column_stack(cols + comb)

    def _column_sets(self, which: str = "all") -> list[int]:
        idx = {c.name: i for i, c in enumerate(self.cases)}
        sets = list(self._case_set)
        comb = [self._case_set[idx[next(iter(cb.factors))]] for cb in self.combinations]
        return sets if which == "cases" else (comb if which == "combinations" else sets + comb)

    def constraints(self, case: str | int = 0) -> ConstraintSet:
        """Resolved supports of a load case (name or index)."""
        k = case if isinstance(case, int) else [c.name for c in self.cases].index(case)
        return self._cs[self._case_set[k]]

    def fixed_dofs(self, case: str | int = 0) -> np.ndarray:
        """Globally fixed DOFs of a case (oblique constraints and springs are not included: see ``constraints``)."""
        return self.constraints(case).fixed

    def _make(self, cs: ConstraintSet, nu: float, element: str | None, solver: Any) -> Elasticity:
        if cs.needs_extension:
            return ConstrainedElasticity(self.domain, cs, nu=nu, element=element, solver=solver)
        return Elasticity(self.domain, cs.fixed, nu=nu, element=element, solver=solver)

    def physics_per_support_set(self, nu: float = 0.3, element: str | None = None,
                                solver_factory: Callable[[], Any] | None = None,
                                which: str = "all") -> list[tuple[Elasticity, np.ndarray]]:
        """One physics per distinct support set with the column indices of ``F(which)`` it solves."""
        sets = self._column_sets(which)
        out = []
        for k in sorted(set(sets)):
            cols = np.flatnonzero(np.asarray(sets) == k)
            out.append((self._make(self._cs[k], nu, element, solver_factory() if solver_factory else None), cols))
        return out

    def physics(self, nu: float = 0.3, element: str | None = None,
                solver_factory: Callable[[], Any] | None = None, which: str = "all") -> Any:
        """Physics for ``Study``: Elasticity / ConstrainedElasticity, or PerCaseSupports for several support sets."""
        groups = self.physics_per_support_set(nu, element, solver_factory, which)
        return groups[0][0] if len(groups) == 1 else PerCaseSupports(groups)

    # ------------------------------------------------------------------ non-design regions
    def bc_nodes(self, supports: bool = True, loads: bool = True) -> np.ndarray:
        parts = []
        if supports:
            for cs in self._cs.values():
                parts += [p[2] for p in cs.per_support]
        if loads:
            for lst in self.applied.values():
                parts += [a.nodes for a in lst if a.kind != "body force"]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def keep_mask(self, thickness: float | None = None, supports: bool = True, loads: bool = True) -> np.ndarray:
        """Frozen pads / rings: active elements whose centre lies within ``thickness`` (≥ 2 h, default 2 h) of a
        support or load node. Returns a bool array of ``domain.shape`` (pass as ``Domain(keep=…)``)."""
        d = self.domain
        t = max(2.0 * d.h if thickness is None else float(thickness), 2.0 * d.h)
        nodes = self.bc_nodes(supports, loads)
        out = np.zeros(d.ne, bool)
        if len(nodes):
            act = d.active_elements
            dist, _ = cKDTree(self.info.X[nodes]).query(d.element_centers()[act], distance_upper_bound=t + 1e-9 * d.h)
            out[act[dist <= t + 1e-9 * d.h]] = True
        return out.reshape(d.shape)

    # ------------------------------------------------------------------ checks
    def _rigid_modes(self, cs: ConstraintSet) -> list[tuple[int, list[str]]]:
        """Per face-connected component: (number of nodes, descriptions of unrestrained rigid-body modes)."""
        from scipy import ndimage
        d, info = self.domain, self.info
        lab, ncomp = ndimage.label(d.active)
        lab_act = lab.ravel()[d.active_elements]
        cn, cd = cs.constrained_directions(d.dim)
        out = []
        for k in range(1, ncomp + 1):
            nodes = np.unique(info.elem_nodes[lab_act == k])
            inset = np.zeros(d.nn, bool)
            inset[nodes] = True
            X = info.X[nodes]
            c = X.mean(axis=0)
            L = max(float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))), d.h)
            rows = []
            m = inset[cn] if len(cn) else np.zeros(0, bool)
            if m.any():
                rows.append(np.einsum("ia,iak->ik", cd[m], _rb_basis(info.X[cn[m]], c, L)))
            if len(cs.spring_nodes):
                ms = inset[cs.spring_nodes]
                if ms.any():
                    K = cs.spring_K[ms]
                    tr = np.maximum(np.trace(K, axis1=1, axis2=2), 1e-300)
                    Phi = _rb_basis(info.X[cs.spring_nodes[ms]], c, L)
                    rows.append(np.einsum("iab,ibk->iak", K / tr[:, None, None], Phi).reshape(-1, Phi.shape[2]))
            nrb = 6 if d.dim == 3 else 3
            if rows:
                C = np.concatenate(rows)
                _, s, Vt = np.linalg.svd(C, full_matrices=True)
                rank = int((s > 1e-8 * max(s[0], 1e-300)).sum())
                N = Vt[rank:]
            else:
                N = np.eye(nrb)
            descr = []
            if len(N):
                # readable basis: reduced row echelon form of the null space
                R = N.copy()
                piv_row = 0
                for col in range(nrb):
                    if piv_row >= len(R):
                        break
                    p = piv_row + int(np.argmax(np.abs(R[piv_row:, col])))
                    if abs(R[p, col]) < 1e-9:
                        continue
                    R[[piv_row, p]] = R[[p, piv_row]]
                    R[piv_row] /= R[piv_row, col]
                    for r_ in range(len(R)):
                        if r_ != piv_row:
                            R[r_] -= R[r_, col] * R[piv_row]
                    piv_row += 1
                descr = [_describe_mode(th, c, L, d.dim) for th in R]
            out.append((len(nodes), descr))
        return out

    def check(self, rtol: float = 1e-9) -> CheckReport:
        """All checks (see module notes); the report lists errors, warnings and infos."""
        rep = CheckReport()
        d, info = self.domain, self.info
        dim = d.dim
        add = rep.issues.append
        for ci, c in enumerate(self.cases):
            cs = self._cs[self._case_set[ci]]
            fvec = self.case_vector(c.name)
            for a in self.applied[c.name]:
                Fi = a.intended_force
                scale = np.linalg.norm(Fi) + (np.linalg.norm(a.intended_moment) / info.L
                                              if a.intended_moment is not None else 0.0)
                scale = max(scale, 1e-300)
                eF = np.linalg.norm(a.force - Fi) / scale
                if eF > rtol:
                    add(Issue("error", "resultant-force", f"{a.name}: applied {a.force} != intended {Fi} "
                              f"(rel. {eF:.2e})", c.name))
                if a.intended_moment is not None:
                    eM = np.linalg.norm(a.moment(X=info.X) - a.intended_moment) / info.L / scale
                    if eM > rtol:
                        add(Issue("error", "resultant-moment", f"{a.name}: applied moment {a.moment(X=info.X)} != "
                                  f"intended {a.intended_moment} (rel. {eM:.2e})", c.name))
                dead = a.nodes[~info.node_active[a.nodes]]
                if len(dead):
                    add(Issue("error", "inactive-node", f"{a.name}: {len(dead)} node(s) are not part of the active "
                              "domain - the solver drops their forces", c.name))
                if len(a.nodes) == 1:
                    add(Issue("warning", "single-node-load", f"{a.name}: applied on one node (singular stress, grey "
                              "load blob); use PointLoad/SurfaceForce", c.name))
                touched_void = a.nodes[np.isin(a.nodes, self._void_nodes())]
                if len(touched_void) and a.kind != "body force":
                    add(Issue("warning", "load-on-void", f"{a.name}: {len(touched_void)} node(s) only touch void "
                              "(forced empty) elements", c.name))
                for note in a.notes:
                    lvl = "info" if note.startswith("info: ") else "warning"
                    add(Issue(lvl, "load-note", f"{a.name}: {note.removeprefix('info: ')}", c.name))
            # loads on constrained directions (the solver removes them)
            cn, cd = cs.constrained_directions(dim)
            if len(cn):
                fnod = fvec.reshape(d.nn, dim)
                comp = np.einsum("ia,ia->i", cd, fnod[cn])
                lost = (comp[:, None] * cd).sum(axis=0)
                tot = np.abs(fvec).sum()
                if tot > 0 and np.abs(comp).sum() > 1e-12 * tot:
                    add(Issue("warning", "load-on-support", f"{np.abs(comp).sum() / tot:.1%} of the nodal load acts "
                              f"on constrained directions and is taken by the support (resultant lost {lost})",
                              c.name))
        for k in range(len(self._sets)):
            cs = self._cs[k]
            cases = [c.name for c, s in zip(self.cases, self._case_set, strict=True) if s == k]
            tag = ", ".join(cases)
            for nn_, modes in self._rigid_modes(cs):
                if modes:
                    add(Issue("error", "rigid-body", f"{len(modes)} unrestrained rigid-body mode(s) of a component "
                              f"with {nn_} nodes: " + "; ".join(modes), tag))
            if len(cs.multiply_constrained):
                add(Issue("warning", "overlapping-supports", f"{len(cs.multiply_constrained)} node(s) belong to more "
                          f"than one support ({cs.redundant_rows} redundant constraint row(s)); check that this "
                          "is intended", tag))
            ns = int(self.info.surface.sum())
            ncon = len(np.unique(np.concatenate([p[2] for p in cs.per_support])))
            if ncon > 0.5 * ns:
                add(Issue("warning", "over-constrained", f"supports cover {ncon / ns:.0%} of the surface nodes - "
                          "a stiff, over-constrained model", tag))
            for name, _kind, nodes, _ in cs.per_support:
                if np.isin(nodes, self._void_nodes()).any():
                    add(Issue("warning", "support-on-void", f"{name}: nodes that only touch void elements", tag))
        return rep

    def _void_nodes(self) -> np.ndarray:
        d = self.domain
        if not d.void.any():
            return np.zeros(0, np.int64)
        act = d.active_elements
        en = self.info.elem_nodes
        solidish = ~d.void.ravel()[act]
        good = np.zeros(d.nn, bool)
        good[en[solidish].ravel()] = True
        return np.flatnonzero(self.info.node_active & ~good)

    # ------------------------------------------------------------------ summary
    def summary(self, rtol: float = 1e-9) -> str:
        """Human-readable tables: per load case the loads (intended vs applied resultant, moment about the load's
        reference point, relative error), per support set the supports; last line: verdict and next step."""
        info = self.info
        rep = self.check(rtol)
        lines: list[str] = []
        for c in self.cases:
            rows, errs = [], []
            for a in self.applied[c.name]:
                F, Fi = a.force, a.intended_force
                M = a.moment(X=info.X)
                scale = max(np.linalg.norm(Fi) + (np.linalg.norm(a.intended_moment) / info.L
                                                  if a.intended_moment is not None else 0.0), 1e-300)
                e = np.linalg.norm(F - Fi) / scale
                if a.intended_moment is not None:
                    e = max(e, np.linalg.norm(M - a.intended_moment) / info.L / scale)
                errs.append(e)
                rows.append([a.name, a.kind, len(a.nodes), Fi, F, M, e])
            if not rows:                                   # a case without loads (zero load vector)
                lines += [f'Load case "{c.name}": no loads', ""]
                continue
            ok = max(errs) <= rtol
            lines.append(f'Load case "{c.name}": {len(rows)} load(s), resultants '
                         f'{"match" if ok else "DO NOT match"} the intended values (max rel. error {max(errs):.1e})')
            lines.append(_format_loads(rows, info.L))
            lines.append("")
        for k, (_, sup) in enumerate(self._sets):
            cs = self._cs[k]
            cases = [c.name for c, s in zip(self.cases, self._case_set, strict=True) if s == k]
            free = sum(len(m) for _, m in self._rigid_modes(cs))
            lines.append(f"Supports of {', '.join(cases)}: {len(sup)} support(s), "
                         f"unrestrained rigid-body modes: {free}")
            rows2 = []
            for name, kind, nodes, nrows in cs.per_support:
                rows2.append([name, kind, len(nodes), nrows])
            lines.append(_table(["Support", "Kind", "Nodes", "Constraint rows"], rows2, "llrr"))
            lines.append(f"  fixed DOFs {len(cs.fixed)}; oblique-frame nodes {len(cs.frames)}; spring nodes "
                         f"{len(cs.spring_nodes)}")
            lines.append("")
        verdict = "OK" if rep.ok else "NOT OK"
        nxt = "print(bc.check()) for the messages" if rep.issues else "build the physics with bc.physics()"
        lines.append(f"Check {verdict}: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s) - next: {nxt}.")
        return "\n".join(lines)


def _fmt_vec(v: np.ndarray, d: int) -> str:
    return "(" + ", ".join(f"{x:.{d}f}" for x in np.where(np.abs(v) < 0.5 * 10.0 ** (-d), 0.0, v)) + ")"


def _decimals(vals: list[np.ndarray]) -> int:
    m = max([float(np.abs(v).max()) for v in vals if np.size(v)] + [0.0])
    return 0 if m == 0 else int(max(0, 3 - math.floor(math.log10(m))))


def _format_loads(rows: list[list[Any]], L: float) -> str:
    dF = _decimals([r[3] for r in rows] + [r[4] for r in rows])
    fmax = max(float(np.abs(np.r_[r[3], r[4]]).max()) for r in rows)
    dM = _decimals([r[5] for r in rows] + [np.array([fmax * L])])    # moments shown on the scale F*L
    body = [[r[0], r[1], r[2], _fmt_vec(r[3], dF), _fmt_vec(r[4], dF), _fmt_vec(r[5], dM), f"{r[6]:.1e}"]
            for r in rows]
    return _table(["Load", "Kind", "Nodes", "F intended [F]", "F applied [F]", "M about ref. [F*L]", "Rel. error"],
                  body, "llrrrrr")


def _table(headers: list[str], rows: list[list[Any]], align: str) -> str:
    """Plain-text table: two-space gutters, no vertical rules, numbers right-aligned."""
    cells = [[str(x) for x in r] for r in rows]
    for r in cells:
        r[0] = r[0] if len(r[0]) <= 38 else r[0][:35] + "..."
    w = [max(len(headers[k]), *(len(r[k]) for r in cells)) if cells else len(headers[k]) for k in range(len(headers))]
    just = lambda s, k: s.ljust(w[k]) if align[k] == "l" else s.rjust(w[k])  # noqa: E731
    out = ["  " + "  ".join(just(h, k) for k, h in enumerate(headers)),
           "  " + "  ".join("-" * w[k] for k in range(len(headers)))]
    out += ["  " + "  ".join(just(s, k) for k, s in enumerate(r)) for r in cells]
    return "\n".join(out)


def clearance_mask(domain: Domain, shape: Shape, threshold: float = 0.5) -> np.ndarray:
    """Void mask for a clearance volume (bolt head, tool access): active elements more than ``threshold`` covered
    by ``shape``. Pass as ``Domain(void=…)``."""
    from toporlogy.geometry import coverage
    return (coverage(shape, domain) > threshold) & domain.active
