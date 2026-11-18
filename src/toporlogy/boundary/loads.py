"""Loads: nodal, remote, point, surface, traction, pressure, bearing, torque, body force.
"""
from __future__ import annotations

import math
import scipy.sparse as sp
import torch
from collections.abc import Sequence
from dataclasses import dataclass, field
from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import Elasticity
from toporlogy.geometry import Bore
from toporlogy.optimization.responses import EvalContext, Response
from typing import Any

import numpy as np

from toporlogy.boundary.constrained import PerCaseSupports
from toporlogy.boundary.part import EmptySelectionError, PartInfo, _cross, _mom, _vec, part_info
from toporlogy.boundary.selectors import InSphere, NodeIds, OnCylinder, Selector


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
    scale = max(float(np.linalg.norm(target)), 1e-300)
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

    def __init__(self, where: Selector, force: Sequence[float], point: Sequence[float] | np.ndarray,
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

    def __init__(self, where: Selector, moment: Any, point: Sequence[float] | np.ndarray | None = None,
                 weights: str = "area", name: str | None = None) -> None:
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

    def __init__(self, point: Sequence[float] | np.ndarray, force: Sequence[float], radius: float | None = None,
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
    def forward(ctx: Any, E_active: torch.Tensor, F: torch.Tensor, model: Elasticity) -> torch.Tensor:
        E_np = E_active.detach().cpu().double().numpy()
        F_np = F.detach().cpu().double().numpy()
        U = model.solve_numpy(E_np, F_np)
        ctx.model, ctx.U, ctx.E, ctx.token = model, U, E_np, model._token
        return torch.from_numpy(U).to(dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, Any, None]:
        model = ctx.model
        if model._token != ctx.token:
            model.factor(ctx.E)
        G = gU.detach().cpu().double().numpy()
        lam = model._solve_free(G)
        model.n_adjoint_solves += 1
        dE = -model.geom * model.element_energy(ctx.U, lam).sum(axis=1)
        return torch.from_numpy(dE), torch.from_numpy(lam), None


def design_load_solve(physics: Elasticity, E_active: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Differentiable ``U = K(E)⁻¹ F`` where the load ``F`` (ndof, n_cases) is a tensor too (∂/∂F = λ, the adjoint).

    With :class:`PerCaseSupports` (cases with different supports) each column group is solved by its own physics and
    the columns are put back in case order (torch ops, so the gradient flows through both)."""
    F2 = F[:, None] if F.ndim == 1 else F
    if isinstance(physics, PerCaseSupports):
        if F2.shape[1] != physics.n_cases:
            raise ValueError(f"F has {F2.shape[1]} columns, the support groups cover {physics.n_cases}")
        parts = [design_load_solve(p, E_active, F2[:, torch.as_tensor(c, device=F2.device)]) for p, c in physics.groups]
        return torch.cat(parts, dim=1).index_select(1, physics._inv.to(F2.device))
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
        # torch accepts array_like indices at run time; its stub declares only Tensor
        self._G = torch.sparse_coo_tensor(np.vstack([Gc.row, Gc.col]),  # type: ignore[arg-type]
                                          Gc.data, Gc.shape, dtype=torch.float64)

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
