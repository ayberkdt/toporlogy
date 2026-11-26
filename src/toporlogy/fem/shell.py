"""Flat-facet shell of rectangular facets in 3-D space (six DOFs per node, global axes).

Per node (u, v, w, θx, θy, θz): displacement and right-hand rotation vector in global axes. Each facet is an a × b
rectangle with corners X1..X4 in the corner order (−1,−1), (1,−1), (1,1), (−1,1); its frame is e_x = (X2 − X1)/a,
e_y = (X4 − X1)/b, e_z = e_x × e_y, and local DOFs are R d per node with R = [e_x; e_y; e_z] (rows), so
K_global = Tᵀ K_local T. In the facet frame:

* membrane: bilinear plane stress, D = t/(1 − ν²) [[1, ν, 0], [ν, 1, 0], [0, 0, (1 − ν)/2]], 2 × 2 Gauss;
* bending: Mindlin–Reissner as in :mod:`toporlogy.fem.plate` generalised to rectangles (plate DOFs (w, βx, βy),
  γ = ∇w − β, bending 2 × 2, transverse shear k_s G t with one point). The plate rotations are the slopes of the
  normal, βx = −θy and βy = θx (a rigid rotation θ moves r by θ × r, so it gives γ = 0 only with these signs);
* drilling: Hughes–Brezzi type penalty ½ γ_d ∫ (θz − ω)² dA with ω = ½ (∂v/∂x − ∂u/∂y), bilinear θz, 2 × 2 Gauss,
  γ_d = ``drill`` · G · t (G = 1/(2(1 + ν)), default ``drill`` = 1e-1: the value closest to a 3-D solid reference of a
  clamped curved segment, DB0-DB5 in ``docs/olcumler/kabuk_onkayit.md``; it was 1e-3 before). A rigid in-plane rotation
  (ω = θz) costs no energy; the term keeps θz of a flat sheet from being singular.

Element matrices are for unit modulus; :class:`FlatShell` assembles K = Σ_e E_e K_e⁰ and solves with fixed DOFs as a
torch autograd function (dc/dE_e = −λ_eᵀ K_e⁰ u_e), so a design gradient is available. Pre-registered checks:
KB1-KB11 in ``docs/olcumler/kabuk_onkayit.md``. Not here: Study / YAML integration, non-rectangular facets.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from toporlogy.fem.plate import SHEAR_FACTOR

__all__ = ["ShellMesh", "FlatShell", "facet_frame", "membrane_matrix", "bending_matrix", "drilling_matrix",
           "facet_matrix", "flat_rectangle", "cylinder_surface", "DRILL"]

DRILL = 1e-1
_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_G = 1.0 / np.sqrt(3.0)
_GAUSS = [(p * _G, q * _G) for p in (-1, 1) for q in (-1, 1)]
_RECT_TOL = 1e-9


def _check(t: float, nu: float) -> None:
    if not (np.isfinite(t) and t > 0):
        raise ValueError(f"shell thickness t must be > 0, got {t}")
    if not -1.0 < nu < 0.5:
        raise ValueError(f"Poisson ratio must be in (-1, 0.5), got {nu}")


def _check_drill(drill: float) -> None:
    if not (np.isfinite(drill) and drill >= 0):
        raise ValueError(f"drilling scale must be finite and >= 0, got {drill}")


def _shape(xi: float, eta: float, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """N (4,) and ∂N/∂(x, y) (2, 4) at (ξ, η) of an a × b rectangle."""
    c = _CORNERS
    N = (1 + c[:, 0] * xi) * (1 + c[:, 1] * eta) / 4.0
    dxi = c[:, 0] * (1 + c[:, 1] * eta) / 4.0
    deta = c[:, 1] * (1 + c[:, 0] * xi) / 4.0
    return N, np.vstack([dxi * (2.0 / a), deta * (2.0 / b)])


def _check_ab(a: float, b: float) -> None:
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        raise ValueError(f"facet edges must be > 0, got a={a}, b={b}")


def membrane_matrix(a: float, b: float, t: float, nu: float) -> np.ndarray:
    """(8, 8) unit-modulus plane-stress stiffness of an a × b rectangle of thickness t; DOFs (u, v) per corner."""
    _check(t, nu)
    _check_ab(a, b)
    D = t / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    K = np.zeros((8, 8))
    for xi, eta in _GAUSS:
        _, dN = _shape(xi, eta, a, b)
        B = np.zeros((3, 8))
        B[0, 0::2] = dN[0]
        B[1, 1::2] = dN[1]
        B[2, 0::2] = dN[1]
        B[2, 1::2] = dN[0]
        K += B.T @ D @ B * (a * b / 4.0)
    return 0.5 * (K + K.T)


def bending_matrix(a: float, b: float, t: float, nu: float) -> np.ndarray:
    """(12, 12) unit-modulus Mindlin plate stiffness of an a × b rectangle; DOFs (w, βx, βy) per corner, bending
    2 × 2 Gauss, transverse shear one point."""
    _check(t, nu)
    _check_ab(a, b)
    Db = t ** 3 / (12.0 * (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ds = SHEAR_FACTOR * t / (2.0 * (1 + nu)) * np.eye(2)
    detJ = a * b / 4.0
    K = np.zeros((12, 12))
    for xi, eta in _GAUSS:
        _, dN = _shape(xi, eta, a, b)
        B = np.zeros((3, 12))
        B[0, 1::3] = dN[0]
        B[1, 2::3] = dN[1]
        B[2, 1::3] = dN[1]
        B[2, 2::3] = dN[0]
        K += B.T @ Db @ B * detJ
    N, dN = _shape(0.0, 0.0, a, b)
    Bs = np.zeros((2, 12))
    Bs[0, 0::3] = dN[0]
    Bs[0, 1::3] = -N
    Bs[1, 0::3] = dN[1]
    Bs[1, 2::3] = -N
    K += Bs.T @ Ds @ Bs * 4.0 * detJ
    return 0.5 * (K + K.T)


def drilling_matrix(a: float, b: float, t: float, nu: float, drill: float = DRILL) -> np.ndarray:
    """(12, 12) unit-modulus drilling penalty of an a × b rectangle; DOFs (u, v, θz) per corner (module notes)."""
    _check(t, nu)
    _check_ab(a, b)
    _check_drill(drill)
    gamma = drill * t / (2.0 * (1 + nu))
    K = np.zeros((12, 12))
    for xi, eta in _GAUSS:
        N, dN = _shape(xi, eta, a, b)
        g = np.zeros(12)
        g[0::3] = 0.5 * dN[1]                     # θz − ω = Σ N θz − ½ ∂v/∂x + ½ ∂u/∂y
        g[1::3] = -0.5 * dN[0]
        g[2::3] = N
        K += gamma * np.outer(g, g) * (a * b / 4.0)
    return 0.5 * (K + K.T)


def _local_matrix(a: float, b: float, t: float, nu: float, drill: float) -> np.ndarray:
    """(24, 24) facet stiffness in the facet frame, local DOFs (u, v, w, θx, θy, θz) per corner."""
    K = np.zeros((24, 24))
    mem = 6 * np.arange(4)[:, None] + np.array([0, 1])[None, :]
    im = mem.ravel()
    K[np.ix_(im, im)] += membrane_matrix(a, b, t, nu)
    P = np.zeros((12, 24))                        # (w, βx, βy) = P d_local: βx = −θy, βy = θx
    for i in range(4):
        P[3 * i, 6 * i + 2] = 1.0
        P[3 * i + 1, 6 * i + 4] = -1.0
        P[3 * i + 2, 6 * i + 3] = 1.0
    K += P.T @ bending_matrix(a, b, t, nu) @ P
    idr = (6 * np.arange(4)[:, None] + np.array([0, 1, 5])[None, :]).ravel()
    K[np.ix_(idr, idr)] += drilling_matrix(a, b, t, nu, drill)
    return K


def facet_frame(X: Any) -> tuple[np.ndarray, float, float]:
    """(R (3, 3) with rows e_x, e_y, e_z; a; b) of a rectangular facet with corners ``X`` (4, 3); a ValueError for a
    degenerate or non-rectangular (skew, warped) facet."""
    X = np.asarray(X, float)
    if X.shape != (4, 3) or not np.isfinite(X).all():
        raise ValueError(f"facet corners must be a finite (4, 3) array, got shape {X.shape}")
    e1, e4 = X[1] - X[0], X[3] - X[0]
    a, b = float(np.linalg.norm(e1)), float(np.linalg.norm(e4))
    L = max(a, b)
    if not L > 0 or min(a, b) <= _RECT_TOL * L:
        raise ValueError(f"degenerate facet: edge lengths a={a:.3e}, b={b:.3e}")
    if np.linalg.norm(X[2] - X[0] - e1 - e4) > _RECT_TOL * L or abs(e1 @ e4) > _RECT_TOL * a * b:
        raise ValueError("facet is not a rectangle (corners must be X1, X2 = X1 + a e_x, X3 = X2 + b e_y, "
                         "X4 = X1 + b e_y with e_x ⟂ e_y)")
    ex, ey = e1 / a, e4 / b
    ez = np.cross(ex, ey)
    return np.vstack([ex, ey, ez]), a, b


def facet_matrix(X: Any, t: float, nu: float = 0.3, drill: float = DRILL) -> np.ndarray:
    """(24, 24) unit-modulus stiffness of the rectangular facet with corners ``X`` (4, 3) in global axes; DOFs
    (u, v, w, θx, θy, θz) per corner."""
    _check(float(t), float(nu))
    _check_drill(float(drill))
    R, a, b = facet_frame(X)
    T = np.kron(np.eye(8), R)
    return T.T @ _local_matrix(a, b, float(t), float(nu), float(drill)) @ T


@dataclass(frozen=True)
class ShellMesh:
    """Surface of rectangular facets: ``nodes`` (nn, 3), ``facets`` (ne, 4) node indices in corner order."""

    nodes: np.ndarray
    facets: np.ndarray

    def __post_init__(self) -> None:
        X = np.asarray(self.nodes, float)
        F = np.asarray(self.facets)
        if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
            raise ValueError(f"nodes must be a finite (nn, 3) array, got shape {X.shape}")
        if F.ndim != 2 or F.shape[1] != 4 or len(F) == 0 or not np.issubdtype(F.dtype, np.integer):
            raise ValueError(f"facets must be a non-empty integer (ne, 4) array, got shape {F.shape}")
        if F.min() < 0 or F.max() >= len(X):
            raise ValueError(f"facet node index out of range [0, {len(X)})")
        if any(len(set(f)) != 4 for f in F.tolist()):
            raise ValueError("a facet repeats a node")
        object.__setattr__(self, "nodes", X)
        object.__setattr__(self, "facets", F.astype(np.int64))

    @property
    def nn(self) -> int:
        return len(self.nodes)

    @property
    def ne(self) -> int:
        return len(self.facets)


def flat_rectangle(lx: float, ly: float, nx: int, ny: int) -> ShellMesh:
    """lx × ly rectangle in the xy-plane from the origin, nx × ny facets (a = lx/nx, b = ly/ny), normal +z; node
    i + (nx + 1) j at (i a, j b, 0)."""
    if not (lx > 0 and ly > 0) or nx < 1 or ny < 1:
        raise ValueError(f"need lx, ly > 0 and nx, ny >= 1, got {lx}, {ly}, {nx}, {ny}")
    x, y = np.linspace(0.0, lx, nx + 1), np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    n0 = (i + (nx + 1) * j).ravel()
    facets = np.column_stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return ShellMesh(nodes, facets)


def cylinder_surface(radius: float, length: float, n_theta: int, n_z: int, angle: float | None = None) -> ShellMesh:
    """Circular cylinder about the z axis, z ∈ [0, length], n_theta × n_z facets with nodes on the radius.
    ``angle`` None: closed tube (n_theta ≥ 3, node i + n_theta j at θ = 2π i / n_theta); a value in (0, 2π): open
    segment θ ∈ [0, angle] (node i + (n_theta + 1) j). Facet normals point outwards (internal pressure p > 0)."""
    if not (radius > 0 and length > 0) or n_z < 1:
        raise ValueError(f"need radius, length > 0 and n_z >= 1, got {radius}, {length}, {n_z}")
    closed = angle is None
    if closed:
        if n_theta < 3:
            raise ValueError(f"a closed tube needs n_theta >= 3, got {n_theta}")
        theta = 2 * np.pi * np.arange(n_theta) / n_theta
    else:
        assert angle is not None
        if not 0 < angle < 2 * np.pi:
            raise ValueError(f"segment angle must be in (0, 2π) (angle=None for a closed tube), got {angle}")
        if n_theta < 1:
            raise ValueError(f"n_theta must be >= 1, got {n_theta}")
        theta = angle * np.arange(n_theta + 1) / n_theta
    z = np.linspace(0.0, length, n_z + 1)
    nt = len(theta)
    T, Z = np.meshgrid(theta, z)
    nodes = np.column_stack([radius * np.cos(T.ravel()), radius * np.sin(T.ravel()), Z.ravel()])
    i, j = np.meshgrid(np.arange(n_theta), np.arange(n_z))
    i, j = i.ravel(), j.ravel()
    ip = (i + 1) % nt
    facets = np.column_stack([i + nt * j, ip + nt * j, ip + nt * (j + 1), i + nt * (j + 1)])
    return ShellMesh(nodes, facets)


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E: torch.Tensor, F: torch.Tensor, model: FlatShell) -> torch.Tensor:
        u, lu = model._forward(E.detach().cpu().double().numpy(), F.detach().cpu().double().numpy())
        ctx.model, ctx.u, ctx.lu, ctx.dtype = model, u, lu, E.dtype
        return torch.from_numpy(u).to(dtype=F.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, Any, None]:
        m: FlatShell = ctx.model
        G = g.detach().cpu().double().numpy()
        lam = np.zeros_like(G)
        lam[m.free] = ctx.lu.solve(G[m.free])
        dE = -np.einsum("ea,eab,eb->e", lam[m.edof], m.KE, ctx.u[m.edof])
        return torch.from_numpy(dE).to(dtype=ctx.dtype), torch.from_numpy(lam).to(dtype=g.dtype), None


class FlatShell:
    """Flat-facet shell on ``mesh`` with thickness ``t``, Poisson ratio ``nu`` and drilling scale ``drill``.
    ``fixed``: fixed global DOFs (6 per node: u, v, w, θx, θy, θz), held at 0. :meth:`solve` gives the DOF vector
    (6 nn,) for facet moduli E (ne,) and a load (6 nn,), differentiable in both."""

    def __init__(self, mesh: ShellMesh, t: float, nu: float = 0.3, fixed: Sequence[int] | np.ndarray = (),
                 drill: float = DRILL) -> None:
        _check(float(t), float(nu))
        _check_drill(float(drill))
        self.mesh, self.t, self.nu, self.drill = mesh, float(t), float(nu), float(drill)
        self.ndof = 6 * mesh.nn
        self.edof = (6 * mesh.facets[:, :, None] + np.arange(6)[None, None, :]).reshape(mesh.ne, 24)
        frames = []
        for k, f in enumerate(mesh.facets):
            try:
                frames.append(facet_frame(mesh.nodes[f]))
            except ValueError as err:
                raise ValueError(f"facet {k}: {err}") from None
        self.frames = np.stack([R for R, _, _ in frames])                       # (ne, 3, 3)
        self.sizes = np.array([[a, b] for _, a, b in frames])                   # (ne, 2)
        local: dict[tuple[float, float], np.ndarray] = {}
        KE = np.empty((mesh.ne, 24, 24))
        for k, (R, a, b) in enumerate(frames):
            if (a, b) not in local:
                local[(a, b)] = _local_matrix(a, b, self.t, self.nu, self.drill)
            T = np.kron(np.eye(8), R)
            KE[k] = T.T @ local[(a, b)] @ T
        self.KE = KE
        fx = np.unique(np.asarray(fixed, np.int64))
        if fx.size and (fx.min() < 0 or fx.max() >= self.ndof):
            raise ValueError(f"fixed DOF index out of range [0, {self.ndof})")
        self.fixed = fx
        self.free = np.setdiff1d(np.arange(self.ndof), fx)
        self._rows = np.repeat(self.edof, 24, axis=1).ravel()
        self._cols = np.tile(self.edof, (1, 24)).ravel()
        self._cache: tuple[np.ndarray, Any] | None = None

    def _moduli(self, E: Any) -> np.ndarray:
        e = np.asarray(E, float)
        if e.shape != (self.mesh.ne,):
            raise ValueError(f"facet moduli must have shape ({self.mesh.ne},), got {e.shape}")
        if not np.isfinite(e).all():
            raise ValueError("facet moduli must be finite")
        return e

    def stiffness(self, E: Any) -> sp.csr_matrix:
        """Assembled K (6 nn, 6 nn) for facet moduli ``E`` (ne,), before applying the fixed DOFs."""
        e = self._moduli(E)
        vals = (e[:, None] * self.KE.reshape(self.mesh.ne, -1)).ravel()
        return sp.csr_matrix((vals, (self._rows, self._cols)), shape=(self.ndof, self.ndof))

    def pressure_load(self, p: float) -> np.ndarray:
        """Consistent nodal load (6 nn,) of a uniform pressure ``p`` along each facet normal e_z (p a b / 4 · e_z per
        corner)."""
        F = np.zeros(self.ndof)
        f = p * (self.sizes[:, 0] * self.sizes[:, 1] / 4.0)[:, None] * self.frames[:, 2, :]     # (ne, 3)
        for c in range(4):
            for d in range(3):
                np.add.at(F, 6 * self.mesh.facets[:, c] + d, f[:, d])
        return F

    def _forward(self, E: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, Any]:
        e = self._moduli(E)
        if F.shape != (self.ndof,) or not np.isfinite(F).all():
            raise ValueError(f"load must be a finite ({self.ndof},) vector, got shape {F.shape}")
        if self._cache is not None and np.array_equal(self._cache[0], e):
            lu = self._cache[1]
        else:
            K = self.stiffness(e).tocsc()
            lu = spla.splu(K[self.free][:, self.free].tocsc())
            self._cache = (e.copy(), lu)
        u = np.zeros(self.ndof)
        u[self.free] = lu.solve(F[self.free])
        return u, lu

    def solve(self, E: Any, F: Any) -> torch.Tensor:
        """DOF vector (6 nn,) for facet moduli ``E`` (ne,) and load ``F`` (6 nn,); differentiable in both."""
        Et = torch.as_tensor(E, dtype=torch.float64)
        Ft = torch.as_tensor(np.asarray(F, float) if not isinstance(F, torch.Tensor) else F, dtype=torch.float64)
        out: torch.Tensor = _Solve.apply(Et, Ft, self)
        return out
