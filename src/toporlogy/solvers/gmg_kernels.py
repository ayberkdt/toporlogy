"""Numerical kernels of the geometric multigrid solver (:mod:`toporlogy.solvers.gmg`): torch only, any device.

Data layout (the numbering of :class:`toporlogy.domain.Domain`): a nodal field of a grid with ``shape = (nx, ny,
nz)`` elements is a tensor ``(nx+1, ny+1, nz+1, 3)`` whose C-order flattening is the DOF order ``3n + d``; an element
field is ``(nx, ny, nz)`` flattened to ``e = (ix·ny + iy)·nz + iz``. Element DOFs follow :data:`CORNERS` (the H8 corner
order of :mod:`toporlogy.fem.elements`).

* Element gather / scatter by eight strided slices (no index arrays).
* Trilinear prolongation ``P`` and ``R = Pᵀ`` on a 2:1 grid, axis by axis.
* Element-level Galerkin coarse operators: trilinear ``P`` restricted to one coarse element only touches that
  element's 8 nodes, so ``K_c[E] = Σ_children P_cᵀ (M K_child M) P_c`` (M = free-DOF mask) equals the global
  ``P̃ᵀ K P̃`` with the masked prolongation ``P̃ = M_f P M_c``.
* Chebyshev smoother on ``D⁻¹A`` (Saad, Iterative Methods, Alg. 12.1 — general knowledge, not re-read in this session).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["CORNERS", "gather", "scatter", "node_max", "local_P", "children", "prolong", "restrict", "Level",
           "galerkin_from_scale", "galerkin_from_elements", "edof_grid", "chebyshev", "power_lmax"]

CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def gather(u: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Nodal field (nx+1, ny+1, nz+1, 3) → element vectors (ne, 24) in :data:`CORNERS` order."""
    nx, ny, nz = shape
    return torch.stack([u[a:a + nx, b:b + ny, c:c + nz] for a, b, c in CORNERS], dim=3).reshape(-1, 24)


def scatter(fe: torch.Tensor, shape: tuple[int, int, int], out: torch.Tensor) -> torch.Tensor:
    """Add element vectors (ne, 24) into the nodal field ``out`` (in place) and return it."""
    nx, ny, nz = shape
    v = fe.view(nx, ny, nz, 8, 3)
    for k, (a, b, c) in enumerate(CORNERS):
        out[a:a + nx, b:b + ny, c:c + nz] += v[:, :, :, k]
    return out


def node_max(E: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Per node, the largest value of ``E`` (ne,) over the (up to 8) adjacent elements → (nx+1, ny+1, nz+1)."""
    nx, ny, nz = shape
    Ep = torch.zeros((nx + 2, ny + 2, nz + 2), dtype=E.dtype, device=E.device)
    Ep[1:-1, 1:-1, 1:-1] = E.view(nx, ny, nz)
    out = torch.zeros((nx + 1, ny + 1, nz + 1), dtype=E.dtype, device=E.device)
    for a, b, c in CORNERS:
        out = torch.maximum(out, Ep[a:a + nx + 1, b:b + ny + 1, c:c + nz + 1])
    return out


def local_P() -> np.ndarray:
    """(8, 24, 24): trilinear interpolation from the 24 DOFs of a coarse element to the 24 DOFs of child c,
    child c = 4a + 2b + cz at offset (a, b, cz) inside the coarse element."""
    P = np.zeros((8, 24, 24))
    for ci in range(8):
        off = (ci // 4, (ci // 2) % 2, ci % 2)
        for k, kc in enumerate(CORNERS):
            t = [(off[m] + kc[m]) / 2.0 for m in range(3)]
            for j, jc in enumerate(CORNERS):
                w = 1.0
                for m in range(3):
                    w *= t[m] if jc[m] else 1.0 - t[m]
                for d in range(3):
                    P[ci, 3 * k + d, 3 * j + d] = w
    return P


def children(a: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Fine element array (nx·ny·nz, ...) → (ncoarse, 8, ...) with child order 4a + 2b + cz."""
    nx, ny, nz = shape
    rest = tuple(a.shape[1:])
    v = a.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2, *rest)
    perm = (0, 2, 4, 1, 3, 5) + tuple(range(6, 6 + len(rest)))
    return v.permute(*perm).reshape((nx // 2) * (ny // 2) * (nz // 2), 8, *rest)


def _interp_axis(a: torch.Tensor, ax: int) -> torch.Tensor:
    a = a.movedim(ax, 0)
    n = a.shape[0]
    out = a.new_empty((2 * n - 1,) + tuple(a.shape[1:]))
    out[0::2] = a
    out[1::2] = 0.5 * (a[:-1] + a[1:])
    return out.movedim(0, ax)


def _restrict_axis(f: torch.Tensor, ax: int) -> torch.Tensor:
    f = f.movedim(ax, 0)
    c = f[0::2].clone()
    half = 0.5 * f[1::2]
    c[:-1] += half
    c[1:] += half
    return c.movedim(0, ax)


def prolong(uc: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a coarse nodal field (2:1 grid)."""
    for ax in range(3):
        uc = _interp_axis(uc, ax)
    return uc.contiguous()


def restrict(f: torch.Tensor) -> torch.Tensor:
    """Transpose of :func:`prolong`."""
    for ax in range(3):
        f = _restrict_axis(f, ax)
    return f.contiguous()


def edof_grid(shape: tuple[int, int, int]) -> np.ndarray:
    """(ne, 24) element DOFs of a full box grid (same numbering as :attr:`Domain.edof`)."""
    nx, ny, nz = shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cols = [(((ix + a) * (ny + 1) + (iy + b)) * (nz + 1) + (iz + c)).reshape(-1) for a, b, c in CORNERS]
    conn = np.stack(cols, 1)
    return (3 * conn[:, :, None] + np.arange(3)).reshape(-1, 24)


class Level:
    """One grid level. Either the fine level (shared ``KE`` (24, 24), element scale ``s`` (ne,), DOF mask applied
    on the fly) or a Galerkin level (per-element matrices ``Ke`` (ne, 24, 24) whose fixed rows/columns are already
    zeroed). Fixed DOFs act as identity rows/columns."""

    def __init__(self, shape: tuple[int, int, int], free: torch.Tensor, s: torch.Tensor | None = None,
                 KE: torch.Tensor | None = None, Ke: torch.Tensor | None = None) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.free = free
        self.fixed = 1.0 - free
        self.s, self.KE, self.Ke = s, KE, Ke
        self.dinv: torch.Tensor | None = None
        self.lmax: float = float("nan")

    @property
    def ndof(self) -> int:
        nx, ny, nz = self.shape
        return 3 * (nx + 1) * (ny + 1) * (nz + 1)

    def raw(self, u: torch.Tensor) -> torch.Tensor:
        ue = gather(u, self.shape)
        if self.Ke is None:
            fe = (ue @ self.KE) * self.s[:, None]
        else:
            fe = torch.bmm(self.Ke, ue.unsqueeze(2)).squeeze(2)
        return scatter(fe, self.shape, torch.zeros_like(u))

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        if self.Ke is None:
            return self.free * self.raw(self.free * u) + self.fixed * u
        return self.raw(u) + self.fixed * u

    def raw_diag(self) -> torch.Tensor:
        nx, ny, nz = self.shape
        if self.Ke is None:
            de = self.s[:, None] * torch.diagonal(self.KE)[None, :]
        else:
            de = torch.diagonal(self.Ke, dim1=1, dim2=2)
        return scatter(de.contiguous(), self.shape,
                       torch.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=de.dtype, device=de.device))

    def diag(self) -> torch.Tensor:
        return self.free * self.raw_diag() + self.fixed

    def to(self, dtype: torch.dtype) -> Level:
        cv = (lambda t: None if t is None else t.to(dtype))  # noqa: E731
        L = Level(self.shape, self.free.to(dtype), cv(self.s), cv(self.KE), cv(self.Ke))
        L.dinv, L.lmax = cv(self.dinv), self.lmax
        return L

    def element_matrices(self) -> torch.Tensor:
        """(ne, 24, 24) masked element matrices (fine level: formed explicitly — small grids only)."""
        if self.Ke is not None:
            return self.Ke
        fe = gather(self.free, self.shape)
        return self.s[:, None, None] * self.KE[None] * fe[:, :, None] * fe[:, None, :]


def galerkin_from_scale(s: torch.Tensor, KE: torch.Tensor, shape: tuple[int, int, int], free: torch.Tensor,
                        Pl: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Coarse element matrices (nc, 24, 24) of ``P̃ᵀ (M K M) P̃`` for the fine level ``K = Σ s_e KE``.

    Unmasked children use ``Σ_c s_c (P_cᵀ KE P_c)`` (one small GEMM); coarse elements with a child that touches a
    fixed fine DOF and has s > 0 are recomputed with the mask, in chunks.
    """
    A8 = torch.einsum("cji,jk,ckl->cil", Pl, KE, Pl).reshape(8, 576)
    Sc = children(s, shape)                                        # (nc, 8)
    Kc = (Sc @ A8).view(-1, 24, 24)
    fe = gather(free, shape)                                       # (ne, 24)
    bad = (fe.amin(1) < 1) & (s > 0)
    idx = torch.nonzero(children(bad, shape).any(1)).flatten()
    if idx.numel():
        fc = children(fe, shape)                                   # (nc, 8, 24)
        for i in range(0, idx.numel(), chunk):
            j = idx[i:i + chunk]
            m = fc[j]
            Km = Sc[j][:, :, None, None] * KE[None, None] * m[:, :, :, None] * m[:, :, None, :]
            Kc[j] = torch.einsum("cji,mcjk,ckl->mil", Pl, Km, Pl)
    return Kc


def galerkin_from_elements(Ke: torch.Tensor, shape: tuple[int, int, int], Pl: torch.Tensor,
                           chunk: int = 8192) -> torch.Tensor:
    """Coarse element matrices from (already masked) fine element matrices ``Ke`` (ne, 24, 24)."""
    Kc = children(Ke, shape)                                       # (nc, 8, 24, 24)
    out = Ke.new_empty((Kc.shape[0], 24, 24))
    for i in range(0, Kc.shape[0], chunk):
        out[i:i + chunk] = torch.einsum("cji,mcjk,ckl->mil", Pl, Kc[i:i + chunk], Pl)
    return out


def mask_elements(Ke: torch.Tensor, free: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Zero the rows/columns of fixed DOFs in element matrices (in place) and return them."""
    fe = gather(free, shape)
    Ke.mul_(fe[:, :, None]).mul_(fe[:, None, :])
    return Ke


def power_lmax(L: Level, its: int = 15, seed: int = 0) -> float:
    """Largest eigenvalue of ``D⁻¹A`` by power iteration (Rayleigh quotient in the D inner product)."""
    g = torch.Generator(device=L.free.device).manual_seed(seed)
    v = torch.rand(L.free.shape, generator=g, dtype=L.free.dtype, device=L.free.device) * L.free
    dinv = L.dinv
    assert dinv is not None
    lam = 0.0
    for _ in range(its):
        Av = L.apply(v)
        lam = float((v * Av).sum() / (v * v / dinv).sum())
        w = dinv * Av
        v = w / w.norm()
    return lam


def chebyshev(L: Level, x: torch.Tensor | None, b: torch.Tensor, degree: int, lo: float, hi: float) -> torch.Tensor:
    """``degree`` Chebyshev steps on ``D⁻¹A`` over [lo, hi]·λmax; ``x = None`` means a zero start (saves one apply)."""
    dinv = L.dinv
    assert dinv is not None
    lmax, lmin = hi * L.lmax, lo * L.lmax
    theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
    sigma = theta / delta
    rho = 1.0 / sigma
    r = b if x is None else b - L.apply(x)
    d = dinv * r / theta
    x = d if x is None else x + d
    for _ in range(degree - 1):
        r = r - L.apply(d)
        rho_new = 1.0 / (2.0 * sigma - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * dinv * r
        rho = rho_new
        x = x + d
    return x


def rigid_body_modes(node_shape: tuple[int, int, int]) -> np.ndarray:
    """(3·nnodes, 6) translations and rotations of a grid with unit spacing (centred), DOF order 3n + d."""
    X = np.indices(node_shape).reshape(3, -1).T.astype(float)
    X -= X.mean(0)
    n = X.shape[0]
    B = np.zeros((3 * n, 6))
    for d in range(3):
        B[d::3, d] = 1.0
    x, y, z = X.T
    B[0::3, 3], B[1::3, 3] = -y, x
    B[1::3, 4], B[2::3, 4] = -z, y
    B[0::3, 5], B[2::3, 5] = z, -x
    return B


def assemble_csr(Ke: torch.Tensor, shape: tuple[int, int, int]) -> Any:
    """Full (all-DOF) SciPy CSR matrix of a Galerkin level from its masked element matrices."""
    import scipy.sparse as sp
    ed = edof_grid(shape)
    nx, ny, nz = shape
    n = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    data = Ke.detach().to("cpu", torch.float64).numpy().reshape(-1)
    rows = np.repeat(ed, 24, axis=1).reshape(-1)
    cols = np.tile(ed, (1, 24)).reshape(-1)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A
