"""A symmetric positive definite system on the grid: K(c) = Σ_m Σ_e c_{m,e} K_m⁰ with unit element matrices K_m⁰.

:class:`GridSPD` assembles K (the one sparse scatter of :mod:`toporlogy.fem.thermoelastic`,
:mod:`toporlogy.fem.homogenization` and :mod:`toporlogy.fem.transient_heat`), applies prescribed values and
solves; :meth:`GridSPD.solve` is a torch autograd function with derivatives with respect to the coefficients
(dc_{m,e} = −λ_eᵀ K_m⁰ u_e) and the right-hand side (λ). ``KE`` is one matrix (nloc, nloc) with coefficients (n_el,)
or a stack (n_mats, nloc, nloc) with coefficients (n_mats, n_el). The factorisation of the last coefficients is
kept, so repeated solves with the same matrix (time steps) factorise once.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from toporlogy.core.domain import Domain, require_grid

__all__ = ["GridSPD", "apply_elements"]


def apply_elements(KE: np.ndarray, coef: torch.Tensor, u: torch.Tensor, edof: np.ndarray, n: int) -> torch.Tensor:
    """K(c) u by element products, differentiable in ``coef`` and ``u`` (KE and coef as for :class:`GridSPD`)."""
    KEs = np.asarray(KE, float)
    if KEs.ndim == 2:
        KEs, coef = KEs[None], coef[None]
    ue = u[torch.as_tensor(edof)]                                              # (n_el, nloc)
    fe = torch.einsum("me,mab,eb->ea", coef, torch.as_tensor(KEs, dtype=u.dtype), ue)
    return torch.zeros(n, dtype=u.dtype).index_add(0, torch.as_tensor(np.asarray(edof).ravel()), fe.reshape(-1))


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, coef: torch.Tensor, rhs: torch.Tensor, model: GridSPD) -> torch.Tensor:
        u, lu = model._forward(coef.detach().cpu().double().numpy(), rhs.detach().cpu().double().numpy())
        ctx.model, ctx.u, ctx.lu = model, u, lu
        ctx.dtype, ctx.device = coef.dtype, coef.device
        return torch.from_numpy(u).to(dtype=rhs.dtype, device=rhs.device)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, Any, None]:
        m: GridSPD = ctx.model
        G = g.detach().cpu().double().numpy()
        lam = np.zeros_like(G)
        lam[m.free] = ctx.lu.solve(G[m.free])
        ue, le = ctx.u[m.edof], lam[m.edof]                       # (n_el, nloc)
        dcoef = -np.einsum("ea,mab,eb->me", le, m._KEs, ue)
        if m.KE.ndim == 2:
            dcoef = dcoef[0]
        return (torch.from_numpy(dcoef).to(dtype=ctx.dtype, device=ctx.device),
                torch.from_numpy(lam).to(dtype=g.dtype, device=g.device), None)


class GridSPD:
    """K(c) u = r on the grid (module notes), prescribed values ``u[fixed] = values`` (default 0) and torch
    derivatives with respect to c and r."""

    def __init__(self, domain: Domain, KE: np.ndarray, edof: np.ndarray, n: int, fixed: Sequence[int] | np.ndarray,
                 values: Any = 0.0) -> None:
        require_grid(domain, "GridSPD")
        self.domain, self.KE, self.edof, self.n = domain, np.asarray(KE, float), np.asarray(edof, np.int64), int(n)
        if self.KE.ndim not in (2, 3):
            raise ValueError(f"KE must be (nloc, nloc) or (n_mats, nloc, nloc), got {self.KE.shape}")
        self._KEs = self.KE if self.KE.ndim == 3 else self.KE[None]
        self.fixed = np.unique(np.asarray(fixed, np.int64))
        self.free = np.setdiff1d(np.arange(self.n), self.fixed)
        self.values = np.broadcast_to(np.asarray(values, float), self.fixed.shape).copy()
        nl = self.edof.shape[1]
        self._rows = np.repeat(self.edof, nl, axis=1).ravel()
        self._cols = np.tile(self.edof, (1, nl)).ravel()
        self._cache: tuple[np.ndarray, Any, Any] | None = None
        self.n_factorizations = 0

    def _coef(self, coef: Any) -> np.ndarray:
        c = np.asarray(coef, float)
        want = (self.edof.shape[0],) if self.KE.ndim == 2 else (self._KEs.shape[0], self.edof.shape[0])
        if c.shape != want:
            raise ValueError(f"coefficients must have shape {want}, got {c.shape}")
        return c.reshape(self._KEs.shape[0], -1)

    def matrix(self, coef: Any) -> sp.csr_matrix:
        c = self._coef(coef)
        vals = np.einsum("me,mk->ek", c, self._KEs.reshape(self._KEs.shape[0], -1)).ravel()
        return sp.csr_matrix((vals, (self._rows, self._cols)), shape=(self.n, self.n))

    def _forward(self, coef: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, Any]:
        c = self._coef(coef)
        if rhs.shape != (self.n,):
            raise ValueError(f"right-hand side must have {self.n} entries, got {rhs.shape}")
        if not (np.isfinite(c).all() and np.isfinite(rhs).all()):
            raise ValueError("coefficients and right-hand side must be finite")
        if self._cache is not None and np.array_equal(self._cache[0], c):
            _, lu, Kfx = self._cache
        else:
            K = self.matrix(coef).tocsc()
            lu = spla.splu(K[self.free][:, self.free].tocsc())
            Kfx = K[self.free][:, self.fixed]
            self._cache = (c.copy(), lu, Kfx)
            self.n_factorizations += 1
        u = np.zeros(self.n)
        u[self.fixed] = self.values
        u[self.free] = lu.solve(rhs[self.free] - Kfx @ u[self.fixed])
        return u, lu

    def solve(self, coef: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """u (n,) with gradients to ``coef`` and ``rhs`` (the prescribed values are constants)."""
        out: torch.Tensor = _Solve.apply(coef, rhs, self)
        return out
