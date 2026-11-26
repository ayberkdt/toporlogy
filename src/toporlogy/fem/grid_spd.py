"""A symmetric positive definite system on the grid with one unit element matrix: K(c) = Σ_e c_e K_e⁰.

:class:`GridSPD` assembles K (the one sparse scatter of :mod:`toporlogy.fem.thermoelastic` and
:mod:`toporlogy.fem.homogenization`), applies prescribed values and solves; :meth:`GridSPD.solve` is a torch autograd
function with derivatives with respect to the coefficients c (−λ_eᵀ K_e⁰ u_e) and the right-hand side (λ).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from toporlogy.core.domain import Domain

__all__ = ["GridSPD"]


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
        dcoef = -np.einsum("ea,ab,eb->e", le, m.KE, ue)
        return (torch.from_numpy(dcoef).to(dtype=ctx.dtype, device=ctx.device),
                torch.from_numpy(lam).to(dtype=g.dtype, device=g.device), None)


class GridSPD:
    """K(c) u = r on the grid with K = Σ_e c_e K_e⁰ (one unit element matrix ``KE`` incl. its geometry factor),
    prescribed values ``u[fixed] = values`` (default 0) and torch derivatives with respect to c and r."""

    def __init__(self, domain: Domain, KE: np.ndarray, edof: np.ndarray, n: int, fixed: Sequence[int] | np.ndarray,
                 values: Any = 0.0) -> None:
        self.domain, self.KE, self.edof, self.n = domain, np.asarray(KE, float), np.asarray(edof, np.int64), int(n)
        self.fixed = np.unique(np.asarray(fixed, np.int64))
        self.free = np.setdiff1d(np.arange(self.n), self.fixed)
        self.values = np.broadcast_to(np.asarray(values, float), self.fixed.shape).copy()
        nl = self.edof.shape[1]
        self._rows = np.repeat(self.edof, nl, axis=1).ravel()
        self._cols = np.tile(self.edof, (1, nl)).ravel()

    def matrix(self, coef: np.ndarray) -> sp.csr_matrix:
        vals = (np.asarray(coef, float)[:, None] * self.KE.ravel()[None, :]).ravel()
        return sp.csr_matrix((vals, (self._rows, self._cols)), shape=(self.n, self.n))

    def _forward(self, coef: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, Any]:
        if coef.shape != (self.edof.shape[0],):
            raise ValueError(f"coefficients must have {self.edof.shape[0]} entries, got {coef.shape}")
        if rhs.shape != (self.n,):
            raise ValueError(f"right-hand side must have {self.n} entries, got {rhs.shape}")
        if not (np.isfinite(coef).all() and np.isfinite(rhs).all()):
            raise ValueError("coefficients and right-hand side must be finite")
        K = self.matrix(coef).tocsc()
        u = np.zeros(self.n)
        u[self.fixed] = self.values
        b = rhs[self.free] - K[self.free][:, self.fixed] @ u[self.fixed]
        lu = spla.splu(K[self.free][:, self.free].tocsc())
        u[self.free] = lu.solve(b)
        return u, lu

    def solve(self, coef: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """u (n,) with gradients to ``coef`` and ``rhs`` (the prescribed values are constants)."""
        out: torch.Tensor = _Solve.apply(coef, rhs, self)
        return out
