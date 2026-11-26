"""Design-variable chain x → x̄ (docs/mimari_plan.md §2): each link is a ``torch.nn.Module`` on element grids.

The usual chain (the order of the legacy voxel driver, which ``tests/toporlogy`` keeps as a regression oracle):

``Embed`` (design vector → grid, keep = 1, void = 0, outside = 0) → ``DensityFilter`` → [AM filter] →
``HeavisideProjection`` → ``Passive`` (keep = 1, void = 0, outside = 0 again).

Gradients come from autograd; every link has a finite-difference test (``tests/toporlogy``). Links that change during a
run (β of the projection) expose plain attributes that a schedule sets.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as tF

from toporlogy.references import cite
from toporlogy.core.domain import Domain, require_grid

__all__ = ["Chain", "Embed", "DensityFilter", "HeavisideProjection", "Passive", "standard_chain"]


def _t(a: np.ndarray, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


class Embed(torch.nn.Module):
    """Design vector (n_design,) → element grid of ``domain.shape``: keep 1, void 0, outside 0.

    ``n_fields`` > 1 (several design fields per element, :mod:`toporlogy.design.phases`): the design vector has
    ``n_fields · n_design`` entries, field-major (field k is ``x[k·n_design:(k+1)·n_design]``), and the output is
    ``(n_fields, *domain.shape)``; every field gets the same keep / void / outside values. The later links act on
    each field alone. The instance then declares ``n_in`` (the length :class:`~toporlogy.optimization.study.Study`
    reads from the first link).
    """

    idx: torch.Tensor
    base: torch.Tensor

    def __init__(self, domain: Domain, n_fields: int = 1) -> None:
        super().__init__()
        if int(n_fields) < 1:
            raise ValueError("n_fields >= 1 is required")
        self.shape = domain.shape
        self.n_fields = int(n_fields)
        if self.n_fields > 1:
            self.n_in = self.n_fields * domain.n_design
        self.register_buffer("idx", torch.as_tensor(domain.design_elements, dtype=torch.long))
        self.register_buffer("base", _t(domain.keep.astype(float)).reshape(-1))

    def _one(self, x: torch.Tensor) -> torch.Tensor:
        g = self.base.clone().to(x.dtype)
        g = g.index_put((self.idx,), x)
        return g.reshape(self.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_fields == 1:
            return self._one(x)
        if x.numel() != self.n_fields * self.idx.numel():
            raise ValueError(f"design vector has {x.numel()} entries, {self.n_fields} fields need "
                             f"{self.n_fields * self.idx.numel()}")
        return torch.stack([self._one(xk) for xk in x.reshape(self.n_fields, -1)])


@cite("bourdin_2001", note="linear cone weights max(0, r_min − dist), normalised by the weight sum")
class DensityFilter(torch.nn.Module):
    """Density filter x̃ = H x / Hs with cone weights ``max(0, rmin − dist)`` (dist in elements).

    Only active elements take part (weights over elements outside the part are zero), exactly as
    the legacy voxel driver's filter: implemented as a convolution of ``x·active`` divided by the
    convolution of ``active``. Elements outside the part get 0.

    ``periodic``: the weights wrap across opposite faces of the box (a periodic cell, as in
    :mod:`toporlogy.fem.homogenization`); needs every element active. Pre-registered checks:
    ``docs/olcumler/periyodik_filtre_onkayit.md`` (PF1-PF5).
    """

    kernel: torch.Tensor
    mask: torch.Tensor
    hs: torch.Tensor

    def __init__(self, domain: Domain, rmin: float, backend: str = "auto", periodic: bool = False) -> None:
        super().__init__()
        require_grid(domain, "DensityFilter", "MeshFilterLink")
        if not rmin > 0:
            raise ValueError("rmin > 0 is required")
        if backend not in ("auto", "sparse", "conv"):
            raise ValueError("backend: auto | sparse | conv")
        self.periodic = bool(periodic)
        if self.periodic and not (np.all(domain.active) and not np.any(np.asarray(domain.void))):
            raise ValueError("density_filter periodic: every element of the box must be active (no void or outside "
                             "region); the box is the periodic cell")
        self.rmin = float(rmin)
        self.dim = domain.dim
        self.shape = domain.shape
        self.backend = backend
        r = math.ceil(rmin) - 1
        if self.periodic and r >= min(domain.shape):
            raise ValueError(f"density_filter periodic: rmin {rmin} needs more than {r} elements along every axis")
        self.pad = r
        ax = np.arange(-r, r + 1)
        grids = np.meshgrid(*([ax] * self.dim), indexing="ij")
        w = np.maximum(0.0, rmin - np.sqrt(sum(g.astype(float) ** 2 for g in grids)))
        self.register_buffer("kernel", _t(w)[None, None])
        m = _t(domain.active.astype(float))
        self.register_buffer("mask", m)
        hs = self._conv(m)
        self.register_buffer("hs", torch.where(m > 0, hs, torch.ones_like(hs)))
        self._H: Any = None
        self._Ht: Any = None
        self._offsets = [(tuple(int(v) for v in o), float(w[tuple(np.asarray(o) + r)]))
                         for o in np.argwhere(w > 0) - r]
        self._active = domain.active

    def _conv(self, g: torch.Tensor) -> torch.Tensor:
        conv = tF.conv3d if self.dim == 3 else tF.conv2d
        if self.periodic:
            gp = tF.pad(g[None, None], (self.pad,) * (2 * self.dim), mode="circular")
            return conv(gp, self.kernel.to(g.dtype))[0, 0]
        return conv(g[None, None], self.kernel.to(g.dtype), padding=self.pad)[0, 0]

    def matrix(self) -> sp.csr_matrix:
        """Sparse H / Hs (box elements × box elements, rows and columns of active elements only)."""
        if self._H is None:
            act = self._active
            shape = np.array(self.shape)
            ijk = np.argwhere(act)
            e = np.asarray(np.ravel_multi_index(tuple(ijk.T), self.shape))
            rows, cols, vals = [], [], []
            for off, wt in self._offsets:
                k = ijk + np.asarray(off)
                if self.periodic:
                    k = k % shape
                ok = np.asarray(np.all((k >= 0) & (k < shape), axis=1))
                ok[ok] = act[tuple(k[ok].T)]
                rows.append(e[ok])
                cols.append(np.ravel_multi_index(k[ok].T, self.shape))
                vals.append(np.full(int(ok.sum()), wt))
            n = int(np.prod(self.shape))
            H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
            hs = self.hs.numpy().ravel()
            H = sp.diags(1.0 / hs) @ H
            self._H, self._Ht = H.tocsr(), H.T.tocsr()
        return self._H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == self.dim + 1:                 # several design fields (Embed n_fields > 1): each alone
            return torch.stack([self.forward(xk) for xk in x])
        use_sparse = self.backend == "sparse" or (self.backend == "auto" and x.device.type == "cpu")
        if use_sparse:
            self.matrix()
            return _SparseApply.apply(x, self._H, self._Ht).reshape(self.shape)
        m = self.mask.to(x.dtype)
        return self._conv(x * m) / self.hs.to(x.dtype) * m


class _SparseApply(torch.autograd.Function):
    """y = H x (x flattened, float64 CPU); backward Hᵀ g."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, H: sp.csr_matrix, Ht: sp.csr_matrix) -> torch.Tensor:
        ctx.Ht = Ht
        ctx.dtype, ctx.shape = x.dtype, x.shape
        y = H @ x.detach().reshape(-1).double().numpy()
        return torch.from_numpy(y).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, None, None]:
        gx = ctx.Ht @ g.detach().reshape(-1).double().numpy()
        return torch.from_numpy(gx).to(ctx.dtype).reshape(ctx.shape), None, None


@cite("ferrari_sigmund_2020", note="Eq. (1) tanh projection")
class HeavisideProjection(torch.nn.Module):
    """x̂ = (tanh(βη) + tanh(β(x̃ − η))) / (tanh(βη) + tanh(β(1 − η))); β and η are plain attributes.

    Element-wise, so several design fields ``(n_fields, *shape)`` are projected with the same β and η."""

    def __init__(self, beta: float = 1.0, eta: float = 0.5) -> None:
        super().__init__()
        self.beta, self.eta = float(beta), float(eta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, e = self.beta, self.eta
        a, d = math.tanh(b * e), math.tanh(b * (1.0 - e))
        return (a + torch.tanh(b * (x - e))) / (a + d)


class Passive(torch.nn.Module):
    """Force keep = 1, void = 0, outside the part = 0 (gradient through those elements is zero).

    Several design fields ``(n_fields, *shape)`` broadcast: every field is set; with
    :class:`~toporlogy.design.phases.MultiPhaseSIMP` keep elements are then the last (stiffest) phase."""

    keep: torch.Tensor
    off: torch.Tensor

    def __init__(self, domain: Domain) -> None:
        super().__init__()
        self.register_buffer("keep", torch.as_tensor(domain.keep))
        self.register_buffer("off", torch.as_tensor(domain.void | ~domain.active))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        return torch.where(self.keep, one, torch.where(self.off, zero, x))


class Chain(torch.nn.Module):
    """Links applied in order; ``chain[i]`` and ``chain.find(type)`` give access to a link (e.g. to set β)."""

    def __init__(self, links: Iterable[torch.nn.Module]) -> None:
        super().__init__()
        self.links = torch.nn.ModuleList(list(links))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for link in self.links:
            x = link(x)
        return x

    def __getitem__(self, i: int) -> torch.nn.Module:
        return self.links[i]

    def find(self, kind: type[torch.nn.Module]) -> torch.nn.Module | None:
        for link in self.links:
            if isinstance(link, kind):
                return link
        return None


def standard_chain(domain: Domain, rmin: float, projection: bool = True, beta: float = 1.0,
                   eta: float = 0.5) -> Chain:
    """Embed → DensityFilter(rmin) → [HeavisideProjection(β, η)] → Passive."""
    require_grid(domain, "standard_chain", "a Chain of Embed, MeshFilterLink, Passive")
    links: list[torch.nn.Module] = [Embed(domain), DensityFilter(domain, rmin)]
    if projection:
        links.append(HeavisideProjection(beta, eta))
    links.append(Passive(domain))
    return Chain(links)
