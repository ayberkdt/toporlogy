"""Helmholtz (PDE) density filter on the regular grid.

``−r² Δ x̃ + x̃ = x`` with homogeneous Neumann conditions on the boundary of the part (Lazarov & Sigmund 2011),
discretised with one unknown per active element (cell-centred finite volumes, unit spacing since lengths are in
elements):

    x̃_i + r² Σ_{j ∈ N(i)} (x̃_i − x̃_j) = x_i ,        N(i): active face neighbours of element i,

so ``A x̃ = x`` with ``A = I + r² L`` (L: graph Laplacian of the active elements). A is symmetric positive definite
with unit column sums, hence

* the filter is volume preserving, Σ x̃ = Σ x, on any part shape (the cone filter is not: it renormalises at walls);
* 0 ≤ x ≤ 1 implies 0 ≤ x̃ ≤ 1 (A is an M-matrix);
* the derivative is the same solve, ``∂J/∂x = A⁻¹ ∂J/∂x̃``.

``r = rmin / (2√3)`` is the relation attributed to Lazarov & Sigmund (only the record of the paper was seen in this
work, ``docs/literatur_2026-11-21.md``). What it means was derived and measured here: the impulse response of the
PDE filter has variance 2r² = rmin²/6 per axis in any dimension, the cone kernel of radius rmin has rmin²/6 in 1-D,
3 rmin²/20 in 2-D and 2 rmin²/15 in 3-D. The two filters therefore have the same second moment only in 1-D; in 2-D
the PDE filter is 11 % and in 3-D 25 % wider in variance (continuum values; ``tests/toporlogy/test_pde_filter.py``
checks the discrete ones), and its kernel is peaked at the centre with exponential tails instead of compact support.
A minimum size calibrated for the cone filter (:mod:`toporlogy.optimization.robust`) does NOT carry over.

Why have it besides :class:`~toporlogy.design.chain.DensityFilter`: the cost does not grow with the radius (one
sparse factorisation, reused in every iteration, instead of a stencil of (2 rmin)^dim entries), which matters for the
large radii of robust and maximum-size formulations; and the Neumann condition is the natural one for parts with
holes and curved walls. The known price is the boundary: a Neumann wall attracts material ("sticking"), see Clausen &
Andreassen (2017) and Wallin et al. (2020) for padding and Robin-type remedies — not implemented here.

The factorisation uses SciPy's SuperLU (CPU, float64). The link runs on CPU tensors only.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from toporlogy.core.domain import Domain
from toporlogy.references import cite

__all__ = ["HelmholtzFilter", "helmholtz_matrix"]


def helmholtz_matrix(active: np.ndarray, r: float) -> tuple[sp.csc_matrix, np.ndarray]:
    """``A = I + r² L`` over the active elements of a grid mask and their box element numbers (ascending)."""
    act = np.asarray(active, bool)
    num = np.full(act.shape, -1, np.int64)
    el = np.flatnonzero(act.ravel())
    num.ravel()[el] = np.arange(el.size)
    rows, cols = [], []
    for ax in range(act.ndim):
        a = [slice(None)] * act.ndim
        b = [slice(None)] * act.ndim
        a[ax], b[ax] = slice(0, -1), slice(1, None)
        both = act[tuple(a)] & act[tuple(b)]
        i, j = num[tuple(a)][both], num[tuple(b)][both]
        rows += [i, j]
        cols += [j, i]
    n = el.size
    if rows:
        r_, c_ = np.concatenate(rows), np.concatenate(cols)
        W = sp.csr_matrix((np.ones(r_.size), (r_, c_)), shape=(n, n))
    else:
        W = sp.csr_matrix((n, n))
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
    return (sp.identity(n, format="csc") + (r * r) * L).tocsc(), el


@cite("lazarov_sigmund_2011", note="Helmholtz filter with Neumann conditions; record seen, text not read")
class HelmholtzFilter(torch.nn.Module):
    """PDE filter ``x̃ = (I + r² L)⁻¹ x`` on the active elements; elements outside the part get 0.

    Parameters
    ----------
    domain : Domain
    rmin : float
        Radius of the cone filter it stands in for, in elements; ``r = rmin / (2√3)``.
    """

    def __init__(self, domain: Domain, rmin: float) -> None:
        super().__init__()
        if not rmin > 0:
            raise ValueError("rmin > 0 is required")
        self.rmin = float(rmin)
        self.r = self.rmin / (2.0 * math.sqrt(3.0))
        self.shape = domain.shape
        A, self._el = helmholtz_matrix(domain.active, self.r)
        self._lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
        self._n_box = int(np.prod(self.shape))

    def apply_numpy(self, x: np.ndarray) -> np.ndarray:
        """Filter a box array (``domain.shape`` or flat); used by forward and backward (A is symmetric)."""
        out = np.zeros(self._n_box)
        out[self._el] = self._lu.solve(np.asarray(x, float).reshape(-1)[self._el])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            raise ValueError("HelmholtzFilter runs on CPU tensors (SuperLU factorisation)")
        return _Solve.apply(x, self).reshape(self.shape)


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, link: HelmholtzFilter) -> torch.Tensor:
        ctx.link, ctx.dtype, ctx.shape = link, x.dtype, x.shape
        return torch.from_numpy(link.apply_numpy(x.detach().double().numpy())).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, None]:
        gx = ctx.link.apply_numpy(g.detach().double().numpy())
        return torch.from_numpy(gx).to(ctx.dtype).reshape(ctx.shape), None
