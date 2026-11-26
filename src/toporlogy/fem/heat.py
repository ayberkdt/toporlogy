"""Steady scalar conduction on the regular grid with an autograd-aware linear solve.

``K(k) T = Q`` with ``K = Σ_e k_e · s · KE`` over the active elements, one unknown (temperature) per node.
``KE`` is the unit-conductivity matrix of the bilinear square (4×4) or the trilinear unit cube (8×8),
``∫ ∇Nᵀ ∇N dV`` integrated exactly by 2-point Gauss rules per axis. The scale is ``s = h`` in 3-D (cube of edge
h: gradients ∝ 1/h, volume h³) and ``s = thickness`` in 2-D (independent of h) — the same convention as
:class:`toporlogy.fem.Elasticity`. ``k_e`` already contains the element's solid fraction (see
:class:`ConductivityInterp`).

Nodes and corners follow :class:`toporlogy.core.domain.Domain`; the node numbers of element e are
``domain.edof[e, ::dim] // dim``.

The differentiable solve reuses the autograd function of :mod:`toporlogy.fem.elastic` unchanged (forward:
factorise + solve all cases; backward: adjoint with the same factorisation, self-adjoint shortcut when ``∂J/∂T`` is a
multiple of ``Q``, refactorisation only if another forward refactorised in between). :class:`HeatConduction` offers
the interface that function needs (``solve_numpy``, ``factor``, ``_solve_free``, ``free``, ``geom``,
``element_energy``, ``_token``, counters).

Uses in this library: virtual-temperature connectivity constraints (:mod:`toporlogy.optimization.connectivity`).
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy import ndimage

from toporlogy.core.domain import Domain
from toporlogy.fem.elastic import _LinearSolve
from toporlogy.solvers.direct import DirectSolver
from toporlogy.solvers.ordering import nested_dissection_blocks

__all__ = ["conduction_matrix", "element_nodes", "assemble_conduction", "HeatConduction", "ConductivityInterp"]

_GAUSS2 = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))       # 2-point rule on [0, 1], weights 1/2


def conduction_matrix(dim: int) -> np.ndarray:
    """Unit-conductivity element matrix ``∫ ∇Nᵀ ∇N`` of the unit square (dim 2, 4×4) or unit cube (dim 3, 8×8).

    Corner order of :class:`toporlogy.core.domain.Domain`. 2-point Gauss per axis is exact (the integrand is at most
    quadratic in each coordinate). Closed form in 2-D: ``[[4, −1, −2, −1], …] / 6``.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    corners = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) if dim == 2 else
               np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]))
    n = len(corners)
    KE = np.zeros((n, n))
    w = 0.5 ** dim
    for xi_t in itertools.product(_GAUSS2, repeat=dim):
        xi = np.asarray(xi_t)
        # N_a = Π_d l(c_ad, ξ_d) with l(1, ξ) = ξ, l(0, ξ) = 1 − ξ
        lin = np.where(corners == 1, xi, 1.0 - xi)                    # (n, dim)
        dlin = np.where(corners == 1, 1.0, -1.0)                       # (n, dim)
        G = np.empty((n, dim))
        for d in range(dim):
            others = np.prod(np.delete(lin, d, axis=1), axis=1)
            G[:, d] = dlin[:, d] * others
        KE += w * G @ G.T
    return KE


def element_nodes(domain: Domain, elements: np.ndarray | None = None) -> np.ndarray:
    """(n, 2^dim) node numbers of the given elements (default: all box elements), corner order of the domain."""
    e = np.arange(domain.ne) if elements is None else np.asarray(elements, np.int64)
    return domain.edof[e][:, ::domain.dim] // domain.dim


def assemble_conduction(domain: Domain, k_active: np.ndarray) -> sp.csr_matrix:
    """Full (nn × nn) conduction matrix for conductivities ``k_active`` of the active elements (tests, inspection)."""
    en = element_nodes(domain, domain.active_elements)
    KE = conduction_matrix(domain.dim)
    s = domain.h if domain.dim == 3 else domain.thickness
    k = np.broadcast_to(np.asarray(k_active, float) * s, (len(en),))
    nl = en.shape[1]
    r = np.repeat(en, nl, axis=1).ravel()
    c = np.tile(en, (1, nl)).ravel()
    v = (k[:, None] * KE.ravel()[None, :]).ravel()
    return sp.csr_matrix((v, (r, c)), shape=(domain.nn, domain.nn))


class HeatConduction:
    """Steady conduction on a :class:`Domain` with prescribed zero temperature at ``fixed`` nodes.

    Parameters
    ----------
    domain : Domain
    fixed : sequence of int
        Node numbers with T = 0 (heat sinks). Nodes not touched by any active element are removed as well.
    solver : object with ``factor(A_upper)`` / ``solve(B)``, optional
        Default :class:`~toporlogy.solvers.DirectSolver` ("auto"); SuperLU then eliminates in nested dissection
        order (:mod:`toporlogy.solvers.ordering`).

    Raises
    ------
    ValueError
        If a node-connected part of the active region contains no fixed node (the matrix would be singular for any
        positive conductivity).

    Notes
    -----
    The "DOFs" of this model are node numbers; loads ``Q`` are nodal heat flows of shape (nn,) or (nn, n_cases).
    """

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, solver: Any = None) -> None:
        self.domain = domain
        self.KE = conduction_matrix(domain.dim)
        self.geom = domain.h if domain.dim == 3 else domain.thickness
        self.elements = domain.active_elements
        self.edof = element_nodes(domain, self.elements)              # (na, 2^dim) node numbers
        fixed_arr = np.unique(np.asarray(fixed, np.int64))
        if fixed_arr.size and (fixed_arr.min() < 0 or fixed_arr.max() >= domain.nn):
            raise ValueError("fixed node out of range")
        active_nodes = np.unique(self.edof)
        self.fixed = np.intersect1d(fixed_arr, active_nodes)
        if self.fixed.size == 0:
            raise ValueError("no fixed node on the active region: the conduction matrix is singular")
        self._check_sinks(active_nodes)
        self.free = np.setdiff1d(active_nodes, self.fixed)
        if self.free.size == 0:
            raise ValueError("no free node")
        nn, nf, nl = domain.nn, len(self.free), self.KE.shape[0]
        gmap = np.full(nn, -1, np.int64)
        gmap[self.free] = np.arange(nf)
        r = gmap[np.repeat(self.edof, nl, axis=1).ravel()]
        c = gmap[np.tile(self.edof, (1, nl)).ravel()]
        keep = (r >= 0) & (c >= 0) & (r <= c)
        self._sel = np.flatnonzero(keep)
        key = r[keep] * nf + c[keep]
        ukey, inv = np.unique(key, return_inverse=True)
        self._inv = inv.astype(np.int64)
        self._P = sp.csr_matrix((np.zeros(len(ukey)), (ukey // nf, ukey % nf)), shape=(nf, nf))
        self.solver = solver if solver is not None else DirectSolver()
        self._order: np.ndarray | None = None
        self._blocks: np.ndarray | None = None
        self._token = 0
        self._factored_E: np.ndarray | None = None
        self.n_solves = 0
        self.n_adjoint_solves = 0

    def _check_sinks(self, active_nodes: np.ndarray) -> None:
        d = self.domain
        lab, n = ndimage.label(d.active, structure=np.ones((3,) * d.dim, bool))   # node sharing = vertex contact
        if n <= 1:
            return
        is_fixed = np.zeros(d.nn, bool)
        is_fixed[self.fixed] = True
        comp = lab.ravel()[self.elements]
        hit = np.zeros(n + 1, bool)
        np.logical_or.at(hit, comp, is_fixed[self.edof].any(axis=1))
        if not hit[1:].all():
            raise ValueError(f"{int((~hit[1:]).sum())} of {n} separate parts of the active region have no fixed "
                             "node: the conduction matrix would be singular")

    # ------------------------------------------------------------------ assembly and solve (NumPy)
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def nn(self) -> int:
        return self.domain.nn

    def matrix(self, k_active: np.ndarray) -> sp.csr_matrix:
        """Upper triangle of the reduced (free-node) conduction matrix."""
        s = np.broadcast_to(np.asarray(k_active, float) * self.geom, (self.n_elements,))
        w = (s[:, None] * self.KE.ravel()[None, :]).ravel()[self._sel]
        A = self._P.copy()
        A.data = np.bincount(self._inv, weights=w, minlength=self._P.nnz)
        return A

    def factor(self, k_active: np.ndarray) -> None:
        k_active = np.asarray(k_active, float)
        A = self.matrix(k_active)
        if isinstance(self.solver, DirectSolver):              # SPD: nested dissection of the node grid
            if self._order is None:
                self._order, self._blocks = nested_dissection_blocks(tuple(n + 1 for n in self.domain.shape), 1,
                                                                     self.free)
            self.solver.factor(A, order=self._order, blocks=self._blocks)
        else:
            self.solver.factor(A)
        self._factored_E = k_active.copy()
        self._token += 1

    def _solve_free(self, B_full: np.ndarray) -> np.ndarray:
        X = np.zeros_like(B_full)
        X[self.free] = self.solver.solve(B_full[self.free]).reshape(B_full[self.free].shape)
        return X

    def solve_numpy(self, k_active: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """T (nn, n_cases) for conductivities ``k_active`` (na,) and nodal heat ``Q`` (nn,) or (nn, n_cases)."""
        Q2 = np.asarray(Q, float)
        Q2 = Q2[:, None] if Q2.ndim == 1 else Q2
        if Q2.shape[0] != self.nn:
            raise ValueError(f"Q has {Q2.shape[0]} rows, the domain {self.nn} nodes")
        self.factor(k_active)
        self.n_solves += 1
        return self._solve_free(Q2)

    def element_energy(self, T: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        """(na, n_cases) ``v_eᵀ KE t_e`` per active element and case (unit conductivity, without ``geom``)."""
        T = T[:, None] if T.ndim == 1 else T
        V = T if V is None else (V[:, None] if V.ndim == 1 else V)
        Te, Ve = T[self.edof], V[self.edof]
        return np.einsum("eac,ab,ebc->ec", Ve, self.KE, Te)

    # ------------------------------------------------------------------ autograd
    def solve(self, k_active: torch.Tensor, Q: np.ndarray) -> torch.Tensor:
        """Differentiable ``T(k)``: (nn, n_cases) float64 tensor; ``k_active`` (na,) tensor."""
        Q2 = np.asarray(Q, float)
        Q2 = Q2[:, None] if Q2.ndim == 1 else Q2
        return _LinearSolve.apply(k_active, self, Q2)


class ConductivityInterp(torch.nn.Module):
    """``k_e = (k_min + x̄_e^p (1 − k_min)) · k0 · frac_e`` — solid conducts, void insulates.

    ``kmin_ratio`` is the void/solid conductivity ratio; ``penal`` the exponent p (p > 1 makes grey material a poor
    conductor, so a grey film cannot fake a connection).
    """

    frac: torch.Tensor

    def __init__(self, domain: Domain, k0: float = 1.0, penal: float = 3.0, kmin_ratio: float = 1e-3) -> None:
        super().__init__()
        if not 0 < kmin_ratio < 1:
            raise ValueError("0 < kmin_ratio < 1 is required")
        self.k0, self.penal, self.kmin_ratio = float(k0), float(penal), float(kmin_ratio)
        self.register_buffer("frac", torch.as_tensor(domain.frac, dtype=torch.float64))

    def forward(self, xphys: torch.Tensor) -> torch.Tensor:
        km = self.kmin_ratio
        return (km + xphys ** self.penal * (1.0 - km)) * self.k0 * self.frac.to(xphys.dtype)
