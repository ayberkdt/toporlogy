"""Linear elasticity on the regular grid with an autograd-aware linear solve (docs/mimari_plan.md §3.1, D1).

``K(E) U = F`` with ``K = Σ_e E_e · s · KE`` over the active elements, where ``s = h`` in 3-D (unit cube scaled to
edge h: B ∝ 1/h, volume h³) and ``s = thickness`` in 2-D (plane stress, square element of any size). ``E_e`` already
contains the element's solid fraction (the interpolation multiplies by ``frac``).

:meth:`Elasticity.solve` is a ``torch.autograd.Function``: the forward pass factorises and solves; the backward pass
solves the adjoint system with the same factorisation, ``K λ = ∂J/∂U``, and returns
``∂J/∂E_e = −s · λ_eᵀ KE u_e``. When ``∂J/∂U`` is a multiple of ``F`` (compliance), ``λ = a·U`` is used without a
solve (self-adjoint shortcut, exact).

Element matrices: :mod:`toporlogy.fem.elements` (H8, H8IM statically condensed, Q4).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.mechanisms import SolveResidualError, check_supports, fixed_dof_rows
from toporlogy.solvers.direct import DirectSolver

__all__ = ["Elasticity", "element_matrix"]


from toporlogy.fem.elements import element_matrix  # noqa: E402  (re-exported)


class Elasticity:
    """Static linear elasticity on a :class:`Domain` with fixed DOFs.

    Parameters
    ----------
    domain : Domain
    fixed : sequence of int
        Box DOF numbers with zero displacement. DOFs not touched by any active element are removed as well.
    nu : float
        Poisson's ratio.
    element : str, optional
        "H8" (default in 3-D), "H8IM", or "Q4" (2-D).
    solver : object with ``factor(A_upper)`` / ``solve(B)``, optional
        Default :class:`DirectSolver` ("auto").
    """

    def __init__(self, domain: Domain, fixed: Sequence[int] | np.ndarray, nu: float = 0.3, element: str | None = None,
                 solver: Any = None) -> None:
        self.domain = domain
        self.nu = float(nu)
        self.element = element or ("H8" if domain.dim == 3 else "Q4")
        self.KE = element_matrix(domain.dim, self.element, nu)
        self.geom = domain.h if domain.dim == 3 else domain.thickness
        self.elements = domain.active_elements
        self.edof = domain.edof[self.elements]                      # (na, nloc)
        fixed_arr = np.unique(np.asarray(fixed, np.int64))
        if fixed_arr.size and (fixed_arr.min() < 0 or fixed_arr.max() >= domain.ndof):
            raise ValueError("fixed DOF out of range")
        self.fixed = fixed_arr
        self.free = np.setdiff1d(domain.active_dofs, fixed_arr)
        if self.free.size == 0:
            raise ValueError("no free DOF")
        nd, nf, nl = domain.ndof, len(self.free), self.KE.shape[0]
        gmap = np.full(nd, -1, np.int64)
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
        self._token = 0                                            # increments with every factorisation
        self._factored_E: np.ndarray | None = None
        self.n_solves = 0
        self.n_adjoint_solves = 0
        self._kinematics_checked = False
        self._A: sp.csr_matrix | None = None                     # reduced matrix of the last factorisation
        self.last_residual: float | None = None                    # ‖K x − b‖ / ‖b‖ of the last solve

    # ------------------------------------------------------------------ assembly and solve (NumPy)
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def matrix(self, E_active: np.ndarray) -> sp.csr_matrix:
        """Upper triangle of the reduced stiffness for element moduli ``E_active`` (na,)."""
        s = np.broadcast_to(np.asarray(E_active, float) * self.geom, (self.n_elements,))
        w = (s[:, None] * self.KE.ravel()[None, :]).ravel()[self._sel]
        A = self._P.copy()
        A.data = np.bincount(self._inv, weights=w, minlength=self._P.nnz)
        return A

    def _kinematic_rows(self) -> dict[int, np.ndarray]:
        """Support rows per node acting on its displacement (fixed DOFs: unit rows); see :mod:`.mechanisms`."""
        return fixed_dof_rows(self.fixed, self.domain.dim)

    def check_kinematics(self) -> None:
        """Raise :class:`~toporlogy.fem.mechanisms.MechanismError` if the supports leave a zero-energy motion (a
        floating piece, a hinge, a free rotation): the stiffness would be singular and a direct solve would return
        finite nonsense. Runs once, before the first factorisation."""
        check_supports(self.domain, self.elements, self._kinematic_rows())
        self._kinematics_checked = True

    def factor(self, E_active: np.ndarray) -> None:
        if not self._kinematics_checked:
            self.check_kinematics()
        E_active = np.asarray(E_active, float)
        self._A = self.matrix(E_active)
        self.solver.factor(self._A)
        self._factored_E = E_active.copy()
        self._token += 1

    def _solve_free(self, B_full: np.ndarray) -> np.ndarray:
        X = np.zeros_like(B_full)
        b = B_full[self.free]
        x = np.asarray(self.solver.solve(b)).reshape(b.shape)
        self.last_residual = checked_residual(self._A, x, b)
        X[self.free] = x
        return X

    def solve_numpy(self, E_active: np.ndarray, F: np.ndarray) -> np.ndarray:
        """U (ndof, n_cases) for moduli ``E_active`` and loads ``F`` (ndof,) or (ndof, n_cases); no autograd."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        self.factor(E_active)
        self.n_solves += 1
        return self._solve_free(F2)

    def element_energy(self, U: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        """(na, n_cases) ``v_eᵀ KE u_e`` per active element and case (unit modulus, without ``geom``)."""
        U = U[:, None] if U.ndim == 1 else U
        V = U if V is None else (V[:, None] if V.ndim == 1 else V)
        Ue, Ve = U[self.edof], V[self.edof]                         # (na, nloc, nc)
        return np.einsum("eac,ab,ebc->ec", Ve, self.KE, Ue)

    # ------------------------------------------------------------------ autograd
    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        """Differentiable ``U(E)``: (ndof, n_cases) float64 tensor. ``E_active`` (na,) tensor."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        return _LinearSolve.apply(E_active, self, F2)


RESIDUAL_TOL = 1e-6        # a backward-stable direct solve gives ~1e-15; above this the solution is not usable


def checked_residual(A: sp.csr_matrix | None, x: np.ndarray, b: np.ndarray) -> float | None:
    """‖K x − b‖ / ‖b‖ with K = A + Aᵀ − diag(A) (A: upper triangle); raises SolveResidualError above
    :data:`RESIDUAL_TOL`. None when there is no matrix (a custom solver path) or b = 0."""
    if A is None:
        return None
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return 0.0
    x2 = x.reshape(len(x), -1)
    Kx = A @ x2 + A.T @ x2 - A.diagonal()[:, None] * x2
    rel = float(np.linalg.norm(Kx - b.reshape(x2.shape))) / nb
    if not np.isfinite(rel) or rel > RESIDUAL_TOL:
        raise SolveResidualError(f"linear solve residual ‖Kx − b‖/‖b‖ = {rel:.2e} > {RESIDUAL_TOL:g}: the system "
                                 "is singular or numerically not solvable at these moduli")
    return rel


class _LinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E_active: torch.Tensor, model: Elasticity, F: np.ndarray) -> torch.Tensor:
        E_np = E_active.detach().cpu().double().numpy()
        U = model.solve_numpy(E_np, F)
        ctx.model, ctx.F, ctx.U, ctx.E, ctx.token = model, F, U, E_np, model._token
        ctx.device, ctx.dtype = E_active.device, E_active.dtype
        return torch.from_numpy(U).to(device=E_active.device, dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, None, None]:
        model: Elasticity = ctx.model
        G = gU.detach().cpu().double().numpy()
        U, F = ctx.U, ctx.F
        lam = np.zeros_like(U)
        need = []
        fr = model.free
        for c in range(U.shape[1]):
            g, f = G[fr, c], F[fr, c]
            ff = float(f @ f)
            if ff > 0:
                a = float(g @ f) / ff
                if np.linalg.norm(g - a * f) <= 1e-12 * max(float(np.linalg.norm(g)), 1e-300):
                    lam[:, c] = a * U[:, c]                     # self-adjoint: K λ = a F  →  λ = a U
                    continue
            if np.any(g):
                need.append(c)
        if need:
            if model._token != ctx.token:                      # another forward refactorised: restore ours
                model.factor(ctx.E)
            sol = model._solve_free(G[:, need])
            lam[:, need] = sol
            model.n_adjoint_solves += 1
        dE = -model.geom * model.element_energy(U, lam).sum(axis=1)
        return torch.from_numpy(dE).to(device=ctx.device, dtype=ctx.dtype), None, None
