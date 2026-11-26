"""Implicit differentiation of a state defined by a residual: R(y, x) = 0 (05 §1).

:class:`ImplicitState` solves R(y, x) = 0 for y by Newton's method (full step with backtracking on |R|; residual
and step reported separately) and differentiates the solution with the adjoint of the converged Jacobian:

    R_yᵀ λ = ∂J/∂y,        dJ/dx = −λᵀ ∂R/∂x   (a vector-Jacobian product of R in x, by torch).

The Jacobian R_y comes from torch autograd (dense; small problems) or from a user ``jacobian(y, x)`` returning a
SciPy sparse or dense matrix. Several responses of one state reuse one factorisation of the converged Jacobian for
their transposed solves. Errors fail closed with their own types: :class:`NotConverged`, :class:`NonFiniteResidual`,
:class:`SingularJacobian` (all :class:`ImplicitSolveError`). The derivative is that of the solution branch found;
it is valid where R_y is non-singular. Pre-registered checks: ``docs/olcumler/ortuk_turev_onkayit.md`` (I1-I6).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

__all__ = ["ImplicitState", "ImplicitSolveError", "NotConverged", "NonFiniteResidual", "SingularJacobian"]


class ImplicitSolveError(RuntimeError):
    """The state equation could not be solved; ``history`` holds the residual norms."""

    def __init__(self, message: str, history: list[float]) -> None:
        super().__init__(f"{message}; |R| history [{', '.join(f'{h:.3e}' for h in history)}]")
        self.history = history


class NotConverged(ImplicitSolveError):
    pass


class NonFiniteResidual(ImplicitSolveError):
    pass


class SingularJacobian(ImplicitSolveError):
    pass


class _Factor:
    """LU of a dense or sparse matrix with plain and transposed solves."""

    def __init__(self, J: Any, history: list[float]) -> None:
        if sp.issparse(J):
            try:
                self._lu: Any = spla.splu(sp.csc_matrix(J))
            except RuntimeError as e:
                raise SingularJacobian(f"singular Jacobian: {e}", history) from None
            self.sparse = True
        else:
            Jd = np.asarray(J, float)
            if not np.isfinite(Jd).all():
                raise NonFiniteResidual("the Jacobian is not finite", history)
            if np.linalg.cond(Jd) > 1e15:
                raise SingularJacobian(f"singular Jacobian (cond {np.linalg.cond(Jd):.2e})", history)
            self._lu = sla.lu_factor(Jd)
            self.sparse = False

    def solve(self, b: np.ndarray, transpose: bool = False) -> np.ndarray:
        if self.sparse:
            return np.asarray(self._lu.solve(b, trans="T" if transpose else "N"))
        return np.asarray(sla.lu_solve(self._lu, b, trans=1 if transpose else 0))


class _Apply(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, state: ImplicitState, y0: torch.Tensor | None) -> torch.Tensor:
        y = state._newton(x.detach(), None if y0 is None else y0.detach())
        ctx.state, ctx.x, ctx.y, ctx.factor = state, x.detach(), y, None
        return y.clone()

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, None, None]:
        state: ImplicitState = ctx.state
        if ctx.factor is None:
            ctx.factor = _Factor(state._jacobian(ctx.y, ctx.x), [])
            state.n_adjoint_factorizations += 1
        G = g.detach().cpu().double().numpy()
        lam = ctx.factor.solve(G, transpose=state.transpose)
        state.n_adjoint_solves += 1
        with torch.enable_grad():
            xr = ctx.x.clone().requires_grad_(True)
            R = state.residual(ctx.y, xr)
            (gx,) = torch.autograd.grad(R, xr, grad_outputs=-torch.as_tensor(lam, dtype=R.dtype), allow_unused=True)
        return (torch.zeros_like(ctx.x) if gx is None else gx), None, None


class ImplicitState:
    """y(x) with R(y(x), x) = 0. ``residual(y, x)`` maps torch tensors to a tensor of the shape of y.

    ``rtol`` / ``atol``: |R| ≤ atol + rtol · |R(y0)|; ``step_tol``: the Newton step must also be below
    step_tol · (1 + |y|) (a small step alone is not success). ``transpose=False`` exists only for the negative control
    of the tests (it solves R_y λ = g, which is wrong unless R_y is symmetric)."""

    def __init__(self, residual: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 jacobian: Callable[[np.ndarray, np.ndarray], Any] | None = None, rtol: float = 1e-12,
                 atol: float = 1e-14, step_tol: float = 1e-10, max_iter: int = 50, transpose: bool = True) -> None:
        self.residual, self.user_jacobian = residual, jacobian
        self.rtol, self.atol, self.step_tol, self.max_iter, self.transpose = rtol, atol, step_tol, max_iter, transpose
        self.n_forward_factorizations = self.n_adjoint_factorizations = self.n_adjoint_solves = 0
        self.history: list[float] = []

    def _jacobian(self, y: torch.Tensor, x: torch.Tensor) -> Any:
        if self.user_jacobian is not None:
            return self.user_jacobian(y.detach().cpu().double().numpy(), x.detach().cpu().double().numpy())
        J = torch.autograd.functional.jacobian(lambda v: self.residual(v, x), y)
        return J.reshape(y.numel(), y.numel()).detach().cpu().double().numpy()

    def _newton(self, x: torch.Tensor, y0: torch.Tensor | None) -> torch.Tensor:
        y = torch.zeros(self._size(x), dtype=x.dtype) if y0 is None else y0.clone()
        r = self.residual(y, x).detach()
        hist = [float(r.norm())]
        self.history = hist
        if not torch.isfinite(r).all():
            raise NonFiniteResidual("the residual is not finite at the start", hist)
        target = self.atol + self.rtol * hist[0]
        for _ in range(self.max_iter):
            if hist[-1] <= target:
                return y
            F = _Factor(self._jacobian(y, x), hist)
            self.n_forward_factorizations += 1
            dy = torch.as_tensor(F.solve(-r.cpu().double().numpy()), dtype=y.dtype)
            a = 1.0
            for _ in range(30):
                yn = y + a * dy
                rn = self.residual(yn, x).detach()
                if torch.isfinite(rn).all() and float(rn.norm()) <= (1 - 1e-4 * a) * hist[-1]:
                    break
                a *= 0.5
            else:
                raise NotConverged("no decrease of |R| along the Newton direction (line search failed)", hist)
            y, r = yn, rn
            hist.append(float(r.norm()))
            if hist[-1] <= target and float((a * dy).norm()) <= self.step_tol * (1 + float(y.norm())):
                return y
        if hist[-1] <= target:
            return y
        raise NotConverged(f"Newton did not converge in {self.max_iter} iterations", hist)

    def _size(self, x: torch.Tensor) -> int:
        if not hasattr(self, "size"):
            raise ValueError("give y0 or set ImplicitState.size (the number of unknowns)")
        return int(self.size)

    def solve(self, x: torch.Tensor, y0: torch.Tensor | None = None) -> torch.Tensor:
        """The state y for x, differentiable in x."""
        out: torch.Tensor = _Apply.apply(x, self, y0)
        return out
