"""Linear elasticity solved by geometric multigrid (:class:`toporlogy.solvers.gmg.GMGSolver`) — phase A2.

Drop-in physics for :class:`toporlogy.Study` with the autograd contract of :class:`toporlogy.Elasticity`:
``solve(E_active_tensor, F) -> U`` (ndof, n_cases) float64, differentiable in ``E_active``. Forward: GMG-PCG,
warm-started from the previous solution of the same load case. Backward: ``∂J/∂E_e = −h · λ_eᵀ KE u_e`` with the
self-adjoint shortcut ``λ = a·U`` when ``∂J/∂U = a·F`` (compliance), else an adjoint GMG solve with the hierarchy of
the forward operator.

The grid is padded at the upper end of each axis to a multiple of ``2^(n_geometric − 1)`` with zero-modulus
elements; DOFs that no element stiffens (padding, nodes outside a masked part) are pinned, which is the reduced
system of :class:`Elasticity` (its free set is ``active_dofs \\ fixed``).

Fallback to the sparse direct solver (:class:`toporlogy.solvers.DirectSolver` through an :class:`Elasticity`
built on first use): when the problem has fewer than ``direct_below`` active elements, or when CG does not reach
``tol`` within ``maxit`` and the problem has at most ``fallback_max_elements`` active elements. Every fallback is
counted in :attr:`n_fallbacks` and logged in :attr:`fallback_log` with its reason.

Tolerance contract (fail-closed): a solve is accepted only when the recomputed relative residual
‖F − K u‖/‖F‖ ≤ ``tol``. If CG misses it and no direct fallback is allowed (problem larger than
``fallback_max_elements``), :class:`toporlogy.solvers.gmg.GMGConvergenceError` is raised. Only with
``allow_inexact=True`` is the unconverged result returned; it is then counted in :attr:`n_inexact`, logged in
:attr:`fallback_log` (``"inexact": True``) and a ``RuntimeWarning`` is issued.
"""
from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch

from toporlogy.core.domain import Domain
from toporlogy.fem.elements import element_matrix
from toporlogy.fem.mechanisms import check_supports, fixed_dof_rows
from toporlogy.solvers import gmg_kernels as gk
from toporlogy.solvers.gmg import GMGConvergenceError, GMGSolver

__all__ = ["ElasticityGMG"]


class ElasticityGMG:
    """Static linear elasticity on a 3-D :class:`Domain` solved by GMG-PCG.

    Parameters
    ----------
    domain : Domain (3-D)
    fixed : sequence of int
        Box DOF numbers with zero displacement.
    nu : float
    element : {"H8", "H8IM"}
    tol : float
        Relative residual ‖K u − F‖ / ‖F‖ of every forward and adjoint solve.
    maxit : int
        CG iteration cap before the fallback.
    direct_below : int
        Problems with fewer active elements are solved by the direct solver (no GMG).
    fallback_max_elements : int
        Largest problem (active elements) for which the direct fallback is attempted.
    warm_start : bool
        Start CG from the previous solution of the same load case.
    allow_inexact : bool
        Default False (strict, fail-closed): an unconverged solve without a permitted direct fallback raises
        :class:`GMGConvergenceError`. True keeps the unconverged result, marked in :attr:`n_inexact` / the log.
    device : optional
        Torch device of the solver (default CUDA when available).
    **gmg : keyword arguments of :class:`GMGSolver` (coarse, n_geometric, smoother_degree, pin_rel, …).
    """

    def __init__(self, domain: Domain, fixed: Sequence[int], nu: float = 0.3, element: str | None = None,
                 tol: float = 1e-8, maxit: int = 300, direct_below: int = 4000,
                 fallback_max_elements: int = 120_000, warm_start: bool = True, device: Any = None,
                 allow_inexact: bool = False, **gmg: Any) -> None:
        if domain.dim != 3:
            raise ValueError("ElasticityGMG is 3-D only (use Elasticity in 2-D)")
        self.domain = domain
        self.nu = float(nu)
        self.element = element or "H8"
        self.KE = element_matrix(3, self.element, nu)
        self.geom = domain.h
        self.elements = domain.active_elements
        self._kinematics_checked = False
        fixed_arr = np.unique(np.asarray(fixed, np.int64))
        if fixed_arr.size and (fixed_arr.min() < 0 or fixed_arr.max() >= domain.ndof):
            raise ValueError("fixed DOF out of range")
        self.fixed = fixed_arr
        self.tol, self.maxit = float(tol), int(maxit)
        self.direct_below, self.fallback_max_elements = int(direct_below), int(fallback_max_elements)
        self.warm_start = bool(warm_start)
        self.allow_inexact = bool(allow_inexact)
        ng = gmg.get("n_geometric")
        if ng is None:
            ng = gmg["n_geometric"] = self.default_levels(domain.shape, gmg.get("max_coarse_dof", 4000))
        m = 2 ** (int(ng) - 1)
        self.shape_pad = cast("tuple[int, int, int]", tuple(int(-(-s // m) * m) for s in domain.shape))  # dim 3
        self.gmg = GMGSolver(self.shape_pad, self.KE, self.geom, device=device, tol=tol, maxit=maxit, **gmg)
        self.solver = self.gmg
        self.gmg.name = f"gmg-{self.gmg.device.type}"     # type: ignore[attr-defined]
        nx, ny, nz = domain.shape
        self._node_slice = (slice(0, nx + 1), slice(0, ny + 1), slice(0, nz + 1))
        fm = np.zeros(self.gmg.node_shape, bool)
        fm[self._node_slice] = np.isin(np.arange(domain.ndof), fixed_arr).reshape(nx + 1, ny + 1, nz + 1, 3)
        self._fixed_mask = fm
        el = np.zeros(self.shape_pad, bool)
        el[:nx, :ny, :nz] = domain.active
        self._pad_elements = torch.as_tensor(np.flatnonzero(el.ravel()), device=self.gmg.device)
        self._direct: Any = None
        self._free: np.ndarray | None = None
        self._E_set: torch.Tensor | None = None
        self._token = 0
        self._prev: dict[int, torch.Tensor] = {}
        self.n_solves = 0
        self.n_adjoint_solves = 0
        self.n_fallbacks = 0
        self.n_inexact = 0
        self.fallback_log: list[dict[str, Any]] = []
        self.solve_log: list[dict[str, Any]] = []
        self.last_iters = 0

    @property
    def strict(self) -> bool:
        """True unless ``allow_inexact`` (fail-closed tolerance contract)."""
        return not self.allow_inexact

    @staticmethod
    def default_levels(shape: Sequence[int], max_coarse_dof: int = 4000) -> int:
        """Fewest geometric grids whose coarsest (after padding to multiples of 2^(levels−1)) has at most
        ``max_coarse_dof`` DOFs (dense coarse solve)."""
        for lev in range(2, 12):
            m = 2 ** (lev - 1)
            c = [-(-int(s) // m) for s in shape]
            if 3 * int(np.prod([v + 1 for v in c])) <= max_coarse_dof:
                return lev
        return 12

    # ------------------------------------------------------------------ helpers
    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def edof(self) -> np.ndarray:
        return self.domain.edof[self.elements]

    @property
    def free(self) -> np.ndarray:
        r"""Free DOFs of the reduced system (``active_dofs \ fixed``, as in :class:`Elasticity`), cached."""
        if self._free is None:
            self._free = np.setdiff1d(self.domain.active_dofs, self.fixed)
        return self._free

    def _grid_E(self, E_active: torch.Tensor) -> torch.Tensor:
        Eg = torch.zeros(int(np.prod(self.shape_pad)), dtype=torch.float64, device=self.gmg.device)
        Eg[self._pad_elements] = E_active.detach().to(device=self.gmg.device, dtype=torch.float64)
        return Eg

    def _to_pad(self, B: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Box DOF array (ndof, k) → padded nodal (np, 3, k) on the solver device."""
        nx, ny, nz = self.domain.shape
        Bt = torch.as_tensor(B, dtype=torch.float64).to(self.gmg.device)
        k = Bt.shape[1]
        out = torch.zeros(self.gmg.node_shape + (k,), dtype=torch.float64, device=self.gmg.device)
        out[self._node_slice] = Bt.reshape(nx + 1, ny + 1, nz + 1, 3, k)
        return out

    def _from_pad(self, X: torch.Tensor) -> torch.Tensor:
        return X[self._node_slice].reshape(self.domain.ndof, -1)

    def _direct_model(self) -> Any:
        if self._direct is None:
            from toporlogy.fem.elastic import Elasticity
            self._direct = Elasticity(self.domain, self.fixed, nu=self.nu, element=self.element)
        return self._direct

    def _set_operator(self, E_active: torch.Tensor) -> None:
        if self._E_set is not None and self._E_set.shape == E_active.shape and torch.equal(self._E_set, E_active):
            return
        self.gmg.update(self._grid_E(E_active), self._fixed_mask if self.gmg.A64 is None else None)
        self._E_set = E_active.detach().clone()
        self._token += 1

    def _fallback(self, reason: str, kind: str, E_np: np.ndarray, B: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        d = self._direct_model()
        X = d.solve_numpy(E_np, B)
        self.n_fallbacks += 1
        self.fallback_log.append({"reason": reason, "kind": kind, "seconds": time.perf_counter() - t0,
                                  "solve": self.n_solves})
        return X

    def _solve_box(self, E_active: torch.Tensor, B: np.ndarray, kind: str,
                   x0: torch.Tensor | None = None) -> np.ndarray:
        """Solve K(E) X = B (box DOFs, (ndof, k)) by GMG with fallback. Returns X (ndof, k) NumPy."""
        if not self._kinematics_checked:                           # a mechanism would only show as no convergence
            check_supports(self.domain, self.elements, fixed_dof_rows(self.fixed, self.domain.dim))
            self._kinematics_checked = True
        E_cpu = E_active.detach().cpu().double()
        if self.n_elements < self.direct_below:
            return self._fallback(f"small problem: {self.n_elements} < direct_below={self.direct_below} elements",
                                  kind, E_cpu.numpy(), B)
        self._set_operator(E_cpu)
        Bp = self._to_pad(B)
        X = torch.zeros_like(Bp)
        bad = []
        for c in range(B.shape[1]):
            xi = None if x0 is None else x0[..., c]
            X[..., c] = self.gmg.solve(None, Bp[..., c], x0=xi, tol=self.tol, maxit=self.maxit)
            info = dict(self.gmg.last)
            info.update(kind=kind, case=c)
            self.solve_log.append({k: info[k] for k in ("kind", "case", "iters", "requested_tol",
                                                        "achieved_true_rel_residual", "solve_s", "converged")})
            self.last_iters = info["iters"]
            if not info["converged"]:
                bad.append((c, info))
        Xb = self._from_pad(X).cpu().numpy()
        for c, info in bad:
            reason = (f"CG not converged ({kind}, load case {c}): {info['iters']} iterations, achieved true "
                      f"relative residual {info['achieved_true_rel_residual']:.3e} > requested tol {self.tol:.3e}")
            if self.n_elements <= self.fallback_max_elements:
                Xb[:, c] = self._fallback(reason, kind, E_cpu.numpy(), B[:, [c]])[:, 0]
            elif self.allow_inexact:
                self.n_inexact += 1
                warnings.warn(f"ElasticityGMG: {reason}; allow_inexact=True, the unconverged result is returned",
                              RuntimeWarning, stacklevel=3)
                self.fallback_log.append({"reason": reason, "kind": kind, "inexact": True, "seconds": 0.0,
                                          "solve": self.n_solves,
                                          "achieved_true_rel_residual": info["achieved_true_rel_residual"]})
            else:
                raise GMGConvergenceError(
                    f"ElasticityGMG: {reason}; {self.n_elements} active elements > fallback_max_elements="
                    f"{self.fallback_max_elements}, so no direct fallback. Raise maxit, loosen tol, or pass "
                    "allow_inexact=True to accept the inexact result explicitly.")
        return Xb

    # ------------------------------------------------------------------ public
    def solve_numpy(self, E_active: np.ndarray, F: np.ndarray) -> np.ndarray:
        """U (ndof, n_cases) for moduli ``E_active`` (na,) and loads ``F``; no autograd, warm start if enabled."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        x0 = None
        if self.warm_start and len(self._prev) == F2.shape[1]:
            x0 = torch.stack([self._prev[c] for c in range(F2.shape[1])], dim=-1)
        U = self._solve_box(torch.as_tensor(np.asarray(E_active, float)), F2, "forward", x0)
        self.n_solves += 1
        if self.warm_start:
            Up = self._to_pad(U)
            self._prev = {c: Up[..., c] for c in range(U.shape[1])}
        return U

    def element_energy(self, U: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
        """(na, n_cases) ``v_eᵀ KE u_e`` per active element (unit modulus, without ``geom``), on the GMG device."""
        U = U[:, None] if U.ndim == 1 else U
        V = U if V is None else (V[:, None] if V.ndim == 1 else V)
        Up, Vp = self._to_pad(U), self._to_pad(V)
        KE = torch.as_tensor(self.KE, dtype=torch.float64, device=self.gmg.device)
        out = []
        for c in range(U.shape[1]):
            ue = gk.gather(Up[..., c], self.shape_pad)
            ve = gk.gather(Vp[..., c], self.shape_pad)
            out.append(((ve @ KE) * ue).sum(1)[self._pad_elements])
        return torch.stack(out, 1).cpu().numpy()

    def solve(self, E_active: torch.Tensor, F: np.ndarray) -> torch.Tensor:
        """Differentiable ``U(E)``: (ndof, n_cases) float64 tensor. ``E_active`` (na,) tensor."""
        F2 = np.asarray(F, float)
        F2 = F2[:, None] if F2.ndim == 1 else F2
        return _GMGLinearSolve.apply(E_active, self, F2)

    def stats(self) -> dict[str, Any]:
        it = [r["iters"] for r in self.solve_log]
        return {"n_solves": self.n_solves, "n_adjoint_solves": self.n_adjoint_solves,
                "n_fallbacks": self.n_fallbacks, "n_inexact": self.n_inexact,
                "fallback_reasons": [f["reason"] for f in self.fallback_log],
                "cg_iters_mean": float(np.mean(it)) if it else 0.0, "cg_iters_max": int(max(it)) if it else 0,
                "padded_shape": self.shape_pad, "setup": getattr(self.gmg, "last_setup", None)}


class _GMGLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, E_active: torch.Tensor, model: ElasticityGMG, F: np.ndarray) -> torch.Tensor:
        E_np = E_active.detach().cpu().double().numpy()
        U = model.solve_numpy(E_np, F)
        ctx.model, ctx.F, ctx.U, ctx.E = model, F, U, E_np
        ctx.device, ctx.dtype = E_active.device, E_active.dtype
        return torch.from_numpy(U).to(device=E_active.device, dtype=torch.float64)

    @staticmethod
    def backward(ctx: Any, gU: torch.Tensor) -> tuple[Any, None, None]:
        model: ElasticityGMG = ctx.model
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
            Gs = G[:, need].copy()
            Gs[model.fixed] = 0.0
            lam[:, need] = model._solve_box(torch.as_tensor(ctx.E), Gs, "adjoint")
            model.n_adjoint_solves += 1
        dE = -model.geom * model.element_energy(U, lam).sum(axis=1)
        return torch.from_numpy(dE).to(device=ctx.device, dtype=ctx.dtype), None, None
