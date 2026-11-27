"""Topology optimisation under nonlinear Norton creep with the exact discrete history adjoint
(``docs/foundations/prereg/CO_creep_topology.md``, CO1-CO5).

Material: the J2 Norton creep law of :class:`~toporlogy.core.creep.NortonCreep` (backward-Euler radial return,
ṗ = A (q/σ_ref)^n), plane strain. Interpolation: only the elastic moduli are design dependent, K_e = w_e K0,
G_e = w_e G0 with w_e = E_min + x̄_e^p (1 − E_min); A, σ_ref and n stay physical in every element.

Adjoint. The converged step displacement u_n solves R_n(u_n; w, ε_cr,n−1) = Σ_e ∫ Bᵀσ dV − f_n = 0 (Newton in NumPy
with the algorithmic tangent of the update). The step is a torch autograd function whose backward solves
K_tᵀ λ = ū and returns −(∂R/∂w)ᵀλ, −(∂R/∂ε_cr,n−1)ᵀλ from torch autograd of R at the converged u_n (implicit
function theorem). The new creep strain is evaluated in torch; its local root Δp is found numerically
(:func:`toporlogy.core.creep._solve_increment`) and then corrected by one Newton step written in torch,
Δp = Δp* − g(Δp*)/g′(Δp*), which carries the exact first derivative of the root. The march is differentiated
exactly on the fixed time grid.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from toporlogy.core.creep import _solve_increment
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.assembly import Pattern
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.quadrature import rule

__all__ = ["CreepStructure", "CreepResult", "norton_update_torch"]

_EYE = torch.eye(3, dtype=torch.float64)
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(_ONE6, _ONE6) / 3.0
_VA = np.array([0, 1, 2, 1, 0, 0])
_VB = np.array([0, 1, 2, 2, 2, 1])
_IP = [0, 1, 5]


def norton_update_torch(eps: torch.Tensor, ecr_n: torch.Tensor, K: torch.Tensor, G: torch.Tensor, dt: float,
                        A: float, sigma_ref: float, n: float, tol: float = 1e-13, max_iter: int = 200
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-Euler Norton update of points with strains ``eps`` (P, 3, 3), committed creep strains ``ecr_n``
    (P, 3, 3) and moduli ``K``, ``G`` (P,): (σ, ε_cr,n+1, Δp), differentiable in all tensor inputs (module notes)."""
    tr = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
    dev = eps - (tr / 3.0)[:, None, None] * _EYE
    s_tr = 2.0 * G[:, None, None] * (dev - ecr_n)
    q2 = 1.5 * (s_tr * s_tr).sum((-2, -1))
    q_np = np.sqrt(np.maximum(q2.detach().numpy(), 0.0))
    G_np = G.detach().numpy()
    active = (q_np > 0.0) & (dt > 0.0)
    dp = torch.zeros_like(tr)
    if active.any():
        idx = np.flatnonzero(active)
        coef = dt * A

        def rate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = q / sigma_ref
            return coef * u ** n, coef * n * u ** (n - 1.0) / sigma_ref
        x, _r, _it = _solve_increment(q_np[idx], G_np[idx], rate, tol, max_iter)   # type: ignore[arg-type]
        it = torch.as_tensor(idx, dtype=torch.long)
        xs = torch.as_tensor(x, dtype=eps.dtype)
        qt = torch.sqrt(q2[it])
        Ga = G[it]
        qe = qt - 3.0 * Ga * xs
        u = qe / sigma_ref
        h = coef * u ** n
        hp = coef * n * u ** (n - 1.0) / sigma_ref
        x_corr = xs - (xs - h) / (1.0 + 3.0 * Ga * hp)                 # one Newton step: exact derivative of the root
        dp = dp.index_put((it,), x_corr)
        theta = torch.ones_like(tr).index_put((it,), (qt - 3.0 * Ga * x_corr) / qt)
        Nn = torch.zeros_like(eps).index_put((it,), 1.5 * s_tr[it] / qt[:, None, None])
    else:
        theta = torch.ones_like(tr)
        Nn = torch.zeros_like(eps)
    sig = theta[:, None, None] * s_tr + (K * tr)[:, None, None] * _EYE
    ecr = ecr_n + dp[:, None, None] * Nn
    return sig, ecr, dp


def _tangent(eps: np.ndarray, ecr_n: np.ndarray, K: np.ndarray, G: np.ndarray, dt: float, A: float, sref: float,
             n: float) -> np.ndarray:
    """Algorithmic tangent (P, 6, 6) of the update (the formula of ``NortonCreep.update`` with per-point K, G)."""
    P = len(eps)
    tr = np.trace(eps, axis1=1, axis2=2)
    s_tr = 2.0 * G[:, None, None] * (eps - (tr / 3.0)[:, None, None] * np.eye(3) - ecr_n)
    q_tr = math.sqrt(1.5) * np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
    C = K[:, None, None] * np.outer(_ONE6, _ONE6)[None] + 2.0 * G[:, None, None] * _IDEV6[None]
    act = (q_tr > 0.0) & (dt > 0.0)
    if act.any():
        i = np.flatnonzero(act)
        coef = dt * A

        def rate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            u = q / sref
            return coef * u ** n, coef * n * u ** (n - 1.0) / sref
        x, _r, _it = _solve_increment(q_tr[i], G[i], rate, 1e-13, 200)               # type: ignore[arg-type]
        qn = np.maximum(q_tr[i] - 3.0 * G[i] * x, 0.0)
        with np.errstate(all="ignore"):
            _, hp = rate(qn)
            dpp = 1.0 / (3.0 * G[i] + 1.0 / hp)
        theta = qn / q_tr[i]
        Nv = (1.5 * s_tr[i] / q_tr[i][:, None, None])[:, _VA, _VB]
        cf = -4.0 * G[i] ** 2 * (dpp - x / q_tr[i])
        C[i] = (K[i][:, None, None] * np.outer(_ONE6, _ONE6)[None] + 2.0 * (G[i] * theta)[:, None, None] * _IDEV6[None]
                + cf[:, None, None] * Nv[:, :, None] * Nv[:, None, :])
    del P
    return C


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, ecr_prev: torch.Tensor, u0: torch.Tensor, owner: CreepStructure,
                k: int) -> torch.Tensor:
        u, fact = owner._newton(w.detach(), ecr_prev.detach(), u0.detach().numpy().copy(), k)
        ctx.save_for_backward(w.detach(), ecr_prev.detach())
        ctx.owner, ctx.fact, ctx.u, ctx.k = owner, fact, u, k
        return torch.as_tensor(u)

    @staticmethod
    def backward(ctx: Any, ubar: torch.Tensor) -> tuple[Any, Any, None, None, None]:
        own: CreepStructure = ctx.owner
        w, ecr_prev = ctx.saved_tensors
        lam = np.zeros(own.ndof)
        lam[own.free] = ctx.fact.solve_transpose(ubar.detach().numpy()[own.free])
        with torch.enable_grad():
            wr = w.clone().requires_grad_(True)
            er = ecr_prev.clone().requires_grad_(True)
            fint = own._internal(torch.as_tensor(ctx.u), wr, er, own.dts[ctx.k])
            gw, ge = torch.autograd.grad(fint, (wr, er), grad_outputs=-torch.as_tensor(lam), allow_unused=True)
        return gw, ge, None, None, None


@dataclass
class CreepResult:
    u: torch.Tensor                  # (n_steps, ndof)
    compliance: torch.Tensor         # f_nᵀ u_n (n_steps,)
    times: np.ndarray


class CreepStructure:
    """Plane-strain Norton-creep structure on ``mesh`` (module notes): elastic K0, G0 [Pa] scaled per element by
    w_e; creep A [1/s], σ_ref [Pa], n physical. ``times`` (n,) step end times from 0 (non-decreasing; a step of zero
    length is elastic), ``loads`` (n, ndof) nodal forces at those times, ``fixed`` held DOFs."""

    def __init__(self, mesh: Mesh, K0: float, G0: float, A: float, sigma_ref: float, n: float,
                 times: Sequence[float], loads: np.ndarray, fixed: np.ndarray, thickness: float = 1.0,
                 tol: float = 1e-10, max_iter: int = 30) -> None:
        if mesh.dim != 2:
            raise ValueError("CreepStructure: 2-D plane strain meshes only")
        for nm, v in (("K0", K0), ("G0", G0), ("sigma_ref", sigma_ref), ("n", n)):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{nm} must be finite and > 0, got {v!r}")
        if not (math.isfinite(A) and A >= 0):
            raise ValueError(f"A must be finite and >= 0, got {A!r}")
        self.mesh, self.K0, self.G0, self.A, self.sref, self.n = mesh, float(K0), float(G0), float(A), \
            float(sigma_ref), float(n)
        self.times = np.asarray(times, float).reshape(-1)
        if not (self.times[0] >= 0 and np.all(np.diff(self.times) >= 0) and np.all(np.isfinite(self.times))):
            raise ValueError("times must be finite, >= 0 and non-decreasing")
        self.dts = np.diff(np.concatenate([[0.0], self.times]))
        self.ndof = mesh.n_nodes * 2
        self.loads = np.asarray(loads, float).reshape(len(self.times), self.ndof)
        g = kn.geometry(mesh.ref, mesh.cell_points(), rule(mesh.ref.cell, kn.stiffness_degree(mesh.ref)))
        self.B = kn.strain_matrix(g.dNdx)
        self.dV = g.dV * float(thickness)
        self.ne, self.nq = self.dV.shape
        self.cell_dofs = mesh.cell_dofs(2)
        self.pattern = Pattern(self.cell_dofs, (self.ndof, self.ndof))
        self.free = np.setdiff1d(np.unique(self.cell_dofs), np.unique(np.asarray(fixed, np.int64)))
        self.tol, self.max_iter = float(tol), int(max_iter)
        self._Bt, self._dVt = torch.as_tensor(self.B), torch.as_tensor(self.dV)
        self._cd = torch.as_tensor(self.cell_dofs, dtype=torch.long)

    # ---------------------------------------------------------------- kinematics and internal force
    def _eps(self, u: torch.Tensor) -> torch.Tensor:
        e = torch.einsum("eqkj,ej->eqk", self._Bt, u[self._cd]).reshape(-1, 3)
        z = torch.zeros(len(e), dtype=e.dtype)
        return torch.stack([torch.stack([e[:, 0], 0.5 * e[:, 2], z], -1),
                            torch.stack([0.5 * e[:, 2], e[:, 1], z], -1), torch.stack([z, z, z], -1)], -2)

    def _moduli(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wp = w.repeat_interleave(self.nq)
        return self.K0 * wp, self.G0 * wp

    def _internal(self, u: torch.Tensor, w: torch.Tensor, ecr_prev: torch.Tensor, dt: float) -> torch.Tensor:
        K, G = self._moduli(w)
        sig, _e, _dp = norton_update_torch(self._eps(u), ecr_prev, K, G, dt, self.A, self.sref, self.n)
        sv = torch.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1]], -1).reshape(self.ne, self.nq, 3)
        fe = torch.einsum("eqkj,eqk,eq->ej", self._Bt, sv, self._dVt)
        return torch.zeros(self.ndof, dtype=u.dtype).index_add(0, self._cd.reshape(-1), fe.reshape(-1))

    def _newton(self, w: torch.Tensor, ecr_prev: torch.Tensor, u: np.ndarray, k: int) -> tuple[np.ndarray, Any]:
        dt, f = float(self.dts[k]), self.loads[k]
        fr = self.free
        wn = w.numpy()
        Kp, Gp = self.K0 * np.repeat(wn, self.nq), self.G0 * np.repeat(wn, self.nq)
        scale = max(float(np.abs(f).max()), 1e-300)
        fact = None
        for it in range(self.max_iter + 1):
            with torch.no_grad():
                ut = torch.as_tensor(u)
                r = (self._internal(ut, w, ecr_prev, dt).numpy() - f)
                eps = self._eps(ut).numpy()
            C = _tangent(eps, ecr_prev.numpy(), Kp, Gp, dt, self.A, self.sref, self.n)[:, _IP][:, :, _IP]
            Ke = np.einsum("eqki,eqkl,eqlj,eq->eij", self.B, C.reshape(self.ne, self.nq, 3, 3), self.B, self.dV)
            Kg = self.pattern.assemble(Ke).tocsr()
            fact = Factorization(Kg[fr][:, fr], "general")
            if float(np.abs(r[fr]).max()) <= self.tol * scale:
                return u, fact
            if it == self.max_iter:
                raise ArithmeticError(f"creep step {k}: Newton not converged (residual {np.abs(r[fr]).max():.3e})")
            u = u.copy()
            u[fr] -= fact.solve(r[fr])
        raise AssertionError("unreachable")

    # ---------------------------------------------------------------- march
    def response(self, w: torch.Tensor, detach_history: bool = False) -> CreepResult:
        """Displacements of every step for element weights ``w`` (ne,), differentiable; ``detach_history`` cuts the
        derivative through the committed creep strain (negative control CO4 only)."""
        ecr = torch.zeros(self.ne * self.nq, 3, 3, dtype=w.dtype)
        u = torch.zeros(self.ndof, dtype=w.dtype)
        us, cs = [], []
        for k in range(len(self.times)):
            e_prev = ecr.detach() if detach_history else ecr
            u = _Step.apply(w, e_prev, u, self, k)
            K, G = self._moduli(w)
            _s, ecr, _dp = norton_update_torch(self._eps(u), e_prev, K, G, float(self.dts[k]), self.A, self.sref,
                                               self.n)
            us.append(u)
            cs.append(torch.as_tensor(self.loads[k]) @ u)
        return CreepResult(torch.stack(us), torch.stack(cs), self.times.copy())
