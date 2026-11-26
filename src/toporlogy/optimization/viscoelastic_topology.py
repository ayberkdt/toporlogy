"""Topology optimisation of a linear viscoelastic structure with the exact discrete history adjoint
(``docs/foundations/prereg/VO_viscoelastic_topology.md``, VO1-VO6).

Material: one :class:`~toporlogy.core.viscoelastic.GeneralizedMaxwell` solid (independent shear and bulk Prony
series, physical units). Interpolation: every modulus of both spectra in element e is multiplied by
w_e = E_min + x̄_e^p (1 − E_min); the relaxation times are not changed (a solid/void mixture of one viscoelastic
phase scales stiffness, not time). The branch stresses are then s_i = w_e ŝ_i with ŝ_i following the unscaled
recursion of the element's strain history, and one step n → n+1 of the plane-strain problem is linear:

    K_alg(w) Δu = f_{n+1} − f_hist,n(w),   K_alg(w) = Σ_e w_e K_e^alg,
    f_hist,n = Σ_e w_e ∫ Bᵀ [2G∞ dev ε_n + K∞ tr ε_n 1 + Σ r_i ŝ_i,n + Σ r_j p̂_j,n 1] dV,

with r = e^{−Δt/τ}, φ = (1 − r)/(Δt/τ) (φ = 1 at Δt = 0, an instantaneous step) and
C_alg = 2(G∞ + Σ G_i φ_i) P_dev + 3(K∞ + Σ K_j φ_j) P_vol — the recursion of ``GeneralizedMaxwell.update`` (exact for
strain linear in each step). The time march is written in torch and differentiated in reverse mode; the linear
solve is a custom autograd function (K_alg symmetric: the adjoint solve is the same factorisation), so the gradient
is the exact derivative of the discrete objective on the fixed time grid, history terms included.

Isothermal, small strain, plane strain on a 2-D :class:`~toporlogy.fem.mesh.Mesh`. Not here: temperature shift,
3-D, nonlinear materials, adaptive time steps inside the optimisation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from toporlogy.core.viscoelastic import GeneralizedMaxwell, _phi
from toporlogy.fem.mesh import kernels as kn
from toporlogy.fem.mesh.assembly import Pattern
from toporlogy.fem.mesh.linear import Factorization
from toporlogy.fem.mesh.mesh import Mesh
from toporlogy.fem.mesh.quadrature import rule

__all__ = ["ViscoelasticStructure", "ViscoelasticResult", "optimize_viscoelastic"]

_IP = [0, 1, 5]                                         # plane strain [xx, yy, xy] inside the 3-D Voigt order
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(_ONE6, _ONE6) / 3.0


class _Solve(torch.autograd.Function):
    """x = K(w)⁻¹ b on the free DOFs, K(w) = Σ w_e Ke (symmetric); backward: K λ = ḡ, w̄_e = −λ_eᵀ Ke x_e, b̄ = λ."""

    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, b: torch.Tensor, owner: ViscoelasticStructure, key: float) -> torch.Tensor:
        fact = owner._factor(w.detach().cpu().numpy(), key)
        x = fact.solve(b.detach().cpu().numpy())
        ctx.fact, ctx.owner, ctx.key, ctx.x = fact, owner, key, x
        return torch.as_tensor(x, dtype=b.dtype)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, Any, None, None]:
        lam = ctx.fact.solve(g.detach().cpu().numpy())
        own = ctx.owner
        L = np.zeros(own.ndof)
        X = np.zeros(own.ndof)
        L[own.free], X[own.free] = lam, ctx.x
        Le, Xe = L[own.cell_dofs], X[own.cell_dofs]
        dw = -np.einsum("ea,eab,eb->e", Le, own._Ke[ctx.key], Xe)
        return torch.as_tensor(dw, dtype=g.dtype), torch.as_tensor(lam, dtype=g.dtype), None, None


@dataclass
class ViscoelasticResult:
    u: torch.Tensor                  # displacements (n_steps, ndof)
    compliance: torch.Tensor         # f_nᵀ u_n per step (n_steps,)
    times: np.ndarray


class ViscoelasticStructure:
    """Plane-strain linear viscoelastic structure on ``mesh`` with the element-scaled material of the module notes.

    ``times`` (n,): end times of the steps (non-decreasing, from t = 0, a step of zero length is an instantaneous
    step); ``loads`` (n, ndof): the external nodal forces at those times (piecewise linear in between); ``fixed``:
    the DOFs held at zero."""

    def __init__(self, mesh: Mesh, material: GeneralizedMaxwell, times: Sequence[float], loads: np.ndarray,
                 fixed: np.ndarray, thickness: float = 1.0) -> None:
        if mesh.dim != 2:
            raise ValueError("ViscoelasticStructure: 2-D plane strain meshes only")
        if not isinstance(material, GeneralizedMaxwell) or material.shift is not None:
            raise ValueError("ViscoelasticStructure: an isothermal GeneralizedMaxwell material is required")
        self.mesh, self.mat = mesh, material
        self.times = np.asarray(times, float).reshape(-1)
        if not (np.all(np.isfinite(self.times)) and self.times[0] >= 0.0 and np.all(np.diff(self.times) >= 0.0)):
            raise ValueError("times must be finite, >= 0 and non-decreasing")
        self.dts = np.diff(np.concatenate([[0.0], self.times]))
        self.ndof = mesh.n_nodes * 2
        self.loads = np.asarray(loads, float).reshape(len(self.times), self.ndof)
        g = kn.geometry(mesh.ref, mesh.cell_points(), rule(mesh.ref.cell, kn.stiffness_degree(mesh.ref)))
        self.B = kn.strain_matrix(g.dNdx)                                    # (ne, nq, 3, nloc)
        self.dV = g.dV * float(thickness)
        self.ne, self.nq = self.dV.shape
        self.cell_dofs = mesh.cell_dofs(2)
        self.pattern = Pattern(self.cell_dofs, (self.ndof, self.ndof))
        used = np.unique(self.cell_dofs)
        self.free = np.setdiff1d(used, np.unique(np.asarray(fixed, np.int64)))
        sh, bk = material.shear, material.bulk
        self._Ke: dict[float, np.ndarray] = {}
        self._coef: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for dt in np.unique(self.dts):
            a_s, a_k = dt / sh.taus, dt / bk.taus
            r_s, r_k, ph_s, ph_k = np.exp(-a_s), np.exp(-a_k), _phi(a_s), _phi(a_k)
            Gt, Kt = sh.inf + float(ph_s @ sh.amplitudes), bk.inf + float(ph_k @ bk.amplitudes)
            C6 = Kt * np.outer(_ONE6, _ONE6) + 2.0 * Gt * _IDEV6
            C = C6[np.ix_(_IP, _IP)]
            self._Ke[float(dt)] = np.einsum("eqki,kl,eqlj,eq->eij", self.B, C, self.B, self.dV)
            self._coef[float(dt)] = (r_s, r_k, 2.0 * sh.amplitudes * ph_s, bk.amplitudes * ph_k)
        self._fcache: dict[tuple[float, bytes], Factorization] = {}
        self._Bt = torch.as_tensor(self.B)
        self._dVt = torch.as_tensor(self.dV)
        self._cd = torch.as_tensor(self.cell_dofs, dtype=torch.long)
        self._free_t = torch.as_tensor(self.free, dtype=torch.long)

    def _factor(self, w: np.ndarray, key: float) -> Factorization:
        k = (key, w.tobytes())
        f = self._fcache.get(k)
        if f is None:
            if len(self._fcache) > 8:
                self._fcache.clear()
            K = self.pattern.assemble(w[:, None, None] * self._Ke[key]).tocsr()
            f = Factorization(K[self.free][:, self.free], "spd")
            self._fcache[k] = f
        return f

    def _strain(self, u: torch.Tensor) -> torch.Tensor:
        """Plane-strain Voigt [xx, yy, γxy] at the points (ne, nq, 3)."""
        return torch.einsum("eqkj,ej->eqk", self._Bt, u[self._cd])

    def response(self, w: torch.Tensor, detach_history: bool = False) -> ViscoelasticResult:
        """Displacements of every step for element weights ``w`` (ne,) (differentiable); ``detach_history`` cuts the
        derivative through the stored history (the negative control VO5 only)."""
        sh, bk = self.mat.shear, self.mat.bulk
        dt_ = w.dtype
        ns, nk = len(sh.taus), len(bk.taus)
        eps = torch.zeros(self.ne, self.nq, 3, dtype=dt_)          # [xx, yy, γxy]; ε_zz = 0 (plane strain)
        shat = torch.zeros(self.ne, self.nq, ns, 4, dtype=dt_)      # deviatoric branches [xx, yy, zz, xy] (tensor)
        phat = torch.zeros(self.ne, self.nq, nk, dtype=dt_)
        u = torch.zeros(self.ndof, dtype=dt_)
        us, cs = [], []
        for n, dt in enumerate(self.dts):
            r_s, r_k, c_s, c_k = (torch.as_tensor(v, dtype=dt_) for v in self._coef[float(dt)])
            e_h = eps.detach() if detach_history else eps
            s_h = shat.detach() if detach_history else shat
            p_h = phat.detach() if detach_history else phat
            tr = e_h[..., 0] + e_h[..., 1]
            dev = torch.stack([e_h[..., 0] - tr / 3, e_h[..., 1] - tr / 3, -tr / 3, 0.5 * e_h[..., 2]], -1)
            S = (2.0 * sh.inf) * dev + torch.einsum("b,eqbc->eqc", r_s, s_h)          # deviatoric stress (tensor)
            P = bk.inf * tr + torch.einsum("b,eqb->eq", r_k, p_h)
            sig = torch.stack([S[..., 0] + P, S[..., 1] + P, S[..., 3]], -1)           # [xx, yy, xy]
            fe = torch.einsum("eqkj,eqk,eq->ej", self._Bt, sig, self._dVt) * w[:, None]
            fh = torch.zeros(self.ndof, dtype=dt_).index_add(0, self._cd.reshape(-1), fe.reshape(-1))
            rhs = torch.as_tensor(self.loads[n], dtype=dt_)[self._free_t] - fh[self._free_t]
            du_f = _Solve.apply(w, rhs, self, float(dt))
            du = torch.zeros(self.ndof, dtype=dt_).index_put((self._free_t,), du_f)
            u = u + du
            de = self._strain(du)
            dtr = de[..., 0] + de[..., 1]
            ddev = torch.stack([de[..., 0] - dtr / 3, de[..., 1] - dtr / 3, -dtr / 3, 0.5 * de[..., 2]], -1)
            shat = r_s[None, None, :, None] * shat + c_s[None, None, :, None] * ddev[:, :, None, :]
            phat = r_k[None, None, :] * phat + c_k[None, None, :] * dtr[..., None]
            eps = eps + de
            us.append(u)
            cs.append(torch.as_tensor(self.loads[n], dtype=dt_) @ u)
        return ViscoelasticResult(torch.stack(us), torch.stack(cs), self.times.copy())


@dataclass
class OptimizeResult:
    x: np.ndarray
    xphys: np.ndarray
    objective: float
    volume: float
    history: dict[str, list[float]] = field(default_factory=dict)


def optimize_viscoelastic(structure: ViscoelasticStructure, volume: float, rmin: float, step_index: int = -1,
                          penal: float = 3.0, emin: float = 1e-9, max_iter: int = 40, move: float = 0.2,
                          x0: np.ndarray | None = None) -> OptimizeResult:
    """Minimise the compliance f_nᵀ u_n at step ``step_index`` (−1: the last, the service time) subject to the cell
    volume fraction ≤ ``volume``; centroid density filter of radius ``rmin``; SIMP weights; MMA; fixed iterations."""
    from toporlogy.optimization.mesh_topology import MeshDensityFilter
    from toporlogy.optimizers.mma import MMA
    filt = MeshDensityFilter(structure.mesh, rmin)
    vol = structure.dV.sum(axis=1)
    vw = vol / vol.sum()
    x = np.full(structure.ne, float(volume)) if x0 is None else np.asarray(x0, float).copy()
    opt = MMA(move=move)
    hist: dict[str, list[float]] = {"f0": [], "volume": [], "change": []}
    c0 = None
    for _ in range(int(max_iter)):
        xp = filt(x)
        wt = torch.as_tensor(emin + xp ** penal * (1.0 - emin), dtype=torch.float64).requires_grad_(True)
        J = structure.response(wt).compliance[step_index]
        J.backward()
        assert wt.grad is not None
        gw = wt.grad.numpy()
        dc = filt.backward(gw * penal * xp ** (penal - 1.0) * (1.0 - emin))
        c = float(J)
        c0 = c if c0 is None else c0
        v = float(vw @ xp)
        hist["f0"].append(c)
        hist["volume"].append(v)
        xn = opt.step(x, c / c0, dc / c0, np.array([v / volume - 1.0]), (filt.backward(vw) / volume)[None, :])
        hist["change"].append(float(np.abs(xn - x).max()))
        x = xn
    xp = filt(x)
    wt = torch.as_tensor(emin + xp ** penal * (1.0 - emin), dtype=torch.float64)
    with torch.no_grad():
        c = float(structure.response(wt).compliance[step_index])
    return OptimizeResult(x, xp, c, float(vw @ xp), hist)
