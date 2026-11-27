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

__all__ = ["ViscoelasticStructure", "ViscoelasticResult", "RobustViscoelastic", "optimize_viscoelastic", "multistart",
           "perturbed_starts", "MultiStartResult"]

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
    the DOFs held at zero. A material with a time–temperature shift needs ``temperatures`` (n,) [K] at the step ends
    and the initial ``temperature0``; each step then uses the reduced increment Δξ of T linear in the step
    (``prereg/VT_viscoelastic_temperature.md``)."""

    def __init__(self, mesh: Mesh, material: GeneralizedMaxwell, times: Sequence[float], loads: np.ndarray,
                 fixed: np.ndarray, thickness: float = 1.0, temperatures: Sequence[float] | None = None,
                 temperature0: float | None = None) -> None:
        if mesh.dim != 2:
            raise ValueError("ViscoelasticStructure: 2-D plane strain meshes only")
        if not isinstance(material, GeneralizedMaxwell):
            raise ValueError("ViscoelasticStructure: a GeneralizedMaxwell material is required")
        self.mesh, self.mat = mesh, material
        self.times = np.asarray(times, float).reshape(-1)
        if not (np.all(np.isfinite(self.times)) and self.times[0] >= 0.0 and np.all(np.diff(self.times) >= 0.0)):
            raise ValueError("times must be finite, >= 0 and non-decreasing")
        self.dts = np.diff(np.concatenate([[0.0], self.times]))
        if material.shift is None:
            if temperatures is not None:
                raise ValueError("temperatures given for a material without a time-temperature shift")
            self.dxis = self.dts.copy()
        else:
            if temperatures is None or temperature0 is None:
                raise ValueError("a shifted material needs temperatures (one per step, kelvin) and temperature0")
            Tn = np.asarray(temperatures, float).reshape(-1)
            if Tn.shape != self.times.shape or not np.all(np.isfinite(Tn)) or np.any(Tn <= 0.0):
                raise ValueError(f"temperatures must be {len(self.times)} finite values > 0 K")
            Tp = np.concatenate([[float(temperature0)], Tn[:-1]])
            self.dxis = np.array([float(material.shift.reduced_time_increment(a, b, d))
                                  for a, b, d in zip(Tp, Tn, self.dts, strict=True)])
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
        for dt in np.unique(self.dxis):                     # reduced increments (= Δt when isothermal)
            a_s, a_k = dt / sh.taus, dt / bk.taus
            r_s, r_k, ph_s, ph_k = np.exp(-a_s), np.exp(-a_k), _phi(a_s), _phi(a_k)
            Gt, Kt = sh.inf + float(ph_s @ sh.amplitudes), bk.inf + float(ph_k @ bk.amplitudes)
            C6 = Kt * np.outer(_ONE6, _ONE6) + 2.0 * Gt * _IDEV6
            C = C6[np.ix_(_IP, _IP)]
            self._Ke[float(dt)] = np.einsum("eqki,kl,eqlj,eq->eij", self.B, C, self.B, self.dV)
            self._coef[float(dt)] = (r_s, r_k, 2.0 * sh.amplitudes * ph_s, bk.amplitudes * ph_k)
        self._fcache: dict[tuple[float, bytes], Factorization] = {}
        self.n_solves = 0
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

    def _step(self, n: int, w: torch.Tensor, eps: torch.Tensor, shat: torch.Tensor, phat: torch.Tensor,
              u: torch.Tensor, detach_history: bool) -> tuple[torch.Tensor, ...]:
        """One step n → n+1 of the march: (ε, ŝ, p̂, u, f_nᵀu_n) after the step."""
        sh, bk = self.mat.shear, self.mat.bulk
        dt_ = w.dtype
        dt = self.dxis[n]
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
        self.n_solves += 1
        du = torch.zeros(self.ndof, dtype=dt_).index_put((self._free_t,), du_f)
        u = u + du
        de = self._strain(du)
        dtr = de[..., 0] + de[..., 1]
        ddev = torch.stack([de[..., 0] - dtr / 3, de[..., 1] - dtr / 3, -dtr / 3, 0.5 * de[..., 2]], -1)
        shat = r_s[None, None, :, None] * shat + c_s[None, None, :, None] * ddev[:, :, None, :]
        phat = r_k[None, None, :] * phat + c_k[None, None, :] * dtr[..., None]
        eps = eps + de
        return eps, shat, phat, u, torch.as_tensor(self.loads[n], dtype=dt_) @ u

    def response(self, w: torch.Tensor, detach_history: bool = False,
                 checkpoint_every: int | None = None) -> ViscoelasticResult:
        """Displacements of every step for element weights ``w`` (ne,) (differentiable); ``detach_history`` cuts the
        derivative through the stored history (the negative control VO5 only). ``checkpoint_every`` = k splits the
        march into segments of k steps recomputed in the backward pass (``torch.utils.checkpoint``; only the segment
        boundary states are kept; ``prereg/D3_checkpoint.md``)."""
        sh, bk = self.mat.shear, self.mat.bulk
        dt_ = w.dtype
        ns, nk = len(sh.taus), len(bk.taus)
        eps = torch.zeros(self.ne, self.nq, 3, dtype=dt_)          # [xx, yy, γxy]; ε_zz = 0 (plane strain)
        shat = torch.zeros(self.ne, self.nq, ns, 4, dtype=dt_)      # deviatoric branches [xx, yy, zz, xy] (tensor)
        phat = torch.zeros(self.ne, self.nq, nk, dtype=dt_)
        u = torch.zeros(self.ndof, dtype=dt_)
        us, cs = [], []
        nsteps = len(self.dts)
        if checkpoint_every is None:
            for n in range(nsteps):
                eps, shat, phat, u, c = self._step(n, w, eps, shat, phat, u, detach_history)
                us.append(u)
                cs.append(c)
            return ViscoelasticResult(torch.stack(us), torch.stack(cs), self.times.copy())
        k = int(checkpoint_every)
        if k < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every!r}")
        from torch.utils.checkpoint import checkpoint

        for a in range(0, nsteps, k):
            b = min(a + k, nsteps)

            def segment(w: torch.Tensor, eps: torch.Tensor, shat: torch.Tensor, phat: torch.Tensor,
                        u: torch.Tensor, a: int = a, b: int = b) -> tuple[torch.Tensor, ...]:
                seg_u, seg_c = [], []
                for n in range(a, b):
                    eps, shat, phat, u, c = self._step(n, w, eps, shat, phat, u, detach_history)
                    seg_u.append(u)
                    seg_c.append(c)
                return eps, shat, phat, u, torch.stack(seg_u), torch.stack(seg_c)
            eps, shat, phat, u, su, sc = checkpoint(segment, w, eps, shat, phat, u, use_reentrant=False)
            us.extend(su.unbind(0))
            cs.extend(sc.unbind(0))
        return ViscoelasticResult(torch.stack(us), torch.stack(cs), self.times.copy())


class RobustViscoelastic:
    """Mean + κ · std over material samples (``docs/foundations/prereg/VU_viscoelastic_uncertain.md``).

    ``structures``: :class:`ViscoelasticStructure` objects that share the mesh, loads, supports and time grid and
    differ only in the material sample; ``weights`` (≥ 0, sum 1; default equal). :meth:`response` returns, per time
    step, J_rob = Σ_s ω_s J_s + κ sqrt(Σ_s ω_s (J_s − mean)²) as ``compliance``, so :func:`optimize_viscoelastic`
    accepts it in place of one structure. The samples are given by the caller (a deterministic set). The std term is
    not differentiable where all samples coincide (zero variance); κ > 0 there gives a non-finite gradient."""

    def __init__(self, structures: Sequence[ViscoelasticStructure], weights: Sequence[float] | None = None,
                 kappa: float = 0.0) -> None:
        st = list(structures)
        if not st:
            raise ValueError("RobustViscoelastic needs at least one structure")
        s0 = st[0]
        for s in st[1:]:
            if (s.mesh is not s0.mesh and not (np.array_equal(s.mesh.points, s0.mesh.points)
                                               and np.array_equal(s.mesh.cells, s0.mesh.cells))):
                raise ValueError("RobustViscoelastic: the samples must share the mesh")
            if not (np.array_equal(s.times, s0.times) and np.array_equal(s.loads, s0.loads)
                    and np.array_equal(s.free, s0.free)):
                raise ValueError("RobustViscoelastic: the samples must share times, loads and supports")
        w = np.full(len(st), 1.0 / len(st)) if weights is None else np.asarray(weights, float).reshape(-1)
        if w.shape != (len(st),) or not np.all(np.isfinite(w)) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-12:
            raise ValueError("weights must be one finite non-negative value per sample, summing to 1")
        if not (np.isfinite(kappa) and kappa >= 0.0):
            raise ValueError(f"kappa must be finite and >= 0, got {kappa!r}")
        self.structures, self.weights, self.kappa = st, w, float(kappa)
        self.mesh, self.dV, self.ne = s0.mesh, s0.dV, s0.ne

    def samples(self, w: torch.Tensor) -> torch.Tensor:
        """J_s at every step (n_samples, n_steps)."""
        return torch.stack([s.response(w).compliance for s in self.structures])

    def response(self, w: torch.Tensor) -> ViscoelasticResult:
        Js = self.samples(w)
        if len(self.structures) == 1 and self.kappa == 0.0:
            rob = Js[0]
        else:
            om = torch.as_tensor(self.weights, dtype=Js.dtype)[:, None]
            mean = (om * Js).sum(0)
            rob = mean + self.kappa * torch.sqrt((om * (Js - mean) ** 2).sum(0)) if self.kappa > 0 else mean
        return ViscoelasticResult(torch.empty(0), rob, self.structures[0].times.copy())


@dataclass
class OptimizeResult:
    x: np.ndarray
    xphys: np.ndarray
    objective: float                 # the objective of the returned design (max over ``steps`` for a min-max run)
    volume: float
    history: dict[str, list[float]] = field(default_factory=dict)
    objectives: np.ndarray | None = None      # J at each of the objective steps for the returned design
    converged: bool = False
    iterations: int = 0
    termination: str = "max_iter"             # max_iter | design_change | kkt
    kkt_residual: float | None = None         # diagnostics.kkt_estimate of the last analysed iterate


def optimize_viscoelastic(structure: ViscoelasticStructure | RobustViscoelastic, volume: float, rmin: float,
                          step_index: int = -1,
                          penal: float = 3.0, emin: float = 1e-9, max_iter: int = 40, move: float = 0.2,
                          x0: np.ndarray | None = None, steps: Sequence[int] | None = None,
                          tol: float = 0.0, kkt_tol: float | None = None) -> OptimizeResult:
    """Minimise the compliance f_nᵀ u_n at step ``step_index`` (−1: the last, the service time), or with ``steps``
    (several step indices) the worst of them, min max_k J_k, by the bound formulation min z s.t.
    J_k/c_ref − z ≤ 0 (z an extra variable in [0, 1], c_ref = 1.5 max_k J_k(x0); ``prereg/VR_viscoelastic_robust.md``);
    volume fraction ≤ ``volume``; centroid density filter of radius ``rmin``; SIMP weights; MMA. Stops after
    ``max_iter`` iterations, when the largest change of the variables is below ``tol``, or — with ``kkt_tol`` — when
    the first-order optimality residual of the current iterate (:func:`~toporlogy.optimization.diagnostics.
    kkt_estimate` on the normalised problem passed to MMA) is ≤ ``kkt_tol`` (``prereg/KS_kkt_stop.md``)."""
    from toporlogy.optimization.diagnostics import kkt_estimate
    from toporlogy.optimization.mesh_topology import MeshDensityFilter
    from toporlogy.optimizers.mma import MMA
    filt = MeshDensityFilter(structure.mesh, rmin)
    vol = structure.dV.sum(axis=1)
    vw = vol / vol.sum()
    ks = [int(step_index)] if steps is None else [int(k) for k in steps]
    if not ks:
        raise ValueError("steps must name at least one step")
    minmax = len(ks) > 1
    x = np.full(structure.ne, float(volume)) if x0 is None else np.asarray(x0, float).copy()
    opt = MMA(move=move)
    hist: dict[str, list[float]] = {"f0": [], "volume": [], "change": []}
    c0: float | None = None
    z = 0.0
    converged = False
    termination = "max_iter"
    kkt: float | None = None
    it = 0

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xp = filt(x)
        wt = torch.as_tensor(emin + xp ** penal * (1.0 - emin), dtype=torch.float64).requires_grad_(True)
        comp = structure.response(wt).compliance
        Js, grads = [], []
        for j, k in enumerate(ks):
            g, = torch.autograd.grad(comp[k], wt, retain_graph=j < len(ks) - 1)
            Js.append(float(comp[k]))
            grads.append(filt.backward(g.numpy() * penal * xp ** (penal - 1.0) * (1.0 - emin)))
        return np.array(Js), np.array(grads), xp

    while it < int(max_iter):
        it += 1
        Js, dJ, xp = evaluate(x)
        v = float(vw @ xp)
        gv = (filt.backward(vw) / volume)[None, :]
        if not minmax:
            c = float(Js[0])
            c0 = c if c0 is None else c0
            hist["f0"].append(c)
            hist["volume"].append(v)
            if kkt_tol is not None:
                kkt = float(kkt_estimate(x, dJ[0] / c0, np.array([v / volume - 1.0]), gv)["residual"])
                if kkt <= kkt_tol:
                    converged, termination = True, "kkt"
                    break
            xn = opt.step(x, c / c0, dJ[0] / c0, np.array([v / volume - 1.0]), gv)
            change = float(np.abs(xn - x).max())
        else:
            if c0 is None:
                c0 = 1.5 * float(Js.max())
                z = float(Js.max()) / c0
            hist["f0"].append(float(Js.max()))
            hist["volume"].append(v)
            y = np.concatenate([x, [z]])
            df0 = np.zeros(len(y))
            df0[-1] = 1.0
            g = np.concatenate([[v / volume - 1.0], Js / c0 - z])
            dg = np.zeros((1 + len(ks), len(y)))
            dg[0, :-1] = gv[0]
            dg[1:, :-1] = dJ / c0
            dg[1:, -1] = -1.0
            if kkt_tol is not None:
                kkt = float(kkt_estimate(y, df0, g, dg)["residual"])
                if kkt <= kkt_tol:
                    converged, termination = True, "kkt"
                    break
            yn = opt.step(y, z, df0, g, dg)
            change = float(np.abs(yn - y).max())
            xn, z = yn[:-1], float(yn[-1])
        hist["change"].append(change)
        x = xn
        if change < tol:
            converged, termination = True, "design_change"
            break
    xp = filt(x)
    wt = torch.as_tensor(emin + xp ** penal * (1.0 - emin), dtype=torch.float64)
    with torch.no_grad():
        comp = structure.response(wt).compliance
    Jf = np.array([float(comp[k]) for k in ks])
    res = OptimizeResult(x, xp, float(Jf.max()), float(vw @ xp), hist, Jf, converged, it, termination, kkt)
    if minmax:
        res.history["z"] = [z]
        res.history["c_ref"] = [float(c0 if c0 is not None else 1.0)]
    return res


@dataclass
class MultiStartResult:
    best: OptimizeResult
    runs: list[tuple[float, float | None, str, bool]] = field(default_factory=list)   # objective, KKT, stop, feasible
    best_index: int = -1


def perturbed_starts(n: int, volume: float, n_cells: int, amplitude: float = 0.1, seed: int = 0) -> list[np.ndarray]:
    """``n`` seeded random starts around the uniform design: x = V + a (r − mean r), r uniform in [−1, 1], clipped
    to [0, 1] and re-centred so that the mean is exactly V (iterated a few times; raises if it cannot)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n)):
        r = rng.uniform(-1.0, 1.0, n_cells)
        x = volume + amplitude * (r - r.mean())
        for _it in range(50):
            x = np.clip(x, 0.0, 1.0)
            d = volume - x.mean()
            if abs(d) <= 1e-13:
                break
            x = x + d
        if abs(x.mean() - volume) > 1e-12 or x.min() < 0.0 or x.max() > 1.0:
            raise ValueError("perturbed_starts: could not keep the volume fraction inside [0, 1]")
        out.append(x)
    return out


def multistart(structure: ViscoelasticStructure | RobustViscoelastic, starts: Sequence[np.ndarray | None],
               volume: float, rmin: float, **kw: Any) -> MultiStartResult:
    """Run :func:`optimize_viscoelastic` from every start (an x0 array or None = uniform) with the same arguments and
    return the best feasible result by its returned objective (``prereg/MS_multistart.md``); the runs' objective, KKT
    residual, termination and feasibility are listed. Deterministic for fixed inputs."""
    if not starts:
        raise ValueError("multistart needs at least one start")
    results = [optimize_viscoelastic(structure, volume, rmin, x0=None if s is None else np.asarray(s, float), **kw)
               for s in starts]
    runs = [(r.objective, r.kkt_residual, r.termination, bool(r.volume <= volume * (1.0 + 1e-6))) for r in results]
    feas = [i for i, rr in enumerate(runs) if rr[3]]
    if not feas:
        raise ValueError("multistart: no run ended feasible")
    i_best = min(feas, key=lambda i: runs[i][0])
    return MultiStartResult(results[i_best], runs, i_best)
