"""Coupled J2 creep and rate-independent J2 plasticity of one material point with ONE local residual (WP13 C04/C08,
WP14 R01; ``docs/foundations/CREEP_PLASTICITY.md``, derivations in ``docs/foundations/prereg/C8_creep_plasticity.md``).

Scope: ε = ε_e + ε_cr + ε_p (ε_cr, ε_p deviatoric), isotropic elasticity with physical K, G [Pa], both inelastic flows
along the same radial direction N = 3s/(2q), q = √(3/2 s:s):

* creep (always active for q > 0, no threshold): ṗ_cr = A f(T) (q/σ_ref)^n, ε̇_cr = ṗ_cr N (as :class:`NortonCreep`);
* plasticity (rate independent): f_y = q − κ(α) ≤ 0, κ = σ_y + Hα, α̇ ≥ 0, α̇ f_y = 0, ε̇_p = α̇ N (as
  :class:`PhysicalJ2` with linear hardening).

Units: Pa, s, K (absolute), J/mol; tension positive; Voigt order [xx, yy, zz, yz, xz, xy] (stress physical shear, strain
engineering shear). Parameters are synthetic in all tests; nothing here is material data. ``yield_stress=None`` or
``math.inf`` switches the plastic branch off exactly (no tolerance represents ∞).

Backward Euler, radial. Trial s_tr = 2G dev(ε − ε_cr,n − ε_p,n), q_tr; the end stress is s = θ s_tr with
q = q_tr − 3GΛ, Λ = Δp_cr + Δα. Unknowns x = Δp_cr, y = Δα; residuals r1 = x − h(q) (h(q) = Δt A F̄ (q/σ_ref)^n) and, in
the plastic branch, r2 = q − κ_n − H y. r2 is linear, so y = (q_tr − κ_n − 3G x)/(3G + H) is eliminated exactly (Schur
complement of the 2×2 Newton system) and x solves g(x) = x − h(q(x)) = 0 with q(x) = κ_n + w (q_tr − κ_n − 3G x),
w = H/(3G + H), g' = 1 + 3G w h' ≥ 1: the safeguarded Newton/bisection of :mod:`toporlogy.core.creep`. Active set: the
creep-only root x_c (y = 0) gives q_c = q_tr − 3G x_c; the step is plastic iff q_c > κ_n (the plastic root then has
x ≤ x_c and y > 0, see the preregistration). In the plastic branch q_{n+1} = σ_y + Hα_{n+1} holds to rounding.

Algorithmic tangent C = K 1⊗1 + 2Gθ I_dev − 4G² (Λ' − Λ/q_tr) N⊗N, Λ' = dΛ/dq_tr: creep-only x' = h'/(1 + 3G h'),
plastic x' = w h'/(1 + 3G w h'), y' = (1 − 3G x')/(3G + H), Λ' = x' + y'. Energy ψ = ½ σ:(ε − ε_cr − ε_p) + ½Hα²;
dissipation D = q (x + y) − Δψ_h = D_cr + D_p ≥ 0, D_cr = q x, D_p = q y − Δψ_h ≥ σ_y y.
Limits: A = 0 is :class:`PhysicalJ2` (linear hardening), σ_y = ∞ is :class:`NortonCreep`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse, require_step,
                                         small_strain_history_evaluate)
from toporlogy.core.creep import ArrheniusFactor, NortonCreep, _pos, _solve_increment

__all__ = ["CreepPlasticJ2"]

_DEFAULT_REQUEST = MaterialRequest()
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
_I6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = _I6 - np.outer(_ONE6, _ONE6) / 3.0


class CreepPlasticJ2:
    """J2 creep + rate-independent J2 plasticity with linear isotropic hardening, one local residual (module notes).

    ``K``, ``G`` [Pa]; creep ``A`` [1/s] ≥ 0 (rate at q = σ_ref, T = T_ref), ``sigma_ref`` [Pa] > 0, ``n`` > 0, optional
    ``arrhenius`` (:class:`ArrheniusFactor`, then the step temperatures are required; the factor is the mean over the
    step of the linearly varying T, as in :class:`NortonCreep`); plasticity ``yield_stress`` σ_y [Pa] > 0 (``None`` or
    ``math.inf``: no plasticity) and ``hardening`` H [Pa] ≥ 0. The physical time step is required in every request
    (``require_step``); Δt = 0 gives a purely plastic update.

    State: ``creep_strain`` (n, 3, 3), ``p_cr`` (n,), ``plastic_strain`` (n, 3, 3), ``alpha`` (n,). Diagnostics:
    ``delta_p_cr``, ``delta_alpha``, ``q`` (end of step), ``plastic`` (1 where the step is plastic), ``yield_residual``
    (q − κ(α_{n+1})), ``creep_residual`` (|g|/cap of the creep-only root), ``plastic_residual`` (same for the plastic
    root), ``local_iterations``, ``creep_active``, ``dissipation_creep``, ``dissipation_plastic``, ``inelastic_work``
    (q Δp_cr + q Δα = σ:Δε_in), ``plastic_work`` (σ:Δε_p). Only the algorithmic tangent exists."""

    kinematics = "small"
    conservative = False
    physical = True
    name = "creep_plastic_j2"
    state_fields: tuple[str, ...] = ("creep_strain", "p_cr", "plastic_strain", "alpha")

    def __init__(self, K: float, G: float, A: float, sigma_ref: float, n: float, yield_stress: float | None,
                 hardening: float = 0.0, arrhenius: ArrheniusFactor | None = None, local_tol: float = 1e-13,
                 local_max_iter: int = 200) -> None:
        # NortonCreep validates K, G, A, sigma_ref, n, arrhenius, tolerances and owns the creep law (one definition).
        self._creep = NortonCreep(K, G, A, sigma_ref, n, arrhenius=arrhenius, local_tol=local_tol,
                                  local_max_iter=local_max_iter)
        self.K, self.G, self.A, self.sigma_ref, self.n = self._creep.K, self._creep.G, self._creep.A, \
            self._creep.sigma_ref, self._creep.n
        self.arrhenius = arrhenius
        self.local_tol, self.local_max_iter = self._creep.local_tol, self._creep.local_max_iter
        self._needs_temperature = arrhenius is not None
        self.plastic = not (yield_stress is None or float(yield_stress) == math.inf)
        self.sy = _pos(yield_stress, "yield_stress [Pa]") if self.plastic else math.inf
        self.H = _pos(hardening, "hardening [Pa]", strict=False)
        self.Ce6 = self.K * np.outer(_ONE6, _ONE6) + 2.0 * self.G * _IDEV6

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"creep_strain": np.zeros((n, 3, 3)), "p_cr": np.zeros(n), "plastic_strain": np.zeros((n, 3, 3)),
                "alpha": np.zeros(n)}

    def _psi_h(self, alpha: np.ndarray) -> np.ndarray:
        return 0.5 * self.H * alpha ** 2

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        step = require_step(self, request, temperature=self._needs_temperature)
        if request.tangent_kind != "algorithmic":
            raise ValueError(f"{self.name}: only the algorithmic tangent is defined (tangent_kind='algorithmic')")
        eps = np.asarray(eps, float)
        n = len(eps)
        K, G, H, tol, maxit = self.K, self.G, self.H, self.local_tol, self.local_max_iter
        ecr_n, p_n = np.asarray(committed["creep_strain"]), np.asarray(committed["p_cr"])
        ep_n, a_n = np.asarray(committed["plastic_strain"]), np.asarray(committed["alpha"])
        eye = np.eye(3)
        tr = np.trace(eps, axis1=1, axis2=2)
        s_tr = 2.0 * G * (eps - (tr / 3.0)[:, None, None] * eye - ecr_n - ep_n)
        q_tr = math.sqrt(1.5) * np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
        factory = self._creep._rate_factory(committed, step, n)

        # ---- creep-only root (y = 0): the candidate of the active set
        creep = (q_tr > 0.0) & (step.dt > 0.0)
        idx = np.flatnonzero(creep)
        x = np.zeros(n)                                # Δp_cr
        y = np.zeros(n)                                # Δα
        q_new = q_tr.copy()
        theta = np.ones(n)
        res_c, res_p = np.zeros(n), np.zeros(n)
        its = np.zeros(n)
        rate_c = None
        if idx.size:
            rate_c = factory(idx)
            xc, rc, ic = _solve_increment(q_tr[idx], G, rate_c, tol, maxit)
            x[idx], res_c[idx], its[idx] = xc, rc, ic
            q_new[idx] = np.maximum(q_tr[idx] - 3.0 * G * xc, 0.0)
            theta[idx] = q_new[idx] / q_tr[idx]
        kappa_n = self.sy + H * a_n if self.plastic else np.full(n, math.inf)
        pl = (q_new > kappa_n) if self.plastic else np.zeros(n, bool)       # plastic iff q_c > κ_n (strict)
        cbr = creep & ~pl                                                     # creep-only branch with evolution

        # ---- plastic branch: same scalar solve with the effective modulus G w (q_tr -> q0)
        ip = np.flatnonzero(pl)
        lam_p = np.zeros(n)                            # Λ' of the plastic branch
        if ip.size:
            rate_p = factory(ip)
            kn, qt = kappa_n[ip], q_tr[ip]
            if H > 0.0:
                w = H / (3.0 * G + H)
                q0 = kn + w * (qt - kn)
                xp, rp, itp = _solve_increment(q0, G * w, rate_p, tol, maxit)
                res_p[ip], its[ip] = rp, itp
            else:
                w = 0.0
                xp = rate_p(kn)[0]                     # q = σ_y exactly: x = h(σ_y)
            yp = (qt - kn - 3.0 * G * xp) / (3.0 * G + H)
            x[ip], y[ip] = xp, yp
            theta[ip] = 1.0 - 3.0 * G * (xp + yp) / qt
            q_new[ip] = theta[ip] * qt
            res_c[ip] = 0.0
            with np.errstate(all="ignore"):
                _, hp = rate_p(np.maximum(q_new[ip], 0.0))
                xd = 1.0 / (3.0 * G + 1.0 / (w * hp))          # x' = w h'/(1 + 3G w h'); h' = 0 or w = 0 -> 0
            lam_p[ip] = xd + (1.0 - 3.0 * G * xd) / (3.0 * G + H)

        # ---- assemble
        N = np.zeros((n, 3, 3))
        tidx = np.flatnonzero(cbr | pl)
        N[tidx] = 1.5 * s_tr[tidx] / q_tr[tidx][:, None, None]
        lam = x + y
        dec, dep = x[:, None, None] * N, y[:, None, None] * N
        sig = theta[:, None, None] * s_tr + (K * tr)[:, None, None] * eye
        ecr, ep = ecr_n + dec, ep_n + dep
        a_new = a_n + y
        C = None
        if request.tangent:
            C = np.broadcast_to(self.Ce6, (n, 6, 6)).copy()
            if tidx.size:
                lp = lam_p[tidx].copy()
                ic_ = np.flatnonzero(cbr)
                if ic_.size:
                    with np.errstate(all="ignore"):
                        _, hp_c = rate_c(np.maximum(q_new[idx], 0.0))                    # type: ignore[misc]
                        dpp = 1.0 / (3.0 * G + 1.0 / hp_c)
                    full = np.zeros(n)
                    full[idx] = dpp
                    lp[np.searchsorted(tidx, ic_)] = full[ic_]
                nv = N[tidx][:, _VA, _VB]                                       # tensor components, not doubled
                coef = -4.0 * G ** 2 * (lp - lam[tidx] / q_tr[tidx])
                C[tidx] = (K * np.outer(_ONE6, _ONE6)[None] + 2.0 * G * theta[tidx][:, None, None] * _IDEV6[None]
                           + coef[:, None, None] * nv[:, :, None] * nv[:, None, :])
        energy = None
        if request.energy:
            energy = 0.5 * np.einsum("nij,nij->n", sig, eps - ecr - ep) + self._psi_h(a_new)
        dpsi = self._psi_h(a_new) - self._psi_h(a_n)
        diss = q_new * lam - dpsi
        diag = {"delta_p_cr": x, "delta_alpha": y, "q": q_new, "plastic": pl.astype(float),
                "yield_residual": q_new - (self.sy + H * a_new) if self.plastic else np.zeros(n),
                "creep_residual": res_c, "plastic_residual": res_p, "local_iterations": its,
                "creep_active": creep.astype(float), "dissipation_creep": q_new * x,
                "dissipation_plastic": q_new * y - dpsi, "inelastic_work": q_new * lam,
                "plastic_work": np.einsum("nij,nij->n", sig, dep)}
        trial = {"creep_strain": ecr, "p_cr": p_n + x, "plastic_strain": ep, "alpha": a_new}
        return HistoryUpdate(sig, C, trial, diag, energy, diss)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)
