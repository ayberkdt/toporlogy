"""Physical rate-independent and rate-dependent small-strain plasticity (WP14): :class:`PhysicalJ2`,
:class:`Perzyna`, :class:`DruckerPrager` and the Eyring rate law (:func:`eyring_rate`, :func:`eyring_stress`).

Units are SI and **physical**: bulk and shear modulus K, G, yield stress and hardening in Pa, time in s. Nothing is
scaled by a Young modulus (the legacy :class:`toporlogy.core.constitutive.J2Plasticity` is the unit-modulus law of the
FE physics and is not touched); the models set ``physical = True`` and the FE physics that scale every law by the
element modulus refuse them (``docs/foundations/MATERIAL_CONTRACT.md``). All constants in the tests are synthetic.

Conventions: tension-positive Cauchy stress σ, small strain, Voigt order [xx, yy, zz, yz, xz, xy] (stress physical
shear, strain engineering shear), q = √(3/2) ‖dev σ‖, pressure p = −tr σ / 3 (compression positive), equivalent
plastic strain p_eq (``alpha``) with ṗ_eq = λ̇ for an associated J2 flow. All three plasticity models implement the
history contract (``update(eps (n, 3, 3), committed, request) -> HistoryUpdate``; ``evaluate`` is the shared
small-strain helper) and return the algorithmic tangent of the update that was performed, the stored energy
ψ = ½ ε_e:σ + ψ_h and the dissipation increment D = σ_{n+1}:Δε_p − Δψ_h (the energy balance of the step).

:class:`PhysicalJ2`: f = q − κ(p_eq), κ = σ_y + H p_eq + (σ_∞ − σ_y)(1 − e^(−δ p_eq)), radial return, rate independent.
:class:`Perzyna`: ṗ_eq = (1/τ0) ⟨f/σ_ref⟩^m with f = q − κ(p_eq), κ = σ_y + H p_eq, backward Euler (needs the physical
time step, ``require_step``). :class:`DruckerPrager`: f = q − a p − b(κ), b = b0 + hκ, flow potential g = q − a_g p
(associated for a_g = a), cone and apex return. Not implemented: finite strain, softening/localisation control,
temperature (thermal coupling), tabulated rate/temperature data. The derivations are in the preregistration
``docs/foundations/prereg/R_rate.md`` and the scope in ``docs/foundations/RATE_DEPENDENT_CONSTITUTIVE.md``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse,
                                         ReturnMappingFailure, require_step, small_strain_history_evaluate)

__all__ = ["BOLTZMANN", "AVOGADRO", "GAS_CONSTANT", "PhysicalJ2", "Perzyna", "DruckerPrager", "eyring_rate",
           "eyring_stress", "eyring_log_rate", "eyring_stress_from_log_rate", "eyring_rate_molar",
           "eyring_stress_molar"]

BOLTZMANN = 1.380649e-23                  # J/K, exact (SI 2019)
AVOGADRO = 6.02214076e23                  # 1/mol, exact (SI 2019)
GAS_CONSTANT = AVOGADRO * BOLTZMANN       # J/(mol K) = 8.31446261815324

_DEFAULT_REQUEST = MaterialRequest()
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
_I6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])             # symmetric identity, Voigt (engineering-shear strain)
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = _I6 - np.outer(_ONE6, _ONE6) / 3.0
_EYE = np.eye(3)
_SQ32 = math.sqrt(1.5)


def _finite_pos(name: str, v: float, strict: bool = True) -> float:
    x = float(v)
    if not (math.isfinite(x) and (x > 0.0 if strict else x >= 0.0)):
        raise ValueError(f"{name} must be finite and {'> 0' if strict else '>= 0'}, got {v!r}")
    return x


def _elastic_trial(K: float, G: float, eps: np.ndarray, ep_n: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trial elastic strain ε − ε_p^n, tr, its deviator, s_tr = 2G dev, q_tr."""
    ee = eps - ep_n
    tr = np.trace(ee, axis1=1, axis2=2)
    dev = ee - (tr / 3.0)[:, None, None] * _EYE
    s_tr = 2.0 * G * dev
    q_tr = _SQ32 * np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
    return ee, tr, dev, s_tr, q_tr


class _SmallStrainBase:
    kinematics = "small"
    conservative = False
    physical = True
    name = "plastic"
    state_fields: tuple[str, ...] = ("plastic_strain", "alpha")

    def __init__(self, K: float, G: float) -> None:
        self.K = _finite_pos("K", K)
        self.G = _finite_pos("G", G)
        self.Ce6 = self.K * np.outer(_ONE6, _ONE6) + 2.0 * self.G * _IDEV6

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"plastic_strain": np.zeros((n, 3, 3)), self.state_fields[1]: np.zeros(n)}

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)

    def _radial_tangent(self, nhat: np.ndarray, theta: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """K 1⊗1 + 2Gθ I_dev − coef n̂⊗n̂ (n, 6, 6) at the points with unit deviatoric direction ``nhat`` (n, 3, 3)."""
        nv = nhat[:, _VA, _VB]
        return (self.K * np.outer(_ONE6, _ONE6)[None] + 2.0 * self.G * theta[:, None, None] * _IDEV6[None]
                - coef[:, None, None] * nv[:, :, None] * nv[:, None, :])


# ===================================================================================================== PhysicalJ2
class PhysicalJ2(_SmallStrainBase):
    """Rate-independent von Mises plasticity with isotropic hardening in physical units (no E scaling).

    Parameters: bulk and shear modulus ``K``, ``G`` [Pa], ``yield_stress`` σ_y > 0 [Pa], linear ``hardening`` H ≥ 0
    [Pa], optional Voce saturation ``saturation_stress`` σ_∞ ≥ σ_y [Pa] with ``saturation_rate`` δ > 0 (dimensionless:
    per unit equivalent plastic strain). Yield function f = q − κ(p), κ(p) = σ_y + H p + (σ_∞ − σ_y)(1 − e^(−δp)).
    Backward-Euler radial return: with trial s* = 2G dev(ε − ε_p^n), q* = √(3/2)‖s*‖, f* = q* − κ(p_n) > 0 the plastic
    increment Δp solves q* − 3GΔp − κ(p_n + Δp) = 0 (closed form for linear hardening, Newton started left of the root
    of a convex decreasing function otherwise); s = θ s*, θ = 1 − 3GΔp/q*, ε_p += √(3/2) Δp n̂. Algorithmic tangent
    C = K 1⊗1 + 2Gθ I_dev − 2G θ̄ n̂⊗n̂, θ̄ = 3G/(3G + κ'(p_{n+1})) − (1 − θ). State: ``plastic_strain`` (n, 3, 3),
    ``alpha`` = p (n,). Stored energy ½ ε_e:σ + ψ_h(p), ψ_h = ½Hp² + (σ_∞ − σ_y)(p − (1 − e^(−δp))/δ); dissipation
    D = q Δp − Δψ_h ≥ σ_y Δp ≥ 0. A time step in the request is ignored (rate independent)."""

    name = "physical_j2"

    def __init__(self, K: float, G: float, yield_stress: float, hardening: float = 0.0,
                 saturation_stress: float | None = None, saturation_rate: float = 0.0,
                 local_tol: float = 1e-13, local_max_iter: int = 50) -> None:
        super().__init__(K, G)
        self.sy = _finite_pos("yield_stress", yield_stress)
        self.H = _finite_pos("hardening", hardening, strict=False)
        self.s_inf = None if saturation_stress is None else _finite_pos("saturation_stress", saturation_stress)
        self.delta = float(saturation_rate)
        if self.s_inf is not None and not (self.s_inf >= self.sy and math.isfinite(self.delta) and self.delta > 0):
            raise ValueError("PhysicalJ2: saturation_stress >= yield_stress and saturation_rate > 0 are required")
        if self.s_inf is None and self.delta != 0.0:
            raise ValueError("PhysicalJ2: saturation_rate given without saturation_stress")
        self.local_tol, self.local_max_iter = float(local_tol), int(local_max_iter)

    def kappa(self, p: np.ndarray) -> np.ndarray:
        k = self.sy + self.H * p
        if self.s_inf is not None:
            k = k + (self.s_inf - self.sy) * (-np.expm1(-self.delta * p))
        return k

    def dkappa(self, p: np.ndarray) -> np.ndarray:
        d = np.full_like(np.asarray(p, float), self.H)
        if self.s_inf is not None:
            d = d + (self.s_inf - self.sy) * self.delta * np.exp(-self.delta * p)
        return d

    def hardening_energy(self, p: np.ndarray) -> np.ndarray:
        e = 0.5 * self.H * p ** 2
        if self.s_inf is not None:
            e = e + (self.s_inf - self.sy) * (p + np.expm1(-self.delta * p) / self.delta)
        return e

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        ep_n = np.asarray(committed["plastic_strain"])
        p_n = np.asarray(committed["alpha"])
        n = len(eps)
        ee, tre, _dev, s_tr, q_tr = _elastic_trial(self.K, self.G, eps, ep_n)
        f_tr = q_tr - self.kappa(p_n)
        plastic = f_tr > 0.0
        dp = np.zeros(n)
        its = np.zeros(n)
        if plastic.any():
            qp, pp, fp = q_tr[plastic], p_n[plastic], f_tr[plastic]
            if self.s_inf is None:
                dp[plastic] = fp / (3.0 * self.G + self.H)
                its[plastic] = 1
            else:
                x = fp / (3.0 * self.G + self.dkappa(pp))            # left of the root (convex decreasing g)
                for k in range(self.local_max_iter + 1):
                    g = qp - 3.0 * self.G * x - self.kappa(pp + x)
                    if np.all(np.abs(g) <= self.local_tol * self.sy):
                        break
                    if k == self.local_max_iter:
                        raise ReturnMappingFailure(f"PhysicalJ2 return mapping not converged in "
                                                   f"{self.local_max_iter} local iterations")
                    x = x + g / (3.0 * self.G + self.dkappa(pp + x))
                    its[np.flatnonzero(plastic)] += 1
                dp[plastic] = x
        nhat = np.zeros_like(s_tr)
        with np.errstate(invalid="ignore", divide="ignore"):
            nhat[plastic] = s_tr[plastic] / np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))[plastic][:, None, None]
        theta = np.ones(n)
        theta[plastic] = 1.0 - 3.0 * self.G * dp[plastic] / q_tr[plastic]
        sig = theta[:, None, None] * s_tr + (self.K * tre)[:, None, None] * _EYE
        dep = _SQ32 * dp[:, None, None] * nhat
        ep = ep_n + dep
        p_new = p_n + dp
        q = theta * q_tr
        C = None
        if request.tangent:
            C = np.broadcast_to(self.Ce6, (n, 6, 6)).copy()
            if plastic.any():
                th = theta[plastic]
                thb = 3.0 * self.G / (3.0 * self.G + self.dkappa(p_new[plastic])) - (1.0 - th)
                C[plastic] = self._radial_tangent(nhat[plastic], th, 2.0 * self.G * thb)
        energy = None
        if request.energy:
            energy = 0.5 * np.einsum("nij,nij->n", sig, eps - ep) + self.hardening_energy(p_new)
        diss = q * dp - (self.hardening_energy(p_new) - self.hardening_energy(p_n))
        diag = {"yield_residual": np.where(plastic, q - self.kappa(p_new), np.minimum(f_tr, 0.0)),
                "plastic": plastic.astype(float), "local_iterations": its, "delta_p": dp,
                "plastic_work": np.einsum("nij,nij->n", sig, dep)}
        return HistoryUpdate(sig, C, {"plastic_strain": ep, "alpha": p_new}, diag, energy, diss)


# ======================================================================================================== Perzyna
def _safe_newton(fun: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], lo: np.ndarray, hi: np.ndarray,
                 x0: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Root of an increasing function in [lo, hi] (g(lo) <= 0 <= g(hi)): Newton steps safeguarded by the bracket."""
    lo, hi = lo.copy(), hi.copy()
    x = np.clip(x0, lo, hi)
    for _ in range(max_iter):
        g, dg = fun(x)
        hi = np.where(g > 0.0, x, hi)
        lo = np.where(g < 0.0, x, lo)
        done = (np.abs(g) <= tol) | ((hi - lo) <= 4.0 * np.finfo(float).eps * hi)
        if done.all():
            return x
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            xn = x - g / dg
        xn = np.where(np.isfinite(xn) & (xn > lo) & (xn < hi), xn, 0.5 * (lo + hi))
        x = np.where(done, x, xn)
    g, _dg = fun(x)
    if not np.all((np.abs(g) <= tol) | ((hi - lo) <= 8.0 * np.finfo(float).eps * hi)):
        raise ReturnMappingFailure(f"Perzyna scalar solve not converged in {max_iter} iterations")
    return x


def _perzyna_split(ln_beta: np.ndarray, m: float, tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, y), x + y = 1: root of x = β (1 − x)^m, x = ΞΔp/f_tr (end-of-step overstress f = f_tr y), for ln β given.
    β ≤ 1: solved for x (relative accuracy of Δp at tiny steps); β > 1: solved for y (relative accuracy of the
    overstress in the rate-independent limit). g is monotone, the bracket is analytic (see the preregistration)."""
    k = len(ln_beta)
    x = np.empty(k)
    y = np.empty(k)
    small = ln_beta <= 0.0
    if small.any():
        beta = np.exp(ln_beta[small])
        lo = beta * (1.0 - beta) ** m
        hi = beta
        # scaled G(x) = x/β − (1 − x)^m, increasing; Newton start: first-order solution
        def fun(xx: np.ndarray, b: np.ndarray = beta) -> tuple[np.ndarray, np.ndarray]:
            return xx / b - (1.0 - xx) ** m, 1.0 / b + m * (1.0 - xx) ** (m - 1.0)
        xs = _safe_newton(fun, lo, hi, beta / (1.0 + m * beta), tol, max_iter)
        x[small], y[small] = xs, 1.0 - xs
    big = ~small
    if big.any():
        lb = ln_beta[big]
        ymax = np.exp(-lb / m)
        ylo = ymax * (1.0 - ymax) ** (1.0 / m)
        ys = ymax.copy()
        todo = ymax > 1e-300
        if todo.any():
            def fun2(yy: np.ndarray, lbb: np.ndarray = lb[todo]) -> tuple[np.ndarray, np.ndarray]:
                with np.errstate(divide="ignore", under="ignore"):
                    t = np.exp(lbb + m * np.log(yy))
                return yy + t - 1.0, 1.0 + m * t / yy
            ys[todo] = _safe_newton(fun2, ylo[todo], ymax[todo], ymax[todo], tol, max_iter)
        y[big], x[big] = ys, 1.0 - ys
    return x, y


class Perzyna(_SmallStrainBase):
    """Perzyna overstress viscoplasticity with linear isotropic hardening, physical units, small strain.

    ṗ = (1/τ0) ⟨f/σ_ref⟩^m, f = q − κ(p), κ = σ_y + H p, associated J2 flow ε̇_p = ṗ (3/2) s/q. Parameters: K, G,
    ``yield_stress`` σ_y > 0 [Pa], ``hardening`` H ≥ 0 [Pa], ``tau0`` τ0 > 0 [s] (viscous time), ``sigma_ref`` > 0 [Pa]
    (stress scale of the overstress), ``m`` > 0 (rate exponent; m = 1 is linear viscosity). The strain rate itself
    never enters: the response depends on the physical time step Δt = ``request.step.dt`` (``require_step``).

    Backward Euler. Trial q*, f* = q* − κ(p_n). f* ≤ 0 or Δt = 0: elastic, no flow. Otherwise Δp solves
    Δp − (Δt/τ0) ⟨(f* − ΞΔp)/σ_ref⟩^m = 0, Ξ = 3G + H; the left side is strictly increasing in Δp and changes sign in
    (0, f*/Ξ], so a bracketed safeguarded Newton solve (scalar, vectorised, tolerance ``local_tol`` on the
    normalised residual) always converges. With x = ΞΔp/f*, y = 1 − x (the end overstress is f = f* y): x = β (1 − x)^m,
    β = (ΔtΞ/(τ0σ_ref)) (f*/σ_ref)^(m−1); for m = 1 Δp = f*/(σ_ref τ0/Δt + Ξ). Consistent tangent: dΔp/dq* = χ =
    m (1 − y)/(Ξ (y + m (1 − y))), C = K 1⊗1 + 2Gθ I_dev − 6G² (χ − Δp/q*) n̂⊗n̂, θ = 1 − 3GΔp/q*. As τ0 → 0 (y → 0)
    this is :class:`PhysicalJ2` (χ → 1/Ξ); as τ0 → ∞ (y → 1) the response is elastic (χ → 0). For m < 1 the rate law is
    not Lipschitz at f = 0 (the solve stays bracketed). State ``plastic_strain`` (n, 3, 3), ``alpha`` = p. Energy
    ½ ε_e:σ + ½Hp²; dissipation D = q Δp − ½H(p_{n+1}² − p_n²) ≥ σ_y Δp ≥ 0. Diagnostics: ``overstress``
    (q − κ(p_{n+1})), ``delta_p``, ``plastic_rate`` Δp/Δt."""

    name = "perzyna"

    def __init__(self, K: float, G: float, yield_stress: float, hardening: float, tau0: float, sigma_ref: float,
                 m: float, local_tol: float = 1e-14, local_max_iter: int = 200) -> None:
        super().__init__(K, G)
        self.sy = _finite_pos("yield_stress", yield_stress)
        self.H = _finite_pos("hardening", hardening, strict=False)
        self.tau0 = _finite_pos("tau0", tau0)
        self.sigma_ref = _finite_pos("sigma_ref", sigma_ref)
        self.m = _finite_pos("m", m)
        self.local_tol, self.local_max_iter = float(local_tol), int(local_max_iter)

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        dt = require_step(self, request).dt
        ep_n = np.asarray(committed["plastic_strain"])
        p_n = np.asarray(committed["alpha"])
        n = len(eps)
        _ee, tre, _dev, s_tr, q_tr = _elastic_trial(self.K, self.G, eps, ep_n)
        f_tr = q_tr - (self.sy + self.H * p_n)
        Xi = 3.0 * self.G + self.H
        flow = (f_tr > 0.0) & (dt > 0.0)
        dp = np.zeros(n)
        y = np.ones(n)
        chi = np.zeros(n)
        if flow.any():
            ln_beta = (math.log(dt * Xi / (self.tau0 * self.sigma_ref))
                       + (self.m - 1.0) * np.log(f_tr[flow] / self.sigma_ref))
            xs, ys = _perzyna_split(ln_beta, self.m, self.local_tol, self.local_max_iter)
            dp[flow] = f_tr[flow] / Xi * xs
            y[flow] = ys
            chi[flow] = self.m * xs / (Xi * (ys + self.m * xs))
        nhat = np.zeros_like(s_tr)
        with np.errstate(invalid="ignore", divide="ignore"):
            nhat[flow] = s_tr[flow] / np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))[flow][:, None, None]
        theta = np.ones(n)
        theta[flow] = 1.0 - 3.0 * self.G * dp[flow] / q_tr[flow]
        sig = theta[:, None, None] * s_tr + (self.K * tre)[:, None, None] * _EYE
        dep = _SQ32 * dp[:, None, None] * nhat
        ep = ep_n + dep
        p_new = p_n + dp
        q = theta * q_tr
        C = None
        if request.tangent:
            C = np.broadcast_to(self.Ce6, (n, 6, 6)).copy()
            if flow.any():
                coef = 6.0 * self.G ** 2 * (chi[flow] - dp[flow] / q_tr[flow])
                C[flow] = self._radial_tangent(nhat[flow], theta[flow], coef)
        psi_h = lambda p: 0.5 * self.H * p ** 2          # noqa: E731
        energy = None
        if request.energy:
            energy = 0.5 * np.einsum("nij,nij->n", sig, _ee - dep) + psi_h(p_new)
        diss = q * dp - (psi_h(p_new) - psi_h(p_n))
        over = q - (self.sy + self.H * p_new)
        diag = {"yield_residual": over, "overstress": over, "plastic": (dp > 0.0).astype(float), "delta_p": dp,
                "plastic_rate": dp / dt if dt > 0.0 else np.zeros(n), "plastic_work": np.einsum("nij,nij->n", sig, dep)}
        return HistoryUpdate(sig, C, {"plastic_strain": ep, "alpha": p_new}, diag, energy, diss)


# ================================================================================================= Drucker–Prager
class DruckerPrager(_SmallStrainBase):
    """Drucker–Prager pressure-sensitive plasticity with linear isotropic hardening, small strain, physical units.

    f = q − a p − b(κ), p = −tr σ/3 (compression positive), b = b0 + h κ [Pa]; ``a`` > 0 is the pressure coefficient
    (uniaxial tension yields at b/(1 + a/3), compression at b/(1 − a/3), hence 0 < a < 3), ``b0`` > 0 [Pa] the
    shear-like strength, ``hardening`` h ≥ 0 [Pa]. Flow potential g = q − a_g p, ``a_flow`` = a_g in (0, a]; ``None`` =
    a (associated). ε̇_p = λ̇ ((3/2) s/q + (a_g/3) 1), κ̇ = λ̇ (κ is the equivalent plastic strain for a_g = 0);
    plastic volume change tr ε̇_p = a_g λ̇ (dilation under flow). Refused: a ≤ 0, a ≥ 3, a_g ≤ 0 (the apex return is
    undefined), a_g > a (dilatancy beyond friction), b0 ≤ 0, h < 0.

    Return mapping from the elastic trial (s*, p*, q*): f* = q* − a p* − b(κ_n) ≤ 0 is elastic. Cone branch
    (q* − 3GΔλ ≥ 0): linear, Δλ = f*/Ξ, Ξ = 3G + K a a_g + h, s = (1 − 3GΔλ/q*) s*, p = p* + K a_g Δλ. Apex branch
    (otherwise): s = 0, Δλ = −(a p* + b_n)/(a K a_g + h), Δε_p = dev ε_e* + (a_g Δλ/3) 1, σ = (b(κ_{n+1})/a) 1 (the
    state has no deviatoric stress; the flow direction is any element of the apex normal cone, fixed by s = 0).
    Algorithmic tangent: cone C = K 1⊗1 + 2Gθ I_dev − 3G√(2/3) n̂ ⊗ (g_λ − Δλ 2G√(3/2) n̂/q*) − K a_g 1⊗g_λ,
    g_λ = (2G√(3/2) n̂ + aK 1)/Ξ (non-symmetric unless a_g = a); apex C = hK/(aK a_g + h) 1⊗1 (zero for h = 0). The
    derivative is one-sided at the cone/apex switch. State ``plastic_strain`` (n, 3, 3), ``kappa`` (n,). Energy
    ½ ε_e:σ + ½hκ²; dissipation D = σ:Δε_p − Δψ_h; on the cone D/λ̇ = b0 + (a − a_g) p, which is positive for a_g = a
    and for h = 0, but **not** guaranteed for a_g < a with h > 0 near the apex in tension: the value is returned, not
    clipped. Diagnostics: ``apex`` (1/0), ``delta_lambda``, ``yield_residual``, ``plastic_volume_strain``."""

    name = "drucker_prager"
    state_fields = ("plastic_strain", "kappa")

    def __init__(self, K: float, G: float, a: float, b0: float, hardening: float = 0.0,
                 a_flow: float | None = None) -> None:
        super().__init__(K, G)
        self.a = _finite_pos("a", a)
        if not self.a < 3.0:
            raise ValueError(f"DruckerPrager: a must be < 3 (no compressive yield stress for a >= 3), got {a!r}")
        self.b0 = _finite_pos("b0", b0)
        self.h = _finite_pos("hardening", hardening, strict=False)
        self.a_g = self.a if a_flow is None else _finite_pos("a_flow", a_flow)
        if self.a_g > self.a:
            raise ValueError(f"DruckerPrager: a_flow must be in (0, a], got {a_flow!r} > a = {a!r}")

    def b(self, kappa: np.ndarray) -> np.ndarray:
        return self.b0 + self.h * kappa

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        K, G, a, ag, h = self.K, self.G, self.a, self.a_g, self.h
        ep_n = np.asarray(committed["plastic_strain"])
        k_n = np.asarray(committed["kappa"])
        n = len(eps)
        ee, tre, dev, s_tr, q_tr = _elastic_trial(K, G, eps, ep_n)
        p_tr = -K * tre
        f_tr = q_tr - a * p_tr - self.b(k_n)
        plastic = f_tr > 0.0
        Xi = 3.0 * G + K * a * ag + h
        dl = np.zeros(n)
        dl[plastic] = f_tr[plastic] / Xi
        with np.errstate(divide="ignore", invalid="ignore"):
            cone = plastic & (q_tr - 3.0 * G * dl >= 0.0)
        apex = plastic & ~cone
        dl[apex] = -(a * p_tr[apex] + self.b(k_n[apex])) / (a * K * ag + h)
        theta = np.ones(n)
        theta[cone] = 1.0 - 3.0 * G * dl[cone] / q_tr[cone]
        theta[apex] = 0.0
        s = theta[:, None, None] * s_tr
        p = p_tr + K * ag * dl
        k_new = k_n + dl
        p[apex] = -self.b(k_new[apex]) / a
        nhat = np.zeros_like(s_tr)
        nrm = np.sqrt(np.einsum("nij,nij->n", s_tr, s_tr))
        nhat[cone] = s_tr[cone] / nrm[cone][:, None, None]
        dep = np.zeros_like(s_tr)
        dep[cone] = dl[cone][:, None, None] * (_SQ32 * nhat[cone] + (ag / 3.0) * _EYE)
        dep[apex] = dev[apex] + (ag * dl[apex] / 3.0)[:, None, None] * _EYE
        ep = ep_n + dep
        sig = s - p[:, None, None] * _EYE
        C = None
        if request.tangent:
            C = np.broadcast_to(self.Ce6, (n, 6, 6)).copy()
            if cone.any():
                nv = nhat[cone][:, _VA, _VB]
                gl = (2.0 * G * _SQ32 * nv + a * K * _ONE6[None]) / Xi
                w = gl - (dl[cone] * 2.0 * G * _SQ32 / q_tr[cone])[:, None] * nv
                C[cone] = (K * np.outer(_ONE6, _ONE6)[None] + 2.0 * G * theta[cone][:, None, None] * _IDEV6[None]
                           - 3.0 * G * math.sqrt(2.0 / 3.0) * nv[:, :, None] * w[:, None, :]
                           - K * ag * _ONE6[None, :, None] * gl[:, None, :])
            if apex.any():
                C[apex] = (h * K / (a * K * ag + h) if h > 0 else 0.0) * np.outer(_ONE6, _ONE6)[None]
        psi_h = lambda k: 0.5 * h * k ** 2               # noqa: E731
        energy = None
        if request.energy:
            energy = 0.5 * np.einsum("nij,nij->n", sig, ee - dep) + psi_h(k_new)
        wp = np.einsum("nij,nij->n", sig, dep)
        diss = wp - (psi_h(k_new) - psi_h(k_n))
        q_new = theta * q_tr
        yres = np.where(plastic, q_new - a * p - self.b(k_new), np.minimum(f_tr, 0.0))
        diag = {"yield_residual": yres, "plastic": plastic.astype(float), "apex": apex.astype(float),
                "delta_lambda": dl, "plastic_work": wp, "plastic_volume_strain": np.trace(dep, axis1=1, axis2=2)}
        return HistoryUpdate(sig, C, {"plastic_strain": ep, "kappa": k_new}, diag, energy, diss)


# ============================================================================================================ Eyring
def _eyring_x(tau: Any, V: float, T_: float, what: str) -> np.ndarray:
    V = _finite_pos(f"{what}: activation volume", V)
    T_ = _finite_pos(f"{what}: temperature [K]", T_)
    t = np.asarray(tau, float)
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{what}: the stress must be finite")
    return t * V / (BOLTZMANN * T_)


def eyring_rate(tau: Any, A: float, V_act: float, T: float) -> np.ndarray:
    """Eyring/sinh flow rate A sinh(τ V*/(k_B T)) [1/s] for the stress ``tau`` [Pa] (any sign: odd in τ), the
    pre-exponential rate ``A`` > 0 [1/s], the activation volume ``V_act`` > 0 per event [m³] and the absolute
    temperature ``T`` > 0 [K]. ``A`` is the rate at unit argument divided by sinh 1 and is a calibration constant (it
    carries the barrier exp(−ΔH/(k_B T)) and attempt frequency); this function does not model its temperature
    dependence. The result overflows to ±inf only where the true value exceeds the float range (|x| > ~710 + ln(1/A));
    use :func:`eyring_log_rate` beyond that."""
    A = _finite_pos("eyring_rate: A", A)
    x = _eyring_x(tau, V_act, T, "eyring_rate")
    ax = np.abs(x)
    with np.errstate(over="ignore", under="ignore"):
        mag = np.where(ax < 1e-3, np.sinh(ax), 0.5 * np.exp(ax) * (-np.expm1(-2.0 * ax)))
    return np.sign(x) * A * mag


def eyring_stress(rate: Any, A: float, V_act: float, T: float) -> np.ndarray:
    """Inverse of :func:`eyring_rate`: the stress [Pa] that gives the flow ``rate`` [1/s], τ = (k_B T/V*)
    asinh(rate/A); stable for any finite rate (asinh, no overflow), odd in the rate."""
    A = _finite_pos("eyring_stress: A", A)
    V = _finite_pos("eyring_stress: activation volume", V_act)
    T_ = _finite_pos("eyring_stress: temperature [K]", T)
    r = np.asarray(rate, float)
    if not np.all(np.isfinite(r)):
        raise ValueError("eyring_stress: the rate must be finite")
    return BOLTZMANN * T_ / V * np.arcsinh(r / A)


def eyring_log_rate(tau: Any, A: float, V_act: float, T: float) -> np.ndarray:
    """ln |rate| of :func:`eyring_rate` by the log-sinh form ln A + |x| − ln 2 + ln(1 − e^(−2|x|)), x = τ V*/(k_B T);
    finite for any x ≠ 0 (−inf at τ = 0). The sign of the rate is sign(τ)."""
    A = _finite_pos("eyring_log_rate: A", A)
    x = np.abs(_eyring_x(tau, V_act, T, "eyring_log_rate"))
    with np.errstate(divide="ignore"):
        return math.log(A) + x - math.log(2.0) + np.log(-np.expm1(-2.0 * x))


def eyring_stress_from_log_rate(log_rate: Any, A: float, V_act: float, T: float) -> np.ndarray:
    """Stress [Pa] ≥ 0 of the rate e^L (L = ``log_rate`` = ln |rate|) without forming the rate: asinh(e^(L − ln A)) =
    ln A' + ln(1 + √(1 + A'^−2)) for A' = e^(L − ln A) ≥ 1, direct otherwise; inverse of :func:`eyring_log_rate`."""
    A = _finite_pos("eyring_stress_from_log_rate: A", A)
    V = _finite_pos("eyring_stress_from_log_rate: activation volume", V_act)
    T_ = _finite_pos("eyring_stress_from_log_rate: temperature [K]", T)
    L = np.asarray(log_rate, float) - math.log(A)
    if np.any(np.isnan(L)):
        raise ValueError("eyring_stress_from_log_rate: log_rate must not be NaN")
    with np.errstate(over="ignore", under="ignore"):
        big = L + np.log1p(np.sqrt(1.0 + np.exp(-2.0 * np.maximum(L, 0.0))))
        x = np.where(L >= 0.0, big, np.arcsinh(np.exp(np.minimum(L, 0.0))))
    return BOLTZMANN * T_ / V * x


def eyring_rate_molar(tau: Any, A: float, V_molar: float, T: float) -> np.ndarray:
    """:func:`eyring_rate` with a molar activation volume ``V_molar`` [m³/mol]: x = τ V_m/(R T), R = N_A k_B."""
    return eyring_rate(tau, A, _finite_pos("eyring_rate_molar: molar volume", V_molar) / AVOGADRO, T)


def eyring_stress_molar(rate: Any, A: float, V_molar: float, T: float) -> np.ndarray:
    """:func:`eyring_stress` with a molar activation volume ``V_molar`` [m³/mol]."""
    return eyring_stress(rate, A, _finite_pos("eyring_stress_molar: molar volume", V_molar) / AVOGADRO, T)
