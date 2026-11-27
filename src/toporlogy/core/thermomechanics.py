"""Adiabatic thermomechanics of plastic dissipation at the material point (WP14 R05, WP11 P04):
:class:`AdiabaticPerzyna`.

A Perzyna overstress J2 point (:class:`toporlogy.core.viscoplastic.Perzyna`) whose yield stress depends on the
temperature, σ_y(T) (a :class:`toporlogy.core.properties.PropertyFunction`, Pa), and whose temperature is an internal
state heated by the plastic **dissipation** with a Taylor–Quinney fraction β ∈ [0, 1]:

    ρ c_p(T) Ṫ = β Ḋ_p,    Ḋ_p = σ:ε̇_p − ψ̇_h = q ṗ − ψ̇_h  (≥ 0),    ψ_h = ½ H p².

Units: SI. Stress, K, G, H in Pa; time s; T absolute kelvin; ρ kg/m³; c_p J/(kg K) (``PropertyFunction``); energies per
unit volume J/m³. Tension-positive stress, Voigt [xx, yy, zz, yz, xz, xy] (stress physical shear, strain engineering
shear). Every constant in the tests is synthetic.

Conventions (derivations in ``docs/foundations/prereg/R5_adiabatic.md``):

* The existing ``Perzyna.update`` returns the dissipation D = qΔp − Δψ_h, i.e. the plastic work σ:Δε_p minus the stored
  hardening energy; its diagnostic ``plastic_work`` is the total σ:Δε_p. **Only D heats**: the stored hardening
  energy is excluded (heat from the total plastic work would count Δψ_h twice, once as heat and once in ψ).
* Discrete heat equation (backward Euler in T, exact enthalpy increment so a temperature-dependent c_p is consistent):
  ρ ∫_{T_n}^{T_{n+1}} c_p dT = β D_{n+1}(T_{n+1}), with D evaluated at σ_y(T_{n+1}).
* The remainder (1 − β) D is **stored cold work** (defect energy not represented in ψ_h): the accounting state
  ``cold_work`` accumulates D − Q per step; it does not feed back on the flow. Per step (trapezoid work W):
  W = Δψ + Q + Δw_c + ½(σ_{n+1} − σ_n):Δε_p, the last term being the O(Δt²) integrator mismatch (it vanishes with dt).
* The returned ``energy`` is the mechanical stored energy ½ε_e:σ + ½Hp² only. The thermal part of the free energy and
  the thermoelastic and thermoplastic coupling terms are **neglected** (classical Taylor–Quinney model: one-way
  dissipation → heat, plus the temperature dependence of σ_y), so this is not a complete two-way thermodynamic model.

Coupled local update (monolithic): per point the scalar ΔT = T_{n+1} − T_n ≥ 0 solves
R(ΔT) = ρ [h(T_n + ΔT) − h(T_n)] − β D(T_n + ΔT) = 0 (h = ∫c_p dT) by Brent's method on the bracket [0, ΔT1] (ΔT1 =
β D(T_n)/(ρ c_p(T_n)), doubled while R < 0); R(0) = −β D(T_n) ≤ 0. If D(T_n) = 0 (elastic) or β = 0 the isothermal
update at T_n is the result, so β = 0 is **bitwise** the existing ``Perzyna`` with σ_y(T_n). Convergence is reported by
the diagnostic ``heat_residual`` = ρΔh − βD of the returned state. Leaving the validity interval of σ_y or c_p raises
``ValueError`` (no clamp, no extrapolation). Algorithmic tangent: the isothermal Perzyna tangent at σ_y(T_{n+1}) plus
the rank-one coupling ∂σ/∂T ⊗ dT/dε from the implicit function theorem (both factors are ∝ the deviatoric
direction n̂, so the tangent stays symmetric). Not implemented: heat
conduction (the spatial coupling), thermoelastic coupling, T-dependent moduli or hardening, finite strain, inertia.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import brentq

from toporlogy.core import tensor as T
from toporlogy.core.constitutive import (HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse,
                                         ReturnMappingFailure, TimeStep, require_step,
                                         small_strain_history_evaluate)
from toporlogy.core.properties import CP_UNITS, PropertyFunction
from toporlogy.core.viscoplastic import Perzyna

__all__ = ["AdiabaticPerzyna"]

_DEFAULT_REQUEST = MaterialRequest()
_VA = np.array([p[0] for p in T.VOIGT_3D])
_VB = np.array([p[1] for p in T.VOIGT_3D])
_SQ32 = math.sqrt(1.5)
_EYE = np.eye(3)
_MAX_EXPAND = 60


def _finite(name: str, v: Any, strict: bool = True) -> float:
    x = float(v)
    if not (math.isfinite(x) and (x > 0.0 if strict else x >= 0.0)):
        raise ValueError(f"{name} must be finite and {'> 0' if strict else '>= 0'}, got {v!r}")
    return x


class AdiabaticPerzyna:
    """Adiabatic Perzyna J2 point with σ_y(T) and the Taylor–Quinney fraction ``taylor_quinney`` = β ∈ [0, 1].

    Parameters: ``K``, ``G``, ``hardening`` H ≥ 0, ``tau0``, ``sigma_ref``, ``m`` as in :class:`Perzyna`;
    ``yield_stress`` a positive :class:`PropertyFunction` of T with units ``"Pa"`` (κ = σ_y(T) + H p); ``density``
    ρ > 0 [kg/m³]; ``specific_heat`` a positive :class:`PropertyFunction` in J/(kg K); ``temperature0`` the initial
    absolute temperature [K] (must lie in the validity interval of both functions). State ``plastic_strain``
    (n, 3, 3), ``alpha`` = p, ``temperature`` (n,) [K], ``cold_work`` (n,) [J/m³]. The temperature is **internal**: a
    time step that carries a temperature is rejected (the physical step ``request.step.dt`` is required).
    Diagnostics (per point): ``temperature`` (T_{n+1}), ``delta_T``, ``heat`` (Q = ρΔh, J/m³),
    ``cold_work_increment`` (D − Q), ``heat_residual`` (ρΔh − βD), ``thermal_evaluations`` (number of isothermal
    Perzyna evaluations in the solve) and the Perzyna diagnostics (``delta_p``, ``overstress``, ``plastic_work`` =
    σ:Δε_p including the stored hardening energy, ...). ``dissipation`` is D (J/m³, excludes the stored hardening
    energy), ``energy`` ½ε_e:σ + ½Hp²."""

    name = "adiabatic_perzyna"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields: tuple[str, ...] = ("plastic_strain", "alpha", "temperature", "cold_work")

    def __init__(self, K: float, G: float, yield_stress: PropertyFunction, hardening: float, tau0: float,
                 sigma_ref: float, m: float, density: float, specific_heat: PropertyFunction, taylor_quinney: float,
                 temperature0: float, local_tol: float = 1e-14, local_max_iter: int = 200) -> None:
        if not isinstance(yield_stress, PropertyFunction):
            raise ValueError("yield_stress must be a PropertyFunction of the absolute temperature")
        if yield_stress.units != "Pa":
            raise ValueError(f"yield_stress: units must be 'Pa', got {yield_stress.units!r}")
        if not yield_stress.positive:
            raise ValueError("yield_stress must be built with positive=True (sigma_y(T) > 0 on its whole domain)")
        if not isinstance(specific_heat, PropertyFunction):
            raise ValueError("specific_heat must be a PropertyFunction of the absolute temperature")
        if specific_heat.units not in CP_UNITS:
            raise ValueError(f"specific_heat: units must be one of {CP_UNITS}, got {specific_heat.units!r}")
        if not specific_heat.positive:
            raise ValueError("specific_heat must be built with positive=True (c_p > 0 is required)")
        b = float(taylor_quinney)
        if not (math.isfinite(b) and 0.0 <= b <= 1.0):
            raise ValueError(f"taylor_quinney must be a finite number in [0, 1], got {taylor_quinney!r}")
        self.beta = b
        self.rho = _finite("density", density)
        self.T0 = _finite("temperature0 [K]", temperature0)
        self.sy_fn = yield_stress
        self.cp_fn = specific_heat
        # constructing the isothermal kernel at T_0 validates K, G, H, tau0, sigma_ref, m (and the domains at T_0)
        proto = self._inner(float(yield_stress(self.T0)), K, G, hardening, tau0, sigma_ref, m, local_tol,
                            local_max_iter)
        specific_heat(self.T0)
        self._args = (float(K), float(G), float(hardening), float(tau0), float(sigma_ref), float(m),
                      float(local_tol), int(local_max_iter))
        self.K, self.G, self.H = proto.K, proto.G, proto.H
        self.m = proto.m
        self.Ce6 = proto.Ce6

    @staticmethod
    def _inner(sy: float, K: float, G: float, H: float, tau0: float, sigma_ref: float, m: float, local_tol: float,
               local_max_iter: int) -> Perzyna:
        return Perzyna(K, G, sy, H, tau0, sigma_ref, m, local_tol=local_tol, local_max_iter=local_max_iter)

    def isothermal(self, temperature: float) -> Perzyna:
        """The existing :class:`Perzyna` kernel with σ_y(``temperature``) (the β = 0 model at that temperature)."""
        K, G, H, tau0, sref, m, tol, mit = self._args
        return self._inner(float(self.sy_fn(temperature)), K, G, H, tau0, sref, m, tol, mit)

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"plastic_strain": np.zeros((n, 3, 3)), "alpha": np.zeros(n),
                "temperature": np.full(n, self.T0), "cold_work": np.zeros(n)}

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)

    # ------------------------------------------------------------------------------------------ heat source
    def _heat_source(self, dissipation: np.ndarray, diag: Mapping[str, np.ndarray]) -> np.ndarray:
        """The energy increment that β multiplies: the plastic dissipation D (hardening energy excluded).
        Overridden only by the negative control of the tests (total plastic work); the tangent assumes it."""
        return dissipation

    # ------------------------------------------------------------------------------------------ the point solve
    def _solve_point(self, eps1: np.ndarray, cm: dict[str, np.ndarray], Tn: float, req: MaterialRequest
                     ) -> tuple[HistoryUpdate, float, int]:
        """One point at the T_{n+1} that closes the heat equation: (isothermal update, T_{n+1}, evaluations)."""
        count = [0]

        def run(T: float) -> HistoryUpdate:
            count[0] += 1
            return self.isothermal(T).update(eps1, cm, req)

        u0 = run(Tn)
        src0 = float(self._heat_source(u0.dissipation, u0.diagnostics)[0])      # type: ignore[arg-type]
        if self.beta == 0.0 or src0 <= 0.0:
            return u0, Tn, count[0]
        rho, cp, beta = self.rho, self.cp_fn, self.beta

        def resid(x: float) -> float:
            T1 = Tn + x
            u = run(T1)
            assert u.dissipation is not None
            return rho * float(cp.integral(Tn, T1)) - beta * float(self._heat_source(u.dissipation, u.diagnostics)[0])

        hi = beta * src0 / (rho * float(cp(Tn)))
        k = 0
        while resid(hi) < 0.0:
            hi *= 2.0
            k += 1
            if k > _MAX_EXPAND:
                raise ReturnMappingFailure(f"adiabatic heat equation: no bracket found after {_MAX_EXPAND} doublings "
                                           f"(T_n = {Tn} K, heat source {src0} J/m^3)")
        x = brentq(resid, 0.0, hi, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=500)
        T1 = Tn + x
        return run(T1), T1, count[0]

    def _coupling(self, u: HistoryUpdate, p_n: float, T1: float) -> tuple[np.ndarray, np.ndarray]:
        """(∂σ/∂T (6,), dT/dε (6,)) of one flowing point by the implicit function theorem (record, derivation 5)."""
        G, H, m = self.G, self.H, self.m
        sig = u.stress[0]
        dev = sig - (np.trace(sig) / 3.0) * _EYE
        nrm = math.sqrt(float(np.einsum("ij,ij->", dev, dev)))
        nv = (dev / nrm)[_VA, _VB]
        q = _SQ32 * nrm
        dp = float(u.diagnostics["delta_p"][0])
        sy = float(self.sy_fn(T1))
        Xi = 3.0 * G + H
        q_tr = q + 3.0 * G * dp
        f_tr = q_tr - sy - H * p_n
        x = Xi * dp / f_tr
        chi = m * x / (Xi * ((1.0 - x) + m * x))
        sy1 = float(self.sy_fn.derivative(T1))
        D_dp = q_tr - H * p_n - (6.0 * G + H) * dp
        k = 2.0 * G * _SQ32
        dD_de = (dp + D_dp * chi) * k * nv
        dD_dT = -D_dp * chi * sy1
        dT_de = self.beta * dD_de / (self.rho * float(self.cp_fn(T1)) - self.beta * dD_dT)
        return k * chi * sy1 * nv, dT_de

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        step = require_step(self, request)
        if step.temperature is not None:
            raise ValueError("adiabatic_perzyna: the temperature is an internal state; pass TimeStep(t, dt) without "
                             "a temperature")
        n = len(eps)
        ep, al = np.asarray(committed["plastic_strain"]), np.asarray(committed["alpha"])
        Tn_all, wc = np.asarray(committed["temperature"], float), np.asarray(committed["cold_work"], float)
        inner = MaterialRequest(energy=request.energy, tangent=request.tangent, step=TimeStep(step.t, step.dt))
        sig = np.empty((n, 3, 3))
        ep_new, al_new, T_new, wc_new = np.empty((n, 3, 3)), np.empty(n), np.empty(n), np.empty(n)
        C = np.empty((n, 6, 6)) if request.tangent else None
        energy = np.empty(n) if request.energy else None
        diss = np.empty(n)
        diag: dict[str, np.ndarray] = {}
        for i in range(n):
            cm = {"plastic_strain": ep[i:i + 1], "alpha": al[i:i + 1]}
            Tn = float(Tn_all[i])
            u, T1, evals = self._solve_point(eps[i:i + 1], cm, Tn, inner)
            sig[i] = u.stress[0]
            ep_new[i], al_new[i], T_new[i] = u.trial_state["plastic_strain"][0], u.trial_state["alpha"][0], T1
            D = float(u.dissipation[0])                                         # type: ignore[index]
            heat = self.rho * float(self.cp_fn.integral(Tn, T1))
            wc_new[i] = wc[i] + (D - heat)
            diss[i] = D
            if energy is not None:
                energy[i] = float(u.energy[0])                                  # type: ignore[index]
            if C is not None:
                Ci = u.tangent[0].copy()                                        # type: ignore[index]
                if self.beta > 0.0 and float(u.diagnostics["delta_p"][0]) > 0.0:
                    dsig_dT, dT_de = self._coupling(u, float(al[i]), T1)
                    Ci = Ci + np.outer(dsig_dT, dT_de)
                C[i] = Ci
            src = float(self._heat_source(u.dissipation, u.diagnostics)[0])     # type: ignore[arg-type]
            point = dict(u.diagnostics)
            point.update({"temperature": np.array([T1]), "delta_T": np.array([T1 - Tn]), "heat": np.array([heat]),
                          "cold_work_increment": np.array([D - heat]),
                          "heat_residual": np.array([heat - self.beta * src]),
                          "thermal_evaluations": np.array([float(evals)])})
            for k, v in point.items():
                diag.setdefault(k, np.empty(n))[i] = float(np.asarray(v).reshape(-1)[0])
        trial = {"plastic_strain": ep_new, "alpha": al_new, "temperature": T_new, "cold_work": wc_new}
        return HistoryUpdate(sig, C, trial, diag, energy, diss)
