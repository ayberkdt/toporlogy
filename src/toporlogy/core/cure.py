"""Thermoset cure kinetics, glass transition, reaction heat, chemical shrinkage and crystallisation kinetics at the
material-point level (WP16 TS01-TS03, WP15 TP03; docs/foundations/CURE_AND_CRYSTALLISATION.md, pre-registration
docs/foundations/prereg/TS_cure.md).

Units (physical): time in seconds, temperature absolute in kelvin, rate constants in 1/s, Arrhenius activation energy
Q in J/mol (R = 8.314462618 J/(mol K)), reaction heat in J per kg of resin (exothermic positive), c_p in J/(kg K),
density in kg/m^3, strains dimensionless (small strain, tension positive). Every number in the tests is synthetic.

* Cure kinetics :class:`NthOrder` (dα/dt = k(T)(1−α)^n) and :class:`KamalSourour` (dα/dt = (k1 + k2 α^m)(1−α)^n),
  conversion α ∈ [0, 1]. :meth:`CureKinetics.advance` is backward Euler with the temperature linear in the step. For
  f ≥ 0 and f(1) = 0 the implicit equation x − α_n − h f(x) = 0 changes sign on [α_n, 1] (value −h f(α_n) ≤ 0 at α_n,
  1 − α_n ≥ 0 at 1), so a root in [α_n, 1] exists and the bracketed Newton iteration used here never leaves that
  bracket: 0 ≤ α_n ≤ α_{n+1} ≤ 1 holds by construction for any step size, without clipping. A solve that does not
  converge raises; inputs outside the contract raise ``ValueError``. The scheme is first-order accurate.
* :class:`DiBenedetto` glass transition Tg(α) with derivative and inverse.
* :class:`ReactionHeat` (heat of a step ΔH Δα, source ρ ΔH dα/dt, adiabatic temperature from the enthalpy of a
  :class:`~toporlogy.core.properties.PropertyFunction` c_p that is the sensible heat only).
* :class:`ChemicalShrinkage` eigenstrain β (α − α_gel)_+ (volumetric, β < 0 for shrinkage; β/3 per axis).
* Crystallisation: :func:`avrami` (X = 1 − exp(−k t^n), k in s^−n, X the relative transformed fraction) and
  :class:`Nakamura` (dX/dt = n K(T)(1−X)[−ln(1−X)]^{(n−1)/n}, K = k^{1/n}) integrated in θ = ∫K dt so that
  X = 1 − exp(−θ^n): exact for a K of the property-function family when T is linear in the step, no singularity
  at X = 0.
* :class:`CuringThermoElasticPoint`: σ = C(T, α):(ε − ε_th(T) 1 − ε_ch(α) 1/3) with the state α (one-way coupling,
  synthetic CHILE-type interpolation of the stiffness)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from toporlogy.core.constitutive import (HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse,
                                         ReturnMappingFailure, require_step, small_strain_history_evaluate)
from toporlogy.core.properties import CP_UNITS, Constant, IsotropicModuliOfT, PropertyFunction, ThermalExpansion

__all__ = ["R_GAS", "Arrhenius", "CureKinetics", "NthOrder", "KamalSourour", "isothermal_first_order",
           "isothermal_nth_order", "finite_completion_time", "DiBenedetto", "ReactionHeat", "ChemicalShrinkage",
           "avrami", "avrami_time", "nakamura_K_from_avrami", "Nakamura", "CuringThermoElasticPoint"]

R_GAS = 8.314462618                                   # J/(mol K)
RATE_UNITS = ("1/s", "s^-1", "s^{-1}")
_EPS = float(np.finfo(float).eps)
_MAX_ITER = 400


def _out(a: np.ndarray) -> Any:
    return float(a) if a.ndim == 0 else a


def _alpha_arr(a: Any, what: str = "alpha") -> np.ndarray:
    v = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v < 0.0) or np.any(v > 1.0):
        raise ValueError(f"{what} must be finite and within [0, 1], got {np.asarray(a)!r}")
    return v


def _temp_arr(T: Any, what: str = "temperature") -> np.ndarray:
    v = np.asarray(T, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError(f"{what} must be finite and > 0 K (absolute), got {np.asarray(T)!r}")
    return v


def _dt_arr(dt: Any) -> np.ndarray:
    v = np.asarray(dt, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError(f"dt must be finite and >= 0 s, got {np.asarray(dt)!r}")
    return v


def _pow(x: np.ndarray, p: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.power(x, p)


# ------------------------------------------------------------------------------------------------ rate constants
@dataclass(frozen=True)
class Arrhenius:
    """k(T) = A exp(−Q/(R T)): ``A`` in 1/s (>= 0), ``Q`` in J/mol (>= 0), T absolute (K)."""

    A: float
    Q: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.A) and self.A >= 0.0):
            raise ValueError(f"Arrhenius A must be finite and >= 0 (1/s), got {self.A!r}")
        if not (math.isfinite(self.Q) and self.Q >= 0.0):
            raise ValueError(f"Arrhenius Q must be finite and >= 0 (J/mol), got {self.Q!r}")

    @classmethod
    def from_reference(cls, k_ref: float, T_ref: float, Q: float) -> Arrhenius:
        """The law with k(T_ref) = ``k_ref`` (1/s, > 0) and activation energy ``Q`` (J/mol)."""
        if not (math.isfinite(k_ref) and k_ref > 0.0):
            raise ValueError(f"k_ref must be finite and > 0 (1/s), got {k_ref!r}")
        if not (math.isfinite(T_ref) and T_ref > 0.0):
            raise ValueError(f"T_ref must be finite and > 0 K, got {T_ref!r}")
        if not (math.isfinite(Q) and Q >= 0.0):
            raise ValueError(f"Q must be finite and >= 0 (J/mol), got {Q!r}")
        A = k_ref * math.exp(Q / (R_GAS * T_ref))
        if not math.isfinite(A):
            raise ValueError("the pre-exponential factor overflows: check Q and T_ref")
        return cls(A, Q)

    def k(self, T: Any) -> np.ndarray:
        Ta = _temp_arr(T)
        return self.A * np.exp(-self.Q / (R_GAS * Ta))

    def dk_dT(self, T: Any) -> np.ndarray:
        Ta = _temp_arr(T)
        return self.k(Ta) * self.Q / (R_GAS * Ta ** 2)


# ------------------------------------------------------------------------------------------------ cure kinetics
class CureKinetics:
    """Base of the cure laws dα/dt = f(α, T). Subclasses give ``_f`` (rate), ``_df`` (∂f/∂α) and ``_dfdT``."""

    def _f(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _df(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _dfdT(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def rate(self, alpha: Any, T: Any) -> Any:
        """dα/dt in 1/s."""
        return _out(self._f(_alpha_arr(alpha), _temp_arr(T)))

    def drate_dalpha(self, alpha: Any, T: Any) -> Any:
        """∂f/∂α (1/s); infinite where a non-integer exponent makes the derivative singular (α = 0 with m < 1,
        α = 1 with n < 1)."""
        return _out(self._df(_alpha_arr(alpha), _temp_arr(T)))

    def drate_dT(self, alpha: Any, T: Any) -> Any:
        """∂f/∂T (1/(s K))."""
        return _out(self._dfdT(_alpha_arr(alpha), _temp_arr(T)))

    # ------------------------------------------------------------------ the step
    def _backward_euler(self, a: np.ndarray, T: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Root of g(x) = x − a − h f(x, T) in [a, 1] by a bracketed Newton iteration (iterates only inside the
        bracket). g(a) = −h f(a) <= 0 and g(1) = 1 − a >= 0 for f >= 0, f(1) = 0."""
        with np.errstate(over="ignore", invalid="ignore"):
            done = (-h * self._f(a, T)) == 0.0                # no change (h = 0, f = 0: dt 0, equilibrium, α = 1)
        x = a.copy()
        lo, hi = a.copy(), np.ones_like(a)
        for _ in range(_MAX_ITER):
            if done.all():
                return x
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                g = x - a - h * self._f(x, T)
                gp = 1.0 - h * self._df(x, T)
                lo = np.where(~done & (g < 0.0), x, lo)
                hi = np.where(~done & (g > 0.0), x, hi)
                cand = x - g / gp
            newton_ok = np.isfinite(cand) & np.isfinite(gp) & (cand >= lo) & (cand <= hi)
            cand = np.where(newton_ok, cand, 0.5 * (lo + hi))
            exact = ~done & (g == 0.0)
            conv = ~done & (exact | (np.abs(cand - x) <= 4.0 * _EPS * np.abs(cand)))
            x = np.where(done | exact, x, cand)
            done = done | conv
        raise ReturnMappingFailure(f"implicit cure step did not converge in {_MAX_ITER} iterations")

    def advance(self, alpha_n: Any, T_n: Any, T_np1: Any, dt: Any, substeps: int = 1) -> Any:
        """α_{n+1} after the step dt (s) with the temperature linear from ``T_n`` to ``T_np1`` (K).

        Backward Euler on dα/dt = f(α, T): the rate is taken at the end temperature of the step (``substeps`` m > 1
        divides the step into m equal sub-steps with linearly interpolated temperature; first-order accurate in dt/m).
        The result lies in [α_n, 1] by construction (see the module docstring), is α_n for dt = 0, and is returned
        as a float for scalar input or an array for array input (broadcast). α_n outside [0, 1], dt < 0, T <= 0 or
        non-finite values raise ``ValueError``; the committed α_n is never modified."""
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError(f"substeps must be an integer >= 1, got {substeps!r}")
        scalar = all(np.ndim(v) == 0 for v in (alpha_n, T_n, T_np1, dt))
        a = _alpha_arr(alpha_n, "alpha_n")
        T0, T1, h = _temp_arr(T_n, "T_n"), _temp_arr(T_np1, "T_np1"), _dt_arr(dt)
        a, T0, T1, h = (np.array(v, dtype=float) for v in np.broadcast_arrays(a, T0, T1, h))
        m = int(substeps)
        for j in range(1, m + 1):
            Tj = T1 if j == m else T0 + (T1 - T0) * (j / m)
            a = self._backward_euler(a, Tj, h / m)
        return float(a) if scalar else a


class NthOrder(CureKinetics):
    """dα/dt = k(T) (1 − α)^n, k Arrhenius (1/s), n > 0 (non-integer allowed; n < 1 completes in finite time)."""

    def __init__(self, k: Arrhenius, n: float) -> None:
        if not isinstance(k, Arrhenius):
            raise ValueError("k must be an Arrhenius law")
        if not (math.isfinite(n) and n > 0.0):
            raise ValueError(f"reaction order n must be finite and > 0, got {n!r}")
        self.k, self.n = k, float(n)

    def _f(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.k.k(T) * _pow(1.0 - a, self.n)

    def _df(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        return -self.n * self.k.k(T) * _pow(1.0 - a, self.n - 1.0)

    def _dfdT(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        return self.k.dk_dT(T) * _pow(1.0 - a, self.n)


class KamalSourour(CureKinetics):
    """dα/dt = (k1(T) + k2(T) α^m)(1 − α)^n, k1, k2 Arrhenius (1/s), m >= 0, n > 0. With k1 = 0 the state α = 0 is an
    equilibrium (no initiation): start from α > 0 or k1 > 0."""

    def __init__(self, k1: Arrhenius, k2: Arrhenius, m: float, n: float) -> None:
        if not (isinstance(k1, Arrhenius) and isinstance(k2, Arrhenius)):
            raise ValueError("k1 and k2 must be Arrhenius laws")
        if not (math.isfinite(m) and m >= 0.0):
            raise ValueError(f"autocatalytic exponent m must be finite and >= 0, got {m!r}")
        if not (math.isfinite(n) and n > 0.0):
            raise ValueError(f"reaction order n must be finite and > 0, got {n!r}")
        self.k1, self.k2, self.m, self.n = k1, k2, float(m), float(n)

    def _f(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        return (self.k1.k(T) + self.k2.k(T) * _pow(a, self.m)) * _pow(1.0 - a, self.n)

    def _df(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        if self.m == 0.0 or self.k2.A == 0.0:
            t1: Any = 0.0
        else:
            t1 = self.k2.k(T) * self.m * _pow(a, self.m - 1.0) * _pow(1.0 - a, self.n)
        t2 = (self.k1.k(T) + self.k2.k(T) * _pow(a, self.m)) * self.n * _pow(1.0 - a, self.n - 1.0)
        return t1 - t2

    def _dfdT(self, a: np.ndarray, T: np.ndarray) -> np.ndarray:
        return (self.k1.dk_dT(T) + self.k2.dk_dT(T) * _pow(a, self.m)) * _pow(1.0 - a, self.n)


# ------------------------------------------------------------------------------------------------ isothermal forms
def isothermal_first_order(alpha0: Any, k: float, t: Any) -> Any:
    """Exact n = 1 isothermal conversion α(t) = 1 − (1 − α0) exp(−k t); k in 1/s (>= 0), t in s (>= 0)."""
    a0 = _alpha_arr(alpha0, "alpha0")
    tt = _dt_arr(t)
    if not (math.isfinite(k) and k >= 0.0):
        raise ValueError(f"k must be finite and >= 0 (1/s), got {k!r}")
    return _out(a0 + (1.0 - a0) * (-np.expm1(-k * tt)))


def isothermal_nth_order(alpha0: Any, k: float, n: float, t: Any) -> Any:
    """Exact isothermal n-th order conversion: (1−α)^{1−n} = (1−α0)^{1−n} + (n−1) k t (n ≠ 1; exp for n = 1). For
    n < 1 the conversion reaches 1 at t* = :func:`finite_completion_time` and stays 1 afterwards (the equilibrium
    branch of the ODE; the solution is not unique there)."""
    a0 = _alpha_arr(alpha0, "alpha0")
    tt = _dt_arr(t)
    if not (math.isfinite(k) and k >= 0.0):
        raise ValueError(f"k must be finite and >= 0 (1/s), got {k!r}")
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError(f"n must be finite and > 0, got {n!r}")
    if n == 1.0:
        return _out(a0 + (1.0 - a0) * (-np.expm1(-k * tt)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = np.power(1.0 - a0, 1.0 - n) + (n - 1.0) * k * tt
        res = 1.0 - np.power(u, 1.0 / (1.0 - n))
    if n < 1.0:
        res = np.where(u <= 0.0, 1.0, res)
    return _out(np.asarray(res, dtype=float))


def finite_completion_time(alpha0: float, k: float, n: float) -> float:
    """Time t* (s) at which the isothermal n-th order law reaches α = 1: (1−α0)^{1−n}/((1−n) k) for n < 1, ∞ for
    n >= 1 or k = 0."""
    a0 = float(_alpha_arr(alpha0, "alpha0"))
    if not (math.isfinite(k) and k >= 0.0 and math.isfinite(n) and n > 0.0):
        raise ValueError("k must be finite >= 0 and n finite > 0")
    if n >= 1.0 or k == 0.0:
        return math.inf
    return (1.0 - a0) ** (1.0 - n) / ((1.0 - n) * k)


# ------------------------------------------------------------------------------------------------ glass transition
@dataclass(frozen=True)
class DiBenedetto:
    """(Tg − Tg0)/(Tg∞ − Tg0) = λ α/(1 − (1 − λ) α), temperatures in kelvin (0 < Tg0 < Tg∞), λ > 0 (dimensionless;
    λ = 1 is the linear law). Parameters are inputs of one resin; none is inferred here."""

    Tg0: float
    Tg_inf: float
    lam: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.Tg0) and math.isfinite(self.Tg_inf) and 0.0 < self.Tg0 < self.Tg_inf):
            raise ValueError(f"need finite 0 < Tg0 < Tg_inf (kelvin), got {self.Tg0!r}, {self.Tg_inf!r}")
        if not (math.isfinite(self.lam) and self.lam > 0.0):
            raise ValueError(f"lambda must be finite and > 0, got {self.lam!r}")

    def Tg(self, alpha: Any) -> Any:
        a = _alpha_arr(alpha)
        return _out(np.asarray(self.Tg0 + (self.Tg_inf - self.Tg0) * self.lam * a / (1.0 - (1.0 - self.lam) * a)))

    def dTg_dalpha(self, alpha: Any) -> Any:
        """(Tg∞ − Tg0) λ / (1 − (1 − λ) α)²  (K per unit conversion)."""
        a = _alpha_arr(alpha)
        return _out(np.asarray((self.Tg_inf - self.Tg0) * self.lam / (1.0 - (1.0 - self.lam) * a) ** 2))

    def alpha_at_Tg(self, T: Any) -> Any:
        """The conversion at which Tg(α) = T (T in [Tg0, Tg∞]): α = x/(λ + (1 − λ) x), x = (T − Tg0)/(Tg∞ − Tg0)."""
        Ta = _temp_arr(T)
        if np.any(Ta < self.Tg0) or np.any(Ta > self.Tg_inf):
            raise ValueError(f"T must lie in [Tg0, Tg_inf] = [{self.Tg0}, {self.Tg_inf}] K, got {np.asarray(T)!r}")
        x = (Ta - self.Tg0) / (self.Tg_inf - self.Tg0)
        return _out(np.asarray(x / (self.lam + (1.0 - self.lam) * x)))


# ------------------------------------------------------------------------------------------------ reaction heat
@dataclass(frozen=True)
class ReactionHeat:
    """Total heat of reaction ``delta_h`` in J per kg of RESIN (> 0, exothermic). The heat released in a step is
    ΔH Δα (J/kg resin), the volumetric source ρ_resin ΔH dα/dt (W/m^3 of resin). It is the energy of the
    transformation; the c_p of the thermal equation is the sensible heat only (no latent term), so the reaction heat
    enters once, as the source."""

    delta_h: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.delta_h) and self.delta_h > 0.0):
            raise ValueError(f"delta_h must be finite and > 0 (J/kg of resin, exothermic), got {self.delta_h!r}")

    def heat(self, alpha_n: Any, alpha_np1: Any) -> Any:
        """ΔH (α_{n+1} − α_n) in J/kg; the conversion is irreversible, α_{n+1} < α_n raises."""
        a0, a1 = _alpha_arr(alpha_n, "alpha_n"), _alpha_arr(alpha_np1, "alpha_np1")
        if np.any(a1 < a0):
            raise ValueError("conversion decreased within a step: the cure reaction is irreversible")
        return _out(np.asarray(self.delta_h * (a1 - a0)))

    def power_density(self, rho: float, dalpha_dt: Any) -> Any:
        """q = ρ ΔH dα/dt in W/m^3 (ρ in kg/m^3 of the resin the heat is referred to)."""
        if not (math.isfinite(rho) and rho > 0.0):
            raise ValueError(f"rho must be finite and > 0 (kg/m^3), got {rho!r}")
        r = np.asarray(dalpha_dt, dtype=float)
        if not np.all(np.isfinite(r)) or np.any(r < 0.0):
            raise ValueError("dalpha_dt must be finite and >= 0 (1/s)")
        return _out(rho * self.delta_h * r)

    def adiabatic_temperature(self, T0: float, alpha_n: float, alpha_np1: float, cp: PropertyFunction) -> float:
        """Temperature (K) after an adiabatic conversion from α_n to α_{n+1} starting at ``T0``: the root T1 of
        ∫_{T0}^{T1} c_p dT = ΔH Δα. ``cp`` must be a strictly positive property function in J/(kg K)."""
        if not isinstance(cp, PropertyFunction) or cp.units not in CP_UNITS or not cp.positive:
            raise ValueError(f"cp must be a PropertyFunction built with positive=True and units in {CP_UNITS}")
        T0f = float(_temp_arr(T0, "T0"))
        q = float(self.heat(alpha_n, alpha_np1))
        c0 = float(cp.value(T0f))                                            # domain check of T0
        if q == 0.0:
            return T0f
        if isinstance(cp, Constant):
            T1 = T0f + q / c0
        else:
            hi = cp.domain[1]
            if not math.isfinite(hi) or float(cp.integral(T0f, hi)) < q:
                raise ValueError("the adiabatic temperature leaves the validity interval of c_p")
            T1 = float(brentq(lambda T: float(cp.integral(T0f, T)) - q, T0f, hi, xtol=1e-13, rtol=4 * _EPS,
                              maxiter=200))
        cp.value(T1)
        return float(T1)


# ------------------------------------------------------------------------------------------------ chemical shrinkage
@dataclass(frozen=True)
class ChemicalShrinkage:
    """Chemical eigenstrain of the cure, small strain: volumetric ε_v(α) = β max(α − α_gel, 0), ``beta`` the volumetric
    strain per unit conversion (β < 0 shrinkage) and ``alpha_gel`` ∈ [0, 1) the conversion from which the shrinkage
    produces strain (the stress-free reference conversion is a model choice; default 0 counts all of it). The linear
    strain per axis is ε_v/3 (small strain; finite strain would be (1 + ε_v)^{1/3} − 1 and is not implemented).
    The thermal strain is a separate function of T: ``beta`` is the shrinkage at constant temperature."""

    beta: float
    alpha_gel: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta):
            raise ValueError(f"beta must be finite, got {self.beta!r}")
        if not (math.isfinite(self.alpha_gel) and 0.0 <= self.alpha_gel < 1.0):
            raise ValueError(f"alpha_gel must lie in [0, 1), got {self.alpha_gel!r}")

    def volumetric(self, alpha: Any) -> Any:
        a = _alpha_arr(alpha)
        return _out(np.asarray(self.beta * np.maximum(a - self.alpha_gel, 0.0)))

    def linear(self, alpha: Any) -> Any:
        """ε_v/3 per axis."""
        return _out(np.asarray(self.volumetric(alpha)) / 3.0)

    def volumetric_increment(self, alpha_n: Any, alpha_np1: Any) -> Any:
        return _out(np.asarray(self.volumetric(alpha_np1)) - np.asarray(self.volumetric(alpha_n)))

    def linear_increment(self, alpha_n: Any, alpha_np1: Any) -> Any:
        return _out(np.asarray(self.volumetric_increment(alpha_n, alpha_np1)) / 3.0)


# ------------------------------------------------------------------------------------------------ crystallisation
def _avrami_check(k: float, n: float) -> None:
    if not (math.isfinite(k) and k > 0.0):
        raise ValueError(f"Avrami k must be finite and > 0 (s^-n), got {k!r}")
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError(f"Avrami n must be finite and > 0, got {n!r}")


def avrami(t: Any, k: float, n: float) -> Any:
    """Isothermal relative transformed fraction X(t) = 1 − exp(−k t^n); k in s^−n (K = k^{1/n} in 1/s), t >= 0 s.
    X is the relative crystallinity (X = 1 is the end of the transformation, not an absolute crystal fraction)."""
    _avrami_check(k, n)
    tt = _dt_arr(t)
    return _out(-np.expm1(-k * np.power(tt, n)))


def avrami_time(X: Any, k: float, n: float) -> Any:
    """Inverse of :func:`avrami`: t = (−ln(1 − X)/k)^{1/n} for X ∈ [0, 1); X = 1 needs infinite time and raises."""
    _avrami_check(k, n)
    x = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x >= 1.0):
        raise ValueError(f"X must lie in [0, 1), got {np.asarray(X)!r}")
    return _out(np.power(-np.log1p(-x) / k, 1.0 / n))


def nakamura_K_from_avrami(k: float, n: float) -> float:
    """K = k^{1/n} (1/s) from the Avrami constant k (s^−n)."""
    _avrami_check(k, n)
    return float(k ** (1.0 / n))


class Nakamura:
    """Nakamura non-isothermal crystallisation dX/dt = n K(T)(1 − X)[−ln(1 − X)]^{(n−1)/n} with K(T) = k(T)^{1/n} a
    :class:`~toporlogy.core.properties.PropertyFunction` in 1/s (value >= 0 on the path; zero outside the
    crystallisation window is allowed) and n the Avrami exponent.

    The state is θ = ∫K dt (>= 0, non-decreasing), X = 1 − exp(−θ^n) = −expm1(−θ^n) ∈ [0, 1]: this is the same equation
    without the start-up singularity at X = 0 (the right side is 0 at X = 0 for n > 1, so an integration in X never
    leaves X = 0). For T linear in the step the increment of θ is dt (∫_{T_n}^{T_{n+1}} K dT)/(T_{n+1} − T_n),
    exact for the constant, piecewise-linear and polynomial laws (a step with |ΔT| <= 1e-8 T uses K at the midpoint).
    A negative increment (K < 0 on the step) raises: melting and re-crystallisation are not modelled."""

    def __init__(self, n: float, K: PropertyFunction) -> None:
        if not (math.isfinite(n) and n > 0.0):
            raise ValueError(f"Avrami exponent n must be finite and > 0, got {n!r}")
        if not isinstance(K, PropertyFunction) or K.units not in RATE_UNITS:
            raise ValueError(f"K must be a PropertyFunction with units in {RATE_UNITS}")
        self.n, self.K = float(n), K

    def theta_increment(self, T_n: Any, T_np1: Any, dt: Any) -> Any:
        """∫K dt over the step (dimensionless), T linear in the step."""
        scalar = all(np.ndim(v) == 0 for v in (T_n, T_np1, dt))
        T0, T1, h = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(_temp_arr(T_n, "T_n"),
                                                                              _temp_arr(T_np1, "T_np1"), _dt_arr(dt)))
        small = np.abs(T1 - T0) <= 1e-8 * np.maximum(T0, T1)
        den = np.where(small, 1.0, T1 - T0)
        mean = np.where(small, np.asarray(self.K.value(0.5 * (T0 + T1))), np.asarray(self.K.integral(T0, T1)) / den)
        inc = h * mean
        if np.any(inc < 0.0):
            raise ValueError("negative increment of theta: K(T) < 0 on the step (melting is not modelled)")
        return float(inc) if scalar else inc

    def advance_theta(self, theta_n: Any, T_n: Any, T_np1: Any, dt: Any) -> Any:
        th = np.asarray(theta_n, dtype=float)
        if not np.all(np.isfinite(th)) or np.any(th < 0.0):
            raise ValueError(f"theta must be finite and >= 0, got {th!r}")
        out = th + np.asarray(self.theta_increment(T_n, T_np1, dt))
        return float(out) if out.ndim == 0 else out

    def X_of_theta(self, theta: Any) -> Any:
        th = np.asarray(theta, dtype=float)
        if np.any(np.isnan(th)) or np.any(th < 0.0):
            raise ValueError("theta must be >= 0")
        return _out(np.asarray(-np.expm1(-np.power(th, self.n))))

    def theta_of_X(self, X: Any) -> Any:
        x = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x > 1.0):
            raise ValueError(f"X must lie in [0, 1], got {np.asarray(X)!r}")
        with np.errstate(divide="ignore"):
            return _out(np.asarray(np.power(-np.log1p(-x), 1.0 / self.n)))

    def advance(self, X_n: Any, T_n: Any, T_np1: Any, dt: Any) -> Any:
        """X_{n+1} for the step (in [0, 1] by construction; X_n for a zero increment; 1 stays 1)."""
        th = np.asarray(self.theta_of_X(X_n))
        inc = np.asarray(self.theta_increment(T_n, T_np1, dt))
        X1 = np.asarray(self.X_of_theta(th + inc))
        out = np.where(inc == 0.0, np.asarray(X_n, dtype=float), X1)
        return float(out) if out.ndim == 0 else out

    def rate(self, X: Any, T: Any) -> Any:
        """dX/dt = n K (1 − X)[−ln(1 − X)]^{(n−1)/n} (1/s). At X = 0 it is 0 for n > 1 and infinite for n < 1 (the
        singularity that the θ form avoids); at X = 1 it is 0."""
        x = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x > 1.0):
            raise ValueError(f"X must lie in [0, 1], got {np.asarray(X)!r}")
        Kv = np.asarray(self.K.value(T), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.n * Kv * (1.0 - x) * np.power(-np.log1p(-x), (self.n - 1.0) / self.n)
        return _out(np.asarray(np.where(x == 1.0, 0.0, r)))


# ------------------------------------------------------------------------------------------------ the material point
_DEFAULT_REQUEST = MaterialRequest()


class CuringThermoElasticPoint:
    """Small-strain cure/thermoelastic material point with a prescribed temperature (one-way coupling T → α → σ):

        σ = C(T, α) : (ε − ε_th(T) 1 − ε_ch,lin(α) 1),    ε_ch,lin = ε_v/3,

    C = K 1⊗1 + 2G I_dev with K(T, α) = (1 − α) K_u(T) + α K_c(T) and G likewise (``moduli_uncured``,
    ``moduli_cured``: :class:`IsotropicModuliOfT` in one unit). This synthetic CHILE-type interpolation (cure hardening,
    instantaneous linear elastic) is an approximation with a numerical minimum stiffness of the uncured state, not a
    physical claim: the uncured resin carries no solid shear stress, and the reference of the strain
    (ε_th = 0 at ``expansion.T_ref``, ε_ch = 0 at ``shrinkage.alpha_gel``) is the stress-free state. State: α
    (``alpha0`` initial). The temperature and its previous value come from the time step; α_{n+1} =
    ``kinetics.advance(α_n, T_n, T_{n+1}, dt, substeps)`` does not depend on ε. Algorithmic tangent C(T_{n+1}, α_{n+1}).
    Energy and dissipation are not defined (None). Physical units (``physical = True``)."""

    name = "curing_thermoelastic"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields: tuple[str, ...] = ("alpha",)

    def __init__(self, kinetics: CureKinetics, moduli_uncured: IsotropicModuliOfT, moduli_cured: IsotropicModuliOfT,
                 expansion: ThermalExpansion, shrinkage: ChemicalShrinkage, alpha0: float = 0.0,
                 substeps: int = 1) -> None:
        if not isinstance(kinetics, CureKinetics):
            raise ValueError("kinetics must be a CureKinetics")
        if not (isinstance(moduli_uncured, IsotropicModuliOfT) and isinstance(moduli_cured, IsotropicModuliOfT)):
            raise ValueError("moduli_uncured and moduli_cured must be IsotropicModuliOfT")
        if moduli_uncured.units != moduli_cured.units:
            raise ValueError(f"the moduli must share one unit, got {moduli_uncured.units!r} and {moduli_cured.units!r}")
        if not isinstance(expansion, ThermalExpansion):
            raise ValueError("expansion must be a ThermalExpansion")
        if not isinstance(shrinkage, ChemicalShrinkage):
            raise ValueError("shrinkage must be a ChemicalShrinkage")
        self.alpha0 = float(_alpha_arr(alpha0, "alpha0"))
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError(f"substeps must be an integer >= 1, got {substeps!r}")
        self.kinetics, self.moduli_u, self.moduli_c = kinetics, moduli_uncured, moduli_cured
        self.expansion, self.shrinkage, self.substeps = expansion, shrinkage, int(substeps)
        self.units = moduli_uncured.units

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"alpha": np.full(n, self.alpha0)}

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        step = require_step(self, request, temperature=True)
        n = len(eps)
        T1 = np.broadcast_to(np.asarray(step.temperature, dtype=float), (n,))
        T0 = np.broadcast_to(np.asarray(step.temperature_prev, dtype=float), (n,))
        a1 = np.asarray(self.kinetics.advance(np.asarray(committed["alpha"], dtype=float), T0, T1, step.dt,
                                              self.substeps), dtype=float)
        Ku, Gu = np.asarray(self.moduli_u.K(T1), dtype=float), np.asarray(self.moduli_u.G(T1), dtype=float)
        Kc, Gc = np.asarray(self.moduli_c.K(T1), dtype=float), np.asarray(self.moduli_c.G(T1), dtype=float)
        K, G = (1.0 - a1) * Ku + a1 * Kc, (1.0 - a1) * Gu + a1 * Gc
        e_th = np.broadcast_to(np.asarray(self.expansion.strain(T1), dtype=float), (n,))
        e_ch = np.broadcast_to(np.asarray(self.shrinkage.linear(a1), dtype=float), (n,))
        e = eps - (e_th + e_ch)[:, None, None] * np.eye(3)
        tr = np.trace(e, axis1=1, axis2=2)
        dev = e - (tr / 3.0)[:, None, None] * np.eye(3)
        sig = (K * tr)[:, None, None] * np.eye(3) + 2.0 * G[:, None, None] * dev
        C = None
        if request.tangent:
            C = ((1.0 - a1)[:, None, None] * np.asarray(self.moduli_u.stiffness(T1), dtype=float)
                 + a1[:, None, None] * np.asarray(self.moduli_c.stiffness(T1), dtype=float))
        diag = {"alpha": a1.copy(), "temperature": np.array(T1), "thermal_strain_trace": 3.0 * e_th,
                "chemical_strain_trace": 3.0 * e_ch}
        return HistoryUpdate(sig, C, {"alpha": a1}, diag)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)
