"""Temperature-dependent material properties, thermal strain and a thermoelastic material point (WP11 P03/P04/P06).

Contract of this module (docs/foundations/TEMPERATURE_PROPERTIES.md, pre-registration
docs/foundations/prereg/P_properties.md):

* Temperature is absolute, in kelvin. A property function is evaluated at T > 0 inside its declared validity interval
  and nowhere else: the extrapolation policy is ``"error"`` (the only one implemented) and an out-of-range, NaN or
  non-positive temperature raises ``ValueError``. There is no clamp.
* Property functions: :class:`Constant`, :class:`PiecewiseLinear`, :class:`Pchip`, :class:`Polynomial` (in T − T_ref).
  Each has ``value``, ``derivative``, ``integral(T0, T1)`` (exact for the piecewise-linear and polynomial laws, through
  the antiderivative for PCHIP), a validity interval, a units string, a source string and an optional positivity
  requirement. Units are documented strings, checked against a small accepted set by the consumers (modulus: Pa, kPa,
  MPa, GPa; CTE: 1/K; c_p: J/(kg K)); the dimensional parser of ``configuration.units`` is not used here (it knows
  neither J nor W nor parenthesised units).
* :class:`IsotropicModuliOfT`: exactly two independent functions among E, nu, K, G; the other moduli and their
  T-derivatives follow by the chain rule through the formulas of WP04 E01. A temperature where K ≤ 0 or G ≤ 0 is
  refused.
* Thermal strain (small strain): ε_th(T) = ∫_{T_ref}^{T} α(T') dT' with α the instantaneous (differential) CTE, as an
  isotropic scalar (:class:`ThermalExpansion`) or a symmetric 3×3 tensor (:class:`ThermalExpansionTensor`); the secant
  CTE α_sec = ε_th/(T − T_ref) is a different function and the two are never exchanged silently. Finite thermal strain
  (multiplicative stretch) is NOT implemented.
* :class:`ThermoElasticPoint`: σ = C(T) : (ε − ε_th(T) I), a small-strain history-contract model with a prescribed
  temperature (one-way coupling).

All numbers in the tests of this module are synthetic."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from toporlogy.core.constitutive import (HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse, require_step,
                                         small_strain_history_evaluate)

__all__ = ["PropertyFunction", "Constant", "PiecewiseLinear", "Pchip", "Polynomial", "IsotropicModuliOfT",
           "ModuliAtT", "ThermalExpansion", "ThermalExpansionTensor", "Enthalpy", "ThermoElasticPoint"]

PRESSURE_UNITS = ("Pa", "kPa", "MPa", "GPa")
DIMENSIONLESS_UNITS = ("1", "-", "dimensionless")
CTE_UNITS = ("1/K", "K^-1")
CP_UNITS = ("J/(kg K)", "J/(kg*K)", "J kg^-1 K^-1")
_CELSIUS = ("c", "degc", "celsius", "°c", "f", "degf", "°f", "fahrenheit", "delta_degc")

_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_I6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])               # symmetric identity, Voigt (engineering-shear strain)
_IDEV6 = _I6 - np.outer(_ONE6, _ONE6) / 3.0
_KEYS3 = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "yz": (1, 2), "xz": (0, 2), "xy": (0, 1)}


def _out(a: np.ndarray) -> Any:
    return a[()] if a.ndim == 0 else a


class PropertyFunction:
    """Base of the property laws: a scalar function of the absolute temperature on a validity interval.

    ``units`` and ``source`` are documented strings; ``domain`` = (T_lo, T_hi) in kelvin (T_hi may be inf);
    ``extrapolation`` must be ``"error"``; ``positive`` rejects a law that is not strictly positive on its domain
    (checked exactly for the table laws and the polynomial). ``smoothness`` states the differentiability class;
    :meth:`nonsmooth` flags the temperatures where the derivative is not unique."""

    smoothness = ""

    def __init__(self, units: str, source: str, domain: tuple[float, float], extrapolation: str, positive: bool,
                 temperature_units: str) -> None:
        if not isinstance(units, str) or not units.strip():
            raise ValueError(f"units must be a non-empty string, got {units!r}")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"source must be a non-empty string (use 'synthetic' for invented numbers), "
                             f"got {source!r}")
        if extrapolation != "error":
            raise ValueError(f"extrapolation policy {extrapolation!r}: only 'error' is implemented (no silent clamp "
                             "or extrapolation)")
        if not isinstance(temperature_units, str) or temperature_units.strip().lower() in _CELSIUS \
                or temperature_units.strip() != "K":
            raise ValueError(f"temperature units must be 'K' (absolute kelvin), got {temperature_units!r}; degrees "
                             "Celsius are ambiguous between an absolute temperature and a difference and are refused")
        lo, hi = float(domain[0]), float(domain[1])
        if not (math.isfinite(lo) and lo >= 0.0 and hi > lo and not math.isnan(hi)):
            raise ValueError("validity interval must satisfy 0 <= T_lo < T_hi (kelvin, T_hi may be inf), "
                             f"got {domain!r}")
        self.units = units.strip()
        self.source = source.strip()
        self.domain = (lo, hi)
        self.extrapolation = extrapolation
        self.positive = bool(positive)
        self.temperature_units = "K"

    # ------------------------------------------------------------------ evaluation
    def _prep(self, T: Any) -> np.ndarray:
        a = np.asarray(T, dtype=float)
        if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError(f"temperature must be finite and > 0 K (absolute), got {np.asarray(T)!r}")
        lo, hi = self.domain
        bad = (a < lo) | (a > hi)
        if np.any(bad):
            raise ValueError(f"temperature {float(a[bad].flat[0])!r} K is outside the validity interval "
                             f"[{lo}, {hi}] K of this property ({self.source}); extrapolation policy 'error'")
        return a

    def value(self, T: Any) -> Any:
        return _out(self._value(self._prep(T)))

    def __call__(self, T: Any) -> Any:
        return self.value(T)

    def derivative(self, T: Any) -> Any:
        """dp/dT (units per kelvin). At a non-smooth temperature (:meth:`nonsmooth`) the one-sided value stated by the
        subclass is returned."""
        return _out(self._derivative(self._prep(T)))

    def integral(self, T0: Any, T1: Any) -> Any:
        """∫_{T0}^{T1} p dT (T1 < T0 gives the negative). Both limits must lie in the validity interval."""
        return _out(self._integral(self._prep(T0), self._prep(T1)))

    def nonsmooth(self, T: Any) -> Any:
        """True where p is not differentiable (an interior knot of a piecewise-linear table)."""
        return _out(np.zeros(self._prep(T).shape, bool))

    # ------------------------------------------------------------------ to override
    def _value(self, T: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _integral(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _require_positive(self, minimum: float, what: str) -> None:
        if self.positive and not minimum > 0.0:
            raise ValueError(f"{what}: positive=True but the minimum on the validity interval is {minimum:.6g}")


class Constant(PropertyFunction):
    """p(T) = c on ``domain`` (default (0, inf): every T > 0). Derivative 0, integral c (T1 − T0)."""

    smoothness = "C-infinity"

    def __init__(self, c: float, units: str, source: str = "unspecified", domain: tuple[float, float] = (0.0, math.inf),
                 positive: bool = False, extrapolation: str = "error", temperature_units: str = "K") -> None:
        super().__init__(units, source, domain, extrapolation, positive, temperature_units)
        self.c = float(c)
        if not math.isfinite(self.c):
            raise ValueError(f"constant must be finite, got {c!r}")
        self._require_positive(self.c, "Constant")

    def _value(self, T: np.ndarray) -> np.ndarray:
        return np.full(T.shape, self.c)

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        return np.zeros(T.shape)

    def _integral(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        return self.c * (T1 - T0)


def _knots(T: Any, y: Any, what: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(T, dtype=float)
    v = np.asarray(y, dtype=float)
    if x.ndim != 1 or v.shape != x.shape:
        raise ValueError(f"{what}: knots and values must be 1-D arrays of the same length, got {x.shape} and {v.shape}")
    if len(x) < 2:
        raise ValueError(f"{what}: at least 2 knots are required, got {len(x)}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
        raise ValueError(f"{what}: knots and values must be finite")
    if x[0] <= 0.0:
        raise ValueError(f"{what}: knot temperatures are absolute (kelvin) and must be > 0, got {x[0]!r}")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError(f"{what}: knots must be strictly increasing (no duplicates, no decreasing order)")
    return x, v


class PiecewiseLinear(PropertyFunction):
    """Linear interpolation of a table (knots strictly increasing, kelvin). Validity interval = [first, last] knot.
    C0 only: the derivative is the slope of the segment; at an interior knot it is not unique and ``side`` selects
    the right-hand ("right", default) or left-hand ("left") slope; at the first (last) knot the slope of the only
    adjacent segment. :meth:`nonsmooth` is True at interior knots. The integral is exact (sum of trapezoids)."""

    smoothness = "C0 (derivative discontinuous at interior knots)"

    def __init__(self, T: Any, values: Any, units: str, source: str = "unspecified", positive: bool = False,
                 extrapolation: str = "error", temperature_units: str = "K", side: str = "right") -> None:
        x, v = _knots(T, values, "PiecewiseLinear")
        super().__init__(units, source, (float(x[0]), float(x[-1])), extrapolation, positive, temperature_units)
        if side not in ("right", "left"):
            raise ValueError("side: 'right' or 'left'")
        self.x, self.y, self.side = x, v, side
        self._require_positive(float(v.min()), "PiecewiseLinear")

    def _value(self, T: np.ndarray) -> np.ndarray:
        return np.interp(T, self.x, self.y)

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        n = len(self.x)
        idx = np.searchsorted(self.x, T, side="right" if self.side == "right" else "left") - 1
        idx = np.clip(idx, 0, n - 2)
        return (self.y[idx + 1] - self.y[idx]) / (self.x[idx + 1] - self.x[idx])

    def nonsmooth(self, T: Any) -> Any:
        a = self._prep(T)
        return _out(np.isin(a, self.x[1:-1]))

    def _integral(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        lo, hi = np.minimum(T0, T1), np.maximum(T0, T1)
        total = np.zeros(np.broadcast(lo, hi).shape)
        for i in range(len(self.x) - 1):
            a = np.clip(lo, self.x[i], self.x[i + 1])
            b = np.clip(hi, self.x[i], self.x[i + 1])
            slope = (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
            ya = self.y[i] + slope * (a - self.x[i])
            yb = self.y[i] + slope * (b - self.x[i])
            total = total + 0.5 * (b - a) * (ya + yb)
        return np.where(T1 >= T0, total, -total)


class Pchip(PropertyFunction):
    """Piecewise cubic Hermite interpolation (Fritsch-Carlson, ``scipy.interpolate.PchipInterpolator``): monotone on
    every interval where the data are monotone, passes through the knots, never leaves [min y, max y]. C1; the second
    derivative jumps at knots. Validity interval = [first, last] knot. The integral uses the exact antiderivative of
    the cubic pieces."""

    smoothness = "C1 (second derivative discontinuous at knots)"

    def __init__(self, T: Any, values: Any, units: str, source: str = "unspecified", positive: bool = False,
                 extrapolation: str = "error", temperature_units: str = "K") -> None:
        x, v = _knots(T, values, "Pchip")
        super().__init__(units, source, (float(x[0]), float(x[-1])), extrapolation, positive, temperature_units)
        self.x, self.y = x, v
        self._p = PchipInterpolator(x, v, extrapolate=False)
        self._dp = self._p.derivative()
        self._F = self._p.antiderivative()
        # PCHIP is monotone on each interval, so its extrema are knot values
        self._require_positive(float(v.min()), "Pchip")

    def _value(self, T: np.ndarray) -> np.ndarray:
        return np.asarray(self._p(T), dtype=float)

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        return np.asarray(self._dp(T), dtype=float)

    def _integral(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        return np.asarray(self._F(T1), dtype=float) - np.asarray(self._F(T0), dtype=float)


class Polynomial(PropertyFunction):
    """p(T) = Σ_k c_k (T − T_ref)^k, ``coefficients`` (c_0, c_1, …) with the unit of p per K^k; the validity interval
    ``domain`` is required (finite). Exact derivative and integral; ``positive`` is checked exactly (end points and the
    real roots of p' inside the domain)."""

    smoothness = "C-infinity"

    def __init__(self, coefficients: Any, T_ref: float, domain: tuple[float, float], units: str,
                 source: str = "unspecified", positive: bool = False, extrapolation: str = "error",
                 temperature_units: str = "K") -> None:
        super().__init__(units, source, domain, extrapolation, positive, temperature_units)
        if not math.isfinite(self.domain[1]):
            raise ValueError("Polynomial: a finite validity interval is required")
        c = np.atleast_1d(np.asarray(coefficients, dtype=float))
        if c.ndim != 1 or len(c) == 0 or not np.all(np.isfinite(c)):
            raise ValueError("Polynomial: coefficients must be a non-empty finite 1-D sequence")
        if not (math.isfinite(T_ref) and T_ref > 0.0):
            raise ValueError(f"Polynomial: T_ref must be a finite absolute temperature > 0 K, got {T_ref!r}")
        self.T_ref = float(T_ref)
        self.coefficients = c
        self._poly = np.polynomial.Polynomial(c)
        self._dpoly = self._poly.deriv() if len(c) > 1 else np.polynomial.Polynomial([0.0])
        self._ipoly = self._poly.integ()
        if self.positive:
            xs = [self.domain[0] - self.T_ref, self.domain[1] - self.T_ref]
            if len(c) > 2:
                r = self._dpoly.roots()
                xs += [float(z.real) for z in r if abs(complex(z).imag) <= 1e-12 * max(1.0, abs(complex(z)))
                       and xs[0] < float(z.real) < xs[1]]
            self._require_positive(float(min(self._poly(np.array(xs)))), "Polynomial")

    def _value(self, T: np.ndarray) -> np.ndarray:
        return self._poly(T - self.T_ref)

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        return self._dpoly(T - self.T_ref)

    def _integral(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        return self._ipoly(T1 - self.T_ref) - self._ipoly(T0 - self.T_ref)


# ------------------------------------------------------------------------------------------------ isotropic moduli
@dataclass(frozen=True)
class ModuliAtT:
    """E, ν, K, G, λ (arrays at the evaluated temperatures; the moduli in the units of the inputs, ν dimensionless)."""

    E: Any
    nu: Any
    K: Any
    G: Any
    lam: Any


def _pair_partials(keys: frozenset[str], a: np.ndarray, b: np.ndarray, first: str, second: str
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K, G and ∂K/∂p1, ∂K/∂p2, ∂G/∂p1, ∂G/∂p2 for the pair (p1 = ``first``, p2 = ``second``, values a, b)."""
    v = {first: a, second: b}
    one, zero = np.ones_like(a), np.zeros_like(a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if keys == {"K", "G"}:
            K, G = v["K"], v["G"]
            parts = {("K", "K"): one, ("K", "G"): zero, ("G", "K"): zero, ("G", "G"): one}
        elif keys == {"E", "nu"}:
            E, nu = v["E"], v["nu"]
            K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
            parts = {("K", "E"): 1 / (3 * (1 - 2 * nu)), ("K", "nu"): 2 * E / (3 * (1 - 2 * nu) ** 2),
                     ("G", "E"): 1 / (2 * (1 + nu)), ("G", "nu"): -E / (2 * (1 + nu) ** 2)}
        elif keys == {"E", "K"}:
            E, K = v["E"], v["K"]
            G = 3 * K * E / (9 * K - E)
            parts = {("K", "E"): zero, ("K", "K"): one, ("G", "E"): 27 * K ** 2 / (9 * K - E) ** 2,
                     ("G", "K"): -3 * E ** 2 / (9 * K - E) ** 2}
        elif keys == {"E", "G"}:
            E, G = v["E"], v["G"]
            K = E * G / (3 * (3 * G - E))
            parts = {("K", "E"): G ** 2 / (3 * G - E) ** 2, ("K", "G"): -E ** 2 / (3 * (3 * G - E) ** 2),
                     ("G", "E"): zero, ("G", "G"): one}
        elif keys == {"K", "nu"}:
            K, nu = v["K"], v["nu"]
            G = 3 * K * (1 - 2 * nu) / (2 * (1 + nu))
            parts = {("K", "K"): one, ("K", "nu"): zero, ("G", "K"): 3 * (1 - 2 * nu) / (2 * (1 + nu)),
                     ("G", "nu"): -9 * K / (2 * (1 + nu) ** 2)}
        else:                                                                   # {"G", "nu"}
            G, nu = v["G"], v["nu"]
            K = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))
            parts = {("K", "G"): 2 * (1 + nu) / (3 * (1 - 2 * nu)), ("K", "nu"): 2 * G / (1 - 2 * nu) ** 2,
                     ("G", "G"): one, ("G", "nu"): zero}
    return (K, G, parts[("K", first)], parts[("K", second)], parts[("G", first)], parts[("G", second)])


class IsotropicModuliOfT:
    """Isotropic elastic moduli of temperature from exactly TWO independent property functions among E, nu, K, G
    (keyword arguments), e.g. ``IsotropicModuliOfT(E=E_of_T, nu=nu_of_T)``. K(T) and G(T) follow from the pair; then
    E = 9KG/(3K+G), ν = (3K − 2G)/(2(3K + G)), λ = K − 2G/3 (WP04 E01) and the T-derivatives by the chain rule
    (∂E/∂K = 9G²/(3K+G)², ∂E/∂G = 27K²/(3K+G)², ∂ν/∂K = 9G/(2(3K+G)²), ∂ν/∂G = −9K/(2(3K+G)²)). Redundant data
    (three or four of the moduli) are refused: the redundancy cannot be checked against measurement uncertainty here and
    inconsistent copies must not be averaged silently. A temperature at which K ≤ 0, G ≤ 0 (or E ≤ 0, ν outside
    (−1, 1/2) for the inputs that carry them) is refused with ``ValueError``; the constructor also samples 257
    temperatures over the common validity interval (a sampled, not a certified, admissibility check), and every
    evaluation checks the temperatures asked exactly. Units: the moduli inputs must share one units string from
    Pa, kPa, MPa, GPa; ν is dimensionless."""

    def __init__(self, **two: PropertyFunction) -> None:
        keys = frozenset(two)
        allowed = {"E", "nu", "K", "G"}
        if not keys <= allowed:
            raise ValueError(f"unsupported modulus {sorted(keys - allowed)}: use two of E, nu, K, G")
        if len(keys) != 2:
            raise ValueError(f"give exactly two independent moduli functions among E, nu, K, G; got {sorted(keys)} "
                             "(redundant moduli are refused)")
        for k, f in two.items():
            if not isinstance(f, PropertyFunction):
                raise ValueError(f"{k} must be a PropertyFunction, got {type(f).__name__}")
            if k == "nu":
                if f.units not in DIMENSIONLESS_UNITS:
                    raise ValueError(f"nu must be dimensionless (units one of {DIMENSIONLESS_UNITS}), got {f.units!r}")
            elif f.units not in PRESSURE_UNITS:
                raise ValueError(f"{k}: modulus units must be one of {PRESSURE_UNITS}, got {f.units!r}")
        mod_units = {f.units for k, f in two.items() if k != "nu"}
        if len(mod_units) > 1:
            raise ValueError(f"the moduli inputs must share one unit, got {sorted(mod_units)}: convert first")
        self.units = mod_units.pop()
        self.keys = keys
        self._order = tuple(sorted(keys))
        self._f = {k: two[k] for k in self._order}
        lo = max(f.domain[0] for f in self._f.values())
        hi = min(f.domain[1] for f in self._f.values())
        if not hi > lo:
            raise ValueError("the validity intervals of the two moduli functions do not overlap")
        self.domain = (lo, hi)
        self.source = "; ".join(f"{k}: {f.source}" for k, f in self._f.items())
        probe = np.linspace(max(lo, 1.0), hi, 257) if math.isfinite(hi) else np.array([max(lo, 1.0)])
        self._pair(probe)                                                         # raises at the first bad T

    # ------------------------------------------------------------------
    def _pair(self, T: Any) -> tuple[np.ndarray, ...]:
        a_name, b_name = self._order
        Ta = np.asarray(T, dtype=float)
        a = np.asarray(self._f[a_name].value(Ta), dtype=float)
        b = np.asarray(self._f[b_name].value(Ta), dtype=float)
        da = np.asarray(self._f[a_name].derivative(Ta), dtype=float)
        db = np.asarray(self._f[b_name].derivative(Ta), dtype=float)
        K, G, Ka, Kb, Ga, Gb = _pair_partials(self.keys, a, b, a_name, b_name)
        ok = np.isfinite(K) & np.isfinite(G) & (K > 0.0) & (G > 0.0)
        for name, arr in ((a_name, a), (b_name, b)):
            if name == "E":
                ok &= arr > 0.0
            elif name == "nu":
                ok &= (arr > -1.0) & (arr < 0.5)
        if not np.all(ok):
            i = int(np.flatnonzero(~ok.reshape(-1))[0])
            T_bad = float(np.broadcast_to(Ta, ok.shape).reshape(-1)[i])
            raise ValueError(f"inadmissible isotropic moduli at T = {T_bad}"
                             f" K: {a_name} = {float(a.reshape(-1)[i]):.6g}, {b_name} = {float(b.reshape(-1)[i]):.6g}"
                             f" give K = {float(K.reshape(-1)[i]):.6g}, G = {float(G.reshape(-1)[i]):.6g}"
                             " (K > 0 and G > 0 are required)")
        dK = Ka * da + Kb * db
        dG = Ga * da + Gb * db
        return K, G, dK, dG, a, b

    def moduli(self, T: Any) -> ModuliAtT:
        """E, ν, K, G, λ at T."""
        K, G, *_ = self._pair(T)
        return self._assemble(K, G)

    def dmoduli(self, T: Any) -> ModuliAtT:
        """dE/dT, dν/dT, dK/dT, dG/dT, dλ/dT at T (chain rule)."""
        K, G, dK, dG, *_ = self._pair(T)
        s = 3 * K + G
        dE = 9 * G ** 2 / s ** 2 * dK + 27 * K ** 2 / s ** 2 * dG
        dnu = 9 * G / (2 * s ** 2) * dK - 9 * K / (2 * s ** 2) * dG
        return ModuliAtT(_out(dE), _out(dnu), _out(dK), _out(dG), _out(dK - 2 * dG / 3))

    @staticmethod
    def _assemble(K: np.ndarray, G: np.ndarray) -> ModuliAtT:
        s = 3 * K + G
        return ModuliAtT(_out(9 * K * G / s), _out((3 * K - 2 * G) / (2 * s)), _out(K), _out(G), _out(K - 2 * G / 3))

    def value(self, name: str, T: Any) -> Any:
        return getattr(self.moduli(T), "lam" if name == "lambda" else name)

    def derivative(self, name: str, T: Any) -> Any:
        return getattr(self.dmoduli(T), "lam" if name == "lambda" else name)

    def E(self, T: Any) -> Any:
        return self.moduli(T).E

    def nu(self, T: Any) -> Any:
        return self.moduli(T).nu

    def K(self, T: Any) -> Any:
        return self.moduli(T).K

    def G(self, T: Any) -> Any:
        return self.moduli(T).G

    def lam(self, T: Any) -> Any:
        return self.moduli(T).lam

    def stiffness(self, T: Any) -> np.ndarray:
        """(…, 6, 6) Voigt stiffness K 1⊗1 + 2G I_dev (stress physical shear, strain engineering shear)."""
        K, G, *_ = self._pair(T)
        return K[..., None, None] * np.outer(_ONE6, _ONE6) + 2.0 * G[..., None, None] * _IDEV6

    def dstiffness(self, T: Any) -> np.ndarray:
        """dC/dT = dK/dT 1⊗1 + 2 dG/dT I_dev."""
        _K, _G, dK, dG, *_ = self._pair(T)
        return dK[..., None, None] * np.outer(_ONE6, _ONE6) + 2.0 * dG[..., None, None] * _IDEV6


# ------------------------------------------------------------------------------------------------ thermal strain
def _check_cte(f: PropertyFunction, what: str) -> None:
    if not isinstance(f, PropertyFunction):
        raise ValueError(f"{what} must be a PropertyFunction, got {type(f).__name__}")
    if f.units not in CTE_UNITS:
        raise ValueError(f"{what}: units must be one of {CTE_UNITS}, got {f.units!r}")


def _check_tref(T_ref: float, f: PropertyFunction, what: str) -> float:
    if not (isinstance(T_ref, (int, float, np.floating)) and math.isfinite(T_ref) and T_ref > 0.0):
        raise ValueError(f"T_ref must be a finite absolute temperature > 0 K, got {T_ref!r}")
    lo, hi = f.domain
    if not lo <= T_ref <= hi:
        raise ValueError(f"T_ref = {T_ref} K is outside the validity interval [{lo}, {hi}] of {what}")
    return float(T_ref)


class ThermalExpansion:
    """Isotropic thermal strain relative to the stress-free reference temperature ``T_ref`` (kelvin):

        ε_th(T) = ∫_{T_ref}^{T} α(T') dT'     (α the INSTANTANEOUS, differential CTE, 1/K),

    :meth:`strain`; the secant CTE α_sec(T) = ε_th/(T − T_ref) is :meth:`secant` (its limit at T = T_ref is α(T_ref);
    a temperature within 1e-9 K of T_ref returns it). Data given as secant CTE (a table of α_sec(T) relative to this
    T_ref) are taken with :meth:`from_secant`: ε_th = α_sec ΔT and α = α_sec + ΔT dα_sec/dT. Small strain only: the
    additive decomposition ε = ε_el + ε_th; finite thermal strain is not implemented."""

    def __init__(self, alpha: PropertyFunction, T_ref: float) -> None:
        _check_cte(alpha, "alpha")
        self._alpha = alpha
        self._secant_fn: PropertyFunction | None = None
        self.T_ref = _check_tref(T_ref, alpha, "alpha")
        self.domain = alpha.domain
        self.units = alpha.units
        self.source = alpha.source

    @classmethod
    def from_secant(cls, alpha_secant: PropertyFunction, T_ref: float) -> ThermalExpansion:
        """Thermal expansion from secant (average) CTE data referred to ``T_ref``."""
        _check_cte(alpha_secant, "alpha_secant")
        obj = cls.__new__(cls)
        obj._alpha = None  # type: ignore[assignment]
        obj._secant_fn = alpha_secant
        obj.T_ref = _check_tref(T_ref, alpha_secant, "alpha_secant")
        obj.domain = alpha_secant.domain
        obj.units = alpha_secant.units
        obj.source = alpha_secant.source
        return obj

    def strain(self, T: Any) -> Any:
        """ε_th(T) (dimensionless)."""
        if self._secant_fn is not None:
            T_a = self._secant_fn._prep(T)
            return _out(np.asarray(self._secant_fn.value(T_a)) * (T_a - self.T_ref))
        return self._alpha.integral(self.T_ref, T)

    def alpha(self, T: Any) -> Any:
        """Instantaneous CTE α(T) (1/K)."""
        if self._secant_fn is not None:
            T_a = self._secant_fn._prep(T)
            return _out(np.asarray(self._secant_fn.value(T_a))
                        + (T_a - self.T_ref) * np.asarray(self._secant_fn.derivative(T_a)))
        return self._alpha.value(T)

    def secant(self, T: Any) -> Any:
        """Secant CTE ε_th/(T − T_ref) (1/K); α(T_ref) at T = T_ref."""
        if self._secant_fn is not None:
            return self._secant_fn.value(T)
        T_a = self._alpha._prep(T)
        dT = T_a - self.T_ref
        small = np.abs(dT) <= 1e-9
        eps = np.asarray(self._alpha.integral(self.T_ref, T_a))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(small, self._alpha.value(self.T_ref), eps / np.where(small, 1.0, dT))
        return _out(out)


class ThermalExpansionTensor:
    """Symmetric 3×3 thermal strain ε_th,ij(T) = ∫_{T_ref}^{T} α_ij dT from per-component instantaneous CTE functions
    ``alpha`` = {"xx", "yy", "zz", "yz", "xz", "xy": PropertyFunction} (missing components are zero; at least one is
    required) given in the material axes; ``rotation`` R (3×3 orthogonal, det +1; columns = material axes in global
    coordinates) maps them to global axes, ε = R ε_mat Rᵀ. :meth:`strain` returns (…, 3, 3)."""

    def __init__(self, alpha: Mapping[str, PropertyFunction], T_ref: float, rotation: Any = None) -> None:
        if not alpha:
            raise ValueError("alpha: at least one component of xx, yy, zz, yz, xz, xy is required")
        bad = set(alpha) - set(_KEYS3)
        if bad:
            raise ValueError(f"alpha: unknown component(s) {sorted(bad)}; use {sorted(_KEYS3)}")
        for k, f in alpha.items():
            _check_cte(f, f"alpha[{k!r}]")
        lo = max(f.domain[0] for f in alpha.values())
        hi = min(f.domain[1] for f in alpha.values())
        if not hi > lo:
            raise ValueError("the validity intervals of the CTE components do not overlap")
        self.domain = (lo, hi)
        if not (isinstance(T_ref, (int, float, np.floating)) and math.isfinite(T_ref) and T_ref > 0.0
                and lo <= T_ref <= hi):
            raise ValueError(f"T_ref = {T_ref!r} must be an absolute temperature inside [{lo}, {hi}] K")
        self.T_ref = float(T_ref)
        self._alpha = dict(alpha)
        if rotation is None:
            self.R = np.eye(3)
        else:
            R = np.asarray(rotation, dtype=float)
            if (R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-12)
                    or abs(np.linalg.det(R) - 1) > 1e-12):
                raise ValueError("rotation must be a proper orthogonal 3x3 matrix")
            self.R = R

    def _assemble(self, comp: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
        A = np.zeros(shape + (3, 3))
        for k, v in comp.items():
            i, j = _KEYS3[k]
            A[..., i, j] = v
            A[..., j, i] = v
        return np.einsum("ij,...jk,lk->...il", self.R, A, self.R)

    def strain(self, T: Any) -> np.ndarray:
        """ε_th(T), (…, 3, 3), global axes."""
        shape = np.asarray(T).shape
        return self._assemble({k: np.asarray(f.integral(self.T_ref, T)) for k, f in self._alpha.items()}, shape)

    def alpha(self, T: Any) -> np.ndarray:
        """Instantaneous CTE tensor (…, 3, 3), global axes."""
        shape = np.asarray(T).shape
        return self._assemble({k: np.asarray(f.value(T)) for k, f in self._alpha.items()}, shape)


class Enthalpy:
    """Specific enthalpy relative to ``T_ref``: H(T) = ∫_{T_ref}^{T} c_p dT (J/kg) for a strictly positive c_p
    (``cp.positive`` must be True; units J/(kg K)). Latent heat is not included."""

    def __init__(self, cp: PropertyFunction, T_ref: float) -> None:
        if not isinstance(cp, PropertyFunction):
            raise ValueError("cp must be a PropertyFunction")
        if cp.units not in CP_UNITS:
            raise ValueError(f"cp: units must be one of {CP_UNITS}, got {cp.units!r}")
        if not cp.positive:
            raise ValueError("cp must be built with positive=True (c_p > 0 is required)")
        self.cp = cp
        self.T_ref = _check_tref(T_ref, cp, "cp")
        self.units = "J/kg"

    def H(self, T: Any) -> Any:
        return self.cp.integral(self.T_ref, T)


# ------------------------------------------------------------------------------------------------ the material point
_DEFAULT_REQUEST = MaterialRequest()


class ThermoElasticPoint:
    """Small-strain thermoelastic material point with a prescribed temperature (one-way coupling):

        σ = C(T) : (ε − ε_th(T) 1),   C(T) = K(T) 1⊗1 + 2G(T) I_dev,   ε_th = ∫_{T_ref}^{T} α dT,

    ``moduli`` :class:`IsotropicModuliOfT`; ``alpha`` an instantaneous-CTE :class:`PropertyFunction`, a
    :class:`ThermalExpansion` (also from secant data) or a :class:`ThermalExpansionTensor`; ``T_ref`` the stress-free
    temperature (required for a bare function; must equal the one of a thermal-expansion object). Physical units
    (``physical = True``: stresses in the units of the moduli; not for ``NonlinearElasticity``). The temperature comes
    from the time step (``require_step(..., temperature=True)``); the dt is not used (elastic, rate independent).
    Stateless. The algorithmic tangent is C(T) at the step temperature (the update is linear in ε at fixed T). Energy
    and dissipation are not defined for this prescribed-temperature model (None); the elastic strain energy
    ½ (ε − ε_th):C:(ε − ε_th) is returned as the diagnostic ``elastic_strain_energy`` (it is not the Helmholtz free
    energy: the temperature dependence of C is not accounted for, so it is not a potential of σ at varying T)."""

    name = "thermoelastic"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields: tuple[str, ...] = ()

    def __init__(self, moduli: IsotropicModuliOfT, alpha: PropertyFunction | ThermalExpansion | ThermalExpansionTensor,
                 T_ref: float | None = None) -> None:
        if not isinstance(moduli, IsotropicModuliOfT):
            raise ValueError("moduli must be an IsotropicModuliOfT")
        if isinstance(alpha, PropertyFunction):
            if T_ref is None:
                raise ValueError("T_ref (the stress-free temperature, kelvin) is required with a bare CTE function")
            alpha = ThermalExpansion(alpha, T_ref)
        elif isinstance(alpha, (ThermalExpansion, ThermalExpansionTensor)):
            if T_ref is not None and float(T_ref) != alpha.T_ref:
                raise ValueError(f"T_ref = {T_ref} differs from the reference temperature {alpha.T_ref} K of alpha")
        else:
            raise ValueError("alpha must be a PropertyFunction, ThermalExpansion or ThermalExpansionTensor")
        self.moduli = moduli
        self.expansion = alpha
        self.T_ref = alpha.T_ref
        self.units = moduli.units

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {}

    def thermal_strain_tensor(self, T: Any, n: int) -> np.ndarray:
        """ε_th(T) as (n, 3, 3) for a scalar or (n,) temperature."""
        Tn = np.broadcast_to(np.asarray(T, dtype=float), (n,))
        e = np.asarray(self.expansion.strain(Tn))
        if e.ndim == 1:
            return e[:, None, None] * np.eye(3)
        return e

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        step = require_step(self, request, temperature=True)
        n = len(eps)
        T = np.broadcast_to(np.asarray(step.temperature, dtype=float), (n,))
        K = np.asarray(self.moduli.K(T), dtype=float)
        G = np.asarray(self.moduli.G(T), dtype=float)
        eth = self.thermal_strain_tensor(T, n)
        e = eps - eth
        tr = np.trace(e, axis1=1, axis2=2)
        dev = e - (tr / 3.0)[:, None, None] * np.eye(3)
        sig = (K * tr)[:, None, None] * np.eye(3) + 2.0 * G[:, None, None] * dev
        C = None
        if request.tangent:
            C = K[:, None, None] * np.outer(_ONE6, _ONE6) + 2.0 * G[:, None, None] * _IDEV6
        diag = {"thermal_strain_trace": np.trace(eth, axis1=1, axis2=2), "temperature": np.array(T),
                "elastic_strain_energy": 0.5 * np.einsum("nij,nij->n", sig, e)}
        return HistoryUpdate(sig, C, {}, diag)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)
