"""Fractional viscoelasticity at the material point: Mittag-Leffler function, springpot models, Prony bridge (WP12 V05).

Physical units throughout (SI: Pa, s; angular frequency ω in rad/s, convention e^{iωt}). Derivations, the validated
ranges and the error budget: ``docs/foundations/FRACTIONAL_VISCOELASTICITY.md``; preregistration:
``docs/foundations/prereg/VF_fractional.md``. Same conventions as :mod:`toporlogy.core.viscoelastic`: the relaxation
modulus G(t) [Pa] (stress under γ0 H(t)) is not 1/J(t), the creep compliance J(t) [1/Pa] (strain under σ0 H(t)); t ≥ 0
is the time after the step.

**Mittag-Leffler function** :func:`mittag_leffler`: E_{α,β}(z) = Σ z^k/Γ(αk+β) for real z ≤ 0, 0 < α ≤ 1, β > 0. Method
(x = −z): the power series when x^{1/α} ≤ 5 (the largest term is ≈ exp(x^{1/α}), cancellation ≤ 2 digits); otherwise the
Hankel contour deformed onto the negative real axis, i.e. the real integral
E_{α,β}(−x) = (1/(πα))∫₀^∞ e^{-u^{1/α}} u^{(1−β)/α} [u sin πβ + x sin π(β−α)]/(u² + 2xu cos πα + x²) du (β < α+1),
with β ≥ α+1 reached by the upward recurrence E_{α,β+α}(z) = (E_{α,β}(z) − 1/Γ(β))/z, and the asymptotic expansion
−Σ z^{-k}/Γ(β−αk) for x ≥ 1e8. α = 1: exp and its integer-β recurrence.

**Models** (relaxation time τ, c = E τ^α the springpot constant, u = t/τ, E_α = E_{α,1}):
fractional Maxwell G = E E_α(−u^α), J = (1/E)(1 + u^α/Γ(1+α)); fractional Kelvin–Voigt J = (1/E) u^α E_{α,α+1}(−u^α),
G = E(1 + u^{-α}/Γ(1−α)) (α < 1; the ideal α = 1 has no finite relaxation modulus and is refused); fractional Zener
(G∞ parallel to a fractional Maxwell branch G1) G = G∞ + G1 E_α(−u^α), J = 1/G0 + (G1/G0²) u^α E_{α,α+1}(−λu^α),
λ = G∞/G0, G0 = G∞ + G1; complex modulus G* = G∞ + G1 z/(1+z), z = (iωτ)^α, with G'' ≥ 0 and the loss peak at ωτ = 1.
α = 1 is the standard linear solid of :mod:`toporlogy.core.viscoelastic`.

**Memory approximation** :func:`fit_prony_fractional_zener`, :func:`prony_error_budget`: a non-negative Prony series
(:class:`~toporlogy.core.viscoelastic.PronySeries`, hence a :class:`~toporlogy.core.viscoelastic.GeneralizedMaxwell`)
fitted to the fractional relaxation modulus over a stated time window, with its measured error.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import nnls
from scipy.special import cosdg, gamma, gammaln, rgamma, sindg

from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries, _pos, _time

__all__ = ["mittag_leffler", "fractional_maxwell_relaxation", "fractional_maxwell_creep",
           "fractional_kelvin_voigt_creep", "fractional_kelvin_voigt_relaxation", "fractional_zener_relaxation",
           "fractional_zener_creep", "fractional_zener_modulus", "fractional_kelvin_voigt_modulus", "is_passive",
           "FractionalPronyFit", "fit_prony_fractional_zener", "prony_error_budget"]

_X_SERIES = 5.0            # series when x^{1/α} <= this (largest term ≈ e^5)
_X_ASYMPTOTIC = 1e8        # asymptotic expansion from here (x² would overflow the integrand far beyond)
_U_LIMIT = 45.0            # integral cut at u^{1/α} = 45 (e^{-45} = 3e-20 relative)
_QUAD_REL = 1e-12


# ----------------------------------------------------------------------------------------------- input checks
def _alpha(alpha: Any) -> float:
    a = float(alpha)
    if not (math.isfinite(a) and 0.0 < a <= 1.0):
        raise ValueError(f"fractional order alpha must be finite and in (0, 1], got {alpha!r}")
    return a


def _nonneg(name: str, x: Any) -> float:
    v = float(x)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {x!r}")
    return v


def _sinpi(t: float) -> float:
    return float(sindg(180.0 * t))


def _cospi(t: float) -> float:
    return float(cosdg(180.0 * t))


# ----------------------------------------------------------------------------------------------- Mittag-Leffler
def _ml_series(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Σ_k (−x)^k/Γ(ak+b) for x ≥ 0 with x^{1/a} ≤ 5, terms added until below 1e-22 of the first one."""
    if x.size == 0:
        return np.zeros(0)
    xmax = float(np.max(x))
    lx = np.log(np.where(x > 0.0, x, 1.0))
    K = 32
    while True:
        k = np.arange(K, dtype=float)
        last = np.exp(float(K - 1) * math.log(xmax) - gammaln(a * (K - 1) + b)) if xmax > 0.0 else 0.0
        if last < 1e-22 * float(rgamma(b)) or K >= 40000:
            break
        K *= 2
    sign = np.where(k % 2 == 0, 1.0, -1.0)
    # k = 0 term handled separately so x = 0 is exact (ln 0 avoided)
    with np.errstate(over="ignore", under="ignore"):
        terms = np.exp(k[None, 1:] * lx[:, None] - gammaln(a * k[1:] + b)[None, :]) * sign[None, 1:]
    terms = np.where(x[:, None] > 0.0, terms, 0.0)
    return float(rgamma(b)) + terms.sum(axis=1)


def _ml_integral(x: float, a: float, b: float) -> float:
    """E_{a,b}(−x) for 0 < a < 1, 0 < b < a+1 from the real integral on the negative axis (module notes)."""
    sb, sba, ca = _sinpi(b), _sinpi(b - a), _cospi(a)
    e = (1.0 - b) / a
    inv_a = 1.0 / a
    U = _U_LIMIT ** a

    def f(u: float) -> float:
        return math.exp(-u ** inv_a) * (u * sb + x * sba) / (u * u + 2.0 * x * u * ca + x * x)

    m = min(x, U)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        if e == 0.0:
            i1 = quad(f, 0.0, m, epsabs=0.0, epsrel=_QUAD_REL, limit=400)[0]
        else:
            i1 = quad(f, 0.0, m, weight="alg", wvar=(e, 0.0), epsabs=0.0, epsrel=_QUAD_REL, limit=400)[0]
        i2 = 0.0
        if m < U:
            i2 = quad(lambda u: u ** e * f(u), m, U, epsabs=0.0, epsrel=_QUAD_REL, limit=400)[0]
    return (i1 + i2) / (math.pi * a)


def _ml_asymptotic(x: np.ndarray, a: float, b: float, n: int = 6) -> np.ndarray:
    """E_{a,b}(−x) ≈ Σ_{k=1}^n (−1)^{k+1} x^{-k}/Γ(b−ak), x large (error O(x^{-n-1}))."""
    out = np.zeros_like(x)
    for k in range(1, n + 1):
        out += (1.0 if k % 2 else -1.0) * x ** (-k) * float(rgamma(b - a * k))
    return out


def _ml_alpha_lt1(x: np.ndarray, a: float, b: float) -> np.ndarray:
    out = np.empty_like(x)
    ser = x ** (1.0 / a) <= _X_SERIES
    out[ser] = _ml_series(x[ser], a, b)
    asy = ~ser & (x >= _X_ASYMPTOTIC)
    out[asy] = _ml_asymptotic(x[asy], a, b)
    mid = ~ser & ~asy
    if np.any(mid):
        m = int(math.ceil((b - 1.0) / a - 1e-12)) if b > 1.0 else 0
        b0 = b - m * a
        xs = x[mid]
        val = np.array([_ml_integral(float(xi), a, b0) for xi in xs])
        bc = b0
        for _ in range(m):                                   # E_{a,bc+a}(−x) = (1/Γ(bc) − E_{a,bc}(−x))/x
            val = (float(rgamma(bc)) - val) / xs
            bc += a
        out[mid] = val
    return out


def _ml_alpha1(x: np.ndarray, b: float) -> np.ndarray:
    """α = 1: E_{1,1}(−x) = e^{-x}; integer β ≥ 2 by the recurrence (x > 1) or the series (x ≤ 1, no cancellation)."""
    n = int(round(b))
    if abs(b - n) > 1e-12 or n < 1:
        raise ValueError(f"alpha = 1 is implemented for integer beta >= 1 only, got beta = {b!r}")
    if n == 1:
        return np.exp(-x)
    out = np.empty_like(x)
    ser = x <= 1.0
    out[ser] = _ml_series(x[ser], 1.0, float(n))
    xs = x[~ser]
    val = np.exp(-xs)
    for j in range(1, n):
        val = (float(rgamma(float(j))) - val) / xs
    out[~ser] = val
    return out


def mittag_leffler(z: Any, alpha: float, beta: float = 1.0) -> np.ndarray:
    """Two-parameter Mittag-Leffler function E_{α,β}(z) = Σ_{k≥0} z^k/Γ(αk+β) for **real z ≤ 0**, 0 < α ≤ 1, β > 0.

    Method (module notes): series for |z|^{1/α} ≤ 5, the real integral of the deformed Hankel contour otherwise
    (``scipy.integrate.quad``, relative 1e-12), upward recurrence in β for β ≥ α+1, the asymptotic expansion for
    |z| ≥ 1e8; α = 1: exp (β = 1) and the integer-β recurrence. Validated against mpmath to ≥ 30 digits for
    |z| ≤ 1e3 (relative 1e-10, α ∈ {0.3, 0.5, 0.8, 1}; see the preregistration); z > 0 is refused (a different
    branch of the problem: the function grows like exp(z^{1/α}) and cancellation is not an issue there, but it is
    not needed and not validated). Returns an array of the shape of ``z``."""
    a = _alpha(alpha)
    b = _pos("beta", beta)
    zz = np.asarray(z, float)
    if not np.all(np.isfinite(zz)):
        raise ValueError("z must be finite")
    if np.any(zz > 0.0):
        raise ValueError("mittag_leffler is implemented for real z <= 0 only")
    x = (-zz).reshape(-1)
    if a == 1.0:
        out = _ml_alpha1(x, b)
    else:
        out = _ml_alpha_lt1(x, a, b)
    return out.reshape(zz.shape)


# ----------------------------------------------------------------------------------------------- scalar models
def _u_alpha(t: Any, tau: float, alpha: float) -> np.ndarray:
    return (_time(t) / tau) ** alpha


def fractional_maxwell_relaxation(t: Any, E: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Maxwell (spring E [Pa] in series with a springpot E τ^α s^α): G(t) = E E_α(−(t/τ)^α) [Pa]; a fluid
    (G → 0 like t^{-α}); α = 1 is `maxwell_relaxation` with η = E τ."""
    E, tau, a = _pos("E", E), _pos("tau", tau), _alpha(alpha)
    return E * mittag_leffler(-_u_alpha(t, tau, a), a)


def fractional_maxwell_creep(t: Any, E: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Maxwell creep compliance J(t) = (1/E)(1 + (t/τ)^α/Γ(1+α)) [1/Pa]; α = 1: `maxwell_creep`."""
    E, tau, a = _pos("E", E), _pos("tau", tau), _alpha(alpha)
    return (1.0 + _u_alpha(t, tau, a) / float(gamma(1.0 + a))) / E


def fractional_kelvin_voigt_creep(t: Any, E: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Kelvin–Voigt (spring E parallel to a springpot E τ^α s^α) creep compliance
    J(t) = (1/E)(1 − E_α(−u^α)) = (1/E) u^α E_{α,α+1}(−u^α) [1/Pa], u = t/τ (second form: no cancellation at small u);
    J(0) = 0; α = 1: `kelvin_voigt_creep` with η = E τ."""
    E, tau, a = _pos("E", E), _pos("tau", tau), _alpha(alpha)
    ua = _u_alpha(t, tau, a)
    return ua * mittag_leffler(-ua, a, a + 1.0) / E


def fractional_kelvin_voigt_relaxation(t: Any, E: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Kelvin–Voigt relaxation modulus G(t) = E (1 + (t/τ)^{-α}/Γ(1−α)) [Pa] for 0 < α < 1 and t > 0
    (singular, integrable, at 0+). α = 1 (ideal Kelvin–Voigt: impulsive stress) and t = 0 are refused."""
    E, tau, a = _pos("E", E), _pos("tau", tau), _alpha(alpha)
    if a == 1.0:
        raise ValueError("fractional Kelvin-Voigt with alpha = 1 has no finite relaxation modulus (the step response "
                         "is E H(t) + eta delta(t)); use fractional_kelvin_voigt_creep")
    tt = _time(t)
    if np.any(tt == 0.0):
        raise ValueError("fractional Kelvin-Voigt relaxation modulus is singular at t = 0; t must be > 0")
    return E * (1.0 + (tt / tau) ** (-a) / float(gamma(1.0 - a)))


def _zener_params(G_inf: float, G1: float, tau: float, alpha: float) -> tuple[float, float, float, float]:
    return _nonneg("G_inf", G_inf), _pos("G1", G1), _pos("tau", tau), _alpha(alpha)


def fractional_zener_relaxation(t: Any, G_inf: float, G1: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Zener relaxation modulus G(t) = G∞ + G1 E_α(−(t/τ)^α) [Pa]; G∞ ≥ 0 (G∞ = 0: fractional Maxwell),
    G(0) = G∞ + G1; α = 1: `sls_relaxation`."""
    G_inf, G1, tau, a = _zener_params(G_inf, G1, tau, alpha)
    return G_inf + G1 * mittag_leffler(-_u_alpha(t, tau, a), a)


def fractional_zener_creep(t: Any, G_inf: float, G1: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Zener creep compliance J(t) = 1/G0 + (G1/G0²) u^α E_{α,α+1}(−λ u^α) [1/Pa], G0 = G∞ + G1,
    λ = G∞/G0 (equal to 1/G∞ − (G1/(G∞ G0)) E_α(−λ u^α) for G∞ > 0; G∞ = 0 gives the fractional Maxwell creep);
    α = 1: `sls_creep`."""
    G_inf, G1, tau, a = _zener_params(G_inf, G1, tau, alpha)
    G0 = G_inf + G1
    lam = G_inf / G0
    ua = _u_alpha(t, tau, a)
    return 1.0 / G0 + (G1 / (G0 * G0)) * ua * mittag_leffler(-lam * ua, a, a + 1.0)


def _z_over_1pz(omega: Any, tau: float, a: float) -> tuple[np.ndarray, np.ndarray]:
    """(Re, Im) of z/(1+z), z = (iωτ)^α = u e^{iπα/2}, u = (ωτ)^α: Im = u s/(1+2uc+u²) ≥ 0,
    Re = (uc+u²)/(1+2uc+u²); for u > 1 the u → 1/u form is used (no overflow)."""
    w = np.asarray(omega, float)
    if not (np.all(np.isfinite(w)) and np.all(w >= 0.0)):
        raise ValueError("omega must be finite and >= 0 (rad/s)")
    u = (w * tau) ** a
    c, s = _cospi(0.5 * a), _sinpi(0.5 * a)
    big = u > 1.0
    v = np.where(big, 1.0 / np.where(big, u, 1.0), u)
    den = 1.0 + 2.0 * v * c + v * v
    im = v * s / den
    re = np.where(big, (c * v + 1.0) / den, (v * c + v * v) / den)
    return re, im


def fractional_zener_modulus(omega: Any, G_inf: float, G1: float, tau: float, alpha: float) -> np.ndarray:
    """Complex modulus G*(ω) = G' + iG'' = G∞ + G1 (iωτ)^α/(1 + (iωτ)^α) [Pa] for e^{iωt}, ω ≥ 0 in rad/s.
    G'' = G1 (ωτ)^α sin(πα/2)/(1 + 2(ωτ)^α cos(πα/2) + (ωτ)^{2α}) ≥ 0 (passive), maximal at ωτ = 1 with value
    (G1/2) tan(πα/4); G∞ = 0 is the fractional Maxwell modulus; α = 1 equals ``viscoelastic.complex_modulus`` of the
    one-branch Prony series."""
    G_inf, G1, tau, a = _zener_params(G_inf, G1, tau, alpha)
    re, im = _z_over_1pz(omega, tau, a)
    return (G_inf + G1 * re) + 1j * (G1 * im)


def fractional_kelvin_voigt_modulus(omega: Any, E: float, tau: float, alpha: float) -> np.ndarray:
    """Fractional Kelvin–Voigt complex modulus E(1 + (iωτ)^α): E' = E(1 + (ωτ)^α cos(πα/2)), E'' = E (ωτ)^α
    sin(πα/2)."""
    E, tau, a = _pos("E", E), _pos("tau", tau), _alpha(alpha)
    w = np.asarray(omega, float)
    if not (np.all(np.isfinite(w)) and np.all(w >= 0.0)):
        raise ValueError("omega must be finite and >= 0 (rad/s)")
    u = (w * tau) ** a
    return E * (1.0 + u * _cospi(0.5 * a)) + 1j * (E * u * _sinpi(0.5 * a))


def is_passive(modulus: Any, rtol: float = 1e-14) -> bool:
    """True if the loss modulus of the complex moduli is non-negative, G'' ≥ −rtol |G*| (passivity of e^{iωt} response:
    the dissipation per cycle π G'' γ0² cannot be negative)."""
    M = np.asarray(modulus, complex)
    return bool(np.all(M.imag >= -rtol * np.abs(M)))


# ----------------------------------------------------------------------------------------------- Prony bridge
@dataclass(frozen=True, eq=False)
class FractionalPronyFit:
    """A non-negative Prony approximation of a fractional Zener relaxation modulus over a time window.

    ``series``: the fitted :class:`PronySeries` (``inf`` ≥ 0, amplitudes ≥ 0, ``n_branches`` relaxation times
    log-spaced on ``tau_span``); ``window``: (t_lo, t_hi) [s] where the approximation is claimed; ``rel_error_time`` =
    max over 4001 log-spaced t in the window of |G_N − G|/G (independent of the fit points); ``rel_error_modulus`` =
    max over 1201 log-spaced ω in ``omega_window`` = [10/t_hi, 0.1/t_lo] rad/s of |G*_N − G*|/|G*|.
    Outside the window nothing is claimed (the exponential sum cannot follow a power law forever)."""

    series: PronySeries
    G_inf: float
    G1: float
    tau: float
    alpha: float
    window: tuple[float, float]
    tau_span: tuple[float, float]
    n_branches: int
    rel_error_time: float
    rel_error_modulus: float
    omega_window: tuple[float, float]

    def to_generalized_maxwell(self, bulk: PronySeries) -> GeneralizedMaxwell:
        """The 3-D :class:`GeneralizedMaxwell` with this shear series (the fractional kernel is a shear kernel) and the
        given ``bulk`` series (e.g. purely elastic, ``PronySeries(K, [], [])``)."""
        return GeneralizedMaxwell(self.series, bulk)


def _prony_fit_on(target_fit: np.ndarray, t_fit: np.ndarray, taus: np.ndarray, G0: float) -> tuple[float, np.ndarray]:
    A = np.concatenate([np.ones((t_fit.size, 1)), np.exp(-t_fit[:, None] / taus[None, :])], axis=1)
    w = G0 / target_fit                                       # row weights: relative error, columns in units of G0
    sol, _ = nnls(A * w[:, None], np.ones(t_fit.size), maxiter=100 * A.shape[1])
    return float(sol[0] * G0), sol[1:] * G0


def prony_error_budget(G_inf: float, G1: float, tau: float, alpha: float, n_branches: Sequence[int],
                       window: tuple[float, float] | None = None, tau_span: tuple[float, float] | None = None,
                       n_fit: int = 801, n_eval: int = 4001, n_omega: int = 1201) -> list[FractionalPronyFit]:
    """Fit a non-negative Prony series to the fractional Zener relaxation modulus for each entry of ``n_branches`` and
    report the error of each (the memory-approximation error budget).

    Method: relaxation times τ_i log-spaced on ``tau_span`` (default [t_lo, 10 t_hi]; with N−1 a power of two the
    grids of the successive N are nested), columns {1, e^{-t/τ_i}}, rows at ``n_fit`` log-spaced times of ``window``
    (default [1e-4 τ, 1e4 τ]) weighted by 1/G(t) (relative error), solved by non-negative least squares
    (``scipy.optimize.nnls``); G∞' = the coefficient of the constant column. The errors are measured on ``n_eval``
    times and ``n_omega`` frequencies that are not the fit points. The target is computed once for all N."""
    G_inf, G1, tau, a = _zener_params(G_inf, G1, tau, alpha)
    t_lo, t_hi = (1e-4 * tau, 1e4 * tau) if window is None else (float(window[0]), float(window[1]))
    if not (math.isfinite(t_lo) and math.isfinite(t_hi) and 0.0 < t_lo < t_hi):
        raise ValueError("window must be finite with 0 < t_lo < t_hi")
    s_lo, s_hi = (t_lo, 10.0 * t_hi) if tau_span is None else (float(tau_span[0]), float(tau_span[1]))
    if not (math.isfinite(s_lo) and math.isfinite(s_hi) and 0.0 < s_lo < s_hi):
        raise ValueError("tau_span must be finite with 0 < tau_lo < tau_hi")
    ns = [int(n) for n in n_branches]
    if not ns or min(ns) < 2:
        raise ValueError("n_branches: every entry must be an integer >= 2")
    if min(n_fit, n_eval, n_omega) < 2:
        raise ValueError("n_fit, n_eval and n_omega must be >= 2")
    G0 = G_inf + G1
    t_fit = np.geomspace(t_lo, t_hi, n_fit)
    t_ev = np.geomspace(t_lo, t_hi, n_eval)
    om_lo, om_hi = 10.0 / t_hi, 0.1 / t_lo
    om = np.geomspace(om_lo, om_hi, n_omega)
    G_fit = fractional_zener_relaxation(t_fit, G_inf, G1, tau, a)
    G_ev = fractional_zener_relaxation(t_ev, G_inf, G1, tau, a)
    Gstar = fractional_zener_modulus(om, G_inf, G1, tau, a)
    out: list[FractionalPronyFit] = []
    for n in ns:
        taus = np.geomspace(s_lo, s_hi, n)
        inf, amp = _prony_fit_on(G_fit, t_fit, taus, G0)
        series = PronySeries(inf, amp, taus)
        e_t = float(np.max(np.abs(series.relaxation(t_ev) - G_ev) / G_ev))
        e_w = float(np.max(np.abs(series.complex_modulus(om) - Gstar) / np.abs(Gstar)))
        out.append(FractionalPronyFit(series, G_inf, G1, tau, a, (t_lo, t_hi), (s_lo, s_hi), n, e_t, e_w,
                                      (om_lo, om_hi)))
    return out


def fit_prony_fractional_zener(G_inf: float, G1: float, tau: float, alpha: float, n_branches: int,
                               window: tuple[float, float] | None = None,
                               tau_span: tuple[float, float] | None = None) -> FractionalPronyFit:
    """One fit of :func:`prony_error_budget` with ``n_branches`` exponentials (see there for the method, the window and
    the error definitions); the result carries the measured ``rel_error_time`` and ``rel_error_modulus``."""
    return prony_error_budget(G_inf, G1, tau, alpha, [n_branches], window, tau_span)[0]
