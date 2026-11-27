"""Linear viscoelasticity: scalar reference models, Prony series, 3-D generalized Maxwell, shift factors, DMA (WP12).

Physical units throughout (SI: Pa, Pa·s, s, K, J/mol); every parameter here is a material input, never a design
variable scaled by an element modulus. The models set ``physical = True`` and need a :class:`TimeStep`
(``docs/foundations/MATERIAL_CONTRACT.md``). Derivations and the list of what is verified:
``docs/foundations/RHEOLOGY.md``;
preregistration: ``docs/foundations/prereg/V_viscoelastic.md``.

Scalar reference functions (closed forms, **reference functions** for tests and calibration; not constitutive models)
distinguish the creep compliance J(t) [1/Pa] (strain under σ0 H(t)) from the relaxation modulus E_R(t) [Pa] (stress
under
ε0 H(t)); J is not 1/E_R. t ≥ 0 is the time after the step; the value at t = 0 is the one at t = 0+.

:class:`PronySeries` M(t) = M∞ + Σ M_i e^{-t/τ_i} (shear G or bulk K). :class:`GeneralizedMaxwell` combines an
independent
shear and bulk series in 3-D small strain, σ = 2G∞ e + Σ s_i + (K∞ θ + Σ p_i) 1, with the update that is exact for a
strain that is piecewise linear in (reduced) time. Shift factors (:class:`WLFShift`, :class:`ArrheniusShift`) act as a
horizontal shift of the relaxation times (fixed amplitudes) through the reduced time ξ = ∫dt/a_T. :func:`dma_moduli`
gives the complex moduli for e^{iωt}.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from toporlogy.core.constitutive import HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse, \
    require_step, small_strain_history_evaluate

__all__ = ["R_GAS", "maxwell_relaxation", "maxwell_creep", "kelvin_voigt_creep", "kelvin_voigt_relaxation",
           "sls_relaxation", "sls_creep", "sls_creep_time", "burgers_creep", "burgers_relaxation",
           "recovery_strain", "PronySeries", "GeneralizedMaxwell", "WLFShift", "ArrheniusShift", "DMAResult",
           "complex_modulus", "dma_moduli", "storage_loss_tan", "frequency_hz_to_rad"]

R_GAS = 8.314462618                                   # J/(mol K), CODATA 2018 (exact in the 2019 SI)

_I6 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])        # Voigt symmetric identity (engineering-shear strain)
_ONE6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_IDEV6 = _I6 - np.outer(_ONE6, _ONE6) / 3.0
_REQ = MaterialRequest()


# ----------------------------------------------------------------------------------------------- input checks
def _pos(name: str, x: Any) -> float:
    v = float(x)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {x!r}")
    return v


def _time(t: Any) -> np.ndarray:
    a = np.asarray(t, float)
    if not (np.all(np.isfinite(a)) and np.all(a >= 0.0)):
        raise ValueError("t is the time after the step and must be finite and >= 0")
    return a


# ----------------------------------------------------------------------------------------------- scalar models
def maxwell_relaxation(t: Any, E: float, eta: float) -> np.ndarray:
    """Maxwell (spring E [Pa] in series with dashpot η [Pa s]) relaxation modulus E_R = E e^{-tE/η} [Pa]."""
    E, eta = _pos("E", E), _pos("eta", eta)
    return E * np.exp(-_time(t) * (E / eta))


def maxwell_creep(t: Any, E: float, eta: float) -> np.ndarray:
    """Maxwell creep compliance J = 1/E + t/η [1/Pa] (the dashpot flows without bound: a fluid)."""
    E, eta = _pos("E", E), _pos("eta", eta)
    return 1.0 / E + _time(t) / eta


def kelvin_voigt_creep(t: Any, E: float, eta: float) -> np.ndarray:
    """Kelvin–Voigt (spring E parallel to dashpot η) creep compliance J = (1 − e^{-tE/η})/E [1/Pa]; J(0) = 0."""
    E, eta = _pos("E", E), _pos("eta", eta)
    return -np.expm1(-_time(t) * (E / eta)) / E


def kelvin_voigt_relaxation(t: Any, E: float, eta: float) -> np.ndarray:
    """Refused. The Kelvin–Voigt response to an ideal strain step is the distribution E H(t) + η δ(t) (an impulse of
    viscous stress at t = 0); no finite ordinary relaxation modulus exists, and a finite curve would be wrong."""
    raise ValueError("Kelvin-Voigt has no finite relaxation modulus: the step response is E H(t) + eta delta(t) "
                     "(impulsive viscous stress); use kelvin_voigt_creep, or a standard linear solid / generalized "
                     "Maxwell model for relaxation")


def sls_relaxation(t: Any, E_inf: float, E1: float, tau: float) -> np.ndarray:
    """Standard linear solid (Maxwell branch E1, τ parallel to a spring E∞): E_R = E∞ + E1 e^{-t/τ}, E0 = E∞ + E1."""
    E_inf, E1, tau = _pos("E_inf", E_inf), _pos("E1", E1), _pos("tau", tau)
    return E_inf + E1 * np.exp(-_time(t) / tau)


def sls_creep_time(E_inf: float, E1: float, tau: float) -> float:
    """Retardation time τ_c = τ E0/E∞ of the standard linear solid."""
    E_inf, E1, tau = _pos("E_inf", E_inf), _pos("E1", E1), _pos("tau", tau)
    return tau * (E_inf + E1) / E_inf


def sls_creep(t: Any, E_inf: float, E1: float, tau: float) -> np.ndarray:
    """SLS creep compliance J = 1/E∞ − (1/E∞ − 1/E0) e^{-t/τ_c}, τ_c = τ E0/E∞; 1/E∞ − 1/E0 = E1/(E∞ E0)."""
    tc = sls_creep_time(E_inf, E1, tau)
    E0 = E_inf + E1
    return 1.0 / E_inf - (E1 / (E_inf * E0)) * np.exp(-_time(t) / tc)


def burgers_creep(t: Any, E_M: float, eta_M: float, E_K: float, eta_K: float) -> np.ndarray:
    """Burgers (Maxwell E_M, η_M in series with Kelvin–Voigt E_K, η_K) creep J = 1/E_M + t/η_M + (1 −
    e^{-tE_K/η_K})/E_K."""
    E_M, eta_M, E_K, eta_K = _pos("E_M", E_M), _pos("eta_M", eta_M), _pos("E_K", E_K), _pos("eta_K", eta_K)
    tt = _time(t)
    return 1.0 / E_M + tt / eta_M - np.expm1(-tt * (E_K / eta_K)) / E_K


def burgers_relaxation(t: Any, E_M: float, eta_M: float, E_K: float, eta_K: float) -> np.ndarray:
    """Burgers relaxation modulus: a sum of two decaying exponentials, E_R(0) = E_M, E_R(∞) = 0 (a fluid).

    Laplace: s² Ê_R Ĵ = 1 with Ĵ from :func:`burgers_creep` gives Ê_R(s) = (1 + τ_K s)/(a s² + b s + c), τ_K = η_K/E_K,
    a = τ_K/E_M, b = 1/E_M + τ_K/η_M + 1/E_K, c = 1/η_M; the roots s_{1,2} are real and negative for positive
    parameters, E_R(t) = Σ_j (1 + τ_K s_j)/(a (s_j − s_k)) e^{s_j t}. A repeated root (D = 0) is refused."""
    E_M, eta_M, E_K, eta_K = _pos("E_M", E_M), _pos("eta_M", eta_M), _pos("E_K", E_K), _pos("eta_K", eta_K)
    tt = _time(t)
    tK = eta_K / E_K
    a, b, c = tK / E_M, 1.0 / E_M + tK / eta_M + 1.0 / E_K, 1.0 / eta_M
    D = b * b - 4.0 * a * c
    if not D > 0.0:
        raise ValueError("Burgers relaxation: repeated relaxation time (discriminant 0) is not implemented")
    q = -0.5 * (b + math.sqrt(D))                                   # cancellation-free roots of a s² + b s + c
    s1, s2 = q / a, c / q
    c1 = (1.0 + tK * s1) / (a * (s1 - s2))
    c2 = (1.0 + tK * s2) / (a * (s2 - s1))
    return c1 * np.exp(s1 * tt) + c2 * np.exp(s2 * tt)


def recovery_strain(creep: Any, t: Any, sigma0: float, t_unload: float) -> np.ndarray:
    """Strain of a creep-and-recovery test by linear superposition: ε(t) = σ0 [J(t) − J(t − t_u) H(t − t_u)]; ``creep``
    is a callable J(t) (e.g. ``lambda t: sls_creep(t, ...)``)."""
    tt = _time(t)
    tu = float(t_unload)
    if not (math.isfinite(tu) and tu >= 0.0):
        raise ValueError("t_unload must be finite and >= 0")
    after = tt >= tu
    out = np.asarray(creep(tt), float).copy()
    out = np.where(after, out - np.asarray(creep(np.where(after, tt - tu, 0.0)), float), out)
    return float(sigma0) * out


# ----------------------------------------------------------------------------------------------- Prony series
@dataclass(frozen=True, eq=False)
class PronySeries:
    """M(t) = ``inf`` + Σ ``amplitudes``_i e^{-t/``taus``_i} [Pa] (a shear or a bulk relaxation modulus).

    ``inf`` ≥ 0 (0: a fluid in that mode, flagged by :attr:`is_fluid`; the quasistatic stiffness is then singular and
    is not
    hidden by a numerical stiffness), amplitudes ≥ 0 (zero allowed), τ > 0 [s], instantaneous modulus
    M0 = inf + Σ amplitudes > 0. All values finite; invalid input raises."""

    inf: float
    amplitudes: np.ndarray
    taus: np.ndarray

    def __post_init__(self) -> None:
        inf = float(self.inf)
        A = np.array(self.amplitudes, float).reshape(-1) if np.ndim(self.amplitudes) <= 1 else None
        tau = np.array(self.taus, float).reshape(-1) if np.ndim(self.taus) <= 1 else None
        if A is None or tau is None:
            raise ValueError("PronySeries: amplitudes and taus must be 1-D")
        if A.shape != tau.shape:
            raise ValueError(f"PronySeries: amplitudes {A.shape} and taus {tau.shape} differ in length")
        if not (math.isfinite(inf) and inf >= 0.0):
            raise ValueError(f"PronySeries: equilibrium modulus must be finite and >= 0, got {self.inf!r}")
        if not (np.all(np.isfinite(A)) and np.all(A >= 0.0)):
            raise ValueError("PronySeries: branch amplitudes must be finite and >= 0 (no negative branch)")
        if not (np.all(np.isfinite(tau)) and np.all(tau > 0.0)):
            raise ValueError("PronySeries: relaxation times must be finite and > 0")
        if not inf + float(A.sum()) > 0.0:
            raise ValueError("PronySeries: the instantaneous modulus (inf + sum of amplitudes) must be > 0")
        A.setflags(write=False)
        tau.setflags(write=False)
        object.__setattr__(self, "inf", inf)
        object.__setattr__(self, "amplitudes", A)
        object.__setattr__(self, "taus", tau)

    @classmethod
    def from_weights(cls, instantaneous: float, weights: Any, taus: Any) -> PronySeries:
        """Normalised convention: M0 = ``instantaneous`` [Pa], branch weights g_i = M_i/M0 with Σ g_i ≤ 1, equilibrium
        M∞ = M0 (1 − Σ g_i). Σ g_i > 1 is refused (never mixed with the physical-modulus convention)."""
        M0 = _pos("instantaneous", instantaneous)
        g = np.array(weights, float).reshape(-1)
        if not (np.all(np.isfinite(g)) and np.all(g >= 0.0) and float(g.sum()) <= 1.0):
            raise ValueError("PronySeries.from_weights: weights must be finite, >= 0 and sum to at most 1")
        return cls(M0 * (1.0 - float(g.sum())), M0 * g, np.array(taus, float).reshape(-1))

    @property
    def instantaneous(self) -> float:
        return self.inf + float(self.amplitudes.sum())

    @property
    def is_fluid(self) -> bool:
        """True when the equilibrium modulus is 0 (the mode flows under a constant stress)."""
        return self.inf == 0.0

    @property
    def viscosity(self) -> float:
        """∫₀^∞ (M(t) − M∞) dt = Σ M_i τ_i [Pa s]; the steady-flow viscosity when ``is_fluid``."""
        return float(np.dot(self.amplitudes, self.taus))

    def relaxation(self, t: Any) -> np.ndarray:
        """M(t) [Pa] at t ≥ 0 (array)."""
        tt = _time(t)
        return self.inf + (self.amplitudes * np.exp(-tt[..., None] / self.taus)).sum(-1)

    def complex_modulus(self, omega: Any) -> np.ndarray:
        """M*(ω) = M∞ + Σ M_i iωτ_i/(1 + iωτ_i) (convention e^{iωt}, ω in rad/s)."""
        return complex_modulus(self, omega)


# ----------------------------------------------------------------------------------------------- shift factors
_GL5_XR, _GL5_WR = np.polynomial.legendre.leggauss(5)
_GL5_X: np.ndarray = 0.5 * (_GL5_XR + 1.0)             # nodes on [0, 1]
_GL5_W: np.ndarray = 0.5 * _GL5_WR
_MAX_DLN_PER_SUB = 0.5                                 # |Δ ln a_T| per quadrature sub-interval
_MAX_SUB = 200000


class _Shift:
    """Base of the time–temperature shift factors: a_T = τ(T)/τ(T_ref); ξ(t) = ∫ dt/a_T(T(t))."""

    T_ref: float
    T_min: float | None
    T_max: float | None

    def _check_range(self, T: np.ndarray) -> None:
        if not (np.all(np.isfinite(T)) and np.all(T > 0.0)):
            raise ValueError("shift factor: temperatures are absolute (kelvin) and must be finite and > 0")
        if self.T_min is not None and np.any(self.T_min > T) or self.T_max is not None and np.any(self.T_max < T):
            raise ValueError(f"shift factor: temperature outside the calibrated range [{self.T_min}, {self.T_max}] K")

    def ln_aT(self, T: Any) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def aT(self, T: Any) -> np.ndarray:
        """a_T(T) = exp(ln a_T); may overflow to inf / underflow to 0 for extreme T (use :meth:`ln_aT` there)."""
        with np.errstate(over="ignore"):
            return np.exp(self.ln_aT(T))

    def reduced_time_increment(self, T_prev: Any, T_next: Any, dt: Any) -> np.ndarray:
        """Δξ = ∫₀^Δt ds/a_T(T(s)) for T varying **linearly** in the step from ``T_prev`` to ``T_next`` (kelvin,
        scalars or
        arrays that broadcast), Δt = ``dt`` s. Isothermal steps (T_prev == T_next): exactly Δt exp(−ln a_T). Otherwise
        Gauss–Legendre with 5 points on equal sub-intervals chosen so that |Δ ln a_T| ≤ 0.5 on each (error ≪ 1e-12
        relative for this smooth integrand); ln a_T is monotone in T, so the end values bound the variation. Refuses a
        non-finite result (a_T so small that ξ overflows)."""
        T0 = np.asarray(T_prev, float)
        T1 = np.asarray(T_next, float)
        dt_a = np.asarray(dt, float)
        if not (np.all(np.isfinite(dt_a)) and np.all(dt_a >= 0.0)):
            raise ValueError("reduced time: dt must be finite and >= 0")
        l0, l1 = self.ln_aT(T0), self.ln_aT(T1)
        shape = np.broadcast(T0, T1, dt_a).shape
        l0, l1 = np.broadcast_to(l0, shape), np.broadcast_to(l1, shape)
        dtb = np.broadcast_to(dt_a, shape)
        nsub = max(1, int(math.ceil(float(np.max(np.abs(l1 - l0), initial=0.0)) / _MAX_DLN_PER_SUB)))
        if nsub > _MAX_SUB:
            raise ValueError("reduced time: ln a_T varies by more than 1e5 in one step; refine the step")
        T0b, T1b = np.broadcast_to(T0, shape), np.broadcast_to(T1, shape)
        acc = np.zeros(shape)
        with np.errstate(over="ignore"):
            for k in range(nsub):
                x = (k + _GL5_X) / nsub
                for xi, wi in zip(x, _GL5_W, strict=True):
                    acc = acc + wi * np.exp(-self.ln_aT(T0b + (T1b - T0b) * xi))
            acc = acc * dtb / nsub
            iso = np.exp(-l0) * dtb
        out = np.where(T0b == T1b, iso, acc)
        if not np.all(np.isfinite(out)):
            raise ValueError("reduced time increment overflows: a_T is too small at this temperature (ln a_T = "
                             f"{float(np.min(np.minimum(l0, l1))):.6g}); the model is outside its range")
        return out


@dataclass(frozen=True)
class WLFShift(_Shift):
    """Williams–Landel–Ferry: log10 a_T = −C1 (T − T_ref)/(C2 + T − T_ref); T in kelvin, C2 in K, C1 > 0, C2 > 0.
    A temperature with C2 + T − T_ref ≤ 0 (the singularity and below) is refused; optional calibrated range
    [``T_min``, ``T_max``]."""

    T_ref: float
    C1: float
    C2: float
    T_min: float | None = None
    T_max: float | None = None

    def __post_init__(self) -> None:
        for k in ("T_ref", "C1", "C2"):
            object.__setattr__(self, k, _pos(k, getattr(self, k)))
        _check_range_args(self)

    def ln_aT(self, T: Any) -> np.ndarray:
        T = np.asarray(T, float)
        self._check_range(T)
        dT = T - self.T_ref
        den = self.C2 + dT
        if np.any(den <= 0.0):
            raise ValueError(f"WLF: C2 + T - T_ref <= 0 (T = {float(np.min(T)):.6g} K, singular at "
                             f"{self.T_ref - self.C2:.6g} K)")
        return -math.log(10.0) * self.C1 * dT / den


@dataclass(frozen=True)
class ArrheniusShift(_Shift):
    """Arrhenius: ln a_T = (E_a/R)(1/T − 1/T_ref), T in kelvin, E_a > 0 in J/mol, R = 8.314462618 J/(mol K)."""

    T_ref: float
    Ea: float
    T_min: float | None = None
    T_max: float | None = None

    def __post_init__(self) -> None:
        for k in ("T_ref", "Ea"):
            object.__setattr__(self, k, _pos(k, getattr(self, k)))
        _check_range_args(self)

    def ln_aT(self, T: Any) -> np.ndarray:
        T = np.asarray(T, float)
        self._check_range(T)
        return (self.Ea / R_GAS) * (self.T_ref - T) / (T * self.T_ref)


def _check_range_args(s: Any) -> None:
    for k in ("T_min", "T_max"):
        v = getattr(s, k)
        if v is not None:
            object.__setattr__(s, k, _pos(k, v))
    if s.T_min is not None and s.T_max is not None and not s.T_min < s.T_max:
        raise ValueError("T_min must be < T_max")


# ----------------------------------------------------------------------------------------------- stable kernels
def _phi(a: np.ndarray) -> np.ndarray:
    """φ(a) = (1 − e^{-a})/a with φ(0) = 1."""
    a = np.asarray(a, float)
    safe = np.where(a > 0.0, a, 1.0)
    return np.where(a > 0.0, -np.expm1(-safe) / safe, 1.0)


def _g3_series() -> np.ndarray:
    """Taylor coefficients d_j (j ≥ 1) of g3(a) = (a − q − q²/2)/a², q = 1 − e^{-a}, as exact fractions → floats."""
    K = 30
    fact = [1] * (K + 3)
    for k in range(1, K + 3):
        fact[k] = fact[k - 1] * k
    q = [Fraction(0)] + [Fraction((-1) ** (k + 1), fact[k]) for k in range(1, K + 3)]
    N = []
    for k in range(K + 3):
        q2 = sum((q[i] * q[k - i] for i in range(1, k)), Fraction(0))
        N.append((Fraction(1) if k == 1 else Fraction(0)) - q[k] - q2 / 2)
    assert N[1] == 0 and N[2] == 0
    return np.array([float(N[k + 2]) for k in range(1, K + 1)])       # d_j = N_{j+2}, j = 1..K


_G3_D = _g3_series()


def _g3(a: np.ndarray) -> np.ndarray:
    """g3(a) = (a − q − q²/2)/a² = a ∫₀¹ ((1 − e^{-ax})/a)² dx with q = −expm1(−a) (≈ a/3 for small a); Taylor series
    for
    a < 1/4 (the direct form cancels as a³), closed form otherwise. g3(0) = 0."""
    a = np.asarray(a, float)
    out = np.zeros_like(a)
    small = a < 0.25
    x = a[small]
    acc = np.zeros_like(x)
    for d in _G3_D[::-1]:
        acc = acc * x + d
    out[small] = acc * x
    big = ~small
    ab = a[big]
    q = -np.expm1(-ab)
    out[big] = (ab - q - 0.5 * q * q) / (ab * ab)
    return out


# ----------------------------------------------------------------------------------------------- generalized Maxwell
class GeneralizedMaxwell:
    """3-D small-strain generalized Maxwell (Prony) model with independent shear and bulk series.

    σ = 2G∞ e + Σ_i s_i + (K∞ θ + Σ_j p_j) 1, e = dev ε, θ = tr ε; ṡ_i + s_i/τ_i = 2G_i ė, ṗ_j + p_j/τ_j = K_j θ̇.
    Update over a step of reduced time Δξ (= Δt without ``shift``) for strain linear in the step, exact for that strain:
    a = Δξ/τ, r = e^{-a}, φ = (1 − e^{-a})/a:  s ← r s + 2G_i φ Δe,  p ← r p + K_j φ Δθ. Algorithmic tangent
    C = (K∞ + ΣK_jφ_j) 1⊗1 + 2(G∞ + ΣG_iφ_i) I_dev in the Voigt convention of :mod:`toporlogy.core.constitutive`
    (= 2(G∞+ΣG_iφ_i) P_dev + 3(K∞+ΣK_jφ_j) P_vol); ``tangent_kind="continuum"`` returns the instantaneous tangent (φ =
    1).
    Stored energy ψ = G∞ e:e + ½K∞θ² + Σ s_i:s_i/(4G_i) + Σ p_j²/(2K_j) (branches with zero amplitude contribute 0);
    dissipation increment: the exact integral over the step of Σ s_i:s_i/(2G_iτ_i) + Σ p_j²/(K_jτ_j) along the exact
    branch
    stresses of the linear strain (closed form, RHEOLOGY.md); ≥ 0 and computed without differencing work and energy.

    ``shift`` (:class:`WLFShift` or :class:`ArrheniusShift`, None = isothermal) replaces Δt by the reduced increment Δξ
    of the
    step with T linear between ``request.step.temperature_prev`` and ``temperature`` (horizontal shift, fixed
    amplitudes;
    the strain is taken linear in ξ within the step, exact for constant T in the step). State: ``strain_prev`` (n, 3,
    3),
    ``s`` (n, nb_shear, 3, 3), ``p`` (n, nb_bulk). The initial state is a body at rest with zero strain at t = 0. Needs
    the
    time step (and the temperature with a shift). Diagnostics: ``reduced_time_increment`` (n,)."""

    name = "generalized_maxwell"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields = ("strain_prev", "s", "p")

    def __init__(self, shear: PronySeries, bulk: PronySeries, shift: _Shift | None = None) -> None:
        if not (isinstance(shear, PronySeries) and isinstance(bulk, PronySeries)):
            raise ValueError("GeneralizedMaxwell: shear and bulk must be PronySeries")
        if shift is not None and not isinstance(shift, _Shift):
            raise ValueError("GeneralizedMaxwell: shift must be a WLFShift, an ArrheniusShift or None")
        self.shear, self.bulk, self.shift = shear, bulk, shift

    # ---------------------------------------------------------------- properties
    @property
    def is_fluid(self) -> bool:
        """True if either equilibrium modulus is 0 (no solid long-time stiffness in that mode)."""
        return self.shear.is_fluid or self.bulk.is_fluid

    def equilibrium_moduli(self) -> tuple[float, float]:
        """(K∞, G∞): the long-time (quasistatic) moduli; G∞ = 0 or K∞ = 0 means the long-time stiffness is singular."""
        return self.bulk.inf, self.shear.inf

    def instantaneous_moduli(self) -> tuple[float, float]:
        """(K0, G0)."""
        return self.bulk.instantaneous, self.shear.instantaneous

    def instantaneous_tangent(self) -> np.ndarray:
        """6×6 instantaneous (dt → 0) tangent, K0 1⊗1 + 2 G0 I_dev."""
        K0, G0 = self.instantaneous_moduli()
        return K0 * np.outer(_ONE6, _ONE6) + 2.0 * G0 * _IDEV6

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"strain_prev": np.zeros((n, 3, 3)), "s": np.zeros((n, len(self.shear.taus), 3, 3)),
                "p": np.zeros((n, len(self.bulk.taus)))}

    # ---------------------------------------------------------------- update
    def _dxi(self, step: Any, n: int) -> np.ndarray:
        if self.shift is None:
            return np.full(n, step.dt)
        return np.array(np.broadcast_to(self.shift.reduced_time_increment(step.temperature_prev, step.temperature,
                                                                          step.dt), (n,)))

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray], request: MaterialRequest = _REQ
               ) -> HistoryUpdate:
        """Stress (n, 3, 3), algorithmic tangent (n, 6, 6), trial state, stored energy and dissipation increment at the
        3-D
        strains ``eps`` (n, 3, 3) for the step ``request.step``; ``committed`` is not modified."""
        if request.tangent_kind not in ("algorithmic", "continuum"):
            raise ValueError("tangent_kind: algorithmic | continuum")
        step = require_step(self, request, temperature=self.shift is not None)
        eps = np.asarray(eps, float)
        n = eps.shape[0]
        dxi = self._dxi(step, n)
        eye = np.eye(3)
        tr = np.trace(eps, axis1=1, axis2=2)
        dev = eps - (tr / 3.0)[:, None, None] * eye
        ep = np.asarray(committed["strain_prev"])
        trp = np.trace(ep, axis1=1, axis2=2)
        de = dev - (ep - (trp / 3.0)[:, None, None] * eye)
        dth = tr - trp
        s0, p0 = np.asarray(committed["s"]), np.asarray(committed["p"])
        Gi, ts = self.shear.amplitudes, self.shear.taus
        Kj, tk = self.bulk.amplitudes, self.bulk.taus

        a_s = dxi[:, None] / ts[None, :]
        a_k = dxi[:, None] / tk[None, :]
        r_s, r_k = np.exp(-a_s), np.exp(-a_k)
        phi_s, phi_k = _phi(a_s), _phi(a_k)
        s1 = r_s[..., None, None] * s0 + (2.0 * Gi[None, :] * phi_s)[..., None, None] * de[:, None]
        p1 = r_k * p0 + (Kj[None, :] * phi_k) * dth[:, None]
        sig = (2.0 * self.shear.inf) * dev + s1.sum(1) + (self.bulk.inf * tr + p1.sum(1))[:, None, None] * eye

        C = None
        if request.tangent:
            if request.tangent_kind == "continuum":
                C = np.broadcast_to(self.instantaneous_tangent(), (n, 6, 6)).copy()
            else:
                Gt = self.shear.inf + phi_s @ Gi
                Kt = self.bulk.inf + phi_k @ Kj
                C = Kt[:, None, None] * np.outer(_ONE6, _ONE6)[None] + 2.0 * Gt[:, None, None] * _IDEV6[None]

        inv4G = np.where(Gi > 0.0, 0.25 / np.where(Gi > 0.0, Gi, 1.0), 0.0)
        inv2K = np.where(Kj > 0.0, 0.5 / np.where(Kj > 0.0, Kj, 1.0), 0.0)
        psi = (self.shear.inf * np.einsum("nij,nij->n", dev, dev) + 0.5 * self.bulk.inf * tr ** 2
               + np.einsum("nbij,nbij,b->n", s1, s1, inv4G) + np.einsum("nb,nb,b->n", p1, p1, inv2K)) \
            if request.energy else None

        # exact dissipation increment over the step: D = Σ 2G_i [½(1−r²) m:m + a φ² m:Δe + g3 Δe:Δe],
        # m = s_n/(2G_i); bulk likewise with K_j, scalars (module notes; RHEOLOGY.md)
        m = s0 * np.where(Gi > 0.0, 0.5 / np.where(Gi > 0.0, Gi, 1.0), 0.0)[None, :, None, None]
        mm = np.einsum("nbij,nbij->nb", m, m)
        md = np.einsum("nbij,nij->nb", m, de)
        dd = np.einsum("nij,nij->n", de, de)[:, None]
        qs = -np.expm1(-a_s)
        Ds = (2.0 * Gi[None, :] * (0.5 * qs * (2.0 - qs) * mm + a_s * phi_s ** 2 * md + _g3(a_s) * dd)).sum(1)
        mk = p0 * np.where(Kj > 0.0, 1.0 / np.where(Kj > 0.0, Kj, 1.0), 0.0)[None, :]
        qk = -np.expm1(-a_k)
        Dk = (Kj[None, :] * (0.5 * qk * (2.0 - qk) * mk ** 2 + a_k * phi_k ** 2 * mk * dth[:, None]
                             + _g3(a_k) * dth[:, None] ** 2)).sum(1)
        trial = {"strain_prev": eps.copy(), "s": s1, "p": p1}
        return HistoryUpdate(sig, C, trial, {"reduced_time_increment": dxi}, psi, Ds + Dk)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _REQ) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)


# ----------------------------------------------------------------------------------------------- DMA
def frequency_hz_to_rad(f_hz: Any) -> np.ndarray:
    """ω = 2π f [rad/s] from f in Hz. The functions of this module take ω in rad/s."""
    return 2.0 * math.pi * np.asarray(f_hz, float)


def _maxwell_terms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x²/(1+x²), x/(1+x²)) for x = ωτ ≥ 0, overflow-free for large x."""
    small = x <= 1.0
    xs = np.where(small, x, 1.0)
    xb = np.where(small, 1.0, x)
    inv = 1.0 / xb
    re = np.where(small, xs * xs / (1.0 + xs * xs), 1.0 / (1.0 + inv * inv))
    im = np.where(small, xs / (1.0 + xs * xs), inv / (1.0 + inv * inv))
    return re, im


def complex_modulus(series: PronySeries, omega: Any) -> np.ndarray:
    """M*(ω) = M' + i M'' of a Prony series, M' = M∞ + Σ M_i (ωτ_i)²/(1 + (ωτ_i)²), M'' = Σ M_i ωτ_i/(1 + (ωτ_i)²) ≥ 0
    (convention e^{iωt}: ε = ε0 e^{iωt} gives σ = M* ε; ω ≥ 0 in rad/s)."""
    w = np.asarray(omega, float)
    if not (np.all(np.isfinite(w)) and np.all(w >= 0.0)):
        raise ValueError("omega must be finite and >= 0 (rad/s)")
    re, im = _maxwell_terms(w[..., None] * series.taus)
    return (series.inf + (series.amplitudes * re).sum(-1)) + 1j * (series.amplitudes * im).sum(-1)


def storage_loss_tan(M: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(storage M', loss M'', tan δ = M''/M') of complex moduli; tan δ is NaN where M' ≤ 0."""
    M = np.asarray(M, complex)
    st, lo = M.real, M.imag
    with np.errstate(divide="ignore", invalid="ignore"):
        td = np.where(st > 0.0, lo / np.where(st > 0.0, st, 1.0), np.nan)
    return st, lo, td


@dataclass(frozen=True)
class DMAResult:
    """Complex moduli at the angular frequencies ``omega`` [rad/s] (e^{iωt}): shear ``G``, bulk ``K``, Young ``E`` =
    9K*G*/(3K*+G*) and Poisson ``nu`` = (3K*−2G*)/(2(3K*+G*)) of the isotropic operational response (NaN where
    3K*+G* = 0). Not the pointwise time functions of the relaxation spectra."""

    omega: np.ndarray
    G: np.ndarray
    K: np.ndarray
    E: np.ndarray
    nu: np.ndarray


def dma_moduli(shear: PronySeries, bulk: PronySeries, omega: Any) -> DMAResult:
    """Complex shear, bulk, Young and Poisson moduli of an isotropic generalized Maxwell material at ω [rad/s]."""
    w = np.asarray(omega, float)
    G, K = complex_modulus(shear, w), complex_modulus(bulk, w)
    den = 3.0 * K + G
    with np.errstate(divide="ignore", invalid="ignore"):
        E = 9.0 * K * G / den
        nu = (3.0 * K - 2.0 * G) / (2.0 * den)
    return DMAResult(w, G, K, E, nu)
