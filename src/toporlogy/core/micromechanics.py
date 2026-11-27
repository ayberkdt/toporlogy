"""Constituent-to-composite micromechanics of isotropic phases: bounds, spherical-inclusion mean-field schemes,
fraction conversion and admissibility of scalar interpolations (WP18; derivation and checks in
``docs/foundations/MICROMECHANICS.md`` and ``docs/foundations/prereg/MM_micromechanics.md``).

Phases and units
----------------
A phase is isotropic with bulk modulus K and shear modulus G (any consistent units): 3-D K, or in 2-D the plane bulk
modulus κ = λ + μ (plane stress E / (2(1 − ν))) with G = μ. A void is K = G = 0; every other phase has K > 0 and G > 0
(:class:`IsoPhase`). Fractions are volume fractions in [0, 1] summing to 1 (:func:`check_fractions`). Tensors use the
Voigt order [xx, yy, zz, yz, xz, xy] (stress physical shear, strain engineering shear) at the interface; the algebra is
done in Mandel form (:mod:`toporlogy.core.tensor`).

Bounds
------
:func:`voigt` (uniform strain) and :func:`reuss` (uniform stress) hold for every microstructure. :func:`hs_bounds` gives
the Hashin–Shtrikman–Walpole bounds in the shift form

    K = [Σ f_i / (K_i + K*)]⁻¹ − K*,   G = [Σ f_i / (G_i + ζ)]⁻¹ − ζ,

with K* = 4 G0 / 3, ζ = G0 (9 K0 + 8 G0) / (6 (K0 + 2 G0)) (3-D; 2-D: K* = G0, ζ = G0 K0 / (K0 + 2 G0)). The upper
bound takes (K0, G0) = (largest K, largest G among the phases present), the lower bound the smallest values; the two
need not belong to the same phase (crossing). Valid for a statistically isotropic composite of linear elastic isotropic
phases; **no claim for anisotropic composites** (layered cells etc.: use Voigt/Reuss). For a crossed pair the bulk
bounds are attained by coated-sphere assemblages (checked in the tests); the shear bound was not checked against an
attaining microstructure. Ordering K_R ≤ K_HS− ≤ K_HS+ ≤ K_V (same for G) holds for every input.

Mean-field schemes
------------------
:func:`dilute_sphere` and :func:`mori_tanaka_sphere` treat spherical isotropic inclusions in an isotropic matrix with
the Eshelby sphere (:func:`eshelby_sphere`), full 6 × 6 matrix solves. They are mean-field approximations, not exact
effective properties of any given microstructure; Mori–Tanaka with the stiffer (both K and G) matrix equals the HS upper
bound, with the softer matrix the HS lower bound.

Interpolation
-------------
:func:`interpolate_moduli` interpolates K and G of two isotropic solids independently (positive, with derivatives).
:func:`interpolation_admissibility` and :func:`required_interpolation_parameters` decide when a scalar SIMP / RAMP law
r(x) applied to a solid/void mixture with fixed ν stays under the HS upper bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np

from toporlogy.core import tensor as T
from toporlogy.core.linear_elastic import isotropic_moduli, plane_bulk_modulus
from toporlogy.references import cite

__all__ = ["IsoPhase", "VOID", "HSBounds", "EffectiveIsotropic", "InterpolatedModuli", "InterpolationCheck",
           "check_fractions", "voigt", "reuss", "hs_bounds", "eshelby_sphere", "dilute_sphere", "mori_tanaka_sphere",
           "volume_to_mass_fractions", "mass_to_volume_fractions", "mixture_density", "interpolate_moduli",
           "hs_void_ramp_parameters", "required_interpolation_parameters", "interpolation_admissibility"]

#: tolerance on |Σ f_i − 1|
FRACTION_SUM_TOL = 1e-9


# ============================================================================================== phases, fractions
@dataclass(frozen=True)
class IsoPhase:
    """Isotropic phase: bulk modulus ``K`` and shear modulus ``G`` (both > 0), or the void (both 0)."""

    K: float
    G: float

    def __post_init__(self) -> None:
        K, G = float(self.K), float(self.G)
        if not (math.isfinite(K) and math.isfinite(G)):
            raise ValueError(f"phase moduli must be finite, got K = {K}, G = {G}")
        if K < 0 or G < 0 or ((K == 0) != (G == 0)):
            raise ValueError(f"a phase needs K > 0 and G > 0, or K = G = 0 (void); got K = {K:.6g}, G = {G:.6g}")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "G", G)

    @classmethod
    def from_E_nu(cls, E: float, nu: float) -> IsoPhase:
        """3-D phase from Young's modulus and Poisson's ratio (:func:`isotropic_moduli` rules)."""
        m = isotropic_moduli(E=E, nu=nu)
        return cls(m.K, m.G)

    @property
    def is_void(self) -> bool:
        return self.K == 0.0

    @property
    def nu(self) -> float:
        """3-D Poisson ratio (not defined for a void)."""
        if self.is_void:
            raise ValueError("a void has no Poisson ratio")
        return (3 * self.K - 2 * self.G) / (2 * (3 * self.K + self.G))


#: the void phase (K = G = 0)
VOID = IsoPhase(0.0, 0.0)


def _moduli(phases: Sequence[IsoPhase]) -> tuple[np.ndarray, np.ndarray]:
    if len(phases) < 1 or not all(isinstance(p, IsoPhase) for p in phases):
        raise ValueError("phases must be a non-empty sequence of IsoPhase")
    return np.array([p.K for p in phases]), np.array([p.G for p in phases])


def check_fractions(f: Any, n: int | None = None) -> np.ndarray:
    """Volume fractions as a float array (last axis = phases): finite, each in [0, 1] (no clipping), Σ = 1 within
    :data:`FRACTION_SUM_TOL`; ``n`` the expected number of phases. Raises ValueError otherwise."""
    f = np.asarray(f, float)
    if f.ndim < 1 or f.shape[-1] < 1:
        raise ValueError("fractions need a last axis with one entry per phase")
    if n is not None and f.shape[-1] != n:
        raise ValueError(f"expected {n} fractions, got {f.shape[-1]}")
    if not np.all(np.isfinite(f)):
        raise ValueError("fractions must be finite")
    if np.any(f < 0) or np.any(f > 1):
        raise ValueError("every fraction must lie in [0, 1]")
    if np.any(np.abs(f.sum(axis=-1) - 1.0) > FRACTION_SUM_TOL):
        raise ValueError(f"fractions must sum to 1 (tolerance {FRACTION_SUM_TOL:g})")
    return f


def _out(x: np.ndarray) -> Any:
    return float(x) if np.ndim(x) == 0 else x


# ============================================================================================== Voigt, Reuss, HS
@cite("voigt_1889", note="uniform-strain average; text not read, standard definition")
def voigt(f: Any, phases: Sequence[IsoPhase]) -> tuple[Any, Any]:
    """Voigt (uniform strain) bulk and shear moduli Σ f_i K_i, Σ f_i G_i; ``f`` (..., n_phases)."""
    K, G = _moduli(phases)
    f = check_fractions(f, len(phases))
    return _out(f @ K), _out(f @ G)


def _shifted_harmonic(a: np.ndarray, f: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """[Σ f_i / (a_i + shift)]⁻¹ − shift over the phases with f_i > 0; a term with a_i + shift = 0 (void, zero shift)
    makes the result 0 (the limit)."""
    present = f > 0
    denom = a + shift[..., None]
    zero = present & (denom <= 0)
    term = np.where(present & ~zero, f / np.where(denom > 0, denom, 1.0), 0.0)
    S = term.sum(axis=-1)
    return np.where(zero.any(axis=-1), 0.0, 1.0 / np.where(S > 0, S, 1.0)) - shift + 0.0


@cite("reuss_1929", note="uniform-stress average; text not read, standard definition")
def reuss(f: Any, phases: Sequence[IsoPhase]) -> tuple[Any, Any]:
    """Reuss (uniform stress) bulk and shear moduli [Σ f_i / K_i]⁻¹, [Σ f_i / G_i]⁻¹; 0 if a void has f > 0."""
    K, G = _moduli(phases)
    f = check_fractions(f, len(phases))
    zero = np.zeros(f.shape[:-1])
    return _out(_shifted_harmonic(K, f, zero)), _out(_shifted_harmonic(G, f, zero))


def _shifts(K0: np.ndarray, G0: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(K*, ζ) of the comparison medium (K0, G0); both 0 for the void."""
    if dim == 3:
        Ks = 4.0 * G0 / 3.0
        den = 6.0 * (K0 + 2.0 * G0)
        num = G0 * (9.0 * K0 + 8.0 * G0)
    elif dim == 2:
        Ks = G0
        den = K0 + 2.0 * G0
        num = G0 * K0
    else:
        raise ValueError("dim must be 2 or 3")
    return Ks, np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


@dataclass(frozen=True)
class HSBounds:
    """Hashin–Shtrikman–Walpole bounds (floats for one fraction vector, arrays for a batch)."""

    K_lower: Any
    K_upper: Any
    G_lower: Any
    G_upper: Any


@cite("hashin_shtrikman_1963", note="shift (Walpole) form derived in docs/foundations/prereg/MM_micromechanics.md; "
                                    "text not read, bulk bound checked against coated spheres in the tests")
def hs_bounds(f: Any, phases: Sequence[IsoPhase], dim: int = 3) -> HSBounds:
    """HS bounds of a statistically isotropic composite of isotropic ``phases`` with volume fractions ``f``
    (..., n_phases); ``dim`` 3 (K) or 2 (κ, the plane bulk modulus). The comparison medium of the upper (lower)
    bound has the largest (smallest) K and the largest (smallest) G among the phases with f > 0, which may belong to
    different phases. A void phase with f > 0 gives lower bounds 0. Not valid for anisotropic composites."""
    K, G = _moduli(phases)
    f = check_fractions(f, len(phases))
    present = f > 0
    out = {}
    for name, pick, fill in (("upper", np.max, -np.inf), ("lower", np.min, np.inf)):
        K0 = pick(np.where(present, K, fill), axis=-1)
        G0 = pick(np.where(present, G, fill), axis=-1)
        Ks, zeta = _shifts(K0, G0, dim)
        out[name] = (_shifted_harmonic(K, f, Ks), _shifted_harmonic(G, f, zeta))
    return HSBounds(_out(out["lower"][0]), _out(out["upper"][0]), _out(out["lower"][1]), _out(out["upper"][1]))


# ============================================================================================== Eshelby sphere, MT
def _projectors() -> tuple[np.ndarray, np.ndarray]:
    PV = np.zeros((6, 6))
    PV[:3, :3] = 1.0 / 3.0
    return PV, np.eye(6) - PV


def eshelby_sphere(nu: float) -> np.ndarray:
    """Eshelby tensor of a sphere in an isotropic matrix of Poisson ratio ``nu`` as a Mandel 6 × 6 matrix
    S = α P_V + β P_D, α = (1 + ν) / (3 (1 − ν)), β = 2 (4 − 5ν) / (15 (1 − ν)) (strain of the inclusion = S ×
    eigenstrain)."""
    nu = float(nu)
    if not (math.isfinite(nu) and -1.0 < nu < 0.5):
        raise ValueError("-1 < nu < 0.5 is required")
    PV, PD = _projectors()
    return (1 + nu) / (3 * (1 - nu)) * PV + 2 * (4 - 5 * nu) / (15 * (1 - nu)) * PD


def _mandel_stiffness(p: IsoPhase) -> np.ndarray:
    PV, PD = _projectors()
    return 3.0 * p.K * PV + 2.0 * p.G * PD


@dataclass(frozen=True)
class EffectiveIsotropic:
    """Effective moduli of a mean-field scheme: ``K``, ``G``, ``E``, ``nu``, the stiffness ``C`` (Voigt 6 × 6) and
    ``isotropy_error`` = ‖C − isotropic part‖ / ‖C‖ in Mandel form (rounding level for isotropic phases)."""

    K: float
    G: float
    E: float
    nu: float
    C: np.ndarray
    isotropy_error: float
    method: str


def _inclusion_scheme(f: float, matrix: IsoPhase, inclusion: IsoPhase, scheme: str) -> EffectiveIsotropic:
    f = float(check_fractions([1.0 - float(f), float(f)])[1])
    if matrix.is_void:
        raise ValueError("the matrix must not be a void")
    Cm, Ci = _mandel_stiffness(matrix), _mandel_stiffness(inclusion)
    I6 = np.eye(6)
    S = eshelby_sphere(matrix.nu)
    A_dil = np.linalg.solve(I6 + S @ np.linalg.solve(Cm, Ci - Cm), I6)          # [I + S Cm⁻¹ (Ci − Cm)]⁻¹
    if scheme == "dilute":
        A = A_dil
    elif scheme == "mori_tanaka":
        A = A_dil @ np.linalg.solve((1.0 - f) * I6 + f * A_dil, I6)               # A_dil [(1−f) I + f A_dil]⁻¹
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    C = Cm + f * (Ci - Cm) @ A
    PV, PD = _projectors()
    e = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    K = float(e @ C @ e / 9.0)
    G = float(np.trace(PD @ C) / 10.0)
    iso = 3.0 * K * PV + 2.0 * G * PD
    err = float(np.linalg.norm(C - iso) / np.linalg.norm(C))
    return EffectiveIsotropic(K, G, 9 * K * G / (3 * K + G), (3 * K - 2 * G) / (2 * (3 * K + G)),
                              T.mandel_to_stiffness(C), err, scheme)


def dilute_sphere(f: float, matrix: IsoPhase, inclusion: IsoPhase) -> EffectiveIsotropic:
    """Dilute (non-interacting) spherical inclusions of volume fraction ``f`` in an isotropic ``matrix``:
    C = C_m + f (C_i − C_m) A, A = [I + S C_m⁻¹ (C_i − C_m)]⁻¹. Accurate to first order in f only."""
    return _inclusion_scheme(f, matrix, inclusion, "dilute")


def mori_tanaka_sphere(f: float, matrix: IsoPhase, inclusion: IsoPhase) -> EffectiveIsotropic:
    """Mori–Tanaka estimate for spherical inclusions of volume fraction ``f`` in an isotropic ``matrix``:
    A_MT = A_dil [(1 − f) I + f A_dil]⁻¹, C = C_m + f (C_i − C_m) A_MT. Equals the HS upper (lower) bound when the
    matrix has the largest (smallest) K and G. A mean-field estimate; no claim for an arbitrary microstructure."""
    return _inclusion_scheme(f, matrix, inclusion, "mori_tanaka")


# ============================================================================================== fractions
def _densities(rho: Any, n: int) -> np.ndarray:
    rho = np.asarray(rho, float)
    if rho.shape != (n,) or not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError(f"densities must be {n} finite non-negative numbers")
    return rho


def mixture_density(v: Any, rho: Any) -> float:
    """ρ = Σ v_i ρ_i of volume fractions ``v`` and phase densities ``rho`` (a void has ρ = 0)."""
    v = check_fractions(v)
    if v.ndim != 1:
        raise ValueError("one fraction vector expected")
    return float(v @ _densities(rho, v.shape[0]))


def volume_to_mass_fractions(v: Any, rho: Any) -> np.ndarray:
    """Mass fractions w_i = v_i ρ_i / ρ. The mixture must have positive density."""
    v = check_fractions(v)
    rho_c = mixture_density(v, rho)
    if rho_c <= 0:
        raise ValueError("the mixture density must be positive")
    return v * np.asarray(rho, float) / rho_c


def mass_to_volume_fractions(w: Any, rho: Any) -> np.ndarray:
    """Volume fractions v_i = (w_i / ρ_i) / Σ_j (w_j / ρ_j). Every density must be > 0: a void has no mass and its
    volume cannot be recovered from mass fractions (refused)."""
    w = check_fractions(w)
    if w.ndim != 1:
        raise ValueError("one fraction vector expected")
    rho = _densities(rho, w.shape[0])
    if np.any(rho <= 0):
        raise ValueError("every density must be > 0 to convert mass fractions to volume fractions")
    u = w / rho
    return u / u.sum()


# ========================================================================================= interpolation of two solids
@dataclass(frozen=True)
class InterpolatedModuli:
    """K(x), G(x) and dK/dx, dG/dx of two interpolated isotropic solids; ``E``, ``nu``, ``dE`` (3-D conversions)
    are None for ``dim = 2``."""

    K: np.ndarray
    G: np.ndarray
    dK: np.ndarray
    dG: np.ndarray
    E: np.ndarray | None
    nu: np.ndarray | None
    dE: np.ndarray | None


SCHEMES = ("voigt", "reuss", "hs_upper", "hs_lower")


def interpolate_moduli(x: Any, phase_a: IsoPhase, phase_b: IsoPhase, scheme: str = "hs_upper",
                       dim: int = 3) -> InterpolatedModuli:
    """K(x) and G(x) of a mixture with fraction ``x`` of ``phase_b`` (x = 0 gives ``phase_a``), interpolated
    independently by the ``scheme`` ("voigt", "reuss", "hs_upper", "hs_lower"; bounds as in :func:`hs_bounds` for the
    two phases) with the exact derivatives. Both phases must be non-void (K, G > 0): K(x), G(x) > 0 on [0, 1]."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if phase_a.is_void or phase_b.is_void:
        raise ValueError("interpolate_moduli needs two non-void phases (use a small positive stiffness for a soft one)")
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
        raise ValueError("x must lie in [0, 1]")
    res = []
    for a, b, which in ((phase_a.K, phase_b.K, "K"), (phase_a.G, phase_b.G, "G")):
        if scheme == "voigt":
            val, der = (1 - x) * a + x * b, np.full_like(x, b - a)
        else:
            if scheme == "reuss":
                shift = 0.0
            else:
                pick = max if scheme == "hs_upper" else min
                Ks, zeta = _shifts(np.array(pick(phase_a.K, phase_b.K)), np.array(pick(phase_a.G, phase_b.G)), dim)
                shift = float(Ks if which == "K" else zeta)
            S = (1 - x) / (a + shift) + x / (b + shift)
            val, der = 1.0 / S - shift, -(1.0 / (b + shift) - 1.0 / (a + shift)) / S ** 2
        res.append((val, der))
    (K, dK), (G, dG) = res
    if dim == 3:
        E = 9 * K * G / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        dE = (9 * G ** 2 * dK + 27 * K ** 2 * dG) / (3 * K + G) ** 2
    else:
        E = nu = dE = None
    return InterpolatedModuli(K, G, dK, dG, E, nu, dE)


# ========================================================================================= SIMP / RAMP admissibility
def _solid_moduli(nu: float, dim: int, plane: str) -> tuple[float, float]:
    """(K, G) of the solid with E = 1: 3-D K; 2-D κ of the plane convention."""
    if dim == 3:
        m = isotropic_moduli(E=1.0, nu=nu)
        return m.K, m.G
    if dim == 2:
        return plane_bulk_modulus(1.0, nu, plane), 1.0 / (2.0 * (1.0 + nu))
    raise ValueError("dim must be 2 or 3")


def hs_void_ramp_parameters(nu: float, dim: int = 3, plane: str = "stress") -> tuple[float, float]:
    """(q_K, q_G) with K_HS(x) = K x / (1 + q_K (1 − x)), G_HS(x) = G x / (1 + q_G (1 − x)) for a solid/void mixture
    (x = solid fraction): q_K = K / K*, q_G = G / ζ of the solid as comparison medium."""
    K, G = _solid_moduli(nu, dim, plane)
    Ks, zeta = _shifts(np.array(K), np.array(G), dim)
    return float(K / Ks), float(G / zeta)


@cite("bendsoe_sigmund_1999", note="condition re-derived in the prereg (RAMP form of the void HS bound); text not read")
def required_interpolation_parameters(nu: float, dim: int = 3, plane: str = "stress") -> tuple[float, float]:
    """(p_min, q_min): smallest SIMP exponent p and smallest RAMP parameter q for which x^p resp. x / (1 + q (1 − x))
    times the solid's K and G stays below the HS upper bound of a solid/void mixture for all x in [0, 1]. p_min = 1 +
    max(q_K, q_G) and q_min = p_min − 1. 2-D plane stress: p_min = max(2/(1 − ν), 4/(1 + ν))."""
    q = max(hs_void_ramp_parameters(nu, dim, plane))
    return 1.0 + q, q


@dataclass(frozen=True)
class InterpolationCheck:
    """Result of :func:`interpolation_admissibility`: ``margin_K`` / ``margin_G`` are the smallest values over the grid
    of (HS upper − interpolated) / solid modulus (negative = violation), ``x_worst`` the grid point of the smaller."""

    admissible: bool
    margin_K: float
    margin_G: float
    x_worst: float
    p_min: float
    q_min: float


def interpolation_admissibility(law: str, param: float, nu: float, dim: int = 3, plane: str = "stress",
                                n: int = 2001, tol: float = 1e-12) -> InterpolationCheck:
    """Grid check that the scalar law r(x) (``law`` "simp": x^param; "ramp": x / (1 + param (1 − x))) applied to the
    moduli of a solid with Poisson ratio ``nu`` stays below the HS upper bound of the solid/void mixture on ``n`` points
    of [0, 1] (violation = margin < −``tol``). The analytic thresholds are returned as ``p_min`` and ``q_min``."""
    param = float(param)
    if law == "simp":
        if not (math.isfinite(param) and param > 0):
            raise ValueError("SIMP exponent must be > 0")
        r = lambda x: x ** param                                            # noqa: E731
    elif law == "ramp":
        if not (math.isfinite(param) and param >= 0):
            raise ValueError("RAMP parameter must be >= 0")
        r = lambda x: x / (1.0 + param * (1.0 - x))                         # noqa: E731
    else:
        raise ValueError("law must be 'simp' or 'ramp'")
    if n < 3:
        raise ValueError("n must be at least 3")
    K, G = _solid_moduli(nu, dim, plane)
    x = np.linspace(0.0, 1.0, int(n))
    hs = hs_bounds(np.stack([1.0 - x, x], axis=-1), [VOID, IsoPhase(K, G)], dim)
    mK = (hs.K_upper - r(x) * K) / K
    mG = (hs.G_upper - r(x) * G) / G
    p_min, q_min = required_interpolation_parameters(nu, dim, plane)
    worst = np.minimum(mK, mG)
    return InterpolationCheck(bool(worst.min() >= -tol), float(mK.min()), float(mG.min()),
                              float(x[int(np.argmin(worst))]), p_min, q_min)
