"""Thermoplastic solid-state transitions, crystallinity-dependent moduli, latent heat and crystallisation shrinkage at
the material-point level (WP15 TP1, TP2, TP5; docs/foundations/THERMOPLASTICS.md, pre-registration
docs/foundations/prereg/TP_thermoplastic.md).

Units (physical): time in s, temperature absolute in K, moduli in the units of the inputs, CTE in 1/K, latent heat L in
J/m^3 of the material (exothermic positive; L = rho * dH with dH in J/kg), density in kg/m^3, c_p in J/(kg K), strains
dimensionless (small strain, tension positive). Every number in the tests is synthetic, not material data.

* :class:`LogisticTransition`: a smooth, explicitly parameterised transition of a property across a glass transition,
  p(T) = p_g + (p_r - p_g) s(T) (form "linear", e.g. CTE) or ln p = ln p_g + (ln p_r - ln p_g) s(T) (form "log",
  e.g. a modulus that falls by decades), s(T) = 1 / (1 + exp(-(T - T_mid)/w)). **This is a synthetic phenomenological
  form, not a universal law**: T_mid and the width w are required parameters without defaults (T_g depends on grade,
  rate and the test that defines it), the validity interval is required and there is no extrapolation.
* :func:`mix_isotropic` and :class:`CrystallineMixture`: K(phi), G(phi) of an amorphous/crystalline two-phase solid by
  a stated rule ("voigt", "reuss", "hill", "hs_lower", "hs_upper"), built on :mod:`toporlogy.core.micromechanics`.
* :class:`CrystallisingThermoElasticPoint`: sigma = C(T, X) : (eps - eps_th(T) 1 - eps_cr(X) 1), the state X being the
  relative crystallinity advanced with :class:`toporlogy.core.cure.Nakamura` at the step temperatures.
* :func:`adiabatic_temperature_rise`: dT = L dX / (rho c_p).

Not implemented: melt flow, melting and re-crystallisation, physical ageing, finite strain, a two-way thermal coupling.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.special import expit

from toporlogy.core import micromechanics as mm
from toporlogy.core.constitutive import (HistoryUpdate, Kinematics, MaterialRequest, MaterialResponse, require_step,
                                         small_strain_history_evaluate)
from toporlogy.core.cure import Nakamura
from toporlogy.core.properties import (_IDEV6, _ONE6, IsotropicModuliOfT, PropertyFunction, ThermalExpansion)

__all__ = ["LogisticTransition", "MIXING_RULES", "mix_isotropic", "CrystallineMixture",
           "CrystallisingThermoElasticPoint", "adiabatic_temperature_rise"]

MIXING_RULES = ("voigt", "reuss", "hill", "hs_lower", "hs_upper")
_FORMS = ("linear", "log")


# ------------------------------------------------------------------------------------------------ glass transition
class LogisticTransition(PropertyFunction):
    """Property across a glass transition: s(T) = 1/(1 + exp(-(T - T_mid)/w)) = expit((T - T_mid)/w), and

        form "linear":  p = p_g + (p_r - p_g) s          (CTE, density, ...; value at T_mid is the arithmetic mean),
        form "log":     p = p_g (p_r/p_g)^s              (a modulus; needs p_g, p_r > 0; value at T_mid is sqrt(p_g p_r)),

    p_g the glassy and p_r the rubbery plateau (their order is free: a modulus falls, a CTE rises), T_mid the midpoint
    and w > 0 the width (K) of the logistic; s is 1/2 at T_mid, its largest slope is 1/(4 w). The plateaus are
    asymptotes that are never reached; the value at the validity limits is whatever the formula gives there.

    This is a synthetic phenomenological form. It is not a universal law, not a WLF/Arrhenius shift (those belong to
    :mod:`toporlogy.core.viscoelastic`), it has no rate dependence and T_mid is not "the" T_g of a material (T_g depends
    on the grade, the heating rate or frequency and the test that defines it). ``T_mid``, ``width`` and ``domain`` have
    no defaults on purpose; the validity interval must be finite and contain T_mid; an evaluation outside it raises.
    The derivative is analytic: p' = (p_r - p_g) s (1 - s)/w (linear), p ln(p_r/p_g) s (1 - s)/w (log). The integral
    (needed for the thermal strain of a CTE) is exact for the form "linear":
    p_g (T1 - T0) + (p_r - p_g) w [softplus((T1 - T_mid)/w) - softplus((T0 - T_mid)/w)]; it is not available for "log"."""

    smoothness = "C-infinity"

    def __init__(self, p_glassy: float, p_rubbery: float, T_mid: float, width: float, domain: tuple[float, float],
                 units: str, source: str, form: str = "linear", positive: bool = False, extrapolation: str = "error",
                 temperature_units: str = "K") -> None:
        if form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
        super().__init__(units, source, domain, extrapolation, positive or form == "log", temperature_units)
        lo, hi = self.domain
        if not math.isfinite(hi):
            raise ValueError("a transition needs a finite validity interval (no open-ended domain)")
        pg, pr, Tm, w = (float(v) for v in (p_glassy, p_rubbery, T_mid, width))
        if not all(math.isfinite(v) for v in (pg, pr, Tm, w)):
            raise ValueError("p_glassy, p_rubbery, T_mid and width must be finite")
        if pg == pr:
            raise ValueError("p_glassy == p_rubbery is not a transition")
        if not w > 0.0:
            raise ValueError(f"width must be > 0 K, got {width!r}")
        if not lo <= Tm <= hi:
            raise ValueError(f"T_mid = {Tm} K must lie inside the validity interval [{lo}, {hi}] K")
        if form == "log" and not (pg > 0.0 and pr > 0.0):
            raise ValueError(f"form 'log' needs positive plateaus, got {pg!r} and {pr!r}")
        self._require_positive(min(pg, pr), "LogisticTransition")
        self.p_g, self.p_r, self.T_mid, self.width, self.form = pg, pr, Tm, w, form
        self._lam = math.log(pr / pg) if form == "log" else 0.0

    def _s(self, T: np.ndarray) -> np.ndarray:
        return np.asarray(expit((T - self.T_mid) / self.width))

    def _value(self, T: np.ndarray) -> np.ndarray:
        s = self._s(T)
        if self.form == "log":
            return self.p_g * np.exp(self._lam * s)
        return self.p_g + (self.p_r - self.p_g) * s

    def _derivative(self, T: np.ndarray) -> np.ndarray:
        s = self._s(T)
        ds = s * (1.0 - s) / self.width
        if self.form == "log":
            return self.p_g * np.exp(self._lam * s) * self._lam * ds
        return (self.p_r - self.p_g) * ds

    def _integral(self, T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
        if self.form == "log":
            raise ValueError("the integral of the form 'log' is not available (it is needed for a CTE only; use "
                             "form 'linear' for a CTE)")
        w = self.width
        soft = np.logaddexp(0.0, (T1 - self.T_mid) / w) - np.logaddexp(0.0, (T0 - self.T_mid) / w)
        return self.p_g * (T1 - T0) + (self.p_r - self.p_g) * w * soft


# ------------------------------------------------------------------------------------------------ mixing rule
def mix_isotropic(phi: float, amorphous: mm.IsoPhase, crystal: mm.IsoPhase, rule: str = "hs_lower") -> tuple[float, float]:
    """(K, G) of an isotropic amorphous/crystalline mixture with the crystalline volume fraction phi in [0, 1].

    "voigt" / "reuss": uniform strain / stress averages (bounds of every microstructure). "hill": their arithmetic
    mean (inside [Reuss, Voigt] by construction; an estimate, not a bound, and not an isotropy-consistent scheme).
    "hs_lower" / "hs_upper": the Hashin-Shtrikman-Walpole bounds (statistically isotropic two-phase composite; with a
    crystal stiffer in K and G the lower bound is the Mori-Tanaka estimate with a soft amorphous matrix). The rule is a
    modelling assumption, not derived from a morphology. phi = 0 and phi = 1 return the pure phases."""
    if rule not in MIXING_RULES:
        raise ValueError(f"rule must be one of {MIXING_RULES}, got {rule!r}")
    if not isinstance(amorphous, mm.IsoPhase) or not isinstance(crystal, mm.IsoPhase):
        raise ValueError("amorphous and crystal must be IsoPhase")
    if amorphous.is_void or crystal.is_void:
        raise ValueError("a void is not a polymer phase")
    p = float(phi)
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"crystalline fraction must lie in [0, 1], got {phi!r}")
    f = np.array([1.0 - p, p])
    phases = [amorphous, crystal]
    if rule == "voigt":
        K, G = mm.voigt(f, phases)
    elif rule == "reuss":
        K, G = mm.reuss(f, phases)
    elif rule == "hill":
        Kv, Gv = mm.voigt(f, phases)
        Kr, Gr = mm.reuss(f, phases)
        K, G = 0.5 * (Kv + Kr), 0.5 * (Gv + Gr)
    else:
        b = mm.hs_bounds(f, phases)
        K, G = (b.K_lower, b.G_lower) if rule == "hs_lower" else (b.K_upper, b.G_upper)
    return float(K), float(G)


class CrystallineMixture:
    """K(T, phi), G(T, phi) of an amorphous and a crystalline phase, each an :class:`IsotropicModuliOfT` of temperature
    (one pressure unit for both), combined by :func:`mix_isotropic` at every temperature. phi is the ABSOLUTE
    crystalline volume fraction. The validity interval is the intersection of the two phases'. Evaluation loops over the
    points (the micromechanics bounds are scalar)."""

    def __init__(self, amorphous: IsotropicModuliOfT, crystal: IsotropicModuliOfT, rule: str = "hs_lower") -> None:
        if not (isinstance(amorphous, IsotropicModuliOfT) and isinstance(crystal, IsotropicModuliOfT)):
            raise ValueError("amorphous and crystal must be IsotropicModuliOfT")
        if amorphous.units != crystal.units:
            raise ValueError(f"the phases must share one unit, got {amorphous.units!r} and {crystal.units!r}")
        if rule not in MIXING_RULES:
            raise ValueError(f"rule must be one of {MIXING_RULES}, got {rule!r}")
        lo, hi = max(amorphous.domain[0], crystal.domain[0]), min(amorphous.domain[1], crystal.domain[1])
        if not hi > lo:
            raise ValueError("the validity intervals of the two phases do not overlap")
        self.amorphous, self.crystal, self.rule = amorphous, crystal, rule
        self.units, self.domain = amorphous.units, (lo, hi)

    def moduli(self, T: Any, phi: Any) -> tuple[np.ndarray, np.ndarray]:
        """(K, G) arrays for broadcast temperatures T (K) and crystalline fractions phi."""
        Ta, pa = np.broadcast_arrays(np.asarray(T, dtype=float), np.asarray(phi, dtype=float))
        Ka, Ga = np.atleast_1d(self.amorphous.K(Ta)), np.atleast_1d(self.amorphous.G(Ta))
        Kc, Gc = np.atleast_1d(self.crystal.K(Ta)), np.atleast_1d(self.crystal.G(Ta))
        K, G = np.empty(Ta.size), np.empty(Ta.size)
        for i, p in enumerate(pa.reshape(-1)):
            K[i], G[i] = mix_isotropic(p, mm.IsoPhase(float(Ka[i]), float(Ga[i])),
                                       mm.IsoPhase(float(Kc[i]), float(Gc[i])), self.rule)
        return K.reshape(Ta.shape), G.reshape(Ta.shape)


# ------------------------------------------------------------------------------------------------ latent heat
def adiabatic_temperature_rise(latent_heat: float, dX: Any, rho: float, cp: float) -> Any:
    """dT = L dX / (rho c_p) (K) for a latent heat L (J/m^3, exothermic positive), a transformed increment dX >= 0
    (the same state X as the point; L is per unit X), density rho (kg/m^3) and a constant c_p (J/(kg K)) that is the
    sensible heat only (the latent heat enters once, here). Melting (dX < 0) is not modelled and raises."""
    for name, v in (("latent_heat", latent_heat), ("rho", rho), ("cp", cp)):
        if not (isinstance(v, (int, float, np.floating)) and math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be a finite number > 0, got {v!r}")
    d = np.asarray(dX, dtype=float)
    if not np.all(np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError(f"dX must be finite and >= 0 (melting is not modelled), got {np.asarray(dX)!r}")
    out = latent_heat * d / (rho * cp)
    return float(out) if out.ndim == 0 else out


# ------------------------------------------------------------------------------------------------ the material point
_DEFAULT_REQUEST = MaterialRequest()


class CrystallisingThermoElasticPoint:
    """Small-strain thermoelastic material point that crystallises at a prescribed temperature (one-way T -> X -> sigma):

        sigma = C(T, X) : (eps - eps_th(T) 1 - eps_cr(X) 1),    eps_cr = beta_c (X - X_ref) / 3  (per axis),

    C = K 1x1 + 2 G I_dev with (K, G) from ``mixture`` at the temperature and the absolute crystalline fraction
    phi = X_max X; ``expansion`` the thermal strain from the stress-free temperature T_ref (the crystalline state does
    not change the CTE here). The state X in [0, 1] is the RELATIVE crystallinity advanced with ``nakamura`` at the
    step temperatures (X_{n+1} = nakamura.advance(X_n, T_n, T_{n+1}, dt); it does not depend on eps and is never
    clipped); ``X0`` is the initial value and the stress-free reference of the crystallisation strain
    (eps_cr = beta_c (X - X0)/3). ``beta_c`` < 0 is the volumetric crystallisation shrinkage per unit X (a synthetic
    parameter; small strain, isotropic). ``latent_heat`` L > 0 (J/m^3 per unit X, exothermic) gives the diagnostic
    ``latent_heat`` = L (X_{n+1} - X_n) (J/m^3 released in the step): it does not feed back into the temperature.
    The algorithmic tangent is C(T_{n+1}, X_{n+1}) (the update is linear in eps at the advanced state). Energy and
    dissipation are not defined (None). Physical units (``physical = True``)."""

    name = "crystallising_thermoelastic"
    kinematics = "small"
    conservative = False
    physical = True
    state_fields: tuple[str, ...] = ("X",)

    def __init__(self, nakamura: Nakamura, mixture: CrystallineMixture, expansion: ThermalExpansion, beta_c: float,
                 latent_heat: float, X_max: float = 1.0, X0: float = 0.0) -> None:
        if not isinstance(nakamura, Nakamura):
            raise ValueError("nakamura must be a Nakamura")
        if not isinstance(mixture, CrystallineMixture):
            raise ValueError("mixture must be a CrystallineMixture")
        if not isinstance(expansion, ThermalExpansion):
            raise ValueError("expansion must be a ThermalExpansion")
        for name, v in (("beta_c", beta_c), ("latent_heat", latent_heat), ("X_max", X_max), ("X0", X0)):
            if not (isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and math.isfinite(v)):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        if not beta_c < 0.0:
            raise ValueError(f"beta_c is a shrinkage coefficient and must be < 0, got {beta_c!r}")
        if not latent_heat > 0.0:
            raise ValueError(f"latent_heat must be > 0 (exothermic crystallisation), got {latent_heat!r}")
        if not 0.0 < X_max <= 1.0:
            raise ValueError(f"X_max must lie in (0, 1], got {X_max!r}")
        if not 0.0 <= X0 <= 1.0:
            raise ValueError(f"X0 must lie in [0, 1], got {X0!r}")
        self.nakamura, self.mixture, self.expansion = nakamura, mixture, expansion
        self.beta_c, self.latent_heat, self.X_max, self.X0 = float(beta_c), float(latent_heat), float(X_max), float(X0)
        self.units = mixture.units

    def initial_state(self, n: int) -> dict[str, np.ndarray]:
        return {"X": np.full(n, self.X0)}

    def update(self, eps: np.ndarray, committed: Mapping[str, np.ndarray],
               request: MaterialRequest = _DEFAULT_REQUEST) -> HistoryUpdate:
        step = require_step(self, request, temperature=True)
        n = len(eps)
        T1 = np.broadcast_to(np.asarray(step.temperature, dtype=float), (n,))
        T0 = np.broadcast_to(np.asarray(step.temperature_prev, dtype=float), (n,))
        X_n = np.asarray(committed["X"], dtype=float)
        X1 = np.atleast_1d(np.asarray(self.nakamura.advance(X_n, T0, T1, step.dt), dtype=float))
        K, G = self.mixture.moduli(T1, self.X_max * X1)
        e_th = np.broadcast_to(np.asarray(self.expansion.strain(T1), dtype=float), (n,))
        e_cr = self.beta_c * (X1 - self.X0) / 3.0
        e = eps - (e_th + e_cr)[:, None, None] * np.eye(3)
        tr = np.trace(e, axis1=1, axis2=2)
        dev = e - (tr / 3.0)[:, None, None] * np.eye(3)
        sig = (K * tr)[:, None, None] * np.eye(3) + 2.0 * G[:, None, None] * dev
        C = None
        if request.tangent:
            C = K[:, None, None] * np.outer(_ONE6, _ONE6) + 2.0 * G[:, None, None] * _IDEV6
        diag = {"X": X1.copy(), "temperature": np.array(T1), "thermal_strain_trace": 3.0 * e_th,
                "crystallisation_strain_trace": 3.0 * e_cr, "latent_heat": self.latent_heat * (X1 - X_n),
                "bulk_modulus": K, "shear_modulus": G}
        return HistoryUpdate(sig, C, {"X": X1}, diag)

    def evaluate(self, kin: Kinematics, params: Mapping[str, Any] | None, committed: Mapping[str, np.ndarray],
                 request: MaterialRequest = _DEFAULT_REQUEST) -> MaterialResponse:
        return small_strain_history_evaluate(self, kin, committed, request)
