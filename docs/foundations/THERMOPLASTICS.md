# Thermoplastics at the material point: glass transition, crystallinity, latent heat, crystallisation shrinkage

Status: slice TP_thermoplastic (WP15 TP01/TP02 property model, TP04 shrinkage and latent heat, acceptance cases TP1, TP2,
TP5). Code: `src/toporlogy/core/thermoplastic.py`. Tests: `tests/toporlogy/test_thermoplastic.py`. Preregistration and
first results: `prereg/TP_thermoplastic.md`. Every parameter in the tests is **synthetic**; none is material data and
none may be used for a real polymer. No Tg, Tm or modulus of a real grade is claimed or implied.

## Verified scope

| Piece | What is verified (check IDs in the prereg) | Class / function |
|---|---|---|
| Glass-transition property model | exact values at the midpoint and at T_m ± w ln 3, monotonicity, peak slope at T_m, derivative vs FD, exact integral vs `quad`, refusal outside the validity interval, no default for T_mid / width / domain (TP2a-f) | `LogisticTransition` |
| Prescribed E(T) and α(T) through the transition | ramp / hold / cool history through `drive_material_point` vs an independent closed form with `quad` for ε_th; the secant-for-instantaneous mistake is detected (TP1a-c) | `LogisticTransition` with `IsotropicModuliOfT`, `ThermalExpansion`, `ThermoElasticPoint` |
| Crystallinity-dependent moduli | five rules inside Voigt/Reuss (HS rules inside the HS bounds), strictly increasing, exact endpoints, closed forms, refusals (MX1-MX4) | `mix_isotropic`, `CrystallineMixture` (built on `core.micromechanics`) |
| Crystallising material point | free stress zero and strain = ε_th + β_c ΔX/3 with cooling, constrained bar and fully constrained tension for shrinkage, reference state X0, boundedness without clipping, wrong shrinkage sign detected, contract, tangent, rollback (TP5b-h) | `CrystallisingThermoElasticPoint` |
| Latent heat | per-step heat L ΔX, telescoping sum, isothermal `quad` balance, adiabatic ΔT = L ΔX/(ρ c_p) (TP5a) | diagnostic `latent_heat`, `adiabatic_temperature_rise` |

## Conventions

Time s; temperature absolute K; moduli in the units of the inputs (one unit for both phases); CTE 1/K; L in J/m³ per unit X
(L = ρ ΔH when the specific heat is ΔH in J/kg), exothermic positive; ρ in kg/m³; c_p in J/(kg K) and the sensible heat only (the
latent heat enters once, as the source/diagnostic; never also in c_p). Small strain, tension positive, Voigt
[xx, yy, zz, yz, xz, xy].

X is the **relative** crystallinity advanced by `core.cure.Nakamura` (θ = ∫K dt, X = 1 − exp(−θ^n); exact for the
property-function family when T is linear in the step; not clipped). φ = X_max X is the absolute crystalline volume
fraction used by the mixing rule. β_c < 0 is the volumetric crystallisation strain per unit **X** (X_max is absorbed into
it); the linear strain per axis is β_c (X − X0)/3, X0 the stress-free reference state. A point with β_c ≥ 0 is refused.

## Glass-transition model

s(T) = 1/(1 + e^{−(T−T_m)/w}); form "linear" p = p_g + (p_r − p_g) s (CTE), form "log" p = p_g (p_r/p_g)^s (a modulus, positive by
construction). This is a **synthetic phenomenological form, not a universal law**: it has no rate dependence, no WLF or
Arrhenius shift (use `core.viscoelastic` for a time-temperature shift), and T_m is not "the" T_g of a material (T_g depends on
grade, heating rate or frequency, and on whether it is defined by DSC, DMA or dilatometry). The validity interval is finite and
required, an evaluation outside it raises (no clamp, no extrapolation), the plateaus are asymptotes that are never reached.
The derivative is analytic; the integral (for the thermal strain) is exact for the form "linear" and refused for "log".

## Material point

σ = C(T, X):(ε − ε_th(T) 1 − β_c (X − X0)/3 · 1), C = K 1⊗1 + 2G I_dev with (K, G) from the mixture at the step temperature and
φ = X_max X. One-way coupling: T is prescribed, X does not depend on ε, and the latent heat is a **diagnostic**
(`latent_heat` = L (X_{n+1} − X_n), J/m³) that is not fed back into the temperature. The algorithmic tangent is C(T_{n+1}, X_{n+1}).
Energy and dissipation are not defined (None). The CTE does not depend on X.

Mixing rules: "voigt", "reuss" (bounds of every microstructure), "hill" (their mean; an estimate, not a bound), "hs_lower",
"hs_upper" (Hashin–Shtrikman–Walpole, statistically isotropic two-phase solid; with a stiffer crystal the lower bound is the
Mori–Tanaka estimate with a soft amorphous matrix). The default is "hs_lower"; the rule is a modelling assumption, not derived
from a morphology, and the crystal is assumed stiffer than the amorphous phase in K and G (not checked at construction).

## NOT implemented

* Melt flow and melt rheology (Carreau–Yasuda / Cross), injection moulding, fibre orientation transport: a softened elastic
  solid is not a melt. No solid-to-melt hand-off.
* Melting and re-crystallisation: X is non-decreasing; a negative θ increment raises. No enthalpy convention for melting.
* Physical ageing and rejuvenation (fictive temperature, free volume).
* Rate dependence of the transition and of the moduli (see `core/viscoelastic`, `core/creep`, `core/viscoplastic`).
* A two-way thermal coupling (heat equation with the latent-heat source; the FE slice TP6).
* Crystallinity-dependent CTE, temperature-dependent shrinkage coefficient, anisotropic or fibre-composite shrinkage (the
  shrinkage here is isotropic and does not hold for a fibre composite), residual stress after a full cycle in an FE model.
* Finite strain (small strain only).
* Any claim that one law represents all PEEK, PA or other grades; no calibration to measured data (DMA, DSC, dilatometry).
