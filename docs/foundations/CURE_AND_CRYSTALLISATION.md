# Cure kinetics, glass transition, reaction heat, chemical shrinkage and crystallisation (material point)

Status: slice TS_cure (WP16 TS01-TS03, WP15 TP03). Code: `src/toporlogy/core/cure.py`. Tests:
`tests/toporlogy/test_cure.py`. Preregistration and first results: `prereg/TS_cure.md`. Every parameter in the tests is
**synthetic**; none is material data and none may be used for a real resin or polymer.

## Units and conventions

Time in s; temperature absolute in K; rate constants in 1/s; Arrhenius k = A exp(−Q/(R T)) with Q in J/mol and
R = 8.314462618 J/(mol K); conversion α ∈ [0, 1]; reaction heat in J per kg of resin (exothermic positive); c_p in
J/(kg K) from a `PropertyFunction` (`core/properties.py`, `positive=True`); density in kg/m³; small strain, tension positive.

## Laws

| Law | Form | Class / function |
|---|---|---|
| n-th order | dα/dt = k(T)(1−α)^n, n > 0 (non-integer allowed) | `NthOrder` |
| Kamal–Sourour | dα/dt = (k1(T) + k2(T) α^m)(1−α)^n, m ≥ 0, n > 0 | `KamalSourour` |
| Isothermal closed forms | n = 1: α = α0 + (1−α0)(1 − e^{−kt}); n ≠ 1: (1−α)^{1−n} = (1−α0)^{1−n} + (n−1)kt; for n < 1 the completion time t* = (1−α0)^{1−n}/((1−n)k), α = 1 afterwards | `isothermal_first_order`, `isothermal_nth_order`, `finite_completion_time` |
| DiBenedetto | (Tg−Tg0)/(Tg∞−Tg0) = λα/(1−(1−λ)α); dTg/dα = (Tg∞−Tg0)λ/(1−(1−λ)α)²; inverse α = x/(λ+(1−λ)x) | `DiBenedetto` |
| Reaction heat | step heat ΔH Δα (J/kg), source q = ρ ΔH dα/dt (W/m³), adiabatic temperature from ∫c_p dT = ΔH Δα | `ReactionHeat` |
| Chemical shrinkage | volumetric ε_v = β max(α−α_gel, 0) (β < 0 shrinkage), linear ε_v/3 per axis (small strain) | `ChemicalShrinkage` |
| Avrami | X = 1 − exp(−k t^n), k in s^−n, X the **relative** transformed fraction; inverse t = (−ln(1−X)/k)^{1/n} | `avrami`, `avrami_time` |
| Nakamura | dX/dt = n K(T)(1−X)[−ln(1−X)]^{(n−1)/n}, K = k^{1/n} (1/s) | `Nakamura`, `nakamura_K_from_avrami` |
| Material point | σ = C(T,α):(ε − ε_th(T)1 − ε_ch,lin(α)1), state α | `CuringThermoElasticPoint` |

## Integrators

**Cure step** `CureKinetics.advance(alpha_n, T_n, T_np1, dt, substeps=1)`: backward Euler on dα/dt = f(α, T) with
the rate taken at the end temperature of the step (T linear in the step; `substeps` m divides the step and
interpolates T linearly). *Bound by construction:* for f ≥ 0 and f(1, T) = 0 the function g(x) = x − α_n − h f(x)
satisfies g(α_n) = −h f(α_n) ≤ 0 and g(1) = 1 − α_n ≥ 0, so a root exists in [α_n, 1] and the Newton iteration of the
code only accepts iterates inside a shrinking bracket inside that interval (bisection otherwise: non-finite or
out-of-bracket Newton steps, singular derivatives at α = 0 with m < 1 or α = 1 with n < 1). The result is monotone and
bounded for **any** step size; nothing is clipped; α_n = α_{n+1} for dt = 0, at equilibrium (α = 1, or α = 0 with
k1 = 0) the state does not move. A solve that does not converge raises `ReturnMappingFailure`; invalid inputs
(α ∉ [0, 1], NaN, dt < 0, T ≤ 0, substeps < 1) raise `ValueError`. The method is first-order accurate (measured
order 0.94 to 1.0, TS1d/TS2a); a higher-order scheme would not keep the bound without a step rejection and is not
provided. The explicit Euler step of the same law with dt = 100 s overshoots to α = 8.2 (negative control TS1e).
Stiff kinetics (k dt ≫ 1) are handled by the implicit step (α → 1 bounded, accuracy in time not claimed).

**Crystallisation** `Nakamura`: state θ = ∫K dt, X = 1 − exp(−θ^n) = −expm1(−θ^n) ∈ [0, 1]. This is the Nakamura
equation without the start-up singularity: the right side in X is 0 at X = 0 for n > 1 (an integration in X never
starts: negative control TP4d) and infinite for n < 1. For T linear in the step the increment is
dt (∫_{T_n}^{T_{n+1}} K dT)/(T_{n+1} − T_n), taken from `PropertyFunction.integral`, which is exact for constant,
piecewise-linear (also across knots) and polynomial K; a Pchip K is exact as well by the properties module (antiderivative of the cubic pieces; not exercised in this slice).
θ is non-decreasing for K ≥ 0; a negative increment raises (melting and re-crystallisation are not modelled). The X
↔ θ round trip of `advance` costs about 1 ulp; keep θ as the state (`advance_theta`) for long histories.

## Reaction heat, no double counting

c_p is the sensible heat only; the reaction energy enters once, as the source ρ ΔH dα/dt of the thermal equation, or as
ΔH Δα in an adiabatic balance ∫c_p dT = ΔH Δα. A c_p that already contains the reaction (an apparent c_p from DSC
across the exotherm) must not be combined with this source. The heat of a step telescopes to ΔH (α_N − α_0) for any
path (TS4c). The coupled heat equation and a thermal FE are not part of this slice.

## Chemical shrinkage, stress-free reference

β is the volumetric strain per unit conversion at constant temperature (the thermal strain is a separate function of
T, `ThermalExpansion`; no double counting). The conversion `alpha_gel` from which shrinkage produces strain is a model
choice (default 0: all of it); no universal gel conversion exists, and pre-gel shrinkage creating residual stress is
not assumed. With ε = ε_th + ε_ch (free) the stress is 0 (TS5b); fully constrained, σ = −3 K(T, α) ε_ch,lin 1, a tension
for shrinkage (TS5c). `CuringThermoElasticPoint` interpolates K and G linearly between an uncured and a cured
`IsotropicModuliOfT` (synthetic CHILE-type, instantaneous elastic response): the uncured state has a numerical minimum
stiffness, which is not a physical claim (the uncured resin carries no solid shear stress) and whose influence on a real
problem would have to be measured. The temperature is prescribed (one-way T → α → σ); α does not depend on ε.

## Verified scope (all synthetic)

TS1 first-order closed form, monotone and bounded, first-order convergence, explicit-Euler overshoot control, n-th order
closed forms with the finite completion time; TS2 Kamal–Sourour with a temperature ramp against `solve_ivp`
(rtol 1e-12), bounded at coarse steps, autocatalytic start, rate derivatives vs FD; TS3 DiBenedetto endpoints, derivative
and inverse; TS4 adiabatic rise (50 K), variable-c_p adiabatic cure against a coupled ODE, heat budget; TS5 free and
constrained shrinkage with thermal strain and gel conversion; TS6 trial/rollback, history, restart, order dependence,
invalid input; TP3 Avrami (X(10 s) = 1 − e^{−1}); TP4 Nakamura reduces to Avrami at constant T and matches an
independent ODE for a varying temperature across a kink of K(T).

## NOT implemented (not claimed)

* Vitrification-controlled (diffusion-limited) kinetics, a diffusion factor, and any kinetic law fitted to data (no DSC
  fitting, no calibration, no uncertainty).
* Gelation mechanics: gel point detection, evolving network and reference state, cure-dependent viscoelastic (Prony)
  response, pre-gel stress-free handling beyond the scalar `alpha_gel`; CHILE is only the synthetic interpolation above.
* Residual stress, warpage, spring-in and any FE use: no coupling to the FE layer (WP17), no heat equation, no finite-strain
  chemical shrinkage, no density change.
* Melting, re-crystallisation, absolute crystallinity (X is relative), a bell-shaped K(T) law (K is a user property function),
  crystallisation shrinkage and latent heat of crystallisation, physical aging, melt rheology and melt flow.
* Real material parameters of any kind: synthetic constants only; nothing here supports a process or certification claim.
