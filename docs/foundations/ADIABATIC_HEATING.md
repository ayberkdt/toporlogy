# Adiabatic heating by plastic dissipation (WP14 R05, WP11 P04)

Code: `src/toporlogy/core/thermomechanics.py` (`AdiabaticPerzyna`). Tests: `tests/toporlogy/test_adiabatic.py`.
Preregistration, amendments and first results: `prereg/R5_adiabatic.md`. All numbers in the tests are **synthetic** and
are not material data. Scope: a homogeneous material point (no heat conduction, no FE coupling).

## Model

`AdiabaticPerzyna(K, G, yield_stress, hardening, tau0, sigma_ref, m, density, specific_heat, taylor_quinney,
temperature0)`: the existing `Perzyna` overstress J2 law with κ = σ_y(T) + H p, where σ_y is a positive
`PropertyFunction` in Pa and c_p a positive `PropertyFunction` in J/(kg K) (both with a validity interval, no
extrapolation: a temperature outside it raises `ValueError`). The temperature is an internal state (kelvin) and a time
step that carries a temperature is rejected. State: `plastic_strain`, `alpha`, `temperature`, `cold_work`.

## Which quantity heats

`Perzyna.update` returns the dissipation D = qΔp − [ψ_h(p_{n+1}) − ψ_h(p_n)], ψ_h = ½Hp². Its diagnostic
`plastic_work` is the total plastic work σ:Δε_p = qΔp. They differ by the stored hardening energy: W_p − D = Δψ_h
(checked: Σ plastic_work − Σ D = ½Hp_N² to 1e-9 relative). **Only D heats**; heating with W_p would count the stored
hardening energy twice (once in ψ, once as heat). The negative control in the tests shows the heat check detecting it.

## Equations and conventions

* Heat: ρ ∫_{T_n}^{T_{n+1}} c_p dT = β D_{n+1}(T_{n+1}), β = `taylor_quinney` ∈ [0, 1] (enforced). The enthalpy
  integral is used (`PropertyFunction.integral`), so a temperature-dependent c_p needs no midpoint rule. D is evaluated
  at σ_y(T_{n+1}) (fully implicit in T). Thermal part of the free energy and thermoelastic/thermoplastic coupling
  terms are neglected (classical Taylor–Quinney model): the returned `energy` is the mechanical ½ε_e:σ + ½Hp².
* Energy budget per step, with the trapezoid work W = ½(σ_n + σ_{n+1}):Δε:
  W − Δψ − D = ½(σ_n − σ_{n+1}):Δε_p (exact identity; this is the backward-Euler/trapezoid mismatch, O(Δt²) per step,
  O(Δt) over a history). Heat Q = ρΔh = βD. The remainder (1 − β)D is **stored cold work**, accumulated in the
  accounting state `cold_work` as D − Q (β = 0: all of D; β = 1: none). `cold_work` does not feed back on the flow;
  the hardening ψ_h does, through κ.
* Diagnostics per point: `temperature`, `delta_T`, `heat` (Q), `cold_work_increment` (D − Q), `heat_residual`
  (ρΔh − β·source, the convergence check of the solve), `thermal_evaluations`, plus those of `Perzyna`.

## Coupled local update (monolithic)

Per point the scalar ΔT solves R(ΔT) = ρ[h(T_n + ΔT) − h(T_n)] − βD(T_n + ΔT) = 0 by Brent's method on the bracket
[0, ΔT1], ΔT1 = βD(T_n)/(ρc_p(T_n)), doubled while R < 0 (R(0) = −βD(T_n) ≤ 0), rtol 4ε. The result is the isothermal
Perzyna update at the converged T_{n+1}. If D(T_n) = 0 or β = 0 the isothermal update at T_n is the result, so
**β = 0 is bitwise the existing `Perzyna`** with σ_y(T_0). Points are solved in a Python loop (the scalar solve is per
point; the model is a material-point model, not tuned for large batches). Cost: tens of isothermal Perzyna evaluations
per step (reported in `thermal_evaluations`).

Algorithmic tangent (implicit function theorem on R(T, ε) = 0): the isothermal Perzyna tangent at σ_y(T_{n+1}) plus
∂σ/∂T ⊗ dT/dε, with ∂σ/∂T = 2G√(3/2) χ σ_y' n̂, dT/dε = β (Δp + D_Δp χ) 2G√(3/2) n̂ / (ρc_p − β ∂D/∂T). Both
factors are proportional to n̂, so the result stays symmetric.

## Verified scope (see the record for the measured values)

β = 0 bitwise Perzyna (regression); no feedback with constant σ_y; per-step energy identity; history closure with dt;
heat closure against the closed-form antiderivative of a linear c_p; adiabatic ΔT and stress against an independent
`solve_ivp` integration of the (p, T) ODE system for uniaxial strain-rate control; thermal softening amount against the
ODE; β ∈ [0, 1] and parameter validation; rollback/purity; tangent against central finite differences; negative control
(total plastic work as heat) detected by the energy check.

## Not implemented / limits

Heat conduction q = −k∇T and any FE coupling (the temperature field of a body); thermoelastic coupling and
temperature-dependent moduli, hardening modulus or rate parameters; a temperature- or strain-dependent β (a constant
here; the physically observed dependence is not modelled); finite strain; inertia (dynamic localisation); latent or
reaction heat (kept separate from plastic dissipation so as not to double count). Because the temperature rise is
instantaneous and local, this is the adiabatic limit only (fast loading); at slow rates conduction removes the heat.
