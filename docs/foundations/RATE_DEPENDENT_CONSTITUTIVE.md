# Rate-dependent and pressure-sensitive constitutive laws (WP14 slice)

Code: `src/toporlogy/core/viscoplastic.py`. Tests: `tests/toporlogy/test_viscoplastic.py`. Preregistration and first
results: `prereg/R_rate.md`. Contract: `MATERIAL_CONTRACT.md`. All numbers in the tests are **synthetic** and are not
material data.

## Conventions and units

SI units, physical parameters (no scaling by a Young modulus; the legacy unit-modulus `J2Plasticity` is unchanged and
remains the law of the FE physics). Stress and moduli in Pa, time in s, temperature (Eyring only) in K, activation
volume in m³ per event (molar variant: m³/mol). Tension-positive stress, `p = −tr σ / 3`, Voigt
[xx, yy, zz, yz, xz, xy] (stress physical shear, strain engineering shear), `q = √(3/2)‖dev σ‖`. The models set
`physical = True`; `NonlinearElasticity` refuses such models (contract rule), so they are material-point and
history-driver models (`drive_material_point`, optionally with `times`), not yet FE physics (WP17).

## Laws

* `PhysicalJ2(K, G, yield_stress, hardening, saturation_stress=None, saturation_rate=0)`: f = q − κ(p),
  κ = σ_y + Hp + (σ_∞ − σ_y)(1 − e^(−δp)), radial return, algorithmic tangent, stored energy and dissipation. Rate
  independent (a time step is ignored). Doubling E at fixed σ_y does not change the yield stress (P2).
* `Perzyna(K, G, yield_stress, hardening, tau0, sigma_ref, m)`: ṗ = (1/τ0)⟨f/σ_ref⟩^m, f = q − (σ_y + Hp), associated
  J2 flow, backward Euler with a bracketed safeguarded scalar solve (hardening H ≥ 0 makes the residual monotone),
  consistent tangent `K 1⊗1 + 2Gθ I_dev − 6G²(χ − Δp/q*) n̂⊗n̂`. Needs `request.step` (`require_step`); Δt = 0 gives
  no flow; below yield there is no flow (unlike Norton creep). τ0 → 0 tends to `PhysicalJ2`, τ0 → ∞ to elastic.
* `eyring_rate(tau, A, V_act, T) = A sinh(τV*/(k_B T))` and the inverse `eyring_stress` (asinh), `eyring_log_rate` and
  `eyring_stress_from_log_rate` (log-sinh forms for arguments beyond the float range of sinh, |x| > ~710),
  `eyring_rate_molar` / `eyring_stress_molar` (molar volume with R = N_A k_B). `A` is a calibration constant;
  the temperature dependence of A is not modelled. The Eyring law is a stand-alone function: it is **not** wired into
  `Perzyna` (the combination "rate-dependent yield stress from an Eyring law" is not implemented).
* `DruckerPrager(K, G, a, b0, hardening=0, a_flow=None)`: f = q − a p − b(κ), b = b0 + hκ, flow potential
  g = q − a_g p (`a_flow` = a_g, default a: associated), κ̇ = λ̇. Smooth cone return (linear) and apex return, consistent
  tangent (non-symmetric for a_g < a). Uniaxial tension yields at b/(1 + a/3), compression at b/(1 − a/3). Refused:
  a ≤ 0, a ≥ 3, a_g ≤ 0 (the apex return is undefined), a_g > a, b0 ≤ 0, h < 0.

## Verified scope (checks of `prereg/R_rate.md`)

R1 physical J2 against closed forms (uniaxial, load–unload–reverse, pure shear, hydrostatic, Voce) and the legacy law;
R2 Perzyna active branch against a closed form (m = 1) and brentq roots (m = 2, 0.5), FD of the full 6×6 tangent;
R3 rate sweep 1e-3, 0.1, 1 1/s against an ODE reference, time convergence (first order at 1 1/s, errors 1.4e4 to 1.1e3 Pa; at 0.1 1/s the errors fall from 0.6 to 2.4e-4 Pa;
at 1e-3 1/s the response sits on the quasi-static steady state, backward Euler agrees with the ODE to roundoff and the
preregistered convergence-ratio check is a recorded failure in the ledger); R4 stress relaxation and no flow below yield; R5 Eyring
round trip, extreme arguments, units; R6 Drucker–Prager yield strengths, pressure sign, hydrostatic and apex branch,
tangent FD, symmetry; R7 dissipation, bitwise rollback and restart, energy balance. Rate sensitivity (end stress
increasing with rate) was verified for the selected synthetic parameters only; it is not a polymer trend.

## Admissibility of non-associated flow

Not automatic. On the cone the dissipation per unit multiplier is `b0 + (a − a_g) p`: positive for a_g = a, and for
h = 0 for every state on the cone (p ≥ −b0/a); for a_g < a with h > 0 it can become negative near the apex in tension
(`R6g` reproduces a closed-form negative dissipation for a = 0.6, a_g = 0.06, h = 50 MPa). The model returns the
dissipation as computed and does not clip or reject; the caller must check it.

## Not implemented

Finite strain (multiplicative F = Fe Fvp, objectivity), strain softening and localisation control (nonlocal, gradient,
crack band; local softening is mesh dependent), thermal coupling (temperature-dependent parameters, Taylor–Quinney
heat), tabulated σ_y(κ, rate, T), Cowper–Symonds and Johnson–Cook, Norton creep combined with Perzyna, Drucker–Prager
caps, cavitation and damage, parameter and state VJP (design sensitivity) of these models, FE integration (WP17), YAML
configuration. The Perzyna model with m < 1 is not Lipschitz at f = 0 (the solve stays bracketed). The apex branch is
one-sided in the tangent at the cone/apex switch.
