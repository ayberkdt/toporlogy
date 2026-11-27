# Coupled cure slab and residual stress (WP16 TS04 stage 2, WP17 H04/H05, TS7)

Module `src/toporlogy/fem/mesh/cure_slab.py`; pre-registration `docs/foundations/prereg/TS7_cure_slab.md` (with the first
results); tests `tests/toporlogy/test_cure_slab.py`. Every number in the tests is **synthetic**: none is material data,
and nothing here is a validation of a composite process.

## What is implemented

* **Thermal problem**: ρ c_p Ṫ = ∇·(k∇T) + ρ ΔH α̇ for a neat resin (constant ρ, c_p, k; c_p is the sensible heat and the
  reaction heat enters once, as the source). Linear elements (the verified use is a Q4 strip `rectangle(1, n, Lx, L)`,
  y through the thickness), consistent capacity, Robin and Dirichlet surfaces, backward Euler in time.
* **Cure state** per quadrature point, advanced with `core.cure` (`kinetics.advance`, bounded backward Euler, rate at the
  end temperature, one sub-step) at the interpolated temperature.
* **Coupling**: monolithic Newton on the nodal temperatures with the cure state eliminated at each quadrature point
  (Jacobian from the implicit-function derivative dα/dT = Δt f_T/(1 − Δt f_α)). Stop when ‖R_free‖∞ ≤ 1e-12 · scale +
  rounding noise; at most 25 iterations. A rejected step (no convergence, kinetics contract violated, singular solve)
  raises `StepRejected` and changes nothing (T, α, t, history are committed only after convergence; committed arrays are
  read-only).
* **Energy budget** per step (stored, reaction, Robin inflow at the end of the step, Dirichlet reservoir heat).
* **One-way residual stress** (`residual_stress`): the accepted thermal history is replayed through `HistoryProblem` with
  `CuringThermoElasticPoint` (thermal and chemical eigenstrain, CHILE-type stiffness), plane strain, constrained in-plane
  (u_x = 0 everywhere) or with rigid-body supports only. The same quadrature points and interpolation are used, so the
  cure state of the mechanical model equals the thermal one (measured: bitwise).

Coupling level (WP17 H05): heat + cure source (level 3) and a one-way mechanical solve (level 2 chain). Omitted feedback:
mechanical dissipation, thermoelastic coupling, deformation-dependent conduction, chemical shrinkage acting on the heat.

## What is verified (first results in the prereg; all thresholds met on the first run)

| Check | Result |
|---|---|
| TS7a adiabatic limit, k = 0.2 and k = 0, closed recurrence (Q = 0) and scalar `brentq` coupled solve (Q ≠ 0) | rise error ≤ 2.2e-14 and 2.0e-13 (threshold 1e-10); nodal spread ≤ 3.4e-13 K |
| TS7b reference accuracy (N = 400 vs 800, rtol 1e-9 vs 1e-11) | 2.4e-7 / 9.6e-9 of the exotherm |
| TS7b time order (BE, n = 32) | p = 1.00, 1.02 (pairs 25→12.5, 12.5→6.25) |
| TS7b space order (Q4, time error extrapolated away) | p = 2.03 (4→8), 2.01 (8→16); E_T(16) = 3.9e-4 |
| TS7b Dirichlet surface | E_T = 7.7e-4, E_α = 3.9e-4 |
| TS7b negative control (no exotherm) | E_T = 1.0, E_α = 0.20: detected |
| TS7c energy balance per step | max mismatch 3.7e-12; without the reaction term 1.0; with the explicit-in-time flux 9.2e-3 |
| TS7d rollback | bitwise unchanged, retry equals a fresh run |
| TS7e uniform constrained / free | σ = −E e/(1−ν) to 8.9e-15 of the scale; free slab 1.8e-15 |
| TS7f one-way consistency | α identical, element mean of σ_yy 5.5e-15 of σ_xx |
| TS7g Newton | ≤ 3 iterations per step (limit 8), Jacobian vs central FD 3.5e-11 |

The independent reference is a method of lines (finite differences on 800 intervals, `solve_ivp` BDF, rate law coded in the
test): a different spatial discretisation, time integrator, source quadrature and cure integrator. The solver reproduces it
with the orders expected of backward Euler (first) and linear elements (second).

## Mechanics: what the free slab means

`constraint="free"` leaves rigid-body supports only. At a uniform state this gives exactly zero stress (checked). With a
through-thickness gradient the free strip is a finite block, not an infinite plate: it carries self-equilibrated stress and
is not the "free plate" of lamination theory (no curvature or in-plane stretch DOF is imposed). The stress-free reference is
T_ref of the expansion law and α_gel of the shrinkage law; the model does not decide when the network forms.

`HistoryProblem` stops on a force-relative residual. A state that is nearly stress free has rounding-level internal forces,
so `residual_stress` accepts a `force_scale` (N) applied to a supported DOF as a reference force scale (it does not change
the stresses). The free-slab check uses 1e3 N; this is a set-up choice recorded in the prereg.

## NOT implemented / not claimed

* Vitrification-controlled (diffusion-limited) kinetics, and any dependence of k, c_p, ρ on T or α.
* Viscoelastic stress relaxation during cure; the stiffness is the synthetic CHILE interpolation, which is not a physical
  claim for an uncured resin.
* Finite strain (chemical and thermal eigenstrain are small-strain, linear = volumetric/3).
* Release (de-moulding) event, tool-part contact, warpage and spring-in, laminate curvature (WP18/19 CLT), 3-D meshes.
* Design sensitivities, adaptive time stepping, YAML integration.
* No claim of composite process validation: no measured data are used.
