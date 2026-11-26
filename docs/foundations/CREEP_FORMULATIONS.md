# Irreversible creep: formulations and verification (WP13 slice C_creep)

Code: `src/toporlogy/core/creep.py`. Tests: `tests/toporlogy/test_creep.py`. Preregistration and first results:
`prereg/C_creep.md`. Ledger entries of the two failed predictions: `ledger_additions/C_creep.json`.
All constants in tests and examples are **synthetic**; nothing here is material data. Recoverable (viscoelastic)
creep is a different mechanism and a different module; do not add a Norton law to a Prony model for the same
measured compliance (double counting).

## Units and conventions

Stress, moduli: Pa. Time: s. Temperature: absolute, K. Activation energy Q: J/mol, R = 8.314462618 J/(mol K).
A: 1/s (the rate at q = σ_ref, T = T_ref). Tension positive; Voigt order [xx, yy, zz, yz, xz, xy], physical shear
stress, engineering shear strain. Physical time comes from `MaterialRequest.step = TimeStep(t, dt, T, T_prev)`;
a load factor is never a time. `physical = True`: the models refuse the legacy element-modulus scaling of
`NonlinearElasticity`.

## Laws

ε = ε_e + ε_cr, σ = K tr ε 1 + 2G (dev ε − ε_cr), ε_cr deviatoric.
q = √(3/2 s:s), N = ∂q/∂σ = 3s/(2q), ε̇_cr = ṗ N, p ≥ 0 accumulated scalar creep strain.
Hydrostatic stress gives s = 0 and no creep (pressure-sensitive creep would be a different law).

* **Norton** (`NortonCreep`): ṗ = A f(T) (q/σ_ref)^n, A ≥ 0, σ_ref > 0, n > 0.
  Arrhenius f(T) = exp[−Q/R (1/T − 1/T_ref)] (`ArrheniusFactor`), evaluated as a log; `factor` refuses an exponent
  above 700 instead of returning inf. At constant uniaxial stress σ and T: ε_cr,axial = sign(σ) A f (|σ|/σ_ref)^n t,
  lateral −½ of it.
* **Norton–Bailey** (`NortonBaileyCreep`), constant-stress law ε_cr = B (q/σ_ref)^n (t_load/t_ref)^m, m > 0
  (m < 1 primary creep, m = 1 steady). Isothermal (B is the value at the temperature of the data).
  * time hardening: ṗ = c(q) Φ'(t_load), c = B (q/σ_ref)^n, Φ = (t/t_ref)^m. Backward Euler uses the *exact
    increment* ΔΦ = Φ(t_load + Δt) − Φ(t_load), evaluated as (t_load/t_ref)^m expm1[m log1p(Δt/t_load)] (and
    (Δt/t_ref)^m at t_load = 0): no ε at t = 0 although Φ' is singular there for m < 1.
  * strain hardening, from the same constant-stress law: at constant c, p = c (t/t_ref)^m ⇒ the time is a function
    of p, ṗ = (m/t_ref) c^{1/m} p^{1−1/m} (singular at p = 0 for m < 1). With w = p^{1/m} the law becomes
    ẇ = c(q)^{1/m}/t_ref, regular at w = 0, so Δp = (w_n + Δt c(q)^{1/m}/t_ref)^m − p_n is again exact for constant q
    and handles p = 0 without a regularisation.
  * The two variants coincide at constant stress and differ after a stress change (time hardening keeps the
    clock, strain hardening keeps the accumulated strain). Synthetic example (tests C5b): q = 1e7 Pa to 100 s,
    then 2e7 Pa to 400 s: p = 2.85e-3 (time) vs 4.39e-3 (strain), 54 % apart.
  * `t_load` is a state field: the time since the load was applied, +Δt for every committed step whether or not a
    load acts (a clock, not a load detector); it starts at 0 at the first step and is restarted by
    `reset_load_time(committed, t, mask)`. It is not the absolute time of the `TimeStep`. For a model without a
    load history reset (a new load level, a reload after a long unloaded period) the user decides whether the clock
    is reset; the model does not guess.

## Backward Euler update (isotropic elasticity, radial)

s_tr = 2G (dev ε − ε_cr,n), q_tr = √(3/2)‖s_tr‖, N = 3 s_tr/(2 q_tr) (the direction of the trial deviator is the
direction at the end of the step). Both laws give a creep increment h(q) of the step as a function of the **end**
stress q (non-negative, non-decreasing, h(0) = 0):

* Norton: h = Δt A F̄ (q/σ_ref)^n, F̄ the Arrhenius mean over the step (below).
* Norton–Bailey time hardening: h = B ΔΦ (q/σ_ref)^n; strain hardening: h = (w_n + δ(q))^m − w_n^m,
  δ = (Δt/t_ref)(B (q/σ_ref)^n)^{1/m}.

With q = q_tr − 3GΔp the update is s = θ s_tr, θ = q/q_tr, ε_cr ← ε_cr + Δp N, p ← p + Δp, and Δp is the root of

  g(Δp) = Δp − h(q_tr − 3GΔp) = 0.

g' = 1 + 3G h' ≥ 1, g(0) = −h(q_tr) ≤ 0, g(q_tr/3G) = q_tr/3G > 0 (h(0) = 0): the root is unique, in
[0, cap], cap = min(q_tr/3G, h(q_tr)) (h non-decreasing). The solver is a safeguarded Newton iteration started at
cap, with bisection whenever the Newton point leaves the maintained bracket or is not finite (h' = inf or nan at
q = 0 for n < 1), stopping at |g| ≤ `local_tol` · cap (default 1e-13; since g' ≥ 1 this bounds the error of Δp)
and raising `ReturnMappingFailure` after `local_max_iter` iterations. Δt = 0 or q_tr = 0: no evolution (state and
stress are the elastic ones, tangent = elastic). No clipping of any physical input; invalid parameters raise.
Powers are evaluated without overflow trapping (inf is a valid upper value of h; the bracket handles it).

Dissipation increment σ_{n+1}:Δε_cr = q Δp ≥ 0 (no stored creep energy); `energy` is the elastic strain energy
½ σ:(ε − ε_cr).

## Algorithmic tangent (with the change of direction)

Since Δp depends only on q_tr, s = θ(q_tr) s_tr, ds_tr = 2G dev dε, dq_tr = N:dε. Hence
ds = 2Gθ dev dε + 2G θ' s_tr (N:dε), with s_tr = (2q_tr/3) N and θ' = −3G(Δp' q_tr − Δp)/q_tr², where
Δp' = dΔp/dq_tr follows from differentiating g = 0: Δp'(1 + 3G h') = h' ⇒ Δp' = h'/(1 + 3G h') (h' at the end
stress). Therefore

  C = dσ/dε = K 1⊗1 + 2Gθ I_dev − 4G² (Δp' − Δp/q_tr) N⊗N.

The N-direction change is the factor θ on the whole deviatoric part (and the Δp/q_tr term); the "scalar" tangent
with N frozen, K 1⊗1 + 2G I_dev − 4G² Δp' N⊗N, is not the derivative of the update (test C3e shows its FD error is
0.15). Limits: h' → ∞ gives C = K 1⊗1 + 2Gθ (I_dev − (2/3) N⊗N); n = 1 gives an isotropic tangent. In Voigt form
N⊗N uses the tensor components of N (not doubled), consistent with engineering-shear strain and physical-shear stress.

## Temperature in a step

`ArrheniusFactor.step_average(T0, T1)` = ∫₀¹ f(T0 + (T1−T0)s) ds, T linear in time over the step
(`TimeStep.temperature_prev` → `temperature`): composite 8-point Gauss–Legendre, the number of equal panels doubled
(1, 2, 4, … ≤ 1024) until two successive estimates agree to a relative 1e-12 for all points, otherwise
`ConstitutiveFailure`. At fixed stress (stress control) the step average is the exact ∫A f(T(t))dt of the step; when q
and T vary together within a step the model applies the mean factor to the end-of-step stress law (first order, like
backward Euler), so the step must resolve a simultaneous change of q and T. `integrate_creep` provides that
resolution through its error control. K and G are constants here (temperature-dependent moduli: not implemented).

## Adaptive driver for long holds

`integrate_creep(model, control, segments, output_times, rtol, atol, ...)` with piecewise-constant mixed targets
(`LoadSegment`, T linear in each segment) uses `drive_material_point` for every sub-step:

* one attempt = one full step on a scratch copy of the committed state and two half steps on the real
  `StateStore`; error e = ‖ε_cr,full − ε_cr,halves‖_F / (rtol ‖ε_cr‖_F + atol); the two-half-steps result is
  kept when e ≤ 1; next step Δt · clip(0.9 e^(−1/2), 0.2, 5) (×5 if e = 0), never shrunk by an event-clipped step
  unless the error says so;
* segment ends and output times are events: no step crosses one, a step that reaches one ends on it exactly (t is
  assigned the event time), so output times and a release time are in the accepted grid bitwise;
* a rejected step (e > 1 or a failed local/mixed-control solve) is rolled back with `store.restore(checkpoint)` +
  `rollback()`; nothing of it stays in the committed state;
* cap: `max_steps` attempts and a resolvable step size, otherwise `CreepStepLimit`; log-spaced outputs
  1e-6 … 1e8 s need ~300 accepted steps for the synthetic relaxation, not 1e14;
* restart: pass `store`, `strain0`, `t0`, `dt_next` (all in `CreepRun`) and the later segments; the result equals the
  uninterrupted run bitwise (test C7c);
* the accepted grid contains both half-step ends with their targets and temperatures, so it can be replayed by one
  `drive_material_point` call (test C7b).

**rtol is a local tolerance.** In the synthetic relaxation the global error of ε_cr at the outputs was
2.1e-3 / 6.7e-4 / 2.1e-4 for rtol 1e-4 / 1e-5 / 1e-6 (≈ 0.2 √rtol), and the error of the long-time stress tail
relative to the *current* stress is larger than relative to q0 (the controller measures ε_cr). No lifetime or
long-time extrapolation of short curves is implied; numerical convergence here says nothing about material-data
validity.

## What is verified (synthetic, first run 2026-11-27; details and numbers in the preregistration)

| check | result |
|---|---|
| C2 uniaxial rate 8e-6 1/s, strain 8e-4 at 100 s, compression sign, hydrostatic zero, shear, Arrhenius, dt = 0 / q_tr = 0, invalid input | PASS (errors ≤ 1.4e-14) |
| C3 held-strain relaxation n = 1 and n = 3 vs closed form; BE first order (orders 0.99–1.00 for n = 1, 0.98–1.00 for n = 3); tangent vs central FD (2.6e-11) with a negative control (0.15); local residual; high exponent / near-zero stress; dissipation | PASS; one preregistered claim failed (C3g full relaxation, see ledger) |
| C4 step-average vs `quad` (≤ 4.9e-15), ramp T at fixed stress with 1–8 steps (≤ 3.9e-14); end-point rule is detected as wrong (error 4.0) | PASS |
| C5 exact increment from t = 0 (1.1e-14), time vs strain hardening vs closed forms and `solve_ivp` (≤ 4.2e-14), NB relaxation first order in Φ, NB tangents vs FD (≤ 8.8e-12) | PASS |
| C6 long relaxation 1e-6…1e8 s, 325 accepted steps, outputs on the grid; hold–release with exact event (35 steps, p error 6.1e-12) | steps/events PASS; **preregistered global accuracy 1e-3 FAIL** (2.1e-3), strict xfail |
| C7 rollback bitwise, rejected steps leave no trace (replay bitwise), restart bitwise (Norton relaxation and NB stress step) | PASS |

## Not implemented

Tertiary creep and creep damage; creep rupture and life correlations (Larson–Miller etc.); creep–plasticity
coupling (WP13 C8: no coupled local residual or consistent coupled tangent exists, and sequential independent
updates are not offered); Garofalo / sinh and diffusion-mechanism laws; temperature-dependent K, G and
temperature-dependent n, Q; pressure-sensitive creep; anisotropic creep; finite-strain creep; FE coupling and
adjoint through the creep history (WP17/WP20). `core.creep.NOT_IMPLEMENTED` lists them.
