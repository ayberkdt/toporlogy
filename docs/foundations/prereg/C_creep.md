# C_creep — Irreversible creep: Norton, Norton–Bailey, temperature, long-time integration (preregistration)

Preregistered 2026-11-27 01:25 +0300, before any creep test was run. Base SHA `25d8df6` (branch
`worktree-agent-a99280705bb8b5bb5`, fast-forwarded to `25d8df6`). Responsible module: `core/creep.py` (new).
Tests: `tests/toporlogy/test_creep.py`. Design and formulas: `docs/foundations/CREEP_FORMULATIONS.md`.
Plan: WP13 (`13_CREEP.md`), acceptance rows C2–C7 of `23_ACCEPTANCE_CASES.md`; C1 (recoverable viscoelastic creep)
belongs to another agent, C8 (creep–plasticity) is explicitly NOT implemented. All constants are synthetic.

## Models and conventions (what is being tested)

Small strain, isotropic elasticity with physical K, G [Pa]; ε = ε_e + ε_cr; σ = K tr ε 1 + 2G (dev ε − ε_cr),
ε_cr deviatoric. q = √(3/2 s:s), N = 3s/(2q), ε̇_cr = ṗ N.

* `NortonCreep(K, G, A, sigma_ref, n, arrhenius=None)`: ṗ = A f(T) (q/σ_ref)^n, A [1/s]. Backward Euler, radial:
  g(Δp) = Δp − Δt A F̄ ((q_tr − 3GΔp)/σ_ref)^n = 0 on [0, min(q_tr/3G, Δt A F̄ (q_tr/σ_ref)^n)], safeguarded
  Newton/bisection, tolerance |g| ≤ 1e-13 × upper bracket end (g' ≥ 1, so the error in Δp is ≤ |g|). F̄ = mean of
  the Arrhenius factor over the step for T linear in t (composite 8-point Gauss–Legendre, panel doubling until two
  successive estimates agree to 1e-12 relative, at most 1024 panels).
* `NortonBaileyCreep(K, G, B, sigma_ref, n, m, t_ref, hardening="time"|"strain")`: constant-stress law
  ε_cr = B (q/σ_ref)^n (t/t_ref)^m. Time hardening: Δp = B (q_{n+1}/σ_ref)^n [Φ(t_load+Δt) − Φ(t_load)],
  Φ = (t/t_ref)^m (exact increment, no ε at t = 0). Strain hardening derived from the same law:
  w = p^{1/m} obeys ẇ = c(q)^{1/m}/t_ref, c = B (q/σ_ref)^n, hence Δp = (w_n + Δt c(q_{n+1})^{1/m}/t_ref)^m − p_n.
  Both are implicit in q_{n+1}. `t_load` is a state field (time since load start, +Δt per committed step, reset by
  `reset_load_time`).
* Algorithmic tangent (derived in the doc, tested here against FD): C = K 1⊗1 + 2Gθ I_dev − 4G²(Δp' − Δp/q_tr) N⊗N,
  θ = q/q_tr, Δp' = h'/(1 + 3G h'), h(q) the increment as a function of the end stress.

## Checks

Notation: "rel" = max-norm error divided by max-norm of the reference. G = 8e9 Pa, K = 1.7e10 Pa (synthetic).
Uniaxial stress via `drive_material_point` with all six components stress-controlled (lateral stresses 0), `tol`
= 1e-6 Pa.

| ID | Hypothesis | Reference (independent of production code) | Threshold |
|---|---|---|---|
| CR1 | `ArrheniusFactor`: factor(T_ref) = 1; factor matches exp[−Q/R(1/T−1/T_ref)] with R = 8.314462618 written in the test; `dfactor` equals a central FD (h = 1e-3 K); `log_factor` finite where the factor overflows; invalid input (Q < 0, T ≤ 0, T_ref ≤ 0, non-finite) raises | formula in the test; FD | rel 1e-13; derivative rel 1e-8; exact raise |
| C2a | A = 1e-6 1/s, σ_ref = 10 MPa, n = 3, uniaxial σ = 20 MPa for 100 s in one step: axial creep strain 8e-4 (rate 8e-6 1/s), lateral −4e-4, p = 8e-4, dissipation q Δp = 1.6e4 J/m³ | closed form A (q/σ_ref)^n t, N = diag(1, −½, −½) | rel 1e-9 on each |
| C2b | compression σ = −20 MPa: axial −8e-4, lateral +4e-4, p = +8e-4 (accumulated scalar positive), dissipation ≥ 0 | same | rel 1e-9; p > 0 exact |
| C2c | hydrostatic stress (σ = 1e8 I) gives zero creep strain and p | exact | abs ≤ 1e-30 |
| C2d | pure shear stress τ = 1e7: q = √3 τ, γ_cr = ε-engineering = 2 ε_xy, p = A (q/σ_ref)^n t; the step is stress controlled | closed form: N_xy = 3τ/(2q) = √3/2 | rel 1e-9 |
| C2e | Arrhenius at T_ref (factor 1 → 8e-4 again) and T = T_ref + 60 K (Q = 5e4 J/mol, T_ref = 300 K), isothermal | A f(T) from the formula in the test | rel 1e-9 |
| C2f | Δt = 0: stress is the elastic response to ε − ε_cr_n, creep strain and p unchanged bitwise; q_tr = 0 (zero deviatoric stress), Δt > 0: no evolution bitwise, tangent = elastic | exact | `np.array_equal` |
| C2g | invalid parameters raise (K, G ≤ 0, A < 0, n ≤ 0, σ_ref ≤ 0, B < 0, m ≤ 0, t_ref ≤ 0, bad hardening name); missing step raises; arrhenius model without temperature raises; negative Δt impossible via TimeStep | exact | ValueError |
| C3a | held deviatoric strain, n = 1, A = 1e-6, σ_ref = 1e7: q(t) = q0 exp(−3G A t/σ_ref), q0 = 3G e0 = 2e7 Pa (e0 = 2e7/(3G)); backward Euler with 1600 steps to 40 s | closed form | rel 2e-3 on q at 40 s (BE error of the finest grid ~ dt/2 · rate² t, dt = 0.025 s) |
| C3b | same, n = 3: q(t) = [q0^−2 + 2·3G A t/σ_ref³]^{−1/2} | closed form | rel 2e-3 at t = 40 s, dt = 0.025 s |
| C3c | BE first-order convergence, n = 3, horizon 40 s, dt = 4, 2, 1, 0.5, 0.25 s: error in q(40 s) decreases monotonically; the observed orders log2(e_k/e_{k+1}) of the three finest pairs lie in [0.9, 1.1]; also n = 1 | closed form of C3a/C3b | order in [0.9, 1.1] |
| C3d | tangent against central FD of the stress with respect to each Voigt strain component (engineering shear), state with nonzero creep strain, Δt = 30 s, n = 3, θ ∈ [0.3, 0.9] (set-up assertion: active, direction-changing branch); h swept over {1e-4, 1e-5, 1e-6, 1e-7}·1e-3 | FD of the production stress (a derivative check, not an independent stress) | best-h max-norm error / max-norm C ≤ 1e-6 |
| C3e | negative control: the tangent with the direction of N frozen, C_fro = C_e − 4G²Δp' N⊗N (no θ and no Δp/q_tr term), written in the test from Δp' = h'/(1+3Gh'), is detected as wrong by the same FD check | FD | error of C_fro ≥ 1e-3 (detects), error of the production tangent ≤ 1e-6 |
| C3f | local root residual: for 30 random states and n ∈ {0.7, 1, 3, 8}, recomputing r = Δp − Δt A (q_new/σ_ref)^n from the returned stress and state (q_new from the returned stress, Δp from the p increment) | independent recomputation in the test | |r|/Δp ≤ 1e-10; root in [0, q_tr/3G] |
| C3g | n = 20 with q_tr = 1.5σ_ref (Δt = 1 s) and q_tr = 1e-3σ_ref: finite stress, state and tangent, p ≥ 0, q_new ≥ 0, residual as C3f; for q_tr = 10σ_ref, n = 20 the update relaxes (q_new < 1e-3 q_tr) without NaN | exact finiteness; residual | as stated |
| C3h | dissipation q Δp ≥ 0 and = σ_new : Δε_cr (tensor contraction) | contraction in the test | rel 1e-12 |
| C4a | `step_average(T0, T1)` against `scipy.integrate.quad` of f(T0 + (T1−T0)s), s ∈ [0,1], epsabs = epsrel = 1e-13: ramps 300→400 K, 300→600 K, 600→300 K, constant T, Q = 6e4 and 2e5 J/mol | quad | rel 1e-10 |
| C4b | variable-T ramp at fixed stress (20 MPa uniaxial), 300→400 K over 10 s, Q = 6e4 J/mol, T_ref = 300 K, steps k = 1, 2, 4, 8: p(10 s) = A (q/σ_ref)^n ∫ f(T(t)) dt | `scipy.integrate.quad`, epsrel 1e-13, in the test | rel 1e-9 for every k (time refinement: results for k = 1 and k = 8 agree to 1e-9) |
| C4c | negative control: the end-point rule Σ Δt f(T_{k+1}) with one step has an error ≥ 1e-2 against the same quad, i.e. the check discriminates | quad | error ≥ 1e-2 |
| C5a | Norton–Bailey time hardening, m = 0.5, B = 1e-4, t_ref = 10 s, n = 3, σ_ref = 1e7, q = 2e7 (c = 8e-4): p(t_k) = c (t_k/t_ref)^m on the grid {1e-6, 0.37, 5, 100} s (first step from t = 0, no ε), and a single step 0 → 1e-6 s | closed form | rel 1e-9 |
| C5b | stress step at t1 = 100 s from q = 1e7 to 2e7, to t = 400 s; both variants. Time hardening p = c1 Φ(t1) + c2 (Φ(t) − Φ(t1)); strain hardening p = (c1^{1/m} t1/t_ref + c2^{1/m} (t − t1)/t_ref)^m. Production matches the closed form of its variant; `solve_ivp` (DOP853, rtol 1e-12, atol 1e-18) of the rate forms ṗ = c m t^{m−1}/t_ref^m from t1 (with p(t1) = c1 Φ(t1)) and ṗ = (m/t_ref) c^{1/m} p^{1−1/m} confirms the closed forms; the two variants differ | closed forms + solve_ivp | production vs closed form rel 1e-9; ivp vs closed form rel 1e-8; relative difference between variants ≥ 0.2 |
| C5c | NB time-hardening relaxation under held deviatoric strain, grid uniform in Φ (t_k = (k Φ_end/N)², Φ_end = 40 s^{m}, t_ref = 1 s, B = 1e-6, m = 0.5, n = 3), N = 4, 8, 16, 32: q(Φ) = [q0^−2 + 2·3G B Φ/σ_ref³]^{−1/2}; orders in [0.9, 1.1] for the two finest pairs | closed form | as stated |
| C5d | tangent FD for both NB variants (state with nonzero p, t_load > 0, Δt = 50 s), as C3d | FD | ≤ 1e-6 |
| C5e | m = 1 time hardening with B, t_ref equals Norton with A = B/t_ref (consistency of two production paths; **regression, not independent**); `reset_load_time` sets t_load to the requested value for the masked points and leaves the rest; t_load advances by Δt per committed step | production vs production; exact | rel 1e-12; exact |
| C6a | long relaxation 1e-6 … 1e8 s (n = 3, held deviatoric strain, q0 = 2e7, A = 1e-6, σ_ref = 1e7), adaptive driver with rtol = 1e-4, atol = 1e-12, 29 log-spaced outputs (2 per decade); output times are in the accepted grid exactly | closed form q(t); ε_cr(t) = (q0 − q)/(3G) (held strain, deviatoric) | accepted steps ≤ 2000; ε_cr rel error ≤ 1e-3 at every output; |q − q_ref|/q0 ≤ 1e-3; a uniform grid of the same resolution would need 1e14 steps (reported, not tested) |
| C6b | long hold + step release (NB, stress controlled, q = 2e7, B = 1e-4, m = 0.5, t_ref = 10 s) hold to 1e4 s, release (σ = 0), run to 1e8 s, outputs log-spaced; release time exactly in the grid; p(t) = c (min(t, t_rel)/t_ref)^m; stress after release ≤ 1e-2 Pa; irreversible strain stays | closed form | accepted steps ≤ 300; p rel 1e-8 |
| C7a | rollback: after one committed step, `checkpoint`; a second step is taken and the store is restored; committed arrays equal the checkpoint bitwise; repeating the step reproduces stress and state bitwise | exact | `np.array_equal` |
| C7b | rejected steps leave no trace: adaptive relaxation run with dt0 so large that ≥ 1 rejection happens (`n_rejected ≥ 1`); replaying the recorded accepted grid with per-step targets and temperatures through a single `drive_material_point` call gives the same final stress, creep strain, p bitwise | replay of the accepted grid (not a second adaptive run) | `np.array_equal` |
| C7c | restart: the run with segments (S1, S2) equals S1 followed by a new call with the checkpointed store, `strain0`, `t0`, carried `dt_next` for S2 (Norton relaxation with segment boundary at 10 s; NB stress step 1e7→2e7 at 100 s), on the accepted grid, stress, creep strain and t_load | the uninterrupted run | `np.array_equal` |
| C8 | creep–plasticity coupling, tertiary creep and rupture are NOT implemented: `core.creep.NOT_IMPLEMENTED` lists them; no coupled model class exists | — | exact |

Expected failure modes (stated before the run): (i) C3a/C3b thresholds assume the BE error ~ dt·(rate·t)/2 and may be
marginal; (ii) C3c orders at the coarsest pair may be outside [0.9, 1.1], which is why only the three finest pairs
are required; (iii) C6a tolerances (1e-3) are the global-error prediction of step doubling with rtol 1e-4 and about
a few hundred steps; if they fail they are reported, not loosened.

Run: `python -m pytest tests/toporlogy/test_creep.py -q -p no:randomly -p no:cacheprovider` (TMP/TEMP and
`--basetemp` under `.claude/tmp-toporlogy/C_creep`). Resource cap: the single-point tests are milliseconds; C6a
≤ 2000 accepted steps × 3 material-point solves.

## First result (2026-11-27, first run of `tests/toporlogy/test_creep.py` after the commit of this file)

Measured with the preregistered thresholds unchanged. 47 passed, 2 strict xfail.

| ID | Result | Measured |
|---|---|---|
| CR1 | PASS | factor vs formula and derivative vs FD within thresholds; overflow raises, log finite; invalid input raises |
| C2a | PASS | rel errors 1.3e-14 (axial, lateral), 1.4e-14 (p), 1.4e-14 (dissipation) |
| C2b | PASS | 1.3e-14, 1.4e-14 |
| C2c | PASS | creep strain 0.0 and p 0.0 exactly |
| C2d | PASS | 8.3e-16 (p), 7.2e-16 (xy component) |
| C2e | PASS | 1.4e-14 (T_ref), 2.1e-15 (360 K) |
| C2f | PASS | bitwise no evolution; tangent equals the elastic one |
| C2g | PASS | all invalid inputs raise ValueError |
| C3a | PASS | 2.9e-6 (n = 1, dt = 0.025 s) |
| C3b | PASS | 5.8e-5 (n = 3) |
| C3c | PASS | n = 1 orders 0.9956, 0.9978, 0.9989, 0.9994; n = 3 orders 0.970, 0.985, 0.992, 0.996 |
| C3d | PASS | theta = 0.609 (set-up assertion held); best FD error 2.6e-11 |
| C3e | PASS | frozen-direction tangent FD error 0.150 (detected) |
| C3f | PASS | n in {0.7, 1, 3, 8}, 30 random states each |
| C3g | **FAIL (xfail) for the sub-claim "q_tr = 10 sigma_ref, n = 20 relaxes below 1e-3 q_tr"**; rest PASS | measured q_new/q_tr = 0.1505. The root agrees with an independent brentq root to 1e-10 (post hoc check), so the claim was wrong (the rate balances the elastic unloading), not the code. Kept as strict xfail, ledger entry added |
| C3h | PASS | |
| C4a | PASS | worst relative error 4.9e-15 |
| C4b | PASS | 2.1e-15, 3.3e-15, 3.9e-14, 3.2e-15 for k = 1, 2, 4, 8; k = 1 vs 8: 1.1e-15 |
| C4c | PASS | end-point rule error 3.99 (discriminates) |
| C5a | PASS | 1.1e-14; single step 0 -> 1e-6 s: 2.1e-16 |
| C5b | PASS | production vs closed form 1.4e-15 (time), 5.9e-16 (strain); solve_ivp vs closed form 4.2e-14, 1.4e-14; variants differ by 54 % (p = 2.846e-3 vs 4.393e-3) |
| C5c | PASS | errors 2.1e-2, 1.1e-2, 5.7e-3, 2.9e-3; orders 0.931, 0.963, 0.981 |
| C5d | PASS | best FD error 8.8e-12 (time), 6.8e-12 (strain) |
| C5e | PASS | m = 1 vs Norton difference 0.0 (regression, not independent); clock semantics exact |
| C6a | steps and events PASS (325 accepted, 0 rejected, 29 outputs exactly on the grid); **global accuracy FAIL (xfail)** | max eps_cr relative error 2.10e-3 at t = 100 s (preregistered 1e-3), max abs(q - q_ref)/q0 = 1.16e-3 (preregistered 1e-3). Step doubling controls the local error; post hoc: 6.7e-4 at rtol 1e-5, 2.1e-4 at 1e-6 (~0.2 sqrt(rtol)). Threshold not loosened; ledger entry added |
| C6b | PASS | 35 accepted steps (1 rejected), p error 6.1e-12, stress after release 1.1e-7 Pa, release time in the grid |
| C7a | PASS | bitwise |
| C7b | PASS | rejections occurred in both runs; replay bitwise (Norton relaxation; mixed control with a temperature ramp) |
| C7c | PASS | bitwise (Norton relaxation restart at 10 s; NB stress step restart at 100 s incl. t_load) |
| C8 | PASS | `NOT_IMPLEMENTED` lists creep-plasticity, tertiary, rupture; no coupled class |

Set-up changes after the first run (inputs, not thresholds): C7b uses `rtol = 1e-4` instead of `1e-6` (the first run took
130 s with 1e-6 in the mixed-control temperature case; the property tested, rejection + replay, does not depend on rtol);
the Arrhenius step average is memoised per step in `NortonCreep` and the Gauss–Legendre nodes are computed once (speed,
no change of values). C3g and C6a were split into the preregistered claim (xfail) and a post hoc characterisation
(labelled post hoc in the test docstring).

Raw evidence: pytest output of the run (`python -m pytest tests/toporlogy/test_creep.py -q -p no:randomly`); measured
values reproduced by a one-off script from the same test helpers. Limitations: synthetic constants only; the step-doubling
error is local; no FE coupling; Norton–Bailey is isothermal.
