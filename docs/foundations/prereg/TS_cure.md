# TS_cure — cure kinetics, glass transition, reaction heat, chemical shrinkage, Avrami / Nakamura (preregistration)

Preregistered 2026-11-27 01:54 +0300 (output of `date "+%Y-%m-%d %H:%M %z"`), before any test of this slice was run.
Base: `89e40bd`. Module `core/cure.py` (new), tests `tests/toporlogy/test_cure.py`, docs
`docs/foundations/CURE_AND_CRYSTALLISATION.md`. Plan rows: WP16 TS01-TS03 (TS04-TS07 FE coupling are NOT in this
slice), WP15 TP03 (crystallisation kinetics); acceptance cases TS1-TS6, TP3, TP4 of 23_ACCEPTANCE_CASES.md. Every
number is **synthetic**; none is material data.

Conventions: temperature absolute (K); rate constants in 1/s; Arrhenius k = A exp(−Q/(R T)), Q in J/mol, R = 8.314462618
J/(mol K); conversion α ∈ [0, 1]; ΔH_total in J per kg of resin, exothermic positive; chemical shrinkage β is the
volumetric strain per unit conversion (β < 0 shrinkage), small strain, linear part β/3; tension positive; Voigt
[xx, yy, zz, yz, xz, xy]. Avrami: X = 1 − exp(−k t^n), k in s^−n, X the RELATIVE transformed fraction.

## Design statements that the checks test

* `advance` is backward Euler on dα/dt = f(α, T_{n+1}) (T linear in the step; `substeps` m ≥ 1 sub-divides the step and
  linearly interpolates T). For f ≥ 0, f(1, T) = 0 (exponent n > 0), g(x) = x − α_n − h f(x, T) has g(α_n) = −h f(α_n) ≤ 0
  and g(1) = 1 − α_n ≥ 0, so a root exists in [α_n, 1] by the intermediate value theorem. The code solves for it with a
  safeguarded Newton iteration that keeps a bracket [lo, hi] ⊆ [α_n, 1] and only accepts iterates inside it. The result
  is therefore in [α_n, 1] by construction (monotone, bounded, unconditionally in h); nothing is clipped. A solve that
  does not converge raises `ReturnMappingFailure`; an input outside the contract raises `ValueError`.
* Nakamura is integrated in θ = ∫K dt. With X = 1 − exp(−θ^n): dX/dt = n θ^{n−1} K exp(−θ^n) = n K (1−X)
  [−ln(1−X)]^{(n−1)/n}. For T linear in the step ∫K dt over the step = dt · (∫_{T_n}^{T_{n+1}} K dT)/(T_{n+1} − T_n),
  exact for constant, piecewise-linear and polynomial K (the `PropertyFunction.integral`). θ is non-decreasing for K ≥ 0
  and X ∈ [0, 1] by construction (`-expm1(-θ^n)`). No epsilon at X = 0.
* Chemical strain ε_ch(α) = β max(α − α_gel, 0) (volumetric), linear β/3 per axis, `alpha_gel` ∈ [0, 1) default 0
  (the stress-free conversion is model-defined, not assumed).
* `CuringThermoElasticPoint`: σ = C(T, α):(ε − ε_th(T) 1 − ε_ch,lin(α) 1), C = K 1⊗1 + 2G I_dev with K = (1−α)K_u(T)
  + α K_c(T), G likewise (synthetic CHILE-type linear interpolation; the uncured state has a numerical minimum
  stiffness, not a physical claim). State α, one-way coupling (T prescribed, α does not depend on ε). Algorithmic
  tangent = C(T_{n+1}, α_{n+1}).

## Checks

Independence: closed forms derived here, `scipy.integrate.solve_ivp` (DOP853, rtol 1e-12, atol 1e-14) written in the
test with the test's own rate law and Arrhenius formula, `scipy.integrate.quad`, a dense/hand numpy stiffness, central
FD written in the test. A comparison with a recurrence of the production scheme derived below is labelled
"derivation" (the recurrence is a hand-derived closed form of backward Euler, not a call of the production code).

| ID | Hypothesis | Reference | Metric / threshold |
|---|---|---|---|
| TS1a | n = 1, α0 = 0.2, k = 0.1 1/s: α(10 s) = 1 − 0.8 e^{−1} (closed form `isothermal_first_order`) | closed form in the test | rel ≤ 1e-14 |
| TS1b | backward Euler, n = 1, constant k: 1 − α_{j+1} = (1 − α_j)/(1 + k Δt); the production `advance` equals this recurrence for 10 steps of 1 s and for a T-dependent k (k(T_{j+1}) from the test's own Arrhenius, Q ≠ 0) | derivation (hand recurrence) | abs ≤ 1e-12 |
| TS1c | monotone and bounded: every α_j ∈ [0.2, 1], α_{j+1} ≥ α_j, for Δt = 0.01, 1, 100 s | the sequence | exact inequalities |
| TS1d | first-order time convergence: Δt = 2, 1, 0.5, 0.25, 0.125, 0.0625; error vs the exact α(10); with s(x) = (t/Δt)(x − ln(1+x)), x = kΔt, err = −0.8 e^{−1}(e^{s} − 1) ≈ −0.0147 Δt: observed order p = log2(err(Δt)/err(Δt/2)) in [0.9, 1.1] for every pair with Δt ≤ 1; err(0.0625) ≤ 1.2e-3; err(0.0625)/(0.8 e^{−1} · 0.05 · 0.0625) within 5 % of 1 | closed form (this is the derivation) | as stated |
| TS1e | negative control: explicit Euler α + Δt k (1 − α), Δt = 100 s, written in the test, gives α > 1 (value 8.2, derived: 0.2 + 100·0.1·0.8); the production `advance` with the same step stays in [0.2, 1] | test explicit Euler | exact inequality; α_explicit = 8.2 rel ≤ 1e-14 |
| TS1f | `isothermal_nth_order`: n = 2, α0 = 0.2, k = 0.05: (1−α)^{−1} = (1−α0)^{−1} + k t; n = 0.5, α0 = 0.2, k = 0.05: (1−α)^{0.5} = (1−α0)^{0.5} − 0.5 k t, reaching 1 at t* = (1−α0)^{0.5}/(0.5 k) and α = 1 for t ≥ t* (documented equilibrium branch); vs `solve_ivp` for t < t*, closed form t* | solve_ivp rtol 1e-12; closed form | rel ≤ 1e-9 (ODE), t* rel ≤ 1e-14, α(t ≥ t*) = 1 exact |
| TS2a | Kamal–Sourour k1 = 0.01, k2 = 0.05 1/s at T_ref = 400 K, Q1 = 60 kJ/mol, Q2 = 50 kJ/mol, m = 1, n = 2, ramp T(t) = 380 + 0.2 t K, t ∈ [0, 300 s], α0 = 0 and α0 = 0.1; reference `solve_ivp` of the test's own rate law at the step times | ODE | Δt = 300/N, N = 240, 480, 960: observed order p over (240→480) and (480→960) in [0.8, 1.2]; max-over-steps abs error at N = 960 ≤ 2e-2 |
| TS2b | boundedness at coarse steps: N = 3 and N = 1 (Δt = 100, 300 s): all α ∈ [α0, 1], nondecreasing | the sequence | exact inequalities |
| TS2c | autocatalytic start: k1 = 0, α0 = 0 stays exactly 0 (trivial equilibrium, documented), α0 = 1e-3 grows; α0 = 1 stays 1 | exact | exact |
| TS2d | ∂f/∂α and ∂f/∂T (production `drate_dalpha`, `drate_dT`) vs central FD in the test (h_α = 1e-6, h_T = 1e-3 K) for NthOrder (n = 1.5) and Kamal (m = 1.3, n = 2.2) at α = 0.3, 0.7 and T = 400 K | FD | rel ≤ 1e-6 |
| TS3 | DiBenedetto Tg0 = 250 K, Tg∞ = 420 K, λ = 0.5: Tg(0) = 250, Tg(0.5) = 250 + 170/3, Tg(1) = 420; dTg/dα = (Tg∞ − Tg0) λ / (1 − (1−λ)α)² = 85, 170·0.5/0.5625 = 151.111..., 340 at α = 0, 0.5, 1 (derivation: with D = 1 − (1−λ)α, d(λα/D)/dα = λ/D²); FD (central h = 1e-5 at α = 0.25, 0.5, 0.75; 3-point one-sided h = 1e-4 at the ends); λ = 1 gives the linear law; inverse `alpha_at_Tg` round trip | closed form; FD | values and derivative rel ≤ 1e-14; FD rel ≤ 1e-6; round trip rel ≤ 1e-13 |
| TS4a | ΔH = 200 kJ/kg, c_p = 2 kJ/(kg K) constant, Δα = 0.5: adiabatic rise 50 K; heat of a step ΔH Δα; negative Δα refused | arithmetic | rel ≤ 1e-14 |
| TS4b | c_p(T) = 1800 + 4 (T − 300) J/(kg K) (`Polynomial`, domain [300, 500]), ΔH = 2e5, Δα = 0.5, T0 = 300: 2ΔT² + 1800ΔT − 1e5 = 0, ΔT = (−1800 + √(1800² + 8e5))/4 = 52.4937...; adiabatic Kamal cure (Arrhenius as TS2, T0 = 350 K) integrated by `solve_ivp` as the system (α, T) with T' = ΔH α'/c_p(T); at 25 output times the production `adiabatic_temperature(T0, 0, α(t), c_p)` equals the ODE temperature | closed form; ODE | quadratic rel ≤ 1e-12; ODE T rel (of the rise) ≤ 1e-9 |
| TS4c | heat budget: Σ over steps of ΔH Δα_j telescopes to ΔH (α_N − α_0) (rel ≤ 1e-12); the volumetric source q = ρ ΔH dα/dt integrated by `quad` over the exact first-order solution (ρ = 1200 kg/m³) equals ρ ΔH (α(T) − α0) (rel ≤ 1e-10); c_p is sensible heat only (the reaction heat enters once, as the source) | quad; closed form | as stated |
| TS5a | ChemicalShrinkage β = −0.03: volumetric ε(α = 1) = −0.03, linear −0.01 (β/3); increments telescope along a path (Σ = β(α_N − α_0)/3, rel ≤ 1e-13); `alpha_gel = 0.3`: zero for α ≤ 0.3, β(α − 0.3) above; shrinkage sign negative | closed form | rel ≤ 1e-14 |
| TS5b | free shrinkage + thermal: all six stresses controlled to 0, T ramp 300 → 340 K (T_ref = 300, CTE 5e-5 1/K), α from the n = 1 recurrence of TS1b (k = 0.1, Q = 0, 20 steps of 1 s): stress = 0, ε = [e, e, e, 0, 0, 0], e = 5e-5 (T − 300) + (β/3) α_j | derivation (hand recurrence) | strain abs ≤ 1e-12; stress abs ≤ 1e-5 Pa (driver tol 1e-6) |
| TS5c | fully constrained (all strains 0), T = T_ref: σ_xx = σ_yy = σ_zz = −3 K(α_j) (β/3) α_j, K(α) = (1−α)K_u + α K_c with K = E/(3(1−2ν)) (E_u = 1e5 Pa, ν_u = 0.45, E_c = 3e9 Pa, ν_c = 0.35), positive (tension) for shrinkage; shear 0; with `alpha_gel` = 0.3: zero stress while α_j ≤ 0.3 | closed form with the recurrence | rel ≤ 1e-12 |
| TS5d | contract and tangent: `physical = True`, `kinematics = "small"`, `state_fields = ("alpha",)`, no step / no temperature raises, energy and dissipation None; tangent equals the dense K 1⊗1 + 2G I_dev at (T_{n+1}, α_{n+1}) (rel ≤ 1e-12) and the FD dσ/dε at fixed committed state (h = 1e-7, rel ≤ 1e-6) | dense numpy; FD | as stated |
| TS6a | rejection: `evaluate` with a step produces trial α ≠ committed α while `store.committed["alpha"]` is unchanged (bitwise) before commit and after `rollback`; committed arrays read-only; repeating the evaluation reproduces the trial (deterministic) | exact | `np.array_equal` |
| TS6b | the committed α is constant during the Newton loop of a stress-controlled step (probe wrapper records it at every evaluation); one commit per step | exact | exact |
| TS6c | history: `drive_material_point` with times and temperatures (ramp 380 → 420 K in 20 s, hold 10 s; n = 1 Arrhenius k_ref = 0.05 at 400 K, Q = 70 kJ/mol) equals the recurrence with the test's own k(T_{j+1}) (abs ≤ 1e-12); restart from a checkpointed store gives bitwise the same α; negative control: hold-then-ramp gives a different α_N (difference > 1e-3) | derivation; exact | as stated |
| TS6d | invalid input raises ValueError: α outside [0, 1] or NaN, dt < 0, T ≤ 0 or NaN, Q < 0, A < 0, n ≤ 0, m < 0, non-positive reference rate or T_ref, DiBenedetto λ ≤ 0 / Tg∞ ≤ Tg0 / Tg0 ≤ 0 / α outside [0, 1], ΔH ≤ 0 or non-finite, c_p without `positive=True` or wrong units, adiabatic rise leaving the c_p domain, β ≥ 0 allowed only as swelling (β > 0 accepted: documented, not refused), alpha_gel ∉ [0, 1), substeps < 1, Avrami/Nakamura k ≤ 0, n ≤ 0, t < 0, X outside [0, 1], K with wrong units, step increment of θ < 0 | exact | exact |
| TP3a | Avrami k = 0.01 s^−2, n = 2: X(10 s) = 1 − e^{−1} (k t^n = 1); X(0) = 0; monotone; `avrami_time` inverse round trip t = 10 s; for n = 1, 2, 2.5, 3 with K = 0.1 1/s (k = K^n): X(5 s) = 1 − exp(−k 5^n) | closed form | rel ≤ 1e-13 |
| TP3b | Nakamura rate at constant T: n K (1−X) (−ln(1−X))^{(n−1)/n} at t = 10 s equals the derivative of Avrami n k t^{n−1} (1−X) = 0.02·10·e^{−1} = 0.2 e^{−1} | closed form | rel ≤ 1e-13 |
| TP4a | constant T: `Nakamura.advance_theta` / `advance` from X = 0 reaches Avrami with 1 step of 10 s and with 100 steps of 0.1 s (independent of the step size: exact in θ), n = 2, K = 0.1 | closed form | abs ≤ 1e-13 |
| TP4b | varying T: K(T) piecewise-linear (K(350) = 0.01, K(400) = 0.1, K(450) = 0.01 1/s), T(t) = 440 − 4 t K over 20 s (crosses the kink at t = 10 s), n = 2, X0 = 0; production in 40 steps of 0.5 s vs θ' = K(T(t)) by `solve_ivp` (two segments split at the kink), X = 1 − exp(−θ^n) | ODE in θ | abs ≤ 1e-10 |
| TP4c | varying T, X-form ODE: with X0 = 0.01 (so the right side is smooth) `solve_ivp` of dX/dt = n K(T(t)) (1−X)(−ln(1−X))^{(n−1)/n} (two segments) vs production | ODE in X | abs ≤ 1e-8 |
| TP4d | negative control: the X-form ODE from X0 = 0 with n = 2 has right side 0 at X = 0 (explicit Euler / `solve_ivp` stay at X = 0 and miss Avrami by > 0.5 at t = 10 s: 1 − e^{−1} = 0.632); the θ-form is exact | test explicit Euler | X_naive = 0 exact; miss > 0.5 |

Expected limiting behaviour: backward Euler is only first order; the order window tolerates pre-asymptotic drift.
Failure modes to watch: sign of ε_ch in the constrained stress; the n = 1 recurrence using k(T_{j+1}) (end-of-step
temperature) while a different convention is documented; the Nakamura exponent (n−1)/n versus n−1.

Run: `python -m pytest tests/toporlogy/test_cure.py -q -p no:randomly -p no:cacheprovider` (TMP/TEMP and basetemp under
`.claude/tmp-toporlogy/TS_cure`).

First result (append after run): see below.
