# R_rate — physical J2, Perzyna viscoplasticity, Eyring rate law, Drucker–Prager (preregistration)

Preregistered 2026-11-27 01:25 +0300, before any test of this slice was written or run. Base SHA `25d8df6`
(branch of worktree `agent-af97c1b4bfe578628`). Modules: `core/viscoplastic.py` (new: `PhysicalJ2`, `Perzyna`,
`DruckerPrager`, `eyring_*`), tests `tests/toporlogy/test_viscoplastic.py`, doc
`docs/foundations/RATE_DEPENDENT_CONSTITUTIVE.md`. Plan: WP14 (R01–R04), acceptance rows R1–R7 and P2 of
`23_ACCEPTANCE_CASES.md`. The legacy unit-modulus `J2Plasticity` is not changed. All constants are **synthetic**, no
material data.

Conventions: tension-positive stress; Voigt [xx, yy, zz, yz, xz, xy] (stress physical shear, strain engineering
shear); q = √(3/2) ‖dev σ‖; p = −tr σ/3 (compression positive); K, G in Pa; time in s.

## Laws and derivations (what the production code must implement)

* **PhysicalJ2**: f = q − κ(p), κ(p) = σ_y + H p + (σ_∞ − σ_y)(1 − e^(−δp)); radial return Δp from
  q_tr − 3GΔp − κ(p_n + Δp) = 0; algorithmic tangent K 1⊗1 + 2Gθ I_dev − 2G θ̄ n̂⊗n̂, θ = 1 − 3GΔp/q_tr,
  θ̄ = 3G/(3G + κ') − (1 − θ). Stored energy ½ε_e:σ + ψ_h(p), ψ_h = ½Hp² + (σ_∞ − σ_y)(p − (1 − e^(−δp))/δ);
  dissipation increment D = q Δp − [ψ_h(p_{n+1}) − ψ_h(p_n)] (= σ_{n+1}:Δε_p − Δψ_h).
* **Perzyna** (linear hardening κ = σ_y + Hp): ṗ = (1/τ0) ⟨f/σ_ref⟩^m, f = q − κ(p). Backward Euler:
  Δp − (Δt/τ0) ⟨(f_tr − ΞΔp)/σ_ref⟩^m = 0, Ξ = 3G + H, f_tr = q_tr − κ(p_n). With y = 1 − ΞΔp/f_tr ∈ (0, 1]
  (f = f_tr y is the end-of-step overstress): g(y) = 1 − y − β y^m = 0, β = (Δt Ξ/(τ0 σ_ref)) (f_tr/σ_ref)^(m−1);
  g is strictly decreasing, g(0) = 1 > 0, g(1) = −β < 0. m = 1 closed form: Δp = f_tr/(σ_ref τ0/Δt + Ξ). Consistent
  tangent: dΔp/dq_tr = χ = m(1 − y)/(Ξ (y + m(1 − y))), C = K 1⊗1 + 2Gθ I_dev − 6G² (χ − Δp/q_tr) n̂⊗n̂. Limits: y → 0
  (τ0 → 0) χ → 1/Ξ (PhysicalJ2), y → 1 (τ0 → ∞) χ → 0. Dissipation D = qΔp − Δψ_h, ψ_h = ½Hp² (≥ σ_y Δp ≥ 0).
  Δt = 0 gives no flow.
* **Eyring**: rate = A sinh(x), x = τ V/(k_B T); inverse τ = (k_B T/V) asinh(rate/A); log|rate| =
  ln A + |x| − ln 2 + ln(1 − e^(−2|x|)) (log-sinh); k_B = 1.380649e-23 J/K and R = N_A k_B (SI 2019, values read from
  `scipy.constants` in this session: k = 1.380649e-23, N_A = 6.02214076e23, N_A k = 8.31446261815324).
* **DruckerPrager**: f = q − a p − b(κ), b = b0 + hκ, g = q − a_g p, κ̇ = λ̇, ε̇_p = λ̇ ((3/2) s/q + (a_g/3) 1),
  ψ_h = ½hκ². Cone branch (linear): Ξ = 3G + K a a_g + h, Δλ = f_tr/Ξ, f_tr = q_tr − a p_tr − b(κ_n), q = q_tr − 3GΔλ,
  p = p_tr + K a_g Δλ, valid if q ≥ 0. Apex branch (q_tr − 3GΔλ_cone < 0): s = 0, Δλ = (b_n + a p_tr)(−1)/(a K a_g + h),
  Δε_p = dev ε_e,tr + (a_g Δλ/3) 1, σ = (b(κ_{n+1})/a) 1. Cone tangent: C = K 1⊗1 + 2Gθ I_dev
  − 3G√(2/3) n̂ (g_λ − Δλ 2G√(3/2) n̂/q_tr)ᵀ − K a_g 1 g_λᵀ, g_λ = (2G√(3/2) n̂ + aK 1)/Ξ; symmetric iff a_g = a.
  Apex tangent: hK/(aKa_g + h) 1⊗1. Refused: a ≤ 0, a ≥ 3 (no compression yield), a_g ≤ 0 or a_g > a (a_g = 0: the
  apex return is undefined), b0 ≤ 0, h < 0, K or G ≤ 0.
  Admissibility: D/λ̇ = b0 + (a − a_g) p on the cone (q = ap + b ⇒ D/λ̇ = q − a_g p − hκ); for a_g = a it is b0 > 0;
  for a_g < a with h > 0 it can be negative near the apex in tension (not claimed admissible; checked in R6g).

## Checks

| ID | Hypothesis | Reference (independent of production) | Threshold |
|---|---|---|---|
| R1a | PhysicalJ2 uniaxial, lateral stress free (drive_material_point, 60 steps to ε = 0.03), E = 2 GPa, ν = 0.35, σ_y = 20 MPa, H = 100 MPa: σ = min(Eε, (σ_y + Hε)/(1 + H/E)) | closed form (exact for finite steps: radial path) | max abs error / σ_y ≤ 1e-9 |
| R1b (P2) | doubling E (ν, σ_y, H fixed): elastic slope ×2, yield stress unchanged; also the closed form of R1a with E' | bisection of the elastic uniaxial trial on `diagnostics["plastic"]`; slope from the first step | yield: rel ≤ 1e-9 vs 20 MPa for both E; slope ratio 2 within 1e-12; curve error ≤ 1e-9 |
| R1c | load–unload–reverse: unload to ε = 0.015 elastic (σ = σ_peak − E Δε, p unchanged), reverse to ε = −0.02 isotropic closed form p = (2p1 − ε − σ_y/E)/(1 + H/E) | closed form | ≤ 1e-9 σ_y |
| R1d | pure shear: τ = (γ + √3σ_y/H)/(1/G + 3/H) after yield; hydrostatic strain 0.5: no yield, σ = 3K e 1; Voce uniaxial: κ(p) = ε − p... root of κ(p) = E(ε − p) | closed form / `scipy.optimize.brentq` | ≤ 1e-9 σ_y |
| R1e | PhysicalJ2(K, G, σ_y, H [, Voce]) = E_ref × legacy `J2Plasticity(ν, σ_y/E, H/E, ...)` stress, state alpha, on a 3-D multiaxial two-step path (cross-implementation consistency, not a closed form) | legacy `J2Plasticity` | rel ≤ 1e-10 |
| R2a | Perzyna m = 1: Δp (production alpha − alpha_n) equals f_tr/(σ_ref τ0/Δt + Ξ) on a multiaxial 2-step path with trial computed in the test with dense numpy; m = 2 and m = 0.5 equal a `scipy.optimize.brentq` root of the scalar residual (xtol 1e-18, rtol 4ε) | closed form / brentq | Δp rel ≤ 1e-11 (m = 1), ≤ 1e-9 (m = 2, 0.5); stress ≤ 1e-9 σ_y |
| R2b | full 6×6 tangent equals the central finite difference of stress (h = 1e-6 strain, engineering shear for shear components) for m = 1, 2, 0.5 under a multiaxial plastic step from a plastic committed state (also PhysicalJ2) | FD of production stress | max abs diff / max abs C ≤ 1e-6 |
| R2c | below yield: stress = elastic, state unchanged, dissipation 0; Δt = 0: no flow (state identical, stress = trial); missing step → ValueError | elastic closed form | exact (≤ 1e-12 rel) |
| R3a | uniaxial ramp to ε = 0.03 at rates 1e-3, 0.1, 1 s⁻¹, n = 25, 50, 100, 200 increments, Perzyna τ0 = 1 s, σ_ref = 10 MPa, m = 2, E = 2 GPa, ν = 0.35, σ_y = 20 MPa, H = 100 MPa: end stress error vs ODE ṗ = (1/τ0)⟨(E(ε(t) − p) − σ_y − Hp)/σ_ref⟩^m (`scipy.integrate.solve_ivp`, Radau, rtol 1e-12, atol 1e-16): error decreases with every doubling by a factor ≥ 1.5, err(200) ≤ err(25)/5, Richardson 2σ_200 − σ_100 within 3e-3 rel of the ODE value | solve_ivp | as stated |
| R3b | rate sensitivity (selected synthetic parameters, not a polymer claim): end stress strictly increasing with rate for ODE and for production (n = 200); PhysicalJ2 is rate independent (same end stress at all three rates to 1e-12 rel) — negative control of the metric | ODE / PhysicalJ2 | strict ordering; 1e-12 |
| R3c | τ0 = 1e-12 s: Perzyna → PhysicalJ2 (max abs stress difference / max stress ≤ 1e-5, rate 1 s⁻¹, n = 100); τ0 = 1e12 s: Perzyna → elastic (≤ 1e-6 rel, plastic strain ≤ 1e-9) | PhysicalJ2 / elastic closed form | as stated |
| R4a | strain hold at ε = 0.02 for 1e4 s (ramp 20 steps at 1 s⁻¹, then 120 geometric steps): overstress f = σ − κ(p) positive and non-increasing during the hold (tolerance 1e-9 σ_y), stress non-increasing, end stress within 1e-5 rel of the rate-independent equilibrium (σ_y + Hε)/(1 + H/E) (continuous-time estimate of f(1e4 s) ≈ τ0 σ_ref²/(Ξ̃ t) ≈ 5 Pa) | closed form | as stated |
| R4b | hold below yield (ε = 0.005, 1e6 s): no flow (p ≡ 0), stress constant to 1e-12 rel — unlike Norton creep | elastic closed form | exact |
| R5a | Eyring round trip τ → rate → τ for x = τV/(k_BT) ∈ ±{1e-8, 1e-3, 0.5, 3, 50, 300, 700} (V = 1e-27 m³, T = 300 K, A = 1e-3 s⁻¹); rate vs `mpmath.sinh` at 50 digits | mpmath | rel ≤ 1e-12 (round trip), ≤ 1e-13 + 5e-16·x (value; amended before any run, see below) |
| R5b | extreme: x = ±5000 and rate = 1e300, 1e-300: no RuntimeWarning (warnings as errors), log-rate = ln A + |x| − ln 2 (abs ≤ 1e-12 rel to |x|), `eyring_stress_from_log_rate` round trip rel ≤ 1e-12, `eyring_stress(1e300)` = (k_BT/V)(ln 2 + 300 ln 10 − ln A) to 1e-12 rel; negative control: naive `arcsinh(sinh(x))` at x = 800 is not finite | mpmath / math | as stated |
| R5c | units: stress ∝ T at fixed rate (doubling T doubles τ, 1e-14); low-stress Newtonian limit rate = A x (rel 1e-8 at x = 1e-6); high-stress ln limit; molar variant equals atomic with V = V_m/N_A (1e-12); invalid A, V, T ≤ 0 → ValueError | scipy.constants, closed forms | as stated |
| R6a | DruckerPrager uniaxial (lateral free), a = 0.6, b0 = 15 MPa, E = 2 GPa, ν = 0.35, a_g ∈ {a, 0.2}: perfect plasticity plateau tension b/(1 + a/3) and compression b/(1 − a/3); with h = 50 MPa the full curve min(Eε, σ_p), σ_p = (b0 + hε/(1 + a_g/3))/((1 + a/3) + h/(E(1 + a_g/3))) (tension) and the compression analogue with (1 − a_g/3), (1 − a/3) | closed form (exact for finite steps) | max abs error / b0 ≤ 1e-9 |
| R6b | pressure sign (negative control): tension yield < compression yield for a > 0; the sign-flipped closed forms (b/(1 − a/3) in tension) exceed the production tension plateau by ≥ 5 % (the metric of R6a detects them) | R6a | ≥ 5 % |
| R6c | hydrostatic: compression never yields (σ = 3Ke, state unchanged, up to e = −0.05); tension below e_apex = b0/(3aK) elastic, above on the apex branch: s = 0 (≤ 1e-9 σ-scale), σ_iso = (b0 + hΔλ)/a, Δλ = (3aKe − b0)/(aKa_g + h), tr ε_p = a_g Δλ, for h = 0 and h = 50 MPa, a_g ∈ {a, 0.2} | closed form | rel ≤ 1e-12 |
| R6d | consistent tangent = central FD (h = 1e-6) on the cone branch (multiaxial plastic step from a plastic committed state), associated and non-associated, h = 0 and h > 0; on the apex branch (hydrostatic tension + small deviatoric perturbation staying on the apex); associated tangent symmetric (‖C − Cᵀ‖/‖C‖ ≤ 1e-10), non-associated not (≥ 1e-3); negative control: the elastic tangent fails the FD metric (≥ 1e-2) | FD | ≤ 1e-6 |
| R6e | refusals: a ≤ 0, a ≥ 3, a_g ≤ 0, a_g > a, b0 ≤ 0, h < 0, K or G ≤ 0, non-finite → ValueError | — | exact |
| R6f | associated flow: dissipation ≥ −1e-6 J/m³ on a cyclic multiaxial path; non-associated with h = 0 likewise; ε_p volumetric = a_g Δλ on the cone | — | stated |
| R6g | documented non-admissible region: a = 0.6, a_g = 0.06, h = 50 MPa, apex step e = 0.1: D = Δλ(a_g b0/a + hΔλ(a_g/a − ½)) < 0 and equals the closed form | closed form | rel ≤ 1e-12, sign negative |
| R7a | dissipation ≥ −1e-6 J/m³ in every step of cyclic multiaxial paths for PhysicalJ2 (linear and Voce), Perzyna (dt > 0 and dt = 0) | — | stated |
| R7b | rollback/restart: repeated `evaluate` from the same committed state is bitwise identical and leaves the committed arrays unchanged; rejected trial (`rollback`) then the same step gives the same bits; 10 steps in one `drive_material_point` call equal 5 + restart from `checkpoint()` with `strain0` (stress, strain, states `np.array_equal`) for PhysicalJ2, Perzyna (with times), DruckerPrager | — | bitwise |
| R7c | energy balance (extra): ∫σ:dε (trapezoid) − ΔΨ − ΣD → 0 under refinement, uniaxial load–unload of PhysicalJ2, n = 100/200/400: relative to ∫ = |W|: ≤ 1e-3 at n = 400 and decreasing by ≥ 1.5 per doubling | — (the discrepancy is ½ΣΔσ:Δε_p, first order) | stated |
| R7d | `evaluate` (Kinematics, 3-D and plane strain) equals `update` Voigt-packed; `physical = True`, `NonlinearElasticity` refuses the models (contract CT7, only the attribute is checked here) | — | exact |

## Amendments before the first run

Made on 2026-11-27 01:50 +0300 after the prereg commit and **before any test of this slice was executed** (no result
was seen): (1) R5a value threshold: the argument x = τV/(k_B T) is formed in floating point (four roundings, ≤ 4.4e-16
relative), and d ln sinh(x)/d ln x = x coth x ≈ x, so the relative error of any float evaluation of sinh is
≈ x · 4.4e-16 (1.5e-13 at x = 700); the bound 1e-13 was an error in the error analysis. The threshold is
1e-13 + 5e-16·x. (2) R6d normalisation: max abs diff divided by max(max|C|, 1e-3 K) (the apex tangent of h = 0 is
exactly zero, the plain ratio would be 0/0).

Run: `python -m pytest tests/toporlogy/test_viscoplastic.py -q -p no:randomly`. First results are appended below.

## First result (2026-11-27, 45 passed, 1 xfailed)

| ID | Result | Measured |
|---|---|---|
| R1a | PASS | max error / σ_y 3.7e-16 |
| R1b (P2) | PASS | yield stress rel error 4.4e-16 for E and 2E; slope ratio 2 within 2e-12; E-scaled-yield control detected |
| R1c, R1d, R1e | PASS | closed forms ≤ 1e-9 σ_y; legacy cross-check ≤ 1e-10 |
| R2a | PASS | Δp rel error 0 (m = 1), 8.3e-16 (m = 2), 2.9e-16 (m = 0.5) |
| R2b | PASS | FD tangent error 9.8e-12 (m = 1), 4.0e-11 (m = 2), 2.9e-12 (m = 0.5); elastic-tangent control 4.5e-2, 9.8e-2, 2.3e-2 |
| R2c | PASS | elastic below yield, dt = 0 no flow, missing step raises ValueError |
| R3a, rate 0.1 | PASS | end-stress error 0.61, 0.020, 1.5e-3, 2.4e-4 Pa for n = 25, 50, 100, 200; Richardson 3.9e-11 |
| R3a, rate 1 | PASS | error 1.35e4, 5.47e3, 2.43e3, 1.14e3 Pa (ratios 2.5, 2.2, 2.1); Richardson 4.7e-6 |
| R3a, rate 1e-3 | **FAIL, xfail(strict)** | errors 1.9e-8, 3.7e-8, 3.7e-8, 1.5e-8 Pa: 1 ulp of 2.2e7 Pa, no first-order series. The hypothesis, not the production code, is wrong: after the yield transient the response is on the quasi-static steady state (constant overstress, p linear in t) where backward Euler is exact. Ledger entry `ledger_additions/R_rate.json`; a post hoc check (not preregistered) asserts agreement with the ODE to 1e-12 relative |
| R3b, R3c | PASS | ODE end stresses 2.22e7 < 2.48e7 < 3.12e7 Pa; PhysicalJ2 rate independent to 1e-12; τ0 limits within thresholds |
| R4a | PASS | overstress 7.9e6 Pa at hold start, 5.45 Pa at 1e4 s (continuous estimate 4.8 Pa); end stress 2.5e-7 rel to equilibrium |
| R4b | PASS | no flow below yield |
| R5a, R5b, R5c | PASS | round trip 1.7e-16; value error 0.044 of the amended threshold; extreme arguments without warnings |
| R6a | PASS | closed-form error / b0 ≤ 1.2e-15 (tension, compression, associated and non-associated, h = 0 and 50 MPa) |
| R6b | PASS | flipped-sign tension strength is 1.5 times the production plateau |
| R6c | PASS | apex and hydrostatic closed forms ≤ 1e-12 |
| R6d | PASS | cone FD error ≤ 9.5e-11 (associated), ≤ 7.8e-11 (non-associated); asymmetry 1.5e-16 (associated), 0.22 (non-associated); apex FD ≤ 1e-6 |
| R6e, R6f, R6g | PASS | refusals; dissipation ≥ −1e-6 J/m³; inadmissible-region closed form matches |
| R7a, R7b, R7d | PASS | dissipation ≥ −1e-6 J/m³; bitwise rollback and restart; evaluate packing and plane strain |
| R7c | PASS | energy-balance discrepancy / W: 1.3e-3, 5.7e-4, 2.8e-4 for n = 100, 200, 400 |

Set-up fix after the first run (no threshold touched): R6f used a strain (STEP1) that lies on the apex branch for the
non-associated model, while its plastic-volume check needs the cone branch; it now uses a strain with a small trace
and asserts the branch.

## Integrator note (merge, 2026-11-27)

The time written in "Amendments before the first run" ("01:50 +0300") is later than the commit that holds the
amendment (4e2daaa, committer time 01:31 +0300); the text time is not reliable. Order by commit time: prereg 021ffbc
(01:26) → amendment 4e2daaa (01:31) → code, tests and first results f4018f7 (01:42).
