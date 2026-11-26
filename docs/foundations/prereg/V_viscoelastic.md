# V_viscoelastic — linear viscoelasticity, generalized Maxwell, DMA, time–temperature shift (preregistration)

Preregistered 2026-11-27 01:27 +0300 (output of `date "+%Y-%m-%d %H:%M %z"`), before any test of
`tests/toporlogy/test_viscoelastic.py` was run. Base SHA `25d8df6` (branch of this worktree fast-forwarded to it).
Responsible module: `src/toporlogy/core/viscoelastic.py` (new). Documentation: `docs/foundations/RHEOLOGY.md`.
Plan sources (data): WP12 V01–V06 and acceptance rows V1–V10, C1 of `23_ACCEPTANCE_CASES.md`.
All parameters below are **synthetic**; none is a measurement of a real polymer. SI units (Pa, s, K, J/mol).

## Model and conventions (what is being tested)

Prony series M(t) = M∞ + Σ M_i e^{−t/τ_i}, M∞ ≥ 0, M_i ≥ 0, τ_i > 0, M0 = M∞ + ΣM_i > 0, for shear (G) and bulk
(K) independently. σ = 2G∞ e + Σ s_i + (K∞ θ + Σ p_i) 1, ṡ_i + s_i/τ_i = 2G_i ė, ṗ_i + p_i/τ_i = K_i θ̇. For strain
piecewise linear in (reduced) time over a step: a = Δξ/τ, r = e^{−a}, φ(a) = −expm1(−a)/a (φ(0) = 1),
s_{n+1} = r s_n + 2G_i φ Δe, p_{n+1} = r p_n + K_i φ Δθ. Tangent 2(G∞+ΣG_iφ_i) P_dev + 3(K∞+ΣK_iφ_i) P_vol (Voigt of
`core/constitutive.py`: stress physical shear, strain engineering shear). Energy ψ = G∞ e:e + ½K∞θ² + Σ s_i:s_i/(4G_i)
+ Σ p_i²/(2K_i); dissipation rate Σ s_i:s_i/(2G_iτ_i) + Σ p_i²/(K_iτ_i), **integrated exactly over the step** for the
piecewise-linear strain (closed form of a quadratic form in (s_n, Δe), stated in RHEOLOGY.md), not as work − Δψ.
Shift: horizontal only, fixed amplitudes, Δξ = ∫dt/a_T(T(t)) with T linear in the step (isothermal: Δξ = Δt/a_T, exact;
otherwise Gauss–Legendre 5 points on sub-intervals with |Δ ln a_T| ≤ 0.5 each); the update treats the strain as linear in
ξ over the step (exact for T constant in the step; an O(h) effect on stiff branches, O(h²) locally otherwise).

## Synthetic parameter sets

| name | shear (G∞; G_i; τ_i) | bulk (K∞; K_i; τ_i) |
|---|---|---|
| A (V2, V5, V6, V9) | 0.5e6; (1.0e6, 0.5e6); (2, 30) | 2.0e6; (1.5e6, 0.5e6, 1.0e6); (0.5, 8, 200) |
| W (V3, wide Δt/τ) | 0.3e6; (0.4e6, 0.6e6, 0.5e6); (1e-4, 1, 1e4) | 1.0e6; (1.0e6, 2.0e6); (1e-3, 50) |
| AN (V4 analytic) | 1.0e6; none (elastic) | 1.0e6; (3.0e6); (10) |
| NP (V4 non-proportional) | 0.5e6; (0.3e6, 0.2e6); (4, 60) | 1.0e6; (2.0e6, 1.0e6); (10, 150) |
| D (V8 time domain) | 0.6e6; (0.8e6); (5) | 1.5e6; (2.0e6); (2) |
| P (C1b, proportional) | E∞ = 1e6, E1 = 2e6, ν = 0.3 → G, K by isotropic relations; τ = 10 on both | same |
| Z (V6, V10 zero branch) | 0.5e6; (1.0e6, 0.0); (2, 7) | 2.0e6; (0.0, 1.0e6); (1, 9) |

Scalar sets (V1): Maxwell E = 1e6, η = 1e7; Kelvin–Voigt E = 1e6, η = 1e7; SLS E∞ = 1e6, E1 = 2e6, τ = 10;
Burgers E_M = 2e6, η_M = 4e7, E_K = 1e6, η_K = 5e6.

## Checks

| ID | Hypothesis | Independent reference | Metric / threshold |
|---|---|---|---|
| V1a | Scalar creep J(t) and relaxation E_R(t) of Maxwell, KV (creep only), SLS, Burgers equal the response of the defining ODE (spring/dashpot constitutive equations, stated in the test), t ∈ [0, 10 τ_max] | `solve_ivp` DOP853 rtol 1e-13 of the element ODEs; creep: σ = σ0 H(t); relaxation: ε = ε0 H(t) with the jump condition of each model | max rel. error ≤ 1e-9 (relative to the curve maximum) |
| V1b | special values: Maxwell E_R(η/E) = E/e, J(0) = 1/E; SLS E_R(0) = E∞+E1, E_R(∞) = E∞, J(0) = 1/E0, J(∞) = 1/E∞, τ_c = τE0/E∞; Burgers J(0) = 1/E_M and slope at ∞ 1/η_M, E_R(0) = E_M; KV J(0) = 0, J(∞) = 1/E; KV relaxation refuses (ValueError); input validation | hand values | rel. ≤ 1e-12 |
| V1c | s² Ê_R(s) Ĵ(s) = 1 for Maxwell, SLS, Burgers at s ∈ {0.05, 0.3, 2.0} | Laplace transforms by `scipy.integrate.quad` of the production curves | rel. ≤ 1e-7 |
| V1d | unloading: ε(t) = σ0[J(t) − J(t−t_u)H(t−t_u)] SLS recovers to 0, Maxwell leaves σ0 t_u/η, Burgers leaves σ0(t_u/η_M) | closed form of the residual | rel. ≤ 1e-12 (SLS: |ε(∞)| ≤ 1e-12 σ0/E∞) |
| V2 | strain step at t = 0 (dt = 0), hold with irregular dt; pure shear γ_xy: σ_xy = G(t)γ0, other components 0; pure volume: σ = K(t)θ 1; set A, t up to 2e3 | exact exponentials (hand sum in the test) | rel. ≤ 1e-10 on nonzero values; zero components ≤ 1e-10 of the scale |
| V3 | ramp/hold/unload with irregular dt, 1e-10 ≤ Δt/τ ≤ 1e6 (set W, 6 strain components, includes volume), stresses of all 6 components at all nodes | viscous-strain ODE e_v' = (e − e_v)/τ_i, θ_v' = (θ − θ_v)/τ_i, per linear segment, `solve_ivp` Radau rtol 1e-12; oracle control: stress-state ODE oracle and Van Loan `expm` oracle agree with it ≤ 1e-9 of the scale; production uses no sub-stepping | max abs. error ≤ 1e-8 of max‖σ‖∞ over the path |
| V4a | analytic set AN: uniaxial stress relaxation (strain-controlled ε_xx step, σ_yy = σ_zz = 0) equals the SLS E_R(t) = E∞' + (E0' − E∞')e^{−t/τ'} with E∞' = 9K∞G/(3K∞+G), E0' = 9K0G/(3K0+G), τ' = τ(3K0+G)/(3K∞+G) (derived from the operational modulus Ē(s) = 9K̄Ḡ/(3K̄+Ḡ), in the test); creep (σ_xx = σ0 step): ε_xx = σ0 J_SLS(t) with the same parameters, τ_c = τ'E0'/E∞'; log-spaced steps, m ∈ {25, 50, 100} per decade from 1e-3 to 1e3 s | closed forms | at m = 100: max rel. error ≤ 5e-4 of the curve scale; observed order between 50 → 100 ≥ 1.6 |
| V4b | set NP: relaxation and creep (uniaxial stress) of non-proportional spectra equal the DAE oracle (branch ODEs with the constraint σ̇_yy = 0 differentiated, instantaneous initial conditions, `solve_ivp` Radau rtol 1e-12) | DAE oracle; second oracle for the creep: ε_xx = σ0(J_G/3 + J_K/9) with J from residues of Laplace-domain Q(s)/(sN(s)) (`numpy.roots`), oracle vs oracle ≤ 1e-8 | same as V4a (5e-4, order ≥ 1.6) |
| V4c | the production history of V4b: σ_yy, σ_zz ≤ 1e-8 σ0 (or ε0·E0) at every node; re-evaluating σ at all nodes from the production strain nodes with the V3 oracle reproduces the production σ (all components) | V3 oracle | ≤ 1e-8 of scale |
| V4d (negative control) | the pointwise formula E(t) = 9K(t)G(t)/(3K(t)+G(t)) does **not** give the uniaxial relaxation modulus for non-proportional spectra: on AN, max_t \|E_ptw − E_R\|/E_R ≥ 3e-2 (t ∈ [1, 100] s); for proportional spectra (set P) it does coincide: ≤ 1e-12. The lateral strain ratio −ε_yy/ε_xx of the production run on AN is time dependent (relative spread ≥ 5e-2 over t ∈ [1e-3, 1e3]), on P constant 0.3 to ≤ 1e-9 | closed forms | as stated |
| V5 | algorithmic tangent equals the central FD of the production stress for each of the 6 Voigt strain components, on a non-trivial committed state, with and without shift/temperature; h ∈ {1e-3, 1e-4, 1e-5, 1e-6}; first-order Taylor remainder ≤ 1e-12 ‖σ‖; instantaneous tangent (dt = 0) equals K0 1⊗1 + 2G0 I_dev built from the isotropic stiffness C11 = K+4G/3, C12 = K−2G/3, C44 = G; `drive_material_point` with a mixed path converges in ≤ 1 Newton iteration per step (consistent tangent) | central finite differences | ‖C_fd − C‖max ≤ 1e-6 ‖C‖max for every h |
| V6 | energy closure: for a piecewise-linear path (24 random segments, seed 20261127, ε ≤ 1e-3, set A and set Z) refined by 1, 2, 4, 8 sub-steps per segment (the path is identical), production ψ_end and ΣD equal the oracle ψ(end) and ∫rate dt (augmented ODE); the oracle satisfies W = ψ + D; each D increment ≥ −1e-12 ψ_max, ψ ≥ 0; the trapezoid external work Σ½(σ_n+σ_{n+1}):Δε of the production stresses converges to ψ_end + ΣD with order ≥ 1.6 (ratio of successive residuals ≥ 3) and final residual ≤ 1e-3 of ψ_end + ΣD; no NaN with zero branches | augmented `solve_ivp` Radau rtol 1e-12 with accumulators for ψ, ∫rate dt, ∫σ:ε̇ dt | ψ, ΣD: rel. ≤ 1e-8 each level; oracle closure rel. ≤ 1e-8 |
| V7a | a_T(T_ref) = 1 exactly (WLF, Arrhenius); values at T ≠ T_ref (WLF T_ref = 300 K, C1 = 17.44, C2 = 51.6 K; Arrhenius T_ref = 300 K, E_a = 8e4 J/mol, R = 8.314462618) | mpmath at 50 digits | rel. ≤ 1e-13 |
| V7b | WLF refuses C2 + T − T_ref ≤ 0 (T = T_ref − C2 and below), T ≤ 0, and T outside the calibrated range when given; constructors reject C1 ≤ 0, C2 ≤ 0, E_a ≤ 0, T_ref ≤ 0 | — | exact (ValueError) |
| V7c | reduced time of a step with T linear: WLF 290 → 350 K in 100 s, Arrhenius 250 → 400 K in 1 s (Δ ln a_T ≈ 14), isothermal; overflow (Arrhenius at T = 1e-3 K) raises ValueError; underflow limit finite | `scipy.integrate.quad` of 1/a_T(T(t)) epsrel 1e-13 (mpmath for isothermal) | rel. ≤ 1e-11 (isothermal 1e-14) |
| V7d | with shift: (i) T = T_ref throughout is bitwise equal to the unshifted model (regression, labelled); (ii) isothermal T ≠ T_ref equals the V3 oracle with τ_i ← a_T τ_i (set A, ramp/hold path), ≤ 1e-8 of scale; (iii) thermal cycle with ramp, hold, a temperature jump (dt = 0 step) and strain ramp, nodes at 0, 100, …, 600 s, Arrhenius T_ref = 320 K, E_a = 6e4, set A with τ's as stated, refined by m ∈ {4, 8, 16, 32, 64} sub-steps per segment: error against the oracle with time-varying τ_i(t) = τ_i a_T(T(t)) (strain and T linear per segment, Radau rtol 1e-12) decreases monotonically, the m = 64 error ≤ 5e-3 of max‖σ‖∞, ratio err(32)/err(64) ≥ 1.8 | oracle with a_T(T(t)) inside the ODE (independent of the reduced-time update) | as stated; observed order reported |
| V8a | DMA single branch, ωτ = 1: G' = G∞ + G1/2, G'' = G1/2, tan δ = G''/G'; limits ω → 0 (G∞, G'' → 0) and ω → ∞ (G0) at ωτ = 1e-9 / 1e9; G'' ≥ 0 for multi-branch sets at 200 log-spaced ω; G'' of a single branch peaks at ωτ = 1 (argmax of a 1e-3-spaced log grid within 1e-3 in ln ω); K* likewise; ω in rad/s (f = ω/2π conversion helper) | hand values | rel. ≤ 1e-14 (limits 1e-8) |
| V8b | time domain, set D, ω = 0.6 rad/s: simple shear γ_xy = γ0 sin ωt (γ0 = 1e-3), 200 steps per cycle, 30 cycles: last-cycle fit σ_xy = γ0(G' sin + G'' cos) recovers G*, ΣD of the last cycle = π G'' γ0², ψ periodic (start/end of last cycle equal) | hand G*, π G'' γ0² | rel. ≤ 1e-3 of \|G*\| (energy: 1e-3) |
| V8c | uniaxial stress, ε_xx = ε0 sin ωt, σ_yy = σ_zz = 0 (drive_material_point), same set and cycles: the fits give E*(ω) = 9K*G*/(3K*+G*) and ν*(ω) = (3K*−2G*)/(2(3K*+G*)) with ε_yy = −Im(ν* ε0 e^{iωt}); identities E* = 2G*(1+ν*) = 3K*(1−2ν*) | time-domain fit | rel. ≤ 2e-3 of \|E*\|, \|ν*\| ; identities 1e-14 |
| V9 | (a) restart: 10 steps in one call equal 5 + 5 steps with `store` restored from `checkpoint()` and `strain0` taken in a dwell (constant strain, dt = 30 s) (regression of the contract, labelled); (b) hold subdivision: one step of Δt = 10 equals ten steps of 1, and a random split of the dwell, state and stress to rel. ≤ 1e-12 (semigroup property of the exact update); (c) a rejected trial (`rollback`, missing step ValueError, negative dt ValueError) leaves the committed arrays bitwise unchanged and read-only; evaluating twice from the same committed state is bitwise identical; the committed arrays are not modified by `update` | the uninterrupted run; the exact exponential semigroup | exact (a, c); 1e-12 (b) |
| V10 | invalid input rejected with ValueError: negative branch amplitude, τ ≤ 0, NaN/inf, negative equilibrium modulus, all moduli zero (M0 = 0), shape mismatch, normalized weights with Σg > 1, a shift of a wrong type; zero equilibrium shear accepted and flagged `is_fluid`, no regularisation: shear stress of a held strain decays to ≤ 1e-12 of the instantaneous value for t ≥ 40 τ_max, the shear creep rate at t ≫ τ_max equals σ/η_G with η_G = ΣG_iτ_i (rel. ≤ 1e-6), `equilibrium_moduli()` returns exactly (K∞, 0); zero-amplitude branches give finite energy and dissipation | hand values | as stated |
| C1a | SLS E∞ = 1 MPa, E1 = 2 MPa, τ = 10 s, σ0 = 1 kPa: J(t) = 1/E∞ − (1/E∞ − 1/E0)e^{−t/30 s}; creep to 40 s, unload at 40 s by superposition: recoverable strain at 40 s = σ0(1/E0 + (1/E∞ − 1/E0)(1 − e^{−40/30})), strain at t = 40 + 3000 s ≤ 1e-12 σ0/E∞ (full recovery of the solid); Maxwell (E = 1 MPa, η = 1e7 Pa s): residual after unloading at 40 s = σ0·40/η = 4e-3 for all t > 40; the two are not conflated | closed form (scalar functions) | rel. ≤ 1e-12 |
| C1b | 3-D production, set P (proportional spectra, ν = 0.3), stress-controlled uniaxial creep σ0 = 1 kPa to 40 s then σ_xx = 0 (jump at dt = 0): ε_xx equals σ0 J_SLS(t) at all nodes (log-spaced, m ∈ {25, 50, 100} per decade; m = 100 error ≤ 5e-4 of σ0/E∞, order 50 → 100 ≥ 1.6), −ε_yy/ε_xx = 0.3 to 1e-9 at all nodes, strain after 150 s of recovery ≤ 1e-3 of the peak strain | scalar closed forms | as stated |

Negative controls: V4d (pointwise formula), V7d(iii) convergence ratio (a first-order model would fail the ≥ 1.8 ratio only
if no refinement), V9(b) shows that a non-exact recursion would not satisfy the semigroup to 1e-12, V10 invalid inputs.
Regressions (not independent) are labelled in the tests: V7d(i), V9(a), V9(c).

Run: `python -m pytest tests/toporlogy/test_viscoelastic.py -q -p no:randomly -p no:cacheprovider` with `TMP` and `TEMP`
set to `D:/Masaustu/Topology Optimization/.claude/tmp-toporlogy/V_viscoelastic` and `--basetemp` below it. Resource
cap: ≤ 20 000 material-point steps per test, single process.

Not implemented and therefore not verified: temperature-dependent amplitudes (vertical shift), thermodynamically
consistent nonisothermal energy, finite strain, anisotropic or orthotropic spectra, plane stress (refused by the
shared helper), state VJP/JVP, fractional kernels, spectrum fitting.

First result (append after run): see below.
