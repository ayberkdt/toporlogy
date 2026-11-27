# TP_thermoplastic — glass-transition property model, crystallinity-dependent moduli, crystallising material point, latent heat and shrinkage (preregistration)

Preregistered 2026-11-27 05:59 +0300 (output of `date "+%Y-%m-%d %H:%M %z"` pasted by the command that writes this record), before any
test of this slice was run. Base: `1cd3a8f`. Module `core/thermoplastic.py` (written before this record; no numerical test of
it has been executed), tests `tests/toporlogy/test_thermoplastic.py` (written after this record is committed), docs
`docs/foundations/THERMOPLASTICS.md`. Plan rows: WP15 TP01/TP02 (transition model only), TP04 (shrinkage, latent heat
at the material point), TP05 NOT implemented; acceptance cases TP1, TP2, TP5 of 23_ACCEPTANCE_CASES.md. TP3/TP4 (Avrami,
Nakamura) are the slice `TS_cure` and are only reused. Every number is **synthetic**; none is material data and no
Tg, Tm or modulus of a real polymer is claimed.

Conventions: temperature absolute (K); time s; moduli Pa; tension positive; Voigt [xx, yy, zz, yz, xz, xy]; X the
RELATIVE crystallinity in [0, 1] (the Avrami/Nakamura state), φ = X_max X the absolute crystalline volume fraction;
β_c < 0 the volumetric crystallisation strain per unit X; L > 0 the latent heat in J/m³ per unit X (exothermic).

## Design statements and derivations

**Transition.** s(T) = 1/(1 + e^{−(T−T_m)/w}); s(T_m) = 1/2; s(T_m + w ln 3) = 3/4; s(T_m − w ln 3) = 1/4 (e^{ln 3} = 3);
s' = s(1−s)/w, largest at T_m with value 1/(4w). Linear form p = p_g + Δ s, Δ = p_r − p_g: p(T_m) = (p_g+p_r)/2,
p' = Δ s(1−s)/w, ∫_{T0}^{T1} p dT = p_g (T1−T0) + Δ w [sp((T1−T_m)/w) − sp((T0−T_m)/w)] with sp(x) = ln(1+e^x)
(d sp/dx = s). Log form p = p_g (p_r/p_g)^s: p(T_m) = √(p_g p_r), p' = p ln(p_r/p_g) s(1−s)/w.

**Mixing.** φ the crystalline fraction. Voigt K_V = (1−φ)K_a + φ K_c (same for G); Reuss 1/K_R = (1−φ)/K_a + φ/K_c; Hill =
(V + R)/2 (inside [R, V] by construction, not a bound); HS (soft matrix 1, hard inclusion 2, f2 = φ):
K_HS− = K1 + f2/(1/(K2−K1) + 3 f1/(3K1+4G1)), G_HS− = G1 + f2/(1/(G2−G1) + 6 f1 (K1+2G1)/(5 G1 (3K1+4G1))); the upper
bound is the same expression with the roles of the phases exchanged (matrix = the hard phase 2, f_inc = 1−φ).

**Point.** σ = C(T, X):(ε − ε_th(T)1 − ε_cr 1), ε_cr = β_c (X − X0)/3, C = K 1⊗1 + 2G I_dev, E = 9KG/(3K+G).
(i) Fully constrained (ε = 0) at T = T_ref: e = −ε_cr 1, σ_ii = −3K ε_cr = −K β_c (X−X0): **positive (tension)** for
β_c < 0, shear 0. (ii) Uniaxial bar constrained along x only (ε_xx = 0, σ_yy = σ_zz = shear = 0): lateral stress free
gives e_yy = e_zz = −ν e_xx, σ_xx = E e_xx with e_xx = −ε_cr: σ_xx = −E β_c (X−X0)/3, positive for β_c < 0.
(iii) Free (all stresses 0): ε = (ε_th + ε_cr) 1. (iv) Strain-controlled xx, stress-free laterally, X ≡ X0 (no crystallisation):
σ_xx = E(T)(ε_xx − ε_th(T)), ε_yy = ε_th − ν (ε_xx − ε_th) (lateral e_yy = −ν e_xx).
(v) Nakamura with constant K(T) = K0 and exponent n integrates to θ = K0 t, X = 1 − exp(−(K0 t)^n) = 1 − exp(−k t^n),
k = K0^n; with X0 > 0 the state starts at θ0 = (−ln(1−X0))^{1/n}: X = 1 − exp(−(θ0 + K0 t)^n). For a K(T(t)) given by a
table with T linear in t, θ(t) = ∫_0^t K(T(s)) ds (quad in the test).
(vi) Latent heat of a step: L (X_{j} − X_{j−1}) J/m³; Σ_j = L (X_N − X_0). Adiabatic rise ΔT = L ΔX/(ρ c_p) (c_p the
sensible heat only; latent heat enters once). L = 1.2e8 J/m³, ρ = 1300 kg/m³, c_p = 1600 J/(kg K): ΔX = 0.5 gives
ΔT = 6e7/2.08e6 = 28.846153846…153846 K (60/2.08 = 28.8461538…).

## Synthetic constants (all tests)

Transition domain [300, 480] K, T_m = 380 K, w = 6 K. Modulus (form "log"): E_g = 2.0e9 Pa, E_r = 2.0e7 Pa, ν_a = 0.38
(constant). CTE (form "linear"): α_g = 6e-5, α_r = 2e-4 1/K. Crystal: E_c = 5e9 Pa, ν_c = 0.30 (constants). X_max = 0.4,
β_c = −0.06, L, ρ, c_p as above, n = 2. Mixing rule of the point: "hs_lower" (the crystal is stiffer than the amorphous
phase in K and G at every T of the domain: at 300 K K_a = 2.78e9 < K_c = 4.17e9, G_a = 0.725e9 < G_c = 1.92e9).

## Checks

Independence: closed forms derived here and written in the test (own logistic, own softplus integral, own HS formulas,
own Avrami / θ), `scipy.integrate.quad` (epsabs 0, epsrel 1e-13), central finite differences written in the test, a dense
numpy stiffness. A comparison of the batched mixture with a scalar loop of `mix_isotropic` is labelled "regression".
`rel(a, b)` = max|a−b| / max|b|.

| ID | Hypothesis | Reference | Metric / threshold |
|---|---|---|---|
| TP2a | exact points: E(T_m) = √(E_g E_r) = 2e8 Pa; E(T_m ± w ln 3) = E_g (E_r/E_g)^{3/4, 1/4}; α(T_m) = 1.3e-4; α(T_m ± w ln 3) = α_g + {0.75, 0.25}(α_r − α_g); dα/dT(T_m) = (α_r−α_g)/(4w); dE/dT(T_m) = E(T_m) ln(E_r/E_g)/(4w) | closed form (derivation above) | rel ≤ 1e-14 (derivatives ≤ 1e-13) |
| TP2b | monotone: E strictly decreasing, α strictly increasing on a 2001-point grid of the domain; the largest \|α'\| on a 24001-point grid lies within 0.01 K of T_m | grid | exact inequalities |
| TP2c | derivative vs central FD (h = 1e-3 K) at 9 temperatures in [304, 476] K, E and α | FD in the test | rel to max\|p'\| ≤ 1e-6 |
| TP2d | integral of α over [300, 420], [340, 400] equals the softplus closed form and `quad`; `ThermalExpansion(α, T_ref=300).strain` equals `quad`; the form "log" refuses `integral` | closed form; quad | rel ≤ 1e-12 |
| TP2e | refusal: T = 299.9, 480.1, NaN, 0, −1 raise ValueError for value, derivative and integral limits; T = 300 and 480 are accepted; extrapolation policy other than "error" refused | — | exact |
| TP2f | no universal T_g: the constructor has no default for `p_glassy`, `p_rubbery`, `T_mid`, `width`, `domain`, `units`, `source` (inspected); calling without T_mid raises TypeError; ValueError for width ≤ 0, equal plateaus, T_mid outside the domain, infinite domain, form "log" with a non-positive plateau, unknown form, Celsius temperature units; two parameter sets give different curves | — | exact |
| TP1a | ramp / hold / cool history through `drive_material_point` with `ThermoElasticPoint(IsotropicModuliOfT(E=transition, nu=0.38), ThermalExpansion(α transition, 300))`: times t = 1…185 s; T = 300 + t (t ≤ 120), 420 (120 < t ≤ 140), 420 − 2(t−140) (t > 140); ε_xx = 1e-3 t/185 controlled, all other stresses 0: σ_xx = E(T)(ε_xx − ε_th(T)) (statement (iv)), ε_th by `quad` of the test's own α, E by the test's own log-logistic; ε_yy = ε_th − ν(ε_xx − ε_th); E varies by more than a factor 50 over the path | closed form + quad | σ_xx: rel ≤ 1e-9; lateral stresses ≤ 1e-6 Pa; ε_yy abs ≤ 1e-12 |
| TP1b | the same history through `CrystallisingThermoElasticPoint` with K(T) ≡ 0 (no crystallisation, X ≡ 0 stays exactly 0) gives the same σ_xx | the same closed form | rel ≤ 1e-9 |
| TP1c | negative control: the secant-for-instantaneous mistake ε_th = α(T)(T − T_ref) gives σ_xx differing from the correct one by more than 1 % of max\|σ\| (predicted ≈ 8 % near T = 360 K) | closed form | > 1e-2 |
| MX1 | mixing rule inside the bounds: for φ = 0, 0.05, …, 1 and every rule: K_R ≤ K ≤ K_V, G_R ≤ G ≤ G_V (relative slack 1e-12 for rounding), K_R ≤ K_HS− ≤ K_HS+ ≤ K_V (same G), strictly increasing in φ; amorphous E = 3e7 Pa, ν = 0.38, crystal E = 5e9, ν = 0.30 | bounds from the test's closed forms; production `voigt/reuss/hs_bounds` as a second check | exact inequalities |
| MX2 | every rule equals its closed form (statement "Mixing"); endpoints exact: φ = 0 gives the amorphous (K, G), φ = 1 the crystal | closed forms | rel ≤ 1e-12 (rules), ≤ 1e-13 (endpoints) |
| MX3 | refusal: φ < 0, φ > 1, NaN, unknown rule, void phase raise ValueError; `CrystallineMixture` refuses mixed units; batched `moduli` equals a scalar loop of `mix_isotropic` (regression) | — | exact / 1e-14 |
| MX4 | negative control: a rule that exceeds Voigt by 1 % is detected by the bound check used in MX1 | — | check returns False |
| TP5a | isothermal 400 K, K ≡ 0.05 1/s, n = 2, dt = 2 s, 30 steps: diagnostic latent heat of step j equals L[X(t_j) − X(t_{j−1})], X = 1 − exp(−0.0025 t²); Σ = L X(60 s) with X(60) = 1 − e^{−9}; Σ ΔT_j (helper) = L X(60)/(ρ c_p); ∫_0^{60} L Ẋ dt by `quad` (Ẋ = 2·0.0025 t e^{−0.0025 t²}) = L X(60); ΔT(ΔX = 0.5) = 28.846153846153847 K; helper refuses dX < 0, L, ρ, c_p ≤ 0 | closed form; quad | steps rel (to L) ≤ 1e-10; sums ≤ 1e-12; quad ≤ 1e-10; ΔT ≤ 1e-14 |
| TP5b | free (all stresses 0) cooling 440 → 360 K at 2 K/s, 40 steps of 1 s, K(T) = table [300, 400, 480] K → [0, 0.03, 0] 1/s (PiecewiseLinear), ε_th with T_ref = 400: strain = (ε_th(T_k) + β_c X_k/3) on the three normal components, 0 shear, stresses 0; X_k = 1 − exp(−θ_k²), θ_k = ∫_0^{t_k} K(440 − 2s) ds by `quad` (break at s = 20); 0.3 < X_40 < 0.9 (predicted 0.588); crystallisation strain ε − ε_th < 0 for every k ≥ 1 | quad; closed form | strain abs ≤ 1e-11; stress ≤ 1e-5 Pa |
| TP5c | uniaxial bar, ε_xx = 0, lateral free, isothermal 400 K = T_ref, as TP5a: σ_xx = −E_mix(X_j) β_c X_j/3 with E_mix = 9KG/(3K+G) from the test's own HS formulas at φ = 0.4 X_j and the amorphous phase at 400 K from the test's own logistic; σ_xx > 0 for all j ≥ 1; lateral and shear stresses 0 | closed form | rel ≤ 1e-9; lateral ≤ 1e-6 Pa |
| TP5d | fully constrained (ε = 0), same set-up: σ_ii = −K_mix β_c X_j (statement (i)), shear 0 | closed form | rel ≤ 1e-9 |
| TP5e | negative control (sign): the closed form of TP5c with β_c → +\|β_c\| is compressive and differs from the production stress by 2·max\|σ\| (max\|σ_prod − σ_flipped\| / max\|σ_prod\| ≥ 1.9); `CrystallisingThermoElasticPoint(β_c ≥ 0)` raises ValueError | closed form | as stated |
| TP5f | X0 = 0.1 reference: free run at 400 K with the constant K, X = 1 − exp(−(θ0 + K0 t)²), θ0 = √(−ln 0.9), strain = β_c (X − 0.1)/3 (the initial state is stress-free); X path monotone non-decreasing, in [0.1, 1] also for dt = 1000 s (no clipping needed) | closed form | strain abs ≤ 1e-11 |
| TP5g | contract: `physical`, `kinematics = "small"`, `state_fields = ("X",)`, `conservative = False`; no step / step without temperature raises ValueError; energy and dissipation None; tangent equals the dense K 1⊗1 + 2G I_dev at (T_{n+1}, X_{n+1}) (rel ≤ 1e-12) and the FD dσ/dε at fixed committed state (h = 1e-7, rel ≤ 1e-6); invalid parameters (β_c ≥ 0, L ≤ 0, X_max ∉ (0, 1], X0 ∉ [0, 1], wrong types) raise ValueError | dense numpy; FD | as stated |
| TP5h | rollback: `evaluate` with a step gives a trial X ≠ committed X while `store.committed["X"]` is unchanged (bitwise) before commit and after `rollback`; committed arrays read-only; the repeated evaluation reproduces the trial; after `commit` committed equals the trial | exact | `np.array_equal` |

## Not in this slice (stated, not tested)

Melt flow and melt rheology, melting and re-crystallisation (X is non-decreasing; a negative θ increment raises), physical
ageing, rate dependence of the transition, a two-way thermal coupling (the latent heat is a diagnostic and is not fed back
into T), temperature-dependent crystallisation shrinkage or CTE, finite strain, anisotropic or fibre-composite shrinkage,
and the FE slice TP6.

Run: `python -m pytest tests/toporlogy/test_thermoplastic.py -q -p no:randomly`. First results are appended below.
