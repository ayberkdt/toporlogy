# N_nonlinear — WP10 verification slice: finite-strain kinematics, constitutive laws, plane stress, grid patch, path following (N1–N5)

Preregistered 2026-11-27 02:53 +0300, base SHA `5be9284`, before any test of this record was run. Tests:
`tests/toporlogy/test_nonlinear_foundations.py`. Document: `docs/foundations/NONLINEAR_CONTINUUM.md`. No production
code is changed in advance; production changes only if a test exposes a confirmed defect (stated in the first result).

## Already present (read first, not duplicated)

* N1 material point: `test_constitutive.py::test_p1_*` (ψ, S, ℂ of SVK and NeoHooke vs torch autograd of ψ(E), 3-D /
  plane strain / plane stress, ν ∈ {−0.3, 0, 0.3, 0.45}); `test_p2_*` (ψ(QF) = ψ(F) and S(QF) = S(F), 3-D, 20 random Q);
  `test_nonlinear.py::test_v1_*` (FE small load = linear), `test_v2_*` (homogeneous uniaxial tension, closed form),
  `test_v3_*` (K_T vs central FD of f_int), `test_strain_energy_gradient_is_internal_force`.
* N2: plane-stress NeoHooke vs autograd Schur complement in `test_p1_*` (plane = "stress"); FE failure code
  `invalid_deformation` in `test_nonlinear_solvers.py::test_n1_failure_codes`.
* N4: `test_nonlinear_solvers.py::test_a1_a3_*`, `test_a2_*` (two-bar truss vs closed form, limit points), `test_a4_*`
  (load-control negative control), `test_bifurcation.py::test_t1_t2_*`, `test_t3_*`, and the historical R07 defect as the
  strict xfails `test_a1_asymmetric_bifurcation_located_and_classified` / `test_a2_asymmetric_branch_switch` (reason
  `A_REASON`: an accepted point exactly on a singular tangent breaks the predictor K⁻¹q).
* N5: `test_constitutive.py::test_j1_*`–`test_j6_*`, `test_j4b_*` (J2 closed forms, load/unload, consistent tangent vs FD,
  yield surface, dissipation, energy identity), `test_plastic_sensitivity.py` (fixed-active-set gradient, VJP).
* Not present in the repository: a nonlinear element on a distorted mesh. `fem/nonlinear.py` is a Q4/H8 element on the
  regular `Domain` grid (`require_grid`), and `fem/mesh/*` has no finite-strain element. N3 "distorted" is therefore
  NOT VERIFIED and recorded as such; the grid patch below is the verifiable part. Also not present: a bordered /
  previous-tangent predictor at an exactly singular accepted point (`arc_length` predicts with K⁻¹q), and finite-strain
  plasticity / viscoelasticity.

## Conventions and independent references (derived here, rule 11)

Unit modulus E = 1, Poisson ν. λ = ν/((1+ν)(1−2ν)), μ = 1/(2(1+ν)), κ = λ + 2μ/3 = 1/(3(1−2ν)). F = I + H, J = det F,
C = FᵀF, E = ½(C − I). Stresses are tension-positive. Voigt order [xx, yy, zz, yz, xz, xy], stress physical shear,
strain engineering shear (the production ℂ is ∂S/∂E in that Voigt form: S_I = ℂ_IJ E_J with E_J = 2E_cd for shear).

* SVK: ψ = ½λ(tr E)² + μ E:E, S = λ tr E I + 2μ E, P = F S. Written in the test with numpy from λ, μ only.
* NeoHooke: ψ = ½κ(J−1)² + ½μ(J^(−2/3) tr C − 3). With ∂J/∂F = J F⁻ᵀ, ∂J^(−2/3)/∂F = −(2/3) J^(−2/3) F⁻ᵀ and
  ∂tr C/∂F = 2F:
  P = ∂ψ/∂F = κ J(J−1) F⁻ᵀ + μ J^(−2/3) (F − (1/3) tr C · F⁻ᵀ), S = F⁻¹P. Closed form, written in the test; the
  production code computes S from C⁻¹ and dev E, a different expression.
* Cauchy σ = P Fᵀ / J = F S Fᵀ / J; Kirchhoff τ = Jσ. Objectivity under F → QF (Q proper orthogonal): C unchanged, so
  ψ, S unchanged; P → Q P; σ → Q σ Qᵀ.
* Material tangent D_abcd = ∂S_ab/∂E_cd; ℂ_IJ = D_abcd with I = (ab), J = (cd) (for J shear, S_I = D_ab,cd E_cd +
  D_ab,dc E_dc = D_abcd · 2E_cd, hence ℂ_IJ = D_abcd). First-Piola tangent A_iJkL = ∂P_iJ/∂F_kL = δ_ik S_JL +
  F_iA F_kB D_AJBL (from P = F S, dS = D : dE, dE = sym(Fᵀ dF)). A has major symmetry A_iJkL = A_kLiJ.
* Simple shear F = I + γ e₁⊗e₂ (J = 1, tr C = 3 + γ², b = FFᵀ = [[1+γ², γ, 0],[γ, 1, 0],[0, 0, 1]]): NeoHooke
  Cauchy σ = μ(b − (1/3) tr b I) gives σ₁₂ = μγ, σ₁₁ = 2μγ²/3, σ₂₂ = σ₃₃ = −μγ²/3 (volumetric part vanishes at J = 1).
* Diagonal F = diag(λ₁, λ₂, λ₃): NeoHooke P_ii = κ J(J−1)/λ_i + μ J^(−2/3)(λ_i − (Σλ_k²)/(3λ_i)); a spherical stretch
  F = sI gives P = κ s²(s³ − 1) I (the isochoric part is zero).
* Work conjugacy: for any F(t), S:Ė = P:Ḟ = τ:d with Ė = sym(Fᵀ Ḟ), d = sym(Ḟ F⁻¹) (follows from P = FS, τ = FSFᵀ).
  Hencky strain H = ½ ln C = ln U. Coaxial case F(t) = Q diag(λ(t)) Qᵀ (fixed Q, R = I): ½ ln C = Q diag(ln λ) Qᵀ,
  Ḣ = Q diag(λ̇/λ) Qᵀ = d, so τ:Ḣ = τ:d exactly. Non-coaxial (simple shear) Ḣ ≠ d in general.
* Plane stress NeoHooke at in-plane F₂ (2×2, det F₂ > 0): F = blockdiag(F₂, λ₃) (out-of-plane shear is zero for the
  isotropic law); λ₃ solves P₃₃(F₂, λ₃) = 0 from the closed form above with J = det F₂ · λ₃, tr C = ‖F₂‖_F² + λ₃²
  (`scipy.optimize.brentq`, xtol 1e-15). E₃₃ = ½(λ₃² − 1). The in-plane S is (F⁻¹P) in-plane. The condensed tangent is
  d S_αβ / d E_γδ along the constrained path, obtained by central finite differences of this reference.
* J2 (E = 1, κ(α) = σ_y + Hα, α the equivalent plastic strain): hydrostatic strain ε = c I has dev ε = 0, so
  s_tr = 0, f_tr = −σ_y < 0 for every c: elastic, σ = K tr ε I = 3Kc I, ψ = ½ σ:ε = (9/2) K c². At the yield point of the
  isochoric strain ε = e·diag(1, −½, −½): s_tr = 2μ e diag(1, −½, −½), q_tr = √(3/2)‖s_tr‖ = 3μ e, so f_tr = 3μe − σ_y
  and e_y = σ_y/(3μ). Just above e_y, Δγ → 0 and θ → 1, θ̄ → 3μ/(3μ + H); the code's algorithmic tangent is
  ℂ = K 1⊗1 + 2μθ 𝕀_dev − 2μθ̄ n̂⊗n̂, so the jump of ℂ_xxxx across the yield point is
  −2μ · 3μ/(3μ+H) · n̂_xx², with n̂ = diag(1, −½, −½)/√(3/2), n̂_xx² = 2/3. The stress is continuous (C⁰); the tangent
  is not (kink).

## Checks, thresholds, negative controls

Relative error = max|a − b| / max|b| unless "abs". 20 random F per case: H uniform in [−0.3, 0.3]³ˣ³ with
‖H‖₂ ≤ 0.3 and det(I+H) ≥ 0.5, composed with a random proper rotation F = Q(I+H) (seeds fixed in the file).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| N1a | F = I: ψ = 0, S = 0, P = 0 exactly; ℂ(I) = isotropic small-strain ℂ with the Lamé constants above | formulas above | abs 1e-15 (ψ, S); ℂ rel 1e-12 |
| N1b | pure rotation F = Q (J = 1; angles up to π): ψ = 0, S = 0, P = 0, Cauchy = 0 | 20 random Q | abs 1e-13 |
| N1c | simple shear γ ∈ {0.1, 0.5, 1, 2}: ψ, S, P, σ equal the closed forms (SVK, NeoHooke); NeoHooke σ₁₂ = μγ, σ₁₁ = 2μγ²/3, σ₂₂ = σ₃₃ = −μγ²/3; ν ∈ {0.3, 0.45} | formulas above (hand-derived σ) | rel 1e-12 |
| N1d | uniaxial F = diag(λ,1,1), λ ∈ {0.6, 0.8, 1.5, 2.5}; biaxial diag(λ,λ,1), λ ∈ {0.7, 1.4, 2}; spherical sI, s ∈ {0.8, 1.3}: P and ψ equal the closed forms; spherical NeoHooke P = κ s²(s³−1) I | diagonal formulas | rel 1e-12 |
| N1e | objectivity: ψ(QF) = ψ(F), S(QF) = S(F), P(QF) = Q P(F), σ(QF) = Q σ(F) Qᵀ for 50 random (F, Q) (P2 covers ψ and S only; P and σ are new). Negative control: the linear-strain energy ½ε:D:ε of H = Q − I (Q a 30° rotation) is not zero (≥ 1e-3), i.e. the check can fail | production S, P = F S | rel 1e-12; control ≥ 1e-3 |
| N1f | tangent: ℂ symmetric (rel 1e-12); A = δS + F F ℂ equals (i) central FD of the reference P (h = 1e-6, rel 1e-6) and (ii) the torch autograd Hessian ∂²ψ/∂F² of the reference ψ(F) (float64, rel 1e-10); A has major symmetry (rel 1e-12); production S, ψ equal the reference ψ, S (rel 1e-11); SVK and NeoHooke, ν ∈ {0.3, 0.45} | independent numpy/torch ψ(F) | as stated |
| N1g | small-strain limit: H = δ H₀ (fixed H₀, max-norm 1), δ ∈ {1e-1, …, 1e-5}: r(δ) = ‖S − σ_lin(ε)‖/‖σ_lin(ε)‖ with σ_lin = λ tr ε I + 2με, ε = sym H, satisfies r = cδ + o(δ) (S = σ_lin(E) + O(E²), E = ε + ½HᵀH), so the least-squares log–log slope over δ ∈ [1e-5, 1e-2] is in [0.9, 1.1], r(1e-5) ≤ 1e-3, r decreasing; same for ‖ℂ(δH₀) − ℂ_lin‖/‖ℂ_lin‖ (slope in [0.9, 1.1]); SVK: S equals σ_lin(E) exactly (rel 1e-13) | formulas above | as stated |
| N1h | Hencky: `spectral_function(C, "log")/2` equals ½ scipy.linalg.logm(C) and `tensor.hencky_strain(F, "material")` (rel 1e-11), spatial likewise with B; derivative: ½ `spectral_frechet(C, Ċ, "log")` equals central FD (h = 1e-6) of ½ logm(C(F + t Ḟ)) (rel 1e-7), 20 random F, Ḟ (J ≥ 0.2, stretches up to 3) | scipy logm | as stated |
| N1i | work conjugacy: S:Ė = P:Ḟ = τ:d for random (F, Ḟ), SVK and NeoHooke (rel 1e-12, relative to ‖S‖‖Ė‖); coaxial F(t) = Q diag(λ(t)) Qᵀ: τ:Ḣ = τ:d = S:Ė with Ḣ = ½ spectral_frechet (rel 1e-11); negative control: simple shear γ = 1, Ḟ = e₁⊗e₂: τ:Ḣ differs from τ:d by ≥ 1e-3 relative | formulas above | as stated |
| N2a | plane stress NeoHooke, 30 random in-plane F₂ (det F₂ ≥ 0.5) and simple shear γ ∈ {0.5, 1}: diagnostic E₃₃ equals ½(λ₃²−1) of the brentq root (rel 1e-10); in-plane S equals the reference (rel 1e-10); the reference P₃₃ evaluated at the production E₃₃ is zero (abs ≤ 1e-12 · max|P|) | brentq on closed form | as stated |
| N2b | condensed tangent: ℂ_ps (3×3) vs central FD of the reference constrained S_ps (h = 1e-6 in F₂, ΔS = ℂ_ps ΔE_ps with ΔE_ps the exact difference of E from F₂ ± h dF), rel 1e-6; ℂ_ps symmetric (1e-12) | reference | rel 1e-6 |
| N2c | inadmissible trial: NeoHooke 3-D, plane strain, plane stress with det F ≤ 0 (J = 0 exactly: H = −I; J < 0: H = diag(−1.2, 0, 0)) raises `InadmissibleState` with code "invalid_deformation"; the input array (read-only) and a committed state mapping are bytewise unchanged; FE: `NonlinearElasticity._newton` with a huge compression returns converged False, code "invalid_deformation", `rolled_back` True and returns the start displacement bytewise unchanged | — | exact |
| N3a | grid patch (Domain, Q4 plane strain and plane stress with thickness 0.8, H8; NeoHooke and SVK): affine u = (F − I)X (random F of N1 family, rotation included): f_int on interior nodes zero (abs ≤ 1e-11 · max boundary force); Σ f = 0 (same scale); resultant on the face X₁ = min equals −t · A · P e₁ with the reference P (plane stress: σ₃₃ = 0 reference, in-plane P; A the face area; t = 0.8 for 2-D) rel 1e-10 | closed-form P | as stated |
| N3b | grid rigid-rotation patch u = (Q − I)X (Q up to π): all nodal forces ≤ 1e-12 abs; min det F = 1 (1e-12) | — | as stated |
| N3c | NOT VERIFIED: homogeneous patch on a distorted mesh, mesh convergence of a distorted element, "agreement with the grid path" (no distorted nonlinear element exists); recorded in the document, no test | — | — |
| N4a | two-bar truss, step refinement ds ∈ {0.02, 0.01, 0.005, 0.0025}: for each ds, max |λ − P(w)/P̂| ≤ 1e-9 and max |u| ≤ 1e-12 (the thresholds of A1), exactly two critical intervals, located limit points |λ₁ − 1|, |λ₂ + 1| ≤ 1e-8, |w − w₁,₂| ≤ 1e-6 h (T2); the errors are reported per ds | `tests/toporlogy/_truss.py` closed form P(w) | as stated |
| N4b | R07 re-check, measured: model A (U = ½k_a e² + ½k_r u² + (β/6)u³, e = v + u²/2L, k_a = 100, k_r = 1, β = 3, L = 1, q = (0, −1)), arc_length from (0, 0) with ds = 0.01: first Δλ = ds/‖K⁻¹q‖ = 0.01/0.01 = 1, the accepted point (0, −0.01) has K = diag(k_r + k_a e/L, k_a) with e = −0.01 → K₁₁ = 0 exactly; prediction: `ContinuationError` code "step_limit", cause "singular_tangent", last accepted λ within 1e-12 of 1. The defect is expected to be still present; the test asserts the measured failure (characterisation of R07, not a pass of the desired behaviour; the desired behaviour stays in the two strict xfails `A_REASON`) | model A closed form | exact code/cause; λ 1e-12 |
| N4c | NOT VERIFIED: bordered or previous-tangent predictor at a singular point, branch orientation there, repeated-eigenvalue subspace diagnostics at a critical point (not implemented) | — | — |
| N5a | J2 hydrostatic: ε = cI, c ∈ {1, 10, 100}·ε_y: elastic (plastic flag 0), α = 0, ε_p = 0 exactly, σ = 3Kc I (rel 1e-13), tangent = elastic ℂ (rel 1e-13), energy (9/2)Kc² (rel 1e-13), dissipation 0 | formulas above | as stated |
| N5b | nonsmooth at yield: ε = e diag(1, −½, −½), e = e_y(1 ∓ 1e-7): below plastic flag 0 and ℂ = elastic; above flag 1 and the jump ℂ_xxxx(above) − ℂ_xxxx(below) = −2μ · 3μ/(3μ+H) · (2/3) (rel 1e-5); stress and energy continuous (|Δσ_xx| ≤ 1e-6 σ_y); tangent_kind "continuum" above has θ = 1 (the same jump formula, exact); H ∈ {0.05, 0} (perfect plasticity: θ̄ = 1) | formulas above | as stated |

Run: `python -m pytest tests/toporlogy/test_nonlinear_foundations.py -q -p no:randomly` (with TMP/TEMP and --basetemp as in
the brief), a few seconds; no slow tests. Negative controls: N1e (linear strain not objective), N1i (non-coaxial Hencky
rate), N4b-style failure kept as measured, N5 jump (a smooth tangent would fail it).

## First result (run 2026-11-27 02:57 +0300 for the first run, 02:58 +0300 final; SHA b4b9fa9 + test file)

`python -m pytest tests/toporlogy/test_nonlinear_foundations.py -q -p no:randomly` (TMP/TEMP under
`.claude/tmp-toporlogy/N_nonlinear`): 55 passed in 13 s. No production code changed; no defect found.

One set-up correction between the first and second run (no threshold changed): in the first run the SVK identity of N1g
(`S = sigma_lin(E)`, rel 1e-13) failed at 4.4e-13 because the test formed the small displacement gradient as
(I + H) − I, losing log10(1/|H|) digits (the cancellation that `core/constitutive.py` documents). The test now passes H to
the model directly (`evaluate(..., H=H)`). The first run stopped there (`-x`) and had passed N1a–N1f only; the other
checks ran for the first time in the second run.

Measured values (max over the cases of each check):

| ID | Result | Measured |
|---|---|---|
| N1a | PASS | psi, S exactly 0; C(I) vs Lame C: 1.6e-16 |
| N1b | PASS | worst of psi, S, P, Cauchy over 20 rotations up to pi: 1.4e-15 (SVK 8.7e-16) |
| N1c | PASS | worst 2.2e-14 (NeoHooke), 4.0e-16 (SVK), including the hand-derived Cauchy components |
| N1d | PASS | worst 2.2e-15; spherical NeoHooke P = kappa s^2 (s^3−1) I included |
| N1e | PASS | P and Cauchy transform: worst 8.0e-14 (NeoHooke), 1.9e-14 (SVK); control: linear-strain energy of a 30 degree rotation 3.45e-2 (≥ 1e-3) |
| N1f | PASS | values 3.2e-14; C symmetry 2.6e-16; A vs FD 2.6e-10 (limit 1e-6); A vs torch Hessian 4.5e-15 (limit 1e-10); major symmetry 2.5e-16 |
| N1g | PASS | slope of r(S): 1.000 (SVK, nu 0.3 and 0.45), 1.001 and 0.999 (NeoHooke); r(1e-5) = 1.2e-5 to 1.9e-5; NeoHooke tangent slope 1.001 and 1.000. The SVK tangent is the constant D: its deviation from the Lame C is ≤ 7.6e-17 at every delta, so the registered "same slope for C" is degenerate for SVK and is checked as a constant (≤ 1e-13) |
| N1h | PASS | spectral log vs logm 3.6e-15; tensor.hencky_strain vs logm 3.6e-15; derivative vs FD 1.4e-9 |
| N1i | PASS | S:Edot = P:Fdot = tau:d 1.8e-16; coaxial Hencky 8.9e-16; non-coaxial control: tau:Hdot differs from tau:d by 0.744 relative (≥ 1e-3) |
| N2a | PASS | E33 6.2e-15, in-plane S 5.4e-15, max abs P33 / max abs P 1.9e-14 |
| N2b | PASS | condensed tangent vs FD 1.0e-9 (limit 1e-6); symmetry 3.7e-17 |
| N2c | PASS | 3-D, plane strain, plane stress: InadmissibleState code invalid_deformation for J = 0 and J < 0, inputs unchanged; FE: converged False, invalid_deformation, rolled back, start displacement unchanged bytewise |
| N3a | PASS | interior nodal forces 8.8e-15, sum 1.7e-15, face resultant 2.2e-15 relative (six configurations): the Q4/H8 grid kernel including the thickness factor is exact for affine F |
| N3b | PASS | rotation patch: max nodal force 4.7e-16, abs(min det F − 1) 2.3e-15 |
| N3c | NOT VERIFIED | no distorted-mesh nonlinear element exists in the repository; no test |
| N4a | PASS | ds = 0.02, 0.01, 0.005, 0.0025: path vs closed form 1.2e-13, 9.3e-14, 1.1e-13, 1.2e-13; limit-point lambda error 1.7e-12, 1.5e-13, 3.9e-14, 3.3e-12; w error / h 3.2e-13, 2.9e-14, 7.6e-15, 6.4e-13. The errors do not decrease with ds: the point is located by a hyperplane Newton inside the interval, so the remaining error is the rounding floor of the 1e-12 tolerance, not a discretisation error |
| N4b | PASS (failure reproduced) | R07 confirmed still present: ContinuationError step_limit / singular_tangent, 2 accepted points, last lambda = 1.000000000000000, K11 = 0 and det K = 0 exactly. A characterisation of an open defect; the desired behaviour remains in the strict xfails of test_bifurcation.py |
| N4c | NOT VERIFIED | bordered / previous-tangent predictor at a singular point, branch orientation there, repeated-eigenvalue subspace diagnostics are not implemented |
| N5a | PASS | hydrostatic c = 1, 10, 100 eps_y: elastic, alpha = 0, eps_p = 0; sigma, C, psi relative error ≤ 1.7e-16 |
| N5b | PASS | flags 0 then 1; jump of C_xxxx −0.4915213 (H = 0.05) and −0.5128205 (H = 0), equal to the prediction to the printed digits; continuum-tangent jump equal; abs(d sigma_xx) 6.9e-11 (limit 1e-6 sigma_y = 1e-9), abs(d psi) 9.0e-14. The energy continuity limit had no number in the table; the test used 1e-6 sigma_y eps_y = 1e-12, fixed before the first run |

Limitations: the grid patch has a constant element Jacobian, so it does not test distorted geometry; plane stress is verified
for the isotropic neo-Hooke only (the condensation uses sigma_zz alone); SVK plane stress is the 2-D law with the
plane-stress constants, not a condensed 3-D body (as `core/constitutive.py` states). Follow-up (separate
preregistration): a distorted Q4/H8 finite-strain element with patch and convergence tests; a bordered predictor for
`arc_length` and a fix of R07.

## Update to N4b / N4c (2026-11-27 09:11 +0300, R07_bordered)

The N4b row above is the measurement at `5be9284` and stays as measured. `prereg/R07_bordered.md` (base `36c020f`) adds
the bordered / previous-increment predictor to `arc_length`: with it, the same run (model A, ds = 0.01) passes the point
with K11 = 0 exactly and stays on v = -lambda/k_a (R1). `test_n4b_*` now runs with that predictor switched off (renamed
`test_n4b_r07_exactly_singular_accepted_point_breaks_the_old_predictor`, same asserts) and keeps the old failure as a
negative control. N4c is now partly done: the bordered / previous-increment predictor and the branch orientation at a
singular accepted or start point are VERIFIED on model A and the truss (R1-R4 there); repeated-eigenvalue subspace
diagnostics remain NOT VERIFIED. The strict xfails `test_a1_*` / `test_a2_*` of `test_bifurcation.py` still fail, now in
`locate_critical` (crossing eigenvalue exactly 0 at an accepted point, "not bracketed"), not in the predictor (R7).

