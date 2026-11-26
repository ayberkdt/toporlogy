# Rheology: linear viscoelasticity, generalized Maxwell, DMA, time–temperature shift

Status: WP12 slice. Code: `src/toporlogy/core/viscoelastic.py`. Tests: `tests/toporlogy/test_viscoelastic.py`.
Preregistration and first results: `prereg/V_viscoelastic.md`. Contract the model follows: `MATERIAL_CONTRACT.md`.
All numbers in the tests are **synthetic**; nothing here is a measurement of a real polymer and no real material is claimed
to be validated.

## Conventions and units

SI: moduli and stresses in Pa, viscosity in Pa s, time in s, temperature in K (absolute), activation energy in J/mol,
R = 8.314462618 J/(mol K). Tension positive. Voigt order [xx, yy, zz, yz, xz, xy], stress with the physical shear
component, strain with the engineering shear (as `core/constitutive.py`). Mechanical strain ε = sym ∇u (thermal and
other eigenstrains are not subtracted here: the caller passes the mechanical strain). e = dev ε, θ = tr ε.
Models are `physical = True`: they need a `TimeStep` and the FE physics that scale every law by an element modulus refuse
them.

## Scalar reference models (reference functions, not constitutive models)

Creep compliance J(t) [1/Pa] is the strain under σ0 H(t) divided by σ0; relaxation modulus E_R(t) [Pa] is the stress under
ε0 H(t) divided by ε0; values at t = 0 are those at 0+. J is not 1/E_R; in Laplace space s² Ê_R(s) Ĵ(s) = 1.

| model | creep J(t) | relaxation E_R(t) |
|---|---|---|
| Maxwell (E, η) | 1/E + t/η | E e^{-tE/η} |
| Kelvin–Voigt (E, η) | (1 − e^{-tE/η})/E | refused: the step response is E H(t) + η δ(t) (impulse) |
| SLS (E∞, E1, τ), E0 = E∞ + E1 | 1/E∞ − (1/E∞ − 1/E0) e^{-t/τ_c}, τ_c = τ E0/E∞ | E∞ + E1 e^{-t/τ} |
| Burgers (E_M, η_M, E_K, η_K) | 1/E_M + t/η_M + (1 − e^{-tE_K/η_K})/E_K | two exponentials from the roots of a s² + b s + c (docstring of `burgers_relaxation`) |

`recovery_strain(J, t, σ0, t_u)` is the linear superposition σ0 [J(t) − J(t − t_u) H(t − t_u)]. A solid (SLS) recovers fully;
a Maxwell fluid keeps the permanent strain σ0 t_u/η; a Burgers body keeps σ0 t_u/η_M. The recoverable and the permanent
parts must not be conflated.

## Prony series

M(t) = M∞ + Σ M_i e^{-t/τ_i} for the shear modulus G and the bulk modulus K independently (their own amplitudes and times).
Validated: M∞ ≥ 0, M_i ≥ 0 (a zero branch is allowed), τ_i > 0, all finite, M0 = M∞ + ΣM_i > 0. M∞ = 0 is a **fluid** in
that mode: accepted and flagged (`is_fluid`); the long-time stiffness is then genuinely singular and nothing replaces it with
a numerical stiffness; the steady flow viscosity is η = ΣM_iτ_i. The normalised convention is a separate constructor
(`PronySeries.from_weights(M0, g_i, τ_i)` with Σg_i ≤ 1, M∞ = M0(1 − Σg_i)); the two conventions are never mixed.

## 3-D generalized Maxwell

σ = 2G∞ e + Σ_i s_i + (K∞ θ + Σ_j p_j) 1, ṡ_i + s_i/τ_i = 2G_i ė, ṗ_j + p_j/τ_j = K_j θ̇.

**Update.** For a strain that is linear in time over a step Δt, a = Δt/τ, r = e^{-a}, φ(a) = (1 − e^{-a})/a (φ(0) = 1,
evaluated with `expm1`), the exact solution of the branch ODE is

s_i ← r s_i + 2G_i φ_i Δe,  p_j ← r p_j + K_j φ_j Δθ.

Algorithmic tangent (stress is linear in the strain at the step): 2(G∞ + ΣG_iφ_i) P_dev + 3(K∞ + ΣK_jφ_j) P_vol, which in
the Voigt convention above is (K∞ + ΣK_jφ_j) 1⊗1 + 2(G∞ + ΣG_iφ_i) I_dev with I_dev = diag(1, 1, 1, ½, ½, ½) − 1⊗1/3.
`tangent_kind="continuum"` returns the instantaneous (Δt → 0, φ = 1) tangent K0 1⊗1 + 2G0 I_dev. The state is
`strain_prev` (n, 3, 3), `s` (n, n_shear, 3, 3), `p` (n, n_bulk); the previous strain is part of the state, so a restart
needs only the committed state. A step with Δt = 0 is an instantaneous elastic probe (the instantaneous moduli).

**Energy and dissipation.** ψ = G∞ e:e + ½K∞θ² + Σ s_i:s_i/(4G_i) + Σ p_j²/(2K_j); branches with a zero amplitude contribute 0
(no division by zero). The dissipation rate is Σ s_i:s_i/(2G_iτ_i) + Σ p_j²/(K_jτ_j). It is integrated **exactly** over
the step along the exact branch response to the piecewise-linear strain (not as work minus Δψ). With m = s_n/(2G_i),
q = 1 − e^{-a}, g3(a) = (a − q − q²/2)/a² (Taylor series for a < 1/4) the shear branch increment is

ΔD_i = 2G_i [ ½ q(2 − q) m:m + a φ² m:Δe + g3(a) Δe:Δe ],

and the bulk branch the same with K_j and scalars. It is a quadratic form that is non-negative up to rounding. Closure is
not assumed: the tests compare ψ and ΣD with an independent ODE oracle that accumulates ∫ rate dt and ∫ σ:ε̇ dt.

## Uniaxial behaviour is not pointwise

E(t) = 9K(t)G(t)/(3K(t) + G(t)) is **not** the uniaxial relaxation modulus (and 1/E(t) is not the creep compliance) unless
the two spectra are proportional (constant ν). With K̄(s) = sK̂(s) etc. the operational modulus is Ē(s) = 9K̄Ḡ/(3K̄ + Ḡ).
For an elastic shear modulus G and one bulk branch (K∞, K1, τ) it gives the SLS E_R(t) = E∞' + (E0' − E∞')e^{-t/τ'},
E∞' = 9K∞G/(3K∞ + G), E0' = 9K0G/(3K0 + G), τ' = τ(3K0 + G)/(3K∞ + G), which is used as an analytic test and as the
negative control against the pointwise formula. For uniaxial stress the lateral stresses are solved from the history at
every step (`drive_material_point` with the lateral stresses controlled to zero); the creep compliance is exactly
D(t) = J_G(t)/3 + J_K(t)/9 with the shear and bulk creep compliances J_G, J_K of the two spectra. The apparent Poisson ratio is
protocol dependent. Plane stress is refused by the shared history helper (it needs this local solve with the history inside
the element); plane strain embeds the 3-D state.

## Time–temperature superposition (horizontal shift only)

a_T = τ(T)/τ(T_ref), reduced time ξ = ∫dt/a_T(T(t)). WLF: log10 a_T = −C1(T − T_ref)/(C2 + T − T_ref); a temperature with
C2 + T − T_ref ≤ 0 is refused; optional calibrated range [T_min, T_max]. Arrhenius: ln a_T = (E_a/R)(1/T − 1/T_ref), T in
kelvin. Both are evaluated as ln a_T (log domain) and a_T(T_ref) = 1 exactly. The reduced increment of a step with T linear
in the step is exact for an isothermal step (Δt exp(−ln a_T)); otherwise 5-point Gauss–Legendre on sub-intervals with
|Δ ln a_T| ≤ 0.5 each. A reduced increment that overflows raises; one that underflows is the limit 0. `GeneralizedMaxwell`
with a shift uses Δξ in place of Δt, with fixed amplitudes. Approximation: the strain is taken linear in ξ over the step
(exact for constant T in the step; a first-order effect for stiff branches and second-order otherwise under a temperature
ramp; the measured convergence is in the prereg record).

## DMA

Convention e^{iωt}, ω in rad/s (`frequency_hz_to_rad` converts Hz). G*(ω) = G∞ + ΣG_i iωτ_i/(1 + iωτ_i) = G' + iG'', storage
G' = G∞ + ΣG_i(ωτ_i)²/(1 + (ωτ_i)²), loss G'' = ΣG_i ωτ_i/(1 + (ωτ_i)²) ≥ 0, tan δ = G''/G' (NaN where G' ≤ 0); K* the same.
E* = 9K*G*/(3K* + G*) and ν* = (3K* − 2G*)/(2(3K* + G*)) are the operational (harmonic) isotropic moduli, not the pointwise
time functions. For ε = ε0 sin ωt the stress is σ = ε0 (G' sin ωt + G'' cos ωt); the dissipated energy per cycle of a
simple shear γ0 sin ωt is π G'' γ0². Tension, flexure and shear DMA modes are not interchangeable without a geometry
correction, which is not implemented.

## What is verified (see the prereg record for the measured values) and what is not

Verified by the tests (V1–V10, C1): scalar closed forms against the defining ODEs and Laplace identity; independent bulk and
shear relaxation; irregular ramp/hold/unload with Δt/τ from 1e-10 to 1e6 against hereditary oracles; uniaxial stress
relaxation and creep for proportional and non-proportional spectra against analytic, DAE and residue oracles, with a negative
control; tangent against central finite differences; energy and dissipation against an independent oracle and the
trapezoid closure under refinement; shift factors, reduced time, and the nonisothermal update against an ODE with
a_T(T(t)) inside; DMA against closed forms and a time-domain steady cycle; restart, subdivision, rejection; invalid input.

**Not implemented / not verified:** temperature-dependent amplitudes (vertical shift) and a thermodynamically consistent
nonisothermal energy; finite-strain viscoelasticity; anisotropic or orthotropic spectra; plane stress in the model's own
`evaluate`; state VJP/JVP and parameter derivatives; fractional kernels; spectrum fitting and its integration with parameter
identification; creep as a separate WP13 law. The energy and the dissipation are defined at a fixed temperature (isothermal
or with the horizontal shift only; with a shift the dissipation is the dissipation of the reduced-time system). No real polymer
is validated: every parameter set in the tests is synthetic.
