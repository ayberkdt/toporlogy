# Micromechanics of isotropic phases (WP18, slice MM + E7)

Module `src/toporlogy/core/micromechanics.py`; tests `tests/toporlogy/test_micromechanics.py`; derivations and results
`docs/foundations/prereg/MM_micromechanics.md`. All parameters in the tests are **synthetic** (not material data).

## What it does

| Function | Content | Status |
|---|---|---|
| `IsoPhase(K, G)`, `VOID` | isotropic phase (K, G > 0) or void (0, 0); anything else refused | — |
| `check_fractions` | each fraction in [0, 1], sum 1 within 1e-9, no clipping | — |
| `voigt`, `reuss` | uniform strain / uniform stress averages | rigorous for every microstructure |
| `hs_bounds(f, phases, dim)` | Hashin–Shtrikman–Walpole bounds, any number of phases, void allowed, 2-D (κ, μ) or 3-D | statistically isotropic composite, isotropic linear phases |
| `eshelby_sphere`, `dilute_sphere`, `mori_tanaka_sphere` | spherical inclusions in an isotropic matrix, full 6 × 6 Mandel solves | mean-field estimates |
| `mixture_density`, `volume_to_mass_fractions`, `mass_to_volume_fractions` | fraction conversion with densities | exact |
| `interpolate_moduli(x, A, B, scheme)` | independent K(x), G(x) of two solids, Voigt/Reuss/HS±, with dK/dx, dG/dx, E, ν, dE/dx | positive on [0, 1] |
| `required_interpolation_parameters`, `interpolation_admissibility` | when SIMP x^p / RAMP x/(1+q(1−x)) of a solid/void mixture stays below the HS upper bound | derived; grid-checked |

## Bounds: assumptions that matter

* **Shift form.** K = [Σ f_i/(K_i + K*)]⁻¹ − K*, G = [Σ f_i/(G_i + ζ)]⁻¹ − ζ with K* = 4G0/3, ζ = G0(9K0+8G0)/(6(K0+2G0)) (2-D: K* = G0,
  ζ = G0K0/(K0+2G0)). Upper bound: K0, G0 = largest K, largest G among the phases present; lower: smallest. These can belong to
  **different phases**. It is *not* "the HS formula with one phase as reference". For a crossed pair the bulk bounds are still
  attained by the coated-sphere assemblages (measured, test MM1d: upper bound = stiff-G coating, lower bound = soft-G coating,
  to 1e-15); for the shear bound of a crossed pair no attaining microstructure was checked here.
* **Void.** Lower bounds are 0 for any positive void fraction; the upper bound is finite and equals the exact hollow-sphere
  bulk modulus (Lamé solution, test MM1c).
* **No anisotropic claim.** HS bounds describe a statistically isotropic composite. A layered or otherwise anisotropic cell is
  bounded by Voigt/Reuss (Loewner order) only. Test MM4c compares the isotropic parts of a *random* cell with the 2-D HS bounds as
  a measured check, not as a theorem.
* **Ordering.** K_R ≤ K_HS− ≤ K_HS+ ≤ K_V and the same for G holds for every admissible input (monotonicity of the shift form,
  prereg §1; E7 sampled 200 cases including ≥ 20 crossed ones).
* **What was independently checked.** The bulk bound against Hashin's coated spheres solved in the test (ordered pair: both bounds
  attained to 1.8e-15; crossed pair: both coating orders lie inside and equal the two bounds to 1e-15). The shear bound was checked against the textbook two-phase form for ordered pairs (a closed form written from memory,
  agreeing algebraically with the shift form), against the existing void-limit forms of `design/interp.py`, and through the 2-D
  RVE measurement; there is **no independent exact shear solution** in these tests. No literature text was read; the references
  in `src/toporlogy/references` are marked "text not read".

## Mean-field schemes

Eshelby sphere S = α P_V + β P_D (Mandel), α = 3K/(3K+4G), β = 2(4−5ν)/(15(1−ν)). Dilute: A = [I + S C_m⁻¹(C_i − C_m)]⁻¹.
Mori–Tanaka: A_MT = A_dil[(1−f)I + fA_dil]⁻¹, C = C_m + f(C_i − C_m)A_MT.

* Mori–Tanaka with a matrix that has the larger K **and** G (stiff matrix, soft spheres, any contrast, also void) **is** the HS upper
  bound; with the softer matrix it is the HS lower bound (algebraic identity, prereg §2; measured agreement 4e-16). The draft
  brief said "lower bound for a stiff-matrix/soft-inclusion pair": that wording was wrong and is recorded as such.
* The dilute scheme is a first-order estimate: it is exact at f = 0 only. At f = 1 it returns for the test pair (matrix K = 10,
  G = 6; inclusion K = 2, G = 1) K = −4.4, a non-physical value. Use it only at small f.
* Not implemented: self-consistent scheme, ellipsoidal/fibre inclusions, anisotropic matrix, interphase, thermal expansion,
  viscoelastic homogenisation, laminate theory (plan items MM02–MM05 beyond fractions). No claim of effective properties of a
  concrete composite from these formulas.

## Interpolation admissibility

With a void second phase the HS bounds are RAMP-shaped: K_HS(x) = K x/(1 + q_K(1−x)), G_HS(x) = G x/(1 + q_G(1−x)), q_K = K/K*,
q_G = G/ζ (x = solid fraction). Hence a RAMP law with parameter q is admissible iff q ≥ max(q_K, q_G), and SIMP x^p iff p ≥ 1 +
max(q_K, q_G) (proof in prereg §3: x^{p−1}(1+q−qx) ≤ 1 on [0, 1] iff p − 1 ≥ q). Closed forms: 2-D plane stress
p_min = max(2/(1−ν), 4/(1+ν)) (≥ 3, equal to 3 only at ν = 1/3); 3-D p_min = max(3(1−ν)/(2(1−2ν)), 15(1−ν)/(7−5ν)). Examples:
ν = 0.3: 2-D 3.077, 3-D 2.625. This agrees with `design.interp.simp_admissible_penal` and with the condition attributed to Bendsøe &
Sigmund (1999), whose text was **not** read here. Plane strain gives other values (through κ = λ + μ). The statement concerns the
scalar law applied to K and G of a solid/void mixture with fixed ν; it says nothing about the physical realisability of grey
elements, and the modified SIMP with a stiffness floor is not covered.

## Fraction conversion

ρ = Σ v_i ρ_i, w_i = v_i ρ_i/ρ, v_i = (w_i/ρ_i)/Σ_j(w_j/ρ_j). A void (ρ = 0) is allowed in the forward direction (w = 0) and refused
in the inverse direction (its volume is not recoverable from mass fractions).

## Periodic RVE (MM4)

`fem.homogenization.homogenize_elasticity` of a three-layer 2-D plane-stress cell equals the exact layered average (relative
difference 1.1e-14) and lies between Voigt and Reuss in the Loewner order. Five random two-phase cells (24 × 24 elements, E = 1
and 10, exactly 50 % each, same ν = 0.3) lie between Voigt and Reuss, and their isotropic parts κ_cell, μ_cell lie inside the 2-D
HS bounds (measured: κ 2.15–2.25 inside [1.53, 2.60]; μ 1.21–1.26 inside [0.81, 1.36]). This is a measurement on a finite
sample, not a proof that every cell satisfies HS.
