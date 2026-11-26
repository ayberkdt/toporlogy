# Mathematical conventions (WP02)

One convention set for the whole library. Every statement below is taken from the code named in brackets; where a
module uses a different variable, that is said explicitly.

## Indices, Voigt and Mandel

* Voigt order 3-D `[xx, yy, zz, yz, xz, xy]`, 2-D `[xx, yy, xy]` [`core/tensor.py`, `fem/mesh/kernels.strain_matrix`].
* Stress vectors carry the physical shear component σ_ij; strain vectors the engineering shear γ_ij = 2 ε_ij, so
  `stress = C @ strain` for a linear law and σ:ε = σ_vᵀ ε_v [`core/tensor.py` module notes, `core/constitutive.py`].
* Mandel is the orthonormal basis of symmetric tensors (shear components scaled by √2), used for spectra, condition
  numbers and positivity of 4th-order tensors: σ:ε = σ_Mᵀ ε_M [`core/tensor.tensor_to_mandel`,
  `mandel_eigenvalues`, `mandel_condition_number`].
* A non-symmetric tensor (e.g. PK1) is never packed into Voigt: `stress_to_voigt` checks symmetry with
  `SYMMETRY_RTOL = 1e-12` [`core/tensor.py`].

## Stress, strain and pressure

* Tension-positive Cauchy stress. Pressure p = −tr σ / 3 where a pressure is used [`core/viscoplastic.DruckerPrager`].
  The mixed u–p variable of `fem/mesh/mixed.py` is p = −λ div u (σ = 2μ ε − p I, compression positive; equal to the
  mean pressure −tr σ/3 when div u = 0) — a λ-based variable, not a K-based one; λ = 0 is refused there.
* Small strain ε = sym ∇u. Finite strain: H = F − I, Green–Lagrange E = ½(H + Hᵀ + HᵀH) formed from H
  [`core/tensor.green_lagrange`], PK2 S work-conjugate to E.
* Equivalent (von Mises) stress q = √(3/2 s:s), s = dev σ; equivalent plastic/creep strain rate ṗ with
  ε̇_p = ṗ N, N = 3s/(2q) [`core/constitutive.J2Plasticity`, `core/creep`, `core/viscoplastic`].
* 2-D: plane strain (ε_zz = 0) or plane stress (σ_zz = 0) are reductions of the 3-D law; the history models accept
  plane strain only, by embedding [`core/constitutive.small_strain_history_evaluate`].

## Elastic constants

* Isotropic: two independent moduli; K > 0, G > 0 is the canonical stable pair. E = 9KG/(3K+G),
  ν = (3K−2G)/(2(3K+G)), λ = K − 2G/3 [`core/linear_elastic.isotropic_moduli`].
* **Poisson ratio** ν_ij = −ε_j/ε_i under uniaxial stress along i, i.e. −S_ji/S_ii with S = C⁻¹
  [`core/anisotropy.poisson_ratio`, `core/linear_elastic.poisson_uniaxial`]. The stiffness ratio C12/C11 is not
  ν12 (it is ν21 of an uncoupled 2-D C); the homogenisation objective names both explicitly
  (`effective_poisson`, `convention: uniaxial | stiffness_ratio`; `prereg/E04_poisson.md`).
* Aggregate moduli: Voigt/Reuss/Hill and the universal anisotropy index A_U = 5G_V/G_R + K_V/K_R − 6; Zener ratio
  for cubic symmetry only [`core/anisotropy`].

## Units, time and temperature

* The legacy laws are unit-modulus: the FE physics multiplies stress, tangent and energy by the element modulus
  [`core/constitutive.py`]. The physical laws (`physical = True`: properties, viscoelastic, creep, viscoplastic, cure)
  take SI-consistent values (Pa, Pa·s, s, K, J/mol, J/(kg K)) and must not be scaled by the design modulus
  (`NonlinearElasticity` refuses them).
* Time is physical time in seconds (or one declared consistent unit) and enters through
  `MaterialRequest.step = TimeStep(t, dt, T, T_prev)`; a load factor, an arc-length parameter or an optimisation
  iteration is never time [`core/constitutive.TimeStep`, `require_step`].
* Temperatures in Arrhenius and WLF laws are absolute (K); differences ΔT are differences. R = 8.314462618 J/(mol K)
  [`core/viscoelastic`, `core/creep`, `core/cure`].
* Thermal strain from an instantaneous CTE: ε_th = ∫_{T_ref}^{T} α dT; a secant CTE is converted explicitly
  [`core/properties.ThermalExpansion`].

## Symbols with two meanings

| symbol | meaning A | meaning B | how the code separates them |
|---|---|---|---|
| C | stiffness (Voigt/Mandel) | right Cauchy–Green FᵀF | `C6`/`D` for stiffness; `right_cauchy_green` for kinematics |
| α | thermal expansion coefficient | degree of cure | `ThermalExpansion` (CTE) vs the state field `alpha` of `CuringThermoElasticPoint` and the `alpha` arguments of `core/cure` (degree of cure) |
| x, x̄ | design variable, filtered density | — | never a mass density; mass density is ρ [kg/m³] in physical records |
| p | pressure −tr σ/3 | SIMP exponent | `penal` for the SIMP exponent in the new modules |

## Verification hierarchy used in the preregistrations

V0 type/shape/domain validation; V1 exact algebra and invariants; V2 high-precision oracle (mpmath); V3 independent
material point; V4 element patch; V5 manufactured solution; V6 mesh/time convergence; V7 independent solver or
literature; V8 derivatives and adjoint; V9 integration/reproducibility/performance (02_CONVENTIONS_VV). A reference
that shares the production kernel is labelled "partially independent" or "regression".
