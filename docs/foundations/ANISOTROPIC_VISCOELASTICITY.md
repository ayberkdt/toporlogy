# Anisotropic linear viscoelasticity at the material point (WP12 V05)

Module `core/viscoelastic_aniso.py` (`AnisotropicMaxwell`, `isotropic_branch_tensor`); preregistration
`docs/foundations/prereg/VA_anisotropic_ve.md`; tests `tests/toporlogy/test_viscoelastic_aniso.py`. All numbers in the
tests are **synthetic**; nothing here is material data and no real polymer is claimed to be validated.

## Model

σ = C∞ ε + Σ_i s_i,  ṡ_i + s_i/τ_i = C_i ε̇  (small strain, constant properties, isothermal).

* C∞: symmetric positive definite 6×6 (positive semi-definite only with `allow_fluid=True`, then `is_fluid` is True,
  the quasistatic stiffness is singular, and C∞ + ΣC_i must still be positive definite).
* C_i: symmetric positive **semi**-definite 6×6, τ_i > 0 [s]. A zero tensor is allowed.
* Tensors are given in this repository's Voigt convention ([xx, yy, zz, yz, xz, xy], physical shear stress,
  engineering shear strain) and are **checked and used in Mandel form** M = diag(m) C diag(m), m = (1,1,1,√2,√2,√2),
  where σ_M = M ε_M and the Euclidean product of Mandel vectors is σ:ε. Eigenvalues, ranges and pseudo-inverses are
  statements about M; the Voigt matrix is not orthogonally similar to M, so its own eigenvalues and pseudo-inverse are
  not the physical ones when a matrix is singular.
* Refused with `ValueError` (never clipped): asymmetric or non-finite tensors, a Mandel eigenvalue below
  −1e-12 λ_max (`PSD_RTOL`) in any C_i or C∞, singular C∞ without `allow_fluid`, τ ≤ 0 or non-finite, an
  instantaneous stiffness that is not positive definite for a fluid, a rotation that is not a proper rotation.

## Update (exact for strain linear in the step)

a = Δt/τ_i, r = e^{−a}, φ = (1 − e^{−a})/a (stable via `expm1`, φ(0) = 1; `viscoelastic._phi`):

s_i ← r s_i + φ C_i Δε,  σ = C∞ ε + Σ s_i,  algorithmic tangent C∞ + Σ φ_i C_i (Voigt).

`tangent_kind="continuum"` returns the instantaneous tangent C∞ + ΣC_i. dt = 0 is a step jump (φ = 1, r = 1).
A hold step (Δε = 0) of any length is exact, so dt/τ from 1e-10 to 1e6 is stable (VA12).

## Energy and dissipation, singular branch tensors

ψ = ½ ε_Mᵀ M∞ ε_M + Σ ½ s_Mᵀ M_i⁺ s_M, M_i⁺ the Moore–Penrose pseudo-inverse of the Mandel matrix. **Handling of a
singular C_i:** eigen-decomposition of M_i; eigenvalues ≤ `RANGE_RTOL` λ_max (1e-12) define the null space (the
pseudo-inverse is zero there), the rest is inverted. The branch state stays in range(M_i) (it is built from C_i Δε and
decays), so ½ sᵀM⁺s is the exact free energy of the branch; a rank-deficient branch simply stores no energy outside
its range. Limit: the 1e-12 cut treats a genuinely soft but non-zero eigenvalue below that ratio as null.

With s in range, ψ̇_i = −s_iᵀM_i⁺s_i/τ_i + s_i·ε̇, hence the dissipation rate Σ s_iᵀM_i⁺s_i/τ_i ≥ 0 and
work = Δψ + ∫D. The increment of a step is the exact integral along the exact branch stress (q = 1 − e^{−a}):

D_step = Σ_i [ ½ q(2−q) s₀ᵀM⁺s₀ + a φ² s₀·Δε + g3(a) ΔεᵀMΔε ],  g3 = (a − q − q²/2)/a²

(`viscoelastic._g3`, Taylor series for small a). It is ≥ 0 by construction (each term is an integral of a PSD form) and
needs no differencing of work and energy. Isotropic C_i = 3K P_vol + 2G P_dev gives M⁺ = P_vol/(3K) + P_dev/(2G) and
reproduces the formulas of `GeneralizedMaxwell` (a shear-only branch is rank 5, a bulk-only branch rank 1).

## Orientation

`rotation=R` (3×3 proper rotation, material axes → global) rotates C∞ and every C_i once, C′ = R ⊛ C (`tensor.rotate4`),
and the pseudo-inverse by the orthogonal Mandel rotation of R. Objectivity: model(rotation=R) under ε(t) equals
R σ(RᵀεR) Rᵀ of the unrotated model. One orientation per model instance; a per-point orientation field is not
implemented.

## DMA

C*(ω) = C∞ + Σ C_i iωτ_i/(1 + iωτ_i) (e^{iωt}, ω in rad/s): `complex_modulus(ω, form="voigt"|"mandel")`,
`storage_loss(ω)`, `relaxation(t)` = C∞ + ΣC_i e^{−t/τ_i}. The loss tensor Σ C_i x/(1+x²) is positive semi-definite
(passivity) and the energy per cycle of ε = ε₀ sin ωt is π e₀ᵀ M″ e₀ (isotropic shear: π G″ γ₀²).

## Verified (prereg VA_anisotropic_ve; first-run values in the prereg)

Isotropic reduction against `GeneralizedMaxwell` (VA1, cross-implementation), orthotropic step relaxation against
C(t) = C∞ + ΣC_i e^{−t/τ_i} in all six strain directions (VA2), objectivity (VA3), `solve_ivp` hereditary reference
for an irregular multiaxial history on a rotated orthotropic material (VA4), tangent against central FD (VA5), ψ ≥ 0,
D ≥ 0 and D against Gauss–Legendre (VA6), energy closure with exact quadrature and second-order trapezoid refinement
(VA7), refusal list (VA8), bitwise restart and rollback (VA9), DMA passivity and cycle work (VA10), history contract
(VA11), stability against mpmath (VA12), and the negative control: a branch tensor with a negative eigenvalue is
refused and, simulated by ODE, produces negative work per cycle (NC1).

## Not done / limits

No temperature shift (WLF/Arrhenius) for the anisotropic model, no per-point orientation, no plane-stress
condensation (2-D plane strain only through the contract), no state VJP/JVP or adjoint, no finite strain, no
separate shift or spectra per direction beyond what branch tensors with their own τ_i give. Branch tensors are an
input; fitting them to data (and whether a given branch structure is physically supported for a given material) is not
addressed. Every number tested is synthetic.
