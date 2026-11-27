# Tensor contract — spectral functions and aggregate moduli

Scope: `toporlogy.core.spectral` (new; first derivative, second derivative and Hessian-vector product) and the aggregate-moduli additions to `toporlogy.core.anisotropy`. Evidence:
`docs/foundations/prereg/T_spectral.md` and `prereg/T2_second_derivative.md` (preregistrations, first results),
`tests/toporlogy/test_spectral.py`, `tests/toporlogy/test_spectral_second.py`,
`tests/toporlogy/test_moduli_foundations.py`. The rest of the tensor algebra stays as documented in
`core/tensor.py` (this file does not restate it).

## Spectral functions of symmetric tensors

`spectral_function(A, f, p=None)`, `spectral_frechet(A, H, f, p=None)` (JVP), `spectral_vjp(A, G, f, p=None)`,
`spectral_second(A, H, K, f, p=None)`, `spectral_hvp(A, G, H, f, p=None)`.

* **Input**: float64, shape `(..., n, n)`, n = 2 or 3, finite, symmetric to `tensor.SYMMETRY_RTOL` (relative). Other
  shapes (0-d, vectors, 4×4, non-square, empty) and non-symmetric input raise `ValueError`; nothing is silently
  symmetrised (the eigensolver reads the symmetrised copy of an input that already passed the check).
* **Functions** `f`: `exp`, `log`, `sqrt`, `inv_sqrt`, `power` (with finite exponent `p`; `p` with any other `f`
  is an error).
* **Domain**: all but `exp` need SPD. A computed eigenvalue ≤ 0 raises `ValueError`. There is no clipping, no
  eigenvalue shift, no regularisation.
* **Derivative**: L_A[H] = Q [D ⊙ QᵀHQ] Qᵀ with divided differences evaluated by the stable closed forms listed in the
  module docstring (sqrt 1/(√a+√b); log via `log1p`; exp via `expm1`; power via `expm1`/`log1p`); equal eigenvalues use
  the derivative limit f'(λ). No eigenvector of a degenerate subspace enters the result.
* **Second derivative**: `spectral_second(A, H, K, f)` = d²f(A)[H, K] = ∂²/∂s∂t f(A + sH + tK), the full mixed
  derivative (no factor ½), symmetric in H and K, for symmetric H, K. Formula (Daleckii–Krein, second order):
  Q [Σ_k F_ijk (H̃_ik K̃_kj + K̃_ik H̃_kj)] Qᵀ with F_ijk = f[λi, λj, λk] the second divided difference. F is evaluated
  as (D(hi,mid) − D(mid,lo))/(hi − lo) on sorted nodes, by a 12-term Taylor series about the node mean where the spread
  is small (exp: < 1e-2; log, power: < 1e-3 of the mean), by exact closed forms for sqrt and inv_sqrt, and equals f''/2
  for three equal nodes. No eigenvalue is perturbed and no eigenvector of a degenerate subspace enters.
  `spectral_hvp(A, G, H, f)` is the Hessian-vector product of φ(A) = ⟨G, f(A)⟩, i.e. d/dt ∇φ(A + tH) = d²f(A)[H, G];
  ⟨K, hvp⟩ = d²φ(A)[H, K], symmetric in (H, K). Same input, domain and batching rules as the first derivative.
* **Perturbations**: only **symmetric** H (JVP) and symmetric cotangent G (VJP). For symmetric A the map is
  self-adjoint under the Frobenius product, so VJP = the same map. A caller holding a general cotangent passes
  `sym(G)`, which is the gradient on the symmetric subspace. Derivatives on general (non-symmetric) matrices are not
  defined here; the non-symmetric matrix logarithm is out of scope.
* **Batching**: leading axes of A and H broadcast with NumPy rules.

### Derivative support matrix

| f | value | 1st derivative (JVP/VJP) | verified against | 2nd derivative (`spectral_second`, `spectral_hvp`) |
|---|---|---|---|---|
| exp | yes | yes | mpmath 60-digit FD, `scipy.linalg.expm_frechet`, projector oracle | yes |
| log (SPD) | yes | yes | mpmath FD, inverse of `expm_frechet(logm A, ·)`, analytic H/2 at 2I, projector oracle | yes |
| sqrt (SPD) | yes | yes | mpmath FD, Sylvester equation X S + S X = H (`solve_sylvester`), analytic H/(2√2) at 2I | yes |
| inv_sqrt (SPD) | yes | yes | mpmath FD, −S⁻¹XS⁻¹ with the Sylvester X | yes |
| power(p) (SPD) | yes (p = 3, −1.5, 0.25 tested; p = 2, −1, 1, 0 closed forms) | yes | mpmath FD, projector oracle, HA+AH, −A⁻¹HA⁻¹ | yes |

"Verified" means the cases in the preregistration table: spectra diag(1,4,9), Q diag(2,2,5)Qᵀ, 2I, near repeat
(δ = 1e-10), [1,1,1], [1,1,4], [1,4,4] with 40 random rotations each, scales 1e-12…1e12, condition numbers
1, 1e2, 1e4, 1e8 (also 2×2).

Second derivative, verified (`prereg/T2_second_derivative.md`, SD1-SD10) for all five rows: block-matrix oracle
`f([[A,H,0],[0,A,K],[0,0,A]])` — d²f[H,K] = (1,3) block for (H,K) plus (1,3) block for (K,H) — with `scipy.linalg`
expm/sqrtm/logm and mpmath (40 digits) on random SPD 2×2 and 3×3 with κ ∈ {1e1…1e4} (error ≤ 1e-10 for κ ≤ 1e2,
1e-12 κ above), repeated spectra [2,2,5], [3,3,3] and the near repeat [2, 2+1e-10, 5] with random rotations (1e-10),
scalar second divided differences against 80-digit mpmath over gaps 0…1 (1e-11), closed forms (x², x³, x⁻¹, triple
repeat f''/2), an exp word series, central differences of the first derivative and of `spectral_vjp` (step sweeps),
symmetry in (H, K), batching, rotation covariance and refusals. Not claimed: κ > 1e4 for the second derivative, torch /
double backward, float32, overflow guards for e^λ and λ^p.

### Regimes not supported or not claimed

* No torch path. A torch autograd route through `eigh` differentiates eigenvectors, which are not unique at repeated
  eigenvalues, and a double backward is not claimed (the NumPy second derivative above is the supported route). float32 and GPU are not run and not claimed.
* **Near singular** A: `eigh` determines small eigenvalues with absolute error of order ε λ_max. For κ near 1e12 the
  smallest eigenvalue may have no correct digits and may come out ≤ 0 (the call then refuses). The preregistered
  accuracy claim stops at κ = 1e8 (error ≤ 1e-12 + 1e-14 κ relative to the derivative norm). At κ = 1e12 one case was
  measured and only reported: log-derivative relative error 4.0e-6 against a 100-digit mpmath reference
  (`prereg/T_spectral.md`, TS7); this is one sample, not a bound.
* Nearly repeated eigenvalues inside a cluster: eigenvectors are inaccurate, the derivative is not (verified to 1e-10 at
  a gap of 1e-10). No guarantee is made for a gap below ε.
* Non-symmetric perturbations, non-symmetric A, n ∉ {2, 3}: refused.
* Overflow of eᵃ or a^p for extreme eigenvalues is not guarded (infinite or NaN results are the caller's to detect);
  the preregistered scale range 1e-12…1e12 is tested for sqrt and log only.

## Aggregate moduli and anisotropy indices (`core/anisotropy.py`)

* `aggregate_moduli(C6)`: Voigt, Reuss, Hill K and G (and the Hill isotropic E, ν) of a 6×6 Voigt stiffness (engineering
  shear, SPD, any frame) from the invariants K_V = C_iijj/9, G_V = (C_ijij − C_iijj/3)/10, K_R = 1/S_iijj,
  G_R = 5/(2(S_ijij − S_iijj/3)). K_V is the bulk modulus under uniform strain, K_R under uniform hydrostatic stress;
  neither is a directional compressibility.
* `universal_anisotropy_index(C6)` = 5 G_V/G_R + K_V/K_R − 6 ≥ 0, zero for isotropy (to rounding).
* `zener_ratio(C6)` = 2 C44/(C11 − C12), **cubic or isotropic in the given axes only**; any other class, and a cubic
  tensor whose cubic axes are rotated away from the frame, raises `ValueError`.
* Indices describe elastic anisotropy only; nothing is inferred about yield or strength.
* Plane stress: `core/material.plane_stress` (Q = (S_pp)⁻¹) is the Schur complement of C eliminating zz, yz, xz; it was
  tested on a rotated full tensor (E5) rather than re-implemented. Eliminating σ_zz alone is wrong for a coupled rotated
  tensor (negative control, relative difference > 1e-3).
* E1 (isotropic parameter pairs, ν sweep) is covered by the existing tests (`test_l1_*`, `test_p6_*`) and was not
  repeated.
