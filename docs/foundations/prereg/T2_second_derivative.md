# T2_second_derivative — second Fréchet derivative and Hessian-vector product of spectral functions (WP03 T02)

Preregistered 2026-11-27 06:51 +0300 (output of `date`), base SHA `d76ed15`, before any test of this record was run.
Module addition: `core/spectral.py` (`spectral_second`, `spectral_hvp`). Tests:
`tests/toporlogy/test_spectral_second.py`. Contract: `docs/foundations/TENSOR_CONTRACT.md`. First derivative and its
record: `prereg/T_spectral.md` (TS1-TS9, verified).

## Derivation (Daleckii–Krein, second order)

A = Q diag(λ) Qᵀ symmetric, f smooth on the spectrum. With H̃ = QᵀHQ, K̃ = QᵀKQ,

    d²f(A)[H,K] = d²/(ds dt) f(A + sH + tK) |₀ = Q [ M ] Qᵀ,
    M_ij = Σ_k F_ijk ( H̃_ik K̃_kj + K̃_ik H̃_kj ),         F_ijk = f[λi, λj, λk]

the second divided difference (symmetric in all three arguments). Checks written here, derived by hand:
f(x) = x²: F ≡ 1, M = H̃K̃ + K̃H̃ (d²(A²)[H,K] = HK + KH ✓). f(x) = x³: F = λi + λj + λk, M = Λ(HK+KH) + (HK+KH)Λ + HΛK +
KΛH, which are the six ordered words of (A+sH+tK)³ containing one H and one K ✓. Normalisation: **no ½**; the
operator is the full mixed second derivative, symmetric in (H, K) (Taylor: f(A+H) = f + L[H] + ½ d²f[H,H] + …).

Second divided difference for nodes x, y, z: f[x,y,z] = (f[x,y] − f[y,z])/(x − z) with f[x,y] = D(x,y) the first
divided difference of the existing code. Confluent limits: f[a,a,a] = f''(a)/2; f[a,a,c] = (f[a,c] − f'(a))/(c − a);
in general f[x,y,z] = Σ_{k≥0} f^{(k+2)}(m)/(k+2)! · h_k(x−m, y−m, z−m), h_k the complete homogeneous symmetric
polynomial (m any centre; used with m = mean of the nodes).

Stable evaluation (no eigenvalue perturbation, equal nodes exact):
* sqrt: F = −1/((√x+√y)(√x+√z)(√y+√z)) (algebra from D = 1/(√x+√y); no cancellation, valid at equal nodes; at
  x=y=z=s it is −1/(8 s^{3/2}) = f''/2 ✓).
* inv_sqrt: F = (√x+√y+√z)/(√x√y√z (√x+√y)(√x+√z)(√y+√z)) (derived from g = 1/√; at x=y=z=s: 3/(8 s^{5/2}) = f''/2 ✓).
* exp, log, power(p): nodes sorted lo ≤ mid ≤ hi, spread s = hi − lo. If s is not small, F = (D(hi,mid) − D(mid,lo))/(hi − lo)
  (D by the stable forms of the first derivative; the denominator is the largest gap, so the loss is ≈ ε/ρ with ρ the
  relative spread). If small (exp: s < 1e-2; log, power: s/m < 1e-3), the Taylor series about m = mean with 12 terms
  (coefficients exp: e^m/j!; log: (−1)^{j−1}/(j m^j); power: C(p,j) m^{p−j}), truncation ≲ (10 ρ)^{12}.
  All-equal nodes use only the first term: exactly f''/2.

Hessian-vector product: φ(A) = ⟨G, f(A)⟩ (G symmetric). dφ[H] = ⟨G, L_A[H]⟩; the gradient is L_A[G] (self-adjoint).
Its derivative along H is d²f(A)[H, G] (derivative of A ↦ L_A[G] in direction H). Hence
**HVP(A, G, H) = d²f(A)[H, G]**, and ⟨K, HVP⟩ = ⟨G, d²f[H, K]⟩ = d²φ(A)[H, K], symmetric in (H, K).

Block-matrix oracle (independent of the production formula). Let M = [[A,H,0],[0,A,K],[0,0,A]]. For analytic f,
f(M) is block upper triangular with f(A) on the diagonal, L_A[H], L_A[K] on the first superdiagonal, and the (1,3)
block X(H,K) = Q[Σ_k F_ijk H̃_ik K̃_kj]Qᵀ (ordered sum, H before K). Check f = x³: the (1,3) block of M³ is
AHK + HAK + HKA, the three words with H before K ✓. Hence **d²f(A)[H,K] = X(H,K) + X(K,H)**: the (1,3) block of
f(M(H,K)) plus that of f(M(K,H)); one block alone is "half" (not symmetric in general).

## Test IDs, references, thresholds

Relative error = max|a−b| / max|b|. tol(κ) = 1e-10 for κ ≤ 1e2, 1e-12·κ for κ > 1e2 (so 1e-8 at κ = 1e4).

| ID | Check | Reference | Threshold |
|---|---|---|---|
| SD1 | random SPD 3×3 and 2×2 (cond κ ∈ {1e1, 1e2, 1e3, 1e4}), random symmetric H, K, f ∈ {exp, log, sqrt, inv_sqrt, power 3, −1.5, 0.25}: d²f[H,K] vs X(H,K)+X(K,H) of the 3n×3n block: exp `scipy.linalg.expm`; sqrt, log `scipy.linalg.sqrtm/logm`; inv_sqrt and powers mpmath (40 digits; `sqrtm`, `expm(p logm)`) | block matrices | tol(κ); for sqrt/log via scipy the reference itself is limited (defective block): max(tol(κ), 1e-8) |
| SD2 | A = Q diag(2,2,5) Qᵀ, Q diag(3,3,3) Qᵀ, Q diag(2, 2+1e-10, 5) Qᵀ (random Q, 20 each), all f of SD1, vs mpmath block (40 digits, the stored float A, H, K) | mpmath block | tol(κ), κ = 2.5 or 1 → 1e-10 |
| SD3 | scalar level (private `_second_divided_differences` vs mpmath 60 digits from the definition, coincident nodes by the limit formulas above): spectra [a, a+δ, c], δ ∈ {0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1}, a ∈ {0.5, 2, 40}, c ∈ {a + 3, 2.5a}, all f | mpmath divided differences | 1e-11 |
| SD4 | symmetry d²f[H,K] = d²f[K,H] and the result is a symmetric matrix, random SPD and repeated spectra | self-consistency (labelled) | 1e-13 |
| SD5 | closed forms: [3,3,3] gives f''(3)/2 (HK+KH) with f'' written in the test; power 2 → HK+KH; power 3 → six-word sum; power −1 → A⁻¹HA⁻¹KA⁻¹ + A⁻¹KA⁻¹HA⁻¹; exp vs the word series Σ_{p,q,r} A^p H A^q K A^r/(p+q+r+2)! (+ swap), 70 terms, random A (‖A‖ ≤ 5), [2,2,5], [3,3,3] | closed forms and series, dense numpy in the test | 1e-12 (series 1e-11) |
| SD6 | central FD of the **first** derivative: (L_{A+hK}[H] − L_{A−hK}[H])/(2h), h ∈ {1e-1…1e-6}, ‖K‖₂ = ‖H‖₂ = 1, A with spectrum in [1, 5] and [2,2,5]: min over h of relative error ≤ 1e-7, and error(1e-2)/error(1e-3) ≥ 30 (O(h²)); the first derivative is verified in T_spectral (TS2-TS5), so this is independent of the second-derivative formula | FD of verified `spectral_frechet` | 1e-7; slope ratio 30 |
| SD7 | HVP: `spectral_hvp(A,G,H)` vs central FD of `spectral_vjp(A±hH, G)` (min over the h sweep ≤ 1e-7); ⟨K, hvp(G,H)⟩ = ⟨G, d²f[H,K]⟩ and Hessian symmetry ⟨K,hvp(G,H)⟩ = ⟨H,hvp(G,K)⟩ (labelled consistency, same kernel) | FD of verified first derivative; identities | 1e-7; 1e-12 |
| SD8 | batch (5,3,3) with broadcast H (3,3), K (5,3,3) vs a loop of single calls; rotation covariance d²f(RARᵀ)[RHRᵀ,RKRᵀ] = R d²f Rᵀ (labelled) | loop; covariance | 1e-13 |
| SD9 | invalid inputs: non-symmetric H or K or A, NaN, shapes (4,4), (3,), (), mismatched n of H/K/A, unknown f, power without p, p with another f, non-SPD A for log/sqrt/inv_sqrt/power → ValueError; exp of an indefinite A works (vs mpmath block); `spectral_hvp` the same refusals | rule | exact |
| SD10 | negative control: the naive recurrence (D(a,b) − D(b,c))/(a − c) at nodes [2, 2+1e-10, 2+2e-10], log, evaluated in the test, differs from the mpmath value by relative error > 1e-8 (the test can fail) | naive formula written in the test | > 1e-8 |

Not claimed: torch or double backward; float32; κ > 1e4 for the second derivative; overflow of e^λ, λ^p for extreme
eigenvalues (unguarded, as in the first derivative).
