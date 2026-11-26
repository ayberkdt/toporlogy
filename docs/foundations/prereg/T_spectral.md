# T_spectral — spectral functions of symmetric tensors (WP03 T02/T04) and aggregate moduli (WP04 E02/E04/E05)

Preregistered 2026-11-27 01:29 +0300, base SHA `df2ca78`, before any test of this record was run. Modules: new
`core/spectral.py` (`spectral_function`, `spectral_frechet` = JVP, `spectral_vjp`, `SPECTRAL_FUNCTIONS`); additions to
`core/anisotropy.py` (`aggregate_moduli`, `universal_anisotropy_index`, `zener_ratio`). Tests:
`tests/toporlogy/test_spectral.py`, `tests/toporlogy/test_moduli_foundations.py`. Contract:
`docs/foundations/TENSOR_CONTRACT.md`.

## Already present (read first, not duplicated)

* E1 pair constructors and the ν sweep: `isotropic_moduli`, `isotropic_sweep`; tests `test_linear_elastic.py::test_l1_*`
  and `test_anisotropy.py::test_p6_*`. Not repeated here.
* Plane-stress reduction: `core/material.plane_stress` computes Q = (S_pp)⁻¹ from the 3-D compliance, which is
  mathematically the Schur complement of C eliminating zz, yz, xz (verified against `_schur` in
  `test_linear_elastic.py::test_l7_*`). No new function is written; E5 below tests it on a rotated full tensor with a
  3-D solve as reference.

## Mathematical domain and design

A symmetric 2×2 or 3×3 (batched over leading axes), float64, A = Q Λ Qᵀ via `numpy.linalg.eigh`. f ∈ {exp, log, sqrt,
inv_sqrt, power(p)}. log, sqrt, inv_sqrt, power need SPD: any computed eigenvalue ≤ 0 or non-finite → ValueError, never
clipped. exp has no domain restriction. Value f(A) = Q f(Λ) Qᵀ. Fréchet derivative L_A[H] = Q [D ⊙ (QᵀHQ)] Qᵀ with
D_ij = (f(λi) − f(λj))/(λi − λj), D_ii = f'(λi), evaluated by stable forms (never separating eigenvalues): sqrt
1/(√a+√b); inv_sqrt −1/(√a√b(√a+√b)); log log1p((hi−lo)/lo)/(hi−lo) with hi/lo = max/min; exp e^hi · (−expm1(−d))/d,
d = hi−lo; power(p) hi^(p−1) expm1(pℓ)/expm1(ℓ), ℓ = −log1p((hi−lo)/lo); exact equality → f'(λ). Only symmetric
perturbations H are supported (a non-symmetric H → ValueError). VJP: for symmetric A, L is self-adjoint under the
Frobenius product, so the gradient with respect to a symmetric A of ⟨G, f(A)⟩ is L_A[G] for symmetric G. Second
derivatives are NOT implemented (documented, no stub). Torch is not provided (not cheap: needs a custom backward;
no claim is made).

Aggregate moduli of a 6×6 stiffness (Voigt, engineering shear): with C4 = C_ijkl and S4 its inverse,
K_V = C_iijj/9, G_V = (C_ijij − C_iijj/3)/10, K_R = 1/S_iijj, G_R = 5/(2(S_ijij − S_iijj/3)); Hill = ½(V+R);
A_U = 5 G_V/G_R + K_V/K_R − 6 ≥ 0 (zero iff isotropic); Zener Z = 2C44/(C11−C12) defined only when
`symmetry_class` is "cubic" or "isotropic" **in the given frame** (a rotated cubic tensor is refused: its cubic axes
are not aligned and the formula would be wrong).

## Test IDs, references, thresholds

Relative error = max|a−b| / max|b| unless stated.

| ID | Check | Reference (independent of production code) | Threshold |
|---|---|---|---|
| TS1 | f(A) at diag(1,4,9): sqrt = diag(1,2,3), inv_sqrt, log, exp, power(3), power(−1) | closed forms; for random rotated SPD (cond ≤ 1e4, 2×2 and 3×3) mpmath `sqrtm/logm/expm` at 40 digits | 1e-12 |
| TS2 (T3) | JVP at diag(1,4,9), off-diagonal H, f ∈ {sqrt, log} (and exp, inv_sqrt) | analytic divided differences written in the test; mpmath central difference (60 digits, h=1e-20) of mp.sqrtm/logm/expm; Sylvester X S + S X = H with `scipy.linalg.solve_sylvester` for sqrt; double-precision central FD step sweep h = 1e-2…1e-8 (best step) | mpmath and analytic 1e-12; Sylvester 1e-12; FD sweep best-step error ≤ 1e-6 |
| TS3 (T4) | A = Q diag(2,2,5) Qᵀ (random Q), A = 2I: log derivative at 2I = H/2, sqrt derivative = H/(2√2); repeated-eigenvalue projector formula f'(λ1)P1HP1 + f'(λ2)P2HP2 + dd(P1HP2+P2HP1); exp vs `scipy.linalg.expm_frechet`; log via inverting the linear map `expm_frechet(logm(A), ·)`; sqrt via Sylvester with `scipy.linalg.sqrtm`; inv_sqrt via −S⁻¹XS⁻¹; power(2) = HA+AH, power(−1) = −A⁻¹HA⁻¹ | as listed | 1e-12 |
| TS4 | near repeat A = Q diag(2, 2+1e-10, 5) Qᵀ: derivative finite and equal to the mpmath central difference; continuity to the exact repeat (‖L_near − L_exact‖/‖L_exact‖ ≤ 1e-9) | mpmath 60 digits | 1e-10 vs mpmath |
| TS5 | randomly rotated repeated eigenspaces: 40 random Q, spectra [1,1,1], [1,1,4], [1,1+1e-10,4], [1,4,4]: covariance L_{QAQᵀ}[QHQᵀ] = Q L_A[H] Qᵀ (A diagonal exact) and projector oracle (no eigenvectors in the degenerate subspace) | projector closed form; covariance is a symmetry check (labelled) | 1e-12 (exactly repeated), 1e-10 (near repeat) |
| TS6 | VJP/JVP duality ⟨L[H],G⟩ = ⟨H,L[G]⟩ (labelled consistency, same kernel); VJP against double-precision central FD of φ(A)=⟨G, f(A)⟩ over a symmetric basis (independent) | FD of the value function | duality 1e-12; FD 1e-6 |
| TS7 (T4) | scales 1e-12…1e12 (A = s B): sqrt(sB) = √s sqrt(B), log(sB) = log s·I + log B, JVP homogeneity (sqrt: L_{sA}[H] = L_A[H]/√s; log: L_{sA}[H] = L_A[H]/s); condition targets 1, 1e2, 1e4, 1e8 against mpmath FD (error ≤ 1e-12 + 1e-14·κ); cond 1e12 is reported (finite or ValueError, no accuracy claim, not asserted) | scaling laws; mpmath | 1e-10 (scaling), κ-scaled as written |
| TS8 | invalid inputs: eigenvalue ≤ 0 for log/sqrt/inv_sqrt/power(p) (including exactly 0 and −1e-30·scale) → ValueError; exp of an indefinite A works; non-symmetric A or H, NaN/inf, shapes (4,4), (3,), (), (2,3), unknown f, power without p, p with another f → ValueError; batch broadcast of A (5,3,3) against H (3,3), (5,3,3), (2,1,3,3) equals a loop | the rule itself; loop of single calls | exact / 1e-14 |
| TS9 | 2×2 matrices: JVP and values vs mpmath, repeated case | mpmath | 1e-12 |
| MO1 (E4) | isotropic C rotated by random R: K_V = K_R = K, G_V = G_R = G (independent of R), A_U = 0, Hill = K, G, Zener of an aligned isotropic = 1 | K, G of the construction; rotation by explicit einsum in the test | 1e-12 |
| MO2 (E4) | cubic (C11,C12,C44) = (2.0, 1.0, 1.5) (C11−C12 ≠ 2C44, synthetic): Z = 2C44/(C11−C12) = 3 ≠ 1, A_U > 0 and equals closed form 5G_V/G_R − 1 with K_V = K_R = (C11+2C12)/3, G_V = (C11−C12+3C44)/5, G_R = 5(C11−C12)C44/(4C44+3(C11−C12)); a rotated copy has the same aggregate moduli; `zener_ratio` of the rotated copy and of orthotropic / transversely isotropic input → ValueError | closed forms of cubic crystals | 1e-12 |
| MO3 (E4) | random SPD 6×6 (cond ≤ 1e4), 30 samples: K_V, G_V, K_R, G_R against the expanded component formulas K_V = (C11+C22+C33+2(C12+C13+C23))/9, G_V = (C11+C22+C33−(C12+C13+C23)+3(C44+C55+C66))/15, K_R = 1/(S11+S22+S33+2(S12+S13+S23)), G_R = 15/(4(S11+S22+S33)−4(S12+S13+S23)+3(S44+S55+S66)) (S = numpy.linalg.inv in the test); inequalities K_V ≥ K_R, G_V ≥ G_R, A_U ≥ 0 | expanded formulas | 1e-10 (κ ≤ 1e4); inequalities with −1e-12 slack |
| MO4 | Hill = ½(V+R); E_H, ν_H from (K_H, G_H); rotation invariance of all outputs for random SPD tensors; invalid input (non-symmetric, non-SPD, wrong shape) → ValueError | formulas; rotation by einsum | 1e-12 / exact |
| E5 | rotated full 3-D tensor (orthotropic synthetic, random rotation): `material.plane_stress(C) @ ε_p` equals σ_p from the 3-D system with σ_zz=σ_yz=σ_xz=0 solved densely for the out-of-plane strains; the 3-D stress reconstructed from `plane_stress_recovery` has σ_o = 0 to 1e-12 relative; `plane_strain` with ε_zz=ε_yz=ε_xz=0 reproduces σ_p; negative control: Schur complement eliminating only zz differs from plane_stress by > 1e-3 for the coupled rotated tensor (and agrees to 1e-13 for a rotation about z) | dense 6×6 solve in the test | 1e-12; control > 1e-3 |

Negative controls: TS8 (invalid inputs), MO2 (rotated cubic and non-cubic refused), E5 (σ_zz only), TS5 with a
deliberately perturbed eigenvalue pair (A with eigenvalues separated by 1e-3 is not treated as repeated: the
equal-eigenvalue formula applied to it differs from the true derivative by > 1e-6), confirming the test can fail.

Expected limiting behaviour: relative eigenvalue accuracy of `eigh` is ε·λmax/λmin per eigenvalue, so at κ near 1e12 a
small eigenvalue can lose all digits and may come out ≤ 0 (then ValueError, as designed). Eigenvectors inside a
cluster are not accurate; the derivative formula does not depend on them to first order in the cluster width.

Run command: `python -m pytest tests/toporlogy/test_spectral.py tests/toporlogy/test_moduli_foundations.py -q -p no:randomly`
(plus `tests/toporlogy/test_linear_elastic.py tests/toporlogy/test_anisotropy.py` as regression); small matrices,
seconds, negligible memory.

First result (append after run): see below.

## First result (2026-11-27, base 828d9be plus production code and test files)

`python -m pytest tests/toporlogy/test_spectral.py tests/toporlogy/test_moduli_foundations.py
tests/toporlogy/test_linear_elastic.py tests/toporlogy/test_anisotropy.py -p no:randomly`: 201 passed, 2 xfailed (the
two xfails are historical ones of the existing files, untouched). New tests: 117 in `test_spectral.py`, 13 in
`test_moduli_foundations.py`.

| ID | Result | Measured |
|---|---|---|
| TS1 | PASS | within the 1e-12 threshold (pass/fail only; per-case maxima not logged) |
| TS2 (T3) | PASS | analytic, mpmath FD and Sylvester within 1e-12; FD sweep best step <= 1e-6 |
| TS3 (T4) | PASS | 2I limits, projector oracle, expm_frechet, inverse-expm_frechet log, Sylvester sqrt within thresholds |
| TS4 | PASS | near repeat (delta 1e-10) within 1e-10 of mpmath, continuity within 1e-9 |
| TS5 | PASS | 40 rotations x 3 spectra x 7 functions within 1e-12; near-repeat 1e-9; negative control differs by > 1e-6 as required |
| TS6 | PASS | duality 1e-12 (consistency); VJP vs independent FD gradient 1e-6 |
| TS7 | PASS; kappa 1e12 reported only | scale laws 1e-10; kappa 1, 1e2, 1e4, 1e8 within 1e-12 + 1e-14 kappa; kappa 1e12 log-derivative relative error 4.0e-06 vs a 100-digit mpmath eigsy reference (one sample, not asserted) |
| TS8 | PASS | all invalid inputs refused; batch broadcast equals loop |
| TS9 | PASS | 2x2 values 1e-12, derivatives 1e-12 (1e-10 near repeat) |
| MO1-MO4 | PASS | within thresholds (rotated isotropic A_U <= 1e-12; cubic (2, 1, 1.5): Zener 3, A_U closed form 5.6 to 1e-12) |
| E5 | PASS | plane stress/strain vs dense 3-D solve 1e-12; sigma_zz-only control differs by > 1e-3, rotation-about-z control agrees to 1e-13 |

Test set-up corrections (no threshold changed, no production change):
1. mpmath returns `mpc` entries from `sqrtm/logm/expm`; `float()` failed in the first run. Fixed with `mp.re`.
2. TS7 kappa 1e12 report: `mp.logm` does not converge at that condition number (ZeroDivisionError), so the reference
   for this report-only case is a central difference of an `mp.eigsy`-based log at 100 digits. The case is not asserted.

Deviations from the preregistration text: the 2x2 near-repeat case uses the TS4 threshold 1e-10 (the TS9 row said
1e-12 for the whole 2x2 batch; the separated and exactly repeated 2x2 cases use 1e-12). No production fault found, so
no ledger entry. Limitation: only pass/fail against the thresholds is on record, apart from the values stated.
Not done: torch path, second derivatives, float32/GPU (documented in `TENSOR_CONTRACT.md`).
