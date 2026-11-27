# Saddle-point systems: Schur complement and block preconditioning (WP05 L06)

Module `toporlogy.solvers.saddle` (numpy/scipy only; it does not import `toporlogy.fem`), tests
`tests/toporlogy/test_saddle.py`, preregistration and every measured number `prereg/L6_saddle.md`. All numbers below are from
synthetic unit-square problems (E = 1), not material data.

## Convention

`K [u; p] = [f; g]`, `K = [[A, Bᵀ], [B, −C]]`, A SPD (n×n), B (m×n), C SPD or zero (m×m). In the mixed classes
(`MixedElasticity`, `MixedKG`) `A` is the displacement block, `B = −∫ψ div φ`, `C = M/K_bulk` (`M` the pressure mass matrix), and
C = 0 is the exact constraint K_bulk = ∞. Eliminating `u = A⁻¹(f − Bᵀp)`:

    S p = B A⁻¹ f − g,        S = C + B A⁻¹ Bᵀ        (symmetric positive semi-definite)

With A and S positive definite, `inertia(K) = (n, m, 0)` (K ≅ diag(A, −S)); verified on dense systems and on the Taylor–Hood
matrices (`test_mixed_kg.py::test_x2b`).

## API

| Name | What it does |
|---|---|
| `factorised_solver(M, kind="lu"\|"ilu")` | `v ↦ M⁻¹ v` from a sparse LU (exact) or `spilu` (incomplete, AMG-free) |
| `SaddleOperator(A, B, C=None, A_solve=None)` | `matvec` (K v), `schur_apply` (S q = C q + B A⁻¹ Bᵀ q with one A-solve, S never formed), `schur_operator` (matrix-free), `reduced_rhs` (B A⁻¹ f − g), `recover_u`, `matrix()` (assembled sparse K) |
| `block_diagonal_preconditioner(Â, Ŝ, n, m)` | `diag(Â⁻¹, Ŝ⁻¹)`, SPD, for MINRES |
| `block_triangular_preconditioner(Â, Ŝ, B)` | `[[Â, Bᵀ], [0, −Ŝ]]⁻¹`, for GMRES |
| `saddle_solve(op, f, g, preconditioner, S_hat, ...)` | MINRES (`"diagonal"`, `"none"`) or GMRES (`"triangular"`) through `krylov_solve`; explicit residual of the original (f, g), separate u-row and p-row residuals, block-solve counters, restart on the explicit residual, pressure null space by compatibility check and projection |

Both preconditioners count their block solves (`n_A_solves`, `n_S_solves`) so the cost of an iteration is reported, not guessed.

## Why Ŝ = (1/G + 1/K) M (derived in the prereg, checked by dense eigenvalues)

For `MixedKG`, plane strain, `A = A_dev = 2G ∫ dev ε:dev ε` and `C = M/K`:

* **Upper bound.** `‖dev ε‖² ≥ (tr ε)²/6` in plane strain, so `vᵀAv ≥ (G/3)‖div v‖²` and, with `|∫ q div v| ≤ ‖q‖‖div v‖`,
  `qᵀ B A⁻¹ Bᵀ q ≤ (3/G) qᵀMq`. Hence `λ_max(Ŝ⁻¹S) ≤ 3` for every h and K. Measured: 0.83 (ν = 0.3), 0.75 (ν = 0.4999 and K = ∞),
  n = 4, 8, 16 (`L6b1a`).
* **Lower bound, on pressures orthogonal to constants.** `vᵀAv ≤ 2G‖∇v‖²` and the discrete inf-sup condition of Taylor–Hood pairs
  (β_h ≥ β > 0 independent of h; Brezzi–Fortin 1991, Elman–Silvester–Wathen 2014, **cited from memory, not re-read**) give
  `qᵀB A⁻¹ Bᵀ q ≥ β²/(2G) qᵀMq`. The eigenvalues of Ŝ⁻¹S on q ⟂ 1 therefore lie in a fixed interval. Measured with the constant
  removed: [0.11, 0.75] (ν = 0.4999 and K = ∞), [0.39, 0.83] (ν = 0.3), identical to 3 digits for n = 4, 8, 16: h-independent.
* **The constant is the exception (a failed prediction of the prereg, L6b1b).** On a fully constrained boundary `Bᵀ1 = 0`, so
  `S 1 = M 1/K`: the constant is an exact eigenvector of Ŝ⁻¹S with eigenvalue `(1/K)/(1/G + 1/K) = G/(G + K)` (2.0e-4 at
  ν = 0.4999; the dense eigenvalue agrees to 1e-8). The condition number on the whole space is therefore 3.7e3 at ν = 0.4999
  (limit K → ∞: the eigenvalue goes to 0, the mode becomes the gauge null vector, κ = 6.8-6.95 with it removed). One isolated small
  eigenvalue costs a bounded number of extra Krylov iterations; it is not a loss of h-independence.

## Gauge (K = ∞, every boundary DOF constrained)

`Bᵀ1 = 0` and `C = 0`: K has the null vector `(0, 1)` and `Kx = b` is solvable only if `1ᵀg = 0`. `saddle_solve(..., null_vector=1,
gauge_weights=M 1)` checks that `z` is a null vector (Bᵀz, Cz), refuses an incompatible `g` (`ValueError`), runs the Krylov
iteration on `g − z zᵀg/zᵀz` and returns `p − z (wᵀp)/(wᵀz)`, i.e. the discrete `∫p = 0` used by `MixedKG`'s bordered row. No
diagonal shift. The result equals the bordered direct solve (`Factorization` on `[[A, Bᵀ, 0], [B, 0, M1], [0, (M1)ᵀ, 0]]`) to the
Krylov tolerance in 24/24 sweep cases.

## What was measured (T6 and Q9 Taylor–Hood, clamped perturbed unit square, rtol 1e-8, exact LU for Â)

* Block-diagonal MINRES converges in all 24 cases (ν = 0.3, 0.49, 0.4999, 0.5; n = 8, 16, 32): explicit residual ≤ 9.9e-8
  (target 1e-7), counts T6 21-23 / 38-41 / 42-47 / 42-47 and Q9 19-22 / 29-36 / 33-34 / 33-34 (per ν, over n; sum over restarts).
  At fixed ν the count varies by ≤ 1.24 over a 16-fold change in the number of unknowns; ν = 0.4999 and K = ∞ agree within 1.
  The count is **not** independent of ν over 0.3 → 0.49 (21 → 41 for T6): the preregistered max/min ≤ 2 failed for T6 (2.24), passed
  for Q9 (1.89), and the cap of 40 failed for T6 (47).
* Unpreconditioned MINRES, T6 ν = 0.4999 n = 32: 500 iterations leave an explicit residual of 2.4e-2 (negative control). After a fixed
  300 iterations the explicit residual is 6.2e-5 at n = 8 and 4.1e-2 at n = 32 (post hoc).
* Block-triangular GMRES with exact blocks: 1 iteration in all 9 T6 cases (the preconditioned operator has the minimal polynomial of
  degree 2), 13-21 block solves per solve; this is the cheaper variant in iterations, not necessarily in work per iteration.
* Solutions equal the direct `Factorization(…, "symmetric_indefinite")` solve to 3.3e-8 (1-norm, relative); the κ-scaled bound is
  informative below ν = 0.4999 and nearly vacuous at ν = 0.4999, n = 32 (κ̂₁ ~ 1e7).

## Decisions and limits

* MINRES stops on its recursive, *preconditioned* residual estimate; the explicit Euclidean residual can be larger. `saddle_solve`
  therefore recomputes the explicit residual and restarts from the iterate (up to 3 times, rtol/10 each) until it meets `10 rtol`.
  The first version judged convergence against the tightened tolerance and mislabelled converged solutions "max_iterations" (found
  in run 1, fixed).
* **Incomplete LU is for GMRES only.** An ILU factor of an SPD matrix is not symmetric: as Â in the MINRES preconditioner it stopped
  on a wrong estimate (explicit residual 1.7e-6 and 1.3e-6 after 18 and 17 iterations, n = 8, 16; strict xfail L6e). With the
  block-triangular preconditioner and GMRES it converged (2 and 1 iterations). An SPD incomplete choice (incomplete Cholesky,
  symmetric Gauss–Seidel) and AMG were not implemented; the ILU counts carry no mesh-independence claim.
* Tested: 2-D plane strain `MixedKG`, T6–P1 and Q9–Q1 vertex pressure, scalar G and K. Not tested: 3-D, the λ-based class, P0 pressure
  or equal-order (unstable) pairs, heterogeneous G or K, rank-deficient B other than the constant-pressure null space, nonconforming
  boundary data (the data must satisfy the compatibility `1ᵀg = 0`). Ŝ = cM is derived for `MixedKG` only; for the λ-based class
  (A = ∫2μ ε:ε, p = −λ div u) the constants differ and were not derived.
* Not done (L07): preconditioner setup cost, RSS, fill and time sweeps, structured-versus-general solver benchmark. The block-solve
  counters are in place, but no timing or speed claim is made. The generalised symmetric eigenproblem and shift-invert part of L06 was
  not part of this slice.
