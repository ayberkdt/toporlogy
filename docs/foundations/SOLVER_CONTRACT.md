# Solver contract (WP05 slice)

Code: `src/toporlogy/solvers/diagnostics.py`, `src/toporlogy/fem/mesh/linear.py` (`Factorization.solve_transpose`).
Preregistration and results: `prereg/L_solver.md`.

| Need | Function | Requirement enforced |
|---|---|---|
| Is the residual meaningful? | `solve_report` | η_N, Oettli–Prager ω on the original A; κ labelled given / 1-norm estimate; `forward_established` only if κ η_N ≤ 1e-8 |
| Definiteness | `inertia` (small dense) | Bunch–Kaufman LDLᵀ, 2×2 pivot blocks counted by their eigenvalues |
| Krylov solve | `krylov_solve` | CG: spd; MINRES: symmetric; GMRES/BiCGSTAB: any; declared symmetry checked numerically; residual recomputed; status converged / max_iterations / breakdown / not_finite |
| Singular, known null space | `check_compatible`, `solve_with_gauge` | incompatible loads refused; bordered gauge system, no diagonal shift |
| Adjoint solve | `Factorization.solve_transpose` | transposed SuperLU solve for `general`; same as `solve` for symmetric structures |
| Accuracy repair | `iterative_refinement` (named `refine` until the WP09 merge) | residual on the original A, stop at ω ≤ eps, stagnation (no halving) or the cap |

Known limit (L2c, recorded failure): the SPD factorization path accepts a symmetric indefinite matrix without
detecting it. Its answer is still residual-checked. Use `inertia` where definiteness matters.

Not done: block saddle-point preconditioners and Schur complements (L06), PETSc/AMG adapters, benchmark sweeps
(L07), equilibration other than `jacobi_scaled`.
