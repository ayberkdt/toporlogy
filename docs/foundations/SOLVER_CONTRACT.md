# Solver contract (WP05 slice)

Code: `src/toporlogy/solvers/diagnostics.py`, `src/toporlogy/solvers/saddle.py`, `src/toporlogy/solvers/eigen.py`,
`src/toporlogy/fem/mesh/linear.py` (`Factorization.solve_transpose`). Preregistration and results: `prereg/L_solver.md`,
`prereg/L6_saddle.md`, `prereg/L6e_eigen.md`.

| Need | Function | Requirement enforced |
|---|---|---|
| Is the residual meaningful? | `solve_report` | η_N, Oettli–Prager ω on the original A; κ labelled given / 1-norm estimate; `forward_established` only if κ η_N ≤ 1e-8 |
| Definiteness | `inertia` (small dense) | Bunch–Kaufman LDLᵀ, 2×2 pivot blocks counted by their eigenvalues |
| Krylov solve | `krylov_solve` | CG: spd; MINRES: symmetric; GMRES/BiCGSTAB: any; declared symmetry checked numerically; residual recomputed; status converged / max_iterations / breakdown / not_finite |
| Singular, known null space | `check_compatible`, `solve_with_gauge` | incompatible loads refused; bordered gauge system, no diagonal shift |
| Adjoint solve | `Factorization.solve_transpose` | transposed SuperLU solve for `general`; same as `solve` for symmetric structures |
| Accuracy repair | `iterative_refinement` (named `refine` until the WP09 merge) | residual on the original A, stop at ω ≤ eps, stagnation (no halving) or the cap |
| Saddle point `[[A, Bᵀ], [B, −C]]` | `SaddleOperator`, `saddle_solve` (`solvers/saddle.py`) | blocks sparse; S = C + B A⁻¹ Bᵀ applied with a factorised A, never formed; `"diagonal"` = diag(Â⁻¹, Ŝ⁻¹) with MINRES (Â, Ŝ must be SPD), `"triangular"` = [[Â, Bᵀ], [0, −Ŝ]]⁻¹ with GMRES; explicit residual of the original (f, g) with separate u and p rows, restart until it meets 10 rtol, block-solve counters; pressure null space only by `null_vector` (compatibility `zᵀg = 0` checked, projected gauge, no shift). Ŝ = (1/G + 1/K) M for stable mixed pairs: `SADDLE_POINT.md` |

| Generalized symmetric eigenproblem K φ = λ M φ | `eigen_report`, `mac`, `subspace_angles`, `track_modes`, `shift_invert_modes` (`solvers/eigen.py`) | K, M symmetric, M SPD (Cholesky); per pair backward error and Rayleigh consistency, M-orthonormality; clusters judged by block residual and principal angles, never by single vectors; modes tracked by MAC assignment with a crossing report (the sorted index is not an identity); shift explicit, a shift at or numerically at an eigenvalue raises `ShiftError`; semidefinite K only with `assume_psd=True` (shift −1e-6 s), zero modes counted and reported; one extra pair, `cluster_cut` flags an incomplete cluster |
| Buckling pencil (K + λ K_G) φ = 0 | `buckling_load_factors` | K SPD (Cholesky up to 2500 unknowns, else Lanczos evidence), K_G symmetric, indefinite allowed; solved as (−K_G) φ = μ K φ, load factor 1/μ for μ > 0, μ < 0 reported as reversed load, none positive gives an empty result; K-orthonormal modes; load convention in `EIGEN_RELIABILITY.md` |

Known limit (L2c, recorded failure): the SPD factorization path accepts a symmetric indefinite matrix without
detecting it. Its answer is still residual-checked. Use `inertia` where definiteness matters.

Saddle-point limits (`SADDLE_POINT.md`): the Ŝ = cM equivalence is derived for `MixedKG` in 2-D plane strain and holds on pressures
orthogonal to constants (the constant has the eigenvalue G/(G+K)); ILU is valid as Â only with GMRES; no AMG or incomplete Cholesky;
iteration counts, not timings, are recorded.

Eigen limits (`EIGEN_RELIABILITY.md`): a residual is a backward error, not a forward bound (the gap is needed); the MAC of single vectors
is meaningless inside a cluster; a mode entering the tracked window from outside is invisible; the SPD check of K for the buckling pencil
above 2500 unknowns is Lanczos evidence, not a proof.

Not done: non-symmetric and quadratic eigenproblems, PETSc/AMG adapters, benchmark sweeps (L07: time, RSS,
fill), equilibration other than `jacobi_scaled`.
