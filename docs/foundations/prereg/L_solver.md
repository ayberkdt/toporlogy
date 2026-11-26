# L — Linear solve diagnostics (WP05 L01–L05; preregistration)

Preregistered 2026-11-27 01:27 +0300 (clock read with `date`), before the module or the tests exist. Base SHA
`df2ca78`. New module `src/toporlogy/solvers/diagnostics.py`; `fem/mesh/linear.Factorization` gains
`solve_transpose`. The existing paths (DirectSolver, Factorization.solve, jacobi_pcg, condition_1norm,
checked_residual) are not changed.

## Planned API

* `solve_report(A, x, b, condition=None, reference=None) -> SolveReport`: residual r = b − A x on the original A;
  normwise backward error η_N = ‖r‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞); componentwise (Oettli–Prager)
  ω = max_i |r_i| / (|A||x| + |b|)_i (rows with zero denominator and zero residual skipped; zero denominator with
  nonzero residual → ω = inf); κ (given, or the 1-norm estimate of `condition_1norm`), labelled "estimate";
  `forward_established = κ · η_N ≤ 1e-8`; forward error ‖x − x_ref‖∞/‖x_ref‖∞ only when a reference is passed.
* `inertia(A)`: (n₊, n₋, n₀) of a small dense symmetric matrix from `scipy.linalg.ldl` (Bunch–Kaufman, 1×1 and 2×2
  pivot blocks; the eigenvalues of each block counted), zero tolerance n · eps · ‖A‖.
* `krylov_solve(A, b, method, structure, M=None, rtol, maxiter, x0=None) -> KrylovResult`: method cg | minres |
  gmres | bicgstab; cg needs structure "spd", minres "spd" or "symmetric_indefinite", gmres/bicgstab any. A
  structure that claims symmetry is checked numerically (‖A − Aᵀ‖_F ≤ 1e-12 ‖A‖_F) and refused otherwise. Status:
  converged | max_iterations | breakdown | not_finite; the final residual is recomputed explicitly from b − A x.
* `check_compatible(A, b, N, rtol)`: ‖Nᵀ b‖ ≤ rtol ‖N‖ ‖b‖ for an orthonormalised null basis N;
  `solve_with_gauge(A, b, N)`: bordered system [[A, N], [Nᵀ, 0]] [x; μ] = [b; 0] (gauge Nᵀ x = 0), refused if b is
  incompatible. No diagonal shift.
* `refine(A, b, solve, max_iter=10)`: iterative refinement with the residual in float64 on the original A; stops at
  ω ≤ eps, at stagnation (ω not reduced by half) or at max_iter; returns x, the ω history and the stop reason.

## Checks

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| L1a | A = Q diag(logspace(0, −k, n)) Qᵀ (Q from QR of a seeded normal matrix, seed 11), n ∈ {8, 32}, k ∈ {0, 4, 8, 12}, x random (seed 12), b = A x in float64: `numpy.linalg.solve` gives η_N ≤ 1e-14 in every case | — | η_N ≤ 1e-14 |
| L1b | forward accuracy: for k ∈ {0, 4} `forward_established` is True and the forward error against an mpmath 50-digit solve of the same float64 A, b is ≤ 1e-8 and ≤ 10 κ η_N + 1e-15; for k = 12 `forward_established` is False | mpmath `lu_solve` at 50 digits (independent arithmetic) | as stated |
| L1n | negative control: x_bad = x + 1e-2 v_min (v_min the eigenvector of the smallest eigenvalue, k = 12, n = 8) has η_N ≤ 1e-12 but `forward_established` False, and its true forward error vs mpmath is ≥ 1e-3 | mpmath | as stated |
| L2a | block-diagonal symmetric indefinite A = diag([[0,1],[1,0]], [[2,1],[1,3]], −[[2,1],[1,3]]): `inertia` = (3, 3, 0), equal to the sign count of `numpy.linalg.eigvalsh` | eigvalsh | exact |
| L2b | the same A: `krylov_solve(..., "cg", structure="symmetric_indefinite")` raises ValueError; MINRES with structure "symmetric_indefinite" converges, explicit relative residual ≤ 1e-10; `Factorization(A, "symmetric_indefinite").solve` relative residual ≤ 1e-14 | numpy solve | as stated |
| L2c | `Factorization(A, "spd")` on this indefinite A: hypothesis — it raises or its residual check fails (the SPD path never returns silently) | — | an exception is raised |
| L2d | nonsymmetric A = [[3, 1], [−2, 4]]: `krylov_solve` cg / minres with structure "spd" or "symmetric_indefinite" raise ValueError (numerical symmetry check); GMRES converges (rel. residual ≤ 1e-12); `Factorization(A, "general").solve_transpose(b)` equals `numpy.linalg.solve(A.T, b)` | numpy | rel 1e-14 |
| L3 | A = [[1, −1], [−1, 1]], N = [1, 1]/√2: b = [1, −1] compatible, `solve_with_gauge` gives x = [0.5, −0.5]; b = [1, 0] incompatible → `check_compatible` False and `solve_with_gauge` raises | hand solution | abs 1e-14 |
| L4a | multi-RHS: random sparse SPD (n = 50, seed 13) with 4 RHS: `Factorization(A, "spd").solve(B)` column-wise equal to dense solves | numpy dense solve | rel 1e-12 |
| L4b | duality aᵀ A⁻¹ b = bᵀ A⁻ᵀ a for a random nonsymmetric sparse A (n = 40, diagonally dominant, seed 14) with `solve` and `solve_transpose` of one `Factorization(A, "general")` | the identity | rel 1e-12 |
| L4c | Krylov status: `krylov_solve(A_spd, b, "cg", "spd", maxiter=2)` returns status max_iterations, converged False, and its reported residual equals ‖b − A x‖₂/‖b‖₂ recomputed in the test | recomputation | rel 1e-12 |
| L5 | A = D S D, S SPD well conditioned (n = 30, seed 15), D = diag(logspace(−6, 6, n)): x₀ from a float32 solve (`numpy.linalg.solve` in float32) then `refine` with a float64 dense LU: the final ω ≤ 1e-14; the history is non-increasing until the stop, the stop reason is one of converged / stagnated / max_iter, and the iteration count ≤ max_iter | — | as stated |

Run: `python -m pytest tests/toporlogy/test_solver_diagnostics.py -q -p no:randomly`. First result appended below.
