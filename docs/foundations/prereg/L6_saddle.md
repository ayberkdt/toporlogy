# L6_saddle — saddle-point block operators, Schur complement, block preconditioning (WP05 L06; preregistration)

Preregistered 2026-11-27 06:54 +0300 (pasted from `date` by the command that wrote it), before any test of the new module was run.
Base SHA `d76ed15` (branch `worktree-agent-a3b265a58a848b323`). New module `src/toporlogy/solvers/saddle.py` (imports only
numpy, scipy and `solvers.diagnostics.krylov_solve`; no `fem` import), tests `tests/toporlogy/test_saddle.py`, docs
`docs/foundations/SADDLE_POINT.md`, update of `docs/foundations/SOLVER_CONTRACT.md`. Plan: `05_NUMERICAL_LINALG.md` L06,
acceptance row L6 of `23_ACCEPTANCE_CASES.md`. Extends (does not duplicate) X2e of `test_mixed_kg.py`, which measured one
block-diagonal preconditioned MINRES at ν = 0.4999 for n = 8, 16 (T6, rtol 1e-6). All materials synthetic (E = 1, unit
square). The mixed classes are only read (blocks `A`, `B`, `M`, `disp.active`); they are not modified.

## Conventions and derivations (written before the run; rule 11)

System: `K [u; p] = [f; g]`, `K = [[A, Bᵀ], [B, −C]]`, A SPD (n×n), B (m×n), C = M/K_bulk SPD or C = 0 (K_bulk = ∞).
Rows: `A u + Bᵀ p = f`, `B u − C p = g`. From row 1, `u = A⁻¹(f − Bᵀp)`; into row 2: `B A⁻¹ f − B A⁻¹ Bᵀ p − C p = g`, so

    S p = B A⁻¹ f − g,        S = C + B A⁻¹ Bᵀ        (SPD when B has full row rank or C is SPD).

**Inertia.** A SPD and S SPD give inertia(K) = (n, m, 0) (Schur-complement form of Sylvester's law: K ≅ diag(A, −S)).
With C = 0 and Bᵀz = 0 the vector (0, z) is a null vector of K (row 1: Bᵀz = 0; row 2: B·0 − 0 = 0); K x = b has a
solution only if (0, z)ᵀ b = zᵀ g = 0 (compatibility). A gauge wᵀp = 0 with wᵀz ≠ 0 fixes the solution:
p ← p − z wᵀp / wᵀz. For the mixed classes z = 1 (every boundary DOF constrained ⇒ Bᵀ1 = 0) and w = M 1 (discrete ∫ p = 0).

**Exact blocks.** (a) Block diagonal P = diag(A, S): P^{-1/2} K P^{-1/2} = [[I, Z], [Zᵀ, −C̃]] with Z = A^{-1/2} Bᵀ S^{-1/2},
ZᵀZ = I − C̃, C̃ = S^{-1/2} C S^{-1/2}. For C = 0: ZᵀZ = I and the eigenvalues are 1 and (1 ± √5)/2, three distinct values, so
MINRES (exact arithmetic) terminates in ≤ 3 iterations. (b) Block triangular P_t = [[A, Bᵀ], [0, −S]]:
P_t⁻¹ = [[A⁻¹, A⁻¹BᵀS⁻¹], [0, −S⁻¹]] and K P_t⁻¹ = [[I, 0], [B A⁻¹, I]] (second row: B A⁻¹ Bᵀ S⁻¹ + C S⁻¹ = S S⁻¹ = I), so
(K P_t⁻¹ − I)² = 0 and GMRES needs ≤ 2 iterations for any C (the left-preconditioned P_t⁻¹K is similar to K P_t⁻¹, same
minimal polynomial).

**Spectral equivalence of Ŝ = c M with S (stable pairs).** Take `MixedKG`, plane strain, A = A_dev = 2G ∫ dev ε:dev ε on
V_h ⊂ H¹₀, C = M/K, c = 1/G + 1/K. (i) Upper bound. dev ε = ε − (tr ε/3) I₃ and ε_xx² + ε_yy² ≥ (tr ε)²/2 give
‖dev ε‖² = ‖ε‖² − (tr ε)²/3 ≥ (tr ε)²/6, so vᵀAv ≥ (G/3)‖div v‖². With qᵀB A⁻¹Bᵀ q = sup_v (qᵀBv)² / vᵀAv and
|qᵀBv| = |∫ q div v| ≤ ‖q‖ ‖div v‖: qᵀ S₀ q ≤ (3/G) qᵀ M q, S₀ = B A⁻¹ Bᵀ. Hence qᵀ S q ≤ (3/G + 1/K) qᵀMq ≤ 3 c qᵀMq:
**λ_max(Ŝ⁻¹ S) ≤ 3** for every h and every K (including ∞). (ii) Lower bound. vᵀAv ≤ 2G ‖ε(v)‖² ≤ 2G ‖∇v‖² on H¹₀ (since
‖ε‖² = ½‖∇v‖² + ½‖div v‖² ≤ ‖∇v‖²), and the discrete inf-sup condition of Taylor–Hood pairs, sup_v ∫ q div v / ‖∇v‖ ≥ β_h ‖q‖
for q ⟂ constants with β_h ≥ β > 0 independent of h (Brezzi–Fortin 1991; Elman–Silvester–Wathen 2014 ch. 5–6; **cited from
memory, not re-read**), give qᵀ S₀ q ≥ β²/(2G) qᵀMq, so qᵀSq ≥ (β²/(2G) + 1/K) qᵀMq ≥ min(β²/2, 1) c qᵀMq. The eigenvalues
of Ŝ⁻¹S lie in [min(β²/2, 1), 3] uniformly in h and K (for K = ∞ with a constrained boundary on q ⟂ 1; the constant is
the null vector). β_h enters only here; it is measured, not assumed, in L6b1.

## Checks

Dense tests (independent reference: dense numpy written in the test; n = 14, m = 5, seeded; A = QΛQᵀ with eigenvalues in
[1, 10], B = UΣVᵀ with singular values in [1, 3], C = 0 or C = 0.3 (RRᵀ/m + I); known x*, b = K_dense x*).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| L6a1 | `SaddleOperator.matvec` = K v | dense K @ v | rel ∞-norm ≤ 1e-14 |
| L6a2 | `schur_apply(q)` = (C + B A⁻¹ Bᵀ) q without forming S, for C = 0 and C ≠ 0, 3 random q | dense `np.linalg.inv(A)` | rel ≤ 1e-12 |
| L6a3 | `reduced_rhs(f,g)` = B A⁻¹ f − g; dense solve of S p = reduced_rhs, then `recover_u`, gives x* | dense inv, x* | reduced rhs rel ≤ 1e-12; (u, p) rel ≤ 1e-10 |
| L6a4 | sign convention: S SPD (min eig > 0) and `inertia(K)` = (n, m, 0) for both C | `diagnostics.inertia` + dense eigvalsh of S | exact counts |
| L6a5 | exact blocks: MINRES with diag(A⁻¹, S⁻¹) and C = 0 needs ≤ 4 iterations (derived 3, +1 for the recursive stopping estimate); GMRES with the triangular preconditioner ≤ 3 (derived 2, +1) for C = 0 and C ≠ 0; explicit residual ≤ 1e-9 (rtol 1e-10) and x = x* to 1e-8 | derivation above | counts and bounds as stated |
| L6a6 | invalid input rejected: block shape mismatch, wrong f/g length, unknown preconditioner, preconditioned solve without S_hat | — | ValueError |
| L6a7 | gauge, dense: B with 1ᵀB = 0, C = 0, g = B w (compatible), `null_vector = 1`, `gauge_weights = w₀ > 0`: A u + Bᵀp = f, B u = g, w₀ᵀp = 0 and equal to the dense bordered solve [[K, (0,w₀)], [(0,w₀)ᵀ, 0]]; incompatible g (zᵀg ≠ 0), a wrong null vector and wᵀz = 0 raise ValueError | dense bordered system | rel ≤ 1e-9 |
| L6b1 | spectrum of Ŝ⁻¹S, T6 MixedKG, ν ∈ {0.3, 0.4999, 0.5 (K = ∞)}, n ∈ {4, 8, 16}: λ_max ≤ 3 + 1e-10 | dense `scipy.linalg.eigh(S, cM)`, S built with dense solves in the test; at K = ∞ the single zero eigenvalue (constant) is removed and must be ≤ 1e-9 | λ_max ≤ 3 + 1e-10; κ = λ_max/λ_min ≤ 200; κ(n=16)/κ(n=4) ≤ 1.25 |

FE sweep (T6 and Q9 Taylor–Hood `MixedKG`, unit square clamped on the whole boundary, `perturb(m, 0.2/n, seed=200+n)`,
manufactured incompressible-limit load f as in X2, g = B w with w random and ‖g‖ = ‖f‖ so the pressure row is not trivial
(compatible: 1ᵀB w = 0 on a fully constrained boundary); ν ∈ {0.3, 0.49, 0.4999, 0.5}, n ∈ {8, 16, 32}; G = 1/(2(1+ν)),
K = 1/(3(1−2ν)) (E = 1), K = ∞ at ν = 0.5; Â = exact sparse LU of A, Ŝ = (1/G + 1/K) M; rtol = 1e-8, up to 3 restarts
from the explicit residual; ν = 0.5 uses `null_vector = 1`, `gauge_weights = M 1`). "Iterations" = Krylov iterations summed
over restarts.

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| L6c1 | block-diagonal MINRES converges everywhere: status converged | explicit ‖b − Kx‖/‖b‖ recomputed in the test from the assembled K | ≤ 1e-7 (= 10 rtol), also the u-row and p-row residuals separately |
| L6c2 | iteration count independent of h and ν: per element, max(its)/min(its) over the 12 (ν, n) cases ≤ 2.0 and max(its) ≤ 40 | counts | as stated |
| L6c3 | negative control, no preconditioner: plain MINRES at ν = 0.4999, n = 32 with maxiter = 500 and no restart does **not** reach explicit residual ≤ 1e-7 | explicit residual | status ≠ converged |
| L6c4 | negative control, growth: plain MINRES iterations (cap 500, ν = 0.3) at n = 32 divided by n = 8 ≥ 2 | counts | ratio ≥ 2 |
| L6c5 | solution equals the direct solve `Factorization(K, "symmetric_indefinite")` (for ν = 0.5 on the test-built bordered matrix [[A,Bᵀ,0],[B,0,(M1)],[0,(M1)ᵀ,0]], i.e. the ∫p = 0 gauge) | direct solve; bound from the condition estimate κ̂₁ = `condition_1norm_estimate` of the same matrix (an estimate, usually a lower bound, hence the factor 10) | ‖x − x_d‖₁/‖x_d‖₁ ≤ 10 κ̂₁ ‖b − Kx‖₁/‖b‖₁ |
| L6c6 | gauge: at ν = 0.5, `result.gauge` is True and \|wᵀp\| ≤ 1e-10 ‖w‖‖p‖ | — | as stated |
| L6d | block-triangular GMRES, T6, ν ∈ {0.3, 0.4999, 0.5}, n ∈ {8, 16, 32}: converged (explicit ≤ 1e-7), max(its)/min(its) ≤ 2.0, max(its) ≤ 40 | explicit residual, counts | as stated |
| L6e | incomplete-LU Â (`spilu`, drop_tol 1e-4, fill 10; AMG-free), T6 ν = 0.3, n ∈ {8, 16}: converges to explicit ≤ 1e-7 within 200 iterations. No mesh independence is claimed or tested; the counts are reported | explicit residual | as stated |
| L6f | cost counters: n_A_solves = n_S_solves ≥ iterations for every preconditioned solve above | — | as stated |

Run: `python -m pytest tests/toporlogy/test_saddle.py -q -p no:randomly` (TMP/TEMP and basetemp under
`.claude/tmp-toporlogy/L6_saddle`). First results are appended below.
