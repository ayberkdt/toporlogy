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

## First result, run 1 (2026-11-27 06:59 +0300, SHA e935b1a + test file)

Dense (L6a1-a7, constructor check): **PASS**. matvec 9.5e-17; schur_apply 3.9e-16 (C = 0), 3.0e-16 (C SPD); reduced rhs
9.7e-16; (u, p) 1.3e-15; min eig S 0.221 / 0.747; inertia (14, 5, 0) both. Exact blocks: MINRES(diag, C = 0) 3 iterations
(derived 3), GMRES(triangular) 1 iteration (derived ≤ 2) for both C; gauge dense 9 iterations, 2.7e-15 against the bordered
system, |w₀ᵀp| 4.4e-16, the three ValueErrors raised.

FE sweep, run 1 (7 passed, 6 failed):

* L6b1 (combined test): ν = 0.3 and 0.5 pass (κ = 2.6 and 6.8-6.95; λ_max ≤ 0.83 and 0.75, below 3). **ν = 0.4999 FAIL**:
  κ = 3.72e3 / 3.75e3 / 3.75e3 (n = 4, 8, 16) against the preregistered ≤ 200 (the assertion order hid the ratio check at
  0.4999; the checks are separated below).
* L6c1 FAIL on **status** (T6 ν = 0.3 n = 8: "max_iterations" although the explicit residual was 6.2e-10 ≤ 1e-7).
  Cause: a **fault in the production code**: the restart loop tightened rtol by 10 per restart and `krylov_solve` judged
  "converged" against the tightened tolerance (res ≤ 10 · rtol_k), so an iterate that met the original target was reported
  as not converged after the last restart. Fixed in `saddle.py`: converged iff the explicit residual is ≤ 10 · the
  original rtol. No test threshold changed.
* L6c2 (combined test) FAIL on the absolute cap: T6 max iterations 50 > 40 (ratio 50/25 = 2.00, within 2.0). Q9: ratio 1.62,
  max 39, both within bounds. Counts of run 1 (computed with the restart defect above): T6 28/27/25 (ν = 0.3), 47/45/44
  (0.49), 49/48/47 (0.4999), 50/48/47 (0.5) for n = 8/16/32.
* L6c3 PASS (plain MINRES n = 32, ν = 0.4999, 500 iterations: explicit residual 2.4e-2). **L6c4 FAIL**: 500/308 = 1.62 < 2.
* L6c5/c6 PASS on all 24 cases (max ‖x − x_d‖₁/‖x_d‖₁ = 2.1e-9; the κ-scaled bound was ≥ 5.8e-6 in every case; gauge
  |wᵀp| ≤ 1e-15 relative).
* L6d PASS (GMRES, block triangular: converged in 1 iteration in every case, residual ≤ 5.6e-9).
* L6e FAIL (n = 8: 18 iterations, status max_iterations, explicit residual 1.7e-6 > 1e-7; MINRES stopped on its estimate).
  Cause (hypothesis fault, found afterwards): an ILU factor of an SPD matrix is not symmetric, so it is not a valid MINRES
  preconditioner (the module's contract requires an SPD one).

Actions after run 1 (thresholds untouched): the status defect fixed (above); the combined L6b1 and L6c2 tests split into
L6b1a (λ_max ≤ 3), L6b1b (κ ≤ 200), L6b1c (κ growth ≤ 1.25) and L6c2a (ratio ≤ 2), L6c2b (cap ≤ 40) so that each
preregistered number is judged on its own; L6e parametrised by n; post hoc tests added and labelled (constant-mode
eigenvalue: Bᵀ1 = 0 gives S 1 = M 1/K, so 1 is an exact eigenvector of Ŝ⁻¹S with eigenvalue (1/K)/(1/G + 1/K) = G/(G+K),
2.0e-4 at ν = 0.4999; ILU with GMRES; plain MINRES residual after a fixed 300 iterations). Run 2 follows.

## Run 2 (2026-11-27 07:04 +0300, restart defect fixed, tests split; the set of numbers below is the final one)

`python -m pytest tests/toporlogy/test_saddle.py -q -p no:randomly`: **30 passed, 6 xfailed** (strict), 41 s.
Counts (Krylov iterations summed over restarts, rtol 1e-8, n = 8 / 16 / 32):

| | ν = 0.3 | ν = 0.49 | ν = 0.4999 | ν = 0.5 (gauge) |
|---|---|---|---|---|
| T6 | 21 / 23 / 22 | 41 / 39 / 38 | 42 / 42 / 47 | 42 / 42 / 47 |
| Q9 | 20 / 19 / 22 | 29 / 32 / 36 | 34 / 33 / 33 | 34 / 33 / 33 |

| ID | Result | Measured |
|---|---|---|
| L6a1-a4 | PASS | matvec 9.5e-17, schur_apply ≤ 3.9e-16, reduced rhs ≤ 9.7e-16, (u,p) ≤ 1.3e-15, inertia (14, 5, 0) |
| L6a5 | PASS | MINRES(diag, C = 0) 3 iterations, GMRES(triangular) 1 iteration (both C); residual ≤ 5.8e-16 |
| L6a6, L6a7 | PASS | ValueErrors raised; gauge 2.7e-15 against the dense bordered system, \|w₀ᵀp\| 4.4e-16 |
| L6b1a | PASS | λ_max(Ŝ⁻¹S) = 0.829 (ν = 0.3), 0.750 (0.4999), 0.750 (K = ∞), bound 3 |
| L6b1b | PASS ν = 0.3 (κ = 2.6), ν = 0.5 (6.8-6.95); **FAIL (xfail) ν = 0.4999**: κ = 3.72e3 / 3.75e3 / 3.75e3 | the constant pressure is an exact eigenvector, eigenvalue G/(G+K) = 2.0e-4 (post hoc test: equal to 1e-8 for ν = 0.3, 0.49, 0.4999); the rest of the spectrum has κ' ≤ 7 |
| L6b1c | PASS | κ(16)/κ(4) = 1.004 (0.3), 1.007 (0.4999), 0.994 (K = ∞) |
| L6c1 | PASS 24/24 | explicit residual ≤ 9.9e-8 (target 1e-7), u-row ≤ 9.9e-8, p-row ≤ 1.6e-8; 1-3 restarts needed because MINRES stops on its preconditioned recursive estimate |
| L6c2a | Q9 PASS (36/19 = 1.89); **T6 FAIL (xfail)** 47/21 = 2.24 | |
| L6c2b | Q9 PASS (max 36); **T6 FAIL (xfail)** max 47 | |
| L6c3 | PASS | plain MINRES, T6 ν = 0.4999 n = 32: 500 iterations, explicit residual 2.43e-2 |
| L6c4 | **FAIL (xfail)** | 500/308 = 1.62 (capped count) |
| L6c5, L6c6 | PASS 24/24 | ‖x − x_d‖₁/‖x_d‖₁ ≤ 3.3e-8 (bound 10 κ̂₁ ‖r‖₁/‖b‖₁ between 1.1e-3 and 3.7: at ν = 0.4999, n = 32, κ̂₁ ~ 1e7, it is 0.996 (T6) and 3.72 (Q9): there the check carries almost no information); gauge \|wᵀp\| ≤ 2.8e-15 rel. |
| L6d | PASS 9/9 | GMRES, block triangular: 1 iteration in every case (A-solves 13-21 per solve), residual ≤ 5.6e-9 |
| L6e | **FAIL (xfail)** n = 8 and 16 | 18 iterations / 1.7e-6 and 17 iterations / 1.3e-6 (non-symmetric ILU in MINRES) |
| L6f | PASS | n_A_solves = n_S_solves ≥ iterations in all FE solves (e.g. 51 solves for 47 iterations) |
| post hoc (labelled) | PASS | constant-mode eigenvalue; ILU + GMRES converges (2 and 1 iterations, 5.8e-10, 7.1e-9); plain MINRES residual after 300 iterations 6.2e-5 (n = 8) vs 4.1e-2 (n = 32); per-ν count ratio over n ≤ 1.24 (threshold 1.3 set after seeing the counts) |

What the failures say, and their cost: the derivation in this record claimed Ŝ ≃ S on the whole pressure space; the correct
statement is on q ⟂ constants (the constant's eigenvalue is G/(G+K), small for K ≫ G), and it was a wording error of the
hypothesis, not of the code. The cost is bounded: one small isolated eigenvalue, counts flat in h (≤ 1.12 T6, ≤ 1.24 Q9 at
fixed ν) and equal for ν = 0.4999 and 0.5; the ν = 0.3 → ν ≥ 0.49 step doubles the count. Ledger entries:
`docs/foundations/ledger_additions/L6_saddle.json` (6 strict xfails).
