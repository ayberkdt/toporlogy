# X_mixed — K/G mixed elasticity, incompressibility, locking diagnosis, reduced-integration modes (preregistration)

Preregistered 2026-11-27 02:54 +0300 (pasted from `date`), before any new numerical test was run. Base SHA `5be9284`.
Modules: `src/toporlogy/fem/mesh/mixed.py` (new class `MixedKG`; the λ-based `MixedElasticity` keeps its behaviour, its
construction is only split into a private helper), tests `tests/toporlogy/test_mixed_kg.py`, doc
`docs/foundations/MIXED_FEM.md`. Plan: WP08 (X01-X03 of `08_MIXED_FEM_LOCKING.md`), acceptance rows X1-X4 of
`23_ACCEPTANCE_CASES.md`. All materials are synthetic (E = 1 or G = 1, unit square, no unit system).

## Formulation (derived here, tension positive, p = −tr σ / 3)

3-D physical strain; plane strain embeds ε_zz = 0, so tr ε = div u and dev ε = ε − (div u / 3) I₃.
`σ = 2G dev ε − p I`, `p = −K div u` (K finite) or `div u = 0` (K = ∞). Weak form, all (v, q):

    ∫ 2G dev ε(u):ε(v) − ∫ p div v = f(v)          (dev ε:ε(v) = dev ε:dev ε(v))
    −∫ q div u − ∫ q p / K        = 0

so the system is `[[A_dev, Bᵀ],[B, −M/K]]` with `B = −∫ψ div φ`, `M = ∫ψψ` (the same B and M as the λ-based class) and

    A_dev = ∫ 2G ε:ε − (2G/3) ∫ div u div v = A_{2μ} − (2G/3) G_div        (2G ε:ε − (2G/3)(tr ε)² = 2G dev ε:dev ε)

The λ-based class has `A = A_{2μ}` and `p_λ = −λ div u`; it is the same continuous problem but a different pressure:
`p_K = −K div u = p_λ − (2G/3) div u` (K = λ + 2G/3). Pressure variable of `MixedKG`: the mean pressure `−tr σ/3`
(exact for all K); of `MixedElasticity`: `−λ div u` (equals the mean pressure only when div u = 0). K = λ + 2G/3 > 0 is
the only requirement: ν = 0 (λ = 0, K = 2G/3) and ν < 0 are valid.

**Equivalence with the displacement formulation.** Eliminating p (finite K): `p = K M⁻¹ B u`, so the displacement
stiffness is `A_{2μ} − (2G/3) G_div + K D_proj`, `D_proj = ∫ div v Π(div u)` (Π the M-orthogonal projection onto the pressure
space). The displacement element is `A_{2μ} + λ G_div`. Their difference is `−K ∫ (I − Π)div u (I − Π)div v`, which is zero
for all u_h iff `div(V_h) ⊂ pressure space`. That holds for **T3 with P0 pressure** (div u constant per cell) and then
`u_mixed = u_displacement` and `p_K = −K div u_h` cellwise are algebraic identities (round-off only), and also
`u_λ = u_displacement` with `p_K = p_λ − (2G/3) div u_h`. For Taylor–Hood (T6–P1 vertex, Q9–Q1 vertex) the difference
is nonzero; both converge to the exact solution, so the difference is bounded by the sum of the two errors.
Mean pressure from a displacement solution (plane strain): `p = −(σxx + σyy + σzz)/3 = −(1+ν)(σxx+σyy)/3` (σzz = ν(σxx+σyy)).

**Saddle-point inertia** (A_dev positive definite on the free DOFs in 2-D plane strain, M positive definite):
finite K: Schur complement `−M/K − B A⁻¹ Bᵀ` negative definite → inertia (n_u, n_p, 0). K = ∞ and B of full row rank:
(n_u, n_p, 0). K = ∞ with every boundary DOF fixed: Bᵀ1 = 0, S = [[A,Bᵀ],[B,0]] has the null vector (0, 1): inertia
(n_u, n_p − 1, 1); the class's gauge row `∫p = 0` borders it with one constraint of full rank, giving
(n_u + 1, n_p, 0) (bordered saddle: pos = pos(S|ker G) + 1, neg = neg(S|ker G) + 1, S|ker G has (n_u, n_p − 1, 0)).
**Compatibility:** with a fully constrained boundary and K = ∞ the data must satisfy ∮ u·n = 0, i.e. `1ᵀ(−B_c u_c) = 0`.

**Reduced integration.** One-point Q4: strain at the centre has 3 components, so rank ≤ 3 per element; one element has
8 − 3 = 5 zero-energy modes (3 rigid + 2 hourglass). Checkerboard `ux = (−1)^{i+j}` (and the same for uy) has zero
centre strain in every cell (ε_xx ∝ (a_{i+1,j}+a_{i+1,j+1}) − (a_{ij}+a_{i,j+1}) = 0, same for γ with b = 0), so it is in the
kernel of every n×n patch: nullity ≥ 5; rank ≤ 3n² gives nullity ≥ 2(n+1)² − 3n². Full 2×2 Gauss Q4 (plane strain,
PD D): kernel = rigid motions, nullity 3. Selective reduced integration (deviatoric part 2×2 Gauss, volumetric part
one-point): kernel ⊂ kernel of the deviatoric term, and dev ε = 0 pointwise with ε_zz = 0 means ε = 0, so the kernel is the
full-integration kernel, nullity 3.

**Slender-beam locking law (Q4).** Pure-bending field `ux = −κ x y`, `uy = κx²/2`: Q4 interpolates uy linearly in x
between nodes, parasitic shear γ = κ(x_mid − x), mean square κ²hx²/12 per element. Energy ratio parasitic/bending
= (G/E')(hx/D)² with E' = E/(1−ν²) (plane stress), i.e. `d_pred = ((1−ν)/2)(hx/D)²` and `R ≈ 1/(1 + d_pred)`.
It is a trial-field estimate (no relaxation, no Poisson-in-depth term): the upper side `d ≲ d_pred` follows from the
variational argument, the lower side is an allowance. For constant element aspect ratio and refinement, hx/D falls, and so does d, while the
1-norm condition number of K grows like h⁻² (λ_max = O(1) in 2-D elasticity, λ_min ∝ h²).

## Checks

Notation: n×n unit-square meshes, E = 1 (`moduli(ν)` conventions), G = E/(2(1+ν)), K = E/(3(1−2ν)); MMS = `fem.mesh.manufactured`
(u = curl ψ + c∇ψ, exact body force; reference = exact fields). Distortion `perturb(·, 0.2/n, seed)`.

| ID | Hypothesis / metric | Reference (independence) | Threshold and reason |
|---|---|---|---|
| X1a | T3–P0 `MixedKG` at ν ∈ {−0.5, 0, 0.3}: displacement equals the displacement element; p = −K div u_h cellwise | `ElasticProblem` with the plane-strain D (a different assembly path); div u_h by the closed triangle-gradient formula written in the test | max rel. diff ≤ 1e-9 (u), ≤ 1e-8 (p): algebraic identity, κ(K) < 1e4 gives errors ~1e-12 |
| X1a-neg | T6 and Q9 Taylor–Hood at ν = 0.3, n = 4: relative displacement difference to the displacement element ≥ 1e-8 | same | the equivalence test is not vacuous (difference derived nonzero) |
| X1b | ν ∈ {0, 0.3}, T6/Q9 vertex, n = 4, 8, 16: mixed u error vs exact, mixed−displacement difference, pressure error vs `−K·div u_exact`; rates between n = 8 and 16 | MMS exact fields | u error rate ≥ 2.5 and difference rate ≥ 2.5 (theory 3 for P2/P1, Q2/Q1 in L2, allowance 0.5), pressure rate ≥ 1.7 (theory 2, allowance 0.3) |
| X1c | T3–P0, ν = 0.3: λ-based u = K-based u (≤ 1e-9); p_K − p_λ = −(2G/3) div u_h (≤ 1e-8); p_K = mean stress pressure from the displacement stress (≤ 1e-8); `MixedElasticity(λ=0)` still raises, `MixedKG(ν=0)` does not | displacement stress; closed forms | derived identities above |
| X1d | the old class is unchanged: T6 3×3 outputs at λ = 7 and λ = ∞ equal a fingerprint recorded before the refactor | arrays stored in `tests/toporlogy/baseline/mixed_lambda_fingerprint.json`, produced by the unmodified code (regression, not independent) | rtol 1e-13 (bitwise on the recording machine is stated in the first result) |
| X2a | ν = 0.4999 and K = ∞, T6 and Q9 vertex, distorted, n = 8, 16, 32: u L2 rate (16→32) ≥ 2.5; p rate ≥ 1.7 (K = ∞: zero-mean error vs −μΔψ; ν = 0.4999: vs −K div u_exact); K = ∞: ‖div u_h‖_L2 rate ≥ 1.7 (= ‖div(u−u_h)‖ ≤ |u−u_h|₁ = O(h²)) and the weak constraint ‖B_f u_f − rp‖∞ ≤ 1e-10 (‖B‖∞‖u‖∞); u error at ν = 0.4999 ≤ 2 × the error at ν = 0.3 on the same n = 16 mesh (uniform in K/G) | MMS exact | as stated |
| X2b | inertia (`solvers.diagnostics.inertia`) of the saddle matrix, T6 4×4 clamped: finite K (ν = 0.3, 0.4999) = (n_u, n_p, 0); K = ∞ without gauge = (n_u, n_p − 1, 1); with the class gauge row = (n_u + 1, n_p, 0); K = ∞ cantilever (clamped left edge only, T6 4×2) = (n_u, n_p, 0) | derived above | exact counts |
| X2c | inf-sup (`MixedKG.inf_sup`, A_dev) T6/Q9 vertex, clamped, n = 4, 8, 16: exactly one zero mode (constant) and β(16) ≥ 0.5 β(4); negative control Q4–Q1 vertex (equal order) and Q4–P0: ≥ 2 zero modes for every n | dense generalised eigenproblem | β bounded below (stable family); equal order: constant + checkerboard (predicted, not derived exactly for Q1–Q1) |
| X2d | gauge, K = ∞ clamped T6 8×8: `gauge` True, 1ᵀMp ≤ 1e-12 ‖M‖‖p‖; displacement equals `solve_with_gauge` on the un-bordered S with N = (0, 1) to 1e-8 (rel.), pressures equal up to a constant (1e-8 rel.); `check_compatible` True for the load and False for a boundary displacement with net inflow; `MixedKG.solve` raises ValueError on the net-inflow data. Characterisation of the old class: it does not raise on that data (it returns a solution that violates B u = rp) | `solve_with_gauge`, `check_compatible` | 1e-8 / exact |
| X2e | MINRES (`krylov_solve`) on the K-based saddle system (T6 vertex, ν = 0.4999) with the block-diagonal SPD preconditioner diag(A_dev,f, (1/G+1/K) M): converged, iterations(n=16) ≤ 1.5 · iterations(n=8) (h-independence, Silvester–Wathen); unpreconditioned MINRES iterations(16)/iterations(8) ≥ 1.5 (κ ∝ h⁻²) | direct solution | as stated |
| X3a | cantilever (plane stress, L/D = 32, closed-form T&G energy U_ex derived in the test), Q4, ny = 2, nx = 32, 16, 8 (hx/D = 1, 2, 4): d = 1/R − 1 against d_pred: d/d_pred ∈ [0.3, 1.5]; d(2)/d(1) and d(4)/d(2) ∈ [2.5, 6] (law: 4); Q9 same meshes R ≥ 0.95; the test's own Q4 cantilever reproduces `cantilever_case("Q4")` to 1e-10 | closed form | as derived |
| X3b | Q4IM (grid), L/D = 32, square elements, ny = 1, 2, 4: R ≥ 0.95, while Q4 at the same meshes has d/d_pred ∈ [0.3, 1.5] (so the deficit is the parasitic shear, which the incompatible modes remove for rectangles) | closed form | as above |
| X3c | locking is not conditioning: Q4, fixed element aspect 4 (L = 32, ny = 2, 4, 8): d falls by ≥ 2.5 per halving while κ₁(K) grows by ≥ 2.5 per halving; at the worst mesh (ny = 2, nx = 8) Jacobi-scaled solve and one refinement step change u by ≤ 1e-8 relative | closed form; scaled/refined solves | κ ∝ h⁻², d ∝ h² |
| X4a | zero-energy modes (eigenvalues ≤ 1e-10 max of the dense stiffness), plane strain ν = 0.3, unconstrained n×n Q4: full integration exactly 3 (n = 1, 2, 3, 4); one-point integration: n = 1 exactly 5, n ≥ 1 at least max(5, 2(n+1)² − 3n²); selective reduced integration exactly 3 | dense numpy element stiffness written in the test; production `ElasticProblem(degree=1)` count equals the dense count for n = 1…4 | exact counts / bounds |
| X4b | checkerboard displacement vectors lie in the one-point kernel for n = 1…4: ‖K z‖ ≤ 1e-12 ‖K‖ | dense | derived |

Negative controls: X1a-neg, X2c equal-order, X2e unpreconditioned, X3 Q4, X4 reduced integration. Run (per brief rule 5):
`python -m pytest tests/toporlogy/test_mixed_kg.py -q -p no:randomly -p no:cacheprovider` with the TMP/TEMP/basetemp of the brief.
First result appended below.
