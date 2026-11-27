# A_adaptivity — a posteriori error estimation and adaptive T3 refinement (preregistration)

Preregistered 2026-11-27 03:05 +0300, base SHA `67dbf1b`, before any test of the new module was run. Plan source:
WP09 (`09_ADAPTIVITY_ERROR.md`) and acceptance rows A1–A3 of `23_ACCEPTANCE_CASES.md`. Module (new):
`src/toporlogy/fem/mesh/adaptivity.py`; tests: `tests/toporlogy/test_adaptivity.py`; design:
`docs/foundations/ADAPTIVITY.md`. Scope: 2-D Poisson `−∇·(κ∇u) = s` on T3 with Dirichlet data on the whole boundary
(`HeatProblem`). Not done: elasticity estimators, T6/Q4, 3-D, curved boundaries, goal-oriented estimators (stated as
open, not implied).

Disclosure on order: the closed-form effectivity model below was evaluated with a throw-away numpy script that does
not import the production code (it is reproduced in the test file as `_model`), before this record was committed. No
test of `adaptivity.py` has been run. The values it produced are quoted here.

## Definitions (fixed before the run)

Mesh T3, positively oriented. For a cell T: area A_T, diameter h_T = longest edge, constant gradient ∇u_h|_T. For an
edge E: length h_E = |E|.

* Residual indicator (squared, unit = energy): `η_T² = (h_T²/κ_T) ‖s‖²_{L²(T)} + ½ Σ_{E⊂∂T, interior} (h_E/κ̄_E)
  ‖[κ∇u_h·n_E]‖²_{L²(E)} + Σ_{E⊂∂T∩Γ_D} h_E ‖∂_s(g − I_h g)‖²_{L²(E)}`, κ̄_E = max of the two adjacent κ. `div(κ∇u_h)`
  vanishes in a T3 cell (κ piecewise constant), so the element residual is s. The last term is the Dirichlet-data
  term (`g` = exact trace, `I_h g` its nodal interpolant, ∂_s the tangential derivative); it is zero for g = 0 and for
  data that is linear along each boundary edge. Coefficients are all 1: this is an **indicator**, not a bound.
  Reliability/efficiency constants are not proved here and no guaranteed upper bound is claimed; κ-robustness is not
  claimed.
* Zienkiewicz–Zhu (ZZ) indicator: nodal gradient G_i = Σ_{T∋i} A_T ∇u_T / Σ A_T (area-weighted patch average, boundary
  nodes use their own patch), G_h its P1 interpolant, `η_T,ZZ² = κ_T ‖G_h − ∇u_h‖²_{L²(T)}`. No superconvergence
  guarantee is claimed.
* Energy error `|e| = (Σ_T κ_T ‖∇u − ∇u_h‖²_{L²(T)})^{1/2}` (absolute). Effectivity `θ = η/|e|`, η = (Σ η_T²)^{1/2}.
* Dörfler marking: the smallest set M (descending η_T², ties by cell index, stable) with Σ_M η_T² ≥ θ_D Σ η_T²,
  θ_D = 0.5 in every adaptive run.
* DOFs N = number of free nodes (nodes − boundary nodes).
* Refinement: newest-vertex bisection; cells[:,0] is the newest vertex (peak), refinement edge = (c1,c2); closure by
  recursive compatible bisection (neighbour is bisected first if its refinement edge differs); initial labelling: the
  longest edge becomes the refinement edge (ties: lexicographically smallest node pair).

## Reference methods and independence

* A1/A2 errors: dense numpy written in the test: collapsed Gauss–Legendre on triangles (10 × 10) for the exact
  gradient, ∇u_h recomputed in the test from the vertex coordinates (not from the production kernels). L-shape cells
  that touch the origin use a Duffy map with the apex at the origin and s = σ³ (the r^{-2/3} integrand becomes
  polynomial in σ). The quadrature is itself checked (A2f) against `scipy.integrate.quad` of the analytic total energy
  `E² = ∫|∇u|² = 2 ∫_0^{π/4} cos(θ)^{-4/3} dθ` (derived: |∇u|² = (4/9) r^{-2/3}, R(θ) = 1/max(|cos θ|,|sin θ|),
  `(4/9)(3/4) · 3 · ∫_0^{π/2} R^{4/3}`, simplified by symmetry).
* A3: geometry checks written in the test (areas by cross product, edge multiplicity by `np.unique`, hanging nodes by
  a KD-tree search of nodes inside edge half-length balls, barycentric containment of child centroids in parents).

## Tests

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| A1a | Residual-indicator effectivity on the unit square, `u = sin πx sin πy`, `s = 2π² u`, T3, n = 4…64 uniform: θ_res tends to the closed-form model value θ_res∞ = 4√2 = 5.657 (below) | model + exact error | n = 64: θ within ±15 % of 5.657; n = 32: ±25 % |
| A1b | ZZ effectivity tends to 1 (model: recovered gradient exact for quadratic u in the interior of a symmetric patch) | exact error | n = 64: θ_ZZ ∈ [0.85, 1.15]; n = 32: [0.80, 1.20] |
| A1c | Exact energy error of the interpolant-like Galerkin solution: `|e| ≈ 3.489 h` (model constant) with slope 1 | exact error | n = 64: `|e| · 64 ∈ 3.489 · [0.90, 1.10]`; log-log slope between n = 32 and 64 ∈ [0.95, 1.05] |
| A1d | Scaling: κ ≡ 4 with source 4s gives the same u_h, `η²` and `|e|²` both scale by κ, so θ is unchanged (regression, labelled: same kernel) | θ at κ = 1 | relative difference ≤ 1e-9 (n = 16) |
| A1e | Patch test: u = 1 + 2x − 3y (s = 0, exact linear data): u_h exact, every indicator zero | exact | η ≤ 1e-12 (both estimators), `|e| ≤ 1e-12` |
| A2f | Test quadrature: Σ_T ∫_T |∇u|² on the L mesh (round 6) equals the analytic E² | `quad` | relative 1e-8 |
| A2a | Uniform bisection on the L-shape, exact solution `u = r^{2/3} sin(2θ/3)` (θ ∈ [0, 3π/2], harmonic, data = trace interpolated at boundary nodes): energy-error slope in N over rounds with N ∈ [1000, 60000] is ≈ −1/3 | log-log fit | slope ∈ [−0.38, −0.29] |
| A2b | Adaptive (residual + data indicator, θ_D = 0.5, start = uniform round 2, cap N ≤ 40000): slope in N over iterates with N ≥ 1000 approaches −1/2 | log-log fit | slope ∈ [−0.58, −0.42] |
| A2c | Error at comparable DOFs: N_u = DOFs of the uniform round-12 mesh (≈ h = 1/64); N_a = DOFs of the last adaptive iterate with N ≤ N_u; `e_unif(N_u)/e_adapt(N_a)` | exact errors | ≥ 2.5 (preregistered factor) |
| A2d | The estimator detects the under-resolved region: on uniform round 8 (h = 1/16) the Dörfler set M contains every cell touching the origin, and Σ_M e_T² ≥ 0.5 Σ_T e_T² (true local errors) | exact local errors | both |
| A2e | Negative control: indicator without the jump term (only the data term, zero near the corner) cannot find the corner. Same loop, caps and θ_D | exact errors | `e_ctrl(last N ≤ N_u) ≥ 2 · e_adapt(last N ≤ N_u)`, control slope (N ≥ 1000) ≥ −0.40, and the smallest cell diameter touching the origin in the control is ≥ 8× that of the adaptive run |
| A2g | Reported, no threshold: θ_res along the adaptive sequence and ZZ-driven adaptive slope (same loop, `estimator="zz"`) | exact errors | none (recorded) |
| A3a | Conformity: starting from the right-isosceles unit-square and L meshes, 6 rounds of deterministic pseudo-random marking (30 % of cells, fixed seed): every interior edge belongs to exactly 2 cells, boundary edges to 1, no hanging node, all cells positive area, min angle ≥ 45° − 1e-9° (NVB of right isosceles triangles stays right isosceles), area conserved | geometry in test | exact; area relative 1e-14 |
| A3b | Tags and sets: cell tags (tag = 1 for x < 0.5 else 2, interface on mesh lines) and the facet/node sets `left right bottom top` stay exactly the geometric sets after refinement | geometry in test | exact |
| A3c | Transfer: nodal affine field `3 + 2x − 5y` is reproduced at all new nodes (after 2 rounds); cell field inherits the parent value and every child centroid lies in its parent | formula, barycentric test | 1e-14; exact |
| A3d | Determinism and restart: same marking → bitwise equal meshes; adaptive run of 6 iterations equals 3 iterations + restart from the mesh checkpoint of iteration 3 + 3 iterations (meshes and errors bitwise) | the uninterrupted run (regression) | `np.array_equal` |
| A3e | Caps and stop reasons: `max_dofs`, `max_iterations`, `target` each stop the loop with the matching reason; the last solved level never exceeds the DOF cap | the settings | exact |
| A3f | Dörfler: marked set mass ≥ θ_D total, and dropping its smallest member goes below; ties broken by index | brute force on random vectors | exact |

## Model derivation (A1a–A1c)

Locally `u ≈ ½ xᵀHx`, H = [[a,b],[b,c]], mesh of squares of side h split along (0,0)–(1,1). The P1 interpolant gradient
on the lower triangle (0,0),(h,0),(h,h) is `h(a/2, b+c/2)`, on the upper (0,0),(h,h),(0,h) `h(a/2+b, c/2)`; a neighbour
cell with origin x0 adds `H x0`. Normal-flux jumps: vertical edge `h(a+b)`, horizontal `h(b+c)`, diagonal `√2 b h`
(edge length √2 h). Per cell, with h_T = √2 h and s = −(a+c):
`η² = h⁴[(a+b)² + (b+c)² + 4b² + 2(a+c)²]`; the energy error of the interpolant per cell `e² = h⁴ F(a,b,c)` with F
integrated exactly (Gauss) on the two triangles. For `u = sin πx sin πy`: a = c = −π² sin πx sin πy, b = π² cos πx cos πy.
Integrating both densities over the square (40² Gauss points): `Σ e² → 12.176 h²`, `Σ η² → 389.64 h²`,
so `|e| → 3.489 h`, `θ_res∞ = √(389.64/12.176) = 5.657 (= 4√2)`; the local ratio ranges between 4.20 and 10.69. The
P1 Galerkin gradient differs from the interpolant gradient by O(h²) (supercloseness on a uniform mesh), boundary strips
contribute O(h) relatively: hence the ±15 % / ±25 % tolerances. For ZZ the nodal average of the six constant gradients
around an interior node of a quadratic u equals ∇u(x_node) exactly (checked numerically with a random H: |G| < 1e-15),
G_h is then ∇u itself, so η_ZZ = |e| in the interior and θ_ZZ∞ = 1.

## Inputs, caps, run

Unit square: `rectangle(n, n, element="T3")`, n ∈ {4, 8, 16, 32, 64} (≤ 4225 nodes); source and exact solution as
above; Dirichlet 0. L-shape: `(-1,1)² \ [0,1]×[-1,0]` from `rectangle(2,2,2,2,origin=(-1,-1))` minus the removed
cells, labelled by the longest edge; uniform rounds ≤ 14 (≤ 5e4 DOFs), adaptive N ≤ 40000, ≤ 60 iterations, double
precision, no seeds except A3a (`default_rng(2026)`).
Command: `python -m pytest tests/toporlogy/test_adaptivity.py -q -p no:randomly -p no:cacheprovider`.

Expected limiting behaviour / failure modes: the Dirichlet data interpolation error on the coarse outer boundary may
stall the adaptive rate at large N if the data term does not trigger boundary refinement (then A2b/A2c fail and are
recorded); an unlabelled input mesh may not terminate the closure (guarded, raises).

## First result (2026-11-27, ran between 03:05 and 03:16 +0300; the clock was read at 03:16 after the final full run; base SHA 67dbf1b + the new files)

Command: `python -m pytest tests/toporlogy/test_adaptivity.py -q -s -p no:randomly -p no:cacheprovider`.
Final full run: 26 passed, 1 xfailed (63 s). Set-up corrections made before the first run (no threshold touched):
three awkward expressions in the test file were tidied (walrus in A3c, a redundant boolean in A3f, a conditional
`pytest.raises` in A3e). After the first run: the A2d failure was marked `xfail(strict=True)` and a post hoc
measurement test was added; ruff/mypy tidy-ups of the module (an equivalent comparison form, a type alias).

| ID | Result | Measured |
|---|---|---|
| A1a | PASS | θ_res = 5.3649, 5.5265, 5.6006, 5.6316, 5.6450 for n = 4, 8, 16, 32, 64 (model 5.657: −0.2 % at n = 64, −0.45 % at n = 32) |
| A1b | PASS | θ_ZZ = 0.9817, 1.0177, 1.0115, 1.0055, 1.0026 (n = 4 … 64) |
| A1c | PASS | `|e|` = 0.838548, 0.431798, 0.217536, 0.108975, 0.0545137; `|e|·64` = 3.4889 (model 3.4894); slope(32→64) = 0.9993 |
| A1d | PASS | κ = 4 vs 1: relative θ difference ≤ 1e-9 |
| A1e | PASS | patch test: all indicators ≤ 1e-12 |
| A2f | PASS | quadrature sum 1.836226661875 vs analytic 1.836226661875, relative 2.2e-16; finite-difference gradient check ≤ 1e-7 |
| A2a | PASS | uniform energy-error slope −0.3262 (N = 1473 … 48641; `e` = 6.31e-2 … 2.01e-2) |
| A2b | PASS | adaptive slope −0.5038 over 12 iterates with N ≥ 1000 (N = 1294 … 32127, `e` = 2.34e-2 … 4.66e-3); stop reason max_dofs |
| A2c | PASS | uniform N = 12033, e = 3.185e-2; adaptive N = 10354, e = 8.215e-3; factor 3.88 (threshold 2.5) |
| A2d | **FAIL (prediction)** | Dörfler θ = 0.5 on uniform h = 1/16 marks 6 of 1536 cells; 4 of the 6 cells touching the origin are in M (the two cells along the Dirichlet rays, indicator share 0.033 each, are not); true error share of M = 0.479 < 0.5. Kept as `xfail(strict=True)`; ledger entry written. Post hoc (no threshold): θ = 0.8 marks 34 cells, all 6 origin cells, true share 0.853; the centroid of every marked cell lies within 0.2 of the origin. The indicator ranks the corner first (the whole marked set is in the corner patch); the wording "every origin cell and half of the error" was too strong for θ_D = 0.5 |
| A2e | PASS | control (no jump term): e = 0.2764 at N = 7428 vs adaptive 8.215e-3 (factor 33.6); control slope −0.000 over 21 levels; origin cell diameter 0.707 vs 6.1e-5 (factor 1.2e4); control stop reason max_iterations |
| A2g | recorded | ZZ-driven adaptive slope −0.501 over 12 levels, final e = 4.63e-3 at N = 32223; θ_res along the adaptive run rises 2.71 → 3.45 and stays there (it is not the uniform-mesh 5.66: constant indicator weights, graded mesh); θ_ZZ 1.04 → 1.00 |
| A3a | PASS | square and L-shape: 6 random rounds, every edge in 1 or 2 cells, no hanging node, positive areas, min angle ≥ 45°, area and boundary length conserved to 1e-14 |
| A3b | PASS | tags and the four facet and node sets equal the geometric sets in 5 rounds |
| A3c | PASS | affine transfer ≤ 1e-14·10 after two refinements; vector transfer; children inside parents; quadratic is interpolated, not reproduced (error > 1e-4) |
| A3d | PASS | identical arrays on repeat; 3 + 3 iterations with restart equal 6 iterations bitwise (mesh, u, indicators) |
| A3e | PASS | stop reasons max_dofs, max_iterations, target, zero_estimator; initial mesh over the cap and unknown estimator are ValueError |
| A3f | PASS | 200 random integer vectors, brute-force minimality and tie order |

Limitations: the effectivity band holds for this smooth problem on this uniform structured mesh and for unit
indicator weights; the adaptive θ_res (about 3.4) differs from the uniform one (5.6), so the indicator is not a
calibrated error bound. The ±15 % band was derived with an O(h) argument and was met with 0.2 % at n = 64. The
adaptive comparison rests on one problem (L-shape, Laplace), one marking fraction and one start mesh; the Dirichlet
data term is part of the indicator (without it the data interpolation error on the coarse outer boundary is not seen
by the estimator; not measured here). Wall time was not compared (cost per DOF not measured). Follow-up (separate
prereg): elasticity (traction jump, κ→(λ, μ) weights, near-incompressible robustness), T6/Q4, goal-oriented
estimators, curved boundaries.
