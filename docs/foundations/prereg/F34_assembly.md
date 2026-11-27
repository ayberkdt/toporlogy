# F34 — Assembly totals, reactions and a general linear constraint transform (preregistration)

Preregistered 2026-11-27 09:38 +0300, before any new numerical test was written or run. Base SHA `97a0768` (worktree fast-forwarded).
Plan items: WP06 F03/F04 (06_FEM_KERNEL.md), acceptance row F3 (23_ACCEPTANCE_CASES.md: "mass/conduction/capacity
analytic totals; density and cp vary; consistent body and boundary loads total force and moment; nonzero Dirichlet
reaction from original residual. MPC exact constraint and contradictory case rejected").
Test file (written after this commit): `tests/toporlogy/test_fem_assembly_constraints.py`. Write-up after the results:
`docs/foundations/ASSEMBLY_CONSTRAINTS.md`.

## What already exists (read in this session, not assumed)

* `ElasticProblem.mass(rho, lumped)` (consistent / HRZ, rho scalar or per cell), `body_force`, `traction`, `pressure`;
  `HeatProblem.robin`, `flux`; `Dirichlet` with values (n, n_cases); `solve` eliminates `K_ff u_f = f_f − K_fc u_c`
  and returns `reactions = K u − f` at the constrained DOFs (`problem.py`). Tests G3, G4 (force, one linear-traction
  moment, curved facet), G9, M1 (total mass, HRZ), M2 exist in `test_mesh_assembly.py`.
* `ElasticProblem.constrain(kind, ...)` is a placeholder that always raises `UnsupportedBoundaryCondition` (G6 asserts
  it; it is not changed).

## What is missing and added

1. `HeatProblem.capacity(rho_cp, lumped=False)`: heat-capacity matrix `∫ ρ c_p N Nᵀ` (no such method existed).
2. `fem/mesh/constraints.py`: `LinearConstraints` (rows `C u = d`, one or several right-hand-side columns), builders
   `fix`, `tie` (`u_a − u_b = d`), `mpc`, `periodic` (+ `match_nodes`), elimination `eliminate -> Reduction`
   (`u = T z + u0`, `K_r = Tᵀ K T`, `f_r = Tᵀ (f − K u0)`), `solve_constrained(problem, f, constraints)` returning
   `u`, `r = K u − f`, multipliers `λ` with `r = Cᵀ λ`, plus `solve_lagrange` and `solve_penalty` with `cond₁`
   estimates. Errors `ConstraintError` ⊃ `ContradictoryConstraints` (inconsistent right-hand side) and
   `RedundantConstraints` (rank-deficient C: cycles, duplicates; `redundant="drop"` accepts consistent ones).
   Rigid-motion check of the constrained system (`MechanismError`) generalises `_check_supports` to rows `C`.

## Conventions and derived expectations (rule 11)

* Moments: `M_p = Σ_i (x_i − p) × f_i` over nodes, reference `∫ (x − p) × b dV`. 2-D scalar (z) moment
  `(x − p_x) f_y − (y − p_y) f_x`. Loads are applied loads (positive along the vector).
* Reaction `r = K u − f` is the force the support exerts on the body (K u = f + r). Affine field `u = A x + a`,
  `ε = sym A`, `σ = D ε` constant, div-free. Traction `t = σ n_out` on the faces other than the supported face makes
  `u` the exact discrete solution (patch test). For the supported face (outward normal `n_L`, area `S`, centroid `c`):
  `Σ_{nodes on the face} r_i = (S · th) σ n_L`, and
  `Σ x_i × r_i = (S · th) (c × (σ n_L))` (th = thickness in 2-D, 1 in 3-D, S = face length in 2-D). Reason:
  `r_i = ∮_face N_i σ n_L dS` for every node (interior nodes: K u − f = 0; the other faces' nodal loads cancel the
  same faces' share of K u), and `Σ_i N_i x_i = x` on a straight face.
* Heat: `(K T)_i = ∮ N_i κ∇T·n_out dS` for a linear T with no source (integration by parts, ∇·(κ∇T) = 0). `flux(q)`
  loads `∮ N q dS` with q the INWARD flux, so the exact linear field satisfies `K T = Σ_sides flux(side, q_side)`
  with `q_side = κ g · n_out,side` (g = ∇T, κ uniform symmetric tensor); per-side total `q_side · area`.
* Capacity `Σ_ij C_ij = ∫ ρ c_p dV`; mass `Σ_{i,j in comp a,b} M = δ_ab ∫ ρ dV`; first moment
  `Σ_ij x_i M_ij = ∫ ρ x dV` (isoparametric: Σ_i N_i x_i = x).
* Constraint multiplier: stationarity of `½uᵀKu − fᵀu − λᵀ(C u − d)` gives `K u − f = Cᵀ λ`; for a Dirichlet row
  `λ = r_i`. The saddle system `[K Cᵀ; C 0][u; μ] = [f; d]` has `μ = −λ`.
* Periodic cell, lattice vector a, node pair x_B = x_A + a: `u_B − u_A = H a` for macro displacement gradient H;
  the exact solution `u = H x` (+ pin) is linear, so the fluctuation vanishes; average stress
  `V σ̄_ac = Σ_i r_{i,a} x_{i,c}` (f = 0, since `Σ_i N_i x_{i,c} = x_c` gives `Σ_i (K u)_{i,a} x_{i,c} = ∫ σ_ac dV`).
* Quadrature-degree theory used for the curved predictions: moment of a constant body force with the default rule
  needs degree(x · detJ). Curved T6: x deg 2, detJ deg 2 → 4 > default 2 (inexact). Curved T10: x deg 2, detJ deg 3 →
  5 > default 3 (inexact). Curved Q9: x (2,2), detJ (3,3) per direction → (5,5) ≤ 5 of the 3-point rule (exact).
  Force needs detJ only: exact for all. Mass first moment uses the mass rule 2p: curved T6 4 ≥ 4 (exact), Q9 exact,
  curved T10 5 > 4 (inexact). Straight / distorted cells: exact in all cases.

## Checks

Elements `SOLID = Q4, Q9, T3, T6, H8, T4, T10`; meshes `(4,3,2)` cells, interior vertices perturbed by 0.2 h
("distorted"), plus interior higher-order nodes moved by 0.15 h ("curved", elements of degree 2 only). Thickness 0.7 in
2-D. Reference integrals: tensor Gauss–Legendre (numpy, 4 points per direction, exact for the polynomial integrands)
over the box, or `scipy.integrate.quad`; independent per-cell volumes for straight cells (shoelace, determinant, own
trilinear Gauss rule).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| FA1 | consistent mass: component blocks sum to ρV·δ_ab, constant ρ and per-cell ρ (straight cells, independent cell volumes); HRZ lumped keeps the total and is positive | closed form, own cell volumes | rel ≤ 1e-12 |
| FA2 | mass first moment `Σ x_i M_ij = ρ V x̄_box`: distorted all 7 exact; curved T6, Q9 exact; curved T10 NOT exact | closed form | exact: rel ≤ 1e-12; T10 curved: rel > 1e-12 |
| FA3 | `HeatProblem.capacity`: ΣC = ρ c_p V (constant and per-cell ρ c_p), HRZ-lumped total and positivity | as FA1 | rel ≤ 1e-12 |
| FA4 | conduction of a linear field: `K T = Σ flux(side, κ g·n_out)` exactly (tensor κ, distorted and curved), per-side loads sum to q·area | closed form | `‖K T − f‖ ≤ 1e-12 ‖f‖`, totals 1e-12 |
| FA5 | body force b(x) = b0 + B x: force and moment about p equal the analytic integral; distorted exact; curved: force exact, moment exact for Q9, inexact (> 1e-12) for curved T6 and T10 at the default rule and exact (≤ 1e-12) with `degree=8` | Gauss box integral | as stated |
| FA6 | boundary load on the face x = Lx with polynomial traction (degree 2 in the face coordinates): force and moment about p equal analytic (all 7 elements, distorted and curved interior); pressure on a curved 2-D facet (T6, Q9, mid node bulged by a): force `−p th (Δy, 0)` and moment from quad of the parametrisation, default rule; constant traction on the curved facet with `degree=40` vs quad | closed form, quad | rel ≤ 1e-12 |
| FA7 | non-zero Dirichlet reaction of an affine field on the left face (other faces carry `σ n_out`): u reproduces `A x + a` (rel ≤ 1e-10), Σ r and Σ x×r equal analytic (rel ≤ 1e-10); two load cases with different supports (left, then bottom) on the same problem; one call with two per-case value columns | closed form with an explicit Lamé D written in the test | as stated |
| CT1 | `solve_constrained` with Dirichlet rows only equals `ElasticProblem.solve` bitwise (u and reactions, 2 per-case value columns); forced general product path `Tᵀ K T` within 1e-14 | production `solve` (a REGRESSION check, not independent) | `np.array_equal`; 1e-14 |
| CT2 | MPC `u_a − u_b = d` (and a 3-term row) holds in the solution (≤ 1e-12 rel); u and λ equal a dense KKT solve written in the test (≤ 1e-9); sign `μ = −λ` | dense `numpy.linalg.solve` of the saddle system | as stated |
| CT3 | periodic cell (parallelogram image of a rectangle, skew offset a2 = (s Ly, Ly), `H` with a skew part) under macro gradient: `u = H x` (rel ≤ 1e-10), σ̄ from Σ r⊗x equals the explicit-Lamé `D sym H` (rel ≤ 1e-10); Q4, T6 (2-D, distorted interior), H8 (3-D, skew in two directions) | closed form | as stated |
| CT4 | contradictory pair (`u_a = 1`, `u_a = 2`; `u_a − u_b = 1`, `u_b − u_a = 1`) raises `ContradictoryConstraints` naming the rows; consistent duplicate raises `RedundantConstraints` by default and is dropped with `redundant="drop"` with the same solution | — | exact |
| CT5 | cyclic chain `a=b, b=c, c=a` raises `RedundantConstraints` (rank deficiency, rows named); `a−b=1, b−c=1, c−a=1` raises `ContradictoryConstraints`; an acyclic chain a=b=c=d is solved and holds | — | exact |
| CT6 | balance: `Σ_i (Cᵀ λ)_i + Σ f = 0` and Σ moment about p of `Cᵀλ` plus of f is 0, to rounding, for MPC + periodic + Dirichlet; `Cᵀ λ = K u − f` (multiplier recovery residual) | the identity itself | rel ≤ 1e-10 |
| CT7 | Lagrange multiplier solve equals elimination (u ≤ 1e-9, μ = −λ ≤ 1e-8); penalty error ratio `e(α)/e(100α)` ∈ [30, 300] (O(1/α)); `cond₁` of the penalty matrix grows ≥ 10× for 100× α; `cond₁` values are reported | elimination, dense KKT | as stated |
| CN1 | negative: periodic jump with the wrong sign (or without H a) gives `‖u − H x‖/‖H x‖ > 0.1` | — | > 0.1 |
| CN2 | negative: periodic cell without a pin → `MechanismError` naming 2 free rigid motions in 2-D (3 in 3-D): a rigid motion `c + ω×x` meets the homogeneous periodic rows `ω × a_k = 0` only with ω = 0 when the lattice vectors span the space, so exactly the translations are free | — | raises, message count = dim |
| CN3 | negative: reduced load without the `−K u0` term misses the non-zero Dirichlet solution by > 1e-3 relative | production-independent | > 1e-3 |
| CN4 | negative: a constraint on a DOF that belongs to no cell, unmatched or ambiguous periodic nodes, wrong right-hand-side shape → `ValueError` family | — | raises |

Run: `python -m pytest tests/toporlogy/test_fem_assembly_constraints.py -q -p no:randomly -p no:cacheprovider`
with TMP/TEMP and `--basetemp` under `.claude/tmp-toporlogy/F34_assembly`. First results appended below.
