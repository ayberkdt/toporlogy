# Assembly totals, reactions and linear constraints on the general mesh (WP06 F03 / F04 slice)

Record: `docs/foundations/prereg/F34_assembly.md`; tests: `tests/toporlogy/test_fem_assembly_constraints.py`. Code:
`src/toporlogy/fem/mesh/constraints.py` (new, import by module; not exported from the package `__init__`) and
`HeatProblem.capacity` in `problem.py`. All materials and loads in the tests are synthetic.

## What already existed (read, not assumed) and what was added

| Item | Before | Now |
|---|---|---|
| consistent mass, HRZ lumping, body force, traction, pressure | `ElasticProblem.mass / body_force / traction / pressure` | unchanged; analytic totals, moments and curved cases added as tests |
| heat conduction, Robin, flux | `HeatProblem.K / robin / flux` | unchanged; exact-flux test of a linear field |
| heat capacity ∫ρ c_p N Nᵀ | none | `HeatProblem.capacity(rho_cp, lumped=False)` |
| Dirichlet with values, reactions `K u − f` at constrained DOFs, several columns | `Dirichlet`, `solve` | unchanged; non-zero affine-field reaction and two supports on one problem tested |
| MPC, periodicity, general `u = T z + u0` | `ElasticProblem.constrain` raised | `LinearConstraints`, `solve_constrained` (`constrain` still raises, its message now points here) |

## Conventions

* Moment `M_p = Σ_i (x_i − p) × f_i` (2-D: z component `(x − p_x) f_y − (y − p_y) f_x`).
* Reaction `r = K u − f` is the force the supports exert on the body; for an affine field with traction `σ n_out`
  on the free faces, `Σ r = (S th) σ n_L` and `Σ x × r = (S th) c × (σ n_L)` on the supported face.
* Heat: `flux(q)` takes the INWARD flux; the exact linear field has `K T = Σ_sides flux(κ g·n_out)`.
* Multiplier: `r = K u − f = Cᵀ λ`; the saddle-point (Lagrange) system returns `μ = −λ`.

## The transform `u = T z + u0`

`LinearConstraints` holds rows `C u = d` (one or several right-hand-side columns). Builders: `fix`,
`from_dirichlet`, `tie` (`u_a − u_b = d`), `mpc` (one row, any length), `periodic(mesh, minus, plus, offset, jump)`
(matched node sets, any offset vector, prescribed jump, e.g. `macro_jump(H, a) = H a`), `periodic_lattice(mesh,
lattice, H)` (matches `x` and `x + a` automatically); sets combine with `+`.

`eliminate(universe, rtol, redundant)` is sparse Gauss-Jordan: each row is reduced by the slaves found so far, solved
for its largest coefficient and substituted into the earlier slaves, so every slave depends on masters only. Result
`Reduction`: `T`, `u0`, masters, slaves, the independent rows (`C_used`), dropped rows, and `reduce_matrix`
(`Tᵀ K T`), `reduce_load` (`Tᵀ (f − K u0)`), `expand`, `multipliers`. For Dirichlet rows only (`pure_selection`) the
reduced matrix is the sub-matrix `K[m][:, m]` and the load `f_m − K_ms u_s`, the exact expression of
`ElasticProblem.solve`; `reduce_*(general=True)` forces the triple product.

`solve_constrained(problem, f, constraints)` returns `u`, `reactions = K u − f` (all DOFs), `multipliers` (independent
rows), `multiplier_residual = ‖Cᵀλ − r‖ / ‖r‖`, `constraint_forces = Cᵀ λ` and the reduction. DOFs that no cell
touches are not unknowns and a constraint on one is refused. Works for `ElasticProblem` and `HeatProblem`.

### Detection

| Situation | Error |
|---|---|
| row empty after substitution, right-hand side not zero (`u_a = 1`, `u_a = 2`; `u_a − u_b = 1, u_b − u_a = 1`; cycle with non-zero sum) | `ContradictoryConstraints`, names the row and the rows it depends on |
| row empty and consistent (duplicate, cycle `a = b, b = c, c = a`) | `RedundantConstraints` ("rank deficient"), or dropped with `redundant="drop"` |
| rows leave a rigid motion of a piece free (periodic cell without a pin) | `MechanismError` with the count (2 in 2-D) |
| constraint on a DOF no cell touches; unmatched, ambiguous or unequal periodic node sets; right-hand side of the wrong shape | `ConstraintError` / `ValueError` |

`redundant="drop"` is needed for periodic cells: the corner nodes appear in several pairs, so those rows are
dependent but consistent (the tests count at least one dropped row).

### Alternatives and conditioning (measured, synthetic T6 plate, 70 DOFs, 3 rows families)

| Method | System | cond₁ estimate | Accuracy against elimination |
|---|---|---|---|
| elimination | `K_r` (SPD) | 2.09e3 | reference |
| Lagrange (`solve_lagrange`) | saddle `[K Cᵀ; C 0]`, SuperLU with pivoting | 2.11e3 | u 4.6e-15, μ + λ 5.2e-15 |
| penalty α = 1e3 (`solve_penalty`) | `K + α s ĈᵀĈ` | 1.42e6 | error 6.0e-5 |
| penalty α = 1e5 | same | 1.42e8 | error 6.0e-7 |

The penalty error falls as 1/α (ratio 100.0 for 100× α) and the condition number grows as α (ratio 99.9), so
elimination is the default; the saddle system costs nothing in conditioning here (cond₁ 2.11e3) but is indefinite and
larger. The numbers are one problem, not a general statement.

## Findings

1. **The default load rule under-integrates non-constant body forces** (FA5c). `body_force` uses the stiffness degree
   (T3/T4 centroid, H8 2×2×2). With a linear `b(x)` the force resultant is exact for all seven elements but the
   moment is not for T3 (4.8e-2), T4 (7.2e-3) and H8 (7.3e-9); `degree=8` makes all exact (≤ 1.2e-15). Not changed
   here: callers with variable loads should pass `degree`.
2. **Curved T6 and T10 moments of a constant body force are not exact at the default rule** (T6 1.4e-4, T10 2.4e-6;
   Q9 is exact), as the degree count x·detJ predicts; the force resultant is exact; `degree=8` is exact. Curved T10
   mass first moment is also inexact at the mass rule 2p (6.4e-8), T6 and Q9 are exact. The distributed load
   `∫ N_i b detJ` is integrable exactly with a rule of degree p + deg detJ; not changed here.
3. Pressure on a curved 2-D facet is exact at the default facet rule (polynomial `n dS`); a constant traction needs
   `degree=40` against `scipy.integrate.quad` for 1e-12 (the arc length is `√poly`; G4 records the default miss).
4. A reaction of an affine field under non-zero Dirichlet values on one face equals the analytic support force and
   moment to 3.4e-14 / 1.8e-14 on all seven elements (distorted meshes), for two different supports on one problem
   object and for two value columns in one call.

## Not done

* No constraint is built for rotated (oblique) supports; `ConstrainedElasticity` of the regular grid has them.
* `ElasticProblem.constrain(kind, ...)` is unchanged (still raises, G6); use `constraints.py`.
* Matrix-free apply, multi-block mixed assembly and DOF-permutation tests of F04 are not part of this slice.
* Nonlinear (Newton) use of the transform (`Tᵀ r`, `Tᵀ K_t T`) follows from the same formulas but is not wired or
  tested here.
