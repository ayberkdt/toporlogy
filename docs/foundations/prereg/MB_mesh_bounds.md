# MB — kinematic and static boundary conditions on a general-mesh cell: the bounds around the periodic value (preregistration)

Preregistered 2026-11-27 10:10 +0300, base SHA `decc95c` (README edit uncommitted, no code change), before the code is written.
Extends `fem.mesh.homogenization` (MH) with `bc="periodic" | "kubc" | "subc"` in `homogenize_mesh_elasticity`.

* KUBC: u = H x at every boundary node (nodes of facets that belong to one cell only), C_ij = u_iᵀ K u_j / |Y|.
* SUBC: uniform stress σ̄ (unit Voigt stresses): consistent load f = Σ_e ∫ Bᵀ σ̄ dV (equal to the boundary traction
  σ̄ n because div σ̄ = 0), rigid motions held by the fewest DOFs (one node both components, one more component of a
  second node in 2-D; 6 DOFs in 3-D), mean strain ε̄ = (1/|Y|) Σ_e ∫ B u dV = a column of the compliance S,
  C_SUBC = S⁻¹. |Y| is the measure of the mesh (Σ dV), which equals |det lattice| for a periodic cell.
* `lattice` is ignored for KUBC and SUBC (any cell shape); sensitivities only for "periodic" (others refused).

| ID | Check | Threshold |
|---|---|---|
| MB1 | homogeneous material, skew T6 cell and square Q4 cell: KUBC and SUBC give D | rel 1e-10 |
| MB2 | random cell (MH4 skew T6, E ∈ [0.2, 2], seed 5) and the MH5 two-phase T3 cell: C_SUBC ≤ C_PBC ≤ C_KUBC in the Loewner order | min eigenvalue of each difference ≥ −1e-10 max abs C |
| MB3 | the same two cells: the ordering is strict (they differ) | min eigenvalue of C_KUBC − C_SUBC ≥ 1e-6 max abs C |
| MB4 | on the 12 × 10 Q4 rectangle with random E (MH2 case), KUBC and SUBC equal the regular-grid `homogenize_elasticity(bc="kubc" | "subc")` | rel 1e-10 |
| MB5 | refusals: unknown bc, sensitivities with kubc/subc | ValueError |

## First result

Appended below after the first run.

## First result (run 10:11 +0300 by the clock; recorded 2026-11-27 10:12 +0300)

`python -m pytest tests/toporlogy/test_mesh_bounds.py tests/toporlogy/test_mesh_homogenization.py -q -p no:randomly -s`:
21 passed on the first run (6 MB tests, 15 MH regressions); no threshold, input or assertion changed.

| ID | Result | Measured (threshold) |
|---|---|---|
| MB1 | PASS | skew T6: KUBC 2.6e-15, SUBC 1.0e-15; square Q4: KUBC 4.0e-16, SUBC 1.0e-15 (1e-10) |
| MB2 | PASS | min eig / max abs C — skew T6: PBC − SUBC 2.5e-2, KUBC − PBC 9.5e-3; two-phase T3: 3.3e-2, 3.1e-2 (≥ −1e-10) |
| MB3 | PASS | KUBC − SUBC: 5.1e-2 and 7.2e-2 (≥ 1e-6) |
| MB4 | PASS | against the regular grid: KUBC 8.4e-15, SUBC 2.1e-14 (1e-10) |
| MB5 | PASS | refusals raised |
