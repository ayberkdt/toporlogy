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
