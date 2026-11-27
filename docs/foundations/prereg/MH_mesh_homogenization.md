# MH — periodic homogenisation on a general mesh, with design sensitivities (preregistration)

Preregistered 2026-11-27 09:58 +0300, base SHA `9e7ca00` (foundations/round-3 after the L6e/F34 merges; the stamp was written at 09:58, the base line filled in at commit),
before `src/toporlogy/fem/mesh/homogenization.py` and its test are written. Synthetic, dimensionless moduli.

## What is built

`homogenize_mesh_elasticity(mesh, D, lattice, E=None, thickness=1, sensitivities=False)` and
`homogenize_mesh_conduction(mesh, kappa, lattice, k=None, ...)` in `toporlogy.fem.mesh.homogenization`. The cell is the
whole mesh with lattice vectors `lattice` (any, skew allowed); per cell the material is `E_e · D` (`k_e · kappa`).
For each unit macro strain (Voigt, engineering shear; [xx, yy, xy] in 2-D, [xx, yy, zz, yz, xz, xy] in 3-D) the
displacement u = H x + w is found with `fem.mesh.constraints.periodic_lattice(mesh, lattice, H)` plus one pinned node,
solved by elimination (`solve_constrained`, dependent corner rows dropped). Then
C_ij = (1/|Y|) u_iᵀ K u_j, |Y| = |det lattice| (× thickness in 2-D), and dC_ij/dE_e = (1/|Y|) u_iᵉᵀ K_e(D) u_jᵉ
(energy form, self-adjoint). H = [[εxx, γxy/2], [γxy/2, εyy]] (3-D analogous). Conduction: T = g·x + w, k_ij likewise.
Result `MeshHomogenized` (C, dC (n_cells, k, k) or None, the unit-case displacements, |Y|). Public names new and
unique: `homogenize_mesh_elasticity`, `homogenize_mesh_conduction`, `MeshHomogenized`.

## Checks (thresholds fixed here)

| ID | Check | Independent reference | Threshold |
|---|---|---|---|
| MH1 | homogeneous material on a skew cell (2-D shear 0.35: Q4, T3, T6, Q9; 3-D skew Aff of CT3b: H8, T4) | C_eff = D exactly (affine solution is periodic-compatible) | rel 1e-10 (max abs / max abs D) |
| MH2 | 12 × 10 Q4 rectangle, random per-cell E ∈ [0.1, 2] (seed 3), plane stress ν = 0.3 | grid `fem.homogenization.homogenize_elasticity` on `Domain((12, 10))`, cells matched by centroid; also its dC | C rel 1e-10; dC rel 1e-9 |
| MH3 | three layers normal to y (E = 1, 7, 3; thicknesses 3/12, 5/12, 4/12) on a mesh whose nodes are moved in x only (layers stay conforming), T3 and Q9 | Backus layered average written in the test (dense numpy) | rel 1e-9 |
| MH4a | sensitivities on a skew T6 cell with random E: dC_ij/dE_e vs central differences (step 1e-6 · E_e) for 8 random cells, all ij | FD | rel to max abs dC: 1e-6 |
| MH4b | Euler identity Σ_e E_e dC/dE_e = C (C is homogeneous of degree 1 in E) | algebra | rel 1e-12 |
| MH5 | random two-phase T3 cell (E ∈ {1, 10}, 50 %): C_Reuss ≤ C ≤ C_Voigt in the Loewner order | phase-volume Voigt/Reuss from D | min eigenvalue of the differences ≥ −1e-10 max abs C |
| MH6 | conduction: homogeneous skew cell gives kappa (rel 1e-10); MH2 analogue vs grid `homogenize_conduction` (rel 1e-10) | as stated | as stated |
| MH7 | renumbering invariance: random node and cell permutation of the MH4 mesh | C and dC (permuted back) | rel 1e-12 |
| MH8 | refusals: a lattice vector that matches no node pair → `ConstraintError`; E of wrong length or ≤ 0 → ValueError; a degenerate lattice (det 0) → ValueError | — | raised |

## First result

Appended below after the first run.
