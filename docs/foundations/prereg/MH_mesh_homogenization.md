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

## First result (run 10:03 +0300 by the clock; recorded 2026-11-27 10:03 +0300)

Order of events, stated plainly: the stamp line was written at 09:58; the module and the test file were drafted
between 09:58 and 10:02 while the record waited for the integration commit 9e7ca00; the record was committed at
10:02 (11a4828) and **no test of this file was run before that commit**. Command:
`python -m pytest tests/toporlogy/test_mesh_homogenization.py -q -p no:randomly -s`: **15 passed on the first run**,
no threshold, input or assertion changed.

| ID | Result | Measured (threshold) |
|---|---|---|
| MH1 | PASS | Q4 8.1e-16, T3 8.1e-16, T6 1.6e-15, Q9 1.0e-14, H8 1.6e-16, T4 8.2e-16 (1e-10) |
| MH2 | PASS | C 1.4e-14 (1e-10), dC 6.4e-14 (1e-9) against the regular-grid implementation |
| MH3 | PASS | T3 3.7e-15, Q9 7.1e-15 against the layered average (1e-9) |
| MH4a | PASS | worst 1.1e-8 relative to max abs dC over 8 cells × all ij (1e-6) |
| MH4b | PASS | Euler identity 4.2e-15 (1e-12) |
| MH5 | PASS | min eig (C − C_R) 0.766, (C_V − C) 0.567 (≥ −1e-10 max abs C) |
| MH6 | PASS | homogeneous 1.2e-14, against the grid 1.8e-14 (1e-10) |
| MH7 | PASS | C 3.9e-15, dC 1.0e-14 (1e-12) |
| MH8 | PASS | the four refusals raised as registered |

Scope: linear elasticity and conduction, small strain, one periodic cell given as a conforming mesh whose opposite
faces have matching nodes. Not done: a Study response on the general mesh (the grid has
`optimization.homogenization_responses`), KUBC/SUBC bounds on the mesh.

## Robustness addition MH9 (preregistered 2026-11-27 12:31 +0300, base `3a57914`, before the code change)

A lattice that is a sub-period of the mesh (e.g. half the true vectors on a structured mesh) still matches node pairs,
so `periodic_lattice` accepts it, imposes a stronger periodicity and |Y| = |det lattice| is wrong: a silent wrong
answer. A periodic cell tiles the plane (space) with exactly its own measure, so |det lattice| must equal the measure
of the mesh, Σ dV. Change (periodic bc, elasticity and conduction): refuse with ValueError when
| |det L| − Σ dV | > 1e-9 Σ dV. Check MH9: (a) the MH2 rectangle 12 × 10 with lattice diag(6, 10) (half period in x,
nodes match) → ValueError naming both measures; (b) every MH1–MH8 test unchanged; (c) the cell tests (fast) unchanged.

### MH9 result (run 12:32 +0300 by the clock; recorded 2026-11-27 12:32 +0300)

PASS on the first run: lattice diag(6, 10) on the 12 × 10 rectangle refused for elasticity and conduction ("mesh
measure"); MH1–MH8 values unchanged to the printed digits; MB and all fast cell tests pass (34 passed). The 40 × 40
square and rhombic cells of the slow runs pass the new check (measures 1.0 and 0.866025403784). The guard adds a
refusal only; this commit postdates the final clean suite (8e39c7c) and is covered by the targeted run named here.

## Robustness addition MH10 (preregistered 2026-11-27 12:39 +0300, base `dcef880`, before the code change)

Measured before this record (probe at dcef880): `homogenize_mesh_conduction` accepts κ = diag(1, −1) and returns a
negative effective conductivity, and a non-symmetric κ = [[1, 0.5], [0, 1]] is symmetrised in silence (result
[[1, 0.25], [0.25, 1]]). The elasticity path refuses a non-SPD D (`ElasticProblem.set_D`), conduction does not.
Change: κ must be symmetric (|κ − κᵀ| ≤ 1e-12 |κ|) and positive definite (Cholesky), else ValueError. Check MH10:
both probes refused ("symmetric", "positive definite"); κ scalar 2.5 and an SPD anisotropic κ accepted (the
homogeneous cell returns κ, rel 1e-10); MH1–MH9 unchanged.

### MH10 result (run 12:39 +0300 by the clock; recorded 2026-11-27 12:39 +0300)

PASS on the first run: diag(1, −1) refused ("positive definite"), [[1, 0.5], [0, 1]] refused ("symmetric"); the SPD
anisotropic κ and the scalar 2.5 are returned by the homogeneous skew cell to ≤ 1e-10. MH and MB files: 23 passed.
The conduction path is not used by the cell optimisation, so the slow results are unaffected.

## Robustness addition MH11 (preregistered 2026-11-27 12:46 +0300, base `773fd33`, before the code change)

Measured before this record (probe at 773fd33): thickness 0 fails inside the factorisation ("Factor is exactly
singular") and thickness −1 is accepted with a positive C (negative K over a negative volume). The root is that
`fem.mesh.problem.ElasticProblem` does not check the thickness (pre-existing; a follow-up). Change here:
`homogenize_mesh_elasticity` and `homogenize_mesh_conduction` require a finite thickness > 0 (ValueError). Check
MH11: thickness 0, −1 and NaN refused in both functions; MH1–MH10 unchanged.

### MH11 result (run 12:46 +0300 by the clock; recorded 2026-11-27 12:47 +0300)

PASS on the first run: thickness 0, −1 and NaN refused in both functions; MH, MB and the fast cell tests: 43 passed.
The slow runs use thickness 1 (default), so their results stand. Follow-up (not done): the same check in
`fem.mesh.problem.ElasticProblem` / `HeatProblem`, which accept a negative thickness.
