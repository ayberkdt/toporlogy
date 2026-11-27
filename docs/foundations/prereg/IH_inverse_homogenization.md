# IH — inverse homogenisation on a general periodic mesh: bulk-modulus cells against the Hashin–Shtrikman bound (preregistration)

Preregistered 2026-11-27 10:07 +0300, base SHA `7249632`, before `src/toporlogy/optimization/cell_topology.py` and its test are
written or run. Synthetic, dimensionless (E = 1, ν = 0.3, plane stress, thickness 1).

## What is built

* `PeriodicDensityFilter(mesh, lattice, rmin)`: cone weights max(0, rmin − d_ij) · |Ω_j| with d_ij the minimum-image
  centroid distance over the lattice translates n·a (n ∈ {−1, 0, 1}^dim), rows normalised; `backward` = transpose.
* `optimize_cell(mesh, lattice, D, volume, rmin, objective="bulk", penal, emin, max_iter, move, x0, kkt_tol)`:
  x̃ = filter(x), E_e = emin + x̃_e^p (1 − emin), C from `fem.mesh.homogenization.homogenize_mesh_elasticity` with
  its self-adjoint dC, objective maximise the 2-D bulk modulus κ = (C11 + C12 + C21 + C22)/4 (Voigt [xx, yy, xy]),
  subject to the volume fraction Σ|Ω_e| x̃_e / Σ|Ω_e| ≤ volume; MMA (`optimizers.mma.MMA`); stop at `max_iter` or
  at the KKT residual `kkt_tol` (`optimization.diagnostics.kkt_estimate`). Result: x, x̃, κ, C, history, stop.

## Reference (written here before the run)

2-D Hashin–Shtrikman upper bound of a porous solid (void phase stiffness 0), solid volume fraction f:
κ_HS = f κ μ / ((1 − f) κ + μ), κ = E/(2(1 − ν)) and μ = E/(2(1 + ν)) for plane stress. Source: the general 2-D
form κ1 + f2/(1/(κ2 − κ1) + f1/(κ1 + μ1)) written from memory and reduced here with κ2 = 0 (checks: f = 1 gives κ,
f = 0 gives 0, and κ_HS ≤ f κ, the Voigt bound). The bound applies to the SIMP material only if every
intermediate density is realisable: `core.micromechanics.required_interpolation_parameters(0.3, 2, "stress")`
gives p ≥ 3.077 (measured before this record), so **penal = 3.5** throughout; emin = 1e-9.

## Checks (thresholds fixed here)

| ID | Check | Threshold |
|---|---|---|
| IH1 | gradient of κ(x) through filter, SIMP and homogenisation vs central differences (step 1e-6) for 6 random variables at a random design on a 12 × 12 T3 square cell | rel to max abs gradient ≤ 1e-6 |
| IH2 | periodic filter: rows sum to 1 (1e-14); a constant field is reproduced (1e-14); on a 12 × 12 Q4 square cell, filtering a field shifted by one column of cells equals the shifted filtered field (periodicity, 1e-12); ⟨F x, y⟩ = ⟨x, Fᵀ y⟩ (1e-12) | as stated |
| IH3 | square cell, 40 × 40 T3, f = 0.5, rmin = 0.06 (cell side 1), x0 = 0.5 with the cells whose centroid is within 0.2 of the centre set to 0.25, move 0.2, max_iter 300, kkt_tol 1e-4: (a) final volume ≤ 0.5 (1 + 1e-6); (b) κ ≤ κ_HS(0.5) (1 + 1e-6); (c) κ / κ_HS ≥ 0.80 | as stated |
| IH4 | rhombic (60°) cell of the triangular lattice a1 = (1, 0), a2 = (1/2, √3/2), the 40 × 40 T3 square mesh mapped affinely, same settings and start (centre of the rhombus): (a), (b), (c) as IH3 | as stated |
| IH5 | negative control for IH3(b): penal = 1 (intermediate densities not realisable), same start, max_iter 100: κ > 1.05 κ_HS(0.5) (the bound check is informative, not vacuous) | as stated |

Reported, not thresholded: iterations, stop reason, grey fraction (0.1 < x̃ < 0.9), anisotropy C11/C22 and the
isotropy defect |C11 − C12 − 2 C33| / C11 of the rhombic cell.

## First result

Appended below after the first run.
