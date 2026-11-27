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

## First result (IH1/IH2 run 10:09, IH3–IH5 run 10:09–10:14 +0300 by the clock; recorded 2026-11-27 10:17 +0300)

Order of events: the module and the test were written after this record's commit decc95c. They were committed
together with MB in ed3bed5 (an error of grouping, not of content), after the first IH1/IH2 run and while IH3–IH5
were running. No threshold, input or assertion was changed after any run.

| ID | Result | Measured (threshold) |
|---|---|---|
| reference limits | PASS | κ_HS(1) = κ, κ_HS(0) = 0, κ_HS ≤ f κ on 11 points |
| IH1 | PASS | worst 2.0e-8 relative to max abs gradient (1e-6) |
| IH2 | PASS | rows and constant field ≤ 1e-14; periodicity under a one-column shift 5.6e-16 (1e-12); adjoint 0 (1e-12) |
| IH3(a), (b) | PASS | volume 0.50000000; κ 0.063135 ≤ κ_HS 0.185185 |
| **IH3(c)** | **FAIL** | κ / κ_HS = **0.341** (≥ 0.80). 300 iterations (cap), KKT 6.7e-4; grey fraction 1.000, C11/C22 1.0000 |
| IH4(a), (b) | PASS | volume 0.49999903; κ 0.063208 ≤ κ_HS |
| **IH4(c)** | **FAIL** | κ / κ_HS = **0.341**; 300 iterations, KKT 1.0e-2; grey fraction 1.000; isotropy defect 2.1e-3 |
| IH5 | PASS | penal = 1: κ = 0.357143 = 1.93 κ_HS (> 1.05 κ_HS); the bound check is informative |

What happened: both optimisations returned to a uniform grey design (x̃ ≈ 0.5 in every cell; κ = 0.5^3.5 · E/(2(1−ν))
= 0.0631, which equals the measured value). The uniform design is a stationary point of this problem: all
sensitivities are equal and proportional to the volume gradient. The registered start, a central hole at 0.25 in a 0.5
field, did not leave its basin: MMA filled the hole in. The hypothesis "this start leads to a near-optimal cell" is
refuted. The machinery is unaffected: gradient (IH1), filter (IH2), bounds (a, b) and the negative control (IH5) all
hold. Strict xfails (slow); ledger `ledger_additions/IH_inverse_homogenization.json`.

## Post hoc start study (not preregistered; `scripts/ih_starts_posthoc.py`, `IH_starts_posthoc.json`; runs 10:15–10:19 +0300, recorded 2026-11-27 10:20 +0300)

Same settings as IH3/IH4, only the start changed:

| start | κ/κ_HS of the start | κ/κ_HS final | stop | grey fraction | C11/C22, isotropy defect |
|---|---|---|---|---|---|
| square, 0/1 void hole of area 0.5 | 0.787 | **0.841** | KKT 9.8e-5 at 88 it | 0.157 | 1.000; C33/C11 = 0.076 (strongly anisotropic) |
| rhombus (60°), 0/1 void hole of area 0.5 | 0.828 | **0.837** | KKT 9.2e-5 at 76 it | 0.169 | 1.000; \|C11 − C12 − 2C33\|/C11 = 7e-3 (nearly isotropic) |
| square, random in [0, 1] scaled to mean 0.5 (seed 0) | 0.327 | 0.341 | 300 it cap, KKT 5.0e-3 | 1.000 | uniform grey again |

So the start decides: from a 0/1 start MMA reaches KKT-stationary cells at 84 % of the HS bound and improves the
start by 6.8 % (square) and 1.0 % (rhombus). From grey starts, the registered hole and a random field alike, it returns
to the uniform grey design, which is therefore not only stationary but attracting for these settings (filter radius
2.4 elements, p = 3.5). Both designs stay below the bound, which holds for p ≥ 3.08. The 0.80 threshold of IH3(c)/IH4(c)
is met post hoc from the void-hole start. This is not a pass of the registered check: the start is part of the check.
Not established: whether a larger radius, a p-continuation or a projection step would close the remaining 16 %
(published bulk-modulus cells come closer to the bound; not verified here).

Figure of the two post hoc cells, tiled 3 × 3 (base cell outlined): `fig_ih_cells.png` / `.pdf` (`scripts/ih_cells_figure.py`, datawiz audit 0 errors, `fig_ih_cells.denetim.txt`).
