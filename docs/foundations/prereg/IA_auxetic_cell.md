# IA — auxetic cell on a general periodic mesh with projection (preregistration)

Preregistered 2026-11-27 10:39 +0300, base SHA `25adc3c`, before the code is changed or run. The formulation of the historical grid
run `docs/olcumler/auxetic_hucre_onkayit.md` (min ν12 = C12/C11, volume ≤ 0.40, C11 ≥ 0.02 E0, start
x = 0.4 + 0.2 cos(2πx) cos(2πy)) moved to `optimization.cell_topology` on a general mesh, with the IP projection.

## Change

`optimize_cell(..., objective="poisson", min_C11=0.02)`: maximise −C12/C11 (the stiffness-ratio ν12 of the
square-symmetric convention; MATHEMATICAL_CONVENTIONS / E04) with the extra constraint 1 − C11/min_C11 ≤ 0; dC/dx̂ from
the self-adjoint dC/dE of `homogenize_mesh_elasticity`. The bulk path is not changed (IP2 fingerprint stays).

## Checks (thresholds fixed here)

| ID | Check | Threshold |
|---|---|---|
| IA1 | gradients of ν12 and of C11 through filter, projection (β = 8), SIMP and homogenisation vs central differences (step 1e-6), 6 variables, 12 × 12 T3 square cell, random design | rel to max abs gradient ≤ 1e-6 each |
| IA2 | 40 × 40 T3 unit square cell, f = 0.40, rmin 0.06, p 3.5, emin 1e-9, move 0.2, kkt_tol 1e-4, η = 0.5, betas (1, 2, 4, 8, 16, 32), stage_iter 100, the historical start at the centroids: (a) volume of x̂ ≤ 0.40 (1 + 1e-6); (b) C11 ≥ 0.02 (1 − 1e-6); (c) grey fraction (0.1 < x̂ < 0.9) ≤ 0.05; (d) ν12 = C12/C11 ≤ −0.5 | as stated |
| IA3 | the IA2 design thresholded (x̂ ≥ 0.5 → 1, else emin) and homogenised again: ν12 ≤ −0.3 and C positive definite (min eigenvalue > 0) | as stated |
| IA4 | IP2 fingerprint (bulk path) still bitwise | bitwise |

Reported, not thresholded: ν21 = C12/C22, C33, iterations, KKT, whether the solid phase of the thresholded design is
one periodic body.

## First result

Appended below after the first run.
