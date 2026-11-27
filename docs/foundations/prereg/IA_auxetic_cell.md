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

## First result (IA1/IA4 run 10:40, IA2/IA3 run 10:40–10:45 +0300 by the clock; recorded 2026-11-27 10:46 +0300)

All checks PASS on the first run; no threshold, input or assertion changed. IP1, IH1, IH2 unchanged.

| ID | Result | Measured (threshold) |
|---|---|---|
| IA1 | PASS | ν gradient 1.8e-8, C11 gradient 7.9e-9 (1e-6) |
| IA2 | PASS | (a) volume 0.39996392; (b) C11 0.02000 (constraint active); (c) grey 0.0109 (≤ 0.05); (d) **C12/C11 = −1.762** (≤ −0.5). 600 iterations (stage caps), KKT 2.2e-2 at β = 32; C33 7.6e-3 |
| IA3 | PASS | thresholded design: C12/C11 = **−1.660** (≤ −0.3), min eigenvalue of C 6.6e-4 (> 0); the solid is one periodic body |
| IA4 | PASS | the bulk path (IP2 fingerprint) bitwise unchanged |

**Convention, stated explicitly (the trap of E04):** the design is not square-symmetric (C11 = 0.020, C22 = 0.095),
so the registered quantity C12/C11 is not the physical ν12. With S = C⁻¹ of an orthotropic 2-D cell, uniaxial stress
along y gives −ε_x/ε_y = −S12/S22 = C12/C11 = **ν21 = −1.76**, and uniaxial stress along x gives −ε_y/ε_x =
−S12/S11 = C12/C22 = **ν12 = −0.37** (thresholded: −1.66 and −0.38). The cell is auxetic in both directions; the
large number belongs to loading along y. |C12| < √(C11 C22) holds (positive definite), so a value below −1 is
admissible for an anisotropic cell. The registered label "the stiffness-ratio ν12 of the square-symmetric convention"
applies to the check as written (C12/C11 ≤ −0.5); the physical names above are the correct reading.
