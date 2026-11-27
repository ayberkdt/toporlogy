# IAb — auxetic cell with the compliance-based Poisson ratio (preregistration)

Preregistered 2026-11-27 10:54 +0300, base SHA `25b3f71`, before the code is changed or run. Follows IA
(`prereg/IA_auxetic_cell.md`): the stiffness-ratio objective C12/C11 reached −1.76 through normal–shear coupling
while the uniaxial-stress Poisson ratios of that cell are ≈ 0 (+0.0004, +0.005).

## Change

`optimize_cell(..., objective="poisson_stress", min_E1=...)`: maximise −ν12 with ν12 = −S21/S11 from the full
compliance S = C⁻¹ (uniaxial stress along x, shear free), dS/dE_e = −S (dC/dE_e) S; extra constraint
E1 = 1/S11 ≥ min_E1. Bulk and "poisson" paths unchanged (IP2 fingerprint).

## Checks (thresholds fixed here)

| ID | Check | Threshold |
|---|---|---|
| IB1 | gradients of ν12 = −S21/S11 and of E1 = 1/S11 through filter, projection (β = 8), SIMP and homogenisation vs central differences (step 1e-6), 6 variables, 12 × 12 T3 square cell, random design | rel ≤ 1e-6 each |
| IB2 | the IA2 setting (40 × 40 T3 unit square, f = 0.40, rmin 0.06, p 3.5, emin 1e-9, move 0.2, kkt_tol 1e-4, η = 0.5, betas 1…32, stage_iter 100, start 0.4 + 0.2 cos 2πx cos 2πy) with min_E1 = 0.01 instead of min_C11: (a) volume ≤ 0.40 (1 + 1e-6); (b) E1 ≥ 0.01 (1 − 1e-6); (c) grey ≤ 0.05; (d) ν12 = −S21/S11 ≤ −0.3 | as stated |
| IB3 | thresholded design (x̂ ≥ 0.5 → 1, else emin) re-homogenised: ν12 ≤ −0.2 and C positive definite | as stated |
| IB4 | IP2 fingerprint bitwise; IA1 gradients unchanged | as stated |

Reported: ν21 = −S12/S22, C13/C11, C23/C22, whether the solid is one periodic body.

## First result

Appended below after the first run.
