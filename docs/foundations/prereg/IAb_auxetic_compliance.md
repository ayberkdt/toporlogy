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

## First result (IB1/IB4 run 10:55, IB2/IB3 run 10:56–11:08 +0300 by the clock; recorded 2026-11-27 11:09 +0300)

| ID | Result | Measured (threshold) |
|---|---|---|
| IB1 | PASS | ν12 gradient 1.2e-8, E1 gradient 6.4e-9 (1e-6) |
| IB2 | PASS as registered | (a) volume 0.39894; (b) E1 0.01219; (c) grey 0.0453; (d) ν12 = −S21/S11 = −36.1 (≤ −0.3). ν21 = −0.019; C13/C11 −2.3e-4, C23/C22 2.2e-2; 600 iterations, KKT of the last iterate **1.0** (not converged) |
| IB3 | PASS as registered | thresholded ν12 = −16.0 (≤ −0.2); min eigenvalue of C **1.08e-9** (> 0); **solid not one periodic body** |
| IB4 | PASS | IP2 fingerprint bitwise; IA1 unchanged (ν 1.8e-8, C11 7.9e-9) |

**The passes are not a valid auxetic material.** The thresholded cell's smallest stiffness eigenvalue (1.1e-9) is at
the level of the void modulus (emin = 1e-9 E): the solid falls apart into pieces joined only by "void", a near
mechanism, and the extreme ratios (−36, −16) come from that soft mode. The registered thresholds were too weak to
see this: "C positive definite" is satisfied by the void material, and a ν12 bound alone rewards approaching a
mechanism. Thresholds are not changed after the fact; the lesson is recorded: a physical auxetic design needs all
moduli bounded below (E1, E2 and G12 from the full compliance) and the connectivity of the solid checked. Follow-up
record: IAc.

### Test set-up note (2026-11-27 11:53 +0300)

A later re-run of the whole file in parallel with four other optimisation files and the clean suite hit the
project-wide pytest timeout of 900 s (IB2 takes 692 s alone). The slow optimisation test now carries
`@pytest.mark.timeout(2400)`; no threshold, input or assertion changed.
