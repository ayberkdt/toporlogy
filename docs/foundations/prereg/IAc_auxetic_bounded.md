# IAc — auxetic cell with all three moduli bounded (preregistration)

Preregistered 2026-11-27 11:10 +0300, base SHA `a1f6d39`, before the code is changed or run. Follows IAb: the compliance-based
objective with only E1 bounded produced a near mechanism (thresholded min eigenvalue 1.1e-9 = void level, the solid
not one body, ν12 = −36).

## Change

`optimize_cell(..., objective="poisson_stress", min_moduli=(E1_min, E2_min, G12_min))`: three constraints
E1 = 1/S11, E2 = 1/S22, G12 = 1/S33 ≥ their minima (full compliance; one homogenisation per iteration for the three),
d(1/S_ii) = (S dC S)_ii / S_ii². `min_E1` stays as it was (IAb).

## Checks (thresholds fixed here)

| ID | Check | Threshold |
|---|---|---|
| IC0 | the connectivity check used below: a full solid cell → one body; two horizontal solid stripes separated by void rows → two bodies (not one); a single diagonal strip that wraps across the periodic faces → one body | as stated |
| IC1 | gradients of E1, E2, G12 through filter, projection (β = 8), SIMP and homogenisation vs central differences (step 1e-6), 6 variables, 12 × 12 T3 square cell, random design | rel ≤ 1e-6 each |
| IC2 | the IAb setting with min_moduli = (0.01, 0.01, 0.002): (a) volume ≤ 0.40 (1 + 1e-6); (b) E1, E2, G12 ≥ their minima (1 − 1e-6); (c) grey ≤ 0.05; (d) ν12 = −S21/S11 ≤ −0.2 | as stated |
| IC3 | thresholded design (x̂ ≥ 0.5 → 1, else emin): (a) min eigenvalue of C ≥ 1e-4 (10⁵ × the void level); (b) ν12 ≤ −0.1; (c) the solid is one periodic body | as stated |

## First result

Appended below after the first run.

## First result (IC0/IC1 run 11:11, IC2/IC3 run 11:11–11:19 +0300 by the clock; recorded 2026-11-27 11:19 +0300)

All checks PASS on the first run; no threshold, input or assertion changed.

| ID | Result | Measured (threshold) |
|---|---|---|
| IC0 | PASS | full cell one body; two stripes not one body; wrapping diagonal one body |
| IC1 | PASS | E1 6.4e-9, E2 1.4e-8, G12 1.5e-8 (1e-6) |
| IC2 | PASS | (a) volume 0.39999995; (b) E1 0.1375, E2 0.0100 (active), G12 0.00234 (≥ 0.01, 0.01, 0.002); (c) grey 0.0034; (d) **ν12 = −S21/S11 = −2.34** (≤ −0.2); ν21 = −0.170; 596 iterations, KKT 3.2e-4 |
| IC3 | PASS | thresholded: (a) min eigenvalue of C **2.26e-3** (≥ 1e-4; 2 × 10⁶ × the void level); (b) ν12 = −2.27 (≤ −0.1); (c) one periodic body; moduli 0.145, 0.0105, 0.0024 |

Reading: with all three moduli bounded the optimiser no longer reaches a mechanism, and the cell is auxetic under
uniaxial stress in both directions, from the full compliance (shear free): ν12 = −2.34 (stress along x) and
ν21 = −0.17 (stress along y). Consistency of the measured numbers: ν12/E1 = −17.0 = ν21/E2 (compliance symmetry) and
ν12 ν21 = 0.40 < 1 (positive definiteness of the orthotropic in-plane block). The thresholded 0/1 design keeps the
values (−2.27, min eigenvalue 2.3e-3), so the response does not live in grey or void material. Together with IA and
IAb: the stiffness-ratio objective is exploitable by shear coupling, the compliance objective with one modulus by a
mechanism; bounding E1, E2 and G12 removes both.

Figure (tiled 3 × 3, base cell outlined): `fig_iac_cell.png` / `.pdf`; `scripts/ia_design.py bounded` re-runs IC2 deterministically (same ν12 −2.3365, ν21 −0.1699; full C and S in `IAc_design.json`) and `scripts/ih_cells_figure.py iac` draws it (datawiz audit 0 errors). The cell: wavy horizontal beams joined by staggered vertical posts.

### Test set-up note (2026-11-27 11:53 +0300)

A later re-run of the whole file in parallel with four other optimisation files and the clean suite hit the
project-wide pytest timeout of 900 s (IB2 takes 692 s alone). The slow optimisation test now carries
`@pytest.mark.timeout(2400)`; no threshold, input or assertion changed.
