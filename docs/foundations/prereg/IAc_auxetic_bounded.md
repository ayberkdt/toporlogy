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
