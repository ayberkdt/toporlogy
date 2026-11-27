# IP — Heaviside projection with β continuation for the IH cells (preregistration)

Preregistered 2026-11-27 10:25 +0300, base SHA `6dc11a1`, before the projection is added to `optimization/cell_topology.py` and
before any run. Follows the IH post hoc study (prereg/IH_inverse_homogenization.md): from a 0/1 void-hole start the
cells reached κ/κ_HS = 0.841 (square) and 0.837 (rhombus) with grey fractions 0.157 and 0.169.

## Change

`optimize_cell(..., projection=(eta, betas), stage_iter=100)`: x̄ = F x (periodic filter), x̂ = (tanh(βη) +
tanh(β(x̄ − η))) / (tanh(βη) + tanh(β(1 − η))) (the formula of `design.chain.HeavisideProjection`, Ferrari and
Sigmund 2020), SIMP and the volume on x̂. β runs through `betas`; each stage stops at the KKT residual `kkt_tol` or
after `stage_iter` iterations, and MMA is restarted (new asymptotes) at each stage. Without `projection` the
function is unchanged (IH tests unchanged).

## Checks (thresholds fixed here)

| ID | Check | Threshold |
|---|---|---|
| IP1 | gradient of κ through filter, projection (β = 8, η = 0.5), SIMP and homogenisation vs central differences (step 1e-6), 6 variables, 12 × 12 T3 square cell, random design | rel to max abs gradient ≤ 1e-6 |
| IP2 | without `projection`, `optimize_cell` returns the same x as before the change, bitwise, for 5 iterations of the IH3 setting | bitwise |
| IP3 | square cell, settings of the IH post hoc void-hole run (40 × 40 T3, f = 0.5, rmin 0.06, p 3.5, emin 1e-9, move 0.2, kkt_tol 1e-4) with η = 0.5, betas (1, 2, 4, 8, 16, 32), stage_iter 100: (a) volume of x̂ ≤ 0.5 (1 + 1e-6); (b) κ ≤ κ_HS (1 + 1e-6); (c) grey fraction of x̂ (0.1 < x̂ < 0.9) ≤ 0.05; (d) κ/κ_HS ≥ 0.85 | as stated |
| IP4 | rhombic cell, same: (a)–(d) | as stated |

Hypothesis behind (d): with p = 3.5 the grey cells of the post hoc designs are penalised, so a crisper design at the
same volume should be stiffer; not established in advance.

## First result

Appended below after the first run.

## First result (IP1/IP2 run 10:26, IP3/IP4 run 10:26–10:29 +0300 by the clock; recorded 2026-11-27 10:30 +0300)

All four checks PASS on the first run; no threshold, input or assertion changed. IH fast tests unchanged (IH1 2.0e-8,
IH2 5.6e-16).

| ID | Result | Measured (threshold) |
|---|---|---|
| IP1 | PASS | worst 1.1e-8 (1e-6) |
| IP2 | PASS | x after 5 iterations bitwise equal to the 12c05ad baseline; objective bitwise; 5 iterations |
| IP3 square | PASS | (a) volume 0.49999991; (b) κ 0.183242 ≤ κ_HS 0.185185; (c) grey 0.0013 (≤ 0.05); (d) **κ/κ_HS = 0.9895** (≥ 0.85). 600 iterations (every stage at its cap of 100; KKT of the last iterate 1.8e-2 at β = 32); isotropy defect 0.67 (square symmetry, anisotropic) |
| IP4 rhombus | PASS | (a) 0.50000006 (≤ 0.5000005); (b) κ 0.182959 ≤ κ_HS; (c) grey 0.0037; (d) **κ/κ_HS = 0.9880**. 447 iterations (some stages stopped at KKT 1e-4; the last at its cap, KKT 4.4e-2); isotropy defect 3.9e-2 |

Reading: the projection raised the post hoc cells from 0.841 / 0.837 to 0.990 / 0.988 of the Hashin–Shtrikman bound,
with almost no grey (0.1–0.4 % of the cells). The hypothesis behind (d) held: with p = 3.5 the grey cells were the
loss. The bound itself is not crossed (IH5 shows the check would detect it). The last stages end at the iteration cap
with KKT 2–4e-2, typical of a steep projection; the designs are near-0/1, so the number is the value of a nearly
discrete cell, not of a grey intermediate. The start still matters (IH): these runs begin from the 0/1 void hole.
\nFigure of the projected cells, tiled 3 × 3: `fig_ip_cells.png` / `.pdf` (`scripts/ip_designs.py` re-runs IP3/IP4 deterministically — same κ/κ_HS 0.9895 / 0.9880 — and `scripts/ih_cells_figure.py ip`; datawiz audit 0 errors).
