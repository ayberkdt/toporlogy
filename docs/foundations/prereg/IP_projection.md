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
