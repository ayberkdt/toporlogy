# VM — Min–max over a load history whose critical time depends on the design (preregistration)

Preregistered 2026-11-27 02:54 +0300 (clock), before the test exists. Base `919586f`. No production change planned
(uses `optimize_viscoelastic(steps=..., kkt_tol=...)`). Motivation: VR1/KS3 showed that with one constant load the
min–max over time collapses to the last time; here two loads act at different times, so each single-time design
neglects one load.

Set-up: cantilever 2:1, 40 × 20 Q4 (h = 0.05), left edge clamped, VO4 non-proportional material (synthetic).
Load case A: vertical point load at the bottom of the free end, F_A = −1, on t ∈ [0, 10 s]; then removed (linear
ramp to zero between 10 s and 10.5 s); load case B: horizontal point load at the mid-height of the free end,
F_B = +1, ramped in over 10 s → 10.5 s and held to 100 s. Steps: 0⁺ (instant), log-spaced to 10 s, 10.5 s,
log-spaced to 100 s. Objectives J_A = f(10 s)ᵀu(10 s), J_B = f(100 s)ᵀu(100 s). V = 0.4, r_min = 1.5 h, KKT stop 1e-4,
max_iter 1500, uniform start.

| ID | Hypothesis | Threshold |
|---|---|---|
| VM1 | the min–max design has max(J_A, J_B) at least 5 % lower than each of the two single-objective designs (A-only, B-only) | ≤ 0.95 × min(max over the singles) |
| VM2 | at the min–max design both bounds are nearly active: J_A and J_B within 2 % of each other (the trade-off is balanced) | |J_A − J_B| / max ≤ 0.02 |
| VM3 | every run reports KKT residual ≤ 1e-4 or converged = False at max_iter | exact |

Run: `python -m pytest tests/toporlogy/test_viscoelastic_minmax_switch.py -q -p no:randomly -m "slow or not slow"`
(slow). First result appended below.
