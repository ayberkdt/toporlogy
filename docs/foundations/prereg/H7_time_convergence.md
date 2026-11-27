# H7 — Time-step convergence of the FE history solver (WP17 H6; preregistration)

Preregistered 2026-11-27 06:21 +0300 (clock), before the test exists. Base `6385cfe`. No
production change.

Set-up: `HistoryProblem` on a Q4 cantilever 8 × 4 (2 × 1, plane strain, left edge clamped), tip load applied
instantly at t = 0 (a zero-length step) and held to T = 100 s with N uniform steps, N ∈ {10, 20, 40, 80, 160}.
Quantity: tip vertical displacement u(T). Reference: N = 1280 (same solver; the observed order uses differences
e_N = |u_N − u_ref|, and the reference error is ≤ e_160 / 8 under the hypothesised orders — stated, not measured).
Asymptotic window: the last three levels (40, 80, 160), order p = log2(e_40/e_80) and log2(e_80/e_160).

| ID | Model (synthetic) | Hypothesis | Threshold |
|---|---|---|---|
| H7a | `NortonCreep` (K = 3.33 GPa, G = 0.77 GPa, A = 1e-4 1/s, σ_ref = 10 MPa, n = 3), tip load giving bending stresses of order 10 MPa | backward Euler: first order | both window orders in [0.9, 1.1] |
| H7b | `GeneralizedMaxwell` (WP23 V set) | the exact-for-linear-strain recursion: second order in the strain path | both window orders in [1.8, 2.2] |
| H7c | consistency: u_N of both models converges monotonically (e_10 > e_20 > … > e_160) | — | strict |

Run: `python -m pytest tests/toporlogy/test_history_time_convergence.py -q -p no:randomly`.

## First result (run 06:21–06:22 +0300 by the clock; recorded 2026-11-27 06:23 +0300)

2 passed on the first run.

| model | e_10 | e_20 | e_40 | e_80 | e_160 | orders (10→20 … 80→160) |
|---|---|---|---|---|---|---|
| Norton (backward Euler) | 5.17e-8 | 2.57e-8 | 1.27e-8 | 6.13e-9 | 2.86e-9 | 1.008, 1.022, **1.047, 1.099** |
| generalized Maxwell | 1.95e-6 | 7.34e-7 | 2.25e-7 | 5.99e-8 | 1.51e-8 | 1.41, 1.71, **1.907, 1.991** |

Both window orders inside the preregistered bands; errors decrease monotonically (H7c). The last Norton order (1.099)
sits near the band edge, consistent with the reference error (N = 1280) inflating the finest difference; the
generalized Maxwell order rises from 1.41 to 1.99 (pre-asymptotic start after the instantaneous load).
