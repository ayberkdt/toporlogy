# VR — Robust (min–max over service times) viscoelastic topology optimisation (preregistration)

Preregistered 2026-11-27 02:15 +0300 (clock; the first committed text said "02:25", typed before the clock output was read — corrected, the commit 56e23ab time is the evidence), before the code change and the tests. Base `ca0c6cc`. Module
`optimization/viscoelastic_topology.py` (`optimize_viscoelastic` gains `steps=` for several service times with
`objective="max"`, and a convergence stop `tol`). Follows the VO post hoc finding that 40 fixed iterations are not
converged.

## Formulation

min_x max_k J_k(x), J_k = f(t_k)ᵀ u(t_k), k over the chosen steps, as the bound formulation
min_{x, z} z subject to J_k(x)/c_ref − z ≤ 0 (k = 1…m), z ∈ [0, 1] an extra MMA variable,
c_ref = 1.5 · max_k J_k(x0), volume ≤ V. One forward time march gives every J_k; their gradients come from one
reverse pass each through the same graph. Stop when the largest change of (x, z) is < `tol` or at `max_iter`;
the result reports `converged`.

## Checks

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| VR1 | MBB half-beam 30 × 10, VO4 non-proportional material, steps t ∈ {0⁺, 10 s, 100 s}, V = 0.5, r_min = 1.5, tol = 1e-3, max_iter = 300: worst case of the min–max design ≤ the worst case of each of the three single-time designs (each run with the same stop rule), up to a relative 1e-3 | the three single-time runs | max_k J_k(minmax) ≤ (1 + 1e-3) · min_s max_k J_k(single_s) |
| VR2 | at the returned min–max design the bound is active: |z − max_k J_k/c_ref| ≤ 1e-3 · z | — | as stated |
| VR3 | gradients of the m objectives from one forward pass (retain_graph) equal those of m separate single-objective evaluations | separate evaluations | rel 1e-12 |
| VR4 | `tol` stop: a run with tol = 1e-3 reports converged = True with its last change < 1e-3 (if it converged before max_iter); a run with max_iter = 3 and tol = 0 reports converged = False | — | exact |

Run: `python -m pytest tests/toporlogy/test_viscoelastic_robust.py -q -p no:randomly`. First result appended below.

## First result (runs 2026-11-27 02:17 and 02:21 +0300 by the clock)

* First run (02:17): VR3, VR4 passed; VR1 and VR2 errored in the fixture — **test set-up error**: the time grid
  `logspace(-1, 2, 12)` has no point at 10 s (nearest 8.11 s), while VR1 names t = 10 s. Grid changed to
  `logspace(-1, 2, 13)` (10 s and 1 s on the grid); no threshold changed.
* Valid run (02:21): VR2, VR3, VR4 PASS. **VR1 FAILED**: worst case (max over 0⁺, 10 s, 100 s) of the min–max design
  1.21480e-4; single-time designs 1.21382e-4 (0⁺), 1.21229e-4 (10 s), 1.21135e-4 (100 s); min–max +0.28 % over the
  best. Iterations: min–max 95 (converged), singles 57, 76, 134 (all converged at tol 1e-3); volumes 0.5 to 3e-9.
  Explanation (measured, not a fix): for a constant load the compliance grows with time, so the worst case is always
  t = 100 s and the min–max problem reduces to minimising J(100 s); the bound formulation stopped at a worse local
  point. Strict xfail with these values; ledger `ledger_additions/VR_viscoelastic_robust.json`.
* Follow-up (not done): a min–max study with time-varying loads or several load cases, where the active time
  changes with the design; a KKT-residual stop instead of the step-size stop.
