# VR — Robust (min–max over service times) viscoelastic topology optimisation (preregistration)

Preregistered 2026-11-27 02:25 +0300 (clock), before the code change and the tests. Base `ca0c6cc`. Module
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
