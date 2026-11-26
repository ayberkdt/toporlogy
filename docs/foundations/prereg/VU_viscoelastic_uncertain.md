# VU — Viscoelastic topology optimisation under uncertain relaxation spectra (preregistration)

Preregistered 2026-11-27 02:37 +0300 (clock), before the code and the tests. Base `6547332`. Module
`optimization/viscoelastic_topology.py`: new `RobustViscoelastic(structures, weights)` (a set of structures that
share mesh, loads, supports and time grid and differ only in the material sample) with
`robust_objective(w, step, kappa)` = mean_s J_s + κ · std_s J_s (weighted, population std), and `optimize_viscoelastic`
accepting it. The samples are deterministic (a quadrature/sample set given by the caller), not random draws inside
the optimiser.

Uncertainty model for the tests (synthetic): the bulk relaxation amplitude K1 and the bulk time τ_K1 of the VO4
non-proportional material scaled by factors (0.7, 1.0, 1.3) × (0.5, 1, 2) — a 3 × 3 tensor grid with equal weights.

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| VU1 | the robust objective equals the mean and std computed in the test from separate single-sample evaluations | separate evaluations (same kernel: consistency) | rel 1e-13 |
| VU2 | its gradient (autograd through all samples) equals a central FD in a random direction on the 8 × 4 cantilever | FD, steps 1e-2…1e-7, best | rel ≤ 1e-6 |
| VU3 | κ = 0 and one sample reduce exactly to the single-structure objective and gradient | single-structure path | bitwise (`==` on the value, `array_equal` on the gradient) |
| VU4 | MBB 30 × 10, V = 0.5, t_s = 100 s, tol 1e-3, ≤ 300 iterations: the design optimised for mean + 2 std has a sample std of J(t_s) not larger than that of the nominal-sample design (both evaluated on the 9 samples) | — (measurement with a stated hypothesis) | std_robust ≤ std_nominal |
| VU5 | invalid inputs: structures with different meshes, time grids, loads or supports, negative weights, weights not summing to 1, κ < 0 → ValueError | — | exact |

Run: `python -m pytest tests/toporlogy/test_viscoelastic_uncertain.py -q -p no:randomly`. First result appended below.
