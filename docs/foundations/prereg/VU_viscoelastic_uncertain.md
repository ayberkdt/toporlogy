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

## First result (run 2026-11-27 02:38–02:42 +0300 by the clock)

5 passed on the first run (VU1–VU5). VU4 values over the 9 samples, J at t_s = 100 s:

| design | mean | std | mean + 2 std | iterations (stop) |
|---|---|---|---|---|
| robust (objective mean + 2 std) | 1.202647e-4 | 1.029451e-5 | 1.40854e-4 | 101 (tol 1e-3) |
| nominal sample (1.0, 1.0) | 1.198694e-4 | 1.030795e-5 | **1.40485e-4** | 134 (tol 1e-3) |

**The pass is weak and the robust run is not optimal at its own objective**: its std is only 0.13 % smaller, and the
nominal design has the lower mean + 2 std (by 0.26 %). Together with VO6 (non-convergence at 40 iterations) and VR1
(min–max bound formulation 0.28 % worse), this is the third measurement where the step-size stop of MMA (largest
change < 1e-3) leaves a ~0.3 % objective gap, the same size as the effects measured. Conclusion recorded for the
optimisation side: on this problem class the spectral uncertainty of the bulk relaxation changes J(t_s) by ±8.6 %
(std/mean) almost independently of the topology, so topology optimisation can do little against it; and design
comparisons at the 0.1–1 % level need a KKT-residual stop and multiple starts before any claim. Follow-up: KKT
residual stopping criterion for MMA (the optimiser module), multi-start comparisons.

## Correction (2026-11-27, after KS2/KS3)

The explanation above that the step-size stop causes the objective gap is refuted by `prereg/KS_kkt_stop.md`: with
KKT-stopped runs (residual ≤ 1e-4) the same gaps remain; the runs end at different stationary points.
