# KS — KKT-residual stopping for the viscoelastic optimisations (preregistration)

Preregistered 2026-11-27 02:43 +0300 (clock), before the code change and the runs. Base: `foundations/round-3`
after the WP18 merge. Motivation (measured, VO6 / VR1 / VU4): the step-size stop (largest change < 1e-3) left
objective gaps of 0.2–0.3 %, the size of the effects compared.

Change: `optimize_viscoelastic(..., kkt_tol=None)`. With `kkt_tol`, after each evaluation the first-order
optimality of the current iterate is computed by `optimization.diagnostics.kkt_estimate` (multipliers fitted to the
iterate by NNLS on the active constraints; the min–max run on the extended variables (x, z) with its bound
constraints), on the normalised objective and constraints passed to MMA; the run stops when it is ≤ `kkt_tol`
(`converged = True`, `termination = "kkt"`). The step-size stop `tol` stays available. The result reports the last
KKT residual. Nothing else changes (the default behaviour of every existing call is unchanged).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| KS1 | default arguments reproduce the previous behaviour: a 5-iteration run with `kkt_tol=None` gives the same x as the code before the change (fixed seed-free uniform start) | the earlier commit's function, re-implemented call by call in the test (regression) | `np.array_equal` |
| KS2 | VU4 set-up with `kkt_tol = 1e-4`, max_iter 1500: the robust design's mean + 2 std ≤ that of the nominal design (+1e-4 relative) | — | as stated |
| KS3 | VR1 set-up with `kkt_tol = 1e-4`, max_iter 1500: min–max worst case ≤ (1 + 1e-3) × best single-time worst case, every run also KKT-stopped | — | as stated |
| KS4 | every KKT-stopped run of KS2/KS3 reports converged and a final residual ≤ 1e-4 or reports converged = False at max_iter (never a false converged) | — | exact |

KS2/KS3 are slow (several minutes); they are marked `slow` and run once here, the result appended below.
Run: `python -m pytest tests/toporlogy/test_kkt_stop.py -q -p no:randomly -m "slow or not slow"`.
