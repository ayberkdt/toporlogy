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

## First result (KS1 run 02:45, KS2/KS3 run 02:46–02:50 +0300 by the clock)

* KS1 PASS: with default arguments x after 5 iterations equals the function at 78bedbb bitwise. Set-up: the first
  version loaded the old source with `exec` and failed (the module must be in `sys.modules` for dataclasses); after
  that fix it passed, and the reference was then frozen as `tests/toporlogy/baseline/ks1_fingerprint_78bedbb.json` so
  the test does not need git history (a shallow CI checkout lacks 78bedbb).
* KS4 PASS (every run KKT-stopped with residual ≤ 1e-4: robust 6.9e-5 at 100 it, nominal 8.1e-5 at 143 it, min–max
  9.2e-5 at 53 it, singles at 57/76/143 it).
* **KS2 FAILED**: robust mean + 2 std 1.408537e-4 vs nominal 1.404853e-4 (+0.26 %).
* **KS3 FAILED**: min–max worst case 1.215313e-4 vs 1.211352e-4 (+0.33 %).

**This refutes the explanation written in the VU and VR records** ("the step-size stop leaves the gap"). With
first-order optimality reached to 1e-4 the gaps remain: the runs end at different stationary points of a
nonconvex problem. The objective values of the KKT-stopped runs are almost identical to the step-size-stopped ones
(1.40854e-4, 1.40485e-4), so the stop rule was not the cause. Strict xfails (slow); ledger
`ledger_additions/KS_kkt_stop.json`. What remains true: the robust and min–max formulations are implemented with
exact gradients (VR3, VU2) and give KKT-stationary designs; claims of a robust or min–max *benefit* on this problem
are not supported by these runs.
