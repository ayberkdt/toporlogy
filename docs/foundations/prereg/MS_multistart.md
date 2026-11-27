# MS — Multi-start for the time-aware topology optimisation (preregistration)

Preregistered 2026-11-27 06:23 +0300 (clock), before the code and the tests. Base `f40acce`.
Motivation (measured): KS2/KS3, VR1, VU4 and the VM reproducibility note show that MMA from the uniform start ends at
different KKT-stationary points; a warm start from another design reached a better one (KS post hoc).

Change: `optimization.viscoelastic_topology.multistart(structure, starts, **kw)` runs `optimize_viscoelastic` from each
start in `starts` (x0 arrays or None = uniform) with the same keyword arguments, and returns the best result by the
returned objective among the runs that are feasible (volume ≤ V (1 + 1e-6)), plus all runs' (objective, KKT,
termination). A helper `perturbed_starts(n, volume, n_cells, amplitude, seed)` gives volume-preserving seeded random
starts. Deterministic for fixed inputs.

| ID | Hypothesis | Threshold |
|---|---|---|
| MS1 | the returned objective equals the minimum over the feasible runs and the result is the run that attained it | exact |
| MS2 | determinism: two calls with the same inputs give bitwise equal designs | `array_equal` |
| MS3 | VU set-up (9 samples, mean + 2 std, KKT stop 1e-4): starts = [uniform, nominal design, 2 perturbed (seed 0, amplitude 0.1)]: the multi-start robust design's mean + 2 std ≤ that of the nominal design | ≤ (no worse) |
| MS4 | perturbed starts keep the mean equal to the volume fraction (to 1e-12) and lie in [0, 1] | as stated |

Run: `python -m pytest tests/toporlogy/test_multistart.py -q -p no:randomly -m "slow or not slow"` (MS3 slow).

## First result (run 06:24–06:41 +0300 by the clock; recorded 2026-11-27 06:42 +0300)

3 passed on the first run (MS1, MS2, MS4 quick; MS3 17 min). MS3 runs (objective = robust mean + 2 std, KKT, stop):
uniform start 1.408537e-4 (6.9e-5, kkt); nominal-design start 1.404645e-4 (3.4e-5); perturbed seed-0 starts
1.404645e-4 (3.9e-5) and 1.404645e-4 (9.5e-5). Best (index 3) 1.404645e-4 ≤ nominal design 1.404853e-4 (−0.015 %).
Only the uniform start ends in the worse basin; three of four starts find the better one. (The test file was swept
into the C08 integration commit 3393137 by a broad `git add`; its content is the one run here.)
