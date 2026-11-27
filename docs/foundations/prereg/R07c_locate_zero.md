# R07c — Critical point exactly at an accepted path point (preregistration)

Preregistered 2026-11-27 09:16 +0300 (clock), before the change. Base `f6266df`. Finding
(`R07_bordered.md`): with the bordered predictor the historical A1/A2 runs pass the singular point but
`locate_critical` raises "crossing eigenvalue not bracketed (0.000e+00 at point 1, …)": the crossing eigenvalue is
exactly zero at the accepted point, so the sign test f_A f_B < 0 fails although the critical point is that point.

Change (only in the branch that raised before): when f_A f_B ≥ 0 and the eigenvalue at one endpoint satisfies the
convergence test of the bisection (|f| ≤ max(eig_tol · max(|f_A|, |f_B|), 16 eps ‖K‖₁)), that endpoint is returned
as the critical point (converged, t = 0 or 1, no extra evaluation); otherwise the old ValueError. Paths where the old
code succeeded are not affected (the branch is reached only where it raised).

| ID | Hypothesis | Threshold |
|---|---|---|
| R7c1 | the historical strict xfails test_a1 and test_a2[plus], test_a2[minus] of test_bifurcation now pass with their original assertions; if they do, their xfail markers and ledger entries are removed together (promotion with this record as evidence) | pass |
| R7c2 | the existing bifurcation, continuation and nonlinear-foundations tests pass unchanged | pass |
| R7c3 | an endpoint whose eigenvalue is not within the tolerance still raises the old ValueError (negative control) | exact |

Run: `python -m pytest tests/toporlogy/test_bifurcation.py tests/toporlogy/test_continuation_singular.py tests/toporlogy/test_nonlinear_foundations.py -q -p no:randomly --runxfail -k "a1 or a2"` then the full files.

## First result (runs 09:16–09:20 +0300 by the clock; recorded 2026-11-27 09:21 +0300)

* **R7c1 PASS**: with `--runxfail` the historical strict xfails test_a1 and test_a2[plus], test_a2[minus] pass with
  their original assertions (6 passed with their post hoc variants). **Promoted**: the two xfail markers (one is
  parametrised) and the three ledger entries removed together, with this record and `R07_bordered.md` as evidence;
  the historical A_REASON text stays in test_bifurcation.py.
* R7c2 PASS: test_bifurcation, test_continuation_singular, test_nonlinear_foundations, test_nonlinear_solvers,
  test_catalog_ledger: 127 passed, 7 xfailed (other historical entries).
* R7c3: the first version of the negative control shifted the eigenvalues by +1, which moved the second endpoint
  (−1) onto exactly zero, so the endpoint was legitimately accepted — a **test set-up error**; with a +2 shift (values
  2 and 1, both away from zero) the old ValueError is raised: PASS.

R07 (exact singular accepted point) is resolved for the predictor and the critical-point location on these models.
