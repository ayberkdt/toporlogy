# F2 / F4 — causes of the historical T4/T10 rate misses (R04) and the curved T10 patch test (R05)

Preregistration: `prereg/F4_rates.md` (with all first results and times). Tests: `tests/toporlogy/test_fem_rates_investigation.py`.
Ledger additions: `ledger_additions/F4_rates.json` (two strict xfails). No production code was changed: no defect was found.
All numbers below are from the runs of 2026-11-27 04:45-05:25 +0300 recorded in the preregistration.

## R05 — curved T10 patch test and the quadrature degree: RESOLVED CAUSE

* Derivation (prereg): u = A x + a lies in the isoparametric space on every map, so the patch can only fail through an
  inexact internal-force integral. The integrand is cof(J) grad_xi N_i, a polynomial of degree 3 on a curved T10 cell, whose
  degree-3 part is proportional to c^2 (c = mid-node displacement). Degree 2 (the production default for T10, the 4-point
  Keast rule) is one degree short; degree 3 is exact.
* F2-1 (sympy, exact rational arithmetic): degree exactly 3, every degree-3 term carries c^2; Keast cell-integral error
  1.35e-5 at c = 0.05 h and 5.40e-5 at c = 0.1 h (ratio 4.0000); degree-3 rule 3.5e-18; c = 0: 1.7e-18.
* F2-2: historical numbers reproduced (u 1.683e-4, stress 2.625e-3 at degree 2; 1.5e-15 at degree 3).
* F2-3 (5 seeds): degree 2 u error 9.5e-5 .. 1.7e-4, degrees 3, 4, 6, 8: u <= 4.8e-15, stress <= 2.0e-13. The patch test is exact
  to rounding from degree 3 up, on curved cells. It does not stay exact for non-constant stress fields (rational integrand;
  not claimed or tested).
* F2-4: degree-2 u error 1.2e-15 (c = 0), 9.95e-6, 4.08e-5, 1.68e-4 for c = 0, 0.025, 0.05, 0.1 h: quadratic in c (ratios 4.10, 4.12).
* Consequence: the 1.7e-4 of G2c is the missing degree, not a kernel defect. For curved T10 meshes use `degree=3` (G2b does).
  The default `stiffness_degree` is exact on affine tets and was left as is.
* Set-up notes: at curved 0.1 h with vertices 0.2 h, 12 of the first 17 seeds give an inverted cell at some tested rule degree;
  the accepted seeds are 3, 7, 13, 17, 19. This mesh family is close to the validity edge.

## R04 — T4 L2 rate 1.796 and T10 energy rate 1.788: STILL PRE-ASYMPTOTIC (no defect), per cause

Reproduced bitwise first (F4-0): T4 (4, 8, 16) L2 1.795927, energy 0.919555; T10 (2, 4, 8) L2 2.922593, energy 1.787746; two runs equal.

| Cause | Measurement | Result |
|---|---|---|
| (a) reference / error-integral | F4-5: Manufactured u, grad, f vs sympy; independent dense error integral vs `errors(degree=24)` | agree to 1.9e-15 / 1.7e-15 / 5.1e-15 and <= 8.4e-15: not the cause |
| (c) quadrature of the error norm and of the load | F4-2: error degree +6, +12, load degree +6 | pair rate changes <= 6.7e-8 (T4), 1.1e-6 (T10); the gaps to 1.8 are 0.004 and 0.012: not the cause |
| (b) mesh draw | F4-3: seeds 200..500 and an undistorted mesh | T4 L2 p(8,16): 1.796 (historical), 1.818, 1.804, 1.816, 1.812; undistorted 1.860. T10 energy p(4,8): 1.788 (hist.), 1.800, 1.786, 1.814, 1.782; undistorted 1.825 |
| (d) pre-asymptotic | F4-1 interpolation without a solve; F4-4 further solved levels | see below |

* T4 L2: the historical draw is the lowest of five draws (1.796 to 1.818, spread 0.022); the threshold 1.8 lies inside that
  spread, so the miss by 0.004 is a mesh-draw accident on top of a rate that is still rising: solved p(16,24) = 1.9229
  (historical 1.92), interpolation 1.918, 1.978, 1.991 for (8,16), (16,24), (24,32). F4-1 (i) failed (interpolation 1.918 vs Galerkin
  1.796, difference 0.122 > 0.10): the solved L2 error is further from its asymptote than the interpolation error at n = 8-16.
* T10 energy: 4 of 5 draws are below 1.8 (1 of the 4 new ones passes: F4-3 failed), the undistorted mesh gives 1.825, and the
  interpolation energy rate is 1.814, 1.899, 1.949, 1.961 for (4,8), (8,12), (12,16), (16,24); solved p(8,12) = 1.8888 (historical 1.89;
  interpolation 1.899). The T10 miss is a genuine pre-asymptotic rate of the P2 energy error at n = 4-8, not a draw accident.
* Not resolvable here: solved levels T4 n = 32 and T10 n = 16 (peak working set 3.1 and 3.3 GB at 47 k DOF with SuperLU, 2.9 GB
  free). The asymptote is therefore supported by the solved pairs up to 24 / 12 and by the interpolation sequence up to 32 / 24 only.
* The historical xfails and thresholds are unchanged. Recommendation (not applied): these two V3 checks need one more level or
  a draw-averaged rate to be meaningful at threshold p - 0.2.
