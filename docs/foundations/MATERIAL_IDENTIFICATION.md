# Material identification from test data (WP20 D03/D05)

Module `src/toporlogy/optimization/material_identification.py`, tests `tests/toporlogy/test_material_identification.py`,
record `docs/foundations/prereg/D_identification.md`. It extends `optimization/identification.py` (`Observations`,
`identify`, `weighted_rms`). **Every number in the tests is synthetic. No real material data exists in this repository
slice; real-data validation (WP23 D8) is NOT VERIFIED.**

## Methods

| Problem | Reference fit | Nonlinear fit |
|---|---|---|
| Relaxation M(t) = M∞ + Σ A_i e^{−t/τ_i}, Pa, s | fixed log-spaced τ: non-negative least squares on (X x − y)/σ (`scipy.optimize.nnls`), σ_i = rel_sigma · y_i | θ = (M∞, A_i, ln τ_i), analytic Jacobian, `identify` from a stated list of start sets (amplitudes of a start = NNLS at its τ), lowest cost wins, τ sorted |
| Norton ṗ = A f(T)(σ/σ_ref)^n, f = exp[−Q/R (1/T − 1/T_ref)] | ordinary least squares of ln ṗ on (1, ln σ/σ_ref, −(1/T − 1/T_ref)/R): exact for noise-free data | `identify` on ṗ (σ_i = rel_sigma · ṗ_i), analytic Jacobian, default start = the log-linear solution |

A is the rate at σ_ref and T_ref (user-supplied, recorded in the fit). Time in s, stress / modulus in Pa, T in K, rate in
1/s, Q in J/mol (R = 8.314462618, as `core/creep.py`).

## Identifiability

`identifiability_report` takes the weighted Jacobian (∂(model/σ)/∂θ), scales the columns by typical parameter sizes
(Prony: the largest modulus; Norton: 1, 1, R·T_ref so that Q' = Q/(R T_ref) is dimensionless) and reports

* singular values, condition number (inf when the smallest is 0), rank at `RANK_RTOL` = 1e-8;
* the **weakest combination**: the right singular vector of the smallest singular value (unit norm in the scaled
  coordinates, largest component positive), named per parameter, and a one-line `describe()`;
* `std_nominal` (assumes the stated σ) and `std_residual` (× the residual standard deviation, only for m > p), both in
  parameter units; both are `None` when the Jacobian is rank deficient at `RANK_RTOL`: no number is better than a wrong
  number;
* `practically_unidentifiable`: condition > `COND_LIMIT` = 1e4 (an arbitrary but stated flag).

The report uses the full design, not the active set of the sign constraint, so a degeneracy cannot hide behind an NNLS
zero. It is local and linearised: it assumes the model and the noise level are right and says nothing about model error.
A fit with `identifiable == False` returns numbers (NNLS or minimum-norm), but no recovery is claimed.

Verified degeneracies (synthetic): relaxation data only on 1e-3 … 1e-1 s with branches at 10 s and 1000 s (condition
8.0e8; weakest combination ≈ +0.71 A_1000 − 0.70 M∞); Norton data at one temperature (exactly rank 2 of 3; weakest
combination = the derived (g, 0, −1)/√(1+g²) mixing ln A and Q'). The same designs with well-spaced times (condition 21)
or two temperatures are identifiable. A derivation that failed: the preregistered expectation of a finite nominal std for
the first case (the covariance is undefined at that rank; xfail with ledger entry).

## Data records and provenance

`MaterialDataRecord` carries the arrays, one unit string per array (known units in `UNIT_FACTORS`; degC is refused),
`source` (`synthetic` / `measured`), a non-empty `origin`, a `license` (required for measured data), a sha256 checksum of the
arrays (serialisation documented in `_checksum`) and a `ValidityBox` (default: the data range of time, or of stress and
temperature). The fits accept SI records only; other units must go through the explicit `convert_to_si`. A fit keeps the
record's box and checksum, and `predict` / `predict_rate` raise `ValidityBoxError` outside the closed box (no
extrapolation). Missing provenance, unknown or wrong-kind units, non-finite / non-positive / empty / mismatched arrays, a
wrong checksum and a box that does not contain the data all raise `ValueError`.

Negative control: the same Norton rates written per hour and labelled "1/s" fit with A larger by exactly 3600 (n and Q
unchanged); labelled "1/h" they are refused until converted.

## What is verified (synthetic only)

Noise-free recovery of Prony amplitudes (fixed true τ) and of Norton A, n, Q (log-linear and nonlinear) to ≤ 1e-13
relative; free-time recovery from three stated start sets; held-out prediction at times / conditions inside the box;
1 % multiplicative noise: the true parameters inside 3 linearised standard deviations (seed 7, z within 0.6), the
standard deviation equals a dense numpy reference, held-out error 0.2 %. Not verified: any real material, correlated or
non-Gaussian noise, model discrepancy, profile likelihood or bootstrap, joint creep + relaxation fits, WLF / cure kinetics,
Stats-Lab adapters.
