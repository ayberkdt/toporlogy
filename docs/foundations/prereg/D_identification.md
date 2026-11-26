# D_identification — inverse identification of material parameters from synthetic data (preregistration)

Preregistered 2026-11-27 02:17 +0300 (clock), base SHA `79623a7`, before any new test below was run. Module
`optimization/material_identification.py` (new; it extends `optimization/identification.py` — `Observations`, `identify`,
`weighted_rms`, `verify_model_jacobian` are reused, not copied; `core/creep.py::ArrheniusFactor` is used for the Arrhenius
factor of the nonlinear Norton model). Tests `tests/toporlogy/test_material_identification.py`. Plan: WP20 D03/D05,
WP23 D4–D7. D8 (real data) is **NOT VERIFIED**: there are no licensed measurements; nothing here is material data and
every number below is synthetic.

## Models and conventions (derived here, used by the expected values)

* Relaxation: M(t) = M∞ + Σ_i A_i e^{−t/τ_i}, Pa, t in s (the series of `core/viscoelastic.PronySeries`). Weights
  σ_i = rel_sigma · y_i (default 0.01). With fixed τ the fit is the non-negative least-squares problem
  min ‖(X x − y)/σ‖, x ≥ 0, X = [1, e^{−t/τ_1}, …] (`scipy.optimize.nnls`). Free times: θ = (M∞, A_i, ln τ_i), analytic
  Jacobian (∂/∂ln τ_i = A_i (t/τ_i) e^{−t/τ_i}), `identify`, deterministic list of start sets, best cost wins, times
  returned sorted.
* Norton (uniaxial): ṗ = A f(T) (σ/σ_ref)^n, f = exp[−Q/R (1/T − 1/T_ref)] (as `core/creep.py`). In the log domain
  ln ṗ = ln A + n ln(σ/σ_ref) + Q' g(T), Q' = Q/(R T_ref), g(T) = −(T_ref/T − 1) = 1 − T_ref/T: **linear** in
  β = (ln A, n, Q'). Log-linear least squares is the reference fit; the nonlinear fit uses `identify` on ṗ with
  σ_i = rel_sigma ṗ_i.
* Identifiability report: weighted Jacobian J_q = (∂m/∂θ_j) scale_j / σ_i; SVD; condition = s_max/s_min (inf when
  s_min = 0); rank at 1e-8 (`RANK_RTOL`); weakest combination = right singular vector of s_min, unit norm in the scaled
  coordinates q = θ/scale; `std_nominal` = sqrt(diag (J_qᵀJ_q)⁻¹) · scale (assumes the stated σ), `std_residual` the same
  times the residual standard deviation s = sqrt(RSS_w/(m − p)) (None when m ≤ p); both None when rank deficient.
  `practically_unidentifiable` = condition > `COND_LIMIT` = 1e4. The report is computed on the **full** design (all
  columns), not on the active set of the sign constraint, so a degeneracy cannot hide behind an NNLS zero.
* Data record: arrays, one unit string per array, source ∈ {synthetic, measured}, origin (non-empty text), license
  (required for measured), validity box (default: data range per box-able array: time, stress, temperature),
  sha256 checksum of the arrays (names sorted; per array: name UTF-8, 0x00, dtype string, shape repr, 0x00, C-order
  bytes). Fits require SI units (s, Pa, K, 1/s); other known units are converted only by the explicit `convert_to_si`.
  Prediction outside the box raises `ValidityBoxError` (a ValueError). Temperature unit: kelvin only.

## Checks

Common synthetic truth (written in the test with plain numpy, not with the production code): relaxation M∞ = 2.0e8 Pa,
A = (1.5e9, 8.0e8, 4.0e8) Pa, τ = (0.1, 10, 1000) s, times `logspace(-3, 5, 81)` s. Norton A = 2.0e-9 1/s at
σ_ref = 75 MPa, T_ref = 600 K, n = 4.5, Q = 2.5e5 J/mol; stresses {50, 75, 100} MPa × temperatures {560, 600, 640} K
(9 points, stress-major order).

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| D4a | NNLS with the true τ fixed recovers M∞ and the three amplitudes from noise-free data | generating parameters | max relative error ≤ 1e-8 |
| D4b | held-out prediction at 37 other times `logspace(-2.7, 4.7, 37)` (inside the box) | closed-form sum of exponentials | relative ≤ 1e-8 |
| D4c | free-time fit from the three start sets τ0 ∈ {(0.03, 3, 300), (0.3, 30, 3000), (0.03, 30, 300)} recovers τ and amplitudes (noise-free); held-out prediction | generating parameters / closed form | τ, amplitudes, M∞ relative ≤ 1e-5; held-out relative ≤ 1e-6 |
| D4d | contrast for D5: the report of the fixed-τ fit on the full-range data is full rank with condition < 1e3 | — | condition < 1e3, rank 4 of 4, not flagged |
| D4e | analytic Jacobian of the free-time model equals central differences (`verify_model_jacobian`) | central differences | ≤ 1e-6 |
| D5a | Prony data restricted to t ∈ [1e-3, 1e-1] s (31 log-spaced points), fixed τ = (0.1, 10, 1000): the columns 1, e^{−t/10}, e^{−t/1000} agree up to a t² term of size ≈ 5e-3 t² ≤ 5e-5 and, with the fast column cancelling the t² content, up to t³ terms ≲ 1e-3·t³; so s_min ≲ 1e-5 s_max-scale and condition ≳ 1e5. Taylor solution of the near-null vector (orders 0–2 cancelled): fast-branch weight ~1e-5, others ≈ (equilibrium −0.67, A_10 −0.07, A_1000 +0.74). No recovery is claimed. | — | condition > 1e4 and flagged `practically_unidentifiable`; squared weight of the weakest vector on {equilibrium, A_10, A_1000} ≥ 0.99, \|component A_1000\| ≥ 0.5, \|component equilibrium\| ≥ 0.4; `describe()` names these parameters; `std_nominal` of A_1000 (rel_sigma 0.01) exceeds the amplitude 4.0e8 Pa (ratio > 1) |
| D5b | Norton data at one temperature T = 640 K, three stresses: the Q' column is the constant g = 1 − 600/640 = 0.0625 times the ln A column, so J v = 0 for v = (g, 0, −1)/√(1+g²) in the order (ln A, n, Q'): exactly rank 2 of 3, covariance None, `identifiable` False | analytic null vector | condition > 1e12 (or inf); rank 2; weakest vector equals ±v to 1e-8 componentwise; with two temperatures the same data is full rank |
| D6a | log-linear fit of noise-free Norton data recovers A, n, Q | generating parameters | relative ≤ 1e-8 each |
| D6b | nonlinear fit through `identify`, start (3A, n + 0.7, 0.6 Q), noise-free | generating parameters | relative ≤ 1e-8 each |
| D6c | 1 % multiplicative noise ṗ_obs = ṗ (1 + 0.01 z), z = `default_rng(7).standard_normal(9)`: the true (ln A, n, Q) lie within 3 `std_residual`; `std_residual` equals an independent dense numpy s²(XᵀX)⁻¹ in the test; held-out points (62.5 MPa, 580 K) and (90 MPa, 620 K) inside the box: relative prediction error vs the noise-free law ≤ 5 % (the measured values are reported) | numpy lstsq + covariance written in the test; noise-free closed form | 3 std; std equality 1e-8 relative; 5 % |
| D6d | the nonlinear noisy estimate (same data, start from the log-linear estimate) differs from the log-linear one by ≤ 0.1 `std_residual` in (ln A, n, Q) (the two estimators differ at second order in the noise) | log-linear fit | 0.1 std |
| D6e | unit negative control: the data in 1/h (rate numerically 3600× larger) with the unit label "1/h" is refused by the fit; `convert_to_si` makes the fit exact; the same numbers labelled "1/s" fit with ln A shifted by ln 3600, n and Q unchanged (the unit check is what catches it) | derivation: ṗ_h = 3600 ṗ_s ⇒ ln A' = ln A + ln 3600 | A' = 3600 A and n, Q to 1e-8 relative |
| D7a | unit errors raise ValueError: unknown unit, unit of the wrong kind (stress labelled "s"), a missing unit, "degC" | — | exact |
| D7b | provenance errors raise: empty origin, source not in {synthetic, measured}, measured without license | — | exact |
| D7c | non-finite, non-positive, empty or mismatched-length arrays raise; a given checksum that does not match raises; an explicit box that does not contain the data raises | — | exact |
| D7d | prediction outside the box (t = 1e6 s; stress 200 MPa; T = 700 K; non-finite input) raises `ValidityBoxError`/ValueError, the box edges themselves are allowed | — | exact |
| D7e | checksum equals a hash computed independently in the test from the documented serialisation, is stable, and changes when one element changes by one ulp | hashlib in the test | exact |
| D8 | held-out independent experiment with licensed real data | — | **NOT VERIFIED**: no licensed data; recorded as a skipped test with that reason |

Note on D5a: the fixed-τ NNLS fit of rank-deficient data may return any non-negative solution; the check
asserts only the report, not a recovered value.

Negative controls: D5 (the degenerate designs are the control for D4), D6e, D7. The noise seed (7) and the 5 % / 3-std
thresholds are fixed here and not tuned. A noise draw that puts a true value outside 3 std is a legitimate (≈ 5 %) outcome
and would be recorded as a failed check, not re-seeded.

Run: `python -m pytest tests/toporlogy/test_material_identification.py -q -p no:randomly -p no:cacheprovider` with
TMP/TEMP and `--basetemp` under `.claude/tmp-toporlogy/D_identification`. Resource cap: seconds, < 100 MB.

First result: appended below after the first run.


## First result (appended 2026-11-27 02:25 +0300 by the clock; the runs were made between the prereg commit 02da4c4 at 02:17 and 02:25, the run time itself was not read from the clock)

Command: `python -m pytest tests/toporlogy/test_material_identification.py -q -p no:randomly -p no:cacheprovider -s`.
Final state: 28 passed, 1 skipped (D8), 1 xfailed (strict). Measured values (a scratch script re-evaluated the fits;
`ruff` and `mypy` clean on the module):

| ID | First result | Measured |
|---|---|---|
| D4a | PASS | max relative error 7.8e-16 (≤ 1e-8) |
| D4b | **set-up error on the first run, then PASS** | see "Set-up fixes"; held-out relative error 4.4e-16 (≤ 1e-8) |
| D4c | PASS | τ 2.2e-16, amplitudes 2.2e-16, M∞ 0, held-out 3.3e-16; 3 of 3 starts agree; weighted rms 8.9e-15 |
| D4d | PASS | condition 20.7, rank 4 of 4 (< 1e3) |
| D4e | PASS | analytic vs central-difference Jacobian ≤ 1e-6 |
| D5a (condition, flags, weakest combination, describe) | PASS | condition 8.0e8 (> 1e4), rank 3 of 4 at 1e-8; weakest vector (equilibrium, A_0.1, A_10, A_1000) = (−0.703, 1.1e-6, −0.007, +0.711): squared weight on the slow parameters 0.99995, \|A_1000\| 0.711, \|equilibrium\| 0.703; `describe()` names A_1000 and equilibrium (the third prereg'd component, A_10, is 0.007, below the 0.05 print threshold; the check asserted only the two dominant names) |
| D5a (std of A_1000 above the amplitude) | **FAIL → xfail(strict)** | `std_nominal` is None: the condition 8.0e8 puts the smallest singular value below `RANK_RTOL` · s_max, so by design (as in `identify`) no covariance is returned. The prereg'd estimate "condition ≳ 1e5" was a lower bound and the prediction that a finite std exists was wrong. Post hoc (not preregistered, labelled in the test): with `rank_rtol=1e-12` the same design gives std/amplitude = 3.5e6. Ledger entry in `ledger_additions/D_identification.json` |
| D5b | PASS | rank 2 of 3, condition 2.3e16, weakest vector (−0.0624, −1.1e-16, +0.9981) = −v of the derived v = (g, 0, −1)/√(1+g²), g = 0.0625 (match ≤ 1e-8); std None; two temperatures: rank 3, identifiable |
| D6a | PASS | relative errors A 2.0e-14, n 7.8e-16, Q 2.2e-16 |
| D6b | PASS | A 1.8e-15, n 2.2e-16, Q 2.2e-16 from the start (3A, n+0.7, 0.6Q); 10 evaluations |
| D6c | PASS | seed 7: estimate (ln A, n, Q) = (−20.0317, 4.50264, 249898) vs truth (−20.0301, 4.5, 250000), std_residual (2.7e-3, 9.3e-3, 242), z = (−0.57, 0.28, −0.42); std equals the dense numpy reference to 1e-8; held-out relative prediction errors −1.3e-3 and −1.7e-3 (≤ 5 %) |
| D6d | PASS | \|nonlinear − log-linear\| / std = (0.025, 0.013, 0.0009) (≤ 0.1) |
| D6e | PASS | unit "1/h" refused ("convert_to_si"); converted data: A exact; mislabelled "1/s": A/A_true = 3600.0000000000327, n, Q unchanged to 1e-14 |
| D7a–D7f | PASS | all invalid inputs raise (ValueError / ValidityBoxError); checksum equals the independent hash |
| D8 | NOT VERIFIED | skipped test with that reason; no licensed data |

### Set-up fixes (no threshold changed)

* D4b/D4c held-out times: the preregistered `logspace(-2.7, 4.7, 37)` has 3 points (k = 0, 18, 36) that coincide exactly with grid points
  (the first runs failed my own "not in the grid" assertion; removing the two end points was not enough, k = 18 is exactly
  10 s). The held-out set became the 80 geometric midpoints of the training grid, `sqrt(t_i t_{i+1})`. The thresholds
  (1e-8 fixed, 1e-6 free) are unchanged.
* The module uses `R_GAS` from `core/creep.py` and writes the log of the Arrhenius law directly (so a negative noisy Q
  can still be predicted); `ArrheniusFactor` enters as a labelled consistency cross-check in the tests, not in the fit
  (the prereg text said the nonlinear model uses it; that was a design statement, not a check).
* D5a `describe()` threshold: the report prints components ≥ 0.05; see the table.

Follow-up (separate preregistration): D8 with licensed data; profile likelihood / bootstrap; joint creep + relaxation fits.
