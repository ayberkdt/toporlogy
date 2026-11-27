# Elastic conditioning: what float64 can and cannot say about a stiffness tensor (E3, E6, R02)

Record: `docs/foundations/prereg/E3_conditioning.md` (preregistered before the run, first results appended).
Tests: `tests/toporlogy/test_elastic_conditioning.py`. Raw data: `docs/foundations/E3_conditioning.json`
(`python docs/foundations/scripts/e3_conditioning.py`). Everything here is **synthetic** (seeded random orthogonal Q,
Mandel spectra geometric from 1 to kappa): no material data, no physical claim.

Reference: mpmath, 50 digits (100 for R02), on the **float64 inputs exactly**. kappa is the Mandel condition number
lambda_max / lambda_min; u = 2^-53 = 1.1e-16. "Error" is the largest over the sampled quantities (20 random strains,
20 random directions, 20 random direction pairs, all 6 engineering nu_ij) of the relative error of the float64 result.

## Accuracy regime table

| kappa | kappa u | S = C^-1 (normwise) | E(n) | nu(n,m) | energy, soft strain | `check_positive_definite` | `forward_accuracy_established` | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.1e-16 | 3e-31 | 3.7e-16 | 8.6e-17 | 7.8e-17 | accepts | established | asserted <= 100 kappa u |
| 1e2 | 1.1e-14 | 1.0e-15 | 1.2e-15 | 1.6e-15 | 2.3e-16 | accepts | established | asserted |
| 1e6 | 1.1e-10 | 1.3e-11 | 1.3e-11 | 5.3e-12 | 7.4e-12 | accepts | established | asserted |
| 1e10 | 1.1e-6 | 1.6e-7 | 1.6e-7 | 2.6e-8 | 1.7e-7 | accepts | established | asserted (bound 1.1e-4) |
| 1e12 | 1.1e-4 | 5.6e-6 | 5.6e-6 | 4.4e-7 | 3.6e-5 | accepts | **not established** | reported, not asserted |
| 1e14 | 1.1e-2 | 7.2e-4 | 7.2e-4 | 5.4e-4 | 2.8e-4 | **refuses** ("numerically singular") | **undecidable** | reported, not asserted |

* Measured error / (kappa u) is at most 0.20 for kappa = 1e2 ... 1e10 (0.33 at 1e12, 0.70 componentwise at 1e14): the
  preregistered constant c = 100 is a generous *bound*, not a fitted one. The first-order scale kappa u is what the data
  follow (S error grows 1e-15 -> 1e-11 -> 1e-7 for kappa 1e2 -> 1e6 -> 1e10).
* Random-strain energy stays at rounding level at every kappa (the quadratic form is dominated by the stiff
  eigendirections); the soft strain, the eigenvector of lambda_min, loses kappa u. Quote the soft-strain row.
* Reciprocity nu_ij / E_i = nu_ji / E_j holds to <= 4.7e-17 (relative to max |S|) at every level, 8.6e-17 at 1e14; the
  mp residual is <= 1e-44. **This is not evidence that nu is accurate**: `ElasticTensor` symmetrises S, and
  nu_ij / E_i = -S_ji is then symmetric by construction; the accuracy of nu is the nu columns above.
* A valid anisotropic tensor with nu12 = 5, nu13 = -3 (orthotropic, E = (100, 1, 1), full SPD, synthetic) is accepted;
  a tensor with nu12 = 5 at E1 = 20 (det [[.05, -.25], [-.25, 1]] < 0) is rejected. Admissibility is the full SPD test,
  not an interval for nu_ij (`test_ec6c`, `test_ec6b`).

## What is certified, what is not

| Statement | Status | Basis |
|---|---|---|
| C SPD in exact arithmetic implies strong ellipticity with w_min/2 <= lambda(Q(n)) <= w_max for **every** unit n | **proved** (derivation in the prereg: eps = sym(a (x) n), ‖eps‖_F^2 in [1/2, 1]) | checked on 700 sampled directions per level; the bound is attained at kappa = 1 |
| isotropic acoustic eigenvalues {G, G, K + 4G/3}, c = sqrt(lambda / rho) | proved (closed form), verified to 1e-14 | `test_ac3`, `test_ac4` (units: Pa / (kg m^-3) = m^2 s^-2) |
| `check_positive_definite` accepts only SPD matrices (as stored) | **tested, not proved**: 0 false acceptances in 300 + 40 seeded matrices with lambda_min / lambda_max near 1e-18 ... 1e-10 or up to kappa 1e36 | EC7, R02b |
| float64 forward error <= c kappa u for the listed quantities | tested for kappa <= 1e10, **sampled** directions and strains with c = 100; not a worst-case bound (a direction avoiding the soft eigenvector can lose kappa^2 u; not tested) | EC1-EC5 |
| `strong_ellipticity` and `directional_extrema` results | **sampled / refined**, labelled so; the SPD bound above is the only certificate | `test_ac5` |
| accuracy for kappa > 1e10 | **not established**; refused outright above ~7e13 | table |
| fixed-load-axis Poisson extrema vs a dense transverse circle | covered by `test_anisotropy.py::test_p4_poisson_extrema_over_m`, not repeated | prereg AC6 |

`forward_accuracy_established(C6)` (new, `core/linear_elastic.py`) returns kappa, kappa u and a verdict:
"established" (kappa <= 1e11), "not_established", "undecidable" (|lambda_min / lambda_max| < 64 eps: the sign of
lambda_min cannot be decided from the stored matrix) or "not_positive_definite". It is a gate for reports, not an
error bound; `check_positive_definite` gained a docstring note of its own limit.

## R02: extreme Cholesky parameters (kappa ~ 1e35)

The historical input is not in this repository. Reconstruction: `theta_diag = linspace(-20, 20, 6)`, no off-diagonals:
exact kappa = e^80 = 5.5e34, which reproduces the order of magnitude of the risk text.

* The parameterisation M = L L^T (L lower triangular, positive diagonal exp(theta)) is **SPD by construction** in exact
  arithmetic for every finite theta. The stored float64 matrix was SPD in mp for all 41 cases tested.
* The map theta -> C is accurate **normwise**: ‖C_float - C_ref‖_F / ‖C_ref‖_F <= 2.04 u over 40 dense samples with exact
  kappa from 7.9e13 to 2.3e36. Its smallest eigenvalue is not: relative error of lambda_min from 8e-16 to 5.6e19 (a few
  samples exceed 1 by many orders of magnitude): the float matrix has the right norm and no meaningful small spectrum.
* Float64 **cannot decide SPD** there by a spectral criterion: every sample with kappa above ~7e13 is refused by
  `check_positive_definite` ("numerically singular"; 0 of 40 accepted), and `cholesky_parameters` of such a tensor
  raises. Plain float Cholesky of the 40 dense matrices of this family *succeeded* in every case (the existing xfail `test_anisotropy.py::test_p10_cholesky_parametrisation` records 37 of 1000 theta failing it, kappa 7.6e16 .. 1e35, in an N(0, 10) family); where it succeeds that is no evidence of
  usable accuracy; the library criterion (lambda_min / lambda_max > 64 eps) is the conservative one.
* No blanket acceptance was added. Consequence for use: a parameter vector reaching kappa > 1e10 should be bounded
  (clip the log-diagonal range) or the result flagged "forward accuracy not established"; the repository does not do
  this automatically.

### R02 status text (for the integrator; `risk_register.json` not edited here)

Proposed `status`: "VERIFIED_SCOPE (reproduced as a reconstruction: kappa 5.5e34 is SPD by construction and in mp, but
undecidable in float64 above kappa ~7e13; check_positive_definite refuses it and had no false acceptance in 340 seeded
cases; map accuracy normwise 2 u; no blanket acceptance)"; `evidence`: "docs/foundations/ELASTIC_CONDITIONING.md,
prereg E3_conditioning R02a/R02b"; `reproduction`: "python -m pytest tests/toporlogy/test_elastic_conditioning.py -k 'r02 or ec7'".
R03 (direction extrema sampled only) is unchanged: the SPD bound above certifies positivity of Q, not extrema.

## Failed or corrected predictions (full text in the prereg)

* AC2 at kappa = 1: the proved bound is attained and float eigenvalues fall 2.6e-16 below the preregistered 4 u slack:
  strict xfail + ledger entry; a labelled post-hoc test with 16 u passes.
* nu rows at kappa = 1: reference nu = 0 (isotropic), relative error undefined; metric changed to
  |x - ref| / max(|ref|, 1) after the first run (threshold unchanged).
* The prereg quoted kappa = 2.3333 for the isotropic E = 1, nu = 0.3 tensor; the derivation gives 3.25 (test asserts 3.25).
