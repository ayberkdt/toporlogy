# E3_conditioning — precision of anisotropic elasticity at extreme conditioning, acoustic tensor (preregistration)

Preregistered 2026-11-27 08:23 +0300, before any new numerical test of this slice was run. Base SHA `9ecc143`. Plan: WP04 E03,
acceptance rows E3 and E6 of WP23, risk R02. Modules read (not changed by this record): `core/linear_elastic.py`
(`check_positive_definite`, `engineering_constants`, `poisson_uniaxial`, `acoustic_tensor`, `strong_ellipticity`,
`wave_speeds`, `ElasticTensor`), `core/anisotropy.py` (`young_modulus`, `poisson_ratio`, `cholesky_parameters`,
`stiffness_from_cholesky`), `core/tensor.py` (Mandel conversions). Test file `tests/toporlogy/test_elastic_conditioning.py`.
All tensors here are **synthetic** (random orthogonal Q, seeded); nothing is material data.

## Production addition (the only one)

`linear_elastic.forward_accuracy_established(C6, kind="stiffness")` -> dict {"condition_number", "kappa_u", "rcond",
"verdict"}. With w the Mandel eigenvalues, rc = w_min / w_max, u = eps/2 = 2^-53, s = `SINGULAR_RCOND` = 64 eps:
rc <= -s -> "not_positive_definite"; |rc| < s -> "undecidable" (the sign of w_min cannot be decided in float64 from the
stored matrix); rc >= s and kappa <= `FORWARD_ACCURACY_KAPPA_LIMIT` = 1e11 -> "established"; otherwise
"not_established". (The "undecidable" band mirrors `check_positive_definite`, which refuses rc <= s.) The
limit 1e11 is chosen one decade above the largest asserted level (1e10) so that kappa = 1e10 is not on the boundary
(eigenvalue noise ~ u kappa ~ 1e-6 relative); it means c kappa u <= 100 * 1.1e-5 ~ 1e-3 (about 3 correct digits) for
c = 100. `check_positive_definite` gets a docstring note of its own limit (acceptance needs rc > 64 eps, so an
exactly SPD matrix with kappa > ~7e13 is refused; accepted implies the float matrix is SPD, tested as EC7).

## Reference method and its independence

mpmath 1.3.0, `mp.dps = 50` (kappa <= 1e14: 36 digits left after the conditioning loss; the R02 cases with kappa up to
1e35 use `mp.dps = 100`, 65 digits left). The reference takes the **float64 inputs exactly** (C6 as stored,
strain/direction vectors as stored) and computes everything in multiprecision with formulas written in the test:
Mandel image M = diag(m) C diag(m), m = (1,1,1,sqrt2,sqrt2,sqrt2) in 50 digits, `mp.inverse`, `mp.eigsy`, `mp.cholesky`.
No toporlogy function is called by the reference. Float64 production results are the toporlogy functions named per row.
The energy row compares the production Mandel path with a plain float64 quadratic form: there is no production energy
function, so EC1's float side is "production Mandel conversion + numpy matmul" (a regression of that path, labelled as
such); its reference is independent.

## Derivations (rule 11)

* Energy. Voigt strain eps_v (engineering shear gamma = 2 eps_ij), stress sigma_v = C eps_v (physical shear stress).
  eps:sigma = eps_xx sigma_xx + ... + 2(eps_yz sigma_yz + ...) = eps_v . sigma_v, so W = 1/2 eps_v^T C eps_v. In Mandel
  form e = eps_v / m (shear entries gamma/sqrt2), M = diag(m) C diag(m): e^T M e = eps_v^T C eps_v. Reference W is
  computed from eps_v and C directly (no Mandel image) in mp; the float side goes through `T.stiffness_to_mandel`.
* Soft strain: e_soft = eigenvector of the smallest eigenvalue of M_ref (mp), eps_v = m * e_soft rounded to float64, so
  W_soft = lambda_min |e|^2 / 2 (+ O(u^2 lambda_max)): the adversarial strain for which the float error is ~kappa u.
* Directional properties. Voigt compliance S (engineering shear). Uniaxial stress sigma_ij = n_i n_j has
  sigma_v = a(n) = (n1^2, n2^2, n3^2, n2 n3, n1 n3, n1 n2); strain along m: eps_mm = eps_ij m_i m_j = a(m) . eps_v (the
  gamma entries enter as gamma m_i m_j). Hence 1/E(n) = a^T S a, and nu(n, m) = -eps_mm / eps_nn
  = -(a(m)^T S a(n)) / (a(n)^T S a(n)) (m perpendicular to n). Engineering constants: nu_ij = -S_ji / S_ii (stress along
  i, contraction along j), E_i = 1/S_ii, so reciprocity nu_ij / E_i = -S_ji = -S_ij = nu_ji / E_j holds exactly for a
  symmetric S. `poisson_uniaxial(C, i, j)` = -x_j / x_i with C x = e_i, equal to nu_ij.
* Acoustic tensor. Q_ik(n) = C_ijkl n_j n_l. For unit a, a^T Q a = eps:C:eps with eps = sym(a (x) n), and
  ||eps||_F^2 = (1 + (a.n)^2)/2 in [1/2, 1]. With C SPD (Mandel eigenvalues w_min <= w_max): for every unit n
  **w_min / 2 <= lambda(Q(n)) <= w_max**. This is a proof, not a sample: SPD implies strong ellipticity with this
  margin for all n.
* Isotropic. C_ijkl = (K - 2G/3) d_ij d_kl + G (d_ik d_jl + d_il d_jk) gives Q = G I + (K + G/3) n (x) n, eigenvalues
  {G, G, K + 4G/3}; phase speeds c = sqrt(lambda / rho), unit Pa / (kg m^-3) = m^2 s^-2. Scaling: c ~ rho^(-1/2) and
  c ~ sqrt(s) under C -> s C.
* Poisson outside the isotropic range. Orthotropic, E = (100, 1, 1), S12 = -nu12 / E1, S13 = -nu13 / E1, S23 = -nu23 / E2;
  nu12 = 5, nu13 = -3, nu23 = 0.2, G = 0.4: normal block [[.01,-.05,.03],[-.05,1,-.2],[.03,-.2,1]] has leading minors
  .01, .0075, det = .0068 > 0 (hand computation), so SPD. The control nu12 = 5 with E1 = 20 has [[.05,-.25],[-.25,1]],
  det = .05 - .0625 < 0: not SPD.
* Cholesky parameterisation. M = L L^T, L lower triangular with diag exp(theta_i): SPD in exact arithmetic for every
  finite theta (det L = prod exp(theta_i) > 0). For L = diag(exp(theta)) (no off-diagonals) kappa = exp(2 (theta_max -
  theta_min)); theta_diag = linspace(-20, 20, 6): kappa = exp(80) = 5.5e34 (the historical "~1e35" of risk R02; the
  historical input itself is not in this repository, this is a reconstruction of its order of magnitude).

## Checks and thresholds (u = 2^-53 = 1.11e-16; kappa = reference condition number of the float64 input, from mp)

Levels: designed kappa in {1, 1e2, 1e6, 1e10, 1e12, 1e14}; Mandel eigenvalues geometric from 1 to kappa, M = Q diag Q^T,
Q from `np.linalg.qr` of a seeded normal matrix (`default_rng(20261127)`), M symmetrised, C6 = `T.mandel_to_stiffness(M)`.
**Asserted** for kappa <= 1e10: error <= c kappa u with **c = 100** (one c for every row). 1e12 and 1e14: measured and
reported only ("forward accuracy not established"), no assertion beyond finiteness and the verdict row EC8.

| ID | Hypothesis | Metric | Threshold |
|---|---|---|---|
| EC1 | float energy 1/2 eps^T C eps vs mp: random eps (20) and soft eps | max rel. error | <= 100 kappa u (kappa <= 1e10) |
| EC2 | float compliance S = sym(inv(C6)) (the `ElasticTensor.compliance_material` recipe) vs mp inverse of the same C6 | normwise Frobenius error / norm in Mandel; componentwise max reported only | <= 100 kappa u |
| EC3 | `young_modulus(S, n)` for 20 random unit n (S float) vs mp 1/(a^T S_ref a) | max rel. error | <= 100 kappa u (sampled directions, **not** a worst-case bound: an n whose a(n) avoids the soft eigenvector can reach kappa^2 u; named, not tested) |
| EC4 | `poisson_ratio(S, n, m)` (20 random orthonormal pairs), `engineering_constants` nu_ij (6), `poisson_uniaxial(C6, i, j)` (6) vs mp | max rel. error per path | <= 100 kappa u |
| EC5 | reciprocity: r = max_(i<j) abs(nu_ij / E_i - nu_ji / E_j) / max abs(S) from `engineering_constants`; mp reference residual | r (float), r_mp | float: <= 100 kappa u (kappa <= 1e10); mp: <= 1e-30 |
| EC6a | `check_positive_definite`: accepts kappa in {1,1e2,1e6,1e10,1e12}; **rejects** kappa = 1e14 (rc = 1e-14 < 64 eps = 1.42e-14: "numerically singular") although mp proves it SPD; verdict of `forward_accuracy_established`: established (<= 1e10), not_established (1e12), undecidable (1e14) | exact | exact |
| EC6b | intentionally non-SPD: Mandel spectrum with lambda_min = -1e-1, -1e-6, -1e-12 (relative to lambda_max); also an exactly singular one (lambda_min = 0) and S with nu12 = 5, E1 = 20 | `NotPositiveDefiniteError` for every one; `ElasticTensor` refuses; verdict "not_positive_definite" or "undecidable" (never "established") | exact |
| EC6c | valid anisotropic: orthotropic S above (nu12 = 5, nu13 = -3) accepted by `ElasticTensor.from_compliance`; `engineering_constants` returns nu12 = 5, nu13 = -3 (rel 1e-12); mp eigenvalues of S all > 0 | exact / rel | 1e-12 |
| EC7 | soundness of acceptance: 300 seeded matrices, lambda_min / lambda_max = s * 10^U(-18, -10), s = +-1, rest of the spectrum 1..10; whenever `check_positive_definite` accepts, mp Cholesky of the float Mandel matrix (the matrix it sees, symmetrised) succeeds | false acceptances | 0 |
| EC8 | verdict helper: values above for kappa = 1, 1e2, 1e6, 1e10 -> "established", 1e12 -> "not_established", 1e14 -> "undecidable"; isotropic E = 1, nu = 0.3 -> kappa = 2.3333 ("established") | exact | exact |
| AC1 | acoustic tensor symmetric: 500 Fibonacci + 200 seeded random n, SPD tensors kappa in {1, 1e2, 1e6, 1e10} | max abs(Q - Q^T) / max abs(Q) | <= 1e-14 |
| AC2 | acoustic tensor positive definite and **within the proved bounds** w_min/2 <= lambda(Q) <= w_max for all sampled n (kappa <= 1e10, w from mp, bound relaxed by 4u w_max for rounding) | violations | 0 |
| AC3 | isotropic Q eigenvalues {G, G, K + 4G/3} (K = 1.7e11 Pa, G = 8.0e10 Pa, synthetic, and K = 3, G = 1), 50 random n, vs the closed form above | max rel. error | <= 1e-14 |
| AC4 | `wave_speeds`: c = sqrt(lambda / rho) equals sqrt(G/rho), sqrt((K + 4G/3)/rho) (rho = 7800 kg/m^3, speeds in m/s); rho x 4 halves c; C x 9 triples c; C in GPa x 1e9 equals C in Pa; rho <= 0 or nan -> ValueError; indefinite Q (G < 0) -> NaN | rel | 1e-14 (scaling 1e-15) |
| AC5 | `strong_ellipticity(C)` label is "sampled"/"refined" (never a certificate word), min eigenvalue >= w_min/2 (the proved bound) and <= w_max, SPD kappa 1e6 | exact label, inequality | exact |
| AC6 | Fixed-load-axis Poisson extrema against a dense transverse circle: already covered by `test_anisotropy.py::test_p4_poisson_extrema_over_m`; not repeated | — | — |
| R02a | reconstruction: theta_diag = linspace(-20, 20, 6), no off-diagonals: mp kappa in [1e34, 1e36] (analytic 5.5e34); `stiffness_from_cholesky` returns a finite matrix; `check_positive_definite` raises `NotPositiveDefiniteError`; `cholesky_parameters` of it raises; verdict "undecidable" | exact | exact |
| R02b | dense family: ranges r in {8, 12, 16, 20}, theta_diag = r * linspace(-1, 1, 6) permuted, theta_ij = N(0,1) exp(theta_ii), 10 seeds per range (40 matrices), mp dps = 100: (i) exact L L^T has positive mp Cholesky (SPD by construction) for all; (ii) every sample with exact kappa >= 1e15 is **rejected** by `check_positive_definite`; (iii) no false acceptance (accepted -> mp-SPD float Mandel matrix); (iv) the map theta -> C is accurate normwise: Frobenius error of C_float vs C_ref in Mandel <= 100 u for all samples. Reported only: fraction with float Cholesky failure, fraction accepted, smallest-eigenvalue relative error | counts | (i) all, (ii) all, (iii) 0, (iv) all |

Hypotheses to be falsified: that the sampled-direction error stays within 100 kappa u (EC3/EC4); that the library
rejects kappa = 1e14 (EC6a); that no non-SPD matrix is accepted (EC7, R02b-iii). Failures follow the brief (xfail
strict + ledger additions in `docs/foundations/ledger_additions/E3_conditioning.json`).

Run: `python -m pytest tests/toporlogy/test_elastic_conditioning.py -q -p no:randomly` (TMP/TEMP and basetemp as the brief).
First results are appended below.
