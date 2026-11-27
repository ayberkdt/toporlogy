# L6e_eigen — reliability checks for generalized symmetric eigenproblems and the buckling pencil (WP05 L06, eigen part; preregistration)

Preregistered 2026-11-27 09:38 +0300 (pasted from `date` by the command that wrote it), before any test of the new module was run.
Base SHA `97a0768` (branch `worktree-agent-ac0c2d9348f0ba32f`). New module `src/toporlogy/solvers/eigen.py` (imports only numpy, scipy;
no `toporlogy.fem` import, checked by an AST scan in the test file), tests `tests/toporlogy/test_eigen.py`, docs
`docs/foundations/EIGEN_RELIABILITY.md`, update of `docs/foundations/SOLVER_CONTRACT.md`. Plan: `05_NUMERICAL_LINALG.md` L06 (eigen
paragraph). It extends, and does not replace, `fem/modal.py` (Lanczos shift-invert with Rayleigh-Ritz clean-up, one extra mode for
cluster detection in the backward pass) and `fem/buckling.py` (the pencil (-K_G) phi = mu K phi solved by Lanczos with K^-1 as
Minv); neither is modified. All data synthetic (E, rho, I and A are model units, no material data).

## Definitions and derivations (written before the run; rule 11)

Pencil `K phi = lambda M phi`, K and M real symmetric, M SPD (checked by Cholesky). Norms: vectors 2-norm, matrices 2-norm
(dense `norm(., 2)` for n <= 400, otherwise a Lanczos estimate, never above the true value by more than its tolerance).

* Residual of one pair: `eta_i = ||K phi_i - lam_i M phi_i|| / ((||K|| + |lam_i| ||M||) ||phi_i||)`. A backward error: phi_i, lam_i is
  an exact eigenpair of (K + dK, M + dM) with ||dK||/||K|| and ||dM||/||M|| of order eta_i (Parlett, the standard result for the
  symmetric pencil; **cited from memory, not re-read**; only the definition is used for pass/fail).
* M-orthonormality: `||Phi^T M Phi - I||_F`.
* Rayleigh consistency: `rho_i = phi_i^T K phi_i / phi_i^T M phi_i`, `r_i = |phi_i^T K phi_i - lam_i phi_i^T M phi_i| / ((||K|| + |lam_i| ||M||) ||phi_i||^2)`.
* Clusters: sorted eigenvalues i, i+1 are in one cluster when `|lam_i - lam_{i+1}| <= cluster_rtol max(|lam_i|, |lam_{i+1}|) + 1e-12 ||K||/||M||`
  (default cluster_rtol = 1e-6). Inside a cluster of size > 1 individual vectors are not unique; the identity of the cluster is its
  subspace. Cluster block residual (basis independent): `Theta = (Phi_c^T M Phi_c)^-1 Phi_c^T K Phi_c`,
  `R_c = K Phi_c - M Phi_c Theta`, normalised by `(||K|| + max|lam| ||M||) ||Phi_c||_F`.
* Principal angles in the M inner product: orthonormalise both bases (`Q^T M Q = I`); cosines are the singular values of `Q_a^T M Q_b`;
  the sines are the singular values of `M^(1/2) (Q_b - Q_a Q_a^T M Q_b)`, evaluated without squaring (thin QR of the residual
  block S = Q_s R_s, then the singular values of `chol(Q_s^T M Q_s)^T R_s`, which equal those of `M^(1/2) S`), and the angle is
  `arctan2(sin, cos)` (sines ascending paired with cosines descending, since sin^2 + cos^2 = 1 on the same pair). Squaring the Gram
  matrix would put a noise floor of about 1e-8 under the angle and make the threshold 1e-8 unmeasurable.
* MAC: `MAC_ij = (a_i^T M b_j)^2 / ((a_i^T M a_i)(b_j^T M b_j))`, in [0, 1], invariant to sign and scale.
* Tracking: consecutive steps matched by the assignment maximising the sum of MAC (Hungarian, `scipy.optimize.linear_sum_assignment`);
  M between two steps is the mean of the two mass matrices when they differ. A crossing between steps s-1 and s is a pair of tracks whose
  column order is reversed. A track is ambiguous at a step when its MAC is below `mac_min` (0.9) or exceeds the next best in its row
  by less than 0.2. Limit stated in the docs: a mode entering the window from outside is invisible, so track k + extra modes.
* Shift-invert about sigma uses the operator `(K - sigma M)^-1 M`, whose eigenvalues are `1/(lam_i - sigma)`; it is self-adjoint in the M
  inner product, so its spectral radius is `1/min|lam_i - sigma|` and its 1-norm is at least that. With a sparse LU of K - sigma M, the
  1-norm estimate `g` of this operator (`onenormest`, matvec `lu.solve(M v)`, rmatvec `M lu.solve(v)`, valid because K - sigma M is
  symmetric) gives `1/g <= min|lam_i - sigma|`. The shift is **refused** (`ShiftError`, a `ValueError`) when the LU reports an exactly
  singular matrix or `1/g <= refuse_rtol * s`, with `s = ||K||_1 / ||M||_1` and refuse_rtol = 1e-8. The test is conservative: it
  can refuse a shift that is merely close, never accept one at an eigenvalue.
* Semidefinite K (free-free): the eigenvalues are >= 0, so any sigma < 0 gives an SPD K - sigma M and `|lam - sigma|` increases with lam: the
  k eigenvalues nearest sigma are the k lowest. With `assume_psd=True` and `sigma=None` the shift is `-1e-6 s`. A zero mode is
  reported when `|lam| <= zero_tol * s` (zero_tol = 1e-9); its null-space residual `||K phi||/(||K|| ||phi||)` is reported. A count
  differing from an optional `expected_zero` is reported (`zero_count_ok = False`), not hidden.
* Cluster cut: the wrapper computes k + 1 eigenpairs, returns k and reports `cluster_cut = True` when the extra eigenvalue is in the
  cluster of a returned one (or equidistant from sigma as the k-th). A cut cluster has an incomplete subspace.
* Buckling pencil: reference load P, `K u = P`, buckling at the load lambda P where `(K + lambda K_G) phi = 0`, K_G built from the stress of
  the reference state (tension positive, so compression makes K_G negative along the mode). With `mu = 1/lambda` the problem is
  `(-K_G) phi = mu K phi`. **Assumptions** (the solver checks the first two): K symmetric positive definite (the structure is stable
  unloaded); K_G symmetric, in general indefinite; the load factor of interest is the smallest positive lambda = 1/(largest positive mu).
  mu > 0 means compression in the reference direction; mu < 0 is a buckling under the reversed load (lambda = 1/mu < 0, reported
  separately, not a load factor); mu ~ 0 is a direction that does not buckle. Normalisation `phi^T K phi = 1`, so mu = -phi^T K_G phi.

Closed forms for the checks (derived here, evaluated in the test with `math.sqrt`; the beam roots by `scipy.optimize.brentq`):

1. 2-DOF buckling: `K = [[4, 1], [1, 3]]` (det 11, trace 7: SPD), `K_G = [[-2, 1], [1, 1]]` (det -3: one positive, one negative
   eigenvalue: indefinite). `K + lambda K_G = [[4 - 2 lambda, 1 + lambda], [1 + lambda, 3 + lambda]]`,
   det = (4 - 2 lambda)(3 + lambda) - (1 + lambda)^2 = 11 - 4 lambda - 3 lambda^2 = 0, so `lambda = (-4 +/- sqrt 148)/6`:
   lambda_+ = 1.360921 (the buckling load factor) and lambda_- = -2.694255 (reversed load; mu = -0.371153). The mu of the problem
   are 1/lambda_+ = 0.734785 and 1/lambda_- = -0.371153. Wrong sign convention (K - lambda K_G): det = 11 + 4 lambda - 3 lambda^2,
   lambda = (4 +/- sqrt 148)/6, lowest positive 2.694255 != 1.360921: the sign of K_G matters and the test discriminates it.
2. Free-free Euler-Bernoulli beam (in plane, DOF u, v, theta per node, L = 1, E = A = rho = 1, I = 2e-4): `lambda_n = (beta_n L)^4 E I / (rho A L^4)`
   with `cos x cosh x = 1`, x = beta_n L; the three zero eigenvalues are the rigid modes (u = 1; v = 1; v = x, theta = 1). The thin
   section keeps the first axial eigenvalue (pi^2 E/rho / L^2 = 9.87) above the fourth bending one (about 8.0 from lambda = x^4 I, x = 14.137),
   so the lowest six eigenvalues are 3 zeros and 3 bending values.
3. Square membrane, Q1 tensor structure: with the 1-D pencil `(K1, M1)` of eigenpairs `(mu_i, psi_i)`, `K = K1 (x) M1 + M1 (x) K1`,
   `M = M1 (x) M1` has eigenpairs `(mu_i + mu_j, psi_i (x) psi_j)` (apply K to the product: (mu_i + mu_j) M1 psi_i (x) M1 psi_j). So
   mu_i + mu_j = mu_j + mu_i: every pair i != j is an exactly double eigenvalue, and the subspace is span{psi_i (x) psi_j, psi_j (x) psi_i}.
   The lowest six are (1,1), (1,2)=(2,1), (2,2), (1,3)=(3,1), provided mu_1 + mu_3 < mu_2 + mu_3 (true, mu increasing): clusters of sizes 1, 2, 1, 2.
4. Crossing families: `K(t) = M Psi diag(a(t), b(t), ...) Psi^T M` with `Psi^T M Psi = I` fixed (Psi = M^(-1/2) Q for a fixed rotation Q): eigenvalues
   exactly `a(t), b(t), ...` with the **fixed** eigenvectors Psi columns, so the identity of a mode is the column of Psi regardless of
   the sorted index. No coupling: the eigenvalues truly cross (no avoided crossing).

## Checks

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| E0 | layer rule: `solvers/eigen.py` imports no `toporlogy.fem`, no torch | AST scan of the module source | no such import |
| E1a | `eigen_report` on the dense `scipy.linalg.eigh(K, M)` result (n = 30, seeded random SPD K, M) | the report's own definitions, evaluated by the test with numpy | max eta <= 1e-13, orth <= 1e-12, max r <= 1e-13 |
| E1b | shift-invert wrapper (sparse, n = 80, sigma = -0.5, k = 6) vs the dense `eigh(K, M)` | `scipy.linalg.eigh(K, M)` in the test | lam rel <= 1e-9; MAC(wrapper, dense) diagonal >= 1 - 1e-9 (simple eigenvalues); report max eta <= 1e-10 |
| E1c | negative controls of the report: phi_1 perturbed by 1e-6 (relative), lam_2 shifted by 1e-6 (relative), Phi scaled by 2 | the report | eta_1 > 1e-8; r_2 > 1e-8; orth > 1; `passed()` False for each |
| E2a | repeated eigenvalues: membrane (12 x 12 interior Q1 grid, n = 144) lowest six: lam equal to the Kronecker values mu_i + mu_j (1-D `eigh` in the test) and clusters from the report equal [[0], [1, 2], [3], [4, 5]] | Kronecker structure, 1-D `scipy.linalg.eigh(K1, M1)` | lam rel <= 1e-10; clusters exactly as stated |
| E2b | wrapper (k = 6) cluster subspaces vs the analytic Kronecker subspaces, in the M inner product | span{psi_i (x) psi_j, psi_j (x) psi_i} | max principal angle <= 1e-8 for both doubles; report cluster block residual <= 1e-10 |
| E2c | negative control: rotating the wrapper's double-cluster basis by 37 degrees changes the individual vectors (MAC of the rotated vector with the original <= 0.7, cos^2 37 deg = 0.638) but not the subspace | principal angle of the rotated vs original block | individual MAC <= 0.7; angle <= 1e-8; report residuals still <= 1e-10 |
| E2d | cluster cut: k = 5 cuts the second double; `cluster_cut` True; k = 6 and k = 4 do not cut (k = 4 ends before the first member of the double: extra = 5th, its eigenvalue is the double's lower member; it is not equal to any of the first four) | structure above | flags as stated |
| E3a | free-free beam (20 elements): dense `eigvalsh(K)` has exactly 3 values with `abs <= 1e-9 ||K||` (independent count); wrapper with `assume_psd=True`, `expected_zero=3`, k = 6: `n_zero == 3`, `zero_count_ok`, zero-mode residual <= 1e-10, zero eigenvalues `abs <= 1e-9 s` | dense `eigvalsh` count in the test | as stated |
| E3b | the three zero modes span the analytic rigid space {u = 1}, {v = 1}, {v = x, theta = 1} | principal angles in the M inner product, rigid vectors built in the test | angle <= 1e-8 |
| E3c | the next three eigenvalues match `(x_n)^4 E I` with x_n the first three roots of `cos x cosh x = 1` (`brentq` in the test, brackets (4, 5), (7, 8), (10, 11)) | closed form | rel <= 1e-3 (cubic Hermite elements, 20 elements) |
| E3d | negative controls: sigma = 0 on the free-free pencil raises `ShiftError`; `expected_zero=2` gives `zero_count_ok == False`; `assume_psd=False` with `sigma=None` raises ValueError | — | as stated |
| E3e | clamped beam (node 0 fixed): K SPD, sigma = 0 accepted, `n_zero == 0`, lowest three bending eigenvalues vs `(x_n)^4 E I`, x_n roots of `cos x cosh x = -1` (brackets (1.5, 2.5), (4, 5), (7, 8)) | closed form | rel <= 1e-3 |
| E3f | shift refusal on a known spectrum: K = diag(1..60), M = I: sigma = 3 exactly raises; sigma = 3(1 + 1e-12) raises; sigma = 3(1 + 1e-3) is accepted and returns the nearest eigenvalues 3, 2, 4, 1, 5 (k = 5) in ascending order with lam abs error <= 1e-9 | exact spectrum | as stated |
| E4a | 2-DOF family, a(t) = 1 + t, b(t) = 3 - t, 8 steps on [0, 2] (t = 0, 2/7, ..., the crossing t = 1 is between steps 3 and 4), non-identity M, Psi from a fixed rotation: `track_modes` follows the columns of Psi | columns of Psi | tracked MAC to Psi column >= 1 - 1e-10 at every step; every consecutive MAC >= 1 - 1e-10; exactly one crossing, at step 4, tracks (0, 1); tracked lam = a(t), b(t) to 1e-12 |
| E4b | negative control, index matching: the MAC of sorted-index mode 0 between steps 3 and 4 is <= 1e-12 (Psi columns M-orthogonal) and the lam of sorted-index mode 0 differs from a(t) by > 0.5 somewhere | MAC definition | as stated |
| E4c | 3-DOF family, a = 1 + t, b = 2.5, c = 4.5 - 0.5 t, 12 steps on [0, 5], M full SPD: crossings at t = 1.5 (a,b), 7/3 (a,c), 4 (b,c): reported as steps 4, 6, 9 with the track pairs (0,1), (0,2), (1,2); no ambiguity | analytic crossing times | as stated |
| E4d | family without crossing (a = 1 + t, b = 3 + t) has no crossing and the identity assignment | — | none reported |
| E5a | 2-DOF buckling (closed form 1): `buckling_load_factors(K, KG, k=1)` lowest positive load factor, K_G inertia (1, 1, 0), reversed load factor reported | `(-4 + sqrt 148)/6`, `(-4 - sqrt 148)/6` | rel <= 1e-12; inertia exact; reverse rel <= 1e-12; K-orthonormality <= 1e-12 |
| E5b | assumptions enforced: K indefinite ([[1, 2], [2, 1]]), K non-symmetric, K_G zero (no buckling: empty load factors and a note, not a division by zero) | — | ValueError, ValueError, empty |
| E5c | negative control: the opposite sign convention (K_G replaced by -K_G) gives the lowest positive load factor 2.694255, not 1.360921 | closed form 1 | differs by > 1 |
| E5d | clamped-free column of the repository (`T.Domain`, `T.Elasticity`, `LinearBuckling`; b = 4, L = 160, t = 1, E = 1000, nu = 0, Q4, unit compressive resultant) with 2, 4, 8 elements across: K from `Elasticity.matrix`, K_G from `LinearBuckling.geometric_matrix`; lowest positive load factor of the wrapper vs `P_E = pi^2 E I/(4 L^2)`, I = b^3 t/12 | Euler closed form | err(8) < 0.01 and err(8) < err(4) < err(2) (the repository's own measured values 0.123, 0.030, 0.006 motivate the size; the closed form is the reference) |
| E5e | independent solve at nb = 2 (n = 486): dense `scipy.linalg.eigh(-K_G, K)` in the test vs the wrapper's Lanczos path (`dense_max` lowered to force it), lowest 3 positive mu | dense eigh in the test | rel <= 1e-8 |
| E5f | regression (labelled so, same K and K_G, second call to the repository kernel): wrapper mu_1 vs `LinearBuckling.modes` mu_1 at nb = 4 | `LinearBuckling.modes` | rel <= 1e-8 |
| E6 | invalid input rejected: non-square, shape mismatch, NaN, K or M non-symmetric, M not SPD, k out of range, sigma not finite | — | ValueError (ShiftError for shifts) |

Run: `python -m pytest tests/toporlogy/test_eigen.py -q -p no:randomly` (TMP/TEMP and basetemp under
`.claude/tmp-toporlogy/L6e_eigen`). First results are appended below.
