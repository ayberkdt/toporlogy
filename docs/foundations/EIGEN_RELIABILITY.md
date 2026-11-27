# Eigenproblem reliability (WP05 L06, eigen part)

Code: `src/toporlogy/solvers/eigen.py` (numpy and scipy only; no `toporlogy.fem`, checked by test E0). Preregistration and results:
`prereg/L6e_eigen.md`; tests `tests/toporlogy/test_eigen.py`. All data synthetic (model units). The production solves in
`fem/modal.py` and `fem/buckling.py` are unchanged; `eigen_report` can be applied to their output.

## Problems and assumptions

| Problem | Pencil | Assumptions (enforced) |
|---|---|---|
| Modal | K phi = lambda M phi | K, M symmetric (relative asymmetry <= 1e-12); M SPD (Cholesky); K may be semidefinite |
| Buckling | (K + lambda K_G) phi = 0, solved as (-K_G) phi = mu K phi, mu = 1/lambda | K symmetric positive definite (Cholesky up to 2500 unknowns, above that the smallest eigenvalue by Lanczos: evidence, not proof); K_G symmetric, indefinite in general |

Buckling load convention: K_G is built from the stress of the reference load (tension positive), so compression makes K_G negative
along the mode and mu > 0. The load factor is 1/mu for the largest mu > 0 and means buckling at lambda times the reference load.
mu < 0 is buckling under the reversed load (reported as `reverse_load_factors`, dense path only, never as a load factor);
mu close to 0 does not buckle; no positive mu returns an empty result with a note. Modes are K-orthonormal, mu = -phi^T K_G phi.
A wrong sign of K_G is not detectable from the matrices alone; E5c shows that it changes the answer (2.694 instead of 1.361 in the 2-DOF case).

## What `eigen_report(K, M, lam, Phi)` measures

* Per pair: backward error `eta = ||K phi - lam M phi|| / ((||K|| + |lam| ||M||) ||phi||)` (2-norms) and Rayleigh consistency
  `r = |phi^T K phi - lam phi^T M phi| / ((||K|| + |lam| ||M||) ||phi||^2)`.
* `||Phi^T M Phi - I||_F`.
* Clusters: sorted eigenvalues within `cluster_rtol` (1e-6, relative, plus 1e-12 ||K||/||M||). A cluster of size > 1 has no unique vectors:
  it gets a basis-independent block residual, and with `reference=(lam_ref, Phi_ref)` a comparison by principal angles
  (`subspace_angles`, M inner product, sines computed without squaring so angles of 1e-15 are resolved). Single vectors of a cluster are
  never compared.
* `passed(...)` applies tolerances (default 1e-8 each). A small residual says the pair is an exact eigenpair of a nearby pencil; it does
  not say which eigenvalue it is (a missing part of the spectrum is invisible to it). Completeness of a cluster is the job of the
  cluster-cut flag below.

## MAC and tracking

`mac(A, B, M)` is (a_i^T M b_j)^2 / ((a_i^T M a_i)(b_j^T M b_j)). `track_modes(lams, Phis, M)` matches consecutive steps by the
assignment maximising the sum of MAC, so a track is a physical mode, not a sorted index. It reports crossings (track pairs whose column
order reversed, with the step), the MAC of each track with itself, and ambiguous tracks (MAC < 0.9 or within 0.2 of the second best).
Limits: a mode entering the window from outside cannot be seen (track k + extra modes); an avoided crossing with a small gap and a coarse
step looks like a crossing or like none depending on the step, so the MAC values and the ambiguity flag are the diagnostic, not a guarantee;
inside a cluster the MAC of single vectors is meaningless, use subspaces.
Measured (E4): in an exact crossing family the tracked MAC to the analytic eigenvector is 1 - 1e-15 or better, the index-matched MAC across the
crossing is 5e-30.

## Shift-invert

`shift_invert_modes(K, M, k, sigma=None, assume_psd=False, ...)`:

* The shift is an explicit choice. `sigma=None` is allowed only with `assume_psd=True`; the shift is then -1e-6 s (s = ||K||_1/||M||_1), below
  the spectrum of a semidefinite K, so K - sigma M is SPD and the k nearest eigenvalues are the k lowest.
* A shift at or numerically at an eigenvalue is refused with `ShiftError`: an exactly singular LU, or a 1-norm estimate g of
  (K - sigma M)^-1 M (spectral radius = 1/min|lam - sigma|, so 1/g <= min|lam - sigma|) with 1/g <= 1e-8 s. Conservative: it can refuse a
  shift that is only close, never accept one at an eigenvalue. The estimate is returned as `shift_distance`.
* Zero modes (|lam| <= 1e-9 s) are reported with count, indices and the null-space residual ||K phi||/(||K|| ||phi||). `expected_zero`
  is compared, not enforced (`zero_count_ok`).
* One extra pair is computed; `cluster_cut` is True when it belongs to the cluster of a returned pair (the returned subspace of that cluster is incomplete).
* Pairs are cleaned by Rayleigh-Ritz in their span, sorted ascending by lambda, and carry their `eigen_report`.

## Measured (first run, `prereg/L6e_eigen.md`)

Free-free 20-element Euler-Bernoulli beam: 3 zero modes found with the default shift (sigma = -1.6e-3), residual 9e-17, angle to the analytic
rigid space 3.6e-14, sigma = 0 refused; first three bending eigenvalues within 4e-6, 3e-5, 1e-4 of the closed form. Clamped-free column of the
repository (Q4, 2, 4, 8 elements across): |P_cr/P_Euler - 1| = 0.125, 0.031, 0.0074 (Euler is the reference; this is the discretisation error of
the repository's element, not of the eigen solver). Square membrane doubles: subspace angles 6e-15.

## Not done

Non-symmetric or damped (quadratic) eigenproblems; Lanczos for the reversed-load buckling modes; a proof of K positive definiteness
for large n (Lanczos evidence only); a bound on the distance of a computed eigenvalue to the true spectrum (a residual is a backward error;
the forward error also needs the gap); coupling of `track_modes` to the topology-optimisation loop.
