# Nonlinear continuum mechanics: what is verified and what is not (WP10 slice N1–N5)

Record: `docs/foundations/prereg/N_nonlinear.md` (preregistered 2026-11-27 02:53 +0300; first result appended there).
New tests: `tests/toporlogy/test_nonlinear_foundations.py` (55 tests, 13 s). All values below are from that run.
Units: unit Young modulus, synthetic parameters (not material data).

## Implemented formulations

* Total Lagrangian, F = I + ∇u, E = ½(FᵀF − I) formed from H = F − I without cancellation, S = ∂ψ/∂E, P = F S.
  Elements: Q4 (2-D, plane stress or plane strain, thickness) and H8 on the regular `Domain` grid only
  (`fem/nonlinear.py`; non-grid domains are refused).
* Laws: Saint Venant–Kirchhoff (SVK; moderate strain, softens without bound in compression) and compressible
  neo-Hooke ψ = ½κ(J−1)² + ½μ(J^(−2/3) tr C − 3) (`core/constitutive.py`); plane stress for neo-Hooke by a local scalar
  Newton on S₃₃ = 0 with the condensed tangent (valid for this isotropic law only; SVK plane stress is the 2-D law with
  the plane-stress constants).
* Small-strain J2 plasticity with isotropic hardening through the history contract (no finite-strain plasticity).
* Path following: arc length, displacement control, critical-point location (`solvers/continuation.py`).

## Verified in this slice (independent references)

| Item | Reference | Result |
|---|---|---|
| F = I, pure rotation (to π) | zero stress and energy | ≤ 1.4e-15 |
| Simple shear, uniaxial / biaxial / spherical stretch | closed forms for ψ, S, P, Cauchy (hand-derived in the record) | ≤ 2.2e-14 |
| Objectivity of ψ, S, P, Cauchy under F → QF | invariance / transformation, 50 random pairs | ≤ 8e-14; negative control (linear strain energy of a 30° rotation 3.5e-2) |
| Tangent ∂S/∂E, pushed to ∂P/∂F | central FD of an independent P; torch Hessian of an independent ψ(F); major symmetry | FD 2.6e-10, autograd 4.5e-15 |
| Small-strain limit | S − σ_lin(ε) over σ_lin, δ from 1e-1 to 1e-5 | log–log slope 0.999 to 1.001 (expected 1) |
| Hencky strain with `core.spectral` | ½ logm(C) (scipy); derivative vs FD | 3.6e-15; 1.4e-9 |
| Work conjugacy S:Ė = P:Ḟ = τ:d; Hencky rate in the coaxial case | identities | ≤ 8.9e-16; non-coaxial negative control 0.74 |
| Plane-stress neo-Hooke local solution σ₃₃ = 0 | brentq root of the closed-form P₃₃ | 1.9e-14 |
| Condensed plane-stress tangent | central FD of the constrained S | 1.0e-9 |
| Inadmissible trial J ≤ 0 | raises `InadmissibleState` (code `invalid_deformation`); FE Newton rolls back, start displacement unchanged | exact |
| Affine patch on the grid element (Q4 plane stress/strain, H8; SVK and neo-Hooke) | interior nodal forces zero, face resultant = −t A P e₁ | ≤ 8.8e-15 |
| Rigid-rotation patch (to π) | zero nodal forces, det F = 1 | ≤ 4.7e-16 |
| Two-bar truss snap-through, step refinement ds = 0.02 … 0.0025 | closed-form P(w), limit points | curve 1.2e-13; limit point 3.3e-12 (rounding floor, no ds trend) |
| J2 hydrostatic strain | never yields, σ = 3Kc | 1.7e-16 |
| J2 yield kink | jump of the algorithmic tangent −2μ·3μ/(3μ+H)·(2/3) | equal to printed digits; stress C⁰ |

Earlier tests that already covered parts of N1–N5 (autograd tangent, FE small load, uniaxial tension, K_T vs FD, truss
arc length, load-control negative control, J2 closed forms, consistent tangent, energy identity, plastic sensitivities)
were not repeated; the record lists them.

## Not verified / not implemented

* **Distorted-mesh nonlinear element (N3).** There is no finite-strain element on a general Q4/T6/H8/T10 mesh; the grid
  kernel has a constant element Jacobian. Homogeneous deformation patches on distorted meshes, mesh convergence of such an
  element and "agreement with the grid path" are NOT VERIFIED. `fem/mesh` is small-strain only.
* **R07, the predictor at an exactly singular tangent, is fixed (R07_bordered); two neighbours are not.** When the
  predictor solve K⁻¹q fails, `arc_length` now takes the direction from the null vector of the bordered system
  [[K, −q], [Δxᵀ_prev, ψ² qᵀq Δλ_prev]] (orientation ⟨t, t_prev⟩ > 0), or from the previous increment where that matrix is
  singular too (a bifurcation point: the null space of [K, −q] has dimension two and no unique direction exists); a start
  without a previous increment uses the null vector of [K, −q] (oriented by `direction`; a bifurcation start without
  `start_tangent` still fails, nothing is invented). The corrector is unchanged, K is not regularised. Model A with
  ds = 0.01 (the N4b case) now passes the accepted point with K₁₁ = 0 exactly and stays on v = −λ/k_a to 0.0; the truss
  started exactly at its limit point follows the closed form to 9.1e-14 (1.1e-13 upward), leaves it downward, turns at the
  second limit point (λ_min = −0.9905 on the sampled path) and rises. The change is bitwise neutral wherever the predictor
  solve succeeds (sha256 fingerprints of the truss, model A and model S runs equal before and after); a nearly singular
  tangent with a tiny nonzero pivot keeps the old predictor, which is accurate there. The switch is the solver's own failure
  (`LinearSolveFailure` or a non-finite K⁻¹q), not a condition-number threshold; the rank decision on the dense bordered
  matrix (n + 1 ≤ 4000) uses singular values ≤ (n + 1) ε σ_max.
  Not fixed: (1) the strict xfails `test_a1_*` / `test_a2_*` of `test_bifurcation.py` still fail, now in `locate_critical`,
  which raises "crossing eigenvalue not bracketed" when the critical eigenvalue is exactly 0 at an accepted point; (2) the
  corrector is unchanged: a predicted point within a few ulps of a limit point (ψ = 0) can raise an uncaught
  `ZeroDivisionError` in the root-angle selection (2 of 6 sampled offsets); (3) `displacement_control` has the same
  `K⁻¹q` predictor and was not changed. Repeated-eigenvalue subspace diagnostics are NOT VERIFIED (not implemented).
  The truss test uses a test-only subclass that sets tangent entries ≤ 1e-12 ‖K‖₁ to 0.0 (the rounding of an exact zero at
  w₁); the truss cannot be driven onto its limit point by ds alone.
* **Finite-strain plasticity and viscoelasticity are NOT implemented.** J2 is small strain (no multiplicative split, no
  objective rate); replacing ε by a logarithmic strain is not a valid substitute and no claim is made. Viscoelastic and
  rate-dependent laws are small-strain history models outside `NonlinearElasticity` (it refuses `physical = True`).
* Updated Lagrangian formulation, follower loads, mixed finite-strain elements, anisotropic neo-Hooke, contact: not provided.
* Plane stress is verified for the isotropic neo-Hooke only; an anisotropic law would need the full Schur complement with
  out-of-plane shear.
* SVK is not robust under large compression (admissible region not enforced beyond J > 0); the tests use stretches
  0.6 to 2.5 for the closed forms and are not a statement about SVK at J → 0.

## Test-set-up note

A small H must be passed as H, not as (I + H) − I: forming F − I loses log₁₀(1/|H|) digits (first N1g run: 4.4e-13
instead of 1e-16). This is the cancellation documented in `core/constitutive.py`, not a defect.
