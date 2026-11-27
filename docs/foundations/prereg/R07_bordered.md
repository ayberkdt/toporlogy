# R07_bordered — bordered / previous-tangent predictor of `arc_length` at an exactly singular tangent (preregistration)

Preregistered 2026-11-27 09:03 +0300, base SHA `36c020f`, committed before any test of this record was run. Module:
`src/toporlogy/solvers/continuation.py` (`arc_length` predictor only). Tests:
`tests/toporlogy/test_continuation_singular.py`. Document: `docs/foundations/NONLINEAR_CONTINUUM.md`. Closes risk R07 as
measured by N4b in `prereg/N_nonlinear.md` (`arc_length` accepts the point (0, −0.01), λ = 1 of model A with ds = 0.01 where
K₁₁ = 0 exactly; the next predictor K⁻¹q fails, `ContinuationError` code `step_limit`, cause `singular_tangent`).

What was done before this record (no new test run): the existing code was read (`arc_length`, `_solve2`, N4a/N4b,
`test_bifurcation.py` A1/A2 and their strict xfails, `docs/vv_kutugu.json` entries with reason "tekil teğet"), and the
fingerprints below were computed with the unchanged code (a script in the scratchpad, not a test). Their clock time was
not captured; they precede the production change. A scan of ds near w₁ on the truss (setup search with the unchanged
code, not a test) showed that the truss cannot be driven onto the limit point by ds alone: with ψ = 0 the predicted and the
accepted point have the same v (the constraint Δxᵀ Δx = ds² and u = 0 fix it), so the corrector's first Newton solve is at
the singular K itself and either rejects the step or (some ds within a few ulps of w₁, ZeroDivisionError at the angle
selection of the two roots, 2 of 6 sampled ulp offsets (+2 and +3 ulp of w₁; a scan of ±3000 ulps reported the same exception for many offsets and never an accepted K₁₁ = 0.0)) raises an uncaught `ZeroDivisionError`. That corrector behaviour is
NOT changed here (task: corrector unchanged) and is reported as a finding. Consequence for the design of the truss tests:
the exactly singular truss point is used as a START point (below), and the mid-path exactly singular accepted point is
model A, where the predicted point is already on the path and the corrector needs no solve (N4b).

## Derivation (rule 11)

Residual R(x, λ) = f(x) − λ q, tangent K = ∂R/∂x, ∂R/∂λ = −q. An equilibrium path direction t = (tₓ, t_λ) satisfies
K tₓ − q t_λ = 0, i.e. t ∈ null [K, −q] (n × (n+1)). Metric of the arc-length constraint (Crisfield, ψ ≥ 0):
‖t‖_G² = tₓᵀtₓ + ψ² qᵀq t_λ²; the predictor is Δ = ds t/‖t‖_G, so ΔxᵀΔx + ψ²Δλ² qᵀq = ds² as in the old predictor.

* Regular K: t ∝ (K⁻¹q, 1) (the old predictor, unchanged).
* Bordered system with the previous increment p = (pₓ, p_λ) (‖·‖_G, from the last accepted step or `start_tangent`):
  B = [[K, −q], [pₓᵀ, ψ² qᵀq p_λ]], B t = e_{n+1}. Then K tₓ − q t_λ = 0 and ⟨p, t⟩_G = 1 > 0, so the orientation
  (tᵀ t_prev > 0) is fixed by construction. B is nonsingular iff null [K, −q] is one-dimensional and ⟨p, ·⟩_G does not
  vanish on it: a limit point (q ∉ range K, nullity of K = 1) gives a one-dimensional null space (t = (φ, 0) with Kφ = 0),
  B regular. A bifurcation point (q ∈ range K) gives a two-dimensional null space, B singular: no unique tangent exists
  and none is invented; the fallback is the previous increment p itself (Δ = ds p/‖p‖_G), the continuation of the path
  that arrived, with the corrector unchanged.
* Without a previous increment (exactly singular start, no `start_tangent`): the null vector of [K, −q] by SVD when the
  numerical rank is n (null space one-dimensional), t scaled by ‖·‖_G. Orientation: sign of t_λ·`direction` when
  |t_λ| ‖q‖ > 1e-12 ‖K‖₁ ‖tₓ‖ (a real load component), otherwise (a limit point, t_λ = 0) the sign of the largest
  component of tₓ times `direction` (the convention of `_signed` in `locate_critical`). Null dimension ≥ 2 (bifurcation
  start) → no predictor, the old failure (no arbitrary unique mode).

Switch criterion (what is "singular"): the predictor solve K⁻¹q raises `LinearSolveFailure` or returns a non-finite
vector, i.e. exactly the cases in which the old predictor produced no step. Where the solve succeeds the old code runs
unchanged, so the change is bitwise neutral wherever K is not singular to the linear solver (a pivot test of the solver
itself; a nearly singular K with a tiny nonzero pivot keeps the old predictor, which is accurate there: dλ = ds/‖K⁻¹q‖
is tiny and Δx = dλ K⁻¹q is the null direction with norm ds). Numerical rank of B (and of [K, −q]): singular values
σ ≤ (n+1) ε σ_max count as zero (numpy `matrix_rank` rule). Dense only: n + 1 ≤ 4000 (a ≤ 128 MB matrix); above that the
bordered system is skipped and the previous increment is used (a start without an increment then fails as before).

## Checks

| ID | Hypothesis | Reference (independent of production code) | Threshold |
|---|---|---|---|
| R1 | model A (N4b: U = ½k_a e² + ½k_r u² + (β/6)u³, e = v + u²/2L, k_a = 100, k_r = 1, β = 3, L = 1, q = (0, −1)), ds = 0.01, from (0,0): the first step is Δλ = 1 to (0, −0.01) with K₁₁ = 1 + k_a v = 0.0 exactly (precondition, asserted); B has a zero row (bifurcation) → previous-increment predictor; the path continues on the primary path v = −λ/k_a, u = 0. Points λ_k = k (k = 0, 1, 2; stop at λ ≥ 1.5, N4b/A-test stop). Predictor of step 2 = (0, −0.01, 1) from p = (0, −0.01, 1)/‖(0,−0.01)‖·ds. | the closed form of the primary path (R_v = k_a v + λ = 0) | no `ContinuationError`; 3 points; abs(λ_k − k) ≤ 1e-12; abs(v_k + λ_k/k_a) ≤ 1e-12; abs(u_k) ≤ 1e-14; every step has Δλ > 0 and Δv < 0; record of step 2 carries the note "predictor: previous increment" |
| R2 | truss (N4: a = 1, h = 0.1, EA = 1, P̂ = P_max; closed form P(w) = EA w (2h − w)(h − w)/L³, w₁ = h(1 − 1/√3), K_vv ∝ P′(w), K_vv(w₁) = 0 exactly in exact arithmetic; the float value ~1e-18 is the rounding of an exact zero and is set to 0.0 by a test-only subclass for entries ≤ 1e-12 ‖K‖₁, which changes no other entry) started AT the limit point x₀ = (0, −w₁), λ₀ = P(w₁)/P̂ = 1, with `start_tangent` = (0, ∓1; 0), ds = 0.01 to w ≥ 2.5 h: B regular (det = K_uu P̂ ≠ 0), t = (0, ∓1, 0): the first step moves along the start tangent (Δv = ∓ds, |Δλ| ≤ 1e-12 predicted), the path leaves the limit point on the closed form; for the downward start (w increasing) λ decreases (the limit point is a load maximum), then turns at the second limit point (λ_min ≈ −1) and increases. | closed form P(w)/P̂ | max abs(λ − P(w)/P̂) ≤ 1e-9 (the N4a threshold); abs(u) ≤ 1e-12; sign of the first Δv equals the start tangent's; for w increasing: λ₁ < 1, exactly one interior minimum of λ, −1 − 1e-9 ≤ λ_min ≤ −0.97 (sample spacing: ½ ·|P″/P̂|(ds/2)² ≈ 0.012 from the closed form P″, derived before the run) and the last λ > λ_min |
| R3 | the same exactly singular truss start WITHOUT `start_tangent`: null vector (0, 1, 0) of [K, −q] = [[K_uu, 0, 0], [0, 0, P̂]] (rank 2 = n), t_λ = 0, so the orientation is the largest-component rule × `direction`: direction = +1 → v increases (w decreases toward the origin), direction = −1 → v decreases. | closed form P(w)/P̂ | max abs(λ − P(w)/P̂) ≤ 1e-9; abs(u) ≤ 1e-12; sign of Δv₁ = direction·(+1); |Δv₁| = ds within 1e-12; λ₁ < 1 |
| R4 | exactly singular BIFURCATION start: model A with k_r = 0 at the origin (K = diag(0, k_a), q ∈ range K, null [K, −q] two-dimensional): (a) without `start_tangent` the old failure remains (ContinuationError, step_limit, singular_tangent; nothing invented); (b) with `start_tangent` = ((0, −0.01), 1) the path continues on v = −λ/k_a through the singular start | the closed form of the primary path | (a) exact code/cause; (b) abs(v_k + λ_k/k_a) ≤ 1e-12, abs(λ_k − k) ≤ 1e-12, λ ≥ 1.5 reached |
| R5 | neutrality: where K is regular the path with the new code equals the path with the fix disabled BITWISE (all accepted points: λ, x, ds, iterations, residual, index, det sign; every record's code, notes, residuals, backward errors, updates; `rejected`): truss ds = 0.01 (N4a run, 26 points, with the dense spectrum) and model A ds = 0.0137 (A post hoc run, 3 points) and model S ds = 0.01. Cross-check by a sha256 fingerprint of the unchanged code (values below) recomputed after the change. | the same run with the fix disabled; the fingerprints of the code at `36c020f` | `np.array_equal` / equal reprs; fingerprints equal |
| R6 | negative control: with the fix disabled (the new helper replaced by one returning no predictor) the old failure reappears in R1 (model A ds = 0.01: ContinuationError, step_limit, singular_tangent, 2 accepted points, last λ = 1 within 1e-12), in R2 and in R3 (step_limit / singular_tangent at the start) | the N4b record | exact code/cause |
| R7 | promotion of the strict xfails `test_a1_asymmetric_bifurcation_located_and_classified` and `test_a2_asymmetric_branch_switch[plus|minus]` (ds = 0.01): they are promoted only if they pass unchanged after the fix (their thresholds are those of `catallanma_onkayit.md`, untouched); otherwise they stay xfail with the measured reason updated by the integrator | the closed-form model A (A1/A2 of the bifurcation record) | their own thresholds, unchanged |

N4b (`test_nonlinear_foundations.py::test_n4b_*`) asserts the old failure and would turn false with the fix. It is kept
as the characterisation of the old behaviour by running it with the fix disabled (same asserts, same thresholds), and its
record note is updated. No threshold changes.

Fingerprints of the unchanged code (`36c020f`; sha256 over the raw float64 bytes of every accepted point's λ, x, ds,
iterations, residual and the repr of det sign / index, and the repr of every record's load factor, converged, code,
rolled_back, notes, residuals, backward errors, updates, plus `rejected`):

| Run | points / rejected | sha256 |
|---|---|---|
| truss, ds = 0.01, rtol 1e-12, eta_tol 0, stop −v ≥ 2.5 h, dense spectrum | 26 / 0 | `01cca0230060f690504c58f74b356e253a73c9cbbecb59212de97adbe57fa839` |
| model A, ds = 0.0137, rtol 1e-12, stop λ ≥ 1.5, dense spectrum | 3 / 0 | `42ceadb4a916a7b7badbd67133cd7c5dcc6151d129cf60fcfa6ae4a127d020a9` |
| model S, ds = 0.01, rtol 1e-12, stop λ ≥ 1.5, dense spectrum | 3 / 0 | `054a2f96d2569b719990bd74f150e8072c74c748e3a4f96eaa9abc602c6499b7` |

Out of scope, not changed: `displacement_control` (its predictor has the same `K⁻¹q` and the same latent defect, with
its own bordered corrector), the corrector (including the uncaught `ZeroDivisionError` described above), repeated
eigenvalue subspace diagnostics, branch switching from a singular accepted point.

Run: `python -m pytest tests/toporlogy/test_continuation_singular.py -q -p no:randomly` (plus the directly related
`test_nonlinear_foundations.py -k n4`, `test_bifurcation.py`, `test_nonlinear_solvers.py`). First result appended below.
