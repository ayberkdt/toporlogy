# CO — Topology optimisation under nonlinear Norton creep with an exact history adjoint (preregistration)

Preregistered 2026-11-27 03:23 +0300 (clock, written by the same command), before the module and its tests exist.
Base `6c45d19`. New module `src/toporlogy/optimization/creep_topology.py`. WP20 D2 (history adjoint
through a nonlinear material), WP13 C06 (structural creep objective).

## Model, interpolation, discretisation

Plane strain on a 2-D `Mesh`; per element the elastic moduli are K_e = w_e K0, G_e = w_e G0 with
w_e = E_min + x̄_e^p (1 − E_min); the creep law ṗ = A (q/σ_ref)^n keeps its physical A, σ_ref, n in every element
(no hidden scaling of the creep parameters by the design modulus, WP00 §4). Backward Euler radial return of
`core.creep.NortonCreep` (same equations), per point K_e, G_e.

## Adjoint

Per step n the converged displacement u_n solves R_n(u_n; w, ε_cr,n−1) = Σ_e ∫ Bᵀ σ dV − f_n = 0; forward Newton in
NumPy with the algorithmic tangent. The step is a custom autograd function: backward solves K_tᵀ λ = ū on the free
DOFs and returns −(∂R/∂w)ᵀλ and −(∂R/∂ε_cr,n−1)ᵀλ by torch autograd of R at the converged u_n (implicit function
theorem). The new creep strain ε_cr,n = Φ(ε(u_n), ε_cr,n−1, w) is evaluated in torch; its local scalar root Δp is
found numerically and then corrected by one Newton step written in torch, Δp = Δp* − g(Δp*)/g′(Δp*), which has the
exact first derivative of the root at Δp* (g(Δp*) = 0 to the local tolerance). The whole march is therefore
differentiated exactly on the fixed time grid. Objective J = f(t_s)ᵀ u(t_s).

## Checks

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| CO1 | the torch local update equals `NortonCreep.update` (stress, creep strain, Δp) for random strains and states, 3 n values (1, 3, 5) | production NumPy update (same equations, different implementation: consistency) | rel 1e-12 |
| CO2 | uniform design w = c (c = 1, 0.4) on a distorted Q4 8 × 4 cantilever, load ramp then hold, 10 steps to 1000 s: u at every step equals `HistoryProblem` with `NortonCreep(cK0, cG0, A, σ_ref, n)` | partially independent (different global algebra) | rel 1e-9 |
| CO3 | dJ/dx in a random direction (random grey x, seed 4) equals central FD | FD steps 1e-2…1e-7, best | rel ≤ 1e-6 |
| CO4 | negative control: detaching ε_cr,n−1 (no history derivative) changes the gradient by ≥ 1e-3 relative | — | ≥ 1e-3 |
| CO5 | MBB 30 × 10, V = 0.5, service time 1000 s, creep strong enough that the service compliance exceeds the elastic one by ≥ 20 % for the uniform design: the creep-optimised design (warm start from the elastic-compliance design, KKT stop 1e-4) has J(t_s) ≤ J(t_s) of the elastic design; the relative gain is reported | — | ≤ (no worse) |

Synthetic parameters: K0 = 3.33 GPa, G0 = 0.77 GPa, A = 1e-6 1/s, σ_ref = 10 MPa (n given per check); loads scaled
so the solid stresses are 5–20 MPa. Run: `python -m pytest tests/toporlogy/test_creep_topology.py -q -p no:randomly`.

## First result (recorded 2026-11-27 03:28 +0300 by the clock; runs between 03:25 and this time)

* CO1: FAILED on the first run for a **test set-up error** — the test built K, G with `torch.full((P,), K0)`, which
  is float32 (stress off by 9 Pa in 2e8); with float64 inputs CO1 PASS (n = 1, 3, 5) at the preregistered 1e-12.
* CO2 PASS: 5.9e-16 (c = 1 and 0.4) against `HistoryProblem` + `NortonCreep`.
* CO3 PASS: best FD error 7.1e-11 (h = 1e-6); errors 7e-4, 7e-6, 7e-8, 7e-10, 7e-11, 2e-9 for h = 1e-2 … 1e-7.
* **CO4 FAILED**: detaching the history changes the gradient by only 6.3e-4 (< 1e-3): at the 2e5 tip load creep is
  weak, so the negative control is insensitive. Strict xfail; ledger. Post hoc CO4b at 5× the load: 1.3e-2 and FD
  error 5.8e-10 (PASS, labelled post hoc).
* **CO5 not run as specified — the preregistered set-up is self-contradictory**: with A = 1e-6 1/s, σ_ref = 10 MPa
  and solid stresses 5–20 MPa the uniform design's service/elastic compliance ratio is 1.0007–1.018 (measured for
  F = 2e6, 5e6, 1e7), so the condition "≥ 20 %" cannot be met (the elastic moduli scale with w, the creep law does
  not). A set-up calibration (A = 1e-5, 1e-4; F = 5e6, 1e7: ratios 1.045, 1.17, 1.40, 2.39) was measured before the
  replacement preregistration CO6 (`CO6_creep_topology.md`); no CO5/CO6 optimisation was run before it.
