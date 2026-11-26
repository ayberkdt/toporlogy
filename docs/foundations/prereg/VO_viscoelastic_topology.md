# VO — Topology optimisation of a linear viscoelastic structure with an exact history adjoint (preregistration)

Preregistered 2026-11-27 02:05 +0300 (clock), before the module and its tests exist. Base: `foundations/round-3`
after the WP12 merge (GeneralizedMaxwell of `core/viscoelastic.py`). New module
`src/toporlogy/optimization/viscoelastic_topology.py`. WP20 D1/D2 (history gradients) and WP17 C06 (structural
creep objective), small strain, isothermal, fixed accepted time grid.

## Model and derivation

Material: one generalized Maxwell solid (independent shear and bulk Prony series, physical moduli in Pa, times in
s). Design interpolation: in element e every modulus of both spectra is multiplied by w_e = E_min + x̄_e^p (1 − E_min)
with the relaxation times unchanged. Justification: a solid/void mixture of one viscoelastic phase scales stiffness,
not time; no yield stress, viscosity-like parameter or relaxation time is scaled independently. Under this
interpolation the branch stresses are s_i = w_e ŝ_i where ŝ_i obeys the WP12 recursion with unit scaling, so the
element response is w_e × (the unscaled law applied to the element's strain history).

Discrete problem (plane strain on a general 2-D `Mesh`, Q4/T3/…; the same recursion as
`GeneralizedMaxwell.update`, Δξ = Δt): for each step n → n+1 with r_i = e^{−Δt/τ_i}, φ_i = (1 − r_i)/(Δt/τ_i)

    K_alg(w) (u_{n+1} − u_n) = f_{n+1} − f_int,hist,n(w),
    K_alg(w) = Σ_e w_e ∫ Bᵀ C_alg B dV,  C_alg = 2(G∞ + Σ G_i φ_i) P_dev + 3(K∞ + Σ K_j φ_j) P_vol,
    f_int,hist,n = Σ_e w_e ∫ Bᵀ [2G∞ dev ε_n + K∞ tr ε_n 1 + Σ r_i ŝ_i,n + Σ r_j p̂_j,n 1] dV,

ŝ_{i,n+1} = r_i ŝ_{i,n} + 2G_i φ_i Δe, p̂_{j,n+1} = r_j p̂_{j,n} + K_j φ_j Δθ (exact for strain linear in each step).
Objective: compliance at the service time, J = f(t_s)ᵀ u(t_s), and optionally the time average Σ_n ω_n f_nᵀ u_n.
Gradient: reverse-mode through the whole recursion (torch autograd) with a custom linear-solve function whose
backward is K_alg λ = ḡ (K_alg symmetric), dJ/dw_e = −λ_eᵀ K_e^alg (u_{n+1} − u_n)_e plus the history terms that
autograd carries through ŝ, p̂ and ε_n. This is the exact derivative of the discrete objective on the fixed grid.

## Checks

| ID | Hypothesis | Reference (independence) | Threshold |
|---|---|---|---|
| VO1 | forward: uniform design w_e = c (c = 1 and c = 0.37) on a distorted Q4 mesh (8 × 4, seed 3), creep load ramp + hold, 12 irregular steps: u at every step equals `fem.mesh.history.HistoryProblem` with `GeneralizedMaxwell` whose moduli are scaled by c | different implementation of the global algebra and of the material update (NumPy, Newton) — partially independent (shares the published recursion) | rel 1e-10 (max over steps of ‖Δu‖∞/‖u‖∞) |
| VO2 | gradient of J(t_s) w.r.t. the 32 design variables (random grey x, seed 4, SIMP p = 3) equals central finite differences in a random direction | central FD, steps 1e-2…1e-7 (6 steps), best step | rel ≤ 1e-6 at the best step |
| VO3 | proportional spectra (K_j/K∞ = G_i/G∞ with equal times): u_n = α_n u_el(w) with α_n a scalar sequence independent of the design, hence ∇J(t_s) = α_s ∇c_el where c_el is the elastic compliance with the instantaneous moduli; checked as ‖∇J − (J/c_el) ∇c_el‖ / ‖∇J‖ | derived above; c_el and ∇c_el from an independent elastic solve (`ElasticProblem`) and the self-adjoint formula | ≤ 1e-10 |
| VO4 | non-proportional spectra (bulk relaxes, shear does not: G(t) = G∞ = 1 MPa, K(t) = 0.5 + 4.5 e^{−t/10 s} MPa): the same ratio test gives ≥ 1e-3 (the time dependence is not a scalar factor) | — | ≥ 1e-3 |
| VO5 | negative control: the gradient with the history state detached (stop-gradient on ŝ, p̂, ε_n) differs from the full gradient by ≥ 1e-3 relative in the VO2 set-up | full gradient | ≥ 1e-3 |
| VO6 | MBB half-beam 30 × 10 Q4, volume 0.5, filter radius 1.5 h, 40 MMA iterations, non-proportional set of VO4, service time t_s = 100 s: the design minimising J(t_s) and the design minimising the instantaneous compliance J(0⁺) differ by mean |Δx̄| ≥ 0.01; both runs keep the volume constraint ≤ 0.5 + 1e-3 | — (measurement with a stated hypothesis) | as stated |

Run: `python -m pytest tests/toporlogy/test_viscoelastic_topology.py -q -p no:randomly`. First result appended below.

## First result (run 2026-11-27 02:07 +0300 by the clock, after the VO module was written)

`test_viscoelastic_topology.py`: 7 passed on the first run (VO1 for c = 1 and 0.37, VO2–VO6). Measured values
(`docs/foundations/VO_measurements.json`, script `scripts/vo_measure.py`):

| ID | measured | threshold |
|---|---|---|
| VO1 | 6.0e-14 (c = 1), 1.6e-14 (c = 0.37) | 1e-10 |
| VO2 | best FD error 5.5e-10 (h = 1e-6); 6.7e-10 at 1e-5; second-order FD truncation visible from 1e-2 to 1e-4 | 1e-6 |
| VO3 | 8.5e-14 (proportional spectra: ∇J = (J/c_el) ∇c_el) | ≤ 1e-10 |
| VO4 | 0.146 (non-proportional: no scalar factor) | ≥ 1e-3 |
| VO5 | 3.78 (detached history gradient is wrong by 378 %) | ≥ 1e-3 |
| VO6 | mean abs Δx̄ = 0.068 at 40 iterations; volumes 0.49977, 0.49999 | ≥ 0.01; ≤ 0.501 |

**VO6 pass is not evidence of a better design.** At 40 iterations the design optimised for the service time has a
*higher* service-time compliance (1.2502e-4) than the instantaneous-compliance design (1.2118e-4): the service-time
run had not converged (last max design change 0.133).

**Post hoc study** (`scripts/vo6_posthoc.py`, `docs/foundations/VO6_posthoc.json`; not preregistered):

| run | J(t_s = 100 s) | J(0⁺) | mean abs Δx̄ between the two designs |
|---|---|---|---|
| 40 it, service-time objective | 1.2502e-4 | 7.677e-5 | 0.068 |
| 40 it, instantaneous objective | 1.2118e-4 | 7.322e-5 | |
| 120 it, service-time objective (last change 4.3e-3) | **1.20917e-4** | 7.335e-5 | 0.022 |
| 120 it, instantaneous objective (last change 1.3e-6) | 1.21162e-4 | **7.318e-5** | |
| warm start from the 40-it instantaneous design, 80 it service-time | 1.20916e-4 | 7.335e-5 | |

At convergence the service-time design is 0.20 % better at t_s and 0.18 % worse at 0⁺ than the instantaneous
design: a real but small trade-off for this synthetic bulk-relaxing material on this MBB beam. The design
difference that VO6 measured at 40 iterations was mostly non-convergence. Follow-up: a convergence-based stop
for VO-type runs and a material with stronger non-proportionality before any claim about design benefit.
