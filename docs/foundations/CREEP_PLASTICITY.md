# Coupled creep and rate-independent plasticity (`CreepPlasticJ2`)

Module `src/toporlogy/core/creep_plasticity.py`, tests `tests/toporlogy/test_creep_plasticity.py`, preregistration
`docs/foundations/prereg/C8_creep_plasticity.md` (derivations, thresholds, first results). WP13 C04/C08, WP14 R01.
All numbers in the tests are **synthetic**; nothing here is material data.

## Decomposition and laws

Small strain, isotropic elasticity (K, G in Pa), ε = ε_e + ε_cr + ε_p, both inelastic strains deviatoric, no thermal strain
(ε_th is not part of this slice). With q = √(3/2 s:s) and the common direction N = 3s/(2q):

| mechanism | law | internal variables |
|---|---|---|
| creep (irreversible, J2) | ṗ_cr = A f(T) (q/σ_ref)^n, ε̇_cr = ṗ_cr N, active for every q > 0 | ε_cr, p_cr |
| plasticity (rate independent, J2, linear isotropic hardening) | f_y = q − (σ_y + Hα) ≤ 0, α̇ ≥ 0, α̇ f_y = 0, ε̇_p = α̇ N | ε_p, α |

The two mechanisms are different physical processes with their own variables; neither is represented twice (creep is not
a yield overstress: the plastic branch has no rate term, and `Perzyna` is a separate model that must not be combined with
this one for the same mechanism). Parameters: `CreepPlasticJ2(K, G, A, sigma_ref, n, yield_stress, hardening, arrhenius=None)`
with A [1/s] ≥ 0, σ_ref [Pa], n > 0, σ_y [Pa] > 0, H [Pa] ≥ 0; `arrhenius` is the :class:`ArrheniusFactor` of
`core/creep.py` (no implicit unit conversion; the step temperatures are then required). **σ_y = ∞ is a flag**
(`yield_stress=None` or `math.inf`), not a large number: the plastic branch is skipped exactly. The physical time step
(`MaterialRequest(step=TimeStep(...))`) is required in every request; Δt = 0 gives a purely plastic update.
State fields: `creep_strain`, `p_cr`, `plastic_strain`, `alpha`.

## One local residual (no sequential splitting)

Backward Euler, radial (both flows are along the trial direction N = 3 s_tr/(2 q_tr), s = θ s_tr). Unknowns x = Δp_cr ≥ 0,
y = Δα ≥ 0, Λ = x + y, end stress q = q_tr − 3GΛ, h(q) = Δt A F̄ (q/σ_ref)^n (F̄: step mean of the Arrhenius factor):

    r1 = x − h(q) = 0,     plastic branch: r2 = q − κ_n − H y = 0   (κ_n = σ_y + H α_n),   y ≥ 0, f_y ≤ 0, y f_y = 0.

Newton Jacobian of (r1, r2) in (x, y): [[1 + 3G h', 3G h'], [−3G, −(3G + H)]]. r2 is linear, so y is eliminated exactly,
y = (q_tr − κ_n − 3G x)/(3G + H); the Schur complement is the scalar g(x) = x − h(q(x)), q(x) = κ_n + w (q_tr − κ_n − 3Gx),
w = H/(3G + H), g' = 1 + 3G w h' ≥ 1, solved by the safeguarded Newton/bisection of `core/creep.py` (bracketed, unique root).

Active set (derived, not assumed): the creep-only root x_c (y = 0) gives q_c = q_tr − 3G x_c; the step is plastic iff
q_c > κ_n. Then q(x_c) < q_c, so the plastic root has x ≤ x_c and y ≥ (q_c − κ_n)/(3G + H) > 0 (complementarity holds);
otherwise the creep-only solution satisfies all KKT conditions. In the plastic branch q_{n+1} = σ_y + Hα_{n+1} holds to
rounding for any x (y is computed from x by the linear equation). H = 0: x = h(σ_y), y = (q_tr − σ_y)/(3G) − x.

A sequential update (creep, then plastic return) evaluates the creep law at the trial stress outside the yield surface;
at Δt = 20 s from a virgin state with q_tr = 3e7 Pa (synthetic) it overestimates Δp_cr by a factor 3.2 and its end state
violates the creep residual by 71 % (test C8l). Starting on the yield surface at a fixed strain rate the splitting error
falls first order with Δt (ratio 4.1-4.4 per factor 4 in the test), so it is a consistent but not an exact scheme.

## Algorithmic tangent

With s = θ s_tr, θ = 1 − 3GΛ/q_tr, dq_tr = 2G N:dε, Λ' = dΛ/dq_tr:

    C = K 1⊗1 + 2Gθ I_dev − 4G² (Λ' − Λ/q_tr) N⊗N      (the −Λ/q_tr term is the change of direction of N)

creep-only branch Λ' = h'/(1 + 3G h'); plastic branch x' = w h'/(1 + 3G w h'), y' = (1 − 3G x')/(3G + H),
Λ' = x' + y' = 1/(3G + H) + w x'. A frozen-direction tangent and a plastic tangent that ignores creep (Λ' = 1/(3G + H)) are
wrong and are detected by the FD check (test C8g). The tangent is one-sided at the switch q_c = κ_n (as for any
plasticity). Only the algorithmic tangent is defined (`tangent_kind="continuum"` raises).

## Energy and dissipation

ψ = ½ σ:(ε − ε_cr − ε_p) + ½Hα². Returned dissipation D = q(Δp_cr + Δα) − Δψ_h = D_cr + D_p with D_cr = q Δp_cr ≥ 0 and
D_p = q Δα − Δψ_h = σ_y Δα + ½ H Δα² ≥ 0 (diagnostics `dissipation_creep`, `dissipation_plastic`); the inelastic work
q(Δp_cr + Δα) = σ:Δε_in is `inelastic_work`. There is no stored creep energy.

## Limits and what is verified

* A = 0 is `PhysicalJ2` with linear hardening: agreement ≤ 2e-15 relative (not bitwise: s_tr and the plastic increment are
  rounded differently).
* σ_y = ∞ is `NortonCreep`: bitwise (stress, tangent, energy, dissipation, state), isothermal and with an Arrhenius ramp.
* Constant uniaxial stress above yield: instantaneous plastic strain (σ_0 − σ_y)/H, then plasticity stops (q = κ, q̇ = 0 ⇒ α̇ = 0)
  and creep runs at the Norton rate; agrees with the closed form to 3.8e-14 (T = 360 K) and with a `quad` Arrhenius ramp to 1.6e-14.
* Held-strain relaxation from a plastic state and a ramp-hold history (plastic and creeping together) against independent
  `solve_ivp` rate systems with active-set events: first-order convergence, errors 6e-5 to 8e-4 at Δt = 0.025-0.03 s.
* Tangent against central FD in the plastic-and-creeping and the creep-only state: ≤ 2.7e-11.
* Separation: Δε_cr = Δp_cr N, Δε_p = Δα N, each counted once; stress from the dense elastic law of ε − ε_cr − ε_p.
* Rollback and restart (mixed control, Arrhenius ramp, plastic and creeping) bitwise.

## Not implemented / limits of validity

Nonlinear (Voce) hardening, kinematic hardening and softening; thermal strain ε_th; temperature-dependent K, G, σ_y or H;
rate-dependent (Perzyna/Eyring) plasticity combined with creep; creep damage, tertiary creep, rupture; finite strain;
pressure-sensitive yield or creep; recoverable (viscoelastic) creep (a different mechanism, never added by default).
The creep constants A, σ_ref, n and the plastic constants σ_y, H must be calibrated to disjoint parts of the data (a
constant-load creep test above yield contains both); the model does not decide which mechanism explains a given curve.
`core/creep.py` still lists "creep-plasticity coupling" in `NOT_IMPLEMENTED`; that entry predates this module and is
left to the integrator (the C8 check of `test_creep.py` asserts it).
