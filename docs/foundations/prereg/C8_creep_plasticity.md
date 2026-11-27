# C8_creep_plasticity — coupled J2 creep + rate-independent plasticity, one local residual (preregistration)

Preregistered 2026-11-27 06:27 +0300, before any test of this slice was run. Base SHA `6385cfe` (branch
`worktree-agent-a83cb5a75b182c13e`, fast-forwarded to it). Responsible module: `core/creep_plasticity.py` (new, class
`CreepPlasticJ2`). Tests: `tests/toporlogy/test_creep_plasticity.py`. Doc: `docs/foundations/CREEP_PLASTICITY.md`.
Plan: WP13 C04 (`13_CREEP.md`: "solve coupled local residual with consistent tangent and energy, not sequential
independent updates unless splitting error quantified; avoid duplicating a mechanism with multiple laws") and C08 row of
`23_ACCEPTANCE_CASES.md` ("creep/plastic interaction only if derived; distinct mechanisms, energy and no double
counting"), WP14 R01 (physical J2: yield and hardening in Pa, no E scaling). All constants are synthetic, nothing here is
material data. This slice does not edit `core/creep.py`; its `NOT_IMPLEMENTED` entry for creep-plasticity (and the C8
test of `test_creep.py` that asserts it) is now stale and is left for the integrator.

## 1. Model and derivations (what is tested)

Small strain, isotropic elasticity (K, G in Pa), ε = ε_e + ε_cr + ε_p, ε_cr and ε_p deviatoric,
σ = K tr ε 1 + 2G dev(ε − ε_cr − ε_p). q = √(3/2 s:s), N = 3s/(2q) (N:N = 3/2, s:N = q). Flow, both along the SAME N:

    ε̇_cr = ṗ_cr N,  ṗ_cr = A f(T) (q/σ_ref)^n  (always active when q > 0, no threshold),
    ε̇_p  = α̇ N,    f_y = q − κ(α) ≤ 0, α̇ ≥ 0, α̇ f_y = 0,  κ(α) = σ_y + Hα  (H ≥ 0),

one set of internal variables (ε_cr, ε_p, p_cr, α), no mechanism is represented twice (creep is not yield overstress, so
the plastic branch carries no rate term; Perzyna is a different model). σ_y → ∞ is a flag (`yield_stress=None` or
`math.inf`): the plastic branch is skipped, no tolerance is used to represent ∞.

**Backward Euler, one local system.** Trial: s_tr = 2G dev(ε − ε_cr,n − ε_p,n), q_tr = √(3/2)‖s_tr‖, N = 3s_tr/(2q_tr)
(both flows are radial, the direction of s_tr is kept, s = θ s_tr). Unknowns x = Δp_cr ≥ 0, y = Δα ≥ 0, Λ = x + y,
q = q_tr − 3GΛ, h(q) = Δt A F̄ (q/σ_ref)^n (F̄ the step-mean Arrhenius factor as in `NortonCreep`). Residuals

    r1 = x − h(q),   plastic branch only: r2 = q − κ_n − H y = 0 (κ_n = σ_y + Hα_n), complementarity y ≥ 0, f_y ≤ 0, y f_y = 0.

Full 2×2 Jacobian of (r1, r2) in (x, y): [[1 + 3G h', 3G h'], [−3G, −(3G + H)]]. r2 is linear: eliminating
y = (q_tr − κ_n − 3G x)/(3G + H) gives q(x) = κ_n + w (q_tr − κ_n − 3G x), w = H/(3G + H), i.e. the scalar
g(x) = x − h(q(x)), g' = 1 + 3G w h' ≥ 1 (the Schur complement of the 2×2 system, so the scalar Newton step IS the 2×2
Newton step). Active set (derived, not assumed): solve the creep-only problem (y = 0) for x_c, q_c = q_tr − 3G x_c. The
branch is plastic iff q_c > κ_n. Proof that this is consistent: if q_c > κ_n then q(x_c) < q_c (since w < 1) so
g(x_c) ≥ 0 hence the plastic root x_p ≤ x_c and y_p = (q_tr − κ_n − 3G x_p)/(3G + H) ≥ (q_c − κ_n)/(3G + H) > 0; if
q_c ≤ κ_n the creep-only solution satisfies all KKT conditions. For H = 0: q = κ_n = σ_y, x = h(σ_y),
y = (q_tr − σ_y)/(3G) − x. The scalar root is found by the safeguarded Newton/bisection of `core/creep.py` (bracket
[0, min(cap, h)], g(0) ≤ 0, g(h(q(0))) ≥ 0). The consistency q_{n+1} = σ_y + Hα_{n+1} holds to rounding in the plastic
branch for ANY x, because y is computed from x by the linear equation.

**Limits.** A = 0: h ≡ 0, x = 0, y = (q_tr − κ_n)/(3G + H) = PhysicalJ2 (linear hardening). σ_y = ∞: NortonCreep.

**Algorithmic tangent.** s = θ s_tr, θ = 1 − 3GΛ/q_tr; with dq_tr = 2G N:dε, ds_tr = 2G I_dev dε and Λ' = dΛ/dq_tr:

    C = K 1⊗1 + 2Gθ I_dev − 4G² (Λ' − Λ/q_tr) N⊗N     (the −Λ/q_tr term is the change of direction of N),

creep-only branch Λ' = x' = h'/(1 + 3G h') (= NortonCreep), plastic branch x' = w h'/(1 + 3G w h') (g_x x' + g_{q_tr} = 0,
g_{q_tr} = −w h'), y' = (1 − 3G x')/(3G + H), Λ' = x' + y' = 1/(3G + H) + w x'. Checks: h' = 0 gives
4G²(1/(3G+H) − Λ/q_tr) N⊗N, which is PhysicalJ2's 2G(3G/(3G+H) − (1 − θ)) n̂⊗n̂ because n̂⊗n̂ = (2/3) N⊗N and
1 − θ = 3GΛ/q_tr; H = 0 gives x' = 0, y' = 1/(3G). The tangent is one-sided at the switch q_c = κ_n (as for any
plasticity); FD checks stay away from it.

**Dissipation and energy.** ψ = ½ σ:(ε − ε_cr − ε_p) + ½Hα². Returned dissipation D = q(x + y) − ½H(α_{n+1}² − α_n²)
= D_cr + D_p, D_cr = q x ≥ 0, D_p = q y − Δψ_h = σ_y y + ½ H y² ≥ 0 (q = σ_y + H(α_n + y)); the inelastic work
q(x + y) = σ:Δε_in (s:N = q) is a diagnostic. Equals PhysicalJ2 (A = 0) and NortonCreep (σ_y = ∞) in the limits.

## 2. Constants

K = 1.7e10, G = 8e9 (E = 9KG/(3K + G) = 2.0746e10, ν = (3K − 2G)/(2(3K + G)) = 0.29661), A = 1e-6 1/s, σ_ref = 1e7,
n = 3, σ_y = 1.5e7, H = 2e9 unless a check says otherwise. Voigt order [xx, yy, zz, yz, xz, xy]; uniaxial tension has
N = diag(1, −½, −½). "rel" = max-norm error / max-norm of the reference. References are closed forms derived below,
`scipy.integrate.solve_ivp` (DOP853, rtol 1e-12, atol 1e-18), `scipy.integrate.quad`, `scipy.optimize.brentq`, dense
numpy in the test; a second call of production kernels is labelled "regression".

## 3. Checks

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| C8a | σ_y = ∞ (`None` and `math.inf`) reproduces `NortonCreep` (**regression, not independent**): stress, tangent, energy, dissipation, creep_strain, p_cr, plastic_strain = 0, alpha = 0, for 3 random batches of 40 multiaxial states with nonzero creep strain, isothermal and with an Arrhenius ramp 300→340 K; plus a 40-step `drive_material_point` history | `NortonCreep.update` | `np.array_equal` (bitwise) |
| C8b | A = 0 reproduces `PhysicalJ2` (**regression**): 3-cycle multiaxial strain-controlled path of 36 steps through `drive_material_point` (H = 2e9 and H = 0), and 50 random states with `update` directly (stress, tangent, energy, dissipation, plastic_strain, alpha); creep_strain and p_cr stay exactly 0 | `PhysicalJ2` (K, G, σ_y, H) | rel ≤ 1e-13 each (bitwise reported, not required); creep arrays exactly 0 |
| C8c | Constant uniaxial stress σ0 = 2e7 > σ_y, H = 2e9, stress controlled (all six components, `drive_material_point`, tol 1e-6 Pa), first step Δt = 0 (load), then holds to t = 5, 25, 100 s. Derivation: Δt = 0 gives creep 0 and a pure plastic return with q = σ0 = κ(α0): α0 = (σ0 − σ_y)/H = 2.5e-3. During a hold the stress is held, q = σ0 = κ(α0) so f_y = 0 and q̇ = 0 ⇒ Hα̇ = 0 (consistency) ⇒ α̇ = 0, plasticity stops; creep runs at the constant rate c = A f(T)(σ0/σ_ref)^n = 8e-6 1/s (T_ref). In the discrete scheme q_{n+1} = σ0 is imposed, so Δp_cr = Δt c exactly and Δα = 0 up to rounding. Closed forms: axial total strain σ0/E + α0 + c t; lateral −νσ0/E − α0/2 − c t/2; ε_cr axial c t; ε_p axial α0; p_cr = c t; α = α0; dissipation of the load step σ_y α0 + ½Hα0² (= q α0 − ½Hα0²), of a hold step σ0 c Δt. Variants: isothermal T = 360 K with Arrhenius (Q = 5e4 J/mol, T_ref = 300 K, c multiplied by f(360) written in the test); T ramp 300→400 K over 10 s after the load step with k = 1 and 4 steps: p_cr(10 s) = A (σ0/σ_ref)^n ∫_0^10 f(T(t)) dt by `quad` (epsabs = epsrel = 1e-13), and the step average makes it exact per step | closed forms, `quad` | rel ≤ 1e-9 on every listed quantity (α and ε_p unchanged to rel 1e-9 of α0); compression σ0 = −2e7 gives the sign-flipped strains and positive p_cr, α (rel 1e-9) |
| C8d | Held-strain relaxation from a plastic state. ε = e1 diag(1, −½, −½), e1 = 3e-3 applied in a Δt = 0 step (q_tr = 3G e1 = 7.2e7): α0 = (3G e1 − σ_y)/(3G + H), q0 = σ_y + Hα0 (= 1.9385e7). Then held 60 s with Δt = 0.025 s. Rate argument: with the held strain q̇ = 3G(ė − ṗ_cr − α̇) = −3G(ṗ_cr + α̇); in the plastic mode q̇ = Hα̇ ⇒ α̇ = −3G ṗ_cr/(3G + H) < 0, impossible, so α̇ = 0 and q relaxes by creep alone: q(t) = [q0^(1−n) + (n − 1) 3G A t/σ_ref^n]^(1/(1−n)), p_cr = (q0 − q)/(3G). Independent second reference: `solve_ivp` of the system (p, α) with the active-set logic of §4. Production: q at 60 s and p_cr at 60 s vs closed form; α and ε_p after the load step stay bitwise constant (plastic flag 0 in every hold step); closed form vs `solve_ivp` agree; BE order from Δt = 0.2, 0.1, 0.05 s on q(60) | closed form, `solve_ivp` | rel ≤ 2e-3 at Δt = 0.025 (prediction ~1e-4, as C3b of `C_creep`); closed form vs ivp rel ≤ 1e-9; α, ε_p `np.array_equal`; the two observed orders log2(e_k/e_{k+1}) in [0.9, 1.1] |
| C8e | Plastic AND creeping together: strain ramp e(t) = ė t, ė = 1e-4 1/s on [0, 30 s] (diag(1, −½, −½) direction, fully strain controlled), then held on [30, 60]; production with N = 250, 500, 1000 equal steps per phase. Reference: `solve_ivp` with events of the rate system of §4 (elastic until q reaches κ(α) at t_y, plastic with α̇ = 3G(ė − r(q))/(3G + H) while α̇ > 0, elastic again when α̇ reaches 0; in the hold phase it starts elastic). Compared: q, p_cr, α at t = 30 and 60. Set-up assertions on the reference: 0 < t_y < 30, α(30) > 0, p_cr(30) > 0 (both mechanisms ran during the ramp), α(60) = α(30) | `solve_ivp` + events | at N = 1000: rel error ≤ 2e-3 for each of q, p_cr, α at both times; observed orders of the pairs (250, 500), (500, 1000) in [0.8, 1.2] for q(60); α(60) == α(30) bitwise in production |
| C8f | Tangent against central FD of the production stress w.r.t. each Voigt strain (engineering shear), h ∈ {1e-4, 1e-5, 1e-6, 1e-7}·1e-3, best h: (i) plastic-and-creeping states (q_tr = 1.6 κ_n, Δt = 30 s, committed creep and plastic strain nonzero, random multiaxial direction, 12 states each) for (n, H) = (3, 2e9), (1, 2e9), (3, 0); (ii) creep-only states (q_tr = 0.8 κ_n) for n = 3, H = 2e9 with nonzero plastic strain. Set-up assertions: plastic flag 1, x > 0, y > 0, x/y ≥ 0.05 in (i); plastic flag 0, θ ≤ 0.97 in (ii); `tangent_kind="continuum"` raises ValueError | FD of the production stress (derivative check, not an independent stress) | max over states of max-norm error / max-norm C ≤ 1e-6 |
| C8g | Negative controls of the tangent, same FD check: (i) frozen direction C = K 1⊗1 + 2G I_dev − 4G² Λ' N⊗N (no θ, no −Λ/q_tr) in (i) states of C8f; (ii) creep ignored in the plastic part, Λ' = 1/(3G + H) with correct θ and −Λ/q_tr term, states with H = 1e10, Δt = 50 s, q_tr = 1.6 κ_n, n = 3 (predicted error from Λ'_true − Λ'_wrong = w x' ≈ 3e-12 Pa⁻¹: about 3e-2) | FD | error of each wrong tangent ≥ 1e-3 (detects); production tangent ≤ 1e-6 on the same states |
| C8h | Separation and no double counting, 60 random states (n ∈ {1, 3}, H ∈ {0, 2e9}, Δt ∈ {0, 0.5, 30}, plastic and creep-only both occurring, ≥ 10 of each branch): with N = 1.5 s_tr/q_tr computed in the test from eps and the committed state: ε_cr,new − ε_cr,n = Δp_cr N; ε_p,new − ε_p,n = Δα N; their sum = (Δp_cr + Δα) N; √(2/3)‖Δε_cr‖_F = Δp_cr and the same for plastic (so each increment is counted once); σ = C_e:(ε − ε_cr,new − ε_p,new) with a dense C_e written in the test; q_new = q_tr − 3G(Δp_cr + Δα) with q_new from the returned stress | dense numpy in the test | rel ≤ 1e-12 (max-norm of the increment / of its reference) |
| C8i | Local system and KKT conditions on the same 60 states plus 100 more with n = 0.7 and 8: r1 = Δp_cr − Δt A F (q_new/σ_ref)^n recomputed with q_new from the returned stress, |r1| ≤ 1e-10 · max(Δp_cr, cap·1e-3); plastic states: |q_new − (σ_y + Hα_new)| ≤ 1e-12 σ_y; elastic-creep states: q_new ≤ κ_n (1 + 1e-12); Δα ≥ 0, Δp_cr ≥ 0, Δα · max(f_y, 0) small (≤ 1e-12 σ_y Δα); the active set agrees with an independent `brentq` solution of both branches (creep-only root, then plastic root of g(x) on [0, x_max]; same reduction, independent root finder) in Δp_cr and Δα | recomputation, `brentq` | thresholds as stated; branch agreement exact (same set of states), Δp_cr, Δα rel ≤ 1e-9 |
| C8j | Dissipation: D_cr = q Δp_cr ≥ 0; D_p ≥ σ_y Δα (1 − 1e-9); returned dissipation = D_cr + D_p rel 1e-12 with ψ_h = ½Hα² written in the test; q(Δp_cr + Δα) = σ_new:(Δε_cr + Δε_p) by tensor contraction rel 1e-12; every step of the C8b cyclic path with creep on (A = 1e-6, Δt = 2 s) has D ≥ −1e-9 q(Δp_cr + Δα) | formulas and contraction in the test | as stated |
| C8k | Rollback/restart: after 3 committed steps of a strain-controlled path with plasticity and creep, `checkpoint`; a 4th step is taken; the store is restored; committed arrays equal the checkpoint bitwise; repeating the step reproduces stress and state bitwise; a path (strain controlled, then mixed control with a temperature ramp) split at step 7 and resumed with the checkpointed store, `strain0`, `t0`, `temperature0` equals the uninterrupted path bitwise (stress, all four state fields, dissipation) (**regression**) | the uninterrupted run | `np.array_equal` |
| C8l | Negative control: sequential split (creep first, then plastic return, written in the test with `brentq` and the formula y = (q_c − κ_n)/(3G + H)) differs from the coupled update. (i) Virgin state, uniaxial ε with q_tr = 3e7, Δt = 20 s, σ_y = 1.5e7, H = 2e9: x_seq = h(q_c) at the creep-only root q_c ≈ 2.4e7, x_coupled from the production with q_new ≈ 1.6e7; (ii) the sequential end state does not satisfy the coupled residual r1; (iii) consistency of the splitting error: start on the yield surface (q_n = κ(α0), α0 = 1e-3, ε_p = α0 N0, ε_cr = 0) and load at ė = 1e-4 1/s with Δt = 8, 2, 0.5, 0.125 s (q_tr = κ_n + 3G ė Δt); the relative difference |x_seq − x_coupled|/x_coupled falls ∝ Δt | `brentq` sequential reference | (i) |x_seq − x_cpl|/x_cpl ≥ 0.5; (ii) |r1|/x_seq ≥ 0.5 for the sequential end state; (iii) monotone decrease with Δt and ratio of the last pair ≥ 2 (predicted ≈ 4); the coupled production passes C8i's residual on the same inputs |
| C8m | Validation and degenerate input: K, G ≤ 0, A < 0, n ≤ 0, σ_ref ≤ 0, σ_y ≤ 0 or NaN, H < 0 or non-finite raise ValueError; `yield_stress=None` and `math.inf` give identical results; missing step raises; Arrhenius without temperature raises; Δt = 0 gives Δp_cr = 0 bitwise and still returns plastically; q_tr = 0 (hydrostatic strain) gives bitwise no evolution and the elastic tangent; hydrostatic strain does not yield | exact | ValueError / `np.array_equal` |

## 4. Reference ODE of C8d / C8e (independent of the production code)

Unknown z = (p_cr, α), prescribed e(t) and ė(t), q = 3G(e − p_cr − α), r(q) = A (max(q, 0)/σ_ref)^n. Elastic mode:
ṗ_cr = r(q), α̇ = 0. Plastic mode (q = σ_y + Hα kept by consistency, q̇ = Hα̇ = 3G(ė − ṗ_cr − α̇)):
ṗ_cr = r(q), α̇ = 3G(ė − r(q))/(3G + H). Transitions: elastic → plastic when q − κ(α) rises through 0; plastic →
elastic when α̇ falls through 0. `solve_ivp` events, DOP853, rtol 1e-12, atol 1e-18.

## 5. Expected failure modes (stated before the run)

(i) C8e and C8d thresholds assume the BE error ~ Δt · O(rate) of earlier `C_creep` runs; the elastic→plastic switch step
may reduce the observed order in C8e (hence [0.8, 1.2]). (ii) C8c in hold steps sits exactly on the switch (q = κ_n):
either branch gives the same step to rounding; the mixed-control Newton may take more iterations. (iii) C8f(ii)
thresholds assume θ ≤ 0.97 is attainable at Δt = 30 s, q_tr = 0.8 κ_n; if the set-up assertion fails the input is changed
and recorded. (iv) C8b bitwise equality with `PhysicalJ2` is not required: the expressions s_tr, ε_p increment (√(3/2) n̂
vs N) differ in rounding. (v) The `NortonCreep` identity C8a is expected bitwise by construction (same helper
functions, same operation order when ε_p = 0).

Run: `python -m pytest tests/toporlogy/test_creep_plasticity.py -q -p no:randomly -p no:cacheprovider` with TMP/TEMP and
`--basetemp` under `.claude/tmp-toporlogy/C8_creep_plasticity`. Resource cap: single points are milliseconds; C8e
≤ 2 × 1750 material-point steps, C8d ≤ 2400 + 3 × ~1000 steps.


## First result (2026-11-27 06:39 +0300, first run of `tests/toporlogy/test_creep_plasticity.py` after the commit of this file, SHA 42cb74a)

Measured with the preregistered thresholds unchanged: 31 passed, 0 xfail, 20 s. One set-up fix before the first full
pass: the independent `brentq` calls in C8i/C8l passed `rtol = 4.5e-16`, which scipy rejects (minimum 4 eps = 8.9e-16);
changed to `rtol = 1e-15` (an input of the reference solver, no threshold touched). Measured values were then
re-read with a one-off script (`scratchpad/c8/measure.py`, same helpers, post hoc, not a new test).

| ID | Result | Measured |
|---|---|---|
| C8a | PASS | bitwise for `None` and `math.inf`, isothermal and Arrhenius ramp 300 to 340 K, 3 batches x dt in {0, 30} and a 40-step history (creep occurred) |
| C8b | PASS | rel 1.7e-16 (stress), 2.8e-16 (tangent), 1.5e-16 (energy), 1.1e-16 (dissipation), 3.2e-16 (plastic strain), 1.3e-16 (alpha) for H = 2e9; H = 0: <= 5.5e-16; 36-step cyclic path <= 1.8e-15 (final alpha 1.3e-2, 2.2e-2); 37 of 50 points plastic. Bitwise equality does NOT hold (False for stress, tangent, plastic strain), as predicted in section 5 (iv) |
| C8c | PASS | closed forms: axial 1.6e-15, lateral 1.7e-15, creep strain 5.3e-15, alpha 2.3e-15, dissipation 2.5e-15 (T_ref); 3.3e-14, 3.4e-14, 3.8e-14, 2.3e-15, 3.8e-14 (360 K); compression identical; Newton 2-7 iterations per step; ramp vs `quad` 1.6e-14 (k = 1), 2.0e-15 (k = 4). alpha0 = 2.5e-3 |
| C8d | PASS | alpha0 = 2.1923e-3, q0 = 1.9385e7 Pa; at dt = 0.025: q rel 6.0e-5, p_cr rel 1.3e-4 (prediction ~1e-4); closed form vs ivp p_cr 7.2e-15; alpha, plastic strain bitwise constant, plastic flag 0 in all 2400 hold steps; errors 4.76e-4, 2.38e-4, 1.19e-4, orders 0.9988, 0.9994 |
| C8e | PASS | reference: t_y = 6.304 s, alpha(30) = 2.077e-3, p_cr(30) = 1.254e-4, p_cr(60) = 2.779e-4; N = 1000: t = 30: q 9.8e-6, p_cr 8.1e-4, alpha 4.5e-5; t = 60: q 4.9e-5, p_cr 2.2e-4, alpha 4.5e-5 (threshold 2e-3); orders of q(60): 0.9993, 0.9992; alpha(60) == alpha(30) bitwise |
| C8f | PASS | plastic-and-creeping worst best-h error 2.0e-11 (n = 3, H = 2e9), 2.1e-11 (n = 1), 1.9e-11 (H = 0); x/y ranges 0.50-0.86, 0.15, 0.37; creep-only 2.7e-11 with theta in [0.888, 0.915]; `tangent_kind="continuum"` raises |
| C8g | PASS | doc tangent re-derived in the test vs production 1.6e-16; frozen direction error >= 0.131 (detected); creep-ignored plastic tangent error between 1.06e-2 and 2.80e-2 (predicted ~3e-2 in the mean, min above the 1e-3 threshold); production best-h error <= 1.1e-11 on the same states |
| C8h | PASS | all increments, stress recomputation, equivalent-strain identity and q balance within 1e-12 over the two parameter sets (the test asserts >= 10 of each branch; exact counts not recorded) |
| C8i | PASS | max r1 residual 1.6e-13 (relative to max(x, 1e-3 cap)); |q - kappa|/sigma_y <= 7.5e-16 on plastic points; active set identical to brentq on every point; brentq difference in x, y <= 1.2e-13 (relative). Note: for n = 8, dt = 30 all 20 points relax to the creep-only branch (0 plastic), the plastic branch of n = 8 is covered at dt = 0 and 0.5 |
| C8j | PASS | D_cr >= 0, D_p >= sigma_y y (1 - 1e-9), sum and contraction identities 1e-12; cyclic path with creep (dt = 2 s) has plastic steps and creep, every D >= -1e-9 W |
| C8k | PASS | restore bitwise, repeated step bitwise, restart at step 7 bitwise (stress, dissipation, all four state fields; mixed control, Arrhenius ramp; plastic steps >= 2 and creep occurred) |
| C8l | PASS | (i) x_seq = 2.6457e-4, x_coupled = 8.196e-5, relative difference 2.23 (threshold 0.5); y_seq = 3.327e-4, y_coupled = 5.013e-4; sequential end state breaks r1 by 0.709 (threshold 0.5); coupled r1 5.0e-16; q_seq = 1.5665e7, q_coupled = 1.6003e7 Pa; (iii) relative differences 3.64, 0.859, 0.194, 0.0470 for dt = 8, 2, 0.5, 0.125; ratios 4.24, 4.43, 4.13 (threshold of the last pair 2) |
| C8m | PASS | all invalid inputs raise ValueError; `None` and `math.inf` identical; dt = 0 plastic with bitwise no creep; hydrostatic strain bitwise no evolution, elastic tangent |

Not verified: FE use (the model implements the history contract but is not wired into the YAML/mesh paths);
temperature-dependent K, G, sigma_y; nonlinear hardening. Limitations: synthetic constants; the BE accuracy checks are
first order and use fixed small steps (no adaptive driver for this model: `integrate_creep` takes any history model but
was not run with it here); the C8e/C8d references share the rate-system reasoning of section 4 with the derivation, they
are independent of the production code but not of the model assumptions.
