# R5_adiabatic — adiabatic heating of a Perzyna J2 material point with Taylor–Quinney fraction and σ_y(T) (preregistration)

Preregistered 2026-11-27 06:26 +0300, before any test of this slice was written or run. Base SHA `6385cfe` (worktree
`agent-aa31719c5b7bf8bfd`). Module: `core/thermomechanics.py` (new: `AdiabaticPerzyna`), tests
`tests/toporlogy/test_adiabatic.py`, doc `docs/foundations/ADIABATIC_HEATING.md`. Plan: WP14 R05 (thermomechanical
coupling: heat fraction is not all plastic work, stored hardening energy, no duplicate heating, adiabatic homogeneous
benchmark) and WP11 P04 (dissipation vs plastic work, heat conversion fraction, energy-budget residual per increment
with dt convergence, one-way imposed T is not two-way thermodynamics). All constants are **synthetic**, no material
data. FE heat conduction (q = −k∇T, the spatial coupling) is **not** in this slice.

Conventions: tension-positive stress; Voigt [xx, yy, zz, yz, xz, xy] (stress physical shear, strain engineering
shear); q = √(3/2)‖dev σ‖; K, G, σ_y, H in Pa; time in s; T absolute in K; ρ in kg/m³; c_p in J/(kg K); energies per
unit volume in J/m³.

## Model and derivations (what the production code must implement)

State per point: ε_p, p (= `alpha`), T, and the accounting variable `cold_work` w_c (J/m³, does not feed back).
Yield f = q − κ(p, T), κ = σ_y(T) + H p (H constant, so ψ_h = ½Hp² does not depend on T; **the thermal part of the free
energy and the thermoelastic/thermoplastic coupling terms are neglected, as in the classical Taylor–Quinney model; the
stored energy returned is the mechanical ψ = ½ε_e:σ + ½Hp² only**). Flow ṗ = (1/τ0)⟨f/σ_ref⟩^m (Perzyna, existing).

1. **Which quantity the existing model returns.** `Perzyna.update` returns D = qΔp − [ψ_h(p_{n+1}) − ψ_h(p_n)], with
   σ_{n+1}:Δε_p = qΔp (J2 flow, ε_p direction n̂ is deviatoric: σ:Δε_p = s:√(3/2) Δp n̂ = √(3/2)‖s‖Δp = qΔp). So D is
   already the plastic **dissipation** (plastic work minus the stored hardening energy); `diagnostics["plastic_work"]`
   = σ_{n+1}:Δε_p = qΔp is the total plastic work. W_p − D = Δψ_h exactly. Heat uses D only.
2. **Heat equation (adiabatic, homogeneous).** ρ c_p(T) Ṫ = β Ḋ_p. Discrete: ρ ∫_{T_n}^{T_{n+1}} c_p dT = β D_{n+1}(T_{n+1})
   (exact enthalpy increment, so a temperature-dependent c_p is consistent without a midpoint rule), D evaluated with
   σ_y(T_{n+1}) (backward Euler, fully implicit in T). β ∈ [0, 1].
3. **Coupled local update, monolithic.** Unknown the scalar ΔT = T_{n+1} − T_n ≥ 0 per point. R(ΔT) = ρ [h(T_n + ΔT) − h(T_n)]
   − β D(T_n + ΔT; ε_{n+1}), h = ∫c_p dT; R(0) = −βD(T_n) ≤ 0. Bracket [0, ΔT1], ΔT1 = βD(T_n)/(ρ c_p(T_n)) doubled while
   R < 0; root by Brent (`scipy.optimize.brentq`, rtol 4ε, xtol 1e-300), i.e. the converged ΔT is accurate to roundoff in
   ΔT. D = 0 or β = 0 at T_n ⇒ ΔT = 0 and the **first** (isothermal) Perzyna call is the result, so β = 0 is bitwise the
   existing Perzyna with σ_y(T_0). Convergence check: after the solve the residual `heat_residual` = ρΔh − βD is reported
   per step; leaving the validity interval of σ_y or c_p raises `ValueError` (no clamp).
4. **Energy budget per step.** Mechanical work (trapezoid) W_n = ½(σ_n + σ_{n+1}):Δε. Since Δψ_e = ½(σ_n + σ_{n+1}):Δε_e
   exactly for a symmetric elastic C, and D = σ_{n+1}:Δε_p − Δψ_h,
   **W_n − Δψ_n − D_n = ½(σ_n − σ_{n+1}):Δε_p** (Δψ = Δψ_e + Δψ_h). This identity is exact to roundoff; the
   right-hand side is the integrator's own (backward-Euler vs trapezoid) work mismatch and is O(Δt²) per step, O(Δt) over
   a history (σ_{n+1} − σ_n = O(Δt), Δε_p = O(Δt), n steps). Heat Q_n = ρΔh_n = βD_n. Convention for the remainder:
   (1 − β) D_n is **stored cold work** (energy of the dislocation/defect structure that is not represented in ψ_h), carried
   in the accounting state `cold_work` by w_c,n+1 = w_c,n + (D_n − Q_n), so that W_n = Δψ_n + Q_n + Δw_c,n + (mismatch).
   The two hardening channels are distinct: ψ_h (ψ) feeds back through κ; w_c does not.
5. **Algorithmic tangent of the actual update** (implicit function theorem on R(T, ε) = 0). With χ = dΔp/dq_tr
   (existing), Δp, q_tr = q + 3GΔp, D_Δp = ∂D/∂Δp = q_tr − Hp_n − (6G + H)Δp (D = Δp(q_tr − Hp_n − (3G + H/2)Δp)),
   n̂_v = deviatoric direction (Voigt, physical shear), σ_y' = dσ_y/dT:
   ∂Δp/∂T = −χσ_y'; ∂σ/∂T|_ε = 2G√(3/2) χ σ_y' n̂_v; ∂D/∂ε_v = (Δp + D_Δp χ) 2G√(3/2) n̂_v; ∂D/∂T = −D_Δp χ σ_y';
   dT/dε_v = β ∂D/∂ε_v /(ρ c_p(T) − β ∂D/∂T); **C = C_Perzyna(σ_y(T_{n+1})) + ∂σ/∂T ⊗ dT/dε_v** (non-symmetric; zero
   extra term for elastic points or β = 0). χ is recomputed from outputs: x = ΞΔp/f_tr, y = 1 − x, χ = m x/(Ξ(y + m x)).
6. **Reference ODE (written in the test, independent of production).** Uniaxial tension, lateral stress free, ε(t) = ε̇ t:
   σ = E_u(ε − p), E_u = 9KG/(3K + G) (p = axial plastic strain); f = σ − σ_y(T) − Hp; ṗ = ⟨f⟩^m/(τ0 σ_ref^m);
   ρ c_p(T) Ṫ = β (σ − Hp) ṗ (σ:ε̇_p = σṗ, minus the stored-hardening rate Hpṗ). `solve_ivp` Radau, rtol 1e-12, atol 1e-16.
   σ_y(T), c_p(T) are re-written in the test as explicit linear formulas (not the production property objects).

Synthetic parameters: E = 2 GPa, ν = 0.35, H = 100 MPa, τ0 = 1 s, σ_ref = 10 MPa, m = 2, ρ = 1200 kg/m³,
σ_y(T) = 20 MPa − 1 MPa/K (T − 293.15 K) on [273.15, 310] K, c_p(T) = 1000 + 2 (T − 293.15) J/(kg K) on [200, 600] K,
T_0 = 293.15 K. Uniaxial ramp ε̇ = 1 s⁻¹ to ε = 0.08 (drive_material_point, mixed control, times k·0.08/n).

## Checks

| ID | Hypothesis | Reference (independent of production) | Threshold |
|---|---|---|---|
| AD1a | β = 0 is bitwise the existing `Perzyna` with σ_y(T_0): 3-step multiaxial path (load, load, partial unload), stress, tangent, energy, dissipation, plastic strain, alpha, T (= T_0) identical | `Perzyna(σ_y = σ_y(T_0))`, **regression** (same kernel, different construction) | `np.array_equal`, 0 difference |
| AD1b | constant σ_y(T) (`Constant`) with β = 1: stress, ε_p, alpha bitwise equal to the β = 0 run (heat does not feed back), T rises | β = 0 run, regression; T rise > 0 | `array_equal`; ΔT > 1e-3 K |
| AD2a | per-step identity W_n − Δψ_n − D_n = ½(σ_n − σ_{n+1}):Δε_p (β = 1, 200 steps; W from stress/strain history, ψ from the energy output) | the identity of derivation 4 (algebra; uses production σ, ε_p, ψ, D) | max abs residual / max W_n ≤ 1e-9 |
| AD2b | closure over the history converges with dt: R(n) = Σ(W − Δψ − D) = ΣW − ψ_N − ΣD, n = 25, 50, 100, 200 (β = 1 and β = 0.4) | trapezoid work computed in the test | \|R\|/ΣW ≤ 5e-3 at n = 200 (estimate 0.16/n ≈ 8e-4 from ½ΔσΔε/n with margin); ratio \|R(n)\|/\|R(2n)\| ≥ 1.5 for each doubling |
| AD2c | heat closure: per step \|ρΔh − βD\| ≤ 1e-8 max_n(βD_n) and over the history \|ρ[h(T_N) − h(T_0)] − βΣD\| ≤ 1e-9 βΣD, h computed in the test as the closed-form antiderivative of the linear c_p; Σ cold_work = (1 − β) ΣD to 1e-9 relative; T non-decreasing; elastic steps have ΔT = 0 exactly | closed-form antiderivative | as stated |
| AD2d | hardening excluded from heat: Σ plastic_work − ΣD = ½ H p_N² (telescoping) to 1e-9 relative; Q_total ≤ ΣD | closed form | as stated |
| AD3 | adiabatic ΔT_end of the production (n = 25, 50, 100, 200) vs the ODE of derivation 6, β = 1 and β = 0.5: relative error in ΔT decreases by ≥ 1.5 per doubling (first order in Δt), Richardson 2ΔT_200 − ΔT_100 within 3e-3 relative of the ODE ΔT; same for the end stress (3e-3 relative to σ_y) | `scipy.integrate.solve_ivp` | as stated |
| AD4 | thermal softening: σ(β = 1) ≤ σ(β = 0) at every step and strictly smaller from the first plastic step on; the reduction σ_end(β = 0) − σ_end(β = 1) equals the ODE reduction within 5 % of that reduction (n = 200); also recorded (not a threshold): ratio to \|σ_y'\|ΔT_end | ODE (β = 0 and β = 1) | as stated |
| AD5 | validation: β ∈ [0, 1] enforced (−0.1, 1 + 1e-12, nan, inf rejected; 0 and 1 accepted); ρ ≤ 0, T_0 ≤ 0, σ_y not in Pa or not positive, c_p units wrong or not positive, a non-property object, tau0/sigma_ref/m/H invalid → `ValueError`; a step carrying a temperature is rejected (T is internal); a missing step is rejected; a probe or step that leaves the validity interval of σ_y raises (no clamp) | — | `ValueError` each |
| AD6 | rollback and purity: `StateStore` trial then `rollback` leaves committed T, ε_p, alpha, cold_work bitwise unchanged; two `evaluate` calls from the same committed state agree bitwise; the committed arrays are read-only and unchanged by `update`; a step that raises (domain) leaves the committed state unchanged | — | `array_equal` |
| AD7 | negative control: a subclass that overrides the heat source with the total plastic work σ:Δε_p (incl. stored hardening). Predicted: Σ(Q − βD) = β·ψ_h(p_N) exactly (telescoping), relative residual of the heat check ψ_h(p_N)/ΣD ≥ 2 %, for β = 1 the cumulative cold work is negative (= −ψ_h(p_N)); ΔT_wrong − ΔT_right ≈ ψ_h(p_N)/(ρ c_p) within 10 % (second-order path difference). The production model passes the same checks (residual ≤ 1e-9, cold work ≥ 0) | the telescoping identity | as stated |
| AD8 | tangent (derivation 5) equals the central finite difference of the production stress (h = 1e-7, engineering shear for shear columns) from a plastic committed state, β = 1, multiaxial; negative control: the isothermal Perzyna tangent (no ∂σ/∂T term) misses the FD by ≥ 1e-3 of max\|C\| | FD of production stress | max abs diff / max abs C ≤ 1e-6 |

## Expected limiting behaviour, failure modes, negative control, run command

β = 0 → Perzyna; σ_y' = 0 → no feedback; Δt → 0 → ODE. Failure modes to look for: bracket not found when D(T) increases
with T (not expected with σ_y' < 0), non-symmetric tangent breaking mixed-control Newton (the Newton uses the production
tangent; convergence is a check by itself). Negative control is AD7 (and the tangent control in AD8). Run:
`pytest tests/toporlogy/test_adiabatic.py -p no:cacheprovider -p no:randomly` with TMP/TEMP and `--basetemp` under
`.claude/tmp-toporlogy/R5_adiabatic`, only this file and `test_viscoplastic.py` (regression of the reused kernel).

## First results (appended after the run)

(none yet)
