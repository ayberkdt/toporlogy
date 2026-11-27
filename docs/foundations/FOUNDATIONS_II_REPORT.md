# Foundations II — integration report (WP24 draft)

Branch `foundations/round-3` from `f1ff694` (main after PR #23). Every number below is copied from a
preregistration record in `docs/foundations/prereg/` (first-result sections) or a log in `docs/foundations/logs/`;
the record is named in each row. Synthetic parameters throughout; no real material is claimed. Live tracker:
`STATUS.md` / `status.json`.

## What was built and verified

| Slice | Content | Verified scope (record) | Failed preregistered checks |
|---|---|---|---|
| P01 contract | `TimeStep`, `MaterialRequest.step`, history evaluate helper, timed/restartable material-point driver; FE refuses physical models | CT1–CT8 (`P01_contract.md`) | — |
| E04 Poisson | `poisson_uniaxial`; `effective_poisson` needs `convention` | PO1–PO6; historical auxetic "ν12 = −0.95" is ν21, ν12 = −0.84 (`E04_poisson.md`) | — |
| L solver | backward/forward error report, inertia, Krylov with structural checks, gauge solve, refinement, transpose solve | L1–L5 (`L_solver.md`) | L2c: SPD path does not detect an indefinite matrix (answer residual-checked) |
| P properties | E(T)/ν(T)/K(T)/G(T) from two functions, validity box, ε_th = ∫α dT, thermoelastic point | PP1–PP17 (`P_properties.md`) | — |
| V viscoelastic | Maxwell/KV/SLS/Burgers, 3-D generalized Maxwell (independent K, G spectra), WLF/Arrhenius, DMA | V1–V10, C1 (`V_viscoelastic.md`) | — |
| VA anisotropic viscoelasticity | generalized Maxwell with PSD branch tensors (Mandel-checked, pseudo-inverse on singular ranges), orientation, DMA passivity | VA1–VA12, NC1 (`VA_anisotropic_ve.md`): isotropic reduction 3e-16 vs GeneralizedMaxwell, hereditary reference 5.3e-12, tangent 7e-14 | — |
| VF fractional | Mittag-Leffler E_α, E_α,β (vs mpmath ≤ 1.5e-13), fractional Maxwell/KV/Zener closed forms (vs numerical inverse Laplace ~1e-16), passivity, Prony bridge to the generalized Maxwell with an error budget | VF1–VF11 (`VF_fractional.md`) | VF9 monotone Prony error at α = 0.3 |
| TP thermoplastics | logistic glass-transition property model, crystallinity mixing inside the bounds, crystallising thermoelastic point with latent heat and shrinkage | TP1, TP2, TP5, MX1–MX4 (`TP_thermoplastic.md`) | wording: secant-CTE control 25.9 % (predicted ~8 %) |
| C8 creep–plasticity | coupled J2 creep + rate-independent plasticity, one local residual, active set, consistent tangent | C8a–m (`C8_creep_plasticity.md`); sequential split overestimates Δp_cr 3.2× at Δt = 20 s | — |
| R5 adiabatic | Perzyna with σ_y(T), Taylor–Quinney heating from the dissipation (not the plastic work), coupled tangent | AD1–AD8 (`R5_adiabatic.md`); roundoff allowances added to two inequality checks after a 6e-15 failure (recorded) | — |
| C creep | Norton, Norton–Bailey, Arrhenius, adaptive long holds | C2–C7 (`C_creep.md`) | C3g (n = 20 relaxation prediction), C6a (global error of local step control) |
| R rate | physical J2, Perzyna, Eyring, Drucker–Prager | R1–R7 (`R_rate.md`) | R3a (1-ulp steady state) |
| T2 second derivative | d²f(A)[H, K] by second divided differences (stable confluent limits), Hessian-vector product | SD1–SD10 vs a 3×3 block-matrix oracle (≤ 1.9e-12 at κ = 1e4) (`T2_second_derivative.md`) | wording: O(h²) slope check inapplicable to a cubic |
| L6 saddle point | block operator, Schur complement, block-diagonal (MINRES) and block-triangular (GMRES) preconditioners with a scaled pressure mass | L6a, b1a/c, c1, c3, c5/6, d, f (`L6_saddle.md`); MINRES converges in 24/24 Taylor–Hood cases up to ν = 0.5; a restart-status fault fixed | L6b1b (constant pressure is an exact eigenvector), L6c2a/b (T6 counts 47), L6c4 (capped statistic), L6e (ILU not symmetric) |
| L6e eigenproblem reliability | `solvers/eigen.py`: per-pair backward error and Rayleigh consistency, cluster-aware block residual and principal angles, MAC mode tracking with crossing report, shift-invert with an explicit shift (a shift at an eigenvalue is refused) and zero-mode report, buckling pencil (−K_G) φ = μ K φ with stated load convention; `fem/modal.py` and `fem/buckling.py` unchanged | E0–E5 (`L6e_eigen.md`), 19 passed on the first run: residual 7.2e-16, angles to analytic double-eigenvalue subspaces ≤ 6.4e-15 (single-vector MAC 0.638 after a 37° rotation, subspace angle 4.9e-16), free-free beam 3 zero modes (residual 9.3e-17), tracking through crossings (index matching gives MAC 4.6e-30), 2-DOF buckling closed form to 1e-12, Euler column error 0.0074 at 8 elements | — |
| F34 assembly totals, reactions, constraints | `fem/mesh/constraints.py`: `LinearConstraints` (fix, tie, MPC, periodic with macro jump), elimination T u_m + u0 with refusal of contradictory/redundant sets, Lagrange and penalty alternatives; `HeatProblem.capacity` | FA1–FA7, CT1–CT8, CN1–CN4 (`F34_assembly.md`), 94 passed, 8 skipped (curved variants of uncurvable elements): totals and reactions ≤ 1e-13, skew periodic cells give the homogeneous solution ≤ 3e-15 on six elements; finding: the default load rule under-integrates the moment of non-constant body forces (T3 4.8e-2, T4 7.2e-3, H8 7.3e-9) and of constant ones on curved T6/T10 — predicted in amendment 1, default not changed | — |
| MH mesh homogenisation | `fem/mesh/homogenization.py`: periodic cell as a general mesh (skew lattice), unit macro strains through `periodic_lattice`, C = u_iᵀ K u_j / \|Y\| and self-adjoint dC/dE_e; conduction analogue | MH1–MH8 (`MH_mesh_homogenization.md`), 15 passed on the first run: homogeneous skew cells ≤ 1e-14 on six elements, regular-grid implementation C 1.4e-14 / dC 6.4e-14, layered average 7.1e-15, FD 1.1e-8, renumbering 1e-14 | — |
| MB mesh bounds | `bc="kubc"` / `"subc"` in `homogenize_mesh_elasticity`: boundary u = H x; uniform stress through its consistent load, fewest DOFs against rigid motion, C = S⁻¹ | MB1–MB5 (`MB_mesh_bounds.md`), all pass on the first run: homogeneous cells ≤ 2.6e-15, C_SUBC ≤ C_PBC ≤ C_KUBC strictly on two cells, regular-grid bounds 2.1e-14 | — |
| IH inverse homogenisation | `optimization/cell_topology.py`: periodic density filter (minimum-image), SIMP p = 3.5 (above the HS-admissible 3.08), MMA, bulk-modulus objective on square and rhombic cells | IH1 gradient 2.0e-8, IH2 filter periodicity 5.6e-16, IH3/IH4 (a, b) volume and κ ≤ κ_HS, IH5 negative control (p = 1 gives 1.93 κ_HS) (`IH_inverse_homogenization.md`) | IH3(c), IH4(c): κ/κ_HS = 0.341 < 0.80 — the registered start returned to the uniform grey stationary point; post hoc from a 0/1 void-hole start: 0.841 (square), 0.837 (rhombus, nearly isotropic), KKT-stationary |
| IP projection for the IH cells | tanh projection with β continuation (1 → 32), MMA restarted per stage, volume on the projected field | IP1–IP4 (`IP_projection.md`), all pass on the first run: gradient 1.1e-8, unprojected path bitwise unchanged, **κ/κ_HS = 0.9895 (square) and 0.9880 (rhombus)** with 0.1–0.4 % grey cells, bound not crossed | — |
| IA auxetic cell | `objective="poisson"` (maximise −C12/C11) with C11 ≥ 0.02 and the IP projection on a general periodic mesh; formulation of the historical grid run | IA1–IA4 (`IA_auxetic_cell.md`), all pass on the first run: C12/C11 = −1.76 (thresholded −1.66, positive definite, one periodic body), grey 1.1 %; physically ν21 = −1.76 (stress along y) and ν12 = −0.37 (stress along x): auxetic in both directions | — |
| NS shared Newton loop (cleanup) | `fem/mesh/_newton.newton_free` used by `cure_slab` and `nonlinear_heat`; each keeps its residual, stop scale and noise model | NS1–NS4 (`NS_newton_shared.md`): six fingerprints bitwise equal to the pre-change code (1f898bf); negative control resolves one ulp | — |
| E3 conditioning | float64 accuracy of energy, compliance, directional E and ν vs mpmath for κ = 1 … 1e14; `forward_accuracy_established` | errors ≤ 0.20 κu to κ = 1e10; R02 resolved: κ = 5.5e34 Cholesky case SPD in mpmath, undecidable in float64, correctly refused; acoustic tensor SPD bound proved (`E3_conditioning.md`) | AC2 at κ = 1 (4u slack exceeded by 2.6e-16); ν metric at κ = 1 changed after the first run (reference ν = 0; recorded) |
| T spectral | f(A) with Fréchet derivative/JVP/VJP at repeated eigenvalues; VRH, A_U, Zener | TS1–TS9, MO1–4, E5 (`T_spectral.md`) | — |
| F4 rates / F2fix | historical T4/T10 rate misses and curved T10 patch investigated | R05 cause: curved T10 needs degree 3 (default now 3 for curved T10, patch exact to 1.5e-15); R04: T4 a mesh-draw accident, T10 pre-asymptotic, no defect (`F4_RATES.md`, `F2fix_curved_t10_degree.md`) | F4-1 T4 interpolation gap, F4-3 T10 draws (strict xfails) |
| TK tensor | cofactor/det derivative without inverse; rotation log near π | TK1–TK5; **fixed an O(π − θ) axis error** of the existing near-π branch (`TK_cofactor_rotation.md`) | — |
| TS cure | n-th order/Kamal–Sourour (bounded implicit), DiBenedetto, reaction heat, shrinkage, Avrami/Nakamura | TS1–TS6, TP3–TP4 (`TS_cure.md`) | — |
| NH nonlinear heat | transient/steady conduction with k(T), ρc_p(T), enthalpy form, exact Newton tangent, rollback, energy report | NH1, NH3–NH6, NH8, NH9 (`NH_nonlinear_heat.md`): energy balance 2.4e-13, capacity form 1–2 % imbalance (control), tangent 3.4e-11; two production defects fixed | NH2 T3/Q9 coarse-level rates, NH7 (nodes exact, no h² slope) |
| TS7 cure slab | transient heat + cure exotherm (monolithic Newton, backward Euler), residual stress through the history solver | TS7a–g (`TS7_cure_slab.md`): time order 1.00, space order 2.03 vs an independent method-of-lines BDF reference; energy balance 3.7e-12 per step | — |
| MM micromechanics | Voigt/Reuss, HS, Eshelby sphere (from memory, cross-checked), Mori–Tanaka, SIMP/RAMP HS admissibility | MM1–MM4, E7, MMI1–2 (`MM_micromechanics.md`) | wording: MT with a stiff matrix is the HS **upper** bound |
| D identification | Prony (NNLS, free times), Norton (log-linear, nonlinear), identifiability SVD, data records with units and validity box | D4–D7; D8 NOT VERIFIED (`D_identification.md`) | D5a covariance at κ = 8e8 |
| H history FE | small-strain history solid on a general mesh, commit-after-convergence, rollback, reactions | H0, H1, H3–H5, H6a–e, H8a–e: every physical model (Norton, Perzyna, generalized Maxwell, physical J2, Drucker–Prager non-associated, creep–plasticity, adiabatic Perzyna, curing, crystallising) agrees with its material point to ≤ 1.8e-11 (`H_history.md`, `H6_physical_models_fe.md`, `H8_more_models_fe.md`) | H2 (1-D series reference wrong in 3-D) |
| X mixed | K-based u–p (`MixedKG`, valid at ν = 0), inf–sup, locking, spurious modes | X1–X4 (`X_mixed.md`); **λ-based solve now refuses incompatible incompressible data** (`XF_mixed_compatibility.md`) | X2e plain-MINRES control |
| A adaptivity | residual and ZZ indicators, Dörfler marking, newest-vertex bisection (T3, Poisson) | A1–A3; adaptive rate −0.504 vs uniform −0.326 on the L-shape (`A_adaptivity.md`) | A2d wording at θ = 0.5 |
| P05 material cards | dict/YAML cards with explicit units, provenance, validity box for 8 model families | MC1–MC8 (`P05_material_cards.md`); a consistently mislabelled stress unit is undetectable (stated) | — |
| M mesh | validation (duplicates, non-manifold, orientation by id), quality with a Bernstein Jacobian certificate, inverse map / point location, renumbering, VTU roundtrip | M1–M5 (`M_mesh.md`); gmsh import NOT VERIFIED | — |
| HY YAML | `analysis: {type: history}` configs: material card, time schedule, service-time / min–max objectives, CLI validate/run, VTU + JSON output | HY1–HY9 (`HY_history_yaml.md`) | — |
| HT temperature YAML | `temperature: {table}` in the history YAML for shifted viscoelastic and Arrhenius creep | YAML = API bitwise; refusals with paths (`HT_history_yaml_temperature.md`) | HT4 first run: two HY7 messages lost the word "isothermal" (message fixed) |
| D3 checkpoint | segment checkpointing of the viscoelastic adjoint | same gradient, 17.5× less saved memory, 2× solves (`D3_checkpoint.md`) | — |
| N nonlinear | objectivity, plane stress, patches, truss, Hencky via spectral log; bordered predictor at exactly singular points | N1–N5 (`N_nonlinear.md`); **R07 resolved**: bordered predictor (R1–R6, bitwise neutral elsewhere; `R07_bordered.md`) and a critical point exactly at an accepted point (`R07c_locate_zero.md`); the historical A1/A2 strict xfails pass and were promoted | — |

## Optimisation and robustness

| Study | Result (record) |
|---|---|
| VO: viscoelastic topology optimisation, exact discrete history adjoint | forward vs history FE 6e-14; gradient vs FD 5.5e-10; proportional spectra ⇒ ∇J = (J/c_el)∇c_el to 8.5e-14, non-proportional breaks it (0.146); detached history wrong by 378 % (`VO_viscoelastic_topology.md`) |
| VO6 service time vs instantaneous design | at convergence 0.20 % gain at t_s, 0.18 % loss at 0⁺ — a real but small trade-off; the 40-iteration difference was non-convergence (post hoc) |
| VR min–max over service times, one constant load | VR1 failed: reduces to the last time; min–max 0.28 % worse (`VR_viscoelastic_robust.md`) |
| VU robust (mean + 2 std over 9 spectrum samples) | weak pass; the uncertainty (±8.6 %) is nearly design-independent (`VU_viscoelastic_uncertain.md`) |
| KS KKT-residual stop | KS2/KS3 failed: KKT-stationary designs still differ by 0.26–0.33 %: different local optima, not the stop rule; warm start from the nominal design improves it by 0.015 % (`KS_kkt_stop.md`) |
| CO creep topology optimisation (Norton, exact history adjoint via implicit step + local root correction) | forward vs history FE 5.9e-16; gradient vs FD 7.1e-11; CO4 negative control insensitive at weak creep (failed, recorded; 1.3e-2 at strong creep, post hoc); CO6: creep design **28.4 % lower** J(1000 s) than the elastic design, 54 % higher J(0); the first run hit the iteration cap, a post hoc continuation converged (KKT 9.8e-5) at the same 28.40 % (`CO_creep_topology.md`, `CO6_creep_topology.md`) |
| CC creep-strain constraint | accumulated creep strain p_cr (exact vs history FE 4.7e-15) and its p-norm (gradient vs FD 6.4e-11) as an extra MMA constraint; CC3 FAILED: the halved limit was not reached (27 % over at the iteration cap), the max creep strain fell 32 % at +90 % compliance; post hoc continuation (0.9, 0.8, 0.7 × p-norm) is feasible at each step: peak creep strain −13/−25/−33 % for +0.5/+2.5/+8.3 % compliance (iteration-capped, not converged) (`CC_creep_constraint.md`) |
| VT / CT temperature histories | viscoelastic optimisation with WLF/Arrhenius reduced time; creep optimisation with an Arrhenius factor | forward vs history FE 6.9e-14 / 4.3e-16; gradients vs FD 1.9e-10 / 6.5e-10; time–temperature superposition bitwise (`VT_viscoelastic_temperature.md`, `CT_creep_temperature.md`) |
| H7 time convergence | FE history solver: backward Euler creep order 1.05–1.10, generalized Maxwell 1.91–1.99 (`H7_time_convergence.md`) |
| MS multi-start | best of uniform / warm / perturbed starts; on the robust problem 3 of 4 starts reach the better basin, robust design 0.015 % better than nominal (`MS_multistart.md`) |
| VM min–max over a two-load history | min–max design J_A = 3.68e-5, J_B = 3.67e-5 (balanced to 0.075 %); single-phase designs reach 903 and 2008 on the other phase (`VM_minmax_switching.md`) |

Conclusion for the optimisation side: the gradients and formulations are exact and verified; the benefit of
time-aware, robust or min–max design depends on the problem — negligible for one constant load on a linear
viscoelastic material, large under nonlinear creep (28 %), decisive when the critical time depends on the design
(min–max). Design comparisons below ~0.5 % need multi-start.

Figure of the designs: `FIG_DESIGNS.md` (datawiz audit 0 errors).

## Regression evidence

* Baseline marker at f1ff694: 8 passed (bitwise); collection 2584 tests (`logs/`).
* Fast suite at 79623a7: 2747 passed, 55 xfailed, 2 failed — both snapshot tests, artefacts of regenerating the
  snapshots in the working tree during the run (pass after regeneration).
* **Clean-worktree fast suite at cb6e41e: 3566 passed, 5 skipped, 67 xfailed, 0 failed, exit 0** (52:47;
  `logs/fast_suite_clean_cb6e41e.txt`). Earlier clean runs: de4dd3b (1 failure in the example snapshot, fixed),
  c5c87d4 (0 failures).
* Clean-worktree fast suite at c5c87d4: 2960 passed, 5 skipped, 58 xfailed, 0 failed, exit 0** (36:51;
  `logs/fast_suite_clean_c5c87d4.txt`). Later commits (WP09 adaptivity, material cards, creep optimisation) were
  checked with their own files and the meta tests; a final clean run covers them.
* ruff and mypy clean at each integration commit (mypy: 170 files).

## Not done / open

`Study` integration of the physical history models (a separate YAML path exists: HY); FE coupling of
creep / plasticity with the design density (only linear viscoelasticity is design-coupled); WP07 external mesh I/O
(meshio/gmsh absent), WP19 shell drilling (memory-blocked), WP05 L06
saddle-point preconditioners and benchmarks; real material data (D8, R12). Remote push and PR need the user's approval.
