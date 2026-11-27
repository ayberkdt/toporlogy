# Foundations II — live status

Actual HEAD: `12c05ad`  
Branch / dirty state: `foundations/round-3` / dirty  
Updated at: 2026-11-27 10:27 +0300  
Environment / resource budget: Windows 11, Python 3.12.1, numpy 2.2.6, scipy 1.14.1, torch 2.5.1+cu121 (CUDA present, not exercised), mpmath 1.3.0, scikit-fem 12.0.2, meshio absent; 12 logical CPUs; 16.7 GB RAM, about 2-4 GB free; C: 5.6 GB free (temp files on D:)  
Last successful integrated checkpoint: cb6e41e (clean-worktree fast suite 3566 passed / 67 xfailed / 0 failed; ruff and mypy clean; records check clean)  
Current unit: IP projection with beta continuation (IH follow-up); final clean-worktree suite

| ID | Priority | Dependencies | State | Evidence / SHA | Blocker / next action |
|---|---|---|---|---|---|
| WP01-baseline | P0 | — | VERIFIED_SCOPE | logs/collect_f1ff694.txt, logs/baseline_marker_f1ff694.txt, logs/ruff_df2ca78.txt (pass), logs/mypy_df2ca78.txt (1 error, fixed) | full fast suite at the integration checkpoint |
| WP02-conventions | P0 | WP01-baseline | IN_PROGRESS | MATERIAL_CONTRACT.md, MATHEMATICAL_CONVENTIONS.md | keep current as modules are added (continuous) |
| P01-contract | P0 | — | VERIFIED_SCOPE | prereg/P01_contract.md CT1-CT8 pass (df2ca78) | FE coupling under WP17 |
| E04-poisson | P0 | — | VERIFIED_SCOPE | prereg/E04_poisson.md PO1-PO6 pass; E04_poisson_migration.json (df2ca78) | README label |
| L-solver | P0 | — | VERIFIED_SCOPE | prereg/L_solver.md: L1-L5 pass except L2c (FAILED, strict xfail, ledger) (f49b623) | opt-in SPD verification; L06 saddle-point preconditioners |
| P-properties | P0 | P01-contract | VERIFIED_SCOPE | prereg/P_properties.md PP1-PP17 pass (939d04e, merged 07bd589) | material-card YAML schema (P05) not done |
| V-viscoelastic | P0 | P01-contract | VERIFIED_SCOPE | prereg/V_viscoelastic.md V1-V10, C1 pass (c53ce2d) | temperature-dependent amplitudes, VJP of the material point |
| C-creep | P0 | P01-contract | VERIFIED_SCOPE | prereg/C_creep.md: C2-C7 pass except C3g, C6a predictions (strict xfail, ledger) (9a02203) | creep-plasticity coupling (C8) not implemented |
| R-rate | P0 | P01-contract | VERIFIED_SCOPE | prereg/R_rate.md R1-R7 pass except R3a at 1e-3 1/s (strict xfail) (f4018f7) | Eyring inside Perzyna; VJP |
| T-spectral | P0 | — | VERIFIED_SCOPE | prereg/T_spectral.md TS1-TS9, MO1-MO4, E5 pass (da98d82) | second derivatives; torch path |
| WP06-fem-kernel | P0 | — | VERIFIED_SCOPE | F1-F3 existing tests; F4 rates and curved-T10 rule: prereg/F4_rates.md; F3/F4 totals, reactions, constraint transform: prereg/F34_assembly.md (FA1-FA7, CT1-CT8, CN1-CN4 pass) | oblique supports, matrix-free apply (F6), constraint transform inside Newton |
| WP17-history-fem | P1 | V-viscoelastic, P01-contract | VERIFIED_SCOPE | prereg/H_history.md H0,H1,H3,H4,H5 pass; H2 FAILED (hypothesis), post hoc H2b pass (e70fa8e, f711936) | design density coupling of physical history models beyond linear viscoelasticity |
| WP16-cure | P1/P2 | P-properties | VERIFIED_SCOPE | prereg/TS_cure.md TS1-TS6; prereg/TS7_cure_slab.md TS7a-g pass (coupled slab) | vitrification kinetics, viscoelastic relaxation during cure |
| WP15-thermoplastics | P1/P2 | P-properties, V-viscoelastic | VERIFIED_SCOPE | prereg/TS_cure.md TP3, TP4 pass (Avrami/Nakamura only) | TP1/TP2/TP5/TP6 |
| WP20-identification | P1/P2 | V-viscoelastic, C-creep | VERIFIED_SCOPE | covered by D-identification: prereg/D_identification.md (D4-D6; D5a strict xfail) | none |
| VO-viscoelastic-topology | P1 | V-viscoelastic, WP17-history-fem | VERIFIED_SCOPE | prereg/VO_viscoelastic_topology.md VO1-VO6 pass; post hoc: 0.2 % gain at convergence | Study/YAML integration |
| MM-micromechanics | P2 | E04-poisson | VERIFIED_SCOPE | prereg/MM_micromechanics.md MM1-MM4, E7, MMI1-2 pass (c9f019b); Eshelby components from memory, cross-checked by coated/hollow sphere closed forms | self-consistent, ellipsoidal inclusions |
| D-identification | P1/P2 | V-viscoelastic, C-creep | VERIFIED_SCOPE | prereg/D_identification.md D4-D7 pass except D5a covariance prediction (strict xfail); D8 NOT VERIFIED (skip) | real licensed data |
| VR/VU/KS robust optimisation | P1 | VO-viscoelastic-topology | FAILED | prereg/VR_viscoelastic_robust.md (VR1 FAILED), VU_viscoelastic_uncertain.md (5 passed), KS_kkt_stop.md (KS1, KS4 pass; KS2, KS3 FAILED: the gaps are different stationary points, not the stop rule) | none (robust/min-max benefit not supported on this problem; recorded) |
| X-mixed | P1 | L-solver | VERIFIED_SCOPE | prereg/X_mixed.md X1-X4 pass except X2e plain-MINRES control (strict xfail); XF: lambda-based solve refuses incompatible data | 3-D MixedKG tests, B-bar/EAS |
| N-nonlinear | P1 | T-spectral | VERIFIED_SCOPE | prereg/N_nonlinear.md N1-N5 pass; R07 measured still present | bordered predictor for exact singular points |
| A-adaptivity | P1 | WP06-fem-kernel | VERIFIED_SCOPE | prereg/A_adaptivity.md A1-A3 pass except A2d wording (strict xfail) | elasticity, T6, 3-D |
| M-mesh | P1 | — | VERIFIED_SCOPE | prereg/M_mesh.md M1, M2, M4, M5 refusal, M3a VTU pass; M3b gmsh NOT VERIFIED | mixed-cell meshes, periodic M6 |
| P05-material-cards | P0 | P-properties | VERIFIED_SCOPE | prereg/P05_material_cards.md MC1-MC8 pass | grade/test-method/uncertainty fields |
| HY-history-yaml | P1 | VO-viscoelastic-topology, CO-creep-topology | VERIFIED_SCOPE | prereg/HY_history_yaml.md HY1-HY9 pass | robust set, traction loads, --resume |
| CO-creep-topology | P1 | C-creep | VERIFIED_SCOPE | prereg/CO_creep_topology.md CO1-CO3 pass, CO4 FAILED (strict xfail), CO6 pass: 28.4 % service gain (run at the iteration cap) | converged creep optimum (post hoc continuation running) |
| D3-checkpoint | P1 | VO-viscoelastic-topology | VERIFIED_SCOPE | prereg/D3_checkpoint.md D3a-D3d pass (17.5x less saved memory, 2x solves) | checkpointing for the creep march |
| VT-CT-temperature-histories | P1 | — | VERIFIED_SCOPE | prereg/VT_viscoelastic_temperature.md, CT_creep_temperature.md: all pass | — |
| HT-history-yaml-temperature | P1 | — | VERIFIED_SCOPE | prereg/HT_history_yaml_temperature.md | — |
| VF-fractional | P2 | — | VERIFIED_SCOPE | prereg/VF_fractional.md (VF9 strict xfail) | FE kernel wiring |
| TP-thermoplastic | P1/P2 | — | VERIFIED_SCOPE | prereg/TP_thermoplastic.md | TP6 FE slice |
| TS7-cure-slab | P1/P2 | — | VERIFIED_SCOPE | prereg/TS7_cure_slab.md | viscoelastic relaxation during cure |
| C8-creep-plasticity | P0 | — | VERIFIED_SCOPE | prereg/C8_creep_plasticity.md | T-dependent K, G, sigma_y |
| R5-adiabatic | P1 | — | VERIFIED_SCOPE | prereg/R5_adiabatic.md | FE heat coupling |
| H7-H8-history-fe | P1 | — | VERIFIED_SCOPE | prereg/H7_time_convergence.md, H8_more_models_fe.md | — |
| MS-multistart | P1 | — | VERIFIED_SCOPE | prereg/MS_multistart.md | — |
| CC-creep-constraint | P1 | — | FAILED | prereg/CC_creep_constraint.md (CC1, CC2, CC4 pass; CC3 FAILED: halved limit infeasible); post hoc continuation feasible at 0.9/0.8/0.7 (CC_continuation_posthoc.json) | none (failure recorded as strict xfail) |
| F4-rates/F2fix | P0 | — | VERIFIED_SCOPE | F4_RATES.md, prereg/F2fix_curved_t10_degree.md | — |
| T2-second-derivative | P0 | — | VERIFIED_SCOPE | prereg/T2_second_derivative.md | κ > 1e4, torch |
| L6-saddle | P0 | — | VERIFIED_SCOPE | prereg/L6_saddle.md (6 strict xfails) | 3-D, eigenproblems (L06 rest), L07 |
| E3-conditioning | P0 | — | VERIFIED_SCOPE | prereg/E3_conditioning.md; R02 resolved | — |
| NH-nonlinear-heat | P1 | — | VERIFIED_SCOPE | prereg/NH_nonlinear_heat.md (5 strict xfails) | share the Newton loop with cure_slab |
| R07-bordered | P1 | — | VERIFIED_SCOPE | prereg/R07_bordered.md (12 passed first run), prereg/R07c_locate_zero.md (R7c1-R7c3 pass; A1/A2 strict xfails promoted) | none |
| VA-anisotropic-ve | P2 | — | VERIFIED_SCOPE | prereg/VA_anisotropic_ve.md (20 passed first run) | none |
| WP19-shells | P1/P2 | WP06-fem-kernel | BLOCKED | — | drilling reference needs >= 3.5 GB free memory (R08) |
| L6e-eigen | P0 | L-solver | VERIFIED_SCOPE | prereg/L6e_eigen.md (E0-E5, 19 passed first run); EIGEN_RELIABILITY.md | non-symmetric and quadratic eigenproblems |
| NS-newton-shared | P1 | — | VERIFIED_SCOPE | prereg/NS_newton_shared.md (NS1-NS4 pass; bitwise neutral vs 1f898bf) | none |
| MH-mesh-homogenization | P1 | WP06-fem-kernel | VERIFIED_SCOPE | prereg/MH_mesh_homogenization.md (MH1-MH8, 15 passed first run) | Study response on the general mesh; KUBC/SUBC on the mesh |
| MB-mesh-bounds | P1 | MH-mesh-homogenization | VERIFIED_SCOPE | prereg/MB_mesh_bounds.md (MB1-MB5 pass first run; C_SUBC <= C_PBC <= C_KUBC) | none |
| IH-inverse-homogenization | P1 | MH-mesh-homogenization | FAILED | prereg/IH_inverse_homogenization.md (IH1, IH2, IH3/4 a-b, IH5 pass; IH3(c), IH4(c) FAILED: uniform grey stationary point); post hoc 0/1 start 0.841/0.837 of HS | IP projection (running) |

## Next executable units

1. final clean-worktree suite
2. WP19 shell drilling reference when >= 3.5 GB RAM is free
3. push foundations/round-3 and open a PR (needs the user's approval)

## Historical failed hypotheses and unresolved risks

- L2c: SPD factorization path does not detect a symmetric indefinite matrix (solution residual-checked and correct)
- H2: 1-D series reference wrong for a 3-D two-material bar (interface constraint); H3 preregistered reaction sign wrong (wording)
- R3a at 1e-3 1/s: one-ulp error on a steady state where backward Euler is exact (hypothesis wrong)
- C3g: n=20 relaxation predicted below 1e-3 q_tr, reaches 0.15 (independent root agrees); C6a: adaptive global error 2.1e-3 > 1e-3 at rtol 1e-4 (step doubling controls the local error)
- historical (not re-run): T4 L2 rate 1.796, T10 energy rate 1.788, shell drilling DL3

## Changes to public API / physical conventions

- EffectivePoisson / YAML effective_poisson: `convention` required (uniaxial | stiffness_ratio); historical value of the auxetic cell is nu21
- MaterialRequest.step (TimeStep); drive_material_point(times, temperatures, t0, temperature0, store, strain0); NonlinearElasticity refuses physical=True models
- Factorization.solve_transpose; solvers.diagnostics (new)
- core.properties (new)
- core.viscoelastic, core.creep, core.viscoplastic, core.spectral, core.cure (new); core.anisotropy aggregate_moduli, universal_anisotropy_index, zener_ratio
- fem.mesh.history.HistoryProblem (new); optimization.viscoelastic_topology (new)

## Tests actually run (with exit code)

- collect-only f1ff694: 2584 collected, exit 0
- pytest -m baseline f1ff694: 8 passed, exit 0
- ruff check src tests df2ca78: exit 0
- mypy df2ca78: exit 1 (1 error in poisson_uniaxial, fixed)
- test_material_contract.py: 8 passed
- test_poisson_convention + homogenization + start_design + auxetic: 41 passed, 1 xfailed
- test_solver_diagnostics.py: 17 passed, 1 failed (L2c, now strict xfail)
- subagent slices: P_properties 24 passed; R_rate 45 passed 1 xfailed; T_spectral 201 passed 2 xfailed (with existing); C_creep 56 passed 2 xfailed; V_viscoelastic 28 passed; TS_cure 65 passed (with properties, contract)
- test_history_fem.py: 6 passed 1 xfailed
- test_viscoelastic_topology.py: 7 passed (first run)
- ledger/api/architecture/catalog meta tests after merges: 35 passed 1 xfailed (after the diagnostics layer fix)
- ruff check src tests: clean after fixes; mypy: clean after fixes (166 files)
- fast suite (-m 'not slow') started at 79623a7: 2747 passed, 4 skipped, 55 xfailed, 2 FAILED, exit 1, 32 min (logs/fast_suite_79623a7.txt). The 2 failures (api_surface signature of optimize_viscoelastic, vv_risk_kapsam render) are artefacts of regenerating snapshots in the working tree while the suite ran; both pass on the regenerated state (meta tests 102 passed). Lesson: the release suite runs in a separate clean worktree of the commit.
- CLEAN-WORKTREE fast suite (-m 'not slow') at c5c87d4: 2960 passed, 5 skipped, 137 deselected, 58 xfailed, 0 failed, exit 0, 36:51 (logs/fast_suite_clean_c5c87d4.txt)
- benchmarks at f12a243: docs/foundations/BENCHMARKS.md (measurements only)
- CLEAN-WORKTREE fast suite at de4dd3b: 3212 passed, 5 skipped, 155 deselected, 60 xfailed, 1 FAILED, exit 1, 37:51 (logs/fast_suite_clean_de4dd3b.txt). The failure: test_units U1 collected the new examples/foundations YAML files (material cards and history configs) into the Study config snapshot; fixed at aa62547 by excluding that folder (HY8 covers the history configs); snapshot unchanged, test_units + test_history_yaml 99 passed.
- CLEAN-WORKTREE fast suite at cb6e41e: 3566 passed, 5 skipped, 158 deselected, 67 xfailed, 0 failed, exit 0, 52:47 (logs/fast_suite_clean_cb6e41e.txt)

## Resource and permission blockers

- CI logs (R11): no gh CLI on this machine
- shell drilling reference (R08): memory
- real material data (R12): none licensed
- push/PR: needs the user's explicit approval
