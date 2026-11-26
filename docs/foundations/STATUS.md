# Foundations II — live status

Actual HEAD: `70b48a4`  
Branch / dirty state: `foundations/round-3` / dirty  
Updated at: 2026-11-27 02:09 +0300  
Environment / resource budget: Windows 11, Python 3.12.1, numpy 2.2.6, scipy 1.14.1, torch 2.5.1+cu121 (CUDA present, not exercised), mpmath 1.3.0, scikit-fem 12.0.2, meshio absent; 12 logical CPUs; 16.7 GB RAM, about 2-4 GB free; C: 5.6 GB free (temp files on D:)  
Last successful integrated checkpoint: f1ff694 (baseline marker 8 passed, collection 2584 tests; logs/collect_f1ff694.txt, logs/baseline_marker_f1ff694.txt)  
Current unit: VO record; integration fast suite

| ID | Priority | Dependencies | State | Evidence / SHA | Blocker / next action |
|---|---|---|---|---|---|
| WP01-baseline | P0 | — | VERIFIED_SCOPE | logs/collect_f1ff694.txt, logs/baseline_marker_f1ff694.txt, logs/ruff_df2ca78.txt (pass), logs/mypy_df2ca78.txt (1 error, fixed) | full fast suite at the integration checkpoint |
| WP02-conventions | P0 | WP01-baseline | IN_PROGRESS | MATERIAL_CONTRACT.md | MATHEMATICAL_CONVENTIONS.md |
| P01-contract | P0 | — | VERIFIED_SCOPE | prereg/P01_contract.md CT1-CT8 pass (df2ca78) | FE coupling under WP17 |
| E04-poisson | P0 | — | VERIFIED_SCOPE | prereg/E04_poisson.md PO1-PO6 pass; E04_poisson_migration.json (df2ca78) | README label |
| L-solver | P0 | — | VERIFIED_SCOPE | prereg/L_solver.md: L1-L5 pass except L2c (FAILED, strict xfail, ledger) (f49b623) | opt-in SPD verification; L06 saddle-point preconditioners |
| P-properties | P0 | P01-contract | VERIFIED_SCOPE | prereg/P_properties.md PP1-PP17 pass (939d04e, merged 07bd589) | material-card YAML schema (P05) not done |
| V-viscoelastic | P0 | P01-contract | VERIFIED_SCOPE | prereg/V_viscoelastic.md V1-V10, C1 pass (c53ce2d) | temperature-dependent amplitudes, VJP of the material point |
| C-creep | P0 | P01-contract | VERIFIED_SCOPE | prereg/C_creep.md: C2-C7 pass except C3g, C6a predictions (strict xfail, ledger) (9a02203) | creep-plasticity coupling (C8) not implemented |
| R-rate | P0 | P01-contract | VERIFIED_SCOPE | prereg/R_rate.md R1-R7 pass except R3a at 1e-3 1/s (strict xfail) (f4018f7) | Eyring inside Perzyna; VJP |
| T-spectral | P0 | — | VERIFIED_SCOPE | prereg/T_spectral.md TS1-TS9, MO1-MO4, E5 pass (da98d82) | second derivatives; torch path |
| WP06-fem-kernel | P0 | — | PLANNED | — | F1 reference-element contract tests; F4 reproduce T4/T10 historical rates |
| WP17-history-fem | P1 | V-viscoelastic, P01-contract | VERIFIED_SCOPE | prereg/H_history.md H0,H1,H3,H4,H5 pass; H2 FAILED (hypothesis), post hoc H2b pass (e70fa8e, f711936) | design density coupling of physical history models beyond linear viscoelasticity |
| WP16-cure | P1/P2 | P-properties | VERIFIED_SCOPE | prereg/TS_cure.md TS1-TS6 pass (9532240) | FE residual-stress coupling |
| WP15-thermoplastics | P1/P2 | P-properties, V-viscoelastic | VERIFIED_SCOPE | prereg/TS_cure.md TP3, TP4 pass (Avrami/Nakamura only) | TP1/TP2/TP5/TP6 |
| WP20-identification | P1/P2 | V-viscoelastic, C-creep | PLANNED | — | Prony and Norton synthetic fits D4-D6 |
| VO-viscoelastic-topology | P1 | V-viscoelastic, WP17-history-fem | IN_PROGRESS | prereg/VO_viscoelastic_topology.md | record VO1-VO6 and the post hoc VO6 study |
| WP07-WP10, WP18, WP19 | P1/P2 | WP06-fem-kernel | DEFERRED | — | after the P0 gates |

## Next executable units

1. fast suite at the integration checkpoint (one run, background)
2. WP20 D4-D6 synthetic Prony/Norton identification
3. WP06 F4 reproduction of the historical T4/T10 rates

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

## Resource and permission blockers

- CI logs (R11): no gh CLI on this machine
- shell drilling reference (R08): memory
- real material data (R12): none licensed
- push/PR: needs the user's explicit approval
