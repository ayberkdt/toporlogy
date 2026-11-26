# Foundations II — live status

Actual HEAD: `07bd589`  
Branch / dirty state: `foundations/round-3` / dirty  
Updated at: 2026-11-27 01:39 +0300  
Environment / resource budget: Windows 11, Python 3.12.1, numpy 2.2.6, scipy 1.14.1, torch 2.5.1+cu121 (CUDA present, not exercised), mpmath 1.3.0, scikit-fem 12.0.2, meshio absent; 12 logical CPUs; 16.7 GB RAM, about 2-4 GB free; C: 5.6 GB free (temp files on D:)  
Last successful integrated checkpoint: f1ff694 (baseline marker 8 passed, collection 2584 tests; logs/collect_f1ff694.txt, logs/baseline_marker_f1ff694.txt)  
Current unit: integration of the WP11-WP14 material slices

| ID | Priority | Dependencies | State | Evidence / SHA | Blocker / next action |
|---|---|---|---|---|---|
| WP01-baseline | P0 | — | VERIFIED_SCOPE | logs/collect_f1ff694.txt, logs/baseline_marker_f1ff694.txt, logs/ruff_df2ca78.txt (pass), logs/mypy_df2ca78.txt (1 error, fixed) | full fast suite at the integration checkpoint |
| WP02-conventions | P0 | WP01-baseline | IN_PROGRESS | MATERIAL_CONTRACT.md | MATHEMATICAL_CONVENTIONS.md |
| P01-contract | P0 | — | VERIFIED_SCOPE | prereg/P01_contract.md CT1-CT8 pass (df2ca78) | FE coupling under WP17 |
| E04-poisson | P0 | — | VERIFIED_SCOPE | prereg/E04_poisson.md PO1-PO6 pass; E04_poisson_migration.json (df2ca78) | README label |
| L-solver | P0 | — | IN_PROGRESS | prereg/L_solver.md: 17 passed, L2c FAILED (strict xfail) | commit; SOLVER_CONTRACT.md |
| P-properties | P0 | P01-contract | VERIFIED_SCOPE | prereg/P_properties.md PP1-PP17 pass (939d04e, merged 07bd589) | material-card YAML schema (P05) not done |
| V-viscoelastic | P0 | P01-contract | IN_PROGRESS | — | subagent branch |
| C-creep | P0 | P01-contract | IN_PROGRESS | — | subagent branch |
| R-rate | P0 | P01-contract | IN_PROGRESS | — | subagent branch |
| T-spectral | P0 | — | IN_PROGRESS | — | subagent branch |
| WP06-fem-kernel | P0 | — | PLANNED | — | F1 reference-element contract tests; F4 reproduce T4/T10 historical rates |
| WP17-history-fem | P1 | V-viscoelastic, P01-contract | PLANNED | — | minimal grid/mesh history adapter with physical models (H1, H2) |
| WP16-cure | P1/P2 | P-properties | PLANNED | — | material-point cure kinetics TS1-TS5 |
| WP15-thermoplastics | P1/P2 | P-properties, V-viscoelastic | PLANNED | — | Avrami TP3/TP4 |
| WP20-identification | P1/P2 | V-viscoelastic, C-creep | PLANNED | — | Prony and Norton synthetic fits D4-D6 |
| WP07-WP10, WP18, WP19 | P1/P2 | WP06-fem-kernel | DEFERRED | — | after the P0 gates |

## Next executable units

1. merge the WP12/WP13/WP14/T subagent branches as they report; regenerate api_surface, catalog and ledger
2. WP06 F1/F4 reference-element and convergence checks
3. WP17 minimal history adapter

## Historical failed hypotheses and unresolved risks

- L2c: SPD factorization path does not detect a symmetric indefinite matrix (solution residual-checked and correct)
- historical (not re-run): T4 L2 rate 1.796, T10 energy rate 1.788, shell drilling DL3

## Changes to public API / physical conventions

- EffectivePoisson / YAML effective_poisson: `convention` required (uniaxial | stiffness_ratio); historical value of the auxetic cell is nu21
- MaterialRequest.step (TimeStep); drive_material_point(times, temperatures, t0, temperature0, store, strain0); NonlinearElasticity refuses physical=True models
- Factorization.solve_transpose; solvers.diagnostics (new)
- core.properties (new)

## Tests actually run (with exit code)

- collect-only f1ff694: 2584 collected, exit 0
- pytest -m baseline f1ff694: 8 passed, exit 0
- ruff check src tests df2ca78: exit 0
- mypy df2ca78: exit 1 (1 error in poisson_uniaxial, fixed)
- test_material_contract.py: 8 passed
- test_poisson_convention + homogenization + start_design + auxetic: 41 passed, 1 xfailed
- test_solver_diagnostics.py: 17 passed, 1 failed (L2c, now strict xfail)

## Resource and permission blockers

- CI logs (R11): no gh CLI on this machine
- shell drilling reference (R08): memory
- real material data (R12): none licensed
- push/PR: needs the user's explicit approval
