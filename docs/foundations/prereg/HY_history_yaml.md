# HY — History-dependent topology optimisation from YAML / the CLI (preregistration)

Preregistered 2026-11-27 04:07 +0300, before any HY test was run. Base SHA `92cd73e` (fast-forwarded into this
worktree; the branch started at f1ff694). Modules: `configuration/history_build.py` (new), small dispatch hooks in
`configuration/resolve.py` (`read_config`, `resolve`), `configuration/build.py` (`build_setup`),
`execution/cli.py` (`validate`, `run`), `execution/runner.py` (`run_history_setup`). The grid path
(`resolve._resolve`, `build.build`) and the mesh path (`mesh_build.py`) are not changed.
Design and key list: `docs/foundations/HISTORY_YAML.md`.

## Dispatch key (decision, justified)

The mesh path dispatches on `domain.type` (mesh | shell) because there the *domain* is what differs from the grid
path. A history study uses the same mesh generator keys as the mesh path; what differs is the *analysis* (a time
march with a stored history instead of one static solve). A `domain.type: history` would overload the word that
already means "this generator, static elasticity". So the dispatch is a new top-level key
`analysis: {type: history}`. `analysis` is in none of the existing schemas (`SCHEMA_KEYS`, `MESH_KEYS`; checked by
grep), so no existing config can change meaning. `resolve()` and `build_setup()` test it first (before the `units` /
`schema_version` handling of the grid path, because `units` has a different meaning here).

Material cards: a card's `parameters.shift` (WLF) and `parameters.arrhenius` describe a temperature dependence that
`ViscoelasticStructure` / `CreepStructure` do not have (isothermal). The existing synthetic cards
`generalized_maxwell.yaml` and `norton.yaml` carry those blocks, so a history config refuses them (HY7) and the
examples use two new isothermal synthetic cards in the same directory.

## Conventions fixed before the run (rule 11)

* SI inside: lengths m, forces N, times s, stresses Pa. `units` may declare `{name, factor}` (factor to SI) for
  `length`, `force`, `time`; every config number of that kind is multiplied by the factor once at resolve time.
  A bare non-SI name (`mm`) is an error (no factor). The card has its own units (`load_material_card`).
* Load vector at step time t_k: F_k = s(t_k) F, with F the nodal vector of the load, s the dimensionless time
  function. `force: [fx, fy]` is the **total** force vector (components along +x, +y; positive = along the axis). A
  load on a node set splits it equally: each of the n nodes of the set gets force / n (same rule as the mesh path
  `distribute: nodes`); a point load acts on the nearest used node (first one on a tie). Compliance is
  J_k = F_kᵀ u_k (work of the load, positive for a load that does positive work on the structure), the objective of
  `ViscoelasticStructure`.
* Time functions: `ramp: t_r` s(t) = min(t / t_r, 1); `hold: true` s(t) = 1 for all t; `table: [[t, s], ...]`
  piecewise-linear through the knots, constant at the last value after it (first knot at t = 0). Knot and ramp times
  must lie on the step grid (otherwise the discrete load would not contain the kink).
* Step grid: `times: [t1, ...]` strictly increasing, >= 0; or `times: {logspace: [t0, t1, n], events: [...]}` =
  the n points t0 (t1/t0)^(k/(n-1)) plus the explicit events; a logspace point within relative 1e-9 of an event is
  replaced by the event (the event keeps its exact value). "On the grid" = within relative 1e-9 of a grid time; the
  grid entry is then used.
* Objective: `compliance_at: t` -> `step_index` = index of t on the grid; `max_compliance_at: [t, ...]` (>= 2
  distinct times) -> `steps` (the min-max bound formulation of `optimize_viscoelastic`).
* Volume constraint `upper` is the volume fraction in (0, 1], cell-volume weighted
  (`optimize_viscoelastic(volume=)`).
* `chain: [{type: density_filter, rmin}]`, rmin in length units (m after the factor), the centroid cone filter of
  `MeshDensityFilter`.
* Plane strain only: `analysis.plane: strain` (default); `stress` is an error. 3-D (`domain.generator: box`) is an
  error.

## Checks

| ID | Hypothesis | Reference | Threshold |
|---|---|---|---|
| HY1 | Viscoelastic: the YAML-built problem equals the same problem built with the Python API: load array, compliance and gradient dJ/dw at a seeded random design, and a 3-iteration `optimize_viscoelastic` run (x, xphys, objective, f0 history), for `compliance_at` and for `max_compliance_at` | the API objects built in the test (REGRESSION / equivalence of two construction paths, not an independent physics check) | bitwise (`np.array_equal`, `torch.equal`) |
| HY2 | Creep: the same for `CreepStructure` (Norton card), 2 iterations | API objects, K0, G0 from `isotropic_moduli(E, nu)` | bitwise |
| HY3 | Independent anchor of loads and time functions: a set load [0, -F] on `right` of a 4 x 2 mesh puts -F/3 s(t_k) on each of the 3 right nodes' y-DOFs and nothing elsewhere; ramp, hold and table values at the grid times equal the closed forms: ramp t_r = 10: s(5) = 0.5, s(10) = 1, s(20) = 1; table [[0,0],[10,1],[100,1],[200,0.5]]: s(10) = 1, s(50) = 1, s(150) = 0.75, s(200) = 0.5, s(300) = 0.5; hold: 1 | the numbers above, written in the test as literals | relative 1e-14 |
| HY4 | Units: a config in mm / kN / h with factors equals the same config in m / N / s: load array, times, start-design compliance | the SI config (equivalence; the factors are applied once) | loads, times 1e-13 relative; compliance 1e-9 relative |
| HY5 | Grid: `{logspace: [1, 1e3, 7], events: [10.0, 250.0]}` gives 8 times: 1, 1e3^(1/6), 10.0 (exactly, the event), 1e3^(3/6), 100 (1e3^(4/6)), 250.0 (exactly), 1e3^(5/6), 1000 | the closed form t0 (t1/t0)^(k/6) | relative 1e-12, events exact |
| HY6 | CLI: `validate` of both examples exits 0; `run` of each (reduced mesh, 2 iterations, written to a tmp folder with `--out`) writes `design.vtu` (n_cells cells, cell data xphys, x, weight, point data displacement at the objective step), the JSON summary, `summary.json`, `history.jsonl` with 2 lines, `result.npz`, `config.resolved.yaml`; the VTU xphys equals result.npz xphys; summary objective is finite and the exit code is 1 (not converged at 2 iterations) | files and the numbers inside them (summary objective vs `structure.response` at the returned design: same kernel, regression) | exact equality of the two xphys arrays; objective bitwise vs the API recomputation; exit code 1 |
| HY7 | Every refusal raises `ConfigError` with the key path and a message that names the problem (list in the test file: unknown top-level and nested keys, 3-D, plane stress, H8 element, unsupported / shifted / Arrhenius material, broken card, missing card file, objective time off the grid, ramp / table time off the grid, two time-function forms, non-SI unit without factor, bad factor, traction distribution, unknown sets, rigid motion, zero load, non-increasing times, bad logspace, volume bound out of range, both objective forms, missing chain, optimizer other than mma, unsupported post step, card validity time range violated, analysis type other than history) | the expected path of each case, written in the test | exact path match |
| HY8 | Idempotence: `resolve(resolve(cfg)) == resolve(cfg)` for both examples, and the `config.resolved.yaml` of a run rebuilds a problem with the same start-design compliance | the first resolve | equality / bitwise |
| HY9 | Isolation: `is_history_config` is false for the grid example and the mesh example; both still resolve through their own paths; the existing `test_mesh_yaml*.py` pass unchanged | existing tests | pass |

Run: `python -m pytest tests/toporlogy/test_history_yaml.py tests/toporlogy/test_mesh_yaml.py -q -p no:randomly`
(`TMP`/`TEMP` and `--basetemp` under `.claude/tmp-toporlogy/HY`). First result appended below.
