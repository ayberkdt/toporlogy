# History-dependent topology optimisation from YAML / the CLI

Preregistration and checks: `prereg/HY_history_yaml.md` (HY1-HY9, tests `tests/toporlogy/test_history_yaml.py`).
Code: `src/toporlogy/configuration/history_build.py` (resolve + build), dispatch hooks in `configuration/resolve.py`,
`configuration/build.py`, `execution/cli.py`, `execution/runner.py` (`run_history_setup`). All example materials and
loads are **synthetic**: invented numbers, no engineering claim.

```
python -m toporlogy validate examples/foundations/viscoelastic_service.yaml
python -m toporlogy run      examples/foundations/viscoelastic_service.yaml --out runs/ve
python -m toporlogy run      examples/foundations/creep_service.yaml --set optimizer.max_iter=5
python -m toporlogy show     runs/ve
```

## What it is

The Python API `ViscoelasticStructure` / `CreepStructure` + `optimize_viscoelastic` (exact discrete history adjoint on
a fixed time grid, `docs/foundations/prereg/VO_*.md`, `CO_*.md`) behind a config. A config with a top-level
`analysis: {type: history}` is resolved and built by `history_build.py`; the grid path and the mesh path
(`mesh_build.py`) are not touched. The material card decides the model: `generalized_maxwell` -> linear viscoelastic,
`norton` -> Norton creep. 2-D plane strain on a rectangle mesh (Q4 or T3).

**Why `analysis` and not `domain.type`.** The mesh path dispatches on `domain.type` because there the domain is what
differs from the grid path. Here the domain keys are the mesh path's; what differs is the analysis (a time march with a
stored history). `analysis` is in no existing schema, so no existing config changes meaning. `resolve()` tests for it
first, before the grid path's `units` / `schema_version` handling (`units` means something else here).

## Keys

Unknown keys at any level are errors with the key path (and a "did you mean" where one is close).

| Key | Meaning |
|---|---|
| `name` | run name [config file stem] |
| `analysis` | **required**. `type: history`; `plane: strain` [strain] (`stress` is refused) |
| `units` | [SI: `{length: m, force: N, time: s}`]. Another unit needs its factor to SI: `{length: {name: mm, factor: 1.0e-3}}`. A bare non-SI name (`mm`) is refused. Applied once at resolve time to lengths, forces and times of the config (not to the card, which has its own units). The resolved config is in SI. |
| `domain` | `generator: rectangle` (a 3-D `box` is refused), `n: [nx, ny]`, `size: [lx, ly]` (**required**), `element: Q4 \| T3` [Q4], `perturb` [0], `seed` [0], `thickness` [1 length unit: with `length: mm` the default is 1 mm, say `thickness` explicitly] |
| `material` | **required**. A path to a card YAML (relative to the config file) or a card mapping (`optimization/material_cards.py`, `docs/foundations/MATERIAL_CARDS.md`). Only `generalized_maxwell` without `parameters.shift` and `norton` without `parameters.arrhenius` (the structures are isothermal). The card's `validity.time`, if declared, is checked against the step times; a Norton card's `validity.stress` is **not** checked (the stress is unknown before the run). |
| `supports` | **required**. `[{set: left\|right\|bottom\|top, fix: x\|y\|all or a list}]`; the rigid motions must be stopped |
| `loads` | **required**. `[{set \| point, force: [fx, fy], time: <function>}]`, summed. `force` is the total force vector (N); on a `set` it is split equally over the set's nodes (`distribute: nodes`, `traction` is refused); a `point` acts on the nearest node. |
| `times` | **required**. A strictly increasing list of end times (>= 0), or `{logspace: [t0, t1, n], events: [..]}`: n geometric points plus the events; a logspace point within 1e-9 relative of an event becomes the event. A time 0 is an instantaneous first step. |
| `objective` | **required**. `{compliance_at: t}` (the compliance F_n . u_n at step time t) or `{max_compliance_at: [t, ...]}` (>= 2 times, min-max). |
| `constraints` | **required**: exactly one `{response: volume, upper: fraction}`, upper in (0, 1], cell-volume weighted |
| `optimizer` | `{type: mma, max_iter: 40, tol: 0.0, kkt_tol: null, move: 0.2}`; `kkt_tol` stops on the first-order residual (`prereg/KS_kkt_stop.md`) |
| `interpolation` | `{type: simp, penal: 3.0, emin_ratio: 1e-9}`; E = emin + x^p (1 - emin) scales every modulus of the material, the relaxation times are unchanged |
| `chain` | **required**: exactly one `{type: density_filter, rmin: <length>}` (centroid cone filter) |
| `post` | [[]] `{type: vtu, name}` and / or `{type: summary, name}` |
| `output` | `{folder: runs}` default parent of the run folder |
| `provenance` | free mapping, passed through |

Anything else the grid or mesh path knows (`load_cases`, `combinations`, `physics`, `schedule`, `robust`,
`connectivity`, `run`, `schema_version`) is refused by name; `run`'s options live in `optimizer`.

### Time functions (`loads[i].time`, exactly one form, **required**)

| Form | s(t) |
|---|---|
| `{ramp: t_r}` | min(t / t_r, 1): linear from 0 at t = 0 to 1 at t_r, then held |
| `{hold: true}` | 1 for all t (applied over the first step) |
| `{table: [[t, s], ...]}` | piecewise linear through the knots; the first knot is at t = 0; constant at the last value afterwards |

The load at step time t_k is s(t_k) F. The march treats the load as piecewise linear **between step times**, so a
`ramp` end time, a table knot (other than t = 0) and every objective time must be step times; otherwise the config is
refused (`... is not a step-grid time (neighbours ...)`). With a logspace grid add such times to `times.events`.

## Output

`run` writes into the run folder (`--out`, default `<output.folder>/<date-time>_<name>`; an existing finished run is
never overwritten):

* `config.resolved.yaml` (SI, the card inline; resolving it again gives it back), `summary.json` (the keys of the other
  paths: status, objective, volume verdict, iterations, termination; `show` and `compare` read it),
  `history.jsonl` (`it`, `f0`, `volume`, `change`), `result.npz` (`x`, `xphys`, `times`, `compliance` of the returned
  design at every step time);
* `post: vtu` -> `<name>.vtu`: cells with cell data `xphys` (filtered density of the returned design), `x` (design
  variable), `weight` (the SIMP stiffness factor), point data `displacement_step<k>` at each objective step k;
* `post: summary` -> `<name>.json`: compliance at every step time for the start design (uniform at the volume bound) and
  for the returned design, the objective steps, volume, termination.

Exit code: 0 converged and feasible, 1 not converged (also: iteration cap reached) or volume constraint violated,
2 config error. The objective in `summary.json` is the returned design's (`objective_source: returned_design`).

## Not here (explicit limits)

* `--sweep` and `--resume` (refused with a message); no checkpoints.
* `optimize_viscoelastic` has no per-iteration callback, so the iteration table is printed after the optimisation.
* 3-D, plane stress, other materials (`norton_bailey`, plasticity, curing, temperature shift / Arrhenius), traction
  loads, several constraints, other optimisers / interpolations / chain links, robust (uncertain-material) sets
  (`RobustViscoelastic` stays Python-only), adaptive time steps (the grid is fixed).
* The existing cards `generalized_maxwell.yaml` and `norton.yaml` carry a WLF shift / Arrhenius factor and are refused
  here; the examples use `materials/generalized_maxwell_isothermal.yaml` and `materials/norton_isothermal.yaml`
  (the same numbers without the temperature block).
* A Norton design is only meaningful while the small-strain assumption holds; the example's creep strain at 1e4 s is
  synthetic and not checked against it.
