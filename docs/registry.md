# Component registry — how to add a component

`toporlogy.configuration.registry` maps `(kind, name)` to a constructor. A config names a component either as a plain string
(all defaults) or as a mapping `{type: <name>, <param>: <value>, ...}`. Parameters, defaults and types are read from
the constructor's signature, so a new component needs **one `register` call**. `python -m toporlogy list`
shows everything that is registered.

## One line

```python
from toporlogy.configuration.registry import register

register("link", "am_filter", AMFilter, context=("domain",))      # AMFilter(domain, direction="+z", ...)
register("response", "displacement", Displacement)                # Displacement(dof, case=0)
register("optimizer", "gcmma", "toporlogy.optimizers.gcmma:GCMMA")  # lazy: imported on first use
```

Put the call at the bottom of the module that defines the component. The module must be imported before the
config is resolved: either `toporlogy` imports it, or the registry imports it lazily through a
`"package.module:attr"` string (preferred for optional parts; register the string in the area's module under
`toporlogy/configuration/components/` or in a module that is imported anyway).

Built-in constructors live in `toporlogy/configuration/components/`, one module per area (`core`, `geometry`,
`boundary`, `design`, `physics`, `responses`, `schedules`) plus `configuration/post_steps.py`;
`components.register_all()` registers them in a fixed order, which `python -m toporlogy list` follows.

## Kinds and what the constructor returns

| kind            | returns                                                                 | used at            |
|-----------------|-------------------------------------------------------------------------|--------------------|
| `domain`        | `Domain`                                                                | `domain`           |
| `shape`         | `geometry.Shape` with attribute `op` (add / cut / intersect)            | csg `shapes`, `keep`/`void`, `in_shape`, finalize `preserve` |
| `selector`      | `boundary.Selector`, or node numbers (int array; empty selection → raise `ValueError`) | `where:` of supports/loads |
| `support`       | `boundary.Support`, or fixed DOF numbers (converted to `boundary.Fixed`) | `supports[]`, `load_cases[].supports[]` |
| `load`          | `boundary.Load`, or a load vector `(ndof,)` of one case (converted to `boundary.NodalForces`) | `load_cases[].loads[]` |
| `link`          | `torch.nn.Module` on the element grid                                   | `chain[]`          |
| `interpolation` | `torch.nn.Module` x̄ → E                                                 | `interpolation`    |
| `solver`        | object with `factor(A_upper)` / `solve(B)`                              | `physics.solver`   |
| `physics`       | object with `solve(E_active, F)`, `free`, `fixed`                       | `physics`          |
| `response`      | `Response` (has `name`, `needs`, `__call__(ctx)`)                        | `objective`, `constraints[].response` |
| `optimizer`     | `Optimizer`                                                             | `optimizer`        |
| `schedule`      | `Schedule`                                                              | `schedule`         |
| `post`          | callable `step(result, study, run_dir) -> dict | None` (dict goes into `summary.json["post"][name]`; optional keys: `summary` = one CLI line, `passed: false` = exit 1) | `post[]` |

## Constructor conventions

* **User parameters** are the keyword parameters of the constructor. Type: from `types={...}`, else from the
  default value (float, int, bool, str, list), else from the annotation text. No default → required.
  Inputs are lenient: `"1e-6"` (PyYAML reads it as text) is accepted for a float, `2` for a float, `yes/no` for a
  bool. Names are case- and dash-insensitive (`Density-Filter` = `density_filter`).
* **Build context.** If the first parameter is called `ctx`, it receives the `BuildContext`
  (`domain`, `material` dict, `fixed` DOFs, `physics`, `chain`, `interp`, `n_cases`, `case_weights`, `run` dict,
  resolved `config`). Alternatively `context=("domain",)` passes `ctx.domain` as the keyword `domain` (handy for
  existing classes such as `DensityFilter(domain, rmin)`). Context parameters are not user parameters.
* **Nested components.** `nested={"where": "selector"}` makes a parameter a component of another kind
  (`"selector[]"` for a list); it is resolved and built first and the constructor receives the built object.
* **Units.** `units={"rmin": "elements"}` is shown by `list` and belongs in the docstring too.
* **Errors.** Raise `ValueError` with a plain message; the config layer prefixes it with the component's key path
  (`supports.0.where: selector plane: plane x = 3.3 selects no node ...`).

Build order (each step may read what the previous ones put into `ctx`): domain → robust thresholds (`ctx.robust`)
→ supports → loads → `boundary.BoundaryConditions` (`ctx.bc`, `ctx.fixed`; connectivity interfaces `ctx.interfaces`, ties
→ new domain) → `BoundaryConditions.check()` (errors raise `ConfigError`) → physics (with its solver) → chain links (`ctx.links`
while building, then `ctx.chain`; `manufacturing.validate_chain`) → interpolation → objective (`ctx.objective`) →
constraints (`ctx.constraints`) → optimizer → schedule → post steps.

A response may carry `config_upper` (and `config_scale`): its default constraint bound when the config gives
`upper: null` (`local_volume` with `t_max`, `min_frequency`, the connectivity responses).

## Feature components (registered in `configuration/components/*` and `configuration/post_steps.py`)

`python -m toporlogy list <kind>` prints every parameter with its default and unit.

| kind | names |
|------|-------|
| domain | `box` (+ `keep`, `void` shapes), `csg` (shapes in order by `op`, h, padding, bounds, min_frac, connectivity), `stl` (path, h, scale, padding) |
| shape | `box`, `ball` (circle in 2-D, sphere in 3-D), `cylinder`, `capsule`, `halfspace`, `polygon`, `prism`, `slot`, `bore` (auto-detected: `index` 1-based or `near`) |
| selector | `plane`, `box`, `point` (node arrays, kept); `all`, `surface`, `in_box`, `on_plane` (axis + at or point + normal, `facing`), `half_space`, `on_cylinder`, `bore` (`index` / `near` / the only one), `in_sphere`, `in_shape`, `at_points`, `nearest`, `node_ids`, `union` / `intersect` / `subtract` (`of: [selectors]`) |
| support | `fixed` (dofs: axes or vectors), `roller`, `symmetry`, `pinned_bore`, `elastic` |
| load | `force` (total spread equally, kept), `point` (spread, zero moment), `surface_force`, `traction`, `pressure`, `bearing`, `remote_force`, `remote_moment`, `torque`, `body_force` |
| link | `am_filter`, `casting`, `symmetry` / `extrusion` / `pattern` (tying: first link, `embed` is put after it), `group_average`, `helmholtz_filter` (PDE filter, rmin in elements) |
| interpolation | `two_range_simp` |
| physics | `modal` (n_modes, mass linear / two_range, added_mass + added_mass_at; load_cases optional), `elasticity_gmg` (GMG-PCG, 3-D, axis-aligned fixed DOFs and one support set only; never a `solver`) |
| response | `robust_compliance`, `dilated_volume`, `local_volume` (`t_max` in length → radius and bound), `grey_level`, `conduction_connectivity`, `coupling_compliance`, `frequency`, `eigenvalue`, `min_eigenvalue_ks`, `maximize` (negates a nested response), `min_frequency`, `stress` (aggregated relaxed von Mises / sigma_lim; upper default 1; `exclude` shapes), `stress_augmented_lagrangian` (objective + local stress constraints), `buckling` (KS of 1/λ; `load_factor` sets the bound), `displacement` (selector + direction), `weighted_sum` |
| optimizer | `gcmma` (conservative inner iterations; any number of constraints), `simpl` (mirror descent; one volume-type constraint) |
| schedule | `robust_continuation`, `multiplier_update` (augmented-Lagrangian multipliers; optional β continuation alongside) |
| post | `connectivity_gate`, `measure_lengthscale`, `finalize` (PostParams fields + `preserve`), `manufacturing_checks` (CheckThresholds fields), `reverify` (gmsh; bc selectors as callables), `export`, `discrete_check` (objective and constraints of the thresholded design) |

Top-level blocks handled by `config.py`: `robust` (min_solid / min_void in length → rmin written into the chain,
thresholds; a `heaviside_projection` is added when missing), `connectivity` (one body via
`connectivity.one_body`: interfaces, ties, constraint, gate, repair), `combinations`, per-case `supports`,
`run.move_beta_ref`. Post steps that fail are recorded in `summary.json` (`passed: false`) and make the run exit 1;
`manufacturing_checks`, `export` and `reverify` need a `finalize` before them. Examples: `examples/*.yaml`.

## Chain ends

If the configured chain contains no `embed` link, `embed` is put first; if it contains no `passive`, `passive` is put
last. A link that maps a reduced variable vector to the design vector (symmetry, extrusion) must come before
`embed`, so such a chain lists `embed` explicitly. The resolved config always shows the full chain.

## Example: a selector

```python
import numpy as np
from toporlogy.configuration.registry import register

def sphere(ctx, centre: list, radius: float):
    """Part nodes within `radius` (length) of `centre`."""
    hit = ctx.domain.nodes_where(lambda X: np.linalg.norm(X - np.asarray(centre, float), axis=1) <= radius)
    if hit.size == 0:
        raise ValueError(f"sphere r = {radius:g} at {centre} selects no node")
    return hit

register("selector", "sphere", sphere, units={"centre": "length", "radius": "length"})
```

```yaml
supports:
  - {type: fixed, where: {type: sphere, centre: [0, 5, 2], radius: 1.5}}
```

## Testing a new component

Register it, build a small config with `toporlogy.configuration.build.build_setup(raw)`, and check the built object
(`tests/toporlogy/test_config.py::test_register_new_components_one_line` is a template). Remove test-only components with
`REGISTRY.unregister(kind, name)`. Links and responses also need the finite-difference check
(`toporlogy.optimization.diagnostics.gradcheck`). [`deriving_methods.md`](deriving_methods.md) walks through a new
local criterion, a new response, a new link and a new optimizer.
