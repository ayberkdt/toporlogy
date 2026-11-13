# Component registry — how to add a component

`aerokit.topo.registry` maps `(kind, name)` to a constructor. A config names a component either as a plain string
(all defaults) or as a mapping `{type: <name>, <param>: <value>, ...}`. Parameters, defaults and types are read from
the constructor's signature, so a new component needs **one `register` call**. `python -m aerokit.topo list`
shows everything that is registered.

## One line

```python
from aerokit.topo.registry import register

register("link", "am_filter", AMFilter, context=("domain",))      # AMFilter(domain, direction="+z", ...)
register("response", "displacement", Displacement)                # Displacement(dof, case=0)
register("optimizer", "gcmma", "aerokit.topo.optim.gcmma:GCMMA")  # lazy: imported on first use
```

Put the call at the bottom of the module that defines the component. The module must be imported before the
config is resolved: either `aerokit.topo` imports it, or the registry imports it lazily through a
`"package.module:attr"` string (preferred for optional parts; register the string in `registry._register_core`
or in a module that is imported anyway).

## Kinds and what the constructor returns

| kind            | returns                                                                 | used at            |
|-----------------|-------------------------------------------------------------------------|--------------------|
| `domain`        | `Domain`                                                                | `domain`           |
| `selector`      | node numbers (int array; empty selection → raise `ValueError`)          | `where:` of supports/loads |
| `support`       | fixed DOF numbers                                                       | `supports[]`       |
| `load`          | load vector `(ndof,)` of one load case                                  | `load_cases[].loads[]` |
| `link`          | `torch.nn.Module` on the element grid                                   | `chain[]`          |
| `interpolation` | `torch.nn.Module` x̄ → E                                                 | `interpolation`    |
| `solver`        | object with `factor(A_upper)` / `solve(B)`                              | `physics.solver`   |
| `physics`       | object with `solve(E_active, F)`, `free`, `fixed`                       | `physics`          |
| `response`      | `Response` (has `name`, `needs`, `__call__(ctx)`)                        | `objective`, `constraints[].response` |
| `optimizer`     | `Optimizer`                                                             | `optimizer`        |
| `schedule`      | `Schedule`                                                              | `schedule`         |
| `post`          | callable `step(result, study, run_dir) -> dict | None` (dict goes into `summary.json["post"][name]`) | `post[]` |

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

Build order (each step may read what the previous ones put into `ctx`): domain → supports (`ctx.fixed`) → loads →
physics (with its solver) → chain links (`ctx.chain`) → interpolation → objective → constraints → optimizer →
schedule → post steps.

## Chain ends

If the configured chain contains no `embed` link, `embed` is put first; if it contains no `passive`, `passive` is put
last. A link that maps a reduced variable vector to the design vector (symmetry, extrusion) must come before
`embed`, so such a chain lists `embed` explicitly. The resolved config always shows the full chain.

## Example: a selector

```python
import numpy as np
from aerokit.topo.registry import register

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

Register it, build a small config with `aerokit.topo.config.build_setup(raw)`, and check the built object
(`tests/topo/test_config.py::test_register_new_components_one_line` is a template). Remove test-only components with
`REGISTRY.unregister(kind, name)`. Links and responses also need the finite-difference check
(`aerokit.topo.checks.gradcheck`).
