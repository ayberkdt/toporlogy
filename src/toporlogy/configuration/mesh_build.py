"""A config whose ``domain.type`` is ``mesh``: a density study on a general mesh, resolved and built by its own path
(``docs/olcumler/ag_topoloji_onkayit.md``, AY1-AY6). :func:`~toporlogy.configuration.resolve.resolve` and
:func:`~toporlogy.configuration.build.build` hand such a config to :func:`resolve_mesh` / :func:`build_mesh`; the grid
path (``resolve._resolve``, ``build.build``, :mod:`toporlogy.boundary`) does not see it.

Schema (minimal scope; defaults in brackets):

.. code-block:: yaml

    name: mesh_cantilever          # [config file stem]
    domain: {type: mesh, generator: rectangle, n: [48, 16], size: [3.0, 1.0], element: Q4, perturb: 0.0, seed: 0}
                                   # *required*; generator rectangle (2-D: element Q4 | T3 [Q4]) or box (3-D: H8 | T4
                                   # [H8]); perturb: interior vertices moved by up to this length (fem.mesh.perturb)
    material: {E0: 1.0, nu: 0.3, plane: stress}                     # unit-modulus D: plane stress (2-D), isotropic
    physics: elasticity            # [elasticity] -> MeshElasticity (thickness 1)
    supports:                      # *required*, >= 1: named node set and the fixed components
      - {set: left, fix: [x, y]}   # sets: left right bottom top (2-D) back front (3-D); fix: axis letters or all
    loads:                         # *required*, >= 1, one load case: the total force spread equally over the nodes
      - {point: [3.0, 0.0], force: [0, -1]}         # of a set, or applied at the node nearest a point
    chain: [{type: density_filter, rmin: 0.125}]    # *required*; rmin in LENGTH units (the grid path: elements);
                                                    # embed / passive added at the ends if absent
    interpolation: simp            # [simp] penal, emin_ratio as on the grid path
    objective: compliance          # [compliance]
    constraints: [{response: volume, upper: 0.4}]   # volume -> MeshVolume (cell-volume weighted)
    optimizer: mma                 # [mma]
    run: {max_iter: 100, tol: 0.01, seed: 0, feasibility_tol: 0.001}
    output: {folder: runs, snapshot_every: 0}       # snapshots are the flat x̄ vector (no grid assumed)

Anything else (``load_cases``, ``combinations``, ``robust``, ``connectivity``, ``schedule``, ``post`` steps,
``units``, other links, responses, objectives, optimizers, physics or interpolations, ``material`` orthotropic /
strength / phases / rho / catalog, ``plane: strain``, ``run.start`` / ``checkpoint_every`` / ``move_beta_ref``) is
a :class:`ConfigError` that names the key and says the mesh path does not support it yet. No post steps run, so a
run folder holds what :func:`~toporlogy.execution.runner.run_setup` writes (``config.resolved.yaml``, ``setup.json``,
``history.jsonl``, ``result.npz``, ``result.json``, ``summary.json``) and no VTU or STL file.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from toporlogy.configuration.build import Setup, _resultant
from toporlogy.configuration.registry import BuildContext, ConfigError, Param, REGISTRY, join, normalise_name, suggest
from toporlogy.configuration.resolve import _list, _section
from toporlogy.configuration.schema import _CONSTRAINT, _OUTPUT, _RUN

__all__ = ["is_mesh_config", "resolve_mesh", "build_mesh", "MESH_KEYS"]

MESH_KEYS = ("name", "domain", "material", "physics", "supports", "loads", "chain", "interpolation", "objective",
             "constraints", "optimizer", "run", "output", "post", "provenance", "schema_version", "units")
_ELEMENTS = {"rectangle": ("Q4", "T3"), "box": ("H8", "T4")}
_SETS = ("left", "right", "bottom", "top", "back", "front")
_AXES = "xyz"
_NOT_YET = "is not supported by the mesh path (domain type mesh) yet"
_DOMAIN = {"type": Param("type", "mesh", str), "generator": Param("generator", "rectangle", str),
           "n": Param("n"), "size": Param("size"),
           "element": Param("element", None, str, nullable=True), "perturb": Param("perturb", 0.0, float),
           "seed": Param("seed", 0, int)}
_MATERIAL = {"E0": Param("E0", 1.0, float), "nu": Param("nu", 0.3, float), "plane": Param("plane", "stress", str)}


def is_mesh_config(raw: Any) -> bool:
    """True when ``raw`` is a mapping whose ``domain`` is a mapping with ``type: mesh``."""
    d = raw.get("domain") if isinstance(raw, dict) else None
    return isinstance(d, dict) and normalise_name(str(d.get("type", ""))) == "mesh"


def _not_yet(errors: list[tuple[str, str]], path: str, what: str | None = None) -> None:
    errors.append((path, f"{what or path} {_NOT_YET}"))


def _number_list(raw: Any, path: str, n: int, errors: list[tuple[str, str]], cast: Any = float) -> list[Any]:
    if not isinstance(raw, list) or len(raw) != n:
        errors.append((path, f"expected a list of {n} numbers, got {raw!r}"))
        return []
    try:
        out = [cast(v) for v in raw]
    except (TypeError, ValueError):
        errors.append((path, f"expected a list of {n} numbers, got {raw!r}"))
        return []
    if not all(math.isfinite(float(v)) for v in out):
        errors.append((path, f"expected finite numbers, got {raw!r}"))
        return []
    return out


def _resolve_domain(raw: Any, errors: list[tuple[str, str]]) -> dict[str, Any] | None:
    d = _section(raw, _DOMAIN, "domain", errors, "domain type mesh")
    gen = normalise_name(d.get("generator", "rectangle"))
    if gen not in _ELEMENTS:
        errors.append(("domain.generator", f'rectangle | box, got "{d.get("generator")}"'))
        return None
    dim = 2 if gen == "rectangle" else 3
    d.update(type="mesh", generator=gen)
    n = _number_list(d.get("n"), "domain.n", dim, errors, int) if "n" in d else []
    if n and min(n) < 1:
        errors.append(("domain.n", f"element counts must be >= 1, got {n}"))
    size = _number_list(d.get("size"), "domain.size", dim, errors) if "size" in d else []
    if size and min(size) <= 0:
        errors.append(("domain.size", f"lengths must be > 0, got {size}"))
    d["n"], d["size"] = n, size
    el = d.get("element") or _ELEMENTS[gen][0]
    if el not in _ELEMENTS[gen]:
        errors.append(("domain.element", f"generator {gen} takes {' | '.join(_ELEMENTS[gen])} on the mesh path, "
                                         f"got {el!r}"))
    d["element"] = el
    if "perturb" in d and not (math.isfinite(d["perturb"]) and d["perturb"] >= 0):
        errors.append(("domain.perturb", f"must be a finite length >= 0, got {d['perturb']}"))
    d["dim"] = dim                                       # removed again below (not part of the resolved config)
    return d


def _resolve_material(raw: Any, dim: int, errors: list[tuple[str, str]]) -> dict[str, Any]:
    raw = {} if raw is None else raw
    if isinstance(raw, dict):
        extra = {k: v for k, v in raw.items() if k not in _MATERIAL}
        for k, v in extra.items():
            if k in ("rho", "orthotropic", "strength", "phases", "catalog"):
                if v is not None:
                    _not_yet(errors, join("material", k))
            else:
                errors.append((join("material", k), f"unknown key in material{suggest(k, list(_MATERIAL)) or ''} "
                                                    f"(mesh path keys: {', '.join(_MATERIAL)})"))
        raw = {k: v for k, v in raw.items() if k in _MATERIAL}
    m = _section(raw, _MATERIAL, "material", errors, "material")
    if "E0" in m and not (math.isfinite(m["E0"]) and m["E0"] > 0):
        errors.append(("material.E0", f"must be > 0, got {m['E0']}"))
    if "nu" in m and not -1.0 < m["nu"] < 0.5:
        errors.append(("material.nu", f"must lie in (-1, 0.5), got {m['nu']}"))
    if m.get("plane") is not None and normalise_name(m["plane"]) != "stress":
        _not_yet(errors, "material.plane", f"material.plane {m['plane']!r}")
    return m


def _resolve_physics(raw: Any, errors: list[tuple[str, str]]) -> dict[str, Any]:
    spec = {"type": raw} if isinstance(raw, str) else raw
    if spec is None:
        return {"type": "elasticity"}
    if not isinstance(spec, dict) or normalise_name(str(spec.get("type", ""))) != "elasticity":
        _not_yet(errors, "physics", f"physics {spec.get('type') if isinstance(spec, dict) else spec!r}")
    elif set(spec) - {"type"}:
        for k in sorted(set(spec) - {"type"}):
            _not_yet(errors, join("physics", k), f"physics parameter {k}")
    return {"type": "elasticity"}


def _resolve_supports(raw: Any, dim: int, errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    sets = _SETS[:2 * dim]
    for i, s in enumerate(_list(raw, "supports", errors, 1)):
        p = join("supports", i)
        if not isinstance(s, dict):
            errors.append((p, "expected a mapping {set, fix}"))
            continue
        for k in s:
            if k not in ("set", "fix"):
                errors.append((join(p, k), f"unknown key of a mesh support (keys: set, fix)"
                                           f"{suggest(k, ['set', 'fix'])}"))
        name = s.get("set")
        if name not in sets:
            errors.append((join(p, "set"), f"a named node set of the mesh ({', '.join(sets)}), got {name!r}"))
        fix = s.get("fix", "all")
        letters = list(_AXES[:dim]) if fix == "all" else [fix] if isinstance(fix, str) else fix
        if not isinstance(letters, list) or not letters or any(c not in _AXES[:dim] for c in letters):
            errors.append((join(p, "fix"), f"axis letters out of {list(_AXES[:dim])} or all, got {fix!r}"))
            letters = []
        out.append({"set": name, "fix": sorted(set(letters), key=_AXES.index)})
    return out


def _resolve_loads(raw: Any, dim: int, errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    sets = _SETS[:2 * dim]
    for i, ld in enumerate(_list(raw, "loads", errors, 1)):
        p = join("loads", i)
        if not isinstance(ld, dict):
            errors.append((p, "expected a mapping {set | point, force}"))
            continue
        for k in ld:
            if k not in ("set", "point", "force"):
                errors.append((join(p, k), f"unknown key of a mesh load (keys: set, point, force)"
                                           f"{suggest(k, ['set', 'point', 'force'])}"))
        if ("set" in ld) == ("point" in ld):
            errors.append((p, "give exactly one of set (a named node set) and point (coordinates)"))
            continue
        r: dict[str, Any] = {}
        if "set" in ld:
            if ld["set"] not in sets:
                errors.append((join(p, "set"), f"a named node set of the mesh ({', '.join(sets)}), got {ld['set']!r}"))
            r["set"] = ld["set"]
        else:
            r["point"] = _number_list(ld["point"], join(p, "point"), dim, errors)
        if "force" not in ld:
            errors.append((join(p, "force"), "required key of a mesh load is missing"))
        else:
            r["force"] = _number_list(ld["force"], join(p, "force"), dim, errors)
        out.append(r)
    return out


def _resolve_chain(raw: Any, errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if raw is None:
        errors.append(("chain", "required on the mesh path: [{type: density_filter, rmin: <length>}] (rmin is a "
                                "length here, so there is no default)"))
        return []
    links: list[dict[str, Any]] = []
    for i, lk in enumerate(_list(raw, "chain", errors)):
        p = join("chain", i)
        spec = {"type": lk} if isinstance(lk, str) else lk
        t = normalise_name(str(spec.get("type", ""))) if isinstance(spec, dict) else ""
        if t in ("embed", "passive"):
            if set(spec) - {"type"}:
                _not_yet(errors, p, f"parameters of link {t}")
            links.append({"type": t})
        elif t == "density_filter":
            for k in set(spec) - {"type", "rmin"}:
                errors.append((join(p, k), "unknown parameter of the mesh density_filter (parameters: rmin)"))
            try:
                rmin = float(spec["rmin"])
            except KeyError:
                errors.append((join(p, "rmin"), "required on the mesh path (a length; the grid default 1.5 is in "
                                                "elements)"))
                continue
            except (TypeError, ValueError):
                errors.append((join(p, "rmin"), f"expected a length, got {spec['rmin']!r}"))
                continue
            if not (math.isfinite(rmin) and rmin > 0):
                errors.append((join(p, "rmin"), f"must be a length > 0, got {rmin}"))
            links.append({"type": "density_filter", "rmin": rmin})
        else:
            _not_yet(errors, join(p, "type"), f"chain link {spec.get('type') if isinstance(spec, dict) else lk!r} "
                                              "(mesh path links: embed, density_filter, passive)")
    kinds = [lk["type"] for lk in links]
    if "embed" in kinds[1:] or kinds.count("embed") > 1:
        errors.append(("chain", "embed must be the first link (once)"))
    if "passive" in kinds[:-1] or kinds.count("passive") > 1:
        errors.append(("chain", "passive must be the last link (once)"))
    if "embed" not in kinds:
        links.insert(0, {"type": "embed"})
    if "passive" not in kinds:
        links.append({"type": "passive"})
    return links


def _resolve_component(kind: str, raw: Any, path: str, allowed: str, errors: list[tuple[str, str]]) -> Any:
    spec = {"type": raw} if isinstance(raw, str) else raw
    t = normalise_name(str(spec.get("type", ""))) if isinstance(spec, dict) else ""
    if t != allowed:
        _not_yet(errors, path, f"{path} {spec.get('type') if isinstance(spec, dict) else raw!r} (mesh path: {allowed})")
        return None
    return REGISTRY.resolve(kind, spec, path, errors)


def _resolve_constraints(raw: Any, errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for i, c in enumerate(_list(raw, "constraints", errors)):
        cp = join("constraints", i)
        if isinstance(c, dict) and "response" not in c and "type" in c:     # shorthand {type: volume, upper: 0.4}
            c = {"response": {k: v for k, v in c.items() if k not in _CONSTRAINT},
                 **{k: v for k, v in c.items() if k in _CONSTRAINT}}
        if not isinstance(c, dict) or "response" not in c:
            errors.append((cp, "expected a mapping with response and upper"))
            continue
        head = _section({k: v for k, v in c.items() if k != "response"}, _CONSTRAINT, cp, errors, "constraint")
        spec = {"type": c["response"]} if isinstance(c["response"], str) else c["response"]
        rp = join(cp, "response")
        if not isinstance(spec, dict) or normalise_name(str(spec.get("type", ""))) != "volume":
            _not_yet(errors, rp, f"response {spec.get('type') if isinstance(spec, dict) else spec!r} (mesh path "
                                 "constraint: volume)")
            continue
        for k in sorted(set(spec) - {"type"}):
            _not_yet(errors, join(rp, k), f"volume parameter {k} (the mesh volume is weighted by the cell volumes)")
        up = head.get("upper")
        if up is None:
            errors.append((join(cp, "upper"), "volume has no default bound; set upper"))
        elif not 0.0 < float(up) <= 1.0:
            errors.append((join(cp, "upper"), f"volume is a fraction of the part: upper must be in (0, 1], got {up}"))
        out.append({"response": {"type": "volume"}, **head})
    return out


def resolve_mesh(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a mesh config and fill every default (the counterpart of ``resolve._resolve``); every problem found
    is listed in one :class:`ConfigError`."""
    errors: list[tuple[str, str]] = []
    from toporlogy.configuration.schema import SCHEMA_KEYS
    for k in raw:
        if k in MESH_KEYS:
            continue
        if k in SCHEMA_KEYS:
            if raw[k] is not None and raw[k] != [] and raw[k] != "none":   # null / [] asks for nothing; {} does
                _not_yet(errors, k, "load_cases (use loads: one load case)" if k == "load_cases" else None)
        else:
            errors.append((k, f"unknown top-level key{suggest(k, MESH_KEYS) or ''}"))
    if raw.get("units") is not None:
        _not_yet(errors, "units", "units (values with units)")
    for k in ("supports", "loads"):
        if k not in raw:
            errors.append((k, "required key is missing"))
    r: dict[str, Any] = {"name": str(raw.get("name") or "study")}
    d = _resolve_domain(raw["domain"], errors)
    dim = 2 if d is None else d.pop("dim")
    r["domain"] = d
    r["material"] = _resolve_material(raw.get("material"), dim, errors)
    r["physics"] = _resolve_physics(raw.get("physics"), errors)
    r["supports"] = _resolve_supports(raw.get("supports"), dim, errors) if "supports" in raw else []
    r["loads"] = _resolve_loads(raw.get("loads"), dim, errors) if "loads" in raw else []
    r["chain"] = _resolve_chain(raw.get("chain"), errors)
    r["interpolation"] = _resolve_component("interpolation", raw.get("interpolation", "simp"), "interpolation",
                                            "simp", errors)
    r["objective"] = _resolve_component("response", raw.get("objective", "compliance"), "objective", "compliance",
                                        errors)
    r["constraints"] = _resolve_constraints(raw.get("constraints", []), errors)
    r["optimizer"] = _resolve_component("optimizer", raw.get("optimizer", "mma"), "optimizer", "mma", errors)
    run = r["run"] = _section(raw.get("run"), _RUN, "run", errors, "run")
    if run.get("start") is not None:
        _not_yet(errors, "run.start", "run.start (a perturbed start over grid indices)")
    if run.get("checkpoint_every"):
        _not_yet(errors, "run.checkpoint_every")
    if run.get("move_beta_ref") is not None:
        _not_yet(errors, "run.move_beta_ref", "run.move_beta_ref (there is no schedule)")
    if run.get("max_iter", 1) < 1:
        errors.append(("run.max_iter", "must be >= 1"))
    if run.get("tol", 1) < 0:
        errors.append(("run.tol", "must be >= 0"))
    r["output"] = _section(raw.get("output"), _OUTPUT, "output", errors, "output")
    for i, _p in enumerate(_list(raw.get("post", []), "post", errors)):
        _not_yet(errors, join("post", i), "post steps (VTU, STL, checks assume the element grid)")
    r["post"] = []
    r["schedule"] = None                                 # what the CLI reads of a grid config, here always none
    if "provenance" in raw:
        r["provenance"] = raw["provenance"]
    if errors:
        raise ConfigError(_tagged(errors))
    return r


def _tagged(errors: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Every message says it comes from the mesh path."""
    return [(p, m if "mesh" in m else f"{m} (domain type mesh)") for p, m in errors]


# ============================================================================================== build
def _mesh(dom: dict[str, Any]) -> Any:
    from toporlogy.fem.mesh import box, perturb, rectangle
    n, size, el = dom["n"], dom["size"], dom["element"]
    m = rectangle(n[0], n[1], size[0], size[1], element=el) if dom["generator"] == "rectangle" else \
        box(n[0], n[1], n[2], size[0], size[1], size[2], element=el)
    if dom["perturb"] > 0:
        m = perturb(m, dom["perturb"], seed=dom["seed"])
    return m


def _check_rigid(mesh: Any, fixed: np.ndarray) -> None:
    """Every rigid motion of the (connected) mesh must be stopped by the fixed DOFs."""
    from toporlogy.fem.mesh.problem import rigid_modes
    used = mesh.used_nodes()
    R = rigid_modes(mesh.points[used], mesh.dim)
    rows = (mesh.dim * used[:, None] + np.arange(mesh.dim)).ravel()
    Rc = R[np.isin(rows, fixed)]
    rank = 0 if not len(Rc) else int((np.linalg.svd(Rc, compute_uv=False) > 1e-8).sum())
    if rank < R.shape[1]:
        raise ConfigError(f"the supports leave {R.shape[1] - rank} rigid motion(s) free; the system is singular",
                          "supports")


def build_mesh(r: dict[str, Any]) -> Setup:
    """Construct a resolved mesh config (:func:`resolve_mesh`) into a :class:`~toporlogy.configuration.build.Setup`:
    ``MeshDomain``, ``MeshElasticity``, a chain ``Embed, MeshFilterLink…, Passive``, the interpolation, compliance,
    ``MeshVolume`` constraints and MMA."""
    try:
        return _build_mesh(r)
    except ConfigError as e:
        raise ConfigError(_tagged(e.errors)) from e.__cause__


def _build_mesh(r: dict[str, Any]) -> Setup:
    from toporlogy.design.chain import Chain, Embed, Passive
    from toporlogy.fem.elements import D_iso, D_plane_stress
    from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity, MeshFilterLink, MeshVolume
    from toporlogy.optimization.study import Constraint, Study

    dom = r["domain"]
    try:
        mesh = _mesh(dom)
        d: Any = MeshDomain(mesh)             # Embed / Passive / Study read only what MeshDomain offers (MD1-MD5)
    except ValueError as e:                   # an inverted or degenerate cell after perturb
        raise ConfigError(f"domain type mesh: {e}", "domain.perturb" if dom["perturb"] > 0 else "domain") from None
    dim, notes = d.dim, [f"mesh: {dom['generator']} {' x '.join(map(str, dom['n']))} {dom['element']}, "
                         f"{mesh.n_cells} cells, {mesh.n_nodes} nodes" + (f", perturbed by up to {dom['perturb']:g} "
                                                                          f"(seed {dom['seed']})" if dom["perturb"] > 0
                                                                          else "")]
    fixed_parts = []
    for i, s in enumerate(r["supports"]):
        nodes = mesh.node_sets[s["set"]]
        dofs = mesh.dofs(nodes, [_AXES.index(c) for c in s["fix"]])
        if not dofs.size:
            raise ConfigError(f'node set "{s["set"]}" is empty', join(join("supports", i), "set"))
        fixed_parts.append(dofs)
    fixed = np.unique(np.concatenate(fixed_parts))
    _check_rigid(mesh, fixed)
    F = np.zeros((d.ndof, 1))
    for j, ld in enumerate(r["loads"]):
        f = np.asarray(ld["force"], float)
        if "set" in ld:
            nodes = mesh.node_sets[ld["set"]]
            if not nodes.size:
                raise ConfigError(f'node set "{ld["set"]}" is empty', join(join("loads", j), "set"))
            F.reshape(-1, dim)[nodes] += f / len(nodes)
        else:
            used = mesh.used_nodes()
            dist = np.linalg.norm(mesh.points[used] - np.asarray(ld["point"], float), axis=1)
            k = int(used[np.argmin(dist)])
            F.reshape(-1, dim)[k] += f
            notes.append(f"load {j + 1} at node {k} ({', '.join(f'{v:g}' for v in mesh.points[k])}), "
                         f"{float(dist.min()):.3g} from the given point")
    if not np.any(F):
        raise ConfigError("load case has zero load", "loads")
    warnings = []
    loaded_fixed = np.intersect1d(np.flatnonzero(F[:, 0] != 0), fixed)
    if loaded_fixed.size:
        warnings.append(f"{loaded_fixed.size} loaded DOF(s) are also fixed; their load does no work")

    m = r["material"]
    D = D_plane_stress(m["nu"]) if dim == 2 else D_iso(m["nu"])
    physics = MeshElasticity(d, D, fixed)
    links: list[Any] = []
    for i, lk in enumerate(r["chain"]):
        if lk["type"] == "embed":
            links.append(Embed(d))
        elif lk["type"] == "passive":
            links.append(Passive(d))
        else:
            try:
                links.append(MeshFilterLink(d, lk["rmin"]))
            except ValueError as e:
                raise ConfigError(f"link density_filter: {e}", join("chain", i)) from None
    chain = Chain(links)
    ctx = BuildContext(config=r, domain=d, material=dict(m), run=dict(r["run"]), n_cases=1, case_weights=[1.0],
                       physics=physics, chain=chain, links=links, fixed=fixed)
    ctx.interp = REGISTRY.build("interpolation", r["interpolation"], ctx, "interpolation")
    objective = ctx.objective = REGISTRY.build("response", r["objective"], ctx, "objective")
    constraints, cnames = ctx.constraints, []
    for i, c in enumerate(r["constraints"]):
        cp = join("constraints", i)
        resp = MeshVolume(d)
        if c.get("name"):
            resp.name = c["name"]
        if resp.name == objective.name or resp.name in cnames:
            raise ConfigError(f'response name "{resp.name}" is used twice; set {cp}.name', cp)
        cnames.append(resp.name)
        constraints.append(Constraint(resp, float(c["upper"]), c.get("scale")))
    optimizer = REGISTRY.build("optimizer", r["optimizer"], ctx, "optimizer")
    if not constraints and getattr(optimizer, "min_constraints", 1) > 0:
        raise ConfigError(f"optimizer {r['optimizer']['type']} needs at least one constraint", "constraints")
    try:
        study = Study(d, physics, F, chain, ctx.interp, objective, constraints, optimizer, None)
    except ValueError as e:
        raise ConfigError(str(e), "constraints" if "constraint" in str(e) else "") from e
    study.config = r
    cases = [{"name": "loads", "weight": 1.0, "resultant": _resultant(F[:, 0], dim),
              "n_nodes": int(np.any(F[:, 0].reshape(-1, dim) != 0, axis=1).sum()), "combination": False}]
    return Setup(study, r, fixed, cases, cnames, [], warnings, None, None, {}, {"notes": notes})
