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
    material: {E0: 1.0, nu: 0.3, plane: stress}     # unit-modulus D; plane (2-D only): stress | strain
                                                    # (fem.elements.D_plane_stress | D_plane_strain); 3-D isotropic
    physics: elasticity            # [elasticity] -> MeshElasticity (thickness 1)
    supports:                      # *required*, >= 1: named node set and the fixed components
      - {set: left, fix: [x, y]}   # sets: left right bottom top (2-D) back front (3-D); fix: axis letters or all
    load_cases:                    # *required*, >= 1; column c of F; compliance = sum_c weight_c F_c.U_c
      - name: tip                  # [case<i>]
        weight: 1.0                # [1.0]
        loads:                     # >= 1, summed:
          - {point: [3.0, 0.0], force: [0, -1]}     # at the node nearest the point
          - {set: right, force: [0, -1], distribute: nodes}
                                   # distribute nodes [default]: the total force split equally over the set's nodes;
                                   # traction: a uniform traction force / |facets| over the set's boundary facets
                                   # (edges in 2-D, faces in 3-D), consistent nodal forces (ElasticProblem.traction)
    chain: [{type: density_filter, rmin: 0.125}]    # *required*; rmin in LENGTH units (the grid path: elements);
                                                    # embed / passive added at the ends if absent
    interpolation: simp            # [simp] penal, emin_ratio as on the grid path
    objective: compliance          # [compliance] weights null = the load-case weights
    constraints: [{response: volume, upper: 0.4}]   # volume -> MeshVolume (cell-volume weighted)
    optimizer: mma                 # [mma]
    run: {max_iter: 100, tol: 0.01, seed: 0, feasibility_tol: 0.001}
    output: {folder: runs, snapshot_every: 0}       # snapshots are the flat x̄ vector (no grid assumed)
    post: [{type: vtu, name: result}]               # [[]] <name>.vtu: the mesh cells (VTK triangle, quad, tetra,
                                                    # hexahedron) with cell data xphys and E, point data
                                                    # displacement_<case> (n, 3; z = 0 in 2-D)

Anything else (``combinations``, per-case ``supports``, ``robust``, ``connectivity``, ``schedule``, other ``post``
steps, ``units``, other links, responses, objectives, optimizers, physics or interpolations, ``material``
orthotropic / strength / phases / rho / catalog, ``run.start`` / ``checkpoint_every`` / ``move_beta_ref``) is a
:class:`ConfigError` that names the key and says the mesh path does not support it yet; a top-level ``loads`` list is
refused with a pointer to ``load_cases`` (``docs/olcumler/ag_topoloji_onkayit.md``, AX1-AX8).

``domain.type: shell`` (``docs/olcumler/kabuk_onkayit.md``, SY1-SY6, SX1-SX6) takes a small parallel branch of the
same functions: the flat-facet shell of :mod:`toporlogy.fem.shell` as a
:class:`~toporlogy.optimization.shell_study.ShellDomain` with :class:`~toporlogy.optimization.shell_study.ShellPhysics`.
Its own keys:

.. code-block:: yaml

    domain: {type: shell, surface: cylinder, radius: 1.0, angle: 90.0, length: 1.0, n: [16, 12], thickness: 0.05}
                                   # cylinder about z (angle in DEGREES, (0, 360); null: closed tube) or
                                   # {type: shell, surface: rectangle, size: [lx, ly], n: [nx, ny], thickness: t}
    material: {E0: 1.0, nu: 0.3}   # no plane: a shell is plane stress
    physics: {type: shell, drill: 0.1}              # [shell; drill = toporlogy.fem.shell.DRILL], drill > 0
    supports:                      # node sets: cylinder bottom top (+ start end for an open segment),
      - {set: bottom, fix: all}    # rectangle left right bottom top; fix: u v w rx ry rz (global axes) or all
    load_cases:                    # as on the mesh path; the loads of a case are summed:
      - name: radial
        loads:
          - {pressure: 1.0}        # along the facet normals (FlatShell.pressure_load)
          - {point: [0.7071, 0.7071, 1.0], force: [-0.7071, -0.7071, 0]}  # at the nearest node
          - {set: top, force: [0, 0, 1], distribute: traction}
                                   # nodes [default]: split equally over the set's nodes; traction: a uniform line
                                   # load along the set's boundary facet edges, q L_e / 2 to each edge end
                                   # (FlatShell.line_load)
    post: [{type: vtu}]            # facets as VTK quads; point data displacement_<case> (u, v, w) and
                                   # rotation_<case> (rx, ry, rz)

``chain`` (rmin a length: the 3-D chord distance between facet centres), ``interpolation``, ``objective``,
``constraints`` (``volume``: area weighted), ``optimizer``, ``run``, ``output`` and the refusals are those of the mesh
path; messages name the shell path.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from toporlogy.configuration.build import Setup, _resultant
from toporlogy.configuration.registry import BuildContext, ConfigError, Param, REGISTRY, join, normalise_name, suggest
from toporlogy.configuration.resolve import _list, _section
from toporlogy.configuration.schema import _CASE, _CONSTRAINT, _OUTPUT, _RUN

__all__ = ["is_mesh_config", "resolve_mesh", "build_mesh", "MESH_KEYS"]

MESH_KEYS = ("name", "domain", "material", "physics", "supports", "load_cases", "chain", "interpolation", "objective",
             "constraints", "optimizer", "run", "output", "post", "provenance", "schema_version", "units")
_DISTRIBUTE = ("nodes", "traction")
_PLANES = ("stress", "strain")
_ELEMENTS = {"rectangle": ("Q4", "T3"), "box": ("H8", "T4")}
_SETS = ("left", "right", "bottom", "top", "back", "front")
_AXES = "xyz"
_NOT_YET = "is not supported by the mesh path (domain type mesh) yet"
_DOMAIN = {"type": Param("type", "mesh", str), "generator": Param("generator", "rectangle", str),
           "n": Param("n"), "size": Param("size"),
           "element": Param("element", None, str, nullable=True), "perturb": Param("perturb", 0.0, float),
           "seed": Param("seed", 0, int)}
_MATERIAL = {"E0": Param("E0", 1.0, float), "nu": Param("nu", 0.3, float), "plane": Param("plane", "stress", str)}
_SHELL_DOMAIN = {"type": Param("type", "shell", str), "surface": Param("surface", type=str),
                 "radius": Param("radius", None, float, nullable=True),
                 "angle": Param("angle", None, float, nullable=True),
                 "length": Param("length", None, float, nullable=True), "size": Param("size", None, nullable=True),
                 "n": Param("n"), "thickness": Param("thickness", type=float)}
_SURFACE_KEYS = {"cylinder": ("radius", "angle", "length"), "rectangle": ("size",)}
_SHELL_MATERIAL = {"E0": Param("E0", 1.0, float), "nu": Param("nu", 0.3, float)}
_SHELL_FIX = ("u", "v", "w", "rx", "ry", "rz")


def _domain_type(raw: Any) -> str:
    d = raw.get("domain") if isinstance(raw, dict) else None
    return normalise_name(str(d.get("type", ""))) if isinstance(d, dict) else ""


def is_mesh_config(raw: Any) -> bool:
    """True when ``raw`` is a mapping whose ``domain`` is a mapping with ``type: mesh`` or ``type: shell`` (both are
    resolved and built here, not by the grid path)."""
    return _domain_type(raw) in ("mesh", "shell")


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


def _resolve_material(raw: Any, errors: list[tuple[str, str]], schema: dict[str, Param] = _MATERIAL,
                      path_name: str = "mesh") -> dict[str, Any]:
    raw = {} if raw is None else raw
    if isinstance(raw, dict):
        extra = {k: v for k, v in raw.items() if k not in schema}
        for k, v in extra.items():
            if k in ("rho", "orthotropic", "strength", "phases", "catalog"):
                if v is not None:
                    _not_yet(errors, join("material", k))
            else:
                errors.append((join("material", k), f"unknown key in material{suggest(k, list(schema)) or ''} "
                                                    f"({path_name} path keys: {', '.join(schema)})"))
        raw = {k: v for k, v in raw.items() if k in schema}
    m = _section(raw, schema, "material", errors, "material")
    if "E0" in m and not (math.isfinite(m["E0"]) and m["E0"] > 0):
        errors.append(("material.E0", f"must be > 0, got {m['E0']}"))
    if "nu" in m and not -1.0 < m["nu"] < 0.5:
        errors.append(("material.nu", f"must lie in (-1, 0.5), got {m['nu']}"))
    if "plane" in m:
        plane = normalise_name(str(m["plane"]))
        if plane not in _PLANES:
            errors.append(("material.plane", f"stress | strain on the mesh path, got {m['plane']!r}"))
        m["plane"] = plane
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


def _resolve_cases(raw: Any, errors: list[tuple[str, str]], loads: Any) -> list[dict[str, Any]]:
    """``load_cases: [{name, weight, loads}]`` of both branches; ``loads(raw, path, errors)`` resolves one list."""
    cases = []
    for i, c in enumerate(_list(raw, "load_cases", errors, 1)):
        cp = join("load_cases", i)
        if not isinstance(c, dict):
            errors.append((cp, "expected a mapping {name, weight, loads}"))
            continue
        if c.get("supports") is not None:
            _not_yet(errors, join(cp, "supports"), "per-case supports (the supports above hold for every case)")
        head = _section({k: v for k, v in c.items() if k not in ("loads", "supports")}, _CASE, cp, errors,
                        "load case")
        w = head.get("weight", 1.0)
        if not math.isfinite(w):
            errors.append((join(cp, "weight"), f"must be finite, got {w}"))
        if "loads" not in c:
            errors.append((join(cp, "loads"), "required key of a load case is missing"))
        cases.append({"name": head.get("name") or f"case{i + 1}", "weight": w,
                      "loads": loads(c.get("loads"), join(cp, "loads"), errors) if "loads" in c else []})
    names = [c["name"] for c in cases]
    for i, n in enumerate(names):
        if n in names[:i]:
            errors.append((join(join("load_cases", i), "name"), f'load case name "{n}" is used twice'))
    return cases


def _set_load(ld: dict[str, Any], p: str, sets: tuple[str, ...], what: str,
              errors: list[tuple[str, str]]) -> dict[str, Any]:
    """{set, distribute} of a load on a named node set (the force is resolved by the caller)."""
    if ld["set"] not in sets:
        errors.append((join(p, "set"), f"a named node set of the {what} ({', '.join(sets)}), got {ld['set']!r}"))
    how = normalise_name(str(ld.get("distribute", "nodes")))
    if how not in _DISTRIBUTE:
        errors.append((join(p, "distribute"), f"nodes (equal share per node) | traction (uniform traction, consistent "
                                              f"nodal forces), got {ld.get('distribute')!r}"))
    return {"set": ld["set"], "distribute": how}


def _resolve_loads(raw: Any, path: str, dim: int, errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    sets = _SETS[:2 * dim]
    keys = ["set", "point", "force", "distribute"]
    for i, ld in enumerate(_list(raw, path, errors, 1)):
        p = join(path, i)
        if not isinstance(ld, dict):
            errors.append((p, "expected a mapping {set, force, distribute} or {point, force}"))
            continue
        for k in ld:
            if k not in keys:
                errors.append((join(p, k), f"unknown key of a mesh load (keys: {', '.join(keys)}){suggest(k, keys)}"))
        if ("set" in ld) == ("point" in ld):
            errors.append((p, "give exactly one of set (a named node set) and point (coordinates)"))
            continue
        r: dict[str, Any] = {}
        if "set" in ld:
            r.update(_set_load(ld, p, sets, "mesh", errors))
        else:
            if "distribute" in ld:
                errors.append((join(p, "distribute"), "distribute applies to a load on a set; a point load acts on "
                                                      "the nearest node (mesh path)"))
            r["point"] = _number_list(ld["point"], join(p, "point"), dim, errors)
        if "force" not in ld:
            errors.append((join(p, "force"), "required key of a mesh load is missing"))
        else:
            r["force"] = _number_list(ld["force"], join(p, "force"), dim, errors)
        out.append(r)
    return out


def _resolve_post(raw: Any, errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for i, s in enumerate(_list(raw, "post", errors)):
        p = join("post", i)
        spec = {"type": s} if isinstance(s, str) else s
        t = normalise_name(str(spec.get("type", ""))) if isinstance(spec, dict) else ""
        if t != "vtu":
            _not_yet(errors, p, f"post step {spec.get('type') if isinstance(spec, dict) else s!r} (mesh path post "
                                "steps: vtu)")
            continue
        for k in sorted(set(spec) - {"type", "name"}):
            _not_yet(errors, join(p, k), f"vtu parameter {k} (mesh path vtu parameters: name)")
        name = spec.get("name", "result")
        if not isinstance(name, str) or not name or any(c in name for c in "/\\:"):
            errors.append((join(p, "name"), f"a file name without a folder (mesh path), got {name!r}"))
        out.append({"type": "vtu", "name": name})
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
        if k == "loads":
            errors.append((k, "the mesh path takes load_cases: [{name, weight, loads: [...]}], not a top-level "
                              "loads list"))
        elif k in SCHEMA_KEYS:
            if raw[k] is not None and raw[k] != [] and raw[k] != "none":   # null / [] asks for nothing; {} does
                _not_yet(errors, k)
        else:
            errors.append((k, f"unknown top-level key{suggest(k, MESH_KEYS) or ''}"))
    if raw.get("units") is not None:
        _not_yet(errors, "units", "units (values with units)")
    for k in ("supports", "load_cases"):
        if k not in raw:
            errors.append((k, "required key is missing"))
    r: dict[str, Any] = {"name": str(raw.get("name") or "study")}
    kind = _domain_type(raw)
    cases = raw.get("load_cases")
    if kind == "shell":
        sd = r["domain"] = _resolve_shell_domain(raw["domain"], errors)
        r["material"] = _resolve_material(raw.get("material"), errors, _SHELL_MATERIAL, "shell")
        r["physics"] = _resolve_shell_physics(raw.get("physics"), errors)
        sets = _shell_sets(sd)
        r["supports"] = _resolve_shell_supports(raw.get("supports"), sets, errors) if "supports" in raw else []
        r["load_cases"] = _resolve_cases(cases, errors, lambda v, p, e: _resolve_shell_loads(v, p, sets, e)) \
            if "load_cases" in raw else []
    else:
        d = _resolve_domain(raw["domain"], errors)
        dim = 2 if d is None else d.pop("dim")
        r["domain"] = d
        r["material"] = _resolve_material(raw.get("material"), errors)
        if dim == 3 and r["material"].get("plane") == "strain":
            errors.append(("material.plane", "plane strain is a 2-D model; a 3-D mesh (generator box) is "
                                             "isotropic 3-D elasticity"))
        r["physics"] = _resolve_physics(raw.get("physics"), errors)
        r["supports"] = _resolve_supports(raw.get("supports"), dim, errors) if "supports" in raw else []
        r["load_cases"] = _resolve_cases(cases, errors, lambda v, p, e: _resolve_loads(v, p, dim, e)) \
            if "load_cases" in raw else []
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
    r["post"] = _resolve_post(raw.get("post", []), errors)
    r["schedule"] = None                                 # what the CLI reads of a grid config, here always none
    if "provenance" in raw:
        r["provenance"] = raw["provenance"]
    if errors:
        raise ConfigError(_tagged(errors, kind))
    return r


def _tagged(errors: list[tuple[str, str]], kind: str = "mesh") -> list[tuple[str, str]]:
    """Every message says which path it comes from (mesh, or shell: the shared messages are renamed)."""
    if kind != "shell":
        return [(p, m if "mesh" in m else f"{m} (domain type mesh)") for p, m in errors]
    out = []
    for p, m in errors:
        m = m.replace("mesh path (domain type mesh)", "shell path (domain type shell)").replace("mesh path",
                                                                                                  "shell path")
        out.append((p, m if "shell" in m else f"{m} (domain type shell)"))
    return out


# ============================================================================================== shell (resolve)
def _resolve_shell_domain(raw: Any, errors: list[tuple[str, str]]) -> dict[str, Any]:
    d = _section(raw, _SHELL_DOMAIN, "domain", errors, "domain type shell")
    surface = normalise_name(str(d.get("surface") or ""))
    if surface not in _SURFACE_KEYS:
        if "surface" in d:
            errors.append(("domain.surface", f'cylinder | rectangle, got "{d.get("surface")}"'))
        return {"type": "shell", "surface": d.get("surface"), "n": [], "thickness": d.get("thickness")}
    own = _SURFACE_KEYS[surface]
    for k in sorted(set(_SURFACE_KEYS["cylinder"] + _SURFACE_KEYS["rectangle"]) - set(own)):
        if isinstance(raw, dict) and k in raw:
            errors.append((join("domain", k), f"not a parameter of surface {surface} (its keys: "
                                              f"{', '.join(own)}, n, thickness; domain type shell)"))
    out: dict[str, Any] = {"type": "shell", "surface": surface}
    n = _number_list(d.get("n"), "domain.n", 2, errors, int) if "n" in d else []
    if n and min(n) < 1:
        errors.append(("domain.n", f"facet counts must be >= 1, got {n}"))
    if surface == "cylinder":
        for k in ("radius", "length"):
            v = d.get(k)
            if v is None:
                errors.append((join("domain", k), "required key of a cylinder surface is missing"))
            elif not (math.isfinite(v) and v > 0):
                errors.append((join("domain", k), f"must be a length > 0, got {v}"))
            out[k] = v
        a = d.get("angle")
        if a is not None and not (math.isfinite(a) and 0 < a < 360):
            errors.append(("domain.angle", f"segment angle in degrees must lie in (0, 360) (null: a closed tube), "
                                           f"got {a}"))
        out["angle"] = a
        if a is None and n and n[0] < 3:
            errors.append(("domain.n", f"a closed tube needs n_theta >= 3, got {n[0]}"))
    else:
        size: list[Any] = []
        if d.get("size") is None:
            errors.append(("domain.size", "required key of a rectangle surface is missing"))
        else:
            size = _number_list(d.get("size"), "domain.size", 2, errors)
            if size and min(size) <= 0:
                errors.append(("domain.size", f"lengths must be > 0, got {size}"))
        out["size"] = size
    t = d.get("thickness")
    if t is not None and not (math.isfinite(t) and t > 0):
        errors.append(("domain.thickness", f"must be a length > 0, got {t}"))
    out.update(n=n, thickness=t)
    return out


def _shell_sets(dom: dict[str, Any]) -> tuple[str, ...]:
    """Node set names of the resolved shell surface (as :func:`toporlogy.fem.shell.cylinder_surface` /
    :func:`toporlogy.fem.shell.flat_rectangle` name them)."""
    if dom.get("surface") == "rectangle":
        return ("left", "right", "bottom", "top")
    if dom.get("surface") == "cylinder":
        return ("bottom", "top") + (("start", "end") if dom.get("angle") is not None else ())
    return ()


def _resolve_shell_physics(raw: Any, errors: list[tuple[str, str]]) -> dict[str, Any]:
    from toporlogy.fem.shell import DRILL
    spec = {"type": raw} if isinstance(raw, str) else ({"type": "shell"} if raw is None else raw)
    out: dict[str, Any] = {"type": "shell", "drill": DRILL}
    if not isinstance(spec, dict) or normalise_name(str(spec.get("type", ""))) != "shell":
        _not_yet(errors, "physics", f"physics {spec.get('type') if isinstance(spec, dict) else spec!r} (shell path "
                                    "physics: shell)")
        return out
    for k in sorted(set(spec) - {"type", "drill"}):
        _not_yet(errors, join("physics", k), f"shell physics parameter {k}")
    if "drill" in spec:
        try:
            out["drill"] = Param("drill", DRILL, float).coerce(spec["drill"], "physics.drill")
        except ConfigError as e:
            errors.extend(e.errors)
            return out
        if not (math.isfinite(out["drill"]) and out["drill"] > 0):
            errors.append(("physics.drill", f"must be a finite number > 0 (0 leaves the drilling rotation of a "
                                            f"flat facet singular), got {out['drill']}"))
    return out


def _resolve_shell_supports(raw: Any, sets: tuple[str, ...], errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for i, s in enumerate(_list(raw, "supports", errors, 1)):
        p = join("supports", i)
        if not isinstance(s, dict):
            errors.append((p, "expected a mapping {set, fix}"))
            continue
        for k in s:
            if k not in ("set", "fix"):
                errors.append((join(p, k), f"unknown key of a shell support (keys: set, fix)"
                                           f"{suggest(k, ['set', 'fix'])}"))
        name = s.get("set")
        if name not in sets:
            errors.append((join(p, "set"), f"a named node set of the shell surface ({', '.join(sets)}), "
                                           f"got {name!r}"))
        fix = s.get("fix", "all")
        comps = list(_SHELL_FIX) if fix == "all" else [fix] if isinstance(fix, str) else fix
        if not isinstance(comps, list) or not comps or any(c not in _SHELL_FIX for c in comps):
            errors.append((join(p, "fix"), f"shell components out of {list(_SHELL_FIX)} or all, got {fix!r}"))
            comps = []
        out.append({"set": name, "fix": sorted(set(comps), key=_SHELL_FIX.index)})
    return out


def _resolve_shell_loads(raw: Any, path: str, sets: tuple[str, ...],
                         errors: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["pressure", "point", "set", "force", "distribute"]
    forms = "{pressure}, {point, force} or {set, force, distribute}"
    for i, ld in enumerate(_list(raw, path, errors, 1)):
        p = join(path, i)
        if not isinstance(ld, dict):
            errors.append((p, f"expected a mapping {forms}"))
            continue
        for k in ld:
            if k not in keys:
                errors.append((join(p, k), f"unknown key of a shell load (keys: {', '.join(keys)}){suggest(k, keys)}"))
        if "set" in ld:
            if "pressure" in ld or "point" in ld:
                errors.append((p, f"a shell load is one of {forms}"))
                continue
            r = _set_load(ld, p, sets, "shell surface", errors)
            if "force" not in ld:
                errors.append((join(p, "force"), "required key of a shell load on a set is missing"))
                continue
            out.append({**r, "force": _number_list(ld["force"], join(p, "force"), 3, errors)})
            continue
        if "distribute" in ld:
            errors.append((join(p, "distribute"), "distribute applies to a load on a set (shell path)"))
        if "pressure" in ld:
            if "point" in ld or "force" in ld:
                errors.append((p, "a shell load is either {pressure} or {point, force}, not both"))
                continue
            try:
                pr = Param("pressure", type=float).coerce(ld["pressure"], join(p, "pressure"))
            except ConfigError as e:
                errors.extend(e.errors)
                continue
            if not math.isfinite(pr):
                errors.append((join(p, "pressure"), f"must be finite, got {pr}"))
            out.append({"pressure": pr})
            continue
        if "point" not in ld or "force" not in ld:
            errors.append((p, "a shell load needs pressure, or point and force"))
            continue
        out.append({"point": _number_list(ld["point"], join(p, "point"), 3, errors),
                    "force": _number_list(ld["force"], join(p, "force"), 3, errors)})
    return out


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
    ``MeshDomain``, ``MeshElasticity`` (``domain.type: shell``: ``ShellDomain``, ``ShellPhysics``), a chain
    ``Embed, MeshFilterLink…, Passive``, the interpolation, compliance, ``MeshVolume`` constraints and MMA."""
    kind = "shell" if r["domain"].get("type") == "shell" else "mesh"
    try:
        return _build_shell(r) if kind == "shell" else _build_mesh(r)
    except ConfigError as e:
        raise ConfigError(_tagged(e.errors, kind)) from e.__cause__


def _build_mesh(r: dict[str, Any]) -> Setup:
    from toporlogy.fem.elements import D_iso, D_plane_strain, D_plane_stress
    from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity

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
    m = r["material"]
    if dim == 3:
        D = D_iso(m["nu"])
    else:
        D = D_plane_strain(m["nu"]) if m["plane"] == "strain" else D_plane_stress(m["nu"])
    physics = MeshElasticity(d, D, fixed)
    prob = physics.problem
    cases = r["load_cases"]
    F = np.zeros((d.ndof, len(cases)))
    for c, case in enumerate(cases):
        label = f"{case['name']} load" if len(cases) > 1 else "load"
        Fc = np.zeros(d.ndof)
        for j, ld in enumerate(case["loads"]):
            lp = join(join(join("load_cases", c), "loads"), j)
            f = np.asarray(ld["force"], float)
            if "set" not in ld:
                used = mesh.used_nodes()
                dist = np.linalg.norm(mesh.points[used] - np.asarray(ld["point"], float), axis=1)
                k = int(used[np.argmin(dist)])
                Fc.reshape(-1, dim)[k] += f
                notes.append(f"{label} {j + 1} at node {k} ({', '.join(f'{v:g}' for v in mesh.points[k])}), "
                             f"{float(dist.min()):.3g} from the given point")
            elif ld["distribute"] == "traction":                # uniform traction force / |facets|, consistent
                facets = mesh.facet_sets.get(ld["set"], np.zeros((0, 2), np.int64))
                if not len(facets):
                    raise ConfigError(f'facet set "{ld["set"]}" is empty', join(lp, "set"))
                fg, _ = prob.facet_geometry(facets)
                area = float(fg.dS.sum()) * (prob.thickness if dim == 2 else 1.0)
                Fc += prob.traction(facets, f / area)
            else:                                               # equal share per node
                nodes = mesh.node_sets[ld["set"]]
                if not nodes.size:
                    raise ConfigError(f'node set "{ld["set"]}" is empty', join(lp, "set"))
                Fc.reshape(-1, dim)[nodes] += f / len(nodes)
        F[:, c] = Fc
    _check_loads(F)
    return _finish(r, d, physics, F, fixed, dim, dim, _loaded_fixed(F, fixed), notes)


def _check_loads(F: np.ndarray) -> None:
    for c in range(F.shape[1]):
        if not np.isfinite(F[:, c]).all():
            raise ConfigError("load case has a load that is not finite (NaN or Inf)", join("load_cases", c))
        if not np.any(F[:, c]):
            raise ConfigError("load case has zero load", join("load_cases", c))


def _loaded_fixed(F: np.ndarray, fixed: np.ndarray) -> list[str]:
    loaded_fixed = np.intersect1d(np.flatnonzero(np.any(F != 0, axis=1)), fixed)
    return [f"{loaded_fixed.size} loaded DOF(s) are also fixed; their load does no work"] if loaded_fixed.size else []


def _finish(r: dict[str, Any], d: Any, physics: Any, F: np.ndarray, fixed: np.ndarray, per_node: int, nforce: int,
            warnings: list[str], notes: list[str]) -> Setup:
    """The part both branches share: chain, interpolation, objective, volume constraints, optimizer, Study, Setup.
    ``per_node``: DOFs per node of F, ``nforce``: how many of them are force components (the resultant)."""
    from toporlogy.design.chain import Chain, Embed, Passive
    from toporlogy.optimization.mesh_study import MeshFilterLink, MeshVolume
    from toporlogy.optimization.study import Constraint, Study

    m = r["material"]
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
    weights = [float(c["weight"]) for c in r["load_cases"]]
    ctx = BuildContext(config=r, domain=d, material=dict(m), run=dict(r["run"]), n_cases=F.shape[1],
                       case_weights=weights, physics=physics, chain=chain, links=links, fixed=fixed)
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
    cases = []
    for c, case in enumerate(r["load_cases"]):
        Fn = F[:, c].reshape(-1, per_node)
        cases.append({"name": case["name"], "weight": weights[c],
                      "resultant": _resultant(Fn[:, :nforce].ravel(), nforce),
                      "n_nodes": int(np.any(Fn != 0, axis=1).sum()), "combination": False})
    post = [(p["type"], _vtu_step(p["name"])) for p in r["post"]]
    return Setup(study, r, fixed, cases, cnames, post, warnings, None, None, {}, {"notes": notes})


def _vtu_step(name: str) -> Any:
    """Post step ``vtu`` of both branches: ``<name>.vtu`` of the cells (mesh: ``mesh.points`` with z = 0 in 2-D and
    ``mesh.cells`` as VTK triangle / quad / tetra / hexahedron; shell: ``mesh.nodes`` and ``mesh.facets`` as VTK
    quads), cell data ``xphys`` and ``E`` = interp(x̄), point data per load case ``displacement_<c>`` (n, 3) and, on a
    shell, ``rotation_<c>`` (n, 3: rx, ry, rz)."""

    def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
        import torch

        from toporlogy.core.vtu import VTK_HEXAHEDRON, VTK_QUAD, VTK_TETRA, VTK_TRIANGLE, write_cells_vtu
        d = study.domain
        shell = not hasattr(d.mesh, "cells")
        x = np.asarray(res.xphys, float).reshape(-1)
        cell: dict[str, Any] = {"xphys": x}
        point: dict[str, Any] = {}
        notes: list[str] = []
        try:
            cell["E"] = np.asarray(study.interp(torch.as_tensor(x)).detach().numpy(), float).reshape(-1)
        except (ValueError, TypeError, RuntimeError) as e:
            notes.append(f"E not written: {type(e).__name__}: {e}")
        per = 6 if shell else d.dim
        U = None if res.U is None else np.asarray(res.U, float).reshape(d.ndof, -1)
        n = d.mesh.nn if shell else d.mesh.n_nodes
        for c in range(0 if U is None else U.shape[1]):
            assert U is not None
            Uc = U[:, c].reshape(n, per)
            u = np.zeros((n, 3))
            u[:, :min(per, 3)] = Uc[:, :3]
            point[f"displacement_{c}"] = u
            if shell:
                point[f"rotation_{c}"] = Uc[:, 3:].copy()
        if shell:
            pts, cells, kind, element = d.mesh.nodes, d.mesh.facets, VTK_QUAD, "flat-facet shell quad"
        else:
            types = {"T3": VTK_TRIANGLE, "Q4": VTK_QUAD, "T4": VTK_TETRA, "H8": VTK_HEXAHEDRON}
            pts, cells, element = d.mesh.points, d.mesh.cells, d.mesh.element
            kind = types[element]
        path = write_cells_vtu(f"{folder}/{name}.vtu", pts, cells, kind, point, cell,
                               meta={"design": "optimized", "element": element})
        return {"file": f"{name}.vtu", "design": "optimized", "cell_arrays": sorted(cell),
                "point_arrays": sorted(point), "notes": notes, "path": path,
                "summary": f"{name}.vtu: {len(cell)} cell and {len(point)} point arrays"
                + (f" ({'; '.join(notes)})" if notes else "")}
    return step


# ============================================================================================== shell (build)
def _shell_rigid_modes(X: np.ndarray) -> np.ndarray:
    """(6 nn, 6) rigid motions of a shell with nodes X (nn, 3): translations, then rotations ω (translation part
    ω × (x − x̄), rotation part ω); coordinates centred and scaled to unit size."""
    n = len(X)
    Y = X - X.mean(axis=0)
    Y = Y / max(float(np.abs(Y).max()), 1e-300)
    cols = []
    for k in range(6):
        c = np.zeros((n, 6))
        if k < 3:
            c[:, k] = 1.0
        else:
            w = np.eye(3)[k - 3]
            c[:, :3] = np.cross(w, Y)
            c[:, 3:] = w
        cols.append(c.ravel())
    return np.column_stack(cols)


def _build_shell(r: dict[str, Any]) -> Setup:
    from toporlogy.fem.shell import FlatShell, cylinder_surface, flat_rectangle
    from toporlogy.optimization.shell_study import ShellDomain, ShellPhysics

    dom = r["domain"]
    n = dom["n"]
    if dom["surface"] == "cylinder":
        angle = None if dom["angle"] is None else math.radians(dom["angle"])
        mesh = cylinder_surface(dom["radius"], dom["length"], n[0], n[1], angle=angle)
        desc = (f"cylinder R {dom['radius']:g}, " + ("closed tube" if angle is None else f"{dom['angle']:g} deg")
                + f", L {dom['length']:g}")
    else:
        mesh = flat_rectangle(dom["size"][0], dom["size"][1], n[0], n[1])
        desc = f"rectangle {dom['size'][0]:g} x {dom['size'][1]:g}"
    d: Any = ShellDomain(mesh)
    notes = [f"shell: {desc}, {n[0]} x {n[1]} facets, t {dom['thickness']:g}, {mesh.ne} facets, {mesh.nn} nodes"]
    fixed_parts = []
    for s in r["supports"]:
        nodes = mesh.node_sets[s["set"]]
        fixed_parts.append((6 * nodes[:, None] + np.array([_SHELL_FIX.index(c) for c in s["fix"]])).ravel())
    fixed = np.unique(np.concatenate(fixed_parts))
    Rm = _shell_rigid_modes(mesh.nodes)[fixed]
    rank = int((np.linalg.svd(Rm, compute_uv=False) > 1e-8).sum()) if len(Rm) else 0
    if rank < 6:
        raise ConfigError(f"the supports leave {6 - rank} rigid motion(s) free; the system is singular", "supports")
    shell = FlatShell(mesh, dom["thickness"], r["material"]["nu"], fixed, drill=r["physics"]["drill"])
    cases = r["load_cases"]
    F = np.zeros((shell.ndof, len(cases)))
    for c, case in enumerate(cases):
        label = f"{case['name']} load" if len(cases) > 1 else "load"
        Fc = np.zeros(shell.ndof)
        for j, ld in enumerate(case["loads"]):
            lp = join(join(join("load_cases", c), "loads"), j)
            if "pressure" in ld:
                Fc += shell.pressure_load(ld["pressure"])
            elif "set" in ld:
                nodes = mesh.node_sets[ld["set"]]
                f = np.asarray(ld["force"], float)
                if not nodes.size:
                    raise ConfigError(f'node set "{ld["set"]}" is empty', join(lp, "set"))
                if ld["distribute"] == "traction":         # uniform line load along the set's boundary edges
                    try:
                        Fc += shell.line_load(nodes, f)
                    except ValueError as e:
                        raise ConfigError(f"distribute traction: {e}", join(lp, "set")) from None
                else:                                       # equal share per node (translations)
                    Fc.reshape(-1, 6)[nodes, :3] += f / len(nodes)
            else:
                dist = np.linalg.norm(mesh.nodes - np.asarray(ld["point"], float), axis=1)
                k = int(np.argmin(dist))
                Fc[6 * k:6 * k + 3] += np.asarray(ld["force"], float)
                notes.append(f"{label} {j + 1} at node {k} ({', '.join(f'{v:g}' for v in mesh.nodes[k])}), "
                             f"{float(dist.min()):.3g} from the given point")
        F[:, c] = Fc
    _check_loads(F)
    return _finish(r, d, ShellPhysics(shell), F, fixed, 6, 3, _loaded_fixed(F, fixed), notes)
