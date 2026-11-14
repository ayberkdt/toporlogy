"""Declarative studies: a YAML/JSON config → :class:`~aerokit.topo.study.Study`, run folders, overrides, sweeps.

Schema (every key optional unless marked *required*; defaults in brackets). Components are a registered name or a
mapping ``{type: <name>, <param>: <value>}`` (:mod:`aerokit.topo.registry`; ``python -m aerokit.topo list`` shows
every component with its parameters and defaults).

Units: any consistent set. Lengths (``h``, ``origin``, selector coordinates, ``t_max``, ``min_solid``) share one
unit, e.g. m; ``E0`` is force/length² (Pa with N and m); forces are force units (N); ``rho`` is mass/length³;
``rmin`` and the ``radius`` of ``local_volume`` are in elements.

.. code-block:: yaml

    name: mbb                      # [config file stem] run-folder suffix
    domain: {type: box, shape: [60, 20], h: 1.0}                    # *required*; also csg, stl (keep / void shapes)
    material: {E0: 1.0, nu: 0.3, rho: 1.0}                          # [1.0, 0.3, 1.0]
    physics: {type: elasticity, element: null, solver: direct}      # element null = H8 (3-D) / Q4 (2-D); or modal
    supports:                                                       # *required*, >= 1 (bc supports)
      - {type: fixed, where: {type: plane, axis: x, at: min}, dofs: x}
    load_cases:                                                     # *required* (optional for physics modal)
      - name: tip                  # [case<i>]
        weight: 1.0                # [1.0] used by compliance
        supports: null             # [null] own supports of this case (null = the supports above)
        loads: [{type: force, where: {type: point, at: [min, max]}, force: [0, -1]}]
    combinations: []               # [[]] {name, factors: {case: factor}, weight}: extra columns (same supports)
    chain: [{type: density_filter, rmin: 1.5}]                      # embed / passive added at the ends if absent
    interpolation: simp            # [simp]
    objective: compliance          # [compliance]
    constraints:                   # [[]]  response <= upper, g = (value - upper) * scale (scale null = 1/|upper|)
      - {response: volume, upper: 0.5, scale: null, name: null}    # upper null: the response's own bound
    optimizer: oc                  # [oc]
    schedule: null                 # [null] e.g. {type: beta_continuation, every: 25}
    robust: null                   # [null] {min_solid, min_void, eta_e, eta_d, eta_i, beta, kernel, eta_max}
    connectivity: null             # [null] {interfaces, ties, constraint, eta, bound, sink, gate, repair, ...}
    run: {max_iter: 100, tol: 0.01, seed: 0, feasibility_tol: 0.001, move_beta_ref: null}
    output: {folder: runs, snapshot_every: 0}
    post: []                       # [[]] post-processing components

``tol`` is the max design change for convergence; ``feasibility_tol`` the relative constraint violation still
reported as satisfied; ``seed`` seeds NumPy and torch; ``move_beta_ref`` (null = off) shrinks the optimizer's move
limit to move0 · min(1, move_beta_ref / β) after every β step of the schedule. The resolved config (all defaults
filled, auto links explicit) is written as ``config.resolved.yaml`` with a ``provenance`` block (git revision,
time, source, overrides); loading it again builds the identical study.

Boundary conditions. Supports and loads build a :class:`aerokit.topo.bc.BoundaryConditions` (selectors, supports,
loads, per-case supports, combinations); its ``check()`` runs in :func:`build` — an error (e.g. an unrestrained
rigid-body mode) is a :class:`ConfigError`; warnings are kept in :attr:`Setup.bc_report`. The physics comes from it
(``ConstrainedElasticity`` for oblique supports and springs, ``PerCaseSupports`` for per-case supports).

``robust``: ``min_solid`` / ``min_void`` (length) set the density filter radius and the thresholds
(:func:`aerokit.topo.robust.robust_chain_settings`; the ``rmin`` written into the resolved chain), or give
``eta_e`` / ``eta_d`` directly. A ``heaviside_projection`` link is added when the chain has none. The thresholds are
shared by ``robust_compliance``, ``dilated_volume``, ``local_volume(source: dilated)`` and ``robust_continuation``.

``connectivity`` (one body): ``interfaces`` (name → selector; null = one per support and per load),
``ties`` (section in elements of frozen minimal ties, null / 0 = none; the interface cells are frozen as well,
otherwise a void interface cell cuts the tie), ``constraint`` (conduction | coupling | none),
``eta``, ``bound`` (null = 20 / 2.5), ``sink``, ``options`` (extra response parameters), ``gate`` (adds the
``connectivity_gate`` post step: verdict in ``summary.json``, exit 1 on FAIL), ``repair`` (bridge | none),
``repair_section``, ``threshold``.

Public functions: :func:`read_config`, :func:`resolve`, :func:`build`, :func:`study_from_dict`, :func:`load_config`,
:func:`apply_overrides`, :func:`parse_sweep`, :func:`sweep_combinations`, :func:`run_setup`, :func:`read_run`.
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aerokit.topo.registry import REGISTRY, BuildContext, ConfigError, Param, join, normalise_name, suggest

__all__ = ["ConfigError", "Setup", "SCHEMA_KEYS", "read_config", "resolve", "build", "build_setup",
           "study_from_dict", "load_config", "apply_overrides", "parse_sweep", "sweep_combinations", "run_setup",
           "write_resolved", "read_run", "status_of", "STATUS_OK"]

SCHEMA_KEYS = ("name", "domain", "material", "physics", "supports", "load_cases", "combinations", "chain",
               "interpolation", "objective", "constraints", "optimizer", "schedule", "robust", "connectivity", "run",
               "output", "post", "provenance")
_REQUIRED = ("domain", "supports")                     # load_cases too, unless physics is modal

_MATERIAL = {"E0": Param("E0", 1.0, float, unit="force/length^2"), "nu": Param("nu", 0.3, float),
             "rho": Param("rho", 1.0, float, unit="mass/length^3")}
_RUN = {"max_iter": Param("max_iter", 100, int), "tol": Param("tol", 0.01, float),
        "seed": Param("seed", 0, int), "feasibility_tol": Param("feasibility_tol", 1e-3, float),
        "move_beta_ref": Param("move_beta_ref", None, float, nullable=True)}
_OUTPUT = {"folder": Param("folder", "runs", str), "snapshot_every": Param("snapshot_every", 0, int)}
_CASE = {"name": Param("name", None, str, nullable=True), "weight": Param("weight", 1.0, float)}
_COMBINATION = {"name": Param("name"), "factors": Param("factors"), "weight": Param("weight", 1.0, float)}
_CONSTRAINT = {"upper": Param("upper", None, float, nullable=True),
               "scale": Param("scale", None, float, nullable=True), "name": Param("name", None, str, nullable=True)}
_ROBUST = {"min_solid": Param("min_solid", None, float, nullable=True, unit="length"),
           "min_void": Param("min_void", None, float, nullable=True, unit="length"),
           "eta_e": Param("eta_e", None, float, nullable=True), "eta_d": Param("eta_d", None, float, nullable=True),
           "eta_i": Param("eta_i", None, float, nullable=True), "beta": Param("beta", None, float, nullable=True),
           "kernel": Param("kernel", "discrete", str), "eta_max": Param("eta_max", 0.75, float)}
_CONNECTIVITY = {"interfaces": Param("interfaces", None, None, nullable=True), "ties": Param("ties", 1, None),
                 "constraint": Param("constraint", "conduction", str, nullable=True),
                 "eta": Param("eta", 0.7, float, nullable=True), "bound": Param("bound", None, float, nullable=True),
                 "sink": Param("sink", None, str, nullable=True), "options": Param("options", None, None,
                                                                                   nullable=True),
                 "gate": Param("gate", True, bool), "repair": Param("repair", "none", str),
                 "repair_section": Param("repair_section", 2, int), "threshold": Param("threshold", 0.5, float)}
_DEFAULTS: dict[str, Any] = {"physics": "elasticity", "chain": [{"type": "density_filter", "rmin": 1.5}],
                             "interpolation": "simp", "objective": "compliance", "constraints": [],
                             "optimizer": "oc", "schedule": None, "post": []}
_TYING = ("symmetry", "extrusion", "pattern")         # links mapping a reduced vector → design vector (before embed)
_NEEDS_FINALIZE = ("manufacturing_checks", "export", "reverify")
STATUS_OK = "ok"


# ============================================================================================== reading
def read_config(path: str | Path) -> dict[str, Any]:
    """Raw config mapping from a ``.yaml`` / ``.yml`` / ``.json`` file (YAML parser otherwise; it reads JSON too)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigError(f"cannot parse {p.name}: {e}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p.name}: the top level must be a mapping, got {type(data).__name__}")
    data.setdefault("name", p.stem)
    return data


# ============================================================================================== resolution
def _section(raw: Any, schema: dict[str, Param], path: str, errors: list[tuple[str, str]],
             what: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        errors.append((path, f"expected a mapping ({what}), got {type(raw).__name__}"))
        raw = {}
    for k in raw:
        if k not in schema:
            errors.append((join(path, k), f"unknown key in {what}{suggest(k, list(schema)) or ''}"
                           f" (keys: {', '.join(schema)})"))
    for k, p in schema.items():
        if k in raw:
            try:
                out[k] = p.coerce(raw[k], join(path, k))
            except ConfigError as e:
                errors.extend(e.errors)
        elif p.required or (p.default is None and not p.nullable):
            errors.append((join(path, k), f"required key of {what} is missing"))
        else:
            out[k] = p.default
    return out


def _list(raw: Any, path: str, errors: list[tuple[str, str]], minimum: int = 0) -> list[Any]:
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        errors.append((path, f"expected a list, got {type(raw).__name__}"))
        return []
    if len(raw) < minimum:
        errors.append((path, f"at least {minimum} entr{'y' if minimum == 1 else 'ies'} required"))
    return raw


def _type_of(spec: Any) -> str | None:
    if isinstance(spec, str):
        return normalise_name(spec)
    if isinstance(spec, dict) and "type" in spec:
        return normalise_name(spec["type"])
    return None


def _h_dim(domain: dict[str, Any] | None, errors: list[tuple[str, str]]) -> tuple[float, int] | None:
    """(h, dim) of a resolved domain spec without building it (robust.min_solid needs them at resolve time)."""
    if domain is None:
        return None
    t, h = domain["type"], domain.get("h")
    dim = None
    if t == "box":
        dim = len(domain["shape"])
    elif t == "stl":
        dim = 3
    elif isinstance(domain.get("shapes"), list) and domain["shapes"]:
        try:
            dim = int(REGISTRY.build("shape", domain["shapes"][0], BuildContext(), "domain.shapes.0").dim)
        except ConfigError as e:
            errors.extend(e.errors)
            return None
    if h is None or dim is None:
        errors.append(("robust.min_solid", f'cannot find h and the dimension of domain type "{t}" before building '
                                           "it; give chain rmin and robust eta_e / eta_d instead"))
        return None
    return float(h), dim


def _robust_settings(rb: dict[str, Any], h: float, dim: int) -> dict[str, float]:
    from aerokit.topo.robust import robust_chain_settings
    return robust_chain_settings(rb["min_solid"], rb["min_void"], h=h, dim=dim, eta_max=rb["eta_max"],
                                 kernel=rb["kernel"])


def _resolve_robust(raw: dict[str, Any], r: dict[str, Any], chain_raw: list[Any],
                    errors: list[tuple[str, str]]) -> list[Any]:
    """Resolve the robust block; with min_solid write rmin into the density_filter links of ``chain_raw``."""
    if raw.get("robust") is None:
        r["robust"] = None
        return chain_raw
    rb = r["robust"] = _section(raw["robust"], _ROBUST, "robust", errors, "robust")
    if rb.get("min_solid") is None:
        if rb.get("min_void") is not None:
            errors.append(("robust.min_void", "needs min_solid as well"))
        return chain_raw
    for k in ("eta_e", "eta_d"):
        if rb.get(k) is not None:
            errors.append((join("robust", k), "give min_solid / min_void or eta_e / eta_d, not both"))
    hd = _h_dim(r.get("domain"), errors)
    if hd is None:
        return chain_raw
    try:
        s = _robust_settings(rb, *hd)
    except ValueError as e:
        errors.append(("robust", str(e)))
        return chain_raw
    out, hit = [], False
    for i, lk in enumerate(chain_raw):
        if _type_of(lk) == "density_filter":
            lk = {"type": lk} if isinstance(lk, str) else dict(lk)
            hit = True
            given = lk.get("rmin")
            if given is not None:
                try:
                    g = float(given)
                except (TypeError, ValueError):
                    g = None
                if g is None or abs(g - s["rmin"]) > 1e-9 * s["rmin"]:
                    errors.append((join(join("chain", i), "rmin"), f"robust.min_solid sets rmin = {s['rmin']:.6g}; "
                                   "remove rmin here or min_solid in the robust block"))
            lk["rmin"] = s["rmin"]
        out.append(lk)
    if not hit:
        errors.append(("chain", "robust.min_solid needs a density_filter link"))
    return out


def _resolve_connectivity(raw: dict[str, Any], errors: list[tuple[str, str]]) -> dict[str, Any] | None:
    if raw.get("connectivity") is None:
        return None
    cb = _section(raw["connectivity"], _CONNECTIVITY, "connectivity", errors, "connectivity")
    t = cb.get("ties")
    if t is None or t is False or t == 0:
        cb["ties"] = None
    elif isinstance(t, dict):
        if set(t) - {"section"} or "section" not in t:
            errors.append(("connectivity.ties", "expected a section in elements, e.g. ties: 1 or ties: {section: 1}"))
        else:
            cb["ties"] = int(t["section"])
    elif isinstance(t, int) and not isinstance(t, bool) and t >= 1:
        cb["ties"] = int(t)
    else:
        errors.append(("connectivity.ties", f"expected a section >= 1 (elements), null or false, got {t!r}"))
    c = cb.get("constraint")
    if c is not None and normalise_name(c) not in ("conduction", "coupling", "none"):
        errors.append(("connectivity.constraint", f'conduction | coupling | none, got "{c}"'))
    cb["constraint"] = None if c is None or normalise_name(c) == "none" else normalise_name(c)
    if cb.get("repair") not in ("none", "bridge"):
        errors.append(("connectivity.repair", f'none | bridge, got "{cb.get("repair")}"'))
    ifc = cb.get("interfaces")
    if ifc is not None:
        if not isinstance(ifc, dict) or len(ifc) < 2:
            errors.append(("connectivity.interfaces", "expected a mapping name -> selector with at least 2 entries"))
            cb["interfaces"] = None
        else:
            cb["interfaces"] = {str(k): REGISTRY.resolve("selector", v, join("connectivity.interfaces", str(k)),
                                                         errors) for k, v in ifc.items()}
    if cb.get("options") is not None and not isinstance(cb["options"], dict):
        errors.append(("connectivity.options", "expected a mapping of response parameters"))
    return cb


def resolve(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw config and fill every default; raises :class:`ConfigError` listing all problems found."""
    errors: list[tuple[str, str]] = []
    if not isinstance(raw, dict):
        raise ConfigError("the config must be a mapping")
    for k in raw:
        if k not in SCHEMA_KEYS:
            errors.append((k, f"unknown top-level key{suggest(k, SCHEMA_KEYS) or ''}"))
    r: dict[str, Any] = {"name": str(raw.get("name") or "study")}
    R = REGISTRY.resolve

    r["domain"] = R("domain", raw["domain"], "domain", errors) if "domain" in raw else None
    r["material"] = _section(raw.get("material"), _MATERIAL, "material", errors, "material")
    r["physics"] = R("physics", raw.get("physics", _DEFAULTS["physics"]), "physics", errors)
    modal = r["physics"] is not None and r["physics"]["type"] == "modal"
    for k in _REQUIRED + (() if modal else ("load_cases",)):
        if k not in raw:
            errors.append((k, "required key is missing"))
    r["supports"] = [R("support", s, join("supports", i), errors)
                     for i, s in enumerate(_list(raw.get("supports"), "supports", errors, 1 if "supports" in raw
                                                 else 0))]
    cases = []
    lc_min = 0 if modal else (1 if "load_cases" in raw else 0)
    for i, c in enumerate(_list(raw.get("load_cases"), "load_cases", errors, lc_min)):
        cp = join("load_cases", i)
        if not isinstance(c, dict):
            errors.append((cp, "expected a mapping with loads (and optional name, weight, supports)"))
            continue
        head = _section({k: v for k, v in c.items() if k not in ("loads", "supports")}, _CASE, cp, errors,
                        "load case")
        if "loads" not in c:
            errors.append((join(cp, "loads"), "required key of load case is missing"))
        loads = [R("load", s, join(join(cp, "loads"), j), errors)
                 for j, s in enumerate(_list(c.get("loads"), join(cp, "loads"), errors, 1 if "loads" in c else 0))]
        sup = None
        if c.get("supports") is not None:
            sup = [R("support", s, join(join(cp, "supports"), j), errors)
                   for j, s in enumerate(_list(c["supports"], join(cp, "supports"), errors, 1))]
        cases.append({"name": head.get("name") or f"case{i + 1}", "weight": head.get("weight", 1.0),
                      "supports": sup, "loads": loads})
    names_seen = [c["name"] for c in cases]
    for i, n in enumerate(names_seen):
        if n in names_seen[:i]:
            errors.append((join(join("load_cases", i), "name"), f'load case name "{n}" is used twice'))
    r["load_cases"] = cases
    combos = []
    for i, cb in enumerate(_list(raw.get("combinations"), "combinations", errors)):
        cp = join("combinations", i)
        s = _section(cb, _COMBINATION, cp, errors, "combination")
        f = s.get("factors")
        if f is not None:
            if not isinstance(f, dict) or not f:
                errors.append((join(cp, "factors"), "expected a mapping load case name -> factor"))
                f = {}
            fac = {}
            for k, v in f.items():
                if k not in names_seen:
                    errors.append((join(join(cp, "factors"), str(k)), f'unknown load case{suggest(k, names_seen)}'))
                try:
                    fac[str(k)] = Param(str(k), None, float).coerce(v, join(join(cp, "factors"), str(k)))
                except ConfigError as e:
                    errors.extend(e.errors)
            s["factors"] = fac
        if s.get("name") is not None:
            s["name"] = str(s["name"])
            if s["name"] in names_seen:
                errors.append((join(cp, "name"), f'name "{s["name"]}" is used twice'))
            names_seen.append(s["name"])
        combos.append(s)
    r["combinations"] = combos

    rb_raw = raw.get("robust")
    sized = isinstance(rb_raw, dict) and rb_raw.get("min_solid") is not None
    chain_raw = _list(raw.get("chain", [{"type": "density_filter"}] if sized else _DEFAULTS["chain"]), "chain",
                      errors)
    chain_raw = _resolve_robust(raw, r, chain_raw, errors)
    links = [R("link", s, join("chain", i), errors) for i, s in enumerate(chain_raw)]
    if all(lk is not None for lk in links):
        if not any(lk["type"] == "embed" for lk in links):
            lead = 0
            while lead < len(links) and links[lead]["type"] in _TYING:
                lead += 1
            links.insert(lead, R("link", "embed", "chain", errors))
        if r["robust"] is not None and not any(lk["type"] == "heaviside_projection" for lk in links):
            at = next((i for i, lk in enumerate(links) if lk["type"] == "passive"), len(links))
            links.insert(at, R("link", {"type": "heaviside_projection", "beta": 1.0,
                                        "eta": r["robust"].get("eta_i") or 0.5}, "chain", errors))
        if not any(lk["type"] == "passive" for lk in links):
            links.append(R("link", "passive", "chain", errors))
    r["chain"] = links
    r["interpolation"] = R("interpolation", raw.get("interpolation", _DEFAULTS["interpolation"]), "interpolation",
                           errors)
    r["objective"] = R("response", raw.get("objective", _DEFAULTS["objective"]), "objective", errors)

    cons = []
    for i, c in enumerate(_list(raw.get("constraints", []), "constraints", errors)):
        cp = join("constraints", i)
        if isinstance(c, dict) and "response" not in c and "type" in c:     # shorthand {type: volume, upper: 0.5}
            c = {"response": {k: v for k, v in c.items() if k not in _CONSTRAINT},
                 **{k: v for k, v in c.items() if k in _CONSTRAINT}}
        if not isinstance(c, dict):
            errors.append((cp, "expected a mapping with response and upper"))
            continue
        if "response" not in c:
            errors.append((join(cp, "response"), "required key of constraint is missing"))
            continue
        head = _section({k: v for k, v in c.items() if k != "response"}, _CONSTRAINT, cp, errors, "constraint")
        cons.append({"response": R("response", c["response"], join(cp, "response"), errors), **head})
    r["constraints"] = cons
    r["optimizer"] = R("optimizer", raw.get("optimizer", _DEFAULTS["optimizer"]), "optimizer", errors)
    sch = raw.get("schedule")
    r["schedule"] = None if sch in (None, "none") else R("schedule", sch, "schedule", errors)
    r["connectivity"] = _resolve_connectivity(raw, errors)
    r["run"] = _section(raw.get("run"), _RUN, "run", errors, "run")
    r["output"] = _section(raw.get("output"), _OUTPUT, "output", errors, "output")
    post_raw = list(_list(raw.get("post", []), "post", errors))
    cb = r["connectivity"]
    if cb is not None and cb.get("gate") and not any(_type_of(p) == "connectivity_gate" for p in post_raw):
        post_raw.insert(0, {"type": "connectivity_gate", "threshold": cb["threshold"], "repair": cb["repair"],
                            "section": cb["repair_section"]})
    r["post"] = [R("post", s, join("post", i), errors) for i, s in enumerate(post_raw)]
    seen: list[str] = []
    for i, p in enumerate(r["post"]):
        if p is None:
            continue
        if p["type"] in _NEEDS_FINALIZE and "finalize" not in seen:
            errors.append((join("post", i), f"post step {p['type']} needs a finalize step before it"))
        seen.append(p["type"])
    if "provenance" in raw:
        r["provenance"] = raw["provenance"]
    run = r["run"]
    if run.get("max_iter", 1) < 1:
        errors.append(("run.max_iter", "must be >= 1"))
    if run.get("tol", 1) < 0:
        errors.append(("run.tol", "must be >= 0"))
    if errors:
        raise ConfigError(errors)
    return r


# ============================================================================================== build
@dataclass
class Setup:
    """A built study with what the CLI prints about it."""

    study: Any
    config: dict[str, Any]                     # resolved
    fixed: np.ndarray
    cases: list[dict[str, Any]]                # name, weight, resultant, n_nodes
    constraint_names: list[str]
    post: list[tuple[str, Callable[..., Any]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bc: Any = None                             # aerokit.topo.bc.BoundaryConditions
    bc_report: Any = None                      # its CheckReport (no errors: those raise ConfigError)
    interfaces: dict[str, int] = field(default_factory=dict)      # connectivity interface name → elements
    info: dict[str, Any] = field(default_factory=dict)            # robust settings, ties, bores, notes

    @property
    def run_options(self) -> dict[str, Any]:
        return dict(self.config["run"])

    def bc_summary(self) -> str:
        """Load and support tables of :meth:`aerokit.topo.bc.BoundaryConditions.summary` (its last line, the
        verdict for the Python API, left out: the CLI prints the check itself)."""
        if self.bc is None:
            return ""
        lines = [ln for ln in self.bc.summary().splitlines() if not ln.startswith("Check ")]
        if not self.config.get("load_cases"):            # modal analysis: the placeholder case has no load
            first = next((i for i, ln in enumerate(lines) if ln.startswith("Supports of")), 0)
            lines = [ln.replace("Supports of modal:", "Supports:") for ln in lines[first:]]
        return "\n".join(lines)


def _resultant(f: np.ndarray, dim: int) -> list[float]:
    return [float(v) for v in f.reshape(-1, dim).sum(axis=0)]


def _as_supports(obj: Any, dim: int, name: str) -> list[Any]:
    """A built support: a bc.Support, or fixed DOF numbers (components that predate bc) → Fixed per direction."""
    from aerokit.topo.bc import Fixed, NodeIds, Support
    if isinstance(obj, Support):
        return [obj]
    dofs = np.unique(np.asarray(obj, np.int64).reshape(-1))
    if not dofs.size:
        raise ValueError("support fixes no DOF")
    return [Fixed(NodeIds(dofs[dofs % dim == k] // dim), [k], name=f"{name} {'xyz'[k]}")
            for k in range(dim) if np.any(dofs % dim == k)]


def _as_load(obj: Any, dim: int, ndof: int, name: str) -> Any:
    """A built load: a bc.Load, or a load vector (ndof,) (components that predate bc) → NodalForces."""
    from aerokit.topo.bc import Load, NodalForces, NodeIds
    if isinstance(obj, Load):
        return obj
    f = np.asarray(obj, float).reshape(-1)
    if f.shape != (ndof,):
        raise ValueError(f"load vector has {f.size} entries, the domain {ndof} DOFs")
    fn = f.reshape(-1, dim)
    nodes = np.flatnonzero(np.any(fn != 0, axis=1))
    if not nodes.size:
        raise ValueError("load is zero")
    return NodalForces(NodeIds(nodes), fn[nodes], name=name)


def _supports(ctx: BuildContext, specs: Sequence[dict[str, Any]], path: str, prefix: str) -> list[Any]:
    from aerokit.topo.bc import EmptySelectionError, part_info
    out = []
    info = part_info(ctx.domain)
    for i, s in enumerate(specs):
        sp = join(path, i)
        built = REGISTRY.build("support", s, ctx, sp)
        default = f"{prefix}{i + 1} {s['type']}"
        objs = _as_supports(built, ctx.domain.dim, default)
        for o in objs:
            if o is built and s.get("name") is None:
                o.name = default
            try:
                o.directions(info)
                o.springs(info)
            except (EmptySelectionError, ValueError) as e:
                raise ConfigError(f"support {s['type']}: {e}", sp) from None
        out += objs
    return out


def _loads(ctx: BuildContext, specs: Sequence[dict[str, Any]], path: str, prefix: str) -> list[Any]:
    from aerokit.topo.bc import EmptySelectionError, part_info
    out = []
    info = part_info(ctx.domain)
    for j, ld in enumerate(specs):
        lp = join(path, j)
        default = f"{prefix}{j + 1} {ld['type']}"
        o = _as_load(REGISTRY.build("load", ld, ctx, lp), ctx.domain.dim, ctx.domain.ndof, default)
        if ld.get("name") is None:
            o.name = default
        try:
            o.apply(info)
        except (EmptySelectionError, ValueError) as e:
            raise ConfigError(f"load {ld['type']}: {e}", lp) from None
        out.append(o)
    return out


def _boundary_conditions(domain: Any, supports: list[Any], cases: list[Any], combos: list[Any]) -> Any:
    from aerokit.topo.bc import BoundaryConditions
    try:
        return BoundaryConditions(domain, supports, cases, combos)
    except ValueError as e:
        raise ConfigError(str(e), "load_cases") from None


def _default_interfaces(bcs: Any) -> dict[str, np.ndarray]:
    """One interface per support and per load (bc node sets → elements touching them); body forces left out."""
    from aerokit.topo.bc import BodyForce
    from aerokit.topo.connectivity import elements_touching_nodes
    d = bcs.domain
    out: dict[str, np.ndarray] = {}
    multi = len(bcs.cases) > 1
    for i, c in enumerate(bcs.cases):
        if i == 0 or c.supports is not None:
            for name, _kind, nodes, _rows in bcs.constraints(i).per_support:
                key = name if c.supports is None or not multi else f"{c.name}.{name}"
                if key not in out and len(nodes):
                    out[key] = elements_touching_nodes(d, nodes)
        for ld, a in zip(c.loads, bcs.applied[c.name], strict=True):
            if isinstance(ld, BodyForce) or getattr(ld, "_no_interface", False) or not len(a.nodes):
                continue
            key = a.name if not multi else f"{c.name}.{a.name}"
            out.setdefault(key, elements_touching_nodes(d, a.nodes))
    return {k: m for k, m in out.items() if m.any()}


def _explicit_interfaces(ctx: BuildContext, specs: dict[str, Any]) -> dict[str, np.ndarray]:
    from aerokit.topo.bc import EmptySelectionError
    from aerokit.topo.connectivity import elements_touching_nodes
    from aerokit.topo.registry import as_selector
    out = {}
    for name, spec in specs.items():
        path = join("connectivity.interfaces", name)
        try:
            nodes = as_selector(REGISTRY.build("selector", spec, ctx, path)).select(ctx.domain)
        except EmptySelectionError as e:
            raise ConfigError(f"selector: {e}", path) from None
        out[name] = elements_touching_nodes(ctx.domain, nodes)
    return out


def _check_bc(bcs: Any, cases: list[dict[str, Any]]) -> Any:
    """bc.check(); errors → ConfigError at supports / load_cases.<i>[.supports]."""
    rep = bcs.check()
    if rep.errors:
        names = [c["name"] for c in cases]
        errs = []
        for it in rep.errors:
            k = names.index(it.case) if it.case in names else None
            if it.code in ("rigid-body", "overlapping-supports", "over-constrained", "support-on-void"):
                path = join(join("load_cases", k), "supports") if k is not None and \
                    cases[k].get("supports") is not None else "supports"
            else:
                path = join("load_cases", k) if k is not None else "load_cases"
            errs.append((path, f"{it.code}: {it.message}"))
        raise ConfigError(errs)
    return rep


def build(resolved: dict[str, Any]) -> Setup:
    """Construct every component of a resolved config (see :func:`resolve`) into a :class:`Setup`."""
    from aerokit.topo.bc import AllNodes, Combination, LoadCase
    from aerokit.topo.chain import Chain, DensityFilter, HeavisideProjection
    from aerokit.topo.manufacturing import GroupAverage, Tying, validate_chain
    from aerokit.topo.registry import SpreadForce
    from aerokit.topo.study import Constraint, Study

    r = resolved
    modal = r["physics"]["type"] == "modal"
    combos_r = r.get("combinations") or []
    weights = [float(c["weight"]) for c in r["load_cases"]] + [float(c["weight"]) for c in combos_r]
    ctx = BuildContext(config=r, material=dict(r["material"]), run=dict(r["run"]),
                       n_cases=max(len(weights), 1), case_weights=weights or [1.0])
    B = REGISTRY.build
    info: dict[str, Any] = {"notes": []}
    d = ctx.domain = B("domain", r["domain"], ctx, "domain")

    rb = r.get("robust")
    if rb is not None:
        from aerokit.topo.robust import RobustSet
        if rb.get("min_solid") is not None:
            s = _robust_settings(rb, d.h, d.dim)
            info["robust"] = {k: float(v) for k, v in s.items()}
            eta_e, eta_d = s["eta_e"], s["eta_d"]
        else:
            eta_e = 0.75 if rb.get("eta_e") is None else rb["eta_e"]
            eta_d = rb.get("eta_d")
            info["robust"] = {"eta_e": eta_e, "eta_d": 1 - eta_e if eta_d is None else eta_d}
        try:
            ctx.robust = RobustSet(eta_e, eta_d, rb.get("eta_i"), rb.get("beta"))
        except ValueError as e:
            raise ConfigError(str(e), "robust") from None

    # ---------------------------------------------------------------- boundary conditions
    default_sup = _supports(ctx, r["supports"], "supports", "S")
    lcs = []
    for i, c in enumerate(r["load_cases"]):
        cp = join("load_cases", i)
        many = len(r["load_cases"]) > 1
        sup = None if c.get("supports") is None else _supports(ctx, c["supports"], join(cp, "supports"),
                                                               f"{c['name']}.S" if many else "S")
        lcs.append(LoadCase(c["name"], _loads(ctx, c["loads"], join(cp, "loads"), "L"), sup))
    if not lcs:                                         # modal analysis without static loads
        dummy = SpreadForce(AllNodes(), np.zeros(d.dim), "no static load")
        dummy._no_interface = True
        lcs = [LoadCase("modal", [dummy])]
    combos = [Combination(cb["name"], dict(cb["factors"])) for cb in combos_r]
    bcs = _boundary_conditions(d, default_sup, lcs, combos)

    cb = r.get("connectivity")
    plan = None
    if cb is not None and cb.get("interfaces"):
        ctx.bc = bcs
        ctx.interfaces = _explicit_interfaces(ctx, cb["interfaces"])
    else:
        ctx.interfaces = _default_interfaces(bcs)
    if cb is not None:
        from aerokit.topo.connectivity import one_body
        if len(ctx.interfaces) < 2:
            raise ConfigError(f"connectivity needs at least 2 interfaces, found {len(ctx.interfaces)}",
                              "connectivity.interfaces")
        opts = dict(cb.get("options") or {})
        if cb["constraint"] == "coupling":
            opts.setdefault("nu", ctx.material["nu"])
        d0 = d
        if cb["ties"]:
            # tie_regions returns the tie WITHOUT the interface cells: an interface cell that is not frozen can go
            # void and cut the tie (measured on examples/topo/split_connectivity2d.yaml: 2 bodies). Freeze them.
            from aerokit.topo.connectivity import with_keep
            pads = np.logical_or.reduce(list(ctx.interfaces.values())) & d.active
            info["frozen_interface_cells"] = int((pads & ~d.keep).sum())
            d0 = with_keep(d, pads)
        try:
            plan = one_body(d0, ctx.interfaces, ties=cb["ties"], constraint=cb["constraint"], eta=cb["eta"],
                            bound=cb["bound"], sink=cb["sink"], threshold=cb["threshold"],
                            repair_section=cb["repair_section"], **opts)
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            raise ConfigError(f"connectivity: {e}", "connectivity") from None
        ctx.interfaces = plan.interfaces
        if plan.ties is not None:
            info["ties"] = int(plan.ties.sum())
            d = ctx.domain = plan.domain
            bcs = _boundary_conditions(d, default_sup, lcs, combos)
    ctx.bc = bcs
    rep = _check_bc(bcs, r["load_cases"])
    if modal:
        F = d.load_vector(1)
        cases: list[dict[str, Any]] = []
    else:
        F = bcs.F("all")
        cases = []
        for i, name in enumerate(bcs.names):
            if not np.any(F[:, i]):
                path = join("load_cases", i) if i < len(r["load_cases"]) else \
                    join("combinations", i - len(r["load_cases"]))
                raise ConfigError("load case has zero load", path)
            cases.append({"name": name, "weight": weights[i], "resultant": _resultant(F[:, i], d.dim),
                          "n_nodes": int(np.any(F[:, i].reshape(-1, d.dim) != 0, axis=1).sum()),
                          "combination": i >= len(r["load_cases"])})
    ctx.n_cases = F.shape[1]
    if modal:
        ctx.case_weights = [1.0]
    ctx.fixed = np.unique(np.concatenate([bcs.constraints(i).fixed for i in range(len(bcs.cases))]))
    warnings = []
    loaded_fixed = np.intersect1d(np.flatnonzero(np.any(F != 0, axis=1)), ctx.fixed)
    if loaded_fixed.size:
        warnings.append(f"{loaded_fixed.size} loaded DOF(s) are also fixed; their load does no work")

    # ---------------------------------------------------------------- physics, chain, responses
    ctx.solver = None
    ctx.physics = B("physics", r["physics"], ctx, "physics")
    for i, lk in enumerate(r["chain"]):
        ctx.links.append(B("link", lk, ctx, join("chain", i)))
    ctx.chain = Chain(ctx.links)
    try:
        validate_chain(ctx.chain)
    except ValueError as e:
        raise ConfigError(str(e), "chain") from None
    if any(isinstance(lk, Tying) for lk in ctx.links) and not any(isinstance(lk, GroupAverage) for lk in ctx.links):
        warnings.append("the chain has a tying link but no group_average after density_filter; the filtered field "
                        "is not tied exactly near box faces")
    ctx.interp = B("interpolation", r["interpolation"], ctx, "interpolation")
    objective = ctx.objective = B("response", r["objective"], ctx, "objective")
    constraints, cnames = ctx.constraints, []
    for i, c in enumerate(r["constraints"]):
        cp = join("constraints", i)
        resp = B("response", c["response"], ctx, join(cp, "response"))
        if c.get("name"):
            resp.name = c["name"]
        if resp.name == objective.name or resp.name in cnames:
            raise ConfigError(f'response name "{resp.name}" is used twice; set {cp}.name', cp)
        upper = c.get("upper")
        if upper is None:
            upper = getattr(resp, "config_upper", None)
            if upper is None:
                raise ConfigError(f"response {c['response']['type']} has no default bound; set upper",
                                  join(cp, "upper"))
        scale = c.get("scale") if c.get("scale") is not None else getattr(resp, "config_scale", None)
        cnames.append(resp.name)
        constraints.append(Constraint(resp, float(upper), scale))
    if plan is not None:
        for con in plan.constraints:
            if con.key in cnames or con.key == objective.name:
                raise ConfigError(f'constraint name "{con.key}" is used twice', "connectivity")
            cnames.append(con.key)
            constraints.append(con)
    if ctx.robust is not None:
        if ctx.chain.find(DensityFilter) is None:
            raise ConfigError("robust realisations need a density_filter link", "chain")
        if ctx.robust.beta is None and ctx.chain.find(HeavisideProjection) is None:
            raise ConfigError("robust realisations need a heaviside_projection link (or robust.beta)", "chain")
    optimizer = B("optimizer", r["optimizer"], ctx, "optimizer")
    if not constraints and getattr(optimizer, "min_constraints", 1) > 0:
        raise ConfigError(f"optimizer {r['optimizer']['type']} needs at least one constraint", "constraints")
    schedule = None if r["schedule"] is None else B("schedule", r["schedule"], ctx, "schedule")
    if r["schedule"] is not None and r["schedule"]["type"] == "beta_continuation" and \
            ctx.chain.find(HeavisideProjection) is None:
        warnings.append("schedule beta_continuation finds no heaviside_projection link; beta stays fixed")
    post = [(p["type"], B("post", p, ctx, join("post", i))) for i, p in enumerate(r["post"])]
    try:
        study = Study(d, ctx.physics, F, ctx.chain, ctx.interp, objective, constraints, optimizer, schedule,
                      move_beta_ref=r["run"].get("move_beta_ref"))
    except ValueError as e:
        raise ConfigError(str(e), "constraints" if "constraint" in str(e) else "") from e
    study.config = r
    bores = ctx.extra.get("bores_found")
    if bores:
        from aerokit.topo.registry import _describe_bores
        info["bores"] = _describe_bores(bores)
    info["notes"] += ctx.extra.get("notes", [])
    return Setup(study, r, ctx.fixed, cases, cnames, post, warnings, bcs, rep,
                 {k: int(m.sum()) for k, m in (ctx.interfaces or {}).items()}, info)


def build_setup(raw: dict[str, Any], overrides: Sequence[str] = ()) -> Setup:
    """:func:`apply_overrides` → :func:`resolve` → :func:`build`."""
    return build(resolve(apply_overrides(raw, overrides)))


def study_from_dict(d: dict[str, Any], overrides: Sequence[str] = ()) -> Any:
    """Study from a raw config mapping; the resolved config is attached as ``study.config``."""
    return build_setup(d, overrides).study


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Any:
    """Study from a YAML/JSON file (``study.config`` = resolved config; run options in ``study.config["run"]``)."""
    return study_from_dict(read_config(path), overrides)


# ============================================================================================== overrides and sweeps
def _split_assignment(text: str, flag: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f'expected key.path=value, got "{text}"', flag)
    return key.strip(), value.strip()


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text) if text != "" else None
    except yaml.YAMLError:
        return text


def set_path(d: dict[str, Any], key: str, value: Any, flag: str = "--set") -> None:
    """Set ``d[a][b][0][c] = value`` for ``key = "a.b.0.c"`` (list indices as integers); creates mapping keys."""
    parts = key.split(".")
    node: Any = d
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        where = ".".join(parts[:i + 1])
        if isinstance(node, list):
            if not part.lstrip("-").isdigit():
                raise ConfigError(f'"{where}": a list needs an integer index', flag)
            k = int(part)
            if not -len(node) <= k < len(node):
                raise ConfigError(f'"{where}": index out of range (list has {len(node)} entries)', flag)
            if last:
                node[k] = value
            else:
                if isinstance(node[k], str):            # component given by name → expand to a mapping
                    node[k] = {"type": node[k]}
                node = node[k]
        elif isinstance(node, dict):
            if last:
                node[part] = value
            else:
                nxt = node.get(part)
                if nxt is None:
                    nxt = node[part] = {}
                elif isinstance(nxt, str):
                    nxt = node[part] = {"type": nxt}
                node = nxt
        else:
            raise ConfigError(f'"{where}": cannot descend into {type(node).__name__}', flag)


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str] | Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Copy of ``raw`` with ``key.path=value`` assignments applied (values parsed as YAML scalars)."""
    out = copy.deepcopy(raw)
    for item in overrides:
        if isinstance(item, tuple):
            set_path(out, item[0], item[1])
        else:
            k, v = _split_assignment(item, "--set")
            set_path(out, k, _parse_value(v))
    return out


def _split_values(text: str) -> list[str]:
    """Split on commas outside brackets: ``"1,[2,3],4"`` → ``["1", "[2,3]", "4"]``."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [s.strip() for s in out if s.strip() != ""]


def parse_sweep(items: Sequence[str]) -> list[tuple[str, list[Any]]]:
    """``["a.b=1,2", "c=x,y"]`` → ``[("a.b", [1, 2]), ("c", ["x", "y"])]``."""
    out = []
    for it in items:
        k, v = _split_assignment(it, "--sweep")
        vals = [_parse_value(s) for s in _split_values(v)]
        if not vals:
            raise ConfigError(f'"{k}": no values', "--sweep")
        if any(k == o[0] for o in out):
            raise ConfigError(f'"{k}" is swept twice', "--sweep")
        out.append((k, vals))
    return out


def sweep_combinations(sweep: Sequence[tuple[str, list[Any]]]) -> list[list[tuple[str, Any]]]:
    """Cartesian product, first key varying slowest."""
    keys = [k for k, _ in sweep]
    return [list(zip(keys, combo, strict=True)) for combo in itertools.product(*[v for _, v in sweep])]


def combination_label(combo: Sequence[tuple[str, Any]]) -> str:
    """Folder-safe label ``chain.1.rmin=1.5__run.tol=0.01``."""
    def safe(v: Any) -> str:
        s = json.dumps(v, separators=(",", ":")) if not isinstance(v, str) else v
        return "".join(ch if ch.isalnum() or ch in "._-+=" else "-" for ch in s)
    return "__".join(f"{k}={safe(v)}" for k, v in combo)


# ============================================================================================== running
def write_resolved(resolved: dict[str, Any], folder: str | Path, source: str | None = None,
                   overrides: Sequence[str] = ()) -> Path:
    """Write ``config.resolved.yaml`` with a provenance block; returns its path."""
    import aerokit.topo as _t
    from aerokit.topo.record import git_revision

    out = {k: v for k, v in resolved.items() if k != "provenance"}
    out["provenance"] = {"git": git_revision(Path(_t.__file__).parent), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "source": source, "overrides": list(overrides)}
    p = Path(folder) / "config.resolved.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def status_of(summary: dict[str, Any]) -> str:
    """"ok", or the problems joined by ", ": "not converged", "constraint violated", "<post step> failed" (a post
    step whose result has ``passed: false``, e.g. connectivity verdict FAIL, failed manufacturing checks, an error)."""
    flags = []
    if not summary.get("converged", False):
        flags.append("not converged")
    if any(not c.get("satisfied", True) for c in summary.get("constraints", [])):
        flags.append("constraint violated")
    for name, out in (summary.get("post") or {}).items():
        if isinstance(out, dict) and out.get("passed") is False:
            flags.append(f"{name} failed")
    return ", ".join(flags) or STATUS_OK


def run_setup(setup: Setup, folder: str | Path, progress: Callable[[int, dict[str, Any]], None] | None = None,
              source: str | None = None, overrides: Sequence[str] = (),
              post_progress: Callable[[str, dict[str, Any] | None, float], None] | None = None) -> dict[str, Any]:
    """Run a built study into ``folder``; returns the summary (also written to ``summary.json``).

    Folder content: ``config.resolved.yaml``, ``setup.json``, ``history.jsonl``, ``result.npz``, ``result.json``
    (:class:`~aerokit.topo.record.Recorder`), ``snapshots.npz`` (if ``output.snapshot_every`` > 0), ``summary.json``,
    and what the post steps write (STL, reports, ``result_repaired.npz``).
    ``progress(it, row)`` gets ``row = {objective, <constraint names>, change, <schedule state>, seconds}``;
    ``post_progress(name, out, seconds)`` is called before each post step (``out`` None) and after it. A post step
    that raises is recorded as ``{"passed": false, "error": ...}`` and the remaining steps still run.
    """
    import torch

    from aerokit.topo.checks import grey_fraction
    from aerokit.topo.record import Recorder

    folder = Path(folder)
    r, st = setup.config, setup.study
    write_resolved(r, folder, source, overrides)
    run = r["run"]
    np.random.seed(int(run["seed"]))                 # global seeds for user components that draw random numbers
    torch.manual_seed(int(run["seed"]))
    st.recorder = Recorder(folder, r["output"]["snapshot_every"], meta={"name": r["name"], "source": source})
    t0 = time.perf_counter()

    def cb(it: int, ev: Any, change: float) -> None:
        if progress is None:
            return
        row = {"objective": ev.f0, **{n: ev.values[n] for n in setup.constraint_names}, "change": change,
               **st.schedule.state(), "seconds": time.perf_counter() - t0}
        progress(it, row)

    res = st.run(max_iter=int(run["max_iter"]), tol=float(run["tol"]), callback=cb)
    ftol = float(run["feasibility_tol"])
    cons = []
    for c, n in zip(st.constraints, setup.constraint_names, strict=True):
        v = float(res.history[n][-1]) if res.history[n] else float("nan")
        ok = bool(v <= c.upper + ftol * max(abs(c.upper), 1e-12))
        cons.append({"name": n, "value": v, "upper": c.upper, "satisfied": ok})
    summary: dict[str, Any] = {
        "name": r["name"], "objective_name": st.objective.name, "objective": res.objective,
        "constraints": cons, "iterations": res.iterations, "max_iter": int(run["max_iter"]),
        "converged": bool(res.converged), "seconds": res.seconds,
        "grey_fraction": grey_fraction(res.xphys, st.domain.active), "tol": float(run["tol"]),
        "config": "config.resolved.yaml", "source": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "post": {}}
    rep = setup.bc_report
    if rep is not None:
        summary["bc_check"] = {"errors": len(rep.errors), "warnings": len(rep.warnings),
                               "issues": [str(i) for i in rep.issues]}
    for name, step in setup.post:
        key, k = name, 2
        while key in summary["post"]:
            key, k = f"{name}_{k}", k + 1
        if post_progress is not None:
            post_progress(key, None, 0.0)
        tp = time.perf_counter()
        try:
            out = step(res, st, folder)
        except Exception as e:  # noqa: BLE001 - a failing post step must not lose the optimised design
            out = {"passed": False, "error": f"{type(e).__name__}: {e}", "summary": f"ERROR {type(e).__name__}: {e}"}
        if out:
            summary["post"][key] = out
        if post_progress is not None:
            post_progress(key, out or {}, time.perf_counter() - tp)
    summary["status"] = status_of(summary)
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def write_sweep_csv(path: str | Path, rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> Path:
    """``sweep.csv``: index, folder, swept keys, status, objective, constraint values, iterations, seconds, grey."""
    cnames: list[str] = []
    for row in rows:
        for c in row.get("constraints", []):
            if c["name"] not in cnames:
                cnames.append(c["name"])
    header = ["index", "folder", *keys, "status", "objective", *cnames, "iterations", "converged", "seconds",
              "grey_fraction"]
    p = Path(path)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            cv = {c["name"]: c["value"] for c in row.get("constraints", [])}
            w.writerow([row["index"], row["folder"], *[json.dumps(row["values"][k]) for k in keys], row["status"],
                        row.get("objective"), *[cv.get(n, "") for n in cnames], row.get("iterations"),
                        row.get("converged"), row.get("seconds"), row.get("grey_fraction")])
    return p


# ============================================================================================== reading run folders
def read_run(folder: str | Path) -> dict[str, Any]:
    """Summary of a finished run: ``summary.json`` (CLI runs) or rebuilt from ``result.json`` + ``history.jsonl``
    (runs recorded from Python with :class:`~aerokit.topo.record.Recorder`). Adds ``history`` (list of rows)."""
    from aerokit.topo.checks import grey_fraction

    f = Path(folder)
    if not f.is_dir():
        raise FileNotFoundError(f"run folder not found: {f}")
    hist_path = f / "history.jsonl"
    history = [json.loads(line) for line in hist_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if hist_path.is_file() else []
    if (f / "summary.json").is_file():
        s = json.loads((f / "summary.json").read_text(encoding="utf-8"))
    elif (f / "result.json").is_file():
        res = json.loads((f / "result.json").read_text(encoding="utf-8"))
        settings = res.get("settings", {})
        last = history[-1] if history else {}
        cons = [{"name": n, "value": last.get(n, float("nan")), "upper": u,
                 "satisfied": bool(last.get(n, float("inf")) <= u + 1e-3 * max(abs(u), 1e-12))}
                for n, u in settings.get("constraints", [])]
        grey = None
        if (f / "result.npz").is_file():
            with np.load(f / "result.npz") as z:
                grey = grey_fraction(z["xphys"])
        s = {"name": f.name, "objective_name": settings.get("objective", "objective"),
             "objective": res.get("objective"), "constraints": cons, "iterations": res.get("iterations"),
             "max_iter": None, "converged": res.get("converged"), "seconds": res.get("seconds"),
             "grey_fraction": grey}
        s["status"] = status_of(s)
    else:
        raise FileNotFoundError(f"{f} has neither summary.json nor result.json (is the run finished?)")
    s["history"] = history
    s["folder"] = str(f)
    setup = f / "setup.json"
    if setup.is_file():
        meta = json.loads(setup.read_text(encoding="utf-8"))
        s.setdefault("git", meta.get("git"))
        s.setdefault("created", meta.get("created"))
    return s
