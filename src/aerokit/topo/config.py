"""Declarative studies: a YAML/JSON config → :class:`~aerokit.topo.study.Study`, run folders, overrides, sweeps.

Schema (every key optional unless marked *required*; defaults in brackets). Components are a registered name or a
mapping ``{type: <name>, <param>: <value>}`` (:mod:`aerokit.topo.registry`; ``python -m aerokit.topo list`` shows
every component with its parameters and defaults).

Units: any consistent set. Lengths (``h``, ``origin``, selector coordinates) share one unit, e.g. m; ``E0`` is
force/length² (Pa with N and m); forces are force units (N); ``rho`` is mass/length³; ``rmin`` is in elements.

.. code-block:: yaml

    name: mbb                      # [config file stem] run-folder suffix
    domain: {type: box, shape: [60, 20], h: 1.0}                    # *required*
    material: {E0: 1.0, nu: 0.3, rho: 1.0}                          # [1.0, 0.3, 1.0]
    physics: {type: elasticity, element: null, solver: direct}      # element null = H8 (3-D) / Q4 (2-D)
    supports:                                                       # *required*, >= 1
      - {type: fixed, where: {type: plane, axis: x, at: min}, dofs: x}
    load_cases:                                                     # *required*, >= 1
      - name: tip                  # [case<i>]
        weight: 1.0                # [1.0] used by compliance
        loads: [{type: force, where: {type: point, at: [min, max]}, force: [0, -1]}]
    chain: [{type: density_filter, rmin: 1.5}]                      # embed / passive added at the ends if absent
    interpolation: simp            # [simp]
    objective: compliance          # [compliance]
    constraints:                   # [[]]  response <= upper, g = (value - upper) * scale (scale null = 1/|upper|)
      - {response: volume, upper: 0.5, scale: null, name: null}
    optimizer: oc                  # [oc]
    schedule: null                 # [null] e.g. {type: beta_continuation, every: 25}
    run: {max_iter: 100, tol: 0.01, seed: 0, feasibility_tol: 0.001}
    output: {folder: runs, snapshot_every: 0}
    post: []                       # [[]] post-processing components

``tol`` is the max design change for convergence; ``feasibility_tol`` the relative constraint violation still
reported as satisfied; ``seed`` seeds NumPy and torch. The resolved config (all defaults filled, auto links
explicit) is written as ``config.resolved.yaml`` with a ``provenance`` block (git revision, time, source, overrides);
loading it again builds the identical study.

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

from aerokit.topo.registry import REGISTRY, BuildContext, ConfigError, Param, join, suggest

__all__ = ["ConfigError", "Setup", "SCHEMA_KEYS", "read_config", "resolve", "build", "build_setup",
           "study_from_dict", "load_config", "apply_overrides", "parse_sweep", "sweep_combinations", "run_setup",
           "write_resolved", "read_run", "status_of", "STATUS_OK"]

SCHEMA_KEYS = ("name", "domain", "material", "physics", "supports", "load_cases", "chain", "interpolation",
               "objective", "constraints", "optimizer", "schedule", "run", "output", "post", "provenance")
_REQUIRED = ("domain", "supports", "load_cases")

_MATERIAL = {"E0": Param("E0", 1.0, float, unit="force/length^2"), "nu": Param("nu", 0.3, float),
             "rho": Param("rho", 1.0, float, unit="mass/length^3")}
_RUN = {"max_iter": Param("max_iter", 100, int), "tol": Param("tol", 0.01, float),
        "seed": Param("seed", 0, int), "feasibility_tol": Param("feasibility_tol", 1e-3, float)}
_OUTPUT = {"folder": Param("folder", "runs", str), "snapshot_every": Param("snapshot_every", 0, int)}
_CASE = {"name": Param("name", None, str, nullable=True), "weight": Param("weight", 1.0, float)}
_CONSTRAINT = {"upper": Param("upper", None, float), "scale": Param("scale", None, float, nullable=True),
               "name": Param("name", None, str, nullable=True)}
_DEFAULTS: dict[str, Any] = {"physics": "elasticity", "chain": [{"type": "density_filter", "rmin": 1.5}],
                             "interpolation": "simp", "objective": "compliance", "constraints": [],
                             "optimizer": "oc", "schedule": None, "post": []}
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
        elif p.default is None and not p.nullable:
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


def resolve(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw config and fill every default; raises :class:`ConfigError` listing all problems found."""
    errors: list[tuple[str, str]] = []
    if not isinstance(raw, dict):
        raise ConfigError("the config must be a mapping")
    for k in raw:
        if k not in SCHEMA_KEYS:
            errors.append((k, f"unknown top-level key{suggest(k, SCHEMA_KEYS) or ''}"))
    for k in _REQUIRED:
        if k not in raw:
            errors.append((k, "required key is missing"))
    r: dict[str, Any] = {"name": str(raw.get("name") or "study")}
    R = REGISTRY.resolve

    r["domain"] = R("domain", raw["domain"], "domain", errors) if "domain" in raw else None
    r["material"] = _section(raw.get("material"), _MATERIAL, "material", errors, "material")
    r["physics"] = R("physics", raw.get("physics", _DEFAULTS["physics"]), "physics", errors)
    r["supports"] = [R("support", s, join("supports", i), errors)
                     for i, s in enumerate(_list(raw.get("supports"), "supports", errors, 1 if "supports" in raw
                                                 else 0))]
    cases = []
    for i, c in enumerate(_list(raw.get("load_cases"), "load_cases", errors, 1 if "load_cases" in raw else 0)):
        cp = join("load_cases", i)
        if not isinstance(c, dict):
            errors.append((cp, "expected a mapping with loads (and optional name, weight)"))
            continue
        head = _section({k: v for k, v in c.items() if k != "loads"}, _CASE, cp, errors, "load case")
        if "loads" not in c:
            errors.append((join(cp, "loads"), "required key of load case is missing"))
        loads = [R("load", s, join(join(cp, "loads"), j), errors)
                 for j, s in enumerate(_list(c.get("loads"), join(cp, "loads"), errors, 1 if "loads" in c else 0))]
        cases.append({"name": head.get("name") or f"case{i + 1}", "weight": head.get("weight", 1.0),
                      "loads": loads})
    names_seen = [c["name"] for c in cases]
    for i, n in enumerate(names_seen):
        if n in names_seen[:i]:
            errors.append((join(join("load_cases", i), "name"), f'load case name "{n}" is used twice'))
    r["load_cases"] = cases

    links = [R("link", s, join("chain", i), errors)
             for i, s in enumerate(_list(raw.get("chain", _DEFAULTS["chain"]), "chain", errors))]
    if all(lk is not None for lk in links):
        if not any(lk["type"] == "embed" for lk in links):
            links.insert(0, R("link", "embed", "chain", errors))
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
    r["run"] = _section(raw.get("run"), _RUN, "run", errors, "run")
    r["output"] = _section(raw.get("output"), _OUTPUT, "output", errors, "output")
    r["post"] = [R("post", s, join("post", i), errors)
                 for i, s in enumerate(_list(raw.get("post", []), "post", errors))]
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

    @property
    def run_options(self) -> dict[str, Any]:
        return dict(self.config["run"])


def _resultant(f: np.ndarray, dim: int) -> list[float]:
    return [float(v) for v in f.reshape(-1, dim).sum(axis=0)]


def build(resolved: dict[str, Any]) -> Setup:
    """Construct every component of a resolved config (see :func:`resolve`) into a :class:`Setup`."""
    from aerokit.topo.chain import Chain, HeavisideProjection
    from aerokit.topo.study import Constraint, Study

    r = resolved
    ctx = BuildContext(config=r, material=dict(r["material"]), run=dict(r["run"]), n_cases=len(r["load_cases"]),
                       case_weights=[float(c["weight"]) for c in r["load_cases"]])
    B = REGISTRY.build
    d = ctx.domain = B("domain", r["domain"], ctx, "domain")
    fixed_parts = [np.asarray(B("support", s, ctx, join("supports", i)), np.int64)
                   for i, s in enumerate(r["supports"])]
    ctx.fixed = np.unique(np.concatenate(fixed_parts)) if fixed_parts else np.zeros(0, np.int64)
    F = d.load_vector(ctx.n_cases)
    cases = []
    for i, c in enumerate(r["load_cases"]):
        for j, ld in enumerate(c["loads"]):
            path = join(join(join("load_cases", i), "loads"), j)
            f = np.asarray(B("load", ld, ctx, path), float).reshape(-1)
            if f.shape != (d.ndof,):
                raise ConfigError(f"load vector has {f.size} entries, the domain {d.ndof} DOFs", path)
            F[:, i] += f
        if not np.any(F[:, i]):
            raise ConfigError("load case has zero load", join("load_cases", i))
        cases.append({"name": c["name"], "weight": c["weight"], "resultant": _resultant(F[:, i], d.dim),
                      "n_nodes": int(np.any(F[:, i].reshape(-1, d.dim) != 0, axis=1).sum())})
    loaded_fixed = np.intersect1d(np.flatnonzero(np.any(F != 0, axis=1)), ctx.fixed)
    warnings = []
    if loaded_fixed.size:
        warnings.append(f"{loaded_fixed.size} loaded DOF(s) are also fixed; their load does no work")

    ctx.solver = None
    ctx.physics = B("physics", r["physics"], ctx, "physics")
    ctx.chain = Chain([B("link", lk, ctx, join("chain", i)) for i, lk in enumerate(r["chain"])])
    ctx.interp = B("interpolation", r["interpolation"], ctx, "interpolation")
    objective = B("response", r["objective"], ctx, "objective")
    constraints, cnames = [], []
    for i, c in enumerate(r["constraints"]):
        cp = join("constraints", i)
        resp = B("response", c["response"], ctx, join(cp, "response"))
        if c.get("name"):
            resp.name = c["name"]
        if resp.name == objective.name or resp.name in cnames:
            raise ConfigError(f'response name "{resp.name}" is used twice; set {cp}.name', cp)
        cnames.append(resp.name)
        constraints.append(Constraint(resp, float(c["upper"]), c.get("scale")))
    optimizer = B("optimizer", r["optimizer"], ctx, "optimizer")
    if not constraints and getattr(optimizer, "min_constraints", 1) > 0:
        raise ConfigError(f"optimizer {r['optimizer']['type']} needs at least one constraint", "constraints")
    schedule = None if r["schedule"] is None else B("schedule", r["schedule"], ctx, "schedule")
    if r["schedule"] is not None and r["schedule"]["type"] == "beta_continuation" and \
            ctx.chain.find(HeavisideProjection) is None:
        warnings.append("schedule beta_continuation finds no heaviside_projection link; beta stays fixed")
    post = [(p["type"], B("post", p, ctx, join("post", i))) for i, p in enumerate(r["post"])]
    try:
        study = Study(d, ctx.physics, F, ctx.chain, ctx.interp, objective, constraints, optimizer, schedule)
    except ValueError as e:
        raise ConfigError(str(e), "constraints" if "constraint" in str(e) else "") from e
    study.config = r
    return Setup(study, r, ctx.fixed, cases, cnames, post, warnings)


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
    """"ok", "not converged", "constraint violated" or both joined by ", "."""
    flags = []
    if not summary.get("converged", False):
        flags.append("not converged")
    if any(not c.get("satisfied", True) for c in summary.get("constraints", [])):
        flags.append("constraint violated")
    return ", ".join(flags) or STATUS_OK


def run_setup(setup: Setup, folder: str | Path, progress: Callable[[int, dict[str, Any]], None] | None = None,
              source: str | None = None, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Run a built study into ``folder``; returns the summary (also written to ``summary.json``).

    Folder content: ``config.resolved.yaml``, ``setup.json``, ``history.jsonl``, ``result.npz``, ``result.json``
    (:class:`~aerokit.topo.record.Recorder`), ``snapshots.npz`` (if ``output.snapshot_every`` > 0), ``summary.json``.
    ``progress(it, row)`` gets ``row = {objective, <constraint names>, change, <schedule state>, seconds}``.
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
    for name, step in setup.post:
        out = step(res, st, folder)
        if out:
            summary["post"][name] = out
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
