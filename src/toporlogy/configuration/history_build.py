"""A config with a top-level ``analysis: {type: history}``: topology optimisation of a time-dependent (linear
viscoelastic or Norton creep) plane-strain structure with the exact discrete history adjoint, resolved and built by
its own path (``docs/foundations/HISTORY_YAML.md``, prereg ``docs/foundations/prereg/HY_history_yaml.md``).
:func:`~toporlogy.configuration.resolve.resolve` / :func:`~toporlogy.configuration.build.build_setup` hand such a config
to :func:`resolve_history` / :func:`build_history`; the grid path and the mesh path (``mesh_build.py``) do not see it.

The dispatch key is ``analysis`` (not ``domain.type``): the domain keys are those of the mesh path, what differs is the
analysis. Schema (defaults in brackets; SI inside, see ``units``):

.. code-block:: yaml

    name: viscoelastic_service         # [config file stem]
    analysis: {type: history, plane: strain}     # *required*; plane [strain]: strain only
    units: {length: m, force: N, time: s}        # [SI]; another unit needs its factor to SI:
                                                 #   {length: {name: mm, factor: 1.0e-3}}
    domain: {generator: rectangle, n: [24, 8], size: [3.0, 1.0], element: Q4, perturb: 0.0, seed: 0, thickness: 1.0}
                                       # *required* n, size; generator rectangle (2-D) only; element Q4 | T3 [Q4];
                                       # thickness in length units [1.0]
    material: materials/card.yaml      # *required*: a path (relative to the config file) or a material card mapping
                                       # (optimization.material_cards); model generalized_maxwell (no shift) -> linear
                                       # viscoelastic, norton (no arrhenius) -> Norton creep
    supports: [{set: left, fix: all}]  # *required*, >= 1 (the mesh path form: set left|right|bottom|top, fix x|y|all)
    loads:                             # *required*, >= 1, summed
      - {point: [3.0, 0.0], force: [0, -1.0e+3], time: {ramp: 10.0}}
      - {set: right, force: [0, -5.0e+2], time: {table: [[0, 0], [10, 1], [100, 1], [200, 0.5]]}}
                                       # force: total force vector; a set splits it equally over its nodes; a point
                                       # load acts on the nearest node; time (*required*): exactly one of
                                       #   ramp: t_r    s(t) = min(t / t_r, 1)
                                       #   hold: true   s(t) = 1
                                       #   table: [[t, s], ...]   piecewise linear, first knot t = 0, then constant
                                       # F(t_k) = s(t_k) F; ramp / knot times must be step-grid times
    times: [1, 3, 10, 30, 100, 300]    # *required*; end times of the steps, strictly increasing; or
                                       # {logspace: [t0, t1, n], events: [10.0, 250.0]} (n geometric points from t0 to
                                       # t1, plus the events; a point within 1e-9 relative of an event is the event)
    objective: {compliance_at: 300}    # *required*; or {max_compliance_at: [30, 300]} (>= 2 times; min-max); the times
                                       # must be step-grid times
    constraints: [{response: volume, upper: 0.4}]   # *required*, exactly one volume constraint (fraction in (0, 1])
    optimizer: {type: mma, max_iter: 40, tol: 0.0, kkt_tol: null, move: 0.2}   # [mma with these defaults]
    interpolation: {type: simp, penal: 3.0, emin_ratio: 1.0e-9}                # [simp with these defaults]
    chain: [{type: density_filter, rmin: 0.125}]    # *required*: exactly one density filter, rmin a length
    post: [{type: vtu, name: design}, {type: summary, name: history_summary}]   # [[]]
    output: {folder: runs}             # [runs]

Anything else is a :class:`~toporlogy.configuration.registry.ConfigError` with the key path: unknown keys, a 3-D
domain, plane stress, an unsupported or temperature-dependent material, a time that is not a step-grid time, a
non-SI unit without a factor, ``distribute: traction``, ... (``docs/foundations/HISTORY_YAML.md``).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporlogy.configuration.mesh_build import _AXES, _SETS, _check_rigid, _mesh, _number_list, _resolve_supports
from toporlogy.configuration.registry import ConfigError, join, normalise_name, suggest
from toporlogy.configuration.resolve import _UniqueKeyLoader, _list

__all__ = ["HISTORY_KEYS", "HistorySetup", "is_history_config", "anchor_paths", "resolve_history", "build_history"]

HISTORY_KEYS = ("name", "analysis", "units", "domain", "material", "supports", "loads", "times", "temperature",
                "objective",
                "constraints", "optimizer", "interpolation", "chain", "post", "output", "provenance")
_SI = {"length": "m", "force": "N", "time": "s"}
_DOMAIN_KEYS = ("type", "generator", "n", "size", "element", "perturb", "seed", "thickness")
_LOAD_KEYS = ("set", "point", "force", "time", "distribute")
_TIME_FORMS = ("ramp", "hold", "table")
_OPT_KEYS = ("type", "max_iter", "tol", "kkt_tol", "move")
_INTERP_KEYS = ("type", "penal", "emin_ratio")
_POST_TYPES = ("vtu", "summary")
_GRID_TOL = 1e-9                                      # relative tolerance of "this time is on the step grid"
_NOT_YET = "is not supported by the history path (analysis type history) yet"
_GRID_ONLY = ("load_cases", "combinations", "physics", "schedule", "robust", "connectivity", "run", "schema_version")
_Errors = list[tuple[str, str]]


def is_history_config(raw: Any) -> bool:
    """True when ``raw`` is a mapping with a top-level ``analysis`` key (any value: a wrong ``analysis.type`` is then
    reported by :func:`resolve_history` instead of as an unknown key of the grid path)."""
    return isinstance(raw, dict) and "analysis" in raw


def anchor_paths(raw: dict[str, Any], base: str | Path) -> None:
    """Make a relative ``material`` card path absolute against the directory ``base`` of the config file (in place;
    :func:`~toporlogy.configuration.resolve.read_config` calls it for a history config)."""
    m = raw.get("material")
    if isinstance(m, str) and m.strip() and not Path(m).expanduser().is_absolute():
        raw["material"] = str(Path(base) / Path(m).expanduser())


# ============================================================================================== small readers
def _num(v: Any, path: str, errors: _Errors, what: str = "a number", positive: bool = False,
         nonneg: bool = False) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        hint = (" (YAML 1.1 reads an exponent without a sign as text: write 5.0e+6, not 5.0e6)"
                if isinstance(v, str) and re.fullmatch(r"[-+]?\d+(\.\d*)?[eE]\d+", v.strip()) else "")
        errors.append((path, f"expected {what}, got {v!r}{hint}"))
        return None
    x = float(v)
    if not math.isfinite(x):
        errors.append((path, f"must be finite, got {v!r}"))
        return None
    if positive and not x > 0.0:
        errors.append((path, f"must be > 0, got {x:g}"))
        return None
    if nonneg and x < 0.0:
        errors.append((path, f"must be >= 0, got {x:g}"))
        return None
    return x


def _int(v: Any, path: str, errors: _Errors, minimum: int) -> int | None:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
        errors.append((path, f"must be an integer >= {minimum}, got {v!r}"))
        return None
    return int(v)


def _mapping(raw: Any, path: str, keys: tuple[str, ...], errors: _Errors, what: str) -> dict[str, Any]:
    """``raw`` as a dict with every key in ``keys`` (others are errors at ``path.key``)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append((path, f"expected a mapping ({what}), got {type(raw).__name__}"))
        return {}
    for k in raw:
        if k not in keys:
            errors.append((join(path, k), f"unknown key of {what}{suggest(str(k), list(keys)) or ''} "
                                          f"(keys: {', '.join(keys)})"))
    return {k: v for k, v in raw.items() if k in keys}


def _snap(t: float, grid: np.ndarray) -> int | None:
    """Index of the grid time equal to ``t`` within :data:`_GRID_TOL` (relative), else None."""
    i = int(np.argmin(np.abs(grid - t)))
    return i if abs(grid[i] - t) <= _GRID_TOL * max(abs(t), abs(grid[i]), 1e-300) else None


def _off_grid(t: float, grid: np.ndarray) -> str:
    lo, hi = grid[grid <= t], grid[grid > t]
    near = ", ".join(f"{v:g}" for v in ([lo[-1]] if lo.size else []) + ([hi[0]] if hi.size else []))
    return (f"{t:g} s is not a step-grid time (neighbours {near}); add it to times (times.events with a logspace "
            f"grid): a load kink or an objective time between two steps is not represented")


# ============================================================================================== sections
def _resolve_analysis(raw: Any, errors: _Errors) -> dict[str, str]:
    a = _mapping(raw, "analysis", ("type", "plane"), errors, "analysis")
    if raw is not None and not isinstance(raw, dict):
        return {"type": "history", "plane": "strain"}
    t = normalise_name(str(a.get("type", "")))
    if t != "history":
        errors.append(("analysis.type", f"history is the only analysis type here, got {a.get('type')!r}"))
    plane = normalise_name(str(a.get("plane", "strain")))
    if plane == "stress":
        errors.append(("analysis.plane", "plane stress is not supported by the history path (the structures are "
                                         "plane strain); use plane: strain"))
    elif plane != "strain":
        errors.append(("analysis.plane", f"strain is the only plane state of the history path, got "
                                         f"{a.get('plane')!r}"))
    return {"type": "history", "plane": "strain"}


def _resolve_units(raw: Any, errors: _Errors) -> dict[str, tuple[str, float]]:
    out = {k: (v, 1.0) for k, v in _SI.items()}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        errors.append(("units", f"expected a mapping of unit kind to unit, got {type(raw).__name__}"))
        return out
    for k, v in raw.items():
        p = join("units", k)
        if k not in _SI:
            errors.append((p, f"unknown unit kind{suggest(str(k), list(_SI)) or ''} (kinds: {', '.join(_SI)}; the "
                              "material card carries its own units)"))
        elif isinstance(v, str):
            if v != _SI[k]:
                errors.append((p, f"{v!r} is not the SI unit of {k} ({_SI[k]}); a non-SI unit needs its factor to "
                                  f"{_SI[k]}: {{name: {v}, factor: <number>}}"))
        elif isinstance(v, dict):
            for kk in v:
                if kk not in ("name", "factor"):
                    errors.append((join(p, kk), "unknown key (keys: name, factor)"))
            name = v.get("name")
            if not (isinstance(name, str) and name.strip()):
                errors.append((join(p, "name"), f"required: the unit's name, got {name!r}"))
            if "factor" not in v:
                errors.append((join(p, "factor"), f"required: the factor that converts {name or 'the unit'} to "
                                                  f"{_SI[k]} (a non-SI unit without a factor is refused)"))
            else:
                f = _num(v["factor"], join(p, "factor"), errors, "a number", positive=True)
                if f is not None and isinstance(name, str) and name.strip():
                    out[k] = (name.strip(), f)
        else:
            errors.append((p, f"expected a unit name or {{name, factor}}, got {v!r}"))
    return out


def _resolve_domain(raw: Any, lf: float, errors: _Errors) -> dict[str, Any]:
    d = _mapping(raw, "domain", _DOMAIN_KEYS, errors, "domain")
    if raw is None:
        errors.append(("domain", "required key is missing"))
    if "type" in d and normalise_name(str(d["type"])) != "mesh":
        errors.append(("domain.type", f"omit it (or mesh) here: the history path takes the mesh generator keys, "
                                      f"got {d['type']!r}"))
    gen = normalise_name(str(d.get("generator", "rectangle")))
    if gen == "box":
        errors.append(("domain.generator", "3-D is not supported by the history path (plane strain, generator "
                                           "rectangle only)"))
    elif gen != "rectangle":
        errors.append(("domain.generator", f'rectangle is the only generator of the history path, got '
                                           f'"{d.get("generator")}"'))
    out: dict[str, Any] = {"generator": "rectangle"}
    n = _number_list(d.get("n"), "domain.n", 2, errors, int) if "n" in d else []
    if "n" not in d:
        errors.append(("domain.n", "required key of the domain is missing"))
    if n and min(n) < 1:
        errors.append(("domain.n", f"element counts must be >= 1, got {n}"))
    size = _number_list(d.get("size"), "domain.size", 2, errors) if "size" in d else []
    if "size" not in d:
        errors.append(("domain.size", "required key of the domain is missing"))
    if size and min(size) <= 0:
        errors.append(("domain.size", f"lengths must be > 0, got {size}"))
    out["n"], out["size"] = n, [v * lf for v in size]
    el = d.get("element") or "Q4"
    if el not in ("Q4", "T3"):
        errors.append(("domain.element", f"Q4 | T3 for the 2-D history path (3-D elements are not supported), "
                                         f"got {el!r}"))
    out["element"] = el
    per = _num(d.get("perturb", 0.0), "domain.perturb", errors, "a length", nonneg=True)
    out["perturb"] = 0.0 if per is None else per * lf
    seed = _int(d.get("seed", 0), "domain.seed", errors, 0)
    out["seed"] = 0 if seed is None else seed
    th = _num(d.get("thickness", 1.0), "domain.thickness", errors, "a length", positive=True)
    out["thickness"] = 1.0 * lf if th is None else th * lf
    return out


def _read_card(raw: Any, errors: _Errors) -> dict[str, Any] | None:
    """The material mapping: given inline, or read from the YAML file named by a string."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        errors.append(("material", "required: a material card mapping or the path of a card YAML file, got "
                                   f"{raw!r}"))
        return None
    p = Path(raw).expanduser()
    if not p.is_file():
        errors.append(("material", f"material card file not found: {p}"))
        return None
    try:
        data = yaml.load(p.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as e:
        errors.append(("material", f"cannot parse the card file {p.name}: {e}"))
        return None
    if not isinstance(data, dict):
        errors.append(("material", f"the card file {p.name} must hold a mapping, got {type(data).__name__}"))
        return None
    return data


def _resolve_material(raw: Any, errors: _Errors, thermal: bool = False) -> tuple[dict[str, Any] | None, Any]:
    """(the card as a mapping in its own units, the loaded :class:`MaterialCard`). ``thermal``: the config has a
    temperature schedule; a shifted / Arrhenius card is accepted only then
    (``prereg/HT_history_yaml_temperature.md``)."""
    from toporlogy.optimization.material_cards import load_material_card
    card = _read_card(raw, errors)
    if card is None:
        return None, None
    try:
        mc = load_material_card(card)
    except ValueError as e:
        msg = str(e)
        m = re.match(r"^([A-Za-z_][\w.\[\]]*): (.*)$", msg, re.S)
        errors.append((join("material", m.group(1)) if m else "material", m.group(2) if m else msg))
        return None, None
    if mc.kind not in ("generalized_maxwell", "norton"):
        errors.append(("material.model", f"{mc.kind!r} is not supported by the history path (generalized_maxwell: "
                                         "linear viscoelastic; norton: Norton creep)"))
        return None, None
    has_t = ((mc.kind == "generalized_maxwell" and mc.model.shift is not None)
             or (mc.kind == "norton" and mc.model.arrhenius is not None))
    if has_t and not thermal:
        key = "shift" if mc.kind == "generalized_maxwell" else "arrhenius"
        errors.append((f"material.parameters.{key}", f"without a temperature schedule the history path is "
                                                     f"isothermal; a temperature-dependent card ({key}) needs "
                                                     "temperature: {table: [[t, T], ...]} (kelvin)"))
        return None, None
    if thermal and not has_t:
        errors.append(("temperature", "a temperature schedule needs a card with a temperature dependence "
                                      "(generalized_maxwell with shift, or norton with arrhenius)"))
        return None, None
    return mc.to_dict(), mc


def _resolve_temperature(raw: Any, grid: np.ndarray | None, tf: float, errors: _Errors) -> dict[str, Any] | None:
    """{table: [[t (s), T (K)], ...]}: first point at t = 0, strictly increasing t, covering the last step time; T > 0.
    No extrapolation (piecewise linear between the points)."""
    if not isinstance(raw, dict) or set(raw) != {"table"}:
        errors.append(("temperature", f"expected {{table: [[t, T], ...]}} (t in the time unit, T in kelvin), got "
                                      f"{raw!r}"))
        return None
    tab = raw["table"]
    if not isinstance(tab, list) or len(tab) < 2:
        errors.append(("temperature.table", "expected at least two [t, T] points"))
        return None
    ts, Ts = [], []
    for i, row in enumerate(tab):
        if not (isinstance(row, list) and len(row) == 2):
            errors.append((join("temperature.table", i), f"expected [t, T], got {row!r}"))
            return None
        t = _num(row[0], join(join("temperature.table", i), 0), errors, "a time", nonneg=True)
        T = _num(row[1], join(join("temperature.table", i), 1), errors, "an absolute temperature (K)",
                 positive=True)
        if t is None or T is None:
            return None
        ts.append(t * tf)
        Ts.append(T)
    if ts[0] != 0.0:
        errors.append(("temperature.table.0.0", f"the schedule must start at t = 0, got {ts[0]:g}"))
        return None
    if any(b <= a for a, b in zip(ts[:-1], ts[1:], strict=True)):
        errors.append(("temperature.table", "the times must be strictly increasing"))
        return None
    if grid is not None and grid.size and ts[-1] < float(grid[-1]) * (1.0 - _GRID_TOL):
        errors.append(("temperature.table", f"the schedule ends at {ts[-1]:g} s, before the last step time "
                                            f"{float(grid[-1]):g} s (no extrapolation)"))
        return None
    return {"table": [[t, T] for t, T in zip(ts, Ts, strict=True)]}


def _resolve_times(raw: Any, tf: float, errors: _Errors) -> np.ndarray | None:
    if raw is None:
        errors.append(("times", "required key is missing: a list of end times or {logspace: [t0, t1, n], events: "
                                "[...]}"))
        return None
    if isinstance(raw, dict):
        for k in raw:
            if k not in ("logspace", "events"):
                errors.append((join("times", k), f"unknown key{suggest(str(k), ['logspace', 'events']) or ''} "
                                                 "(keys: logspace, events)"))
        ls = raw.get("logspace")
        if not (isinstance(ls, list) and len(ls) == 3):
            errors.append(("times.logspace", f"expected [t0, t1, n], got {ls!r}"))
            return None
        t0 = _num(ls[0], "times.logspace.0", errors, "a time", positive=True)
        t1 = _num(ls[1], "times.logspace.1", errors, "a time", positive=True)
        n = _int(ls[2], "times.logspace.2", errors, 2)
        if t0 is None or t1 is None or n is None:
            return None
        if not t0 < t1:
            errors.append(("times.logspace", f"need 0 < t0 < t1, got [{t0:g}, {t1:g}]"))
            return None
        ev: list[float] = []
        for i, e in enumerate(_list(raw.get("events"), "times.events", errors)):
            v = _num(e, join("times.events", i), errors, "a time", nonneg=True)
            if v is not None:
                ev.append(v * tf)
        ev_arr = np.array(sorted(ev))
        if ev_arr.size > 1 and np.any(np.diff(ev_arr) <= _GRID_TOL * ev_arr[1:]):
            errors.append(("times.events", "the events must be distinct"))
            return None
        pts = np.geomspace(t0 * tf, t1 * tf, n)
        keep = [p for p in pts if not (ev_arr.size and np.any(np.abs(ev_arr - p) <= _GRID_TOL * max(abs(p), 1e-300)))]
        grid = np.sort(np.concatenate([np.array(keep), ev_arr]))
        return grid
    if not isinstance(raw, list) or not raw:
        errors.append(("times", f"expected a non-empty list of times or {{logspace, events}}, got {raw!r}"))
        return None
    vals = [_num(v, join("times", i), errors, "a time", nonneg=True) for i, v in enumerate(raw)]
    if any(v is None for v in vals):
        return None
    arr = np.array(vals, float) * tf
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        errors.append(("times", "the times must be strictly increasing"))
        return None
    return arr


def _resolve_time_function(raw: Any, path: str, grid: np.ndarray | None, tf: float,
                           errors: _Errors) -> dict[str, Any] | None:
    if raw is None:
        errors.append((path, "required key of a load is missing: time: {ramp: t_r} | {hold: true} | "
                             "{table: [[t, s], ...]}"))
        return None
    if not isinstance(raw, dict):
        errors.append((path, f"expected a mapping with one of ramp, hold, table, got {raw!r}"))
        return None
    for k in raw:
        if k not in _TIME_FORMS:
            errors.append((join(path, k), f"unknown key of a time function{suggest(str(k), list(_TIME_FORMS)) or ''} "
                                          f"(forms: {', '.join(_TIME_FORMS)})"))
    forms = [k for k in raw if k in _TIME_FORMS]
    if len(forms) != 1:
        if len(forms) > 1:
            errors.append((path, f"give exactly one of ramp, hold, table, got {forms}"))
        elif not any(k not in _TIME_FORMS for k in raw):
            errors.append((path, "give exactly one of ramp, hold, table"))
        return None
    f = forms[0]
    if f == "hold":
        if raw["hold"] is not True:
            errors.append((join(path, "hold"), "hold must be true (a constant load); a changing load is a ramp or "
                                               f"a table, got {raw['hold']!r}"))
            return None
        return {"hold": True}
    if f == "ramp":
        t = _num(raw["ramp"], join(path, "ramp"), errors, "a time", positive=True)
        if t is None or grid is None:
            return None
        i = _snap(t * tf, grid)
        if i is None:
            errors.append((join(path, "ramp"), _off_grid(t * tf, grid)))
            return None
        return {"ramp": float(grid[i])}
    tb = raw["table"]
    tp = join(path, "table")
    if not isinstance(tb, list) or len(tb) < 2:
        errors.append((tp, f"expected a list of at least 2 [t, s] knots, got {tb!r}"))
        return None
    knots: list[list[float]] = []
    ok = True
    for j, kn in enumerate(tb):
        kp = join(tp, j)
        if not (isinstance(kn, list) and len(kn) == 2):
            errors.append((kp, f"expected [t, s], got {kn!r}"))
            ok = False
            continue
        t = _num(kn[0], join(kp, 0), errors, "a time", nonneg=True)
        s = _num(kn[1], join(kp, 1), errors, "a number")
        if t is None or s is None:
            ok = False
            continue
        t *= tf
        if j == 0 and t != 0.0:
            errors.append((kp, f"the first knot must be at t = 0 (the load is s(0) there), got t = {t:g} s"))
            ok = False
        elif j > 0 and not t > knots[-1][0]:
            errors.append((kp, "knot times must be strictly increasing"))
            ok = False
        elif j > 0 and grid is not None:
            i = _snap(t, grid)
            if i is None:
                errors.append((kp, _off_grid(t, grid)))
                ok = False
            else:
                t = float(grid[i])
        knots.append([t, s])
    return {"table": knots} if ok else None


def _resolve_loads(raw: Any, grid: np.ndarray | None, f: dict[str, tuple[str, float]],
                   errors: _Errors) -> list[dict[str, Any]]:
    out = []
    lf, ff, tf = f["length"][1], f["force"][1], f["time"][1]
    for i, ld in enumerate(_list(raw, "loads", errors, 1)):
        p = join("loads", i)
        if not isinstance(ld, dict):
            errors.append((p, "expected a mapping {set | point, force, time}"))
            continue
        for k in ld:
            if k not in _LOAD_KEYS:
                errors.append((join(p, k), f"unknown key of a history load{suggest(str(k), list(_LOAD_KEYS)) or ''} "
                                           f"(keys: {', '.join(_LOAD_KEYS)})"))
        r: dict[str, Any] = {}
        if ("set" in ld) == ("point" in ld):
            errors.append((p, "give exactly one of set (a named node set) and point (coordinates)"))
        elif "set" in ld:
            if ld["set"] not in _SETS[:4]:
                errors.append((join(p, "set"), f"a named node set of the mesh ({', '.join(_SETS[:4])}), got "
                                               f"{ld['set']!r}"))
            how = normalise_name(str(ld.get("distribute", "nodes")))
            if how == "traction":
                errors.append((join(p, "distribute"), f"traction {_NOT_YET}; use nodes (the total force split "
                                                      "equally over the set's nodes)"))
            elif how != "nodes":
                errors.append((join(p, "distribute"), f"nodes is the only distribution here, got "
                                                      f"{ld.get('distribute')!r}"))
            r.update(set=ld["set"], distribute="nodes")
        else:
            if "distribute" in ld:
                errors.append((join(p, "distribute"), "distribute applies to a load on a set; a point load acts on "
                                                      "the nearest node"))
            pt = _number_list(ld["point"], join(p, "point"), 2, errors)
            r["point"] = [v * lf for v in pt]
        if "force" not in ld:
            errors.append((join(p, "force"), "required key of a history load is missing"))
        else:
            fv = _number_list(ld["force"], join(p, "force"), 2, errors)
            r["force"] = [v * ff for v in fv]
            if fv and not any(fv):
                errors.append((join(p, "force"), "zero force"))
        tfn = _resolve_time_function(ld.get("time"), join(p, "time"), grid, tf, errors)
        if tfn is not None:
            r["time"] = tfn
        out.append(r)
    return out


def _resolve_objective(raw: Any, grid: np.ndarray | None, tf: float, errors: _Errors) -> dict[str, Any]:
    o = _mapping(raw, "objective", ("compliance_at", "max_compliance_at"), errors, "objective")
    if raw is None:
        errors.append(("objective", "required key is missing: {compliance_at: t} | {max_compliance_at: [t, ...]}"))
        return {}
    if len(o) != 1:
        if o:
            errors.append(("objective", "give exactly one of compliance_at and max_compliance_at"))
        elif isinstance(raw, dict) and not raw:
            errors.append(("objective", "give one of compliance_at, max_compliance_at"))
        return {}
    if "compliance_at" in o:
        t = _num(o["compliance_at"], "objective.compliance_at", errors, "a time", nonneg=True)
        if t is None or grid is None:
            return {}
        i = _snap(t * tf, grid)
        if i is None:
            errors.append(("objective.compliance_at", _off_grid(t * tf, grid)))
            return {}
        return {"compliance_at": float(grid[i])}
    ts = o["max_compliance_at"]
    if not isinstance(ts, list) or len(ts) < 2:
        errors.append(("objective.max_compliance_at", f"expected a list of at least 2 times (one time: use "
                                                      f"compliance_at), got {ts!r}"))
        return {}
    idx: list[int] = []
    for j, t in enumerate(ts):
        v = _num(t, join("objective.max_compliance_at", j), errors, "a time", nonneg=True)
        if v is None or grid is None:
            continue
        i = _snap(v * tf, grid)
        if i is None:
            errors.append((join("objective.max_compliance_at", j), _off_grid(v * tf, grid)))
        elif i in idx:
            errors.append((join("objective.max_compliance_at", j), f"{v:g} is listed twice"))
        else:
            idx.append(i)
    if grid is None or len(idx) != len(ts):
        return {}
    return {"max_compliance_at": [float(grid[i]) for i in idx]}


def _resolve_constraints(raw: Any, errors: _Errors) -> list[dict[str, Any]]:
    cs = _list(raw, "constraints", errors)
    if len(cs) != 1:
        errors.append(("constraints", f"exactly one volume constraint is required on the history path, got {len(cs)}"))
        return []
    c, p = cs[0], "constraints.0"
    if not isinstance(c, dict):
        errors.append((p, "expected a mapping {response: volume, upper: fraction}"))
        return []
    for k in c:
        if k not in ("response", "upper"):
            errors.append((join(p, k), f"{k} {_NOT_YET}; keys: response, upper"))
    resp = c.get("response")
    t = normalise_name(str(resp.get("type", ""))) if isinstance(resp, dict) else normalise_name(str(resp))
    if t != "volume":
        errors.append((join(p, "response"), f"volume is the only constraint of the history path, got {resp!r}"))
    elif isinstance(resp, dict) and set(resp) - {"type"}:
        errors.append((join(p, "response"), f"volume parameters {sorted(set(resp) - {'type'})} {_NOT_YET}"))
    if "upper" not in c:
        errors.append((join(p, "upper"), "volume has no default bound; set upper"))
        return []
    up = _num(c["upper"], join(p, "upper"), errors, "a volume fraction")
    if up is not None and not 0.0 < up <= 1.0:
        errors.append((join(p, "upper"), f"volume is a fraction of the part: upper must be in (0, 1], got {up:g}"))
        return []
    return [] if up is None else [{"response": "volume", "upper": up}]


def _resolve_optimizer(raw: Any, errors: _Errors) -> dict[str, Any]:
    spec = {"type": raw} if isinstance(raw, str) else raw
    o = _mapping(spec, "optimizer", _OPT_KEYS, errors, "the optimizer")
    if normalise_name(str(o.get("type", "mma"))) != "mma":
        errors.append(("optimizer.type", f"mma is the only optimizer of the history path, got {o.get('type')!r}"))
    out: dict[str, Any] = {"type": "mma"}
    mi = _int(o.get("max_iter", 40), "optimizer.max_iter", errors, 1)
    out["max_iter"] = 40 if mi is None else mi
    tol = _num(o.get("tol", 0.0), "optimizer.tol", errors, "a change tolerance", nonneg=True)
    out["tol"] = 0.0 if tol is None else tol
    kt = o.get("kkt_tol")
    if kt is not None:
        kt = _num(kt, "optimizer.kkt_tol", errors, "a KKT residual bound", positive=True)
    out["kkt_tol"] = kt
    mv = _num(o.get("move", 0.2), "optimizer.move", errors, "a move limit", positive=True)
    if mv is not None and mv > 1.0:
        errors.append(("optimizer.move", f"must be in (0, 1], got {mv:g}"))
    out["move"] = 0.2 if mv is None else mv
    return out


def _resolve_interpolation(raw: Any, errors: _Errors) -> dict[str, Any]:
    spec = {"type": raw} if isinstance(raw, str) else raw
    o = _mapping(spec, "interpolation", _INTERP_KEYS, errors, "the interpolation")
    if normalise_name(str(o.get("type", "simp"))) != "simp":
        errors.append(("interpolation.type", f"simp is the only interpolation of the history path, got "
                                             f"{o.get('type')!r}"))
    pen = _num(o.get("penal", 3.0), "interpolation.penal", errors, "a penalty")
    if pen is not None and pen < 1.0:
        errors.append(("interpolation.penal", f"must be >= 1, got {pen:g}"))
    em = _num(o.get("emin_ratio", 1e-9), "interpolation.emin_ratio", errors, "a stiffness ratio")
    if em is not None and not 0.0 < em < 1.0:
        errors.append(("interpolation.emin_ratio", f"must be in (0, 1), got {em:g}"))
    return {"type": "simp", "penal": 3.0 if pen is None else pen, "emin_ratio": 1e-9 if em is None else em}


def _resolve_chain(raw: Any, lf: float, errors: _Errors) -> list[dict[str, Any]]:
    if raw is None:
        errors.append(("chain", "required on the history path: [{type: density_filter, rmin: <length>}]"))
        return []
    links = _list(raw, "chain", errors, 1)
    if len(links) > 1:
        errors.append(("chain", f"exactly one density_filter link is supported (embed / passive links "
                                f"are not supported by the history path), got {len(links)}"))
        return []
    out: list[dict[str, Any]] = []
    for i, lk in enumerate(links):
        p = join("chain", i)
        spec = {"type": lk} if isinstance(lk, str) else lk
        if not isinstance(spec, dict):
            errors.append((p, "expected {type: density_filter, rmin: <length>}"))
            continue
        t = normalise_name(str(spec.get("type", "")))
        if t != "density_filter":
            errors.append((join(p, "type"), f"chain link {spec.get('type')!r} {_NOT_YET} (history path link: "
                                            "density_filter)"))
            continue
        for k in spec:
            if k not in ("type", "rmin"):
                errors.append((join(p, k), "unknown parameter of the density_filter (parameters: rmin)"))
        if "rmin" not in spec:
            errors.append((join(p, "rmin"), "required: the filter radius, a length"))
            continue
        rmin = _num(spec["rmin"], join(p, "rmin"), errors, "a length", positive=True)
        if rmin is not None:
            out.append({"type": "density_filter", "rmin": rmin * lf})
    return out


def _resolve_post(raw: Any, errors: _Errors) -> list[dict[str, Any]]:
    out = []
    for i, s in enumerate(_list(raw, "post", errors)):
        p = join("post", i)
        spec = {"type": s} if isinstance(s, str) else s
        if not isinstance(spec, dict):
            errors.append((p, "expected {type: vtu | summary, name: <file stem>}"))
            continue
        t = normalise_name(str(spec.get("type", "")))
        if t not in _POST_TYPES:
            errors.append((join(p, "type"), f"post step {spec.get('type')!r} {_NOT_YET} (history path post steps: "
                                            f"{', '.join(_POST_TYPES)})"))
            continue
        for k in sorted(set(spec) - {"type", "name"}):
            errors.append((join(p, k), f"unknown parameter of the {t} step (parameters: name)"))
        name = spec.get("name", "design" if t == "vtu" else "history_summary")
        if not isinstance(name, str) or not name or any(c in name for c in "/\\:") or name in ("summary", "result"):
            errors.append((join(p, "name"), f"a file stem without a folder, not 'summary' or 'result' (the run "
                                            f"writes those), got {name!r}"))
        out.append({"type": t, "name": name})
    return out


# ============================================================================================== resolve
def resolve_history(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a history config and fill every default (values converted to SI); every problem found is listed in one
    :class:`ConfigError` with the key path of each. The result is itself a valid input (resolving it again gives it
    back; units are then SI)."""
    errors: _Errors = []
    for k in raw:
        if k in HISTORY_KEYS:
            continue
        if k == "load_cases":
            errors.append((k, "the history path takes loads: [{set | point, force, time}] with a time function, not "
                              "load_cases"))
        elif k == "run":
            errors.append((k, "the history path takes its run options in optimizer: {max_iter, tol, kkt_tol, move}"))
        elif k in _GRID_ONLY:
            errors.append((k, f"{k} {_NOT_YET}"))
        else:
            errors.append((str(k), f"unknown top-level key{suggest(str(k), list(HISTORY_KEYS)) or ''} "
                                   f"(history keys: {', '.join(HISTORY_KEYS)})"))
    r: dict[str, Any] = {"name": str(raw.get("name") or "study")}
    r["analysis"] = _resolve_analysis(raw.get("analysis"), errors)
    f = _resolve_units(raw.get("units"), errors)
    r["units"] = dict(_SI)
    r["domain"] = _resolve_domain(raw.get("domain"), f["length"][1], errors)
    r["material"], card = _resolve_material(raw.get("material"), errors, thermal="temperature" in raw)
    if "supports" not in raw:
        errors.append(("supports", "required key is missing"))
    r["supports"] = _resolve_supports(raw.get("supports"), 2, errors) if "supports" in raw else []
    grid = _resolve_times(raw.get("times"), f["time"][1], errors)
    r["times"] = [] if grid is None else [float(t) for t in grid]
    if "temperature" in raw:
        tsch = _resolve_temperature(raw.get("temperature"), grid, f["time"][1], errors)
        if tsch is not None:
            r["temperature"] = tsch
            if card is not None and "temperature" in card.validity.ranges and grid is not None:
                tab = np.array(tsch["table"])
                try:
                    card.check(temperature=np.concatenate([tab[:, 1], np.interp(grid, tab[:, 0], tab[:, 1])]))
                except ValueError as e:
                    errors.append(("temperature", f"outside the validity of the material card: {e}"))
    if "loads" not in raw:
        errors.append(("loads", "required key is missing"))
    r["loads"] = _resolve_loads(raw.get("loads"), grid, f, errors)
    if card is not None and grid is not None and "time" in card.validity.ranges:
        pos = grid[grid > 0.0]
        if pos.size:
            try:
                card.check(time=pos)
            except ValueError as e:
                errors.append(("times", f"outside the validity of the material card: {e}"))
    r["objective"] = _resolve_objective(raw.get("objective"), grid, f["time"][1], errors)
    r["constraints"] = _resolve_constraints(raw.get("constraints"), errors)
    r["optimizer"] = _resolve_optimizer(raw.get("optimizer"), errors)
    r["interpolation"] = _resolve_interpolation(raw.get("interpolation"), errors)
    r["chain"] = _resolve_chain(raw.get("chain"), f["length"][1], errors)
    r["post"] = _resolve_post(raw.get("post"), errors)
    out = _mapping(raw.get("output"), "output", ("folder",), errors, "output")
    folder = out.get("folder", "runs")
    if not isinstance(folder, str) or not folder:
        errors.append(("output.folder", f"expected a folder name, got {folder!r}"))
        folder = "runs"
    r["output"] = {"folder": folder}
    if "provenance" in raw:
        r["provenance"] = raw["provenance"]
    if errors:
        raise ConfigError([(p, m if "history" in m else f"{m} (analysis type history)") for p, m in errors])
    return r


# ============================================================================================== build
@dataclass
class HistorySetup:
    """A built history study: ``structure`` (a :class:`~toporlogy.optimization.viscoelastic_topology.
    ViscoelasticStructure` or :class:`~toporlogy.optimization.creep_topology.CreepStructure`), ``steps`` (the objective
    step indices on ``times``), ``options`` (the keyword arguments of
    :func:`~toporlogy.optimization.viscoelastic_topology.optimize_viscoelastic`), what the CLI prints, and the
    post steps."""

    structure: Any
    config: dict[str, Any]                      # resolved
    kind: str                                   # "viscoelastic" | "creep"
    card: Any                                   # MaterialCard
    mesh: Any
    fixed: np.ndarray
    times: np.ndarray
    steps: list[int]
    options: dict[str, Any]
    loads: list[dict[str, Any]]                 # label, resultant, nodes, time function (for the CLI)
    post: list[tuple[str, Callable[..., Any]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def objective_name(self) -> str:
        return "compliance" if len(self.steps) == 1 else "max_compliance"

    def optimize(self) -> Any:
        """:func:`~toporlogy.optimization.viscoelastic_topology.optimize_viscoelastic` with this study's options."""
        from toporlogy.optimization.viscoelastic_topology import optimize_viscoelastic
        o = self.options
        return optimize_viscoelastic(self.structure, volume=o["volume"], rmin=o["rmin"], step_index=o["step_index"],
                                     penal=o["penal"], emin=o["emin"], max_iter=o["max_iter"], move=o["move"],
                                     steps=o["steps"], tol=o["tol"], kkt_tol=o["kkt_tol"])

    def weights(self, xphys: np.ndarray) -> Any:
        """SIMP weights w = emin + x̄^p (1 − emin) as a float64 tensor (the optimiser's map)."""
        import torch
        o = self.options
        return torch.as_tensor(o["emin"] + np.asarray(xphys, float) ** o["penal"] * (1.0 - o["emin"]),
                               dtype=torch.float64)


def _scale(tf: dict[str, Any], t: np.ndarray) -> np.ndarray:
    """The time function s(t) of a resolved load at the times ``t``."""
    if "ramp" in tf:
        return np.array([min(float(v) / tf["ramp"], 1.0) for v in t])
    if "hold" in tf:
        return np.ones(len(t))
    kn = np.asarray(tf["table"], float)
    return np.interp(t, kn[:, 0], kn[:, 1])


def _time_text(tf: dict[str, Any]) -> str:
    if "ramp" in tf:
        return f"ramp {tf['ramp']:g} s"
    if "hold" in tf:
        return "hold"
    return "table " + " ".join(f"({t:g}, {s:g})" for t, s in tf["table"])


def build_history(r: dict[str, Any]) -> HistorySetup:
    """Construct a resolved history config (:func:`resolve_history`) into a :class:`HistorySetup`. Every
    :class:`ConfigError` message says it comes from the history path."""
    try:
        return _build_history(r)
    except ConfigError as e:
        tagged = [(p, m if "history" in m else f"{m} (analysis type history)") for p, m in e.errors]
        raise ConfigError(tagged) from e.__cause__


def _build_history(r: dict[str, Any]) -> HistorySetup:
    from toporlogy.optimization.creep_topology import CreepStructure
    from toporlogy.optimization.material_cards import load_material_card
    from toporlogy.optimization.mesh_topology import MeshDensityFilter
    from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure

    dom = r["domain"]
    try:
        mesh = _mesh({**dom, "generator": "rectangle"})
    except ValueError as e:                       # an inverted or degenerate cell after perturb
        raise ConfigError(f"domain: {e}", "domain.perturb" if dom["perturb"] > 0 else "domain") from None
    notes = [f"mesh: rectangle {' x '.join(map(str, dom['n']))} {dom['element']}, {mesh.n_cells} cells, "
             f"{mesh.n_nodes} nodes" + (f", perturbed by up to {dom['perturb']:g} (seed {dom['seed']})"
                                        if dom["perturb"] > 0 else "")]
    parts = []
    for i, s in enumerate(r["supports"]):
        dofs = mesh.dofs(mesh.node_sets[s["set"]], [_AXES.index(c) for c in s["fix"]])
        if not dofs.size:
            raise ConfigError(f'node set "{s["set"]}" is empty', join(join("supports", i), "set"))
        parts.append(dofs)
    fixed = np.unique(np.concatenate(parts))
    _check_rigid(mesh, fixed, "supports")

    times = np.asarray(r["times"], float)
    ndof = 2 * mesh.n_nodes
    loads = np.zeros((len(times), ndof))
    info: list[dict[str, Any]] = []
    for j, ld in enumerate(r["loads"]):
        lp = join("loads", j)
        f = np.asarray(ld["force"], float)
        F = np.zeros(ndof)
        if "set" in ld:
            nodes = mesh.node_sets[ld["set"]]
            if not nodes.size:
                raise ConfigError(f'node set "{ld["set"]}" is empty', join(lp, "set"))
            F.reshape(-1, 2)[nodes] += f / len(nodes)
            label, n_nodes = f"set {ld['set']}", int(len(nodes))
        else:
            used = mesh.used_nodes()
            dist = np.linalg.norm(mesh.points[used] - np.asarray(ld["point"], float), axis=1)
            k = int(used[np.argmin(dist)])
            F.reshape(-1, 2)[k] += f
            label, n_nodes = f"point at node {k}", 1
            notes.append(f"load {j + 1} acts at node {k} ({', '.join(f'{v:g}' for v in mesh.points[k])}), "
                         f"{float(dist.min()):.3g} from the given point")
        s = _scale(ld["time"], times)
        loads += s[:, None] * F[None, :]
        info.append({"label": label, "resultant": [float(v) for v in f], "nodes": n_nodes,
                     "time": _time_text(ld["time"]), "peak": float(np.abs(s).max())})
    if not np.isfinite(loads).all():
        raise ConfigError("the load history is not finite (NaN or Inf)", "loads")
    if not np.any(loads):
        raise ConfigError("zero load: every load vanishes at every step time", "loads")
    warnings = []
    hit = np.intersect1d(np.flatnonzero(np.any(loads != 0, axis=0)), fixed)
    if hit.size:
        warnings.append(f"{hit.size} loaded DOF(s) are also fixed; their load does no work")

    try:
        mc = load_material_card(r["material"])
    except ValueError as e:
        raise ConfigError(str(e), "material") from None
    thick = float(dom["thickness"])
    try:
        tkw: dict[str, Any] = {}
        if "temperature" in r:
            tab = np.asarray(r["temperature"]["table"], float)
            tkw = {"temperatures": np.interp(times, tab[:, 0], tab[:, 1]).tolist(), "temperature0": float(tab[0, 1])}
        if mc.kind == "generalized_maxwell":
            kind = "viscoelastic"
            structure: Any = ViscoelasticStructure(mesh, mc.model, times.tolist(), loads, fixed, thickness=thick,
                                                   **tkw)
        else:
            kind = "creep"
            m = mc.model
            if tkw:
                tkw["arrhenius"] = m.arrhenius
            structure = CreepStructure(mesh, m.K, m.G, m.A, m.sigma_ref, m.n, times.tolist(), loads, fixed,
                                       thickness=thick, **tkw)
    except ValueError as e:
        raise ConfigError(str(e), "times" if "times" in str(e) else "material") from None
    grid_idx = {float(t): i for i, t in enumerate(times)}
    obj = r["objective"]
    steps = [grid_idx[obj["compliance_at"]]] if "compliance_at" in obj else \
        [grid_idx[t] for t in obj["max_compliance_at"]]
    opt, ip, ch = r["optimizer"], r["interpolation"], r["chain"][0]
    try:
        MeshDensityFilter(mesh, ch["rmin"])
    except ValueError as e:
        raise ConfigError(f"link density_filter: {e}", "chain.0.rmin") from None
    options = {"volume": float(r["constraints"][0]["upper"]), "rmin": float(ch["rmin"]), "step_index": steps[0],
               "steps": steps if len(steps) > 1 else None, "penal": float(ip["penal"]),
               "emin": float(ip["emin_ratio"]), "max_iter": int(opt["max_iter"]), "move": float(opt["move"]),
               "tol": float(opt["tol"]), "kkt_tol": opt["kkt_tol"]}
    post = [(p["type"], _vtu_step(p["name"]) if p["type"] == "vtu" else _summary_step(p["name"])) for p in r["post"]]
    return HistorySetup(structure, r, kind, mc, mesh, fixed, times, steps, options, info, post, warnings, notes)


# ============================================================================================== post steps
# a step is called as step(result, setup, folder, final) with the OptimizeResult and ``final``, the
# ViscoelasticResult / CreepResult of the returned design (run_history_setup computes it once)
def _vtu_step(name: str) -> Callable[..., dict[str, Any]]:
    """``<name>.vtu``: the mesh cells (VTK quad / triangle) with cell data ``xphys`` (filtered density of the returned
    design), ``x`` (design variable), ``weight`` (SIMP stiffness factor) and point data ``displacement_step<k>`` (n, 3;
    z = 0) at every objective step k."""

    def step(res: Any, setup: HistorySetup, folder: Any, final: Any) -> dict[str, Any]:
        from toporlogy.core.vtu import VTK_QUAD, VTK_TRIANGLE, write_cells_vtu
        mesh = setup.mesh
        xp = np.asarray(res.xphys, float).reshape(-1)
        cell = {"xphys": xp, "x": np.asarray(res.x, float).reshape(-1),
                "weight": np.asarray(setup.weights(xp).numpy(), float)}
        point: dict[str, Any] = {}
        u = final.u.detach().numpy()
        for k in setup.steps:
            d = np.zeros((mesh.n_nodes, 3))
            d[:, :2] = u[k].reshape(-1, 2)
            point[f"displacement_step{k}"] = d
        kind = VTK_QUAD if mesh.element == "Q4" else VTK_TRIANGLE
        path = write_cells_vtu(f"{folder}/{name}.vtu", mesh.points, mesh.cells, kind, point, cell,
                               meta={"design": "optimized", "element": mesh.element, "analysis": "history",
                                     "material": setup.kind, "objective_steps": list(setup.steps),
                                     "objective_times": [float(setup.times[k]) for k in setup.steps]})
        return {"file": f"{name}.vtu", "design": "optimized", "cell_arrays": sorted(cell),
                "point_arrays": sorted(point), "path": path,
                "summary": f"{name}.vtu: {len(cell)} cell and {len(point)} point arrays"}
    return step


def _summary_step(name: str) -> Callable[..., dict[str, Any]]:
    """``<name>.json``: the compliance at every step time for the start design (uniform at the volume bound) and for
    the returned design, the objective steps, the volume and the density range."""

    def step(res: Any, setup: HistorySetup, folder: Any, final: Any) -> dict[str, Any]:
        import json

        import torch
        o = setup.options
        with torch.no_grad():
            start = setup.structure.response(setup.weights(np.full(setup.structure.ne, o["volume"]))).compliance
        c0, c1 = start.numpy(), final.compliance.detach().numpy()
        data = {"model": setup.kind, "card": setup.card.name, "synthetic": setup.card.synthetic,
                "times_s": [float(t) for t in setup.times], "objective": setup.objective_name,
                "objective_steps": list(setup.steps), "objective_times_s": [float(setup.times[k]) for k in setup.steps],
                "compliance_start_design": [float(v) for v in c0], "compliance_returned_design": [float(v) for v in c1],
                "volume": float(res.volume), "volume_upper": o["volume"], "xphys_min": float(np.min(res.xphys)),
                "xphys_max": float(np.max(res.xphys)), "iterations": int(res.iterations),
                "termination": res.termination, "converged": bool(res.converged)}
        p = Path(folder) / f"{name}.json"
        p.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        return {"file": p.name, "path": str(p),
                "summary": f"{p.name}: compliance at {len(c1)} times; objective "
                           f"{float(c0[setup.steps].max()):.4e} -> {float(c1[setup.steps].max()):.4e}"}
    return step
