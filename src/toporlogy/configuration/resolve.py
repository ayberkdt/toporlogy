"""Reading, resolving and validating configs; overrides (``--set``) and sweeps (``--sweep``).
"""
from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from toporlogy.configuration.registry import BuildContext, ConfigError, Param, REGISTRY, join, normalise_name, suggest
from toporlogy.configuration.schema import (
    SCHEMA_KEYS, _CASE, _COMBINATION, _CONNECTIVITY, _CONSTRAINT, _DEFAULTS, _MATERIAL, _NEEDS_FINALIZE, _OUTPUT,
    _REQUIRED, _ROBUST, _RUN, _TYING,
)


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
    from toporlogy.optimization.robust import robust_chain_settings
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
