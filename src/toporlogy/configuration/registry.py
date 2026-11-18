"""Component registry: every building block of a declarative study is found here by kind and name.

A *component* is a constructor registered under ``(kind, name)``. A config refers to it as a name (all defaults) or
as a mapping ``{type: <name>, <param>: <value>, ...}``. Parameters, defaults and types are read from the
constructor's signature, so registering a new component is one line (see ``docs/registry.md``)::

    from toporlogy.configuration.registry import register
    register("link", "am_filter", AMFilter, context=("domain",))

Kinds (:data:`KINDS`): ``domain``, ``shape``, ``selector``, ``support``, ``load``, ``link``, ``interpolation``,
``solver``, ``physics``, ``response``, ``optimizer``, ``schedule``, ``post``.

Core components are registered by ``_register_core``; the feature modules (:mod:`toporlogy.boundary`, ``geometry``,
``robust``, ``lengthscale``, ``connectivity``, ``manufacturing``, ``dynamics``, ``fem.modal``, ``post``) by
``_register_features`` (``post`` is imported only when one of its steps is used).

Constructor conventions
-----------------------
* A first parameter named ``ctx`` receives the :class:`BuildContext` (domain, material, chain, fixed DOFs, ...).
* Names listed in ``context=`` are taken from the :class:`BuildContext` attribute of the same name and passed as
  keyword arguments (e.g. ``context=("domain",)`` for ``DensityFilter(domain, rmin)``).
* Every other parameter is a user parameter. Its type comes from ``types=``, else from the default value, else from
  the annotation text (float, int, bool, str, list); without a default it is required.
* ``nested={"where": "selector"}`` makes a parameter a component of another kind (``"selector[]"``: a list of them);
  it is resolved and built recursively and the constructor receives the built object.
* ``units={"h": "length"}`` documents units (shown by ``python -m toporlogy list``).

What each kind's constructor returns: domain → :class:`~toporlogy.core.domain.Domain`; shape →
:class:`~toporlogy.geometry.Shape` (with an ``op`` attribute: add / cut / intersect); selector →
:class:`~toporlogy.boundary.Selector` or node numbers (int array); support → :class:`~toporlogy.boundary.Support` or
fixed DOF numbers; load → :class:`~toporlogy.boundary.Load` or a load vector (ndof,) of one case; link →
``torch.nn.Module``; interpolation → ``torch.nn.Module`` x̄ → E; solver → object with ``factor`` / ``solve``; physics →
object with ``solve(E_active, F)``; response → :class:`~toporlogy.optimization.responses.Response`; optimizer →
:class:`~toporlogy.optimizers.base.Optimizer`; schedule → :class:`~toporlogy.optimizers.schedule.Schedule`;
post → callable ``step(result, study, run_dir) -> dict | None`` run after the optimisation.
"""
from __future__ import annotations

import difflib
import importlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


__all__ = ["KINDS", "REQUIRED", "ConfigError", "Param", "Entry", "BuildContext", "Registry", "REGISTRY", "register",
           "get", "names", "normalise_name", "suggest", "join"]

KINDS: dict[str, str] = {
    "domain": "design domain builder -> Domain",
    "shape": "geometric shape (CSG primitive; op: add | cut | intersect)",
    "selector": "node selection of the part",
    "support": "support (zero displacement, roller, pin, springs)",
    "load": "load of one load case",
    "link": "chain link (design -> physical density)",
    "interpolation": "physical density -> element modulus",
    "solver": "linear solver",
    "physics": "physics model (state solve)",
    "response": "objective or constraint response",
    "optimizer": "optimizer",
    "schedule": "continuation schedule",
    "post": "post-processing step after the run",
}


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()


class ConfigError(Exception):
    """One or more config problems, each ``(path, message)``; ``path`` is dotted (``chain.1.rmin``)."""

    def __init__(self, errors: Sequence[tuple[str, str]] | str, path: str = "") -> None:
        self.errors: list[tuple[str, str]] = [(path, errors)] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(f"{p}: {m}" if p else m for p, m in self.errors))


def normalise_name(name: str) -> str:
    """Case-, dash- and space-insensitive component name (``"Density-Filter"`` → ``"density_filter"``)."""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def suggest(word: str, options: Sequence[str]) -> str:
    """``'; did you mean "x"?'`` for the closest option, else ``""``."""
    hit = difflib.get_close_matches(str(word), list(options), n=1, cutoff=0.6)
    return f'; did you mean "{hit[0]}"?' if hit else ""


def join(path: str, key: str | int) -> str:
    return f"{path}.{key}" if path else str(key)


# ---------------------------------------------------------------------------------------------- parameters
_TYPE_NAMES = {float: "number", int: "integer", bool: "true/false", str: "text", list: "list", None: "any"}


_TRUE, _FALSE = {"true", "yes", "on", "1"}, {"false", "no", "off", "0"}


@dataclass
class Param:
    """One user parameter of a component."""

    name: str
    default: Any = REQUIRED
    type: Any = None                 # float | int | bool | str | list | None (any)
    nullable: bool = False
    nested: str | None = None        # kind of a nested component ("selector", "selector[]")
    unit: str = ""

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    def coerce(self, value: Any, path: str) -> Any:
        """Validate ``value`` and return it in canonical form; raises :class:`ConfigError`.

        Lenient on input (Postel): numbers given as text (PyYAML reads ``1e-6`` as text), ints for floats, integral
        floats for ints, yes/no/on/off for booleans, a scalar for a list of one.
        """
        if value is None:
            if self.nullable or self.type is None:
                return None
            raise ConfigError(f"expected {_TYPE_NAMES[self.type]}, got null", path)
        t = self.type
        if t is None:
            return _plain(value)
        bad = ConfigError(f"expected {_TYPE_NAMES[t]}, got {_show(value)}", path)
        if t is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
                return value.strip().lower() in _TRUE
            raise bad
        if t is float:
            if isinstance(value, bool):
                raise bad
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    raise bad from None
            raise bad
        if t is int:
            if isinstance(value, bool):
                raise bad
            if isinstance(value, int):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    raise bad from None
            raise bad
        if t is str:
            if isinstance(value, str):
                return value
            raise bad
        if t is list:
            if isinstance(value, list | tuple):
                return _plain(list(value))
            return [_plain(value)]
        return value


def _plain(v: Any) -> Any:
    """YAML-safe plain data (tuples → lists, NumPy scalars → Python)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _show(v: Any) -> str:
    s = repr(v) if not isinstance(v, str) else f'"{v}"'
    return s if len(s) <= 40 else s[:37] + "..."


def _annotation_type(ann: Any) -> tuple[Any, bool]:
    """(type, nullable) from an annotation (string under ``from __future__ import annotations``)."""
    if ann is inspect.Parameter.empty:
        return None, False
    s = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    nullable = "None" in s or "Optional" in s
    for key, t in (("bool", bool), ("float", float), ("int", int), ("str", str)):
        if key in s and not any(w in s for w in ("Sequence", "list", "tuple", "[", "Any")):
            return t, nullable
    if any(w in s for w in ("Sequence", "list", "tuple")):
        return list, nullable
    return None, nullable


def _param_from_signature(p: inspect.Parameter, types: dict[str, Any], nested: dict[str, str],
                          units: dict[str, str]) -> Param:
    default = REQUIRED if p.default is inspect.Parameter.empty else p.default
    ann_t, ann_null = _annotation_type(p.annotation)
    if p.name in types:
        t = types[p.name]
    elif default is REQUIRED or default is None:
        t = ann_t
    elif isinstance(default, bool):
        t = bool
    elif isinstance(default, int):
        t = float if ann_t is float else int
    elif isinstance(default, float):
        t = float
    elif isinstance(default, str):
        t = str
    elif isinstance(default, list | tuple):
        t = list
    else:
        t = None
    if isinstance(default, tuple):
        default = list(default)
    return Param(p.name, default, t, nullable=default is None or ann_null, nested=nested.get(p.name),
                 unit=units.get(p.name, ""))


# ---------------------------------------------------------------------------------------------- entries
@dataclass
class Entry:
    """A registered component."""

    kind: str
    name: str
    target: Callable[..., Any] | str          # callable or lazy "package.module:attr"
    context: tuple[str, ...] = ()
    doc: str = ""
    nested: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    types: dict[str, Any] = field(default_factory=dict)
    _params: dict[str, Param] | None = None
    _wants_ctx: bool = False

    def resolve_target(self) -> Callable[..., Any]:
        if isinstance(self.target, str):
            mod, _, attr = self.target.partition(":")
            self.target = getattr(importlib.import_module(mod), attr)
        return self.target

    @property
    def params(self) -> dict[str, Param]:
        if self._params is None:
            fn = self.resolve_target()
            sig = inspect.signature(fn)
            out: dict[str, Param] = {}
            plist = list(sig.parameters.values())
            if plist and plist[0].name == "ctx":
                self._wants_ctx = True
                plist = plist[1:]
            for p in plist:
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name in self.context:
                    continue
                out[p.name] = _param_from_signature(p, self.types, self.nested, self.units)
            for k in self.nested:
                if k not in out:
                    raise TypeError(f"{self.kind} {self.name}: nested parameter {k!r} not in the signature")
            self._params = out
        return self._params

    @property
    def summary(self) -> str:
        if self.doc:
            return self.doc.strip().splitlines()[0]
        fn = self.resolve_target()
        d = inspect.getdoc(fn) or ""
        return d.splitlines()[0] if d else ""

    def construct(self, ctx: BuildContext, params: dict[str, Any]) -> Any:
        fn = self.resolve_target()
        _ = self.params                                  # sets _wants_ctx
        kw = {name: getattr(ctx, name) for name in self.context}
        kw.update(params)
        return fn(ctx, **kw) if self._wants_ctx else fn(**kw)


@dataclass
class BuildContext:
    """What constructors may read while a study is built (filled step by step by
    :mod:`toporlogy.configuration.build`)."""

    config: dict[str, Any] = field(default_factory=dict)       # resolved config
    domain: Any = None
    material: dict[str, float] = field(default_factory=dict)
    fixed: Any = None                                           # fixed DOF numbers (after supports)
    solver: Any = None
    physics: Any = None
    chain: Any = None
    interp: Any = None
    n_cases: int = 1
    case_weights: list[float] = field(default_factory=list)
    run: dict[str, Any] = field(default_factory=dict)
    path: str = ""                                              # config path of the component being built
    extra: dict[str, Any] = field(default_factory=dict)
    bc: Any = None                                              # toporlogy.boundary.BoundaryConditions (after loads)
    robust: Any = None                                          # shared optimization.robust.RobustSet or None
    interfaces: Any = None                                      # name -> bool element mask (connectivity)
    links: list[Any] = field(default_factory=list)              # chain links built so far
    objective: Any = None
    constraints: list[Any] = field(default_factory=list)        # Constraint objects built so far


# ---------------------------------------------------------------------------------------------- registry
class Registry:
    """``(kind, name) → Entry``; lookups are case- and dash-insensitive and suggest the closest name on a miss."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Entry]] = {k: {} for k in KINDS}

    def register(self, kind: str, name: str, target: Callable[..., Any] | str | None = None, *,
                 context: Sequence[str] = (), doc: str = "", nested: dict[str, str] | None = None,
                 units: dict[str, str] | None = None, types: dict[str, Any] | None = None,
                 replace: bool = False) -> Any:
        """Register ``target`` (a callable or lazy ``"module:attr"``) as component ``name`` of ``kind``.

        Without ``target`` this returns a decorator. Re-registering a name raises unless ``replace=True``.
        """
        if kind not in self._entries:
            raise ValueError(f'unknown kind "{kind}"{suggest(kind, list(self._entries))}')

        def deco(fn: Callable[..., Any] | str) -> Any:
            key = normalise_name(name)
            if key in self._entries[kind] and not replace:
                raise ValueError(f"{kind} {key!r} is already registered (pass replace=True)")
            self._entries[kind][key] = Entry(kind, key, fn, tuple(context), doc, dict(nested or {}),
                                             dict(units or {}), dict(types or {}))
            return fn

        return deco if target is None else deco(target)

    def unregister(self, kind: str, name: str) -> None:
        self._entries[kind].pop(normalise_name(name), None)

    def kinds(self) -> list[str]:
        return list(self._entries)

    def names(self, kind: str) -> list[str]:
        return sorted(self._entries[kind])

    def get(self, kind: str, name: str, path: str = "") -> Entry:
        if kind not in self._entries:
            raise ConfigError(f'unknown kind "{kind}"{suggest(kind, list(self._entries))}', path)
        key = normalise_name(name)
        try:
            return self._entries[kind][key]
        except KeyError:
            opts = self.names(kind)
            hint = suggest(key, opts) or (f" (registered: {', '.join(opts)})" if opts else " (none registered)")
            raise ConfigError(f'unknown {kind} "{name}"{hint}', path) from None

    # ------------------------------------------------------------------ specs
    def resolve(self, kind: str, spec: Any, path: str, errors: list[tuple[str, str]]) -> dict[str, Any] | None:
        """Normalise a component spec to ``{"type": name, **all parameters}``; problems are appended to ``errors``."""
        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, dict):
            errors.append((path, f'expected a {kind} name or a mapping with "type", got {_show(spec)}'))
            return None
        if "type" not in spec:
            errors.append((join(path, "type"), f"missing; {kind} names: {', '.join(self.names(kind)) or 'none'}"))
            return None
        try:
            entry = self.get(kind, spec["type"], join(path, "type"))
        except ConfigError as e:
            errors.extend(e.errors)
            return None
        params = entry.params
        out: dict[str, Any] = {"type": entry.name}
        for key in spec:
            if key != "type" and key not in params:
                hint = suggest(key, list(params)) or \
                    (f" (parameters: {', '.join(params)})" if params else " (it has no parameters)")
                errors.append((join(path, key), f"unknown parameter of {kind} {entry.name}{hint}"))
        for pname, p in params.items():
            ppath = join(path, pname)
            if pname not in spec:
                if p.required:
                    errors.append((ppath, f"required parameter of {kind} {entry.name} is missing"))
                    continue
                value = p.default
                if p.nested and value is not None:
                    value = self._resolve_nested(p.nested, value, ppath, errors)
                out[pname] = _plain(value)
                continue
            value = spec[pname]
            if p.nested and value is not None:
                out[pname] = self._resolve_nested(p.nested, value, ppath, errors)
                continue
            try:
                out[pname] = p.coerce(value, ppath)
            except ConfigError as e:
                errors.extend(e.errors)
        return out

    def _resolve_nested(self, nk: str, value: Any, path: str, errors: list[tuple[str, str]]) -> Any:
        if nk.endswith("[]"):
            if not isinstance(value, list):
                errors.append((path, f"expected a list of {nk[:-2]} components"))
                return None
            return [self.resolve(nk[:-2], v, join(path, i), errors) for i, v in enumerate(value)]
        return self.resolve(nk, value, path, errors)

    def build(self, kind: str, resolved: dict[str, Any], ctx: BuildContext, path: str) -> Any:
        """Construct a resolved spec; constructor errors are reported as :class:`ConfigError` at ``path``."""
        entry = self.get(kind, resolved["type"], join(path, "type"))
        params: dict[str, Any] = {}
        for pname, p in entry.params.items():
            v = resolved.get(pname, p.default)
            if p.nested and v is not None:
                if p.nested.endswith("[]"):
                    v = [self.build(p.nested[:-2], s, ctx, join(join(path, pname), i)) for i, s in enumerate(v)]
                else:
                    v = self.build(p.nested, v, ctx, join(path, pname))
            params[pname] = v
        old, ctx.path = ctx.path, path
        try:
            return entry.construct(ctx, params)
        except ConfigError:
            raise
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            raise ConfigError(f"{kind} {entry.name}: {e}", path) from e
        finally:
            ctx.path = old


REGISTRY = Registry()


register = REGISTRY.register


get = REGISTRY.get


names = REGISTRY.names
