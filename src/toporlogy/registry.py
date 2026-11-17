"""Component registry: every building block of a declarative study is found here by kind and name.

A *component* is a constructor registered under ``(kind, name)``. A config refers to it as a name (all defaults) or
as a mapping ``{type: <name>, <param>: <value>, ...}``. Parameters, defaults and types are read from the
constructor's signature, so registering a new component is one line (see ``docs/registry.md``)::

    from toporlogy.registry import register
    register("link", "am_filter", AMFilter, context=("domain",))

Kinds (:data:`KINDS`): ``domain``, ``shape``, ``selector``, ``support``, ``load``, ``link``, ``interpolation``,
``solver``, ``physics``, ``response``, ``optimizer``, ``schedule``, ``post``.

Core components are registered by ``_register_core``; the feature modules (:mod:`toporlogy.bc`, ``geometry``,
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

What each kind's constructor returns: domain → :class:`~toporlogy.domain.Domain`; shape →
:class:`~toporlogy.geometry.Shape` (with an ``op`` attribute: add / cut / intersect); selector →
:class:`~toporlogy.bc.Selector` or node numbers (int array); support → :class:`~toporlogy.bc.Support` or fixed
DOF numbers; load → :class:`~toporlogy.bc.Load` or a load vector (ndof,) of one case; link → ``torch.nn.Module``;
interpolation → ``torch.nn.Module`` x̄ → E; solver → object with ``factor`` / ``solve``; physics → object with
``solve(E_active, F)``; response → :class:`~toporlogy.responses.Response`; optimizer →
:class:`~toporlogy.optim.base.Optimizer`; schedule → :class:`~toporlogy.optim.schedule.Schedule`;
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
           "get", "names", "normalise_name", "suggest", "as_selector", "checked_selector", "find_bore", "SpreadForce",
           "bc_callables", "bc_mapping", "selector_callable"]

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
    """What constructors may read while a study is built (filled step by step by :mod:`toporlogy.config`)."""

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
    bc: Any = None                                              # toporlogy.bc.BoundaryConditions (after loads)
    robust: Any = None                                          # shared toporlogy.robust.RobustSet (or None)
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


# ============================================================================================== shared helpers
def _axis(a: Any, dim: int) -> int:
    """Axis index from "x" / "y" / "z" (also "+x", "-z") or 0 / 1 / 2."""
    if isinstance(a, str) and a.strip().lower().lstrip("+-") in ("x", "y", "z"):
        k = "xyz".index(a.strip().lower().lstrip("+-"))
    elif isinstance(a, int | float) and not isinstance(a, bool) and float(a).is_integer():
        k = int(a)
    else:
        raise ValueError(f"axis must be x, y, z (or 0, 1, 2), got {a!r}")
    if not 0 <= k < dim:
        raise ValueError(f"axis {a!r} does not exist in {dim}-D")
    return k


def _bounds(d: Any) -> tuple[np.ndarray, np.ndarray]:
    lo = d.origin
    return lo, lo + d.h * np.asarray(d.shape, float)


def _coord(v: Any, k: int, d: Any, default: float | None = None) -> float:
    """Coordinate k from a number, "min", "max" or "mid" (of the domain box); null → ``default``."""
    lo, hi = _bounds(d)
    if v is None:
        if default is None:
            raise ValueError("a coordinate is null")
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "min":
            return float(lo[k])
        if s == "max":
            return float(hi[k])
        if s in ("mid", "centre", "center"):
            return float(0.5 * (lo[k] + hi[k]))
        try:
            return float(s)
        except ValueError:
            raise ValueError(f'coordinate must be a number, "min", "mid" or "max", got {v!r}') from None
    return float(v)


def _point(v: Any, d: Any, what: str = "point") -> np.ndarray:
    """A point of the domain's dimension; entries may be numbers, "min", "mid" or "max"."""
    if not isinstance(v, list | tuple) or len(v) != d.dim:
        raise ValueError(f"{what} needs {d.dim} coordinates, got {v!r}")
    return np.array([_coord(c, k, d) for k, c in enumerate(v)])


def _vector(v: Any, dim: int, what: str = "direction") -> np.ndarray:
    """Unit vector from an axis spec ("x", "-z", 1) or components."""
    if isinstance(v, str) or (isinstance(v, int | float) and not isinstance(v, bool)):
        sign = -1.0 if isinstance(v, str) and v.strip().startswith("-") else 1.0
        e = np.zeros(dim)
        e[_axis(v, dim)] = sign
        return e
    a = np.asarray(v, float).reshape(-1)
    if a.size != dim or not np.linalg.norm(a) > 0:
        raise ValueError(f"{what} must be an axis or a non-zero vector with {dim} components, got {v!r}")
    return a / np.linalg.norm(a)


def _nonempty(nodes: np.ndarray, what: str, d: Any) -> np.ndarray:
    if nodes.size == 0:
        lo, hi = _bounds(d)
        raise ValueError(f"{what} selects no node of the part (part box {np.round(lo, 6).tolist()} to "
                         f"{np.round(hi, 6).tolist()}, node spacing h = {d.h:g})")
    return nodes


def _directions(dofs: Any, dim: int) -> list[Any]:
    """Constrained directions: "all", "x", "xy", [x, z], [0, 2] (axes) or [[1, 1, 0]] (vectors)."""
    items: list[Any]
    if isinstance(dofs, str):
        s = dofs.strip().lower()
        if s == "all":
            return list(range(dim))
        items = list(s.replace(",", "").replace(" ", ""))
    elif isinstance(dofs, list | tuple):
        items = list(dofs)
    else:
        items = [dofs]
    axes = sorted({_axis(v, dim) for v in items if not isinstance(v, list | tuple)})
    vecs = [_vector(v, dim) for v in items if isinstance(v, list | tuple)]
    if not axes and not vecs:
        raise ValueError("dofs is empty")
    return [*axes, *[v.tolist() for v in vecs]]


def as_selector(v: Any) -> Any:
    """A built selector (bc.Selector, or node numbers from a node-array selector) as a bc.Selector."""
    from toporlogy.bc import NodeIds, Selector
    if isinstance(v, Selector):
        return v
    a = np.asarray(v)
    if a.ndim == 1 and a.dtype.kind in "iu":
        return NodeIds(a.astype(np.int64))
    raise ValueError(f"expected a selector, got {type(v).__name__}")


def checked_selector(ctx: BuildContext, v: Any, key: str = "where") -> Any:
    """:func:`as_selector`, and the selection must not be empty (reported at ``<component path>.<key>``)."""
    from toporlogy.bc import EmptySelectionError
    s = as_selector(v)
    try:
        s.select(ctx.domain)
    except EmptySelectionError as e:
        raise ConfigError(f"selector: {e}", join(ctx.path, key)) from None
    return s


# ============================================================================================== core components
# Imported here (not at the top) so that toporlogy's import-order guard (PARDISO before torch) runs first.
def _register_core() -> None:
    from toporlogy.chain import DensityFilter, Embed, HeavisideProjection, Passive
    from toporlogy.domain import Domain
    from toporlogy.interp import RAMP, SIMP
    from toporlogy.optim.oc import OC
    from toporlogy.optim.schedule import BetaContinuation
    from toporlogy.responses import Compliance, Volume
    from toporlogy.solvers.direct import DirectSolver

    # ---------------------------------------------------------------- domain
    def box(ctx: BuildContext, shape: list, h: float = 1.0, origin: list | None = None, thickness: float = 1.0,
            keep: Any = None, void: Any = None) -> Domain:
        """Full rectangular grid: shape (nx, ny) or (nx, ny, nz) elements of edge h; keep / void: shapes
        (elements more than half covered are frozen solid / empty)."""
        if len(shape) not in (2, 3):
            raise ValueError(f"shape needs 2 or 3 entries, got {len(shape)}")
        try:
            shp = [int(s) for s in shape]
        except (TypeError, ValueError):
            raise ValueError("shape entries must be integers") from None
        d = Domain(shp, h=float(h), origin=None if origin is None else [float(o) for o in origin],
                   thickness=float(thickness))
        if keep is None and void is None:
            return d
        km, vm = (None if s is None else _region_mask(s, d) for s in (keep, void))
        return Domain(shp, h=d.h, origin=d.origin, keep=km, void=vm, thickness=d.thickness)

    register("domain", "box", box, nested={"keep": "shape[]", "void": "shape[]"},
             units={"h": "length", "origin": "length", "thickness": "length"})

    # ---------------------------------------------------------------- selectors (node sets of the part)
    def plane(ctx: BuildContext, axis: Any, at: Any) -> np.ndarray:
        """Nodes on the plane (3-D) or line (2-D) coordinate[axis] = at; at: number, "min" or "max"."""
        d = ctx.domain
        k = _axis(axis, d.dim)
        c = _coord(at, k, d)
        tol = 1e-6 * d.h
        return _nonempty(d.nodes_where(lambda X: np.abs(X[:, k] - c) <= tol), f"plane {'xyz'[k]} = {c:g}", d)

    def box_sel(ctx: BuildContext, min: list | None = None, max: list | None = None) -> np.ndarray:  # noqa: A002
        """Nodes inside the closed box [min, max]; entries may be numbers, "min", "max" or null (unbounded)."""
        d = ctx.domain
        lo = [-np.inf] * d.dim if min is None else min
        hi = [np.inf] * d.dim if max is None else max
        if len(lo) != d.dim or len(hi) != d.dim:
            raise ValueError(f"min and max need {d.dim} entries")
        a = np.array([_coord(v, k, d, -np.inf) for k, v in enumerate(lo)])
        b = np.array([_coord(v, k, d, np.inf) for k, v in enumerate(hi)])
        tol = 1e-6 * d.h
        return _nonempty(d.nodes_where(lambda x: np.all((x >= a - tol) & (x <= b + tol), axis=1)),
                         f"box {a.tolist()} to {b.tolist()}", d)

    def point(ctx: BuildContext, at: list) -> np.ndarray:
        """The part node nearest to ``at`` (entries: numbers, "min" or "max"); must lie within one element edge."""
        d = ctx.domain
        if len(at) != d.dim:
            raise ValueError(f"at needs {d.dim} entries")
        p = np.array([_coord(v, k, d) for k, v in enumerate(at)])
        cand = d.nodes_where(lambda X: np.ones(len(X), bool))
        dist = np.linalg.norm(d.node_coords()[cand] - p, axis=1)
        i = int(np.argmin(dist))
        if dist[i] > d.h * (1 + 1e-9):
            raise ValueError(f"no part node within h = {d.h:g} of {p.tolist()} (nearest at {dist[i]:g})")
        return cand[i:i + 1]

    register("selector", "plane", plane, units={"at": "length"})
    register("selector", "box", box_sel, units={"min": "length", "max": "length"})
    register("selector", "point", point, units={"at": "length"})

    # ---------------------------------------------------------------- supports and loads
    from toporlogy.bc import Fixed

    def fixed(ctx: BuildContext, where: Any, dofs: Any = "all", name: str | None = None) -> Fixed:
        """Zero displacement of the selected nodes in the directions ``dofs`` ("all", "x", "xy", [x, z], or
        vectors [[1, 1, 0]])."""
        return Fixed(checked_selector(ctx, where), _directions(dofs, ctx.domain.dim), name=name)

    def force(ctx: BuildContext, where: Any, force: list, name: str | None = None) -> Any:
        """Total force vector (Fx, Fy[, Fz]) spread equally over the selected nodes."""
        if len(force) != ctx.domain.dim:
            raise ValueError(f"force needs {ctx.domain.dim} components, got {len(force)}")
        return SpreadForce(checked_selector(ctx, where), [float(f) for f in force], name)

    register("support", "fixed", fixed, nested={"where": "selector"}, types={"dofs": None})
    register("load", "force", force, nested={"where": "selector"}, units={"force": "force (total)"})

    # ---------------------------------------------------------------- chain links
    register("link", "embed", Embed, context=("domain",))
    register("link", "density_filter", DensityFilter, context=("domain",), types={"rmin": float},
             units={"rmin": "elements"})
    register("link", "heaviside_projection", HeavisideProjection)
    register("link", "passive", Passive, context=("domain",))

    # ---------------------------------------------------------------- interpolation (E0 from material)
    def simp(ctx: BuildContext, penal: float = 3.0, emin_ratio: float = 1e-6) -> SIMP:
        """Modified SIMP E = (Emin + x^p (E0 - Emin)) frac; E0 from material.E0."""
        return SIMP(ctx.domain, E0=ctx.material["E0"], penal=penal, emin_ratio=emin_ratio)

    def ramp(ctx: BuildContext, q: float = 8.0, emin_ratio: float = 1e-6) -> RAMP:
        """RAMP (experimental); E0 from material.E0."""
        return RAMP(ctx.domain, E0=ctx.material["E0"], q=q, emin_ratio=emin_ratio)

    register("interpolation", "simp", simp)
    register("interpolation", "ramp", ramp)

    # ---------------------------------------------------------------- solver and physics
    register("solver", "direct", DirectSolver)

    def elasticity(ctx: BuildContext, element: str | None = None, solver: Any = "direct") -> Any:
        """Static linear elasticity; nu from material.nu; element null = H8 (3-D) / Q4 (2-D). Oblique supports and
        springs use bc.ConstrainedElasticity, load cases with their own supports bc.PerCaseSupports."""
        spec = (ctx.config.get("physics") or {}).get("solver")
        made = [0]

        def factory() -> Any:              # first physics gets the built solver, further ones a fresh one each
            made[0] += 1
            if made[0] == 1 or not isinstance(spec, dict):
                return solver
            return REGISTRY.build("solver", spec, ctx, join(ctx.path, "solver"))

        return ctx.bc.physics(nu=ctx.material["nu"], element=element, solver_factory=factory)

    register("physics", "elasticity", elasticity, nested={"solver": "solver"})

    # ---------------------------------------------------------------- responses
    def compliance(ctx: BuildContext, weights: list | None = None) -> Compliance:
        """Weighted compliance sum_c w_c F_c.U_c; weights null = the load-case weights."""
        return Compliance(_case_weights(ctx, weights))

    register("response", "compliance", compliance)
    register("response", "volume", Volume)

    # ---------------------------------------------------------------- optimizers
    def oc(ctx: BuildContext, move: float = 0.2, l1: float = 0.0, l2: float = 1e9, bisection_tol: float = 1e-4,
           max_bisect: int = 80, dv_scale: float | None = None) -> OC:
        """Optimality criteria (one constraint); dv_scale null = number of active elements (classic bisection)."""
        return OC(move=move, l1=l1, l2=l2, bisection_tol=bisection_tol, max_bisect=max_bisect,
                  dv_scale=float(ctx.domain.n_active) if dv_scale is None else dv_scale)

    register("optimizer", "oc", oc)
    register("optimizer", "mma", "toporlogy.optim.mma:MMA")

    # ---------------------------------------------------------------- schedules
    def beta_continuation(ctx: BuildContext, beta0: float = 1.0, beta_max: float = 16.0, every: int = 25,
                          tol: float | None = None) -> BetaContinuation:
        """beta doubles every `every` iterations or when change <= tol (null = run.tol)."""
        return BetaContinuation(ctx.chain, beta0, beta_max, every,
                                float(ctx.run.get("tol", 0.01)) if tol is None else tol)

    register("schedule", "beta_continuation", beta_continuation)


def _case_weights(ctx: BuildContext, weights: list | None) -> list[float] | None:
    w = ctx.case_weights if weights is None else [float(v) for v in weights]
    if len(w) != ctx.n_cases:
        raise ValueError(f"{len(w)} weights for {ctx.n_cases} load cases")
    return None if all(v == 1.0 for v in w) else w


def _region_mask(shapes: Sequence[Any], d: Any) -> np.ndarray:
    """Elements of ``d`` more than half covered by the CSG of ``shapes`` (list of built shapes with ``op``)."""
    from toporlogy.geometry import coverage
    return coverage(_combine(shapes), d) > 0.5


def _combine(shapes: Sequence[Any]) -> Any:
    from toporlogy.geometry import csg
    if not shapes:
        raise ValueError("at least one shape is required")
    ops = [(getattr(s, "op", "add"), s) for s in shapes]
    if ops[0][0] != "add":
        raise ValueError(f'the first shape must have op "add", got "{ops[0][0]}"')
    return csg(ops)


# ============================================================================================== feature components
from toporlogy.bc import Applied, Load  # noqa: E402  (toporlogy is initialised before this module)


class SpreadForce(Load):
    """Total force spread equally over the selected nodes (the ``force`` load; not area-weighted)."""

    kind = "spread"

    def __init__(self, where: Any, force: Sequence[float], name: str | None = None) -> None:
        self.where, self.F = where, np.asarray(force, float)
        self.name = name or f"Force {where!r}"

    def apply(self, info: Any) -> Applied:
        n = self.where.select(info)
        if self.F.size != info.dim:
            raise ValueError(f"{self.name}: force needs {info.dim} components")
        f = np.broadcast_to(self.F / len(n), (len(n), info.dim)).copy()
        return Applied(self.name, self.kind, n, f, self.F.copy(), info.X[n].mean(axis=0), None, [])


def _bores(ctx: BuildContext, min_diameter: float | None, max_roundness: float, min_length: float | None) -> list:
    from toporlogy.geometry import find_bores
    key = ("bores", id(ctx.domain), min_diameter, max_roundness, min_length)
    if key not in ctx.extra:
        ctx.extra[key] = find_bores(ctx.domain, min_diameter, max_roundness, min_length)
        ctx.extra["bores_found"] = ctx.extra[key]
    return ctx.extra[key]


def _describe_bores(bores: Sequence[Any]) -> str:
    def f(v: Any) -> str:
        return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"
    return "; ".join(f"{b.id} r {b.radius:.4g} at {f(b.center)}" + (f" axis {'xyz'[int(np.argmax(np.abs(b.axis)))]}"
                     if b.axis is not None else "") for b in bores) or "none"


def find_bore(ctx: BuildContext, index: int | None, near: list | None, min_diameter: float | None = None,
              max_roundness: float = 0.35, min_length: float | None = None) -> Any:
    """An auto-detected bore: ``index`` (1-based, = id "B<index>"), ``near`` (point: nearest axis), or the only one."""
    bores = _bores(ctx, min_diameter, max_roundness, min_length)
    if not bores:
        raise ValueError("no bore found in the part (find_bores: enclosed round holes, diameter >= 3 h)")
    if index is not None:
        if not 1 <= int(index) <= len(bores):
            raise ValueError(f"bore index {index} out of range; found: {_describe_bores(bores)}")
        return bores[int(index) - 1]
    if near is not None:
        p = _point(near, ctx.domain, "near")

        def dist(b: Any) -> float:
            q = p - b.center
            if b.axis is not None:
                q = q - (q @ b.axis) * b.axis
            return float(np.linalg.norm(q))
        return min(bores, key=dist)
    if len(bores) > 1:
        raise ValueError(f"{len(bores)} bores found, give index or near: {_describe_bores(bores)}")
    return bores[0]


def _is_cylinder(s: Any, what: str) -> Any:
    from toporlogy.bc import OnCylinder
    if not isinstance(s, OnCylinder):
        raise ValueError(f"{what} needs a bore / cylinder selector (type bore or on_cylinder), got {s!r}")
    return s


def _robust_set(ctx: BuildContext) -> Any:
    from toporlogy.robust import RobustSet
    if ctx.robust is None:
        ctx.robust = RobustSet()
        ctx.extra.setdefault("notes", []).append("robust: no robust block; default thresholds eta_e 0.75, eta_d 0.25")
    return ctx.robust


def _field_source(ctx: BuildContext, source: str | None) -> Any:
    if source in (None, "xphys", "nominal"):
        return None
    if source not in ("eroded", "intermediate", "dilated"):
        raise ValueError(f'source: xphys | eroded | intermediate | dilated, got "{source}"')
    return _robust_set(ctx).field(source)


def _interfaces(ctx: BuildContext) -> dict[str, np.ndarray]:
    if not ctx.interfaces:
        raise ValueError("no interface regions (connectivity.interfaces, or supports and loads)")
    return ctx.interfaces


def _post_state(ctx: BuildContext) -> dict[str, Any]:
    return ctx.extra.setdefault("post_state", {})


def _design(state: dict[str, Any], res: Any) -> tuple[np.ndarray, str]:
    """x̄ for the geometry steps and its label: the repaired variant of ``res`` when a connectivity_gate repaired
    this result ("repaired"), else ``res.xphys`` ("optimized"). ``res`` itself is never modified, so its x, xphys,
    objective, constraints and U keep describing the optimised design."""
    d = state.get("design")
    if d is not None and d["source"] is res:
        return d["xphys"], d["label"]
    return res.xphys, "optimized"


def _reevaluate(study: Any, xphys: np.ndarray, run: dict[str, Any]) -> dict[str, Any]:
    """Objective and constraints of a repaired density (Study.evaluate_density), with the run_setup feasibility
    rule. Constraints that need the design chain are listed with ``value: None`` and the reason; ``feasible`` is
    False if an evaluated constraint is violated, None if none is violated but some could not be evaluated."""
    ev = study.evaluate_density(xphys)
    ftol = float(run.get("feasibility_tol", 1e-3))
    cons = []
    for c in study.constraints:
        if c.key in ev.values:
            v = ev.values[c.key]
            cons.append({"name": c.key, "value": v, "upper": c.upper,
                         "satisfied": bool(v <= c.upper + ftol * max(abs(c.upper), 1e-12))})
        else:
            cons.append({"name": c.key, "value": None, "upper": c.upper, "satisfied": None,
                         "not_evaluated": ev.skipped[c.key]})
    obj = study.objective.name
    feasible: bool | None = True
    if any(c["satisfied"] is False for c in cons):
        feasible = False
    elif any(c["satisfied"] is None for c in cons):
        feasible = None
    return {"objective": ev.values.get(obj), "objective_name": obj, "constraints": cons, "feasible": feasible,
            **({"objective_not_evaluated": ev.skipped[obj]} if obj in ev.skipped else {})}


def _rel(files: dict[str, str], folder: Any) -> dict[str, str]:
    from pathlib import Path
    out = {}
    for k, v in files.items():
        try:
            out[k] = str(Path(v).relative_to(Path(folder)))
        except ValueError:
            out[k] = str(v)
    return out


def _register_features() -> None:
    from toporlogy import bc as B
    from toporlogy import geometry as G
    from toporlogy import manufacturing as MF
    from toporlogy.domain import Domain

    length = "length"

    # ---------------------------------------------------------------- shapes (geometry CSG primitives)
    def _op(s: Any, op: str) -> Any:
        o = str(op).strip().lower()
        if o not in ("add", "cut", "intersect"):
            raise ValueError(f'op: add | cut | intersect, got "{op}"')
        s.op = o
        return s

    def box_shape(lo: list, hi: list, op: str = "add") -> Any:
        """Axis-aligned box lo <= x <= hi (2-D rectangle or 3-D box)."""
        return _op(G.Box(lo, hi), op)

    def ball(center: list, radius: float, op: str = "add") -> Any:
        """Sphere (3-D) or circle (2-D)."""
        return _op(G.Ball(center, radius), op)

    def cylinder(p0: list, p1: list, radius: float, op: str = "add") -> Any:
        """Capped circular cylinder with axis p0 -> p1 (3-D)."""
        return _op(G.Cylinder(p0, p1, radius), op)

    def capsule(p0: list, p1: list, radius: float, op: str = "add") -> Any:
        """Points within radius of the segment p0-p1 (2-D rounded slot / 3-D capsule)."""
        return _op(G.Capsule(p0, p1, radius), op)

    def halfspace(point: list, normal: list, op: str = "intersect") -> Any:
        """(x - point).normal <= 0 (material opposite to the normal); unbounded: use with op intersect or cut."""
        return _op(G.HalfSpace(point, normal), op)

    def polygon(vertices: list, op: str = "add") -> Any:
        """2-D simple polygon (vertices in order)."""
        return _op(G.Polygon(vertices), op)

    def prism(polygon: list, axis: Any = "z", range: list = (0.0, 1.0), op: str = "add") -> Any:  # noqa: A002
        """Polygon extruded along axis over range [lo, hi]; polygon (x, y) for z, (x, z) for y, (y, z) for x."""
        return _op(G.Prism(polygon, axis, (float(range[0]), float(range[1]))), op)

    def slot(p0: list, p1: list, radius: float, depth: list, op: str = "add") -> Any:
        """Rounded slot: the stadium of half-width radius about p0-p1 extruded along the vector depth (3-D)."""
        return _op(G.Slot(p0, p1, radius, depth), op)

    def bore_shape(ctx: BuildContext, index: int | None = None, near: list | None = None, extra_radius: float = 0.0,
                   extend: float = 0.0, through: bool = True, op: str = "cut") -> Any:
        """An auto-detected bore of the part as a cylinder (finalize: preserve it exactly). index 1-based (B1 ...),
        near = point (nearest axis); through: infinite in post-processing."""
        if ctx.domain is None:
            raise ValueError("a bore shape needs a built domain (use it in post steps or selectors)")
        b = find_bore(ctx, index, near)
        s = _op(b.shape(extra_radius, extend), op)
        s.through, s.bore = bool(through), b
        return s

    for name, fn in (("box", box_shape), ("ball", ball), ("sphere", ball), ("circle", ball),
                     ("cylinder", cylinder), ("capsule", capsule), ("halfspace", halfspace),
                     ("polygon", polygon), ("prism", prism), ("slot", slot)):
        register("shape", name, fn, types={"axis": None},
                 units={k: length for k in ("lo", "hi", "center", "radius", "p0", "p1", "point", "vertices",
                                            "polygon", "range", "depth")})
    register("shape", "bore", bore_shape, units={"near": length, "extra_radius": length, "extend": length})

    # ---------------------------------------------------------------- domains from geometry
    def csg_domain(ctx: BuildContext, shapes: Any, h: float, padding: int = 0, bounds: list | None = None,
                   sub: int = 4, min_frac: float = 0.05, connectivity: str | None = "largest", keep: Any = None,
                   void: Any = None, thickness: float = 1.0) -> Domain:
        """Part = CSG of shapes (applied in order by their op), voxelised with partial volumes on a grid of edge h
        (padding empty layers; bounds [[lo], [hi]] for unbounded shapes); keep / void: shapes."""
        shape = _combine(shapes)
        bb = None if bounds is None else (list(bounds[0]), list(bounds[1]))
        d = G.voxelize(shape, h, padding=padding, bounds=bb, sub=sub, min_frac=min_frac, connectivity=connectivity,
                       thickness=thickness)
        return _with_regions(d, keep, void)

    def stl_domain(ctx: BuildContext, path: str, h: float, scale: float = 1.0, padding: int = 1, sub: int = 3,
                   min_frac: float = 0.05, connectivity: str | None = "largest", bounds: list | None = None,
                   keep: Any = None, void: Any = None) -> Domain:
        """Part = closed triangle surface (binary or ASCII STL; vertices x scale, in the unit of h; a relative path
        is taken from the working directory), voxelised with partial volumes."""
        from pathlib import Path
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"STL file not found: {p.resolve()}")
        V, F = G.read_stl(p)
        bb = None if bounds is None else (list(bounds[0]), list(bounds[1]))
        d = G.voxelize_mesh(np.asarray(V, float) * float(scale), F, h, padding=padding, sub=sub, min_frac=min_frac,
                            connectivity=connectivity, bounds=bb)
        return _with_regions(d, keep, void)

    def _with_regions(d: Domain, keep: Any, void: Any) -> Domain:
        if keep is None and void is None:
            return d
        km = d.keep | (_region_mask(keep, d) if keep is not None else False)
        vm = d.void | (_region_mask(void, d) if void is not None else False)
        out = Domain(d.shape, d.h, origin=d.origin, frac=d.frac, keep=km, void=vm, thickness=d.thickness)
        out.voxelization_info = d.voxelization_info
        return out

    regions = {"keep": "shape[]", "void": "shape[]"}
    register("domain", "csg", csg_domain, nested={"shapes": "shape[]", **regions},
             units={"h": length, "bounds": length, "thickness": length})
    register("domain", "stl", stl_domain, nested=regions, units={"h": length, "bounds": length})

    # ---------------------------------------------------------------- selectors (bc)
    def all_nodes(ctx: BuildContext) -> Any:
        """Every node of the part."""
        return B.AllNodes()

    def surface(ctx: BuildContext) -> Any:
        """Nodes on the part surface."""
        return B.SurfaceNodes()

    def in_box(ctx: BuildContext, lo: list | None = None, hi: list | None = None, surface: bool = False,
               tol: float | None = None) -> Any:
        """Nodes with lo <= x <= hi; entries numbers, "min", "mid", "max" or null (unbounded); surface: only
        surface nodes."""
        d = ctx.domain
        lo_ = [None] * d.dim if lo is None else lo
        hi_ = [None] * d.dim if hi is None else hi
        if len(lo_) != d.dim or len(hi_) != d.dim:
            raise ValueError(f"lo and hi need {d.dim} entries")
        a = [_coord(v, k, d, -np.inf) for k, v in enumerate(lo_)]
        b = [_coord(v, k, d, np.inf) for k, v in enumerate(hi_)]
        return B.InBox(a, b, surface=surface, tol=tol)

    def _plane(ctx: BuildContext, axis: Any, at: Any, point: list | None, normal: Any) -> tuple[np.ndarray, Any]:
        d = ctx.domain
        if axis is not None:
            if point is not None or normal is not None:
                raise ValueError("give axis + at, or point + normal")
            k = _axis(axis, d.dim)
            p = d.origin.astype(float).copy()
            p[k] = _coord(at if at is not None else "min", k, d)
            return p, k
        if point is None or normal is None:
            raise ValueError("give axis + at, or point + normal")
        return _point(point, d), _vector(normal, d.dim, "normal")

    def on_plane(ctx: BuildContext, axis: Any = None, at: Any = None, point: list | None = None,
                 normal: Any = None, tol: float | None = None, facing: str | None = "any",
                 max_angle: float = 60.0) -> Any:
        """Surface nodes on a plane (axis + at, or point + normal) within tol (default h/2) whose outward normal is
        (anti)parallel to it ("any"), along it ("out"), against it ("in") or not filtered (null)."""
        p, n = _plane(ctx, axis, at, point, normal)
        return B.OnPlane(p, n, tol=tol, facing=facing, max_angle=max_angle)

    def half_space(ctx: BuildContext, point: list, normal: Any, surface: bool = False,
                   tol: float | None = None) -> Any:
        """Nodes with (x - point).normal <= tol."""
        return B.HalfSpaceNodes(_point(point, ctx.domain), _vector(normal, ctx.domain.dim, "normal"), surface, tol)

    def on_cylinder(ctx: BuildContext, p0: list, radius: float, p1: list | None = None, band: Any = None,
                    facing: str | None = "inward", max_angle: float = 60.0, extend: float = 0.0) -> Any:
        """Surface nodes on a cylinder face (3-D axis p0 -> p1; 2-D circle about p0): a bore ("inward") or a boss
        ("outward"); band = radial tolerance (default 1.5 h inside / 0.75 h outside)."""
        d = ctx.domain
        q1 = None if p1 is None else _point(p1, d, "p1")
        return B.OnCylinder(_point(p0, d, "p0"), q1, radius, band=band, facing=facing, max_angle=max_angle,
                            extend=extend)

    def bore(ctx: BuildContext, index: int | None = None, near: list | None = None, band: Any = None,
             facing: str | None = "inward", max_angle: float = 60.0, extend: float = 0.0,
             min_diameter: float | None = None) -> Any:
        """Wall of an auto-detected bore (find_bores): index 1-based (B1, B2 ...), near = a point (nearest axis),
        or the only bore of the part."""
        b = find_bore(ctx, index, near, min_diameter)
        return B.OnCylinder.from_bore(b, facing=facing, band=band, max_angle=max_angle, extend=extend)

    def in_sphere(ctx: BuildContext, center: list, radius: float, surface: bool = False) -> Any:
        """Nodes within radius of center."""
        return B.InSphere(_point(center, ctx.domain, "center"), radius, surface)

    def in_shape(ctx: BuildContext, shape: Any, surface: bool = False, tol: float | None = None) -> Any:
        """Nodes inside a shape (sdf <= tol)."""
        return B.InShape(shape, surface, tol)

    def at_points(ctx: BuildContext, points: list, tol: float | None = None) -> Any:
        """Nodes at the given points (each must hit a node within tol, default 1e-6 h)."""
        return B.AtPoints([_point(p, ctx.domain) for p in points], tol)

    def nearest(ctx: BuildContext, points: list, k: int = 1, surface: bool = True) -> Any:
        """The k nearest (surface) nodes to each point."""
        return B.Nearest([_point(p, ctx.domain) for p in points], k, surface)

    def node_ids(ctx: BuildContext, ids: list) -> Any:
        """Explicit node numbers (advanced)."""
        return B.NodeIds([int(i) for i in ids])

    def _fold(of: Sequence[Any], op: str) -> Any:
        if len(of) < 2:
            raise ValueError("of needs at least two selectors")
        out = as_selector(of[0])
        for s in of[1:]:
            t = as_selector(s)
            out = out | t if op == "|" else (out & t if op == "&" else out - t)
        return out

    def union(ctx: BuildContext, of: Any) -> Any:
        """Nodes selected by any of the selectors."""
        return _fold(of, "|")

    def intersect(ctx: BuildContext, of: Any) -> Any:
        """Nodes selected by all selectors."""
        return _fold(of, "&")

    def subtract(ctx: BuildContext, of: Any) -> Any:
        """Nodes of the first selector minus those of the others."""
        return _fold(of, "-")

    pts = {"lo": length, "hi": length, "point": length, "p0": length, "p1": length, "center": length,
           "radius": length, "near": length, "points": length, "tol": length, "band": length, "at": length,
           "extend": length, "min_diameter": length, "max_angle": "deg"}
    for name, fn, nest in (("all", all_nodes, {}), ("surface", surface, {}), ("in_box", in_box, {}),
                           ("on_plane", on_plane, {}), ("half_space", half_space, {}),
                           ("on_cylinder", on_cylinder, {}), ("bore", bore, {}), ("in_sphere", in_sphere, {}),
                           ("in_shape", in_shape, {"shape": "shape"}), ("at_points", at_points, {}),
                           ("nearest", nearest, {}), ("node_ids", node_ids, {}),
                           ("union", union, {"of": "selector[]"}), ("intersect", intersect, {"of": "selector[]"}),
                           ("subtract", subtract, {"of": "selector[]"})):
        sig = inspect.signature(fn).parameters
        register("selector", name, fn, nested=nest, types={k: None for k in ("axis", "at", "normal", "band")
                                                          if k in sig},
                 units={k: u for k, u in pts.items() if k in sig})

    # ---------------------------------------------------------------- supports (bc)
    def roller(ctx: BuildContext, where: Any, normal: Any, name: str | None = None) -> Any:
        """Zero displacement along normal (axis or vector) at the selected nodes; free in the tangent plane."""
        return B.Roller(checked_selector(ctx, where), _vector(normal, ctx.domain.dim, "normal"), name)

    def symmetry(ctx: BuildContext, axis: Any = None, at: Any = None, point: list | None = None, normal: Any = None,
                 tol: float | None = None, name: str | None = None) -> Any:
        """Symmetry plane (axis + at, or point + normal): zero normal displacement of the surface nodes on it."""
        p, n = _plane(ctx, axis, at, point, normal)
        s = B.Symmetry(p, n, tol=tol, name=name)
        checked_selector(ctx, s.where, "at" if axis is not None else "point")
        return s

    def pinned_bore(ctx: BuildContext, bore: Any, axial: bool = False, tangential: bool = False,
                    name: str | None = None) -> Any:
        """Pin in a bore: radial displacement zero, rotation about the pin free (axial / tangential: also fix those)."""
        sel = _is_cylinder(checked_selector(ctx, bore, "bore"), "pinned_bore")
        return B.PinnedBore(sel, axial=axial, tangential=tangential, name=name)

    def elastic(ctx: BuildContext, where: Any, k: float = 0.0, direction: Any = "normal", total: float | None = None,
                name: str | None = None) -> Any:
        """Springs to ground: stiffness per area k x nodal area ("normal", "all" or a vector direction), or a total
        stiffness shared by area."""
        if not k > 0 and total is None:
            raise ValueError("give k > 0 (per area) or total")
        dirn = direction if isinstance(direction, str) and direction in ("normal", "all") else \
            _vector(direction, ctx.domain.dim).tolist()
        return B.ElasticSupport(checked_selector(ctx, where), k=k, direction=dirn, total=total, name=name)

    register("support", "roller", roller, nested={"where": "selector"}, types={"normal": None})
    register("support", "symmetry", symmetry, types={"axis": None, "at": None, "normal": None},
             units={"at": length, "point": length, "tol": length})
    register("support", "pinned_bore", pinned_bore, nested={"bore": "selector"})
    register("support", "elastic", elastic, nested={"where": "selector"}, types={"direction": None},
             units={"k": "force/length^3", "total": "force/length"})

    # ---------------------------------------------------------------- loads (bc)
    def point_load(ctx: BuildContext, at: list, force: list, radius: float | None = None,
                   name: str | None = None) -> Any:
        """Force at a point, spread over the surface nodes within radius (default 1.5 h) with zero moment about it."""
        return B.PointLoad(_point(at, ctx.domain, "at"), force, radius, name)

    def surface_force(ctx: BuildContext, where: Any, force: list, name: str | None = None) -> Any:
        """Total force over the selected surface, shared by geometric nodal area (uniform traction)."""
        return B.SurfaceForce(checked_selector(ctx, where), force, name)

    def traction(ctx: BuildContext, where: Any, traction: list, geometric: bool = True,
                 name: str | None = None) -> Any:
        """Traction vector (force per area) on the complete surface faces of the selection."""
        return B.Traction(checked_selector(ctx, where), traction, geometric, name)

    def pressure(ctx: BuildContext, where: Any, p: float, normals: str = "faces", name: str | None = None) -> Any:
        """Pressure p (positive pushes into the part) on the complete surface faces of the selection."""
        return B.Pressure(checked_selector(ctx, where), p, normals, name)

    def bearing(ctx: BuildContext, bore: Any, force: list, mode: str = "parallel", on: str = "surface",
                name: str | None = None) -> Any:
        """Pin load on a bore: radial part over the loaded half with weight max(0, cos theta) (projected area)."""
        sel = _is_cylinder(checked_selector(ctx, bore, "bore"), "bearing")
        return B.BearingLoad(sel, force, mode=mode, on=on, name=name)

    def remote_force(ctx: BuildContext, where: Any, force: list, point: list, moment: Any = None,
                     weights: str = "area", name: str | None = None) -> Any:
        """Force (and moment) acting at point, carried by the selected nodes (rigid-body least squares)."""
        return B.RemoteForce(checked_selector(ctx, where), force, _point(point, ctx.domain), moment, weights, name)

    def remote_moment(ctx: BuildContext, where: Any, moment: Any, point: list | None = None, weights: str = "area",
                      name: str | None = None) -> Any:
        """Pure moment about point (default: area centroid of the selection); 2-D: scalar, 3-D: (Mx, My, Mz)."""
        p = None if point is None else _point(point, ctx.domain)
        return B.RemoteMoment(checked_selector(ctx, where), moment, p, weights, name)

    def torque(ctx: BuildContext, bore: Any, torque: float, name: str | None = None) -> Any:
        """Torque about a bore axis (right-hand rule; 2-D: about the out-of-plane axis), zero net force."""
        sel = _is_cylinder(checked_selector(ctx, bore, "bore"), "torque")
        return B.Torque(sel, torque, name)

    def body_force(ctx: BuildContext, acceleration: list, density: float | None = None,
                   name: str | None = None) -> Any:
        """Design-independent body force density x acceleration on the whole part (density null = material.rho)."""
        rho = ctx.material.get("rho", 1.0) if density is None else density
        return B.BodyForce(acceleration, rho, name)

    w = {"where": "selector"}
    register("load", "point", point_load, units={"at": length, "force": "force", "radius": length})
    register("load", "surface_force", surface_force, nested=w, units={"force": "force (total)"})
    register("load", "traction", traction, nested=w, units={"traction": "force/length^2"})
    register("load", "pressure", pressure, nested=w, units={"p": "force/length^2"})
    register("load", "bearing", bearing, nested={"bore": "selector"}, units={"force": "force (total)"})
    register("load", "remote_force", remote_force, nested=w, types={"moment": None},
             units={"force": "force", "point": length, "moment": "force*length"})
    register("load", "remote_moment", remote_moment, nested=w, types={"moment": None},
             units={"moment": "force*length", "point": length})
    register("load", "torque", torque, nested={"bore": "selector"}, units={"torque": "force*length"})
    register("load", "body_force", body_force, units={"acceleration": "length/time^2", "density": "mass/length^3"})

    # ---------------------------------------------------------------- chain links (manufacturing)
    def symmetry_link(ctx: BuildContext, planes: Any) -> Any:
        """Mirror tying about planes {x: 30} or [[x, 30], [y, mid]] (coordinate: number, min, mid, max); first
        link of the chain."""
        items = list(planes.items()) if isinstance(planes, dict) else [tuple(p) for p in planes]
        d = ctx.domain
        return MF.Symmetry(d, [(_axis(a, d.dim), _coord(c, _axis(a, d.dim), d)) for a, c in items])

    def pattern(ctx: BuildContext, kind: str = "translational", offset: list | None = None, n: int | None = None,
                center: list | None = None, axis: Any = "z") -> Any:
        """Pattern repetition: translational (offset vector) or cyclic (n sectors about center[, axis])."""
        d = ctx.domain
        if kind == "translational":
            if offset is None:
                raise ValueError("translational pattern needs offset")
            return MF.Pattern.translational(d, offset)
        if kind == "cyclic":
            if n is None or center is None:
                raise ValueError("cyclic pattern needs n and center")
            return MF.Pattern.cyclic(d, n, _point(center, d, "center"), axis)
        raise ValueError(f'kind: translational | cyclic, got "{kind}"')

    def group_average(ctx: BuildContext) -> Any:
        """Re-ties the filtered field to the groups of the chain's tying link (put right after density_filter)."""
        tying = next((lk for lk in ctx.links if isinstance(lk, MF.Tying)), None)
        if tying is None:
            raise ValueError("group_average needs a tying link (symmetry, extrusion, pattern) first in the chain")
        return MF.GroupAverage(tying)

    register("link", "am_filter", MF.AMFilter, context=("domain",), types={"build": None},
             units={"base_coordinate": length})
    register("link", "casting", MF.Casting, context=("domain",), types={"direction": None},
             units={"parting": length})
    register("link", "symmetry", symmetry_link, types={"planes": None}, units={"planes": length})
    register("link", "extrusion", MF.Extrusion, context=("domain",), types={"axis": None})
    register("link", "pattern", pattern, types={"axis": None}, units={"offset": length, "center": length})
    register("link", "group_average", group_average)

    # ---------------------------------------------------------------- interpolation, physics (dynamics)
    def two_range_simp(ctx: BuildContext, penal: float = 3.0, threshold: float = 0.1, q: float = 1.0,
                       emin_ratio: float = 1e-9) -> Any:
        """SIMP x^p above threshold, threshold^(p-q) x^q below (no spurious low-density modes); E0 from material."""
        from toporlogy.dynamics import TwoRangeSIMP
        return TwoRangeSIMP(ctx.domain, E0=ctx.material["E0"], penal=penal, threshold=threshold, q=q,
                            emin_ratio=emin_ratio)

    register("interpolation", "two_range_simp", two_range_simp)

    def modal(ctx: BuildContext, element: str | None = None, n_modes: int = 6, sigma: float = 0.0, tol: float = 0.0,
              mass: str = "linear", rhomin_ratio: float = 0.0, mass_threshold: float = 0.1, mass_q: float = 6.0,
              added_mass: float = 0.0, added_mass_at: Any = None, cluster_tol: float = 1e-3,
              repeated: str = "warn") -> Any:
        """Free vibration K phi = lambda M phi (no static solve; load_cases optional); rho0 = material.rho; mass
        "linear" or "two_range"; added_mass (total) spread over the nodes of added_mass_at. Supports: fixed DOFs
        along axes only, one support set."""
        from toporlogy.dynamics import LinearMass, TwoRangeMass
        from toporlogy.fem.modal import Modal
        bcs, d = ctx.bc, ctx.domain
        sets = {id(bcs.constraints(i)) for i in range(len(bcs.cases))}
        if len(sets) > 1:
            raise ValueError("modal physics needs one support set (no per-case supports)")
        cs = bcs.constraints(0)
        if cs.needs_extension:
            raise ValueError("modal physics supports zero displacements along x / y / z only (no oblique rollers, "
                             "pinned bores or springs)")
        rho0 = float(ctx.material.get("rho", 1.0))
        if mass == "linear":
            m = LinearMass(d, rho0=rho0, rhomin_ratio=rhomin_ratio)
        elif mass == "two_range":
            m = TwoRangeMass(d, rho0=rho0, threshold=mass_threshold, q=mass_q)
        else:
            raise ValueError(f'mass: linear | two_range, got "{mass}"')
        phys = Modal(d, cs.fixed, nu=ctx.material["nu"], element=element, mass=m, n_modes=n_modes, sigma=sigma,
                     tol=tol, cluster_tol=cluster_tol, repeated=repeated)
        if added_mass:
            if added_mass_at is None:
                raise ValueError("added_mass needs added_mass_at (a selector)")
            nodes = checked_selector(ctx, added_mass_at, "added_mass_at").select(d)
            phys.add_point_mass(nodes, float(added_mass) / len(nodes))
        return phys

    def elasticity_gmg(ctx: BuildContext, element: str | None = None, tol: float = 1e-8, maxit: int = 300,
                       direct_below: int = 4000, fallback_max_elements: int = 120_000, warm_start: bool = True,
                       allow_inexact: bool = False) -> Any:
        """Static elasticity by geometric-multigrid PCG (matrix-free; 3-D only; tol = relative residual). Supports:
        zero displacements along x / y / z, one support set (no oblique rollers, pinned bores, springs or per-case
        supports)."""
        bcs, d = ctx.bc, ctx.domain
        if d.dim != 3:
            raise ValueError("elasticity_gmg is 3-D only; use elasticity in 2-D")
        if len({id(bcs.constraints(i)) for i in range(len(bcs.cases))}) > 1:
            raise ValueError("elasticity_gmg needs one support set (no per-case supports); use elasticity")
        if bcs.constraints(0).needs_extension:
            raise ValueError("elasticity_gmg supports zero displacements along x / y / z only (no oblique rollers, "
                             "pinned bores or springs); use elasticity")
        from toporlogy.fem.elastic_gmg import ElasticityGMG
        return ElasticityGMG(d, bcs.constraints(0).fixed, nu=ctx.material["nu"], element=element, tol=tol,
                             maxit=maxit, direct_below=direct_below, fallback_max_elements=fallback_max_elements,
                             warm_start=warm_start, allow_inexact=allow_inexact)

    register("physics", "elasticity_gmg", elasticity_gmg, units={"direct_below": "active elements",
                                                                 "fallback_max_elements": "active elements"})
    register("physics", "modal", modal, nested={"added_mass_at": "selector"},
             units={"added_mass": "mass (total)", "sigma": "eigenvalue"})

    # ---------------------------------------------------------------- responses
    def _need_modal(ctx: BuildContext, what: str) -> None:
        from toporlogy.fem.modal import Modal
        if not isinstance(ctx.physics, Modal):
            raise ValueError(f"{what} needs physics modal")

    def robust_compliance(ctx: BuildContext, realisations: list = ("eroded", "intermediate", "dilated"),
                          aggregate: str = "ks", rho: float = 50.0, p: float = 16.0, scale: float | None = None,
                          weights: list | None = None) -> Any:
        """Worst-case compliance over the eroded / intermediate / dilated realisations (one solve each); thresholds
        from the robust block; weights null = load-case weights."""
        from toporlogy.robust import RobustCompliance
        return RobustCompliance(_robust_set(ctx), weights=_case_weights(ctx, weights), realisations=realisations,
                                aggregate=aggregate, rho=rho, p=p, scale=scale)

    def dilated_volume(ctx: BuildContext, weighted: bool = False) -> Any:
        """Volume of the dilated realisation (robust formulation); robust_continuation keeps the blueprint at the
        constraint's initial bound."""
        from toporlogy.robust import DilatedVolume
        return DilatedVolume(_robust_set(ctx), weighted)

    def local_volume(ctx: BuildContext, t_max: float | None = None, radius: float | None = None,
                     member: str | None = None, p: float = 16.0, aggregate: str = "pmean", adaptive: float = 0.0,
                     outside: Any = "void", source: str = "xphys") -> Any:
        """Maximum member thickness: local volume fraction in a ball <= alpha. t_max (length) sets radius and the
        constraint bound alpha (max_thickness_settings); or radius (elements) with an explicit upper. source:
        xphys | dilated | intermediate | eroded."""
        from toporlogy.lengthscale import LocalVolume, max_thickness_settings
        d = ctx.domain
        s = None
        if t_max is not None:
            s = max_thickness_settings(t_max, d.h, d.dim, radius, member)
            radius = s["radius"]
        elif radius is None:
            raise ValueError("give t_max (length) or radius (elements)")
        r = LocalVolume(radius, p=p, aggregate=aggregate, adaptive=adaptive, outside=outside,
                        source=_field_source(ctx, source))
        if s is not None:
            r.config_upper = float(s["alpha"])
            r.settings = s
        return r

    def grey_level(ctx: BuildContext, source: str = "xphys") -> Any:
        """M_nd = mean 4 x (1 - x) over the design elements (monitor or constraint, e.g. <= 0.05)."""
        from toporlogy.lengthscale import GreyLevel
        return GreyLevel(_field_source(ctx, source))

    def conduction_connectivity(ctx: BuildContext, sink: str | None = None, kmin_ratio: float = 1e-3,
                                penal: float = 3.0, aggregate: str = "pnorm", p: float = 8.0,
                                eta: float | None = 0.7, reference: Any = "solid") -> Any:
        """One body: virtual heat conduction from every interface to the sink (default the first); upper default
        20. Interfaces: connectivity.interfaces, else one per support and load."""
        from toporlogy.connectivity import ConductionConnectivity
        r = ConductionConnectivity(ctx.domain, _interfaces(ctx), sink, kmin_ratio=kmin_ratio, penal=penal,
                                   aggregate=aggregate, p=p, eta=eta, reference=reference)
        r.config_upper = 20.0
        return r

    def coupling_compliance(ctx: BuildContext, held: str | None = None, aggregate: str = "pnorm", p: float = 8.0,
                            eta: float | None = 0.7, reference: Any = "first") -> Any:
        """One body: auxiliary cases with the held interface clamped and unit forces on the others; upper default
        2.5 (x first value)."""
        from toporlogy.connectivity import CouplingCompliance
        r = CouplingCompliance(ctx.domain, _interfaces(ctx), held, nu=ctx.material["nu"], aggregate=aggregate, p=p,
                               eta=eta, reference=reference)
        r.config_upper = 2.5
        return r

    def frequency(ctx: BuildContext, k: int = 1, E_unit: float = 1.0, rho_unit: float = 1.0,
                  length_unit: float = 1.0) -> Any:
        """f_k in Hz (the model's E, rho and length units given in SI multiples)."""
        from toporlogy.dynamics import Frequency
        _need_modal(ctx, "frequency")
        return Frequency(k, E_unit, rho_unit, length_unit)

    def eigenvalue(ctx: BuildContext, k: int = 1) -> Any:
        """lambda_k = omega_k^2 in model units (only differentiable while simple)."""
        from toporlogy.dynamics import Eigenvalue
        _need_modal(ctx, "eigenvalue")
        return Eigenvalue(k)

    def min_eigenvalue_ks(ctx: BuildContext, n_modes: int = 3, rho: float = 50.0, scale: float | None = None,
                          kind: str = "ks") -> Any:
        """Smooth lower bound of min(lambda_1 .. lambda_n) (KS or p-norm); maximise it with type maximize."""
        from toporlogy.dynamics import MinEigenvalueKS
        _need_modal(ctx, "min_eigenvalue_ks")
        return MinEigenvalueKS(n_modes, rho, scale, kind)

    def maximize(ctx: BuildContext, response: Any) -> Any:
        """-response (the study minimises), e.g. objective {type: maximize, response: min_eigenvalue_ks}."""
        from toporlogy.dynamics import Negated
        return Negated(response)

    def min_frequency(ctx: BuildContext, f_min: float, n_modes: int = 1, rho: float = 50.0, E_unit: float = 1.0,
                      rho_unit: float = 1.0, length_unit: float = 1.0) -> Any:
        """Constraint f_1 >= f_min (Hz) as 1 - lambda_low / lambda_min <= 0 (upper 0, scale 1 by default)."""
        from toporlogy.dynamics import FrequencyMargin
        _need_modal(ctx, "min_frequency")
        r = FrequencyMargin(f_min, n_modes, rho, E_unit, rho_unit, length_unit)
        r.config_upper, r.config_scale = 0.0, 1.0
        return r

    register("response", "robust_compliance", robust_compliance, types={"realisations": list})
    register("response", "dilated_volume", dilated_volume)
    register("response", "local_volume", local_volume, types={"outside": None},
             units={"t_max": length, "radius": "elements"})
    register("response", "grey_level", grey_level)
    register("response", "conduction_connectivity", conduction_connectivity, types={"reference": None})
    register("response", "coupling_compliance", coupling_compliance, types={"reference": None})
    register("response", "frequency", frequency)
    register("response", "eigenvalue", eigenvalue)
    register("response", "min_eigenvalue_ks", min_eigenvalue_ks)
    register("response", "maximize", maximize, nested={"response": "response"})
    register("response", "min_frequency", min_frequency, units={"f_min": "Hz"})

    # ---------------------------------------------------------------- schedule
    def robust_continuation(ctx: BuildContext, beta0: float = 1.0, beta_max: float = 32.0, every: int = 50,
                            tol: float | None = None, volume_every: int = 20, target: float | None = None,
                            update_volume: bool = True) -> Any:
        """beta doubles every `every` iterations or when change <= tol (null = run.tol); every volume_every
        iterations the dilated volume bound becomes target V_d / V_i (target null = its initial upper)."""
        from toporlogy.robust import RealisationVolume, Robust, RobustContinuation
        vol = next((c for c in ctx.constraints if isinstance(c.response, RealisationVolume)), None) \
            if update_volume else None
        rescale = [r for r in [ctx.objective, *[c.response for c in ctx.constraints]] if isinstance(r, Robust)]
        return RobustContinuation(ctx.chain, ctx.robust, beta0, beta_max, every,
                                  float(ctx.run.get("tol", 0.01)) if tol is None else tol, volume=vol,
                                  target=(None if vol is None else (vol.upper if target is None else target)),
                                  volume_every=volume_every, rescale=rescale)

    register("schedule", "robust_continuation", robust_continuation)

    # ---------------------------------------------------------------- post steps
    def connectivity_gate(ctx: BuildContext, threshold: float = 0.5, connectivity: Any = "face",
                          robust_thresholds: list | None = (0.3, 0.7), erosion: bool = True, voids: str = "fail",
                          repair: str = "none", section: int = 2) -> Any:
        """One-body gate (analyse_connectivity) over the interfaces; verdict PASS / FRAGILE / FAIL into
        summary.json (FAIL: exit 1). repair bridge: on FAIL join the bodies into a repaired variant
        (result_repaired.npz; later geometry steps use it, the optimisation result is not modified), gate it again
        and re-evaluate its objective and constraints ("repaired_design"); a repair that violates a constraint
        fails the gate."""
        from toporlogy.connectivity import analyse_connectivity, bridge
        if repair not in ("none", "bridge"):
            raise ValueError(f'repair: none | bridge, got "{repair}"')
        masks = _interfaces(ctx)
        state = _post_state(ctx)

        def gate(dom: Any, x: np.ndarray) -> Any:
            return analyse_connectivity(dom, x, masks, threshold, connectivity, robust_thresholds, erosion,
                                        voids=voids)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            x0, label = _design(state, res)
            rep = gate(study.domain, x0)
            out: dict[str, Any] = {"verdict": rep.verdict, "design": label, **_gate_dict(rep)}
            if rep.verdict == "FAIL" and repair == "bridge":        # FRAGILE is one body already: nothing to join
                br = bridge(study.domain, x0, masks, threshold=threshold, section=section)
                rep2 = gate(study.domain, br.xphys)
                np.savez_compressed(f"{folder}/result_repaired.npz", xphys=br.xphys)
                state["design"] = {"source": res, "xphys": br.xphys, "label": "repaired"}
                out = {"verdict": rep2.verdict, "verdict_before_repair": rep.verdict, "design": "repaired",
                       **_gate_dict(rep2),
                       "repair": {"bridges": br.n_bridges, "added_cells": br.added_cells,
                                  "added_volume": br.added_volume, "removed_cells": br.removed_cells,
                                  "file": "result_repaired.npz"},
                       "repaired_design": _reevaluate(study, br.xphys, ctx.run)}
            rd = out.get("repaired_design")
            out["passed"] = out["verdict"] != "FAIL" and (rd is None or rd["feasible"] is not False)
            rp = out.get("repair")
            bad = [c["name"] for c in rd["constraints"] if c["satisfied"] is False] if rd else []
            out["summary"] = (f"{out['verdict']} ({out['n_bodies']} bod{'y' if out['n_bodies'] == 1 else 'ies'}"
                              f"{', unattached ' + ', '.join(out['unattached']) if out['unattached'] else ''})"
                              + (f" after {rp['bridges']} bridge(s), +{rp['added_cells']} cells; was "
                                 f"{out['verdict_before_repair']}" if rp else "")
                              + (f"; repaired design violates {', '.join(bad)}" if bad else ""))
            return out
        return step

    def lengthscale_step(ctx: BuildContext, threshold: float = 0.5, outside: Any = "void",
                         upsample: int | None = None, smooth: float = 0.0, margin: float = 0.0,
                         percentiles: list = (1.0, 5.0, 50.0), radii: list = ()) -> Any:
        """Measure member thickness and void size of the final design (lengths in the unit of h)."""
        from toporlogy.lengthscale import measure_lengthscale
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            d = study.domain
            xp, label = _design(state, res)
            rep = measure_lengthscale(xp, d.h, threshold, outside, upsample, d.active, percentiles, radii,
                                      margin, smooth)
            keys = ("solid_min", "solid_max", "void_min", "void_max", "grey_fraction", "m_nd", "solid_fraction",
                    "resolution")
            out: dict[str, Any] = {k: float(getattr(rep, k)) for k in keys}
            out["design"] = label
            out.update(solid_percentiles=rep.solid_percentiles, void_percentiles=rep.void_percentiles,
                       opening_residue=rep.opening_residue, closing_residue=rep.closing_residue)
            out["summary"] = (f"solid {out['solid_min']:.3g} .. {out['solid_max']:.3g}, void min "
                              f"{out['void_min']:.3g} (resolution {out['resolution']:.3g})")
            return out
        return step

    register("post", "connectivity_gate", connectivity_gate, types={"connectivity": None})
    register("post", "measure_lengthscale", lengthscale_step, types={"outside": None},
             units={"margin": length, "radii": length, "smooth": "elements"})
    register("post", "finalize", "toporlogy.registry:finalize_step",
             units={"sigma": length, "r_open": length, "r_close": length, "r_fillet_min": length})
    register("post", "manufacturing_checks", "toporlogy.registry:checks_step",
             units={"min_wall": length, "plate_level": length, "min_concave_radius": length})
    register("post", "reverify", "toporlogy.registry:reverify_step", units={"E": "force/length^2",
                                                                                "size": length})
    register("post", "export", "toporlogy.registry:export_step")


def _gate_dict(rep: Any) -> dict[str, Any]:
    m = rep.min_neck
    return {"n_bodies": int(rep.n_bodies), "connected": bool(rep.connected), "fragile": bool(rep.fragile),
            "reasons": list(rep.reasons), "unattached": list(rep.unattached),
            "interface_bodies": {k: list(v) for k, v in rep.interface_bodies.items()},
            "body_volume": [float(v) for v in rep.body_volume[:10]], "enclosed_voids": len(rep.enclosed_voids),
            "min_neck": None if m is None else {"between": [m.a, m.b], "cells": m.cells, "area": m.area},
            "checks": [{"label": c.label, "n_bodies": c.n_bodies, "connected": c.connected} for c in rep.checks],
            "text": rep.summary()}


# ============================================================================================== post steps (lazy)
# The post package (and its compiled kernels) is imported only when one of these steps is listed or used; their
# parameters are the fields of PostParams / CheckThresholds (signature built on first access, PEP 562).
_LAZY: dict[str, Any] = {}


def _post_shape(s: Any) -> Any:
    """geometry shape → post primitive (exact for box, ball, cylinder, halfspace; else an implicit wrapper)."""
    from toporlogy import geometry as G
    from toporlogy.post import primitives as P
    if s.dim != 3:
        raise ValueError("post-processing primitives are 3-D")
    if isinstance(s, G.Box):
        return P.Box(s.lo.tolist(), s.hi.tolist())
    if isinstance(s, G.Ball):
        return P.Sphere(s.center.tolist(), s.radius)
    if isinstance(s, G.Cylinder):
        c = 0.5 * (s.p0 + s.p1)
        return P.Cylinder(c.tolist(), s.axis.tolist(), s.radius, None if getattr(s, "through", False) else s.length)
    if isinstance(s, G.HalfSpace):
        return P.Plane(s.point.tolist(), s.normal.tolist())
    return P.Implicit(s.sdf, name=type(s).__name__.lower())


def _roles(ctx: BuildContext, preserve: Sequence[Any] | None) -> list[Any]:
    """preserve: [{role: keep | cut | bound, shape: <shape>, stock: 0, functional: null, name: null}]."""
    from toporlogy.post import primitives as P
    out = []
    for i, item in enumerate(preserve or []):
        path = join(join(ctx.path, "preserve"), i)
        if not isinstance(item, dict) or "role" not in item or "shape" not in item:
            raise ConfigError("expected {role: keep | cut | bound, shape: {...}, stock, functional, name}", path)
        bad = [k for k in item if k not in ("role", "shape", "stock", "functional", "name")]
        if bad:
            raise ConfigError(f"unknown key(s) {bad} (keys: role, shape, stock, functional, name)", path)
        cls = {"keep": P.Keep, "cut": P.Cut, "bound": P.Bound}.get(str(item["role"]).lower())
        if cls is None:
            raise ConfigError(f'role: keep | cut | bound, got "{item["role"]}"', join(path, "role"))
        errors: list[tuple[str, str]] = []
        spec = REGISTRY.resolve("shape", item["shape"], join(path, "shape"), errors)
        if errors:
            raise ConfigError(errors)
        shp = _post_shape(REGISTRY.build("shape", spec, ctx, join(path, "shape")))
        kw: dict[str, Any] = {"stock": float(item.get("stock") or 0.0), "name": str(item.get("name") or
                                                                                   f"{item['role']}{i + 1}")}
        if item.get("functional") is not None:
            kw["functional"] = bool(item["functional"])
        out.append(cls(shp, **kw))
    return out


def _signature_with(fn: Callable[..., Any], dc: type, skip: Sequence[str] = ()) -> Callable[..., Any]:
    import dataclasses
    sig = inspect.signature(fn)
    base = [p for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD]
    extra = []
    for f in dataclasses.fields(dc):
        if f.name in skip:
            continue
        dv = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
        if isinstance(dv, tuple):
            dv = list(dv)
        extra.append(inspect.Parameter(f.name, inspect.Parameter.KEYWORD_ONLY, default=dv, annotation=f.type))
    fn.__signature__ = sig.replace(parameters=[*base, *extra])  # type: ignore[attr-defined]
    return fn


def _make_finalize() -> Callable[..., Any]:
    from toporlogy.post import PostParams

    def finalize_step(ctx: BuildContext, preserve: list | None = None, name: str = "part", write: bool = True,
                      **params: Any) -> Any:
        """Optimised density -> smooth, watertight surface (toporlogy.post.finalize; parameters = PostParams);
        preserve: exact keep / cut / bound shapes with stock; writes <name>.stl and <name>_report.json (3-D)."""
        from toporlogy.post import export_result, finalize
        if ctx.domain.dim != 3:
            raise ValueError("finalize needs a 3-D domain")
        pp = PostParams(**params)
        roles = _roles(ctx, preserve)
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            xp, label = _design(state, res)
            pr = finalize(study.domain, xp, preserve=roles, params=pp)
            state["result"] = pr
            state["result_design"] = label
            rep = pr.report
            files = _rel(export_result(pr, folder, name, ("stl",), print_stock=False), folder) if write else {}
            topo = rep.get("topology_mesh", {})
            out = {"design": label, "triangles": int(len(pr.F)), "volume_mesh": rep.get("volume_mesh"),
                   "volume_vs_target": rep.get("volume_vs_target"), "topology": topo, "edges": rep.get("edges"),
                   "interfaces": rep.get("interfaces"), "seconds": rep.get("seconds", {}).get("total"),
                   "files": files}
            out["summary"] = (f"{out['triangles']} triangles, volume {out['volume_mesh']:.4g} "
                              f"({100 * out['volume_vs_target']:+.2f}% vs target), {topo.get('bodies', '?')} body"
                              + (f" -> {files['stl']}" if files else ""))
            return out
        return step

    return _signature_with(finalize_step, PostParams)


def _make_checks() -> Callable[..., Any]:
    from toporlogy.post import CheckThresholds

    def checks_step(ctx: BuildContext, **thresholds: Any) -> Any:
        """Manufacturing checks of the finalized part (watertight, bodies, voids, wall, overhang, fillet); gated
        values failing make the run exit 1. Needs a finalize step before it."""
        from toporlogy.post import manufacturing_checks
        th = dict(thresholds)
        th["build_direction"] = tuple(float(v) for v in th.get("build_direction", (0, 0, 1)))
        ct = CheckThresholds(**th)
        state = _post_state(ctx)

        def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
            pr = _need_result(state, "manufacturing_checks")
            ck = manufacturing_checks(pr, ct)
            state["checks"] = ck
            failed = [it["name"] for it in ck.items if it["passed"] is False]
            return {"passed": bool(ck.passed), "items": ck.items, "failed": failed,
                    "summary": ("all gated checks pass" if ck.passed else "FAILED: " + ", ".join(failed))
                    + f" ({sum(it['passed'] is not None for it in ck.items)} gated)"}
        return step

    return _signature_with(checks_step, CheckThresholds)


def _need_result(state: dict[str, Any], who: str) -> Any:
    pr = state.get("result")
    if pr is None:
        raise RuntimeError(f"{who} needs a finalize step before it")
    return pr


def export_step(ctx: BuildContext, formats: list = ("stl",), name: str = "part", print_stock: bool = True) -> Any:
    """Write the finalized surface (stl / ply / obj), the print variant (roles with stock) and <name>_report.json
    (pipeline report, checks) into the run folder. Needs a finalize step before it."""
    bad = [f for f in formats if f not in ("stl", "ply", "obj")]
    if bad:
        raise ValueError(f"formats: stl | ply | obj, got {bad}")
    state = _post_state(ctx)

    def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
        from toporlogy.post import export_result
        pr = _need_result(state, "export")
        ck = state.get("checks")
        files = _rel(export_result(pr, folder, name, tuple(formats), checks=ck, print_stock=print_stock,
                                   extra={"reverify": state.get("reverify")} if state.get("reverify") else None),
                     folder)
        return {"files": files, "summary": ", ".join(files.values())}
    return step


def reverify_step(ctx: BuildContext, E: float | None = None, nu: float | None = None, size: float | None = None,
                  surface_edge: float | None = None, timeout: float = 600.0, compare_voxel: bool = True) -> Any:
    """Re-verify the finalized part: gmsh T10 mesh + static solve with the config's supports and loads (as
    callables in physical coordinates); with compare_voxel the change against the voxel model. E / nu null =
    material. Needs a finalize step before it."""
    import importlib.util
    if importlib.util.find_spec("gmsh") is None:
        raise ValueError("reverify needs the gmsh package (not importable)")
    state = _post_state(ctx)
    E_ = float(ctx.material["E0"] if E is None else E)
    nu_ = float(ctx.material["nu"] if nu is None else nu)
    bcm = bc_mapping(ctx.bc)
    supports, cases = bcm["supports"], bcm["cases"]

    def step(res: Any, study: Any, folder: Any) -> dict[str, Any]:
        from toporlogy.post import reverify
        pr = _need_result(state, "reverify")
        xp, label = _design(state, res)                 # the voxel model of the same design that was finalized
        out = reverify(pr, supports, cases, E_, nu_, size=size, surface_edge=surface_edge,
                       domain=study.domain if compare_voxel else None, xphys=xp if compare_voxel else None,
                       timeout=timeout)
        out["design"] = state.get("result_design", label)
        # the T10 / voxel comparison is between two meshes under the SAME mapped conditions; whether those are the
        # optimisation's conditions is bc_fidelity ("equivalent" only when nothing was approximated or left out)
        out["notes"] = bcm["notes"]
        out["bc_fidelity"] = bcm["fidelity"]
        out["bc_mapping"] = bcm["items"]
        state["reverify"] = out
        ch = out.get("change") or []
        out["summary"] = (", ".join(f"{c['name']} compliance x{c['compliance_ratio']:.3f}" for c in ch)
                          or f"{len(out['t10']['cases'])} case(s) solved") + f" ({out['t10']['elements']} T10)"
        if bcm["fidelity"] != "equivalent":
            out["summary"] += f"; {bcm['fidelity']} BC mapping ({len(bcm['items'])} item(s), see bc_mapping)"
        return out
    return step


def bc_mapping(bcs: Any) -> dict[str, Any]:
    """Supports ``[(select, dirs)]`` and load cases ``{name: [(select, force)]}`` for
    :func:`toporlogy.post.reverify` from a :class:`~toporlogy.bc.BoundaryConditions` (the bc selectors as callables
    ``select(X (n, dim)) -> bool`` in physical coordinates), and how faithful that translation is.

    Returns ``{"supports", "cases", "items", "notes", "fidelity"}``. ``items``: one ``{"name", "fidelity",
    "note"}`` per support or load that is not carried over as it is: "approximate" (carried over as a different
    condition: pinned bore as fixed, oblique as fixed in all directions, bearing / remote force as an area-spread
    resultant, first case's supports for every case) or "unsupported" (left out: springs, body forces, pure
    moments, other support types). ``fidelity``: "unsupported" if any item is, else "approximate" if any item is,
    else "equivalent" (every support and load mapped by type with no known approximation; the nodal distribution
    on the tetrahedral mesh still differs from the voxel grid's). Only an "equivalent" re-analysis is the
    optimisation's problem."""
    from toporlogy import bc as B
    info, h = bcs.info, bcs.domain.h
    mapped: list[dict[str, str]] = []

    def note(name: str, fidelity: str, text: str) -> None:
        mapped.append({"name": str(name), "fidelity": fidelity, "note": text})
    sup_list = bcs.cases[0].supports if bcs.cases[0].supports is not None else bcs.supports
    if any(c.supports is not None for c in bcs.cases):
        note("per-case supports", "approximate", "the supports of the first case are used for every case")
    supports = []
    for s in sup_list:
        if isinstance(s, B.ElasticSupport):
            note(s.name, "unsupported", "springs are not re-verified (left free)")
            continue
        if isinstance(s, B.PinnedBore):
            supports.append((selector_callable(s.sel, info, h), (0, 1, 2)))
            note(s.name, "approximate", "pinned bore re-verified as a fixed bore")
            continue
        if isinstance(s, B.Fixed):
            rows = np.eye(info.dim) if s.dirs is None else np.array(
                [np.eye(info.dim)[int(r)] if np.isscalar(r) else np.asarray(r, float) / np.linalg.norm(r)
                 for r in s.dirs])
            ax = [int(np.argmax(np.abs(r))) for r in rows if np.isclose(np.abs(r).max(), 1.0)]
            if len(ax) != len(rows):
                note(s.name, "approximate", "oblique direction(s) re-verified as fixed in all directions")
                ax = [0, 1, 2]
            supports.append((selector_callable(s.where, info, h), tuple(sorted(set(ax)))))
            continue
        note(s.name, "unsupported", f"support type {type(s).__name__} not re-verified")
    if not supports:
        raise ValueError("reverify: no support can be re-verified")
    cases: dict[str, list[Any]] = {}
    for c in bcs.cases:
        items = []
        for ld, a in zip(c.loads, bcs.applied[c.name], strict=True):
            F = np.asarray(a.force, float)
            if isinstance(ld, B.BodyForce) or not np.linalg.norm(F) > 0:
                note(a.name, "unsupported", f"{'body force' if isinstance(ld, B.BodyForce) else 'pure moment'} not "
                                             "re-verified")
                continue
            if isinstance(ld, B.PointLoad):
                sel = B.InSphere(ld.point, 1.5 * h if ld.radius is None else float(ld.radius))
            else:
                sel = getattr(ld, "where", None) or getattr(ld, "sel", None)
            if sel is None:
                note(a.name, "unsupported", "no selection, not re-verified")
                continue
            if isinstance(ld, B.BearingLoad | B.RemoteForce):
                note(a.name, "approximate", "re-verified as the resultant spread by area (no cosine / moment)")
            items.append((selector_callable(sel, info, h), F))
        if items:
            cases[c.name] = items
    if not cases:
        raise ValueError("reverify: no load can be re-verified")
    levels = {it["fidelity"] for it in mapped}
    fidelity = "unsupported" if "unsupported" in levels else "approximate" if levels else "equivalent"
    return {"supports": supports, "cases": cases, "items": mapped,
            "notes": [f"{it['name']}: {it['note']}" for it in mapped], "fidelity": fidelity}


def bc_callables(bcs: Any) -> tuple[list[Any], dict[str, list[Any]], list[str]]:
    """``(supports, cases, notes)`` of :func:`bc_mapping` (kept for callers that need no fidelity)."""
    m = bc_mapping(bcs)
    return m["supports"], m["cases"], m["notes"]


def selector_callable(sel: Any, info: Any, h: float) -> Callable[[np.ndarray], np.ndarray]:
    """``select(X) -> bool`` in physical coordinates for a bc selector: geometric selectors keep their geometry with a
    tolerance of h/2 (the finished surface moves by up to about that), others select points within h of the nodes
    they pick on the grid."""
    from scipy.spatial import cKDTree

    from toporlogy import bc as B
    t = 0.5 * h
    if isinstance(sel, B.InBox):
        lo, hi = sel.lo, sel.hi
        return lambda X: np.all((X - lo >= -t) & (hi - X >= -t), axis=1)        # bounds may be infinite
    if isinstance(sel, B.OnPlane):
        p, n = sel.point, sel.normal
        tol = max(t, 0.5 * h if sel.tol is None else float(sel.tol))
        return lambda X: np.abs((X - p) @ n) <= tol
    if isinstance(sel, B.HalfSpaceNodes):
        p, n = sel.point, sel.normal
        return lambda X: (X - p) @ n <= t
    if isinstance(sel, B.InSphere):
        c, r = sel.center, sel.radius
        return lambda X: np.linalg.norm(X - c, axis=1) <= r + t
    if isinstance(sel, B.InShape):
        shp = sel.shape
        return lambda X: shp.sdf(X) <= t
    if isinstance(sel, B.OnCylinder):
        bi, bo = sel.bands(h)

        def cyl(X: np.ndarray) -> np.ndarray:
            tt, r, _ = sel.coords(X)
            ok = (r >= sel.radius - bi) & (r <= sel.radius + bo)
            if sel.axis is not None:
                ok &= (tt >= -sel.extend - t) & (tt <= sel.length + sel.extend + t)
            return ok
        return cyl
    if isinstance(sel, B.AllNodes | B.SurfaceNodes):
        return lambda X: np.ones(len(X), bool)
    op, a, b = getattr(sel, "op", None), getattr(sel, "a", None), getattr(sel, "b", None)
    if op in ("|", "&", "-") and isinstance(a, B.Selector) and isinstance(b, B.Selector):
        fa, fb = selector_callable(a, info, h), selector_callable(b, info, h)
        if op == "|":
            return lambda X: fa(X) | fb(X)
        if op == "&":
            return lambda X: fa(X) & fb(X)
        return lambda X: fa(X) & ~fb(X)
    nodes = sel.select(info)
    tree = cKDTree(info.X[nodes])
    return lambda X: np.isfinite(tree.query(X, distance_upper_bound=h * (1 + 1e-9))[0])


def __getattr__(name: str) -> Any:                     # PEP 562: lazy post-step constructors
    makers = {"finalize_step": _make_finalize, "checks_step": _make_checks}
    if name in makers:
        if name not in _LAZY:
            _LAZY[name] = makers[name]()
        return _LAZY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_register_core()
_register_features()
