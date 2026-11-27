"""Material cards: a plain (YAML-ready) dict -> one validated physical material model, with units, provenance and a
validity box (WP11 P05; ``docs/foundations/MATERIAL_CARDS.md``, ``docs/foundations/prereg/P05_material_cards.md``).

:func:`load_material_card` takes ``{name, model, units, parameters, provenance, validity}`` and returns a
:class:`MaterialCard` (``.model`` the built model, ``.provenance``, ``.validity`` a
:class:`~toporlogy.optimization.material_identification.ValidityBox` in SI, ``.check(...)``, ``.to_dict()``). Every
problem raises ``ValueError`` whose message starts with the dotted path of the offending key (``parameters.elasticity.E:
...``), the style of :mod:`toporlogy.configuration`.

* Units are never implicit. ``units`` declares one unit per kind used (stress Pa/kPa/MPa/GPa, time s/min/h, temperature
  K, rate 1/s, 1/min, 1/h); the factors are those of
  :data:`~toporlogy.optimization.material_identification.UNIT_FACTORS` and are applied exactly once, when the model
  is built. Celsius/Fahrenheit (as a unit, a unit key, a negative temperature or a string such as ``"25C"``) are refused:
  an offset is not a factor. Fixed-unit quantities are not selectable: dimensionless numbers, activation energy J/mol,
  CTE 1/K, C2 of the WLF law K.
* Isotropic elasticity is exactly two of E, nu, K, G. A third modulus is refused, whether it agrees with the pair or not
  (the message says which).
* ``provenance``: ``source``, ``synthetic`` (bool), ``date`` (YYYY-MM-DD) and, when ``synthetic`` is false,
  ``license``. Nothing is inferred about a real material: the module checks that provenance is stated, not that it is
  true.
* ``validity``: closed ranges ``temperature`` [K], ``time``, ``strain_rate`` (box coordinate ``rate``), ``stress`` (a
  signed range; its largest absolute value is the amplitude bound). Each model requires the ranges it needs.
* Two unit sanity checks run when a stress range is declared: **U1** the stress amplitude bound must not exceed the
  smallest Young modulus of the card; **U2** each strength scale (yield_stress, b0, sigma_ref) divided by the
  amplitude bound must lie in [1e-3, 1e3]. They catch a modulus or a strength labelled with the wrong unit while the
  range is right; a card whose every stress number carries the same wrong label is self-consistent and not caught.

All numbers in this repository's cards are synthetic."""
from __future__ import annotations

import copy
import datetime
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toporlogy.core.creep import ArrheniusFactor, NortonBaileyCreep, NortonCreep
from toporlogy.core.cure import (Arrhenius, ChemicalShrinkage, CuringThermoElasticPoint, KamalSourour, NthOrder)
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.core.properties import (Constant, IsotropicModuliOfT, PiecewiseLinear, PropertyFunction,
                                       ThermalExpansion, ThermoElasticPoint)
from toporlogy.core.viscoelastic import ArrheniusShift, GeneralizedMaxwell, PronySeries, WLFShift
from toporlogy.core.viscoplastic import DruckerPrager, Perzyna, PhysicalJ2
from toporlogy.optimization.material_identification import UNIT_FACTORS, ValidityBox, ValidityBoxError

__all__ = ["MATERIAL_MODELS", "STRENGTH_RATIO_BOUNDS", "MaterialCard", "load_material_card"]

MATERIAL_MODELS = ("thermoelastic", "generalized_maxwell", "norton", "norton_bailey", "physical_j2", "perzyna",
                   "drucker_prager", "curing_thermoelastic")
STRENGTH_RATIO_BOUNDS = (1e-3, 1e3)                  # U2: strength scale / stress amplitude bound
_TOP_KEYS = ("name", "model", "units", "parameters", "provenance", "validity")
_VALIDITY_KEYS = {"temperature": ("temperature", "temperature"), "time": ("time", "time"),
                  "strain_rate": ("rate", "rate"), "stress": ("stress", "stress")}      # key -> (unit kind, coordinate)
_CELSIUS = ("c", "degc", "°c", "celsius", "degf", "°f", "f", "fahrenheit", "delta_degc")
_FIXED = {"1": "dimensionless", "J/mol": "J/mol", "1/K": "1/K"}                         # fixed-unit kinds
_MODULI = ("E", "nu", "K", "G")
_UNSET: Any = object()


def _fail(path: str, msg: str) -> Any:
    raise ValueError(f"{path}: {msg}" if path else msg)


def _celsius(text: Any) -> bool:
    return isinstance(text, str) and text.strip().lower() in _CELSIUS


_CELSIUS_HINT = ("Celsius and Fahrenheit are refused: temperatures are absolute kelvin (an offset is not a unit factor; "
                 "convert yourself and say so in the provenance)")


def _number(path: str, v: Any, sign: str, kind: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        hint = (" (a temperature is a plain number in kelvin; strings such as '25C' are ambiguous and refused)"
                if kind == "temperature" else "")
        if isinstance(v, str) and re.fullmatch(r"[-+]?\d+(\.\d*)?[eE]\d+", v.strip()):
            hint += " (YAML 1.1 reads an exponent without a sign as a string: write 5.0e+6, not 5.0e6)"
        _fail(path, f"expected a number, got {v!r}{hint}")
    x = float(v)
    if not math.isfinite(x):
        _fail(path, f"must be finite, got {v!r}")
    if kind == "temperature" and x < 0.0:
        _fail(path, f"{x:g} is negative: {_CELSIUS_HINT}")
    if sign == "pos" and not x > 0.0:
        _fail(path, f"must be > 0, got {x:g}")
    if sign == "nonneg" and not x >= 0.0:
        _fail(path, f"must be >= 0, got {x:g}")
    return x


class _Ctx:
    def __init__(self, name: str, units: dict[str, str]) -> None:
        self.name, self.units = name, units

    def factor(self, kind: str, path: str) -> float:
        if kind in _FIXED:
            return 1.0
        unit = self.units.get(kind)
        if unit is None:
            _fail(path, f"needs units.{kind}, which is not declared (no implicit units)")
        return UNIT_FACTORS[kind][unit]


class _Block:
    """A mapping being read: every ``take`` records the key as known; :meth:`done` refuses the rest."""

    def __init__(self, ctx: _Ctx, path: str, raw: Any) -> None:
        if not isinstance(raw, Mapping):
            _fail(path, f"must be a mapping, got {type(raw).__name__}")
        self.ctx, self.path, self.raw = ctx, path, dict(raw)
        self.out: dict[str, Any] = {}
        self.seen: set[str] = set()
        self.children: list[_Block] = []

    def p(self, key: str) -> str:
        return f"{self.path}.{key}" if self.path else key

    def has(self, key: str) -> bool:
        self.seen.add(key)
        return key in self.raw

    def _raw(self, key: str, required: bool) -> Any:
        self.seen.add(key)
        if key not in self.raw:
            if required:
                _fail(self.p(key), "missing required key")
            return _UNSET
        return self.raw[key]

    def number(self, key: str, kind: str, sign: str = "pos", required: bool = True,
               bounds: tuple[float, float, bool] | None = None) -> float | None:
        """The value in SI (None when optional and absent); the card-unit value is kept for ``to_dict``."""
        v = self._raw(key, required)
        if v is _UNSET:
            return None
        x = _number(self.p(key), v, sign, kind)
        if bounds is not None and not (bounds[0] <= x < bounds[1] or (bounds[2] and x == bounds[1])):
            _fail(self.p(key), f"must lie in [{bounds[0]:g}, {bounds[1]:g}{']' if bounds[2] else ')'}, got {x:g}")
        self.out[key] = x
        return x * self.ctx.factor(kind, self.p(key))

    def integer(self, key: str, minimum: int, required: bool = False) -> int | None:
        v = self._raw(key, required)
        if v is _UNSET:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
            _fail(self.p(key), f"must be an integer >= {minimum}, got {v!r}")
        self.out[key] = int(v)
        return int(v)

    def array(self, key: str, kind: str, sign: str, required: bool = True) -> np.ndarray | None:
        v = self._raw(key, required)
        if v is _UNSET:
            return None
        if isinstance(v, (str, bytes)) or not hasattr(v, "__len__") or len(v) == 0:
            _fail(self.p(key), f"must be a non-empty list of numbers, got {v!r}")
        xs = [_number(f"{self.p(key)}[{i}]", x, sign, kind) for i, x in enumerate(v)]
        self.out[key] = xs
        return np.array(xs) * self.ctx.factor(kind, self.p(key))

    def choice(self, key: str, options: tuple[str, ...], required: bool = True) -> str | None:
        v = self._raw(key, required)
        if v is _UNSET:
            return None
        if v not in options:
            _fail(self.p(key), f"must be one of {options}, got {v!r}")
        self.out[key] = v
        return str(v)

    def sub(self, key: str, required: bool = True) -> _Block | None:
        v = self._raw(key, required)
        if v is _UNSET:
            return None
        child = _Block(self.ctx, self.p(key), v)
        self.out[key] = child.out
        self.children.append(child)
        return child

    def done(self) -> None:
        for k in self.raw:
            if k not in self.seen:
                _fail(self.p(k), f"unknown key (known here: {sorted(self.seen)})")
        for c in self.children:
            c.done()


def _wrap(path: str, fn: Callable[[], Any]) -> Any:
    """Run a library constructor; its ValueError is re-raised with the path of the block."""
    try:
        return fn()
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


# ------------------------------------------------------------------------------------------------ property values
def _property(b: _Block, key: str, kind: str, sign: str, tdom: tuple[float, float]) -> PropertyFunction:
    """A scalar (constant over the validity temperature range) or a ``{T: [...], values: [...]}`` piecewise-linear
    table; ``kind`` is a unit kind or a fixed unit ("1", "1/K")."""
    raw = b._raw(key, True)
    path = b.p(key)
    units = {"stress": "Pa", "1": "1", "1/K": "1/K"}[kind]
    source = f"material card {b.ctx.name}"
    if isinstance(raw, Mapping):
        t = _Block(b.ctx, path, raw)
        T = t.array("T", "temperature", "pos")
        vals = t.array("values", kind, sign)
        t.done()
        b.out[key] = t.out
        assert T is not None and vals is not None
        if T[0] > tdom[0] or T[-1] < tdom[1]:
            _fail(path, f"table temperatures [{T[0]:g}, {T[-1]:g}] K do not cover validity.temperature "
                        f"[{tdom[0]:g}, {tdom[1]:g}] K")
        return _wrap(path, lambda: PiecewiseLinear(T, vals, units, source, positive=(sign == "pos")))
    x = b.number(key, kind, sign)
    assert x is not None
    return _wrap(path, lambda: Constant(x, units, source, domain=tdom, positive=(sign == "pos")))


# ------------------------------------------------------------------------------------------------ elasticity
@dataclass
class _Info:
    """What a builder reports besides the model: smallest Young modulus [Pa], strength scales [Pa], extra ranges."""

    e_min: float | None = None
    strengths: dict[str, float] = field(default_factory=dict)
    needs: set[str] = field(default_factory=set)


def _read_pair(b: _Block, scalar: bool, tdom: tuple[float, float] | None) -> dict[str, Any]:
    present = [k for k in _MODULI if b.has(k)]
    if len(present) < 2:
        _fail(b.path, f"give exactly two of E, nu, K, G, got {present}")
    vals: dict[str, Any] = {}
    for k in present:
        if k == "nu":
            if scalar:
                nu = b.number("nu", "1", "any")
                assert nu is not None
                if not -1.0 < nu < 0.5:
                    _fail(b.p("nu"), f"must lie in (-1, 0.5), got {nu:g}")
                vals[k] = nu
            else:
                assert tdom is not None
                vals[k] = _property(b, "nu", "1", "any", tdom)
        elif scalar:
            vals[k] = b.number(k, "stress", "pos")
        else:
            assert tdom is not None
            vals[k] = _property(b, k, "stress", "pos", tdom)
    if len(present) > 2:
        first, second = present[0], present[1]
        if scalar:
            full = _wrap(b.path, lambda: isotropic_moduli(**{first: vals[first], second: vals[second]}))
            implied = {"E": full.E, "nu": full.nu, "K": full.K, "G": full.G}
            for extra in present[2:]:
                got, want = vals[extra], implied[extra]
                if abs(got - want) > 1e-9 * max(abs(want), 1e-300):
                    _fail(b.path, f"redundant moduli {present}: {extra} = {got:g} is inconsistent with {first}, {second} "
                                  f"(they imply {extra} = {want:g}); give exactly two")
        _fail(b.path, f"redundant moduli {present}: give exactly two of E, nu, K, G (the extra one is not needed)")
    return vals


def _elastic_scalar(parent: _Block, key: str, info: _Info) -> tuple[float, float]:
    b = parent.sub(key)
    assert b is not None
    vals = _read_pair(b, True, None)
    m = _wrap(b.path, lambda: isotropic_moduli(**vals))
    info.e_min = m.E if info.e_min is None else min(info.e_min, m.E)
    return m.K, m.G


def _elastic_of_T(parent: _Block, key: str, tdom: tuple[float, float], info: _Info) -> IsotropicModuliOfT:
    b = parent.sub(key)
    assert b is not None
    funcs = _read_pair(b, False, tdom)
    mod = _wrap(b.path, lambda: IsotropicModuliOfT(**funcs))
    e = float(np.min(mod.moduli(np.linspace(tdom[0], tdom[1], 33)).E))
    info.e_min = e if info.e_min is None else min(info.e_min, e)
    return mod


# ------------------------------------------------------------------------------------------------ builders
def _build_thermoelastic(b: _Block, tdom: tuple[float, float], info: _Info) -> Any:
    moduli = _elastic_of_T(b, "elasticity", tdom, info)
    alpha = _property(b, "alpha", "1/K", "any", tdom)
    t_ref = b.number("T_ref", "temperature")
    assert t_ref is not None
    return _wrap(b.path, lambda: ThermoElasticPoint(moduli, alpha, t_ref))


def _prony(parent: _Block, key: str) -> PronySeries:
    b = parent.sub(key)
    assert b is not None
    inf = b.number("inf", "stress", "nonneg")
    amp = b.array("amplitudes", "stress", "nonneg")
    taus = b.array("taus", "time", "pos")
    assert inf is not None and amp is not None and taus is not None
    return _wrap(b.path, lambda: PronySeries(inf, amp, taus))


def _build_generalized_maxwell(b: _Block, tdom: tuple[float, float] | None, info: _Info) -> Any:
    shear, bulk = _prony(b, "shear"), _prony(b, "bulk")
    shift = None
    s = b.sub("shift", required=False)
    if s is not None:
        info.needs.add("temperature")
        kind = s.choice("type", ("wlf", "arrhenius"))
        t_ref = s.number("T_ref", "temperature")
        assert t_ref is not None
        if kind == "wlf":
            c1, c2 = s.number("C1", "1"), s.number("C2", "temperature")      # C2 is a temperature DIFFERENCE in K
            assert c1 is not None and c2 is not None
            shift = _wrap(s.path, lambda: WLFShift(t_ref, c1, c2))
        else:
            ea = s.number("Ea", "J/mol")
            assert ea is not None
            shift = _wrap(s.path, lambda: ArrheniusShift(t_ref, ea))
    k0, g0 = bulk.instantaneous, shear.instantaneous
    info.e_min = 9.0 * k0 * g0 / (3.0 * k0 + g0)
    return _wrap(b.path, lambda: GeneralizedMaxwell(shear, bulk, shift))


def _build_norton(b: _Block, tdom: tuple[float, float] | None, info: _Info) -> Any:
    K, G = _elastic_scalar(b, "elasticity", info)
    A = b.number("A", "rate", "nonneg")
    sref = b.number("sigma_ref", "stress")
    n = b.number("n", "1")
    arr = None
    a = b.sub("arrhenius", required=False)
    if a is not None:
        info.needs.add("temperature")
        q, t_ref = a.number("Q", "J/mol", "nonneg"), a.number("T_ref", "temperature")
        assert q is not None and t_ref is not None
        arr = _wrap(a.path, lambda: ArrheniusFactor(q, t_ref))
    assert A is not None and sref is not None and n is not None
    info.strengths["sigma_ref"] = sref
    return _wrap(b.path, lambda: NortonCreep(K, G, A, sref, n, arrhenius=arr))


def _build_norton_bailey(b: _Block, tdom: tuple[float, float] | None, info: _Info) -> Any:
    K, G = _elastic_scalar(b, "elasticity", info)
    B = b.number("B", "1", "nonneg")
    sref, n, m = b.number("sigma_ref", "stress"), b.number("n", "1"), b.number("m", "1")
    t_ref = b.number("t_ref", "time")
    hardening = b.choice("hardening", ("time", "strain"), required=False) or "time"
    assert B is not None and sref is not None and n is not None and m is not None and t_ref is not None
    info.strengths["sigma_ref"] = sref
    return _wrap(b.path, lambda: NortonBaileyCreep(K, G, B, sref, n, m, t_ref, hardening=hardening))


def _build_physical_j2(b: _Block, tdom: tuple[float, float] | None, info: _Info) -> Any:
    K, G = _elastic_scalar(b, "elasticity", info)
    sy = b.number("yield_stress", "stress")
    H = b.number("hardening", "stress", "nonneg", required=False)
    s_inf = b.number("saturation_stress", "stress", required=False)
    delta = b.number("saturation_rate", "1", required=False)
    assert sy is not None
    info.strengths["yield_stress"] = sy
    kw: dict[str, Any] = {}
    if H is not None:
        kw["hardening"] = H
    if s_inf is not None:
        kw["saturation_stress"] = s_inf
    if delta is not None:
        kw["saturation_rate"] = delta
    return _wrap(b.path, lambda: PhysicalJ2(K, G, sy, **kw))


def _build_perzyna(b: _Block, tdom: tuple[float, float] | None, info: _Info) -> Any:
    K, G = _elastic_scalar(b, "elasticity", info)
    sy, H = b.number("yield_stress", "stress"), b.number("hardening", "stress", "nonneg")
    tau0, sref, m = b.number("tau0", "time"), b.number("sigma_ref", "stress"), b.number("m", "1")
    assert sy is not None and H is not None and tau0 is not None and sref is not None and m is not None
    info.strengths.update(yield_stress=sy, sigma_ref=sref)
    return _wrap(b.path, lambda: Perzyna(K, G, sy, H, tau0, sref, m))


def _build_drucker_prager(b: _Block, tdom: tuple[float, float] | None, info: _Info) -> Any:
    K, G = _elastic_scalar(b, "elasticity", info)
    a, b0 = b.number("a", "1"), b.number("b0", "stress")
    H = b.number("hardening", "stress", "nonneg", required=False)
    a_flow = b.number("a_flow", "1", required=False)
    assert a is not None and b0 is not None
    info.strengths["b0"] = b0
    return _wrap(b.path, lambda: DruckerPrager(K, G, a, b0, 0.0 if H is None else H, a_flow))


def _arrhenius_block(parent: _Block, key: str) -> Arrhenius:
    b = parent.sub(key)
    assert b is not None
    A, Q = b.number("A", "rate", "nonneg"), b.number("Q", "J/mol", "nonneg")
    assert A is not None and Q is not None
    return _wrap(b.path, lambda: Arrhenius(A, Q))


def _build_curing(b: _Block, tdom: tuple[float, float], info: _Info) -> Any:
    kb = b.sub("kinetics")
    assert kb is not None
    ktype = kb.choice("type", ("nth_order", "kamal_sourour"))
    if ktype == "nth_order":
        arr = _arrhenius_fields(kb)
        n = kb.number("n", "1")
        assert n is not None
        kinetics = _wrap(kb.path, lambda: NthOrder(arr, n))
    else:
        k1, k2 = _arrhenius_block(kb, "k1"), _arrhenius_block(kb, "k2")
        m, n = kb.number("m", "1", "nonneg"), kb.number("n", "1")
        assert m is not None and n is not None
        kinetics = _wrap(kb.path, lambda: KamalSourour(k1, k2, m, n))
    mu = _elastic_of_T(b, "moduli_uncured", tdom, info)
    mc = _elastic_of_T(b, "moduli_cured", tdom, info)
    alpha = _property(b, "alpha", "1/K", "any", tdom)
    t_ref = b.number("T_ref", "temperature")
    beta = b.number("beta", "1", "any")
    gel = b.number("alpha_gel", "1", "nonneg", required=False, bounds=(0.0, 1.0, False))
    a0 = b.number("alpha0", "1", "nonneg", required=False, bounds=(0.0, 1.0, True))
    substeps = b.integer("substeps", 1)
    assert t_ref is not None and beta is not None
    expansion = _wrap(b.path, lambda: ThermalExpansion(alpha, t_ref))
    shrink = _wrap(b.path, lambda: ChemicalShrinkage(beta, 0.0 if gel is None else gel))
    kw: dict[str, Any] = {}
    if a0 is not None:
        kw["alpha0"] = a0
    if substeps is not None:
        kw["substeps"] = substeps
    return _wrap(b.path, lambda: CuringThermoElasticPoint(kinetics, mu, mc, expansion, shrink, **kw))


def _arrhenius_fields(kb: _Block) -> Arrhenius:
    A, Q = kb.number("A", "rate", "nonneg"), kb.number("Q", "J/mol", "nonneg")
    assert A is not None and Q is not None
    return _wrap(kb.path, lambda: Arrhenius(A, Q))


# model -> (builder, validity ranges always required, builder needs the temperature range as its property domain)
_MODELS: dict[str, tuple[Callable[..., Any], tuple[str, ...], bool]] = {
    "thermoelastic": (_build_thermoelastic, ("temperature",), True),
    "generalized_maxwell": (_build_generalized_maxwell, ("time",), False),
    "norton": (_build_norton, ("stress",), False),
    "norton_bailey": (_build_norton_bailey, ("stress", "time"), False),
    "physical_j2": (_build_physical_j2, ("stress",), False),
    "perzyna": (_build_perzyna, ("stress", "strain_rate"), False),
    "drucker_prager": (_build_drucker_prager, ("stress",), False),
    "curing_thermoelastic": (_build_curing, ("temperature", "time"), True),
}


# ------------------------------------------------------------------------------------------------ card
@dataclass(frozen=True, eq=False)
class MaterialCard:
    """A validated card: ``model`` (the built physical model, SI), ``kind`` (its card name), ``provenance`` (a dict
    copy), ``validity`` (a :class:`ValidityBox` in SI over ``temperature`` K, ``time`` s, ``rate`` 1/s, ``stress``
    Pa, only the declared coordinates), ``units`` as declared. :meth:`check` raises outside the box;
    :meth:`to_dict` returns the card in its declared units (a deep copy that loads to the same card)."""

    name: str
    kind: str
    model: Any
    provenance: dict[str, Any]
    validity: ValidityBox
    units: dict[str, str]
    _spec: dict[str, Any]

    @property
    def synthetic(self) -> bool:
        return bool(self.provenance["synthetic"])

    def check(self, *, temperature: Any = None, time: Any = None, rate: Any = None, stress: Any = None) -> None:
        """Raise :class:`ValidityBoxError` (a ValueError) unless every given coordinate, in SI (K, s, 1/s, Pa), lies in
        the declared closed range; a coordinate the card does not declare raises ValueError."""
        coords = {k: v for k, v in (("temperature", temperature), ("time", time), ("rate", rate), ("stress", stress))
                  if v is not None}
        if not coords:
            raise ValueError("check: give at least one of temperature, time, rate, stress (SI units)")
        try:
            self.validity.check(**coords)
        except ValidityBoxError as e:
            raise ValidityBoxError(f"material card {self.name!r}: {e}") from None
        except ValueError as e:
            raise ValueError(f"material card {self.name!r}: {e}") from None

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._spec)


def _provenance(ctx: _Ctx, raw: Any) -> tuple[_Block, dict[str, Any]]:
    b = _Block(ctx, "provenance", raw)
    synthetic_raw = b._raw("synthetic", True)
    if not isinstance(synthetic_raw, bool):
        _fail("provenance.synthetic", f"must be true or false (a boolean), got {synthetic_raw!r}")
    src = b._raw("source", True)
    if not (isinstance(src, str) and src.strip()):
        _fail("provenance.source", f"must be a non-empty string, got {src!r}")
    if not synthetic_raw and src.strip().lower() == "synthetic":
        _fail("provenance.source", "a card with synthetic: false must name its real source, not 'synthetic'")
    date = b._raw("date", True)
    if isinstance(date, datetime.datetime):
        _fail("provenance.date", f"must be a date YYYY-MM-DD, got {date!r}")
    if isinstance(date, datetime.date):
        date = date.isoformat()
    if not (isinstance(date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)):
        _fail("provenance.date", f"must be a date YYYY-MM-DD, got {date!r}")
    try:
        datetime.date.fromisoformat(date)
    except ValueError:
        _fail("provenance.date", f"is not a calendar date: {date!r}")
    out: dict[str, Any] = {"source": src.strip(), "synthetic": synthetic_raw, "date": date}
    lic = b._raw("license", not synthetic_raw)
    if lic is not _UNSET:
        if not (isinstance(lic, str) and lic.strip()):
            _fail("provenance.license", f"must be a non-empty string, got {lic!r}")
        out["license"] = lic.strip()
    b.done()
    b.out.update(out)
    return b, out


def _read_units(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        _fail("units", f"must be a mapping of unit kind to unit, got {type(raw).__name__}")
    out: dict[str, str] = {}
    for k, v in raw.items():
        path = f"units.{k}"
        if _celsius(k):
            _fail(path, f"unit kind {k!r}: {_CELSIUS_HINT}")
        if k not in UNIT_FACTORS:
            _fail(path, f"unknown key (unit kinds: {tuple(UNIT_FACTORS)})")
        if _celsius(v):
            _fail(path, f"unit {v!r}: {_CELSIUS_HINT}")
        if not isinstance(v, str) or v not in UNIT_FACTORS[k]:
            _fail(path, f"unit {v!r} is not a {k} unit {tuple(UNIT_FACTORS[k])}")
        out[k] = v
    return out


def _read_validity(raw: Any) -> dict[str, tuple[float, float]]:
    """Structural read, card units; unit conversion happens later (it needs the declared units)."""
    if not isinstance(raw, Mapping):
        _fail("validity", f"must be a mapping, got {type(raw).__name__}")
    out: dict[str, tuple[float, float]] = {}
    for k, v in raw.items():
        path = f"validity.{k}"
        if k not in _VALIDITY_KEYS:
            _fail(path, f"unknown key (known: {tuple(_VALIDITY_KEYS)})")
        if isinstance(v, (str, bytes)) or not hasattr(v, "__len__") or len(v) != 2:
            _fail(path, f"must be a [lo, hi] pair, got {v!r}")
        kind = _VALIDITY_KEYS[k][0]
        sign = "any" if k == "stress" else "nonneg"
        lo, hi = (_number(f"{path}[{i}]", x, sign, kind) for i, x in enumerate(v))
        if not lo <= hi:
            _fail(path, f"need lo <= hi, got [{lo:g}, {hi:g}]")
        out[k] = (lo, hi)
    return out


def load_material_card(card: Mapping[str, Any]) -> MaterialCard:
    """Validate ``card`` (a plain dict, e.g. from ``yaml.safe_load``) and build its model; see the module notes.
    Raises ``ValueError`` with the dotted path of the first offending key."""
    if not isinstance(card, Mapping):
        _fail("card", f"must be a mapping, got {type(card).__name__}")
    for k in _TOP_KEYS:
        if k not in card:
            _fail(k, "missing required key")
    for k in card:
        if k not in _TOP_KEYS:
            _fail(str(k), f"unknown key (a card has exactly {_TOP_KEYS})")
    name = card["name"]
    if not (isinstance(name, str) and name.strip()):
        _fail("name", f"must be a non-empty string, got {name!r}")
    kind = card["model"]
    if kind not in _MODELS:
        _fail("model", f"{kind!r} is not a known model {MATERIAL_MODELS}")
    units = _read_units(card["units"])
    ctx = _Ctx(name.strip(), units)
    prov_block, prov = _provenance(ctx, card["provenance"])
    ranges = _read_validity(card["validity"])
    builder, required, needs_tdom = _MODELS[kind]
    for k in required:
        if k not in ranges:
            _fail(f"validity.{k}", f"required by model {kind!r}")
    tdom = ranges.get("temperature")
    if needs_tdom:
        assert tdom is not None
        if not tdom[0] < tdom[1]:
            _fail("validity.temperature", f"model {kind!r} needs lo < hi, got [{tdom[0]:g}, {tdom[1]:g}] K")
    params = _Block(ctx, "parameters", card["parameters"])
    info = _Info()
    model = builder(params, tdom, info)
    params.done()
    for k in info.needs:
        if k not in ranges:
            _fail(f"validity.{k}", f"required by the {kind!r} parameters given (temperature-dependent law)")
    # convert the validity to SI, once, by the declared units
    box: dict[str, tuple[float, float]] = {}
    spec_validity: dict[str, list[float]] = {}
    for k, (lo, hi) in ranges.items():
        ukind, coord = _VALIDITY_KEYS[k]
        f = ctx.factor(ukind, f"validity.{k}")
        box[coord] = (lo * f, hi * f)
        spec_validity[k] = [lo, hi]
    if "stress" in box:
        amp = max(abs(box["stress"][0]), abs(box["stress"][1]))
        if info.e_min is not None and amp > info.e_min:
            _fail("validity.stress", f"amplitude bound {amp:.6g} Pa exceeds the smallest Young modulus "
                                     f"{info.e_min:.6g} Pa of this card (strain > 100 %): a stress or modulus unit is "
                                     "probably mislabelled (check U1)")
        if amp > 0.0:
            lo_r, hi_r = STRENGTH_RATIO_BOUNDS
            for sname, s in info.strengths.items():
                if not lo_r <= s / amp <= hi_r:
                    _fail(f"parameters.{sname}", f"{s:.6g} Pa against the stress amplitude bound {amp:.6g} Pa of "
                                                f"validity.stress is a ratio {s / amp:.3g} outside [{lo_r:g}, {hi_r:g}]: "
                                                "a stress unit is probably mislabelled (check U2)")
    spec = {"name": name.strip(), "model": kind, "units": dict(units), "parameters": params.out,
            "provenance": prov_block.out, "validity": spec_validity}
    return MaterialCard(name.strip(), kind, model, dict(prov), ValidityBox(box), dict(units), copy.deepcopy(spec))
