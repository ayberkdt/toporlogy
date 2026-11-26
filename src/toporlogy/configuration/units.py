"""Unit-checked values in a config: ``units: {length: mm, force: N}`` and values such as ``"210 GPa"`` or ``"-1 kN"``.

Without a top-level ``units`` section nothing changes: every value is a plain number in one consistent set of units
chosen by the user (the library never assumes SI). With it, a parameter whose unit kind is declared (``Param.unit``:
``length``, ``force``, ``force/length^2``, ``mass/length^3``, ``length/time^2``, ``Hz``, …) may be given as text
with a unit; the value is converted by dimensional analysis into the declared system, whose time unit is the second
and whose mass unit follows from force · s² / length (mm, N → tonne). A value whose dimension differs from the
parameter's (a mass for a force, a force for a modulus) is refused with both dimensions named.

Temperatures: ``K`` is a temperature and a temperature difference; ``degC`` is an absolute temperature only and
``delta_degC`` a difference only (an absolute temperature cannot be used as a difference: 20 °C is not 20 K of
difference). Pre-registered checks: ``docs/olcumler/birimler_onkayit.md`` (U1-U8).
"""
from __future__ import annotations

import contextvars
import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = ["UnitSystem", "UnitError", "is_quantity", "parse_quantity", "kind_dimension", "dimension_name", "convert",
           "active", "use", "SCHEMA_VERSIONS", "check_schema_version"]

SCHEMA_VERSIONS = ("1",)
Dim = tuple[int, int, int, int]                       # powers of length, mass, time, temperature

_L, _M, _T, _K = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
_FORCE: Dim = (1, 1, -2, 0)
_PRESSURE: Dim = (-1, 1, -2, 0)
# name -> (SI factor, dimension)
_UNITS: dict[str, tuple[float, Dim]] = {
    "m": (1.0, _L), "mm": (1e-3, _L), "cm": (1e-2, _L), "um": (1e-6, _L), "km": (1e3, _L), "in": (0.0254, _L),
    "ft": (0.3048, _L),
    "kg": (1.0, _M), "g": (1e-3, _M), "t": (1e3, _M), "lb": (0.45359237, _M),
    "s": (1.0, _T), "ms": (1e-3, _T), "min": (60.0, _T), "h": (3600.0, _T),
    "N": (1.0, _FORCE), "mN": (1e-3, _FORCE), "kN": (1e3, _FORCE), "MN": (1e6, _FORCE),
    "lbf": (4.4482216152605, _FORCE),
    "Pa": (1.0, _PRESSURE), "kPa": (1e3, _PRESSURE), "MPa": (1e6, _PRESSURE), "GPa": (1e9, _PRESSURE),
    "bar": (1e5, _PRESSURE), "psi": (6894.757293168361, _PRESSURE),
    "Hz": (1.0, (0, 0, -1, 0)),
    "K": (1.0, _K), "delta_degC": (1.0, _K),
}
_ABSOLUTE_ONLY = {"degC": 273.15}                     # offset to kelvin
_DIFFERENCE_ONLY = {"delta_degC"}
_KINDS: dict[str, Dim] = {"length": _L, "mass": _M, "time": _T, "force": _FORCE, "temperature": _K,
                          "temperature difference": _K}
_NAMES: dict[Dim, str] = {(0, 0, 0, 0): "dimensionless", _L: "length", _M: "mass", _T: "time", _K: "temperature",
                          _FORCE: "force", _PRESSURE: "pressure (force/length^2)", (-3, 1, 0, 0): "mass/length^3",
                          (0, 1, -2, 0): "force/length", (2, 1, -2, 0): "force*length",
                          (-2, 1, -2, 0): "force/length^3", (1, 0, -2, 0): "length/time^2",
                          (0, 0, -1, 0): "frequency (1/time)"}
_NUMBER = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S.*?)\s*$")


class UnitError(ValueError):
    """A value or unit that cannot be read or does not fit its parameter."""


def _add(a: Dim, b: Dim, k: int = 1) -> Dim:
    return (a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2], a[3] + k * b[3])


def _expression(text: str, table: Mapping[str, tuple[float, Dim]]) -> tuple[float, Dim]:
    """Factor and dimension of a product / quotient of names with integer powers ("kg/m^3", "N*m", "N/mm^2")."""
    factor, dim, sign = 1.0, (0, 0, 0, 0), 1
    for tok in re.split(r"(\*|/)", text.replace(" ", "")):
        if tok in ("*", "/"):
            sign = 1 if tok == "*" else -1
            continue
        m = re.fullmatch(r"([A-Za-z_]+)(?:\^([+-]?\d+))?", tok)
        if not m or m.group(1) not in table:
            raise UnitError(f'unknown unit "{tok}" in "{text}" (known: {", ".join(sorted(table))})')
        f, d = table[m.group(1)]
        p = sign * int(m.group(2) or 1)
        factor *= f ** p
        dim = _add(dim, d, p)
        sign = 1
    return factor, dim


def dimension_name(d: Dim) -> str:
    if d in _NAMES:
        return _NAMES[d]
    return " ".join(f"{s}^{p}" for s, p in zip(("L", "M", "T", "Theta"), d, strict=True) if p) or "dimensionless"


def kind_dimension(kind: str) -> Dim | None:
    """Dimension of a declared unit kind (``Param.unit``), None for a kind without a physical unit ("elements",
    "eigenvalue", "selector", …). A parenthesis ("force (total)") is a comment."""
    k = re.sub(r"\(.*?\)", "", kind).strip()
    if not k:
        return None
    if k in _KINDS:
        return _KINDS[k]
    try:
        return _expression(k, {**{n: (1.0, d) for n, d in _KINDS.items() if " " not in n}, **_UNITS})[1]
    except UnitError:
        return None


def is_quantity(text: str) -> bool:
    """True for text of the form "<number> <unit>" ("-1 kN"); keywords such as "max" are not quantities."""
    return _NUMBER.match(text) is not None


def parse_quantity(text: str) -> tuple[float, Dim, str]:
    """(value in SI, dimension, unit text) of "<number> <unit>"; degC is converted to kelvin."""
    m = _NUMBER.match(text)
    if not m:
        raise UnitError(f'expected "<number> <unit>", got "{text}"')
    value, unit = float(m.group(1)), m.group(2)
    if unit in _ABSOLUTE_ONLY:
        return value + _ABSOLUTE_ONLY[unit], _K, unit
    f, d = _expression(unit, _UNITS)
    return value * f, d, unit


@dataclass(frozen=True)
class UnitSystem:
    """The user's consistent units (SI factors). Time is the second; mass = force · s² / length."""

    length: float = 1.0
    force: float = 1.0
    temperature_offset: float = 0.0          # absolute user temperature = kelvin − offset (degC: 273.15)
    names: tuple[str, str, str] = ("m", "N", "K")

    @property
    def mass(self) -> float:
        return self.force / self.length

    @classmethod
    def from_config(cls, spec: Any) -> UnitSystem:
        if not isinstance(spec, Mapping):
            raise UnitError("units: expected a mapping {length: <unit>, force: <unit>, mass: <unit>, temperature: "
                            "K | degC}")
        unknown = set(spec) - {"length", "force", "mass", "temperature"}
        if unknown:
            raise UnitError(f"units: unknown keys {sorted(unknown)} (length, force, mass, temperature)")
        names = {k: str(spec.get(k, d)) for k, d in (("length", "m"), ("force", "N"), ("temperature", "K"))}
        fl, dl = _expression(names["length"], _UNITS)
        ff, df = _expression(names["force"], _UNITS)
        if dl != _L:
            raise UnitError(f'units.length: "{names["length"]}" is a {dimension_name(dl)}, not a length')
        if df != _FORCE:
            raise UnitError(f'units.force: "{names["force"]}" is a {dimension_name(df)}, not a force')
        if names["temperature"] not in ("K", "degC"):
            raise UnitError(f'units.temperature: K or degC, got "{names["temperature"]}"')
        out = cls(fl, ff, _ABSOLUTE_ONLY.get(names["temperature"], 0.0),
                  (names["length"], names["force"], names["temperature"]))
        if "mass" in spec:
            fm, dm = _expression(str(spec["mass"]), _UNITS)
            if dm != _M:
                raise UnitError(f'units.mass: "{spec["mass"]}" is a {dimension_name(dm)}, not a mass')
            if not math.isclose(fm, out.mass, rel_tol=1e-12):
                raise UnitError(f'units are inconsistent: with length {names["length"]} and force {names["force"]} '
                                f"(time in s) the mass unit is {out.mass:g} kg, not {spec['mass']} ({fm:g} kg)")
        return out

    def from_si(self, value: float, dim: Dim) -> float:
        return value / (self.length ** dim[0] * self.mass ** dim[1])


def convert(text: str, kind: str, system: UnitSystem) -> float:
    """``text`` ("210 GPa") as a number of ``system`` for a parameter of unit ``kind``."""
    want = kind_dimension(kind)
    if want is None:
        raise UnitError(f'"{text}": this parameter has no physical unit ({kind or "none"}); give a plain number')
    m = _NUMBER.match(text)
    unit = m.group(2) if m else ""
    value, got, _ = parse_quantity(text)
    if got != want:
        name = dimension_name(want)
        raise UnitError(f'"{text}" is a {dimension_name(got)}; this parameter is a {name}'
                        + ("" if kind in name else f" ({kind})"))
    if want == _K:
        diff = "difference" in kind
        if diff and unit in _ABSOLUTE_ONLY:
            raise UnitError(f'"{text}" is an absolute temperature; this parameter is a temperature difference '
                            f"(use K or delta_degC)")
        if not diff and unit in _DIFFERENCE_ONLY:
            raise UnitError(f'"{text}" is a temperature difference; this parameter is an absolute temperature')
        return value if diff else value - system.temperature_offset
    return system.from_si(value, want)


_ACTIVE: contextvars.ContextVar[UnitSystem | None] = contextvars.ContextVar("toporlogy_units", default=None)


def active() -> UnitSystem | None:
    """The unit system of the config being resolved (None: plain numbers only)."""
    return _ACTIVE.get()


@contextmanager
def use(system: UnitSystem | None) -> Iterator[None]:
    token = _ACTIVE.set(system)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def check_schema_version(v: Any) -> str:
    """The schema version of a config ("1" when absent); refuses one this library does not read."""
    s = "1" if v is None else str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s not in SCHEMA_VERSIONS:
        raise UnitError(f'schema_version "{v}" is not supported (supported: {", ".join(SCHEMA_VERSIONS)})')
    return s
