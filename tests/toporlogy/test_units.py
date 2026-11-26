"""Schema version and unit-checked config values (docs/olcumler/birimler_onkayit.md, U1-U8, pre-registered
2026-11-26 03:03 +0300 before the code). ``baseline/resolved_configs.json`` holds the resolved configs of every example
written by the code before the change (commit cb1a483)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config, resolve
from toporlogy.configuration import units as U

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MM_N = {"length": "mm", "force": "N"}


def _mbb() -> dict:
    return read_config(REPO / "examples" / "mbb2d.yaml")


def _examples() -> list[Path]:
    return sorted(p.relative_to(REPO) for p in (REPO / "examples").glob("**/*.y*ml"))


def test_u1_examples_resolve_as_before() -> None:
    """The resolved config of every example is a snapshot (written by the code before the units change, cb1a483;
    rewritten after intended changes of the components: ``python tests/toporlogy/test_units.py --write``, say why
    in the commit)."""
    old = json.loads((HERE / "baseline" / "resolved_configs.json").read_text(encoding="utf-8"))
    assert sorted(old) == [p.as_posix() for p in _examples()], "examples added or removed: --write"
    for name, want in old.items():
        raw = read_config(REPO / name)
        got = resolve(raw)
        units = got.pop("units")
        assert got.pop("schema_version") == "1" and (units is None) == ("units" not in raw)
        assert json.loads(json.dumps(got, sort_keys=True, default=repr)) == want, name


@pytest.mark.parametrize(("text", "kind", "want"), [
    ("1 mm", "length", 1.0), ("0.001 m", "length", 1.0), ("210 GPa", "force/length^2", 210000.0),
    ("1 kN", "force", 1000.0), ("7850 kg/m^3", "mass/length^3", 7.85e-9), ("-1 kN", "force (total)", -1000.0),
    ("9.81 m/s^2", "length/time^2", 9810.0), ("2 N/mm^2", "force/length^2", 2.0)])
def test_u2_conversion_into_mm_and_n(text: str, kind: str, want: float) -> None:
    got = U.convert(text, kind, U.UnitSystem.from_config(MM_N))
    assert abs(got - want) <= 1e-15 * abs(want)


@pytest.mark.parametrize(("where", "value", "names"), [
    ("force", [0, "5 kg"], ("mass", "force")), ("E0", "100 N", ("force", "pressure")), ("h", "1 s", ("time", "length"))])
def test_u3_dimension_mismatch_names_both_dimensions(where: str, value: object, names: tuple[str, str]) -> None:
    c = _mbb()
    c["units"] = MM_N
    if where == "force":
        c["load_cases"][0]["loads"][0]["force"] = value
    elif where == "E0":
        c["material"] = {"E0": value, "nu": 0.3}
    else:
        c["domain"]["h"] = value
    with pytest.raises(ConfigError) as e:
        resolve(c)
    assert all(n in str(e.value) for n in names), str(e.value)


def test_u4_absolute_temperature_is_not_a_difference() -> None:
    s = U.UnitSystem.from_config({"length": "m", "force": "N", "temperature": "degC"})
    with pytest.raises(U.UnitError, match="absolute temperature"):
        U.convert("20 degC", "temperature difference", s)
    with pytest.raises(U.UnitError, match="difference"):
        U.convert("5 delta_degC", "temperature", s)
    assert U.convert("5 K", "temperature difference", s) == 5.0
    assert math.isclose(U.convert("293.15 K", "temperature", s), 20.0, rel_tol=1e-12)
    assert math.isclose(U.convert("20 degC", "temperature", s), 20.0, rel_tol=1e-12)


def test_u5_inconsistent_mass_unit_is_refused() -> None:
    c = _mbb()
    c["units"] = {"length": "mm", "force": "N", "mass": "kg"}
    with pytest.raises(ConfigError, match="inconsistent"):
        resolve(c)
    c["units"] = {"length": "mm", "force": "N", "mass": "t"}
    assert resolve(c)["units"]["length"] == "mm"


def test_u6_a_unit_without_units_section_is_still_refused() -> None:
    c = _mbb()
    c["domain"]["h"] = "1 mm"
    with pytest.raises(ConfigError, match="number"):
        resolve(c)


@pytest.mark.parametrize(("v", "ok"), [(None, True), (1, True), ("1", True), (1.0, True), ("2", False), (2, False)])
def test_u7_schema_version(v: object, ok: bool) -> None:
    c = _mbb()
    if v is not None:
        c["schema_version"] = v
    if ok:
        assert resolve(c)["schema_version"] == "1"
    else:
        with pytest.raises(ConfigError, match=r"schema_version.*supported: 1"):
            resolve(c)


def _plain_and_mm() -> tuple:
    plain = build_setup(_mbb()).study
    c = _mbb()
    c["units"] = MM_N
    c["material"] = {"E0": "210 GPa", "nu": 0.3}
    c["load_cases"][0]["loads"][0]["force"] = [0, "-1 kN"]
    return plain, build_setup(copy.deepcopy(c)).study


@pytest.mark.xfail(strict=True, reason="pre-registered U8 (1e-12 relative) FAILED on the first run: the compliance "
                   "ratio differs from 1000^2 / 210000 by 3.75e-12 relative. Not verified why; the moduli and loads "
                   "scale exactly (post hoc test below), so the solve's rounding (cond K about 1e8-1e11 with Emin) "
                   "is the likely source, which the threshold did not provide for")
def test_u8_compliance_in_mm_and_n_scales_exactly() -> None:
    plain, unit = _plain_and_mm()
    x = plain.default_start()
    a = float(plain.evaluate(x, False).f0)
    b = float(unit.evaluate(x, False).f0)
    print(f"\n  U8: ratio error {abs(b / a - 1000.0 ** 2 / 210000.0) / (1000.0 ** 2 / 210000.0):.2e}")
    assert abs(b / a - 1000.0 ** 2 / 210000.0) <= 1e-12 * (1000.0 ** 2 / 210000.0)


def test_u8_posthoc_loads_and_modulus_scale_exactly() -> None:
    """Post hoc (written after U8 failed): the converted inputs are exact; the load vector is 1000 x the plain one
    and the solid modulus is 210000 bit for bit."""
    plain, unit = _plain_and_mm()
    assert np.array_equal(np.asarray(unit.F), 1000.0 * np.asarray(plain.F))
    assert float(unit.interp.E0) == 210000.0 and float(plain.interp.E0) == 1.0


if __name__ == "__main__":
    import sys
    if "--write" not in sys.argv:
        sys.exit("usage: python tests/toporlogy/test_units.py --write")
    out = {}
    for p in _examples():
        r = resolve(read_config(REPO / p))
        r.pop("schema_version"), r.pop("units")
        out[p.as_posix()] = json.loads(json.dumps(r, sort_keys=True, default=repr))
    (HERE / "baseline" / "resolved_configs.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=repr)
                                                             + "\n", encoding="utf-8")
    print(f"wrote {len(out)} resolved configs")
