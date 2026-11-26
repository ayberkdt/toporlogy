"""Material data catalog: schema, admissibility and the YAML reference (docs/olcumler/malzeme_katalog_onkayit.md,
C1-C7, pre-registered before the code). Every record is synthetic: no real material value was entered."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from toporlogy.configuration import catalog as C
from toporlogy.configuration import units as U
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config, resolve

REPO = Path(__file__).resolve().parents[2]


def _mbb(material: dict, units: dict | None = None) -> dict:
    c = read_config(REPO / "examples" / "mbb2d.yaml")
    c["material"] = material
    if units is not None:
        c["units"] = units
    return c


def test_c1_c2_every_record_is_complete_and_admissible() -> None:
    ids = [m.id for m in C.MATERIALS]
    assert len(ids) == len(set(ids))
    for m in C.MATERIALS:
        C.validate_material(m)


@pytest.mark.parametrize(("change", "field"), [
    ({"values": {"E0": 1.0, "nu": 0.5, "rho": 1.0}}, "nu"),
    ({"family": "orthotropic", "values": {"E1": 1.0, "E2": 1.0, "E3": 1.0, "G12": 0.4, "G13": 0.4, "G23": 0.4,
                                         "nu12": 0.3, "nu13": 0.3, "nu23": 1.2}}, "orthotropic constants"),
    ({"kind": "typical", "page_or_table": "Table 2"}, "licence"),
    ({"kind": "typical", "licence": "CC-BY-4.0"}, "page_or_table"),
    ({"values": {"E0": 1.0, "nu": 0.3, "rho": 0.0}}, "rho")])
def test_c3_negative_controls_are_refused(change: dict, field: str) -> None:
    base = C.MATERIALS[0]
    bad = dataclasses.replace(base, **change)
    with pytest.raises(ValueError, match=field):
        C.validate_material(bad)


def test_c4_anisotropic_and_auxetic_records_are_accepted() -> None:
    ids = {m.id: m for m in C.MATERIALS}
    high = ids["synthetic_orthotropic_high_nu23"]
    assert high.values["nu23"] > 0.5
    C.validate_material(high)
    aux = ids["synthetic_auxetic_isotropic"]
    assert aux.values["nu"] < 0
    C.validate_material(aux)


@pytest.mark.parametrize("rid", ["synthetic_auxetic_isotropic", "synthetic_orthotropic_ply"])
def test_c5_catalog_reference_equals_writing_the_values(rid: str) -> None:
    m = {r.id: r for r in C.MATERIALS}[rid]
    v = dict(m.values)
    if m.family == "isotropic":
        by_hand = {"E0": v["E0"], "nu": v["nu"], "rho": v["rho"]}
    else:
        by_hand = {"E0": v["E1"], "rho": v["rho"], "orthotropic": {k: v[k] for k in C._ORTHO_KEYS}}
    assert resolve(_mbb({"catalog": rid}))["material"] == resolve(_mbb(by_hand))["material"]


def test_c6_si_record_converts_like_the_units_module() -> None:
    mm_n = {"length": "mm", "force": "N"}
    got = resolve(_mbb({"catalog": "synthetic_si_isotropic"}, mm_n))["material"]
    s = U.UnitSystem.from_config(mm_n)
    assert got["E0"] == U.convert(f"{2.0e11!r} Pa", "force/length^2", s) == 200000.0
    assert got["rho"] == U.convert(f"{8000.0!r} kg/m^3", "mass/length^3", s)


@pytest.mark.parametrize(("material", "units", "text"), [
    ({"catalog": "synthetic_unit_isotropik"}, None, "did you mean"),
    ({"catalog": "synthetic_unit_isotropic", "E0": 2.0}, None, "both"),
    ({"catalog": "synthetic_si_isotropic"}, None, "units")])
def test_c7_errors(material: dict, units: dict | None, text: str) -> None:
    with pytest.raises(ConfigError, match=text):
        resolve(_mbb(material, units))
