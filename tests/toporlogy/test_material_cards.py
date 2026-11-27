"""Material cards (``toporlogy.optimization.material_cards``), checks MC1-MC7 of
``docs/foundations/prereg/P05_material_cards.md`` (committed before the first run of this file).

Every number is synthetic. MC2 is a regression test (the same model kernels are called, from a card and directly); its
independent part is that the direct constructors below use hand-converted SI numbers. MC2b compares with closed forms
written here; MC6 documents what the unit checks catch and what they cannot.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from toporlogy.core.creep import ArrheniusFactor, NortonBaileyCreep, NortonCreep
from toporlogy.core.constitutive import drive_material_point
from toporlogy.core.cure import (Arrhenius, ChemicalShrinkage, CuringThermoElasticPoint, KamalSourour, NthOrder)
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.core.properties import Constant, IsotropicModuliOfT, PiecewiseLinear, ThermalExpansion, ThermoElasticPoint
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries, WLFShift
from toporlogy.core.viscoplastic import DruckerPrager, Perzyna, PhysicalJ2
from toporlogy.optimization.material_cards import MATERIAL_MODELS, MaterialCard, load_material_card
from toporlogy.optimization.material_identification import ValidityBoxError

MAT = Path(__file__).resolve().parents[2] / "examples" / "foundations" / "materials"
CTRL = np.ones(6, bool)
FILES = sorted(MAT.glob("*.yaml"))


def card_of(model: str) -> dict[str, Any]:
    return yaml.safe_load((MAT / f"{model}.yaml").read_text(encoding="utf-8"))


def base() -> dict[str, Any]:
    """The Pa/s/K physical_j2 card the refusal cases mutate (synthetic)."""
    return {"name": "base", "model": "physical_j2",
            "units": {"stress": "Pa", "time": "s", "temperature": "K", "rate": "1/s"},
            "parameters": {"elasticity": {"E": 2.0e9, "nu": 0.35}, "yield_stress": 2.0e7, "hardening": 1.0e8},
            "provenance": {"source": "synthetic test card", "synthetic": True, "date": "2026-11-27",
                           "license": "CC0-1.0"},
            "validity": {"stress": [0.0, 3.0e7]}}


def drive(model: Any, a: float, times: Any, temps: Any, T0: float):
    f = np.arange(1, 6) / 5.0
    strain = np.outer(f, a * np.array([1.0, -0.2, 0.0, 0.0, 0.0, 0.5]))
    return drive_material_point(model, strain, np.zeros((5, 6)), CTRL, times=times, temperatures=temps,
                                temperature0=T0)


# ---------------------------------------------------------------------------------------------------- MC1 / MC7
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_MC1_round_trip(path: Path) -> None:
    card = load_material_card(yaml.safe_load(path.read_text(encoding="utf-8")))
    d = card.to_dict()
    assert load_material_card(d).to_dict() == d
    assert load_material_card(yaml.safe_load(yaml.safe_dump(d))).to_dict() == d
    d["parameters"] = None                                   # to_dict is a copy: mutating it does not touch the card
    assert card.to_dict()["parameters"] is not None


def test_MC7_examples_cover_every_family_and_are_synthetic() -> None:
    # MC7 as preregistered: the example set covers exactly the 8 families (integration note: the HY slice added
    # two isothermal variants, so files are no longer one per family; a variant's file name starts with its family)
    assert len(MATERIAL_MODELS) == 8
    models = set()
    for p in FILES:
        card = load_material_card(yaml.safe_load(p.read_text(encoding="utf-8")))
        assert isinstance(card, MaterialCard)
        assert card.synthetic is True and card.provenance["synthetic"] is True
        assert "synthetic" in card.provenance["source"].lower()
        assert p.stem == card.kind or p.stem.startswith(card.kind + "_")
        models.add(card.kind)
    assert models == set(MATERIAL_MODELS)


# ---------------------------------------------------------------------------------------------------- MC2
def _direct() -> dict[str, tuple[Any, float, Any, Any, float]]:
    """model -> (directly built model from hand-converted SI numbers, strain amplitude, times, temperatures, T0)."""
    def mod_T(E, nu, dom):
        return IsotropicModuliOfT(E=E, nu=Constant(nu, "1", "x", domain=dom))

    K, G = isotropic_moduli(E=7.0e10, nu=0.33).K, isotropic_moduli(E=7.0e10, nu=0.33).G
    Kb, Gb = 3000.0e6, 1100.0e6
    Kj, Gj = isotropic_moduli(E=2000.0e6, nu=0.35).K, isotropic_moduli(E=2000.0e6, nu=0.35).G
    mp = isotropic_moduli(E=2000.0e6, G=740.0e6)
    dom_t = (250.0, 400.0)
    E_t = PiecewiseLinear([250.0, 300.0, 350.0, 400.0], [3200.0e6, 3000.0e6, 2600.0e6, 1800.0e6], "Pa", "x",
                          positive=True)
    dom_c = (300.0, 450.0)
    kin = NthOrder(Arrhenius(5.0e6 * (1.0 / 60.0), 6.0e4), 1.5)
    cure = CuringThermoElasticPoint(
        kin,
        mod_T(Constant(5.0e6, "Pa", "x", domain=dom_c, positive=True), 0.40, dom_c),
        mod_T(Constant(3000.0e6, "Pa", "x", domain=dom_c, positive=True), 0.35, dom_c),
        ThermalExpansion(Constant(5.0e-5, "1/K", "x", domain=dom_c), 300.0), ChemicalShrinkage(-0.06, 0.1),
        alpha0=0.0, substeps=4)
    k5 = np.arange(1, 6)
    return {
        "thermoelastic": (ThermoElasticPoint(mod_T(E_t, 0.35, dom_t), Constant(6.0e-5, "1/K", "x", domain=dom_t),
                                             300.0), 2e-3, 60.0 * k5, [310.0, 330.0, 350.0, 370.0, 390.0], 300.0),
        "generalized_maxwell": (GeneralizedMaxwell(PronySeries(400.0e6, [300.0e6, 200.0e6], [1.0, 100.0]),
                                                   PronySeries(3000.0e6, [500.0e6], [10.0]),
                                                   WLFShift(300.0, 17.44, 51.6)),
                                1e-3, [1.0, 10.0, 100.0, 1000.0, 10000.0], [305.0, 310.0, 315.0, 320.0, 325.0], 300.0),
        "norton": (NortonCreep(K, G, 1.0e-6, 1.0e8, 4.5, arrhenius=ArrheniusFactor(1.5e5, 600.0)), 2e-3,
                   1.0e3 * k5, [600.0, 610.0, 620.0, 630.0, 640.0], 600.0),
        "norton_bailey": (NortonBaileyCreep(Kb, Gb, 0.002, 10.0e6, 3.0, 0.4, 3600.0, hardening="time"), 1e-3,
                          3600.0 * k5, [300.0] * 5, 300.0),
        "physical_j2": (PhysicalJ2(Kj, Gj, 20.0e6, hardening=100.0e6, saturation_stress=30.0e6, saturation_rate=20.0),
                        0.02, 1.0 * k5, [300.0] * 5, 300.0),
        "perzyna": (Perzyna(mp.K, mp.G, 20.0e6, 100.0e6, 0.5, 20.0e6, 2.0), 0.01, 0.1 * k5, [300.0] * 5, 300.0),
        "drucker_prager": (DruckerPrager(2.0e9, 1.0e9, 0.6, 2.0e7, 1.0e8, 0.3), 0.02, 1.0 * k5, [300.0] * 5, 300.0),
        "curing_thermoelastic": (cure, 2e-3, 600.0 * k5, [350.0, 360.0, 370.0, 380.0, 390.0], 300.0),
    }


@pytest.mark.parametrize("name", MATERIAL_MODELS)
def test_MC2_card_model_equals_direct_model_regression(name: str) -> None:
    """Regression (same kernels): bitwise equal stress on a 5-step path."""
    direct, a, times, temps, T0 = _direct()[name]
    card = load_material_card(card_of(name))
    assert type(card.model) is type(direct)
    h_card = drive(card.model, a, np.asarray(times, float), np.asarray(temps, float), T0)
    h_direct = drive(direct, a, np.asarray(times, float), np.asarray(temps, float), T0)
    assert np.abs(h_direct.stress).max() > 0.0
    assert np.array_equal(h_card.stress, h_direct.stress)


def test_MC2c_kamal_sourour_kinetics_regression() -> None:
    c = card_of("curing_thermoelastic")
    c["parameters"]["kinetics"] = {"type": "kamal_sourour", "k1": {"A": 1.0e3, "Q": 5.0e4},
                                   "k2": {"A": 5.0e6, "Q": 6.0e4}, "m": 0.5, "n": 1.5}
    card = load_material_card(c)
    assert card.to_dict()["parameters"]["kinetics"]["type"] == "kamal_sourour"
    direct, a, times, temps, T0 = _direct()["curing_thermoelastic"]
    ks = KamalSourour(Arrhenius(1.0e3 * (1.0 / 60.0), 5.0e4), Arrhenius(5.0e6 * (1.0 / 60.0), 6.0e4), 0.5, 1.5)
    direct = CuringThermoElasticPoint(ks, direct.moduli_u, direct.moduli_c, direct.expansion, direct.shrinkage,
                                      alpha0=0.0, substeps=4)
    t, T = np.asarray(times, float), np.asarray(temps, float)
    assert np.array_equal(drive(card.model, a, t, T, T0).stress, drive(direct, a, t, T, T0).stress)


# ---------------------------------------------------------------------------------------------------- MC2b
def test_MC2b_thermoelastic_uniaxial_strain_closed_form() -> None:
    card = load_material_card(card_of("thermoelastic"))
    e = np.array([1e-3, 2e-3, 3e-3])
    T = np.array([320.0, 350.0, 390.0])
    strain = np.zeros((3, 6))
    strain[:, 0] = e
    h = drive_material_point(card.model, strain, np.zeros((3, 6)), CTRL, times=[1.0, 2.0, 3.0], temperatures=T,
                             temperature0=300.0)
    E = np.interp(T, [250.0, 300.0, 350.0, 400.0], [3200.0e6, 3000.0e6, 2600.0e6, 1800.0e6])
    nu = 0.35
    K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
    eth = 6.0e-5 * (T - 300.0)
    sxx = (K + 4 * G / 3) * e - 3 * K * eth
    syy = (K - 2 * G / 3) * e - 3 * K * eth
    err = max(np.abs(h.stress[:, 0] - sxx).max(), np.abs(h.stress[:, 1] - syy).max(),
              np.abs(h.stress[:, 2] - syy).max(), np.abs(h.stress[:, 3:]).max())
    assert err / np.abs(sxx).max() <= 1e-12, err / np.abs(sxx).max()


def test_MC2b_physical_j2_elastic_range_closed_form() -> None:
    card = load_material_card(card_of("physical_j2"))
    e = np.array([1e-3, 2e-3, 3e-3])
    strain = np.zeros((3, 6))
    strain[:, 0] = e
    h = drive_material_point(card.model, strain, np.zeros((3, 6)), CTRL)
    E, nu = 2000.0e6, 0.35
    c11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    c12 = E * nu / ((1 + nu) * (1 - 2 * nu))
    err = max(np.abs(h.stress[:, 0] - c11 * e).max(), np.abs(h.stress[:, 1] - c12 * e).max())
    assert err / (c11 * e).max() <= 1e-12, err / (c11 * e).max()


# ---------------------------------------------------------------------------------------------------- MC3
def S(path: str, value: Any):
    def f(c: dict[str, Any]) -> None:
        keys = path.split(".")
        d = c
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return f


def D(path: str):
    def f(c: dict[str, Any]) -> None:
        keys = path.split(".")
        d = c
        for k in keys[:-1]:
            d = d[k]
        del d[keys[-1]]
    return f


def many(*fs):
    def f(c: dict[str, Any]) -> None:
        for g in fs:
            g(c)
    return f


NAN = float("nan")
CASES: list[tuple[str, Any, Any, list[str]]] = [
    ("R01", base, S("color", 1), ["color"]),
    ("R02", base, D("provenance"), ["provenance"]),
    ("R03", base, D("validity"), ["validity"]),
    ("R04", base, S("model", "foo"), ["model"]),
    ("R05", base, S("parameters.yeild2", 1.0), ["parameters.yeild2"]),
    ("R06", base, D("parameters.yield_stress"), ["parameters.yield_stress"]),
    ("R07", base, S("units.C", "K"), ["units.C", "Celsius"]),
    ("R08", base, S("units.temperature", "degC"), ["units.temperature", "Celsius"]),
    ("R09", base, S("units.temperature", "C"), ["units.temperature", "Celsius"]),
    ("R10", base, S("units.stress", "psi"), ["units.stress"]),
    ("R11", base, S("units.stress", "s"), ["units.stress"]),
    ("R12", base, D("units.stress"), ["parameters.elasticity.E"]),
    ("R13", lambda: card_of("thermoelastic"), S("validity.temperature", [-20.0, 80.0]), ["validity.temperature"]),
    ("R14", base, S("validity.stress", [5.0, 1.0]), ["validity.stress"]),
    ("R15", base, S("validity.stress", [0.0]), ["validity.stress"]),
    ("R16", base, S("validity.pressure", [0.0, 1.0]), ["validity.pressure"]),
    ("R17", base, S("parameters.elasticity.E", -2.0e9), ["parameters.elasticity.E"]),
    ("R18", base, S("parameters.yield_stress", -2.0e7), ["parameters.yield_stress"]),
    ("R19", base, S("parameters.hardening", -1.0), ["parameters.hardening"]),
    ("R20", base, S("parameters.elasticity.G", 2.0e9 / (2 * 1.35)), ["parameters.elasticity", "redundant"]),
    ("R21", base, S("parameters.elasticity.G", 1.0e9), ["redundant", "inconsistent"]),
    ("R22", base, D("parameters.elasticity.nu"), ["parameters.elasticity"]),
    ("R23", base, S("parameters.elasticity.nu", 0.5), ["parameters.elasticity.nu"]),
    ("R24", base, S("parameters.yield_stress", NAN), ["parameters.yield_stress"]),
    ("R25", base, S("parameters.yield_stress", True), ["parameters.yield_stress"]),
    ("R26", base, S("parameters.yield_stress", "2e7"), ["parameters.yield_stress"]),
    ("R27", base, many(S("provenance.synthetic", False), D("provenance.source")), ["provenance.source"]),
    ("R28", base, many(S("provenance.synthetic", False), D("provenance.license")), ["provenance.license"]),
    ("R29", base, S("provenance.synthetic", "false"), ["provenance.synthetic"]),
    ("R30", base, D("provenance.date"), ["provenance.date"]),
    ("R31", base, S("provenance.date", "27/11/2026"), ["provenance.date"]),
    ("R32", base, many(S("provenance.synthetic", False), S("provenance.source", "synthetic")), ["provenance.source"]),
    ("R33", base, S("name", ""), ["name"]),
    ("R34", lambda: card_of("thermoelastic"), S("parameters.T_ref", -10.0), ["parameters.T_ref"]),
    ("R35", lambda: card_of("thermoelastic"),
     S("parameters.alpha", {"T": [-5.0, 300.0, 400.0], "values": [6e-5, 6e-5, 6e-5]}), ["parameters.alpha.T"]),
    ("R36", lambda: card_of("generalized_maxwell"), S("parameters.shear.taus", [-1.0, 100.0]),
     ["parameters.shear.taus"]),
    ("R37a", lambda: card_of("norton"), S("parameters.n", 0.0), ["parameters.n"]),
    ("R37b", lambda: card_of("norton"), S("parameters.arrhenius.Q", -1.0), ["parameters.arrhenius.Q"]),
    ("R38a", lambda: card_of("thermoelastic"), S("parameters.T_ref", "25C"), ["parameters.T_ref"]),
    ("R38b", lambda: card_of("drucker_prager"), S("parameters.a", 3.0), ["parameters"]),
]


@pytest.mark.parametrize("cid,make,mutate,needles", CASES, ids=[c[0] for c in CASES])
def test_MC3_refusals_name_the_offending_key(cid: str, make: Any, mutate: Any, needles: list[str]) -> None:
    card = make()
    mutate(card)
    with pytest.raises(ValueError) as ei:
        load_material_card(card)
    msg = str(ei.value)
    for needle in needles:
        assert needle in msg, (cid, needle, msg)


def test_MC3_unmutated_cards_load() -> None:
    assert load_material_card(base()).kind == "physical_j2"


# ---------------------------------------------------------------------------------------------------- MC4
def _si_norton_bailey() -> dict[str, Any]:
    c = card_of("norton_bailey")
    c["units"] = {"stress": "Pa", "time": "s"}
    p = c["parameters"]
    p["elasticity"] = {"K": 3000.0e6, "G": 1100.0e6}
    p["sigma_ref"], p["t_ref"] = 10.0e6, 3600.0
    c["validity"] = {"stress": [0.0, 20.0e6], "time": [0.0, 3.6e6]}
    return c


def _si_curing() -> dict[str, Any]:
    c = card_of("curing_thermoelastic")
    c["units"] = {"stress": "Pa", "time": "s", "temperature": "K", "rate": "1/s"}
    p = c["parameters"]
    p["kinetics"]["A"] = 5.0e6 / 60.0
    p["moduli_uncured"]["E"], p["moduli_cured"]["E"] = 5.0e6, 3000.0e6
    c["validity"] = {"temperature": [300.0, 450.0], "time": [0.0, 36000.0]}
    return c


@pytest.mark.parametrize("name,si,a,times,temps,T0", [
    ("norton_bailey", _si_norton_bailey, 1e-3, 3600.0 * np.arange(1, 6), [300.0] * 5, 300.0),
    ("curing_thermoelastic", _si_curing, 2e-3, 600.0 * np.arange(1, 6), [350.0, 360.0, 370.0, 380.0, 390.0], 300.0),
])
def test_MC4_unit_factors(name: str, si: Any, a: float, times: Any, temps: Any, T0: float) -> None:
    declared = load_material_card(card_of(name))
    in_si = load_material_card(si())
    assert declared.to_dict()["units"] != in_si.to_dict()["units"]          # the declared units are kept
    assert declared.units == card_of(name)["units"]
    t, T = np.asarray(times, float), np.asarray(temps, float)
    s1 = drive(declared.model, a, t, T, T0).stress
    s2 = drive(in_si.model, a, t, T, T0).stress
    assert np.abs(s1 - s2).max() / np.abs(s2).max() <= 1e-12
    assert declared.validity.ranges == in_si.validity.ranges


# ---------------------------------------------------------------------------------------------------- MC5
def test_MC5_validity_box_closed_ends_and_si_conversion() -> None:
    p = load_material_card(card_of("perzyna"))                                  # MPa, 1/s
    p.check(stress=0.0), p.check(stress=40.0e6), p.check(rate=1.0e-4), p.check(rate=10.0)
    for kw in ({"stress": 40.0e6 + 1.0}, {"stress": -1.0}, {"rate": 1.0e-5}, {"rate": 10.0001},
               {"stress": np.array([1.0, 5.0e7])}):
        with pytest.raises(ValidityBoxError):
            p.check(**kw)
    c = load_material_card(card_of("curing_thermoelastic"))                      # min -> s
    c.check(temperature=300.0), c.check(temperature=450.0), c.check(time=0.0), c.check(time=36000.0)
    for kw in ({"temperature": 299.9}, {"temperature": 450.1}, {"time": 36000.1}, {"time": -1.0},
               {"temperature": np.array([300.0, 500.0])}):
        with pytest.raises(ValidityBoxError):
            c.check(**kw)
    with pytest.raises(ValueError) as ei:                                         # undeclared coordinate
        p.check(temperature=300.0)
    assert not isinstance(ei.value, ValidityBoxError)
    with pytest.raises(ValueError):
        p.check()
    assert isinstance(ValidityBoxError("x"), ValueError)


# ---------------------------------------------------------------------------------------------------- MC6
def _j2(E: float, sy: float, H: float, hi: float) -> dict[str, Any]:
    c = base()
    c["parameters"] = {"elasticity": {"E": E, "nu": 0.35}, "yield_stress": sy, "hardening": H}
    c["validity"] = {"stress": [0.0, hi]}
    return c


def test_MC6a_modulus_mislabelled_is_caught_by_U1() -> None:
    with pytest.raises(ValueError) as ei:
        load_material_card(_j2(2000.0, 2.0e7, 1.0e8, 3.0e7))               # E truly MPa, labelled Pa; range right
    assert "validity.stress" in str(ei.value) and "U1" in str(ei.value)


def test_MC6b_strength_mislabelled_is_caught_by_U2() -> None:
    with pytest.raises(ValueError) as ei:
        load_material_card(_j2(2.0e9, 20.0, 1.0e8, 3.0e7))                  # yield truly MPa, labelled Pa
    assert "parameters.yield_stress" in str(ei.value) and "U2" in str(ei.value)


def test_MC6c_consistent_mislabel_is_accepted_and_off_by_1e6() -> None:
    right = load_material_card(_j2(2.0e9, 2.0e7, 1.0e8, 3.0e7))
    wrong = load_material_card(_j2(2000.0, 20.0, 100.0, 30.0))             # everything MPa numerals labelled Pa
    a, t = 0.02, 1.0 * np.arange(1, 6)
    s_right = drive(right.model, a, t, [300.0] * 5, 300.0).stress
    s_wrong = drive(wrong.model, a, t, [300.0] * 5, 300.0).stress
    assert np.abs(s_right).max() > 0
    assert np.abs(s_wrong - 1.0e-6 * s_right).max() <= 1e-12 * np.abs(s_wrong).max()


def test_MC6d_without_a_stress_range_no_check_applies() -> None:
    c = card_of("thermoelastic")
    c["units"]["stress"] = "Pa"                                              # numerals are MPa: 1e6 off, undetected
    mislabeled = load_material_card(c)
    right = load_material_card(card_of("thermoelastic"))
    T = [300.0]
    assert mislabeled.model.moduli.K(T[0]) * 1.0e6 == pytest.approx(right.model.moduli.K(T[0]), rel=1e-12)


def test_MC8_card_is_not_mutated_by_loading() -> None:
    c = base()
    before = copy.deepcopy(c)
    load_material_card(c)
    assert c == before
