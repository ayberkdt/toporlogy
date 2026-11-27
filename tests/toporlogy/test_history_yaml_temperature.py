"""Temperature histories in the history YAML path (docs/foundations/prereg/HT_history_yaml_temperature.md, HT1-HT4)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from toporlogy.configuration.history_build import build_history, resolve_history
from toporlogy.configuration.registry import ConfigError
from toporlogy.core.creep import ArrheniusFactor
from toporlogy.core.linear_elastic import isotropic_moduli
from toporlogy.core.viscoelastic import GeneralizedMaxwell, PronySeries, WLFShift
from toporlogy.optimization.creep_topology import CreepStructure
from toporlogy.optimization.viscoelastic_topology import ViscoelasticStructure

from test_history_yaml import (CR_TIMES, MPA, TIMES, _cfg, _cr_card, _mesh_and_fixed, _tip_loads, _ve_card)

TABLE = [[0.0, 300.0], [3.5, 330.0], [300.0, 330.0]]


def _ve_thermal_cfg():
    card = _ve_card()
    card["units"]["temperature"] = "K"
    card["parameters"]["shift"] = {"type": "wlf", "T_ref": 300.0, "C1": 17.4, "C2": 51.6}
    card["validity"]["temperature"] = [290.0, 340.0]
    c = _cfg(card, TIMES, -1.0e3)
    c["temperature"] = {"table": copy.deepcopy(TABLE)}
    return c


def _cr_thermal_cfg():
    card = _cr_card()
    card["units"]["temperature"] = "K"
    card["parameters"]["arrhenius"] = {"Q": 8.0e4, "T_ref": 300.0}
    card["validity"]["temperature"] = [290.0, 340.0]
    c = _cfg(card, CR_TIMES, -5.0e6, n=(6, 3), rmin=0.3, ramp=1.0)
    c["temperature"] = {"table": [[0.0, 300.0], [20.0, 330.0], [100.0, 330.0]]}
    return c


def _J_grad(st, x):
    xt = torch.as_tensor(x).requires_grad_(True)
    J = st.response(1e-9 + xt ** 3 * (1 - 1e-9)).compliance[-1]
    J.backward()
    return float(J), xt.grad.numpy().copy()


def test_ht1_shifted_viscoelastic_yaml_equals_api() -> None:
    st_yaml = build_history(resolve_history(_ve_thermal_cfg())).structure
    mat = GeneralizedMaxwell(PronySeries(1.0 * MPA, [2.0 * MPA, 3.0 * MPA], [1.0, 100.0]),
                             PronySeries(5.0 * MPA, [4.0 * MPA], [10.0]), shift=WLFShift(300.0, 17.4, 51.6))
    m, fixed, tip = _mesh_and_fixed((8, 4))
    tab = np.array(TABLE)
    st_api = ViscoelasticStructure(m, mat, TIMES, _tip_loads(m, tip, TIMES, -1.0e3, 2.0), fixed,
                                   temperatures=np.interp(TIMES, tab[:, 0], tab[:, 1]), temperature0=300.0)
    x = 0.3 + 0.6 * np.random.default_rng(4).random(m.n_cells)
    a, b = _J_grad(st_yaml, x), _J_grad(st_api, x)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_ht2_arrhenius_creep_yaml_equals_api() -> None:
    st_yaml = build_history(resolve_history(_cr_thermal_cfg())).structure
    mo = isotropic_moduli(E=7.0e10, nu=0.33)
    m, fixed, tip = _mesh_and_fixed((6, 3))
    temps = np.interp(CR_TIMES, [0.0, 20.0, 100.0], [300.0, 330.0, 330.0])
    st_api = CreepStructure(m, mo.K, mo.G, 1.0e-6, 1.0e8, 3.0, CR_TIMES, _tip_loads(m, tip, CR_TIMES, -5.0e6, 1.0),
                            fixed, arrhenius=ArrheniusFactor(8.0e4, 300.0), temperatures=temps, temperature0=300.0)
    x = 0.4 + 0.5 * np.random.default_rng(4).random(m.n_cells)
    a, b = _J_grad(st_yaml, x), _J_grad(st_api, x)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


@pytest.mark.parametrize(("case", "path"), [
    ("shift_without_temperature", "material.parameters.shift"),
    ("temperature_without_shift", "temperature"),
    ("not_from_zero", "temperature.table.0.0"),
    ("not_increasing", "temperature.table"),
    ("ends_early", "temperature.table"),
    ("nonpositive_T", "temperature.table.1.1"),
    ("bad_shape", "temperature.table.1"),
    ("outside_validity", "temperature"),
])
def test_ht3_refusals_name_the_path(case, path) -> None:
    c = _ve_thermal_cfg()
    if case == "shift_without_temperature":
        del c["temperature"]
    elif case == "temperature_without_shift":
        c = _cfg(_ve_card(), TIMES, -1.0e3)
        c["temperature"] = {"table": copy.deepcopy(TABLE)}
    elif case == "not_from_zero":
        c["temperature"]["table"][0][0] = 0.1
    elif case == "not_increasing":
        c["temperature"]["table"][2][0] = 3.0
    elif case == "ends_early":
        c["temperature"]["table"][2][0] = 100.0
    elif case == "nonpositive_T":
        c["temperature"]["table"][1][1] = -5.0
    elif case == "bad_shape":
        c["temperature"]["table"][1] = [3.5]
    elif case == "outside_validity":
        c["temperature"]["table"][1][1] = 350.0
    with pytest.raises(ConfigError) as ei:
        resolve_history(c)
    assert path in str(ei.value), str(ei.value)


def test_ht4_resolve_idempotent_with_temperature() -> None:
    r = resolve_history(_ve_thermal_cfg())
    assert resolve_history(copy.deepcopy(r)) == r
