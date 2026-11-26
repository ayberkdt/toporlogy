"""Thermoelasticity as Study physics (docs/olcumler/termoelastik_study_onkayit.md, TS1-TS5; TS6 is the baseline run
and the resolved-config snapshot)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError


def _cfg(dim: int = 2, delta_T: float = 50.0, force: float = -1.0, physics: bool = True) -> dict:
    shape = [12, 6] if dim == 2 else [6, 3, 3]
    at = ["max", "min"] if dim == 2 else ["max", "min", "min"]
    f = [0, force] if dim == 2 else [0, force, 0]
    cfg = {"domain": {"type": "box", "shape": shape, "h": 1.0}, "material": {"E0": 1.0, "nu": 0.3},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": at}, "force": f}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}], "objective": "compliance",
           "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma"}
    if physics:
        cfg["physics"] = {"type": "thermoelastic", "alpha": 1e-2, "delta_T": delta_T}
    return cfg


def _x(st, seed=2):
    return 0.2 + 0.7 * np.random.default_rng(seed).random(st.n_var)


def test_ts1_zero_temperature_equals_elasticity() -> None:
    a = build_setup(_cfg(delta_T=0.0)).study
    b = build_setup(_cfg(physics=False)).study
    x = _x(a)
    ea, eb = a.evaluate(x), b.evaluate(x)
    assert abs(ea.f0 - eb.f0) <= 1e-12 * abs(eb.f0)
    assert np.abs(ea.df0 - eb.df0).max() <= 1e-12 * np.abs(eb.df0).max()


def test_ts2_value_equals_direct_thermoelastic() -> None:
    st = build_setup(_cfg()).study
    x = _x(st)
    ev = st.evaluate(x, False)
    d = st.domain
    E = st.interp(torch.as_tensor(ev.xphys)).reshape(-1)[torch.as_tensor(d.active_elements)]
    te = st.physics.te
    u, F = te.displacement(E, torch.as_tensor(st.physics.T), st.F[:, 0])
    c = float(F @ u)
    assert abs(ev.f0 - c) <= 1e-12 * abs(c)


@pytest.mark.parametrize("dim", [2, 3])
def test_ts3_gradient_finite_difference(dim) -> None:
    st = build_setup(_cfg(dim=dim)).study
    x = _x(st)
    v = np.random.default_rng(7).normal(size=st.n_var)
    ev = st.evaluate(x)
    t = 1e-6
    fd = (st.evaluate(x + t * v, False).f0 - st.evaluate(x - t * v, False).f0) / (2 * t)
    print(f"TS3 {dim}-D: rel {abs(ev.df0 @ v - fd) / abs(fd):.2e}, thermal share "
          f"{st.physics.te.thermal_load(torch.ones(st.domain.n_active), torch.as_tensor(st.physics.T)).norm():.3g}")
    assert abs(ev.df0 @ v - fd) <= 1e-6 * abs(fd)


def test_ts4_yaml_runs() -> None:
    st = build_setup(_cfg()).study
    res = st.run(max_iter=5)
    assert np.isfinite(res.objective) and np.isfinite(res.constraint_values["volume"])


@pytest.mark.parametrize("change", ["two_supports", "oblique", "q4im", "field", "no_alpha"])
def test_ts5_rejects(change, tmp_path) -> None:
    cfg = _cfg()
    if change == "two_supports":
        cfg["load_cases"].append(copy.deepcopy(cfg["load_cases"][0]))
        cfg["load_cases"][1]["supports"] = [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "max"}}]
    elif change == "oblique":
        cfg["supports"].append({"type": "roller", "where": {"type": "point", "at": ["max", "max"]},
                                "normal": [1, 1]})
    elif change == "q4im":
        cfg["physics"]["element"] = "Q4IM"
    elif change == "field":
        from test_element_orientation import ORTHO
        f = tmp_path / "angle.npy"
        np.save(f, np.full((12, 6), 25.0))
        cfg["material"] = {"E0": float(ORTHO["E1"]), "nu": 0.3, "orthotropic": {**ORTHO, "angle_field": str(f)}}
    else:
        del cfg["physics"]["alpha"]
    expected = {"two_supports": "one support set", "oblique": "zero displacements", "q4im": "Q4 and H8",
                "field": "per-element material", "no_alpha": "needs alpha"}[change]
    with pytest.raises((ValueError, ConfigError), match=expected):
        build_setup(cfg)
