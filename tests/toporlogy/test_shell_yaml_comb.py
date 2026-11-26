"""Load combinations and per-case supports on the shell YAML path (docs/olcumler/kabuk_onkayit.md, SC1-SC7,
pre-registered 2026-11-26 23:14 +0300 before the code). The shared parts (anchors SC7, diff predictions) are in
test_mesh_yaml_comb.py (AC7, AC8)."""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest
import torch

import toporlogy.fem.shell as shell_module
from toporlogy.boundary.constrained import PerCaseSupports
from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config

from test_shell_yaml_ext import EXAMPLE, RADIAL, RECT, _cfg

C45 = math.cos(math.pi / 4)
EPS = float(np.finfo(float).eps)
RADIAL0 = {"point": [C45, C45, 0.0], "force": [-C45, -C45, 0.0]}
ALL = "all"


def _segment(cases: list[Any], supports: list[Any] | None = None,
             combos: list[Any] | None = None) -> dict[str, Any]:
    cfg = read_config(EXAMPLE)
    cfg["load_cases"] = cases
    if supports is not None:
        cfg["supports"] = supports
    if combos is not None:
        cfg["combinations"] = combos
    return cfg


def _rect(cases: list[Any], supports: list[Any], combos: list[Any] | None = None) -> dict[str, Any]:
    cfg = _cfg(RECT, supports, cases)
    if combos is not None:
        cfg["combinations"] = combos
    return cfg


def _x(st: Any, seed: int) -> np.ndarray:
    return 0.2 + 0.7 * np.random.default_rng(seed).random(st.n_var)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def _grel(g: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(g - want).max() / np.abs(want).max())


def _scaled(ld: dict[str, Any], f: float) -> dict[str, Any]:
    if "pressure" in ld:
        return {"pressure": f * ld["pressure"]}
    return {**ld, "force": [f * v for v in ld["force"]]}


# ------------------------------------------------------------------ SC1: a combination = the explicitly summed case
SC1: dict[str, tuple[list[Any], list[Any]]] = {
    "segment": ([RADIAL], [{"pressure": 0.5}, {"set": "start", "force": [0.0, 0.2, 0.0], "distribute": "traction"}]),
    "rectangle": ([{"point": [2.0, 1.0, 0.0], "force": [0.0, 0.0, -1.0]}], [{"pressure": 1.0}])}


def _sc1_cfg(case: str, cases: list[Any], combos: list[Any] | None = None) -> dict[str, Any]:
    if case == "segment":
        return _segment(cases, combos=combos)
    return _rect(cases, [{"set": "left", "fix": ALL}], combos)


@pytest.mark.parametrize("case", list(SC1))
def test_sc1_combination_equals_the_explicitly_summed_case(case: str) -> None:
    a, b = SC1[case]
    two = [{"name": "a", "weight": 0.7, "loads": a}, {"name": "b", "weight": 1.3, "loads": b}]
    comb = build_setup(_sc1_cfg(case, copy.deepcopy(two),
                                [{"name": "c", "weight": 0.8, "factors": {"a": 2.0, "b": -0.5}}]))
    c_loads = [_scaled(ld, 2.0) for ld in a] + [_scaled(ld, -0.5) for ld in b]
    explicit = build_setup(_sc1_cfg(case, copy.deepcopy(two) + [{"name": "c", "weight": 0.8, "loads": c_loads}]))
    Fc, Fe = np.asarray(comb.study.F, float), np.asarray(explicit.study.F, float)
    rf = float(np.abs(Fc[:, 2] - Fe[:, 2]).max() / np.abs(Fe[:, 2]).max())
    x = _x(comb.study, 51)
    ec, ee = comb.study.evaluate(x), explicit.study.evaluate(x)
    rv, rg = _rel(ec.f0, ee.f0), _grel(ec.df0, ee.df0)
    rows = [(c["name"], c["weight"], c["combination"]) for c in comb.cases]
    print(f"SC1 {case}: F column {rf:.2e}, value {rv:.2e} (c = {ee.f0:.6e}), gradient {rg:.2e}, cases {rows}")
    assert rows == [("a", 0.7, False), ("b", 1.3, False), ("c", 0.8, True)] and Fc.shape[1] == 3
    assert rf <= 1e-15 and rv <= 1e-12 and rg <= 1e-12


# ------------------------------------------------------------------ SC2: per-case supports = sum of one-case studies
def _sc2_cfg(case: str, cases: list[Any], supports: list[Any]) -> dict[str, Any]:
    return _segment(cases, supports) if case == "segment" else _rect(cases, supports)


SC2: dict[str, tuple[list[Any], list[Any], dict[str, Any], dict[str, Any]]] = {
    "segment": ([{"set": "bottom", "fix": ALL}], [{"set": "top", "fix": ALL}], RADIAL, RADIAL0),
    "rectangle": ([{"set": "left", "fix": ALL}], [{"set": "right", "fix": ALL}],
                  {"point": [2.0, 1.0, 0.0], "force": [0.0, 0.0, -1.0]},
                  {"point": [0.0, 1.0, 0.0], "force": [0.0, 0.0, -1.0]})}


def _cond_bound(st: Any, x: np.ndarray) -> float:
    E = st.interp(st.chain(torch.as_tensor(x))).detach().numpy().reshape(-1)
    out = 0.0
    for g, _ in st.physics.groups:
        sh = g.shell
        K = sh.stiffness(E).toarray()[np.ix_(sh.free, sh.free)]
        out = max(out, EPS * float(np.linalg.cond(K)))
    return out


def _sc2(case: str, wa: float, wb: float) -> tuple[float, float, float]:
    sup_a, sup_b, la, lb = SC2[case]
    two = build_setup(_sc2_cfg(case, [{"name": "a", "weight": wa, "loads": [la]},
                                      {"name": "b", "weight": wb, "supports": sup_b, "loads": [lb]}], sup_a)).study
    sa = build_setup(_sc2_cfg(case, [{"name": "a", "loads": [la]}], sup_a)).study
    sb = build_setup(_sc2_cfg(case, [{"name": "b", "loads": [lb]}], sup_b)).study
    assert isinstance(two.physics, PerCaseSupports) and len(two.physics.groups) == 2
    x = _x(sa, 51)
    e2, ea, eb = two.evaluate(x), sa.evaluate(x), sb.evaluate(x)
    want, gwant = wa * ea.f0 + wb * eb.f0, wa * ea.df0 + wb * eb.df0
    return _rel(e2.f0, want), _grel(e2.df0, gwant), _cond_bound(two, x)


@pytest.mark.parametrize("case", list(SC2))
def test_sc2_per_case_supports_power_of_two_weights(case: str) -> None:
    rv, rg, bound = _sc2(case, 0.5, 2.0)
    print(f"SC2 (i) {case}: value {rv:.2e}, gradient {rg:.2e} (eps cond {bound:.2e})")
    assert rv <= 1e-12 and rg <= 1e-12


@pytest.mark.parametrize("case", list(SC2))
def test_sc2_per_case_supports_other_weights_within_the_roundoff_bound(case: str) -> None:
    rv, rg, bound = _sc2(case, 0.7, 1.3)
    print(f"SC2 (ii) {case}: value {rv:.2e}, gradient {rg:.2e}, bound max(1e-12, eps cond) {max(1e-12, bound):.2e}")
    assert rv <= 1e-12 and rg <= max(1e-12, bound)


# ------------------------------------------------------------------ SC3: grouping and one factorisation per set
def test_sc3_one_shell_and_one_factorisation_per_support_set(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [{"name": "a", "loads": [RADIAL]}, {"name": "b", "supports": [{"set": "top", "fix": ALL}],
                                                "loads": [RADIAL0]},
             {"name": "c", "supports": [{"set": "bottom", "fix": ALL}], "loads": [{"pressure": 0.5}]}]
    st = build_setup(_segment(cases)).study
    ph = st.physics
    cols = [sorted(c.tolist()) for _, c in ph.groups]
    calls = []
    real = shell_module.spla.splu

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(shell_module.spla, "splu", counted)
    st.evaluate(_x(st, 52))
    print(f"SC3: groups {cols}, splu calls in one evaluate {len(calls)}")
    assert isinstance(ph, PerCaseSupports) and cols == [[0, 2], [1]] and len(calls) == 2


# ------------------------------------------------------------------ SC4: gradient against central differences
def test_sc4_gradient_on_a_curved_shell_with_per_case_supports_and_a_combination() -> None:
    cases = [{"name": "a", "weight": 0.7, "loads": [RADIAL]},
             {"name": "b", "weight": 1.3, "supports": [{"set": "top", "fix": ALL}], "loads": [RADIAL0]},
             {"name": "c", "weight": 0.9, "loads": [{"pressure": 0.5}]}]
    st = build_setup(_segment(cases, combos=[{"name": "d", "weight": 0.6, "factors": {"a": 1.0, "c": -0.5}}])).study
    x = _x(st, 42)
    v = np.random.default_rng(43).normal(size=st.n_var)
    g = st.evaluate(x).df0 @ v
    s = 1e-3
    f = {k: st.evaluate(x + k * s * v, False).f0 for k in (-2, -1, 1, 2)}
    fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * s)
    rel = abs(g - fd) / abs(fd)
    print(f"SC4: rel {rel:.2e} (directional derivative {g:.6e})")
    assert isinstance(st.physics, PerCaseSupports) and rel <= 1e-6


# ------------------------------------------------------------------ SC5: rigid motions checked per support set
TOP_W = [{"set": "top", "fix": ["w"]}]


def test_sc5_rigid_motion_is_refused_per_case() -> None:
    with pytest.raises(ConfigError) as err:
        build_setup(_segment([{"name": "a", "loads": [RADIAL]},
                              {"name": "b", "supports": TOP_W, "loads": [RADIAL0]}]))
    hits = [m for p, m in err.value.errors if p == "load_cases.1.supports"]
    print(f"SC5 own supports: {err.value}")
    assert hits and all("rigid motion" in m and "shell" in m for m in hits), err.value.errors
    ok = build_setup(_segment([{"name": "a", "supports": [{"set": "bottom", "fix": ALL}], "loads": [RADIAL]},
                               {"name": "b", "supports": [{"set": "top", "fix": ALL}], "loads": [RADIAL0]}], TOP_W))
    print(f"SC5 unused top-level set: built, {len(ok.study.physics.groups)} groups")
    with pytest.raises(ConfigError) as err2:
        build_setup(_segment([{"name": "a", "loads": [RADIAL]},
                              {"name": "b", "supports": [{"set": "top", "fix": ALL}], "loads": [RADIAL0]}], TOP_W))
    hits2 = [m for p, m in err2.value.errors if p == "supports"]
    print(f"SC5 used top-level set: {err2.value}")
    assert hits2 and all("rigid motion" in m for m in hits2), err2.value.errors


# ------------------------------------------------------------------ SC6: invalid inputs
TWO = [{"name": "a", "loads": [RADIAL]}, {"name": "b", "loads": [{"pressure": 0.5}]}]
TWO_OWN = [{"name": "a", "loads": [RADIAL]}, {"name": "b", "supports": [{"set": "top", "fix": ALL}],
                                              "loads": [RADIAL0]}]
INVALID: dict[str, tuple[list[Any], list[Any] | None, str, str | None]] = {
    "unknown_case": (TWO, [{"name": "c", "factors": {"x": 1.0}}], "combinations.0.factors.x", None),
    "no_name": (TWO, [{"factors": {"a": 1.0}}], "combinations.0.name", None),
    "case_name": (TWO, [{"name": "a", "factors": {"a": 1.0}}], "combinations.0.name", None),
    "empty_factors": (TWO, [{"name": "c", "factors": {}}], "combinations.0.factors", None),
    "infinite_factor": (TWO, [{"name": "c", "factors": {"a": float("inf")}}], "combinations.0.factors.a", None),
    "infinite_weight": (TWO, [{"name": "c", "weight": float("inf"), "factors": {"a": 1.0}}],
                        "combinations.0.weight", None),
    "unknown_key": (TWO, [{"name": "c", "factors": {"a": 1.0}, "scale": 2.0}], "combinations.0.scale", None),
    "mixed_supports": (TWO_OWN, [{"name": "c", "factors": {"a": 1.0, "b": 1.0}}], "combinations.0",
                       "different supports"),
    "zero_combination": ([{"name": "a", "loads": [RADIAL]}, {"name": "a2", "loads": [RADIAL]}],
                         [{"name": "c", "factors": {"a": 1.0, "a2": -1.0}}], "combinations.0", "zero load"),
    "empty_case_supports": ([{"name": "a", "loads": [RADIAL]}, {"name": "b", "supports": [], "loads": [RADIAL0]}],
                            None, "load_cases.1.supports", None),
    "case_support_set": ([{"name": "a", "loads": [RADIAL]},
                          {"name": "b", "supports": [{"set": "left", "fix": ALL}], "loads": [RADIAL0]}], None,
                         "load_cases.1.supports.0.set", None),
    "case_support_component": ([{"name": "a", "loads": [RADIAL]},
                                {"name": "b", "supports": [{"set": "top", "fix": ["x"]}], "loads": [RADIAL0]}], None,
                               "load_cases.1.supports.0.fix", None),
}


@pytest.mark.parametrize("case", list(INVALID))
def test_sc6_invalid_inputs_raise(case: str) -> None:
    cases, combos, path, word = INVALID[case]
    with pytest.raises(ConfigError) as err:
        build_setup(_segment(copy.deepcopy(cases), combos=combos))
    hits = [m for p, m in err.value.errors if p == path]
    print(f"SC6 {case}: {err.value}")
    assert hits and all("shell" in m for m in hits), err.value.errors
    if word is not None:
        assert any(word in m for m in hits), err.value.errors
