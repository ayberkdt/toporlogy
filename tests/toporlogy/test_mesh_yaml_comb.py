"""Load combinations and per-case supports on the mesh YAML path (docs/olcumler/ag_topoloji_onkayit.md, AC1-AC8,
pre-registered 2026-11-26 23:14 +0300 before the code). The shell branch is in test_shell_yaml_comb.py (SC1-SC7);
AC8 (diff predictions) is measured on the commit, not here."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from toporlogy.boundary.constrained import PerCaseSupports
from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config
from toporlogy.optimization.mesh_study import MeshElasticity

from test_mesh_yaml_ext import AX1, _cfg
from test_shell_yaml_ext import SX1, _sx1

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1] / "examples" / "research"
EPS = float(np.finfo(float).eps)

Q4: dict[str, Any] = {"generator": "rectangle", "n": [24, 8], "size": [3.0, 1.0], "element": "Q4", "perturb": 0.02,
                      "seed": 5}
H8: dict[str, Any] = {"generator": "box", "n": [6, 3, 3], "size": [2.0, 1.0, 1.0], "element": "H8"}
RMIN = {"q4": 0.3, "h8": 0.5}
DOM: dict[str, dict[str, Any]] = {"q4": Q4, "h8": H8}
LEFT = [{"set": "left", "fix": "all"}]
RIGHT = [{"set": "right", "fix": "all"}]


def _x(st: Any, seed: int) -> np.ndarray:
    return 0.2 + 0.7 * np.random.default_rng(seed).random(st.n_var)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def _grel(g: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(g - want).max() / np.abs(want).max())


def _with_combinations(cfg: dict[str, Any], combos: list[Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["combinations"] = combos
    return out


# ------------------------------------------------------------------ AC1: a combination = the explicitly summed case
AC1_LOADS = {"q4": ({"point": [3.0, 0.0], "force": [0.0, -1.0]},
                    {"set": "top", "force": [0.3, -0.5], "distribute": "traction"}),
             "h8": ({"point": [2.0, 0.0, 0.0], "force": [0.0, 0.0, -1.0]},
                    {"set": "top", "force": [0.2, 0.0, -0.4], "distribute": "traction"})}


def _scaled(ld: dict[str, Any], f: float) -> dict[str, Any]:
    return {**ld, "force": [f * v for v in ld["force"]]}


@pytest.mark.parametrize("case", list(AC1_LOADS))
def test_ac1_combination_equals_the_explicitly_summed_case(case: str) -> None:
    a, b = AC1_LOADS[case]
    two = [{"name": "a", "weight": 0.7, "loads": [a]}, {"name": "b", "weight": 1.3, "loads": [b]}]
    comb = build_setup(_with_combinations(_cfg(DOM[case], LEFT, two, RMIN[case]),
                                          [{"name": "c", "weight": 0.8, "factors": {"a": 2.0, "b": -0.5}}]))
    explicit = build_setup(_cfg(DOM[case], LEFT, two + [{"name": "c", "weight": 0.8,
                                                         "loads": [_scaled(a, 2.0), _scaled(b, -0.5)]}],
                                RMIN[case]))
    Fc, Fe = np.asarray(comb.study.F, float), np.asarray(explicit.study.F, float)
    rf = float(np.abs(Fc[:, 2] - Fe[:, 2]).max() / np.abs(Fe[:, 2]).max())
    x = _x(comb.study, 41)
    ec, ee = comb.study.evaluate(x), explicit.study.evaluate(x)
    rv, rg = _rel(ec.f0, ee.f0), _grel(ec.df0, ee.df0)
    rows = [(c["name"], c["weight"], c["combination"]) for c in comb.cases]
    print(f"AC1 {case}: F column {rf:.2e}, value {rv:.2e} (c = {ee.f0:.6e}), gradient {rg:.2e}, cases {rows}")
    assert rows == [("a", 0.7, False), ("b", 1.3, False), ("c", 0.8, True)] and Fc.shape[1] == 3
    assert rf <= 1e-15 and rv <= 1e-12 and rg <= 1e-12


# ------------------------------------------------------------------ AC2: per-case supports = sum of one-case studies
AC2_CASES = {"q4": ({"point": [3.0, 0.0], "force": [0.0, -1.0]}, {"point": [0.0, 1.0], "force": [0.0, -1.0]}),
             "h8": ({"point": [2.0, 0.0, 0.0], "force": [0.0, 0.0, -1.0]},
                    {"point": [0.0, 0.0, 1.0], "force": [0.0, 0.0, -1.0]})}


def _cond_bound(st: Any, x: np.ndarray) -> float:
    """eps · max over the support groups of cond(K_ff) at x (dense)."""
    E = st.interp(st.chain(torch.as_tensor(x))).detach().numpy().reshape(-1)
    out = 0.0
    for g, _ in st.physics.groups:
        K = g.problem.pattern.assemble(E[:, None, None] * g.ke)
        Kff = K[g.free][:, g.free].toarray()
        out = max(out, EPS * float(np.linalg.cond(Kff)))
    return out


def _ac2(case: str, wa: float, wb: float) -> tuple[float, float, float]:
    a, b = AC2_CASES[case]
    dom, rmin = DOM[case], RMIN[case]
    two = build_setup(_cfg(dom, LEFT, [{"name": "a", "weight": wa, "loads": [a]},
                                       {"name": "b", "weight": wb, "supports": RIGHT, "loads": [b]}], rmin)).study
    sa = build_setup(_cfg(dom, LEFT, [{"name": "a", "loads": [a]}], rmin)).study
    sb = build_setup(_cfg(dom, RIGHT, [{"name": "b", "loads": [b]}], rmin)).study
    assert isinstance(two.physics, PerCaseSupports) and len(two.physics.groups) == 2
    x = _x(sa, 41)
    e2, ea, eb = two.evaluate(x), sa.evaluate(x), sb.evaluate(x)
    want, gwant = wa * ea.f0 + wb * eb.f0, wa * ea.df0 + wb * eb.df0
    return _rel(e2.f0, want), _grel(e2.df0, gwant), _cond_bound(two, x)


@pytest.mark.parametrize("case", list(AC2_CASES))
def test_ac2_per_case_supports_power_of_two_weights(case: str) -> None:
    rv, rg, bound = _ac2(case, 0.5, 2.0)
    print(f"AC2 (i) {case}: value {rv:.2e}, gradient {rg:.2e} (eps cond {bound:.2e})")
    assert rv <= 1e-12 and rg <= 1e-12


@pytest.mark.parametrize("case", list(AC2_CASES))
def test_ac2_per_case_supports_other_weights_within_the_roundoff_bound(case: str) -> None:
    rv, rg, bound = _ac2(case, 0.7, 1.3)
    print(f"AC2 (ii) {case}: value {rv:.2e}, gradient {rg:.2e}, bound max(1e-12, eps cond) {max(1e-12, bound):.2e}")
    assert rv <= 1e-12 and rg <= max(1e-12, bound)


# ------------------------------------------------------------------ AC3: grouping and one factorisation per set
def test_ac3_one_physics_and_one_factorisation_per_support_set() -> None:
    pa = {"point": [3.0, 0.0], "force": [0.0, -1.0]}
    pb = {"point": [0.0, 1.0], "force": [0.0, -1.0]}
    cases = [{"name": "a", "loads": [pa]}, {"name": "b", "supports": RIGHT, "loads": [pb]},
             {"name": "c", "supports": [{"set": "left", "fix": ["x", "y"]}], "loads": [pb]}]
    s = build_setup(_with_combinations(_cfg(Q4, LEFT, cases), [{"name": "d", "factors": {"a": 1.0, "c": 1.0}}]))
    ph = s.study.physics
    cols = [sorted(c.tolist()) for _, c in ph.groups]
    before = [g.n_solves for g, _ in ph.groups]
    s.study.evaluate(_x(s.study, 41))
    solves = [g.n_solves - n for (g, _), n in zip(ph.groups, before, strict=True)]
    one = build_setup(_cfg(Q4, LEFT, [{"name": "c", "supports": [{"set": "left", "fix": ["x", "y"]}],
                                       "loads": [pa]}])).study.physics
    print(f"AC3: groups {cols}, factorisations per evaluate {solves}, single same-content case {type(one).__name__}")
    assert isinstance(ph, PerCaseSupports) and cols == [[0, 2, 3], [1]] and solves == [1, 1]
    assert type(one) is MeshElasticity


# ------------------------------------------------------------------ AC4: gradient against central differences
def test_ac4_gradient_with_per_case_supports_and_a_combination() -> None:
    cases = [{"name": "a", "weight": 0.7, "loads": [{"point": [3.0, 0.0], "force": [0.0, -1.0]}]},
             {"name": "b", "weight": 1.3, "supports": RIGHT, "loads": [{"point": [0.0, 1.0], "force": [0.0, -1.0]}]},
             {"name": "c", "weight": 0.9, "loads": [{"set": "top", "force": [0.2, -0.3]}]}]
    st = build_setup(_with_combinations(_cfg(Q4, LEFT, cases),
                                        [{"name": "d", "weight": 0.6, "factors": {"a": 1.0, "c": -0.5}}])).study
    x = _x(st, 42)
    v = np.random.default_rng(43).normal(size=st.n_var)
    g = st.evaluate(x).df0 @ v
    s = 1e-3
    f = {k: st.evaluate(x + k * s * v, False).f0 for k in (-2, -1, 1, 2)}
    fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * s)
    rel = abs(g - fd) / abs(fd)
    print(f"AC4: rel {rel:.2e} (directional derivative {g:.6e})")
    assert isinstance(st.physics, PerCaseSupports) and rel <= 1e-6


# ------------------------------------------------------------------ AC5: rigid motions checked per support set
BOTTOM_Y = [{"set": "bottom", "fix": ["y"]}]
PA = {"point": [3.0, 0.0], "force": [0.0, -1.0]}
PB = {"point": [0.0, 1.0], "force": [0.0, -1.0]}


def test_ac5_rigid_motion_is_refused_per_case() -> None:
    with pytest.raises(ConfigError) as err:
        build_setup(_cfg(Q4, LEFT, [{"name": "a", "loads": [PA]},
                                    {"name": "b", "supports": BOTTOM_Y, "loads": [PB]}]))
    hits = [m for p, m in err.value.errors if p == "load_cases.1.supports"]
    print(f"AC5 own supports: {err.value}")
    assert hits and all("rigid motion" in m and "mesh" in m for m in hits), err.value.errors
    ok = build_setup(_cfg(Q4, BOTTOM_Y, [{"name": "a", "supports": LEFT, "loads": [PA]},
                                         {"name": "b", "supports": RIGHT, "loads": [PB]}]))
    print(f"AC5 unused top-level set: built, {len(ok.study.physics.groups)} groups")
    with pytest.raises(ConfigError) as err2:
        build_setup(_cfg(Q4, BOTTOM_Y, [{"name": "a", "loads": [PA]},
                                        {"name": "b", "supports": RIGHT, "loads": [PB]}]))
    hits2 = [m for p, m in err2.value.errors if p == "supports"]
    print(f"AC5 used top-level set: {err2.value}")
    assert hits2 and all("rigid motion" in m for m in hits2), err2.value.errors


# ------------------------------------------------------------------ AC6: invalid inputs
TWO = [{"name": "a", "loads": [PA]}, {"name": "b", "loads": [PB]}]
TWO_OWN = [{"name": "a", "loads": [PA]}, {"name": "b", "supports": RIGHT, "loads": [PB]}]
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
    "zero_combination": ([{"name": "a", "loads": [PA]}, {"name": "a2", "loads": [PA]}],
                         [{"name": "c", "factors": {"a": 1.0, "a2": -1.0}}], "combinations.0", "zero load"),
    "empty_case_supports": ([{"name": "a", "loads": [PA]}, {"name": "b", "supports": [], "loads": [PB]}], None,
                            "load_cases.1.supports", None),
    "case_support_set": ([{"name": "a", "loads": [PA]},
                          {"name": "b", "supports": [{"set": "middle", "fix": "all"}], "loads": [PB]}], None,
                         "load_cases.1.supports.0.set", None),
    "case_support_axis": ([{"name": "a", "loads": [PA]},
                           {"name": "b", "supports": [{"set": "right", "fix": ["z"]}], "loads": [PB]}], None,
                          "load_cases.1.supports.0.fix", None),
}


@pytest.mark.parametrize("case", list(INVALID))
def test_ac6_invalid_inputs_raise(case: str) -> None:
    cases, combos, path, word = INVALID[case]
    cfg = _cfg(Q4, LEFT, cases)
    if combos is not None:
        cfg["combinations"] = combos
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    hits = [m for p, m in err.value.errors if p == path]
    print(f"AC6 {case}: {err.value}")
    assert hits and all("mesh" in m for m in hits), err.value.errors
    if word is not None:
        assert any(word in m for m in hits), err.value.errors


# ------------------------------------------------------------------ AC7: the anchors measured before the code
def _h(a: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def _got(st: Any, x: np.ndarray | None = None) -> tuple[Any, ...]:
    ev = st.evaluate(st.default_start() if x is None else x)
    F = np.asarray(st.F, float)
    return repr(float(ev.f0)), _h(ev.df0), _h(F), F.shape


GRID = {"name": "g", "domain": {"type": "box", "shape": [24, 8], "h": 1.0},
        "supports": [{"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"name": "a", "weight": 0.7, "loads": [{"type": "point", "at": [24, 0], "force": [0, -1]}]},
                       {"name": "b", "weight": 1.3,
                        "supports": [{"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": "max"}}],
                        "loads": [{"type": "point", "at": [0, 8], "force": [0, -1]}]}],
        "combinations": [{"name": "c", "factors": {"a": 2.0}, "weight": 0.5}],
        "optimizer": "mma", "constraints": [{"response": "volume", "upper": 0.4}]}
ANCHOR = {"mesh_cantilever2d": ("1908.3237613774095", "4793a979cd435212", "a627b5d4d61a4200", (1666, 1)),
          "shell_cylinder": ("30131.882810400843", "ce282cdd37f56919", "1581708cfbc900cb", (1326, 1)),
          "ax1_q4": ("455.28516090495594", "07fcc7cfa4c11b66", "f9cf9987d1de49fa", (450, 2)),
          "ax1_h8": ("204.9945594821092", "459c60ad7a8b6822", "9cf26f888e3a2684", (336, 2)),
          "sx1_example": ("11352.463637595167", "75e7cfdc9da9ba7f", "15884aeea9091226", (1326, 2)),
          "grid_per_case": ("3124.069453307912", "8bff2dc9fced752e", "b32d5426148cb843", (450, 3))}


def _anchored(name: str) -> tuple[Any, ...]:
    if name in ("mesh_cantilever2d", "shell_cylinder"):
        return _got(build_setup(read_config(EXAMPLES / f"{name}.yaml")).study)
    if name.startswith("ax1_"):
        dom, sup, a, b, rmin = AX1[name[4:]]
        st = build_setup(_cfg(dom, sup, [{"name": "a", "weight": 0.7, "loads": [a]},
                                         {"name": "b", "weight": 1.3, "loads": [b]}], rmin)).study
        return _got(st, _x(st, 41))
    if name == "sx1_example":
        a, b = SX1["example"]
        st = build_setup(_sx1("example", [{"name": "a", "weight": 0.7, "loads": a},
                                          {"name": "b", "weight": 1.3, "loads": b}])).study
        return _got(st, _x(st, 51))
    st = build_setup(copy.deepcopy(GRID)).study
    assert isinstance(st.physics, PerCaseSupports)
    return _got(st, _x(st, 61))


@pytest.mark.parametrize("name", list(ANCHOR))
def test_ac7_anchored_studies_are_bitwise_unchanged(name: str) -> None:
    got = _anchored(name)
    print(f"AC7 {name}: {got}")
    assert got == ANCHOR[name]
