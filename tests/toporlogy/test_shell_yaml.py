"""YAML for the flat-facet shell domain, the shell branch of ``configuration/mesh_build.py``
(docs/olcumler/kabuk_onkayit.md, SY1-SY4; SY5-SY6 are suite / diff checks). Inputs in the load_cases form since
SX1-SX6 (test_shell_yaml_ext.py); the SY3 rows load_cases and post are supported since then, and a load on a set is
valid (the load_key row now uses an unknown key, edge)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config, resolve
from toporlogy.design.chain import Chain, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.execution.cli import main
from toporlogy.fem.shell import FlatShell, ShellMesh, cylinder_surface, flat_rectangle
from toporlogy.optimization.mesh_study import MeshFilterLink, MeshVolume
from toporlogy.optimization.responses import Compliance
from toporlogy.optimization.shell_study import ShellDomain, ShellPhysics
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "research" / "shell_cylinder.yaml"
C45 = math.cos(math.pi / 4)
FIX = {"u": 0, "v": 1, "w": 2, "rx": 3, "ry": 4, "rz": 5}


def _base() -> dict[str, Any]:
    return read_config(EXAMPLE)


# ------------------------------------------------------------------ SY1: YAML = the same study built in Python
def _cfg(domain: dict[str, Any], supports: list[Any], loads: list[Any], rmin: float, drill: float | None) -> dict:
    physics: Any = "shell" if drill is None else {"type": "shell", "drill": drill}
    return {"name": "s", "domain": {"type": "shell", **domain}, "material": {"E0": 1.0, "nu": 0.3},
            "physics": physics, "supports": supports, "load_cases": [{"loads": loads}],
            "chain": [{"type": "density_filter", "rmin": rmin}],
            "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9}, "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma", "run": {"max_iter": 10}}


CASES: dict[str, dict[str, Any]] = {
    "example": {"mesh": lambda: cylinder_surface(1.0, 1.0, 16, 12, angle=math.pi / 2), "t": 0.05, "drill": None,
                "supports": [("bottom", list(FIX))], "pressure": 0.0,
                "points": [([C45, C45, 1.0], [-C45, -C45, 0.0])], "rmin": 0.15},
    "cylinder_pressure": {"mesh": lambda: cylinder_surface(1.0, 1.0, 12, 8, angle=math.pi / 2), "t": 0.02,
                          "drill": 1e-2, "supports": [("bottom", list(FIX)), ("top", ["u", "v"])], "pressure": 0.5,
                          "points": [([0.0, 1.0, 0.5], [0.0, 0.0, 1.0])], "rmin": 0.2},
    "rectangle": {"mesh": lambda: flat_rectangle(2.0, 1.0, 8, 6), "t": 0.05, "drill": None,
                  "supports": [("left", list(FIX))], "pressure": -1.0, "points": [], "rmin": 0.3},
}


def _yaml_cfg(case: str) -> dict[str, Any]:
    if case == "example":
        return _base()
    c = CASES[case]
    loads: list[Any] = [{"pressure": c["pressure"]}] + [{"point": p, "force": f} for p, f in c["points"]]
    dom = ({"surface": "cylinder", "radius": 1.0, "angle": 90.0, "length": 1.0, "n": [12, 8], "thickness": c["t"]}
           if case == "cylinder_pressure" else
           {"surface": "rectangle", "size": [2.0, 1.0], "n": [8, 6], "thickness": c["t"]})
    return _cfg(dom, [{"set": s, "fix": f} for s, f in c["supports"]], loads, c["rmin"], c["drill"])


def _python_study(case: str) -> Study:
    c = CASES[case]
    m: ShellMesh = c["mesh"]()
    fixed = np.unique(np.concatenate([(6 * m.node_sets[s][:, None] + np.array([FIX[k] for k in f])).ravel()
                                      for s, f in c["supports"]]))
    sh = FlatShell(m, c["t"], 0.3, fixed) if c["drill"] is None else FlatShell(m, c["t"], 0.3, fixed,
                                                                                drill=c["drill"])
    F = np.zeros((sh.ndof, 1))
    if c["pressure"]:
        F[:, 0] += sh.pressure_load(c["pressure"])
    for p, f in c["points"]:
        k = int(np.argmin(np.linalg.norm(m.nodes - np.asarray(p), axis=1)))
        F[6 * k:6 * k + 3, 0] += f
    sd: Any = ShellDomain(m)
    return Study(sd, ShellPhysics(sh), F, Chain([Embed(sd), MeshFilterLink(sd, c["rmin"]), Passive(sd)]),
                 SIMP(sd, E0=1.0, penal=3.0, emin_ratio=1e-9), Compliance(), [Constraint(MeshVolume(sd), 0.4)], MMA())


@pytest.mark.parametrize("case", list(CASES))
def test_sy1_yaml_study_equals_python_study(case: str) -> None:
    ys, ps = build_setup(_yaml_cfg(case)).study, _python_study(case)
    assert np.array_equal(ys.default_start(), ps.default_start())
    x = 0.2 + 0.7 * np.random.default_rng(31).random(ps.n_var)
    ey, ep = ys.evaluate(x), ps.evaluate(x)
    rv = abs(ey.f0 - ep.f0) / abs(ep.f0)
    rg = np.abs(ey.df0 - ep.df0).max() / np.abs(ep.df0).max()
    rvol = abs(ey.values["volume"] - ep.values["volume"]) / abs(ep.values["volume"])
    rdg = np.abs(ey.dg - ep.dg).max() / np.abs(ep.dg).max()
    print(f"SY1 {case}: value {rv:.2e} (c = {ep.f0:.6e}), gradient {rg:.2e}, volume {rvol:.2e}, volume gradient "
          f"{rdg:.2e}")
    assert rv <= 1e-12 and rg <= 1e-12 and rvol <= 1e-12 and rdg <= 1e-12


# ------------------------------------------------------------------ SY2: the example through the CLI
def test_sy2_example_through_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(EXAMPLE)]) == 0
    folder = tmp_path / "run"
    code = main(["run", str(EXAMPLE), "--out", str(folder), "-q"])
    printed = capsys.readouterr().out
    sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hist = [json.loads(ln) for ln in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    vol = sm["constraints"][0]["value"]
    print(f"SY2: exit {code}, status {sm['status']}, {sm['iterations']} it, compliance {hist[0]['f0']:.4e} -> "
          f"{sm['objective']:.4e}, volume {vol:.6f}")
    assert code == (0 if sm["status"] == "ok" else 1)
    assert "RESULT" in printed and np.isfinite(sm["objective"]) and np.isfinite(vol)
    assert sm["objective"] < hist[0]["f0"] and vol <= 0.4 * (1 + 1e-3)
    with np.load(folder / "result.npz") as z:
        assert z["xphys"].shape == (192,)
    resolved = yaml.safe_load((folder / "config.resolved.yaml").read_text(encoding="utf-8"))
    again = resolve(resolved)
    strip = ("provenance", "schema_version", "units")
    assert {k: v for k, v in again.items() if k not in strip} == {k: v for k, v in resolved.items() if k not in strip}
    s1, s2 = build_setup(_base()).study, build_setup(resolved).study
    assert s1.evaluate(s1.default_start()).f0 == s2.evaluate(s2.default_start()).f0


# ------------------------------------------------------------------ SY3: what the shell path does not support
def _with(path: str, value: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(base) if base is not None else _base()
    node: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value
    return cfg


def _closed() -> dict[str, Any]:
    cfg = _base()
    cfg["domain"]["angle"] = None
    return cfg


def _rectangle() -> dict[str, Any]:
    return _yaml_cfg("rectangle")


UNSUPPORTED: dict[str, tuple[str, Any, str, Any]] = {      # id -> (key, value, path in the error, base config)
    "units": ("units", {"length": "mm", "force": "N"}, "units", None),
    "physics_elasticity": ("physics", "elasticity", "physics", None),
    "physics_modal": ("physics", "modal", "physics", None),
    "physics_param": ("physics", {"type": "shell", "shear_factor": 1.0}, "physics.shear_factor", None),
    "helmholtz_link": ("chain", [{"type": "helmholtz_filter", "rmin": 2.0}], "chain.0.type", None),
    "objective_volume": ("objective", "volume", "objective", None),
    "optimizer_oc": ("optimizer", "oc", "optimizer", None),
    "constraint_compliance": ("constraints", [{"response": "compliance", "upper": 1.0}], "constraints.0.response",
                              None),
}
INVALID: dict[str, tuple[str, Any, str, Any]] = {
    "drill_zero": ("physics", {"type": "shell", "drill": 0.0}, "physics.drill", None),
    "drill_negative": ("physics", {"type": "shell", "drill": -1e-3}, "physics.drill", None),
    "surface": ("domain.surface", "sphere", "domain.surface", None),
    "rectangle_radius": ("domain.radius", 1.0, "domain.radius", "rectangle"),
    "cylinder_size": ("domain.size", [1.0, 1.0], "domain.size", None),
    "angle_zero": ("domain.angle", 0.0, "domain.angle", None),
    "angle_360": ("domain.angle", 360.0, "domain.angle", None),
    "n_length": ("domain.n", [16], "domain.n", None),
    "thickness_zero": ("domain.thickness", 0.0, "domain.thickness", None),
    "set_left_on_cylinder": ("supports", [{"set": "left", "fix": "all"}], "supports.0.set", None),
    "set_start_on_tube": ("supports", [{"set": "start", "fix": "all"}], "supports.0.set", "closed"),
    "fix_component": ("supports", [{"set": "bottom", "fix": ["x"]}], "supports.0.fix", None),
    "load_key": ("load_cases.0.loads", [{"edge": "top", "force": [0, 0, 1]}], "load_cases.0.loads.0.edge", None),
    "point_length": ("load_cases.0.loads", [{"point": [1.0, 0.0], "force": [0, 0, 1]}], "load_cases.0.loads.0.point",
                     None),
    "rigid_motion": ("supports", [{"set": "bottom", "fix": ["u"]}], "supports", None),
    "zero_load": ("load_cases.0.loads", [{"point": [1.0, 0.0, 1.0], "force": [0, 0, 0]}], "load_cases.0", None),
    "material_plane": ("material.plane", "stress", "material.plane", None),
}


@pytest.mark.parametrize("case", list(UNSUPPORTED) + list(INVALID))
def test_sy3_unsupported_and_invalid_keys_raise(case: str) -> None:
    key, value, path, base = UNSUPPORTED.get(case) or INVALID[case]
    start = {"rectangle": _rectangle, "closed": _closed}[base]() if base else None
    cfg = _with(key, value, start)
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    hits = [m for p, m in err.value.errors if p == path]
    print(f"SY3 {case}: {err.value}")
    assert hits and all("shell" in m for m in hits), err.value.errors
    if case in UNSUPPORTED:
        assert any("not support" in m for m in hits), err.value.errors


# ------------------------------------------------------------------ SY4: the named node sets of the generators
def test_sy4_named_node_sets() -> None:
    def sel(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask)

    r = flat_rectangle(2.0, 1.0, 4, 3)
    X = r.nodes
    want = {"left": sel(np.abs(X[:, 0]) <= 1e-9), "right": sel(np.abs(X[:, 0] - 2.0) <= 1e-9),
            "bottom": sel(np.abs(X[:, 1]) <= 1e-9), "top": sel(np.abs(X[:, 1] - 1.0) <= 1e-9)}
    ok = sorted(r.node_sets) == sorted(want) and all(np.array_equal(np.sort(r.node_sets[k]), v)
                                                     for k, v in want.items())
    c = cylinder_surface(1.0, 1.0, 8, 6, angle=math.pi / 2)
    X = c.nodes
    th = np.arctan2(X[:, 1], X[:, 0])
    want_c = {"bottom": sel(np.abs(X[:, 2]) <= 1e-9), "top": sel(np.abs(X[:, 2] - 1.0) <= 1e-9),
              "start": sel(np.abs(th) <= 1e-9), "end": sel(np.abs(th - math.pi / 2) <= 1e-9)}
    ok_c = sorted(c.node_sets) == sorted(want_c) and all(np.array_equal(np.sort(c.node_sets[k]), v)
                                                         for k, v in want_c.items())
    t = cylinder_surface(1.0, 1.0, 8, 6)
    X = t.nodes
    want_t = {"bottom": sel(np.abs(X[:, 2]) <= 1e-9), "top": sel(np.abs(X[:, 2] - 1.0) <= 1e-9)}
    ok_t = sorted(t.node_sets) == sorted(want_t) and all(np.array_equal(np.sort(t.node_sets[k]), v)
                                                         for k, v in want_t.items())
    print(f"SY4: rectangle {ok} ({ {k: len(v) for k, v in r.node_sets.items()} }), segment {ok_c} "
          f"({ {k: len(v) for k, v in c.node_sets.items()} }), tube {ok_t} ({sorted(t.node_sets)})")
    assert ok and ok_c and ok_t


def test_shell_mesh_refuses_a_bad_node_set() -> None:
    """Post hoc (the field is new): a node set out of range is refused."""
    m = flat_rectangle(1.0, 1.0, 2, 2)
    with pytest.raises(ValueError, match="node set"):
        ShellMesh(m.nodes, m.facets, {"edge": np.array([0, 99])})
