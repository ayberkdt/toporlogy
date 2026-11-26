"""YAML for a general mesh through its own build path, ``configuration/mesh_build.py``
(docs/olcumler/ag_topoloji_onkayit.md, AY1-AY4; AY5-AY6 are suite / diff checks). Inputs in the load_cases form since
AX1-AX8 (test_mesh_yaml_ext.py); the AY4 rows load_cases, post vtu and plane strain are supported since then."""
from __future__ import annotations

import copy
import json
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
from toporlogy.fem.elements import D_iso, D_plane_stress
from toporlogy.fem.mesh import box, perturb, rectangle
from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity, MeshFilterLink, MeshVolume
from toporlogy.optimization.responses import Compliance
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "research" / "mesh_cantilever2d.yaml"


def _base() -> dict[str, Any]:
    return read_config(EXAMPLE)


def _mesh_cfg(domain: dict[str, Any], supports: list[Any], loads: list[Any], rmin: float) -> dict[str, Any]:
    return {"name": "t", "domain": {"type": "mesh", **domain}, "material": {"E0": 1.0, "nu": 0.3},
            "supports": supports, "load_cases": [{"loads": loads}], "chain": [{"type": "density_filter", "rmin": rmin}],
            "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9}, "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma", "run": {"max_iter": 10}}


# ------------------------------------------------------------------ AY1: YAML = the same study built in Python
def _python_study(case: str) -> Study:
    if case == "example_q4":
        m = perturb(rectangle(48, 16, 3.0, 1.0, "Q4"), 0.01, seed=0)
        fixed = m.dofs(m.node_sets["left"], [0, 1])
        F = np.zeros((2 * m.n_nodes, 1))
        k = int(np.argmin(np.linalg.norm(m.points - [3.0, 0.0], axis=1)))
        F[2 * k + 1, 0] = -1.0
        rmin, D = 0.125, D_plane_stress(0.3)
    elif case == "t3":
        m = perturb(rectangle(24, 8, 3.0, 1.0, "T3"), 0.02, seed=3)
        fixed = m.dofs(m.node_sets["left"], [0, 1])
        F = np.zeros((2 * m.n_nodes, 1))
        nodes = m.node_sets["right"]
        F[2 * nodes + 1, 0] = -1.0 / len(nodes)
        rmin, D = 0.3, D_plane_stress(0.3)
    else:
        m = box(6, 3, 3, 2.0, 1.0, 1.0, "H8")
        fixed = m.dofs(m.node_sets["left"])
        F = np.zeros((3 * m.n_nodes, 1))
        nodes = m.node_sets["right"]
        F[3 * nodes + 2, 0] = -1.0 / len(nodes)
        rmin, D = 0.5, D_iso(0.3)
    md = MeshDomain(m)
    return Study(md, MeshElasticity(md, D, fixed), F, Chain([Embed(md), MeshFilterLink(md, rmin), Passive(md)]),
                 SIMP(md, E0=1.0, penal=3.0, emin_ratio=1e-9), Compliance(), [Constraint(MeshVolume(md), 0.4)], MMA())


def _yaml_cfg(case: str) -> dict[str, Any]:
    if case == "example_q4":
        return _base()
    if case == "t3":
        return _mesh_cfg({"generator": "rectangle", "n": [24, 8], "size": [3.0, 1.0], "element": "T3",
                          "perturb": 0.02, "seed": 3}, [{"set": "left", "fix": ["x", "y"]}],
                         [{"set": "right", "force": [0, -1]}], 0.3)
    return _mesh_cfg({"generator": "box", "n": [6, 3, 3], "size": [2.0, 1.0, 1.0], "element": "H8"},
                     [{"set": "left", "fix": "all"}], [{"set": "right", "force": [0, 0, -1]}], 0.5)


@pytest.mark.parametrize("case", ["example_q4", "t3", "h8"])
def test_ay1_yaml_study_equals_python_study(case: str) -> None:
    ys, ps = build_setup(_yaml_cfg(case)).study, _python_study(case)
    assert np.array_equal(ys.default_start(), ps.default_start())
    x = 0.2 + 0.7 * np.random.default_rng(21).random(ps.n_var)
    ey, ep = ys.evaluate(x), ps.evaluate(x)
    rv = abs(ey.f0 - ep.f0) / abs(ep.f0)
    rg = np.abs(ey.df0 - ep.df0).max() / np.abs(ep.df0).max()
    rvol = abs(ey.values["volume"] - ep.values["volume"]) / abs(ep.values["volume"])
    rdg = np.abs(ey.dg - ep.dg).max() / np.abs(ep.dg).max()
    print(f"AY1 {case}: value {rv:.2e}, gradient {rg:.2e}, volume {rvol:.2e}, volume gradient {rdg:.2e}")
    assert rv <= 1e-12 and rg <= 1e-12 and rvol <= 1e-12 and rdg <= 1e-12


# ------------------------------------------------------------------ AY2: unperturbed Q4 mesh = grid YAML cantilever
GRID = {"name": "g", "domain": {"type": "box", "shape": [24, 8], "h": 0.125}, "material": {"E0": 1.0, "nu": 0.3},
        "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                                  "force": [0, -1]}]}],
        "chain": [{"type": "density_filter", "rmin": 2.0}],
        "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9}, "objective": "compliance",
        "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma", "run": {"max_iter": 10}}


def _grid_and_mesh() -> tuple[Study, Study]:
    mesh = _mesh_cfg({"generator": "rectangle", "n": [24, 8], "size": [3.0, 1.0], "element": "Q4"},
                     [{"set": "left", "fix": ["x", "y"]}], [{"point": [3.0, 0.0], "force": [0, -1]}], 0.25)
    return build_setup(copy.deepcopy(GRID)).study, build_setup(mesh).study


@pytest.mark.xfail(strict=True, reason="pre-registered AY2 FAILED on the first run: centroids 0, value 9.6e-13 "
                   "(start) / 2.0e-13 (random), gradient 2.2e-12 (start, > 1e-12) / 5.8e-13 (random), 10-iteration MMA "
                   "history 1.46e-8 (> 1e-8). cond(K) = 3.0e4 at x = 0.4, eps cond = 6.7e-12: two direct solvers "
                   "(grid splu, mesh Factorization) differ at the roundoff level; thresholds unchanged")
def test_ay2_unperturbed_mesh_equals_grid_yaml() -> None:
    g, m = _grid_and_mesh()
    dc = np.abs(m.domain.centroids - g.domain.element_centers()[g.domain.active_elements]).max()
    out = {"centroids": float(dc)}
    for label, x in (("start", g.default_start()), ("random", 0.2 + 0.7 * np.random.default_rng(22).random(g.n_var))):
        eg, em = g.evaluate(x), m.evaluate(x)
        out[f"{label} value"] = abs(em.f0 - eg.f0) / abs(eg.f0)
        out[f"{label} gradient"] = float(np.abs(em.df0 - eg.df0).max() / np.abs(eg.df0).max())
    hg, hm = g.run(max_iter=10, tol=0.0).history["f0"], m.run(max_iter=10, tol=0.0).history["f0"]
    out["history"] = max(abs(a - b) / abs(a) for a, b in zip(hg, hm, strict=True))
    print("AY2:", {k: f"{v:.2e}" for k, v in out.items()}, f"({len(hg)} iterations)")
    assert out["centroids"] <= 1e-12
    assert all(out[k] <= 1e-12 for k in out if k.endswith(("value", "gradient")))
    assert len(hg) == len(hm) == 10 and out["history"] <= 1e-8


def test_ay2_posthoc_differences_are_within_the_solve_roundoff_bound() -> None:
    """Post hoc (after AY2 failed): at the default start the relative value and gradient differences of the grid and
    the mesh YAML study stay below eps · cond(K), the forward-error scale of a backward-stable solve."""
    import torch
    g, m = _grid_and_mesh()
    x = m.default_start()
    eg, em = g.evaluate(x), m.evaluate(x)
    ph = m.physics
    E = m.interp(torch.as_tensor(x)).detach().numpy().ravel()
    K = ph.problem.pattern.assemble(E[:, None, None] * ph.ke)[ph.free][:, ph.free]
    bound = np.finfo(float).eps * np.linalg.cond(K.toarray())
    rv = abs(em.f0 - eg.f0) / abs(eg.f0)
    rg = np.abs(em.df0 - eg.df0).max() / np.abs(eg.df0).max()
    print(f"AY2 post hoc: value {rv:.2e}, gradient {rg:.2e}, eps cond(K) {bound:.2e}; default starts differ by "
          f"{np.abs(g.default_start() - x).max():.1e}")
    assert rv <= bound and rg <= bound


# ------------------------------------------------------------------ AY3: the example through the CLI
def test_ay3_example_through_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(EXAMPLE)]) == 0
    folder = tmp_path / "run"
    code = main(["run", str(EXAMPLE), "--out", str(folder), "-q"])
    printed = capsys.readouterr().out
    sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    hist = [json.loads(ln) for ln in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    vol = sm["constraints"][0]["value"]
    print(f"AY3: exit {code}, status {sm['status']}, {sm['iterations']} it, compliance {hist[0]['f0']:.4e} -> "
          f"{sm['objective']:.4e}, volume {vol:.6f}")
    assert code == (0 if sm["status"] == "ok" else 1)
    assert "RESULT" in printed and np.isfinite(sm["objective"]) and np.isfinite(vol)
    assert sm["objective"] < hist[0]["f0"] and vol <= 0.4 * (1 + 1e-3)
    with np.load(folder / "result.npz") as z:
        assert z["xphys"].shape == (768,)
    resolved = yaml.safe_load((folder / "config.resolved.yaml").read_text(encoding="utf-8"))
    again = resolve(resolved)
    strip = ("provenance", "schema_version", "units")
    assert {k: v for k, v in again.items() if k not in strip} == {k: v for k, v in resolved.items() if k not in strip}
    s1, s2 = build_setup(_base()).study, build_setup(resolved).study
    assert s1.evaluate(s1.default_start()).f0 == s2.evaluate(s2.default_start()).f0


# ------------------------------------------------------------------ AY4: what the mesh path does not support
def _with(path: str, value: Any) -> dict[str, Any]:
    cfg = _base()
    node: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value
    return cfg


UNSUPPORTED: dict[str, tuple[str, Any, str]] = {      # id -> (config key to set, value, path named in the error)
    "robust": ("robust", {"eta_e": 0.7}, "robust"),
    "connectivity": ("connectivity", {}, "connectivity"),
    "schedule": ("schedule", {"type": "beta_continuation"}, "schedule"),
    "units": ("units", {"length": "mm", "force": "N"}, "units"),
    "heaviside_link": ("chain", [{"type": "density_filter", "rmin": 0.1}, {"type": "heaviside_projection"}],
                       "chain.1.type"),
    "helmholtz_link": ("chain", [{"type": "helmholtz_filter", "rmin": 2.0}], "chain.0.type"),
    "objective_volume": ("objective", "volume", "objective"),
    "constraint_compliance": ("constraints", [{"response": "compliance", "upper": 1.0}], "constraints.0.response"),
    "volume_weighted": ("constraints", [{"response": {"type": "volume", "weighted": True}, "upper": 0.4}],
                        "constraints.0.response.weighted"),
    "optimizer_oc": ("optimizer", "oc", "optimizer"),
    "physics_modal": ("physics", "modal", "physics"),
    "physics_element": ("physics", {"type": "elasticity", "element": "Q4IM"}, "physics.element"),
    "interpolation_ramp": ("interpolation", "ramp", "interpolation"),
    "orthotropic": ("material.orthotropic", {"E1": 1.0}, "material.orthotropic"),
    "strength": ("material.strength", {"tension": 1.0}, "material.strength"),
    "phases": ("material.phases", [{"name": "a", "E": 1.0}], "material.phases"),
    "rho": ("material.rho", 1.0, "material.rho"),
    "run_start": ("run.start", {"type": "random", "amplitude": 0.1}, "run.start"),
    "run_checkpoint": ("run.checkpoint_every", 5, "run.checkpoint_every"),
    "run_move_beta_ref": ("run.move_beta_ref", 4.0, "run.move_beta_ref"),
}
INVALID: dict[str, tuple[str, Any, str]] = {
    "unknown_support_set": ("supports", [{"set": "middle", "fix": ["x"]}], "supports.0.set"),
    "unknown_load_set": ("load_cases.0.loads", [{"set": "nowhere", "force": [0, -1]}], "load_cases.0.loads.0.set"),
    "element_dimension": ("domain.element", "H8", "domain.element"),
    "n_length": ("domain.n", [48], "domain.n"),
    "size_length": ("domain.size", [3.0, 1.0, 1.0], "domain.size"),
    "generator": ("domain.generator", "disk", "domain.generator"),
    "zero_load": ("load_cases.0.loads", [{"point": [3.0, 0.0], "force": [0, 0]}], "load_cases.0"),
    "rigid_motion": ("supports", [{"set": "left", "fix": ["x"]}], "supports"),
    "chain_missing": ("chain", None, "chain"),
    "unknown_top_key": ("symmetry", {"x": 1.5}, "symmetry"),
}


@pytest.mark.parametrize("case", list(UNSUPPORTED) + list(INVALID))
def test_ay4_unsupported_and_invalid_keys_raise(case: str) -> None:
    key, value, path = UNSUPPORTED.get(case) or INVALID[case]
    cfg = _with(key, value)
    if value is None:
        cfg.pop(key)
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    hits = [m for p, m in err.value.errors if p == path]
    print(f"AY4 {case}: {err.value}")
    assert hits and all("mesh" in m for m in hits), err.value.errors
    if case in UNSUPPORTED:
        assert any("not support" in m for m in hits), err.value.errors
