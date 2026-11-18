"""Feature modules through the declarative config / CLI: every registered feature component builds from a minimal
config (and responses evaluate), boundary-condition check in validate (summary, exit 2 when under-supported), the
connectivity block (ties, constraint, gate verdict in summary.json, bridge repair), the robust block (rmin from a
length), and a small finalize / checks / export post chain. Problems are tiny (CPU shared with other jobs)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from toporlogy.configuration.build import Setup, build_setup
from toporlogy.configuration.resolve import resolve
from toporlogy.execution.runner import run_setup
from toporlogy.execution.cli import main
from toporlogy.configuration.registry import BuildContext, ConfigError, REGISTRY


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def plate(**over) -> dict:
    """2-D plate 16 x 8 (h 0.5) with a hole r 2 at (10, 4), clamped at x = 0, tip load at x = 16."""
    raw = {
        "name": "plate",
        "domain": {"type": "csg", "h": 0.5, "shapes": [{"type": "box", "lo": [0, 0], "hi": [16, 8]},
                                                      {"type": "ball", "center": [10, 4], "radius": 2,
                                                       "op": "cut"}]},
        "supports": [{"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": "min"}}],
        "load_cases": [{"name": "tip", "loads": [{"type": "surface_force",
                                                  "where": {"type": "on_plane", "axis": "x", "at": "max"},
                                                  "force": [0, -1]}]}],
        "chain": [{"type": "density_filter", "rmin": 1.5}],
        "constraints": [{"response": "volume", "upper": 0.5}],
        "optimizer": "mma",
        "run": {"max_iter": 2},
    }
    raw.update(over)
    return raw


def box2d(**over) -> dict:
    raw = {"name": "b", "domain": {"type": "box", "shape": [12, 6]},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "point", "at": ["max", "min"], "force": [0, -1]}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}],
           "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma", "run": {"max_iter": 2}}
    raw.update(over)
    return raw


def evaluate(s: Setup) -> None:
    ev = s.study.evaluate(s.study.default_start())
    assert np.isfinite(ev.f0) and np.all(np.isfinite(ev.g)) and np.all(np.isfinite(ev.df0))


# ---------------------------------------------------------------------------------------------- shapes, domains
@pytest.mark.parametrize("spec", [
    {"type": "box", "lo": [0, 0, 0], "hi": [1, 2, 3]}, {"type": "ball", "center": [0, 0], "radius": 1},
    {"type": "ball", "center": [0, 0, 0], "radius": 1},
    {"type": "cylinder", "p0": [0, 0, 0], "p1": [0, 0, 2], "radius": 1},
    {"type": "capsule", "p0": [0, 0], "p1": [2, 0], "radius": 0.5},
    {"type": "halfspace", "point": [0, 0], "normal": [1, 0]},
    {"type": "polygon", "vertices": [[0, 0], [2, 0], [0, 2]]},
    {"type": "prism", "polygon": [[0, 0], [2, 0], [0, 2]], "axis": "z", "range": [0, 1]},
    {"type": "slot", "p0": [0, 0, 0], "p1": [2, 0, 0], "radius": 0.5, "depth": [0, 0, 1]}])
def test_shapes_build(spec):
    errors: list = []
    r = REGISTRY.resolve("shape", spec, "s", errors)
    assert not errors
    s = REGISTRY.build("shape", r, BuildContext(), "s")
    assert s.op in ("add", "intersect") and np.isfinite(s.sdf(np.zeros((1, s.dim)))).all()


def test_csg_and_stl_domains(tmp_path):
    from toporlogy.geometry import Box, write_stl
    s = build_setup(plate())
    d = s.study.domain
    assert d.shape == (32, 16) and d.frac[d.active].min() < 1 and d.n_active < 32 * 16   # hole, partial volumes
    V = np.array([[0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0], [0, 0, 2], [4, 0, 2], [4, 2, 2], [0, 2, 2]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    write_stl(tmp_path / "b.stl", V, F)
    assert Box([0, 0, 0], [1, 1, 1]).dim == 3
    raw = box2d(domain={"type": "stl", "path": str(tmp_path / "b.stl"), "h": 0.5, "padding": 0,
                        "keep": [{"type": "box", "lo": [3, 0, 0], "hi": [4, 2, 2]}]},
                supports=[{"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": 0}}],
                load_cases=[{"loads": [{"type": "surface_force", "where": {"type": "on_plane", "axis": "x",
                                                                           "at": 4}, "force": [0, 0, -1]}]}])
    s = build_setup(raw)
    d = s.study.domain
    assert d.shape == (8, 4, 4) and d.keep.sum() == 2 * 4 * 4
    assert abs(d.frac.sum() * d.h ** 3 - 16.0) < 1e-6


# ---------------------------------------------------------------------------------------------- selectors, supports, loads
SELECTORS = [
    {"type": "on_plane", "axis": "x", "at": "max"},
    {"type": "on_plane", "point": [16, 0], "normal": [1, 0]},
    {"type": "in_box", "lo": [15, None], "hi": [None, None]},
    {"type": "half_space", "point": [15.5, 0], "normal": [-1, 0], "surface": True},
    {"type": "in_sphere", "center": [16, 4], "radius": 1.0, "surface": True},
    {"type": "in_shape", "shape": {"type": "box", "lo": [15.5, 0], "hi": [17, 8]}, "surface": True},
    {"type": "at_points", "points": [[16, 4], [16, 3.5]]},
    {"type": "nearest", "points": [[16.2, 4]], "k": 3},
    {"type": "node_ids", "ids": [32 * 17 + 8]},
    {"type": "union", "of": [{"type": "on_plane", "axis": "x", "at": "max"}, {"type": "point", "at": [16, 8]}]},
    {"type": "intersect", "of": [{"type": "surface"}, {"type": "in_box", "lo": [15.9, 0], "hi": [16, 8]}]},
    {"type": "subtract", "of": [{"type": "on_plane", "axis": "x", "at": "max"},
                                {"type": "in_box", "lo": [0, 0], "hi": [16, 1]}]},
    {"type": "bore", "index": 1}, {"type": "bore", "near": [10, 4]},
    {"type": "on_cylinder", "p0": [10, 4], "radius": 2},
]


@pytest.mark.parametrize("sel", SELECTORS, ids=lambda s: s["type"] + str(len(s)))
def test_selectors_as_load_region(sel):
    raw = plate()
    raw["load_cases"][0]["loads"] = [{"type": "force", "where": sel, "force": [0, -1]}]
    s = build_setup(raw)
    assert abs(s.cases[0]["resultant"][1] + 1) < 1e-12 and s.cases[0]["n_nodes"] >= 1


def test_all_and_surface_selectors():
    raw = plate()
    raw["load_cases"][0]["loads"] = [{"type": "force", "where": "surface", "force": [0, -1]},
                                     {"type": "force", "where": "all", "force": [0, -1]}]
    s = build_setup(raw)
    assert s.cases[0]["n_nodes"] == len(s.study.domain.active_dofs) // 2


@pytest.mark.parametrize("sup", [
    {"type": "roller", "where": {"type": "on_plane", "axis": "y", "at": "min"}, "normal": "y"},
    {"type": "symmetry", "axis": "y", "at": "min"},
    {"type": "pinned_bore", "bore": {"type": "bore"}},
    {"type": "elastic", "where": {"type": "on_plane", "axis": "y", "at": "max"}, "k": 0.1},
    {"type": "fixed", "where": {"type": "point", "at": [16, 0]}, "dofs": [[1, 1]]},
], ids=lambda s: s["type"])
def test_supports(sup):
    """The appended support is built with its kind, on its nodes, and it holds in the solved displacement; adding
    a support or a spring can only stiffen the part, so under the same load and design the compliance falls."""
    raw = plate()
    raw["supports"].append(sup)
    s = build_setup(raw)
    assert s.bc_report.ok
    per = s.bc.constraints(0).per_support
    assert [p[1] for p in per[:1]] == ["fixed"] and len(per) == 2          # the base clamp, then the new one
    name, kind, nodes, rows = per[1]
    X = s.study.domain.node_coords()
    x0 = np.full(s.study.n_var, 0.5)
    ev = s.study.evaluate(x0, gradients=False)
    U = ev.U[:, 0].reshape(-1, 2)
    tol = 1e-12 * np.abs(U).max()
    if sup["type"] in ("roller", "symmetry"):                             # normal y on the edge y = 0
        assert kind == "roller" and set(nodes) == set(np.flatnonzero(np.abs(X[:, 1]) < 1e-9)) and rows == len(nodes)
        assert np.abs(U[nodes, 1]).max() <= tol
    elif sup["type"] == "pinned_bore":                                    # bore r 2 at (10, 4): radial u = 0
        r = np.linalg.norm(X[nodes] - [10.0, 4.0], axis=1)                # corners of cut voxels: within h sqrt 2
        assert kind == "pinned" and rows == len(nodes) and np.all(np.abs(r - 2.0) <= 0.5 * np.sqrt(2) + 1e-9)
        er = (X[nodes] - [10.0, 4.0]) / r[:, None]
        assert np.abs((U[nodes] * er).sum(axis=1)).max() <= tol
        assert np.abs((U[nodes] * er[:, ::-1] * [-1, 1]).sum(axis=1)).max() > 1e3 * tol   # tangential stays free
    elif sup["type"] == "elastic":                                        # a spring: no constraint rows
        assert kind == "spring" and rows == 0 and set(nodes) == set(np.flatnonzero(np.abs(X[:, 1] - 8.0) < 1e-9))
        assert np.abs(U[nodes]).max() > 1e3 * tol
    else:                                       # "dofs": [[1, 1]] is a direction: u . (1, 1) = 0 at (16, 0)
        assert kind == "fixed" and len(nodes) == 1 and rows == 1 and np.allclose(X[nodes[0]], [16.0, 0.0])
        u = U[nodes[0]]
        assert abs(u @ [1.0, 1.0]) <= tol and abs(u @ [1.0, -1.0]) > 1e3 * tol
    base = build_setup(plate()).study.evaluate(x0, gradients=False).f0
    assert ev.f0 < base * (1 - 1e-6), (ev.f0, base)
    evaluate(s)


@pytest.mark.parametrize("load", [
    {"type": "point", "at": [16, 8], "force": [0, -1]},
    {"type": "traction", "where": {"type": "on_plane", "axis": "x", "at": "max"}, "traction": [0, -0.125]},
    {"type": "pressure", "where": {"type": "on_plane", "axis": "y", "at": "max"}, "p": 0.1},
    {"type": "bearing", "bore": {"type": "bore", "index": 1}, "force": [0, -1]},
    {"type": "remote_force", "where": {"type": "on_plane", "axis": "x", "at": "max"}, "force": [0, -1],
     "point": [20, 4]},
    {"type": "remote_moment", "where": {"type": "on_plane", "axis": "x", "at": "max"}, "moment": 1.0},
    {"type": "torque", "bore": {"type": "bore"}, "torque": 1.0},
    {"type": "body_force", "acceleration": [0, -9.81]},
], ids=lambda s: s["type"])
def test_loads(load):
    raw = plate()
    raw["load_cases"][0]["loads"] = [load]
    s = build_setup(raw)
    assert s.bc_report.ok and not [i for i in s.bc_report.issues if i.code.startswith("resultant")]
    a = s.bc.applied["tip"][0]
    assert np.allclose(a.force, a.intended_force, atol=1e-9 * max(1.0, np.abs(a.intended_force).max()))


def test_per_case_supports_and_combinations():
    raw = plate(optimizer="mma")
    raw["load_cases"].append({"name": "side", "weight": 0.5,
                              "supports": [{"type": "fixed", "where": {"type": "on_plane", "axis": "y",
                                                                       "at": "min"}}],
                              "loads": [{"type": "point", "at": [8, 8], "force": [1, 0]}]})
    s = build_setup(raw)
    assert type(s.study.physics).__name__ == "PerCaseSupports" and s.study.F.shape[1] == 2
    evaluate(s)
    raw = plate()
    raw["load_cases"].append({"name": "up", "loads": [{"type": "point", "at": [16, 8], "force": [0, 1]}]})
    raw["combinations"] = [{"name": "both", "factors": {"tip": 1.0, "up": 0.5}, "weight": 2.0}]
    s = build_setup(raw)
    assert s.study.F.shape[1] == 3 and s.cases[2]["combination"] and s.study.objective.weights.tolist() == \
        [1.0, 1.0, 2.0]
    assert np.allclose(s.study.F[:, 2], s.study.F[:, 0] + 0.5 * s.study.F[:, 1])
    raw["combinations"][0]["factors"] = {"tpi": 1.0}
    with pytest.raises(ConfigError, match='did you mean "tip"'):
        build_setup(raw)


# ---------------------------------------------------------------------------------------------- links, interp, physics
@pytest.mark.parametrize("chain", [
    [{"type": "density_filter", "rmin": 1.5}, {"type": "am_filter", "build": "+y"}],
    [{"type": "density_filter", "rmin": 1.5}, {"type": "casting", "direction": "+y"}],
    [{"type": "symmetry", "planes": {"y": "mid"}}, {"type": "density_filter", "rmin": 1.5}, "group_average"],
    [{"type": "extrusion", "axis": "y"}, {"type": "density_filter", "rmin": 1.5}, "group_average"],
    [{"type": "pattern", "offset": [6, 0]}, {"type": "density_filter", "rmin": 1.5}, "group_average"],
    [{"type": "pattern", "kind": "cyclic", "n": 2, "center": ["mid", "mid"]}, {"type": "density_filter", "rmin": 1.5}],
], ids=lambda c: c[0]["type"] if c[0]["type"] != "density_filter" else c[1]["type"])
def test_manufacturing_links(chain):
    """The built chain (not only the resolved config) has the links in order with their parameters, and a random
    design comes out obeying the tying exactly; the untied chain breaks every one of these by ~0.4 (measured
    18 Nov 2026), so each check tells a working tie from a missing one."""
    raw = box2d(chain=chain)
    s = build_setup(raw)
    names = [lk["type"] for lk in s.config["chain"]]
    assert names[-1] == "passive" and "embed" in names
    built = [type(lk).__name__ for lk in s.study.chain.links]
    first = chain[0]["type"]
    want = {"density_filter": ["Embed", "DensityFilter", "AMFilter" if chain[1]["type"] == "am_filter" else "Casting"],
            "symmetry": ["SymmetryTying", "Embed", "DensityFilter", "GroupAverage"],
            "extrusion": ["ExtrusionTying", "Embed", "DensityFilter", "GroupAverage"],
            "pattern": ["PatternTying", "Embed", "DensityFilter"] + (["GroupAverage"] if len(chain) == 3 else [])}[first]
    assert built == want + ["Passive"]
    links = {type(lk).__name__: lk for lk in s.study.chain.links}
    assert links["DensityFilter"].rmin == 1.5
    for k in ("AMFilter", "Casting"):
        if k in links:
            assert (links[k].axis, links[k].sign) == (1, 1)                   # "+y"
    n_var = {"density_filter": 72, "symmetry": 36, "extrusion": 12, "pattern": 36}[first]    # 12 x 6 grid
    assert s.study.n_var == n_var and s.study.domain.n_design == 72
    x = np.random.default_rng(0).uniform(0.0, 1.0, s.study.n_var)
    xp = s.study.evaluate(x, gradients=False).xphys.reshape(12, 6)
    if first == "symmetry":
        assert np.abs(xp - xp[:, ::-1]).max() <= 1e-15
    elif first == "extrusion":
        assert np.abs(xp - xp[:, :1]).max() <= 1e-15
    elif first == "pattern" and len(chain) == 3:                              # translation by 6 cells in x
        assert np.abs(xp[:6] - xp[6:]).max() <= 1e-15
    elif first == "pattern":                                                  # two sectors about the centre
        assert np.abs(xp - xp[::-1, ::-1]).max() <= 1e-15
    evaluate(s)


def test_chain_order_is_validated():
    raw = box2d(chain=[{"type": "am_filter"}, {"type": "density_filter", "rmin": 1.5}])
    with pytest.raises(ConfigError, match="out of order|need a DensityFilter"):
        build_setup(raw)


def test_modal_responses_and_interpolation():
    base = {"name": "m", "domain": {"type": "box", "shape": [16, 4]},
            "physics": {"type": "modal", "n_modes": 4, "added_mass": 3.0, "added_mass_at": {"type": "point",
                                                                                          "at": ["mid", "mid"]}},
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
            "chain": [{"type": "density_filter", "rmin": 1.5}], "interpolation": "two_range_simp",
            "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma"}
    for obj in ({"type": "maximize", "response": {"type": "min_eigenvalue_ks", "n_modes": 3}},
                {"type": "maximize", "response": "eigenvalue"}, {"type": "maximize", "response": "frequency"}):
        s = build_setup(dict(base, objective=obj))
        assert s.cases == [] and type(s.study.physics).__name__ == "Modal"
        evaluate(s)
    raw = dict(base, objective={"type": "maximize", "response": "eigenvalue"},
               constraints=[{"response": "volume", "upper": 0.5}, {"type": "min_frequency", "f_min": 1e-3}])
    s = build_setup(raw)
    c = s.study.constraints[1]
    assert c.upper == 0.0 and c.scale == 1.0
    evaluate(s)
    errs = dict(pytest.raises(ConfigError, build_setup, dict(box2d(), objective="eigenvalue")).value.errors)
    assert "needs physics modal" in errs["objective"]


# ---------------------------------------------------------------------------------------------- robust, length scale
def test_robust_block_sets_rmin_and_round_trips(tmp_path):
    from toporlogy.optimization.robust import robust_chain_settings
    raw = box2d(robust={"min_solid": 3.0}, chain=[{"type": "density_filter"}],
                objective="robust_compliance",
                constraints=[{"response": "dilated_volume", "upper": 0.5},
                             {"response": {"type": "local_volume", "t_max": 3.0, "source": "dilated"}},
                             {"response": "grey_level", "upper": 0.9}],
                schedule={"type": "robust_continuation", "every": 2})
    r = resolve(raw)
    exp = robust_chain_settings(3.0, None, h=1.0, dim=2)
    assert [lk["type"] for lk in r["chain"]] == ["embed", "density_filter", "heaviside_projection", "passive"]
    assert r["chain"][1]["rmin"] == exp["rmin"]
    s = build_setup(raw)
    assert s.study.constraints[1].upper == pytest.approx(s.study.constraints[1].response.settings["alpha"])
    assert s.info["robust"]["eta_e"] == pytest.approx(exp["eta_e"])
    evaluate(s)
    sm = run_setup(s, tmp_path / "r")
    again = resolve(yaml.safe_load((tmp_path / "r" / "config.resolved.yaml").read_text(encoding="utf-8")))
    assert {k: v for k, v in again.items() if k != "provenance"} == s.config
    assert sm["iterations"] == 2
    bad = copy.deepcopy(raw)
    bad["chain"] = [{"type": "density_filter", "rmin": 2.0}]
    errs = dict(pytest.raises(ConfigError, resolve, bad).value.errors)
    assert "robust.min_solid sets rmin" in errs["chain.0.rmin"]


def test_local_volume_by_radius_needs_upper():
    raw = box2d(constraints=[{"response": "volume", "upper": 0.5},
                             {"response": {"type": "local_volume", "radius": 2.0}}])
    errs = dict(pytest.raises(ConfigError, build_setup, raw).value.errors)
    assert "set upper" in errs["constraints.1.upper"]
    raw["constraints"][1]["upper"] = 0.6
    evaluate(build_setup(raw))


# ---------------------------------------------------------------------------------------------- connectivity
def split(nx: int = 30, ny: int = 12, **conn) -> dict:
    """Two supports + two load pads (connectivity.split_benchmark layout, smaller)."""
    keep = [{"type": "box", "lo": [2, 0], "hi": [5, 1]}, {"type": "box", "lo": [25, 0], "hi": [28, 1]},
            {"type": "box", "lo": [2, 5], "hi": [5, 8]}, {"type": "box", "lo": [25, 5], "hi": [28, 8]}]
    return {"name": "split", "domain": {"type": "box", "shape": [nx, ny], "keep": keep},
            "supports": [{"type": "fixed", "where": {"type": "in_box", "lo": [2, 0], "hi": [5, 0]}, "name": "sa"},
                         {"type": "fixed", "where": {"type": "in_box", "lo": [25, 0], "hi": [28, 0]}, "name": "sb"}],
            "load_cases": [{"loads": [
                {"type": "force", "where": {"type": "in_box", "lo": [2, 5], "hi": [5, 8]}, "force": [0, -1]},
                {"type": "force", "where": {"type": "in_box", "lo": [25, 5], "hi": [28, 8]}, "force": [0, -1]}]}],
            "chain": [{"type": "density_filter", "rmin": 1.5}, {"type": "heaviside_projection"}],
            "constraints": [{"response": "volume", "upper": 0.15}], "optimizer": "mma",
            "connectivity": conn, "run": {"max_iter": 1}}


def test_connectivity_block_ties_constraint_interfaces():
    s = build_setup(split(ties=1, constraint="conduction"))
    assert list(s.interfaces) == ["sa", "sb", "L1 force", "L2 force"]
    assert s.info["ties"] > 0 and s.info["frozen_interface_cells"] > 0       # a tie must not end in void cells
    assert s.study.domain.keep.sum() == 2 * 3 + 2 * 9 + s.info["ties"] + s.info["frozen_interface_cells"]
    assert s.constraint_names == ["volume", "connectivity"] and s.study.constraints[1].upper == 20.0
    assert [p["type"] for p in s.config["post"]] == ["connectivity_gate"]
    evaluate(s)
    s = build_setup(split(ties=None, constraint="coupling", bound=3.0, gate=False,
                            interfaces={"a": {"type": "in_box", "lo": [2, 0], "hi": [5, 0]},
                                        "b": {"type": "in_box", "lo": [25, 5], "hi": [28, 8]}}))
    assert list(s.interfaces) == ["a", "b"] and s.config["post"] == [] and s.study.constraints[1].upper == 3.0
    evaluate(s)
    for resp in ("conduction_connectivity", "coupling_compliance"):
        raw = split(ties=None, constraint="none", gate=False)
        raw["constraints"].append({"response": resp})
        evaluate(build_setup(raw))


def test_gate_verdict_in_summary_and_exit_code(capsys, tmp_path):
    """Iteration 1 at volume 0.15 thresholds to the four separate pads: FAIL, exit 1; bridge repair joins them."""
    p = tmp_path / "split.yaml"
    p.write_text(yaml.safe_dump(split(ties=None, constraint="none")), encoding="utf-8")
    code = main(["run", str(p), "--out", str(tmp_path / "a"), "-q"])
    out = capsys.readouterr().out
    sm = json.loads((tmp_path / "a" / "summary.json").read_text(encoding="utf-8"))
    g = sm["post"]["connectivity_gate"]
    assert code == 1 and g["verdict"] == "FAIL" and g["passed"] is False and g["n_bodies"] >= 2
    assert "connectivity_gate failed" in sm["status"] and "post          connectivity_gate  FAIL" in out
    assert out.isascii()
    p.write_text(yaml.safe_dump(split(ties=None, constraint="none", repair="bridge")), encoding="utf-8")
    main(["run", str(p), "--out", str(tmp_path / "b"), "-q"])
    g = json.loads((tmp_path / "b" / "summary.json").read_text(encoding="utf-8"))["post"]["connectivity_gate"]
    assert g["verdict_before_repair"] == "FAIL" and g["verdict"] != "FAIL" and g["repair"]["bridges"] >= 1
    assert (tmp_path / "b" / "result_repaired.npz").is_file()


def test_bridge_repair_is_a_variant_not_a_mutation(tmp_path):
    """The repair leaves the optimisation result alone (x, xphys, objective, final all describe the optimised
    design), stores the repaired density as a variant that later steps of the same result use, re-evaluates its
    objective and constraints, and fails the gate when the added material breaks a constraint."""
    s = build_setup(split(ties=None, constraint="none", repair="bridge"))
    res = s.study.run(max_iter=1, tol=0.0)
    x0, xp0, f0, fin0 = res.x.copy(), res.xphys.copy(), res.objective, dict(res.final.values)
    name, step = s.post[0]
    assert name == "connectivity_gate"
    out = step(res, s.study, tmp_path)
    assert np.array_equal(res.x, x0) and np.array_equal(res.xphys, xp0)
    assert res.objective == f0 and res.final.values == fin0
    rep = np.load(tmp_path / "result_repaired.npz")["xphys"]
    assert not np.array_equal(rep, xp0) and out["design"] == "repaired" and out["verdict_before_repair"] == "FAIL"
    # re-evaluated at the repaired density: volume by the response's own formula, compliance by a fresh solve
    act = s.study.domain.active
    rd = out["repaired_design"]
    vol = next(c for c in rd["constraints"] if c["name"] == "volume")
    assert vol["value"] == pytest.approx(rep[act].sum() / act.sum(), rel=1e-12)
    assert rd["objective"] < f0                                    # more material, stiffer
    # this repair more than doubles the volume (bound 0.15): before, the gate reported passed for it
    assert vol["value"] > 0.15 * 1.01 and vol["satisfied"] is False and rd["feasible"] is False
    assert out["passed"] is False and "repaired design violates volume" in out["summary"]
    again = step(res, s.study, tmp_path)                           # a later step of the same result: the variant
    assert again["design"] == "repaired" and "repair" not in again
    res2 = s.study.run(max_iter=1, tol=0.0)                         # a new result does not inherit the variant
    assert step(res2, s.study, tmp_path)["verdict_before_repair"] == "FAIL"


def test_evaluate_density_matches_evaluate_and_skips_chain_responses():
    """evaluate_density(x̄ of x) equals evaluate(x) for responses of x̄ and U; a response of the chain's filtered
    field (conduction connectivity on the eroded field) gets no value but a reason."""
    s = build_setup(split(ties=1, constraint="conduction"))
    res = s.study.run(max_iter=1, tol=0.0)
    de = s.study.evaluate_density(res.xphys)
    assert de.values["compliance"] == pytest.approx(res.final.values["compliance"], rel=1e-12)
    assert de.values["volume"] == pytest.approx(res.final.values["volume"], rel=1e-12)
    assert "connectivity" not in de.values and "no chain" in de.skipped["connectivity"]
    with pytest.raises(ValueError, match="shape"):
        s.study.evaluate_density(res.xphys.ravel())


# ---------------------------------------------------------------------------------------------- validate output
def test_validate_prints_bc_summary_and_rejects_under_support(capsys, tmp_path):
    p = tmp_path / "plate.yaml"
    raw = plate()
    raw["load_cases"][0]["loads"].append({"type": "bearing", "bore": {"type": "bore"}, "force": [0, -1]})
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code = main(["validate", str(p)])
    out = capsys.readouterr().out
    assert code == 0 and "bc check      OK | 0 errors" in out
    assert 'Load case "tip"' in out and "Supports of tip" in out and "L1 surface_force" in out
    assert "bores         B1 r" in out and out.strip().splitlines()[-1].startswith("NEXT")
    raw = plate(supports=[{"type": "roller", "where": {"type": "on_plane", "axis": "x", "at": "min"},
                           "normal": "x"}])
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code = main(["validate", str(p)])
    o = capsys.readouterr()
    assert code == 2 and o.out == ""
    assert "supports  rigid-body: 1 unrestrained rigid-body mode" in o.err and "translation along (0, 1)" in o.err
    code = main(["run", str(p), "--out", str(tmp_path / "r")])
    assert code == 2 and not (tmp_path / "r").exists()


def test_empty_bc_selector_reports_its_path():
    raw = plate()
    raw["supports"][0]["where"] = {"type": "on_plane", "axis": "x", "at": 30}
    errs = dict(pytest.raises(ConfigError, build_setup, raw).value.errors)
    assert "selected no node" in errs["supports.0.where"]
    raw = plate()
    raw["load_cases"][0]["loads"] = [{"type": "bearing", "bore": {"type": "bore", "index": 3}, "force": [0, 1]}]
    errs = dict(pytest.raises(ConfigError, build_setup, raw).value.errors)
    assert "bore index 3 out of range; found: B1" in errs["load_cases.0.loads.0.bore"]


# ---------------------------------------------------------------------------------------------- post steps
def test_finalize_checks_export_and_lengthscale(tmp_path):
    """A 3-D bar with a bore: finalize (bore preserved with stock), checks, export, length-scale report."""
    raw = {"name": "bar", "domain": {"type": "csg", "h": 1.0,
                                     "shapes": [{"type": "box", "lo": [0, 0, 0], "hi": [14, 8, 4]},
                                                {"type": "cylinder", "p0": [10, 4, -1], "p1": [10, 4, 5],
                                                 "radius": 2.0, "op": "cut"}]},
           "supports": [{"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "bearing", "bore": {"type": "bore"}, "force": [0, -1, 0]}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}], "constraints": [{"response": "volume",
                                                                                "upper": 0.6}],
           "run": {"max_iter": 3},
           "post": [{"type": "measure_lengthscale", "upsample": 1},
                    {"type": "finalize", "upsample": 2, "smooth_iter": 5, "relax_iter": 2,
                     "preserve": [{"role": "cut", "shape": {"type": "bore"}, "stock": 0.3, "name": "bore"}]},
                    {"type": "manufacturing_checks", "max_bodies": 1},
                    {"type": "export", "formats": ["stl", "ply"]}]}
    s = build_setup(raw)
    sm = run_setup(s, tmp_path / "r")
    post = sm["post"]
    assert post["finalize"]["files"] == {"stl": "part.stl", "report": "part_report.json"}
    assert post["finalize"]["triangles"] > 100 and "error" not in post["manufacturing_checks"]
    assert {"stl", "ply", "stl_print", "report"} <= set(post["export"]["files"])
    for f in ("part.stl", "part.ply", "part_print_stock.stl", "part_report.json"):
        assert (tmp_path / "r" / f).stat().st_size > 0
    rep = json.loads((tmp_path / "r" / "part_report.json").read_text(encoding="utf-8"))
    assert "checks" in rep and any(i["name"] == "bore" for i in rep["pipeline"]["interfaces"])
    assert post["measure_lengthscale"]["solid_max"] > 0


def test_post_order_and_reverify_build():
    raw = box2d(post=["export"])
    errs = dict(pytest.raises(ConfigError, resolve, raw).value.errors)
    assert "needs a finalize step" in errs["post.0"]
    errs = dict(pytest.raises(ConfigError, build_setup, box2d(post=["finalize"])).value.errors)
    assert "3-D" in errs["post.0"]
    pytest.importorskip("gmsh")
    raw = {"name": "r", "domain": {"type": "box", "shape": [6, 3, 3]},
           "supports": [{"type": "symmetry", "axis": "y", "at": "min"},
                        {"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "point", "at": ["max", "mid", "mid"], "force": [0, 0, -1]}]}],
           "constraints": [{"response": "volume", "upper": 0.5}], "post": ["finalize", "reverify"]}
    s = build_setup(raw)
    from toporlogy.configuration.boundary_mapping import bc_callables
    sup, cases, notes = bc_callables(s.bc)
    X = s.study.domain.node_coords()
    assert sup[0][1] == (1,) and sup[0][0](X).sum() > 0 and list(cases) == ["case1"]


def test_bc_mapping_labels_what_reverify_cannot_carry_over():
    """bc_mapping: equivalent when every support and load maps by type; a bearing load (area-spread resultant)
    makes it approximate; springs (left out) make it unsupported. bc_callables keeps its (supports, cases,
    notes) contract with the same note texts."""
    from toporlogy.configuration.boundary_mapping import bc_callables, bc_mapping
    m = bc_mapping(build_setup(plate()).bc)
    assert m["fidelity"] == "equivalent" and m["items"] == [] and m["notes"] == []
    raw = plate()
    raw["load_cases"][0]["loads"].append({"type": "bearing", "bore": {"type": "bore"}, "force": [0, -1]})
    m = bc_mapping(build_setup(raw).bc)
    assert m["fidelity"] == "approximate" and [it["fidelity"] for it in m["items"]] == ["approximate"]
    assert "no cosine / moment" in m["items"][0]["note"]
    raw["supports"].append({"type": "elastic", "where": {"type": "on_plane", "axis": "y", "at": "max"}, "k": 0.1,
                            "name": "spring"})
    bcs = build_setup(raw).bc
    m = bc_mapping(bcs)
    assert m["fidelity"] == "unsupported"
    assert {(it["name"], it["fidelity"]) for it in m["items"]} >= {("spring", "unsupported")}
    sup, cases, notes = bc_callables(bcs)
    assert notes == m["notes"] and "spring: springs are not re-verified (left free)" in notes
    assert len(sup) == len(m["supports"]) and list(cases) == list(m["cases"])


def test_move_beta_ref_reaches_study():
    s = build_setup(box2d(run={"max_iter": 2, "move_beta_ref": 2.0}))
    assert s.study.move_beta_ref == 2.0
    assert build_setup(box2d()).study.move_beta_ref is None


def test_list_shows_feature_components(capsys):
    assert main(["list", "post"]) == 0
    out = capsys.readouterr().out
    for name in ("connectivity_gate", "finalize", "manufacturing_checks", "reverify", "export",
                 "measure_lengthscale"):
        assert f"  {name}" in out
    assert "upsample=4" in out and "min_wall=null [length]" in out


def test_elasticity_gmg_physics_and_rejections(capsys, tmp_path):
    pytest.importorskip("toporlogy.fem.elastic_gmg")
    base = {"name": "g", "domain": {"type": "box", "shape": [8, 4, 4]}, "physics": "elasticity_gmg",
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
            "load_cases": [{"loads": [{"type": "point", "at": ["max", "mid", "mid"], "force": [0, 0, -1]}]}],
            "constraints": [{"response": "volume", "upper": 0.5}]}
    s = build_setup(base)
    assert type(s.study.physics).__name__ == "ElasticityGMG"
    evaluate(s)
    bad = {"2-D": (box2d(physics="elasticity_gmg"), "3-D only"),
           "oblique": (dict(base, supports=base["supports"] + [{"type": "roller", "where": {
               "type": "on_plane", "axis": "z", "at": "min"}, "normal": [0, 1, 1]}]), "zero displacements along"),
           "springs": (dict(base, supports=base["supports"] + [{"type": "elastic", "where": {
               "type": "on_plane", "axis": "z", "at": "min"}, "k": 1.0}]), "no oblique rollers"),
           "per-case": (dict(base, load_cases=base["load_cases"] + [{"name": "b", "supports": base["supports"] + [
               {"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "max"}, "dofs": "z"}],
               "loads": [{"type": "point", "at": ["mid", "mid", "max"], "force": [0, 0, -1]}]}]), "one support set")}
    for name, (raw, msg) in bad.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        code = main(["validate", str(p)])
        err = capsys.readouterr().err
        assert code == 2 and "physics" in err and msg in err, (name, err)


def test_repair_whose_constraints_cannot_be_evaluated_does_not_pass(tmp_path):
    """A repaired design whose constraints cannot all be evaluated (responses that need the design chain: robust
    realisations, eroded-field connectivity) used to pass the gate as long as none of the evaluated ones failed;
    on lbracket_bore3d no constraint at all could be evaluated. It is now reported as not verified and fails,
    with the plain volume fraction before and after the repair as information."""
    raw = split(ties=None, constraint="none", repair="bridge")         # splits into the four pads: FAIL, repaired
    raw["constraints"][0]["upper"] = 0.9                    # the volume bound holds; only the chain response is unknown
    raw["constraints"].append({"response": "conduction_connectivity", "upper": 1e6})   # needs the chain (eroded field)
    s = build_setup(raw)
    res = s.study.run(x0=np.full(s.study.n_var, 0.15), max_iter=1, tol=0.0)   # low start: the pads stay apart
    name, step = next((n, st) for n, st in s.post if n == "connectivity_gate")
    out = step(res, s.study, tmp_path)
    assert out["verdict_before_repair"] == "FAIL" and out["verdict"] != "FAIL"
    rd = out["repaired_design"]
    skipped = [c["name"] for c in rd["constraints"] if c["satisfied"] is None]
    assert rd["feasible"] is None and len(skipped) == 1 and "connectivity" in skipped[0]
    assert out["passed"] is False and "not verified" in out["summary"] and "connectivity" in out["summary"]
    vf = rd["volume_fraction_plain"]
    assert vf["repaired"] > vf["optimized"]
