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

from toporlogy import config as C
from toporlogy.cli import main
from toporlogy.registry import REGISTRY, BuildContext, ConfigError


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
                                                      {"type": "circle", "center": [10, 4], "radius": 2,
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


def evaluate(s: C.Setup) -> None:
    ev = s.study.evaluate(s.study.default_start())
    assert np.isfinite(ev.f0) and np.all(np.isfinite(ev.g)) and np.all(np.isfinite(ev.df0))


# ---------------------------------------------------------------------------------------------- shapes, domains
@pytest.mark.parametrize("spec", [
    {"type": "box", "lo": [0, 0, 0], "hi": [1, 2, 3]}, {"type": "ball", "center": [0, 0], "radius": 1},
    {"type": "sphere", "center": [0, 0, 0], "radius": 1}, {"type": "circle", "center": [0, 0], "radius": 1},
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
    s = C.build_setup(plate())
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
    s = C.build_setup(raw)
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
    s = C.build_setup(raw)
    assert abs(s.cases[0]["resultant"][1] + 1) < 1e-12 and s.cases[0]["n_nodes"] >= 1


def test_all_and_surface_selectors():
    raw = plate()
    raw["load_cases"][0]["loads"] = [{"type": "force", "where": "surface", "force": [0, -1]},
                                     {"type": "force", "where": "all", "force": [0, -1]}]
    s = C.build_setup(raw)
    assert s.cases[0]["n_nodes"] == len(s.study.domain.active_dofs) // 2


@pytest.mark.parametrize("sup", [
    {"type": "roller", "where": {"type": "on_plane", "axis": "y", "at": "min"}, "normal": "y"},
    {"type": "symmetry", "axis": "y", "at": "min"},
    {"type": "pinned_bore", "bore": {"type": "bore"}},
    {"type": "elastic", "where": {"type": "on_plane", "axis": "y", "at": "max"}, "k": 0.1},
    {"type": "fixed", "where": {"type": "point", "at": [16, 0]}, "dofs": [[1, 1]]},
], ids=lambda s: s["type"])
def test_supports(sup):
    raw = plate()
    raw["supports"].append(sup)
    s = C.build_setup(raw)
    kinds = {p[1] for p in s.bc.constraints(0).per_support}
    assert len(kinds) >= 1 and s.bc_report.ok
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
    s = C.build_setup(raw)
    assert s.bc_report.ok and not [i for i in s.bc_report.issues if i.code.startswith("resultant")]
    a = s.bc.applied["tip"][0]
    assert np.allclose(a.force, a.intended_force, atol=1e-9 * max(1.0, np.abs(a.intended_force).max()))


def test_per_case_supports_and_combinations():
    raw = plate(optimizer="mma")
    raw["load_cases"].append({"name": "side", "weight": 0.5,
                              "supports": [{"type": "fixed", "where": {"type": "on_plane", "axis": "y",
                                                                       "at": "min"}}],
                              "loads": [{"type": "point", "at": [8, 8], "force": [1, 0]}]})
    s = C.build_setup(raw)
    assert type(s.study.physics).__name__ == "PerCaseSupports" and s.study.F.shape[1] == 2
    evaluate(s)
    raw = plate()
    raw["load_cases"].append({"name": "up", "loads": [{"type": "point", "at": [16, 8], "force": [0, 1]}]})
    raw["combinations"] = [{"name": "both", "factors": {"tip": 1.0, "up": 0.5}, "weight": 2.0}]
    s = C.build_setup(raw)
    assert s.study.F.shape[1] == 3 and s.cases[2]["combination"] and s.study.objective.weights.tolist() == \
        [1.0, 1.0, 2.0]
    assert np.allclose(s.study.F[:, 2], s.study.F[:, 0] + 0.5 * s.study.F[:, 1])
    raw["combinations"][0]["factors"] = {"tpi": 1.0}
    with pytest.raises(ConfigError, match='did you mean "tip"'):
        C.build_setup(raw)


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
    raw = box2d(chain=chain)
    s = C.build_setup(raw)
    names = [lk["type"] for lk in s.config["chain"]]
    assert names[-1] == "passive" and "embed" in names
    if chain[0]["type"] in ("symmetry", "extrusion", "pattern"):
        assert names.index("embed") == 1 and s.study.n_var < s.study.domain.n_design
    evaluate(s)


def test_chain_order_is_validated():
    raw = box2d(chain=[{"type": "am_filter"}, {"type": "density_filter", "rmin": 1.5}])
    with pytest.raises(ConfigError, match="out of order|need a DensityFilter"):
        C.build_setup(raw)


def test_modal_responses_and_interpolation():
    base = {"name": "m", "domain": {"type": "box", "shape": [16, 4]},
            "physics": {"type": "modal", "n_modes": 4, "added_mass": 3.0, "added_mass_at": {"type": "point",
                                                                                          "at": ["mid", "mid"]}},
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
            "chain": [{"type": "density_filter", "rmin": 1.5}], "interpolation": "two_range_simp",
            "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma"}
    for obj in ({"type": "maximize", "response": {"type": "min_eigenvalue_ks", "n_modes": 3}},
                {"type": "maximize", "response": "eigenvalue"}, {"type": "maximize", "response": "frequency"}):
        s = C.build_setup(dict(base, objective=obj))
        assert s.cases == [] and type(s.study.physics).__name__ == "Modal"
        evaluate(s)
    raw = dict(base, objective={"type": "maximize", "response": "eigenvalue"},
               constraints=[{"response": "volume", "upper": 0.5}, {"type": "min_frequency", "f_min": 1e-3}])
    s = C.build_setup(raw)
    c = s.study.constraints[1]
    assert c.upper == 0.0 and c.scale == 1.0
    evaluate(s)
    errs = dict(pytest.raises(ConfigError, C.build_setup, dict(box2d(), objective="eigenvalue")).value.errors)
    assert "needs physics modal" in errs["objective"]


# ---------------------------------------------------------------------------------------------- robust, length scale
def test_robust_block_sets_rmin_and_round_trips(tmp_path):
    from toporlogy.robust import robust_chain_settings
    raw = box2d(robust={"min_solid": 3.0}, chain=[{"type": "density_filter"}],
                objective="robust_compliance",
                constraints=[{"response": "dilated_volume", "upper": 0.5},
                             {"response": {"type": "local_volume", "t_max": 3.0, "source": "dilated"}},
                             {"response": "grey_level", "upper": 0.9}],
                schedule={"type": "robust_continuation", "every": 2})
    r = C.resolve(raw)
    exp = robust_chain_settings(3.0, None, h=1.0, dim=2)
    assert [lk["type"] for lk in r["chain"]] == ["embed", "density_filter", "heaviside_projection", "passive"]
    assert r["chain"][1]["rmin"] == exp["rmin"]
    s = C.build_setup(raw)
    assert s.study.constraints[1].upper == pytest.approx(s.study.constraints[1].response.settings["alpha"])
    assert s.info["robust"]["eta_e"] == pytest.approx(exp["eta_e"])
    evaluate(s)
    sm = C.run_setup(s, tmp_path / "r")
    again = C.resolve(yaml.safe_load((tmp_path / "r" / "config.resolved.yaml").read_text(encoding="utf-8")))
    assert {k: v for k, v in again.items() if k != "provenance"} == s.config
    assert sm["iterations"] == 2
    bad = copy.deepcopy(raw)
    bad["chain"] = [{"type": "density_filter", "rmin": 2.0}]
    errs = dict(pytest.raises(ConfigError, C.resolve, bad).value.errors)
    assert "robust.min_solid sets rmin" in errs["chain.0.rmin"]


def test_local_volume_by_radius_needs_upper():
    raw = box2d(constraints=[{"response": "volume", "upper": 0.5},
                             {"response": {"type": "local_volume", "radius": 2.0}}])
    errs = dict(pytest.raises(ConfigError, C.build_setup, raw).value.errors)
    assert "set upper" in errs["constraints.1.upper"]
    raw["constraints"][1]["upper"] = 0.6
    evaluate(C.build_setup(raw))


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
    s = C.build_setup(split(ties=1, constraint="conduction"))
    assert list(s.interfaces) == ["sa", "sb", "L1 force", "L2 force"]
    assert s.info["ties"] > 0 and s.info["frozen_interface_cells"] > 0       # a tie must not end in void cells
    assert s.study.domain.keep.sum() == 2 * 3 + 2 * 9 + s.info["ties"] + s.info["frozen_interface_cells"]
    assert s.constraint_names == ["volume", "connectivity"] and s.study.constraints[1].upper == 20.0
    assert [p["type"] for p in s.config["post"]] == ["connectivity_gate"]
    evaluate(s)
    s = C.build_setup(split(ties=None, constraint="coupling", bound=3.0, gate=False,
                            interfaces={"a": {"type": "in_box", "lo": [2, 0], "hi": [5, 0]},
                                        "b": {"type": "in_box", "lo": [25, 5], "hi": [28, 8]}}))
    assert list(s.interfaces) == ["a", "b"] and s.config["post"] == [] and s.study.constraints[1].upper == 3.0
    evaluate(s)
    for resp in ("conduction_connectivity", "coupling_compliance"):
        raw = split(ties=None, constraint="none", gate=False)
        raw["constraints"].append({"response": resp})
        evaluate(C.build_setup(raw))


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
    errs = dict(pytest.raises(ConfigError, C.build_setup, raw).value.errors)
    assert "selected no node" in errs["supports.0.where"]
    raw = plate()
    raw["load_cases"][0]["loads"] = [{"type": "bearing", "bore": {"type": "bore", "index": 3}, "force": [0, 1]}]
    errs = dict(pytest.raises(ConfigError, C.build_setup, raw).value.errors)
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
    s = C.build_setup(raw)
    sm = C.run_setup(s, tmp_path / "r")
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
    errs = dict(pytest.raises(ConfigError, C.resolve, raw).value.errors)
    assert "needs a finalize step" in errs["post.0"]
    errs = dict(pytest.raises(ConfigError, C.build_setup, box2d(post=["finalize"])).value.errors)
    assert "3-D" in errs["post.0"]
    pytest.importorskip("gmsh")
    raw = {"name": "r", "domain": {"type": "box", "shape": [6, 3, 3]},
           "supports": [{"type": "symmetry", "axis": "y", "at": "min"},
                        {"type": "fixed", "where": {"type": "on_plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "point", "at": ["max", "mid", "mid"], "force": [0, 0, -1]}]}],
           "constraints": [{"response": "volume", "upper": 0.5}], "post": ["finalize", "reverify"]}
    s = C.build_setup(raw)
    from toporlogy.registry import bc_callables
    sup, cases, notes = bc_callables(s.bc)
    X = s.study.domain.node_coords()
    assert sup[0][1] == (1,) and sup[0][0](X).sum() > 0 and list(cases) == ["case1"]


def test_move_beta_ref_reaches_study():
    s = C.build_setup(box2d(run={"max_iter": 2, "move_beta_ref": 2.0}))
    assert s.study.move_beta_ref == 2.0
    assert C.build_setup(box2d()).study.move_beta_ref is None


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
    s = C.build_setup(base)
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
