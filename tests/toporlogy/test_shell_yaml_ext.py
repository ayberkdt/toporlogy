"""Load cases, line loads along an edge set and VTU on the shell YAML path (docs/olcumler/kabuk_onkayit.md, SX1-SX6,
pre-registered 2026-11-26 22:09 +0300 before the code). vtk is the independent reader of SX4; without it that part
is skipped (NOT VERIFIED). The anchored example (SX6) is checked in test_mesh_yaml_ext.py (AX7)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config
from toporlogy.core.vtu import VTK_QUAD, read_vtu
from toporlogy.execution.cli import main
from toporlogy.fem.shell import FlatShell, flat_rectangle

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "research" / "shell_cylinder.yaml"
C45 = math.cos(math.pi / 4)
NU = 0.3


def _base() -> dict[str, Any]:
    return read_config(EXAMPLE)


def _cfg(domain: dict[str, Any], supports: list[Any], cases: list[Any], rmin: float = 0.3) -> dict[str, Any]:
    return {"name": "s", "domain": {"type": "shell", **domain}, "material": {"E0": 1.0, "nu": NU}, "physics": "shell",
            "supports": supports, "load_cases": cases, "chain": [{"type": "density_filter", "rmin": rmin}],
            "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9}, "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma", "run": {"max_iter": 10}}


RECT = {"surface": "rectangle", "size": [2.0, 1.0], "n": [8, 6], "thickness": 0.05}
RADIAL = {"point": [C45, C45, 1.0], "force": [-C45, -C45, 0.0]}


# ------------------------------------------------------------------ SX1: two cases = weighted sum of one-case studies
def _sx1(case: str, cases: list[Any]) -> dict[str, Any]:
    if case == "example":
        cfg = _base()
        cfg["load_cases"] = cases
        return cfg
    return _cfg(RECT, [{"set": "left", "fix": "all"}], cases)


SX1: dict[str, tuple[Any, ...]] = {"example": ([RADIAL], [{"pressure": 0.5}, {"set": "top", "force": [0, 0, 0.2], "distribute": "traction"}]),
       "rectangle": ([{"point": [2.0, 1.0, 0.0], "force": [0, 0, -1]}], [{"pressure": 1.0}])}


_SX1_FAIL = ("pre-registered SX1 FAILED on the first run for the flat rectangle: value 0.0e+00, gradient 5.59e-12 "
             "(> 1e-12); cond(K_ff) = 4.0e6 at that x, eps cond = 9.0e-10: the weighted adjoint solves K l = w F of "
             "the two-case study and w (K^-1 F) of the one-case studies round differently; thresholds unchanged")


def _sx1_measure(case: str) -> tuple[float, float, Any, np.ndarray]:
    a, b = SX1[case]
    two = build_setup(_sx1(case, [{"name": "a", "weight": 0.7, "loads": a}, {"name": "b", "weight": 1.3, "loads": b}]))
    sa = build_setup(_sx1(case, [{"name": "a", "loads": a}])).study
    sb = build_setup(_sx1(case, [{"name": "b", "loads": b}])).study
    x = 0.2 + 0.7 * np.random.default_rng(51).random(sa.n_var)
    e2, ea, eb = two.study.evaluate(x), sa.evaluate(x), sb.evaluate(x)
    want, gwant = 0.7 * ea.f0 + 1.3 * eb.f0, 0.7 * ea.df0 + 1.3 * eb.df0
    rv = abs(e2.f0 - want) / abs(want)
    rg = np.abs(e2.df0 - gwant).max() / np.abs(gwant).max()
    rows = [(c["name"], c["weight"]) for c in two.cases]
    print(f"SX1 {case}: value {rv:.2e} (c = {want:.6e}), gradient {rg:.2e}, cases {rows}")
    assert rows == [("a", 0.7), ("b", 1.3)] and two.study.F.shape[1] == 2
    return rv, rg, two, x


@pytest.mark.parametrize("case", ["example", pytest.param("rectangle", marks=pytest.mark.xfail(strict=True,
                                                                                               reason=_SX1_FAIL))])
def test_sx1_two_cases_are_the_weighted_sum_of_one_case_studies(case: str) -> None:
    rv, rg, _, _ = _sx1_measure(case)
    assert rv <= 1e-12 and rg <= 1e-12


def test_sx1_posthoc_differences_are_within_the_solve_roundoff_bound() -> None:
    """Post hoc (after SX1 failed for the flat rectangle): there the value is equal and the relative gradient
    difference stays below eps · cond(K_ff), the forward-error scale of a backward-stable solve."""
    rv, rg, two, x = _sx1_measure("rectangle")
    st = two.study
    xphys = st.chain(torch.as_tensor(x)).detach()
    E = st.interp(xphys).detach().numpy().reshape(-1)
    sh = st.physics.shell
    K = sh.stiffness(E).toarray()[np.ix_(sh.free, sh.free)]
    bound = np.finfo(float).eps * np.linalg.cond(K)
    print(f"SX1 post hoc rectangle: value {rv:.2e}, gradient {rg:.2e}, eps cond(K_ff) {bound:.2e}")
    assert rv <= 1e-12 and rg <= bound


# ------------------------------------------------------------------ SX2: resultant and shares of a line load
SX2: dict[str, tuple[Any, ...]] = {"rectangle_right": ({"surface": "rectangle", "size": [2.0, 1.0], "n": [8, 4], "thickness": 0.05}, "left",
                           "right", [1.0, 0.0, 0.0], 4, False),
       "segment_top": ({"surface": "cylinder", "radius": 1.0, "angle": 90.0, "length": 1.0, "n": [16, 12],
                        "thickness": 0.05}, "bottom", "top", [0.0, 0.0, 1.0], 16, False),
       "tube_top": ({"surface": "cylinder", "radius": 1.0, "angle": None, "length": 1.0, "n": [12, 6],
                     "thickness": 0.05}, "bottom", "top", [0.0, 0.0, 1.0], 12, True)}


def _ends(X: np.ndarray, case: str) -> np.ndarray:
    if case == "rectangle_right":
        return np.isclose(X[:, 1], X[:, 1].min()) | np.isclose(X[:, 1], X[:, 1].max())
    th = np.arctan2(X[:, 1], X[:, 0])
    return np.isclose(th, 0.0) | np.isclose(th, math.pi / 2)


@pytest.mark.parametrize("case", list(SX2))
def test_sx2_line_load_resultant_and_nodal_shares(case: str) -> None:
    dom, held, loaded, force, n, closed = SX2[case]
    s = build_setup(_cfg(dom, [{"set": held, "fix": "all"}],
                         [{"loads": [{"set": loaded, "force": force, "distribute": "traction"}]}]))
    m = s.study.domain.mesh
    F6 = np.asarray(s.study.F, float)[:, 0].reshape(-1, 6)
    f = np.asarray(force, float)
    res = float(np.linalg.norm(F6[:, :3].sum(axis=0) - f) / np.linalg.norm(f))
    nodes = m.node_sets[loaded]
    ends = np.zeros(len(nodes), bool) if closed else _ends(m.nodes[nodes], case)
    want = np.where(ends[:, None], f / (2 * n), f / n)
    share = float(np.abs(F6[nodes, :3] - want).max() / np.linalg.norm(f))
    rot_zero = bool((F6[:, 3:] == 0).all())
    outside = np.setdiff1d(np.flatnonzero(np.any(F6 != 0, axis=1)), nodes)
    print(f"SX2 {case}: resultant {res:.2e}, shares {share:.2e} ({int(ends.sum())} ends), rotations zero {rot_zero}, "
          f"loaded nodes outside the set {len(outside)}")
    assert res <= 1e-14 and share <= 1e-14 and rot_zero and not len(outside)


# ------------------------------------------------------------------ SX3: membrane patch test
def _patch_error(distribute: str) -> float:
    P, lx, ly, t = 0.5, 2.0, 1.0, 0.05
    cfg = _cfg({"surface": "rectangle", "size": [lx, ly], "n": [8, 4], "thickness": t},
               [{"set": "left", "fix": ["u", "w", "rx", "ry", "rz"]}, {"set": "bottom", "fix": ["v"]}],
               [{"loads": [{"set": "right", "force": [P, 0.0, 0.0], "distribute": distribute}]}])
    st = build_setup(cfg).study
    m = st.domain.mesh
    d = st.physics.solve(torch.ones(m.ne, dtype=torch.float64), st.F).detach().numpy()[:, 0].reshape(-1, 6)
    eps = P / (ly * t)
    exact = np.zeros_like(d)
    exact[:, 0] = eps * m.nodes[:, 0]
    exact[:, 1] = -NU * eps * m.nodes[:, 1]
    return float(np.abs(d - exact).max() / np.abs(exact[:, 0]).max())


def test_sx3_membrane_patch_test_with_a_line_load() -> None:
    err = _patch_error("traction")
    print(f"SX3 traction: max |d - d_exact| / max |u_exact| = {err:.2e}")
    assert err <= 1e-10


def test_sx3_equal_split_is_not_consistent() -> None:
    err = _patch_error("nodes")
    print(f"SX3 nodes: max |d - d_exact| / max |u_exact| = {err:.2e}")
    assert err >= 1e-3


# ------------------------------------------------------------------ SX4: VTU of the result
SX4: dict[str, tuple[Any, ...]] = {"segment": ({"surface": "cylinder", "radius": 1.0, "angle": 90.0, "length": 1.0, "n": [8, 6],
                    "thickness": 0.05}, "bottom", [[RADIAL], [{"pressure": 0.5}]]),
       "rectangle": ({"surface": "rectangle", "size": [2.0, 1.0], "n": [4, 3], "thickness": 0.05}, "left",
                     [[{"set": "right", "force": [0, 0, -1], "distribute": "traction"}]])}


def _vtu_run(case: str, folder: Path) -> tuple[Any, Any, Any]:
    dom, held, loads = SX4[case]
    cfg = _cfg(dom, [{"set": held, "fix": "all"}], [{"name": f"c{i}", "loads": ld} for i, ld in enumerate(loads)])
    cfg["post"] = [{"type": "vtu"}]
    s = build_setup(cfg)
    res = s.study.run(max_iter=2, tol=0.0)
    name, step = s.post[0]
    assert name == "vtu"
    return s, res, step(res, s.study, folder)


@pytest.mark.parametrize("case", list(SX4))
def test_sx4_vtu_roundtrip_is_exact(case: str, tmp_path: Path) -> None:
    s, res, out = _vtu_run(case, tmp_path)
    st = s.study
    m = st.domain.mesh
    r = read_vtu(tmp_path / out["file"])
    E = st.interp(torch.as_tensor(np.asarray(res.xphys, float))).detach().numpy().reshape(-1)
    U = np.asarray(res.U, float).reshape(st.domain.ndof, -1)
    ok = {"points": np.array_equal(r.points, m.nodes), "cells": np.array_equal(r.connectivity, m.facets),
          "types": bool((r.cell_types == VTK_QUAD).all()) and len(r.cell_types) == m.ne,
          "xphys": np.array_equal(r.cell_data["xphys"], np.asarray(res.xphys, float).reshape(-1)),
          "E": np.array_equal(r.cell_data["E"], E), "cases": U.shape[1] == len(SX4[case][2])}
    for c in range(U.shape[1]):
        d6 = U[:, c].reshape(-1, 6)
        ok[f"displacement_{c}"] = np.array_equal(r.point_data[f"displacement_{c}"], d6[:, :3])
        ok[f"rotation_{c}"] = np.array_equal(r.point_data[f"rotation_{c}"], d6[:, 3:])
    print(f"SX4 {case}: {ok}, point arrays {out['point_arrays']}")
    assert all(ok.values()) and sorted(r.point_data) == out["point_arrays"]


@pytest.mark.parametrize("case", list(SX4))
def test_sx4_vtk_reads_the_facets_with_their_areas(case: str, tmp_path: Path) -> None:
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    s, res, out = _vtu_run(case, tmp_path)
    d = s.study.domain
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(tmp_path / out["file"]))
    reader.Update()
    g = reader.GetOutput()
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(g)
    size.Update()
    area = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Area"))
    total = float(d.cell_volume.sum())
    rel = abs(float(area.sum()) - total) / total
    types = {g.GetCellType(i) for i in range(g.GetNumberOfCells())}
    print(f"SX4 vtk {case}: {g.GetNumberOfCells()} cells, types {types}, min area {area.min():.3e}, total {rel:.2e}")
    assert g.GetNumberOfCells() == d.mesh.ne and types == {VTK_QUAD} and (area > 0).all() and rel <= 1e-12


def test_sx4_cli_run_writes_the_vtu(tmp_path: Path) -> None:
    cfg = _base()
    cfg["post"] = [{"type": "vtu"}]
    cfg["run"]["max_iter"] = 3
    src = tmp_path / "shell_vtu.yaml"
    src.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    folder = tmp_path / "run"
    code = main(["run", str(src), "--out", str(folder), "-q"])
    sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    r = read_vtu(folder / "result.vtu")
    with np.load(folder / "result.npz") as z:
        same = np.array_equal(r.cell_data["xphys"], z["xphys"])
    print(f"SX4 CLI: exit {code}, post {sm['post']['vtu']['summary']}, xphys as result.npz {same}")
    assert sm["post"]["vtu"]["file"] == "result.vtu" and same


# ------------------------------------------------------------------ SX5: what is still refused
def _with(path: str, value: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(_base())
    node: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value
    return cfg


UNSUPPORTED: dict[str, tuple[str, Any, str]] = {
    "combinations": ("combinations", [{"name": "c", "factors": {"radial": 1.0}}], "combinations"),
    "case_supports": ("load_cases.0.supports", [{"set": "bottom", "fix": "all"}], "load_cases.0.supports"),
    "post_export": ("post", [{"type": "export"}], "post.0"),
}
INVALID: dict[str, tuple[str, Any, str]] = {
    "top_level_loads": ("loads", [RADIAL], "loads"),
    "pressure_with_set": ("load_cases.0.loads", [{"pressure": 1.0, "set": "top", "force": [0, 0, 1]}],
                          "load_cases.0.loads.0"),
    "point_distribute": ("load_cases.0.loads", [{**RADIAL, "distribute": "traction"}],
                         "load_cases.0.loads.0.distribute"),
    "distribute_value": ("load_cases.0.loads", [{"set": "top", "force": [0, 0, 1], "distribute": "even"}],
                         "load_cases.0.loads.0.distribute"),
    "unknown_set": ("load_cases.0.loads", [{"set": "left", "force": [0, 0, 1]}], "load_cases.0.loads.0.set"),
}


@pytest.mark.parametrize("case", list(UNSUPPORTED) + list(INVALID))
def test_sx5_unsupported_and_invalid_keys_raise(case: str) -> None:
    key, value, path = UNSUPPORTED.get(case) or INVALID[case]
    with pytest.raises(ConfigError) as err:
        build_setup(_with(key, value))
    hits = [m for p, m in err.value.errors if p == path]
    print(f"SX5 {case}: {err.value}")
    assert hits and all("shell" in m for m in hits), err.value.errors
    if case in UNSUPPORTED:
        assert any("not support" in m for m in hits), err.value.errors
    if case == "top_level_loads":
        assert any("load_cases" in m for m in hits), err.value.errors


def test_sx5_line_load_needs_an_edge() -> None:
    m = flat_rectangle(2.0, 1.0, 4, 2)
    sh = FlatShell(m, 0.05, NU, np.arange(6))
    with pytest.raises(ValueError, match="boundary edge") as err:
        sh.line_load(np.array([0]), [0.0, 0.0, 1.0])
    print(f"SX5 line_load with one node: {err.value}")
