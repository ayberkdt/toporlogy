"""Load cases, consistent traction, plane strain and VTU on the mesh YAML path (docs/olcumler/ag_topoloji_onkayit.md,
AX1-AX8, pre-registered 2026-11-26 22:09 +0300 before the code). vtk is the independent reader of AX5; without it
that part is skipped (NOT VERIFIED). AX8 (diff predictions) is measured on the commit, not here."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from toporlogy.configuration.build import build_setup
from toporlogy.configuration.registry import ConfigError
from toporlogy.configuration.resolve import read_config
from toporlogy.core.vtu import VTK_HEXAHEDRON, VTK_QUAD, VTK_TETRA, VTK_TRIANGLE, read_vtu, write_vtu
from toporlogy.design.chain import Chain, Embed, Passive
from toporlogy.design.interp import SIMP
from toporlogy.execution.cli import main
from toporlogy.fem.elements import D_plane_strain
from toporlogy.fem.mesh import perturb, rectangle
from toporlogy.optimization.mesh_study import MeshDomain, MeshElasticity, MeshFilterLink, MeshVolume
from toporlogy.optimization.responses import Compliance
from toporlogy.optimization.study import Constraint, Study
from toporlogy.optimizers.mma import MMA

from test_vtu import _domain, _fields

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLES = REPO / "examples" / "research"
NU = 0.3
P = 0.5


def _cfg(domain: dict[str, Any], supports: list[Any], cases: list[Any], rmin: float = 0.3,
         plane: str | None = None) -> dict[str, Any]:
    mat: dict[str, Any] = {"E0": 1.0, "nu": NU}
    if plane is not None:
        mat["plane"] = plane
    return {"name": "x", "domain": {"type": "mesh", **domain}, "material": mat, "supports": supports,
            "load_cases": cases, "chain": [{"type": "density_filter", "rmin": rmin}],
            "interpolation": {"type": "simp", "penal": 3.0, "emin_ratio": 1e-9}, "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.4}], "optimizer": "mma", "run": {"max_iter": 10}}


Q4 = {"generator": "rectangle", "n": [24, 8], "size": [3.0, 1.0], "element": "Q4", "perturb": 0.02, "seed": 5}
T3 = {**Q4, "element": "T3"}
H8 = {"generator": "box", "n": [6, 3, 3], "size": [2.0, 1.0, 1.0], "element": "H8"}


# ------------------------------------------------------------------ AX1: two cases = weighted sum of one-case studies
AX1: dict[str, tuple[Any, ...]] = {"q4": (Q4, [{"set": "left", "fix": "all"}], {"point": [3.0, 0.0], "force": [0, -1]},
              {"set": "top", "force": [0.3, -0.5], "distribute": "traction"}, 0.3),
       "h8": (H8, [{"set": "left", "fix": "all"}], {"point": [2.0, 0.0, 0.0], "force": [0, 0, -1]},
              {"set": "top", "force": [0.2, 0.0, -0.4], "distribute": "traction"}, 0.5)}


@pytest.mark.parametrize("case", list(AX1))
def test_ax1_two_cases_are_the_weighted_sum_of_one_case_studies(case: str) -> None:
    dom, sup, a, b, rmin = AX1[case]
    two = build_setup(_cfg(dom, sup, [{"name": "a", "weight": 0.7, "loads": [a]},
                                      {"name": "b", "weight": 1.3, "loads": [b]}], rmin))
    sa = build_setup(_cfg(dom, sup, [{"name": "a", "loads": [a]}], rmin)).study
    sb = build_setup(_cfg(dom, sup, [{"name": "b", "loads": [b]}], rmin)).study
    x = 0.2 + 0.7 * np.random.default_rng(41).random(sa.n_var)
    e2, ea, eb = two.study.evaluate(x), sa.evaluate(x), sb.evaluate(x)
    want, gwant = 0.7 * ea.f0 + 1.3 * eb.f0, 0.7 * ea.df0 + 1.3 * eb.df0
    rv = abs(e2.f0 - want) / abs(want)
    rg = np.abs(e2.df0 - gwant).max() / np.abs(gwant).max()
    rows = [(c["name"], c["weight"]) for c in two.cases]
    print(f"AX1 {case}: value {rv:.2e} (c = {want:.6e}), gradient {rg:.2e}, cases {rows}")
    assert rows == [("a", 0.7), ("b", 1.3)] and two.study.F.shape[1] == 2
    assert rv <= 1e-12 and rg <= 1e-12


# ------------------------------------------------------------------ AX2: resultant and nodal shares of a traction
def _loaded(dom: dict[str, Any], force: list[float], distribute: str = "traction") -> tuple[Any, np.ndarray]:
    s = build_setup(_cfg(dom, [{"set": "left", "fix": "all"}],
                         [{"loads": [{"set": "right", "force": force, "distribute": distribute}]}]))
    m = s.study.domain.mesh
    return m, np.asarray(s.study.F, float)[:, 0].reshape(-1, m.dim)


def _edge_shares(m: Any, Fn: np.ndarray, f: np.ndarray, n: int) -> float:
    nodes = m.node_sets["right"]
    y = m.points[nodes, 1]
    want = np.where((np.isclose(y, y.min()) | np.isclose(y, y.max()))[:, None], f / (2 * n), f / n)
    return float(np.abs(Fn[nodes] - want).max() / np.linalg.norm(f))


def _face_shares(m: Any, Fn: np.ndarray, f: np.ndarray, ny: int, nz: int) -> float:
    nodes = m.node_sets["right"]
    Y, Z = m.points[nodes, 1], m.points[nodes, 2]
    count = (np.where(np.isclose(Y, Y.min()) | np.isclose(Y, Y.max()), 1, 2)
             * np.where(np.isclose(Z, Z.min()) | np.isclose(Z, Z.max()), 1, 2))
    want = count[:, None] * f / (4 * ny * nz)
    return float(np.abs(Fn[nodes] - want).max() / np.linalg.norm(f))


AX2: dict[str, tuple[Any, ...]] = {"q4_48x16": ({"generator": "rectangle", "n": [48, 16], "size": [3.0, 1.0], "element": "Q4"}, [0.3, -1.0]),
       "t3_24x8": ({**T3, "seed": 3}, [0.3, -1.0]),
       "h8_4x2x2": ({"generator": "box", "n": [4, 2, 2], "size": [2.0, 1.0, 1.0], "element": "H8"}, [0.2, 0.5, -1.0]),
       "t4_2x2x2": ({"generator": "box", "n": [2, 2, 2], "size": [2.0, 1.0, 1.0], "element": "T4"}, [0.2, 0.5, -1.0])}


@pytest.mark.parametrize("case", list(AX2))
def test_ax2_traction_resultant_and_nodal_shares(case: str) -> None:
    dom, force = AX2[case]
    m, Fn = _loaded(dom, force)
    f = np.asarray(force, float)
    res = float(np.linalg.norm(Fn.sum(axis=0) - f) / np.linalg.norm(f))
    outside = np.setdiff1d(np.flatnonzero(np.any(Fn != 0, axis=1)), m.node_sets["right"])
    share = {"q4_48x16": lambda: _edge_shares(m, Fn, f, 16), "t3_24x8": lambda: _edge_shares(m, Fn, f, 8),
             "h8_4x2x2": lambda: _face_shares(m, Fn, f, 2, 2), "t4_2x2x2": lambda: 0.0}[case]()
    print(f"AX2 {case}: resultant {res:.2e}, shares {share:.2e}, loaded nodes outside the set {len(outside)}")
    assert res <= 1e-14 and share <= 1e-14 and not len(outside)


# ------------------------------------------------------------------ AX3 / AX4: patch tests with the closed forms
PATCH: dict[str, dict[str, Any]] = {"q4": {"generator": "rectangle", "n": [8, 4], "size": [2.0, 1.0], "element": "Q4", "perturb": 0.05,
                "seed": 1},
         "t3": {"generator": "rectangle", "n": [8, 4], "size": [2.0, 1.0], "element": "T3", "perturb": 0.05,
                "seed": 2},
         "h8": {"generator": "box", "n": [4, 2, 2], "size": [2.0, 1.0, 1.0], "element": "H8", "perturb": 0.03,
                "seed": 3},
         "t4": {"generator": "box", "n": [4, 2, 2], "size": [2.0, 1.0, 1.0], "element": "T4", "perturb": 0.03,
                "seed": 4}}


def _patch_error(case: str, distribute: str = "traction", plane: str = "stress") -> float:
    dom = PATCH[case]
    dim = len(dom["n"])
    sup = [{"set": "left", "fix": ["x"]}, {"set": "bottom", "fix": ["y"]}] + \
        ([{"set": "back", "fix": ["z"]}] if dim == 3 else [])
    force = [P] + [0.0] * (dim - 1)
    s = build_setup(_cfg(dom, sup, [{"loads": [{"set": "right", "force": force, "distribute": distribute}]}],
                         plane=None if dim == 3 else plane))
    st = s.study
    m = st.domain.mesh
    U = st.physics.solve(torch.ones(m.n_cells, dtype=torch.float64), st.F).detach().numpy()[:, 0].reshape(-1, dim)
    X = m.points
    sigma = P / float(np.prod(dom["size"][1:]))
    if dim == 3:
        exact = np.column_stack([sigma * X[:, 0], -NU * sigma * X[:, 1], -NU * sigma * X[:, 2]])
    elif plane == "strain":
        exact = np.column_stack([(1 - NU ** 2) * sigma * X[:, 0], -NU * (1 + NU) * sigma * X[:, 1]])
    else:
        exact = np.column_stack([sigma * X[:, 0], -NU * sigma * X[:, 1]])
    u = m.used_nodes()
    return float(np.abs(U[u] - exact[u]).max() / np.abs(exact[u]).max())


@pytest.mark.parametrize("case", list(PATCH))
def test_ax3_plane_stress_patch_test_with_a_consistent_traction(case: str) -> None:
    err = _patch_error(case)
    print(f"AX3 {case}: max |u - u_exact| / max |u_exact| = {err:.2e}")
    assert err <= 1e-10


def test_ax3_equal_split_is_not_consistent() -> None:
    err = _patch_error("q4", "nodes")
    print(f"AX3 q4 distribute nodes: max |u - u_exact| / max |u_exact| = {err:.2e}")
    assert err >= 1e-3


def _plane_strain_python(element: str) -> Study:
    m = perturb(rectangle(24, 8, 3.0, 1.0, element), 0.02, seed=5)
    fixed = m.dofs(m.node_sets["left"], [0, 1])
    F = np.zeros((2 * m.n_nodes, 1))
    k = int(np.argmin(np.linalg.norm(m.points - [3.0, 0.0], axis=1)))
    F[2 * k + 1, 0] = -1.0
    md: Any = MeshDomain(m)
    return Study(md, MeshElasticity(md, D_plane_strain(NU), fixed), F,
                 Chain([Embed(md), MeshFilterLink(md, 0.3), Passive(md)]), SIMP(md, E0=1.0, penal=3.0, emin_ratio=1e-9),
                 Compliance(), [Constraint(MeshVolume(md), 0.4)], MMA())


@pytest.mark.parametrize("case", ["q4", "t3"])
def test_ax4_plane_strain_yaml_equals_python(case: str) -> None:
    dom = Q4 if case == "q4" else T3
    ys = build_setup(_cfg(dom, [{"set": "left", "fix": ["x", "y"]}],
                          [{"loads": [{"point": [3.0, 0.0], "force": [0, -1]}]}], plane="strain")).study
    ps = _plane_strain_python(case.upper())
    x = 0.2 + 0.7 * np.random.default_rng(42).random(ps.n_var)
    ey, ep = ys.evaluate(x), ps.evaluate(x)
    rv = abs(ey.f0 - ep.f0) / abs(ep.f0)
    rg = np.abs(ey.df0 - ep.df0).max() / np.abs(ep.df0).max()
    print(f"AX4 {case}: value {rv:.2e} (c = {ep.f0:.6e}), gradient {rg:.2e}")
    assert rv <= 1e-12 and rg <= 1e-12


@pytest.mark.parametrize("case", ["q4", "t3"])
def test_ax4_plane_strain_patch_test(case: str) -> None:
    err = _patch_error(case, plane="strain")
    print(f"AX4 {case} plane strain patch: max |u - u_exact| / max |u_exact| = {err:.2e}")
    assert err <= 1e-10


def test_ax4_plane_strain_is_refused_in_3d() -> None:
    cfg = _cfg(H8, [{"set": "left", "fix": "all"}], [{"loads": [{"set": "right", "force": [0, 0, -1]}]}])
    cfg["material"]["plane"] = "strain"
    with pytest.raises(ConfigError) as err:
        build_setup(cfg)
    hits = [m for p, m in err.value.errors if p == "material.plane"]
    print(f"AX4 3-D plane strain: {err.value}")
    assert hits and all("mesh" in m for m in hits)


# ------------------------------------------------------------------ AX5: VTU of the result, every cell type
VTU: dict[str, tuple[Any, ...]] = {"q4": ({"generator": "rectangle", "n": [12, 4], "size": [3.0, 1.0], "element": "Q4", "perturb": 0.02,
               "seed": 6}, 2, VTK_QUAD),
       "t3": ({"generator": "rectangle", "n": [12, 4], "size": [3.0, 1.0], "element": "T3"}, 1, VTK_TRIANGLE),
       "h8": ({"generator": "box", "n": [4, 2, 2], "size": [2.0, 1.0, 1.0], "element": "H8"}, 1, VTK_HEXAHEDRON),
       "t4": ({"generator": "box", "n": [3, 2, 2], "size": [2.0, 1.0, 1.0], "element": "T4"}, 1, VTK_TETRA)}


def _vtu_run(case: str, folder: Path) -> tuple[Any, Any, Any]:
    dom, ncases, _ = VTU[case]
    dim = len(dom["n"])
    down = [0.0] * (dim - 1) + [-1.0]
    loads = [[{"set": "right", "force": down, "distribute": "traction"}],
             [{"set": "top", "force": [0.5] + [0.0] * (dim - 1)}]][:ncases]
    cfg = _cfg(dom, [{"set": "left", "fix": "all"}], [{"name": f"c{i}", "loads": ld} for i, ld in enumerate(loads)],
               rmin=0.4)
    cfg["post"] = [{"type": "vtu"}]
    s = build_setup(cfg)
    res = s.study.run(max_iter=2, tol=0.0)
    name, step = s.post[0]
    assert name == "vtu"
    out = step(res, s.study, folder)
    return s, res, out


@pytest.mark.parametrize("case", list(VTU))
def test_ax5_vtu_roundtrip_is_exact(case: str, tmp_path: Path) -> None:
    s, res, out = _vtu_run(case, tmp_path)
    st = s.study
    m, dim, ncases, kind = st.domain.mesh, st.domain.dim, VTU[case][1], VTU[case][2]
    r = read_vtu(tmp_path / out["file"])
    pts = np.zeros((m.n_nodes, 3))
    pts[:, :dim] = m.points
    E = st.interp(torch.as_tensor(np.asarray(res.xphys, float))).detach().numpy().reshape(-1)
    U = np.asarray(res.U, float).reshape(st.domain.ndof, -1)
    ok = {"points": np.array_equal(r.points, pts), "cells": np.array_equal(r.connectivity, m.cells),
          "types": bool((r.cell_types == kind).all()) and len(r.cell_types) == m.n_cells,
          "xphys": np.array_equal(r.cell_data["xphys"], np.asarray(res.xphys, float).reshape(-1)),
          "E": np.array_equal(r.cell_data["E"], E), "cases": U.shape[1] == ncases}
    for c in range(ncases):
        u = np.zeros((m.n_nodes, 3))
        u[:, :dim] = U[:, c].reshape(-1, dim)
        ok[f"displacement_{c}"] = np.array_equal(r.point_data[f"displacement_{c}"], u)
    print(f"AX5 {case}: {ok}, meta element {r.meta.get('element')}, arrays {out['cell_arrays']} "
          f"{out['point_arrays']}")
    assert all(ok.values()) and r.meta["element"] == m.element and sorted(r.point_data) == out["point_arrays"]


@pytest.mark.parametrize("case", list(VTU))
def test_ax5_vtk_reads_the_cells_with_positive_sizes(case: str, tmp_path: Path) -> None:
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    s, res, out = _vtu_run(case, tmp_path)
    d = s.study.domain
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(tmp_path / out["file"]))
    reader.Update()
    g = reader.GetOutput()
    types = {g.GetCellType(i) for i in range(g.GetNumberOfCells())}
    xp = vtk_to_numpy(g.GetCellData().GetArray("xphys"))
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(g)
    size.Update()
    sz = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Area" if d.dim == 2 else "Volume"))
    total = float(d.cell_volume.sum())
    rel = abs(float(sz.sum()) - total) / total
    print(f"AX5 vtk {case}: {g.GetNumberOfCells()} cells, types {types}, min size {sz.min():.3e}, total {rel:.2e}")
    assert g.GetNumberOfCells() == d.mesh.n_cells and g.GetNumberOfPoints() == d.mesh.n_nodes
    assert types == {VTU[case][2]} and np.array_equal(xp, np.asarray(res.xphys, float).reshape(-1))
    assert (sz > 0).all() and rel <= 1e-12


def test_ax5_cli_run_writes_the_vtu(tmp_path: Path) -> None:
    cfg = read_config(EXAMPLES / "mesh_cantilever2d.yaml")
    cfg["post"] = [{"type": "vtu"}]
    cfg["run"]["max_iter"] = 3
    src = tmp_path / "mesh_vtu.yaml"
    src.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    folder = tmp_path / "run"
    code = main(["run", str(src), "--out", str(folder), "-q"])
    sm = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    r = read_vtu(folder / "result.vtu")
    with np.load(folder / "result.npz") as z:
        same = np.array_equal(r.cell_data["xphys"], z["xphys"])
    print(f"AX5 CLI: exit {code}, post {sm['post']['vtu']['summary']}, xphys as result.npz {same}")
    assert sm["post"]["vtu"]["file"] == "result.vtu" and same


# ------------------------------------------------------------------ AX6: what is still refused
def _with(path: str, value: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(base) if base is not None else read_config(EXAMPLES / "mesh_cantilever2d.yaml")
    node: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value
    return cfg


TIP = {"point": [3.0, 0.0], "force": [0, -1]}
BOX = _cfg(H8, [{"set": "left", "fix": "all"}], [{"loads": [{"set": "right", "force": [0, 0, -1]}]}])
UNSUPPORTED: dict[str, tuple[str, Any, str, Any]] = {
    "post_export": ("post", [{"type": "export"}], "post.0", None),
    "post_finalize": ("post", [{"type": "finalize"}], "post.0", None),
    "vtu_param": ("post", [{"type": "vtu", "fields": ["xphys"]}], "post.0.fields", None),
}
INVALID: dict[str, tuple[str, Any, str, Any]] = {
    "top_level_loads": ("loads", [TIP], "loads", None),
    "duplicate_name": ("load_cases", [{"name": "a", "loads": [TIP]}, {"name": "a", "loads": [TIP]}],
                       "load_cases.1.name", None),
    "infinite_weight": ("load_cases.0.weight", float("inf"), "load_cases.0.weight", None),
    "distribute_value": ("load_cases.0.loads", [{"set": "right", "force": [0, -1], "distribute": "spread"}],
                         "load_cases.0.loads.0.distribute", None),
    "point_distribute": ("load_cases.0.loads", [{**TIP, "distribute": "traction"}],
                         "load_cases.0.loads.0.distribute", None),
    "plane_strain_3d": ("material.plane", "strain", "material.plane", BOX),
    "plane_membrane": ("material.plane", "membrane", "material.plane", None),
}


@pytest.mark.parametrize("case", list(UNSUPPORTED) + list(INVALID))
def test_ax6_unsupported_and_invalid_keys_raise(case: str) -> None:
    key, value, path, base = UNSUPPORTED.get(case) or INVALID[case]
    with pytest.raises(ConfigError) as err:
        build_setup(_with(key, value, base))
    hits = [m for p, m in err.value.errors if p == path]
    print(f"AX6 {case}: {err.value}")
    assert hits and all("mesh" in m for m in hits), err.value.errors
    if case in UNSUPPORTED:
        assert any("not support" in m for m in hits), err.value.errors
    if case == "top_level_loads":
        assert any("load_cases" in m for m in hits), err.value.errors


# ------------------------------------------------------------------ AX7: the anchors measured before the code
ANCHOR = {"mesh_cantilever2d": ("1908.3237613774095", "4793a979cd435212", "a627b5d4d61a4200", (1666, 1)),
          "shell_cylinder": ("30131.882810400843", "ce282cdd37f56919", "1581708cfbc900cb", (1326, 1))}
GRID_VTU = {2: "aa39645340d0cfd74c748d040828f8cc93422058dfd4da0b09ffb30eaeffa1b8",
            3: "3ed5230e38c0241143500535abc7a2a7b83f16098b6fe7789762416e1cee88ce"}


def _h(a: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("example", list(ANCHOR))
def test_ax7_converted_examples_give_the_anchored_study(example: str) -> None:
    st = build_setup(read_config(EXAMPLES / f"{example}.yaml")).study
    ev = st.evaluate(st.default_start())
    F = np.asarray(st.F, float)
    got = (repr(float(ev.f0)), _h(ev.df0), _h(F), F.shape)
    print(f"AX7 {example}: {got}")
    assert got == ANCHOR[example]


@pytest.mark.parametrize("dim", [2, 3])
def test_ax7_grid_vtu_bytes_are_the_anchored_ones(dim: int, tmp_path: Path) -> None:
    d = _domain(dim)
    point, cell = _fields(d)
    p = write_vtu(tmp_path / "g.vtu", d, point, cell, meta={"units": "consistent"})
    sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    print(f"AX7 grid VTU {dim}-D: {sha}")
    assert sha == GRID_VTU[dim]
