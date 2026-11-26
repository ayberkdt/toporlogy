"""VTU export of the active grid (docs/olcumler/vtu_onkayit.md, W1-W6, pre-registered 2026-11-26 02:42 +0300 before
the writer existed). vtk is the independent reader of W2 / W3; without it those checks are skipped (NOT VERIFIED)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from toporlogy.core.domain import Domain
from toporlogy.core.vtu import VOIGT_CONVENTION, VTK_HEXAHEDRON, VTK_QUAD, read_vtu, write_vtu


def _domain(dim: int) -> Domain:
    if dim == 2:
        void = np.zeros((6, 4), bool)
        void[2:4, 1:3] = True                                   # a hole
        frac = np.ones((6, 4))
        frac[0, 0] = 0.4                                        # a partial volume
        return Domain((6, 4), h=0.5, void=void, frac=frac, thickness=0.2)
    void = np.zeros((4, 3, 2), bool)
    void[1, 1, :] = True
    return Domain((4, 3, 2), h=0.7, void=void)


def _fields(d: Domain) -> tuple[dict, dict]:
    rng = np.random.default_rng(d.dim)
    point = {"u": rng.normal(size=(d.nn, 3)), "T": rng.normal(size=d.nn)}
    cell = {"xphys": rng.random(d.shape), "flag": (rng.random(d.n_active) > 0.5),
            "stress": rng.normal(size=(d.n_active, 6 if d.dim == 3 else 3))}
    return point, cell


@pytest.mark.parametrize("dim", [2, 3])
def test_w1_w4_roundtrip_is_exact_and_holds_the_active_elements(dim: int, tmp_path: Path) -> None:
    d = _domain(dim)
    point, cell = _fields(d)
    p = write_vtu(tmp_path / "f.vtu", d, point, cell, meta={"units": "consistent"})
    r = read_vtu(p)
    el = d.active_elements
    assert r.connectivity.shape == (d.n_active, 4 if dim == 2 else 8)
    assert np.array_equal(r.connectivity, d.edof[el][:, ::dim] // dim)                         # W4: order
    assert set(np.unique(r.cell_types)) == {VTK_QUAD if dim == 2 else VTK_HEXAHEDRON}
    assert np.array_equal(r.points[:, :dim], d.node_coords())
    assert np.array_equal(r.point_data["u"], point["u"]) and np.array_equal(r.point_data["T"], point["T"])
    assert np.array_equal(r.cell_data["xphys"], cell["xphys"].ravel()[el])
    assert np.array_equal(r.cell_data["flag"], cell["flag"].astype(np.uint8))
    assert np.array_equal(r.cell_data["stress"], cell["stress"])
    assert r.meta["convention"] == VOIGT_CONVENTION and r.meta["units"] == "consistent"


@pytest.mark.parametrize("dim", [2, 3])
def test_w2_w3_vtk_reads_the_same_arrays_and_positive_cell_sizes(dim: int, tmp_path: Path) -> None:
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    d = _domain(dim)
    point, cell = _fields(d)
    p = write_vtu(tmp_path / "f.vtu", d, point, cell)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(p))
    reader.Update()
    g = reader.GetOutput()
    assert g.GetNumberOfCells() == d.n_active and g.GetNumberOfPoints() == d.nn
    assert {g.GetCellType(i) for i in range(g.GetNumberOfCells())} == {VTK_QUAD if dim == 2 else VTK_HEXAHEDRON}
    assert np.array_equal(vtk_to_numpy(g.GetPointData().GetArray("u")), point["u"])
    assert np.array_equal(vtk_to_numpy(g.GetCellData().GetArray("xphys")), cell["xphys"].ravel()[d.active_elements])
    assert np.array_equal(vtk_to_numpy(g.GetCellData().GetArray("stress")), cell["stress"])
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(g)
    size.Update()
    name = "Area" if dim == 2 else "Volume"
    s = vtk_to_numpy(size.GetOutput().GetCellData().GetArray(name))
    assert (s > 0).all()
    assert abs(s.sum() - d.n_active * d.h ** dim) <= 1e-12 * d.n_active * d.h ** dim


def test_w5_wrong_shapes_and_convention_fail_and_nan_survives(tmp_path: Path) -> None:
    d = _domain(2)
    with pytest.raises(ValueError, match="rows"):
        write_vtu(tmp_path / "a.vtu", d, point_data={"u": np.zeros(d.nn - 1)})
    with pytest.raises(ValueError, match="rows"):
        write_vtu(tmp_path / "b.vtu", d, cell_data={"x": np.zeros(d.n_active + 1)})
    with pytest.raises(ValueError, match="convention"):
        write_vtu(tmp_path / "c.vtu", d, meta={"convention": "Mandel"})
    v = np.arange(d.n_active, dtype=float)
    v[0], v[1], v[2] = math.nan, math.inf, -math.inf
    r = read_vtu(write_vtu(tmp_path / "d.vtu", d, cell_data={"v": v}))
    assert np.array_equal(r.cell_data["v"], v, equal_nan=True)


def test_w6_post_step_writes_the_returned_design(tmp_path: Path) -> None:
    from toporlogy.execution.cli import main
    import yaml
    cfg = {"domain": {"type": "box", "shape": [12, 6], "h": 1.0},
           "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
           "load_cases": [{"loads": [{"type": "force", "where": {"type": "point", "at": ["max", "min"]},
                                      "force": [0, -1]}]}],
           "chain": [{"type": "density_filter", "rmin": 1.5}],
           "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "mma",
           "run": {"max_iter": 3}, "post": [{"type": "vtu"}]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "run"
    main(["run", str(p), "--out", str(out), "-q"])
    r = read_vtu(out / "result.vtu")
    res = np.load(out / "result.npz")
    d = Domain((12, 6))
    assert np.array_equal(r.cell_data["xphys"], res["xphys"].ravel()[d.active_elements])
    assert {"E", "von_mises_solid_0"} <= set(r.cell_data) and "displacement_0" in r.point_data
