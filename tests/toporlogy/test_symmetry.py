"""Symmetry diagnosis (docs/yol_haritasi_2026-11-18.md §2.3, PR-10): e_sym of element fields on analytic symmetric
and asymmetric fields, the geometric mirror distance of surfaces, and the ``symmetry_check`` post step."""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy import Domain
from toporlogy.configuration.build import build_setup
from toporlogy.design.manufacturing import SymmetryTying, mirror_partner
from toporlogy.execution.runner import run_setup
from toporlogy.optimization.diagnostics import field_symmetry


# ---------------------------------------------------------------------------------------------- element fields
def test_symmetric_field_has_zero_error_on_a_grid_line_plane() -> None:
    d = Domain((8, 6))
    X = d.element_centers()
    f = np.exp(-((X[:, 0] - 4.0) ** 2)) * (1.0 + X[:, 1])         # even in x about x = 4 (a grid line)
    out = field_symmetry(d, {"f": f}, "x", 4.0)
    assert out["fields"]["f"]["e_sym"] == 0.0
    assert out["fields"]["f"]["mismatch_cells"] == 0
    assert out["grid_shift_max"] == 0.0 and out["cells_outside_box"] == 0


def test_linear_field_has_the_closed_form_error() -> None:
    """f = (1, 2, 3, 4) about the box middle: R(f) = (4, 3, 2, 1), ‖f − R f‖ / ‖f‖ = √20 / √30 = √(2/3)."""
    d = Domain((4, 1))
    out = field_symmetry(d, {"f": np.array([1.0, 2.0, 3.0, 4.0])}, 0, 2.0, threshold=2.5)
    assert out["fields"]["f"]["e_sym"] == pytest.approx(np.sqrt(2.0 / 3.0), rel=1e-14)
    assert out["fields"]["f"]["mismatch_cells"] == 4                 # every cell flips across 2.5


def test_disjoint_mirror_images_give_sqrt2() -> None:
    d = Domain((8, 6))
    f = (d.element_centers()[:, 0] < 4.0).astype(float)              # left half only
    out = field_symmetry(d, {"f": f}, "x", 4.0)
    assert out["fields"]["f"]["e_sym"] == pytest.approx(np.sqrt(2.0), rel=1e-14)
    assert out["fields"]["f"]["mismatch_cells"] == 48


def test_mask_error_is_the_root_of_the_mismatch_share() -> None:
    """A 0/1 mask with n cells and m mismatched cells: e_sym = √(m / n) (the 228-cell / 0.224 relation of §2.3)."""
    d = Domain((10, 6, 4))
    keep = np.zeros(d.shape, bool)
    keep[2:8, 1:3, :] = True                                         # symmetric about x = 5
    keep[1, 4, 0:3] = True                                           # 3 extra cells on the left only
    out = field_symmetry(d, {"keep": keep.astype(float)}, "x", 5.0)
    m = out["fields"]["keep"]["mismatch_cells"]
    assert m == 6                                                    # the 3 cells and their empty mirror cells
    assert out["fields"]["keep"]["e_sym"] == pytest.approx(np.sqrt(m / keep.sum()), rel=1e-14)
    assert field_symmetry(d, {"keep": keep.astype(float)}, "y", 3.0)["fields"]["keep"]["e_sym"] > 0


def test_off_grid_plane_reports_the_grid_shift_and_cells_leaving_the_box() -> None:
    """Plane x = 4.3 on unit cells: mirrored centres 8.1 − i sit 0.4 from the partner centres; column 0 (x' = 8.1)
    leaves the 8-cell box, so its mirror value is 0."""
    d = Domain((8, 6))
    out = field_symmetry(d, {"one": np.ones(d.ne)}, "x", 4.3)
    assert out["grid_shift_max"] == pytest.approx(0.4, abs=1e-12)
    assert out["cells_outside_box"] == 6
    assert out["fields"]["one"]["e_sym"] == pytest.approx(np.sqrt(6 / 48), rel=1e-14)


def test_zero_field_and_wrong_size() -> None:
    d = Domain((4, 4))
    assert field_symmetry(d, {"z": np.zeros(d.ne)}, "y", 2.0)["fields"]["z"]["e_sym"] == 0.0
    with pytest.raises(ValueError, match="box has 16 elements"):
        field_symmetry(d, {"bad": np.zeros(5)}, "x", 2.0)
    with pytest.raises(ValueError, match="not valid in 2-D"):
        field_symmetry(d, {"z": np.zeros(d.ne)}, "z", 0.0)


def test_partner_is_the_relation_symmetry_tying_ties() -> None:
    d = Domain((9, 5, 3), h=2.5, origin=(1.0, -3.0, 0.0))
    for axis, c in (("x", 12.0), ("y", 1.7), ("z", 4.0)):
        assert np.array_equal(SymmetryTying(d, {axis: c}).maps[0], mirror_partner(d, axis, c))


# ---------------------------------------------------------------------------------------------- surfaces
def _box_mesh(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = np.array([t for a, b, c, e in quads for t in ((a, b, c), (a, c, e))], np.int64)
    return V, F


def test_mirror_distance_of_a_symmetric_and_a_shifted_box() -> None:
    """Box [0, 2] × [0, 1]²: about x = 1 it is symmetric (0); about x = 1.5 its image is [1, 3] × [0, 1]², whose
    face x = 3 lies 1 from the box, so d_H = 1 = 2·0.5 (attained at mirrored vertices, which are samples)."""
    pytest.importorskip("pyvista")
    from toporlogy.post.mesh import mirror_distance
    V, F = _box_mesh((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    sym = mirror_distance(V, F, "x", 1.0)
    assert sym["hausdorff"] < 1e-9 and sym["mean"] < 1e-9 and sym["p95"] < 1e-9
    off = mirror_distance(V, F, 0, 1.5)
    assert off["hausdorff"] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < off["mean"] <= off["p95"] <= off["hausdorff"]
    assert off["bbox_diagonal"] == pytest.approx(np.sqrt(6.0)) and off["samples"] == len(V) + len(F)
    assert mirror_distance(V, F, "y", 0.5)["hausdorff"] < 1e-9
    with pytest.raises(ValueError, match="axis"):
        mirror_distance(V, F, 3, 0.0)


# ---------------------------------------------------------------------------------------------- post step
def _cantilever(post: list) -> dict:
    """2-D cantilever symmetric about y = mid (load at the middle of the free end)."""
    return {"name": "sym", "domain": {"type": "box", "shape": [12, 6], "h": 1.0},
            "supports": [{"type": "fixed", "where": {"type": "plane", "axis": "x", "at": "min"}}],
            "load_cases": [{"name": "tip", "loads": [{"type": "force", "where": {"type": "point",
                                                                                 "at": ["max", "mid"]},
                                                       "force": [0, -1]}]}],
            "chain": [{"type": "density_filter", "rmin": 1.5}], "objective": "compliance",
            "constraints": [{"response": "volume", "upper": 0.5}], "optimizer": "oc",
            "run": {"max_iter": 3}, "post": post}


def test_symmetry_check_step_writes_summary_and_gives_no_verdict(tmp_path) -> None:
    s = build_setup(_cantilever([{"type": "symmetry_check", "planes": [["x", "mid"], ["y", 3.0]]}]))
    assert not any("symmetry" in type(lk).__name__.lower() for lk in s.study.chain.links)   # tying stays opt-in
    sm = run_setup(s, tmp_path / "run")
    out = sm["post"]["symmetry_check"]
    assert "passed" not in out and "symmetry_check failed" not in sm["status"]
    assert out["design"] == "optimized" and out["mesh_design"] is None
    px, py = out["planes"]
    assert (px["axis"], px["coordinate"], py["axis"], py["coordinate"]) == ("x", 6.0, "y", 3.0)
    assert px["fields"]["frac"]["e_sym"] == 0.0 and py["fields"]["frac"]["e_sym"] == 0.0     # full box
    assert px["fields"]["keep"]["e_sym"] == 0.0                                              # no keep: ‖0‖
    assert px["fields"]["xphys"]["e_sym"] > 0.05                     # clamped left edge, free right edge
    x = np.load(tmp_path / "run" / "result.npz")["xphys"]
    assert py["fields"]["xphys"]["e_sym"] == pytest.approx(
        field_symmetry(s.study.domain, {"x": x}, "y", 3.0)["fields"]["x"]["e_sym"], rel=1e-12)
    assert "x=6: e_sym frac 0.000" in out["summary"]


def test_symmetry_check_rejects_empty_planes() -> None:
    from toporlogy.configuration.registry import ConfigError
    with pytest.raises((ConfigError, ValueError), match="at least one plane"):
        build_setup(_cantilever([{"type": "symmetry_check", "planes": {}}]))
