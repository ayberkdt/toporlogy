"""toporlogy.post.finalize on analytic shapes and a synthetic two-member topology.

Pre-registered acceptance (assignment, 13 Nov 2026): volume after the whole pipeline within 0.5 % of the exact
solid, bore radius within 1 % of nominal over the full face, watertight closed manifold, Hausdorff distance to the
raw iso-surface small relative to the fine grid step, thin neck kept (one body) by the adaptive opening.
"""
from __future__ import annotations

import numpy as np
import pytest

from toporlogy import Domain
from toporlogy.post import Cut, CylinderPrimitive, Keep, PostParams, finalize, manufacturing_checks
from toporlogy.post.primitives import BoxPrimitive, element_fraction



def _common(res, exact: float) -> None:
    r = res.report
    assert r["edges"]["open"] == 0 and r["edges"]["non_manifold"] == 0
    assert r["edges"]["misoriented"] == 0 and r["edges"]["non_manifold_vertices"] == 0
    assert abs(r["volume_mesh"] / exact - 1) < 0.005
    h_f = res.grid.h
    assert r["vs_raw"]["hausdorff_final_to_ref"] < 0.6 * h_f
    assert r["vs_raw"]["hausdorff_ref_to_final"] < 0.6 * h_f
    assert r["quality"]["min_angle_deg"] > 15.0
    assert r["topology_mesh"]["bodies"] == r["topology_grid"]["bodies"]


def test_sphere() -> None:
    d = Domain((16, 16, 16), h=1.0)
    c, R = np.array([8.0, 8.0, 8.0]), 5.5
    x = element_fraction(d, lambda X: np.linalg.norm(X - c, axis=1) - R)
    res = finalize(d, x)
    exact = 4 / 3 * np.pi * R ** 3
    _common(res, exact)
    rad = np.linalg.norm(res.V - c, axis=1)
    assert np.abs(rad - R).max() < 0.02 * R
    assert res.report["topology_mesh"]["euler"] == 1


def test_tube_cylinder_with_hole() -> None:
    """Free outer cylinder (from the density), exact coaxial hole (Cut)."""
    d = Domain((14, 14, 10), h=1.0)
    c = np.array([7.0, 7.0, 5.0])
    outer, inner = CylinderPrimitive(c, (0, 0, 1), 5.5), CylinderPrimitive(c, (0, 0, 1), 2.5)
    solid = lambda X: np.maximum(np.maximum(outer.sdf(X), -inner.sdf(X)), np.abs(X[:, 2] - 5.0) - 3.0)  # noqa: E731
    x = element_fraction(d, solid)
    res = finalize(d, x, preserve=[Cut(inner, name="hole")])
    exact = np.pi * (5.5 ** 2 - 2.5 ** 2) * 6.0
    _common(res, exact)
    hole = next(i for i in res.report["interfaces"] if i["name"] == "hole")
    assert hole["radius_rel_error_max"] < 0.01
    assert res.report["topology_mesh"]["tunnels"] == 1


def test_plate_with_bore() -> None:
    d = Domain((16, 16, 6), h=1.0)
    c, R = np.array([8.0, 8.0, 3.0]), 3.5
    bore = CylinderPrimitive(c, (0, 0, 1), R)
    plate = lambda X: np.maximum(np.maximum(1 - X[:, 2], X[:, 2] - 5), -bore.sdf(X))  # noqa: E731
    x = element_fraction(d, plate)
    res = finalize(d, x, preserve=[Cut(bore, name="bore")])
    exact = 16 * 16 * 4 - np.pi * R * R * 4
    _common(res, exact)
    b = next(i for i in res.report["interfaces"] if i["name"] == "bore")
    assert abs(b["radius_mean"] / R - 1) < 0.01
    assert b["radius_rel_error_max"] < 0.01 and b["vertices"] > 200
    env = next(i for i in res.report["interfaces"] if i["name"] == "envelope")
    assert max(abs(env["distance_min"]), abs(env["distance_max"])) < 0.1 * res.grid.h
    ck = manufacturing_checks(res, min_wall=3.5)
    assert ck.passed, ck.table()
    assert abs(ck.measured["wall_ray"]["p5"] - 4.0) < 0.1


def test_neck_keeps_one_body() -> None:
    """Two bars joined by a 2×2-element neck (half thickness h); the start radius 1.6 h would cut it."""
    d = Domain((30, 10, 10), h=1.0)
    bar1, bar2 = BoxPrimitive((1, 2, 2), (13, 8, 8)), BoxPrimitive((17, 2, 2), (29, 8, 8))
    neck = BoxPrimitive((12, 4, 4), (18, 6, 6))
    solid = lambda X: np.minimum(np.minimum(bar1.sdf(X), bar2.sdf(X)), neck.sdf(X))  # noqa: E731
    x = element_fraction(d, solid)
    pads = [Keep(BoxPrimitive((1, 2, 2), (3, 8, 8)), name="support pad"), Keep(BoxPrimitive((27, 2, 2), (29, 8, 8)), name="load pad")]
    res = finalize(d, x, preserve=pads, params=PostParams(r_open=1.6))
    op = res.report["opening"]
    assert op["trials"][0]["accepted"] is False
    assert op["trials"][0]["topology"]["bodies"] == 2          # the first radius would have split the part
    assert op["topology_before"]["bodies"] == 1
    assert 0 < op["r_used"] < 1.0
    assert res.report["topology_mesh"]["bodies"] == 1
    assert res.report["topology_grid"]["bodies"] == 1


@pytest.mark.parametrize("smoothing", ["sinc"])
def test_sinc_and_decimation(smoothing: str) -> None:
    d = Domain((12, 12, 12), h=1.0)
    c, R = np.array([6.0, 6.0, 6.0]), 4.0
    x = element_fraction(d, lambda X: np.linalg.norm(X - c, axis=1) - R)
    res = finalize(d, x, params=PostParams(smoothing=smoothing, target_faces=4000))
    r = res.report
    assert r["decimation"]["applied"] and len(res.F) < 6000
    assert r["edges"]["open"] == 0 and r["edges"]["non_manifold"] == 0
    assert abs(r["volume_mesh"] / (4 / 3 * np.pi * R ** 3) - 1) < 0.01


# ------------------------------------------------------------------------------------- PR-05: CAD envelope
@pytest.mark.parametrize("source", ["csg", "stl"])
def test_cad_envelope_follows_the_curved_part_surface(source: str) -> None:
    """Full-density cylinder (no volume matching): the finished side face is the envelope. Measured 24 Nov 2026 at
    h_f = 1/3: envelope="voxel" max |r − R| = 0.113 (0.34 h_f), "cad" from the CSG shape 0.0025 (0.008 h_f)."""
    import pyvista as pv

    from toporlogy import geometry as G
    R, H, c = 5.3, 6.0, np.array([0.37, 0.21])
    shape = G.Cylinder((c[0], c[1], 0.0), (c[0], c[1], H), R)
    d = G.voxelize(shape, 1.0, padding=1)
    if source == "csg":
        part = shape
    else:                                       # 192-gon: within R·(1 − cos(π/192)) = 7e-4 of the cylinder
        cy = pv.Cylinder(center=(c[0], c[1], H / 2), direction=(0, 0, 1), radius=R, height=H, resolution=192,
                         capping=True).triangulate()
        part = (np.asarray(cy.points, float), np.asarray(cy.faces).reshape(-1, 4)[:, 1:])
    err = {}
    for env in ("voxel", "cad"):
        res = finalize(d, np.ones(d.shape), params=PostParams(upsample=3, envelope=env, volume_match="none"),
                       part=part if env == "cad" else None)
        side = np.abs(res.V[:, 2] - H / 2) < 2.0
        err[env] = float(np.abs(np.hypot(res.V[side, 0] - c[0], res.V[side, 1] - c[1]) - R).max())
        h_f = res.grid.h
    assert err["cad"] < 0.25 * h_f
    assert err["cad"] < 0.1 * err["voxel"]
    print(f"\n  {source}: max |r - R| voxel {err['voxel']:.4f}, cad {err['cad']:.4f} (h_f {h_f:.3f})")


def test_cad_envelope_needs_the_part_geometry() -> None:
    d = Domain((6, 6, 6), h=1.0, frac=np.full((6, 6, 6), 0.5))
    with pytest.raises(ValueError, match="needs the part geometry"):
        finalize(d, np.ones(d.shape), params=PostParams(upsample=2, envelope="cad"))
    with pytest.raises(ValueError, match=r'"voxel" \| "cad"'):
        finalize(d, np.ones(d.shape), params=PostParams(upsample=2, envelope="exact"))


def test_yaml_envelope_cad_needs_a_csg_or_stl_domain() -> None:
    from pathlib import Path

    import yaml

    from toporlogy.configuration.build import build_setup
    from toporlogy.configuration.registry import ConfigError
    ex = Path(__file__).resolve().parents[2] / "examples"
    raw = yaml.safe_load((ex / "lbracket_bore3d.yaml").read_text(encoding="utf-8"))      # domain type csg
    for st in raw["post"]:
        if st["type"] == "finalize":
            st["envelope"] = "cad"
    assert "finalize" in [n for n, _ in build_setup(raw).post]
    raw = yaml.safe_load((ex / "cantilever3d.yaml").read_text(encoding="utf-8"))         # domain type box
    raw["post"] = [{"type": "finalize", "envelope": "cad"}]
    with pytest.raises(ConfigError, match="envelope: cad needs the part geometry"):
        build_setup(raw)
