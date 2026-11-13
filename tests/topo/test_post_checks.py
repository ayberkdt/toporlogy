"""aerokit.topo.post: mesh checks, manufacturing measures, exporters, print variant with machining stock."""
from __future__ import annotations

import numpy as np

from aerokit.topo import Domain
from aerokit.topo.post import Bound, Cut, Cylinder, Plane, finalize, print_variant
from aerokit.topo.post import mesh as M
from aerokit.topo.post import sdf as S
from aerokit.topo.post.checks import overhang, ray_thickness
from aerokit.topo.post.export import read_stl, write_obj, write_ply, write_stl
from aerokit.topo.post.primitives import Box, element_fraction


def _cube(lo: float = 0.0, hi: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    import pyvista as pv
    c = pv.Cube(center=(0.5 * (lo + hi),) * 3, x_length=hi - lo, y_length=hi - lo, z_length=hi - lo)
    V, F = M.from_polydata(c.triangulate().clean())
    if M.volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def test_edge_report_detects_defects() -> None:
    V, F = _cube()
    assert M.is_watertight(F, len(V)) and abs(M.volume(V, F) - 1) < 1e-12
    r = M.edge_report(F[1:], len(V))
    assert r["open"] == 3
    flipped = F.copy()
    flipped[0] = flipped[0, ::-1]
    assert M.edge_report(flipped, len(V))["misoriented"] == 3
    V2, F2 = _cube(1.0, 2.0)                                       # touches the first cube at one vertex
    VV, FF = np.vstack([V, V2]), np.vstack([F, F2 + len(V)])
    VV, FF = M.clean(VV, FF, tol=1e-9)
    assert M.edge_report(FF, len(VV))["non_manifold_vertices"] == 1
    assert M.mesh_topology(VV, FF)["shells"] == 2                  # edge-connected shells: the vertex does not join


def test_overhang_area() -> None:
    V, F = _cube(1.0, 2.0)
    o = overhang(V, F, (0, 0, 1), 45.0, plate_level=0.0)
    assert abs(o["area"] - 1.0) < 1e-12                            # the bottom face, above the plate
    o2 = overhang(V, F, (0, 0, 1), 45.0)                            # plate = part bottom
    assert o2["area"] == 0.0 and abs(o2["plate_contact_area"] - 1.0) < 1e-12


def test_ray_thickness_slab() -> None:
    g = S.Grid(np.array([-6.0, -6.0, -3.0]) + 0.125, 0.25, (48, 48, 24))
    phi = Box((-5, -5, -1), (5, 5, 1)).sdf(g.points()).reshape(g.shape)
    V, F = M.extract(phi, g)
    t = ray_thickness(V, F, phi, g)
    assert abs(t["p5"] - 2.0) < 0.02 and abs(t["median"] - 2.0) < 0.02


def test_writers_round_trip(tmp_path) -> None:
    import pyvista as pv
    V, F = _cube()
    p = write_stl(tmp_path / "c.stl", V, F)
    V2, F2 = read_stl(p)
    assert len(F2) == len(F) and abs(M.volume(V2, F2) - 1) < 1e-6
    for w, ext in ((write_ply, "ply"), (write_obj, "obj")):
        m = pv.read(w(tmp_path / f"c.{ext}", V, F))
        assert m.n_cells == len(F) and abs(abs(m.volume) - 1) < 1e-9


def test_print_variant_stock() -> None:
    d = Domain((14, 14, 6), h=1.0)
    c, R = np.array([7.0, 7.0, 3.0]), 3.0
    bore = Cylinder(c, (0, 0, 1), R)
    plate = lambda X: np.maximum(np.maximum(1 - X[:, 2], X[:, 2] - 5), -bore.sdf(X))  # noqa: E731
    x = element_fraction(d, plate)
    stock = 0.5
    face = Bound(Plane((0, 0, 1), (0, 0, -1)), stock=stock, name="mounting face")
    res = finalize(d, x, preserve=[Cut(bore, stock=stock, name="bore"), face])
    nom = {i["name"]: i for i in res.report["interfaces"]}
    assert nom["mounting face"]["flatness"] < 1e-6 and abs(nom["mounting face"]["offset_mean"]) < 1e-6
    pv_ = print_variant(res)
    rep = {i["name"]: i for i in pv_["report"]["interfaces"]}
    assert abs(rep["bore_print"]["radius_mean"] - (R - stock)) < 0.01 * R
    assert abs(rep["mounting face_print"]["offset_mean"]) < 1e-6           # the moved plane, exactly
    assert pv_["report"]["edges"]["open"] == 0
    added = pv_["report"]["volume"] - pv_["report"]["volume_nominal"]
    # bore ring 4·π(R² − (R − s)²) plus the slab (14² − π(R − s)²)·s, minus the slab edge rounding
    expect = 4 * np.pi * (R ** 2 - (R - stock) ** 2) + (14 * 14 - np.pi * (R - stock) ** 2) * stock
    assert abs(added / expect - 1) < 0.03
