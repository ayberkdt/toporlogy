"""toporlogy.geometry: signed distances, CSG voxelisation with partial volumes, STL I/O and voxelisation,
winding number, bore detection."""
import math

import numpy as np
import pytest

from toporlogy import geometry as g

RNG = np.random.default_rng(0)


def _lbracket() -> g.Shape:
    return g.csg([("add", g.Box([0, 0, 0], [40, 10, 10])), ("add", g.Box([0, 0, 0], [10, 40, 10])),
                  ("cut", g.Cylinder([34, 5, -1], [34, 5, 11], 3))])


def test_primitive_distances_exact():
    b = g.Box([0, 0, 0], [2, 1, 1])
    P = np.array([[1, 0.5, 0.5], [3, 0.5, 0.5], [3, 2, 0.5], [1, 0.5, 0.9]])
    assert np.allclose(b.sdf(P), [-0.5, 1.0, math.sqrt(2), -0.1])
    c = g.Cylinder([0, 0, 0], [1, 1, 1], 0.5)                       # oblique axis
    a = np.array([1, 1, 1]) / math.sqrt(3)
    q = 0.5 * np.array([1, 1, 1]) + 0.8 * np.array([1, -1, 0]) / math.sqrt(2)
    assert c.sdf(q[None])[0] == pytest.approx(0.3, abs=1e-12)
    end = -0.2 * a                                                  # beyond the p0 cap, on the axis
    assert c.sdf(end[None])[0] == pytest.approx(0.2, abs=1e-12)
    sq = g.Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert np.allclose(sq.sdf(np.array([[1, 1], [3, 1], [1, 0.25]])), [-1, 1, -0.25])
    st = g.Capsule([0, 0], [4, 0], 1.0)
    assert np.allclose(st.sdf(np.array([[2, 0], [2, 2], [6, 0]])), [-1, 1, 1])
    sl = g.Slot([0, 0, 0], [4, 0, 0], 1.0, [0, 0, 2])
    assert np.allclose(sl.sdf(np.array([[2, 0, 1], [2, 0, 3], [2, 1.5, 1]])), [-1, 1, 0.5])
    pr = g.Prism([[0, 0], [2, 0], [0, 2]], axis="y", range=(0, 3))    # polygon in (x, z)
    assert pr.inside(np.array([[0.5, 1.0, 0.5]]))[0] and not pr.inside(np.array([[1.5, 1.0, 1.5]]))[0]


def test_csg_sdf_is_1_lipschitz():
    """The voxeliser skips voxels by |sdf| > half diagonal; that is exact only for a 1-Lipschitz sdf."""
    s = _lbracket() | g.Slot([20, 20, 0], [30, 20, 0], 2, [0, 0, 10])
    A = RNG.uniform(-5, 45, (4000, 3))
    B = A + RNG.normal(scale=2.0, size=A.shape)
    ratio = np.abs(s.sdf(A) - s.sdf(B)) / np.linalg.norm(A - B, axis=1)
    assert ratio.max() <= 1 + 1e-12


def test_coverage_equals_brute_force_sampling():
    """Skipping far voxels changes nothing: every voxel sampled with the same sub-grid gives identical fractions."""
    s = _lbracket()
    gshape, origin = g.grid_for(*s.bbox(), 2.5)
    fast = g.coverage(s, (gshape, 2.5, origin), sub=3)
    C = origin + 2.5 * (np.indices(gshape).reshape(3, -1).T + 0.5)
    off = ((np.arange(3) + 0.5) / 3 - 0.5) * 2.5
    O = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    brute = (s.sdf((C[:, None] + O[None]).reshape(-1, 3)) <= 0).reshape(len(C), -1).mean(1).reshape(gshape)
    assert np.array_equal(fast, brute)


def test_voxelize_volumes():
    d = g.voxelize(_lbracket(), 1.0)
    exact = 40 * 10 * 10 + 30 * 10 * 10 - math.pi * 9 * 10
    assert d.frac.sum() / exact - 1 == pytest.approx(0, abs=2e-3)       # measured 4.1e-4
    d2 = g.voxelize(g.Box([0, 0], [10, 6]) - g.Ball([5, 3], 1.5), 0.25, thickness=2.0)
    assert d2.dim == 2 and d2.thickness == 2.0
    assert d2.frac.sum() * 0.0625 / (60 - math.pi * 2.25) - 1 == pytest.approx(0, abs=2e-3)
    ball = g.voxelize(g.Ball([0, 0, 0], 1.0), 0.1, padding=1)
    assert ball.frac.sum() * 1e-3 / (4 / 3 * math.pi) - 1 == pytest.approx(0, abs=2e-3)   # measured −3.1e-4
    assert ball.active[0].sum() == 0                                   # padding layer empty


def test_keep_and_void_shapes_and_connectivity():
    s = g.Box([0, 0, 0], [10, 4, 4])
    d = g.voxelize(s, 1.0, keep=g.Box([0, 0, 0], [2, 4, 4]), void=g.Box([5, 0, 0], [6, 4, 4]))
    assert d.keep.sum() == 2 * 16 and d.void.sum() == 16
    # two cubes touching along an edge only: the edge is a hinge, the smaller cube is dropped
    two = g.Box([0, 0, 0], [4, 4, 4]) | g.Box([4, 4, 0], [6, 6, 4])
    d1 = g.voxelize(two, 1.0)
    assert d1.n_active == 64
    d2 = g.voxelize(two, 1.0, connectivity=None)
    assert d2.n_active == 64 + 16


@pytest.fixture(scope="module")
def sphere_mesh():
    import pyvista as pv
    s = pv.Sphere(radius=1.0, theta_resolution=64, phi_resolution=64).triangulate()
    return np.asarray(s.points, float), np.asarray(s.faces).reshape(-1, 4)[:, 1:]


def test_stl_roundtrip_binary_ascii(tmp_path, sphere_mesh):
    import pyvista as pv
    V, F = sphere_mesh
    p = g.write_stl(tmp_path / "s.stl", V, F)
    V2, F2 = g.read_stl(p)
    assert len(F2) == len(F) and len(V2) == len(V)
    assert g.mesh_volume(V2, F2) == pytest.approx(g.mesh_volume(V, F), rel=1e-6)   # float32 storage
    mesh = pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel())
    mesh.save(tmp_path / "a.stl", binary=False)
    V3, F3 = g.read_stl(tmp_path / "a.stl")
    assert len(F3) == len(F)
    assert g.mesh_volume(V3, F3) == pytest.approx(g.mesh_volume(V, F), rel=1e-6)


def test_stl_voxelisation_volume(tmp_path, sphere_mesh):
    """Closed sphere mesh (64×64) written to STL and voxelised at h = R/10: volume within 1 % of 4/3 π R³."""
    V, F = sphere_mesh
    V2, F2 = g.read_stl(g.write_stl(tmp_path / "s.stl", V, F))
    d = g.voxelize_mesh(V2, F2, 0.1)
    vol = d.frac.sum() * 1e-3
    assert abs(vol / (4 / 3 * math.pi) - 1) < 0.01               # measured −0.24 % (mesh itself: −0.21 %)
    assert abs(vol / g.mesh_volume(V2, F2) - 1) < 0.002
    assert all(v["dropped"] == 0 for v in d.voxelization_info.values())


def test_stl_cylinder_and_winding_number():
    import pyvista as pv
    cy = pv.Cylinder(radius=1, height=3, resolution=96, capping=True).triangulate()
    V, F = np.asarray(cy.points, float), np.asarray(cy.faces).reshape(-1, 4)[:, 1:]
    d = g.voxelize_mesh(V, F, 0.1)
    assert abs(d.frac.sum() * 1e-3 / (math.pi * 3) - 1) < 0.01          # measured −0.02 %
    w = g.winding_number(np.array([[0, 0, 0], [2, 0, 0], [0.9, 0, 0]]), V, F)
    assert np.allclose(w, [1, 0, 1], atol=1e-9)


def test_find_bores_3d_2d_and_clevis():
    d = g.voxelize(_lbracket(), 1.0)
    b = g.find_bores(d)
    assert len(b) == 1
    assert np.allclose(b[0].axis, [0, 0, 1]) and np.allclose(b[0].center, [34, 5, 5], atol=0.1)
    assert b[0].radius == pytest.approx(3.0, rel=0.02)                   # measured 2.985
    assert b[0].segments == [(0.0, 10.0)]
    d2 = g.voxelize(g.Box([0, 0], [10, 6]) - g.Ball([5.1, 3.05], 1.5), 0.25)
    b2 = g.find_bores(d2)
    assert len(b2) == 1 and b2[0].axis is None
    assert np.allclose(b2[0].center, [5.1, 3.05], atol=0.02) and b2[0].radius == pytest.approx(1.5, rel=0.01)
    clevis = g.csg([("add", g.Box([0, 0, 0], [12, 12, 4])), ("add", g.Box([0, 0, 8], [12, 12, 12])),
                    ("add", g.Box([0, 0, 0], [2, 12, 12])),          # back wall joins the two arms
                    ("cut", g.Cylinder([6, 6, -1], [6, 6, 13], 2.5))])
    bc_ = g.find_bores(g.voxelize(clevis, 0.5))
    assert len(bc_) == 1 and len(bc_[0].segments) == 2
    assert isinstance(bc_[0].shape(), g.Cylinder)


def test_binary_stl_with_trailing_bytes_or_solid_header(tmp_path, sphere_mesh):
    """B-12: a binary STL was recognised only if its size was exactly 84 + 50 n; padding after the last record sent it
    to the ASCII parser, which raised. A binary header that starts with "solid" must still read as binary."""
    V, F = sphere_mesh
    p = g.write_stl(tmp_path / "s.stl", V, F)
    raw = p.read_bytes()
    padded = tmp_path / "padded.stl"
    padded.write_bytes(raw + b"\x00" * 13)
    solid = tmp_path / "solid_header.stl"
    solid.write_bytes(b"solid exported by some CAD".ljust(80, b" ") + raw[80:])
    V0, F0 = g.read_stl(p)
    for f in (padded, solid):
        V1, F1 = g.read_stl(f)
        assert np.array_equal(V1, V0) and np.array_equal(F1, F0), f.name


def test_csg_booleans_reject_mixed_dimensions():
    """B-13: Difference refused a 2-D and a 3-D shape, Union and Intersection silently took the first one's dim."""
    sq, cube = g.Box([0, 0], [1, 1]), g.Box([0, 0, 0], [1, 1, 1])
    for op in (g.Union, g.Intersection, g.Difference):
        with pytest.raises(ValueError, match="different dimension"):
            op(sq, cube)
    assert g.Union(cube, g.Box([1, 0, 0], [2, 1, 1])).dim == 3


def _box_mesh(lo, hi):
    """Closed box surface, 8 vertices / 12 triangles, float64 (no float32 rounding of the corners)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


@pytest.mark.parametrize("lo, hi", [((0.1234, 0.2345, 0.0567), (2.3123, 1.7321, 1.3311)),   # faces between samples
                                    ((0.1, 0.2, 0.05), (2.3, 1.7, 1.33))])                   # faces on sample planes
def test_mesh_sdf_grid_is_exact_near_the_surface_and_signed_everywhere(lo, hi):
    V, F = _box_mesh(lo, hi)
    org, h, shape, band = np.array([-0.5, -0.5, -0.5]), 0.1, (35, 30, 25), 0.3
    d = g.mesh_sdf_grid(V, F, org, h, shape, band=band)
    ax = [org[k] + h * np.arange(shape[k]) for k in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    ref = g.Box(lo, hi).sdf(P).reshape(shape)
    near = np.abs(ref) <= band
    assert np.abs(d - ref)[near & (np.abs(ref) > 1e-9)].max() < 1e-12
    on = np.abs(ref) <= 1e-9                                            # samples on the surface: |d| = 0
    assert not on.any() or np.abs(d[on]).max() < 1e-12
    off = np.abs(ref) > 1e-9
    assert (np.sign(d[off]) == np.sign(ref[off])).all()
    assert (np.abs(d[~near]) >= band * (1 - 1e-9)).all()


def test_mesh_sdf_grid_sign_matches_winding_number(sphere_mesh):
    V, F = sphere_mesh
    org, h, shape = np.array([-1.3, -1.3, -1.3]), 0.1, (27, 27, 27)
    d = g.mesh_sdf_grid(V, F, org, h, shape, band=0.25)
    ax = [org[k] + h * np.arange(shape[k]) for k in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    w = g.winding_number(P, V, F)
    clear = np.abs(np.linalg.norm(P, axis=1) - 1.0) > 0.02             # off the (faceted) surface
    assert ((d.ravel() < 0) == (w > 0.5))[clear].all()
